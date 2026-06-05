"""Rules-only auto-trader — no Claude required."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.bankroll import clamp_trade_amount, resolve_bankroll
from agent.config import load_config
from agent.guardrails import Guardrails, TradeIntent, TradingMode
from agent.rules_signals import LiveSignal, get_live_signal, parse_positions
from backtest.engine import run_backtest
from broker.executor import OrderExecutor
from broker.robinhood_client import RobinhoodMCPClient

logger = logging.getLogger(__name__)


class RulesTrader:
    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        dry_run: bool = True,
    ) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.agent_config, self.guardrails_config = load_config(self.base_dir)
        self.dry_run = dry_run
        self.log_dir = self.base_dir / self.agent_config.logging.directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        strategy = self.agent_config.backtest.strategy
        if self.agent_config.engine == "kronos":
            strategy = "kronos_top_k"
        elif self.agent_config.engine == "ensemble":
            strategy = "ensemble_weighted"
        retail = self.agent_config.retail

        report = run_backtest(
            tickers=self.agent_config.watchlist,
            strategy_name=strategy,
            lookback_days=self.agent_config.backtest.lookback_days,
            initial_capital=self.agent_config.backtest.initial_capital,
            benchmark_ticker=retail.benchmark_ticker,
        )

        client = RobinhoodMCPClient(
            mcp_url=self.agent_config.robinhood.mcp_url,
            token_path=str(self.base_dir / ".tokens" / "robinhood_oauth.json"),
        )
        await client.connect()
        account_context = await client.get_account_context()
        executor = OrderExecutor(client)

        br_cfg = self.guardrails_config.bankroll
        bankroll = resolve_bankroll(
            mode=br_cfg.mode,
            initial_usd=br_cfg.initial_usd,
            ceiling_usd=br_cfg.ceiling_usd,
            account_context=account_context,
        )
        positions = parse_positions(account_context)
        if not positions and bankroll.positions:
            positions = bankroll.positions

        evaluated = Guardrails(self.agent_config, self.guardrails_config).evaluate_backtest(
            report.aggregate
        )
        cmp = report.benchmark_comparison
        if cmp:
            wins = sum([cmp.beats_benchmark_cagr, cmp.beats_benchmark_sharpe, cmp.beats_benchmark_calmar])
            evaluated.details["vs_benchmark"] = {
                "summary": cmp.summary,
                "cagr_spread_pct": cmp.cagr_spread_pct,
                "beats_majority": wins >= 2,
            }
        guardrails = Guardrails(
            self.agent_config,
            self.guardrails_config,
            bankroll=bankroll,
            backtest_result=evaluated,
            open_positions=len([v for v in positions.values() if v > 0]),
        )

        signal = get_live_signal(
            strategy,
            self.agent_config.watchlist,
            lookback_days=self.agent_config.backtest.lookback_days,
            benchmark=retail.benchmark_ticker,
            kronos_cfg=self.agent_config.kronos.model_dump(),
        )

        account_number = await executor.ensure_account(account_context)
        trade_plan = self._build_trade_plan(signal, positions, bankroll, guardrails)

        executions: list[dict[str, Any]] = []
        mode = guardrails.effective_mode
        can_trade = guardrails.can_execute_trades()

        if self.dry_run:
            logger.info("Dry-run mode — orders will be reviewed, not placed.")

        for step in trade_plan:
            intent = step["intent"]
            verdict = guardrails.validate_trade(intent)
            step_result = {
                **step,
                "allowed": verdict.allowed,
                "reasons": verdict.reasons,
                "executed": False,
            }

            if not verdict.allowed:
                executions.append(step_result)
                continue

            if self.dry_run or mode != TradingMode.AUTO_EXECUTE or not can_trade.allowed:
                try:
                    review = await executor.place_market_order(
                        account_number,
                        intent.ticker,
                        intent.side,
                        intent.amount_usd,
                        dry_run=True,
                    )
                    step_result["review"] = review
                except Exception as exc:
                    step_result["reasons"] = [*verdict.reasons, str(exc)]
                    step_result["allowed"] = False
                executions.append(step_result)
                continue

            try:
                placed = await executor.place_market_order(
                    account_number,
                    intent.ticker,
                    intent.side,
                    intent.amount_usd,
                    dry_run=False,
                )
                step_result["executed"] = True
                step_result["result"] = placed
            except Exception as exc:
                step_result["reasons"] = [str(exc)]
                step_result["allowed"] = False

            executions.append(step_result)

        await client.disconnect()

        cmp = report.benchmark_comparison
        result = {
            "run_id": run_id,
            "engine": "ensemble" if self.agent_config.engine == "ensemble" else "rules",
            "dry_run": self.dry_run,
            "mode": mode.value,
            "strategy": strategy,
            "signal": {
                "target_ticker": signal.target_ticker,
                "action": signal.action,
                "rationale": signal.rationale,
            },
            "backtest": {
                "passed": evaluated.passed,
                "cagr_pct": evaluated.cagr_pct,
                "sharpe_ratio": evaluated.sharpe_ratio,
                "vs_benchmark": cmp.summary if cmp else "",
            },
            "bankroll": bankroll.to_summary(),
            "positions_before": positions,
            "trade_plan": executions,
            "can_execute": can_trade.allowed,
            "execute_block_reasons": can_trade.reasons,
        }

        out = self.log_dir / f"rules_trade_{run_id}.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Rules trade run saved: %s", out)
        return result

    def _build_trade_plan(
        self,
        signal: LiveSignal,
        positions: dict[str, float],
        bankroll,
        guardrails: Guardrails,
    ) -> list[dict[str, Any]]:
        g = self.guardrails_config
        target = signal.target_ticker
        held = {t: v for t, v in positions.items() if v > 0}
        plan: list[dict[str, Any]] = []

        for ticker, value in held.items():
            if target and ticker == target:
                continue
            amount = round(value, 2)
            if amount <= 0:
                continue
            intent = TradeIntent(
                ticker=ticker,
                side="sell",
                amount_usd=amount,
                order_type="market",
                rationale=f"Exit {ticker} — rules target is {target or 'cash'}.",
            )
            plan.append({"intent": intent, "kind": "exit"})

        if target:
            allocation = bankroll.max_position_usd(g.max_position_pct)
            clamped, notes = clamp_trade_amount(
                allocation,
                target,
                "buy",
                bankroll,
                g.max_position_pct,
                g.max_order_usd,
            )
            current_in_target = held.get(target, 0.0)
            buy_amount = round(max(clamped - current_in_target, 0), 2)
            if buy_amount >= 1.0:
                rationale = signal.rationale
                if notes:
                    rationale += " " + " ".join(notes)
                intent = TradeIntent(
                    ticker=target,
                    side="buy",
                    amount_usd=buy_amount,
                    order_type="market",
                    rationale=rationale,
                )
                plan.append({"intent": intent, "kind": "enter"})

        return plan
