"""Boss-orchestrated trader — three leg agents report, boss executes."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.bankroll import resolve_bankroll
from agent.runtime_state import append_progress
from agent.boss.agent import decide_from_history, load_boss_weights, weights_from_config
from agent.config import load_config
from agent.guardrails import Guardrails, TradeIntent, TradingMode
from agent.rules_signals import LiveSignal, parse_positions
from agent.rules_trader import RulesTrader
from agent.paper_trials import live_promoted_tickers, run_paper_trials
from agent.watchlist import resolve_watchlist
from backtest.engine import run_backtest
from broker.executor import OrderExecutor
from broker.robinhood_client import RobinhoodMCPClient
from data.market_data import fetch_history

logger = logging.getLogger(__name__)


class BossTrader(RulesTrader):
    """RulesTrader extended with boss agent decision layer."""

    async def run(self) -> dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        retail = self.agent_config.retail
        weights_path = self.base_dir / self.agent_config.boss.weights_path
        weights = load_boss_weights(weights_path, fallback=weights_from_config(self.agent_config))

        watchlist, screener_meta = resolve_watchlist(self.base_dir, self.agent_config)
        benchmark = retail.benchmark_ticker.upper()
        promoted = live_promoted_tickers(self.base_dir)
        allowed = list(dict.fromkeys([*watchlist, *promoted, benchmark]))
        guard_cfg = self.guardrails_config.model_copy(update={"allowed_tickers": allowed})

        report = run_backtest(
            tickers=watchlist,
            strategy_name="ensemble_weighted",
            lookback_days=self.agent_config.backtest.lookback_days,
            initial_capital=self.agent_config.backtest.initial_capital,
            benchmark_ticker=retail.benchmark_ticker,
        )

        lookback = max(
            self.agent_config.backtest.lookback_days,
            self.agent_config.hmm.lookback + 10,
        )
        history = fetch_history(watchlist, lookback)
        decision = decide_from_history(history, self.agent_config, weights)

        signal = LiveSignal(
            strategy="boss_ensemble",
            target_ticker=decision.target_ticker,
            previous_ticker=None,
            action="buy" if decision.target_ticker else "hold",
            rationale=decision.rationale,
            in_trend=decision.target_ticker is not None,
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

        evaluated = Guardrails(self.agent_config, guard_cfg).evaluate_backtest(
            report.aggregate
        )
        guardrails = Guardrails(
            self.agent_config,
            guard_cfg,
            bankroll=bankroll,
            backtest_result=evaluated,
            open_positions=len([v for v in positions.values() if v > 0]),
        )

        account_number = await executor.ensure_account(account_context)
        trade_plan = self._build_trade_plan(signal, positions, bankroll, guardrails)

        executions: list[dict[str, Any]] = []
        mode = guardrails.effective_mode
        can_trade = guardrails.can_execute_trades()

        trials_report = run_paper_trials(
            self.base_dir,
            self.agent_config,
            guard_cfg,
            watchlist=watchlist,
            rotation_swaps=screener_meta.get("swaps") or [],
            combined_scores=decision.combined_scores,
            dry_run=self.dry_run,
        )
        promotion_executions: list[dict[str, Any]] = []
        can_promote_live = self.dry_run or (mode == TradingMode.AUTO_EXECUTE and can_trade.allowed)
        if can_promote_live and trials_report.get("enabled") and trials_report.get("promoted_this_run"):
            pt = self.agent_config.paper_trials
            promo_usd = float(getattr(pt, "virtual_usd", 15.0))
            for promo in trials_report["promoted_this_run"]:
                ticker = promo["ticker"]
                try:
                    review = await executor.place_market_order(
                        account_number, ticker, "buy", promo_usd, dry_run=self.dry_run
                    )
                    promotion_executions.append(
                        {"ticker": ticker, "usd": promo_usd, "executed": not self.dry_run, "result": review}
                    )
                    logger.info("Auto-promoted live buy: %s $%.0f", ticker, promo_usd)
                except Exception as exc:
                    promotion_executions.append({"ticker": ticker, "error": str(exc)})

        if self.dry_run:
            logger.info("Boss dry-run — orders reviewed, not placed.")

        for step in trade_plan:
            intent = step["intent"]
            verdict = guardrails.validate_trade(intent)
            step_result = {**step, "allowed": verdict.allowed, "reasons": verdict.reasons, "executed": False}
            if not verdict.allowed:
                executions.append(step_result)
                continue
            if self.dry_run or mode != TradingMode.AUTO_EXECUTE or not can_trade.allowed:
                try:
                    step_result["review"] = await executor.place_market_order(
                        account_number, intent.ticker, intent.side, intent.amount_usd, dry_run=True
                    )
                except Exception as exc:
                    step_result["reasons"] = [*verdict.reasons, str(exc)]
                    step_result["allowed"] = False
                executions.append(step_result)
                continue
            try:
                step_result["result"] = await executor.place_market_order(
                    account_number, intent.ticker, intent.side, intent.amount_usd, dry_run=False
                )
                step_result["executed"] = True
            except Exception as exc:
                step_result["reasons"] = [str(exc)]
                step_result["allowed"] = False
            executions.append(step_result)

        await client.disconnect()

        cmp = report.benchmark_comparison
        result = {
            "run_id": run_id,
            "engine": "boss",
            "dry_run": self.dry_run,
            "mode": mode.value,
            "strategy": "boss_ensemble",
            "boss_weights": {**weights.as_dict(), "source": weights.source, "train_sharpe": weights.train_sharpe},
            "watchlist": watchlist,
            "screener": screener_meta,
            "paper_trials": trials_report,
            "promotion_trades": promotion_executions,
            "leg_reports": decision.leg_reports,
            "combined_scores": decision.combined_scores,
            "executive": decision.executive,
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

        out = self.log_dir / f"boss_trade_{run_id}.json"
        out.write_text(json.dumps(result, indent=2, default=str))

        append_progress(
            self.base_dir,
            equity_usd=float(bankroll.equity_usd),
            mode="live" if not self.dry_run else "dry_run",
            pick=signal.target_ticker,
        )
        return result
