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
from agent.boss.bear_market import detect_market_stress
from agent.config import load_config
from agent.guardrails import Guardrails, TradeIntent, TradingMode, count_meaningful_positions
from agent.rules_signals import LiveSignal, parse_positions
from agent.rules_trader import RulesTrader
from agent.portfolio_plan import build_portfolio_trade_plan, pick_targets
from agent.position_exits import build_exit_plan, parse_position_lots
from agent.options_trader import plan_option_trade
from agent.paper_trials import live_promoted_tickers, run_paper_trials
from agent.launch_schedule import apply_phase_to_guardrails, resolve_trading_phase
from agent.trade_ledger import record_trade, resolve_daily_trade_stats
from agent.watchlist import resolve_watchlist
from backtest.engine import run_backtest
from broker.executor import OrderExecutor
from broker.order_result import extract_action_url, order_error_message, order_succeeded
from broker.robinhood_client import RobinhoodMCPClient
from data.market_data import fetch_history
from data.trajectory import recent_momentum_by_ticker

logger = logging.getLogger(__name__)


class BossTrader(RulesTrader):
    """RulesTrader extended with boss agent decision layer."""

    async def run(self) -> dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        retail = self.agent_config.retail
        weights_path = self.base_dir / self.agent_config.boss.weights_path
        weights = load_boss_weights(weights_path, fallback=weights_from_config(self.agent_config))

        phase = resolve_trading_phase(self.agent_config, self.guardrails_config)
        self.guardrails_config = apply_phase_to_guardrails(self.guardrails_config, phase)
        logger.info("Trading phase: %s — %s", phase.name, phase.message)

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

        boss = self.agent_config.boss
        momentum = recent_momentum_by_ticker(screener_meta.get("details") or [])
        targets = pick_targets(
            decision.combined_scores,
            min_score=boss.min_combined_score,
            mode=boss.portfolio_mode,
            max_picks=boss.max_picks,
            momentum_by_ticker=momentum,
            momentum_weight=boss.momentum_weight,
        )
        primary = targets[0] if targets else decision.target_ticker

        signal = LiveSignal(
            strategy="boss_ensemble",
            target_ticker=primary,
            previous_ticker=None,
            action="buy" if targets else "hold",
            rationale=decision.rationale,
            in_trend=bool(targets),
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

        position_lots = parse_position_lots(account_context)
        held = []
        for ticker, lot in position_lots.items():
            t = ticker.upper()
            qty = float(lot.get("quantity") or 0)
            mv = float(lot.get("market_value") or 0)
            if positions.get(t, 0) > 0 or mv > 0 or qty > 0:
                held.append(t)
        if held:
            missing = [t for t in held if t not in history]
            if missing:
                history = {**history, **fetch_history(missing, lookback)}
            # Some Robinhood payloads only include quantity + avg cost (no market_value).
            # Mark held positions to current market value so sizing/rotation can rebalance.
            for ticker in held:
                if positions.get(ticker, 0) > 0:
                    continue
                lot = position_lots.get(ticker, {})
                qty = float(lot.get("quantity") or 0)
                if qty <= 0:
                    continue
                df = history.get(ticker)
                if df is None or df.empty:
                    continue
                last_px = float(df["Close"].iloc[-1])
                positions[ticker] = round(qty * last_px, 2)

        account_number = await executor.ensure_account(account_context)
        broker_orders = None
        try:
            broker_orders = await client.call_tool(
                "get_equity_orders", {"account_number": account_number}
            )
        except Exception as exc:
            logger.debug("Could not fetch equity orders: %s", exc)

        trades_today, last_trade_at = resolve_daily_trade_stats(
            self.base_dir, broker_orders_payload=broker_orders
        )

        evaluated = Guardrails(self.agent_config, guard_cfg).evaluate_backtest(
            report.aggregate
        )
        guardrails = Guardrails(
            self.agent_config,
            guard_cfg,
            bankroll=bankroll,
            backtest_result=evaluated,
            trades_today=trades_today,
            last_trade_at=last_trade_at,
            positions=positions,
        )

        exit_plan = build_exit_plan(
            position_lots,
            positions=positions,
            history=history,
            agent_cfg=self.agent_config,
            base_dir=self.base_dir,
            executive=decision.executive,
        )
        adjusted_positions = dict(positions)
        for step in exit_plan:
            intent = step["intent"]
            t = intent.ticker.upper()
            held_val = adjusted_positions.get(t, 0.0)
            if held_val <= 0:
                continue
            if intent.amount_usd >= held_val * 0.95:
                adjusted_positions.pop(t, None)
            else:
                adjusted_positions[t] = round(max(held_val - intent.amount_usd, 0), 2)

        entry_plan = build_portfolio_trade_plan(
            targets,
            positions=adjusted_positions,
            bankroll=bankroll,
            guardrails=guardrails,
            rationale=decision.rationale,
            rotate_out=boss.rotate_out or boss.portfolio_mode == "single",
            max_buys_per_run=getattr(boss, "max_buys_per_run", None),
            max_sells_per_run=getattr(boss, "max_sells_per_run", None),
            min_staged_buy_usd=float(getattr(boss, "min_staged_buy_usd", 15.0)),
        )
        # Exits first (stops / profit-taking), then rotation and new entries
        trade_plan = exit_plan + entry_plan

        executions: list[dict[str, Any]] = []
        mode = guardrails.effective_mode
        can_trade = guardrails.can_execute_trades()

        bench_hist = fetch_history([benchmark], min(252, lookback))
        market_stress = detect_market_stress(bench_hist, benchmark)

        trials_report = run_paper_trials(
            self.base_dir,
            self.agent_config,
            guard_cfg,
            watchlist=watchlist,
            rotation_swaps=screener_meta.get("swaps") or [],
            combined_scores=decision.combined_scores,
            dry_run=self.dry_run,
            market_stress=market_stress,
        )
        promotion_executions: list[dict[str, Any]] = []
        broker_block: str | None = None
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
                    ok = self.dry_run or order_succeeded(review)
                    err = order_error_message(review) if not self.dry_run else None
                    if err:
                        broker_block = broker_block or err
                    promotion_executions.append(
                        {
                            "ticker": ticker,
                            "usd": promo_usd,
                            "executed": ok,
                            "broker_error": err,
                            "result": review,
                        }
                    )
                    if ok and not self.dry_run:
                        record_trade(self.base_dir, ticker=ticker, side="buy", amount_usd=promo_usd, executed=True)
                        guardrails.trades_today += 1
                        guardrails.last_trade_at = datetime.now(tz=Guardrails.ET)
                        logger.info("Auto-promoted live buy: %s $%.0f", ticker, promo_usd)
                    elif err:
                        logger.error("Promotion buy failed %s: %s", ticker, err[:200])
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
                placed = await executor.place_market_order(
                    account_number, intent.ticker, intent.side, intent.amount_usd, dry_run=False
                )
                step_result["result"] = placed
                err = order_error_message(placed)
                if err:
                    step_result["executed"] = False
                    step_result["broker_error"] = err
                    step_result["reasons"] = [*verdict.reasons, err]
                    broker_block = broker_block or err
                    logger.error("Order rejected %s %s: %s", intent.side, intent.ticker, err[:200])
                else:
                    step_result["executed"] = True
                    record_trade(
                        self.base_dir,
                        ticker=intent.ticker,
                        side=intent.side,
                        amount_usd=intent.amount_usd,
                        executed=True,
                    )
                    guardrails.trades_today += 1
                    guardrails.last_trade_at = datetime.now(tz=Guardrails.ET)
                    t = intent.ticker.upper()
                    if intent.side.lower() == "buy":
                        prev = guardrails.positions.get(t, 0.0)
                        guardrails.positions[t] = round(prev + intent.amount_usd, 2)
                        if prev < 1.0 and guardrails.positions[t] >= 1.0:
                            guardrails.open_positions = count_meaningful_positions(guardrails.positions)
                    elif intent.side.lower() == "sell":
                        prev = guardrails.positions.get(t, 0.0)
                        remaining = round(max(prev - intent.amount_usd, 0), 2)
                        if remaining >= 1.0:
                            guardrails.positions[t] = remaining
                        else:
                            guardrails.positions.pop(t, None)
                        guardrails.open_positions = count_meaningful_positions(guardrails.positions)
            except Exception as exc:
                step_result["reasons"] = [str(exc)]
                step_result["allowed"] = False
            executions.append(step_result)

        option_executions: list[dict[str, Any]] = []
        opt_cfg = getattr(boss, "option_trading", None)
        if (
            opt_cfg
            and getattr(opt_cfg, "enabled", False)
            and not self.dry_run
            and mode == TradingMode.AUTO_EXECUTE
            and can_trade.allowed
            and bankroll.cash_usd >= float(getattr(opt_cfg, "min_cash_to_trade_usd", 50.0))
        ):
            option_plan = await plan_option_trade(
                client=client,
                targets=targets[: int(getattr(opt_cfg, "max_underlyings_per_run", 1))],
                history=history,
                side=str(getattr(opt_cfg, "side", "call")),
                min_days_to_expiry=int(getattr(opt_cfg, "min_days_to_expiry", 3)),
                max_days_to_expiry=int(getattr(opt_cfg, "max_days_to_expiry", 21)),
            )
            if option_plan:
                placed = await executor.place_option_order(
                    account_number,
                    option_id=option_plan["option_id"],
                    side="buy",
                    position_effect="open",
                    quantity=int(getattr(opt_cfg, "max_contracts_per_run", 1)),
                    dry_run=self.dry_run,
                )
                err = order_error_message(placed)
                option_executions.append(
                    {
                        **option_plan,
                        "executed": err is None,
                        "broker_error": err,
                        "result": placed,
                    }
                )
                if err:
                    broker_block = broker_block or err
                    logger.error("Option order rejected %s %s: %s", option_plan["ticker"], option_plan["option_id"], err[:200])
                else:
                    logger.info(
                        "Option order executed: %s %s %s",
                        option_plan["ticker"],
                        option_plan["type"],
                        option_plan["strike_price"],
                    )

        await client.disconnect()

        cmp = report.benchmark_comparison
        result = {
            "run_id": run_id,
            "engine": "boss",
            "trading_phase": phase.name,
            "phase_message": phase.message,
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
            "market_stress": market_stress,
            "signal": {
                "target_ticker": signal.target_ticker,
                "targets": targets,
                "portfolio_mode": boss.portfolio_mode,
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
            "exit_plan": [
                {
                    "kind": s.get("kind"),
                    "ticker": s["intent"].ticker,
                    "amount_usd": s["intent"].amount_usd,
                    "pnl_pct": s.get("pnl_pct"),
                    "rationale": s["intent"].rationale,
                }
                for s in exit_plan
            ],
            "trade_plan": executions,
            "can_execute": can_trade.allowed,
            "execute_block_reasons": can_trade.reasons,
            "broker_block": broker_block,
            "broker_setup_url": extract_action_url(broker_block) if broker_block else None,
            "option_trades": option_executions,
        }

        out = self.log_dir / f"boss_trade_{run_id}.json"
        out.write_text(json.dumps(result, indent=2, default=str))

        append_progress(
            self.base_dir,
            equity_usd=float(bankroll.equity_usd),
            mode="pilot" if phase.name == "pilot" else ("live" if not self.dry_run else "dry_run"),
            pick=primary,
        )
        return result
