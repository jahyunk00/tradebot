"""Main orchestrator: backtest → Stage 1 daily → Stage 2 weekly synthesis."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.analyzer import MarketAnalyzer
from agent.config import load_config
from agent.digest_store import load_recent_digests, save_daily_digest
from agent.guardrails import Guardrails, TradeIntent, TradingMode
from backtest.engine import run_backtest
from broker.robinhood_client import RobinhoodMCPClient
from data.intelligence import (
    build_intelligence_package,
    compute_changes,
    load_snapshot,
    save_snapshot,
)

logger = logging.getLogger(__name__)


class TradingAgentRunner:
    SNAPSHOT_NAME = "latest_intelligence.json"

    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        connect_robinhood: bool = True,
    ) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.agent_config, self.guardrails_config = load_config(self.base_dir)
        self.connect_robinhood = connect_robinhood
        self.log_dir = self.base_dir / self.agent_config.logging.directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.log_dir / self.SNAPSHOT_NAME

    async def run(self) -> dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info("Starting run %s in mode=%s", run_id, self._effective_mode_label())
        retail = self.agent_config.retail

        # Step 1: Backtest with SPY benchmark comparison
        backtest_report = run_backtest(
            tickers=self.agent_config.watchlist,
            strategy_name=self.agent_config.backtest.strategy,
            lookback_days=self.agent_config.backtest.lookback_days,
            initial_capital=self.agent_config.backtest.initial_capital,
            benchmark_ticker=retail.benchmark_ticker,
        )

        guardrails = Guardrails(self.agent_config, self.guardrails_config)
        evaluated = guardrails.evaluate_backtest(backtest_report.aggregate)
        comparison = backtest_report.benchmark_comparison

        logger.info(
            "Backtest: CAGR=%.1f%% Sharpe=%.2f Calmar=%.2f vs %s CAGR=%.1f%% — %s",
            evaluated.cagr_pct,
            evaluated.sharpe_ratio,
            evaluated.calmar_ratio,
            retail.benchmark_ticker,
            comparison.benchmark.cagr_pct if comparison else 0,
            comparison.summary if comparison else "no benchmark",
        )

        # Step 2: Market intelligence
        previous = load_snapshot(self.snapshot_path)
        intelligence = build_intelligence_package(
            self.agent_config.watchlist,
            news_per_ticker=retail.news_headlines_per_ticker,
            benchmark=retail.benchmark_ticker,
        )
        changes = compute_changes(intelligence, previous)

        account_context: dict[str, Any] | None = None
        if self.connect_robinhood:
            account_context = await self._fetch_account_context()

        # Step 3: Two-stage LLM pipeline
        analyzer = MarketAnalyzer(
            model=self.agent_config.llm.model,
            max_tokens=self.agent_config.llm.max_tokens,
        )
        prior_digests = load_recent_digests(self.log_dir, days=7)

        backtest_summary = {
            "strategy": backtest_report.aggregate.__dict__,
            "per_ticker": {k: v.__dict__ for k, v in backtest_report.per_ticker.items()},
            "benchmark": comparison.benchmark.__dict__ if comparison else {},
            "vs_benchmark": {
                "beats_cagr": comparison.beats_benchmark_cagr if comparison else False,
                "beats_sharpe": comparison.beats_benchmark_sharpe if comparison else False,
                "beats_calmar": comparison.beats_benchmark_calmar if comparison else False,
                "cagr_spread_pct": comparison.cagr_spread_pct if comparison else 0,
                "summary": comparison.summary if comparison else "",
            },
        }
        guardrail_summary = {
            "effective_mode": guardrails.effective_mode.value,
            "force_analyze_only": self.guardrails_config.force_analyze_only,
            "max_order_usd": self.guardrails_config.max_order_usd,
            "allowed_tickers": self.guardrails_config.allowed_tickers,
            "backtest_passed": evaluated.passed,
        }

        daily_digest, weekly_analysis = analyzer.analyze_two_stage(
            watchlist=self.agent_config.watchlist,
            intelligence=intelligence,
            changes=changes,
            daily_digests=prior_digests,
            backtest_summary=backtest_summary,
            guardrail_summary=guardrail_summary,
            account_context=account_context,
        )

        save_snapshot(intelligence, self.snapshot_path)
        save_daily_digest(self.log_dir, daily_digest, {"run_id": run_id, "stage": 1})

        # Step 4: Validate trade intents from Stage 2
        raw_intents = analyzer.extract_trade_intents(weekly_analysis, self.guardrails_config.max_order_usd)
        trade_results: list[dict[str, Any]] = []

        for raw in raw_intents:
            intent = TradeIntent(**raw)
            verdict = guardrails.validate_trade(intent)
            trade_results.append(
                {"intent": raw, "allowed": verdict.allowed, "reasons": verdict.reasons, "executed": False}
            )
            if not verdict.allowed or guardrails.effective_mode != TradingMode.AUTO_EXECUTE:
                logger.info(
                    "Trade blocked or analyze-only: %s %s $%.2f",
                    intent.side,
                    intent.ticker,
                    intent.amount_usd,
                )

        result = {
            "run_id": run_id,
            "mode": guardrails.effective_mode.value,
            "pipeline": "two_stage",
            "daily_digest": daily_digest,
            "analysis": weekly_analysis,
            "intelligence": intelligence,
            "changes": changes,
            "backtest": backtest_summary,
            "trade_intents": trade_results,
            "account_connected": account_context is not None,
            "digests_this_week": len(load_recent_digests(self.log_dir, days=7)),
        }

        log_path = self.log_dir / f"run_{run_id}.json"
        log_path.write_text(json.dumps(result, indent=2, default=str))
        (self.log_dir / f"daily_{run_id}.md").write_text(daily_digest)
        (self.log_dir / f"analysis_{run_id}.md").write_text(weekly_analysis)

        logger.info("Run complete. Logs: %s", log_path)
        return result

    async def _fetch_account_context(self) -> dict[str, Any] | None:
        try:
            client = RobinhoodMCPClient(
                mcp_url=self.agent_config.robinhood.mcp_url,
                oauth_server_url=self.agent_config.robinhood.oauth_server_url,
                token_path=str(self.base_dir / ".tokens" / "robinhood_oauth.json"),
            )
            await client.connect()
            context = await client.get_account_context()
            await client.disconnect()
            return context
        except Exception as exc:
            logger.warning("Robinhood MCP unavailable: %s", exc)
            return None

    def _effective_mode_label(self) -> str:
        g = Guardrails(self.agent_config, self.guardrails_config)
        return g.effective_mode.value
