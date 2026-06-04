#!/usr/bin/env python3
"""Robinhood agentic trading bot — analyze-only by default."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.briefing import BriefingRunner
from agent.config import load_config
from agent.guardrails import Guardrails
from agent.runner import TradingAgentRunner
from backtest.engine import run_backtest


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _print_metrics(label: str, m) -> None:
    print(f"  {label}")
    print(f"    Total return:  {m.total_return_pct:.2f}%")
    print(f"    CAGR:          {m.cagr_pct:.2f}%")
    print(f"    Sharpe:        {m.sharpe_ratio:.2f}")
    print(f"    Sortino:       {m.sortino_ratio:.2f}")
    print(f"    Calmar:        {m.calmar_ratio:.2f}")
    print(f"    Max drawdown:  {m.max_drawdown_pct:.2f}%")
    print(f"    Win rate:      {m.win_rate_pct:.2f}%")
    print(f"    Trade count:   {m.trade_count}")


def cmd_backtest(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    agent_cfg, guard_cfg = load_config(ROOT)
    report = run_backtest(
        tickers=agent_cfg.watchlist,
        strategy_name=agent_cfg.backtest.strategy,
        lookback_days=agent_cfg.backtest.lookback_days,
        initial_capital=agent_cfg.backtest.initial_capital,
        benchmark_ticker=agent_cfg.retail.benchmark_ticker,
    )
    guardrails = Guardrails(agent_cfg, guard_cfg)
    result = guardrails.evaluate_backtest(report.aggregate)
    cmp = report.benchmark_comparison

    print("\n=== BACKTEST RESULTS ===")
    print(f"Strategy: {agent_cfg.backtest.strategy}")
    print(f"Lookback: {agent_cfg.backtest.lookback_days} days")
    print(f"Tickers tested: {list(report.per_ticker.keys())}")
    print()
    _print_metrics("Strategy (aggregate)", result)
    print(f"    PASSED GATE:   {'YES' if result.passed else 'NO'}")
    print()

    if cmp:
        _print_metrics(f"Benchmark ({cmp.benchmark_ticker} buy & hold)", cmp.benchmark)
        print()
        print("  vs Benchmark:")
        print(f"    CAGR spread:   {cmp.cagr_spread_pct:+.2f}%  {'BEAT' if cmp.beats_benchmark_cagr else 'LOSE'}")
        print(f"    Sharpe spread: {cmp.sharpe_spread:+.2f}   {'BEAT' if cmp.beats_benchmark_sharpe else 'LOSE'}")
        print(f"    Calmar spread: {cmp.calmar_spread:+.2f}   {'BEAT' if cmp.beats_benchmark_calmar else 'LOSE'}")
        print(f"    Verdict:       {cmp.summary}")

    print()
    for ticker, metrics in report.per_ticker.items():
        print(f"  {ticker}: CAGR={metrics.cagr_pct:.1f}% Sharpe={metrics.sharpe_ratio:.2f} Calmar={metrics.calmar_ratio:.2f}")


async def cmd_briefing(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    runner = BriefingRunner(ROOT, connect_robinhood=not args.no_robinhood)
    result = await runner.run()

    print("\n" + "=" * 60)
    print("STAGE 1 — Daily Digest")
    print("=" * 60)
    print(result["briefing"])
    print("=" * 60)
    print(f"\nDigests this week: {result['digests_this_week']}")
    print(f"Saved to logs/briefing_{result['run_id']}.md")


async def cmd_run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    runner = TradingAgentRunner(ROOT, connect_robinhood=not args.no_robinhood)
    result = await runner.run()

    print("\n" + "=" * 60)
    print("STAGE 1 — Daily Digest")
    print("=" * 60)
    print(result["daily_digest"][:2000] + ("..." if len(result["daily_digest"]) > 2000 else ""))
    print("\n" + "=" * 60)
    print("STAGE 2 — Weekly Portfolio Synthesis")
    print("=" * 60)
    print(result["analysis"])
    print("=" * 60)

    vs = result["backtest"].get("vs_benchmark", {})
    print(f"\nMode: {result['mode']}")
    print(f"Backtest passed gate: {result['backtest']['strategy']['passed']}")
    print(f"vs SPY: {vs.get('summary', 'N/A')}")
    print(f"Digests this week: {result['digests_this_week']}")
    print(f"Trade intents: {len(result['trade_intents'])}")
    print(f"\nFull reports: logs/daily_{result['run_id']}.md + logs/analysis_{result['run_id']}.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robinhood agentic trading bot for retail investors"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Run historical backtest with SPY benchmark comparison")
    bt.set_defaults(func=cmd_backtest)

    brief = sub.add_parser("briefing", help="Stage 1 daily digest only (~2 min read)")
    brief.add_argument("--no-robinhood", action="store_true", help="Skip Robinhood MCP connection")
    brief.set_defaults(func=lambda a: asyncio.run(cmd_briefing(a)))

    run = sub.add_parser("run", help="Full two-stage pipeline: backtest + daily + weekly synthesis")
    run.add_argument("--no-robinhood", action="store_true", help="Skip Robinhood MCP connection")
    run.set_defaults(func=lambda a: asyncio.run(cmd_run(a)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
