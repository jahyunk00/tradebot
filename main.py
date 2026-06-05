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
from agent.boss.agent import (
    decide_from_history,
    load_boss_weights,
    train_boss_weights,
    weights_from_config,
)
from agent.boss.paper_runner import run_historical_paper_practice, run_paper_session
from agent.boss_trader import BossTrader
from agent.config import load_config
from agent.guardrails import Guardrails
from agent.notify import send_email
from agent.rules_trader import RulesTrader
from agent.runner import TradingAgentRunner
from backtest.engine import compare_strategies, run_backtest


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
    runner = BriefingRunner(
        ROOT,
        connect_robinhood=not args.no_robinhood,
        send_email=not args.no_email,
    )
    result = await runner.run()

    print("\n" + "=" * 60)
    print("STAGE 1 — Daily Digest")
    print("=" * 60)
    print(result["briefing"])
    print("=" * 60)
    print(f"\nRobinhood connected: {result['account_connected']}")
    if result.get("bankroll"):
        b = result["bankroll"]
        print(f"Bankroll: ${b.get('equity_usd')} equity ({b.get('gain_pct', 0):+.1f}%)")
    print(f"Email sent: {result.get('email_sent', False)}")
    print(f"Digests this week: {result['digests_this_week']}")
    print(f"Saved to logs/briefing_{result['run_id']}.md")


async def cmd_run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    runner = TradingAgentRunner(
        ROOT,
        connect_robinhood=not args.no_robinhood,
        send_email=not args.no_email,
    )
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
    print(f"Email sent: {result.get('email_sent', False)}")
    print(f"\nFull reports: logs/daily_{result['run_id']}.md + logs/analysis_{result['run_id']}.md")


async def cmd_test_robinhood(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    from broker.test_connection import save_test_result, test_robinhood_connection

    result = await test_robinhood_connection(ROOT)
    save_test_result(ROOT, result)
    if result.get("status") != "connected":
        sys.exit(1)


def cmd_test_email(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    ok = send_email(
        "Tradebot test email",
        "If you received this, email notifications are working.\n\n— Tradebot",
    )
    print("Email sent successfully." if ok else "Email not sent — check SMTP_* and NOTIFY_EMAIL in .env")
    sys.exit(0 if ok else 1)


def cmd_compare_strategies(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    agent_cfg, guard_cfg = load_config(ROOT)
    guardrails = Guardrails(agent_cfg, guard_cfg)

    rows = compare_strategies(
        tickers=agent_cfg.watchlist,
        lookback_days=agent_cfg.backtest.lookback_days,
        initial_capital=agent_cfg.backtest.initial_capital,
        benchmark_ticker=agent_cfg.retail.benchmark_ticker,
        evaluate_gate=guardrails.evaluate_backtest,
        include_kronos=args.with_kronos,
    )

    print("\n=== STRATEGY COMPARISON (best first) ===")
    print(f"Watchlist: {agent_cfg.watchlist}")
    print(f"Benchmark: {agent_cfg.retail.benchmark_ticker} | Lookback: {agent_cfg.backtest.lookback_days}d\n")
    print(f"{'Strategy':<22} {'CAGR':>7} {'Sharpe':>7} {'Calmar':>7} {'vs SPY':>8} {'Gate':>5} {'Score':>6}")
    print("-" * 72)
    for row in rows:
        cmp = row.benchmark_comparison
        vs = f"{cmp.cagr_spread_pct:+.1f}%"
        gate = "YES" if row.passed_gate else "no"
        print(
            f"{row.strategy:<22} {row.aggregate.cagr_pct:>6.1f}% "
            f"{row.aggregate.sharpe_ratio:>7.2f} {row.aggregate.calmar_ratio:>7.2f} "
            f"{vs:>8} {gate:>5} {row.score:>6.2f}"
        )

    best = rows[0]
    print(f"\nRecommended: **{best.strategy}**")
    print(f"  {best.benchmark_comparison.summary}")
    print(f"\nSet in config.yaml → backtest.strategy: {best.strategy}")


def cmd_kronos_forecast(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    agent_cfg, _ = load_config(ROOT)
    from models.kronos_engine import KronosConfig, kronos_available, rank_watchlist

    if not kronos_available():
        print("Kronos not ready.")
        print("  pip install -r requirements-kronos.txt")
        print("  vendor/Kronos should exist (cloned from github.com/shiyu-coder/Kronos)")
        sys.exit(1)

    cfg = KronosConfig(
        **{
            k: v
            for k, v in agent_cfg.kronos.model_dump().items()
            if k in KronosConfig.__dataclass_fields__
        }
    )
    from data.market_data import fetch_history

    history = fetch_history(agent_cfg.watchlist, max(agent_cfg.backtest.lookback_days, cfg.lookback))
    print("\n=== KRONOS FORECAST RANKINGS ===")
    print(f"Model: {cfg.model_id} | Horizon: {cfg.pred_len} days | T={cfg.temperature}\n")
    ranked = rank_watchlist(history, cfg)
    if not ranked:
        print("No forecasts generated.")
        sys.exit(1)
    for i, (ticker, ret) in enumerate(ranked, 1):
        mark = " ← top pick" if i == 1 else ""
        print(f"  {i}. {ticker}: {ret:+.2f}% forecast{mark}")
    print("\nSet config.yaml → engine: kronos and backtest.strategy: kronos_top_k to trade on this.")


def cmd_ensemble_score(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    agent_cfg, _ = load_config(ROOT)
    from data.market_data import fetch_history
    from models.ensemble_strategy import (
        kronos_cfg_from_agent_config,
        score_watchlist,
        settings_from_agent_config,
    )

    settings = settings_from_agent_config(agent_cfg)
    history = fetch_history(
        agent_cfg.watchlist,
        max(agent_cfg.backtest.lookback_days, settings.hmm_lookback + 10),
    )

    print("\n=== ENSEMBLE (3 legs: Kronos + HMM + third) ===")
    w = settings.weights
    print(
        f"Weights: kronos {w.kronos:.0%} | hmm {w.hmm:.0%} | "
        f"third_leg ({settings.third_leg_model}) {w.third_leg:.0%}\n"
    )

    decision = score_watchlist(
        history,
        settings,
        kronos_cfg=kronos_cfg_from_agent_config(agent_cfg),
        live=True,
    )

    if decision.combined_scores:
        print("Combined scores (0–1):")
        for ticker, score in sorted(decision.combined_scores.items(), key=lambda x: -x[1]):
            mark = " ← pick" if ticker == decision.target_ticker else ""
            print(f"  {ticker}: {score:.3f}{mark}")

    print(f"\nDecision: {decision.target_ticker or 'CASH'}")
    print(f"Reason: {decision.rationale}")
    print("\nDry-run trade: python main.py trade")


def cmd_train_boss(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    agent_cfg, _ = load_config(ROOT)
    print("\n=== TRAIN BOSS (historical practice) ===")
    print(
        f"Training on {agent_cfg.boss.training.lookback_days}d history, "
        f"legs: hmm=on, kronos={'on' if agent_cfg.boss.training.use_kronos else 'off'}, "
        f"third_leg={'on' if agent_cfg.boss.training.use_third_leg else 'off'}"
    )
    print("This may take several minutes...\n")

    weights = train_boss_weights(agent_cfg, base_dir=ROOT)
    print(f"Learned weights → kronos {weights.kronos:.0%}, hmm {weights.hmm:.0%}, third_leg {weights.third_leg:.0%}")
    print(f"Training Sharpe: {weights.train_sharpe:.2f} over {weights.train_samples} rebalance periods")
    print(f"Saved: {agent_cfg.boss.weights_path}")
    print("\nNext: python main.py paper-run --learn   or   python main.py trade")


def cmd_tune_roi(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    print("\n=== TUNE FOR MAX ROI ===")
    print("Retraining boss weights (ROI objective) + grid search — several minutes...\n")
    from agent.boss.tune_roi import tune_for_roi

    result = tune_for_roi(ROOT)
    p = result["best_params"]
    w = result["boss_weights"]
    print(f"Simulated ROI: {result['simulated_roi_pct']:+.1f}%")
    print(f"Best params: min_score={p['min_combined_score']} rebalance={p['rebalance_days']}d cap_boost={p['small_cap_boost']}")
    print(f"Boss weights: kronos {w['kronos']:.0%} hmm {w['hmm']:.0%} third {w['third_leg']:.0%}")
    print(f"Saved: config.yaml + logs/boss_weights.json + logs/roi_tune_result.json")


def cmd_tune_bear(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    target = getattr(args, "target", 5.0)
    print(f"\n=== BEAR MARKET PAPER TEST (target {target}% ROI) ===")
    print("Simulating defensive rules on XBI downtrend periods — several minutes...\n")
    from agent.boss.bear_paper_test import tune_bear_paper

    result = tune_bear_paper(ROOT, target_bear_roi=target, max_rounds=getattr(args, "rounds", 5))
    p = result["best_params"]
    s = result["simulation"]
    s = result["simulation"]
    print(f"Bear ROI: {result['achieved_bear_roi_pct']:+.1f}%  vs XBI {s.get('bear_bench_roi_pct', 0):+.1f}%  alpha {result.get('achieved_bear_alpha_pct', 0):+.1f}%")
    print(f"Met target (alpha >= {target}%): {'YES' if result['met_target'] else 'NO'}")
    print(f"Bear trades: {s.get('bear_trades')}  win rate: {s.get('bear_win_rate_pct')}%  cash in bear: {s.get('bear_cash_pct')}%")
    print(f"Best params: min_news={p.get('min_news_score')} block_metrics={p.get('block_metrics_only')} min_score={p.get('min_combined_score')}")
    print("Saved: config.yaml + logs/bear_paper_tune.json")


def cmd_paper_run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    if args.historical:
        result = run_historical_paper_practice(ROOT, update_weights=args.learn)
        print("\n=== HISTORICAL PAPER PRACTICE ===")
        print(f"Simulated {result['periods']} rebalance periods ({result['trades']} picks)")
        print(f"Paper equity: ${result['final_equity']:.2f} ({result['total_return_pct']:+.2f}%)")
        w = result["boss_weights"]
        print(f"Boss weights: kronos {w['kronos']:.0%}, hmm {w['hmm']:.0%}, third_leg {w['third_leg']:.0%}")
        print(f"Source: {result['weights_source']}")
        return

    result = run_paper_session(ROOT, update_weights=args.learn)
    d = result["decision"]
    print("\n=== PAPER RUN (live quotes) ===")
    print(f"Boss pick: {d['target'] or 'CASH'}")
    print(f"Reason: {d['rationale']}")
    print(f"Paper equity: ${result['portfolio']['equity_usd']:.2f}")
    if args.learn:
        print("Weights updated from this paper session.")


def cmd_boss_score(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    agent_cfg, _ = load_config(ROOT)
    weights_path = ROOT / agent_cfg.boss.weights_path
    weights = load_boss_weights(weights_path, fallback=weights_from_config(agent_cfg))

    lookback = max(agent_cfg.backtest.lookback_days, agent_cfg.hmm.lookback + 10)
    from data.market_data import fetch_history

    history = fetch_history(agent_cfg.watchlist, lookback)
    decision = decide_from_history(history, agent_cfg, weights)

    print("\n=== BOSS + THREE LEG AGENTS ===")
    print(f"Weights ({weights.source}): kronos {weights.kronos:.0%} | hmm {weights.hmm:.0%} | third_leg {weights.third_leg:.0%}")
    print("\nLeg reports:")
    for leg in decision.leg_reports:
        avail = "✓" if leg["available"] else "✗"
        print(f"  [{avail}] {leg['agent_id']}: top={leg.get('top_ticker') or '—'} — {leg.get('note', '')}")
    if decision.combined_scores:
        print("\nBoss combined scores:")
        for ticker, score in sorted(decision.combined_scores.items(), key=lambda x: -x[1]):
            mark = " ← pick" if ticker == decision.target_ticker else ""
            print(f"  {ticker}: {score:.3f}{mark}")
    print(f"\nDecision: {decision.target_ticker or 'CASH'}")
    print(f"Reason: {decision.rationale}")
    print("\nTrain weights: python main.py train-boss")
    print("Paper practice: python main.py paper-run --historical --learn")


async def cmd_trade(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    setup_logging()
    agent_cfg, _ = load_config(ROOT)
    if agent_cfg.engine == "boss" and agent_cfg.boss.enabled:
        trader = BossTrader(ROOT, dry_run=not args.execute)
    else:
        trader = RulesTrader(ROOT, dry_run=not args.execute)
    result = await trader.run()

    sig = result["signal"]
    engine = result.get("engine", "rules")
    title = "BOSS AUTO-TRADER" if engine == "boss" else "RULES AUTO-TRADER"
    print(f"\n=== {title} ===")
    print(f"Engine: {engine} | Dry-run: {result['dry_run']} | Mode: {result['mode']}")
    if engine == "boss":
        bw = result.get("boss_weights", {})
        print(
            f"Boss weights: kronos {bw.get('kronos', 0):.0%} | "
            f"hmm {bw.get('hmm', 0):.0%} | third_leg {bw.get('third_leg', 0):.0%} "
            f"({bw.get('source', 'unknown')})"
        )
        print("\nLeg agents:")
        for leg in result.get("leg_reports", []):
            top = leg.get("top_ticker") or "—"
            avail = "✓" if leg.get("available") else "✗"
            print(f"  [{avail}] {leg.get('agent_id')}: top={top} — {leg.get('note', '')}")
    print(f"Strategy: {result.get('strategy', 'n/a')}")
    print(f"Signal: {sig['action'].upper()} → {sig['target_ticker'] or 'CASH'}")
    print(f"Reason: {sig['rationale']}")
    print(f"\nBacktest gate: {'PASSED' if result['backtest']['passed'] else 'FAILED'}")
    print(f"vs SPY: {result['backtest']['vs_benchmark']}")
    print(f"Bankroll: ${result['bankroll']['equity_usd']} equity, ${result['bankroll']['cash_usd']} cash")

    if result["execute_block_reasons"]:
        print("\nExecution blocked:")
        for r in result["execute_block_reasons"]:
            print(f"  - {r}")

    print("\nTrade plan:")
    if not result["trade_plan"]:
        print("  (no trades needed)")
    for step in result["trade_plan"]:
        intent = step["intent"]
        status = "EXECUTED" if step.get("executed") else ("ALLOWED" if step.get("allowed") else "BLOCKED")
        print(f"  [{status}] {intent.side.upper()} {intent.ticker} ${intent.amount_usd:.2f} — {intent.rationale}")
        for reason in step.get("reasons") or []:
            print(f"           ↳ {reason}")

    log_prefix = "boss_trade" if engine == "boss" else "rules_trade"
    print(f"\nLog: logs/{log_prefix}_{result['run_id']}.json")
    if result["dry_run"]:
        print("\nDry-run only. To place live orders after enabling auto_execute:")
        print("  1. guardrails.yaml → force_analyze_only: false")
        print("  2. config.yaml → mode: auto_execute")
        print("  3. python main.py trade --execute")


def main() -> None:
    import os
    import sys
    from pathlib import Path

    if len(sys.argv) == 1 and (
        os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_SERVICE_NAME")
    ):
        import runpy

        runpy.run_path(
            str(Path(__file__).resolve().parent / "scripts" / "railway_trade.py"),
            run_name="__main__",
        )
        return

    parser = argparse.ArgumentParser(
        description="Robinhood agentic trading bot for retail investors"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Run historical backtest with SPY benchmark comparison")
    bt.set_defaults(func=cmd_backtest)

    brief = sub.add_parser("briefing", help="Stage 1 daily digest only (~2 min read)")
    brief.add_argument("--no-robinhood", action="store_true", help="Skip Robinhood MCP connection")
    brief.add_argument("--no-email", action="store_true", help="Skip email notification")
    brief.set_defaults(func=lambda a: asyncio.run(cmd_briefing(a)))

    run = sub.add_parser("run", help="Full two-stage pipeline: backtest + daily + weekly synthesis")
    run.add_argument("--no-robinhood", action="store_true", help="Skip Robinhood MCP connection")
    run.add_argument("--no-email", action="store_true", help="Skip email notification")
    run.set_defaults(func=lambda a: asyncio.run(cmd_run(a)))

    rh = sub.add_parser("test-robinhood", help="Test Robinhood MCP connection (opens browser for OAuth)")
    rh.set_defaults(func=lambda a: asyncio.run(cmd_test_robinhood(a)))

    em = sub.add_parser("test-email", help="Send a test notification email")
    em.set_defaults(func=cmd_test_email)

    cmp = sub.add_parser("compare-strategies", help="Compare all rule strategies vs SPY and pick best")
    cmp.add_argument(
        "--with-kronos",
        action="store_true",
        help="Include kronos_top_k (slow; downloads model on first run)",
    )
    cmp.set_defaults(func=cmd_compare_strategies)

    kf = sub.add_parser("kronos-forecast", help="Kronos OHLCV forecast rankings for watchlist")
    kf.set_defaults(func=cmd_kronos_forecast)

    ens = sub.add_parser(
        "ensemble-score",
        help="Kronos + HMM + (Bayesian CP or TFT) weighted ensemble",
    )
    ens.set_defaults(func=cmd_ensemble_score)

    boss = sub.add_parser("boss-score", help="Boss agent + three leg agents (live scores)")
    boss.set_defaults(func=cmd_boss_score)

    train = sub.add_parser(
        "train-boss",
        help="Train boss weights on historical walk-forward practice",
    )
    train.set_defaults(func=cmd_train_boss)

    roi = sub.add_parser("tune-roi", help="Retrain boss for max ROI and tune hyperparameters")
    roi.set_defaults(func=cmd_tune_roi)

    bear = sub.add_parser("paper-bear-test", help="Paper-test bear market defense until good ROI")
    bear.add_argument("--target", type=float, default=5.0, help="Target bear-period ROI %% (default 5)")
    bear.add_argument("--rounds", type=int, default=5, help="Max tuning rounds")
    bear.set_defaults(func=cmd_tune_bear)

    paper = sub.add_parser("paper-run", help="Paper portfolio — boss practices without real orders")
    paper.add_argument(
        "--learn",
        action="store_true",
        help="Nudge boss weights from paper outcomes",
    )
    paper.add_argument(
        "--historical",
        action="store_true",
        help="Replay past rebalance dates instead of today's live quotes",
    )
    paper.set_defaults(func=cmd_paper_run)

    trade = sub.add_parser("trade", help="Auto-trader — dry-run by default (boss when engine: boss)")
    trade.add_argument(
        "--execute",
        action="store_true",
        help="Place live orders (requires mode: auto_execute and force_analyze_only: false)",
    )
    trade.set_defaults(func=lambda a: asyncio.run(cmd_trade(a)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
