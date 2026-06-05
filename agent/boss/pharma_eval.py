"""Evaluate how boss training performed on pharma watchlist vs XBI."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.boss.agent import BossWeights, combine_leg_reports
from agent.legs.runner import run_all_legs
from data.market_data import fetch_history


def evaluate_pharma_training(
    agent_cfg: Any,
    weights: BossWeights,
    *,
    base_dir: Path,
) -> dict[str, Any]:
    """
    Walk-forward simulation on pharma watchlist using learned boss weights.
    Compares boss picks vs biotech benchmark (XBI).
    """
    tr = agent_cfg.boss.training
    ph = agent_cfg.pharma
    watchlist = agent_cfg.watchlist
    benchmark = ph.benchmark_ticker

    history = fetch_history([*watchlist, benchmark], tr.lookback_days)
    if benchmark.upper() not in history:
        history = fetch_history(watchlist, tr.lookback_days)

    closes = pd.DataFrame({t: history[t]["Close"] for t in watchlist if t in history})
    bench = history.get(benchmark.upper(), history.get(benchmark))
    bench_close = bench["Close"] if bench is not None and not bench.empty else None

    min_rows = max(252, tr.warmup_days) + 10
    rebalance = tr.rebalance_days
    forward = tr.forward_days

    period_returns: list[float] = []
    bench_returns: list[float] = []
    picks: list[dict[str, Any]] = []

    for i, idx in enumerate(closes.index):
        if i < min_rows or i % rebalance != 0:
            continue
        if i + forward >= len(closes.index):
            break

        slice_hist = {
            t: history[t].loc[:idx]
            for t in watchlist
            if t in history and len(history[t].loc[:idx]) >= min_rows
        }
        if len(slice_hist) < 2:
            continue

        reports = run_all_legs(slice_hist, agent_cfg, live=False)
        decision = combine_leg_reports(
            reports,
            weights,
            min_combined_score=agent_cfg.boss.min_combined_score,
        )

        if agent_cfg.pharma.enabled and decision.combined_scores:
            from data.pharma_intel import apply_pharma_overlay

            adjusted, _ = apply_pharma_overlay(
                decision.combined_scores,
                watchlist,
                max_market_cap_b=ph.max_market_cap_b,
                small_cap_boost=ph.small_cap_boost,
                news_weight=0.0,  # historical eval = price legs only (no live news)
                news_headlines=ph.news_headlines,
            )
            if adjusted:
                pick = max(adjusted.items(), key=lambda x: x[1])[0]
            else:
                pick = decision.target_ticker
        else:
            pick = decision.target_ticker

        if not pick:
            period_returns.append(0.0)
            if bench_close is not None:
                fwd_idx = closes.index[i + forward]
                b0 = float(bench_close.loc[idx])
                b1 = float(bench_close.loc[fwd_idx])
                bench_returns.append((b1 / b0 - 1) if b0 > 0 else 0.0)
            picks.append({"date": str(idx.date()), "pick": None, "return_pct": 0.0})
            continue

        pick = pick.upper()
        fwd_idx = closes.index[i + forward]
        p0 = float(closes.loc[idx, pick])
        p1 = float(closes.loc[fwd_idx, pick])
        ret = (p1 / p0 - 1) * 100 if p0 > 0 else 0.0
        period_returns.append(ret / 100)

        if bench_close is not None:
            b0 = float(bench_close.loc[idx])
            b1 = float(bench_close.loc[fwd_idx])
            bench_returns.append((b1 / b0 - 1) if b0 > 0 else 0.0)

        picks.append({"date": str(idx.date()), "pick": pick, "return_pct": round(ret, 2)})

    strat_arr = np.array(period_returns)
    bench_arr = np.array(bench_returns) if bench_returns else np.array([0.0])

    strat_total = float(np.prod(1 + strat_arr) - 1) * 100 if len(strat_arr) else 0.0
    bench_total = float(np.prod(1 + bench_arr) - 1) * 100 if len(bench_arr) else 0.0

    sharpe = 0.0
    if len(strat_arr) > 1 and strat_arr.std() > 1e-9:
        sharpe = float(strat_arr.mean() / strat_arr.std() * np.sqrt(252 / rebalance))

    win_rate = 0.0
    traded = [p for p in picks if p.get("pick")]
    if traded:
        win_rate = sum(1 for p in traded if p["return_pct"] > 0) / len(traded) * 100

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "watchlist": watchlist,
        "benchmark": benchmark.upper(),
        "periods": len(picks),
        "strategy_total_return_pct": round(strat_total, 2),
        "benchmark_total_return_pct": round(bench_total, 2),
        "vs_benchmark_pct": round(strat_total - bench_total, 2),
        "simulated_sharpe": round(sharpe, 3),
        "win_rate_pct": round(win_rate, 1),
        "learned_weights": weights.as_dict(),
        "recent_picks": picks[-15:],
        "implementation_notes": (
            "Boss applies learned kronos/hmm/third_leg weights on pharma price history. "
            "Live picks also blend pharma news sentiment and favor smaller market caps "
            f"(under ${ph.max_market_cap_b}B)."
        ),
    }

    out = base_dir / agent_cfg.logging.directory / "pharma_train_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    return report
