"""Paper-test strategy during bear market periods and tune defensive params."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agent.boss.agent import BossWeights, combine_leg_reports, load_boss_weights, weights_from_config
from agent.boss.bear_market import detect_market_stress
from agent.config import load_config
from agent.legs.runner import run_all_legs
from data.market_data import fetch_history
from models.signal_classifier import classify_signal, historical_news_proxy, quick_rsi


@dataclass
class _PeriodRow:
    idx: object
    fwd_idx: object
    in_bear: bool
    stress: dict[str, Any]
    decision_scores: dict[str, float]
    leg_reports: list[dict[str, Any]]
    news_by_ticker: dict[str, float]
    kind_by_ticker: dict[str, str]
    rsi_by_ticker: dict[str, float]


def _build_period_cache(agent_cfg: Any, weights: BossWeights) -> tuple[pd.DataFrame, dict, list[_PeriodRow]]:
    """Run legs once per rebalance date; reuse for all param grids."""
    tr = agent_cfg.boss.training
    benchmark = agent_cfg.pharma.benchmark_ticker
    watchlist = agent_cfg.watchlist
    history = fetch_history([*watchlist, benchmark], tr.lookback_days)

    bench_df = history.get(benchmark.upper())
    if bench_df is None:
        bench_df = history.get(benchmark)
    if bench_df is None or bench_df.empty:
        raise ValueError(f"No benchmark {benchmark} data")

    closes = pd.DataFrame({t: history[t]["Close"] for t in watchlist if t in history})
    min_rows = max(252, tr.warmup_days) + 10
    rebalance = tr.rebalance_days
    forward = tr.forward_days
    rows: list[_PeriodRow] = []

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
        bench_slice = bench_df.loc[:idx]
        stress = detect_market_stress({benchmark.upper(): bench_slice}, benchmark)

        reports = run_all_legs(slice_hist, agent_cfg, live=False)
        decision = combine_leg_reports(reports, weights, min_combined_score=0.0)

        news_by: dict[str, float] = {}
        kind_by: dict[str, str] = {}
        rsi_by: dict[str, float] = {}
        for t in decision.combined_scores:
            pick_df = slice_hist.get(t.upper())
            if pick_df is None:
                pick_df = slice_hist.get(t)
            ns = historical_news_proxy(pick_df)
            rsi = quick_rsi(pick_df)
            kind, _ = classify_signal(
                t,
                leg_reports=decision.leg_reports,
                news_score=ns,
                leg_combined_score=decision.combined_scores[t],
                strategist_score=0.5,
            )
            news_by[t.upper()] = ns
            kind_by[t.upper()] = kind
            rsi_by[t.upper()] = rsi

        rows.append(
            _PeriodRow(
                idx=idx,
                fwd_idx=closes.index[i + forward],
                in_bear=stress["is_bear"],
                stress=stress,
                decision_scores=decision.combined_scores,
                leg_reports=decision.leg_reports,
                news_by_ticker=news_by,
                kind_by_ticker=kind_by,
                rsi_by_ticker=rsi_by,
            )
        )

    return closes, history, rows


def _pick_bear_eligible(
    eligible: dict[str, float],
    row: _PeriodRow,
    *,
    min_news: float,
    block_metrics_only: bool,
    oversold_rsi: float,
) -> tuple[str | None, float, str, float]:
    """Try ranked tickers until one passes bear filters."""
    for ticker, _ in sorted(eligible.items(), key=lambda x: -x[1]):
        pick = ticker.upper()
        news_score = row.news_by_ticker.get(pick, 0.0)
        kind = row.kind_by_ticker.get(pick, "WEAK")
        rsi = row.rsi_by_ticker.get(pick, 50.0)

        if news_score <= -0.12 or kind == "WEAK":
            continue
        if block_metrics_only and kind == "METRICS_ONLY" and rsi > oversold_rsi:
            continue
        if row.stress["severity"] >= 0.6 and kind not in ("NEWS_CATALYST", "MIXED") and news_score < min_news:
            continue
        return pick, news_score, kind, rsi
    return None, 0.0, "WEAK", 50.0


def _simulate_from_cache(
    closes: pd.DataFrame,
    history: dict[str, pd.DataFrame],
    rows: list[_PeriodRow],
    *,
    min_news: float,
    block_metrics_only: bool,
    min_score: float,
    oversold_rsi: float = 32.0,
) -> dict[str, Any]:
    bear_rets: list[float] = []
    all_rets: list[float] = []
    bear_trades = 0
    cash_in_bear = 0
    wins = 0

    bear_bench_rets: list[float] = []

    for row in rows:
        eligible = {t: s for t, s in row.decision_scores.items() if s >= min_score}
        if not eligible:
            if row.in_bear:
                bear_rets.append(0.0)
                bear_bench_rets.append(_bench_ret(closes, history, row))
                cash_in_bear += 1
            continue

        if row.in_bear:
            pick, news_score, kind, rsi = _pick_bear_eligible(
                eligible,
                row,
                min_news=min_news,
                block_metrics_only=block_metrics_only,
                oversold_rsi=oversold_rsi,
            )
            if pick is None:
                bear_rets.append(0.0)
                bear_bench_rets.append(_bench_ret(closes, history, row))
                cash_in_bear += 1
                continue
        else:
            best = max(eligible.items(), key=lambda x: x[1])[0]
            pick = best.upper()
            news_score = row.news_by_ticker.get(pick, 0.0)
            kind = row.kind_by_ticker.get(pick, "WEAK")

        p0 = float(closes.loc[row.idx, pick])
        p1 = float(closes.loc[row.fwd_idx, pick])
        ret = (p1 / p0 - 1) if p0 > 0 else 0.0

        if pick in history:
            path = history[pick].loc[row.idx : row.fwd_idx]["Close"]
            if len(path) > 1:
                prices = path.astype(float)
                min_p = float(prices.min())
                max_p = float(prices.max())
                stop_pct = 0.04 if row.in_bear else 0.08
                stop = p0 * (1 - stop_pct)
                if min_p <= stop:
                    ret = stop / p0 - 1
                elif row.in_bear:
                    # Early cut if first half of window dumps
                    mid = prices.iloc[: max(1, len(prices) // 2)]
                    if float(mid.min()) <= p0 * 0.97:
                        ret = min(ret, -0.03)
                    tp_r = 0.04 if kind == "NEWS_CATALYST" else 0.06
                    if max_p >= p0 * (1 + tp_r):
                        ret = max(ret, tp_r)

        all_rets.append(ret)
        if row.in_bear:
            bear_rets.append(ret)
            bear_bench_rets.append(_bench_ret(closes, history, row))
            bear_trades += 1
            if ret > 0:
                wins += 1

    bear_arr = np.array(bear_rets) if bear_rets else np.array([0.0])
    bear_roi = float(np.prod(1 + bear_arr) - 1) * 100
    bench_arr = np.array(bear_bench_rets) if bear_bench_rets else np.array([0.0])
    bench_roi = float(np.prod(1 + bench_arr) - 1) * 100 if len(bench_arr) else 0.0
    win_rate = (wins / bear_trades * 100) if bear_trades else 0.0

    return {
        "bear_roi_pct": round(bear_roi, 2),
        "bear_bench_roi_pct": round(bench_roi, 2),
        "bear_alpha_pct": round(bear_roi - bench_roi, 2),
        "bear_periods": len(bear_rets),
        "bear_trades": bear_trades,
        "bear_cash_pct": round(cash_in_bear / max(len(bear_rets), 1) * 100, 1),
        "bear_win_rate_pct": round(win_rate, 1),
        "all_roi_pct": round(float(np.prod(1 + np.array(all_rets)) - 1) * 100, 2) if all_rets else 0,
    }


def tune_bear_paper(base_dir: Path, *, target_bear_roi: float = 5.0, max_rounds: int = 5) -> dict[str, Any]:
    """
    Grid-search bear defensive params; apply best to config.yaml.
    Re-runs up to max_rounds with expanding search until bear ROI >= target.
    """
    agent_cfg, _ = load_config(base_dir)
    weights = load_boss_weights(
        base_dir / agent_cfg.boss.weights_path,
        fallback=weights_from_config(agent_cfg),
    )

    print("Building rebalance cache (legs run once per period)...")
    closes, history, rows = _build_period_cache(agent_cfg, weights)
    print(f"Cached {len(rows)} rebalance periods — grid search starting...")

    best_result: dict[str, Any] = {"bear_alpha_pct": -999.0, "bear_roi_pct": -999.0}
    best_params: dict[str, Any] = {}

    news_grid = [0.08, 0.12, 0.18, 0.25, 0.30]
    score_grid = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

    oversold = getattr(agent_cfg.bear_mode, "oversold_rsi", 32.0)
    oversold_grid = [28.0, 32.0, 38.0]

    for round_i in range(max_rounds):
        for min_news in news_grid:
            for block_m in (True, False):
                for min_score in score_grid:
                    for os_rsi in oversold_grid:
                        result = _simulate_from_cache(
                            closes,
                            history,
                            rows,
                            min_news=min_news,
                            block_metrics_only=block_m,
                            min_score=min_score,
                            oversold_rsi=os_rsi,
                        )
                        result["params"] = {
                            "min_news_score": min_news,
                            "block_metrics_only": block_m,
                            "min_combined_score": min_score,
                            "oversold_rsi": os_rsi,
                        }
                        if result["bear_alpha_pct"] > best_result.get("bear_alpha_pct", -999):
                            best_result = result
                            best_params = result["params"]

        if best_result.get("bear_alpha_pct", 0) >= target_bear_roi:
            break
        news_grid = [x for x in news_grid if x >= 0.12] + [0.35]
        score_grid = [x for x in score_grid if x >= 0.10]

    _apply_bear_config(base_dir, best_params)

    out = {
        "tuned_at": datetime.now(timezone.utc).isoformat(),
        "target_bear_roi_pct": target_bear_roi,
        "achieved_bear_roi_pct": best_result.get("bear_roi_pct"),
        "achieved_bear_alpha_pct": best_result.get("bear_alpha_pct"),
        "bear_bench_roi_pct": best_result.get("bear_bench_roi_pct"),
        "met_target": best_result.get("bear_alpha_pct", 0) >= target_bear_roi,
        "best_params": best_params,
        "simulation": best_result,
        "rounds_run": round_i + 1,
        "periods_cached": len(rows),
    }
    (base_dir / "logs" / "bear_paper_tune.json").write_text(json.dumps(out, indent=2))
    return out


def _apply_bear_config(base_dir: Path, params: dict[str, Any]) -> None:
    path = base_dir / "config.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    raw.setdefault("bear_mode", {})
    raw["bear_mode"]["enabled"] = True
    raw["bear_mode"]["min_news_score"] = params.get("min_news_score", 0.12)
    raw["bear_mode"]["block_metrics_only"] = params.get("block_metrics_only", True)
    if "oversold_rsi" in params:
        raw["bear_mode"]["oversold_rsi"] = params["oversold_rsi"]
    if "min_combined_score" in params:
        raw.setdefault("boss", {})["min_combined_score"] = params["min_combined_score"]
        raw.setdefault("ensemble", {})["min_combined_score"] = params["min_combined_score"]
    path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))


def _bench_ret(closes: pd.DataFrame, history: dict[str, pd.DataFrame], row: _PeriodRow) -> float:
    """XBI buy-hold return for same window (from cache benchmark in history)."""
    for key in history:
        if key.upper() in ("XBI", "SPY"):
            df = history[key]
            p0 = float(df.loc[row.idx, "Close"])
            p1 = float(df.loc[row.fwd_idx, "Close"])
            return (p1 / p0 - 1) if p0 > 0 else 0.0
    return 0.0
