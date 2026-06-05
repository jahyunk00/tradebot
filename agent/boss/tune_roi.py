"""Tune boss + leg settings for maximum historical ROI (fast cached sweep)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agent.boss.agent import (
    LEG_IDS,
    BossWeights,
    rank_normalize,
    save_boss_weights,
)
from agent.config import load_config
from agent.legs.runner import run_all_legs
from data.market_data import fetch_history


def _build_samples(agent_cfg: Any) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tr = agent_cfg.boss.training
    history = fetch_history(agent_cfg.watchlist, tr.lookback_days)
    closes = pd.DataFrame({t: history[t]["Close"] for t in agent_cfg.watchlist if t in history})
    min_rows = max(252, tr.warmup_days) + 10
    rebalance = tr.rebalance_days
    forward = tr.forward_days

    groups: list[dict[str, Any]] = []
    for i, idx in enumerate(closes.index):
        if i < min_rows or i % rebalance != 0:
            continue
        if i + forward >= len(closes.index):
            break
        slice_hist = {
            t: history[t].loc[:idx]
            for t in agent_cfg.watchlist
            if t in history and len(history[t].loc[:idx]) >= min_rows
        }
        if len(slice_hist) < 2:
            continue
        reports = run_all_legs(slice_hist, agent_cfg, live=False)
        leg_norm = {
            r.agent_id: rank_normalize(r.scores)
            for r in reports
            if r.available and r.scores
        }
        if not leg_norm:
            continue
        fwd_idx = closes.index[i + forward]
        fwd_returns = {}
        for ticker in closes.columns:
            p0 = float(closes.loc[idx, ticker])
            p1 = float(closes.loc[fwd_idx, ticker])
            if p0 > 0:
                fwd_returns[ticker.upper()] = (p1 / p0 - 1) * 100
        if len(fwd_returns) >= 2:
            groups.append({"leg_norm": leg_norm, "fwd_returns": fwd_returns})
    return closes, groups


def _roi_for_weights(
    groups: list[dict[str, Any]],
    w: np.ndarray,
    *,
    min_score: float = 0.0,
) -> float:
    rets: list[float] = []
    for group in groups:
        leg_norm = group["leg_norm"]
        tickers = sorted(set().union(*[set(m.keys()) for m in leg_norm.values()]))
        combined: dict[str, float] = {}
        for ticker in tickers:
            score = sum(
                w[LEG_IDS.index(leg)] * leg_norm[leg].get(ticker, 0)
                for leg in leg_norm
                if leg in LEG_IDS
            )
            combined[ticker] = score
        if not combined:
            rets.append(0.0)
            continue
        best, best_score = max(combined.items(), key=lambda x: x[1])
        if best_score < min_score:
            rets.append(0.0)
        else:
            rets.append(group["fwd_returns"].get(best, 0.0) / 100)
    if not rets:
        return 0.0
    return float(np.prod(1 + np.array(rets)) - 1) * 100


def _train_boss_roi(groups: list[dict[str, Any]], rebalance: int) -> BossWeights:
    from scipy.optimize import minimize

    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def objective(logits: np.ndarray) -> float:
        return -_roi_for_weights(groups, _softmax(logits)) / 100

    x0 = np.log(np.array([0.34, 0.33, 0.33]))
    result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 300})
    w = _softmax(result.x)
    roi = _roi_for_weights(groups, w)

    arr = []
    for group in groups:
        leg_norm = group["leg_norm"]
        tickers = sorted(set().union(*[set(m.keys()) for m in leg_norm.values()]))
        combined = {}
        for ticker in tickers:
            combined[ticker] = sum(
                w[LEG_IDS.index(leg)] * leg_norm[leg].get(ticker, 0)
                for leg in leg_norm
                if leg in LEG_IDS
            )
        if combined:
            pick = max(combined, key=combined.get)
            arr.append(group["fwd_returns"].get(pick, 0.0) / 100)

    sharpe = 0.0
    if len(arr) > 1 and np.std(arr) > 1e-9:
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252 / rebalance))

    return BossWeights(
        kronos=round(float(w[0]), 4),
        hmm=round(float(w[1]), 4),
        third_leg=round(float(w[2]), 4),
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_sharpe=round(sharpe, 3),
        train_samples=len(groups),
        source="roi_optimized",
    )


def _apply_tuned_config(base_dir: Path, params: dict[str, Any], weights: BossWeights) -> None:
    cfg_path = base_dir / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    raw.setdefault("boss", {})["min_combined_score"] = params["min_combined_score"]
    raw.setdefault("ensemble", {})["min_combined_score"] = params["min_combined_score"]
    raw["ensemble"]["rebalance_days"] = params["rebalance_days"]
    raw.setdefault("boss", {}).setdefault("training", {})["rebalance_days"] = params["rebalance_days"]
    raw.setdefault("pharma", {})["small_cap_boost"] = params["small_cap_boost"]
    raw["pharma"]["news_weight"] = 0.10
    raw.setdefault("ensemble", {}).setdefault("weights", {})
    raw["ensemble"]["weights"] = {
        "kronos": weights.kronos,
        "hmm": weights.hmm,
        "third_leg": weights.third_leg,
    }
    cfg_path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))


def tune_for_roi(base_dir: Path) -> dict[str, Any]:
    agent_cfg, _ = load_config(base_dir)
    _, groups = _build_samples(agent_cfg)
    if len(groups) < 12:
        raise ValueError(f"Need more history samples, got {len(groups)}")

    rebalance = agent_cfg.boss.training.rebalance_days
    weights = _train_boss_roi(groups, rebalance)
    save_boss_weights(base_dir / agent_cfg.boss.weights_path, weights)
    w = np.array([weights.kronos, weights.hmm, weights.third_leg])

    best_roi = -999.0
    best_params: dict[str, Any] = {}
    for min_score in (0.0, 0.10, 0.15, 0.20):
        roi = _roi_for_weights(groups, w, min_score=min_score)
        if roi > best_roi:
            best_roi = roi
            best_params = {
                "min_combined_score": min_score,
                "rebalance_days": rebalance,
                "small_cap_boost": agent_cfg.pharma.small_cap_boost,
            }

    _apply_tuned_config(base_dir, best_params, weights)

    out = {
        "best_params": best_params,
        "simulated_roi_pct": round(best_roi, 2),
        "boss_weights": weights.as_dict(),
        "train_sharpe": weights.train_sharpe,
    }
    (base_dir / "logs" / "roi_tune_result.json").write_text(json.dumps(out, indent=2))
    return out
