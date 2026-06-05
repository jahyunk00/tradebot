"""Boss agent — learns how to weight the three leg agents."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.legs.base import LegReport, rank_normalize
from agent.legs.runner import run_all_legs
from data.market_data import fetch_history

logger = logging.getLogger(__name__)

LEG_IDS = ("kronos", "hmm", "third_leg")


@dataclass
class BossWeights:
    kronos: float = 0.34
    hmm: float = 0.33
    third_leg: float = 0.33
    trained_at: str = ""
    train_sharpe: float = 0.0
    train_samples: int = 0
    source: str = "config_default"

    def as_dict(self) -> dict[str, float]:
        return {"kronos": self.kronos, "hmm": self.hmm, "third_leg": self.third_leg}

    def normalized(self, available: dict[str, bool]) -> dict[str, float]:
        raw = {
            "kronos": self.kronos if available.get("kronos") else 0.0,
            "hmm": self.hmm if available.get("hmm") else 0.0,
            "third_leg": self.third_leg if available.get("third_leg") else 0.0,
        }
        total = sum(raw.values())
        if total <= 0:
            active = [k for k, v in available.items() if v]
            if not active:
                return {"hmm": 1.0}
            eq = 1.0 / len(active)
            return {k: eq for k in active}
        return {k: v / total for k, v in raw.items()}


@dataclass
class BossDecision:
    target_ticker: str | None
    combined_scores: dict[str, float] = field(default_factory=dict)
    leg_reports: list[dict[str, Any]] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)
    min_score: float = 0.30
    rationale: str = ""
    pharma_report: dict[str, Any] | None = None
    executive: dict[str, Any] | None = None


def weights_from_config(agent_cfg: Any) -> BossWeights:
    w = agent_cfg.ensemble.weights
    return BossWeights(
        kronos=w.kronos,
        hmm=w.hmm,
        third_leg=w.third_leg,
        source="config_default",
    )


def load_boss_weights(path: Path, fallback: BossWeights | None = None) -> BossWeights:
    if path.exists():
        data = json.loads(path.read_text())
        return BossWeights(
            kronos=float(data.get("kronos", 0.34)),
            hmm=float(data.get("hmm", 0.33)),
            third_leg=float(data.get("third_leg", 0.33)),
            trained_at=data.get("trained_at", ""),
            train_sharpe=float(data.get("train_sharpe", 0)),
            train_samples=int(data.get("train_samples", 0)),
            source=data.get("source", "learned"),
        )
    return fallback or BossWeights()


def save_boss_weights(path: Path, weights: BossWeights) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(weights), indent=2))


def combine_leg_reports(
    reports: list[LegReport],
    weights: BossWeights,
    *,
    min_combined_score: float = 0.30,
) -> BossDecision:
    available = {r.agent_id: r.available and bool(r.scores) for r in reports}
    w = weights.normalized(available)

    leg_norm: dict[str, dict[str, float]] = {}
    for r in reports:
        if r.available and r.scores:
            leg_norm[r.agent_id] = rank_normalize(r.scores)

    tickers = sorted(set().union(*[set(d.keys()) for d in leg_norm.values()]))
    combined: dict[str, float] = {}
    for ticker in tickers:
        score = sum(w.get(leg, 0) * leg_norm[leg].get(ticker, 0) for leg in leg_norm)
        combined[ticker] = round(score, 4)

    leg_payload = [
        {
            "agent_id": r.agent_id,
            "available": r.available,
            "top_ticker": r.top_ticker,
            "scores": r.scores,
            "note": r.note,
        }
        for r in reports
    ]

    if not combined:
        return BossDecision(
            target_ticker=None,
            leg_reports=leg_payload,
            weights_used=w,
            min_score=min_combined_score,
            rationale="Boss: no leg agents produced scores — stay in cash.",
        )

    best, best_score = max(combined.items(), key=lambda x: x[1])
    if best_score < min_combined_score:
        return BossDecision(
            target_ticker=None,
            combined_scores=combined,
            leg_reports=leg_payload,
            weights_used=w,
            min_score=min_combined_score,
            rationale=(
                f"Boss: best {best} score {best_score:.2f} below threshold "
                f"{min_combined_score:.2f} — cash."
            ),
        )

    parts = []
    for r in reports:
        if not r.available or not r.scores:
            continue
        raw = r.scores.get(best)
        norm = leg_norm.get(r.agent_id, {}).get(best)
        wt = w.get(r.agent_id, 0)
        if raw is not None and norm is not None:
            parts.append(f"{r.agent_id}={raw:+.4f}→{norm:.2f}×{wt:.0%}")

    rationale = (
        f"Boss picks {best} (score {best_score:.2f}). "
        f"Learned weights: kronos {w.get('kronos', 0):.0%}, "
        f"hmm {w.get('hmm', 0):.0%}, third_leg {w.get('third_leg', 0):.0%}. "
        f"{' | '.join(parts)}"
    )
    return BossDecision(
        target_ticker=best,
        combined_scores=combined,
        leg_reports=leg_payload,
        weights_used=w,
        min_score=min_combined_score,
        rationale=rationale,
    )


def decide_from_history(
    history: dict[str, pd.DataFrame],
    agent_cfg: Any,
    weights: BossWeights,
) -> BossDecision:
    reports = run_all_legs(history, agent_cfg, live=True)
    decision = combine_leg_reports(
        reports,
        weights,
        min_combined_score=agent_cfg.boss.min_combined_score,
    )

    if getattr(agent_cfg, "pharma", None) and agent_cfg.pharma.enabled and decision.combined_scores:
        from data.pharma_intel import apply_pharma_overlay

        ph = agent_cfg.pharma
        adjusted, pharma_report = apply_pharma_overlay(
            decision.combined_scores,
            agent_cfg.watchlist,
            max_market_cap_b=ph.max_market_cap_b,
            small_cap_boost=ph.small_cap_boost,
            news_weight=ph.news_weight,
            news_headlines=ph.news_headlines,
        )
        decision.combined_scores = adjusted
        decision.pharma_report = {
            "sector_trend": pharma_report.sector_trend,
            "news_scores": pharma_report.news_scores,
            "market_cap_b": pharma_report.market_cap_b,
            "small_cap_boost": pharma_report.small_cap_boost,
            "trending_up": pharma_report.trending_up,
            "trending_down": pharma_report.trending_down,
        }

    from agent.boss.bear_market import apply_bear_defense
    from agent.boss.executive import apply_executive_decision

    decision = apply_executive_decision(decision, history, agent_cfg)
    return apply_bear_defense(decision, history, agent_cfg, live_news=True)


def train_boss_weights(
    agent_cfg: Any,
    *,
    base_dir: Path,
) -> BossWeights:
    """
    Practice on past market data: walk-forward rebalance dates,
    score each leg, optimize boss weights for pick Sharpe ratio.
    """
    from scipy.optimize import minimize

    tr = agent_cfg.boss.training
    watchlist = agent_cfg.watchlist
    history = fetch_history(watchlist, tr.lookback_days)

    closes = pd.DataFrame({t: history[t]["Close"] for t in watchlist if t in history})
    if closes.empty or len(closes) < tr.min_train_samples + 50:
        raise ValueError("Not enough history to train boss weights.")

    min_rows = max(252, tr.warmup_days) + 10
    rebalance = tr.rebalance_days
    forward = tr.forward_days

    # Collect training samples per rebalance date
    rebalance_groups: list[dict[str, Any]] = []

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
        leg_norm = {
            r.agent_id: rank_normalize(r.scores)
            for r in reports
            if r.available and r.scores
        }
        if not leg_norm:
            continue

        fwd_idx = closes.index[i + forward]
        fwd_returns: dict[str, float] = {}
        for ticker in closes.columns:
            p0 = float(closes.loc[idx, ticker])
            p1 = float(closes.loc[fwd_idx, ticker])
            if p0 <= 0:
                continue
            fwd_returns[ticker.upper()] = (p1 / p0 - 1) * 100

        if len(fwd_returns) < 2:
            continue
        rebalance_groups.append({"leg_norm": leg_norm, "fwd_returns": fwd_returns})

    if len(rebalance_groups) < tr.min_train_samples:
        raise ValueError(
            f"Only {len(rebalance_groups)} rebalance periods; need {tr.min_train_samples}."
        )

    def _combined_scores(logits: np.ndarray, leg_norm: dict[str, dict[str, float]]) -> dict[str, float]:
        w = _softmax(logits)
        tickers = sorted(set().union(*[set(m.keys()) for m in leg_norm.values()]))
        combined: dict[str, float] = {}
        for ticker in tickers:
            score = 0.0
            for leg, norm_map in leg_norm.items():
                if leg not in LEG_IDS:
                    continue
                score += w[LEG_IDS.index(leg)] * norm_map.get(ticker, 0)
            combined[ticker] = score
        return combined

    def objective(logits: np.ndarray) -> float:
        period_returns: list[float] = []
        for group in rebalance_groups:
            combined = _combined_scores(logits, group["leg_norm"])
            if not combined:
                period_returns.append(0.0)
                continue
            pick = max(combined.items(), key=lambda x: x[1])[0]
            period_returns.append(group["fwd_returns"].get(pick, 0.0) / 100)

        arr = np.array(period_returns)
        if len(arr) < 2 or arr.std() < 1e-9:
            return 1e6
        sharpe = float(arr.mean() / arr.std() * np.sqrt(252 / rebalance))
        return -sharpe

    x0 = np.log(np.array([0.34, 0.33, 0.33]))
    result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 400})
    w = _softmax(result.x)
    arr = []
    for group in rebalance_groups:
        combined = _combined_scores(result.x, group["leg_norm"])
        if not combined:
            continue
        pick = max(combined, key=combined.get)
        arr.append(group["fwd_returns"].get(pick, 0.0) / 100)

    train_sharpe = 0.0
    if len(arr) > 1 and np.std(arr) > 1e-9:
        train_sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252 / rebalance))

    trained = BossWeights(
        kronos=round(float(w[0]), 4),
        hmm=round(float(w[1]), 4),
        third_leg=round(float(w[2]), 4),
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_sharpe=round(train_sharpe, 3),
        train_samples=len(rebalance_groups),
        source="historical_practice",
    )
    out_path = base_dir / agent_cfg.boss.weights_path
    save_boss_weights(out_path, trained)
    logger.info("Boss weights saved to %s (Sharpe %.2f)", out_path, train_sharpe)

    if getattr(agent_cfg, "pharma", None) and agent_cfg.pharma.enabled:
        from agent.boss.pharma_eval import evaluate_pharma_training

        evaluate_pharma_training(agent_cfg, trained, base_dir=base_dir)
        logger.info("Pharma training report saved to logs/pharma_train_report.json")

    return trained


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def update_weights_from_paper_outcomes(
    weights: BossWeights,
    outcomes: list[dict[str, Any]],
    *,
    learning_rate: float = 0.05,
) -> BossWeights:
    """
    Online nudge after paper runs: reward legs whose top pick matched profitable trades.
    """
    if not outcomes:
        return weights

    rewards = {"kronos": 0.0, "hmm": 0.0, "third_leg": 0.0}
    for row in outcomes:
        pnl = float(row.get("pnl_pct", 0))
        if pnl == 0:
            continue
        for leg in row.get("leg_tops", {}):
            if row.get("leg_tops", {}).get(leg) == row.get("pick"):
                rewards[leg] += pnl

    w = np.array([weights.kronos, weights.hmm, weights.third_leg])
    r = np.array([rewards["kronos"], rewards["hmm"], rewards["third_leg"]])
    if np.abs(r).sum() < 1e-9:
        return weights

    r = r / (np.abs(r).sum() + 1e-9)
    w = w + learning_rate * r
    w = np.maximum(w, 0.05)
    w = w / w.sum()

    return BossWeights(
        kronos=round(float(w[0]), 4),
        hmm=round(float(w[1]), 4),
        third_leg=round(float(w[2]), 4),
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_sharpe=weights.train_sharpe,
        train_samples=weights.train_samples + len(outcomes),
        source="paper_run_learning",
    )
