"""Boss executive layer — merges leg agents + senior strategist TA."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent.boss.agent import BossDecision
from models.market_strategist import StrategistReport, analyze_ticker


def apply_executive_decision(
    decision: BossDecision,
    history: dict[str, pd.DataFrame],
    agent_cfg: Any,
    *,
    top_n: int = 5,
) -> BossDecision:
    """
    Senior strategist review of top leg candidates:
    MACD, RSI, regime, price structure, trend integrity, volume → entry/stops/targets.
    """
    if not decision.combined_scores:
        return decision

    exec_w = getattr(agent_cfg.boss, "strategist_weight", 0.40)
    leg_w = 1.0 - exec_w
    min_score = agent_cfg.boss.min_combined_score

    ranked = sorted(decision.combined_scores.items(), key=lambda x: -x[1])[:top_n]
    strategist_reports: dict[str, StrategistReport] = {}
    final_scores: dict[str, float] = {}

    for ticker, leg_score in ranked:
        t = ticker.upper()
        df = history.get(t)
        if df is None:
            for k, v in history.items():
                if k.upper() == t:
                    df = v
                    break
        if df is None or df.empty:
            final_scores[t] = leg_score * leg_w
            continue
        report = analyze_ticker(t, df)
        if not report:
            final_scores[t] = leg_score * leg_w
            continue
        strategist_reports[t] = report
        # Penalize hard flags
        strat_score = report.score
        if report.flags:
            strat_score *= 0.85
        final_scores[t] = round(leg_w * leg_score + exec_w * strat_score, 4)

    if not final_scores:
        return decision

    best, best_score = max(final_scores.items(), key=lambda x: x[1])
    best_report = strategist_reports.get(best)

    if best_score < min_score:
        return BossDecision(
            target_ticker=None,
            combined_scores=final_scores,
            leg_reports=decision.leg_reports,
            weights_used=decision.weights_used,
            min_score=min_score,
            rationale=_executive_rationale_none(best, best_score, best_report, min_score),
            pharma_report=decision.pharma_report,
            executive=_pack_executive(strategist_reports, final_scores, None),
        )

    rationale = _executive_rationale_pick(
        best, best_score, best_report, decision.weights_used, leg_w, exec_w
    )

    return BossDecision(
        target_ticker=best,
        combined_scores=final_scores,
        leg_reports=decision.leg_reports,
        weights_used=decision.weights_used,
        min_score=min_score,
        rationale=rationale,
        pharma_report=decision.pharma_report,
        executive=_pack_executive(strategist_reports, final_scores, best_report),
    )


def _pack_executive(
    reports: dict[str, StrategistReport],
    final_scores: dict[str, float],
    pick: StrategistReport | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "final_scores": final_scores,
        "candidates": {t: r.to_dict() for t, r in reports.items()},
    }
    if pick and pick.trade_plan:
        tp = pick.trade_plan
        payload["trade_plan"] = {
            "ticker": tp.ticker,
            "entry": tp.entry_price,
            "stop_loss": tp.stop_loss,
            "take_profit_1": tp.take_profit_1,
            "take_profit_2": tp.take_profit_2,
            "take_profit_3": tp.take_profit_3,
            "risk_per_share": tp.risk_per_share,
            "last_price": tp.last_price,
        }
        payload["strategist_summary"] = pick.summary
        payload["dominant_trend"] = pick.dominant_trend
        payload["regime"] = pick.regime
        payload["macd"] = pick.macd_signal
        payload["rsi"] = pick.rsi
        payload["structure"] = pick.price_structure
        payload["volume"] = pick.volume_behavior
    return payload


def _executive_rationale_pick(
    ticker: str,
    score: float,
    report: StrategistReport | None,
    weights: dict[str, float],
    leg_w: float,
    exec_w: float,
) -> str:
    if not report:
        return f"Executive: {ticker} (score {score:.2f}) — legs weighted, strategist data unavailable."

    tp = report.trade_plan
    plan = ""
    if tp:
        plan = (
            f" Entry ~${tp.entry_price:.2f} · Stop ${tp.stop_loss:.2f} · "
            f"Targets ${tp.take_profit_1:.2f} / ${tp.take_profit_2:.2f} / ${tp.take_profit_3:.2f}."
        )

    return (
        f"Executive BUY {ticker} (score {score:.2f} = {leg_w:.0%} legs + {exec_w:.0%} strategist). "
        f"Trend {report.dominant_trend}, regime {report.regime}, RSI {report.rsi:.0f}, MACD {report.macd_signal}, "
        f"integrity {report.trend_integrity:.2f}. {report.volume_behavior}.{plan}"
    )


def _executive_rationale_none(
    ticker: str,
    score: float,
    report: StrategistReport | None,
    min_score: float,
) -> str:
    extra = f" Strategist flags: {', '.join(report.flags)}." if report and report.flags else ""
    return (
        f"Executive: stay in CASH — best {ticker} at {score:.2f} below threshold {min_score:.2f}.{extra}"
    )
