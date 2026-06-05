"""Bear-market detection and defensive executive rules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent.boss.agent import BossDecision
from models.market_strategist import analyze_ticker
from models.signal_classifier import classify_signal, historical_news_proxy, quick_rsi


def detect_market_stress(
    history: dict[str, pd.DataFrame],
    benchmark: str = "XBI",
) -> dict[str, Any]:
    bench_key = benchmark.upper()
    df = history.get(bench_key)
    if df is None:
        df = history.get(benchmark)
    if df is None or df.empty or len(df) < 60:
        return {"is_bear": False, "severity": 0.0, "label": "UNKNOWN", "return_60d_pct": 0.0}

    close = df["Close"].astype(float)
    last = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else last
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
    ret_60 = float(close.pct_change(60).iloc[-1]) * 100 if len(close) > 60 else 0.0
    ret_20 = float(close.pct_change(20).iloc[-1]) * 100 if len(close) > 20 else 0.0

    severity = 0.0
    if last < sma200:
        severity += 0.35
    if last < sma50:
        severity += 0.25
    if ret_60 < -8:
        severity += 0.25
    if ret_20 < -5:
        severity += 0.15
    severity = min(1.0, severity)

    is_bear = severity >= 0.35 or (last < sma200 and ret_60 < 0)

    if severity >= 0.6:
        label = "SEVERE_BEAR"
    elif is_bear:
        label = "BEAR"
    elif ret_20 > 5 and last > sma50:
        label = "BULL"
    else:
        label = "NEUTRAL"

    return {
        "is_bear": is_bear,
        "severity": round(severity, 2),
        "label": label,
        "return_60d_pct": round(ret_60, 2),
        "return_20d_pct": round(ret_20, 2),
        "benchmark": bench_key,
    }


def apply_bear_defense(
    decision: BossDecision,
    history: dict[str, pd.DataFrame],
    agent_cfg: Any,
    *,
    live_news: bool = True,
) -> BossDecision:
    """Adjust picks, stops, and cash rules when the market is under stress."""
    bear_cfg = getattr(agent_cfg, "bear_mode", None)
    if bear_cfg is None or not getattr(bear_cfg, "enabled", True):
        return decision

    benchmark = agent_cfg.pharma.benchmark_ticker if getattr(agent_cfg, "pharma", None) else "XBI"
    stress = detect_market_stress(history, benchmark)
    pharma = decision.pharma_report or {}
    news_scores = pharma.get("news_scores") or {}

    executive = dict(decision.executive or {})
    executive["market_stress"] = stress

    if not decision.combined_scores:
        decision.executive = executive
        return decision

    scores = (decision.executive or {}).get("final_scores") or decision.combined_scores
    ranked = sorted(
        scores.items(),
        key=lambda x: -float(x[1]),
    )

    if stress["is_bear"]:
        bear_cfg = getattr(agent_cfg, "bear_mode", None)
        picked = _select_bear_candidate(
            ranked,
            decision,
            history,
            pharma,
            stress,
            bear_cfg,
            executive,
            live_news=live_news,
        )
        if picked is None:
            decision.target_ticker = None
            decision.rationale = (
                f"BEAR DEFENSE ({stress['label']}): stay in CASH — no ticker passed "
                f"news/metrics filters during downtrend."
            )
            decision.executive = executive
            return decision
        best, kind, kind_note, news_score, strat = picked
        decision.target_ticker = best
        executive["signal_kind"] = kind
        executive["signal_note"] = kind_note
        executive["news_score"] = round(news_score, 3)
        decision.rationale = (
            f"BEAR MODE ({stress['label']}): Executive BUY {best} "
            f"[{kind}: {kind_note}; news {news_score:+.2f}. Tighter stop / faster targets.]"
        )
        if strat and strat.trade_plan:
            tp = strat.trade_plan
            executive["trade_plan"] = {
                "ticker": tp.ticker,
                "entry": tp.entry_price,
                "stop_loss": tp.stop_loss,
                "take_profit_1": tp.take_profit_1,
                "take_profit_2": tp.take_profit_2,
                "take_profit_3": tp.take_profit_3,
                "risk_per_share": tp.risk_per_share,
                "last_price": tp.last_price,
            }
            _adjust_stops_for_regime(executive, stress, kind, bear_cfg)
        decision.executive = executive
        return decision

    best = decision.target_ticker
    if not best:
        decision.executive = executive
        return decision

    df = _get_df(history, best)
    strat = analyze_ticker(best, df) if df is not None else None
    news_score = float(news_scores.get(best, 0.0))
    if not live_news and df is not None:
        news_score = historical_news_proxy(df)

    leg_score = float(decision.combined_scores.get(best, 0))
    strat_score = strat.score if strat else 0.0

    kind, kind_note = classify_signal(
        best,
        leg_reports=decision.leg_reports,
        news_score=news_score,
        leg_combined_score=leg_score,
        strategist_score=strat_score,
    )
    executive["signal_kind"] = kind
    executive["signal_note"] = kind_note
    executive["news_score"] = round(news_score, 3)
    decision.rationale = f"{decision.rationale} [{kind}: {kind_note}]"

    decision.executive = executive
    if strat and decision.target_ticker and executive.get("trade_plan"):
        _adjust_stops_for_regime(executive, stress, kind, bear_cfg)
        decision.executive = executive

    return decision


def _passes_bear_filter(
    kind: str,
    news_score: float,
    rsi: float,
    stress: dict,
    bear_cfg: Any,
) -> bool:
    min_news = getattr(bear_cfg, "min_news_score", 0.12)
    block_metrics = getattr(bear_cfg, "block_metrics_only", True)
    oversold = getattr(bear_cfg, "oversold_rsi", 32.0)

    if news_score <= -0.12 or kind == "WEAK":
        return False
    if block_metrics and kind == "METRICS_ONLY" and rsi > oversold:
        return False
    if stress["severity"] >= 0.6 and kind not in ("NEWS_CATALYST", "MIXED") and news_score < min_news:
        return False
    return True


def _select_bear_candidate(
    ranked: list[tuple[str, float]],
    decision: BossDecision,
    history: dict,
    pharma: dict,
    stress: dict,
    bear_cfg: Any,
    executive: dict,
    *,
    live_news: bool,
):
    news_scores = pharma.get("news_scores") or {}
    for ticker, _ in ranked:
        t = ticker.upper()
        df = _get_df(history, t)
        strat = analyze_ticker(t, df) if df is not None else None
        news_score = float(news_scores.get(t, 0.0))
        if not live_news and df is not None:
            news_score = historical_news_proxy(df)
        leg_score = float(decision.combined_scores.get(t, 0))
        strat_score = strat.score if strat else 0.0
        rsi = strat.rsi if strat else quick_rsi(df)
        kind, kind_note = classify_signal(
            t,
            leg_reports=decision.leg_reports,
            news_score=news_score,
            leg_combined_score=leg_score,
            strategist_score=strat_score,
        )
        if _passes_bear_filter(kind, news_score, rsi, stress, bear_cfg):
            return t, kind, kind_note, news_score, strat
    return None


def _adjust_stops_for_regime(executive: dict, stress: dict, kind: str, bear_cfg: Any) -> None:
    tp = executive.get("trade_plan")
    if not tp:
        return
    entry = float(tp["entry"])
    stop = float(tp["stop_loss"])
    risk = max(entry - stop, 0.01)

    if stress.get("is_bear"):
        atr_tight = getattr(bear_cfg, "stop_tighten_pct", 0.20)
        stop = round(entry - risk * (1 - atr_tight), 2)
        risk = max(entry - stop, 0.01)
        tp["stop_loss"] = stop
        tp["take_profit_1"] = round(entry + risk * getattr(bear_cfg, "tp1_r", 0.75), 2)
        tp["take_profit_2"] = round(entry + risk * getattr(bear_cfg, "tp2_r", 1.5), 2)
        tp["take_profit_3"] = round(entry + risk * getattr(bear_cfg, "tp3_r", 2.0), 2)
        tp["risk_per_share"] = round(risk, 2)
        tp["bear_adjusted"] = True
    elif kind == "NEWS_CATALYST":
        tp["take_profit_1"] = round(entry + risk * 0.8, 2)
        tp["take_profit_2"] = round(entry + risk * 1.6, 2)
        tp["news_fast_exit"] = True


def _get_df(history: dict[str, pd.DataFrame], ticker: str):
    t = ticker.upper()
    if t in history:
        return history[t]
    for k, v in history.items():
        if k.upper() == t:
            return v
    return None
