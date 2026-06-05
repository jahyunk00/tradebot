"""Classify whether a pick is news-driven, metrics-driven, or mixed."""

from __future__ import annotations

from typing import Any, Literal

SignalKind = Literal["NEWS_CATALYST", "METRICS_ONLY", "MIXED", "WEAK", "DEFENSIVE_CASH"]


def metrics_strength_for_ticker(
    ticker: str,
    leg_reports: list[dict[str, Any]],
) -> float:
    """0–1 strength from kronos / hmm / third_leg raw scores."""
    t = ticker.upper()
    vals: list[float] = []
    for leg in leg_reports:
        if not leg.get("available"):
            continue
        raw = (leg.get("scores") or {}).get(t)
        if raw is None:
            continue
        vals.append(float(raw))
    if not vals:
        return 0.0
    # Normalize roughly to 0-1 using tanh scale on returns-like values
    import math

    avg = sum(vals) / len(vals)
    return round(min(1.0, max(0.0, 0.5 + math.tanh(avg / 5) * 0.5)), 3)


def classify_signal(
    ticker: str,
    *,
    leg_reports: list[dict[str, Any]],
    news_score: float = 0.0,
    leg_combined_score: float = 0.0,
    strategist_score: float = 0.0,
) -> tuple[SignalKind, str]:
    """
    Distinguish news catalyst setups vs pure model/metrics setups.
    """
    metrics = max(metrics_strength_for_ticker(ticker, leg_reports), leg_combined_score * 0.5)
    news = float(news_score)
    strat = float(strategist_score)

    if news >= 0.25 and metrics < 0.45:
        return (
            "NEWS_CATALYST",
            f"{ticker}: headline catalyst (news {news:+.2f}) — models lag; use tight stop, fast profit.",
        )
    if metrics >= 0.50 and news < 0.12 and strat >= 0.45:
        return (
            "METRICS_ONLY",
            f"{ticker}: model/metrics driven (legs {metrics:.2f}, news flat) — vulnerable in bear markets.",
        )
    if news >= 0.15 and metrics >= 0.35:
        return (
            "MIXED",
            f"{ticker}: news + metrics align (news {news:+.2f}, metrics {metrics:.2f}).",
        )
    if news <= -0.15:
        return ("WEAK", f"{ticker}: negative news flow — avoid.")
    return ("WEAK", f"{ticker}: weak conviction — news and metrics both soft.")


def quick_rsi(df, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 50.0
    close = df["Close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-9)
    return float(100 - (100 / (1 + rs)))


def historical_news_proxy(df) -> float:
    """Backtest proxy when live headlines unavailable: volume spike + sharp move."""
    if df is None or len(df) < 25:
        return 0.0
    close = df["Close"]
    ret_3 = float(close.pct_change(3).iloc[-1]) if len(close) > 3 else 0.0
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        vol = df["Volume"].astype(float)
        ratio = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]) if vol.rolling(20).mean().iloc[-1] else 1.0
    else:
        ratio = 1.0
    if ret_3 > 0.06 and ratio > 1.4:
        return 0.35
    if ret_3 > 0.03 and ratio > 1.2:
        return 0.18
    if ret_3 < -0.06:
        return -0.25
    return 0.0
