"""Multi-window trajectory scores for universe rotation."""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)


def _window_return_pct(closes, days: int) -> float | None:
    if closes is None or len(closes) < days + 2:
        return None
    start = float(closes.iloc[-days - 1])
    end = float(closes.iloc[-1])
    if start <= 0:
        return None
    return round((end / start - 1) * 100, 2)


def trajectory_score(ticker: str) -> dict[str, Any]:
    """
    Weight recent momentum heavier than stale 1Y — good trajectory = rising lately.
    Weights: 1M 35%, 3M 30%, 6M 20%, 1Y 15%
    """
    try:
        hist = yf.Ticker(ticker).history(period="400d", auto_adjust=True)
        if hist.empty:
            return {"ticker": ticker.upper(), "trajectory": None}
        closes = hist["Close"]
        r1 = _window_return_pct(closes, 21)
        r3 = _window_return_pct(closes, 63)
        r6 = _window_return_pct(closes, 126)
        r12 = _window_return_pct(closes, 252)
        parts = [(r1, 0.35), (r3, 0.30), (r6, 0.20), (r12, 0.15)]
        usable = [(r, w) for r, w in parts if r is not None]
        if not usable:
            return {"ticker": ticker.upper(), "trajectory": None}
        wsum = sum(w for _, w in usable)
        score = sum(r * w for r, w in usable) / wsum
        return {
            "ticker": ticker.upper(),
            "trajectory": round(score, 2),
            "return_1m_pct": r1,
            "return_3m_pct": r3,
            "return_6m_pct": r6,
            "return_1y_pct": r12,
        }
    except Exception as exc:
        logger.debug("trajectory %s failed: %s", ticker, exc)
        return {"ticker": ticker.upper(), "trajectory": None}


def rank_by_trajectory(tickers: list[str]) -> list[dict[str, Any]]:
    scored = [trajectory_score(t) for t in tickers]
    scored = [s for s in scored if s.get("trajectory") is not None]
    scored.sort(key=lambda x: -x["trajectory"])
    return scored
