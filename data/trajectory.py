"""Multi-window trajectory scores for universe rotation."""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# Cap any single window so bad Yahoo data (e.g. MU splits) cannot dominate picks.
MAX_WINDOW_RETURN_PCT = 80.0


def _cap(value: float | None, limit: float = MAX_WINDOW_RETURN_PCT) -> float | None:
    if value is None:
        return None
    return max(min(float(value), limit), -limit)


def _yahoo_1y_pct(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).info or {}
        change = info.get("52WeekChange")
        if change is not None:
            return _cap(float(change) * 100, 120.0)
    except Exception:
        pass
    return None


def _window_return_pct(closes, days: int) -> float | None:
    if closes is None or len(closes) < days + 2:
        return None
    start = float(closes.iloc[-days - 1])
    end = float(closes.iloc[-1])
    if start <= 0:
        return None
    return _cap(round((end / start - 1) * 100, 2))


def trajectory_score(ticker: str) -> dict[str, Any]:
    """
    Recent momentum drives rotation — not stale 1Y blow-off winners.
    Weights: 1M 50%, 3M 35%, 6M 10%, 1Y 5%
    """
    try:
        hist = yf.Ticker(ticker).history(period="400d", auto_adjust=True)
        if hist.empty:
            return {"ticker": ticker.upper(), "trajectory": None}
        closes = hist["Close"]
        r1 = _window_return_pct(closes, 21)
        r3 = _window_return_pct(closes, 63)
        r6 = _window_return_pct(closes, 126)
        r12 = _yahoo_1y_pct(ticker.upper()) or _window_return_pct(closes, 252)
        r12 = _cap(r12, 120.0)
        parts = [(r1, 0.50), (r3, 0.35), (r6, 0.10), (r12, 0.05)]
        usable = [(r, w) for r, w in parts if r is not None]
        if not usable:
            return {"ticker": ticker.upper(), "trajectory": None}
        wsum = sum(w for _, w in usable)
        score = sum(r * w for r, w in usable) / wsum
        recent = recent_momentum_from_parts(r1, r3)
        return {
            "ticker": ticker.upper(),
            "trajectory": round(score, 2),
            "recent_momentum": recent,
            "return_1m_pct": r1,
            "return_3m_pct": r3,
            "return_6m_pct": r6,
            "return_1y_pct": r12,
        }
    except Exception as exc:
        logger.debug("trajectory %s failed: %s", ticker, exc)
        return {"ticker": ticker.upper(), "trajectory": None}


def recent_momentum_from_parts(r1: float | None, r3: float | None) -> float | None:
    """Short-term momentum for live pick ranking (1M + 3M)."""
    parts = [(r1, 0.55), (r3, 0.45)]
    usable = [(r, w) for r, w in parts if r is not None]
    if not usable:
        return None
    wsum = sum(w for _, w in usable)
    return round(sum(r * w for r, w in usable) / wsum, 2)


def recent_momentum_by_ticker(details: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in details:
        t = str(row.get("ticker", "")).upper()
        if not t:
            continue
        rm = row.get("recent_momentum")
        if rm is None:
            rm = recent_momentum_from_parts(row.get("return_1m_pct"), row.get("return_3m_pct"))
        if rm is not None:
            out[t] = float(rm)
    return out


def rank_by_trajectory(tickers: list[str]) -> list[dict[str, Any]]:
    scored = [trajectory_score(t) for t in tickers]
    scored = [s for s in scored if s.get("trajectory") is not None]
    scored.sort(key=lambda x: -x["trajectory"])
    return scored
