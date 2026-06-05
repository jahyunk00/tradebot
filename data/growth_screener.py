"""Screen large-cap stocks with strong trailing growth for the boss watchlist."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Liquid US large caps — starting pool before growth + cap filters
DEFAULT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "V", "UNH", "XOM", "MA", "PG", "JNJ", "HD", "COST", "MRK", "ABBV",
    "CRM", "BAC", "NFLX", "AMD", "PEP", "TMO", "LIN", "CSCO", "ABT", "ACN",
    "WMT", "MCD", "DHR", "TXN", "DIS", "VZ", "INTU", "CMCSA", "ADBE", "PM",
    "NEE", "QCOM", "HON", "AMGN", "COP", "IBM", "GE", "CAT", "LOW", "UNP",
    "SPGI", "BA", "ELV", "GS", "MS", "BLK", "DE", "RTX", "ISRG", "GILD",
    "BKNG", "SYK", "ADP", "VRTX", "MDT", "LMT", "TJX", "CB", "CI",
    "MO", "SO", "DUK", "ZTS", "EQIX", "CME", "SHW", "NOC", "USB", "PNC",
    "EOG", "SLB", "ITW", "HCA", "APD", "EMR", "FDX", "ORCL", "NOW", "PANW",
    "SNPS", "CDNS", "KLAC", "LRCX", "MU", "AMAT", "ADI", "MELI", "UBER", "ABNB",
    "INTC", "SBUX", "CVX", "WFC", "C", "AXP", "TGT", "LULU", "MDLZ", "KO",
    "PEP", "BMY", "PFE", "UNP", "CSX", "NSC", "WM", "NKE", "ORLY", "AZO",
]


def _yahoo_1y_return_pct(ticker: str, cap_cache: dict[str, float | None]) -> tuple[float | None, float | None]:
    """Return (1Y change %, market cap B) from Yahoo fundamentals."""
    try:
        info = yf.Ticker(ticker).info or {}
        cap = round(float(info["marketCap"]) / 1e9, 1) if info.get("marketCap") else None
        cap_cache[ticker] = cap
        change = info.get("52WeekChange")
        if change is not None:
            return round(float(change) * 100, 2), cap
    except Exception:
        pass
    return None, cap_cache.get(ticker)


def _single_ticker_return(ticker: str, lookback_days: int) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period=f"{lookback_days + 30}d", auto_adjust=True)
        return _trailing_return_pct(hist["Close"], lookback_days)
    except Exception:
        return None


def _safe_return(ticker: str, batch_return: float | None, lookback_days: int) -> float | None:
    """Re-fetch if batch data looks like a split glitch (>200% on a mega-cap)."""
    if batch_return is None:
        return _single_ticker_return(ticker, lookback_days)
    if abs(batch_return) > 200:
        verified = _single_ticker_return(ticker, lookback_days)
        return verified if verified is not None else batch_return
    return batch_return


@dataclass
class ScreenerResult:
    tickers: list[str]
    details: list[dict[str, Any]]
    screened_at: str
    source: str


def _trailing_return_pct(closes: pd.Series, days: int = 252) -> float | None:
    if closes is None or len(closes) < 20:
        return None
    series = closes.dropna()
    if len(series) < 20:
        return None
    window = min(days, len(series) - 1)
    start = float(series.iloc[-window - 1])
    end = float(series.iloc[-1])
    if start <= 0:
        return None
    return round((end / start - 1) * 100, 2)


def _market_cap_b(ticker: str, cache: dict[str, float | None]) -> float | None:
    if ticker in cache:
        return cache[ticker]
    try:
        info = yf.Ticker(ticker).info or {}
        cap = info.get("marketCap")
        value = round(float(cap) / 1e9, 1) if cap else None
    except Exception:
        value = None
    cache[ticker] = value
    return value


def screen_large_cap_growth(
    universe: list[str] | None = None,
    *,
    min_market_cap_b: float = 50.0,
    min_return_1y_pct: float = 5.0,
    top_n: int = 15,
    lookback_days: int = 252,
) -> ScreenerResult:
    """
    Rank universe by 1-year price return; keep names above min market cap.
    Returns top_n tickers for boss agent scoring.
    """
    pool = list(dict.fromkeys(t.upper() for t in (universe or DEFAULT_UNIVERSE)))
    if not pool:
        return ScreenerResult([], [], datetime.now(timezone.utc).isoformat(), "empty_universe")

    logger.info("Screening %d tickers (min cap $%.0fB, min 1Y return %.1f%%)...", len(pool), min_market_cap_b, min_return_1y_pct)

    cap_cache: dict[str, float | None] = {}
    candidates: list[dict[str, Any]] = []

    for ticker in pool:
        ret, cap = _yahoo_1y_return_pct(ticker, cap_cache)
        if ret is None:
            ret = _single_ticker_return(ticker, lookback_days)
            if cap is None:
                cap = _market_cap_b(ticker, cap_cache)

        if ret is None or ret < min_return_1y_pct:
            continue

        if cap is not None and cap < min_market_cap_b:
            continue

        candidates.append(
            {
                "ticker": ticker,
                "return_1y_pct": ret,
                "market_cap_b": cap,
            }
        )

    candidates.sort(key=lambda x: (-x["return_1y_pct"], -(x["market_cap_b"] or 0)))
    selected = candidates[:top_n]
    tickers = [c["ticker"] for c in selected]

    logger.info(
        "Screener picked %d: %s",
        len(tickers),
        ", ".join(f"{c['ticker']} ({c['return_1y_pct']:+.1f}%)" for c in selected[:5]),
    )

    return ScreenerResult(
        tickers=tickers,
        details=selected,
        screened_at=datetime.now(timezone.utc).isoformat(),
        source="growth_large_cap",
    )


def _cache_path(base_dir: Path) -> Path:
    return base_dir / "logs" / "screener_cache.json"


def _load_cache(base_dir: Path, max_age_hours: float) -> ScreenerResult | None:
    path = _cache_path(base_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        screened_at = datetime.fromisoformat(data["screened_at"].replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - screened_at).total_seconds() / 3600
        if age_h > max_age_hours:
            return None
        return ScreenerResult(
            tickers=data["tickers"],
            details=data.get("details", []),
            screened_at=data["screened_at"],
            source=data.get("source", "cache"),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cache(base_dir: Path, result: ScreenerResult) -> None:
    path = _cache_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tickers": result.tickers,
                "details": result.details,
                "screened_at": result.screened_at,
                "source": result.source,
            },
            indent=2,
        )
    )


def resolve_growth_watchlist(
    base_dir: Path,
    *,
    enabled: bool,
    universe: list[str] | None,
    min_market_cap_b: float,
    min_return_1y_pct: float,
    top_n: int,
    lookback_days: int,
    cache_hours: float,
    fallback: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Return watchlist + metadata; uses daily cache on Railway/cron."""
    if not enabled:
        return fallback, {"source": "config_watchlist", "tickers": fallback}

    cached = _load_cache(base_dir, cache_hours)
    if cached and cached.tickers:
        return cached.tickers, {
            "source": "cache",
            "screened_at": cached.screened_at,
            "tickers": cached.tickers,
            "details": cached.details,
        }

    result = screen_large_cap_growth(
        universe or DEFAULT_UNIVERSE,
        min_market_cap_b=min_market_cap_b,
        min_return_1y_pct=min_return_1y_pct,
        top_n=top_n,
        lookback_days=lookback_days,
    )

    if not result.tickers:
        logger.warning("Screener returned no tickers — using fallback watchlist.")
        return fallback, {"source": "fallback", "tickers": fallback, "error": "no_screen_matches"}

    _save_cache(base_dir, result)
    return result.tickers, {
        "source": result.source,
        "screened_at": result.screened_at,
        "tickers": result.tickers,
        "details": result.details,
    }
