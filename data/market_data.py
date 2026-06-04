"""Historical market data for backtesting."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_history(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Download adjusted daily OHLCV for each ticker."""
    period = f"{lookback_days}d"
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title)
        df.index = pd.to_datetime(df.index)
        out[ticker.upper()] = df
    return out


def latest_quotes(tickers: list[str]) -> dict[str, dict]:
    """Snapshot of recent price action for analysis prompts."""
    quotes: dict[str, dict] = {}
    for ticker in tickers:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            continue
        close = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
        pct = ((close - prev) / prev * 100) if prev else 0.0
        quotes[ticker.upper()] = {
            "last_close": round(close, 2),
            "prev_close": round(prev, 2),
            "daily_change_pct": round(pct, 2),
        }
    return quotes
