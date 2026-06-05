"""Backtest strategies applied to historical price data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signals(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Long when price above N-day SMA and 5-day return positive."""
    close = df["Close"]
    sma = close.rolling(lookback).mean()
    ret_5d = close.pct_change(5)
    signal = (close > sma) & (ret_5d > 0)
    return signal.astype(int)


def mean_reversion_signals(df: pd.DataFrame, lookback: int = 20, z_threshold: float = 1.5) -> pd.Series:
    """Long when price is more than z_threshold std devs below rolling mean."""
    close = df["Close"]
    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    z = (close - mean) / std.replace(0, np.nan)
    signal = (z < -z_threshold).astype(int)
    return signal


def sma_crossover_signals(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """Long when fast SMA crosses above slow SMA."""
    close = df["Close"]
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()
    signal = (fast_sma > slow_sma).astype(int)
    return signal


def dual_momentum_signals(df: pd.DataFrame, lookback: int = 200) -> pd.Series:
    """Long when price is above long-term SMA (time-series momentum / GTAA style)."""
    close = df["Close"]
    sma = close.rolling(lookback).mean()
    return (close > sma).astype(int)


def portfolio_relative_strength(
    history: dict[str, pd.DataFrame],
    *,
    momentum_days: int = 126,
    trend_days: int = 200,
    top_n: int = 1,
) -> pd.Series:
    """
    Rotate into top-N tickers by trailing return, only if above trend SMA.
    Returns a daily Series of held ticker symbol (or empty string for cash).
    """
    tickers = sorted(history.keys())
    if not tickers:
        return pd.Series(dtype=str)

    closes = pd.DataFrame({t: history[t]["Close"] for t in tickers}).dropna(how="all")
    if closes.empty:
        return pd.Series(dtype=str)

    mom = closes.pct_change(momentum_days)
    trend = closes.rolling(trend_days).mean()
    above_trend = closes > trend

    held = pd.Series("", index=closes.index, dtype=str)
    for idx in closes.index:
        eligible = [t for t in tickers if above_trend.loc[idx, t] and pd.notna(mom.loc[idx, t])]
        if not eligible:
            continue
        ranked = sorted(eligible, key=lambda t: float(mom.loc[idx, t]), reverse=True)
        held.loc[idx] = ranked[0] if top_n >= 1 else ""

    return held


def portfolio_dual_momentum_spy(
    history: dict[str, pd.DataFrame],
    benchmark: str = "SPY",
    lookback: int = 200,
) -> pd.Series:
    """Hold benchmark ETF when above SMA, else cash."""
    bench = history.get(benchmark.upper())
    if bench is None or bench.empty:
        return pd.Series(dtype=str)

    signal = dual_momentum_signals(bench, lookback=lookback)
    held = pd.Series("", index=bench.index, dtype=str)
    held[signal == 1] = benchmark.upper()
    return held


def portfolio_returns_from_holds(
    history: dict[str, pd.DataFrame],
    holds: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Convert daily held-ticker series into strategy returns and position flags."""
    tickers = sorted(history.keys())
    closes = pd.DataFrame({t: history[t]["Close"] for t in tickers}).reindex(holds.index)
    daily_ret = closes.pct_change().fillna(0)

    strat_ret = pd.Series(0.0, index=holds.index)
    for idx in holds.index:
        ticker = holds.loc[idx]
        if ticker and ticker in daily_ret.columns:
            strat_ret.loc[idx] = daily_ret.loc[idx, ticker]

    position = holds.apply(lambda t: 1 if t else 0)
    return strat_ret, position


STRATEGIES = {
    "momentum": momentum_signals,
    "mean_reversion": mean_reversion_signals,
    "sma_crossover": sma_crossover_signals,
    "dual_momentum": dual_momentum_signals,
}

PORTFOLIO_STRATEGIES = {
    "relative_strength": portfolio_relative_strength,
    "dual_momentum_spy": portfolio_dual_momentum_spy,
}


def _portfolio_kronos_top_k(history: dict[str, pd.DataFrame], benchmark: str = "SPY", **kwargs) -> pd.Series:
    from models.kronos_engine import portfolio_kronos_top_k

    return portfolio_kronos_top_k(history, benchmark=benchmark, **kwargs)


def register_kronos_strategy() -> None:
    PORTFOLIO_STRATEGIES["kronos_top_k"] = _portfolio_kronos_top_k


def _portfolio_ensemble_weighted(history: dict[str, pd.DataFrame], benchmark: str = "SPY", **kwargs) -> pd.Series:
    from pathlib import Path

    from agent.config import load_config
    from models.ensemble_strategy import (
        kronos_cfg_from_agent_config,
        portfolio_ensemble_weighted,
        settings_from_agent_config,
    )

    agent_cfg, _ = load_config(Path(__file__).resolve().parent.parent)
    return portfolio_ensemble_weighted(
        history,
        benchmark=benchmark,
        settings=settings_from_agent_config(agent_cfg),
        kronos_cfg=kronos_cfg_from_agent_config(agent_cfg),
    )


def register_ensemble_strategy() -> None:
    PORTFOLIO_STRATEGIES["ensemble_weighted"] = _portfolio_ensemble_weighted


register_kronos_strategy()
register_ensemble_strategy()

ALL_STRATEGY_NAMES = list(STRATEGIES.keys()) + list(PORTFOLIO_STRATEGIES.keys())

__all__ = [
    "STRATEGIES",
    "PORTFOLIO_STRATEGIES",
    "ALL_STRATEGY_NAMES",
    "momentum_signals",
    "mean_reversion_signals",
    "sma_crossover_signals",
    "dual_momentum_signals",
    "portfolio_relative_strength",
    "portfolio_dual_momentum_spy",
    "portfolio_returns_from_holds",
    "register_kronos_strategy",
    "register_ensemble_strategy",
]
