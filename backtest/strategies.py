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


STRATEGIES = {
    "momentum": momentum_signals,
    "mean_reversion": mean_reversion_signals,
    "sma_crossover": sma_crossover_signals,
}
