"""Kronos foundation model wrapper — follows github.com/shiyu-coder/Kronos."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KRONOS_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "Kronos"


@dataclass(frozen=True)
class KronosConfig:
    """Settings aligned with Kronos README (forecasting uses lower temperature)."""

    model_id: str = "NeoQuasar/Kronos-mini"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-2k"
    max_context: int = 512
    lookback: int = 400
    pred_len: int = 5
    rebalance_days: int = 5
    temperature: float = 0.6
    top_p: float = 0.9
    sample_count: int = 3
    device: str | None = None  # auto: cuda > mps > cpu
    min_forecast_return_pct: float = 0.0  # skip buy if forecast below this


def kronos_available() -> bool:
    try:
        import torch  # noqa: F401
        return KRONOS_ROOT.is_dir()
    except ImportError:
        return False


def _ensure_kronos_path() -> None:
    root = str(KRONOS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _load_predictor(cfg: KronosConfig):
    _ensure_kronos_path()
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    device = cfg.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("Loading Kronos model=%s device=%s", cfg.model_id, device)
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_id)
    model = Kronos.from_pretrained(cfg.model_id)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    return predictor


def ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance OHLCV columns to Kronos format."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    needed = ["open", "high", "low", "close"]
    if not all(c in out.columns for c in needed):
        raise ValueError(f"Missing OHLC columns in {list(out.columns)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["amount"] = out["volume"] * out[needed].mean(axis=1)
    return out[needed + ["volume", "amount"]]


def forecast_return_pct(df: pd.DataFrame, cfg: KronosConfig | None = None) -> float | None:
    """
    Forecast near-term return using KronosPredictor (Kronos paper / README workflow).
    Returns predicted % change from last close to final forecast close.
    """
    cfg = cfg or KronosConfig()
    ohlcv = ohlcv_frame(df)
    if len(ohlcv) < cfg.lookback:
        return None

    predictor = _load_predictor(cfg)
    x_df = ohlcv.iloc[-cfg.lookback :].copy()
    x_timestamp = pd.Series(x_df.index)
    y_timestamp = pd.Series(
        pd.bdate_range(start=x_df.index[-1] + pd.Timedelta(days=1), periods=cfg.pred_len)
    )

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=cfg.pred_len,
        T=cfg.temperature,
        top_p=cfg.top_p,
        sample_count=cfg.sample_count,
        verbose=False,
    )

    last_close = float(x_df["close"].iloc[-1])
    forecast_close = float(pred_df["close"].iloc[-1])
    if last_close <= 0:
        return None
    return round((forecast_close / last_close - 1) * 100, 4)


def rank_watchlist(
    history: dict[str, pd.DataFrame],
    cfg: KronosConfig | None = None,
) -> list[tuple[str, float]]:
    """Kronos-style top-K ranking by forecasted return (highest first)."""
    cfg = cfg or KronosConfig()
    scores: list[tuple[str, float]] = []
    for ticker, df in history.items():
        try:
            ret = forecast_return_pct(df, cfg)
            if ret is not None:
                scores.append((ticker.upper(), ret))
        except Exception as exc:
            logger.warning("Kronos forecast failed for %s: %s", ticker, exc)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def pick_top_ticker(
    history: dict[str, pd.DataFrame],
    cfg: KronosConfig | None = None,
) -> tuple[str | None, str]:
    """Select best ticker if forecast return clears threshold; else cash."""
    cfg = cfg or KronosConfig()
    ranked = rank_watchlist(history, cfg)
    if not ranked:
        return None, "Kronos: no forecasts available."
    best_ticker, best_ret = ranked[0]
    if best_ret < cfg.min_forecast_return_pct:
        return None, f"Kronos: best {best_ticker} forecast {best_ret:+.2f}% below threshold."
    detail = ", ".join(f"{t} {r:+.2f}%" for t, r in ranked[:3])
    return best_ticker, f"Kronos top-K: {best_ticker} ({best_ret:+.2f}%). Rankings: {detail}"


def portfolio_kronos_top_k(
    history: dict[str, pd.DataFrame],
    *,
    benchmark: str = "SPY",
    lookback: int = 400,
    pred_len: int = 5,
    rebalance_days: int = 5,
    min_forecast_return_pct: float = 0.0,
) -> pd.Series:
    """
    Weekly-style backtest holds: run Kronos top-1 forecast on rebalance days.
    Follows Kronos finetune demo (TopkDropout / forecast score ranking).
    """
    cfg = KronosConfig(
        lookback=lookback,
        pred_len=pred_len,
        rebalance_days=rebalance_days,
        min_forecast_return_pct=min_forecast_return_pct,
    )

    tickers = sorted(history.keys())
    if not tickers:
        return pd.Series(dtype=str)

    closes = pd.DataFrame({t: history[t]["Close"] for t in tickers}).dropna(how="all")
    if closes.empty:
        return pd.Series(dtype=str)

    held = pd.Series("", index=closes.index, dtype=str)
    current_hold = ""

    min_rows = lookback + pred_len + 5
    for i, idx in enumerate(closes.index):
        if i < min_rows:
            continue
        if i % rebalance_days != 0:
            held.loc[idx] = current_hold
            continue

        slice_history = {
            t: history[t].loc[:idx].tail(lookback + pred_len + 10)
            for t in tickers
            if t in history and len(history[t].loc[:idx]) >= min_rows
        }
        if not slice_history:
            held.loc[idx] = current_hold
            continue

        try:
            target, _ = pick_top_ticker(slice_history, cfg)
            current_hold = target or ""
        except Exception as exc:
            logger.warning("Kronos rebalance at %s failed: %s", idx, exc)
        held.loc[idx] = current_hold

    return held
