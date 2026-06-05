"""Temporal Fusion Transformer (TFT) forecast leg via darts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Suppress lightning logs during quick fits
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass(frozen=True)
class TFTConfig:
    input_chunk_length: int = 60
    output_chunk_length: int = 5
    train_length: int = 200
    hidden_size: int = 16
    lstm_layers: int = 1
    num_attention_heads: int = 2
    n_epochs: int = 5
    batch_size: int = 16
    dropout: float = 0.1


def tft_available() -> bool:
    try:
        import darts  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def tft_forecast_return_pct(
    df: pd.DataFrame,
    cfg: TFTConfig | None = None,
    *,
    ticker: str = "",
) -> float | None:
    """
    Fit a small Temporal Fusion Transformer on recent closes and score
    predicted return over output_chunk_length days.
    """
    import warnings

    import torch
    from darts import TimeSeries
    from darts.models import TFTModel

    cfg = cfg or TFTConfig()
    close = df["Close"].dropna()
    if len(close) < cfg.train_length:
        return None

    series = TimeSeries.from_series(close.tail(cfg.train_length), fill_missing_dates=True, freq="B")
    if len(series) < cfg.input_chunk_length + cfg.output_chunk_length + 5:
        return None

    accelerator = "cpu"
    # MPS has float64 issues with darts TFT; keep CPU for reliability.
    if torch.cuda.is_available():
        accelerator = "gpu"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TFTModel(
            input_chunk_length=cfg.input_chunk_length,
            output_chunk_length=cfg.output_chunk_length,
            hidden_size=cfg.hidden_size,
            lstm_layers=cfg.lstm_layers,
            num_attention_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
            add_relative_index=True,
            batch_size=min(cfg.batch_size, max(8, len(series) // 4)),
            n_epochs=cfg.n_epochs,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            },
        )
        try:
            model.fit(series)
            pred = model.predict(n=cfg.output_chunk_length)
        except Exception as exc:
            logger.warning("TFT fit/predict failed for %s: %s", ticker or "?", exc)
            return None

    last = float(series[-1].values()[0, 0])
    forecast = float(pred[-1].values()[0, 0])
    if last <= 0:
        return None
    pct = (forecast / last - 1) * 100
    if pct != pct:  # NaN
        return None
    return round(pct, 4)


def tft_scores_batch(
    history: dict[str, pd.DataFrame],
    cfg: TFTConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or TFTConfig()
    out: dict[str, float] = {}
    for ticker, df in history.items():
        ret = tft_forecast_return_pct(df, cfg, ticker=ticker)
        if ret is not None:
            out[ticker.upper()] = ret
    return out
