"""Bayesian-style changepoint detection for trend direction scoring."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="ruptures")


def changepoint_available() -> bool:
    try:
        import ruptures  # noqa: F401
        return True
    except ImportError:
        return False


def bayesian_changepoint_direction(
    df: pd.DataFrame,
    *,
    lookback: int = 252,
    penalty: float = 3.0,
    min_segment: int = 10,
) -> float | None:
    """
    Detect the most recent changepoint in daily returns (PELT / BIC-style penalty),
    then score direction as mean return since that point (bullish > 0, bearish < 0).

    Weighted by how recent the changepoint is — fresher regime shifts count more.
    """
    import ruptures as rpt

    close = df["Close"].tail(lookback)
    if len(close) < min_segment * 3:
        return None

    log_ret = np.log(close / close.shift(1)).dropna().values.reshape(-1, 1)
    if len(log_ret) < min_segment * 2:
        return None

    try:
        algo = rpt.Pelt(model="rbf", min_size=min_segment, jump=1).fit(log_ret)
        breakpoints = algo.predict(pen=penalty)
        if not breakpoints or breakpoints[-1] != len(log_ret):
            breakpoints = [b for b in breakpoints if b < len(log_ret)]

        last_cp = breakpoints[-2] if len(breakpoints) >= 2 else 0
        segment = log_ret[last_cp:, 0]
        if len(segment) < 3:
            return None

        direction = float(np.mean(segment))
        vol = float(np.std(segment)) + 1e-6
        sharpe_like = direction / vol

        days_since_cp = len(log_ret) - last_cp
        recency = 1.0 / (1.0 + days_since_cp / 20.0)
        score = sharpe_like * recency
        return round(score, 6)
    except Exception as exc:
        logger.debug("Changepoint detection failed: %s", exc)
        return None
