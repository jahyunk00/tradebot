"""Hidden Markov Model regime scoring for individual tickers."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*Model is not converging.*")


def hmm_available() -> bool:
    try:
        import hmmlearn  # noqa: F401
        return True
    except ImportError:
        return False


def hmm_regime_score(
    df: pd.DataFrame,
    *,
    n_states: int = 3,
    lookback: int = 252,
    vol_window: int = 20,
) -> float | None:
    """
    Fit a Gaussian HMM on [daily return, rolling vol] and score the current regime.

    Higher score = current hidden state has higher expected forward return
    and the model assigns meaningful probability mass to it.
    """
    from hmmlearn.hmm import GaussianHMM

    close = df["Close"].tail(lookback)
    if len(close) < max(60, vol_window + 10):
        return None

    ret = close.pct_change().dropna()
    vol = ret.rolling(vol_window).std().bfill().fillna(0.0)
    aligned = pd.concat([ret, vol], axis=1).dropna()
    if len(aligned) < 50:
        return None

    features = aligned.values.astype(np.float64)
    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=150,
            random_state=42,
            tol=1e-3,
            min_covar=1e-4,
        )
        model.fit(features)
        states = model.predict(features)
        current = int(states[-1])

        state_returns = model.means_[:, 0]
        best_state = int(np.argmax(state_returns))
        current_return = float(state_returns[current])
        best_return = float(state_returns[best_state])

        # Probability of being in the current state now
        posteriors = model.predict_proba(features)
        state_prob = float(posteriors[-1, current])

        # Favor bullish current regime; penalize if current state is weak vs best
        regime_quality = current_return / (abs(best_return) + 1e-6)
        score = current_return * state_prob * (0.5 + 0.5 * max(regime_quality, 0))
        return round(float(score), 6)
    except Exception as exc:
        logger.debug("HMM fit failed: %s", exc)
        return None
