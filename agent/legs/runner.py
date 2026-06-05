"""Run the three specialist leg agents."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from agent.legs.base import LegReport
from models.ensemble_strategy import (
    EnsembleSettings,
    kronos_cfg_from_agent_config,
    settings_from_agent_config,
)
from models.bayesian_changepoint import bayesian_changepoint_direction, changepoint_available
from models.hmm_strategy import hmm_available, hmm_regime_score
from models.kronos_engine import KronosConfig, forecast_return_pct, kronos_available
from models.tft_strategy import tft_available, tft_forecast_return_pct

logger = logging.getLogger(__name__)


def run_kronos_leg(history: dict[str, pd.DataFrame], agent_cfg, *, live: bool) -> LegReport:
    if not kronos_available():
        return LegReport("kronos", available=False, note="Kronos not installed")

    cfg = kronos_cfg_from_agent_config(agent_cfg)
    if not live and not agent_cfg.boss.training.use_kronos:
        return LegReport("kronos", available=False, note="Skipped in training (use_kronos: false)")

    scores: dict[str, float] = {}
    for ticker, df in history.items():
        try:
            ret = forecast_return_pct(df, cfg)
            if ret is not None:
                scores[ticker.upper()] = ret
        except Exception as exc:
            logger.warning("Kronos leg %s: %s", ticker, exc)
    return LegReport(
        "kronos",
        scores=scores,
        available=bool(scores),
        note=f"Forecast horizon {cfg.pred_len}d",
    )


def run_hmm_leg(history: dict[str, pd.DataFrame], settings: EnsembleSettings) -> LegReport:
    if not hmm_available():
        return LegReport("hmm", available=False, note="hmmlearn not installed")

    scores: dict[str, float] = {}
    for ticker, df in history.items():
        s = hmm_regime_score(
            df,
            n_states=settings.hmm_n_states,
            lookback=settings.hmm_lookback,
            vol_window=settings.hmm_vol_window,
        )
        if s is not None:
            scores[ticker.upper()] = s
    return LegReport("hmm", scores=scores, available=bool(scores), note="Gaussian HMM regime")


def run_third_leg(history: dict[str, pd.DataFrame], agent_cfg, settings: EnsembleSettings, *, live: bool) -> LegReport:
    model = settings.third_leg_model
    if not live and not agent_cfg.boss.training.use_third_leg:
        return LegReport("third_leg", available=False, note="Skipped in training")

    train_model = agent_cfg.boss.training.third_leg_model if not live else model
    scores: dict[str, float] = {}

    if train_model == "bayesian_changepoint":
        if not changepoint_available():
            return LegReport("third_leg", available=False, note="ruptures not installed")
        for ticker, df in history.items():
            s = bayesian_changepoint_direction(
                df, lookback=settings.bcp_lookback, penalty=settings.bcp_penalty
            )
            if s is not None:
                scores[ticker.upper()] = s
        return LegReport(
            "third_leg",
            scores=scores,
            available=bool(scores),
            note=f"Bayesian changepoint direction",
        )

    if not tft_available():
        return LegReport("third_leg", available=False, note="darts/TFT not installed")
    for ticker, df in history.items():
        ret = tft_forecast_return_pct(df, settings.tft, ticker=ticker)
        if ret is not None:
            scores[ticker.upper()] = ret
    return LegReport(
        "third_leg",
        scores=scores,
        available=bool(scores),
        note=f"TFT forecast ({model})",
    )


def run_all_legs(
    history: dict[str, pd.DataFrame],
    agent_cfg: Any,
    *,
    live: bool = True,
) -> list[LegReport]:
    settings = settings_from_agent_config(agent_cfg)
    return [
        run_kronos_leg(history, agent_cfg, live=live),
        run_hmm_leg(history, settings),
        run_third_leg(history, agent_cfg, settings, live=live),
    ]
