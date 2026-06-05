"""Weighted ensemble: Kronos + HMM + (Bayesian CP or TFT) → trade decision."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from models.bayesian_changepoint import bayesian_changepoint_direction, changepoint_available
from models.hmm_strategy import hmm_available, hmm_regime_score
from models.tft_strategy import TFTConfig, tft_available, tft_scores_batch

logger = logging.getLogger(__name__)

ThirdLegModel = Literal["bayesian_changepoint", "tft"]


@dataclass(frozen=True)
class EnsembleWeights:
    kronos: float = 0.34
    hmm: float = 0.33
    third_leg: float = 0.33

    def normalized(
        self,
        *,
        use_kronos: bool,
        use_third_leg: bool,
    ) -> dict[str, float]:
        weights = {
            "kronos": self.kronos if use_kronos else 0.0,
            "hmm": self.hmm,
            "third_leg": self.third_leg if use_third_leg else 0.0,
        }
        total = sum(weights.values())
        if total <= 0:
            return {"hmm": 1.0}
        return {k: v / total for k, v in weights.items()}


@dataclass(frozen=True)
class EnsembleSettings:
    weights: EnsembleWeights = EnsembleWeights()
    third_leg_model: ThirdLegModel = "tft"
    rebalance_days: int = 5
    min_combined_score: float = 0.30
    backtest_use_kronos: bool = False
    backtest_use_third_leg: bool = False
    hmm_n_states: int = 3
    hmm_lookback: int = 252
    hmm_vol_window: int = 20
    bcp_lookback: int = 252
    bcp_penalty: float = 3.0
    tft: TFTConfig = TFTConfig()


@dataclass
class EnsembleDecision:
    target_ticker: str | None
    combined_scores: dict[str, float] = field(default_factory=dict)
    leg_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    third_leg_model: str = ""
    rationale: str = ""


def _rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    if len(scores) == 1:
        ticker = next(iter(scores))
        return {ticker: 1.0}
    ranked = sorted(scores.items(), key=lambda x: x[1])
    n = len(ranked)
    return {ticker: i / (n - 1) for i, (ticker, _) in enumerate(ranked)}


def _kronos_scores(history: dict[str, pd.DataFrame], kronos_cfg) -> dict[str, float]:
    from models.kronos_engine import KronosConfig, forecast_return_pct, kronos_available

    if not kronos_available():
        return {}

    cfg = kronos_cfg if isinstance(kronos_cfg, KronosConfig) else KronosConfig(**kronos_cfg)
    out: dict[str, float] = {}
    for ticker, df in history.items():
        try:
            ret = forecast_return_pct(df, cfg)
            if ret is not None:
                out[ticker.upper()] = ret
        except Exception as exc:
            logger.warning("Kronos leg failed for %s: %s", ticker, exc)
    return out


def _hmm_scores(history: dict[str, pd.DataFrame], settings: EnsembleSettings) -> dict[str, float]:
    if not hmm_available():
        if not getattr(_hmm_scores, "_warned", False):
            logger.warning("hmmlearn not installed — HMM leg skipped")
            _hmm_scores._warned = True  # type: ignore[attr-defined]
        return {}

    out: dict[str, float] = {}
    for ticker, df in history.items():
        s = hmm_regime_score(
            df,
            n_states=settings.hmm_n_states,
            lookback=settings.hmm_lookback,
            vol_window=settings.hmm_vol_window,
        )
        if s is not None:
            out[ticker.upper()] = s
    return out


def _third_leg_scores(
    history: dict[str, pd.DataFrame],
    settings: EnsembleSettings,
    *,
    active: bool,
) -> tuple[dict[str, float], str]:
    """Score the configured third leg: bayesian changepoint OR TFT."""
    if not active:
        return {}, settings.third_leg_model

    model = settings.third_leg_model
    if model == "bayesian_changepoint":
        if not changepoint_available():
            if not getattr(_third_leg_scores, "_bcp_warned", False):
                logger.warning("ruptures not installed — Bayesian changepoint leg skipped")
                _third_leg_scores._bcp_warned = True  # type: ignore[attr-defined]
            return {}, model
        out: dict[str, float] = {}
        for ticker, df in history.items():
            s = bayesian_changepoint_direction(
                df,
                lookback=settings.bcp_lookback,
                penalty=settings.bcp_penalty,
            )
            if s is not None:
                out[ticker.upper()] = s
        return out, model

    # tft
    if not tft_available():
        if not getattr(_third_leg_scores, "_tft_warned", False):
            logger.warning("darts not installed — TFT leg skipped")
            _third_leg_scores._tft_warned = True  # type: ignore[attr-defined]
        return {}, model
    return tft_scores_batch(history, settings.tft), model


def score_watchlist(
    history: dict[str, pd.DataFrame],
    settings: EnsembleSettings,
    *,
    kronos_cfg=None,
    use_kronos: bool | None = None,
    use_third_leg: bool | None = None,
    live: bool = False,
) -> EnsembleDecision:
    """
    Three decision makers (weighted):
      1. Kronos — OHLCV foundation model
      2. HMM — hidden Markov regime
      3. Third leg — bayesian_changepoint OR tft (config: third_leg_model)
    """
    from models.kronos_engine import kronos_available

    if use_kronos is None:
        use_kronos = kronos_available() if live else settings.backtest_use_kronos
    if use_third_leg is None:
        use_third_leg = (tft_available() or changepoint_available()) if live else settings.backtest_use_third_leg

    weights = settings.weights.normalized(use_kronos=use_kronos, use_third_leg=use_third_leg)

    kronos_raw = _kronos_scores(history, kronos_cfg) if weights.get("kronos", 0) > 0 and kronos_cfg else {}
    hmm_raw = _hmm_scores(history, settings) if weights.get("hmm", 0) > 0 else {}
    third_raw, third_name = _third_leg_scores(history, settings, active=weights.get("third_leg", 0) > 0)

    leg_scores = {
        "kronos": kronos_raw,
        "hmm": hmm_raw,
        third_name: third_raw,
    }
    leg_norm = {leg: _rank_normalize(raw) for leg, raw in leg_scores.items() if raw}

    # Map third_leg weight to the active model name for scoring
    weight_by_leg: dict[str, float] = {
        "kronos": weights.get("kronos", 0),
        "hmm": weights.get("hmm", 0),
        third_name: weights.get("third_leg", 0),
    }

    tickers = sorted(set().union(*[set(d.keys()) for d in leg_norm.values()]))
    combined: dict[str, float] = {}
    for ticker in tickers:
        score = 0.0
        for leg, w in weight_by_leg.items():
            if w <= 0 or leg not in leg_norm:
                continue
            score += w * leg_norm[leg].get(ticker, 0.0)
        combined[ticker] = round(score, 4)

    if not combined:
        return EnsembleDecision(
            target_ticker=None,
            leg_scores=leg_scores,
            weights_used=weights,
            third_leg_model=third_name,
            rationale="Ensemble: no scores from any leg — stay in cash.",
        )

    best_ticker, best_score = max(combined.items(), key=lambda x: x[1])
    if best_score < settings.min_combined_score:
        return EnsembleDecision(
            target_ticker=None,
            combined_scores=combined,
            leg_scores=leg_scores,
            weights_used=weights,
            third_leg_model=third_name,
            rationale=(
                f"Ensemble: best {best_ticker} score {best_score:.2f} "
                f"below threshold {settings.min_combined_score:.2f} — cash."
            ),
        )

    parts = []
    for leg, w in weight_by_leg.items():
        if w <= 0:
            continue
        raw = leg_scores.get(leg, {}).get(best_ticker)
        norm = leg_norm.get(leg, {}).get(best_ticker)
        if raw is not None and norm is not None:
            parts.append(f"{leg}={raw:+.4f}→{norm:.2f}×{w:.0%}")

    rationale = (
        f"Ensemble pick {best_ticker} (combined {best_score:.2f}). "
        f"Legs: kronos {weights.get('kronos', 0):.0%}, "
        f"hmm {weights.get('hmm', 0):.0%}, "
        f"{third_name} {weights.get('third_leg', 0):.0%}. "
        f"{' | '.join(parts)}"
    )

    return EnsembleDecision(
        target_ticker=best_ticker,
        combined_scores=combined,
        leg_scores=leg_scores,
        weights_used=weights,
        third_leg_model=third_name,
        rationale=rationale,
    )


def portfolio_ensemble_weighted(
    history: dict[str, pd.DataFrame],
    *,
    benchmark: str = "SPY",
    settings: EnsembleSettings | None = None,
    kronos_cfg=None,
) -> pd.Series:
    settings = settings or EnsembleSettings()
    tickers = sorted(history.keys())
    if not tickers:
        return pd.Series(dtype=str)

    closes = pd.DataFrame({t: history[t]["Close"] for t in tickers}).dropna(how="all")
    if closes.empty:
        return pd.Series(dtype=str)

    held = pd.Series("", index=closes.index, dtype=str)
    current = ""
    min_rows = max(settings.hmm_lookback, settings.bcp_lookback, settings.tft.train_length) + 10

    for i, idx in enumerate(closes.index):
        if i < min_rows:
            continue
        if i % settings.rebalance_days != 0:
            held.loc[idx] = current
            continue

        slice_history = {
            t: history[t].loc[:idx]
            for t in tickers
            if t in history and len(history[t].loc[:idx]) >= min_rows
        }
        if not slice_history:
            held.loc[idx] = current
            continue

        decision = score_watchlist(
            slice_history,
            settings,
            kronos_cfg=kronos_cfg,
            use_kronos=settings.backtest_use_kronos,
            use_third_leg=settings.backtest_use_third_leg,
            live=False,
        )
        current = decision.target_ticker or ""
        held.loc[idx] = current

    return held


def settings_from_agent_config(agent_cfg) -> EnsembleSettings:
    e = agent_cfg.ensemble
    w = e.weights
    tft = agent_cfg.tft
    return EnsembleSettings(
        weights=EnsembleWeights(
            kronos=w.kronos,
            hmm=w.hmm,
            third_leg=w.third_leg,
        ),
        third_leg_model=e.third_leg_model,
        rebalance_days=e.rebalance_days,
        min_combined_score=e.min_combined_score,
        backtest_use_kronos=e.backtest_use_kronos,
        backtest_use_third_leg=e.backtest_use_third_leg,
        hmm_n_states=agent_cfg.hmm.n_states,
        hmm_lookback=agent_cfg.hmm.lookback,
        hmm_vol_window=agent_cfg.hmm.vol_window,
        bcp_lookback=agent_cfg.bayesian_changepoint.lookback,
        bcp_penalty=agent_cfg.bayesian_changepoint.penalty,
        tft=TFTConfig(
            input_chunk_length=tft.input_chunk_length,
            output_chunk_length=tft.output_chunk_length,
            train_length=tft.train_length,
            hidden_size=tft.hidden_size,
            lstm_layers=tft.lstm_layers,
            num_attention_heads=tft.num_attention_heads,
            n_epochs=tft.n_epochs,
            batch_size=tft.batch_size,
            dropout=tft.dropout,
        ),
    )


def kronos_cfg_from_agent_config(agent_cfg):
    from models.kronos_engine import KronosConfig

    return KronosConfig(
        **{
            k: v
            for k, v in agent_cfg.kronos.model_dump().items()
            if k in KronosConfig.__dataclass_fields__
        }
    )
