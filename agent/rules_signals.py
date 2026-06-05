"""Live rule-based signals — no LLM required."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.market_data import fetch_history


def _portfolio_strategies() -> dict:
    from backtest.strategies import PORTFOLIO_STRATEGIES

    return PORTFOLIO_STRATEGIES


def _strategies() -> dict:
    from backtest.strategies import STRATEGIES

    return STRATEGIES


@dataclass
class LiveSignal:
    strategy: str
    target_ticker: str | None  # None = move to cash
    previous_ticker: str | None
    action: str  # buy | sell | hold | rotate
    rationale: str
    in_trend: bool = True


def _latest_hold(strategy_name: str, history: dict[str, pd.DataFrame], benchmark: str) -> tuple[str | None, str | None]:
    portfolio = _portfolio_strategies()
    if strategy_name in portfolio:
        fn = portfolio[strategy_name]
        kwargs = {"benchmark": benchmark} if strategy_name == "dual_momentum_spy" else {}
        holds = fn(history, **kwargs)
        if holds.empty:
            return None, None
        current = holds.iloc[-1] or None
        previous = holds.iloc[-2] if len(holds) > 1 else ""
        previous = previous or None
        return current, previous

    # Per-ticker momentum: pick strongest name with active signal
    scores: list[tuple[str, float]] = []
    single = _strategies()
    for ticker, df in history.items():
        if len(df) < 30:
            continue
        signal_fn = single.get(strategy_name, single["momentum"])
        sig = signal_fn(df)
        if int(sig.iloc[-1]) != 1:
            continue
        ret = float(df["Close"].pct_change(20).iloc[-1])
        scores.append((ticker, ret))

    current = max(scores, key=lambda x: x[1])[0] if scores else None
    return current, None


def get_live_signal(
    strategy_name: str,
    watchlist: list[str],
    *,
    lookback_days: int = 365,
    benchmark: str = "SPY",
    kronos_cfg: dict | None = None,
) -> LiveSignal:
    """Compute today's target allocation from the configured rules strategy."""
    if strategy_name == "kronos_top_k":
        return _kronos_live_signal(watchlist, lookback_days, kronos_cfg)

    if strategy_name == "ensemble_weighted":
        return _ensemble_live_signal(watchlist, lookback_days, kronos_cfg)

    tickers = list(dict.fromkeys([*watchlist, benchmark.upper()]))
    history = fetch_history(tickers, lookback_days)
    if not history:
        return LiveSignal(
            strategy=strategy_name,
            target_ticker=None,
            previous_ticker=None,
            action="hold",
            rationale="No market data available.",
            in_trend=False,
        )

    current, previous = _latest_hold(strategy_name, history, benchmark)

    if current is None:
        action = "sell" if previous else "hold"
        rationale = (
            f"{strategy_name}: no qualifying ticker — stay in cash."
            if strategy_name in _portfolio_strategies()
            else f"{strategy_name}: no buy signals on watchlist — stay in cash."
        )
        return LiveSignal(
            strategy=strategy_name,
            target_ticker=None,
            previous_ticker=previous,
            action=action,
            rationale=rationale,
            in_trend=False,
        )

    if previous and previous != current:
        action = "rotate"
        rationale = f"{strategy_name}: rotate {previous} → {current}."
    elif previous == current:
        action = "hold"
        rationale = f"{strategy_name}: continue holding {current}."
    else:
        action = "buy"
        rationale = f"{strategy_name}: enter {current}."

    return LiveSignal(
        strategy=strategy_name,
        target_ticker=current,
        previous_ticker=previous,
        action=action,
        rationale=rationale,
        in_trend=True,
    )


def parse_positions(account_context: dict) -> dict[str, float]:
    """Ticker -> market value from Robinhood MCP account context."""
    import json
    import re

    positions: dict[str, float] = {}
    text = json.dumps(account_context, default=str)
    for match in re.finditer(
        r'"(?:symbol|ticker)"\s*:\s*"([A-Z]{1,5})".{0,200}?"(?:market_value|equity|value)"\s*:\s*"?([\d.]+)"?',
        text,
        re.IGNORECASE,
    ):
        positions[match.group(1).upper()] = float(match.group(2))
    return positions


def _kronos_live_signal(
    watchlist: list[str],
    lookback_days: int,
    kronos_cfg: dict | None,
) -> LiveSignal:
    from agent.config import KronosSettings
    from models.kronos_engine import KronosConfig, kronos_available, pick_top_ticker

    if not kronos_available():
        return LiveSignal(
            strategy="kronos_top_k",
            target_ticker=None,
            previous_ticker=None,
            action="hold",
            rationale=(
                "Kronos not installed. Run: pip install -r requirements-kronos.txt "
                "(vendor/Kronos must exist)."
            ),
            in_trend=False,
        )

    cfg_dict = kronos_cfg or {}
    settings = KronosSettings(**cfg_dict) if cfg_dict else KronosSettings()
    cfg = KronosConfig(
        model_id=settings.model_id,
        tokenizer_id=settings.tokenizer_id,
        max_context=settings.max_context,
        lookback=min(settings.lookback, lookback_days),
        pred_len=settings.pred_len,
        rebalance_days=settings.rebalance_days,
        temperature=settings.temperature,
        top_p=settings.top_p,
        sample_count=settings.sample_count,
        min_forecast_return_pct=settings.min_forecast_return_pct,
        device=settings.device,
    )

    history = fetch_history(watchlist, max(lookback_days, cfg.lookback + cfg.pred_len))
    target, rationale = pick_top_ticker(history, cfg)
    action = "buy" if target else "hold"
    return LiveSignal(
        strategy="kronos_top_k",
        target_ticker=target,
        previous_ticker=None,
        action=action,
        rationale=rationale,
        in_trend=target is not None,
    )


def _ensemble_live_signal(
    watchlist: list[str],
    lookback_days: int,
    kronos_cfg: dict | None,
) -> LiveSignal:
    from pathlib import Path

    from agent.config import load_config
    from models.ensemble_strategy import (
        kronos_cfg_from_agent_config,
        score_watchlist,
        settings_from_agent_config,
    )
    from models.bayesian_changepoint import changepoint_available
    from models.hmm_strategy import hmm_available
    from models.kronos_engine import kronos_available
    from models.tft_strategy import tft_available

    agent_cfg, _ = load_config(Path(__file__).resolve().parent.parent)
    settings = settings_from_agent_config(agent_cfg)
    k_cfg = kronos_cfg_from_agent_config(agent_cfg)

    missing = []
    if not kronos_available():
        missing.append("Kronos")
    if not hmm_available():
        missing.append("HMM (hmmlearn)")
    third = settings.third_leg_model
    if third == "tft" and not tft_available():
        missing.append("TFT (darts)")
    if third == "bayesian_changepoint" and not changepoint_available():
        missing.append("Bayesian CP (ruptures)")

    history = fetch_history(watchlist, max(lookback_days, settings.hmm_lookback + 10))
    decision = score_watchlist(history, settings, kronos_cfg=k_cfg, live=True)

    if missing and not decision.target_ticker:
        rationale = f"Ensemble incomplete — missing: {', '.join(missing)}. {decision.rationale}"
    else:
        rationale = decision.rationale
        if missing:
            rationale += f" Note: renormalized without {', '.join(missing)}."

    action = "buy" if decision.target_ticker else "hold"
    return LiveSignal(
        strategy="ensemble_weighted",
        target_ticker=decision.target_ticker,
        previous_ticker=None,
        action=action,
        rationale=rationale,
        in_trend=decision.target_ticker is not None,
    )
