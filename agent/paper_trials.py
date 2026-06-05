"""Paper-trial pipeline — audition up to 2 names, auto-promote winners to live cash."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

from agent.config import AgentConfig, GuardrailsConfig
from data.trajectory import rank_by_trajectory

logger = logging.getLogger(__name__)


def _state_path(base_dir: Path) -> Path:
    return base_dir / "logs" / "paper_trials.json"


def load_trials_state(base_dir: Path) -> dict[str, Any]:
    path = _state_path(base_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"trials": [], "live_promoted": [], "history": []}


def save_trials_state(base_dir: Path, state: dict[str, Any]) -> None:
    path = _state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


def _last_price(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).info or {}
        for key in ("regularMarketPrice", "currentPrice", "previousClose"):
            if info.get(key):
                return float(info[key])
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def _model_score_for_ticker(ticker: str, combined_scores: dict[str, float] | None) -> float | None:
    if not combined_scores:
        return None
    for k, v in combined_scores.items():
        if k.upper() == ticker.upper():
            return float(v)
    return None


def run_paper_trials(
    base_dir: Path,
    agent_cfg: AgentConfig,
    guard_cfg: GuardrailsConfig,
    *,
    watchlist: list[str],
    rotation_swaps: list[dict[str, Any]],
    combined_scores: dict[str, float] | None,
    dry_run: bool = True,
    market_stress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Maintain max `max_paper_slots` paper auditions. Promote to live cash when
    paper return + model agreement criteria pass. Cap live promotions at max_live_promoted.
    """
    pt = getattr(agent_cfg, "paper_trials", None)
    if pt is None or not getattr(pt, "enabled", False):
        return {"enabled": False}

    max_paper = int(getattr(pt, "max_paper_slots", 2))
    max_live = int(getattr(pt, "max_live_promoted", 2))
    virtual_usd = float(getattr(pt, "virtual_usd", 15.0))
    min_sessions = int(getattr(pt, "min_sessions", 5))
    min_return_pct = float(getattr(pt, "min_return_pct", 3.0))
    drop_below_pct = float(getattr(pt, "drop_below_pct", -4.0))
    min_model_score = float(getattr(pt, "min_model_score", 0.25))

    is_bear = bool(market_stress and market_stress.get("is_bear"))
    if getattr(pt, "bear_fast_track", True) and is_bear:
        min_sessions = int(getattr(pt, "bear_min_sessions", 1))
        min_return_pct = float(getattr(pt, "bear_min_return_pct", 0.5))
        min_model_score = float(getattr(pt, "bear_min_model_score", 0.18))
        logger.info(
            "Bear fast-track trials (%s): min_sessions=%d min_return=%.1f%%",
            market_stress.get("label", "BEAR"),
            min_sessions,
            min_return_pct,
        )

    state = load_trials_state(base_dir)
    campaign_days = int(getattr(pt, "campaign_days", 30))
    if not state.get("campaign_started_at"):
        state["campaign_started_at"] = datetime.now(timezone.utc).isoformat()
        state["campaign_days"] = campaign_days
    trials: list[dict[str, Any]] = state.get("trials") or []
    live_promoted: list[dict[str, Any]] = state.get("live_promoted") or []

    promoted_this_run: list[dict[str, Any]] = []
    dropped_this_run: list[dict[str, Any]] = []

    # Update existing paper trials
    active_trials: list[dict[str, Any]] = []
    for trial in trials:
        if trial.get("status") != "paper":
            continue
        ticker = trial["ticker"].upper()
        price = _last_price(ticker)
        entry = float(trial.get("entry_price") or 0)
        if price and entry > 0:
            trial["return_pct"] = round((price / entry - 1) * 100, 2)
            trial["last_price"] = price
        trial["sessions"] = int(trial.get("sessions", 0)) + 1

        model_score = _model_score_for_ticker(ticker, combined_scores)
        trial["last_model_score"] = model_score

        sessions = trial["sessions"]
        ret = float(trial.get("return_pct") or 0)

        if sessions >= min_sessions and ret >= min_return_pct and (model_score or 0) >= min_model_score:
            if len(live_promoted) < max_live:
                trial["status"] = "promoted"
                trial["promoted_at"] = datetime.now(timezone.utc).isoformat()
                entry_info = {
                    "ticker": ticker,
                    "promoted_at": trial["promoted_at"],
                    "paper_return_pct": ret,
                    "sessions": sessions,
                    "model_score": model_score,
                }
                live_promoted.append(entry_info)
                promoted_this_run.append(entry_info)
                continue

        if sessions >= min_sessions and ret < drop_below_pct:
            trial["status"] = "dropped"
            trial["dropped_at"] = datetime.now(timezone.utc).isoformat()
            dropped_this_run.append({"ticker": ticker, "return_pct": ret})
            continue

        active_trials.append(trial)

    # Fill empty paper slots from rotation newcomers (best trajectory first)
    active_tickers = {t["ticker"].upper() for t in active_trials}
    promoted_tickers = {p["ticker"].upper() for p in live_promoted}
    used = active_tickers | promoted_tickers

    newcomers = [s["in"] for s in rotation_swaps if s.get("in")]
    if not newcomers:
        candidates = rank_by_trajectory([t for t in watchlist if t.upper() not in used])
        newcomers = [c["ticker"] for c in candidates]

    slots_open = max_paper - len(active_trials)
    for ticker in newcomers:
        if slots_open <= 0:
            break
        t = ticker.upper()
        if t in used:
            continue
        price = _last_price(t)
        if not price or price <= 0:
            continue
        active_trials.append(
            {
                "ticker": t,
                "status": "paper",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "entry_price": price,
                "virtual_usd": virtual_usd,
                "sessions": 0,
                "return_pct": 0.0,
                "last_model_score": None,
            }
        )
        used.add(t)
        slots_open -= 1
        logger.info("Paper trial started: %s @ $%.2f", t, price)

    # Trim live_promoted to max (drop worst performer if over cap)
    if len(live_promoted) > max_live:
        live_promoted.sort(key=lambda x: x.get("paper_return_pct", 0))
        live_promoted = live_promoted[-max_live:]

    all_trials = active_trials + [t for t in trials if t.get("status") != "paper"]
    state["trials"] = all_trials[-20:]
    state["live_promoted"] = live_promoted
    state["history"] = (state.get("history") or [])[-50:]
    if promoted_this_run or dropped_this_run:
        state["history"].append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "promoted": promoted_this_run,
                "dropped": dropped_this_run,
            }
        )
    save_trials_state(base_dir, state)

    return {
        "enabled": True,
        "paper_trials": active_trials,
        "live_promoted": live_promoted,
        "promoted_this_run": promoted_this_run,
        "dropped_this_run": dropped_this_run,
        "virtual_usd": virtual_usd,
        "bear_fast_track": is_bear,
        "thresholds": {
            "min_sessions": min_sessions,
            "min_return_pct": min_return_pct,
            "min_model_score": min_model_score,
        },
        "market_stress": market_stress,
    }


def live_promoted_tickers(base_dir: Path) -> list[str]:
    state = load_trials_state(base_dir)
    return [p["ticker"].upper() for p in state.get("live_promoted") or []]
