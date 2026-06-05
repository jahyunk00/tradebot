"""Resolve the effective watchlist (static config, screener, or rotation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from data.growth_screener import resolve_growth_watchlist
from data.universe_rotation import load_roster, rotate_roster


def resolve_watchlist(base_dir: Path, agent_cfg: AgentConfig) -> tuple[list[str], dict[str, Any]]:
    uni = getattr(agent_cfg, "universe", None)
    if uni is None or not getattr(uni, "enabled", False):
        tickers = [t.upper() for t in agent_cfg.watchlist]
        return tickers, {"source": "config_watchlist", "tickers": tickers}

    fallback = [t.upper() for t in agent_cfg.watchlist]
    seed = getattr(uni, "seed_universe", None) or None

    if getattr(uni, "rotation_enabled", True):
        roster_state = load_roster(base_dir)
        current = roster_state.get("tickers") or fallback
        tickers, rot_meta = rotate_roster(
            base_dir,
            current=current,
            universe=seed,
            min_market_cap_b=getattr(uni, "min_market_cap_b", 50.0),
            min_return_1y_pct=getattr(uni, "min_return_1y_pct", 5.0),
            pool_size=getattr(uni, "pool_size", 30),
            roster_size=getattr(uni, "top_n", 15),
            swap_margin_pct=getattr(uni, "swap_margin_pct", 2.0),
        )
        return tickers, rot_meta

    return resolve_growth_watchlist(
        base_dir,
        enabled=True,
        universe=seed,
        min_market_cap_b=getattr(uni, "min_market_cap_b", 50.0),
        min_return_1y_pct=getattr(uni, "min_return_1y_pct", 5.0),
        top_n=getattr(uni, "top_n", 15),
        lookback_days=getattr(uni, "lookback_days", 252),
        cache_hours=getattr(uni, "cache_hours", 20.0),
        fallback=fallback,
    )
