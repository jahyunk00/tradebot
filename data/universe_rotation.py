"""Rotate watchlist — swap laggards for stronger trajectory names."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.growth_screener import DEFAULT_UNIVERSE, screen_large_cap_growth
from data.trajectory import rank_by_trajectory

logger = logging.getLogger(__name__)


def _roster_path(base_dir: Path) -> Path:
    return base_dir / "logs" / "universe_roster.json"


def load_roster(base_dir: Path) -> dict[str, Any]:
    path = _roster_path(base_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"tickers": [], "swaps": [], "updated_at": ""}


def save_roster(base_dir: Path, data: dict[str, Any]) -> None:
    path = _roster_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2))


def rotate_roster(
    base_dir: Path,
    *,
    current: list[str],
    universe: list[str] | None,
    min_market_cap_b: float,
    min_return_1y_pct: float,
    pool_size: int,
    roster_size: int,
    swap_margin_pct: float,
) -> tuple[list[str], dict[str, Any]]:
    """
    Start from current roster (or fresh screen). Swap out names whose trajectory
    trails the best available alternative by swap_margin_pct.
    """
    pool = universe or DEFAULT_UNIVERSE
    screen = screen_large_cap_growth(
        pool,
        min_market_cap_b=min_market_cap_b,
        min_return_1y_pct=min_return_1y_pct,
        top_n=max(pool_size, roster_size),
    )
    ranked_pool = rank_by_trajectory(screen.tickers or [t.upper() for t in pool[:pool_size]])
    by_ticker = {r["ticker"]: r for r in ranked_pool}

    roster = [t.upper() for t in current if t.upper() in by_ticker]
    if len(roster) < roster_size:
        for row in ranked_pool:
            if row["ticker"] not in roster:
                roster.append(row["ticker"])
            if len(roster) >= roster_size:
                break

    swaps: list[dict[str, Any]] = []
    roster_scores = [(t, by_ticker[t]["trajectory"]) for t in roster if t in by_ticker]
    roster_scores.sort(key=lambda x: x[1])

    for outsider in ranked_pool:
        if outsider["ticker"] in roster:
            continue
        if not roster_scores:
            break
        worst_ticker, worst_score = roster_scores[0]
        if outsider["trajectory"] >= worst_score + swap_margin_pct:
            swaps.append(
                {
                    "out": worst_ticker,
                    "in": outsider["ticker"],
                    "out_trajectory": worst_score,
                    "in_trajectory": outsider["trajectory"],
                }
            )
            roster = [outsider["ticker"] if t == worst_ticker else t for t in roster]
            roster_scores[0] = (outsider["ticker"], outsider["trajectory"])
            roster_scores.sort(key=lambda x: x[1])
        else:
            break

    roster = roster[:roster_size]
    details = [by_ticker[t] for t in roster if t in by_ticker]

    meta = {
        "source": "rotation",
        "tickers": roster,
        "details": details,
        "swaps": swaps,
        "pool_screened": len(screen.tickers),
    }
    save_roster(base_dir, {"tickers": roster, "swaps": swaps, "details": details})
    if swaps:
        logger.info(
            "Universe rotation: %s",
            ", ".join(f"{s['out']}→{s['in']}" for s in swaps),
        )
    return roster, meta
