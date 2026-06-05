"""Pilot window before full launch — live trades sized to a small cap, then full bankroll."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from agent.config import AgentConfig, GuardrailsConfig

ET = ZoneInfo("America/New_York")


@dataclass
class TradingPhase:
    name: str  # pilot | live
    bankroll_ceiling_usd: float | None
    max_order_usd: float | None
    live_start_date: str
    message: str


def _today_et() -> date:
    return datetime.now(tz=ET).date()


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def resolve_trading_phase(agent_cfg: AgentConfig, guard_cfg: GuardrailsConfig) -> TradingPhase:
    sched = getattr(agent_cfg, "launch_schedule", None)
    if sched is None or not getattr(sched, "enabled", False):
        return TradingPhase(
            name="live",
            bankroll_ceiling_usd=guard_cfg.bankroll.ceiling_usd,
            max_order_usd=guard_cfg.max_order_usd,
            live_start_date="",
            message="Live rules from guardrails.yaml",
        )

    live_start = _parse_date(getattr(sched, "live_start_date", ""))
    today = _today_et()
    pilot_usd = float(getattr(sched, "pilot_bankroll_usd", 50.0))
    live_start_str = getattr(sched, "live_start_date", "")

    if live_start and today < live_start:
        return TradingPhase(
            name="pilot",
            bankroll_ceiling_usd=pilot_usd,
            max_order_usd=None,
            live_start_date=live_start_str,
            message=(
                f"Pilot until {live_start_str} — live orders sized to ${pilot_usd:.0f} bankroll cap."
            ),
        )

    max_order = guard_cfg.max_order_usd
    if getattr(sched, "live_full_bankroll", True):
        max_order = None

    return TradingPhase(
        name="live",
        bankroll_ceiling_usd=None,
        max_order_usd=max_order,
        live_start_date=live_start_str,
        message=f"Full scale from {live_start_str or 'today'} — entire bankroll (position % caps apply).",
    )


def apply_phase_to_guardrails(guard_cfg: GuardrailsConfig, phase: TradingPhase) -> GuardrailsConfig:
    updates: dict[str, Any] = {}
    bankroll_updates: dict[str, Any] = {}

    if phase.bankroll_ceiling_usd is not None:
        bankroll_updates["ceiling_usd"] = phase.bankroll_ceiling_usd
    elif phase.name == "live":
        bankroll_updates["ceiling_usd"] = None

    if phase.max_order_usd is not None:
        updates["max_order_usd"] = phase.max_order_usd
    elif phase.name == "live":
        updates["max_order_usd"] = None

    if bankroll_updates:
        updates["bankroll"] = guard_cfg.bankroll.model_copy(update=bankroll_updates)

    if updates:
        return guard_cfg.model_copy(update=updates)
    return guard_cfg
