"""Dynamic bankroll — position sizes scale with account equity and profits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BankrollSnapshot:
    """Live or configured account sizing state for one run."""

    mode: str  # fixed | dynamic
    initial_usd: float
    equity_usd: float
    cash_usd: float
    source: str  # robinhood_mcp | config_fallback
    ceiling_usd: float | None = None
    positions: dict[str, float] = field(default_factory=dict)  # ticker -> market value

    @property
    def gain_usd(self) -> float:
        return round(self.equity_usd - self.initial_usd, 2)

    @property
    def gain_pct(self) -> float:
        if self.initial_usd <= 0:
            return 0.0
        return round((self.equity_usd - self.initial_usd) / self.initial_usd * 100, 2)

    def effective_equity(self) -> float:
        if self.ceiling_usd is not None:
            return min(self.equity_usd, self.ceiling_usd)
        return self.equity_usd

    def max_position_usd(self, max_position_pct: float) -> float:
        return round(self.effective_equity() * max_position_pct / 100, 2)

    def max_order_usd(self, max_position_pct: float, absolute_cap: float | None = None) -> float:
        """Largest allowed single order given equity, concentration, and cash."""
        by_pct = self.max_position_usd(max_position_pct)
        cap = by_pct
        if absolute_cap is not None:
            cap = min(cap, absolute_cap)
        if self.cash_usd > 0:
            cap = min(cap, self.cash_usd)
        return round(max(cap, 0), 2)

    def remaining_room_usd(self, ticker: str, max_position_pct: float) -> float:
        """How much more can be added to a ticker before hitting concentration cap."""
        cap = self.max_position_usd(max_position_pct)
        current = self.positions.get(ticker.upper(), 0.0)
        return round(max(cap - current, 0), 2)

    def to_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "initial_usd": self.initial_usd,
            "equity_usd": self.equity_usd,
            "cash_usd": self.cash_usd,
            "gain_usd": self.gain_usd,
            "gain_pct": self.gain_pct,
            "source": self.source,
            "ceiling_usd": self.ceiling_usd,
            "positions": self.positions,
        }


def _find_numbers(obj: Any, path: str = "") -> list[tuple[str, float]]:
    """Walk nested MCP JSON and collect numeric fields with their key paths."""
    found: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            found.extend(_find_numbers(v, p))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(_find_numbers(item, f"{path}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        found.append((path.lower(), float(obj)))
    elif isinstance(obj, str):
        stripped = obj.strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            found.append((path.lower(), float(stripped)))
            return found
        try:
            parsed = json.loads(obj)
            found.extend(_find_numbers(parsed, path))
        except (json.JSONDecodeError, TypeError):
            pass
    return found


def _pick_amount(pairs: list[tuple[str, float]], keywords: tuple[str, ...]) -> float | None:
    for path, val in pairs:
        if val <= 0:
            continue
        if any(kw in path for kw in keywords):
            return val
    return None


def _pick_equity(pairs: list[tuple[str, float]]) -> float | None:
    """Prefer portfolio totals; ignore get_equity_positions tool paths (false 'equity' match)."""
    preferred = (
        "get_portfolio.data.total_value",
        "get_portfolio.data.equity_value",
        "portfolio_value",
        "total_value",
        "account_value",
        "net_liquid",
    )
    for token in preferred:
        for path, val in pairs:
            if val <= 0 or "positions[" in path or "get_equity_positions" in path:
                continue
            if token in path:
                return val
    return None


def _pick_cash(pairs: list[tuple[str, float]]) -> float | None:
    preferred = (
        "buying_power.buying_power",
        "unleveraged_buying_power",
        "get_portfolio.data.cash",
        "available_cash",
        "settled_cash",
        "cash_available",
    )
    for token in preferred:
        for path, val in pairs:
            if val <= 0:
                continue
            if token in path:
                return val
    return _pick_amount(pairs, ("cash", "buying_power"))


def _extract_positions(account_context: dict[str, Any]) -> dict[str, float]:
    """Best-effort parse of ticker -> market value from MCP account data."""
    positions: dict[str, float] = {}
    text = json.dumps(account_context, default=str).lower()

    # Common patterns: "symbol":"AAPL","market_value":123.45
    for match in re.finditer(
        r'"(?:symbol|ticker)"\s*:\s*"([A-Z]{1,5})".{0,120}?"(?:market_value|value|equity)"\s*:\s*([\d.]+)',
        text,
        re.IGNORECASE,
    ):
        positions[match.group(1).upper()] = float(match.group(2))

    return positions


def resolve_bankroll(
    *,
    mode: str,
    initial_usd: float,
    ceiling_usd: float | None,
    account_context: dict[str, Any] | None,
) -> BankrollSnapshot:
    """
    Resolve effective bankroll for sizing.

    dynamic: use live equity/cash from Robinhood MCP when available; profits increase limits.
    fixed: always use initial_usd from config.
    """
    if mode != "dynamic" or not account_context:
        return BankrollSnapshot(
            mode="fixed" if mode != "dynamic" else mode,
            initial_usd=initial_usd,
            equity_usd=initial_usd,
            cash_usd=initial_usd,
            source="config_fallback",
            ceiling_usd=ceiling_usd,
        )

    pairs = _find_numbers(account_context)
    equity = _pick_equity(pairs)
    cash = _pick_cash(pairs)
    positions = _extract_positions(account_context)

    if equity is None and positions:
        equity = sum(positions.values()) + (cash or 0)
    if equity is None and cash is not None:
        equity = cash
    if equity is None:
        equity = initial_usd
    if cash is None:
        cash = equity

    return BankrollSnapshot(
        mode="dynamic",
        initial_usd=initial_usd,
        equity_usd=round(equity, 2),
        cash_usd=round(cash, 2),
        source="robinhood_mcp",
        ceiling_usd=ceiling_usd,
        positions=positions,
    )


def clamp_trade_amount(
    amount_usd: float,
    ticker: str,
    side: str,
    bankroll: BankrollSnapshot,
    max_position_pct: float,
    absolute_order_cap: float | None = None,
) -> tuple[float, list[str]]:
    """Shrink an order to fit dynamic bankroll limits. Returns (amount, adjustment notes)."""
    notes: list[str] = []
    amount = amount_usd

    max_order = bankroll.max_order_usd(max_position_pct, absolute_order_cap)
    if amount > max_order:
        notes.append(f"Reduced ${amount:.2f} → ${max_order:.2f} (equity-based order cap).")
        amount = max_order

    if side.lower() == "buy":
        room = bankroll.remaining_room_usd(ticker, max_position_pct)
        if amount > room:
            notes.append(f"Reduced to ${room:.2f} (max {max_position_pct}% of ${bankroll.effective_equity():.2f} in {ticker}).")
            amount = room

    return round(max(amount, 0), 2), notes
