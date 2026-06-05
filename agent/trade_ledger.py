"""Daily trade counts — ledger file + Robinhood order sync for Railway cron."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agent.runtime_state import _logs_dir

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def _ledger_path(base_dir: Path) -> Path:
    return _logs_dir(base_dir) / "trade_ledger.json"


def load_ledger(base_dir: Path) -> dict[str, Any]:
    path = _ledger_path(base_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"trades": [], "date_et": ""}


def save_ledger(base_dir: Path, ledger: dict[str, Any]) -> None:
    path = _ledger_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, default=str))


def _today_et() -> str:
    return datetime.now(tz=ET).strftime("%Y-%m-%d")


def _parse_order_time(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        s = str(raw).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _walk_orders(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", payload)
    if isinstance(data, dict):
        for key in ("equity_orders", "orders", "results"):
            if isinstance(data.get(key), list):
                return data[key]
    if isinstance(data, list):
        return data
    return []


def trades_from_broker_payload(payload: Any) -> tuple[int, datetime | None]:
    """Count filled equity orders placed today (US/Eastern)."""
    today = _today_et()
    count = 0
    last_at: datetime | None = None
    for order in _walk_orders(payload):
        state = str(order.get("state") or order.get("status") or "").lower()
        if state and state not in ("filled", "confirmed", "executed", "complete", "completed"):
            continue
        ts = _parse_order_time(
            order.get("updated_at")
            or order.get("last_transaction_at")
            or order.get("created_at")
            or order.get("time")
        )
        if ts is None:
            continue
        if ts.astimezone(ET).strftime("%Y-%m-%d") != today:
            continue
        count += 1
        if last_at is None or ts > last_at:
            last_at = ts
    return count, last_at


def resolve_daily_trade_stats(
    base_dir: Path,
    *,
    broker_orders_payload: Any | None = None,
) -> tuple[int, datetime | None]:
    """Prefer live broker orders; fall back to local ledger for the same ET day."""
    today = _today_et()
    ledger = load_ledger(base_dir)
    if ledger.get("date_et") != today:
        ledger = {"date_et": today, "trades": []}

    local_count = len(ledger.get("trades") or [])
    local_last: datetime | None = None
    for row in ledger.get("trades") or []:
        ts = _parse_order_time(row.get("time"))
        if ts and (local_last is None or ts > local_last):
            local_last = ts

    if broker_orders_payload is not None:
        broker_count, broker_last = trades_from_broker_payload(broker_orders_payload)
        count = max(broker_count, local_count)
        last_at = broker_last
        if local_last and (last_at is None or local_last > last_at):
            last_at = local_last
        return count, last_at

    return local_count, local_last


def record_trade(
    base_dir: Path,
    *,
    ticker: str,
    side: str,
    amount_usd: float,
    executed: bool,
) -> None:
    if not executed:
        return
    today = _today_et()
    ledger = load_ledger(base_dir)
    if ledger.get("date_et") != today:
        ledger = {"date_et": today, "trades": []}
    ledger["trades"].append(
        {
            "time": datetime.now(tz=ET).isoformat(),
            "ticker": ticker.upper(),
            "side": side.lower(),
            "amount_usd": round(amount_usd, 2),
        }
    )
    save_ledger(base_dir, ledger)
