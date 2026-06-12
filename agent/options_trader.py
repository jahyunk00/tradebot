"""Option candidate selection for boss trader."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


def _extract_chain_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            chains = data.get("results") or data.get("option_chains") or data.get("chains")
            if isinstance(chains, list) and chains:
                chain = chains[0]
                if isinstance(chain, dict):
                    return chain.get("id") or chain.get("chain_id")
            for key in ("id", "chain_id"):
                if key in data and isinstance(data[key], str):
                    return data[key]
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0].get("id") or payload[0].get("chain_id")
    return None


def _extract_instruments(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            for key in ("instruments", "results", "option_instruments"):
                if isinstance(data.get(key), list):
                    return [x for x in data[key] if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10])
    except ValueError:
        return None


def _pick_instrument(
    instruments: list[dict[str, Any]],
    *,
    underlying_price: float | None,
    side: str,
    min_days: int,
    max_days: int,
) -> dict[str, Any] | None:
    now = datetime.now(tz=ET).replace(tzinfo=None)
    filtered: list[dict[str, Any]] = []
    for ins in instruments:
        exp = _parse_date(ins.get("expiration_date"))
        if exp is None:
            continue
        dte = (exp - now).days
        if dte < min_days or dte > max_days:
            continue
        filtered.append(ins)
    if not filtered:
        return None

    def score(ins: dict[str, Any]) -> tuple[float, int]:
        strike = float(ins.get("strike_price") or 0)
        exp = _parse_date(ins.get("expiration_date")) or now
        dte = (exp - now).days
        if underlying_price is None or strike <= 0:
            return (9999.0, dte)
        if side == "call":
            moneyness = max(strike - underlying_price, 0)
        else:
            moneyness = max(underlying_price - strike, 0)
        return (abs(moneyness), dte)

    filtered.sort(key=score)
    return filtered[0]


async def plan_option_trade(
    *,
    client,
    targets: list[str],
    history: dict[str, pd.DataFrame],
    side: str,
    min_days_to_expiry: int,
    max_days_to_expiry: int,
) -> dict[str, Any] | None:
    for ticker in targets:
        t = ticker.upper()
        try:
            chain_payload = await client.call_tool("get_option_chains", {"underlying_symbol": t})
            chain_id = _extract_chain_id(chain_payload)
            if not chain_id:
                continue

            inst_payload = await client.call_tool(
                "get_option_instruments",
                {
                    "chain_id": chain_id,
                    "type": side,
                    "state": "active",
                    "tradability": "tradable",
                },
            )
            instruments = _extract_instruments(inst_payload)
            if not instruments:
                continue

            px = None
            df = history.get(t)
            if df is not None and not df.empty:
                px = float(df["Close"].iloc[-1])

            pick = _pick_instrument(
                instruments,
                underlying_price=px,
                side=side,
                min_days=min_days_to_expiry,
                max_days=max_days_to_expiry,
            )
            if not pick:
                continue

            option_id = pick.get("id") or pick.get("instrument_id")
            if not option_id:
                continue

            return {
                "ticker": t,
                "option_id": option_id,
                "expiration_date": pick.get("expiration_date"),
                "strike_price": pick.get("strike_price"),
                "type": side,
            }
        except Exception:
            continue
    return None
