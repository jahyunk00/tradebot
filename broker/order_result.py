"""Interpret Robinhood MCP place/review order responses."""

from __future__ import annotations

import re
from typing import Any


def _result_body(response: dict[str, Any]) -> Any:
    if response.get("dry_run"):
        return response.get("review")
    return response.get("result")


def order_error_message(response: dict[str, Any]) -> str | None:
    """Return a human-readable error if the broker rejected the order."""
    if response.get("dry_run"):
        return None

    raw = _result_body(response)
    if raw is None:
        return "Broker returned no order result."

    text = raw if isinstance(raw, str) else str(raw)
    lower = text.lower()

    if "api error" in lower or "blocked because" in lower or "non_field_errors" in lower:
        return text.strip()

    if isinstance(raw, dict):
        err = raw.get("error") or raw.get("errors") or raw.get("non_field_errors")
        if err:
            return str(err)

    return None


def order_succeeded(response: dict[str, Any]) -> bool:
    return order_error_message(response) is None


def extract_action_url(error_text: str) -> str | None:
    match = re.search(r"https://applink\.robinhood\.com/\S+", error_text)
    return match.group(0).rstrip(").,]") if match else None
