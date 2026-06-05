"""Place equity orders via Robinhood MCP."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from broker.robinhood_client import RobinhoodMCPClient, _pick_account_number

logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, client: RobinhoodMCPClient) -> None:
        self.client = client
        self._account_number: str | None = None

    async def ensure_account(self, account_context: dict[str, Any]) -> str:
        if self._account_number:
            return self._account_number
        accounts = account_context.get("get_accounts")
        acct = _pick_account_number(accounts)
        if not acct:
            raise RuntimeError("No agentic Robinhood account found.")
        self._account_number = acct
        return acct

    async def review_market_order(
        self,
        account_number: str,
        symbol: str,
        side: str,
        dollar_amount: float,
    ) -> dict[str, Any]:
        payload = {
            "account_number": account_number,
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": "market",
            "dollar_amount": f"{dollar_amount:.2f}",
            "time_in_force": "gfd",
            "market_hours": "regular_hours",
        }
        return await self.client.call_tool("review_equity_order", payload)

    async def place_market_order(
        self,
        account_number: str,
        symbol: str,
        side: str,
        dollar_amount: float,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "account_number": account_number,
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": "market",
            "dollar_amount": f"{dollar_amount:.2f}",
            "time_in_force": "gfd",
            "market_hours": "regular_hours",
            "ref_id": str(uuid.uuid4()),
        }

        if dry_run:
            review = await self.review_market_order(
                account_number, symbol, side, dollar_amount
            )
            return {"dry_run": True, "review": review, "payload": payload}

        logger.info("Placing %s %s $%.2f", side, symbol, dollar_amount)
        result = await self.client.call_tool("place_equity_order", payload)
        return {"dry_run": False, "result": result, "payload": payload}
