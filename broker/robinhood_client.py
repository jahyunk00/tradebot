"""Robinhood Trading MCP client — read portfolio data and (when enabled) place orders."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import webbrowser
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from broker.token_storage import FileTokenStorage

logger = logging.getLogger(__name__)

ROBINHOOD_MCP_URL = os.getenv("ROBINHOOD_MCP_URL", "https://agent.robinhood.com/mcp/trading")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8765/callback")


def _parse_tool_result(result: Any) -> Any:
    """MCP tools return TextContent blocks — parse JSON when present."""
    if not hasattr(result, "content"):
        return result
    parts: list[Any] = []
    for block in result.content:
        if hasattr(block, "text"):
            try:
                parts.append(json.loads(block.text))
            except (json.JSONDecodeError, TypeError):
                parts.append(block.text)
    if len(parts) == 1:
        return parts[0]
    return parts


def _pick_account_number(accounts_payload: Any) -> str | None:
    """Prefer the Agentic trading account; fall back to default."""
    accounts: list[dict[str, Any]] = []
    if isinstance(accounts_payload, dict):
        data = accounts_payload.get("data", accounts_payload)
        if isinstance(data, dict):
            accounts = data.get("accounts") or []
    if not accounts:
        return None

    agentic = [a for a in accounts if a.get("agentic_allowed")]
    if agentic:
        return str(agentic[0]["account_number"])

    default = [a for a in accounts if a.get("is_default")]
    if default:
        return str(default[0]["account_number"])

    active = [a for a in accounts if a.get("state") == "active" and not a.get("deactivated")]
    if active:
        return str(active[0]["account_number"])

    return str(accounts[0]["account_number"])


async def _handle_redirect(auth_url: str) -> None:
    logger.info("Opening browser for Robinhood OAuth: %s", auth_url)
    webbrowser.open(auth_url)


async def _handle_callback() -> tuple[str, str | None]:
    """Capture OAuth callback on localhost."""
    from aiohttp import web

    result: dict[str, str | None] = {"code": None, "state": None}

    async def callback(request: web.Request) -> web.Response:
        result["code"] = request.query.get("code")
        result["state"] = request.query.get("state")
        return web.Response(text="Robinhood auth complete. You can close this tab.")

    app = web.Application()
    app.router.add_get(urlparse(OAUTH_REDIRECT_URI).path or "/callback", callback)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", urlparse(OAUTH_REDIRECT_URI).port or 8765)
    await site.start()

    for _ in range(120):
        if result["code"]:
            break
        await asyncio.sleep(1)

    await runner.cleanup()
    if not result["code"]:
        raise TimeoutError("OAuth callback timed out after 120 seconds.")
    return result["code"], result["state"]


class RobinhoodMCPClient:
    """Async wrapper around Robinhood's hosted MCP server."""

    def __init__(
        self,
        mcp_url: str | None = None,
        token_path: str = ".tokens/robinhood_oauth.json",
    ) -> None:
        self.mcp_url = mcp_url or ROBINHOOD_MCP_URL
        self.token_path = Path(token_path)
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        storage = FileTokenStorage(self.token_path)
        oauth = OAuthClientProvider(
            server_url=self.mcp_url,
            client_metadata=OAuthClientMetadata(
                client_name="Trading Agent Bot",
                redirect_uris=[AnyUrl(OAUTH_REDIRECT_URI)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope="trading",
            ),
            storage=storage,
            redirect_handler=_handle_redirect,
            callback_handler=_handle_callback,
        )

        transport = await self._stack.enter_async_context(
            streamablehttp_client(self.mcp_url, auth=oauth)
        )
        read_stream, write_stream, _ = transport
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        tools = await self._session.list_tools()
        logger.info("Connected to Robinhood MCP. Tools: %s", [t.name for t in tools.tools])

    async def disconnect(self) -> None:
        await self._stack.aclose()
        self._session = None

    async def list_tools(self) -> list[str]:
        self._ensure_connected()
        tools = await self._session.list_tools()
        return [t.name for t in tools.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self._ensure_connected()
        raw = await self._session.call_tool(name, arguments or {})
        return _parse_tool_result(raw)

    async def get_account_context(self) -> dict[str, Any]:
        """Best-effort fetch of account/portfolio context via available MCP tools."""
        self._ensure_connected()
        context: dict[str, Any] = {"tools_available": await self.list_tools()}

        account_number: str | None = None
        if "get_accounts" in context["tools_available"]:
            try:
                accounts = await self.call_tool("get_accounts", {})
                context["get_accounts"] = accounts
                account_number = _pick_account_number(accounts)
                if account_number:
                    context["account_number"] = account_number
            except Exception as exc:
                logger.warning("Tool get_accounts failed: %s", exc)

        account_args = {"account_number": account_number} if account_number else {}

        for tool_name in context["tools_available"]:
            lower = tool_name.lower()
            if tool_name == "get_accounts":
                continue
            if any(k in lower for k in ("portfolio", "position", "balance", "buying")):
                try:
                    context[tool_name] = await self.call_tool(tool_name, account_args or None)
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", tool_name, exc)
        return context

    def _ensure_connected(self) -> None:
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")
