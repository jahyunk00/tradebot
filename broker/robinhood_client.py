"""Robinhood Trading MCP client — read portfolio data and (when enabled) place orders."""

from __future__ import annotations

import asyncio
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
ROBINHOOD_OAUTH_SERVER_URL = os.getenv("ROBINHOOD_OAUTH_SERVER_URL", "https://agent.robinhood.com")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8765/callback")


def _handle_redirect(auth_url: str) -> None:
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
        oauth_server_url: str | None = None,
        token_path: str = ".tokens/robinhood_oauth.json",
    ) -> None:
        self.mcp_url = mcp_url or ROBINHOOD_MCP_URL
        self.oauth_server_url = oauth_server_url or ROBINHOOD_OAUTH_SERVER_URL
        self.token_path = Path(token_path)
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        storage = FileTokenStorage(self.token_path)
        oauth = OAuthClientProvider(
            server_url=self.oauth_server_url,
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
        return await self._session.call_tool(name, arguments or {})

    async def get_account_context(self) -> dict[str, Any]:
        """Best-effort fetch of account/portfolio context via available MCP tools."""
        self._ensure_connected()
        context: dict[str, Any] = {"tools_available": await self.list_tools()}

        for tool_name in context["tools_available"]:
            lower = tool_name.lower()
            if any(k in lower for k in ("portfolio", "account", "position", "balance", "buying")):
                try:
                    context[tool_name] = await self.call_tool(tool_name, {})
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", tool_name, exc)
        return context

    def _ensure_connected(self) -> None:
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")
