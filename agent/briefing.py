"""Retail investor briefing runner — Stage 1 daily digest."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.analyzer import MarketAnalyzer
from agent.config import load_config
from agent.digest_store import load_recent_digests, save_daily_digest
from broker.robinhood_client import RobinhoodMCPClient
from data.intelligence import (
    build_intelligence_package,
    compute_changes,
    load_snapshot,
    save_snapshot,
)

logger = logging.getLogger(__name__)


class BriefingRunner:
    SNAPSHOT_NAME = "latest_intelligence.json"

    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        connect_robinhood: bool = True,
    ) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.agent_config, self.guardrails_config = load_config(self.base_dir)
        self.connect_robinhood = connect_robinhood
        self.log_dir = self.base_dir / self.agent_config.logging.directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.log_dir / self.SNAPSHOT_NAME

    async def run(self) -> dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        retail = self.agent_config.retail

        previous = load_snapshot(self.snapshot_path)
        intelligence = build_intelligence_package(
            self.agent_config.watchlist,
            news_per_ticker=retail.news_headlines_per_ticker,
            benchmark=retail.benchmark_ticker,
        )
        changes = compute_changes(intelligence, previous)

        account_context: dict[str, Any] | None = None
        if self.connect_robinhood:
            account_context = await self._fetch_account_context()

        analyzer = MarketAnalyzer(
            model=self.agent_config.llm.model,
            max_tokens=self.agent_config.llm.max_tokens,
        )
        # Stage 1 only — daily information extraction
        briefing = analyzer.daily_digest(
            watchlist=self.agent_config.watchlist,
            intelligence=intelligence,
            changes=changes,
            account_context=account_context,
        )

        save_snapshot(intelligence, self.snapshot_path)
        save_daily_digest(self.log_dir, briefing, {"run_id": run_id, "stage": 1})

        result = {
            "run_id": run_id,
            "type": "briefing",
            "stage": 1,
            "intelligence": intelligence,
            "changes": changes,
            "briefing": briefing,
            "account_connected": account_context is not None,
            "digests_this_week": len(load_recent_digests(self.log_dir, days=7)),
        }

        log_path = self.log_dir / f"briefing_{run_id}.json"
        log_path.write_text(json.dumps(result, indent=2, default=str))
        md_path = self.log_dir / f"briefing_{run_id}.md"
        md_path.write_text(briefing)

        logger.info("Stage 1 briefing saved: %s", md_path)
        return result

    async def _fetch_account_context(self) -> dict[str, Any] | None:
        try:
            client = RobinhoodMCPClient(
                mcp_url=self.agent_config.robinhood.mcp_url,
                oauth_server_url=self.agent_config.robinhood.oauth_server_url,
                token_path=str(self.base_dir / ".tokens" / "robinhood_oauth.json"),
            )
            await client.connect()
            context = await client.get_account_context()
            await client.disconnect()
            return context
        except Exception as exc:
            logger.warning("Robinhood MCP unavailable: %s", exc)
            return None
