"""LLM-powered market analyzer — two-stage ICAIF-inspired pipeline."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import anthropic

from agent.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    BRIEFING_SYSTEM_PROMPT,
    STAGE1_DAILY_SYSTEM_PROMPT,
    STAGE2_WEEKLY_SYSTEM_PROMPT,
    build_analysis_prompt,
    build_briefing_prompt,
    build_stage1_daily_prompt,
    build_stage2_weekly_prompt,
)

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required. Copy .env.example to .env and set it.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _complete(self, system: str, user_prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text if message.content else ""
        logger.info("LLM response complete (%d chars)", len(text))
        return text

    def daily_digest(
        self,
        watchlist: list[str],
        intelligence: dict[str, Any],
        changes: dict[str, Any],
        account_context: dict[str, Any] | None = None,
    ) -> str:
        """Stage 1: extract themes, risks, and stock highlights from today's data."""
        prompt = build_stage1_daily_prompt(watchlist, intelligence, changes, account_context)
        return self._complete(STAGE1_DAILY_SYSTEM_PROMPT, prompt)

    def weekly_portfolio(
        self,
        watchlist: list[str],
        daily_digests: list[dict],
        intelligence: dict[str, Any],
        backtest_summary: dict[str, Any],
        guardrail_summary: dict[str, Any],
        account_context: dict[str, Any] | None = None,
    ) -> str:
        """Stage 2: synthesize weekly portfolio view from accumulated daily digests."""
        prompt = build_stage2_weekly_prompt(
            watchlist, daily_digests, intelligence, backtest_summary, guardrail_summary, account_context
        )
        return self._complete(STAGE2_WEEKLY_SYSTEM_PROMPT, prompt)

    def analyze_two_stage(
        self,
        watchlist: list[str],
        intelligence: dict[str, Any],
        changes: dict[str, Any],
        daily_digests: list[dict],
        backtest_summary: dict[str, Any],
        guardrail_summary: dict[str, Any],
        account_context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Run Stage 1 (today) then Stage 2 (weekly synthesis). Returns (daily, weekly)."""
        daily = self.daily_digest(watchlist, intelligence, changes, account_context)
        all_digests = list(daily_digests)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not any(d.get("date") == today for d in all_digests):
            all_digests.append({"date": today, "digest": daily})

        weekly = self.weekly_portfolio(
            watchlist, all_digests, intelligence, backtest_summary, guardrail_summary, account_context
        )
        return daily, weekly

    def briefing(
        self,
        watchlist: list[str],
        intelligence: dict[str, Any],
        changes: dict[str, Any],
        account_context: dict[str, Any] | None = None,
    ) -> str:
        """Quick daily digest — Stage 1 only."""
        prompt = build_briefing_prompt(watchlist, intelligence, changes, account_context)
        return self._complete(BRIEFING_SYSTEM_PROMPT, prompt)

    def analyze(
        self,
        watchlist: list[str],
        intelligence: dict[str, Any],
        changes: dict[str, Any],
        backtest_summary: dict[str, Any],
        guardrail_summary: dict[str, Any],
        account_context: dict[str, Any] | None = None,
        daily_digests: list[dict] | None = None,
    ) -> str:
        """Full analysis — Stage 2 weekly synthesis."""
        prompt = build_analysis_prompt(
            watchlist, intelligence, changes, backtest_summary, account_context, guardrail_summary, daily_digests
        )
        return self._complete(ANALYSIS_SYSTEM_PROMPT, prompt)

    def extract_trade_intents(self, analysis: str, max_order_usd: float) -> list[dict[str, Any]]:
        """Parse structured trade hints from LLM output (best-effort)."""
        intents: list[dict[str, Any]] = []
        pattern = re.compile(
            r"\*\*Action\*\*:\s*(BUY|SELL|HOLD)\s+([A-Z]{1,5}).*?"
            r"\*\*Amount\*\*:\s*\$?([\d,]+(?:\.\d+)?)",
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(analysis):
            side, ticker, amount_str = match.groups()
            if side.upper() == "HOLD":
                continue
            amount = min(float(amount_str.replace(",", "")), max_order_usd)
            intents.append(
                {
                    "ticker": ticker.upper(),
                    "side": side.lower(),
                    "amount_usd": amount,
                    "order_type": "market",
                    "rationale": "Extracted from LLM weekly synthesis",
                }
            )
        return intents
