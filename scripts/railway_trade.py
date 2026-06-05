#!/usr/bin/env python3
"""Railway cron entrypoint — run one live boss trade and exit cleanly."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _bootstrap() -> None:
    """Seed logs, OAuth tokens, and active-investing flag for ephemeral cron runs."""
    from agent.runtime_state import _logs_dir

    logs = _logs_dir(ROOT)
    logs.mkdir(parents=True, exist_ok=True)

    tokens_dir = ROOT / ".tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    token_path = tokens_dir / "robinhood_oauth.json"

    b64 = os.getenv("ROBINHOOD_OAUTH_B64", "").strip()
    raw_json = os.getenv("ROBINHOOD_OAUTH_JSON", "").strip()
    if b64 and not token_path.exists():
        token_path.write_text(base64.b64decode(b64).decode("utf-8"))
    elif raw_json and not token_path.exists():
        token_path.write_text(raw_json)

    weights_path = logs / "boss_weights.json"
    if not weights_path.exists():
        seed = ROOT / "deploy" / "railway" / "boss_weights.json"
        if seed.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_text(seed.read_text())

    state_path = logs / "runtime_state.json"
    if not state_path.exists():
        active = os.getenv("ACTIVE_INVESTING", "false").lower() in ("1", "true", "yes")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "active_investing": active,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "railway_bootstrap",
                },
                indent=2,
            )
        )


def _notify(result: dict) -> None:
    try:
        from agent.notify import send_email

        sig = result.get("signal", {})
        br = result.get("bankroll", {})
        lines = [
            f"Pick: {sig.get('target_ticker') or 'CASH'}",
            f"Reason: {sig.get('rationale', '')[:500]}",
            f"Equity: ${br.get('equity_usd', '?')}",
            f"Mode: {result.get('mode')} · Dry-run: {result.get('dry_run')}",
        ]
        if result.get("execute_block_reasons"):
            lines.append("Blocked:")
            lines.extend(f"  - {r}" for r in result["execute_block_reasons"])
        for step in result.get("trade_plan") or []:
            intent = step.get("intent", {})
            status = "EXECUTED" if step.get("executed") else "BLOCKED"
            lines.append(
                f"[{status}] {intent.get('side', '').upper()} "
                f"{intent.get('ticker')} ${intent.get('amount_usd', 0):.2f}"
            )
        send_email(f"Tradebot Railway run — {result.get('run_id', 'cron')}", "\n".join(lines))
    except Exception as exc:
        logging.getLogger(__name__).warning("Email notify failed: %s", exc)


async def _run() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from agent.boss_trader import BossTrader

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _bootstrap()
    result = await BossTrader(ROOT, dry_run=False).run()
    _notify(result)

    if result.get("execute_block_reasons"):
        logging.info("Run finished with blocks: %s", result["execute_block_reasons"])
    else:
        logging.info("Run finished OK")
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_run()))
    except Exception:
        logging.exception("Railway trade run failed")
        try:
            from agent.notify import send_email

            send_email("Tradebot Railway run FAILED", "Check Railway logs for details.")
        except Exception:
            pass
        raise SystemExit(1)


if __name__ == "__main__":
    main()
