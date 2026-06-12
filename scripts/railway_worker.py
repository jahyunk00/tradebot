#!/usr/bin/env python3
"""Always-on Railway worker — trade on an interval during US market hours."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sleep_seconds() -> int:
    raw = os.getenv("TRADE_LOOP_SECONDS", "300").strip()
    try:
        return max(int(raw), 60)
    except ValueError:
        return 300


def _write_heartbeat(payload: dict) -> None:
    from agent.runtime_state import _logs_dir

    path = _logs_dir(ROOT) / "cron_heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload, indent=2, default=str))


async def _run_once() -> dict:
    from scripts.railway_trade import _bootstrap, _run

    _bootstrap()
    return {"exit_code": await _run(), "finished_at": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    from agent.guardrails import is_us_market_hours

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("railway_worker")
    interval = _sleep_seconds()
    logger.info("Trade worker started — interval %ss during market hours", interval)

    while True:
        if not is_us_market_hours():
            logger.info("Outside market hours — sleeping 5 min")
            time.sleep(300)
            continue
        try:
            outcome = asyncio.run(_run_once())
            _write_heartbeat({"mode": "worker", "interval_sec": interval, **outcome})
            logger.info("Worker cycle complete: %s", outcome)
        except Exception:
            logger.exception("Worker cycle failed")
            _write_heartbeat(
                {
                    "mode": "worker",
                    "interval_sec": interval,
                    "error": "cycle_failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        time.sleep(interval)


if __name__ == "__main__":
    main()
