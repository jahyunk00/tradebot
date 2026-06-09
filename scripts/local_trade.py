#!/usr/bin/env python3
"""Run one live boss trade locally — no Railway, no cron middleman."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from agent.boss_trader import BossTrader
    from agent.launch_schedule import apply_phase_to_guardrails, resolve_trading_phase
    from agent.config import load_config
    from agent.runtime_state import set_active_investing

    dry_run = "--dry-run" in sys.argv
    if not dry_run:
        set_active_investing(ROOT, True)
        os.environ["ACTIVE_INVESTING"] = "true"

    agent_cfg, guard = load_config(ROOT)
    phase = resolve_trading_phase(agent_cfg, guard)
    guard = apply_phase_to_guardrails(guard, phase)

    print(f"\n=== LOCAL TRADE {'(dry-run)' if dry_run else '(LIVE)'} ===")
    print(f"Phase: {phase.name} — {phase.message}\n")

    result = asyncio.run(BossTrader(ROOT, dry_run=dry_run).run())

    sig = result.get("signal") or {}
    print(f"Pick: {sig.get('target_ticker') or 'CASH'}")
    print(f"Mode: {result.get('mode')} · Phase: {result.get('trading_phase')}")
    print(f"Equity: ${result.get('bankroll', {}).get('equity_usd', '?')}")

    if result.get("execute_block_reasons"):
        print("\nBlocked:")
        for r in result["execute_block_reasons"]:
            print(f"  - {r}")

    print("\nTrades:")
    for step in result.get("trade_plan") or []:
        intent = step.get("intent")
        if not intent:
            continue
        tag = "EXECUTED" if step.get("executed") else ("ALLOWED" if step.get("allowed") else "BLOCKED")
        print(f"  [{tag}] {intent.side.upper()} {intent.ticker} ${intent.amount_usd:.2f}")
        for reason in step.get("reasons") or []:
            print(f"         {reason}")

    run_id = result.get("run_id", "local")
    log = ROOT / "logs" / f"boss_trade_{run_id}.json"
    print(f"\nLog: {log}")
    return 0 if not result.get("execute_block_reasons") else 1


if __name__ == "__main__":
    raise SystemExit(main())
