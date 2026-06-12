#!/usr/bin/env python3
"""Railway cron entrypoint — run one live boss trade and exit cleanly."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _bootstrap(*, force_token_refresh: bool = False) -> None:
    """Seed logs, OAuth tokens, and active-investing flag for ephemeral cron runs."""
    from agent.runtime_state import _logs_dir

    logs = _logs_dir(ROOT)
    logs.mkdir(parents=True, exist_ok=True)

    tokens_dir = ROOT / ".tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    token_path = tokens_dir / "robinhood_oauth.json"

    b64 = os.getenv("ROBINHOOD_OAUTH_B64", "").strip()
    raw_json = os.getenv("ROBINHOOD_OAUTH_JSON", "").strip()
    on_railway = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_SERVICE_NAME"))
    # Headless Railway cannot complete OAuth in-browser — always seed from env when set.
    if b64 and (force_token_refresh or on_railway or not token_path.exists()):
        token_path.write_text(base64.b64decode(b64).decode("utf-8"))
    elif raw_json and (force_token_refresh or on_railway or not token_path.exists()):
        token_path.write_text(raw_json)

    weights_path = logs / "boss_weights.json"
    if not weights_path.exists():
        seed = ROOT / "deploy" / "railway" / "boss_weights.json"
        if seed.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_text(seed.read_text())

    state_path = logs / "runtime_state.json"
    existing: dict = {}
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    env_active = os.getenv("ACTIVE_INVESTING", "").strip().lower()
    if env_active in ("1", "true", "yes"):
        active = True
    elif env_active in ("0", "false", "no"):
        active = False
    else:
        # Default ON for Railway so cron/worker keeps trading without manual toggles.
        active = bool(existing.get("active_investing", on_railway or True))
    existing["active_investing"] = active
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    existing["source"] = "railway_env_sync"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(existing, indent=2))
    os.environ.setdefault("ACTIVE_INVESTING", "true" if active else "false")


def _write_heartbeat(payload: dict) -> None:
    from agent.runtime_state import _logs_dir

    path = _logs_dir(ROOT) / "cron_heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _summarize_run(result: dict) -> dict[str, Any]:
    executed = [
        s for s in (result.get("trade_plan") or [])
        if s.get("executed")
    ]
    blocked = [
        s for s in (result.get("trade_plan") or [])
        if not s.get("executed") and s.get("allowed") is False
    ]
    return {
        "run_id": result.get("run_id"),
        "mode": result.get("mode"),
        "can_execute": result.get("can_execute"),
        "execute_block_reasons": result.get("execute_block_reasons") or [],
        "executed_count": len(executed),
        "blocked_count": len(blocked),
        "broker_block": result.get("broker_block"),
        "active_investing": os.getenv("ACTIVE_INVESTING"),
    }


def _notify(result: dict) -> None:
    def _intent_field(intent: object, key: str, default: object = "") -> object:
        if isinstance(intent, dict):
            return intent.get(key, default)
        return getattr(intent, key, default)

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
            side = str(_intent_field(intent, "side", "")).upper()
            ticker = _intent_field(intent, "ticker", "?")
            amount = float(_intent_field(intent, "amount_usd", 0) or 0)
            lines.append(
                f"[{status}] {side} {ticker} ${amount:.2f}"
            )
        send_email(f"Tradebot Railway run — {result.get('run_id', 'cron')}", "\n".join(lines))
    except Exception as exc:
        logging.getLogger(__name__).warning("Email notify failed: %s", exc)


async def _run() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from agent.boss_trader import BossTrader
    from agent.config import load_config
    from agent.guardrails import is_us_market_hours

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Keep Railway logs stable; these warnings are noisy and non-fatal.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Some rows of transmat_ have zero sum.*")
    warnings.filterwarnings("ignore", message=".*Model is not converging.*")
    warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")

    _bootstrap()
    agent_cfg, guard_cfg = load_config(ROOT)
    if guard_cfg.enforce_market_hours and not is_us_market_hours():
        msg = "Outside US market hours — skipping run"
        logging.info(msg)
        _write_heartbeat(
            {
                "skipped": "market_closed",
                "message": msg,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return 0

    logger = logging.getLogger("railway_trade")
    logger.info(
        "Starting trade run (ACTIVE_INVESTING=%s, mode=%s)",
        os.getenv("ACTIVE_INVESTING"),
        agent_cfg.mode,
    )
    try:
        result = await BossTrader(ROOT, dry_run=False).run()
    except BaseException as exc:
        text = str(exc).lower()
        auth_like = (
            "oauth callback timed out" in text
            or "401 unauthorized" in text
            or "invalid_grant" in text
            or "authorization" in text
            or "cancelled via cancel scope" in text
            or "aclose(): asynchronous generator is already running" in text
        )
        if not auth_like:
            raise

        logging.warning(
            "Robinhood auth failed; refreshing token from Railway env and retrying once: %s",
            str(exc)[:300],
        )
        _bootstrap(force_token_refresh=True)
        try:
            result = await BossTrader(ROOT, dry_run=False).run()
        except BaseException as retry_exc:
            logging.error(
                "Robinhood auth still failing after refresh; skipping this cron run: %s",
                str(retry_exc)[:300],
            )
            try:
                from agent.notify import send_email

                send_email(
                    "Tradebot Railway auth blocked",
                    "Robinhood OAuth failed in Railway. Re-run auth locally and update ROBINHOOD_OAUTH_B64.",
                )
            except Exception:
                pass
            return 0

    _notify(result)

    summary = _summarize_run(result)
    _write_heartbeat({**summary, "finished_at": datetime.now(timezone.utc).isoformat()})
    logger.info("RAILWAY_RUN_COMPLETE %s", json.dumps(summary, default=str))

    if result.get("execute_block_reasons"):
        logging.info("Run finished with blocks: %s", result["execute_block_reasons"])
    elif summary["executed_count"] == 0 and not summary.get("broker_block"):
        logging.info("Run finished — no orders executed (plan empty or all reviewed only)")
    else:
        logging.info("Run finished OK — %s orders executed", summary["executed_count"])
    return 0


def main() -> None:
    exit_code = 0
    try:
        exit_code = int(asyncio.run(_run()) or 0)
    except BaseException:
        logging.exception("Railway trade run failed")
        try:
            from agent.notify import send_email

            send_email("Tradebot Railway run FAILED", "Check Railway logs for details.")
        except Exception:
            pass
        # Do not crash the service; cron should continue next cycle.
        exit_code = 0
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
