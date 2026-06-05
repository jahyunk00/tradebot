"""Sync ACTIVE_INVESTING to the tradebot cron service via Railway GraphQL."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

RAILWAY_GRAPHQL = "https://backboard.railway.app/graphql/v2"

_VARIABLE_UPSERT = """
mutation variableUpsert($input: VariableUpsertInput!) {
  variableUpsert(input: $input)
}
"""


def railway_sync_configured() -> bool:
    return bool(
        os.getenv("RAILWAY_API_TOKEN", "").strip()
        and os.getenv("RAILWAY_PROJECT_ID", "").strip()
        and os.getenv("RAILWAY_ENVIRONMENT_ID", "").strip()
        and os.getenv("RAILWAY_CRON_SERVICE_ID", "").strip()
    )


def sync_active_investing(enabled: bool) -> tuple[bool, str]:
    """
    Push ACTIVE_INVESTING to the cron service so the next run picks it up.
    Returns (ok, message).
    """
    token = os.getenv("RAILWAY_API_TOKEN", "").strip()
    project_id = os.getenv("RAILWAY_PROJECT_ID", "").strip()
    environment_id = os.getenv("RAILWAY_ENVIRONMENT_ID", "").strip()
    service_id = os.getenv("RAILWAY_CRON_SERVICE_ID", "").strip()

    if not all([token, project_id, environment_id, service_id]):
        return False, "Railway sync skipped (local-only toggle)"

    value = "true" if enabled else "false"
    payload = {
        "query": _VARIABLE_UPSERT,
        "variables": {
            "input": {
                "projectId": project_id,
                "environmentId": environment_id,
                "serviceId": service_id,
                "name": "ACTIVE_INVESTING",
                "value": value,
            }
        },
    }

    try:
        resp = httpx.post(
            RAILWAY_GRAPHQL,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            msg = body["errors"][0].get("message", str(body["errors"]))
            logger.warning("Railway variableUpsert error: %s", msg)
            return False, f"Railway sync failed: {msg}"
        return True, f"Railway cron updated (ACTIVE_INVESTING={value})"
    except Exception as exc:
        logger.warning("Railway sync failed: %s", exc)
        return False, f"Railway sync failed: {exc}"
