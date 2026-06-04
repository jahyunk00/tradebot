"""Persist daily LLM digests for weekly Stage-2 synthesis (ICAIF two-stage pipeline)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _digest_path(log_dir: Path, day: str) -> Path:
    return log_dir / "daily_digests" / f"{day}.json"


def save_daily_digest(log_dir: Path, digest: str, metadata: dict | None = None) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    digest_dir = log_dir / "daily_digests"
    digest_dir.mkdir(parents=True, exist_ok=True)

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "date": day,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "digest": digest,
        "metadata": metadata or {},
    }
    path = _digest_path(log_dir, day)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_recent_digests(log_dir: Path, days: int = 7) -> list[dict]:
    digest_dir = log_dir / "daily_digests"
    if not digest_dir.exists():
        return []

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days - 1)
    out: list[dict] = []
    for path in sorted(digest_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            digest_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
            if digest_date >= cutoff:
                out.append(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return out
