"""Runtime toggle (live investing) and progress history for the chart."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _logs_dir(base_dir: Path) -> Path:
    """Shared state dir — optional STATE_DIR (e.g. Railway volume at /data)."""
    custom = os.getenv("STATE_DIR", "").strip()
    if custom:
        p = Path(custom)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_probe"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return p
        except OSError:
            pass
    fallback = base_dir / "logs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _state_path(base_dir: Path) -> Path:
    return _logs_dir(base_dir) / "runtime_state.json"


def _progress_path(base_dir: Path) -> Path:
    return _logs_dir(base_dir) / "progress_history.json"


def load_state(base_dir: Path) -> dict[str, Any]:
    path = _state_path(base_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"active_investing": False, "updated_at": ""}


def set_active_investing(base_dir: Path, enabled: bool) -> None:
    path = _state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_state(base_dir)
    existing["active_investing"] = enabled
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(existing, indent=2))


def set_trial_mode(
    base_dir: Path,
    *,
    enabled: bool = True,
    max_order_usd: float = 15.0,
    max_position_pct: float = 10.0,
) -> None:
    path = _state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_state(base_dir)
    existing["trial"] = {
        "enabled": enabled,
        "max_order_usd": max_order_usd,
        "max_position_pct": max_position_pct,
    }
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(existing, indent=2))


def trial_limits(base_dir: Path) -> dict[str, Any]:
    trial = load_state(base_dir).get("trial") or {}
    if not trial.get("enabled"):
        return {}
    return {
        "max_order_usd": float(trial.get("max_order_usd", 15)),
        "max_position_pct": float(trial.get("max_position_pct", 10)),
    }


def is_active_investing(base_dir: Path) -> bool:
    """Env var wins when set explicitly (Railway control panel sync)."""
    env = os.getenv("ACTIVE_INVESTING", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return bool(load_state(base_dir).get("active_investing"))


def append_progress(
    base_dir: Path,
    *,
    equity_usd: float,
    mode: str,
    pick: str | None = None,
) -> None:
    path = _progress_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except json.JSONDecodeError:
            history = []

    history.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "equity_usd": round(equity_usd, 2),
            "mode": mode,
            "pick": pick,
        }
    )
    path.write_text(json.dumps(history[-500:], indent=2, default=str))


def load_progress(base_dir: Path) -> list[dict[str, Any]]:
    path = _progress_path(base_dir)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
