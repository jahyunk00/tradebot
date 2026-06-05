"""Web control panel — toggle active investing, view status, sync to Railway cron."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import load_config
from agent.runtime_state import (
    _logs_dir,
    is_active_investing,
    load_progress,
    load_state,
    set_active_investing,
)
from web.railway_sync import railway_sync_configured, sync_active_investing

app = FastAPI(title="Tradebot Control")
PIN = os.getenv("CONTROL_PIN", "").strip()


def _latest_trade_run() -> dict[str, Any] | None:
    logs = _logs_dir(ROOT)
    files = sorted(logs.glob("boss_trade_*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _fmt_usd(value: float | None, *, fallback: str = "Full bankroll (%)") -> str:
    if value is None:
        return fallback
    return f"${value:.0f}"


def _dashboard_data() -> dict[str, Any]:
    from agent.launch_schedule import apply_phase_to_guardrails, resolve_trading_phase

    agent_cfg, guard = load_config(ROOT)
    phase = resolve_trading_phase(agent_cfg, guard)
    guard = apply_phase_to_guardrails(guard, phase)
    state = load_state(ROOT)
    progress = load_progress(ROOT)
    last_run = _latest_trade_run()
    on = is_active_investing(ROOT)

    equity = guard.bankroll.initial_usd
    if progress:
        equity = float(progress[-1].get("equity_usd", equity))
    elif last_run:
        equity = float(last_run.get("bankroll", {}).get("equity_usd", equity))

    pilot_cap = phase.bankroll_ceiling_usd if phase.name == "pilot" else None

    return {
        "active": on,
        "updated_at": state.get("updated_at", ""),
        "source": state.get("source", "local"),
        "equity_usd": equity,
        "initial_usd": guard.bankroll.initial_usd,
        "max_order_usd": guard.max_order_usd,
        "pilot_bankroll_usd": pilot_cap,
        "trading_phase": phase.name,
        "phase_message": phase.message,
        "max_daily_trades": guard.max_daily_trades,
        "max_open_positions": guard.max_open_positions,
        "progress": progress[-30:],
        "last_run": last_run,
        "railway_sync": railway_sync_configured(),
        "cron_schedule": "Every 30 min · US market hours · Mon–Fri",
    }


def _fmt_time(iso: str) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%b %d, %H:%M UTC")
    except ValueError:
        return iso[:19]


def _progress_svg(points: list[dict[str, Any]], width: int = 360, height: int = 100) -> str:
    if len(points) < 2:
        return ""
    vals = [float(p.get("equity_usd", 0)) for p in points]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    coords = []
    for i, v in enumerate(vals):
        x = 8 + (width - 16) * i / (len(vals) - 1)
        y = height - 8 - (height - 16) * (v - lo) / span
        coords.append(f"{x:.1f},{y:.1f}")
    path = "M " + " L ".join(coords)
    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" aria-hidden="true">
      <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="{" ".join(coords)}"/>
    </svg>"""


def _page(data: dict[str, Any], msg: str = "") -> str:
    on = data["active"]
    status = "LIVE — cron may place real orders" if on else "PAPER — no live orders"
    status_color = "#ef4444" if on else "#22c55e"
    gain_pct = (data["equity_usd"] / data["initial_usd"] - 1) * 100 if data["initial_usd"] else 0
    alert = f"<div class='alert'>{msg}</div>" if msg else ""

    pin_field = (
        """
        <label class="label">PIN</label>
        <input type="password" name="pin" placeholder="Control PIN" required autocomplete="current-password" />
        """
        if PIN
        else ""
    )
    pin_check = (
        "if(!document.querySelector('[name=pin]')?.value){alert('Enter PIN');return false}"
        if PIN
        else ""
    )

    last = data.get("last_run") or {}
    sig = last.get("signal") or {}
    pick = sig.get("target_ticker") or "—"
    run_id = last.get("run_id") or "—"
    mode = last.get("mode") or "—"

    sync_note = (
        "Toggle syncs to Railway cron via API."
        if data["railway_sync"]
        else "Local toggle only — set RAILWAY_API_TOKEN for cloud sync."
    )

    chart = _progress_svg(data.get("progress") or [])

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Tradebot Control</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8;
    --accent: #3b82f6; --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text);
    margin: 0; padding: 1rem; line-height: 1.45;
  }}
  .wrap {{ max-width: 440px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 0.25rem; }}
  .sub {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.1rem 1.2rem; margin-bottom: 0.85rem;
  }}
  .status {{
    font-size: 1.05rem; font-weight: 700; color: {status_color};
    margin: 0.5rem 0 1rem;
  }}
  .metrics {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; }}
  .metric {{ text-align: center; }}
  .metric .val {{ font-size: 1.15rem; font-weight: 700; }}
  .metric .lbl {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; }}
  button {{
    width: 100%; padding: 0.95rem; font-size: 1.05rem; font-weight: 600;
    border: 0; border-radius: 10px; cursor: pointer; margin-top: 0.35rem;
  }}
  .btn-on {{ background: #ef4444; color: white; }}
  .btn-off {{ background: #22c55e; color: #052e16; }}
  input {{
    width: 100%; padding: 0.65rem; margin: 0.35rem 0 0.75rem;
    border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text);
  }}
  .label {{ font-size: 0.8rem; color: var(--muted); }}
  .alert {{
    background: #422006; color: #fde68a; padding: 0.6rem 0.75rem;
    border-radius: 8px; margin-bottom: 0.75rem; font-size: 0.9rem;
  }}
  .row {{ display: flex; justify-content: space-between; font-size: 0.88rem; padding: 0.25rem 0; }}
  .row span {{ color: var(--muted); }}
  small {{ color: var(--muted); font-size: 0.78rem; display: block; margin-top: 0.6rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Boss Agent</h1>
  <p class="sub">Web control · {data["cron_schedule"]}</p>

  <div class="card">
    {alert}
    <div class="status">{status}</div>
    <form method="post" action="/toggle" onsubmit="{pin_check}">
      {pin_field}
      <input type="hidden" name="enable" value="{"false" if on else "true"}" />
      <button type="submit" class="{"btn-off" if on else "btn-on"}">
        Turn {"OFF" if on else "ON"}
      </button>
    </form>
    <small>Updated {_fmt_time(data["updated_at"])} · {sync_note}</small>
  </div>

  <div class="card">
    <div class="metrics">
      <div class="metric"><div class="val">${data["equity_usd"]:.0f}</div><div class="lbl">Equity</div></div>
      <div class="metric"><div class="val">{gain_pct:+.1f}%</div><div class="lbl">Return</div></div>
      <div class="metric"><div class="val">{pick}</div><div class="lbl">Last pick</div></div>
    </div>
    {chart}
  </div>

  <div class="card">
    <strong>Guardrails</strong>
    <div class="row"><span>Phase</span>{data["trading_phase"]}</div>
    <div class="row"><span>Sizing</span>{_fmt_usd(data.get("pilot_bankroll_usd"), fallback=_fmt_usd(data["max_order_usd"], fallback="Full bankroll (position %)"))}</div>
    <div class="row"><span>Trades / day</span>{data["max_daily_trades"]}</div>
    <div class="row"><span>Open positions</span>{data["max_open_positions"]}</div>
    <div class="row"><span>Last cron run</span>{run_id}</div>
    <div class="row"><span>Mode</span>{mode}</div>
  </div>
</div>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    msg = request.query_params.get("msg", "").replace("+", " ")
    return HTMLResponse(_page(_dashboard_data(), msg))


@app.get("/api/status")
def api_status() -> JSONResponse:
    data = _dashboard_data()
    data["updated_at"] = _fmt_time(data["updated_at"])
    return JSONResponse(data)


@app.post("/toggle")
def toggle(enable: str = Form(...), pin: str = Form(default="")) -> RedirectResponse:
    if PIN and pin != PIN:
        return RedirectResponse("/?msg=Wrong+PIN", status_code=303)

    enabled = enable.lower() in ("1", "true", "yes")
    set_active_investing(ROOT, enabled)

    sync_ok, sync_msg = sync_active_investing(enabled)
    label = "ON" if enabled else "OFF"
    msg = f"Active investing {label}"
    if sync_ok:
        msg += " · synced to Railway"
    elif railway_sync_configured():
        msg += f" · {sync_msg}"

    return RedirectResponse(f"/?msg={msg.replace(' ', '+')}", status_code=303)
