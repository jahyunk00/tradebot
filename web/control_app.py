"""Lightweight web control panel — toggle active investing from your phone."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.runtime_state import load_state, set_active_investing

app = FastAPI(title="Tradebot Control")
PIN = os.getenv("CONTROL_PIN", "").strip()


def _page(on: bool, msg: str = "") -> str:
    status = "ON — live trades allowed at cron" if on else "OFF — paper / no live orders"
    color = "#dc2626" if on else "#16a34a"
    alert = f"<p style='color:#b45309'>{msg}</p>" if msg else ""
    pin_field = (
        f"""
        <label>PIN</label>
        <input type="password" name="pin" placeholder="Enter control PIN" required />
        """
        if PIN
        else ""
    )
    pin_check = "if(!document.querySelector('[name=pin]')?.value){alert('Set CONTROL_PIN on Railway');return false}" if PIN else ""
    return f"""<!DOCTYPE html>
<html><head>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Tradebot Control</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 420px; margin: 2rem auto; padding: 1rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 1.25rem; }}
  h1 {{ font-size: 1.35rem; margin: 0 0 0.5rem; }}
  .status {{ font-weight: 700; color: {color}; margin: 1rem 0; }}
  button {{ width: 100%; padding: 0.85rem; font-size: 1rem; border: 0; border-radius: 8px;
            cursor: pointer; margin-top: 0.5rem; }}
  .on {{ background: #dc2626; color: white; }}
  .off {{ background: #16a34a; color: white; }}
  input {{ width: 100%; padding: 0.6rem; margin: 0.5rem 0 1rem; box-sizing: border-box; }}
  small {{ color: #666; }}
</style></head>
<body>
<div class="card">
  <h1>Active investing</h1>
  <small>Boss Agent · Railway</small>
  {alert}
  <p class="status">{status}</p>
  <form method="post" action="/toggle" onsubmit="{pin_check}">
    {pin_field}
    <input type="hidden" name="enable" value="{"false" if on else "true"}" />
    <button type="submit" class="{"off" if on else "on"}">
      Turn {"OFF" if on else "ON"}
    </button>
  </form>
  <p><small>Cron reads this toggle at 9:35 AM ET (Mon–Fri). $15 max per order.</small></p>
</div>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    msg = request.query_params.get("msg", "")
    on = bool(load_state(ROOT).get("active_investing"))
    return HTMLResponse(_page(on, msg))


@app.post("/toggle")
def toggle(enable: str = Form(...), pin: str = Form(default="")) -> RedirectResponse:
    if PIN and pin != PIN:
        return RedirectResponse("/?msg=Wrong+PIN", status_code=303)
    set_active_investing(ROOT, enable.lower() in ("1", "true", "yes"))
    label = "ON" if enable.lower() in ("1", "true", "yes") else "OFF"
    return RedirectResponse(f"/?msg=Active+investing+{label}", status_code=303)
