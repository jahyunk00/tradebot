"""Minimal dashboard — progress chart, active investing toggle, paper run."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from agent.config import load_config
from agent.launch_schedule import resolve_trading_phase
from agent.runtime_state import (
    append_progress,
    load_progress,
    load_state,
    set_active_investing,
)

st.set_page_config(page_title="Boss Agent", layout="wide")

agent_cfg, guard_cfg = load_config(ROOT)
phase = resolve_trading_phase(agent_cfg, guard_cfg)
initial = guard_cfg.bankroll.initial_usd

# Seed chart with starting equity if empty
if not load_progress(ROOT):
    append_progress(ROOT, equity_usd=initial, mode="start", pick=None)

st.title("Boss Agent")
st.caption("3 workers → boss decides")

trial_cap = guard_cfg.max_order_usd
trial_pct = guard_cfg.max_position_pct
pilot_cap = phase.bankroll_ceiling_usd if phase.name == "pilot" else None
if pilot_cap:
    st.info(
        f"**Pilot mode until Monday:** live orders sized to **${pilot_cap:.0f}** bankroll cap "
        f"({trial_pct:.0f}% max per position). Max {guard_cfg.max_daily_trades} trades/day."
    )
elif trial_cap:
    st.info(
        f"**Live trial cap:** ${trial_cap:.0f} max per order ({trial_pct:.0f}% of ~${initial:.0f} account). "
        f"Max {guard_cfg.max_daily_trades} trade/day · {guard_cfg.max_open_positions} open position."
    )

# --- Progress chart ---
history = load_progress(ROOT)
if history:
    df = pd.DataFrame(history)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    st.subheader("Progress")
    st.line_chart(df.set_index("time")["equity_usd"], height=320)
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${latest['equity_usd']:.2f}")
    c2.metric("Return", f"{(latest['equity_usd'] / initial - 1) * 100:+.1f}%")
    c3.metric("Last pick", latest.get("pick") or "cash")
else:
    st.info("Run paper practice to start the chart.")

st.divider()

# --- Active investing toggle ---
state = load_state(ROOT)
investing = st.toggle(
    "Active investing",
    value=state.get("active_investing", False),
    help="OFF = paper practice only. ON = boss may place real Robinhood orders when you run.",
)

if investing != state.get("active_investing"):
    set_active_investing(ROOT, investing)
    st.rerun()

if investing:
    cap_label = (
        f"${pilot_cap:.0f} pilot bankroll"
        if pilot_cap
        else (f"${trial_cap:.0f} per order" if trial_cap else "full bankroll")
    )
    st.warning(f"Live investing ON — sizing: **{cap_label}** ({trial_pct:.0f}% max per position).")
else:
    st.success("Paper mode — no real orders.")

# --- Run ---
if st.button("Paper run practice", type="primary", use_container_width=True):
    with st.spinner("Boss + 3 workers..."):
        try:
            if investing:
                from agent.boss_trader import BossTrader

                result = asyncio.run(BossTrader(ROOT, dry_run=False).run())
                pick = result["signal"]["target_ticker"] or "cash"
                eq = result["bankroll"]["equity_usd"]
                st.session_state["last_executive"] = result.get("executive")
                st.session_state["last_msg"] = f"Live: {pick} · ${eq:.2f}"
            else:
                from agent.boss.paper_runner import run_paper_session

                result = run_paper_session(ROOT, update_weights=True)
                pick = result["decision"]["target"] or "cash"
                eq = result["portfolio"]["equity_usd"]
                st.session_state["last_executive"] = result["decision"].get("executive")
                st.session_state["last_msg"] = f"Paper: {pick} · ${eq:.2f}"
        except Exception as exc:
            st.session_state["last_msg"] = f"Error: {exc}"
    st.rerun()

if investing:
    st.caption("Active investing ON — **Paper run practice** will place real orders.")
if msg := st.session_state.get("last_msg"):
    st.caption(msg)

exec_plan = st.session_state.get("last_executive")
if exec_plan:
    stress = exec_plan.get("market_stress") or {}
    kind = exec_plan.get("signal_kind")
    if stress or kind:
        st.caption(
            f"Market: {stress.get('label', '—')} · "
            f"Signal: **{kind or '—'}** · "
            f"{exec_plan.get('signal_note', '')}"
        )
if exec_plan and exec_plan.get("trade_plan"):
    tp = exec_plan["trade_plan"]
    st.subheader("Executive trade plan")
    st.write(exec_plan.get("strategist_summary", ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"${tp['entry']:.2f}")
    c2.metric("Stop loss", f"${tp['stop_loss']:.2f}")
    c3.metric("Target 1", f"${tp['take_profit_1']:.2f}")
    c4.metric("Target 2/3", f"${tp['take_profit_2']:.2f} / ${tp['take_profit_3']:.2f}")
    st.caption(
        f"Regime: {exec_plan.get('regime')} · MACD: {exec_plan.get('macd')} · "
        f"RSI: {exec_plan.get('rsi')} · {exec_plan.get('volume', '')}"
    )
