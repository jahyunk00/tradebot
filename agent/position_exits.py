"""Live exit rules — stop loss on dips, take profit / trailing on rises."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agent.config import AgentConfig
from agent.guardrails import TradeIntent
from agent.runtime_state import _logs_dir
from models.market_strategist import analyze_ticker


def parse_position_lots(account_context: dict[str, Any]) -> dict[str, dict[str, float]]:
    """
    Parse Robinhood MCP account context → ticker → {quantity, avg_cost, market_value}.
    """
    lots: dict[str, dict[str, float]] = {}
    text = json.dumps(account_context, default=str)

    # Structured walk when positions array is present in JSON
    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            symbol = obj.get("symbol") or obj.get("ticker")
            qty = obj.get("quantity") or obj.get("shares") or obj.get("shares_available_for_sells")
            avg = obj.get("average_buy_price") or obj.get("average_price") or obj.get("cost_basis")
            mv = obj.get("market_value") or obj.get("equity") or obj.get("value")
            if symbol and qty:
                try:
                    q = float(qty)
                    if q > 0:
                        ticker = str(symbol).upper()
                        entry: dict[str, float] = {"quantity": q}
                        if avg is not None:
                            entry["avg_cost"] = float(avg)
                        if mv is not None:
                            entry["market_value"] = float(mv)
                        lots[ticker] = entry
                except (TypeError, ValueError):
                    pass
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(account_context)

    if lots:
        return lots

    # Regex fallback
    for match in re.finditer(
        r'"(?:symbol|ticker)"\s*:\s*"([A-Z]{1,5})".{0,400}?'
        r'"(?:quantity|shares_available_for_sells)"\s*:\s*"?([\d.]+)"?',
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        ticker = match.group(1).upper()
        qty = float(match.group(2))
        if qty <= 0:
            continue
        block = match.group(0)
        avg_m = re.search(r'"average_buy_price"\s*:\s*"?([\d.]+)"?', block)
        mv_m = re.search(r'"market_value"\s*:\s*"?([\d.]+)"?', block)
        lots[ticker] = {
            "quantity": qty,
            **({"avg_cost": float(avg_m.group(1))} if avg_m else {}),
            **({"market_value": float(mv_m.group(1))} if mv_m else {}),
        }
    return lots


def _tracks_path(base_dir: Path) -> Path:
    return _logs_dir(base_dir) / "position_tracks.json"


def load_position_tracks(base_dir: Path) -> dict[str, Any]:
    path = _tracks_path(base_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_position_tracks(base_dir: Path, tracks: dict[str, Any]) -> None:
    path = _tracks_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracks, indent=2, default=str))


def _current_price(
    ticker: str,
    lot: dict[str, float],
    history: dict[str, pd.DataFrame],
) -> float | None:
    t = ticker.upper()
    if t in history and history[t] is not None and not history[t].empty:
        return float(history[t]["Close"].iloc[-1])
    mv = lot.get("market_value")
    qty = lot.get("quantity")
    if mv and qty and qty > 0:
        return float(mv) / float(qty)
    return None


def _strategist_levels(
    ticker: str,
    history: dict[str, pd.DataFrame],
) -> dict[str, float] | None:
    df = history.get(ticker.upper())
    if df is None or df.empty:
        return None
    report = analyze_ticker(ticker.upper(), df)
    if not report or not report.trade_plan:
        return None
    tp = report.trade_plan
    return {
        "stop_loss": float(tp.stop_loss),
        "take_profit_1": float(tp.take_profit_1),
        "take_profit_2": float(tp.take_profit_2),
        "take_profit_3": float(tp.take_profit_3),
    }


def build_exit_plan(
    lots: dict[str, dict[str, float]],
    *,
    positions: dict[str, float],
    history: dict[str, pd.DataFrame],
    agent_cfg: AgentConfig,
    base_dir: Path,
    executive: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Sell intents when price dips (stop) or rises (take profit / trailing stop).
    Runs before new buys so cash is freed on weakness.
    """
    rules = getattr(agent_cfg.boss, "exit_rules", None)
    if rules is None or not getattr(rules, "enabled", True):
        return []

    tracks = load_position_tracks(base_dir)
    plan: list[dict[str, Any]] = []
    exec_tp = (executive or {}).get("trade_plan") or {}

    for ticker, lot in lots.items():
        t = ticker.upper()
        market_value = positions.get(t) or lot.get("market_value") or 0.0
        if market_value <= 0:
            continue

        price = _current_price(t, lot, history)
        if not price or price <= 0:
            continue

        avg_cost = lot.get("avg_cost") or price
        pnl_pct = (price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0.0

        track = tracks.get(t, {})
        peak = float(track.get("peak_price") or price)
        if price > peak:
            peak = price
        track["peak_price"] = round(peak, 4)
        track["entry_price"] = round(float(track.get("entry_price") or avg_cost), 4)
        track["last_price"] = round(price, 4)
        track["updated_at"] = datetime.now(timezone.utc).isoformat()
        tracks[t] = track

        exit_kind: str | None = None
        sell_pct = 1.0
        reason = ""

        stop_pct = float(getattr(rules, "stop_loss_pct", 4.5))
        if pnl_pct <= -stop_pct:
            exit_kind = "stop_loss"
            reason = f"Stop loss: {pnl_pct:+.1f}% (limit −{stop_pct:.1f}%)."

        levels = _strategist_levels(t, history) if getattr(rules, "use_strategist_levels", True) else None
        if levels and price <= levels["stop_loss"]:
            exit_kind = "stop_loss"
            reason = f"Strategist stop: price ${price:.2f} ≤ ${levels['stop_loss']:.2f}."

        tp2 = float(getattr(rules, "take_profit_2_pct", 12.0))
        tp1 = float(getattr(rules, "take_profit_pct", 7.0))
        partial = float(getattr(rules, "partial_take_profit", 0.5))
        took_tp1 = bool(track.get("took_tp1"))

        if exit_kind is None:
            if pnl_pct >= tp2:
                exit_kind = "take_profit"
                reason = f"Take profit 2: {pnl_pct:+.1f}% ≥ {tp2:.1f}% target."
            elif took_tp1 and pnl_pct >= tp1:
                exit_kind = "take_profit"
                reason = f"Take profit follow-through: {pnl_pct:+.1f}%."
            elif not took_tp1 and pnl_pct >= tp1:
                exit_kind = "take_profit"
                sell_pct = partial if partial < 1.0 else 1.0
                reason = f"Take profit 1: {pnl_pct:+.1f}% ≥ {tp1:.1f}% — lock {'partial' if sell_pct < 1 else 'full'}."

        if exit_kind is None and levels and price >= levels["take_profit_1"]:
            exit_kind = "take_profit"
            sell_pct = partial if partial < 1.0 and not took_tp1 else 1.0
            reason = f"Strategist target 1: ${price:.2f} ≥ ${levels['take_profit_1']:.2f}."

        trail_activate = float(getattr(rules, "trailing_activate_pct", 4.0))
        trail_pct = float(getattr(rules, "trailing_stop_pct", 3.5))
        if exit_kind is None and pnl_pct >= trail_activate and peak > 0:
            drawdown_from_peak = (peak - price) / peak * 100
            if drawdown_from_peak >= trail_pct:
                exit_kind = "trailing_stop"
                reason = (
                    f"Trailing stop: −{drawdown_from_peak:.1f}% from peak ${peak:.2f} "
                    f"(trail {trail_pct:.1f}%)."
                )

        if exit_kind is None:
            continue

        amount = round(market_value * sell_pct, 2)
        if amount < 1.0:
            continue

        if exit_kind == "take_profit" and sell_pct < 1.0:
            track["took_tp1"] = True

        plan.append(
            {
                "intent": TradeIntent(
                    ticker=t,
                    side="sell",
                    amount_usd=amount,
                    order_type="market",
                    rationale=reason,
                ),
                "kind": exit_kind,
                "pnl_pct": round(pnl_pct, 2),
                "price": round(price, 2),
            }
        )

    save_position_tracks(base_dir, tracks)
    return plan
