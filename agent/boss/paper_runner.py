"""Paper trading — boss practices without real orders."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agent.boss.agent import (
    combine_leg_reports,
    decide_from_history,
    load_boss_weights,
    save_boss_weights,
    update_weights_from_paper_outcomes,
    weights_from_config,
)
from agent.config import load_config
from agent.legs.runner import run_all_legs
from agent.runtime_state import append_progress
from data.market_data import fetch_history

logger = logging.getLogger(__name__)


def run_historical_paper_practice(
    base_dir: Path,
    *,
    update_weights: bool = True,
) -> dict[str, Any]:
    """
    Walk forward on past market data: each rebalance date the boss picks,
    simulates forward P&L, and optionally nudges weights online.
    """
    agent_cfg, _ = load_config(base_dir)
    tr = agent_cfg.boss.training
    weights_path = base_dir / agent_cfg.boss.weights_path
    weights = load_boss_weights(weights_path, fallback=weights_from_config(agent_cfg))

    history = fetch_history(agent_cfg.watchlist, tr.lookback_days)
    closes = pd.DataFrame({t: history[t]["Close"] for t in agent_cfg.watchlist if t in history})
    min_rows = max(252, tr.warmup_days) + 10
    rebalance = tr.rebalance_days
    forward = tr.forward_days

    equity = float(agent_cfg.backtest.initial_capital)
    outcomes: list[dict[str, Any]] = []
    period_log: list[dict[str, Any]] = []

    for i, idx in enumerate(closes.index):
        if i < min_rows or i % rebalance != 0:
            continue
        if i + forward >= len(closes.index):
            break

        slice_hist = {
            t: history[t].loc[:idx]
            for t in agent_cfg.watchlist
            if t in history and len(history[t].loc[:idx]) >= min_rows
        }
        if len(slice_hist) < 2:
            continue

        reports = run_all_legs(slice_hist, agent_cfg, live=False)
        decision = combine_leg_reports(
            reports,
            weights,
            min_combined_score=agent_cfg.boss.min_combined_score,
        )
        if not decision.target_ticker:
            period_log.append({"date": str(idx.date()), "pick": None, "pnl_pct": 0.0})
            continue

        pick = decision.target_ticker.upper()
        fwd_idx = closes.index[i + forward]
        p0 = float(closes.loc[idx, pick])
        p1 = float(closes.loc[fwd_idx, pick])
        if p0 <= 0:
            continue
        pnl_pct = (p1 / p0 - 1) * 100
        equity *= 1 + pnl_pct / 100

        leg_tops = {r["agent_id"]: r.get("top_ticker") for r in decision.leg_reports}
        outcome = {"pick": pick, "pnl_pct": pnl_pct, "leg_tops": leg_tops}
        outcomes.append(outcome)
        period_log.append({"date": str(idx.date()), "pick": pick, "pnl_pct": round(pnl_pct, 3)})

        if update_weights:
            weights = update_weights_from_paper_outcomes(
                weights,
                [outcome],
                learning_rate=agent_cfg.boss.learning_rate,
            )

    if update_weights and outcomes:
        save_boss_weights(weights_path, weights)

    total_return = (equity / agent_cfg.backtest.initial_capital - 1) * 100
    result = {
        "mode": "historical_paper_practice",
        "periods": len(period_log),
        "trades": len(outcomes),
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_return, 2),
        "boss_weights": weights.as_dict(),
        "weights_source": weights.source,
        "period_log": period_log[-20:],
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = base_dir / agent_cfg.logging.directory / f"paper_practice_{run_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def _load_portfolio(path: Path, initial_cash: float) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "cash_usd": initial_cash,
        "positions": {},
        "equity_usd": initial_cash,
        "trades": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_portfolio(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


def _mark_to_market(state: dict[str, Any], quotes: dict[str, float]) -> float:
    cash = float(state["cash_usd"])
    pos_val = sum(float(qty) * quotes.get(t, 0) for t, qty in state.get("positions", {}).items())
    return round(cash + pos_val, 2)


def run_paper_session(
    base_dir: Path,
    *,
    update_weights: bool = False,
) -> dict[str, Any]:
    agent_cfg, guard_cfg = load_config(base_dir)
    weights_path = base_dir / agent_cfg.boss.weights_path
    paper_path = base_dir / agent_cfg.boss.paper_portfolio_path

    weights = load_boss_weights(weights_path, fallback=weights_from_config(agent_cfg))
    portfolio = _load_portfolio(paper_path, guard_cfg.bankroll.initial_usd)

    lookback = max(
        agent_cfg.backtest.lookback_days,
        agent_cfg.hmm.lookback + 10,
        agent_cfg.boss.training.warmup_days,
    )
    history = fetch_history(agent_cfg.watchlist, lookback)
    decision = decide_from_history(history, agent_cfg, weights)

    quotes = {
        t.upper(): float(history[t]["Close"].iloc[-1])
        for t in agent_cfg.watchlist
        if t in history and not history[t].empty
    }
    equity_before = _mark_to_market(portfolio, quotes)

    trade_log: dict[str, Any] | None = None
    if decision.target_ticker:
        target = decision.target_ticker.upper()
        g = guard_cfg
        max_usd = min(
            portfolio["cash_usd"] * (g.max_position_pct / 100),
            portfolio["cash_usd"],
        )
        if g.max_order_usd:
            max_usd = min(max_usd, g.max_order_usd)

        # Exit other positions
        for ticker in list(portfolio.get("positions", {}).keys()):
            if ticker == target:
                continue
            qty = portfolio["positions"].pop(ticker)
            proceeds = qty * quotes.get(ticker, 0)
            portfolio["cash_usd"] = round(float(portfolio["cash_usd"]) + proceeds, 2)
            portfolio["trades"].append(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "side": "sell",
                    "ticker": ticker,
                    "usd": round(proceeds, 2),
                    "paper": True,
                }
            )

        price = quotes.get(target, 0)
        if price > 0 and max_usd >= 1:
            current_val = portfolio["positions"].get(target, 0) * price
            buy_usd = round(max(max_usd - current_val, 0), 2)
            buy_usd = min(buy_usd, portfolio["cash_usd"])
            if buy_usd >= 1:
                shares = buy_usd / price
                portfolio["cash_usd"] = round(float(portfolio["cash_usd"]) - buy_usd, 2)
                portfolio["positions"][target] = portfolio["positions"].get(target, 0) + shares
                trade_log = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "side": "buy",
                    "ticker": target,
                    "usd": buy_usd,
                    "shares": round(shares, 6),
                    "paper": True,
                }
                portfolio["trades"].append(trade_log)

    equity_after = _mark_to_market(portfolio, quotes)
    pnl_pct = round((equity_after - equity_before) / equity_before * 100, 4) if equity_before else 0
    portfolio["equity_usd"] = equity_after
    portfolio["last_run"] = datetime.now(timezone.utc).isoformat()
    _save_portfolio(paper_path, portfolio)

    leg_tops = {r["agent_id"]: r.get("top_ticker") for r in decision.leg_reports}
    outcome = {
        "pick": decision.target_ticker,
        "pnl_pct": pnl_pct,
        "equity_usd": equity_after,
        "leg_tops": leg_tops,
    }

    if update_weights and decision.target_ticker:
        new_weights = update_weights_from_paper_outcomes(
            weights,
            [outcome],
            learning_rate=agent_cfg.boss.learning_rate,
        )
        save_boss_weights(weights_path, new_weights)
        weights = new_weights

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result = {
        "run_id": run_id,
        "mode": "paper",
        "decision": {
            "target": decision.target_ticker,
            "rationale": decision.rationale,
            "combined_scores": decision.combined_scores,
            "weights_used": decision.weights_used,
            "leg_reports": decision.leg_reports,
            "executive": decision.executive,
        },
        "boss_weights": weights.as_dict(),
        "weights_source": weights.source,
        "portfolio": {
            "cash_usd": portfolio["cash_usd"],
            "positions": {k: round(v, 4) for k, v in portfolio.get("positions", {}).items()},
            "equity_usd": equity_after,
            "pnl_pct_this_run": pnl_pct,
        },
        "trade": trade_log,
    }

    log_path = base_dir / agent_cfg.logging.directory / f"paper_{run_id}.json"
    log_path.write_text(json.dumps(result, indent=2, default=str))

    append_progress(
        base_dir,
        equity_usd=equity_after,
        mode="paper",
        pick=decision.target_ticker,
    )
    return result
