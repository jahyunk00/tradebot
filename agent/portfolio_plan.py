"""Build trade plans for single-stock or multi-stock portfolio modes."""

from __future__ import annotations

from typing import Any

from agent.bankroll import clamp_trade_amount
from agent.guardrails import Guardrails, TradeIntent


def pick_targets(
    combined_scores: dict[str, float] | None,
    *,
    min_score: float,
    mode: str,
    max_picks: int,
) -> list[str]:
    if not combined_scores:
        return []
    ranked = sorted(combined_scores.items(), key=lambda x: -float(x[1]))
    eligible = [(t.upper(), float(s)) for t, s in ranked if float(s) >= min_score]
    if not eligible:
        return []
    if mode == "single":
        return [eligible[0][0]]
    return [t for t, _ in eligible[: max(1, max_picks)]]


def build_portfolio_trade_plan(
    targets: list[str],
    *,
    positions: dict[str, float],
    bankroll,
    guardrails: Guardrails,
    rationale: str,
    rotate_out: bool,
) -> list[dict[str, Any]]:
    """
    single + rotate_out: sell non-target, buy one (classic rotation).
    multi: buy each target up to order cap; optionally sell names dropped from list.
    """
    g = guardrails.guardrails_config
    held = {t.upper(): v for t, v in positions.items() if v > 0}
    plan: list[dict[str, Any]] = []
    target_set = {t.upper() for t in targets}

    if rotate_out:
        for ticker, value in held.items():
            if ticker in target_set:
                continue
            amount = round(value, 2)
            if amount <= 0:
                continue
            plan.append(
                {
                    "intent": TradeIntent(
                        ticker=ticker,
                        side="sell",
                        amount_usd=amount,
                        order_type="market",
                        rationale=f"Rotate out of {ticker} — not in current picks.",
                    ),
                    "kind": "exit",
                }
            )

    for ticker in targets:
        t = ticker.upper()
        allocation = bankroll.max_position_usd(g.max_position_pct)
        clamped, notes = clamp_trade_amount(
            allocation,
            t,
            "buy",
            bankroll,
            g.max_position_pct,
            g.max_order_usd,
        )
        current_in_target = held.get(t, 0.0)
        buy_amount = round(max(clamped - current_in_target, 0), 2)
        if g.max_order_usd:
            buy_amount = min(buy_amount, g.max_order_usd)
        if buy_amount < 1.0:
            continue
        r = rationale
        if notes:
            r += " " + " ".join(notes)
        plan.append(
            {
                "intent": TradeIntent(
                    ticker=t,
                    side="buy",
                    amount_usd=buy_amount,
                    order_type="market",
                    rationale=r,
                ),
                "kind": "enter",
            }
        )
    return plan
