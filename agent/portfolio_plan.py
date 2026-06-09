"""Build trade plans for single-stock or multi-stock portfolio modes."""

from __future__ import annotations

from typing import Any

from agent.bankroll import clamp_trade_amount
from agent.guardrails import Guardrails, TradeIntent


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def pick_targets(
    combined_scores: dict[str, float] | None,
    *,
    min_score: float,
    mode: str,
    max_picks: int,
    momentum_by_ticker: dict[str, float] | None = None,
    momentum_weight: float = 0.35,
) -> list[str]:
    """
    Rank by blended boss score + recent momentum so slow legacy winners
    (e.g. already-up memory names) don't block fresher movers.
    """
    if not combined_scores:
        return []

    eligible = {t.upper(): float(s) for t, s in combined_scores.items() if float(s) >= min_score}
    if not eligible:
        return []

    model_n = _normalize(eligible)
    mom_subset = {t: momentum_by_ticker[t] for t in eligible if momentum_by_ticker and t in momentum_by_ticker}
    mom_n = _normalize(mom_subset) if mom_subset else {}

    mw = max(0.0, min(float(momentum_weight), 0.6))
    blended: dict[str, float] = {}
    for ticker, ms in model_n.items():
        if ticker in mom_n:
            blended[ticker] = (1.0 - mw) * ms + mw * mom_n[ticker]
        else:
            blended[ticker] = ms

    ranked = sorted(blended.items(), key=lambda x: -x[1])
    if mode == "single":
        return [ranked[0][0]]
    return [t for t, _ in ranked[: max(1, max_picks)]]


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
    multi + rotate_out: sell names dropped from top picks, buy up to max_picks.
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
                        rationale=f"Rotate out of {ticker} — not in current top picks.",
                    ),
                    "kind": "exit",
                }
            )

    for ticker in targets:
        t = ticker.upper()
        allocation = bankroll.max_position_usd(g.max_position_pct)
        current_in_target = held.get(t, 0.0)

        # If a target is already oversized, trim first so one winner
        # does not consume most of the bankroll.
        if current_in_target > allocation * 1.02:
            trim_amount = round(current_in_target - allocation, 2)
            if trim_amount >= 1.0:
                plan.append(
                    {
                        "intent": TradeIntent(
                            ticker=t,
                            side="sell",
                            amount_usd=trim_amount,
                            order_type="market",
                            rationale=(
                                f"Trim {t} back to {g.max_position_pct:.0f}% position cap "
                                f"for diversification."
                            ),
                        ),
                        "kind": "rebalance_trim",
                    }
                )
                current_in_target = round(current_in_target - trim_amount, 2)

        clamped, notes = clamp_trade_amount(
            allocation,
            t,
            "buy",
            bankroll,
            g.max_position_pct,
            g.max_order_usd,
        )
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
