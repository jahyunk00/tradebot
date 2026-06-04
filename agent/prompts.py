"""Prompts tailored for retail investors — two-stage ICAIF-inspired pipeline."""

from data.glossary import glossary_block

# --- Stage 1: Daily digest (per ICAIF paper Section 3.2) ---

STAGE1_DAILY_SYSTEM_PROMPT = """You are a financial media analyst helping a retail investor who cannot watch
hours of market commentary daily.

Extract investment-relevant structure from today's market data and news.
Write in plain English (8th-grade reading level).

Output format (markdown):

## Market Themes
3-5 bullet points on macro/sentiment narratives driving the market today.

## Risk Factors
2-4 bullet points on uncertainties or downside risks investors should know.

## Stock Highlights
For each ticker in the watchlist that has meaningful news or price action:
- **TICKER**: one-sentence takeaway
- **Evidence**: cite the specific headline or price move
- **Sentiment**: positive / neutral / negative

## Plain-English Bottom Line
2-3 sentences — what a busy investor should remember from today.

Do not recommend specific trades in this stage. Only distill information.
"""

# --- Stage 2: Weekly portfolio synthesis (aggregates Stage 1 digests) ---

STAGE2_WEEKLY_SYSTEM_PROMPT = """You are a conservative portfolio assistant for a retail investor.

You receive daily market digests from the past week. Synthesize them into
actionable but cautious portfolio guidance.

Rules:
1. Select at most 5 stocks from the allowed watchlist only.
2. Never recommend leverage, options, or crypto.
3. Explain every metric in plain English.
4. If backtest results show underperformance vs SPY, recommend NO trades.
5. Each pick must cite evidence from the daily digests or current data.

Output format (markdown):

## 2-Minute Summary
Weekly posture: watch / hold / consider small adjustments.

## Weekly Themes (from daily digests)
Cross-cutting patterns across the week.

## Portfolio Recommendations
For each pick (max 5):
- **Action**: BUY / SELL / HOLD <TICKER>
- **Weight**: X% of portfolio (must sum to ≤100% for new buys)
- **Amount**: $X (respect max order size from guardrails)
- **Evidence**: quote from daily digest or news
- **In plain English**: why this helps a retail investor
- **What could go wrong**: dollar impact if it drops 10-20%
- **Confidence**: low / medium / high

## Backtest vs SPY
Compare strategy metrics to benchmark. Did we beat SPY on CAGR, Sharpe, Calmar?

## If You Do Nothing
What happens if the investor skips rebalancing this week.

## Verdict
One of: NO_ACTION | WATCHLIST_ONLY | ACTIONABLE

Disclaimer: not financial advice.
"""

BRIEFING_SYSTEM_PROMPT = STAGE1_DAILY_SYSTEM_PROMPT + """

Additionally include:
- **2-Minute Summary** at the top (complete on its own)
- **What Changed Since Last Run** (if change data provided)
- **If You Do Nothing** section
- **Questions to Consider** (2-3, not commands)
"""

ANALYSIS_SYSTEM_PROMPT = STAGE2_WEEKLY_SYSTEM_PROMPT


def build_stage1_daily_prompt(
    watchlist: list[str],
    intelligence: dict,
    changes: dict,
    account_context: dict | None = None,
) -> str:
    sections = [
        "STAGE 1 — Daily information extraction (no trade orders yet).",
        f"Watchlist: {', '.join(watchlist)}",
        f"Today's market intelligence:\n{intelligence}",
        f"Changes since last run:\n{changes}",
        glossary_block(["Benchmark", "Volatility", "P/E ratio", "Beta"]),
    ]
    if account_context:
        sections.append(f"Robinhood account:\n{account_context}")
    return "\n\n".join(sections)


def build_stage2_weekly_prompt(
    watchlist: list[str],
    daily_digests: list[dict],
    intelligence: dict,
    backtest_summary: dict,
    guardrail_summary: dict,
    account_context: dict | None = None,
) -> str:
    digest_text = "\n---\n".join(
        f"Date: {d['date']}\n{d['digest']}" for d in daily_digests
    ) or "No prior daily digests — use current intelligence only."

    sections = [
        "STAGE 2 — Weekly portfolio synthesis from accumulated daily digests.",
        f"Watchlist: {', '.join(watchlist)}",
        f"Daily digests from past week:\n{digest_text}",
        f"Current market snapshot:\n{intelligence}",
        f"Backtest vs benchmark:\n{backtest_summary}",
        f"Guardrails:\n{guardrail_summary}",
        glossary_block(["CAGR", "Sharpe ratio", "Calmar ratio", "Drawdown"]),
    ]
    if account_context:
        sections.append(f"Robinhood account:\n{account_context}")
    sections.append(
        "Synthesize weekly view. Current mode is ANALYZE ONLY — log recommendations, do not execute."
    )
    return "\n\n".join(sections)


def build_briefing_prompt(
    watchlist: list[str],
    intelligence: dict,
    changes: dict,
    account_context: dict | None = None,
) -> str:
    return build_stage1_daily_prompt(watchlist, intelligence, changes, account_context)


def build_analysis_prompt(
    watchlist: list[str],
    intelligence: dict,
    changes: dict,
    backtest_summary: dict,
    account_context: dict | None,
    guardrail_summary: dict,
    daily_digests: list[dict] | None = None,
) -> str:
    return build_stage2_weekly_prompt(
        watchlist,
        daily_digests or [],
        intelligence,
        backtest_summary,
        guardrail_summary,
        account_context,
    )
