"""Plain-English definitions injected into prompts for literacy support."""

GLOSSARY: dict[str, str] = {
    "CAGR": "Compound Annual Growth Rate — average yearly growth if returns were smoothed over time.",
    "Sharpe ratio": "Return per unit of risk. Above 1.0 is generally good; below 0.5 suggests weak risk-adjusted performance.",
    "Calmar ratio": "Return divided by worst drop (drawdown). Higher means strong growth with smaller crashes.",
    "Drawdown": "Peak-to-trough decline — how far an investment fell from its high before recovering.",
    "P/E ratio": "Price divided by earnings. Higher often means investors expect faster growth; compare within same sector.",
    "Market cap": "Total company value on the stock market (share price × shares outstanding).",
    "Beta": "How much a stock moves vs the overall market. Beta > 1 = more volatile than the market.",
    "Diversification": "Spreading money across different stocks/sectors so one bad pick doesn't sink the portfolio.",
    "Concentration risk": "Too much money in one stock or sector — a single bad event hurts badly.",
    "Benchmark": "A standard to compare against, usually SPY (S&P 500) for US stocks.",
    "Volatility": "How much prices swing up and down. Higher volatility = bumpier ride.",
}


def glossary_block(terms: list[str] | None = None) -> str:
    """Return glossary text for prompt injection."""
    keys = terms or list(GLOSSARY.keys())
    lines = [f"- **{k}**: {GLOSSARY[k]}" for k in keys if k in GLOSSARY]
    return "Financial terms reference:\n" + "\n".join(lines)
