"""Pharma / biotech intelligence — news trends and small-cap scoring (no LLM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.intelligence import fetch_fundamentals, fetch_news

POSITIVE_KEYWORDS = (
    "approval",
    "approved",
    "fda",
    "breakthrough",
    "phase 3",
    "phase iii",
    "positive",
    "beat",
    "exceed",
    "partnership",
    "license",
    "acquisition",
    "trial success",
    "efficacy",
    "orphan drug",
    "fast track",
    "priority review",
)

NEGATIVE_KEYWORDS = (
    "failed",
    "failure",
    "delay",
    "delayed",
    "halt",
    "halted",
    "rejection",
    "rejected",
    "warning letter",
    "lawsuit",
    "declined",
    "discontinued",
    "safety concern",
    "adverse",
    "miss",
    "misses",
    "downgrade",
    "bankruptcy",
)


@dataclass
class PharmaTrendReport:
    ticker_scores: dict[str, float] = field(default_factory=dict)
    news_scores: dict[str, float] = field(default_factory=dict)
    market_cap_b: dict[str, float | None] = field(default_factory=dict)
    small_cap_boost: dict[str, float] = field(default_factory=dict)
    headlines: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    sector_trend: str = ""
    trending_up: list[str] = field(default_factory=list)
    trending_down: list[str] = field(default_factory=list)


def _headline_sentiment(title: str) -> float:
    text = title.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(pos + neg, 1)


def score_pharma_news(ticker: str, *, headline_limit: int = 8) -> tuple[float, list[dict[str, str]]]:
    """
    Score recent headlines for pharma catalyst language.
    Returns score in [-1, 1] and headline list.
    """
    headlines = fetch_news(ticker, limit=headline_limit)
    if not headlines:
        return 0.0, []

    sentiments = [_headline_sentiment(h.get("title", "")) for h in headlines]
    # Recency-weighted: newer headlines count more
    weights = [1.0 / (i + 1) for i in range(len(sentiments))]
    total_w = sum(weights)
    score = sum(s * w for s, w in zip(sentiments, weights)) / total_w if total_w else 0.0
    return round(float(score), 4), headlines


def small_cap_multiplier(
    market_cap_b: float | None,
    *,
    max_cap_b: float,
    boost_strength: float,
) -> float:
    """Higher multiplier for smaller market caps (within watchlist)."""
    if market_cap_b is None or market_cap_b <= 0 or max_cap_b <= 0:
        return 1.0
    if market_cap_b >= max_cap_b:
        return 1.0
    # Linear boost: smallest cap → 1 + boost_strength
    fraction = 1.0 - (market_cap_b / max_cap_b)
    return round(1.0 + boost_strength * fraction, 4)


def analyze_pharma_watchlist(
    watchlist: list[str],
    *,
    max_market_cap_b: float = 20.0,
    small_cap_boost: float = 0.15,
    news_headlines: int = 8,
    news_weight: float = 0.20,
) -> PharmaTrendReport:
    """
    Build pharma trend report: news sentiment + small-cap preference + combined scores.
    """
    report = PharmaTrendReport()
    raw_news: dict[str, float] = {}

    for ticker in watchlist:
        t = ticker.upper()
        fund = fetch_fundamentals(t)
        cap = fund.get("market_cap_b")
        report.market_cap_b[t] = cap
        report.small_cap_boost[t] = small_cap_multiplier(
            cap, max_cap_b=max_market_cap_b, boost_strength=small_cap_boost
        )

        news_score, headlines = score_pharma_news(t, headline_limit=news_headlines)
        report.headlines[t] = headlines
        report.news_scores[t] = news_score
        raw_news[t] = news_score

    if raw_news:
        ranked = sorted(raw_news.items(), key=lambda x: x[1], reverse=True)
        mid = len(ranked) // 2
        report.trending_up = [t for t, s in ranked[: max(1, mid)] if s > 0]
        report.trending_down = [t for t, s in ranked if s < 0]

    pos_count = sum(1 for s in raw_news.values() if s > 0.1)
    neg_count = sum(1 for s in raw_news.values() if s < -0.1)
    if pos_count > neg_count:
        report.sector_trend = "Pharma/biotech news skews positive — catalyst activity elevated."
    elif neg_count > pos_count:
        report.sector_trend = "Pharma/biotech news skews negative — caution on trial/regulatory headlines."
    else:
        report.sector_trend = "Pharma/biotech news mixed — rely on price legs + small-cap filter."

    # Normalize news to 0–1 for blending with boss leg scores
    for ticker in watchlist:
        t = ticker.upper()
        ns = raw_news.get(t, 0.0)
        norm_news = (ns + 1.0) / 2.0  # map [-1,1] → [0,1]
        report.ticker_scores[t] = round(norm_news, 4)

    return report


def apply_pharma_overlay(
    combined: dict[str, float],
    watchlist: list[str],
    *,
    max_market_cap_b: float,
    small_cap_boost: float,
    news_weight: float,
    news_headlines: int = 8,
) -> tuple[dict[str, float], PharmaTrendReport]:
    """
    Blend leg-based combined scores with pharma news + small-cap preference.
    """
    report = analyze_pharma_watchlist(
        watchlist,
        max_market_cap_b=max_market_cap_b,
        small_cap_boost=small_cap_boost,
        news_headlines=news_headlines,
        news_weight=news_weight,
    )

    leg_weight = 1.0 - news_weight
    adjusted: dict[str, float] = {}

    for ticker in combined:
        t = ticker.upper()
        news_norm = report.ticker_scores.get(t, 0.5)
        base = combined[ticker]
        blended = leg_weight * base + news_weight * news_norm
        blended *= report.small_cap_boost.get(t, 1.0)
        adjusted[t] = round(blended, 4)

    return adjusted, report
