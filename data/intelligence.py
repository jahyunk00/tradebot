"""Market intelligence — news, fundamentals, benchmarks for retail investors."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

from data.market_data import latest_quotes


def _period_return(ticker: str, days: int) -> float | None:
    hist = yf.Ticker(ticker).history(period=f"{days + 5}d")
    if hist.empty or len(hist) < 2:
        return None
    start_idx = max(0, len(hist) - days - 1)
    start = float(hist["Close"].iloc[start_idx])
    end = float(hist["Close"].iloc[-1])
    if start == 0:
        return None
    return round((end - start) / start * 100, 2)


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    """Key stats a retail investor needs, without raw API jargon."""
    t = yf.Ticker(ticker)
    info = t.info or {}
    return {
        "ticker": ticker.upper(),
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap_b": round(info["marketCap"] / 1e9, 1) if info.get("marketCap") else None,
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield_pct": round(info["dividendYield"] * 100, 2) if info.get("dividendYield") else None,
        "beta": info.get("beta"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "analyst_rating": info.get("recommendationKey"),
    }


def _parse_news_item(item: dict) -> dict[str, str]:
    """Support legacy yfinance news shape and newer nested `content` payloads."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = (
        content.get("title")
        or item.get("title")
        or content.get("summary", "")[:120]
        or ""
    )
    provider = content.get("provider") or {}
    publisher = (
        provider.get("displayName")
        if isinstance(provider, dict)
        else None
    ) or item.get("publisher", "")

    published = ""
    pub_date = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
    if pub_date:
        if isinstance(pub_date, (int, float)):
            published = datetime.fromtimestamp(pub_date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            published = str(pub_date).replace("T", " ").replace("Z", " UTC")[:22]

    link = ""
    for key in ("clickThroughUrl", "canonicalUrl"):
        url_obj = content.get(key) or item.get(key)
        if isinstance(url_obj, dict) and url_obj.get("url"):
            link = url_obj["url"]
            break

    return {
        "title": str(title).strip(),
        "publisher": str(publisher).strip(),
        "published": published,
        "url": link,
    }


def fetch_news(ticker: str, limit: int = 3) -> list[dict[str, str]]:
    """Recent headlines — free public data via yfinance."""
    t = yf.Ticker(ticker)
    raw = t.news or []
    headlines: list[dict[str, str]] = []
    for item in raw[:limit]:
        parsed = _parse_news_item(item)
        if parsed["title"]:
            headlines.append(parsed)
    return headlines


def benchmark_comparison(tickers: list[str], benchmark: str = "SPY") -> dict[str, Any]:
    """Compare watchlist performance vs benchmark over common windows."""
    windows = {"1d": 1, "5d": 5, "30d": 30, "90d": 90}
    bench_returns = {label: _period_return(benchmark, days) for label, days in windows.items()}

    comparisons: dict[str, Any] = {}
    for ticker in tickers:
        ticker_returns = {label: _period_return(ticker, days) for label, days in windows.items()}
        vs_benchmark = {}
        for label in windows:
            t_ret = ticker_returns.get(label)
            b_ret = bench_returns.get(label)
            if t_ret is not None and b_ret is not None:
                vs_benchmark[label] = round(t_ret - b_ret, 2)
        comparisons[ticker.upper()] = {
            "returns": ticker_returns,
            "vs_benchmark_pct": vs_benchmark,
        }

    return {
        "benchmark": benchmark.upper(),
        "benchmark_returns": bench_returns,
        "tickers": comparisons,
    }


def build_intelligence_package(
    watchlist: list[str],
    *,
    news_per_ticker: int = 3,
    benchmark: str = "SPY",
) -> dict[str, Any]:
    """Aggregate everything the LLM needs to replace manual research."""
    quotes = latest_quotes(watchlist)
    fundamentals = {t: fetch_fundamentals(t) for t in watchlist}
    news = {t: fetch_news(t, limit=news_per_ticker) for t in watchlist}
    benchmark_data = benchmark_comparison(watchlist, benchmark)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quotes": quotes,
        "fundamentals": fundamentals,
        "news": news,
        "benchmark_comparison": benchmark_data,
    }


def save_snapshot(package: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package, indent=2, default=str))


def load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_changes(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """Highlight what changed since last run — saves the investor monitoring time."""
    if not previous:
        return {"has_previous": False, "message": "First run — no prior snapshot to compare."}

    changes: dict[str, Any] = {"has_previous": True, "price_moves": {}, "new_headlines": {}}
    prev_quotes = previous.get("quotes", {})
    curr_quotes = current.get("quotes", {})

    for ticker, curr in curr_quotes.items():
        prev = prev_quotes.get(ticker)
        if not prev:
            continue
        delta = round(curr["last_close"] - prev["last_close"], 2)
        delta_pct = round(
            (curr["last_close"] - prev["last_close"]) / prev["last_close"] * 100, 2
        ) if prev["last_close"] else 0
        if abs(delta_pct) >= 0.5:
            changes["price_moves"][ticker] = {
                "from": prev["last_close"],
                "to": curr["last_close"],
                "change_pct": delta_pct,
            }

    prev_news_titles = {
        t: {h["title"] for h in headlines}
        for t, headlines in previous.get("news", {}).items()
    }
    for ticker, headlines in current.get("news", {}).items():
        old = prev_news_titles.get(ticker, set())
        new_items = [h for h in headlines if h["title"] not in old]
        if new_items:
            changes["new_headlines"][ticker] = new_items

    prev_time = previous.get("generated_at", "unknown")
    changes["since"] = prev_time
    return changes
