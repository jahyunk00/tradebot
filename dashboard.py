"""Streamlit dashboard for the trading agent."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from agent.briefing import BriefingRunner
from agent.config import load_config
from agent.runner import TradingAgentRunner
from backtest.engine import run_backtest
from data.intelligence import build_intelligence_package, compute_changes, load_snapshot

LOG_DIR = ROOT / "logs"
SNAPSHOT_PATH = LOG_DIR / "latest_intelligence.json"


def _list_reports(prefix: str) -> list[Path]:
    if not LOG_DIR.exists():
        return []
    return sorted(LOG_DIR.glob(f"{prefix}_*.md"), reverse=True)


def _load_json_reports(prefix: str) -> list[dict]:
    reports = []
    for path in sorted(LOG_DIR.glob(f"{prefix}_*.json"), reverse=True):
        try:
            reports.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return reports


def _run_async(coro):
    return asyncio.run(coro)


st.set_page_config(
    page_title="Trading Agent",
    page_icon="📊",
    layout="wide",
)

st.title("Trading Agent Dashboard")
st.caption("Plain-English market briefings for retail investors — analyze-only by default.")

agent_cfg, guard_cfg = load_config(ROOT)

with st.sidebar:
    st.header("Controls")
    connect_robinhood = st.toggle("Connect Robinhood", value=False)
    st.divider()
    st.subheader("Run agent")
    if st.button("Daily Briefing (2 min)", use_container_width=True):
        with st.spinner("Gathering news and writing briefing..."):
            try:
                result = _run_async(
                    BriefingRunner(ROOT, connect_robinhood=connect_robinhood).run()
                )
                st.session_state["last_briefing"] = result["briefing"]
                st.session_state["last_run_id"] = result["run_id"]
                st.success(f"Briefing saved: briefing_{result['run_id']}.md")
            except Exception as exc:
                st.error(f"Briefing failed: {exc}")
                st.info("Make sure ANTHROPIC_API_KEY is set in trading-agent/.env")

    if st.button("Full Analysis (2-stage)", use_container_width=True):
        with st.spinner("Backtest → Stage 1 daily → Stage 2 weekly..."):
            try:
                result = _run_async(
                    TradingAgentRunner(ROOT, connect_robinhood=connect_robinhood).run()
                )
                st.session_state["last_daily"] = result.get("daily_digest", "")
                st.session_state["last_analysis"] = result["analysis"]
                st.session_state["last_run_id"] = result["run_id"]
                st.success(f"Saved: daily_{result['run_id']}.md + analysis_{result['run_id']}.md")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.info("Make sure ANTHROPIC_API_KEY is set in trading-agent/.env")

    st.divider()
    if st.button("Run Backtest vs SPY", use_container_width=True):
        with st.spinner("Running backtest..."):
            report = run_backtest(
                agent_cfg.watchlist,
                agent_cfg.backtest.strategy,
                agent_cfg.backtest.lookback_days,
                agent_cfg.backtest.initial_capital,
                agent_cfg.retail.benchmark_ticker,
            )
            st.session_state["backtest_report"] = report

    st.divider()
    st.subheader("Mode")
    mode = "analyze_only" if guard_cfg.force_analyze_only else agent_cfg.mode
    st.info(f"**{mode}** — no trades will execute.")

tab_home, tab_briefings, tab_analysis, tab_market, tab_backtest = st.tabs(
    ["Home", "Daily Digests", "Weekly Analysis", "Market Data", "Backtest vs SPY"]
)

with tab_home:
    st.subheader("Start here")
    st.markdown(
        """
        1. Set your **API key** in `trading-agent/.env` (`ANTHROPIC_API_KEY=...`)
        2. Click **Daily Briefing** in the sidebar for a quick 2-minute read
        3. Use **Full Analysis** weekly for deeper review with backtest context

        Reports are also saved to the `logs/` folder as markdown files.
        """
    )

    col1, col2, col3 = st.columns(3)
    briefings = _list_reports("briefing")
    analyses = _list_reports("analysis")
    col1.metric("Briefings saved", len(briefings))
    col2.metric("Analyses saved", len(analyses))
    col3.metric("Watchlist size", len(agent_cfg.watchlist))

    if "last_briefing" in st.session_state:
        st.divider()
        st.subheader("Latest briefing")
        st.markdown(st.session_state["last_briefing"])

with tab_briefings:
    st.subheader("Daily briefings")
    reports = _list_reports("briefing")
    if not reports:
        st.info("No briefings yet. Click **Daily Briefing** in the sidebar.")
    else:
        selected = st.selectbox("Select briefing", reports, format_func=lambda p: p.name)
        if selected:
            st.markdown(selected.read_text())

with tab_analysis:
    st.subheader("Full analysis reports")
    reports = _list_reports("analysis")
    if not reports:
        st.info("No analyses yet. Click **Full Analysis** in the sidebar.")
    else:
        selected = st.selectbox("Select analysis", reports, format_func=lambda p: p.name)
        if selected:
            st.markdown(selected.read_text())

        json_reports = _load_json_reports("run")
        if json_reports:
            latest = json_reports[0]
            st.divider()
            st.subheader("Trade intents (from latest run)")
            intents = latest.get("trade_intents", [])
            if not intents:
                st.write("No trade recommendations in latest run.")
            for item in intents:
                status = "✅ Allowed" if item.get("allowed") else "🚫 Blocked"
                st.write(f"**{status}** — {item.get('intent')}")

with tab_market:
    st.subheader("Live market snapshot")
    if st.button("Refresh market data"):
        st.session_state.pop("market_pkg", None)

    if "market_pkg" not in st.session_state:
        with st.spinner("Fetching quotes, news, and fundamentals..."):
            previous = load_snapshot(SNAPSHOT_PATH)
            pkg = build_intelligence_package(
                agent_cfg.watchlist,
                news_per_ticker=agent_cfg.retail.news_headlines_per_ticker,
                benchmark=agent_cfg.retail.benchmark_ticker,
            )
            changes = compute_changes(pkg, previous)
            st.session_state["market_pkg"] = pkg
            st.session_state["market_changes"] = changes

    pkg = st.session_state.get("market_pkg", {})
    changes = st.session_state.get("market_changes", {})

    if changes.get("has_previous"):
        st.warning(f"Changes since {changes.get('since', 'last run')}")
        if changes.get("price_moves"):
            st.write("**Price moves:**", changes["price_moves"])
        if changes.get("new_headlines"):
            st.write("**New headlines:**", changes["new_headlines"])
    else:
        st.info(changes.get("message", "Run a briefing to start tracking changes."))

    quotes = pkg.get("quotes", {})
    if quotes:
        st.dataframe(
            [
                {
                    "Ticker": t,
                    "Price": q.get("last_close"),
                    "Daily %": q.get("daily_change_pct"),
                }
                for t, q in quotes.items()
            ],
            use_container_width=True,
            hide_index=True,
        )

    benchmark = pkg.get("benchmark_comparison", {})
    if benchmark:
        st.subheader(f"vs {benchmark.get('benchmark', 'SPY')}")
        st.json(benchmark.get("benchmark_returns", {}))

    for ticker in agent_cfg.watchlist:
        t = ticker.upper()
        with st.expander(f"{t} — news & fundamentals"):
            fund = pkg.get("fundamentals", {}).get(t, {})
            news = pkg.get("news", {}).get(t, [])
            st.write("**Fundamentals:**", fund)
            for headline in news:
                st.write(f"- {headline.get('title')} ({headline.get('publisher')})")

with tab_backtest:
    st.subheader("Strategy vs SPY buy-and-hold")
    report = st.session_state.get("backtest_report")
    if not report:
        st.info("Click **Run Backtest vs SPY** in the sidebar.")
    else:
        s = report.aggregate
        b = report.benchmark_comparison.benchmark if report.benchmark_comparison else None
        c = report.benchmark_comparison

        col1, col2, col3 = st.columns(3)
        col1.metric("Strategy CAGR", f"{s.cagr_pct:.1f}%")
        col2.metric("Strategy Sharpe", f"{s.sharpe_ratio:.2f}")
        col3.metric("Strategy Calmar", f"{s.calmar_ratio:.2f}")

        if b and c:
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{c.benchmark_ticker} CAGR", f"{b.cagr_pct:.1f}%", f"{c.cagr_spread_pct:+.1f}%")
            col2.metric(f"{c.benchmark_ticker} Sharpe", f"{b.sharpe_ratio:.2f}", f"{c.sharpe_spread:+.2f}")
            col3.metric(f"{c.benchmark_ticker} Calmar", f"{b.calmar_ratio:.2f}", f"{c.calmar_spread:+.2f}")
            st.info(c.summary)

        st.dataframe(
            [
                {
                    "Ticker": t,
                    "CAGR %": m.cagr_pct,
                    "Sharpe": m.sharpe_ratio,
                    "Calmar": m.calmar_ratio,
                    "Max DD %": m.max_drawdown_pct,
                }
                for t, m in report.per_ticker.items()
            ],
            use_container_width=True,
            hide_index=True,
        )
