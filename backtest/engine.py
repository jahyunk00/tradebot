"""Backtest engine with benchmark comparison (SPY) and extended risk metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agent.guardrails import BacktestResult
from backtest.strategies import (
    ALL_STRATEGY_NAMES,
    PORTFOLIO_STRATEGIES,
    STRATEGIES,
    portfolio_returns_from_holds,
)
from data.market_data import fetch_history


@dataclass
class BenchmarkComparison:
    """Strategy vs buy-and-hold benchmark on the same window."""
    benchmark_ticker: str
    benchmark: BacktestResult
    beats_benchmark_cagr: bool = False
    beats_benchmark_sharpe: bool = False
    beats_benchmark_calmar: bool = False
    cagr_spread_pct: float = 0.0
    sharpe_spread: float = 0.0
    calmar_spread: float = 0.0
    summary: str = ""


@dataclass
class BacktestReport:
    per_ticker: dict[str, BacktestResult]
    aggregate: BacktestResult
    benchmark_comparison: BenchmarkComparison | None = None
    strategy_name: str = ""


@dataclass
class StrategyComparisonRow:
    strategy: str
    aggregate: BacktestResult
    benchmark_comparison: BenchmarkComparison
    passed_gate: bool
    score: float


def _annualized_cagr(cumulative_return: float, trading_days: int) -> float:
    if trading_days <= 0 or cumulative_return <= -1:
        return 0.0
    years = trading_days / 252
    if years <= 0:
        return 0.0
    return float(((1 + cumulative_return) ** (1 / years) - 1) * 100)


def _compute_metrics(returns: pd.Series, trades: pd.Series) -> BacktestResult:
    if returns.empty:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, False)

    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1) if len(cumulative) else 0.0
    total_return_pct = round(total_return * 100, 2)

    trading_days = len(returns.dropna())
    cagr_pct = round(_annualized_cagr(total_return, trading_days), 2)

    daily_ret = returns.dropna()
    sharpe = 0.0
    sortino = 0.0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))
        downside = daily_ret[daily_ret < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = float((daily_ret.mean() / downside.std()) * np.sqrt(252))

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown_pct = float(abs(drawdown.min()) * 100) if len(drawdown) else 0.0

    calmar_ratio = 0.0
    if max_drawdown_pct > 0:
        calmar_ratio = round((cagr_pct / 100) / (max_drawdown_pct / 100), 2)

    trade_changes = trades.diff().fillna(trades)
    entries = trade_changes[trade_changes != 0]
    wins = 0
    total_trades = 0
    for idx in entries.index:
        pos_idx = trades.index.get_loc(idx)
        if pos_idx + 5 >= len(trades):
            continue
        hold_ret = returns.iloc[pos_idx + 1 : pos_idx + 6].sum()
        total_trades += 1
        if hold_ret > 0:
            wins += 1

    win_rate = (wins / total_trades * 100) if total_trades else 0.0

    return BacktestResult(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        calmar_ratio=calmar_ratio,
        max_drawdown_pct=round(max_drawdown_pct, 2),
        win_rate_pct=round(win_rate, 2),
        trade_count=total_trades,
        passed=False,
    )


def _buy_and_hold_returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().fillna(0)


def run_benchmark(
    benchmark_ticker: str,
    lookback_days: int,
    initial_capital: float = 10_000,
) -> BacktestResult:
    history = fetch_history([benchmark_ticker], lookback_days)
    df = history.get(benchmark_ticker.upper())
    if df is None or df.empty:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, False, {"error": "No benchmark data"})

    daily_ret = _buy_and_hold_returns(df)
    position = pd.Series(1, index=daily_ret.index)
    result = _compute_metrics(daily_ret, position)
    result.details = {
        "ticker": benchmark_ticker.upper(),
        "strategy": "buy_and_hold",
        "days": len(df),
        "initial_capital": initial_capital,
        "simulated_final_value": round(initial_capital * (1 + result.total_return_pct / 100), 2),
    }
    return result


def compare_to_benchmark(strategy: BacktestResult, benchmark: BacktestResult) -> BenchmarkComparison:
    cagr_spread = round(strategy.cagr_pct - benchmark.cagr_pct, 2)
    sharpe_spread = round(strategy.sharpe_ratio - benchmark.sharpe_ratio, 2)
    calmar_spread = round(strategy.calmar_ratio - benchmark.calmar_ratio, 2)

    beats_cagr = strategy.cagr_pct > benchmark.cagr_pct
    beats_sharpe = strategy.sharpe_ratio > benchmark.sharpe_ratio
    beats_calmar = strategy.calmar_ratio > benchmark.calmar_ratio

    wins = sum([beats_cagr, beats_sharpe, beats_calmar])
    if wins >= 2:
        summary = f"Strategy beats {benchmark.details.get('ticker', 'benchmark')} on {wins}/3 key metrics."
    elif wins == 1:
        summary = "Mixed results vs benchmark — proceed with caution."
    else:
        summary = "Strategy underperforms benchmark on CAGR, Sharpe, and Calmar — do not auto-trade."

    return BenchmarkComparison(
        benchmark_ticker=str(benchmark.details.get("ticker", "SPY")),
        benchmark=benchmark,
        beats_benchmark_cagr=beats_cagr,
        beats_benchmark_sharpe=beats_sharpe,
        beats_benchmark_calmar=beats_calmar,
        cagr_spread_pct=cagr_spread,
        sharpe_spread=sharpe_spread,
        calmar_spread=calmar_spread,
        summary=summary,
    )


def _run_per_ticker_backtest(
    history: dict[str, pd.DataFrame],
    strategy_name: str,
    initial_capital: float,
) -> tuple[dict[str, BacktestResult], BacktestResult]:
    strategy_fn = STRATEGIES.get(strategy_name, STRATEGIES["momentum"])
    per_ticker: dict[str, BacktestResult] = {}
    all_returns: list[pd.Series] = []

    for ticker, df in history.items():
        if len(df) < 30:
            continue
        signal = strategy_fn(df)
        position = signal.shift(1).fillna(0)
        daily_ret = df["Close"].pct_change().fillna(0)
        strat_ret = daily_ret * position
        metrics = _compute_metrics(strat_ret, position)
        metrics.details = {"ticker": ticker, "days": len(df)}
        per_ticker[ticker] = metrics
        all_returns.append(strat_ret)

    if all_returns:
        combined = pd.concat(all_returns, axis=1).mean(axis=1)
        combined_position = pd.Series(1, index=combined.index)
        aggregate = _compute_metrics(combined, combined_position)
        aggregate.details = {
            "tickers_tested": list(per_ticker.keys()),
            "strategy": strategy_name,
            "initial_capital": initial_capital,
            "simulated_final_value": round(initial_capital * (1 + aggregate.total_return_pct / 100), 2),
        }
    else:
        aggregate = BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, False, {"error": "No data fetched"})

    return per_ticker, aggregate


def _run_portfolio_backtest(
    history: dict[str, pd.DataFrame],
    strategy_name: str,
    initial_capital: float,
    benchmark_ticker: str,
) -> tuple[dict[str, BacktestResult], BacktestResult]:
    portfolio_fn = PORTFOLIO_STRATEGIES[strategy_name]
    kwargs: dict = {}
    if strategy_name == "dual_momentum_spy":
        kwargs["benchmark"] = benchmark_ticker

    holds = portfolio_fn(history, **kwargs)
    strat_ret, position = portfolio_returns_from_holds(history, holds)
    aggregate = _compute_metrics(strat_ret, position)
    aggregate.details = {
        "strategy": strategy_name,
        "initial_capital": initial_capital,
        "simulated_final_value": round(initial_capital * (1 + aggregate.total_return_pct / 100), 2),
        "days": len(strat_ret),
    }
    return {}, aggregate


def run_backtest(
    tickers: list[str],
    strategy_name: str,
    lookback_days: int,
    initial_capital: float = 10_000,
    benchmark_ticker: str = "SPY",
) -> BacktestReport:
    history = fetch_history(tickers, lookback_days)

    if strategy_name in PORTFOLIO_STRATEGIES:
        per_ticker, aggregate = _run_portfolio_backtest(
            history, strategy_name, initial_capital, benchmark_ticker
        )
    else:
        per_ticker, aggregate = _run_per_ticker_backtest(history, strategy_name, initial_capital)

    bench = run_benchmark(benchmark_ticker, lookback_days, initial_capital)
    comparison = compare_to_benchmark(aggregate, bench)

    return BacktestReport(
        per_ticker=per_ticker,
        aggregate=aggregate,
        benchmark_comparison=comparison,
        strategy_name=strategy_name,
    )


def _strategy_score(row: StrategyComparisonRow) -> float:
    cmp = row.benchmark_comparison
    score = row.aggregate.sharpe_ratio
    score += cmp.cagr_spread_pct * 0.05
    score += cmp.sharpe_spread * 0.5
    if row.passed_gate:
        score += 1.0
    if cmp.beats_benchmark_cagr and cmp.beats_benchmark_sharpe:
        score += 0.5
    return round(score, 3)


def compare_strategies(
    tickers: list[str],
    lookback_days: int,
    initial_capital: float,
    benchmark_ticker: str,
    *,
    evaluate_gate,
    include_kronos: bool = False,
) -> list[StrategyComparisonRow]:
    from backtest.strategies import ALL_STRATEGY_NAMES

    names = list(ALL_STRATEGY_NAMES)
    if not include_kronos and "kronos_top_k" in names:
        names.remove("kronos_top_k")

    rows: list[StrategyComparisonRow] = []
    for name in names:
        report = run_backtest(
            tickers=tickers,
            strategy_name=name,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            benchmark_ticker=benchmark_ticker,
        )
        evaluated = evaluate_gate(report.aggregate)
        cmp = report.benchmark_comparison
        assert cmp is not None
        row = StrategyComparisonRow(
            strategy=name,
            aggregate=evaluated,
            benchmark_comparison=cmp,
            passed_gate=evaluated.passed,
            score=0.0,
        )
        row.score = _strategy_score(row)
        rows.append(row)

    rows.sort(key=lambda r: r.score, reverse=True)
    return rows
