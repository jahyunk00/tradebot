"""Configuration loading for the trading agent."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    lookback_days: int = 365
    initial_capital: float = 10_000
    strategy: Literal["momentum", "mean_reversion", "sma_crossover"] = "momentum"
    min_sharpe_ratio: float = 0.5
    min_win_rate_pct: float = 45
    max_drawdown_pct: float = 20
    min_total_return_pct: float = 5


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096


class RobinhoodConfig(BaseModel):
    mcp_url: str = "https://agent.robinhood.com/mcp/trading"
    oauth_server_url: str = "https://agent.robinhood.com"


class LoggingConfig(BaseModel):
    directory: str = "logs"
    level: str = "INFO"


class RetailConfig(BaseModel):
    """Settings that address retail investor constraints."""
    benchmark_ticker: str = "SPY"
    news_headlines_per_ticker: int = 3
    plain_english: bool = True
    include_glossary: bool = True
    max_read_minutes: int = 2  # target length for summary section


class AgentConfig(BaseModel):
    mode: Literal["analyze_only", "auto_execute"] = "analyze_only"
    watchlist: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    robinhood: RobinhoodConfig = Field(default_factory=RobinhoodConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    retail: RetailConfig = Field(default_factory=RetailConfig)


class BacktestGate(BaseModel):
    min_sharpe_ratio: float = 0.5
    min_win_rate_pct: float = 45
    max_drawdown_pct: float = 20
    min_total_return_pct: float = 5


class GuardrailsConfig(BaseModel):
    force_analyze_only: bool = True
    max_order_usd: float = 500
    max_position_pct: float = 15
    max_daily_trades: int = 3
    max_open_positions: int = 5
    allowed_tickers: list[str] = Field(default_factory=list)
    allowed_order_types: list[str] = Field(default_factory=lambda: ["market", "limit"])
    require_backtest_pass: bool = True
    backtest_gate: BacktestGate = Field(default_factory=BacktestGate)
    min_minutes_between_trades: int = 60
    enforce_market_hours: bool = True


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config(base_dir: Path | None = None) -> tuple[AgentConfig, GuardrailsConfig]:
    root = base_dir or Path(__file__).resolve().parent.parent
    agent_cfg = AgentConfig(**_load_yaml(root / "config.yaml"))
    guard_cfg = GuardrailsConfig(**_load_yaml(root / "guardrails.yaml"))
    return agent_cfg, guard_cfg
