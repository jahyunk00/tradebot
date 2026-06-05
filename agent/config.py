"""Configuration loading for the trading agent."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    lookback_days: int = 365
    initial_capital: float = 10_000
    strategy: Literal[
        "momentum",
        "mean_reversion",
        "sma_crossover",
        "dual_momentum",
        "relative_strength",
        "dual_momentum_spy",
        "kronos_top_k",
        "ensemble_weighted",
    ] = "relative_strength"
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


class EnsembleWeightsConfig(BaseModel):
    kronos: float = 0.34
    hmm: float = 0.33
    third_leg: float = 0.33


class HMMSettings(BaseModel):
    n_states: int = 3
    lookback: int = 252
    vol_window: int = 20


class BayesianChangepointSettings(BaseModel):
    lookback: int = 252
    penalty: float = 3.0
    min_segment: int = 10


class TFTSettingsConfig(BaseModel):
    input_chunk_length: int = 60
    output_chunk_length: int = 5
    train_length: int = 200
    hidden_size: int = 16
    lstm_layers: int = 1
    num_attention_heads: int = 2
    n_epochs: int = 5
    batch_size: int = 16
    dropout: float = 0.1


class EnsembleSettingsConfig(BaseModel):
    """Kronos + HMM + (Bayesian changepoint OR TFT) — exactly 3 decision makers."""
    weights: EnsembleWeightsConfig = Field(default_factory=EnsembleWeightsConfig)
    third_leg_model: Literal["bayesian_changepoint", "tft"] = "tft"
    rebalance_days: int = 5
    min_combined_score: float = 0.30
    backtest_use_kronos: bool = False
    backtest_use_third_leg: bool = False


class BossTrainingConfig(BaseModel):
    """Historical walk-forward practice for the boss agent."""
    lookback_days: int = 730
    warmup_days: int = 252
    rebalance_days: int = 5
    forward_days: int = 5
    min_train_samples: int = 12
    use_kronos: bool = False
    use_third_leg: bool = True
    third_leg_model: Literal["bayesian_changepoint", "tft"] = "tft"


class BossSettings(BaseModel):
    """Boss agent orchestrates three leg agents with learned weights."""
    enabled: bool = True
    min_combined_score: float = 0.30
    strategist_weight: float = 0.40
    weights_path: str = "logs/boss_weights.json"
    paper_portfolio_path: str = "logs/paper_portfolio.json"
    learning_rate: float = 0.05
    training: BossTrainingConfig = Field(default_factory=BossTrainingConfig)
    portfolio_mode: Literal["single", "multi"] = "multi"
    max_picks: int = 4
    rotate_out: bool = True
    momentum_weight: float = 0.5


class BearModeSettings(BaseModel):
    """Defensive rules when benchmark is in a downtrend."""
    enabled: bool = True
    min_news_score: float = 0.12
    block_metrics_only: bool = True
    oversold_rsi: float = 32.0
    stop_tighten_pct: float = 0.20
    tp1_r: float = 0.75
    tp2_r: float = 1.5
    tp3_r: float = 2.0


class PharmaSettings(BaseModel):
    """Small/mid-cap pharma focus — news trends + market-cap preference."""
    enabled: bool = True
    max_market_cap_b: float = 20.0       # prefer names under ~$20B
    small_cap_boost: float = 0.15        # score boost for smaller caps
    news_weight: float = 0.20            # blend 20% news into boss pick
    news_headlines: int = 8
    benchmark_ticker: str = "XBI"          # biotech ETF vs SPY


class KronosSettings(BaseModel):
    """Kronos foundation model — github.com/shiyu-coder/Kronos."""
    enabled: bool = False
    model_id: str = "NeoQuasar/Kronos-mini"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-2k"
    max_context: int = 512
    lookback: int = 400
    pred_len: int = 5
    rebalance_days: int = 5
    temperature: float = 0.6
    top_p: float = 0.9
    sample_count: int = 3
    min_forecast_return_pct: float = 0.0
    device: str | None = None


class UniverseSettings(BaseModel):
    """Dynamic watchlist — large caps with strong trailing growth + rotation."""
    enabled: bool = False
    rotation_enabled: bool = True
    min_market_cap_b: float = 50.0
    min_return_1y_pct: float = 5.0
    pool_size: int = 30                       # broad screen pool
    top_n: int = 15                           # active roster size
    swap_margin_pct: float = 2.0              # replace laggard if alt beats by this %
    lookback_days: int = 252
    cache_hours: float = 20.0
    seed_universe: list[str] = Field(default_factory=list)


class PaperTrialsSettings(BaseModel):
    """Paper-audit names; auto-promote winners to live cash."""
    enabled: bool = True
    max_paper_slots: int = 3
    max_live_promoted: int = 3
    virtual_usd: float = 30.0
    campaign_days: int = 30
    min_sessions: int = 2
    min_return_pct: float = 1.5
    drop_below_pct: float = -4.0
    min_model_score: float = 0.22
    bear_fast_track: bool = True
    bear_min_sessions: int = 1
    bear_min_return_pct: float = 0.5
    bear_min_model_score: float = 0.18


class LaunchScheduleSettings(BaseModel):
    """Pilot live trading at reduced size until live_start_date, then full bankroll."""
    enabled: bool = True
    live_start_date: str = "2026-06-09"
    pilot_bankroll_usd: float = 50.0
    live_full_bankroll: bool = True


class AgentConfig(BaseModel):
    mode: Literal["analyze_only", "auto_execute"] = "analyze_only"
    engine: Literal["rules", "llm", "kronos", "ensemble", "boss"] = "boss"
    watchlist: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    paper_trials: PaperTrialsSettings = Field(default_factory=PaperTrialsSettings)
    launch_schedule: LaunchScheduleSettings = Field(default_factory=LaunchScheduleSettings)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    robinhood: RobinhoodConfig = Field(default_factory=RobinhoodConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    retail: RetailConfig = Field(default_factory=RetailConfig)
    kronos: KronosSettings = Field(default_factory=KronosSettings)
    hmm: HMMSettings = Field(default_factory=HMMSettings)
    bayesian_changepoint: BayesianChangepointSettings = Field(default_factory=BayesianChangepointSettings)
    tft: TFTSettingsConfig = Field(default_factory=TFTSettingsConfig)
    ensemble: EnsembleSettingsConfig = Field(default_factory=EnsembleSettingsConfig)
    boss: BossSettings = Field(default_factory=BossSettings)
    pharma: PharmaSettings = Field(default_factory=PharmaSettings)
    bear_mode: BearModeSettings = Field(default_factory=BearModeSettings)


class BacktestGate(BaseModel):
    min_sharpe_ratio: float = 0.5
    min_win_rate_pct: float = 45
    max_drawdown_pct: float = 20
    min_total_return_pct: float = 5


class BankrollConfig(BaseModel):
    """How the bot sizes orders as the account grows or shrinks."""
    mode: Literal["fixed", "dynamic"] = "dynamic"
    initial_usd: float = 100
    ceiling_usd: float | None = None  # optional max equity used for sizing
    require_cash_only: bool = True
    reinvest_profits: bool = True  # dynamic mode uses live equity including gains


class GuardrailsConfig(BaseModel):
    force_analyze_only: bool = True
    bankroll: BankrollConfig = Field(default_factory=BankrollConfig)
    max_order_usd: float | None = None  # optional absolute cap; None = pct-only
    max_position_pct: float = 50
    max_daily_trades: int = 10
    max_open_positions: int = 5
    allowed_tickers: list[str] = Field(default_factory=list)
    allowed_order_types: list[str] = Field(default_factory=lambda: ["market", "limit"])
    require_backtest_pass: bool = True
    require_beats_benchmark: bool = False
    backtest_gate: BacktestGate = Field(default_factory=BacktestGate)
    min_minutes_between_trades: int = 60
    enforce_market_hours: bool = True
    no_same_day_round_trip: bool = True


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config(base_dir: Path | None = None) -> tuple[AgentConfig, GuardrailsConfig]:
    root = base_dir or Path(__file__).resolve().parent.parent
    agent_cfg = AgentConfig(**_load_yaml(root / "config.yaml"))
    guard_cfg = GuardrailsConfig(**_load_yaml(root / "guardrails.yaml"))
    return agent_cfg, guard_cfg
