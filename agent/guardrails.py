"""Hard guardrails — every trade intent must pass before execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from agent.bankroll import BankrollSnapshot, clamp_trade_amount
from agent.config import AgentConfig, GuardrailsConfig


class TradingMode(str, Enum):
    ANALYZE_ONLY = "analyze_only"
    AUTO_EXECUTE = "auto_execute"


@dataclass
class TradeIntent:
    ticker: str
    side: str  # buy | sell
    amount_usd: float
    order_type: str = "market"
    rationale: str = ""


@dataclass
class BacktestResult:
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trade_count: int
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailVerdict:
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def block(cls, *reasons: str) -> GuardrailVerdict:
        return cls(allowed=False, reasons=list(reasons))

    @classmethod
    def allow(cls) -> GuardrailVerdict:
        return cls(allowed=True)


MIN_POSITION_USD = 1.0


def count_meaningful_positions(
    positions: dict[str, float] | None,
    *,
    min_usd: float = MIN_POSITION_USD,
) -> int:
    """Ignore dust holdings (e.g. fractional leftovers) when enforcing position caps."""
    if not positions:
        return 0
    return len([v for v in positions.values() if float(v) >= min_usd])


class Guardrails:
    """Central enforcement layer. analyze-only is the default safe state."""

    ET = ZoneInfo("America/New_York")
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        agent_config: AgentConfig,
        guardrails_config: GuardrailsConfig,
        *,
        bankroll: BankrollSnapshot | None = None,
        backtest_result: BacktestResult | None = None,
        trades_today: int = 0,
        last_trade_at: datetime | None = None,
        open_positions: int = 0,
        positions: dict[str, float] | None = None,
    ) -> None:
        self.agent_config = agent_config
        self.guardrails_config = guardrails_config
        self.bankroll = bankroll
        self.backtest_result = backtest_result
        self.trades_today = trades_today
        self.last_trade_at = last_trade_at
        self.positions = {k.upper(): float(v) for k, v in (positions or {}).items() if float(v) > 0}
        self.open_positions = open_positions or count_meaningful_positions(self.positions)

    def effective_max_order_usd(self) -> float:
        g = self.guardrails_config
        if not self.bankroll:
            return g.max_order_usd or g.bankroll.initial_usd
        return self.bankroll.max_order_usd(g.max_position_pct, g.max_order_usd)

    def bankroll_summary(self) -> dict[str, Any]:
        g = self.guardrails_config
        if not self.bankroll:
            return {"initial_usd": g.bankroll.initial_usd, "mode": g.bankroll.mode}
        b = self.bankroll
        return {
            **b.to_summary(),
            "effective_equity_usd": b.effective_equity(),
            "max_order_usd_now": self.effective_max_order_usd(),
            "max_position_usd_now": b.max_position_usd(g.max_position_pct),
            "note": (
                f"Limits scale with equity. Started ${b.initial_usd:.2f}, "
                f"now ${b.equity_usd:.2f} ({b.gain_pct:+.1f}% P/L)."
                if b.mode == "dynamic"
                else f"Fixed bankroll ${b.initial_usd:.2f}."
            ),
        }

    @property
    def effective_mode(self) -> TradingMode:
        from pathlib import Path

        from agent.runtime_state import is_active_investing

        base = Path(__file__).resolve().parent.parent
        if is_active_investing(base):
            return TradingMode.AUTO_EXECUTE
        if self.guardrails_config.force_analyze_only:
            return TradingMode.ANALYZE_ONLY
        if self.agent_config.mode == "analyze_only":
            return TradingMode.ANALYZE_ONLY
        return TradingMode.AUTO_EXECUTE

    def can_execute_trades(self) -> GuardrailVerdict:
        from pathlib import Path

        from agent.runtime_state import is_active_investing

        base = Path(__file__).resolve().parent.parent
        live_via_toggle = is_active_investing(base)

        if self.effective_mode == TradingMode.ANALYZE_ONLY and not live_via_toggle:
            return GuardrailVerdict.block(
                "Trading is disabled: turn on **Active investing** in the dashboard.",
            )

        if self.guardrails_config.require_backtest_pass:
            if self.backtest_result is None:
                return GuardrailVerdict.block("No backtest result — run backtest before live trading.")
            if not self.backtest_result.passed:
                return GuardrailVerdict.block(
                    "Backtest did not pass guardrail thresholds.",
                    f"Sharpe={self.backtest_result.sharpe_ratio:.2f}, "
                    f"win_rate={self.backtest_result.win_rate_pct:.1f}%, "
                    f"drawdown={self.backtest_result.max_drawdown_pct:.1f}%.",
                )

        if self.guardrails_config.require_beats_benchmark and self.backtest_result:
            details = self.backtest_result.details or {}
            vs = details.get("vs_benchmark")
            if vs and not vs.get("beats_majority"):
                return GuardrailVerdict.block(
                    "Strategy does not beat SPY on enough metrics — tune strategy first.",
                    str(vs.get("summary", "")),
                )

        return GuardrailVerdict.allow()

    def validate_trade(self, intent: TradeIntent) -> GuardrailVerdict:
        execute_check = self.can_execute_trades()
        if not execute_check.allowed:
            return execute_check

        g = self.guardrails_config
        reasons: list[str] = []

        ticker = intent.ticker.upper()
        if ticker not in {t.upper() for t in g.allowed_tickers}:
            reasons.append(f"Ticker {ticker} not in allowed list.")

        if intent.amount_usd <= 0:
            reasons.append("Order amount must be positive.")
        elif intent.side.lower() == "buy":
            max_order = self.effective_max_order_usd()
            if intent.amount_usd > max_order:
                reasons.append(
                    f"Order ${intent.amount_usd:.2f} exceeds dynamic max ${max_order:.2f} "
                    f"(equity ${self.bankroll.equity_usd:.2f} × {g.max_position_pct}% cap)."
                    if self.bankroll
                    else f"Order ${intent.amount_usd:.2f} exceeds max ${max_order:.2f}."
                )
            if self.bankroll:
                room = self.bankroll.remaining_room_usd(ticker, g.max_position_pct)
                if intent.amount_usd > room:
                    reasons.append(
                        f"Order exceeds remaining room in {ticker} (${room:.2f} left at {g.max_position_pct}% cap)."
                    )
                if g.bankroll.require_cash_only and intent.amount_usd > self.bankroll.cash_usd:
                    reasons.append(
                        f"Order ${intent.amount_usd:.2f} exceeds available cash ${self.bankroll.cash_usd:.2f}."
                    )

        if intent.order_type not in g.allowed_order_types:
            reasons.append(f"Order type '{intent.order_type}' not allowed.")

        if self.trades_today >= g.max_daily_trades:
            reasons.append(f"Daily trade limit reached ({g.max_daily_trades}).")

        if intent.side.lower() == "buy" and self.open_positions >= g.max_open_positions:
            already_held = self.positions.get(ticker, 0.0) >= MIN_POSITION_USD
            if not already_held:
                reasons.append(
                    f"Max open positions ({g.max_open_positions}) reached "
                    f"({self.open_positions} meaningful holdings)."
                )

        if self.last_trade_at and g.min_minutes_between_trades > 0:
            elapsed = (datetime.now(tz=self.ET) - self.last_trade_at.astimezone(self.ET)).total_seconds() / 60
            if elapsed < g.min_minutes_between_trades:
                reasons.append(
                    f"Cooldown active: {g.min_minutes_between_trades - elapsed:.0f} min remaining."
                )

        if g.enforce_market_hours and not self._is_market_hours():
            reasons.append("Outside US regular market hours (9:30–16:00 ET).")

        if reasons:
            return GuardrailVerdict.block(*reasons)
        return GuardrailVerdict.allow()

    def evaluate_backtest(self, result: BacktestResult) -> BacktestResult:
        gate = self.guardrails_config.backtest_gate
        passed = (
            result.sharpe_ratio >= gate.min_sharpe_ratio
            and result.win_rate_pct >= gate.min_win_rate_pct
            and result.max_drawdown_pct <= gate.max_drawdown_pct
            and result.total_return_pct >= gate.min_total_return_pct
        )
        result.passed = passed
        self.backtest_result = result
        return result

    def _is_market_hours(self) -> bool:
        return is_us_market_hours()


def is_us_market_hours() -> bool:
    """US regular session Mon–Fri 9:30–16:00 Eastern."""
    now = datetime.now(tz=Guardrails.ET)
    if now.weekday() >= 5:
        return False
    return Guardrails.MARKET_OPEN <= now.time() <= Guardrails.MARKET_CLOSE
