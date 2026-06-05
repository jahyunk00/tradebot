"""Senior strategist — MACD, RSI, regime, structure, volume, entry/stops/targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TradePlan:
    ticker: str
    last_price: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_per_share: float
    reward_risk_1: float


@dataclass
class StrategistReport:
    ticker: str
    score: float  # 0–1 executive quality
    dominant_trend: str
    regime: str
    trend_integrity: float
    macd_signal: str
    macd_histogram: float
    rsi: float
    rsi_zone: str
    price_structure: str
    volume_behavior: str
    volume_ratio: float
    support: float
    resistance: float
    summary: str
    trade_plan: TradePlan | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.trade_plan:
            d["trade_plan"] = asdict(self.trade_plan)
        return d


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - (100 / (1 + rs))
    return float(val.iloc[-1]) if not val.empty and pd.notna(val.iloc[-1]) else 50.0


def _macd(close: pd.Series) -> tuple[float, float, float, str]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    m, s, h = float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    if h > 0 and macd_line.iloc[-1] > signal.iloc[-1]:
        sig = "bullish"
    elif h < 0 and macd_line.iloc[-1] < signal.iloc[-1]:
        sig = "bearish"
    else:
        sig = "neutral"
    return m, s, h, sig


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _trend_integrity(close: pd.Series, window: int = 20) -> float:
    """0–1: higher = cleaner trend (R² of linear fit on recent closes)."""
    tail = close.tail(window)
    if len(tail) < 5:
        return 0.5
    x = np.arange(len(tail))
    y = tail.values.astype(float)
    if np.std(y) < 1e-9:
        return 0.5
    corr = np.corrcoef(x, y)[0, 1]
    return round(float(corr ** 2), 3)


def _price_structure(close: pd.Series) -> tuple[str, float, float]:
    tail = close.tail(60)
    if len(tail) < 20:
        return "insufficient data", 0.0, 0.0
    mid = len(tail) // 2
    h1, h2 = float(tail.iloc[:mid].max()), float(tail.iloc[mid:].max())
    l1, l2 = float(tail.iloc[:mid].min()), float(tail.iloc[mid:].min())
    support = float(tail.tail(10).min())
    resistance = float(tail.tail(10).max())
    if h2 > h1 and l2 > l1:
        return "higher highs & higher lows (uptrend structure)", support, resistance
    if h2 < h1 and l2 < l1:
        return "lower highs & lower lows (downtrend structure)", support, resistance
    return "range / chop — no clear structure", support, resistance


def _regime(close: pd.Series) -> str:
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50
    last = float(close.iloc[-1])
    vol = float(close.pct_change().tail(20).std() or 0)
    if last > sma50 > sma200:
        return "BULL_TREND"
    if last < sma50 < sma200:
        return "BEAR_TREND"
    if vol > 0.025:
        return "HIGH_VOLATILITY"
    return "RANGE_BOUND"


def _volume_behavior(df: pd.DataFrame) -> tuple[str, float]:
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return "volume data unavailable", 1.0
    vol = df["Volume"].astype(float)
    avg = vol.rolling(20).mean().iloc[-1]
    last = vol.iloc[-1]
    ratio = float(last / avg) if avg and avg > 0 else 1.0
    up = float(df["Close"].iloc[-1]) >= float(df["Close"].iloc[-2])
    if ratio >= 1.3 and up:
        return "accumulation — volume confirms up move", ratio
    if ratio >= 1.3 and not up:
        return "distribution — heavy volume on decline", ratio
    if ratio < 0.7:
        return "thin volume — weak conviction", ratio
    return "normal volume", ratio


def _build_trade_plan(
    ticker: str,
    df: pd.DataFrame,
    *,
    support: float,
    resistance: float,
) -> TradePlan:
    last = float(df["Close"].iloc[-1])
    atr = _atr(df)
    sma20 = float(df["Close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else last
    rsi = _rsi(df["Close"])

    # Entry: prefer pullback to SMA20 in uptrend, else last
    if rsi < 55 and last > sma20:
        entry = round(min(last, sma20 * 1.01), 2)
    else:
        entry = round(last, 2)

    stop = round(min(support * 0.98, entry - max(2 * atr, entry * 0.05)), 2)
    if stop >= entry:
        stop = round(entry - max(atr, entry * 0.04), 2)

    risk = max(entry - stop, 0.01)
    tp1 = round(entry + risk, 2)
    tp2 = round(entry + 2 * risk, 2)
    tp3 = round(entry + 3 * risk, 2)
    if resistance and resistance > tp2:
        tp3 = min(tp3, round(resistance * 1.02, 2))
    tp3 = max(tp3, tp2)

    return TradePlan(
        ticker=ticker,
        last_price=round(last, 2),
        entry_price=entry,
        stop_loss=stop,
        take_profit_1=tp1,
        take_profit_2=tp2,
        take_profit_3=tp3,
        risk_per_share=round(risk, 2),
        reward_risk_1=round((tp1 - entry) / risk, 2) if risk else 0,
    )


def analyze_ticker(ticker: str, df: pd.DataFrame) -> StrategistReport | None:
    if df is None or df.empty or len(df) < 60:
        return None

    close = df["Close"].astype(float)
    _, _, macd_hist, macd_sig = _macd(close)
    rsi = _rsi(close)
    regime = _regime(close)
    integrity = _trend_integrity(close)
    structure, support, resistance = _price_structure(close)
    vol_note, vol_ratio = _volume_behavior(df)

    if rsi >= 70:
        rsi_zone = "overbought"
    elif rsi <= 30:
        rsi_zone = "oversold"
    elif rsi >= 55:
        rsi_zone = "bullish momentum"
    else:
        rsi_zone = "neutral"

    if close.iloc[-1] > close.rolling(50).mean().iloc[-1]:
        dominant = "UP"
    elif close.iloc[-1] < close.rolling(50).mean().iloc[-1]:
        dominant = "DOWN"
    else:
        dominant = "SIDEWAYS"

    score = 0.5
    flags: list[str] = []

    if dominant == "UP":
        score += 0.12
    elif dominant == "DOWN":
        score -= 0.15
        flags.append("dominant trend down")

    if macd_sig == "bullish":
        score += 0.10
    elif macd_sig == "bearish":
        score -= 0.12
        flags.append("MACD bearish")

    if 45 <= rsi <= 65:
        score += 0.08
    elif rsi >= 75:
        score -= 0.10
        flags.append("RSI overbought")
    elif rsi <= 35:
        score += 0.05

    if regime == "BULL_TREND":
        score += 0.12
    elif regime == "BEAR_TREND":
        score -= 0.15
        flags.append("bear regime")
    elif regime == "HIGH_VOLATILITY":
        score -= 0.05

    score += integrity * 0.15

    if "accumulation" in vol_note:
        score += 0.08
    elif "distribution" in vol_note:
        score -= 0.10
        flags.append("distribution volume")

    if "uptrend structure" in structure:
        score += 0.10
    elif "downtrend" in structure:
        score -= 0.12

    score = round(max(0.0, min(1.0, score)), 3)
    plan = _build_trade_plan(ticker, df, support=support, resistance=resistance)

    summary = (
        f"{ticker}: {dominant} trend, {regime}, RSI {rsi:.0f} ({rsi_zone}), "
        f"MACD {macd_sig}, integrity {integrity:.2f}, vol {vol_ratio:.1f}× avg."
    )

    return StrategistReport(
        ticker=ticker.upper(),
        score=score,
        dominant_trend=dominant,
        regime=regime,
        trend_integrity=integrity,
        macd_signal=macd_sig,
        macd_histogram=round(macd_hist, 4),
        rsi=round(rsi, 1),
        rsi_zone=rsi_zone,
        price_structure=structure,
        volume_behavior=vol_note,
        volume_ratio=round(vol_ratio, 2),
        support=round(support, 2),
        resistance=round(resistance, 2),
        summary=summary,
        trade_plan=plan,
        flags=flags,
    )


def score_watchlist(history: dict[str, pd.DataFrame]) -> dict[str, StrategistReport]:
    out: dict[str, StrategistReport] = {}
    for ticker, df in history.items():
        r = analyze_ticker(ticker.upper(), df)
        if r:
            out[r.ticker] = r
    return out
