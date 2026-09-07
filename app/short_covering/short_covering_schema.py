"""
app/short_covering/short_covering_schema.py

Pydantic schemas and dataclasses for the Short-Covering Early-Ignition Scanner.
Covers:
- ShortCoveringState: State machine progression (WATCH -> IGNITION_CANDIDATE -> CONFIRMED_IGNITION -> CONTINUATION -> EXHAUSTED)
- ProviderCapability: Explicit matrix of data provider capabilities (Upstox, Fyers, NSE)
- FNOContractInfo: Resolved near/next contract specs
- EODShortPositionCandidate: Output of Layer 1 EOD Engine
- Intraday5mTrigger: 5m candle + OI evaluation
- ShortCoveringSignal: High-confidence early alert contract
- IntradayReplayMetrics: Replay / Backtest result metrics with move consumed & earlyness
- StrategyComparativeBenchmark: Comparative benchmark metrics across baseline strategies
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class ShortCoveringState(str, Enum):
    """Lifecycle state machine for short-covering detection."""
    WATCH = "WATCH"
    IGNITION_CANDIDATE = "IGNITION_CANDIDATE"
    CONFIRMED_IGNITION = "CONFIRMED_IGNITION"
    CONTINUATION = "CONTINUATION"
    EXHAUSTED = "EXHAUSTED"


class ProviderCapability(BaseModel):
    """Capabilities supported by a market data provider."""
    provider_name: str
    supports_5m_price: bool = True
    supports_5m_volume: bool = True
    supports_5m_oi: bool = False
    supports_historical_oi: bool = False
    supports_eod_bhavcopy: bool = False
    oi_resolution_notes: str = ""


PROVIDER_CAPABILITY_MATRIX: Dict[str, ProviderCapability] = {
    "UPSTOX": ProviderCapability(
        provider_name="UPSTOX",
        supports_5m_price=True,
        supports_5m_volume=True,
        supports_5m_oi=True,
        supports_historical_oi=True,
        supports_eod_bhavcopy=False,
        oi_resolution_notes="Upstox V3 candle API supports 5m OHLCV and open_interest directly."
    ),
    "FYERS": ProviderCapability(
        provider_name="FYERS",
        supports_5m_price=True,
        supports_5m_volume=True,
        supports_5m_oi=False,  # Quotes API does not provide OI; requires dedicated futures endpoint if available
        supports_historical_oi=False,
        supports_eod_bhavcopy=False,
        oi_resolution_notes="Fyers quotes API does not provide OI. 5m OI must be routed to Upstox/Bhavcopy."
    ),
    "NSE_EOD": ProviderCapability(
        provider_name="NSE_EOD",
        supports_5m_price=False,
        supports_5m_volume=False,
        supports_5m_oi=False,
        supports_historical_oi=True,
        supports_eod_bhavcopy=True,
        oi_resolution_notes="Authoritative source for daily F&O Bhavcopy and cumulative settlement OI."
    ),
}


class FNOContractInfo(BaseModel):
    """Details of resolved near and next month futures contracts for a symbol."""
    symbol: str
    underlying: str
    near_expiry: date
    next_expiry: Optional[date] = None
    near_instrument_token: Optional[str] = None
    next_instrument_token: Optional[str] = None
    near_trading_symbol: Optional[str] = None
    next_trading_symbol: Optional[str] = None
    lot_size: int = 1
    is_expiry_week: bool = False
    days_to_near_expiry: int = 0


class EODShortPositionCandidate(BaseModel):
    """Layer 1: Output candidate from EOD Short Buildup Scanner."""
    symbol: str
    scan_date: date
    close_price: float
    total_oi: int
    oi_change_pct_1d: float
    oi_buildup_5d_pct: float
    oi_buildup_10d_pct: float
    short_buildup_ratio: float = Field(..., description="Fraction of recent days where price fell and OI rose (SBR)")
    rsi_14: float
    support_level: float
    overhead_resistance: float
    atr_14: float
    daily_volume: int
    sector: Optional[str] = "GENERAL"
    buildup_quality_score: float = Field(..., ge=0.0, le=100.0)
    reasons: List[str] = Field(default_factory=list)


class Intraday5mTrigger(BaseModel):
    """Layer 2: 5-minute bar evaluation record."""
    symbol: str
    timestamp: datetime
    price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    current_oi: int
    oi_change_5m_pct: float
    oi_change_session_pct: float
    nifty_oi_change_5m_pct: float = 0.0
    excess_oi_contraction: float = Field(..., description="Stock OI contraction minus NIFTY OI contraction")
    rel_strength_vs_nifty_pct: float
    is_rollover: bool = False
    volume_surge_ratio: float = 1.0


class ShortCoveringSignal(BaseModel):
    """Complete Signal payload emitted by Layer 2 Early-Ignition Scanner."""
    symbol: str
    timestamp: datetime
    ignition_price: float
    session_open_price: float = 0.0
    vwap: float
    stop_loss: float
    initial_target: float
    risk_reward_ratio: float
    excess_oi_contraction: float
    oi_contraction_session_pct: float
    volume_surge_ratio: float
    rs_vs_nifty_pct: float
    prior_short_score: float
    ignition_score: float = Field(..., ge=0.0, le=100.0)
    grade: str = Field(..., description="A+ / A / B / C")
    state: ShortCoveringState = ShortCoveringState.CONFIRMED_IGNITION
    timeframe_confirmations: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)


class IntradayReplayMetrics(BaseModel):
    """Metrics recorded during intraday point-in-time backtesting."""
    symbol: str
    alert_time: datetime
    alert_price: float
    session_open: float
    session_high: float
    session_low: float
    move_consumed_at_alert_pct: float = Field(
        ...,
        description="Percentage of the total session range already consumed when the alert fired: (Alert - Low) / (High - Low)"
    )
    move_captured_after_alert_pct: float = Field(
        ...,
        description="Percentage of the session upside captured after alert: (High - Alert) / Alert"
    )
    fwd_return_15m_pct: float
    fwd_return_30m_pct: float
    fwd_return_60m_pct: float
    fwd_return_120m_pct: float
    eod_return_pct: float
    mfe_session_pct: float = Field(..., description="Max Favorable Excursion during the session")
    mae_session_pct: float = Field(..., description="Max Adverse Excursion during the session")
    is_false_covering: bool = Field(False, description="True if price breached stop / ignition low within 30m")
    next_day_continuation: bool = Field(False, description="True if next session close > alert price")


class StrategyComparativeBenchmark(BaseModel):
    """Comparative benchmarking results comparing proposed strategy vs baseline models."""
    strategy_name: str
    total_alerts: int
    win_rate_eod_pct: float
    false_covering_rate_pct: float
    next_day_continuation_rate_pct: float
    avg_fwd_return_15m_pct: float
    avg_fwd_return_30m_pct: float
    avg_fwd_return_60m_pct: float
    avg_fwd_return_120m_pct: float
    avg_eod_return_pct: float
    avg_session_mfe_pct: float
    avg_session_mae_pct: float
    median_move_consumed_pct: float
    median_move_captured_pct: float
    profit_factor: float
