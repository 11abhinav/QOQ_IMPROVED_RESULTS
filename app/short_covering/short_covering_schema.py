"""
app/short_covering/short_covering_schema.py

Data schemas and dataclasses for the Short-Covering Early-Ignition Scanner.
Uses standard Python dataclasses for 100% portable, dependency-free execution.
Covers:
- ShortCoveringState: State machine progression (WATCH -> IGNITION_CANDIDATE -> CONFIRMED_IGNITION -> CONTINUATION -> EXHAUSTED)
- ProviderCapability: Explicit matrix of data provider capabilities (Upstox, Fyers, NSE)
- FNOContractInfo: Resolved near/next contract specs
- EODShortPositionCandidate: Output of Layer 1 EOD Engine
- Intraday5mTrigger: 5m candle + OI evaluation
- ShortCoveringSignal: High-confidence early alert contract
- IntradayReplayMetrics: Replay / Backtest result metrics with pre/post alert MFE/MAE and latency
- StrategyComparativeBenchmark: Comparative benchmark metrics across baseline strategies
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum


class ShortCoveringState(str, Enum):
    """Lifecycle state machine for short-covering detection."""
    WATCH = "WATCH"
    IGNITION_CANDIDATE = "IGNITION_CANDIDATE"
    CONFIRMED_IGNITION = "CONFIRMED_IGNITION"
    CONTINUATION = "CONTINUATION"
    EXHAUSTED = "EXHAUSTED"


@dataclass
class BaseModelCompat:
    """Base dataclass providing .dict() helper for compatibility."""
    def dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderCapability(BaseModelCompat):
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
        supports_5m_oi=True,
        supports_historical_oi=True,
        supports_eod_bhavcopy=False,
        oi_resolution_notes="Fyers v3 History API supports 5m OI via 'oi_flag: 1'. Quotes API does not have OI; real-time OI uses Market Depth / OptionChain API."
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


@dataclass
class FNOContractInfo(BaseModelCompat):
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


@dataclass
class EODShortPositionCandidate(BaseModelCompat):
    """Layer 1: Output candidate from EOD Short Buildup Scanner."""
    symbol: str
    scan_date: date
    close_price: float
    total_oi: int
    oi_change_pct_1d: float
    oi_buildup_5d_pct: float
    oi_buildup_10d_pct: float
    short_buildup_ratio: float
    rsi_14: float
    support_level: float
    overhead_resistance: float
    atr_14: float
    daily_volume: int
    sector: Optional[str] = "GENERAL"
    buildup_quality_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class Intraday5mTrigger(BaseModelCompat):
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
    excess_oi_contraction: float = 0.0
    rel_strength_vs_nifty_pct: float = 0.0
    is_rollover: bool = False
    volume_surge_ratio: float = 1.0


@dataclass
class ShortCoveringSignal(BaseModelCompat):
    """Complete Signal payload emitted by Layer 2 Early-Ignition Scanner."""
    symbol: str
    timestamp: datetime
    ignition_price: float
    session_open_price: float = 0.0
    true_ignition_time: Optional[datetime] = None
    alert_latency_minutes: float = 0.0
    vwap: float = 0.0
    stop_loss: float = 0.0
    initial_target: float = 0.0
    risk_reward_ratio: float = 0.0
    excess_oi_contraction: float = 0.0
    oi_contraction_session_pct: float = 0.0
    volume_surge_ratio: float = 1.0
    rs_vs_nifty_pct: float = 0.0
    prior_short_score: float = 0.0
    ignition_score: float = 0.0
    grade: str = "B"
    state: ShortCoveringState = ShortCoveringState.CONFIRMED_IGNITION
    timeframe_confirmations: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class IntradayReplayMetrics(BaseModelCompat):
    """Metrics recorded during intraday point-in-time backtesting."""
    symbol: str
    alert_time: datetime
    true_ignition_time: datetime
    alert_latency_minutes: float
    alert_price: float
    session_open: float
    pre_alert_low: float
    post_alert_high: float
    post_alert_low: float
    total_session_high: float
    pre_alert_move_consumed_pct: float
    eventual_move_consumed_pct: float
    post_alert_mfe_pct: float
    post_alert_mae_pct: float
    fwd_return_15m_pct: float
    fwd_return_30m_pct: float
    fwd_return_60m_pct: float
    fwd_return_120m_pct: float
    eod_return_pct: float
    is_false_covering: bool = False
    next_day_continuation: bool = False


@dataclass
class StrategyComparativeBenchmark(BaseModelCompat):
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
    avg_post_alert_mfe_pct: float
    avg_post_alert_mae_pct: float
    median_pre_alert_move_pct: float
    median_eventual_move_consumed_pct: float
    median_latency_minutes: float
    p25_latency_minutes: float
    p75_latency_minutes: float
    worst_latency_minutes: float
    profit_factor: float
