"""
app/short_covering/short_covering_schema.py

Pydantic schemas and dataclasses for the Short-Covering Early-Ignition Scanner.
Covers:
- FNOContractInfo: Resolved near/next contract specs
- EODShortPositionCandidate: Output of Layer 1 EOD Engine
- Intraday5mTrigger: 5m candle + OI evaluation
- ShortCoveringSignal: High-confidence early alert contract
- IntradayReplayMetrics: Replay / Backtest result metrics
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


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
    timeframe_confirmations: Dict[str, bool] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    state: str = "IGNITION"  # IGNITION | CONFIRMED_CONTINUATION | EXHAUSTION


class IntradayReplayMetrics(BaseModel):
    """Metrics recorded during intraday point-in-time backtesting."""
    symbol: str
    alert_time: datetime
    alert_price: float
    fwd_return_15m_pct: float
    fwd_return_30m_pct: float
    fwd_return_60m_pct: float
    fwd_return_120m_pct: float
    eod_return_pct: float
    mfe_session_pct: float = Field(..., description="Max Favorable Excursion during the session")
    mae_session_pct: float = Field(..., description="Max Adverse Excursion during the session")
    is_false_covering: bool = Field(False, description="True if price breached stop / ignition low within 30m")
    next_day_continuation: bool = Field(False, description="True if next session close > alert price")
