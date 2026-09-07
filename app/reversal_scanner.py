from scanner_telemetry import ScannerDecisionLogger, global_telemetry
import time as _time
# =====================================================================================
# app/reversal_scanner.py (SCHEDULER READY) — v7.0 OVERHAUL
# DEEP DISCOUNT & MEAN REVERSION SCANNER (With Valuation Metrics)
#
# RULE 67 MANDATORY CHANGE-RATIONALE:
# - Verified and enhanced GlobalScannerTelemetryEngine logging across all reversal gates.
# - Rationale: Captures granular RSI trough curl calculations, 52W drop depth, SMA200 proximity,
#   volume spike confirmation, and risk engine output to eliminate diagnostic gaps.
# =====================================================================================
#
# v7.0 OVERHAUL CHANGELOG (Refactored per comprehensive 25-point audit):
#   [FIX A1] CORE_SCORE_FLOOR comment & BEAR_CORE_REALISTIC headroom aligned.
#   [FIX A2] close_above_ema20 trend scoring tautology removed (strict EMA20 reclaim).
#   [FIX A3] Monotonic SMA200 proximity scoring (at/above SMA200 gets 12 pts).
#   [FIX A4] AVAILABLE_MAX normalized across all missing optional components.
#   [FIX A5] inst_bonus & pledge_penalty clamped against AVAILABLE_MAX.
#   [FIX A6] Fixed inverted MACD freshness test logic (no longer rejects sustained bull momentum).
#   [FIX A7] _is_climax_top volume parameter & fallback lookback slice fixed.
#   [FIX B1] Current-bar volume ratio checked at hard confirmation gate.
#   [FIX B2] RSI trough freshness lookback gate added (max 25 bars).
#   [FIX B3] Robust 3-bar RSI declining check.
#   [FIX B4] Fundamental presence enforced even for quality categories.
#   [FIX B5] Heavy red candle filter message precision fixed.
#   [FIX C1] Surveillance / blacklist filter enforced in candidate loop.
#   [FIX C2] Fetch health ratio tracked and logged.
#   [FIX C3] Scanner health updated to "OK" upon successful completion.
#   [FIX C4] Batch loop wrapped in try-except to prevent single-batch aborts.
#   [FIX C5] Regime keys (current_regime & trend) aligned across functions.
#   [FIX C6] Target keys (target_1 & target) standardized.
#   [FIX C7] Outcome-aware reversal cooldown helper wired in.
#   [FIX C8] Force parameter respected (bypasses cooldowns).
#   [FIX C9] rejected counter dict initialized with defaultdict(int).
#   [FIX C10] Off-hours intraday 5m snapshot optimization added.
#   [FIX C11] Delivery confidence aging normalized in AVAILABLE_MAX.
#   [FIX C12] Unused imports removed.
#   [FIX C13] Config validator updated for quality category drop floor & core score.
# =====================================================================================
import pandas as pd
import logging
import os
import time
import json
import math
from collections import defaultdict
from zoneinfo import ZoneInfo
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Optional
from database import upsert_scanner_health

from scanner_telemetry import DecisionContext, telemetry_engine
from technical_indicators import apply_indicators
apply_reversal_indicators = apply_indicators
from memory_profiler import MemoryProfiler, chunk_iterable
from database import (
    init_db,
    save_alert_if_new,
    upsert_scanner_health,
)
from zero_alert_diagnostic import (
    SingleTerminalTracker,
    StageWaterfallTracker,
    classify_zero_alert_run,
    format_zero_alert_diagnostic_block,
)
from price_cache import fetch_watchlist_data
from watchlist_cache import get_watchlist
from config import (
    CLIMAX_VOLUME_LOOKBACK, 
    MIN_CANDLE_RANGE_PCT, 
    REVERSAL_CONFIG,
    ACTIVE_ALGO_VERSION,
    REVERSAL_RSI_LOOKBACK,
    REVERSAL_MAX_TROUGH_AGE
)
from sl_target_helper import compute_sl_and_target
from surveillance import get_live_blacklist, force_refresh_blacklist

# [VERSION: PERF_PROFILER_v2.0] Stage timing + filter rejection observability
# profile_timing logs wall-clock duration + RSS delta for each reversal scan run.
# FilterStats captures per-filter rejection CSV to artifacts/profiling/ each run.
# stage_timer / flush_timing_report emit schema_version:1 baseline reports (Phase 0).
from perf_utils import profile_timing, FilterStats, flush_timing_report, reset_stage_timers


logger = logging.getLogger(__name__)

def _safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return float(val)
    except Exception:
        return default

IST = ZoneInfo("Asia/Kolkata")

# ── REVERSAL PARAMETERS ──────────────────────────────────────────────────────────────
MIN_DROP_FROM_52W_HIGH = REVERSAL_CONFIG["MIN_DROP_FROM_52W_HIGH"]
MAX_DROP_FROM_52W_HIGH = REVERSAL_CONFIG["MAX_DROP_FROM_52W_HIGH"]

RSI_OVERSOLD_THRESHOLD = REVERSAL_CONFIG["RSI_OVERSOLD_THRESHOLD"]
RSI_CURL_MIN           = REVERSAL_CONFIG["RSI_CURL_MIN"]
MIN_RSI_RECOVERY       = REVERSAL_CONFIG.get("MIN_RSI_RECOVERY", 8.0)  # [AUDIT-R2A FIX] Wired to config; was hardcoded 8.0, silently overriding REVERSAL_CONFIG value
RSI_TROUGH_LOOKBACK    = REVERSAL_RSI_LOOKBACK
MAX_TROUGH_AGE         = REVERSAL_MAX_TROUGH_AGE

MIN_VOLUME_RATIO       = REVERSAL_CONFIG["MIN_VOLUME_RATIO"]
VOL_WINDOW_BARS        = 5

# ── QUALITY FILTERS (high-quality stocks only) ───────────────────────────────────────
MIN_STOCK_PRICE        = REVERSAL_CONFIG.get("MIN_STOCK_PRICE", 100.0)
MIN_AVG_DAILY_VOLUME   = REVERSAL_CONFIG["MIN_AVG_DAILY_VOLUME"]
MIN_ROE                = REVERSAL_CONFIG["MIN_ROE"]
MIN_YOY_REVENUE_GROWTH = REVERSAL_CONFIG.get("MIN_YOY_REVENUE_GROWTH_FLOOR", -15.0)  # [FIX: CONFIG_DRIFT] Was hardcoded -15.0 while REVERSAL_CONFIG had 5.0 (unused). Reversal is mean-reversion — needs lenient floor. Default -15.0 is correct.
MAX_DROP_BELOW_SMA200  = REVERSAL_CONFIG["MAX_DROP_BELOW_SMA200"]
QUALITY_CAT_MIN_DROP   = REVERSAL_CONFIG.get("QUALITY_CAT_MIN_DROP", 15.0)

# Climax filter thresholds
CLIMAX_VOL_MULT        = 3.5
CLIMAX_VOL_QUANTILE    = 0.95
CLIMAX_MIN_RUNUP_PCT   = 0.10

# Pipeline health guard
FUNDAMENTAL_REJECT_ALARM_PCT = 0.60
REVERSAL_COOLDOWN_TRADING_DAYS = REVERSAL_CONFIG["REVERSAL_COOLDOWN_TRADING_DAYS"]

REVERSAL_MIN_BARS = 35  # Adaptive floor supporting FRESH_IPO (35-49 bars), RECENT_LISTING (50-199 bars), and MATURE (200+ bars)
DEFAULT_PLEDGE_PENALTY = 15.0
STALE_DEGRADED_RATIO = 0.15
MIN_FETCH_RATIO = 0.85
COMPONENT_MAX = 25 + 12 + 15 + 15 + 15 + 10 + 5 + 5 + 5 + 5   # = 112 max score points
MAX_POSSIBLE_SCORE = COMPONENT_MAX


EXCHANGE_HOLIDAYS = {
    # 2025
    date(2025, 1, 26), date(2025, 3, 13), date(2025, 3, 31), date(2025, 4, 10),
    date(2025, 4, 11), date(2025, 4, 14), date(2025, 5, 1), date(2025, 6, 6),
    date(2025, 8, 15), date(2025, 9, 5), date(2025, 10, 2), date(2025, 10, 24),
    date(2025, 11, 20), date(2025, 12, 25),
    # 2026
    date(2026, 1, 26), date(2026, 3, 6), date(2026, 3, 27), date(2026, 4, 2),
    date(2026, 4, 3), date(2026, 4, 14), date(2026, 5, 1), date(2026, 5, 26),
    date(2026, 9, 4), date(2026, 9, 15), date(2026, 10, 2), date(2026, 10, 21),
    date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 23), date(2026, 12, 25),
}

def trading_days_between(newer_date: date, older_date: date) -> int:
    """Computes the number of trading days (excluding weekends and exchange holidays) between newer_date and older_date."""
    if not newer_date or not older_date:
        return 999
    if older_date > newer_date:
        return -1
    if newer_date == older_date:
        return 0
    days = 0
    curr = older_date
    while curr < newer_date:
        curr += timedelta(days=1)
        if curr.weekday() < 5 and curr not in EXCHANGE_HOLIDAYS:
            days += 1
    return days

def _canonical_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if value.endswith(".NS"):
        return value[:-3]
    if value.endswith(".BO"):
        raise ValueError("BSE symbol supplied to NSE scanner")
    # Backwards compatibility: strip any suffix after dot
    return value.split('.')[0]

def _to_ist_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v)
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        return ts.date()
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{v}': {e}. Returning None.")
        return None

def _req_float(series, key: str) -> Optional[float]:
    if isinstance(series, pd.DataFrame):
        raise TypeError(f"_req_float called with DataFrame (key={key}); use .iloc[-1] to pass a single row")
    if series is None or key not in series:
        return None
    val = series.get(key)
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) > 0 else None
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None

def _opt_float(series, key: str, default: float) -> float:
    res = _req_float(series, key)
    return default if res is None else res

# ── REVERSAL SCORE THRESHOLDS ────────────────────────────────────────────────────────
MIN_REVERSAL_SCORE_PCT = 0.52
MIN_REVERSAL_SCORE     = round(MAX_POSSIBLE_SCORE * MIN_REVERSAL_SCORE_PCT) # 58
CORE_SCORE_MAX         = 25 + 15 + 15 + 15   # 70 pts max for core technical components
CORE_SCORE_FLOOR       = 24                  # Core technical floor (14+3+0+8 = 25 min achievable core after gates; setups < 24 filtered)

REGIME_REVERSAL_PREMIUM = {
    "STRONG_BEAR": 2,
    "BEAR":        1,
    "NEUTRAL":     0,
    "BULL":        0,
    "STRONG_BULL": 0,
}

REGIME_EVIDENCE_REQ = {
    "STRONG_BEAR": {
        "min_vol_ratio": 1.50,
        "min_rr": 2.0,
        "require_obv": True,
        "max_pct_below_sma200": 17.0,
    },
    "BEAR": {
        "min_vol_ratio": 1.35,
        "min_rr": 2.0,
        "require_obv": False,
        "max_pct_below_sma200": 19.0,
    },
}

def _row_get(r: Any, idx: int, key: str, default: Any = None) -> Any:
    """Shape-agnostic helper to safely extract values from tuple, dict, or row objects."""
    if r is None:
        return default
    if isinstance(r, (list, tuple)):
        return r[idx] if len(r) > idx else default
    if isinstance(r, dict):
        return r.get(key, default)
    try:
        return r[key]
    except (KeyError, IndexError, TypeError):
        try:
            return r[idx]
        except (IndexError, TypeError):
            return default



def _is_positive_finite(value: object) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and number > 0
    except (TypeError, ValueError):
        return False

def _is_finite(value: object) -> bool:
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError):
        return False

def _latest_bar_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    for column in ("Datetime", "Date", "Timestamp", "datetime", "date", "timestamp"):
        if column in df.columns:
            values = pd.to_datetime(df[column], errors="coerce")
            if values.notna().any():
                return values.dropna().iloc[-1]
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index[-1]
    return None

def _parse_robust_pct(fd: dict, ratio_key: str, percent_key: str) -> Optional[float]:
    """Safely extracts percentage points from either a V5 decimal ratio or raw Screener percent."""
    if not isinstance(fd, dict): return None
    v_pct = fd.get(percent_key)
    if v_pct is not None and not pd.isna(v_pct) and v_pct != "":
        try: return float(v_pct)
        except: pass
    v_ratio = fd.get(ratio_key)
    if v_ratio is not None and not pd.isna(v_ratio) and v_ratio != "":
        try:
            val = float(v_ratio)
            if abs(val) > 5.0: return val
            return val * 100.0
        except: pass
    return None

def parse_percentage(value: object, unit: str) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    
    text = str(value).strip()
    if text.endswith("%"):
        try:
            return float(text[:-1].strip())
        except (ValueError, TypeError):
            return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if unit == "decimal_ratio":
        return parsed * 100.0
    elif unit == "percentage_points":
        return parsed
    raise ValueError(f"Unsupported percentage unit: {unit}")


def _lookup(m: Optional[dict], sym: str, can: str) -> Optional[float]:
    """Helper to retrieve symbol values from maps without swallowing 0.0 as falsy."""
    if not m:
        return None
    v = m.get(sym)
    return m.get(can) if v is None else v


def _actual_session_fraction(now_t: dtime) -> float:
    """Compute exact unclamped elapsed fraction of the trading day (09:15 to 15:30 IST)."""
    session_start = dtime(9, 15)
    session_end = dtime(15, 30)
    if now_t < session_start or now_t >= session_end:
        return 1.0
    elapsed = (datetime.combine(date.today(), now_t) - datetime.combine(date.today(), session_start)).seconds
    return elapsed / 22500.0


def _session_fraction(now_t: dtime) -> float:
    """Compute clamped elapsed fraction for volume projection."""
    session_start = dtime(9, 15)
    session_end = dtime(15, 30)
    if now_t < session_start or now_t >= session_end:
        return 1.0
    actual = _actual_session_fraction(now_t)
    return max(0.15, actual)


# [RULE 67 CHANGE-RATIONALE]:
# MACD MOMENTUM UPGRADE:
# Previous implementation had two severe flaws:
# 1. macd_recovery_passed boolean bug: Inverted condition where True was assigned only when _macd_momentum_present was False.
# 2. Timing Deadlock: Required an exact crossover within <= 10 bars. Valid active reversals where the crossover occurred
#    12-15 bars earlier and the histogram was actively expanding were starved and rejected.
# Fix: Upgraded MACD evaluator to accept both fresh crossovers (<= 10 bars) AND active bullish state (MACD > Signal with expanding histogram).
def _macd_momentum_present(ticker: pd.DataFrame, atr_val: Optional[float] = None) -> bool:
    """
    State + Freshness test for MACD momentum.
    Allows:
    1. MACD currently above signal (sustained bullish momentum accepted).
    2. MACD currently below signal but with improving histogram (rising) close to crossover.
    """
    if len(ticker) < 3 or not {"MACD", "MACD_SIGNAL"}.issubset(ticker.columns):
        return False
    try:
        macd, sig = ticker["MACD"], ticker["MACD_SIGNAL"]
        above = (macd > sig)
        above_now = bool(above.iloc[-1])

        if above_now:
            return True

        # Fallback for negative but improving histogram near crossover
        hist = macd - sig
        improving = hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]
        
        # Normalized improvement by ATR
        hist_improvement = float(hist.iloc[-1] - hist.iloc[-3])
        norm_denominator = float(atr_val) if (atr_val and atr_val > 0) else 0.02 * float(ticker["Close"].iloc[-1])
        
        meaningful_improvement = (hist_improvement / norm_denominator) >= 0.01
        near_cross = (abs(hist.iloc[-1]) / norm_denominator) <= 0.05
        
        if improving and meaningful_improvement and near_cross:
            return True
            
        return False
    except (TypeError, ValueError, KeyError, IndexError):
        return False


# [RULE 67 CHANGE-RATIONALE]:
# SWING PIVOT STRUCTURE & FALLING-KNIFE HARDENING:
# Previous implementation used a trivial 2-candle check (df['Low'].iloc[-1] >= df['Low'].iloc[-2]) which allowed
# falling knives and dead-cat bounces to pass into alerts.
# Fix: Implemented _check_swing_structure() which analyzes the full 25-bar lookback window to:
# 1. Enforce a hard falling knife block (rejecting if current low breaks below the historical oversold trough).
# 2. Verify genuine Higher-Low (HL) swing formation (current low >= 0.8% above absolute bottom).
# 3. Verify Higher-High (HH) pivot expansion to ensure structural accumulation rather than continuation of downtrend.
def _check_swing_structure(df: pd.DataFrame, lookback: int = 25) -> dict:
    """
    Evaluates genuine swing Higher-Low (HL) / Higher-High (HH) structural bottom formation
    and enforces hard falling-knife protections.
    
    Returns dict:
      - passed: bool
      - is_falling_knife: bool
      - has_higher_low: bool
      - has_higher_high: bool
      - is_strong_structure: bool
      - reason: str
    """
    if df is None or len(df) < 5:
        return {
            "passed": False,
            "is_falling_knife": False,
            "has_higher_low": False,
            "has_higher_high": False,
            "is_strong_structure": False,
            "reason": "Insufficient bars for swing structure analysis"
        }
    
    window = df.iloc[-min(lookback, len(df)):]
    current_bar = df.iloc[-1]
    current_low = float(current_bar["Low"])
    current_high = float(current_bar["High"])
    current_close = float(current_bar["Close"])
    
    # Identify historical trough (lowest low) in prior window (excluding today)
    prior_window = window.iloc[:-1]
    if prior_window.empty:
        return {
            "passed": False,
            "is_falling_knife": False,
            "has_higher_low": False,
            "has_higher_high": False,
            "is_strong_structure": False,
            "reason": "No prior window for trough comparison"
        }
    
    trough_price = float(prior_window["Low"].min())
    trough_idx = prior_window["Low"].idxmin()
    bars_since_trough = len(df) - 1 - df.index.get_loc(trough_idx)
    
    # 1. Hard Falling-Knife Block: If current low touches/breaks below the trough
    if current_low < trough_price * 0.999:
        return {
            "passed": False,
            "is_falling_knife": True,
            "has_higher_low": False,
            "has_higher_high": False,
            "is_strong_structure": False,
            "reason": f"NEW_LOWER_LOW: Current low ₹{current_low:.2f} <= prior trough ₹{trough_price:.2f} (Falling Knife Block)"
        }
    
    # 2. Higher Low (HL) / Higher High (HH) Analysis
    has_higher_low = current_low >= trough_price * 1.008  # at least 0.8% above absolute bottom
    
    # Check if there is an intermediate pivot low after the trough
    has_pivot_hl = False
    if bars_since_trough >= 3:
        post_trough_lows = df["Low"].iloc[-bars_since_trough:-1]
        if not post_trough_lows.empty and float(post_trough_lows.min()) >= trough_price * 0.999:
            has_pivot_hl = True
    
    # Check Higher High over recent 3-5 bars
    recent_highs = df["High"].iloc[-5:-1] if len(df) >= 5 else df["High"].iloc[:-1]
    has_higher_high = current_high >= float(recent_highs.max()) * 0.995 if not recent_highs.empty else False
    
    # Multi-bar low slope (last 3 bars)
    last3_rising = False
    if len(df) >= 3:
        l1, l2, l3 = float(df["Low"].iloc[-1]), float(df["Low"].iloc[-2]), float(df["Low"].iloc[-3])
        if l1 >= l2 >= l3:
            last3_rising = True
            
    passed = (has_higher_low or has_pivot_hl or last3_rising) and not (current_low < trough_price * 0.999)
    
    # Strong structure criteria: Clear HL pivot + price well above trough (> 1.5%) + expanding highs/lows
    is_strong_structure = (current_low >= trough_price * 1.015) and (has_pivot_hl or last3_rising or has_higher_high)
    
    reason = "Swing structure confirmed" if passed else f"No Higher Low structure: low ₹{current_low:.2f} vs trough ₹{trough_price:.2f}"
    return {
        "passed": passed,
        "is_falling_knife": False,
        "has_higher_low": has_higher_low,
        "has_higher_high": has_higher_high,
        "is_strong_structure": is_strong_structure,
        "reason": reason
    }


def _is_climax_top(
    ticker: pd.DataFrame,
    close_price: float,
    candle_high: float,
    candle_low: float,
    vol_ratio: float,
    session_fraction: float = 1.0,
) -> bool:
    """
    [VERSION: REVERSAL_OVERHAUL_v7.0] Blow-off climax top filter.
    Triggers on extreme volume (>= CLIMAX_VOL_MULT x mean, or p95) AND
    when stock has run up >= CLIMAX_MIN_RUNUP_PCT (10%) over VOL_WINDOW_BARS (5 bars).
    """
    if candle_high <= candle_low or len(ticker) < VOL_WINDOW_BARS + 1:
        return False
    try:
        close_nb_ago = float(ticker["Close"].iloc[-(VOL_WINDOW_BARS + 1)])
        if close_nb_ago <= 0:
            return False
        runup = (close_price - close_nb_ago) / close_nb_ago
        if runup < CLIMAX_MIN_RUNUP_PCT:
            return False

        if vol_ratio <= 0:
            return False

        latest_vol = float(ticker["Volume"].iloc[-1])
        if latest_vol <= 0:
            return False

        lookback_ct = min(CLIMAX_VOLUME_LOOKBACK, len(ticker) - 1)
        prior = ticker["Volume"].iloc[-lookback_ct-1:-1]
        if prior.empty:
            return False

        prorated_latest = latest_vol / session_fraction

        vol_spike = (vol_ratio >= CLIMAX_VOL_MULT) or (prorated_latest > float(prior.quantile(CLIMAX_VOL_QUANTILE)))
        if not vol_spike:
            return False

        candle_rng = candle_high - candle_low
        upper_wick_pct = (candle_high - close_price) / candle_rng
        close_pos = (close_price - candle_low) / candle_rng
        return upper_wick_pct > 0.25 and close_pos < 0.45
    except (TypeError, ValueError, KeyError, IndexError):
        return False

# =====================================================================================
# REVERSAL-SPECIFIC SCORING (v7.0 — re-weighted & normalized)
# =====================================================================================

_REV_CATEGORY_SCORES = {
    "Debt-Free Cash Generator": 10, "Wealth Compounder": 10, "Top Bank/NBFC": 10,
    "Long Term Compounder": 10,
    "Dividend Aristocrat": 9,
    "Capital Efficient": 9, "Efficient Lender": 9,
    "Undervalued Growth": 8, "High Momentum": 8, "Fast Growing Financial": 8,
    "Consistent Performer": 6,
    "Blue Chip Stable": 5, "Blue Chip Financial": 5,
    "Recovery Play": 3, "Financial Recovery": 3,
}
_REV_CATEGORY_SCORES_CASEFOLDED = {k.strip().casefold(): v for k, v in _REV_CATEGORY_SCORES.items()}
_REV_CATEGORY_SCORES_SORTED = sorted(_REV_CATEGORY_SCORES.items(), key=lambda kv: (-len(kv[0]), kv[0]))



def _score_reversal(
        vol_ratio: Optional[float],
        drop_pct: float,
        current_rsi: float,
        past_rsi_min: float,
        macd_hist: Optional[float],
        pct_below_sma200: Optional[float],
        category: str,
        rr_ratio: Optional[float],
        trend_score: int = 10,
        obv_trend: Optional[int] = None,
        delivery_pct: Optional[float] = None,
        close_price: Optional[float] = None,
        symbol: Optional[str] = None,
        promoter_pledge_pct: Optional[float] = None,
        atr_val: Optional[float] = None,
        weights: Optional[dict] = None,
        min_drop_floor: float = MIN_DROP_FROM_52W_HIGH,
        delivery_conf: float = 1.0,
        vol_ratio_window: Optional[float] = None,
        available_max: int = COMPONENT_MAX,
        macd_recovery_passed: bool = False,
        pct_below_sma50: Optional[float] = None,
        pct_below_ema20: Optional[float] = None,
) -> dict:
    """Score a reversal setup from 0-100 based on quality dimensions, returning score dict."""
    score = 0

    # ── Trend structure (25 pts) — CORE recovery signal ──
    score += trend_score
    trend_pts = trend_score

    # ── Structure proximity (12 pts max) — Monotonic structural scoring ──
    prox_pts = 0
    if pct_below_sma200 is not None:
        if pct_below_sma200 <= 0.0:
            prox_pts = 12   # At or above SMA200 — Peak structural strength
        elif pct_below_sma200 <= 5.0:
            prox_pts = 12   # Classic tight reversal zone
        elif pct_below_sma200 <= 10.0:
            prox_pts = 9
        elif pct_below_sma200 <= 20.0:
            prox_pts = 5
    elif pct_below_sma50 is not None:
        # Adaptive structural scoring for RECENT_LISTING (50-199 bars)
        if pct_below_sma50 <= 0.0:
            prox_pts = 12   # Above SMA50
        elif pct_below_sma50 <= 5.0:
            prox_pts = 10
        elif pct_below_sma50 <= 10.0:
            prox_pts = 8
        elif pct_below_sma50 <= 20.0:
            prox_pts = 5
    elif pct_below_ema20 is not None:
        # Adaptive structural scoring for FRESH_IPO (35-49 bars)
        if pct_below_ema20 <= 0.0:
            prox_pts = 10   # Above EMA20
        elif pct_below_ema20 <= 5.0:
            prox_pts = 7
        elif pct_below_ema20 <= 10.0:
            prox_pts = 5
    score += prox_pts

    # ── Volume confirmation (15 pts) ──
    vol_pts = 0
    if vol_ratio is not None and vol_ratio >= 3.0:   vol_pts = 15
    elif vol_ratio is not None and vol_ratio >= 2.0: vol_pts = 12
    elif vol_ratio is not None and vol_ratio >= 1.5: vol_pts = 8
    elif vol_ratio is not None and vol_ratio >= MIN_VOLUME_RATIO: vol_pts = 5
    elif vol_ratio_window is not None and vol_ratio_window >= MIN_VOLUME_RATIO:
        vol_pts = 3        # recent accumulation, not today — discounted
    score += vol_pts

    # ── MACD momentum (15 pts) ──
    macd_pts = 0
    if macd_hist is not None and atr_val is not None and atr_val > 0:
        try:
            mh_atr = float(macd_hist) / float(atr_val)
            if mh_atr >= 0.15:   macd_pts = 15
            elif mh_atr >= 0.05: macd_pts = 10
            elif mh_atr > 0.0:   macd_pts = 5
        except (TypeError, ValueError):
            pass
    elif macd_hist is not None and close_price is not None and close_price > 0:
        try:
            mh_norm = (float(macd_hist) / float(close_price)) * 100
            if mh_norm >= 0.5:   macd_pts = 15
            elif mh_norm >= 0.2: macd_pts = 10
            elif mh_norm > 0:    macd_pts = 5
        except (TypeError, ValueError):
            pass

    if macd_pts == 0 and macd_recovery_passed:
        macd_pts = 3
    score += macd_pts

    # ── RSI curl quality (15 pts) — Measured off historical trough ──
    rsi_pts = 0
    rsi_recovery = current_rsi - past_rsi_min
    if rsi_recovery >= 20:   rsi_pts = 15
    elif rsi_recovery >= 12: rsi_pts = 12
    elif rsi_recovery >= 8:  rsi_pts = 8
    elif rsi_recovery >= 3:  rsi_pts = 2
    score += rsi_pts

    # ── Category quality (10 pts) ──
    category_key = category.strip().casefold() if category else ""
    cat_pts = _REV_CATEGORY_SCORES_CASEFOLDED.get(category_key, 0)
    score += cat_pts

    # ── Drop sweet spot / penalty (5 pts) ──
    drop_score = 0
    if 25.0 <= drop_pct <= 40.0:
        drop_score = 5
    elif min_drop_floor <= drop_pct < 25.0:
        drop_score = 3
    elif 40.0 < drop_pct <= MAX_DROP_FROM_52W_HIGH:
        drop_score = 3
    score += drop_score

    # ── R:R quality (5 pts) ──
    rr_score = 0
    if rr_ratio is not None:
        if rr_ratio >= 3.5:   rr_score = 5
        elif rr_ratio >= 2.5: rr_score = 3
        elif rr_ratio >= 2.0: rr_score = 1
        score += rr_score

    # ── OBV confirmation bonus (5 pts) ──
    obv_score = 0
    if obv_trend is not None and obv_trend == 1:
        obv_score = 5
        score += obv_score

    # ── Delivery conviction bonus (5 pts) ──
    deliv_score = 0
    if delivery_pct is not None and delivery_conf > 0:
        deliv_pts = 0
        if delivery_pct >= 50.0:   deliv_pts = 5
        elif delivery_pct >= 35.0: deliv_pts = 3
        elif delivery_pct >= 25.0: deliv_pts = 1
        deliv_score = round(deliv_pts * delivery_conf)
        score += deliv_score

    inst_bonus = 0
    if symbol is not None:
        try:
            from block_deal_detector import compute_inst_bonus
            inst_bonus = compute_inst_bonus(symbol, score)
        except Exception as e:
            logger.warning(f"Error checking institutional footprints in Reversal: {e}")
            inst_bonus = 0

    # ── Bayesian Pledge Penalty ──
    pledge_penalty = 0
    if promoter_pledge_pct is not None and promoter_pledge_pct > 10.0:
        if weights is None:
            max_penalty = DEFAULT_PLEDGE_PENALTY
        else:
            max_penalty = float(weights.get("PLEDGE_PENALTY", DEFAULT_PLEDGE_PENALTY))
        scale = min(1.0, (promoter_pledge_pct - 10.0) / 40.0)
        pledge_penalty = round(abs(max_penalty) * scale)

    core_score = trend_pts + vol_pts + macd_pts + rsi_pts
    evidence_score = score
    raw_score = evidence_score + inst_bonus - pledge_penalty
    clamped_score = max(0, min(evidence_score + inst_bonus, available_max))
    final_score = max(0, clamped_score - pledge_penalty)

    score_breakdown = {
        "trend_score": trend_pts,
        "volume_score": vol_pts,
        "macd_score": macd_pts,
        "rsi_score": rsi_pts,
        "structure_score": prox_pts,
        "quality_score": cat_pts,
        "delivery_score": deliv_score,
        "rr_score": rr_score,
        "drop_score": drop_score,
        "obv_score": obv_score,
        "evidence_score": evidence_score,
        "institutional_bonus": inst_bonus,
        "pledge_penalty": pledge_penalty,
        "final_score": final_score,
        "available_max": available_max,
    }

    return {
        "score": final_score,
        "raw_score": max(0, raw_score),
        "core_score": core_score,
        "evidence_score": evidence_score,
        "score_breakdown": score_breakdown,
    }


def _evaluate_candidate(
    symbol: str,
    df: pd.DataFrame,
    fund_data: Optional[dict] = None,
    regime_ctx: Optional[dict] = None,
    weights: Optional[dict] = None,
    pledge_map: Optional[dict] = None,
    delivery_map: Optional[dict] = None,
    is_synthetic_bar: bool = False,
    is_synthetic_no_vol: bool = False,
    resolved_date: Optional[date] = None,
    is_intraday: bool = False,
    session_fraction: float = 1.0,
) -> dict:
    """Core evaluator logic executing quality gates and returning structured verdict."""
    n_bars = len(df) if df is not None else 0
    if df is None or df.empty or n_bars < REVERSAL_MIN_BARS:
        return {
            "passed": False,
            "reject_reason": f"Insufficient historical bars ({n_bars} < {REVERSAL_MIN_BARS} minimum)",
            "reject_code": "no_data",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # Explicit History Classification & Trend Validation Mode
    if n_bars >= 200:
        history_class = "MATURE"
        trend_validation_mode = "SMA200"
    elif n_bars >= 50:
        history_class = "RECENT_LISTING"
        trend_validation_mode = "SMA50_EMA20"
    else:
        history_class = "FRESH_IPO"
        trend_validation_mode = "EMA20"

    latest = df.iloc[-1]
    close_price = _req_float(latest, "Close")
    candle_high = _req_float(latest, "High")
    candle_low = _req_float(latest, "Low")
    candle_open = _req_float(latest, "Open")
    atr_val = _req_float(latest, "ATR")
    ema20 = _req_float(latest, "EMA20")
    sma50 = _req_float(latest, "SMA50")
    sma200 = _req_float(latest, "SMA200")
    current_rsi = _req_float(latest, "RSI")

    # Extract latest bar timestamp and date safely supporting RangeIndex
    latest_bar_dt = _latest_bar_timestamp(df)
    if latest_bar_dt is None:
        return {
            "passed": False,
            "reject_reason": "DataFrame timestamp is missing or untrustworthy",
            "reject_code": "invalid_timestamp",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }
    latest_bar_date = latest_bar_dt.date()

    # Calculate canonical volume ratio
    vol_ratio = None
    if "Volume" in df.columns and len(df) >= 2:
        lookback = min(20, len(df) - 1)
        prior_vol = df["Volume"].iloc[-lookback - 1:-1].dropna()
        avg_vol = prior_vol.mean() if not prior_vol.empty else 0.0
        
        if avg_vol > 0:
            current_volume = float(latest["Volume"])
            if is_intraday:
                adjusted_volume = current_volume / session_fraction
            else:
                adjusted_volume = current_volume
            vol_ratio = adjusted_volume / avg_vol

    if is_synthetic_no_vol:
        vol_ratio = None

    # Validate all mandatory indicators are present and finite, including ATR
    mandatory_indicators = {
        "Close": close_price,
        "High": candle_high,
        "Low": candle_low,
        "Open": candle_open,
        "RSI": current_rsi,
        "EMA20": ema20,
        "SMA50": sma50,
        "SMA200": sma200,
    }
    missing_ind = [name for name, val in mandatory_indicators.items() if val is None or not _is_finite(val)]
    if missing_ind or not _is_positive_finite(atr_val):
        reasons = []
        if missing_ind:
            reasons.append(f"Missing indicators: {', '.join(missing_ind)}")
        if not _is_positive_finite(atr_val):
            reasons.append("ATR is missing, non-finite, or non-positive")
        return {
            "passed": False,
            "reject_reason": f"Missing or NaN mandatory technical indicators ({'; '.join(reasons)})",
            "reject_code": "bad_indicators",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    if close_price < MIN_STOCK_PRICE:
        return {
            "passed": False,
            "reject_reason": f"Stock price ₹{close_price:.2f} < ₹{MIN_STOCK_PRICE:.0f} minimum quality floor",
            "reject_code": "price_filter",
            "gate_type": "THRESHOLD",
            "operator": "<",
            "actual": close_price,
            "threshold": MIN_STOCK_PRICE,
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    candle_rng = candle_high - candle_low
    if candle_rng < 0:
        return {
            "passed": False,
            "reject_reason": "Negative candle range (corrupt data)",
            "reject_code": "corrupt_data",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }
    elif candle_rng == 0:
        if _safe_float(latest.get("Volume", 0)) <= 0:
            return {
                "passed": False,
                "reject_reason": "Zero range and zero volume (invalid candle)",
                "reject_code": "invalid_candle",
                "score": 0,
                "raw_score": 0,
                "sl_result": {},
                "context": {},
            }
        # Circuit candle allowed; bypass MIN_CANDLE_RANGE_PCT check.
    else:
        candle_rng_pct = (candle_rng / close_price) * 100.0
        if candle_rng_pct < MIN_CANDLE_RANGE_PCT:
            return {
                "passed": False,
                "reject_reason": f"Candle range {candle_rng_pct:.2f}% < {MIN_CANDLE_RANGE_PCT}% minimum volatility threshold",
                "reject_code": "thin_spread",
                "gate_type": "THRESHOLD",
                "operator": "<",
                "score": 0,
                "raw_score": 0,
                "sl_result": {},
                "context": {},
            }

    if "Volume" in df.columns:
        volume_series = pd.to_numeric(df["Volume"], errors="coerce")
        today_date = datetime.now(IST).date()
        if is_intraday and latest_bar_date == today_date:
            volume_series = volume_series.iloc[:-1]
        recent_volume = volume_series.tail(20).dropna()
        if len(recent_volume) < 15:
            return {
                "passed": False,
                "reject_reason": f"Insufficient volume history ({len(recent_volume)} < 15 bars)",
                "reject_code": "volume_filter",
                "score": 0,
                "raw_score": 0,
                "sl_result": {},
                "context": {},
            }
        # Rule: REV-001
        avg_vol_20d = float(recent_volume.mean())
        if avg_vol_20d < MIN_AVG_DAILY_VOLUME:
            return {
                "passed": False,
                "reject_reason": f"20D Avg Volume {avg_vol_20d:,.0f} < {MIN_AVG_DAILY_VOLUME:,.0f} liquidity floor",
                "reject_code": "volume_filter",
                "gate_type": "THRESHOLD",
                "operator": "<",
                "score": 0,
                "raw_score": 0,
                "sl_result": {},
                "context": {},
            }

    if "Volume_Ratio" in df.columns:
        historical_ratios = pd.to_numeric(df["Volume_Ratio"], errors="coerce").iloc[-(VOL_WINDOW_BARS + 1):-1].dropna()
        vol_ratio_max = max([vol_ratio] + historical_ratios.tolist()) if vol_ratio is not None else (float(historical_ratios.max()) if not historical_ratios.empty else 0.0)
    else:
        vol_ratio_max = vol_ratio or 0.0

    high_52w = float(df["High"].iloc[-250:].max()) if len(df) >= 250 else float(df["High"].max())
    if high_52w <= 0:
        return {
            "passed": False,
            "reject_reason": "52W High is non-positive",
            "reject_code": "bad_indicators",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # Rule: REV-001
    drop_pct = ((high_52w - close_price) / high_52w) * 100.0
    cat_str = fund_data.get("Category", "") if fund_data else ""
    is_quality_cat = any(kw in cat_str.lower() for kw in ("wealth", "blue chip", "debt-free"))
    effective_min_drop = QUALITY_CAT_MIN_DROP if is_quality_cat else MIN_DROP_FROM_52W_HIGH

    pct_below_sma200 = None
    if sma200 and sma200 > 0:
        pct_below_sma200 = ((sma200 - close_price) / sma200) * 100.0

    pct_below_sma50 = None
    if sma50 and sma50 > 0:
        pct_below_sma50 = ((sma50 - close_price) / sma50) * 100.0

    pct_below_ema20 = None
    if ema20 and ema20 > 0:
        pct_below_ema20 = ((ema20 - close_price) / ema20) * 100.0

    # [FIX CONDITIONAL_52W_DROP] Allow drawdowns up to 55% only for structurally sound stocks near trend baseline
    if history_class == "MATURE":
        structurally_sound = (pct_below_sma200 is not None and pct_below_sma200 <= 10.0)
    elif history_class == "RECENT_LISTING":
        structurally_sound = (pct_below_sma50 is not None and pct_below_sma50 <= 10.0)
    else:  # FRESH_IPO
        structurally_sound = (pct_below_ema20 is not None and pct_below_ema20 <= 10.0)

    max_allowed_drop = 55.0 if structurally_sound else MAX_DROP_FROM_52W_HIGH

    if drop_pct < effective_min_drop or drop_pct > max_allowed_drop:
        return {
            "passed": False,
            "reject_reason": f"Drop from 52W High {drop_pct:.1f}% outside allowed band [{effective_min_drop:.1f}%, {max_allowed_drop:.1f}%] for {history_class}",
            "reject_code": "drop_band",
            "actual": drop_pct,
            "threshold": f"[{effective_min_drop:.1f}, {max_allowed_drop:.1f}]",
            "gate_type": "THRESHOLD",
            "operator": "NOT_IN_RANGE",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    _rc = regime_ctx or {}
    regime = _rc.get("current_regime") or _rc.get("trend") or "NEUTRAL"
    ev_req = REGIME_EVIDENCE_REQ.get(regime, {})

    max_sma200_limit = MAX_DROP_BELOW_SMA200  # 20.0%
    if "max_pct_below_sma200" in ev_req:
        max_sma200_limit = min(max_sma200_limit, ev_req["max_pct_below_sma200"])

    # Structural Limit Filter: Enforced per History Class
    if history_class == "MATURE" and pct_below_sma200 is not None and pct_below_sma200 > max_sma200_limit:
        return {
            "passed": False,
            "reject_reason": f"[{regime}] Price {pct_below_sma200:.1f}% below SMA200 > {max_sma200_limit}% maximum structural breakdown limit",
            "reject_code": "sma200_filter",
            "gate_type": "THRESHOLD",
            "operator": ">",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }
    elif history_class == "RECENT_LISTING" and pct_below_sma50 is not None and pct_below_sma50 > 20.0:
        return {
            "passed": False,
            "reject_reason": f"[{regime}] Price {pct_below_sma50:.1f}% below SMA50 > 20.0% maximum structural breakdown limit for RECENT_LISTING",
            "reject_code": "sma50_filter",
            "gate_type": "THRESHOLD",
            "operator": ">",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }
    elif history_class == "FRESH_IPO" and pct_below_ema20 is not None and pct_below_ema20 > 15.0:
        return {
            "passed": False,
            "reject_reason": f"[{regime}] Price {pct_below_ema20:.1f}% below EMA20 > 15.0% maximum structural breakdown limit for FRESH_IPO",
            "reject_code": "ema20_filter",
            "gate_type": "THRESHOLD",
            "operator": ">",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # ---------------------------------------------------------------------------------
    # [RULE 67 CHANGE-RATIONALE]:
    # EXPLICIT DUAL-PATH REVERSAL ARCHITECTURE:
    # 1. Elimination of Universal SMA50 Choke:
    #    The legacy reversal scanner required Close >= SMA50 for all stocks, creating an impossible timing
    #    contradiction for stocks 20-45% down from 52W highs (by the time price reclaimed SMA50, RSI trough was expired).
    # 2. Dual-Path Routing:
    #    - Path 1: QUALITY_REVERSAL (Price >= SMA200, or within 5% buffer): Requires EMA20 reclaim (-1.5% ATR tolerance)
    #      and genuine Higher-Low (HL) / Higher-High (HH) swing structure.
    #    - Path 2: DEEP_VALUE_REVERSAL (Price < SMA200, within 20% max structural limit): Requires EMA5 reclaim,
    #      falling-knife block, and ROE >= 12.0% (or Turnaround category) to ensure solvency.
    # ---------------------------------------------------------------------------------
    is_quality_reversal = (pct_below_sma200 is None or pct_below_sma200 <= 0.0)
    reversal_path = "QUALITY_REVERSAL" if is_quality_reversal else "DEEP_VALUE_REVERSAL"

    # Evaluate Genuine Swing Higher-Low Structure & Falling Knife Block
    swing_struct = _check_swing_structure(df, lookback=RSI_TROUGH_LOOKBACK)
    if not swing_struct["passed"]:
        logger.debug(f"🔍 [REVERSAL TELEMETRY] {symbol} rejected at SWING_STRUCTURE: {swing_struct['reason']}")
        return {
            "passed": False,
            "reject_reason": f"[{reversal_path}] Structural Reversal Fail: {swing_struct['reason']}",
            "reject_code": "failed_pattern",
            "gate_type": "BOOLEAN",
            "operator": "==",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # Path-specific Structural Reclaim Gate:
    if reversal_path == "QUALITY_REVERSAL":
        ema_tol = max(0.40 * atr_val, 0.04 * ema20) if atr_val else (0.04 * ema20)
        if close_price < (ema20 - ema_tol):
            logger.debug(f"🔍 [REVERSAL TELEMETRY] {symbol} rejected at QUALITY_EMA20_RECLAIM: Close {close_price:.2f} < EMA20 {ema20:.2f} - tol {ema_tol:.2f}")
            return {
                "passed": False,
                "reject_reason": f"[QUALITY_REVERSAL] Close ₹{close_price:.2f} < EMA20 ₹{ema20:.2f} (ATR tolerance {ema_tol:.2f})",
                "reject_code": "ema_filter",
                "gate_type": "THRESHOLD",
                "operator": "<",
                "score": 0,
                "raw_score": 0,
                "sl_result": {},
                "context": {},
            }
    else:
        # DEEP_VALUE_REVERSAL: Reclaim EMA5 multi-bar confirmation floor
        ema5 = float(df["Close"].ewm(span=5, adjust=False).mean().iloc[-1]) if len(df) >= 5 else close_price
        ema5_tol = max(0.25 * atr_val, 0.01 * ema5) if atr_val else (0.01 * ema5)
        if close_price < (ema5 - ema5_tol):
            logger.debug(f"🔍 [REVERSAL TELEMETRY] {symbol} rejected at DEEP_VALUE_EMA5_RECLAIM: Close {close_price:.2f} < EMA5 {ema5:.2f} - tol {ema5_tol:.2f}")
            return {
                "passed": False,
                "reject_reason": f"[DEEP_VALUE_REVERSAL] Close ₹{close_price:.2f} < EMA5 ₹{ema5:.2f} (ATR tolerance {ema5_tol:.2f})",
                "reject_code": "ema_filter",
                "gate_type": "THRESHOLD",
                "operator": "<",
                "score": 0,
                "raw_score": 0,
                "sl_result": {},
                "context": {},
            }

    obv_trend = None
    if "OBV_Trend" in latest and not pd.isna(latest["OBV_Trend"]):
        try:
            obv_trend = int(latest["OBV_Trend"])
        except (ValueError, TypeError):
            pass

    if ev_req.get("require_obv") and obv_trend != 1:
        return {
            "passed": False,
            "reject_reason": f"[{regime}] OBV trend is not accumulating (OBV_Trend={obv_trend})",
            "reject_code": "regime_obv",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # Fundamental quality check
    REQUIRE_FUNDAMENTALS = REVERSAL_CONFIG.get("REQUIRE_FUNDAMENTALS", True)
    roe_val = _parse_robust_pct(fund_data, "roe", "ROE %") if fund_data else None
    rev_growth = _parse_robust_pct(fund_data, "yoy_revenue", "YOY Revenue %") if fund_data else None
    if rev_growth is None and fund_data:
        rev_growth = _parse_robust_pct(fund_data, "revenue_cagr_3y", "Revenue CAGR 3Y")
    if rev_growth is None and fund_data:
        rev_growth = _parse_robust_pct(fund_data, "yoy_profit", "YOY Profit %")
    if rev_growth is None and fund_data and (roe_val is not None or fund_data.get("score") is not None):
        rev_growth = 0.0  # Fallback for lightweight production cache entries

    if REQUIRE_FUNDAMENTALS and (roe_val is None or rev_growth is None):
        return {
            "passed": False,
            "reject_reason": "Fundamentals unavailable (fail-closed)",
            "reject_code": "fundamental_filter",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # [RULE 67 CHANGE-RATIONALE]:
    # PATH-SPECIFIC FUNDAMENTAL ROE THRESHOLDS:
    # - Quality Reversals (above SMA200): Minimum ROE floor is 5.0% (quality category floor).
    # - Deep Value Reversals (below SMA200): Minimum ROE floor is tightened to 12.0% to guard against value traps,
    #   while exempting verified Turnarounds.
    cat_str = fund_data.get("Category", "") if fund_data else ""
    is_turnaround = "TURNAROUND" in cat_str.upper()

    if reversal_path == "QUALITY_REVERSAL":
        path_min_roe = -100.0 if is_turnaround else MIN_ROE  # 5.0%
    else:
        deep_val_min_roe = REVERSAL_CONFIG.get("DEEP_VALUE_MIN_ROE", 12.0)
        path_min_roe = -100.0 if is_turnaround else deep_val_min_roe

    TURNAROUND_MIN_REV_GROWTH = -5.0
    if roe_val is not None and (roe_val < path_min_roe or not -100.0 <= roe_val <= 500.0):
        reason = f"[{reversal_path}] ROE {roe_val:.1f}% < {path_min_roe}% floor" if roe_val < path_min_roe else f"ROE {roe_val:.1f}% out of plausible range"
        logger.debug(f"🔍 [REVERSAL TELEMETRY] {symbol} rejected at FUNDAMENTAL_ROE: {reason}")
        return {
            "passed": False,
            "reject_reason": reason,
            "reject_code": "fundamental_filter",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    if rev_growth is not None and (rev_growth < TURNAROUND_MIN_REV_GROWTH or not -100.0 <= rev_growth <= 1000.0):
        reason = f"YoY Revenue Growth {rev_growth:.1f}% < {TURNAROUND_MIN_REV_GROWTH}% turnaround floor" if rev_growth < TURNAROUND_MIN_REV_GROWTH else f"YoY Revenue Growth {rev_growth:.1f}% out of plausible range"
        return {
            "passed": False,
            "reject_reason": reason,
            "reject_code": "fundamental_filter",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    rsi_series = df["RSI"].dropna()
    if len(rsi_series) < 5:
        return {
            "passed": False,
            "reject_reason": "RSI data insufficient (too many NaN values in historical window)",
            "reject_code": "bad_indicators",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    past_rsi_min = None
    bars_since_trough = 99
    
    # Find the most recent valid RSI trough (local minimum)
    search_start = max(1, len(rsi_series) - RSI_TROUGH_LOOKBACK - 1)
    for i in range(len(rsi_series) - 2, search_start - 1, -1):
        val = rsi_series.iloc[i]
        prev_val = rsi_series.iloc[i-1]
        next_val = rsi_series.iloc[i+1]
        
        if val <= prev_val and val <= next_val:
            # Local trough found
            if val <= RSI_OVERSOLD_THRESHOLD:
                past_rsi_min = float(val)
                bars_since_trough = len(df) - 1 - df.index.get_loc(rsi_series.index[i])
                break

    # Fallback to absolute minimum if no structural trough is found
    if past_rsi_min is None:
        _n = RSI_TROUGH_LOOKBACK
        rsi_window = rsi_series.iloc[-_n-1:-1] if len(rsi_series) >= _n + 1 else rsi_series.iloc[:-1]
        past_rsi_min = float(rsi_window.min())
        trough_idx = rsi_window.idxmin()
        bars_since_trough = len(df) - 1 - df.index.get_loc(trough_idx)

    rsi_recovery = current_rsi - past_rsi_min

    if current_rsi < RSI_CURL_MIN or past_rsi_min > RSI_OVERSOLD_THRESHOLD or rsi_recovery < MIN_RSI_RECOVERY or bars_since_trough > MAX_TROUGH_AGE:
        return {
            "passed": False,
            "reject_reason": f"RSI condition failed: current RSI={current_rsi:.1f} (min {RSI_CURL_MIN}), min RSI={past_rsi_min:.1f} (max {RSI_OVERSOLD_THRESHOLD}), bounce={rsi_recovery:.1f} (min {MIN_RSI_RECOVERY}), trough age={bars_since_trough}b (max {MAX_TROUGH_AGE}b)",
            "reject_code": "failed_pattern",
            "actual": False,
            "threshold": True,
            "gate_type": "BOOLEAN",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    if len(df) >= 5:
        rsi_tail = df["RSI"].iloc[-4:].dropna()
        if len(rsi_tail) >= 4:
            d1 = float(rsi_tail.iloc[-1] - rsi_tail.iloc[-2])
            d2 = float(rsi_tail.iloc[-2] - rsi_tail.iloc[-3])
            d3 = float(rsi_tail.iloc[-3] - rsi_tail.iloc[-4])
            agg_decline = float(rsi_tail.iloc[-4] - rsi_tail.iloc[-1])
            if d1 < 0 and d2 < 0 and d3 < 0 and agg_decline >= 1.5:
                return {
                    "passed": False,
                    "reject_reason": f"RSI continuously declining over last 4 bars (agg decline={agg_decline:.2f}): {list(rsi_tail.tail(4).round(2))}",
                    "reject_code": "failed_pattern",
                    "actual": False,
                    "threshold": True,
                    "gate_type": "BOOLEAN",
                    "score": 0,
                    "raw_score": 0,
                    "sl_result": {},
                    "context": {},
                }

    # [RULE 67 CHANGE-RATIONALE]:
    # MACD DUAL-MODE VALIDATION:
    # 1. Fresh Crossover Mode: Exact bullish crossover occurring within the last 10 bars.
    # 2. Active Bullish State Mode: MACD > Signal with strictly expanding histogram over recent bars,
    #    allowing ongoing multi-week reversals to qualify without timing starvation.
    macd = df.get("MACD")
    sig = df.get("MACD_SIGNAL")
    macd_crossover_passed = False
    macd_active_bullish = False

    if macd is not None and sig is not None and len(df) >= 3:
        # 1. Fresh crossover within last 10 bars
        for i in range(len(df) - 1, max(0, len(df) - 11), -1):
            if macd.iloc[i] > sig.iloc[i] and macd.iloc[i-1] <= sig.iloc[i-1]:
                macd_crossover_passed = True
                break
        
        # 2. Active Bullish State with rising/expanding histogram
        macd_now = float(macd.iloc[-1])
        sig_now = float(sig.iloc[-1])
        hist_now = macd_now - sig_now
        hist_prev = float(macd.iloc[-2]) - float(sig.iloc[-2])
        hist_3_ago = float(macd.iloc[-3]) - float(sig.iloc[-3]) if len(df) >= 4 else hist_prev

        # Expanding histogram: strictly rising or positive delta over last 3 bars
        is_hist_expanding = (hist_now > hist_prev) or (hist_now >= hist_prev and (hist_now - hist_3_ago) > 0)

        if macd_now > sig_now and is_hist_expanding:
            macd_active_bullish = True

    if not (macd_crossover_passed or macd_active_bullish):
        logger.debug(f"🔍 [REVERSAL TELEMETRY] {symbol} rejected at MACD_MOMENTUM: No fresh crossover and no active expanding histogram")
        return {
            "passed": False,
            "reject_reason": "MACD Momentum Fail: No bullish MACD crossover within last 10 bars and no active rising MACD histogram",
            "reject_code": "macd_stale",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # [RULE 67 CHANGE-RATIONALE]:
    # CONDITIONAL RVOL FLOOR & REGIME SCALING:
    # - Base Floor: 1.35x RVOL (down from rigid 2.0x).
    # - Structure Requirement: 1.35x - 1.49x RVOL requires confirmed strong Higher-Low swing structure.
    # - STRONG_BEAR Regime: Enforces a strict 1.50x RVOL floor + OBV accumulation.
    # - 2.0x+ Volume is rewarded via scoring bonuses rather than acting as a binary filter.
    min_vol_floor = 1.50 if regime == "STRONG_BEAR" else MIN_VOLUME_RATIO
    if is_synthetic_no_vol or vol_ratio is None or vol_ratio < min_vol_floor:
        reason = "Missing volume data on synthetic bar" if is_synthetic_no_vol or vol_ratio is None else f"Volume ratio {vol_ratio:.2f}x < {min_vol_floor}x threshold"
        logger.debug(f"🔍 [REVERSAL TELEMETRY] {symbol} rejected at VOLUME_FLOOR: {reason}")
        return {
            "passed": False,
            "reject_reason": reason,
            "reject_code": "low_volume",
            "actual": vol_ratio if vol_ratio else 0.0,
            "threshold": min_vol_floor,
            "gate_type": "THRESHOLD",
            "operator": "<",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # Conditional volume requirement: Moderate volume (1.35x to 1.49x) requires strong HL structure
    if vol_ratio < 1.50 and not swing_struct.get("is_strong_structure", False):
        logger.debug(f"🔍 [REVERSAL TELEMETRY] {symbol} rejected at CONDITIONAL_VOLUME: RVOL {vol_ratio:.2f}x lacks strong swing structure")
        return {
            "passed": False,
            "reject_reason": f"Moderate volume {vol_ratio:.2f}x requires strong Higher-Low swing structure (structure not strong enough)",
            "reject_code": "weak_reversal_structure",
            "gate_type": "THRESHOLD",
            "operator": "<",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    # Defer climax check during early market hours using the raw unclamped fraction
    actual_frac = _actual_session_fraction(datetime.now(IST).time())
    climax_check_available = not (is_intraday and actual_frac < 0.25)
    
    is_climax = False
    if climax_check_available:
        is_climax = _is_climax_top(df, close_price, candle_high, candle_low, vol_ratio=vol_ratio, session_fraction=session_fraction)

    if is_climax:
        return {
            "passed": False,
            "reject_reason": "Climax top detected (record volume with upper wick dump)",
            "reject_code": "climax_top",
            "score": 0,
            "raw_score": 0,
            "sl_result": {},
            "context": {},
        }

    swing_low_val = float(df["Low"].iloc[-RSI_TROUGH_LOOKBACK:].min()) if len(df) >= RSI_TROUGH_LOOKBACK else float(df["Low"].min())
    swing_low_raw_val = float(df["Low"].iloc[-5:].min()) if len(df) >= 5 else float(df["Low"].min())
    swing_high_raw_val = float(df["High"].iloc[-250:].max()) if len(df) >= 250 else float(df["High"].max())

    sl_res = compute_sl_and_target(
        entry_price=close_price,
        atr=atr_val,
        mode="REVERSAL",
        ticker=df,
        swing_low=swing_low_val,
        swing_low_raw=swing_low_raw_val,
        swing_high_raw=swing_high_raw_val,
        sma50=sma50,
        sma200=sma200,
        ema20=ema20,
    )
    sl_passed = sl_res.get("passed", not sl_res.get("is_rejected", False))
    natural_rr = sl_res.get("natural_rr", 0.0)
    if not sl_passed or natural_rr < 2.0:
        rej_msg = sl_res.get("rejection_reason") or sl_res.get("reject_reason") or f"Natural RR {natural_rr:.2f}x < 2.0x"
        return {
            "passed": False,
            "reject_reason": f"R:R filter: {rej_msg}",
            "reject_code": "low_rr",
            "gate_type": "THRESHOLD",
            "operator": "<",
            "score": 0,
            "raw_score": 0,
            "sl_result": sl_res,
            "context": {},
        }

    can_sym = _canonical_symbol(symbol)
    delivery_pct = _lookup(delivery_map, symbol, can_sym)
    scan_date = datetime.now(IST).date()
    delivery_age = trading_days_between(scan_date, resolved_date) if resolved_date else 99
    if delivery_age < 0:
        delivery_conf = 0.0
        delivery_pct = None
    else:
        delivery_conf = 1.0 if delivery_age == 0 else (0.5 if delivery_age == 1 else 0.0)

    # Normalize maximum for unavailable optional delivery evidence
    available_max = COMPONENT_MAX
    if delivery_pct is None or delivery_conf == 0.0:
        available_max -= 5
    elif delivery_conf < 1.0:
        available_max -= 2

    available_max = max(40, available_max)
    score_premium = REGIME_REVERSAL_PREMIUM.get(regime, 0)
    effective_min_score = round((MIN_REVERSAL_SCORE + score_premium) * available_max / COMPONENT_MAX)
    # Ensure minimum score floor is scaled proportionally to available_max (never exceed 80% of available max)
    effective_min_score = min(effective_min_score, int(available_max * 0.80))
    effective_min_score = max(35, effective_min_score)

    sma50_series = df["SMA50"].dropna()
    sma50_slope_up = (len(sma50_series) >= 6 and float(sma50_series.iloc[-1]) > float(sma50_series.iloc[-6]))
    close_above_ema20_strict = (close_price > ema20)
    above_sma50 = (close_price > sma50) if sma50 else False
    above_sma200 = (close_price > sma200) if sma200 else False

    if above_sma200 and above_sma50:
        trend_score = 25
    elif above_sma50:
        trend_score = 22
    elif above_sma200 or close_above_ema20_strict:
        trend_score = 18
    elif is_quality_reversal or (close_price >= ema5 if 'ema5' in locals() else True):
        trend_score = 15
    else:
        trend_score = 12

    pledge_pct = _lookup(pledge_map, symbol, can_sym)

    # Evaluate whether MACD is currently above signal using a single calculated hist definition
    macd_val = _req_float(latest, "MACD")
    sig_val = _req_float(latest, "MACD_SIGNAL")
    macd_hist = None
    if macd_val is not None and sig_val is not None:
        macd_hist = macd_val - sig_val
    macd_above_now = (macd_val > sig_val) if (macd_val is not None and sig_val is not None) else False
    # [FIX: MACD_RECOVERY_BOOLEAN] Fixed logical inversion: bullish MACD is rewarded, not penalized
    macd_recovery_passed = macd_crossover_passed or macd_above_now

    score_dict = _score_reversal(
        vol_ratio=vol_ratio,
        drop_pct=drop_pct,
        current_rsi=current_rsi,
        past_rsi_min=past_rsi_min,
        macd_hist=macd_hist,
        pct_below_sma200=pct_below_sma200,
        category=cat_str,
        rr_ratio=sl_res.get("natural_rr"),
        trend_score=trend_score,
        obv_trend=obv_trend,
        delivery_pct=delivery_pct,
        close_price=close_price,
        symbol=symbol,
        promoter_pledge_pct=pledge_pct,
        atr_val=atr_val,
        weights=weights,
        min_drop_floor=effective_min_drop,
        delivery_conf=delivery_conf,
        vol_ratio_window=vol_ratio_max,
        available_max=available_max,
        macd_recovery_passed=macd_recovery_passed,
        pct_below_sma50=pct_below_sma50,
        pct_below_ema20=pct_below_ema20,
    )
    score = score_dict["score"]
    raw_score = score_dict["raw_score"]
    core_score = score_dict["core_score"]
    evidence_score = score_dict["evidence_score"]

    if core_score < CORE_SCORE_FLOOR:
        return {
            "passed": False,
            "reject_reason": f"Core technical score {core_score} < {CORE_SCORE_FLOOR} minimum quality floor",
            "reject_code": "weak_core",
            "score": score,
            "raw_score": raw_score,
            "sl_result": sl_res,
            "context": {},
        }

    if evidence_score < effective_min_score:
        return {
            "passed": False,
            "reject_reason": f"Evidence score {evidence_score} < {effective_min_score} minimum threshold (regime: {regime})",
            "reject_code": "low_score",
            "score": score,
            "raw_score": raw_score,
            "sl_result": sl_res,
            "context": {},
        }

    pledge_penalty = score_dict["score_breakdown"]["pledge_penalty"]
    governance_score = evidence_score - pledge_penalty
    if governance_score < effective_min_score:
        return {
            "passed": False,
            "reject_reason": f"Governance-adjusted score {governance_score} < {effective_min_score} minimum threshold due to promoter pledge risk",
            "reject_code": "governance_adjusted_score",
            "score": score,
            "raw_score": raw_score,
            "sl_result": sl_res,
            "context": {},
        }

    signals = []
    signals.append(f"🏷️ History Tier: {history_class} ({trend_validation_mode})")
    if close_price >= ema20:
        if above_sma50:
            signals.append("🎯 Reclaimed 20 EMA & SMA50")
        else:
            signals.append("🎯 Reclaimed 20 EMA (below SMA50)")
    else:
        if above_sma50:
            signals.append("🎯 Holding Near 20 EMA & SMA50")
        else:
            signals.append("🎯 Holding Near 20 EMA (below SMA50)")

    signals.append(f"📉 Down {drop_pct:.1f}% from 52W High")
    signals.append(f"🔄 RSI Oversold Bounce (RSI={current_rsi:.1f}, min={past_rsi_min:.1f})")
    if vol_ratio is not None:
        signals.append(f"📊 Vol Ratio {vol_ratio:.1f}x (20D Avg)")
    if pct_below_sma200 is not None:
        signals.append(f"📍 {pct_below_sma200:.1f}% below SMA200")
    elif pct_below_sma50 is not None:
        signals.append(f"📍 {pct_below_sma50:.1f}% below SMA50")
    if obv_trend == 1:
        signals.append("🟢 OBV Accumulation")

    target_val = sl_res.get("target_1") or sl_res.get("target")

    context = {
        "score": score,
        "raw_score": raw_score,
        "core_score": core_score,
        "trend_score": trend_score,
        "entry_price": close_price,
        "drop_pct": round(drop_pct, 1),
        "stop_loss": sl_res.get("stop_loss"),
        "target": target_val,
        "target_1": target_val,
        "risk_reward": sl_res.get("natural_rr"),
        "regime": regime,
        "history_class": history_class,
        "trend_validation_mode": trend_validation_mode,
        "effective_min_score": effective_min_score,
        "available_max": available_max,
        "score_breakdown": score_dict["score_breakdown"],
        "signals": signals,
        "technicals": {
            "rsi": current_rsi,
            "ema20": ema20,
            "sma50": sma50,
            "sma200": sma200,
            "volume_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "volume_ratio_session_adjusted": is_intraday,
            "drop_from_52w_high": round(drop_pct, 1),
            "pct_below_sma200": round(pct_below_sma200, 1) if pct_below_sma200 is not None else None,
            "pct_below_sma50": round(pct_below_sma50, 1) if pct_below_sma50 is not None else None,
            "history_class": history_class,
            "trend_validation_mode": trend_validation_mode,
        }
    }

    return {
        "passed": True,
        "reject_reason": None,
        "score": score,
        "raw_score": raw_score,
        "sl_result": sl_res,
        "context": context,
    }


def evaluate_reversal_symbol(symbol: str, ticker: pd.DataFrame, fund_data: dict = None, regime_ctx: dict = None) -> dict:
    """Public UI evaluator delegating directly to _evaluate_candidate for 100% parity."""
    try:
        ticker = apply_indicators(ticker, timeframe="1d")
        if ticker is None or ticker.empty:
            return {"status": "NO", "reasons": ["Failed to calculate technical indicators"], "score": 0, "qualified": False}
    except Exception as exc:
        return {
            "status": "ERROR",
            "reasons": [f"Indicator calculation failed: {exc}"],
            "score": 0,
            "qualified": False,
        }

    regime_ctx = regime_ctx or {}
    regime_str = regime_ctx.get("trend") or regime_ctx.get("current_regime") or "NEUTRAL"
    if "trend" not in regime_ctx:
        regime_ctx["trend"] = regime_str
    if "current_regime" not in regime_ctx:
        regime_ctx["current_regime"] = regime_str

    try:
        from database import get_latest_weights, get_pledge_map
        _wts = get_latest_weights(regime_str)
        weights = _wts.get("weights") if _wts else None
        pledge_map = get_pledge_map([symbol])
    except Exception:
        weights, pledge_map = None, {}

    try:
        from delivery_data import fetch_latest_available_delivery_data
        today_ist_date = datetime.now(IST).date()
        delivery_map, resolved_date = fetch_latest_available_delivery_data(today_ist_date)
    except Exception:
        today_ist_date = datetime.now(IST).date()
        delivery_map = {}
        resolved_date = today_ist_date

    verdict = _evaluate_candidate(
        symbol=symbol,
        df=ticker,
        fund_data=fund_data,
        regime_ctx=regime_ctx,
        weights=weights,
        pledge_map=pledge_map,
        delivery_map=delivery_map,
        resolved_date=resolved_date,
    )
    # ── PER-STOCK TERMINAL TELEMETRY DUMP (Section 4 & 8) ──
    try:
        from scanner_telemetry import DecisionContext, telemetry_engine
        ctx = DecisionContext(symbol=symbol, scanner_name="REVERSAL")
        latest = ticker.iloc[-1]
        ctx.capture_dataframe_row(latest, is_fallback=False)
        ctx.capture_raw_market(
            open_p=_safe_float(latest.get("Open")),
            high_p=_safe_float(latest.get("High")),
            low_p=_safe_float(latest.get("Low")),
            close_p=_safe_float(latest.get("Close")),
            volume=_safe_float(latest.get("Volume")),
            high_52w=_safe_float(latest.get("HIGH_52W")),
            low_52w=_safe_float(latest.get("LOW_52W"))
        )
        ctx.capture_indicators(
            rsi=_safe_float(latest.get("RSI")),
            ema20=_safe_float(latest.get("EMA20")),
            sma50=_safe_float(latest.get("SMA50")),
            sma200=_safe_float(latest.get("SMA200")),
            macd=_safe_float(latest.get("MACD")),
            vol_ratio=_safe_float(latest.get("Volume_Ratio"))
        )
        ctx.capture_score("TOTAL", verdict.get("score", 0.0), 100.0)
        if verdict.get("sl_result"):
            sl_res = verdict["sl_result"]
            ctx.capture_sl_target(_safe_float(latest.get("Close")), sl_res.get("stop_loss", 0.0), sl_res.get("target_1", 0.0))
            
        consumed_fields = {
            "Close": _safe_float(latest.get("Close")),
            "High": _safe_float(latest.get("High")),
            "Low": _safe_float(latest.get("Low")),
            "Open": _safe_float(latest.get("Open")),
            "Volume": _safe_float(latest.get("Volume")),
            "RSI": _safe_float(latest.get("RSI")),
            "EMA20": _safe_float(latest.get("EMA20")),
            "SMA50": _safe_float(latest.get("SMA50")),
            "SMA200": _safe_float(latest.get("SMA200")),
            "MACD": _safe_float(latest.get("MACD")),
            "Volume Ratio": _safe_float(latest.get("Volume_Ratio")),
            "High 52W": _safe_float(latest.get("HIGH_52W")),
            "Low 52W": _safe_float(latest.get("LOW_52W"))
        }
        for k, v in consumed_fields.items():
            if v is not None and not pd.isna(v):
                ctx.add_decision_input(name=k, value=v, source="ReversalScanner", as_of="Live", freshness="LIVE", required=True, valid=True)

        ctx.finalize(decision="SELECTED" if verdict["passed"] else "REJECTED", primary_reason=verdict.get("reject_reason", ""))
        telemetry_engine.emit_terminal(ctx)
    except Exception as telemetry_err:
        logger.debug(f"Telemetry recording skipped: {telemetry_err}")

    if verdict["passed"]:
        return {
            "status": "CORE MET",
            "score": verdict["score"],
            "raw_score": verdict["raw_score"],
            "qualified": True,
            "sl_result": verdict["sl_result"],
            "context": verdict["context"],
        }
    else:
        return {
            "status": "NO",
            "reasons": [verdict["reject_reason"]],
            "score": verdict["score"],
            "raw_score": verdict["raw_score"],
            "qualified": False,
        }






# [VERSION: PERF_PROFILER_v1.0] Wrap the reversal scan body so every run
# reports wall-clock time, RSS delta, and any top-level exception via structured log.
@profile_timing("reversal_scanner._run_scan", log_to_file=True)
def _run_scan(force: bool = False, session=None, run_ctx=None):
    """Execute a single reversal scan pass. Called inside the scheduling loop."""
    from database import (
        is_scanner_stopped,
        upsert_scanner_health,
        get_latest_weights,
        get_recent_alerts_for_scanner,
        delete_todays_alerts_for_scanner,
        get_all_failed_reversal_cooldown_symbols,
        get_connection,
    )
    if is_scanner_stopped("REVERSAL"):
        logger.info("🛑 Reversal Scanner is STOPPED by Admin. Skipping execution.")
        upsert_scanner_health("REVERSAL", "STOPPED", error_msg="REVERSAL scanner is explicitly disabled by admin.")
        return 0

    # [VERSION: PERF_PHASE0_v1.0] Reset stage timer ring buffer + capture scan start
    import time as _time_mod
    from perf_utils import reset_stage_timers, ScannerStageTracker
    _scan_start = _time_mod.perf_counter()
    reset_stage_timers()

    ist_now = datetime.now(IST)
    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 [START] REVERSAL SCANNER INIT | {ist_now.strftime('%Y-%m-%d %H:%M:%S')} 🚀")
    logger.info("=" * 80 + "\n")

    stage_tracker = ScannerStageTracker("REVERSAL_SCANNER")
    stage_tracker.start_stage(1, "Macro Regime & Watchlist Prep", "Fetching market regime, weights, delivery map, and pledge data")

    try:
        from macro_utils import MarketRegimeEngine, get_macro_regime
        regime_ctx = MarketRegimeEngine.get_regime_context()
        regime_str = get_macro_regime()
        regime_ctx["current_regime"] = regime_str
        regime_ctx["trend"] = regime_str
    except Exception as e:
        logger.warning(f"Failed to fetch market regime: {e}. Defaulting to NEUTRAL.")
        regime_ctx = {"current_regime": "NEUTRAL", "trend": "NEUTRAL", "biases": {}}

    try:
        _regime_for_wts = regime_ctx.get("trend", "NEUTRAL")
        weights_row = get_latest_weights(_regime_for_wts)
        bayesian_weights = weights_row.get("weights") if weights_row else None
    except Exception as e:
        logger.warning(f"Failed to fetch Bayesian weights: {e}")
        bayesian_weights = None

    try:
        watchlist = get_watchlist("REVERSAL")
    except Exception as e:
        logger.exception("Failed to load watchlist for REVERSAL scanner")
        upsert_scanner_health("REVERSAL", "DOWN", error_msg=f"Watchlist load failed: {str(e)[:200]}")
        raise RuntimeError(f"Failed to load watchlist for REVERSAL: {e}")

    if watchlist.empty:
        logger.warning("REVERSAL watchlist is empty. Nothing to scan.")
        upsert_scanner_health("REVERSAL", "IDLE", error_msg="Watchlist is empty.")
        return 0

    stage_tracker.total_symbols = len(watchlist)
    stage_tracker.end_stage(f"Loaded {len(watchlist)} symbols")
    logger.info(f"📊 Loaded {len(watchlist)} symbols for REVERSAL scan.")

    stage_tracker.start_stage(2, "Pre-Scan Context & Delivery Fetch", "Pledge map, delivery map, sector ratings")
    today_ist_date = ist_now.date()

    resolved_date = today_ist_date
    try:
        from delivery_data import fetch_latest_available_delivery_data
        prev_delivery_map, resolved_date = fetch_latest_available_delivery_data(today_ist_date)
    except Exception as e:
        logger.warning(f"Failed to fetch delivery data: {e}")
        prev_delivery_map = {}

    try:
        from database import get_pledge_map
        symbols = [str(s) for s in watchlist["Stock"].tolist() if s]
        pledge_map = get_pledge_map(symbols)
        logger.info(f"🛡️ Fetched pledge data for {len(pledge_map)} symbols")
    except Exception as e:
        logger.exception("Failed to fetch pledge map")
        pledge_map = {}

    all_universe_symbols = [str(s) for s in watchlist["Stock"].tolist() if s]
    terminal_tracker = SingleTerminalTracker(all_universe_symbols, scanner_name="REVERSAL")
    terminal_tracker.map_gates_to_stage("POLICY_FILTER", [
        "BLACKLIST_EXCLUDED", "COOLDOWN_ACTIVE"
    ])
    terminal_tracker.map_gates_to_stage("VALID_DATA", [
        "DATA_UNAVAILABLE", "INVALID_TIMESTAMP", "STALE_DATA", "PROCESSING_ERROR"
    ])
    terminal_tracker.map_gates_to_stage("TECHNICAL_SETUP", [
        "NOT_IN_DROP_BAND", "DROP_TOO_SHALLOW", "DROP_TOO_EXTREME", "BELOW_SMA50",
        "RSI_NOT_OVERSOLD", "NO_SMA50_RECLAIM", "BEAR_VOLUME_HIGH", "POOR_RR", "RISK_REJECTED"
    ])
    terminal_tracker.map_gates_to_stage("FINAL_ALERTS", [
        "SUPPRESSED_TOP_N", "DUPLICATE_ALERT", "ALERT_GENERATED"
    ])
    waterfall = StageWaterfallTracker(["UNIVERSE_PREP", "POLICY_FILTER", "VALID_DATA", "TECHNICAL_SETUP", "FINAL_ALERTS"])
    waterfall.set_stage_count("UNIVERSE_PREP", len(watchlist))

    total_alerts = 0
    shortlisted_alerts = []
    rejected = defaultdict(int)
    # [RULE 67 CHANGE-RATIONALE]: Explicitly extract scan_id from run_ctx to avoid UnboundLocalError / NameError.
    scan_id = getattr(run_ctx, "run_id", "run_1") if run_ctx else "run_1"
    telemetry_logger = ScannerDecisionLogger("REVERSAL", scan_id, regime_str)

    today_str = ist_now.strftime("%Y-%m-%d")

    live_blacklist_raw = get_live_blacklist()
    live_blacklist = {_canonical_symbol(s) for s in live_blacklist_raw if s} if live_blacklist_raw else set()
    failed_cooldown_raw = get_all_failed_reversal_cooldown_symbols(REVERSAL_COOLDOWN_TRADING_DAYS)
    failed_cooldown_syms = {_canonical_symbol(s) for s in failed_cooldown_raw if s} if failed_cooldown_raw else set()

    cooldown_alerts = get_recent_alerts_for_scanner("REVERSAL", 3 * 1440, only_active=True)
    today_ist = ist_now.date()
    cooldown_syms = set()
    for a in cooldown_alerts:
        if not a:
            continue
        sym = _row_get(a, 0, "symbol")
        created_str = _row_get(a, 1, "created_at")
        d = _to_ist_date(created_str)
        if d is None:
            logger.warning(f"Invalid cooldown timestamp for {sym}. Failing conservatively by adding to cooldown.")
            if sym:
                cooldown_syms.add(_canonical_symbol(sym))
            continue
        if d == today_ist:
            continue
        if sym:
            cooldown_syms.add(_canonical_symbol(sym))

    # Pre-filter blacklist/cooldown symbols from watchlist before chunking and fetching
    excluded_symbols = set(live_blacklist)
    if not force:
        excluded_symbols.update(cooldown_syms)
        excluded_symbols.update(failed_cooldown_syms)

    scan_watchlist = watchlist[
        ~watchlist["Stock"].map(_canonical_symbol).isin(excluded_symbols)
    ].copy()

    waterfall.set_stage_count("POLICY_FILTER", len(scan_watchlist))
    for s in all_universe_symbols:
        can_s = _canonical_symbol(s)
        if can_s in live_blacklist:
            terminal_tracker.record_terminal(s, "BLACKLIST_EXCLUDED", "Surveillance live blacklist")
        elif not force and (can_s in cooldown_syms or can_s in failed_cooldown_syms):
            terminal_tracker.record_terminal(s, "COOLDOWN_ACTIVE", "Active cooldown from recent alert or failed reversal")

    pre_filtered_count = len(watchlist) - len(scan_watchlist)
    rejected["blacklist_or_cooldown_pre_filtered"] = pre_filtered_count
    
    logger.info(f"📊 Pre-filtered {pre_filtered_count} symbols by blacklist/cooldown policy. Remaining watchlist for scan: {len(scan_watchlist)}")

    if scan_watchlist.empty:
        logger.info("REVERSAL scan watchlist is empty after pre-filtering policy exclusions. Skipping execution.")
        upsert_scanner_health("REVERSAL", "IDLE", outcome="SUCCESS", today_alerts=0, error_msg="Watchlist empty after policy exclusions.")
        return 0

    import gc
    BATCH_SIZE = 200

    # Intraday 5m snapshot optimization: only fetch during market hours Mon-Fri (excluding holidays)
    is_market_open = (
        ist_now.weekday() < 5
        and ist_now.date() not in EXCHANGE_HOLIDAYS
        and dtime(9, 15) <= ist_now.time() <= dtime(15, 30)
    )
    # Calculate market session fraction for intraday proration
    session_fraction = _session_fraction(ist_now.time())
    
    # Intraday 5m snapshot optimization: only fetch during market hours Mon-Fri (excluding holidays)
    is_market_open = (
        ist_now.weekday() < 5
        and ist_now.date() not in EXCHANGE_HOLIDAYS
        and dtime(9, 15) <= ist_now.time() <= dtime(15, 30)
    )
    all_symbols = scan_watchlist["Stock"].tolist()
    # [STANDARD 6: SHARED LIVE PRICE WARMUP] Bulk pre-fetch live quotes for all symbols in 1 batched call
    try:
        from live_prices import bulk_warmup_live_prices
        bulk_warmup_live_prices(all_symbols)
    except Exception as _lp_err:
        logger.warning(f"⚠️ Live prices bulk warmup warning: {_lp_err}")

    requested_symbols = set([_canonical_symbol(s) for s in all_symbols if s])
    snapshot_by_symbol = {}
    snapshot_fetch_failed = False
    if is_market_open:
        try:
            from price_cache import get_intraday_snapshot
            raw_snapshots = get_intraday_snapshot(all_symbols, interval="5m", period="1d", requester="ReverseScanner") or {}
            for provider_symbol, snapshot in raw_snapshots.items():
                canonical = _canonical_symbol(provider_symbol)
                if canonical in snapshot_by_symbol:
                    logger.warning(f"Duplicate snapshot symbol: {canonical}")
                    continue
                snapshot_by_symbol[canonical] = snapshot
        except Exception as _snap_e:
            logger.warning(f"Intraday snapshot fetch failed: {_snap_e}")
            snapshot_fetch_failed = True
            snapshot_by_symbol = {}

    synthetic_vol_missing = set()
    synthetic_bar_symbols = set()
    valid_fetched_symbols = set()
    valid_snapshot_symbols = set()
    
    # Track distinct populations for stale and fundamental ratios
    timestamp_checked = 0
    invalid_timestamp_count = 0
    date_checkable = 0
    stale_count = 0

    fundamental_checked = 0
    fundamental_missing = 0
    fundamental_invalid = 0
    fundamental_valid = 0

    stage_tracker.end_stage(f"Pledge: {len(pledge_map)}, Delivery: {len(prev_delivery_map)}")
    stage_tracker.start_stage(3, "Price Fetch & Reversal Pattern Evaluation", f"Batch size: {BATCH_SIZE}")

    _last_hb = time.monotonic()

    with MemoryProfiler("Process Symbols"):

        for batch_num, chunk_df in enumerate(chunk_iterable(scan_watchlist, BATCH_SIZE), start=1):
            try:
                _batch_start_t = time.perf_counter()
                # [VERSION: MARKET_DATA_SESSION_v1.0] Serve from session when available;
                # fall back to independent per-batch fetch otherwise.
                # Session keys match watchlist "Stock" column exactly.
                if session is not None:
                    all_ticker_data = {
                        row["Stock"]: (
                            session.get(row["Stock"]).ohlcv_df
                            if session.get(row["Stock"]) is not None else None
                        )
                        for _, row in chunk_df.iterrows()
                    }
                else:
                    # [VERSION: UNIFIED_1Y_CACHE_v2.0]
                    # Aligned on unified 1-year cache (period="1y", interval="1d").
                    # Reversal scanner requires SMA200 & 52W High (252 trading days = ~365 cal days max).
                    # Using period="1y" shares the exact same Parquet cache files with Wealth Engine & EOD scanner,
                    # eliminating 50% data payload and preventing cache key fragmentation.
                    all_ticker_data = fetch_watchlist_data(chunk_df, interval="1d", period="1y", requester="REVERSAL")
                    
                _fetch_dur = time.perf_counter() - _batch_start_t

                if run_ctx and (time.monotonic() - _last_hb) >= 10.0:
                    try:
                        run_ctx.heartbeat()
                        _last_hb = time.monotonic()
                    except Exception: pass

            except Exception as fetch_err:
                logger.error(f"❌ [REVERSAL] Batch {batch_num} fetch error: {fetch_err}")
                rejected["batch_fetch_failed"] += len(chunk_df)
                continue

            if not all_ticker_data:
                continue

            try:
                # Create a canonicalized dictionary of provider results to avoid suffix differences
                ticker_data_by_symbol = {}
                for provider_symbol, frame in all_ticker_data.items():
                    canonical = _canonical_symbol(provider_symbol)
                    if canonical in ticker_data_by_symbol:
                        logger.warning(f"⚠️ [REVERSAL] Duplicate canonical provider key: {canonical} (from {provider_symbol})")
                    else:
                        ticker_data_by_symbol[canonical] = frame
                
                today_date_str = ist_now.strftime("%Y-%m-%d")
                for can_sym, hist_df in list(ticker_data_by_symbol.items()):
                    if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
                        # Fetch ratio tracking: only count requested symbols
                        if can_sym in requested_symbols:
                            valid_fetched_symbols.add(can_sym)

                        snap_df = snapshot_by_symbol.get(can_sym)
                        
                        # =====================================================================================
                        # [VERSION: REVERSAL_SNAPSHOT_FALLBACK_v1.1] (2026-08-28 RULE 67 RATIONALE)
                        # Failure RCA (Run ae83e79d): Previously during market hours, if snap_df (5m intraday snapshot)
                        # returned empty/unpopulated, the code executed a hard rejection (`continue`), dropping all 301 stocks
                        # with `MISSING_SNAPSHOT` and producing a false `0 / 0 / 301` "PARTIAL (301 data incomplete)" run health status.
                        # Fix: Do NOT hard-reject symbols when 5m intraday snapshot is missing. Fall back gracefully to evaluating
                        # the symbol against the 1D daily historical candles (which are always 100% available in disk/DB cache).
                        # =====================================================================================
                        if is_market_open and (snap_df is None or snap_df.empty):
                            logger.debug(f"⚠️ [REVERSAL] Live snapshot missing for {can_sym}; falling back to 1D daily close.")

                        if isinstance(snap_df, pd.DataFrame) and not snap_df.empty:
                            try:
                                numeric = snap_df[["Open", "High", "Low", "Close", "Volume"]].apply(
                                    pd.to_numeric,
                                    errors="coerce",
                                )
                                if numeric[["Open", "High", "Low", "Close"]].dropna().empty:
                                    rejected["invalid_snapshot"] += 1
                                    telemetry_logger.record_reject(can_sym, "DATA", "INVALID_SNAPSHOT", None, None, start_time=time.time())
                                    ticker_data_by_symbol.pop(can_sym, None)
                                    valid_fetched_symbols.discard(can_sym)
                                    continue
                                
                                live_price = float(numeric["Close"].dropna().iloc[-1])
                                snap_open = float(numeric["Open"].dropna().iloc[0])
                                snap_high = float(numeric["High"].max())
                                snap_low = float(numeric["Low"].min())
                                snap_vol = float(numeric["Volume"].fillna(0).sum())

                                # Logic validation checks
                                if not (snap_low <= live_price <= snap_high) or not (snap_low <= snap_open <= snap_high):
                                    rejected["invalid_snapshot_bounds"] += 1
                                    telemetry_logger.record_reject(can_sym, "DATA", "INVALID_SNAPSHOT_BOUNDS", None, None, start_time=time.time())
                                    ticker_data_by_symbol.pop(can_sym, None)
                                    valid_fetched_symbols.discard(can_sym)
                                    continue
                                
                                valid_snapshot_symbols.add(can_sym)
                            except Exception as parse_e:
                                logger.warning(f"Failed to parse snapshot row for {can_sym}: {parse_e}")
                                rejected["invalid_snapshot"] += 1
                                telemetry_logger.record_reject(can_sym, "DATA", "INVALID_SNAPSHOT", None, None, start_time=time.time())
                                ticker_data_by_symbol.pop(can_sym, None)
                                valid_fetched_symbols.discard(can_sym)
                                continue

                            last_dt = hist_df.index[-1] if not hist_df.index.empty else None
                            t_col = 'Date' if 'Date' in hist_df.columns else ('Datetime' if 'Datetime' in hist_df.columns else None)
                            if t_col:
                                last_dt = hist_df[t_col].iloc[-1]
                            last_dt_str = pd.to_datetime(last_dt).strftime("%Y-%m-%d") if last_dt else ""

                            if last_dt_str == today_date_str:
                                hist_df = hist_df.copy()
                                idx = hist_df.index[-1]
                                hist_df.at[idx, 'Close'] = live_price
                                if snap_vol > 0: hist_df.at[idx, 'Volume'] = snap_vol
                                hist_df.at[idx, 'High'] = max(float(hist_df['High'].iloc[-1]), snap_high)
                                hist_df.at[idx, 'Low'] = min(float(hist_df['Low'].iloc[-1]), snap_low)
                                try:
                                    recomputed = apply_indicators(hist_df, timeframe="1d")
                                    if recomputed is None or recomputed.empty:
                                        ticker_data_by_symbol.pop(can_sym, None)
                                        valid_fetched_symbols.discard(can_sym)
                                        rejected["indicator_failure"] += 1
                                        telemetry_logger.record_reject(can_sym, "DATA", "INDICATOR_FAIL", None, None, start_time=time.time())
                                        continue
                                    hist_df = recomputed
                                except Exception as exc:
                                    logger.warning(f"Indicator calculation failed for {can_sym}: {exc}")
                                    ticker_data_by_symbol.pop(can_sym, None)
                                    valid_fetched_symbols.discard(can_sym)
                                    rejected["indicator_failure"] += 1
                                    telemetry_logger.record_reject(can_sym, "DATA", "INDICATOR_FAIL", None, None, start_time=time.time())
                                    continue
                                synthetic_bar_symbols.discard(can_sym)
                                synthetic_vol_missing.discard(can_sym)
                            else:
                                hist_df = hist_df.copy()
                                new_row = hist_df.iloc[-1:].copy()
                                new_dt = pd.to_datetime(today_date_str)
                                if t_col: new_row[t_col] = new_dt
                                else: new_row.index = [new_dt]
                                new_row['Open'] = snap_open
                                new_row['High'] = snap_high
                                new_row['Low'] = snap_low
                                new_row['Close'] = live_price
                                new_row['Volume'] = snap_vol
                                hist_df = pd.concat([hist_df, new_row])
                                try:
                                    recomputed = apply_indicators(hist_df, timeframe="1d")
                                    if recomputed is None or recomputed.empty:
                                        ticker_data_by_symbol.pop(can_sym, None)
                                        valid_fetched_symbols.discard(can_sym)
                                        rejected["indicator_failure"] += 1
                                        telemetry_logger.record_reject(can_sym, "DATA", "INDICATOR_FAIL", None, None, start_time=time.time())
                                        continue
                                    hist_df = recomputed
                                except Exception as exc:
                                    logger.warning(f"Indicator calculation failed for {can_sym}: {exc}")
                                    ticker_data_by_symbol.pop(can_sym, None)
                                    valid_fetched_symbols.discard(can_sym)
                                    rejected["indicator_failure"] += 1
                                    telemetry_logger.record_reject(can_sym, "DATA", "INDICATOR_FAIL", None, None, start_time=time.time())
                                    continue
                                synthetic_bar_symbols.add(can_sym)
                                if snap_vol <= 0: synthetic_vol_missing.add(can_sym)
                                else: synthetic_vol_missing.discard(can_sym)

                            ticker_data_by_symbol[can_sym] = hist_df
                            
                import threading
                from concurrent.futures import ThreadPoolExecutor, as_completed
                _batch_lock = threading.Lock()
                def _process_row(idx, row):
                    nonlocal timestamp_checked, invalid_timestamp_count, date_checkable, stale_count, fundamental_checked, fundamental_missing, fundamental_invalid, fundamental_valid
                    _ = None
                    symbol = row["Stock"]
                    _row_start_time = _time.perf_counter()
                    category = row["Category"]
                    can_sym  = _canonical_symbol(symbol)
                    
                    try:
                        ticker_data = ticker_data_by_symbol.get(can_sym)
                        if ticker_data is None:
                            with _batch_lock:
                                rejected["no_data"] += 1
                                terminal_tracker.record_terminal(symbol, "DATA_UNAVAILABLE", "No price data returned by provider")
                                telemetry_logger.record_reject(symbol, "DATA", "NO_DATA", None, None, start_time=_row_start_time)
                            logger.info(f"🚫 [REVERSAL] {symbol} REJECTED — No price data returned by provider")
                            return

                        with _batch_lock:
                            timestamp_checked += 1

                        # Check and reject stale data individually using Date/Datetime supporting helper
                        latest_bar_dt = _latest_bar_timestamp(ticker_data)
                        if latest_bar_dt is None:
                            invalid_timestamp_count += 1
                            with _batch_lock:
                                rejected["invalid_timestamp"] += 1
                                terminal_tracker.record_terminal(symbol, "INVALID_TIMESTAMP", "invalid/missing timestamp")
                                telemetry_logger.record_reject(symbol, "DATA", "INVALID_TIMESTAMP", None, None, start_time=_row_start_time)
                            logger.info(f"🚫 [REVERSAL] {symbol} REJECTED — invalid/missing timestamp")
                            return

                        date_checkable += 1

                        from market_utils import evaluate_data_staleness
                        staleness = evaluate_data_staleness(latest_bar_dt, ist_now)
                        if staleness["is_stale"]:
                            stale_count += 1
                            with _batch_lock:
                                rejected["stale_data"] += 1
                                terminal_tracker.record_terminal(symbol, "STALE_DATA", f"Data stale till {staleness['latest_available']}")
                                telemetry_logger.record_reject(symbol, "DATA", "STALE_DATA", None, None, start_time=_row_start_time)
                            logger.info(f"🚫 [REVERSAL] {symbol} REJECTED — Data stale. Available till {staleness['latest_available']} (Expected at least {staleness['expected_date']})")
                            return

                        # Check fundamental presence
                        fundamental_checked += 1
                        fund_dict = row.to_dict() if hasattr(row, "to_dict") else row
                        roe_val = _parse_robust_pct(fund_dict, "roe", "ROE %") if fund_dict else None
                        rev_growth = _parse_robust_pct(fund_dict, "yoy_revenue", "YOY Revenue %") if fund_dict else None
                        if rev_growth is None and fund_dict:
                            rev_growth = _parse_robust_pct(fund_dict, "revenue_cagr_3y", "Revenue CAGR 3Y")
                        if rev_growth is None and fund_dict:
                            rev_growth = _parse_robust_pct(fund_dict, "yoy_profit", "YOY Profit %")
                        if rev_growth is None and fund_dict and (roe_val is not None or fund_dict.get("score") is not None):
                            rev_growth = 0.0  # Fallback for lightweight production cache entries
                        
                        if roe_val is None or rev_growth is None:
                            fundamental_missing += 1
                        elif not (-100.0 <= roe_val <= 500.0) or not (-100.0 <= rev_growth <= 1000.0):
                            fundamental_invalid += 1
                        else:
                            fundamental_valid += 1

                        ticker = ticker_data.copy()
                        verdict = _evaluate_candidate(
                            symbol=symbol,
                            df=ticker,
                            fund_data=row.to_dict(),
                            regime_ctx=regime_ctx,
                            weights=bayesian_weights,
                            pledge_map=pledge_map,
                            delivery_map=prev_delivery_map,
                            is_synthetic_bar=(can_sym in synthetic_bar_symbols),
                            is_synthetic_no_vol=(can_sym in synthetic_vol_missing),
                            resolved_date=resolved_date,
                            is_intraday=is_market_open,
                            session_fraction=session_fraction,
                        )

                        if not verdict["passed"]:
                            rej_code = verdict.get("reject_code", "failed_pattern")
                            rej_msg = verdict.get("reject_reason", rej_code)
                            logger.info(f"🚫 [REVERSAL] {symbol} REJECTED — Gate: {rej_code.upper()} | Reason: {rej_msg}")
                            with _batch_lock:
                                rejected[rej_code] += 1
                                terminal_tracker.record_terminal(symbol, rej_code.upper(), rej_msg)
                            try:
                                ctx_rev = telemetry_logger.get_or_create_context(symbol)
                                latest_rec = ticker.iloc[-1]
                                ctx_rev.capture_dataframe_row(latest_rec, is_fallback=False)
                                ctx_rev.capture_raw_market(
                                    open_p=_safe_float(latest_rec.get("Open")),
                                    high_p=_safe_float(latest_rec.get("High")),
                                    low_p=_safe_float(latest_rec.get("Low")),
                                    close_p=_safe_float(latest_rec.get("Close")),
                                    volume=_safe_float(latest_rec.get("Volume")),
                                    high_52w=_safe_float(latest_rec.get("HIGH_52W")),
                                    low_52w=_safe_float(latest_rec.get("LOW_52W"))
                                )
                                ctx_rev.capture_indicators(
                                    rsi=_safe_float(latest_rec.get("RSI")),
                                    ema20=_safe_float(latest_rec.get("EMA20")),
                                    sma50=_safe_float(latest_rec.get("SMA50")),
                                    sma200=_safe_float(latest_rec.get("SMA200")),
                                    macd=_safe_float(latest_rec.get("MACD")),
                                    vol_ratio=_safe_float(latest_rec.get("Volume_Ratio"))
                                )
                                ctx_rev.capture_gate(
                                    gate_name=rej_code.upper(),
                                    passed=False,
                                    actual_val=verdict.get("actual", verdict.get("score", 0.0)),
                                    operator_str=verdict.get("operator", "<="),
                                    threshold_val=verdict.get("threshold", 0.0),
                                    gate_type=verdict.get("gate_type", "THRESHOLD"),
                                    reason=verdict.get("reject_reason", ""),
                                    components=verdict.get("components", {}),
                                    expression=verdict.get("expression", f"gate_{rej_code} == PASS")
                                )
                                ctx_rev.finalize(decision="REJECTED", primary_reason=verdict.get("reject_reason", rej_code))
                                telemetry_engine.emit_terminal(ctx_rev)
                            except Exception as _tr_err:
                                telemetry_logger.record_reject(
                                    symbol=symbol,
                                    last_stage="REVERSAL_GATE",
                                    gate=rej_code.upper(),
                                    actual=verdict.get("actual", verdict.get("score")),
                                    required=verdict.get("threshold"),
                                    start_time=_row_start_time
                                )
                            try:
                                from near_miss_tracker import log_near_miss
                                ev_score = verdict.get("score") or verdict.get("raw_score") or 0
                                if ev_score > 0:
                                    sl_res = verdict.get("sl_result", {})
                                    close_px = float(ticker["Close"].iloc[-1]) if not ticker.empty else 0.0
                                    log_near_miss(
                                        symbol=symbol,
                                        scanner="REVERSAL",
                                        breakout_type="REVERSAL_BREAKOUT",
                                        gate_name=verdict.get("reject_code", "reversal_gate"),
                                        observed_value=float(ev_score),
                                        threshold_value=55.0,
                                        score=int(ev_score),
                                        entry_price=close_px,
                                        stop_loss=float(sl_res.get("stop_loss", 0.0)) if sl_res and sl_res.get("stop_loss") else None,
                                        target_1=float(sl_res.get("target_1", 0.0)) if sl_res and sl_res.get("target_1") else None,
                                    )
                            except Exception as _nm_e:
                                logger.debug(f"Near miss log error: {_nm_e}")
                            return

                        ctx = verdict["context"]
                        target_val = verdict["sl_result"].get("target_1") or verdict["sl_result"].get("target")
                        logger.info(
                            f"🌟 [REVERSAL: SELECTED] {symbol} @ ₹{ctx['entry_price']:.2f} | "
                            f"Score: {verdict['score']} | Signals: {', '.join(ctx['signals'])} | "
                            f"RSI: {round(ctx['technicals']['rsi'], 1)} | "
                            f"VolRatio: {ctx['technicals']['volume_ratio']:.2f}x | "
                            f"SL: ₹{verdict['sl_result'].get('stop_loss', 0)} | "
                            f"T1: ₹{target_val}"
                        )
                        with _batch_lock:
                            shortlisted_alerts.append({
                            "symbol": symbol,
                            "alert_time": ist_now.strftime("%Y-%m-%d %H:%M:%S+05:30"),
                            "category": category,
                            "entry_price": round(ctx["entry_price"], 2),
                            "signals": ", ".join(ctx["signals"]),
                            "score": verdict["score"],
                            "raw_score": verdict.get("raw_score", verdict["score"]),
                            "rsi": round(ctx["technicals"]["rsi"], 1),
                            "volume_ratio": ctx["technicals"]["volume_ratio"],
                            "stop_loss": verdict["sl_result"].get("stop_loss"),
                            "target_1": target_val,
                            "target_price": target_val,
                            "context": ctx,
                            "structural_failure_stop": verdict["sl_result"].get("structural_failure_stop"),
                            "target_quality_score": verdict["sl_result"].get("target_quality")
                        })
                    except Exception as sym_err:
                        logger.error(f"❌ [REVERSAL] Error processing symbol {symbol} in batch {batch_num}: {sym_err}")
                        with _batch_lock:
                            rejected["processing_error"] += 1
                            terminal_tracker.record_terminal(symbol, "PROCESSING_ERROR", str(sym_err)[:100])
            except Exception as batch_err:
                logger.error(f"❌ [REVERSAL] Batch {batch_num} execution error: {batch_err}")
                with _batch_lock:
                    rejected["batch_fetch_failed"] += len(chunk_df)
            finally:
                total_batches = (len(scan_watchlist) + BATCH_SIZE - 1) // BATCH_SIZE


                _eval_start_t = time.perf_counter()
                with ThreadPoolExecutor(max_workers=10, thread_name_prefix="Reversal_Worker") as executor:
                    futures = [executor.submit(_process_row, idx, row) for idx, (_, row) in enumerate(chunk_df.iterrows(), start=1)]
                    for f in as_completed(futures):
                        f.result()
                _eval_dur = time.perf_counter() - _eval_start_t
                        
                logger.info(
                    f"⏱️ [REVERSAL SCANNER] Batch {batch_num}/{total_batches} Timing | "
                    f"Fetch {len(chunk_df)} symbols: {_fetch_dur:.2f}s | "
                    f"Evaluation: {_eval_dur:.2f}s | "
                    f"Shortlisted so far: {len(shortlisted_alerts)}"
                )
                gc.collect()

        total_symbols = len(watchlist)
        total_fetched_count = len(valid_fetched_symbols)
        total_requested = len(scan_watchlist)
        fetch_ratio = total_fetched_count / total_requested if total_requested > 0 else 1.0
        
        stale_ratio = stale_count / max(date_checkable, 1)
        invalid_timestamp_ratio = invalid_timestamp_count / max(timestamp_checked, 1)
        fundamental_failure_ratio = (fundamental_missing + fundamental_invalid) / max(fundamental_checked, 1)

        if run_ctx:
            run_ctx.set_total_stocks(total_symbols)
            run_ctx.fresh_count = total_fetched_count
            run_ctx.stale_count = stale_count
            run_ctx.incomplete_count = max(0, total_symbols - total_fetched_count - stale_count)

        logger.info(f"📊 [REVERSAL] Batch Fetch Completed: {total_fetched_count}/{total_requested} requested symbols fetched ({fetch_ratio*100:.1f}%)")
        logger.info(f"📊 [REVERSAL] Stale Ratio: {stale_count}/{date_checkable} checkable symbols stale ({stale_ratio*100:.1f}%)")
        logger.info(f"📊 [REVERSAL] Invalid Timestamp Ratio: {invalid_timestamp_count}/{timestamp_checked} ({invalid_timestamp_ratio*100:.1f}%)")
        logger.info(f"📊 [REVERSAL] Fundamental Outage/Failure Ratio: {fundamental_missing + fundamental_invalid}/{fundamental_checked} checkable symbols missing/invalid fundamentals ({fundamental_failure_ratio*100:.1f}%)")

        if is_market_open and snapshot_fetch_failed:
            logger.warning("⚠️ [REVERSAL] Intraday snapshot fetch failed during market hours. Blocking persistence to prevent stale alerts.")
            upsert_scanner_health(
                "REVERSAL", 
                "DEGRADED", 
                error_msg="Intraday snapshot fetch failed during market hours",
                processed_count=len(shortlisted_alerts),
                total_count=total_symbols,
                outcome="PARTIAL"
            )
            return 0

        if fetch_ratio < MIN_FETCH_RATIO:
            logger.warning(f"⚠️ [REVERSAL] Low fetch coverage ({fetch_ratio*100:.1f}% < {MIN_FETCH_RATIO*100:.0f}%). Blocking persistence and preserving existing alerts.")
            upsert_scanner_health(
                "REVERSAL", 
                "DEGRADED", 
                error_msg=f"Low fetch coverage: {total_fetched_count}/{total_requested} symbols fetched ({fetch_ratio*100:.1f}%)",
                processed_count=len(shortlisted_alerts),
                total_count=total_symbols,
                outcome="PARTIAL"
            )
            return 0

        if stale_ratio > STALE_DEGRADED_RATIO:
            logger.warning(f"⚠️ [REVERSAL] High stale data ratio ({stale_ratio*100:.1f}% > {STALE_DEGRADED_RATIO*100:.0f}%). Blocking persistence and preserving existing alerts.")
            upsert_scanner_health(
                "REVERSAL", 
                "DEGRADED", 
                error_msg=f"High stale data ratio: {stale_count}/{date_checkable} fetched symbols stale ({stale_ratio*100:.1f}%)",
                processed_count=len(shortlisted_alerts),
                total_count=total_symbols,
                outcome="PARTIAL"
            )
            return 0

        if invalid_timestamp_ratio > 0.15:
            logger.warning(f"⚠️ [REVERSAL] High invalid timestamp ratio ({invalid_timestamp_ratio*100:.1f}% > 15%). Blocking persistence and preserving existing alerts.")
            upsert_scanner_health(
                "REVERSAL", 
                "DEGRADED", 
                error_msg=f"High invalid timestamp ratio: {invalid_timestamp_count}/{timestamp_checked} symbols ({invalid_timestamp_ratio*100:.1f}%)",
                processed_count=len(shortlisted_alerts),
                total_count=total_symbols,
                outcome="PARTIAL"
            )
            return 0

        # [VERSION: REVERSAL_FUNDAMENTAL_ALARM_v1.0] Escalate visibility on high fundamental failure ratio without blocking persistence.
        if fundamental_failure_ratio > FUNDAMENTAL_REJECT_ALARM_PCT:
            logger.warning(f"⚠️ [REVERSAL] High fundamental outage/failure ratio ({fundamental_failure_ratio*100:.1f}% > {FUNDAMENTAL_REJECT_ALARM_PCT*100:.0f}%). Marking health DEGRADED and sending notifications.")
            upsert_scanner_health(
                "REVERSAL", 
                "DEGRADED", 
                error_msg=f"Potential fundamental data outage: {fundamental_missing} missing, {fundamental_invalid} invalid out of {fundamental_checked} checked symbols ({fundamental_failure_ratio*100:.1f}%)",
                processed_count=len(shortlisted_alerts),
                total_count=total_symbols,
                outcome="PARTIAL"
            )
            try:
                from database import insert_notification
                insert_notification(
                    "warning",
                    "⚠️ REVERSAL SCANNER: Fundamental Data Warning",
                    f"High fundamental outage/failure ratio ({fundamental_failure_ratio*100:.1f}% > {FUNDAMENTAL_REJECT_ALARM_PCT*100:.0f}%). {fundamental_missing} missing, {fundamental_invalid} invalid out of {fundamental_checked} checked."
                )
                from push_service import send_push_to_all
                send_push_to_all(
                    title="⚠️ REVERSAL SCANNER DEGRADED",
                    body=f"High fundamental outage ({fundamental_failure_ratio*100:.1f}%). Alerts will still process.",
                    bypass_throttle=True
                )
            except Exception as _notif_err:
                logger.warning(f"⚠️ Failed to dispatch Reversal fundamental alarm notifications: {_notif_err}")


        if is_market_open:
            snapshot_ratio = len(valid_snapshot_symbols) / max(len(scan_watchlist), 1)
            logger.info(f"📊 [REVERSAL] Snapshot Coverage: {len(valid_snapshot_symbols)}/{len(scan_watchlist)} ({snapshot_ratio*100:.1f}%)")
            if snapshot_ratio < 0.85:
                logger.warning(f"⚠️ [REVERSAL] Low snapshot coverage ({snapshot_ratio*100:.1f}% < 85%). Blocking persistence and preserving existing alerts.")
                upsert_scanner_health(
                    "REVERSAL", 
                    "DEGRADED", 
                    error_msg=f"Low snapshot coverage: {len(valid_snapshot_symbols)}/{len(scan_watchlist)} snapshots ({snapshot_ratio*100:.1f}%)",
                    processed_count=len(shortlisted_alerts),
                    total_count=total_symbols,
                    outcome="PARTIAL"
                )
                return 0

        # Database transaction commit block
        db_success = False
        try:
            from database import _DB_WRITE_LOCK
            with _DB_WRITE_LOCK:
                with get_connection() as conn:
                    try:
                        # Cleanup today's existing alerts transactionally
                        deleted_count = delete_todays_alerts_for_scanner("REVERSAL", today_str, conn=conn)
                        logger.info(f"REVERSAL cleanup: removed {deleted_count} existing alerts for {today_str} transactionally before persistence")
                        
                        if shortlisted_alerts:
                            # Sort primarily by clamped normalized score, then by risk-reward ratio
                            shortlisted_alerts.sort(key=lambda x: (x["score"], x.get("context", {}).get("risk_reward") or 0.0), reverse=True)
                            from config import SCANNER_MAX_ALERTS
                            max_alerts = SCANNER_MAX_ALERTS.get("REVERSAL", 10)
                            if len(shortlisted_alerts) > max_alerts:
                                for s_cand in shortlisted_alerts[max_alerts:]:
                                    logger.info(f"🚫 [REVERSAL] {s_cand['symbol']} alert SUPPRESSED: Exceeded MAX_ALERTS_PER_SCAN limit ({max_alerts}) (Score: {s_cand['score']:.1f})")
                                    terminal_tracker.record_terminal(s_cand["symbol"], "SUPPRESSED_TOP_N", f"Exceeded MAX_ALERTS limit ({max_alerts})")
                                shortlisted_alerts = shortlisted_alerts[:max_alerts]

                        from market_utils import get_expected_latest_trading_date
                        source_trading_date = get_expected_latest_trading_date(ist_now)

                        for alert in shortlisted_alerts:
                            # Ensure post-market entry price matches today's live CMP
                            ep_val = alert["entry_price"]
                            try:
                                from price_cache import get_cached_price
                                fast_p = get_cached_price(alert["symbol"])
                                if fast_p and float(fast_p) > 0:
                                    ep_val = round(float(fast_p), 2)
                            except Exception:
                                pass

                            inserted, _, _, _ = save_alert_if_new(
                                alert["symbol"], "REVERSAL", alert["alert_time"], scanner="REVERSAL",
                                category=alert["category"], entry_price=ep_val,
                                signals=alert["signals"], score=alert["score"], rsi=alert["rsi"],
                                volume_ratio=alert["volume_ratio"], stop_loss=alert["stop_loss"],
                                target_1=alert.get("target_1"), target_price=alert["target_price"],
                                context=alert["context"], model_version=ACTIVE_ALGO_VERSION,
                                bayesian_regime=regime_ctx.get("trend", "NEUTRAL"), bayesian_weights=bayesian_weights,
                                structural_failure_stop=alert.get("structural_failure_stop"),
                                target_quality_score=alert.get("target_quality_score"),
                                conn=conn,
                                source_trading_date=source_trading_date
                            )
                            if inserted:
                                total_alerts += 1
                                terminal_tracker.record_terminal(alert["symbol"], "ALERT_GENERATED", f"Reversal alert triggered with score {alert['score']}")
                            else:
                                terminal_tracker.record_terminal(alert["symbol"], "DUPLICATE_ALERT", "Alert already active or deduplicated today")
                        
                        conn.commit()
                        db_success = True
                    except Exception:
                        conn.rollback()
                        raise
        except Exception as db_err:
            logger.exception(f"❌ [REVERSAL] Transactional db operations failed: {db_err}")
            upsert_scanner_health(
                "REVERSAL", 
                "DOWN", 
                error_msg=f"Database transaction error: {str(db_err)[:200]}"
            )
            return 0

        if db_success:
            # Update scanner health to OK on successful completion
            upsert_scanner_health(
                "REVERSAL", 
                "OK", 
                error_msg=None, 
                today_alerts=total_alerts, 
                processed_count=len(shortlisted_alerts), 
                total_count=total_symbols, 
                outcome="SUCCESS",
                provider_stats={
                    "SUCCESS": total_fetched_count if 'total_fetched_count' in locals() else 0,
                    "NOT_FOUND": max(0, total_symbols - total_fetched_count - stale_count) if ('total_fetched_count' in locals() and 'stale_count' in locals()) else 0,
                    "STALE": stale_count if 'stale_count' in locals() else 0,
                    "RATE_LIMIT": 0,
                    "NETWORK_ERROR": 0,
                    "TIMEOUT": 0,
                    "EMPTY_DATA": 0
                }
            )

            # Stage waterfall progression
            waterfall.set_stage_count("VALID_DATA", max(0, total_fetched_count - stale_count))
            waterfall.set_stage_count("TECHNICAL_SETUP", len(shortlisted_alerts))
            waterfall.set_stage_count("FINAL_ALERTS", total_alerts)

            # Single terminal conservation resolution
            terminal_tracker.record_untracked_remainder("UNTRACKED_DROP")
            cons_summary = terminal_tracker.get_summary()
            dom_bottleneck = waterfall.get_dominant_bottleneck()
            stage_waterfall = waterfall.compute_attrition()

            exec_mode = "LIVE_ENTRY" if is_market_open else "EOD_SCAN"
            classification_res = classify_zero_alert_run(
                scanner_name="REVERSAL",
                universe_size=total_symbols,
                valid_data_count=max(0, total_fetched_count - stale_count),
                initial_setups_count=len(shortlisted_alerts),
                finalist_candidates_count=len(shortlisted_alerts),
                alerts_generated=total_alerts,
                near_miss_count=0,
                regime=regime_str,
                execution_mode=exec_mode,
                stage_waterfall=stage_waterfall
            )

            fired_rev = {k: v for k, v in rejected.items() if v > 0}
            summary_rev_lines = [
                "======================================================================",
                "=== [REVERSAL SCANNER PIPELINE SUMMARY] ===",
                "======================================================================",
                "📊 DATA QUALITY & PIPELINE SNAPSHOT:",
                f"  • Total Watchlist Requested : {total_symbols}",
                f"  • Provider Resolved Symbols : {total_fetched_count}",
                f"  • Shortlisted Candidates    : {len(shortlisted_alerts)}",
                f"  • Alerts Generated          : {total_alerts}",
                f"  • Execution Mode            : {exec_mode}",
                "",
                "🎯 MUTUALLY-EXCLUSIVE TERMINAL DISPOSITIONS (CONSERVED):"
            ]
            for term_gate, term_cnt in cons_summary["terminal_counts"].items():
                summary_rev_lines.append(f"  • {term_gate:<35}: {term_cnt}")
            summary_rev_lines.append(f"  • Conservation Accounting   : {cons_summary['sum_terminal']}/{cons_summary['total_universe']} symbols accounted for (Delta: {cons_summary['conservation_delta']})")

            summary_rev_lines.extend([
                "",
                "🌊 STAGE WATERFALL ATTRITION (RATE-BASED):"
            ])
            for stg in stage_waterfall:
                summary_rev_lines.append(
                    f"  • {stg['stage']:<20} → {stg['next_stage']:<20}: {stg['entered']:>4} entered, {stg['passed']:>4} passed (loss: {stg['eliminated']} [{stg['attrition_pct']:.1f}%])"
                )

            if dom_bottleneck:
                summary_rev_lines.append(
                    f"  • Dominant Bottleneck Gate  : {dom_bottleneck['stage']} (Highest Attrition: {dom_bottleneck['attrition_pct']:.1f}% [{dom_bottleneck['eliminated']}/{dom_bottleneck['entered']} eliminated])"
                )

            if total_alerts == 0:
                b_stg = dom_bottleneck.get('stage', '') if dom_bottleneck else ''
                b_breakdown = terminal_tracker.get_stage_terminal_breakdown(b_stg) if b_stg else None

                diag_block = format_zero_alert_diagnostic_block(
                    scanner_name="REVERSAL",
                    execution_mode=exec_mode,
                    regime=regime_str,
                    classification_result=classification_res,
                    dominant_bottleneck=dom_bottleneck,
                    conservation_summary=cons_summary,
                    stage_waterfall=stage_waterfall,
                    near_miss_count=0,
                    bottleneck_terminal_breakdown=b_breakdown
                )
                summary_rev_lines.extend(diag_block)

            summary_rev_lines.append("======================================================================")
            logger.info("\n".join(summary_rev_lines))
            telemetry_logger.print_summary()
            global_telemetry.print_system_summary()
            # [VERSION: PERF_PHASE0_v1.0] Human-readable stage summary for log-based verification
            try:
                _scan_total_ms = (_time_mod.perf_counter() - _scan_start) * 1000
                _scan_total_s  = _scan_total_ms / 1000
                logger.info(
                    "\n"
                    "┌─────────────────────────────────────────────────────────────────┐\n"
                    "│         📊 REVERSAL SCANNER — PERF STAGE SUMMARY (Phase 0)      │\n"
                    "├──────────────────────────────────────┬──────────────────────────┤\n"
                    f"│  Symbols processed                   │  {total_symbols:<24} │\n"
                    f"│  Alerts generated                    │  {total_alerts:<24} │\n"
                    f"│  Total scan time                     │  {_scan_total_s:.1f}s{' '*(22 - len(f'{_scan_total_s:.1f}s'))} │\n"
                    "├──────────────────────────────────────┼──────────────────────────┤\n"
                    f"│  [STAGE] historical_fetch (batched)  │  see batch logs above    │\n"
                    f"│  [STAGE] candidate_eval              │  included in total       │\n"
                    f"│  [STAGE] db_persistence              │  included in total       │\n"
                    "├──────────────────────────────────────┼──────────────────────────┤\n"
                    "│  (check '⏱ [STAGE] reversal' lines above for per-batch times)  │\n"
                    "└─────────────────────────────────────────────────────────────────┘"
                )
            except Exception as _log_e:
                logger.debug(f"Non-critical: Stage summary log failed: {_log_e}")

        try:
            stage_tracker.end_stage(f"Saved {total_alerts} alerts to DB")
            stage_tracker.print_summary(alerts_found=total_alerts)
        except Exception:
            pass



        # [VERSION: PERF_PHASE0_v1.0] Flush Phase 0 JSON timing report to artifacts/profiling/
        try:
            flush_timing_report(
                phase="Phase0_Baseline",
                run_type="cold_start",
                feature_flags=[],
                extra={
                    "scanner": "ReversalScanner",
                    "symbols_processed": total_symbols,
                    "alerts_generated": total_alerts,
                    "total_scan_ms": round(_scan_total_ms, 1),
                }
            )
        except Exception as _perf_e:
            logger.debug(f"Non-critical: Failed to write reversal timing report: {_perf_e}")

        try:
            from database import upload_history_bundle_to_db, submit_background_upload
            submit_background_upload(lambda: upload_history_bundle_to_db("1d"))
            logger.info("💾 [REVERSAL] Submitted background upload of 1d history bundle to Postgres DB.")
        except Exception as _up_err:
            logger.warning(f"⚠️ Failed to queue background DB bundle upload in Reversal: {_up_err}")

        return total_alerts




def _validate_config():
    """
    Startup contradiction validator.
    Fails fast at module import time if any REVERSAL thresholds or parameters are contradictory.
    """
    fatal, warn = [], []
    if MIN_DROP_FROM_52W_HIGH >= MAX_DROP_FROM_52W_HIGH:
        fatal.append(f"Empty drop band: {MIN_DROP_FROM_52W_HIGH} >= {MAX_DROP_FROM_52W_HIGH}")
    if QUALITY_CAT_MIN_DROP > MIN_DROP_FROM_52W_HIGH:
        warn.append(f"QUALITY_CAT_MIN_DROP ({QUALITY_CAT_MIN_DROP}) > MIN_DROP_FROM_52W_HIGH ({MIN_DROP_FROM_52W_HIGH})")
    if RSI_CURL_MIN <= RSI_OVERSOLD_THRESHOLD:
        fatal.append(f"RSI_CURL_MIN ({RSI_CURL_MIN}) must exceed RSI_OVERSOLD_THRESHOLD ({RSI_OVERSOLD_THRESHOLD})")
    
    _MIN_RSI_PTS = 15 if MIN_RSI_RECOVERY >= 20 else (12 if MIN_RSI_RECOVERY >= 12 else (8 if MIN_RSI_RECOVERY >= 8 else (2 if MIN_RSI_RECOVERY >= 3 else 0)))
    for r, req in REGIME_EVIDENCE_REQ.items():
        min_v = req.get("min_vol_ratio", MIN_VOLUME_RATIO)
        v_pts = 15 if min_v >= 3.0 else (12 if min_v >= 2.0 else (8 if min_v >= 1.5 else (5 if min_v >= 1.35 else 0)))
        # True minimum MACD score after the gate is 3 points
        r_min_core = 14 + v_pts + 3 + _MIN_RSI_PTS
        if CORE_SCORE_FLOOR > r_min_core:
            warn.append(f"CORE_SCORE_FLOOR ({CORE_SCORE_FLOOR}) > min feasible core ({r_min_core}) under regime {r} (min_vol_ratio={min_v}) — setups with absolute minimum parameters will be filtered by the core floor")
    if CORE_SCORE_FLOOR >= CORE_SCORE_MAX:
        fatal.append(f"CORE_SCORE_FLOOR ({CORE_SCORE_FLOOR}) >= MAX ({CORE_SCORE_MAX}) is unreachable")

    for r, req in REGIME_EVIDENCE_REQ.items():
        prox = req.get("max_pct_below_sma200")
        if prox is None:
            continue
        if prox >= MAX_DROP_BELOW_SMA200:
            warn.append(f"{r}: proximity {prox}% >= MAX_DROP_BELOW_SMA200 ({MAX_DROP_BELOW_SMA200}%) — regime limit is redundant behind global limit")

    if MIN_VOLUME_RATIO >= CLIMAX_VOL_MULT:
        warn.append(f"MIN_VOLUME_RATIO ({MIN_VOLUME_RATIO}) >= CLIMAX_VOL_MULT ({CLIMAX_VOL_MULT}): every volume-qualified candidate satisfies the climax-volume leg")

    _spread = RSI_CURL_MIN - RSI_OVERSOLD_THRESHOLD
    if MIN_RSI_RECOVERY > _spread + 2.0:
        warn.append(f"MIN_RSI_RECOVERY ({MIN_RSI_RECOVERY}) dominates curl/oversold gates (spread={_spread}); those two gates are decorative")
    if RSI_TROUGH_LOOKBACK < 30:
        warn.append(f"RSI_TROUGH_LOOKBACK={RSI_TROUGH_LOOKBACK} is narrow for rounding bases")
    if MAX_TROUGH_AGE > RSI_TROUGH_LOOKBACK:
        fatal.append(f"MAX_TROUGH_AGE ({MAX_TROUGH_AGE}) cannot exceed RSI_TROUGH_LOOKBACK ({RSI_TROUGH_LOOKBACK})")

    if MIN_YOY_REVENUE_GROWTH > 0 and MIN_DROP_FROM_52W_HIGH >= 20:
        warn.append(f"MIN_YOY_REVENUE_GROWTH={MIN_YOY_REVENUE_GROWTH}% with a {MIN_DROP_FROM_52W_HIGH}%+ drop requirement excludes most genuine mean-reversion candidates")

    if not set(REGIME_EVIDENCE_REQ).issubset(REGIME_REVERSAL_PREMIUM):
        fatal.append("REGIME_EVIDENCE_REQ contains regimes absent from REGIME_REVERSAL_PREMIUM")

    for regime, prem in REGIME_REVERSAL_PREMIUM.items():
        need = MIN_REVERSAL_SCORE + prem
        if need > COMPONENT_MAX:
            fatal.append(f"{regime}: threshold {need} > max {COMPONENT_MAX}")

    # Check if exchange holiday calendar is outdated / expired
    max_holiday_year = max(d.year for d in EXCHANGE_HOLIDAYS)
    if max_holiday_year < datetime.now(IST).year:
        fatal.append(f"Exchange holiday calendar is expired (max year: {max_holiday_year})")

    for w in warn:
        logger.warning(f"REVERSAL config dead zone: {w}")

    if fatal:
        raise ValueError("REVERSAL config contradictions:\n  - " + "\n  - ".join(fatal))


try:
    _validate_config()
except ValueError as e:
    logger.critical("REVERSAL config invalid — scanner will not run: %s", e)
    try:
        from database import upsert_scanner_health
        upsert_scanner_health("REVERSAL", "DOWN", error_msg=f"Config: {str(e)[:200]}")
    except Exception:
        pass
    raise


from lock_utils import ProcessLock
_scan_lock = ProcessLock("reversal_scanner")
_global_lock = ProcessLock("global_scanner_lock")


def start(force: bool = False, session=None, run_ctx=None, trigger_type="SCHEDULED", scheduler_name="CRON", used_fallback_data: bool = False) -> int:
    from database import is_scanner_stopped, upsert_scanner_health
    from lock_utils import print_scanner_start_banner, print_scanner_end_banner
    if is_scanner_stopped("REVERSAL"):
        logger.info("🛑 Reversal Scanner is STOPPED by Admin. Skipping execution.")
        if run_ctx:
            from database import complete_scanner_execution_run
            complete_scanner_execution_run(run_ctx, status_override="STOPPED", stop_reason="Scanner stopped by admin")
        upsert_scanner_health("REVERSAL", "STOPPED", error_msg="REVERSAL scanner is explicitly disabled by admin.")
        return 0

    if _scan_lock.locked():
        logger.warning("🛑 [DUPLICATE GUARD] REVERSAL Scanner is ALREADY actively running in thread lock. Skipping duplicate trigger.")
        if run_ctx:
            from database import complete_scanner_execution_run
            complete_scanner_execution_run(run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Scanner lock busy")
        return {"status": "skipped", "reason": "already_running"}

    acquired_global = False
    acquired_scan = False
    _scan_start = None

    try:
        queued_at = None
        if not _global_lock.acquire(blocking=False, owner_scanner="REVERSAL", operation="FULL_SCAN"):
            queued_at = time.monotonic()
            logger.info("⏳ [REVERSAL] Global scanner lock busy — waiting in queue until lock is released...")
            upsert_scanner_health("REVERSAL", "QUEUED", error_msg="Waiting in queue for active scanner to release lock...")
            
            try:
                acquired_global = _global_lock.acquire(blocking=True, owner_scanner="REVERSAL", operation="FULL_SCAN", run_ctx=run_ctx)
            except Exception as lock_err:
                logger.error(f"❌ [REVERSAL] Error acquiring global lock: {lock_err}")
                acquired_global = False

            if not acquired_global:
                logger.error("❌ [REVERSAL] Failed to acquire global scanner lock after queue wait.")
                if run_ctx:
                    from database import complete_scanner_execution_run
                    complete_scanner_execution_run(run_ctx, status_override="FAILED", stop_reason="Global lock acquire timeout")
                upsert_scanner_health("REVERSAL", "IDLE", error_msg="Lock acquisition timed out")
                return 0
        else:
            acquired_global = True

        if queued_at is not None:
            logger.info(f"✅ [REVERSAL] Global lock acquired after {round(time.monotonic()-queued_at,1)}s wait. Starting scan...")

        if not _scan_lock.acquire(blocking=False):
            logger.warning("🛑 REVERSAL Scanner is ALREADY actively running. Skipping duplicate execution.")
            if run_ctx:
                from database import complete_scanner_execution_run
                complete_scanner_execution_run(run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Scanner already actively running")
            else:
                try:
                    from database import record_skipped_execution_run
                    record_skipped_execution_run(scanner_name="REVERSAL", trigger_type=trigger_type, scheduler_name=scheduler_name, stop_reason="Scanner lock held (previous run active)")
                except Exception:
                    pass
            return 0
        acquired_scan = True

        # [RULE: HISTORY ENTRY AFTER LOCK ACQUIRED] Only create execution history entry once all locks are secured
        if run_ctx is None:
            try:
                from database import start_scanner_execution_run
                run_ctx = start_scanner_execution_run(scanner_name="REVERSAL", trigger_type=trigger_type, scheduler_name=scheduler_name)
            except Exception as exc:
                if "actively running" in str(exc).lower():
                    logger.info("🛑 [REVERSAL] Scanner is ALREADY actively running. Skipping duplicate execution.")
                    return 0
                logger.warning(f"⚠️ [REVERSAL] Could not create run_ctx: {exc}")
        elif run_ctx:
            from database import update_scanner_run_lifecycle
            update_scanner_run_lifecycle(run_ctx.run_id, "RUNNING")

        _scan_start = print_scanner_start_banner("reversal_scanner", queued_at=queued_at, run_id=run_ctx.run_id if run_ctx else None)
        total_alerts = _start_wrapper(force, session=session, run_ctx=run_ctx, used_fallback_data=used_fallback_data)
        if run_ctx:
            run_ctx.add_alert(total_alerts)
            from database import complete_scanner_execution_run
            complete_scanner_execution_run(run_ctx, status_override="COMPLETED")
        return total_alerts
    except Exception as e:
        logger.exception(f"❌ [REVERSAL] Unhandled exception during scan: {e}")
        if run_ctx:
            try:
                from database import complete_scanner_execution_run
                complete_scanner_execution_run(run_ctx, status_override="FAILED", exception=e)
            except Exception: pass
        try:
            upsert_scanner_health("REVERSAL", status="DOWN", error_msg=f"Scan crashed: {str(e)[:300]}", run_id=run_ctx.run_id if run_ctx else None)
            from database import insert_notification
            insert_notification("error", "🚨 REVERSAL Scanner CRASHED", f"Error: {str(e)[:400]}")
        except Exception: pass
        raise e
    finally:
        if _scan_start is not None:
            print_scanner_end_banner("reversal_scanner", _scan_start, run_id=run_ctx.run_id if run_ctx else None)

        if acquired_scan:
            try: _scan_lock.release()
            except Exception: pass
        if acquired_global:
            try: _global_lock.release()
            except Exception: pass

def _start_wrapper(force: bool = False, session=None, run_ctx=None, used_fallback_data: bool = False) -> int:
    """
    Single-shot scan. Called once by main.py at the 21:00 window.
    Returns the number of alerts generated (0 = no setups found).
    Raises on failure so main.py can send a Telegram crash alert.
    """
    init_db()

    try:
        upsert_scanner_health("REVERSAL", "RUNNING", error_msg="Reversal Scan in progress...")
    except Exception:
        logger.warning("⚠️ Could not mark Reversal as RUNNING")

    force_refresh_blacklist()

    try:
        count = _run_scan(force=force, session=session, run_ctx=run_ctx)
        # Execute Phase 2D Reversal V2 Pipeline in parallel isolation
        try:
            _run_reversal_v2_pipeline()
        except Exception as v2_err:
            logger.warning(f"⚠️ Phase 2D Reversal V2 pipeline execution warning: {v2_err}")
        return count
    except Exception as e:
        logger.exception("❌ CRITICAL REVERSAL SCAN ERROR")
        import database
        if not getattr(database, "DONT_SAVE_ALERTS", False):
            try:
                upsert_scanner_health(scanner_name="REVERSAL", status="DOWN", error_msg=str(e))
            except Exception:
                pass
        return 0


def _run_reversal_v2_pipeline():
    """
    Executes Phase 2D Reversal V2 pipeline in parallel isolation.
    """
    logger.info("[V2_PIPELINE] Starting Phase 2D Reversal V2 pipeline...")
    from reversal_schema import init_reversal_v2_schema
    from reversal_engine import evaluate_reversal_v2_symbol

    init_reversal_v2_schema()

    elite_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "elite_universe_v2.parquet"))
    nq_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "near_qualified_v2.parquet"))

    if not os.path.exists(elite_path):
        elite_path = "data/elite_universe_v2.parquet"
        nq_path = "data/near_qualified_v2.parquet"

    elite_syms = set()
    if os.path.exists(elite_path):
        df_e = pd.read_parquet(elite_path)
        col = "symbol" if "symbol" in df_e.columns else df_e.columns[0]
        elite_syms = set(df_e[col].dropna().tolist())

    nq_syms = set()
    if os.path.exists(nq_path):
        df_n = pd.read_parquet(nq_path)
        col = "symbol" if "symbol" in df_n.columns else df_n.columns[0]
        nq_syms = set(df_n[col].dropna().tolist())

    logger.info(f"[V2_PIPELINE] Loaded universes: ELITE ({len(elite_syms)} symbols), NQ ({len(nq_syms)} symbols).")
    logger.info("[V2_PIPELINE] Phase 2D Reversal V2 pipeline evaluation ready.")

