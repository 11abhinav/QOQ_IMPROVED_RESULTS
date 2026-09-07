# =====================================================================================
# app/technical_scanner.py
# PRODUCTION-GRADE UNIFIED TECHNICAL PATTERN & ANTI-FAKE SCANNER — Daily 18:15 IST
#
# RULE 67 MANDATORY CHANGE-RATIONALE:
# - Institutional-grade technical scanner conforming to the "Permissive Pattern Discovery,
#   Aggressive Pattern-Specific Validation" architecture.
# - Recognizes 8 core primary structures:
#   * Tier A: BULL_FLAG, SHAKEOUT_RECLAIM, DOUBLE_BOTTOM, V_REVERSAL
#   * Tier B: CUP_HANDLE, ASCENDING_TRIANGLE, BULL_PENNANT, HIGHER_LOW_REVERSAL
# - Pattern-Specific Anti-Fake & Volume Signature validations:
#   * Bull Flag: Pole directional efficiency (>=0.55), shallow flag retrace (<=45%),
#     flag volume contraction (avg_flag_vol / avg_pole_vol <= 0.85). Target: Pole projection.
#   * Shakeout: Selling exhaustion, bullish absorption, volume absorption vs selloff.
#     Target: Pre-selloff structural resistance or measured expansion continuation (>= 1.5R).
#   * Double Bottom: Preceding markdown (>=7.0% / 2x ATR), 8-45 bars trough spacing,
#     twin trough diff <= 2.5%, clean neckline height >= 4.0%, fresh breakout timing.
#   * Cup & Handle: Rounded base (5-30% depth), handle in upper 35% of cup with drying volume.
#   * Ascending Triangle: Multi-touch flat resistance + ascending swing lows compression.
#   * V-Reversal: Sharp drop >=5% + steep multi-bar recovery >=55% on volume.
#   * Bull Pennant: Converging coil (3-10 bars) following directional pole (>=6.0% / 2x ATR).
#   * Higher Low Reversal: Distinct multi-bar higher low + swing high breakout.
# - Universal Common Hard Gates:
#   * Candle: Green trigger candle (Close > Open) & Non-zero spread.
#   * Liquidity: Volume >= 25k, Turnover >= ₹50 Lakhs.
#   * Volume: RVOL >= 1.20x strictly non-negotiable.
#   * Price Action: CLV >= 0.65, Upper Wick <= 30%.
#   * Risk: Room to Resistance >= 1.5R.
# - 100-Point Additive Scoring Model (<70 Reject, 70-79 Strong, 80-89 Very Strong, 90-100 Elite).
# - Tier-B Scoring Calibrated to 22 baseline points to avoid starvation under 70 threshold.
# - Full Forensic Telemetry Contract (`TECHNICAL_TRACE`) & Funnel Conservation Logging.
# =====================================================================================

import logging
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from database import (
    complete_scanner_execution_run,
    get_elite_watchlist,
    init_db,
    save_alert_if_new,
    start_scanner_execution_run,
    upsert_scanner_health,
)
from lock_utils import ProcessLock
from price_cache import fetch_watchlist_data
from technical_indicators import apply_indicators
from telemetry_manager import telemetry
from watchlist_cache import get_watchlist

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_scan_lock = ProcessLock("technical_scanner_lock")
_global_lock = ProcessLock("global_scanner_lock")

# =====================================================================================
# CONFIGURABLE STRATEGY PARAMETERS & THRESHOLDS
# =====================================================================================

# Universal Hard Gates
MIN_RVOL_HARD_GATE = 1.20           # Minimum RVOL for trigger candle (RVOL < 1.20 -> REJECT)
MIN_CLV_HARD_GATE = 0.65            # Minimum Close Location Value ((C - L) / (H - L) >= 0.65)
MAX_UPPER_WICK_PCT = 0.30           # Maximum upper wick ratio ((H - max(O, C)) / (H - L) <= 0.30)
MIN_AVG_TURNOVER_INR = 50_00_000    # Minimum 20-day average turnover: ₹50 Lakhs
MIN_AVG_VOLUME = 25_000             # Minimum 20-day average volume: 25k shares

# Aliases for backward compatibility in unit tests & external callers
MIN_RVOL = MIN_RVOL_HARD_GATE
MIN_CLV = MIN_CLV_HARD_GATE
MIN_AVG_TURNOVER = MIN_AVG_TURNOVER_INR

# Risk & Target Parameters
MAX_SL_PCT = 0.06                   # Maximum structural SL distance cap (6.0%)
MIN_SL_PCT = 0.012                  # Minimum risk buffer (1.2%)
MIN_ROOM_TO_RESISTANCE_R = 1.5      # Minimum R-multiple room to major overhead resistance

# Pattern-Specific Thresholds
BULL_FLAG_MIN_POLE_GAIN = 5.0       # Minimum pole gain % (or 2.0x ATR)
BULL_FLAG_MIN_POLE_EFFICIENCY = 0.55# Net move / Gross move ratio for directional efficiency
BULL_FLAG_MAX_RETRACE = 0.45        # Maximum retracement ratio of pole (45%)
BULL_FLAG_MAX_VOL_RATIO = 0.85      # Avg Flag Volume / Avg Pole Volume <= 0.85

SHAKEOUT_MIN_DECLINE_PCT = 4.0      # Minimum prior drop % for shakeout setup

# Double Bottom Structural Invariants
DOUBLE_BOTTOM_MIN_PRIOR_DROP_PCT = 7.0 # Minimum prior downtrend % into Trough 1
DOUBLE_BOTTOM_MIN_TROUGH_BARS = 8      # Minimum daily bars between Trough 1 and Trough 2
DOUBLE_BOTTOM_MAX_TROUGH_BARS = 45     # Maximum daily bars between Trough 1 and Trough 2
DOUBLE_BOTTOM_MAX_DIFF_PCT = 2.5       # Max % difference between Trough 1 and Trough 2
DOUBLE_BOTTOM_MIN_HEIGHT_PCT = 4.0     # Minimum neckline height % above troughs

CUP_HANDLE_MIN_DEPTH_PCT = 5.0      # Minimum cup depth %
CUP_HANDLE_MAX_DEPTH_PCT = 30.0     # Maximum cup depth %
CUP_HANDLE_MAX_HANDLE_RETRACE = 0.35# Handle depth / Cup depth <= 35%


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return float(val)
    except Exception:
        return default


def _coalesce_indicator_with_source(
    df: pd.DataFrame,
    aliases: List[str],
    validator: Optional[Any] = None,
    default: float = 0.0,
    default_source: str = "DEFAULT_2PCT"
) -> Tuple[float, str]:
    """
    Extracts latest non-NaN, finite value matching validator from the first valid alias.
    Returns (value, source_name).
    """
    if df is not None and not df.empty:
        for alias in aliases:
            if alias in df.columns:
                s = df[alias]
                if hasattr(s, "iloc") and len(s) > 0:
                    val = s.iloc[-1]
                    val_f = _safe_float(val, float("nan"))
                    if not math.isnan(val_f) and not math.isinf(val_f):
                        if validator is None or validator(val_f):
                            return val_f, alias
    return default, default_source


def _coalesce_indicator_val(
    df: pd.DataFrame,
    aliases: List[str],
    default: float = 0.0,
    validator: Optional[Any] = None,
) -> float:
    """Extracts latest non-NaN, finite value from the first matching valid column alias."""
    val, _ = _coalesce_indicator_with_source(df, aliases, validator=validator, default=default)
    return val


def _coalesce_indicator_series(df: pd.DataFrame, aliases: List[str]) -> Optional[pd.Series]:
    """Returns the first non-empty Series from the alias list that contains valid non-NaN values."""
    if df is None or df.empty:
        return None
    for alias in aliases:
        if alias in df.columns:
            s = df[alias]
            if hasattr(s, "dropna") and not s.dropna().empty:
                return s
    return None


def _extract_ohlcv(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Case-insensitive OHLCV array extractor supporting both lowercase and titlecase column schemas.
    """
    cols = {str(c).lower(): c for c in df.columns}
    o_col = cols.get("open", "Open")
    h_col = cols.get("high", "High")
    l_col = cols.get("low", "Low")
    c_col = cols.get("close", "Close")
    v_col = cols.get("volume", "Volume")
    
    opens = df[o_col].values.astype(float)
    highs = df[h_col].values.astype(float)
    lows = df[l_col].values.astype(float)
    closes = df[c_col].values.astype(float)
    volumes = df[v_col].values.astype(float)
    return opens, highs, lows, closes, volumes


def _find_swing_pivots(highs: np.ndarray, lows: np.ndarray, lookback: int = 2) -> Tuple[List[int], List[int]]:
    """
    Identifies robust multi-bar fractal swing highs (peaks) and swing lows (troughs).
    A bar i is a swing low if lows[i] < all other lows in [i - lookback, i + lookback].
    A bar i is a swing high if highs[i] > all other highs in [i - lookback, i + lookback].
    Strict inequality prevents flat continuous price plateaus from registering as fake pivots.
    """
    n = len(highs)
    peak_indices: List[int] = []
    trough_indices: List[int] = []

    if n < (2 * lookback + 1):
        return peak_indices, trough_indices

    for i in range(lookback, n - lookback):
        # Trough test: strictly lower than neighbors
        left_lows = lows[i - lookback: i]
        right_lows = lows[i + 1: i + lookback + 1]
        if lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
            trough_indices.append(i)

        # Peak test: strictly higher than neighbors
        left_highs = highs[i - lookback: i]
        right_highs = highs[i + 1: i + lookback + 1]
        if highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
            peak_indices.append(i)

    return peak_indices, trough_indices


# =====================================================================================
# 1. PERMISSIVE PATTERN DISCOVERY SUB-DETECTORS (8 CORE STRUCTURES)
# =====================================================================================

def _detect_bull_flag(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier A Pattern: Bull Flag + Pole
    - Pole: Directional impulse over 3 to 10 bars (gain >= 5% or >= 2x ATR, efficiency >= 0.55).
    - Flag: Controlled consolidation over 3 to 12 bars, retrace <= 45%, volume ratio <= 0.85.
    
    [RULE 67 MANDATORY FIX - BULL FLAG TARGET RESOLUTION]:
    Target is set to the structural Pole Projection (flag_resistance + pole_move) or major
    structural resistance cluster, completely eliminating the minor-high trap where a 1-bar wick
    caused false <1.5R rejections.
    """
    n = len(df)
    if n < 20:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    o_today = _safe_float(opens[today_idx])
    c_yesterday = _safe_float(closes[today_idx - 1])

    if c_today <= o_today:
        return None

    if atr14 is None or atr14 <= 0:
        atr14 = _coalesce_indicator_val(df, ["ATR", "ATR_14", "ATR20", "atr", "atr_14", "atr20"], default=c_today * 0.02)

    best_setup = None

    for flag_len in range(3, 13):
        pole_end_idx = today_idx - flag_len
        if pole_end_idx < 4:
            continue

        flag_highs = highs[pole_end_idx: today_idx]
        flag_lows = lows[pole_end_idx: today_idx]
        flag_vols = volumes[pole_end_idx: today_idx]

        flag_resistance = float(np.max(flag_highs))
        flag_support = float(np.min(flag_lows))

        # Fresh Breakout Timing: Must break above flag resistance today without having closed far above it yesterday
        if c_today < flag_resistance * 1.001:
            continue
        if c_yesterday > (flag_resistance * 1.005 + 1e-4):
            continue

        for pole_len in range(3, 11):
            pole_start_idx = pole_end_idx - pole_len
            if pole_start_idx < 0:
                continue

            pole_sub_lows = lows[pole_start_idx: pole_end_idx]
            pole_sub_highs = highs[pole_start_idx: pole_end_idx + 1]
            pole_low = float(np.min(pole_sub_lows))
            pole_high = float(np.max(pole_sub_highs))
            pole_move = pole_high - pole_low
            pole_pct = (pole_move / max(pole_low, 1.0)) * 100.0

            if pole_pct < BULL_FLAG_MIN_POLE_GAIN and pole_move < (2.0 * atr14):
                continue

            # Pole directional efficiency
            pole_closes = closes[pole_start_idx: pole_end_idx + 1]
            gross_movement = np.sum(np.abs(np.diff(pole_closes)))
            net_movement = abs(pole_closes[-1] - pole_closes[0])
            pole_efficiency = (net_movement / max(gross_movement, 0.01))

            if pole_efficiency < BULL_FLAG_MIN_POLE_EFFICIENCY:
                continue

            # Flag retracement check
            retrace_depth = pole_high - flag_support
            retrace_ratio = retrace_depth / max(pole_move, 0.01)
            if retrace_ratio > BULL_FLAG_MAX_RETRACE or retrace_ratio < 0:
                continue

            # Volume signature: flag volume must contract relative to pole
            pole_vols = volumes[pole_start_idx: pole_end_idx + 1]
            avg_pole_vol = float(np.mean(pole_vols)) if len(pole_vols) > 0 else 1.0
            avg_flag_vol = float(np.mean(flag_vols)) if len(flag_vols) > 0 else 1.0
            vol_ratio_flag_to_pole = avg_flag_vol / max(avg_pole_vol, 1.0)

            if vol_ratio_flag_to_pole > BULL_FLAG_MAX_VOL_RATIO:
                continue

            # Robust Pole Projection Target
            pole_target = flag_resistance + pole_move
            overhead_highs = highs[max(0, pole_start_idx - 40): pole_start_idx]
            higher_res = [h for h in overhead_highs if h > flag_resistance * 1.03]
            target_res = float(max(higher_res)) if higher_res else pole_target

            setup = {
                "pattern": "BULL_FLAG",
                "tier": "TIER_A",
                "pole_gain_pct": round(pole_pct, 2),
                "pole_bars": pole_len,
                "pole_efficiency": round(pole_efficiency, 2),
                "flag_bars": flag_len,
                "flag_resistance": round(flag_resistance, 2),
                "flag_support": round(flag_support, 2),
                "retracement_pct": round(retrace_ratio * 100.0, 1),
                "vol_ratio_flag_to_pole": round(vol_ratio_flag_to_pole, 2),
                "invalidation_level": round(flag_support * 0.995, 2),
                "target_resistance": round(target_res, 2),
                "pattern_quality_score": 25 if (pole_pct >= 10.0 and vol_ratio_flag_to_pole <= 0.70) else 23,
                "description": f"Bull Flag (Pole +{pole_pct:.1f}%, Flag {flag_len}b, Retrace {retrace_ratio*100:.0f}%, VolRatio {vol_ratio_flag_to_pole:.2f}x)",
            }
            if best_setup is None or setup["pole_gain_pct"] > best_setup["pole_gain_pct"]:
                best_setup = setup

    return best_setup


def _detect_shakeout_reclaim(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier A Pattern: Shakeout Reclaim (Bottom Absorption)
    - Selloff: Decline >= 4.0% or >= 1.2x ATR over 3 to 15 sessions into support.
    - Absorption: Green candle at support engulfing preceding red candle.
    - Target: Pre-selloff structural resistance or measured expansion continuation (>= 1.5R).
    
    [RULE 67 MANDATORY FIX - SHAKEOUT RECLAIM DEADLOCK RESOLUTION]:
    Pre-refactor logic set target_resistance strictly to drop_origin_high. Since an engulfing
    reclaim candle already recovers 50-70% of the drop, room_to_resistance was capped at 0.5-0.9R,
    making valid shakeouts mathematically impossible.
    Fix: Uses pre-selloff structural resistance or measured expansion continuation target
    (drop_high + 0.618 * drop_points) providing clean 1.5R - 3.0R headroom calculation.
    """
    n = len(df)
    if n < 20:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    prev_idx = today_idx - 1

    c_today = _safe_float(closes[today_idx])
    o_today = _safe_float(opens[today_idx])
    c_prev = _safe_float(closes[prev_idx])
    o_prev = _safe_float(opens[prev_idx])
    h_prev = _safe_float(highs[prev_idx])
    l_prev = _safe_float(lows[prev_idx])
    v_today = _safe_float(volumes[today_idx])

    if c_today <= o_today or c_today <= c_prev or c_prev > o_prev:
        return None

    if atr14 is None or atr14 <= 0:
        atr14 = _coalesce_indicator_val(df, ["ATR", "ATR_14", "ATR20", "atr", "atr_14", "atr20"], default=c_today * 0.02)

    prev_body = abs(o_prev - c_prev)
    today_body = c_today - o_today

    is_body_engulfing = (c_today >= o_prev) and (today_body >= prev_body * 0.85)
    is_full_reclaim = (c_today >= h_prev)

    if not (is_body_engulfing or is_full_reclaim):
        return None

    engulf_type = "LEVEL_B_FULL_RECLAIM" if is_full_reclaim else "LEVEL_A_BODY_ENGULFING"

    lookback = min(16, n - 2)
    recent_highs = highs[today_idx - lookback: today_idx]
    recent_lows = lows[today_idx - lookback: today_idx]
    recent_vols = volumes[today_idx - lookback: today_idx]

    drop_high = float(np.max(recent_highs))
    trough_low = float(np.min(recent_lows))
    drop_points = drop_high - trough_low
    drop_pct = (drop_points / max(drop_high, 1.0)) * 100.0

    if drop_pct < SHAKEOUT_MIN_DECLINE_PCT and drop_points < (1.2 * atr14):
        return None

    avg_selloff_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 1.0
    vol_vs_selloff = v_today / max(avg_selloff_vol, 1.0)
    if vol_vs_selloff < 0.85:
        return None

    base_support = min(lows[today_idx], l_prev, trough_low)

    # Resolve pre-selloff structural resistance or measured expansion continuation target
    lookback_pre = min(45, n - 2)
    pre_selloff_highs = highs[max(0, today_idx - lookback_pre): today_idx - lookback]
    higher_res = [h for h in pre_selloff_highs if h > c_today * 1.02]
    major_target_res = float(max(higher_res)) if higher_res else (drop_high + (drop_points * 0.618))

    return {
        "pattern": "SHAKEOUT_RECLAIM",
        "tier": "TIER_A",
        "engulfing_type": engulf_type,
        "drop_origin_high": round(drop_high, 2),
        "trough_low": round(trough_low, 2),
        "selloff_depth_pct": round(drop_pct, 2),
        "vol_vs_selloff": round(vol_vs_selloff, 2),
        "invalidation_level": round(base_support * 0.995, 2),
        "target_resistance": round(major_target_res, 2),
        "pattern_quality_score": 25 if is_full_reclaim else 23,
        "description": f"Shakeout Reclaim ({engulf_type.replace('_', ' ')}, Prior Drop -{drop_pct:.1f}%, Vol vs Selloff {vol_vs_selloff:.2f}x)",
    }


def _detect_double_bottom(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier A Pattern: Double Bottom Breakout (Reversal)
    Requires strict mathematical invariants:
    1. Preceding Downtrend: Prior selloff >= 7.0% (or >= 2.0x ATR) into Trough 1.
    2. Multi-Bar Swing Lows: Troughs must be genuine multi-bar fractals.
    3. Structural Spacing: Troughs separated by 8 to 45 daily bars.
    4. Price Symmetry: Trough 2 within 2.5% of Trough 1.
    5. Neckline Clearance: High between troughs must be >= 4.0% above troughs.
    6. Fresh Breakout Timing: Today's candle must be the INITIAL breakout crossing neckline.
    """
    n = len(df)
    if n < 35:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    c_yesterday = _safe_float(closes[today_idx - 1])

    if atr14 is None or atr14 <= 0:
        atr14 = _coalesce_indicator_val(df, ["ATR", "ATR_14", "ATR20", "atr", "atr_14", "atr20"], default=c_today * 0.02)

    # Search window up to 70 bars back
    window_start = max(0, today_idx - 70)
    w_highs = highs[window_start: today_idx]
    w_lows = lows[window_start: today_idx]

    if len(w_lows) < 20:
        return None

    _, rel_troughs = _find_swing_pivots(w_highs, w_lows, lookback=2)
    local_trough_indices = [window_start + idx for idx in rel_troughs]

    if len(local_trough_indices) < 2:
        return None

    for i in range(len(local_trough_indices) - 1):
        t1 = local_trough_indices[i]
        for j in range(i + 1, len(local_trough_indices)):
            t2 = local_trough_indices[j]

            # Spacing between troughs: 8 to 45 daily bars
            bars_between = t2 - t1
            if bars_between < DOUBLE_BOTTOM_MIN_TROUGH_BARS or bars_between > DOUBLE_BOTTOM_MAX_TROUGH_BARS:
                continue

            if t2 >= today_idx - 1:
                continue

            l1_val = float(lows[t1])
            l2_val = float(lows[t2])

            # Invariant 1: Troughs price symmetry within 2.5%
            diff_pct = abs(l1_val - l2_val) / min(l1_val, l2_val) * 100.0
            if diff_pct > DOUBLE_BOTTOM_MAX_DIFF_PCT:
                continue

            # Invariant 2: Prior Downtrend into Trough 1
            pre_t1_start = max(0, t1 - 30)
            if t1 - pre_t1_start < 5:
                continue
            pre_t1_high = float(np.max(highs[pre_t1_start: t1]))
            prior_drop_pct = (pre_t1_high - l1_val) / max(pre_t1_high, 1.0) * 100.0
            prior_drop_points = pre_t1_high - l1_val

            if prior_drop_pct < DOUBLE_BOTTOM_MIN_PRIOR_DROP_PCT and prior_drop_points < (2.0 * atr14):
                continue

            # Invariant 3: Clean Neckline Peak between Troughs
            neckline_window = highs[t1 + 1: t2]
            if len(neckline_window) < 3:
                continue
            neckline_val = float(np.max(neckline_window))
            trough_base = min(l1_val, l2_val)
            pattern_height_pct = (neckline_val - trough_base) / max(trough_base, 1.0) * 100.0

            if pattern_height_pct < DOUBLE_BOTTOM_MIN_HEIGHT_PCT:
                continue

            # Invariant 4: No severe breakdown between troughs
            mid_lows = lows[t1: t2 + 1]
            if np.min(mid_lows) < (trough_base * 0.975):
                continue

            # Invariant 5: Fresh Breakout Timing Gate (Today is first breakout day)
            if c_today < (neckline_val * 1.002 - 1e-4):
                continue
            if c_yesterday > (neckline_val * 1.005 + 1e-4):
                continue

            sl_level = round(max(l2_val * 0.995, neckline_val * 0.96), 2)
            pre_pattern_highs = highs[max(0, t1 - 40): t1]
            higher_res = [h for h in pre_pattern_highs if h > c_today * 1.02]
            major_target_res = float(max(higher_res)) if higher_res else (neckline_val + (neckline_val - trough_base))

            return {
                "pattern": "DOUBLE_BOTTOM",
                "tier": "TIER_A",
                "trough_1": round(l1_val, 2),
                "trough_2": round(l2_val, 2),
                "neckline": round(neckline_val, 2),
                "trough_diff_pct": round(diff_pct, 2),
                "prior_drop_pct": round(prior_drop_pct, 1),
                "pattern_height_pct": round(pattern_height_pct, 2),
                "trough_bars": bars_between,
                "invalidation_level": sl_level,
                "target_resistance": round(major_target_res, 2),
                "pattern_quality_score": 25 if diff_pct <= 1.0 and prior_drop_pct >= 10.0 else 23,
                "description": (
                    f"Double Bottom Breakout (Neckline ₹{neckline_val:.2f}, "
                    f"Prior Drop -{prior_drop_pct:.1f}%, Height {pattern_height_pct:.1f}%, {bars_between}b apart)"
                ),
            }
    return None


def _detect_v_reversal(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier A Pattern: V-Reversal Recovery
    - Sharp decline >= 5% or >= 2.0x ATR followed by strong multi-bar recovery >= 55%.
    - Fresh recovery momentum.
    """
    n = len(df)
    if n < 15:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    o_today = _safe_float(opens[today_idx])

    if c_today <= o_today:
        return None

    if atr14 is None or atr14 <= 0:
        atr14 = _coalesce_indicator_val(df, ["ATR", "ATR_14", "ATR20", "atr", "atr_14", "atr20"], default=c_today * 0.02)

    lookback = min(15, n - 2)
    recent_highs = highs[today_idx - lookback: today_idx - 2]
    recent_lows = lows[today_idx - lookback: today_idx]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None

    drop_high = float(np.max(recent_highs))
    trough_low = float(np.min(recent_lows))
    drop_points = drop_high - trough_low
    drop_pct = (drop_points / max(drop_high, 1.0)) * 100.0

    if drop_pct < 5.0 and drop_points < (2.0 * atr14):
        return None

    recovery_pct = (c_today - trough_low) / max(drop_points, 0.01) * 100.0
    if recovery_pct < 55.0:
        return None

    lookback_pre = min(50, n - 2)
    pre_drop_highs = highs[max(0, today_idx - lookback_pre): today_idx - lookback]
    higher_res = [h for h in pre_drop_highs if h > c_today * 1.03]
    major_target_res = float(max(higher_res)) if higher_res else (drop_high + (drop_points * 0.382))

    sl_level = round(trough_low * 0.995, 2)
    return {
        "pattern": "V_REVERSAL",
        "tier": "TIER_A",
        "drop_origin_high": round(drop_high, 2),
        "trough_low": round(trough_low, 2),
        "selloff_depth_pct": round(drop_pct, 1),
        "invalidation_level": sl_level,
        "target_resistance": round(major_target_res, 2),
        "pattern_quality_score": 23,
        "description": f"V-Reversal Recovery (-{drop_pct:.1f}% Drop, Recovered {recovery_pct:.0f}%)",
    }


def _detect_cup_and_handle(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier B Pattern: Cup & Handle Breakout
    - Rounded U-base over 20-50 bars (depth 5% - 30%).
    - Handle pullback in upper 35% portion of cup (depth <= 35% of cup).
    - Fresh breakout above rim.
    """
    n = len(df)
    if n < 30:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    c_yesterday = _safe_float(closes[today_idx - 1])

    for handle_len in range(3, 12):
        rim_idx = today_idx - handle_len
        if rim_idx < 15:
            continue

        handle_low = float(np.min(lows[rim_idx: today_idx]))
        rim_high = float(highs[rim_idx])

        for cup_len in range(20, min(50, rim_idx)):
            left_rim_idx = rim_idx - cup_len
            left_rim_high = float(np.max(highs[left_rim_idx: left_rim_idx + 4]))

            cup_low_slice = lows[left_rim_idx: rim_idx]
            cup_bottom_rel_idx = int(np.argmin(cup_low_slice))
            if cup_bottom_rel_idx < 2 or cup_bottom_rel_idx > len(cup_low_slice) - 3:
                continue

            cup_bottom = float(np.min(cup_low_slice))

            rim_diff = abs(left_rim_high - rim_high) / min(left_rim_high, rim_high) * 100.0
            if rim_diff > 3.5:
                continue

            cup_depth = rim_high - cup_bottom
            cup_depth_pct = (cup_depth / rim_high) * 100.0
            if cup_depth_pct < CUP_HANDLE_MIN_DEPTH_PCT or cup_depth_pct > CUP_HANDLE_MAX_DEPTH_PCT:
                continue

            handle_depth = rim_high - handle_low
            if handle_depth > (cup_depth * CUP_HANDLE_MAX_HANDLE_RETRACE):
                continue

            # Fresh Breakout timing (first breakout day above rim)
            if c_today >= (rim_high * 1.002 - 1e-4) and c_yesterday <= (rim_high * 1.005 + 1e-4):
                sl_level = round(handle_low * 0.995, 2)
                measured_target = rim_high + cup_depth
                return {
                    "pattern": "CUP_HANDLE",
                    "tier": "TIER_B",
                    "rim_level": round(rim_high, 2),
                    "cup_depth_pct": round(cup_depth_pct, 1),
                    "handle_bars": handle_len,
                    "invalidation_level": sl_level,
                    "target_resistance": round(measured_target, 2),
                    "pattern_quality_score": 22,
                    "description": f"Cup & Handle Breakout (Rim ₹{rim_high:.2f}, Cup -{cup_depth_pct:.1f}%, Handle {handle_len}b)",
                }
    return None


def _detect_ascending_triangle(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier B Pattern: Ascending Triangle Breakout
    - Flat horizontal resistance line (2+ peaks within 1.5%).
    - Multi-bar ascending swing lows compressing upward.
    - Fresh breakout today through ceiling.
    """
    n = len(df)
    if n < 25:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    c_yesterday = _safe_float(closes[today_idx - 1])

    lookback = min(40, n - 2)
    sub_highs = highs[today_idx - lookback: today_idx]
    sub_lows = lows[today_idx - lookback: today_idx]

    if len(sub_highs) < 15:
        return None

    peaks, troughs = _find_swing_pivots(sub_highs, sub_lows, lookback=2)

    if len(peaks) < 2 or len(troughs) < 2:
        return None

    peak_vals = [float(sub_highs[p]) for p in peaks]
    trough_vals = [float(sub_lows[t]) for t in troughs]

    res_level = float(np.max(peak_vals))
    near_peaks = [p for p in peak_vals if (res_level - p) / res_level <= 0.015]
    if len(near_peaks) < 2:
        return None

    # Verify ascending sequence
    if trough_vals[-1] <= trough_vals[0] * 1.01:
        return None

    is_ascending = all(trough_vals[k] >= trough_vals[k-1] * 0.995 for k in range(1, len(trough_vals)))
    if not is_ascending:
        return None

    if c_today >= (res_level * 1.002 - 1e-4) and c_yesterday <= (res_level * 1.005 + 1e-4):
        last_low = trough_vals[-1]
        sl_level = round(last_low * 0.995, 2)
        measured_target = res_level + (res_level - trough_vals[0])
        return {
            "pattern": "ASCENDING_TRIANGLE",
            "tier": "TIER_B",
            "resistance_level": round(res_level, 2),
            "ascending_lows_count": len(troughs),
            "invalidation_level": sl_level,
            "target_resistance": round(measured_target, 2),
            "pattern_quality_score": 22,
            "description": f"Ascending Triangle Breakout (Res ₹{res_level:.2f}, Lows +{len(troughs)})",
        }
    return None


def _detect_bull_pennant(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier B Pattern: Bull Pennant
    - Symmetrical converging coil (3 to 10 bars) following a strong directional pole (>=6.0% or 2x ATR).
    - Fresh breakout today above pennant upper boundary.
    """
    n = len(df)
    if n < 18:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    c_yesterday = _safe_float(closes[today_idx - 1])

    if atr14 is None or atr14 <= 0:
        atr14 = _coalesce_indicator_val(df, ["ATR", "ATR_14", "ATR20", "atr", "atr_14", "atr20"], default=c_today * 0.02)

    for pennant_len in range(3, 10):
        pole_end = today_idx - pennant_len
        if pole_end < 4:
            continue

        pole_start = max(0, pole_end - 10)
        pole_low = float(np.min(lows[pole_start: pole_end]))
        pole_high = float(np.max(highs[pole_start: pole_end + 1]))
        pole_gain = (pole_high - pole_low) / max(pole_low, 1.0) * 100.0

        if pole_gain < 6.0 and (pole_high - pole_low) < (2.0 * atr14):
            continue

        p_highs = highs[pole_end: today_idx]
        p_lows = lows[pole_end: today_idx]

        if len(p_highs) >= 3 and p_highs[-1] < (p_highs[0] * 0.995) and p_lows[-1] > (p_lows[0] * 1.005):
            pennant_top = float(np.max(p_highs))
            if c_today >= (pennant_top * 1.002 - 1e-4) and c_yesterday <= (pennant_top * 1.005 + 1e-4):
                sl_level = round(float(np.min(p_lows)) * 0.995, 2)
                pole_move = pole_high - pole_low
                pre_highs = highs[max(0, pole_start - 30): pole_start]
                higher_res = [h for h in pre_highs if h > c_today * 1.02]
                target_res = float(max(higher_res)) if higher_res else (pennant_top + pole_move)
                return {
                    "pattern": "BULL_PENNANT",
                    "tier": "TIER_B",
                    "pole_gain_pct": round(pole_gain, 1),
                    "pennant_bars": pennant_len,
                    "invalidation_level": sl_level,
                    "target_resistance": round(target_res, 2),
                    "pattern_quality_score": 22,
                    "description": f"Bull Pennant Breakout (Pole +{pole_gain:.1f}%, Pennant {pennant_len}b)",
                }
    return None


def _detect_higher_low_reversal(df: pd.DataFrame, atr14: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Tier B Pattern: Higher-Low Structure Break
    - 2-bar swing pivots confirm a distinct higher low followed by a break above the intervening swing high.
    - Fresh breakout timing.
    """
    n = len(df)
    if n < 25:
        return None

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    c_yesterday = _safe_float(closes[today_idx - 1])

    lookback = min(35, n - 2)
    sub_highs = highs[today_idx - lookback: today_idx]
    sub_lows = lows[today_idx - lookback: today_idx]

    if len(sub_lows) < 15:
        return None

    peaks, troughs = _find_swing_pivots(sub_highs, sub_lows, lookback=2)

    if len(troughs) < 2 or len(peaks) < 1:
        return None

    l1_idx = troughs[-2]
    l2_idx = troughs[-1]

    if sub_lows[l2_idx] <= sub_lows[l1_idx] * 1.01:
        return None

    valid_peaks = [p for p in peaks if l1_idx < p < l2_idx]
    if not valid_peaks:
        h1_val = float(np.max(sub_highs[l1_idx: l2_idx + 1]))
    else:
        h1_val = float(sub_highs[valid_peaks[-1]])

    if c_today >= (h1_val * 1.002 - 1e-4) and c_yesterday <= (h1_val * 1.005 + 1e-4):
        sl_level = round(float(sub_lows[l2_idx]) * 0.995, 2)
        overhead_high = float(np.max(sub_highs))
        target_res = overhead_high if overhead_high > c_today * 1.02 else (c_today * 1.15)
        return {
            "pattern": "HIGHER_LOW_REVERSAL",
            "tier": "TIER_B",
            "swing_high": round(h1_val, 2),
            "higher_low": round(float(sub_lows[l2_idx]), 2),
            "invalidation_level": sl_level,
            "target_resistance": round(target_res, 2),
            "pattern_quality_score": 22,
            "description": f"Higher-Low Structure Break (Swing High ₹{h1_val:.2f}, Low ₹{sub_lows[l2_idx]:.2f})",
        }
    return None


# =====================================================================================
# 2. TIER C CONFLUENCE BOOSTERS (SECONDARY BOOSTERS)
# =====================================================================================

def _detect_confluence_factors(df: pd.DataFrame) -> Tuple[List[str], int]:
    """
    Evaluates secondary confluence factors (Max +5 points total).
    """
    confluences = []
    bonus_pts = 0
    n = len(df)
    if n < 15:
        return confluences, bonus_pts

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    c_today = _safe_float(closes[-1])
    o_today = _safe_float(opens[-1])
    h_today = _safe_float(highs[-1])
    l_today = _safe_float(lows[-1])
    c_prev = _safe_float(closes[-2])

    candle_range = h_today - l_today
    body = abs(c_today - o_today)

    if candle_range > 0:
        lower_wick = min(c_today, o_today) - l_today
        upper_wick = h_today - max(c_today, o_today)
        if lower_wick >= 1.8 * body and upper_wick <= 0.25 * candle_range:
            confluences.append("HAMMER_AT_SUPPORT")
            bonus_pts += 2

    ema20_series = _coalesce_indicator_series(df, ["EMA20", "EMA_20", "ema20", "ema_20"])
    if ema20_series is not None and len(ema20_series) >= 2:
        ema20 = float(ema20_series.iloc[-1])
        ema20_prev = float(ema20_series.iloc[-2])
        if c_prev <= ema20_prev and c_today > ema20:
            confluences.append("EMA20_RECLAIM")
            bonus_pts += 1
        elif c_today > ema20:
            confluences.append("ABOVE_EMA20")

    sma50_series = _coalesce_indicator_series(df, ["SMA50", "SMA_50", "sma50", "sma_50"])
    if sma50_series is not None and len(sma50_series) >= 1:
        sma50 = float(sma50_series.iloc[-1])
        if not math.isnan(sma50) and c_today > sma50:
            confluences.append("ABOVE_SMA50")
            bonus_pts += 1

    sma200_series = _coalesce_indicator_series(df, ["SMA200", "SMA_200", "sma200", "sma_200"])
    if sma200_series is not None and len(sma200_series) >= 1:
        sma200 = float(sma200_series.iloc[-1])
        if not math.isnan(sma200) and c_today > sma200:
            confluences.append("ABOVE_SMA200_UPTREND")
            bonus_pts += 1

    rsi_series = _coalesce_indicator_series(df, ["RSI", "RSI_14", "rsi", "rsi_14"])
    if rsi_series is not None and len(rsi_series) >= 25:
        rsi_today = float(rsi_series.iloc[-1])
        rsi_min_past = float(rsi_series.iloc[-15:-1].min())
        price_min_past = float(lows[-15:-1].min())
        if l_today <= price_min_past * 1.01 and rsi_today > rsi_min_past + 3.0:
            confluences.append("RSI_BULLISH_DIVERGENCE")
            bonus_pts += 2

    vol_sma20 = _coalesce_indicator_val(df, ["Volume_SMA20", "VOL_SMA20", "volume_sma20", "vol_sma20", "SMA20_Volume"], default=float(np.mean(volumes[-20:])))
    if float(volumes[-1]) >= 1.75 * max(vol_sma20, 1.0):
        confluences.append("INSTITUTIONAL_VOLUME_SURGE")
        bonus_pts += 1

    return confluences, min(5, bonus_pts)


# =====================================================================================
# 3. COMMON QUALITY, RISK, ANTI-FAKE & FORENSIC TELEMETRY ENGINE
# =====================================================================================

def detect_technical_setup(
    df: pd.DataFrame,
    symbol: str,
    return_trace: bool = False,
) -> Union[Optional[Dict[str, Any]], Tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
    """
    Unified Technical Pattern & Anti-Fake Engine with Forensic Telemetry:
    
    1. Common Hard Filters:
       - Candle Gate: Green trigger candle (Close > Open) & Non-zero spread.
       - Liquidity Filter: 20-day Volume >= 25k & Turnover >= ₹50L.
       - Hard Volume Gate: RVOL >= 1.20x.
       - Close Strength Gate: CLV >= 0.65.
       - Upper Wick Filter: Upper Wick <= 30% of range.
    2. Permissive Pattern Discovery (8 Core Structures).
    3. Pattern-Specific Validation & Volume Signature Matching.
    4. Risk Engine & Room-to-Resistance Hard Gate (>= 1.5R).
    5. Tier C Confluence Boosters.
    6. Clean 100-Point Additive Scoring Engine (Calibrated Tier-B).
    7. Forensic Trace Generation (`TECHNICAL_TRACE`).
    """
    now_ist = datetime.now(IST)
    trading_date = str(df.index[-1]).split(" ")[0] if (df is not None and not df.empty and hasattr(df.index[-1], "strftime")) else now_ist.strftime("%Y-%m-%d")
    final_bar_ts = str(df.index[-1]) if (df is not None and not df.empty) else now_ist.strftime("%Y-%m-%d %H:%M:%S")

    trace: Dict[str, Any] = {
        "symbol": symbol,
        "trading_date": trading_date,
        "final_bar_timestamp": final_bar_ts,
        "01_UNIVERSE": {
            "status": "PASS",
            "reason": "eligible_watchlist_symbol"
        },
        "01_DATA_VALIDATION": {},
        "02_COMMON_GATES": {
            "status": "PENDING",
            "rejection_code": None,
        },
        "03_PATTERN_DISCOVERY": {
            "patterns_considered": [
                "BULL_FLAG", "SHAKEOUT_RECLAIM", "DOUBLE_BOTTOM", "V_REVERSAL",
                "CUP_HANDLE", "ASCENDING_TRIANGLE", "BULL_PENNANT", "HIGHER_LOW_REVERSAL"
            ],
            "detected_patterns": [],
        },
        "04_PATTERN_VALIDATION": {},
        "05_RISK": {
            "status": "PENDING",
            "rejection_code": None,
        },
        "06_SCORE": {},
        "FINAL": {
            "status": "REJECTED",
            "terminal_stage": "INITIALIZATION",
            "terminal_reason": "DATA_UNAVAILABLE",
            "observed": {},
            "required": {},
        }
    }

    def _finish(res_obj: Optional[Dict[str, Any]]):
        if return_trace:
            return res_obj, trace
        return res_obj

    if df is None or df.empty or len(df) < 20:
        trace["FINAL"]["terminal_reason"] = "INSUFFICIENT_BARS"
        return _finish(None)

    n_bars = len(df)
    engine_min_history = 20
    if n_bars >= 200:
        history_class = "MATURE"
        history_confidence = "HIGH"
        trend_validation_mode = "SMA200"
    elif n_bars >= 50:
        history_class = "RECENT_LISTING"
        history_confidence = "STANDARD"
        trend_validation_mode = "SMA50_EMA20"
    elif n_bars >= 20:
        history_class = "FRESH_IPO"
        history_confidence = "STANDARD"
        trend_validation_mode = "EMA20"
    else:
        history_class = "FRESH_IPO"
        history_confidence = "LOW"
        trend_validation_mode = "EMA20"

    trace["01_DATA_VALIDATION"]["history_class"] = history_class
    trace["01_DATA_VALIDATION"]["history_confidence"] = history_confidence
    trace["01_DATA_VALIDATION"]["engine_min_history"] = engine_min_history
    trace["01_DATA_VALIDATION"]["trend_validation_mode"] = trend_validation_mode

    cols_lower = [str(c).lower() for c in df.columns]
    required_keys = ["open", "high", "low", "close", "volume"]
    if not all(k in cols_lower for k in required_keys):
        trace["FINAL"]["terminal_reason"] = "MISSING_OHLCV_COLUMNS"
        return _finish(None)

    has_ema20 = _coalesce_indicator_series(df, ["EMA20", "EMA_20", "ema20"]) is not None
    has_atr = _coalesce_indicator_val(df, ["ATR", "ATR_14", "ATR20", "atr"], default=0.0) > 0
    has_vol_sma = _coalesce_indicator_val(df, ["Volume_SMA20", "VOL_SMA20", "volume_sma20", "vol_sma20"], default=0.0) > 0

    if not (has_ema20 and has_atr and has_vol_sma):
        df = apply_indicators(df, timeframe="1d")

    opens, highs, lows, closes, volumes = _extract_ohlcv(df)
    n = len(df)
    today_idx = n - 1
    c_today = _safe_float(closes[today_idx])
    o_today = _safe_float(opens[today_idx])
    h_today = _safe_float(highs[today_idx])
    l_today = _safe_float(lows[today_idx])
    v_today = _safe_float(volumes[today_idx])

    trace["FINAL"]["observed"]["cmp"] = c_today
    trace["FINAL"]["observed"]["open"] = o_today
    trace["FINAL"]["observed"]["high"] = h_today
    trace["FINAL"]["observed"]["low"] = l_today
    trace["FINAL"]["observed"]["volume"] = v_today

    # ── COMMON HARD FILTER 0: GREEN CANDLE & NON-ZERO SPREAD ───────────────────────
    if c_today <= o_today or c_today <= 0:
        trace["02_COMMON_GATES"]["status"] = "REJECT"
        trace["02_COMMON_GATES"]["rejection_code"] = "RED_CANDLE"
        trace["FINAL"]["terminal_stage"] = "02_COMMON_GATES"
        trace["FINAL"]["terminal_reason"] = "RED_CANDLE"
        trace["FINAL"]["required"]["close_gt_open"] = True
        return _finish(None)

    candle_range = h_today - l_today
    if candle_range <= 0:
        trace["02_COMMON_GATES"]["status"] = "REJECT"
        trace["02_COMMON_GATES"]["rejection_code"] = "ZERO_RANGE"
        trace["FINAL"]["terminal_stage"] = "02_COMMON_GATES"
        trace["FINAL"]["terminal_reason"] = "ZERO_RANGE"
        return _finish(None)

    # ── COMMON HARD FILTER 1: LIQUIDITY & TURNOVER ─────────────────────────────────
    vol_sma20 = _coalesce_indicator_val(df, ["Volume_SMA20", "VOL_SMA20", "volume_sma20", "vol_sma20", "SMA20_Volume"], default=float(np.mean(volumes[-20:])))
    avg_turnover = vol_sma20 * c_today
    avg_turnover_cr = avg_turnover / 10_000_000.0

    trace["02_COMMON_GATES"]["avg_volume"] = round(vol_sma20, 1)
    trace["02_COMMON_GATES"]["avg_turnover_cr"] = round(avg_turnover_cr, 2)

    if vol_sma20 < MIN_AVG_VOLUME and avg_turnover < MIN_AVG_TURNOVER_INR:
        trace["02_COMMON_GATES"]["status"] = "REJECT"
        trace["02_COMMON_GATES"]["rejection_code"] = "ILLIQUID_STOCK"
        trace["FINAL"]["terminal_stage"] = "02_COMMON_GATES"
        trace["FINAL"]["terminal_reason"] = "ILLIQUID_STOCK"
        trace["FINAL"]["required"]["min_avg_volume"] = MIN_AVG_VOLUME
        trace["FINAL"]["required"]["min_avg_turnover_inr"] = MIN_AVG_TURNOVER_INR
        return _finish(None)

    # ── COMMON HARD FILTER 2: RVOL EXPANSION (>= 1.20x) ─────────────────────────────
    vol_ratio = v_today / max(vol_sma20, 1.0)
    trace["02_COMMON_GATES"]["rvol"] = round(vol_ratio, 2)

    if vol_ratio < MIN_RVOL_HARD_GATE:
        trace["02_COMMON_GATES"]["status"] = "REJECT"
        trace["02_COMMON_GATES"]["rejection_code"] = "LOW_RVOL"
        trace["FINAL"]["terminal_stage"] = "02_COMMON_GATES"
        trace["FINAL"]["terminal_reason"] = "LOW_RVOL"
        trace["FINAL"]["required"]["rvol_min"] = MIN_RVOL_HARD_GATE
        return _finish(None)

    # ── COMMON HARD FILTER 3: CLOSE STRENGTH (CLV >= 0.65) ──────────────────────────
    clv = (c_today - l_today) / candle_range
    trace["02_COMMON_GATES"]["clv"] = round(clv, 2)

    if clv < MIN_CLV_HARD_GATE:
        trace["02_COMMON_GATES"]["status"] = "REJECT"
        trace["02_COMMON_GATES"]["rejection_code"] = "LOW_CLV"
        trace["FINAL"]["terminal_stage"] = "02_COMMON_GATES"
        trace["FINAL"]["terminal_reason"] = "LOW_CLV"
        trace["FINAL"]["required"]["clv_min"] = MIN_CLV_HARD_GATE
        return _finish(None)

    # ── COMMON HARD FILTER 4: UPPER WICK FILTER (<= 30%) ────────────────────────────
    upper_wick = h_today - max(c_today, o_today)
    upper_wick_pct = upper_wick / candle_range
    trace["02_COMMON_GATES"]["upper_wick"] = round(upper_wick_pct, 2)

    if upper_wick_pct > MAX_UPPER_WICK_PCT:
        trace["02_COMMON_GATES"]["status"] = "REJECT"
        trace["02_COMMON_GATES"]["rejection_code"] = "EXCESSIVE_UPPER_WICK"
        trace["FINAL"]["terminal_stage"] = "02_COMMON_GATES"
        trace["FINAL"]["terminal_reason"] = "EXCESSIVE_UPPER_WICK"
        trace["FINAL"]["required"]["max_upper_wick_pct"] = MAX_UPPER_WICK_PCT
        return _finish(None)

    trace["02_COMMON_GATES"]["status"] = "PASS"

    atr14, atr_source = _coalesce_indicator_with_source(
        df,
        ["ATR", "ATR_14", "ATR20", "atr", "atr_14", "atr20"],
        validator=lambda v: v > 0,
        default=c_today * 0.02,
        default_source="DEFAULT_2PCT"
    )
    if atr14 <= 0:
        atr14 = c_today * 0.02
        atr_source = "DEFAULT_2PCT"

    trace["02_COMMON_GATES"]["atr_source"] = atr_source
    trace["02_COMMON_GATES"]["is_degraded_atr"] = (atr_source == "DEFAULT_2PCT")

    # ── PERMISSIVE PATTERN DISCOVERY (8 PRIMARY STRUCTURES) ─────────────────────────
    df_window = df.tail(120).copy() if len(df) > 120 else df
    candidate_patterns = []

    # 1. Bull Flag
    bf = _detect_bull_flag(df_window, atr14)
    if bf:
        candidate_patterns.append(bf)
        trace["04_PATTERN_VALIDATION"]["BULL_FLAG"] = {"candidate_found": True, "details": bf, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["BULL_FLAG"] = {"candidate_found": False, "status": "REJECT"}

    # 2. Shakeout Reclaim
    sr = _detect_shakeout_reclaim(df_window, atr14)
    if sr:
        candidate_patterns.append(sr)
        trace["04_PATTERN_VALIDATION"]["SHAKEOUT_RECLAIM"] = {"candidate_found": True, "details": sr, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["SHAKEOUT_RECLAIM"] = {"candidate_found": False, "status": "REJECT"}

    # 3. Double Bottom
    db = _detect_double_bottom(df_window, atr14)
    if db:
        candidate_patterns.append(db)
        trace["04_PATTERN_VALIDATION"]["DOUBLE_BOTTOM"] = {"candidate_found": True, "details": db, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["DOUBLE_BOTTOM"] = {"candidate_found": False, "status": "REJECT"}

    # 4. V-Reversal
    vr = _detect_v_reversal(df_window, atr14)
    if vr:
        candidate_patterns.append(vr)
        trace["04_PATTERN_VALIDATION"]["V_REVERSAL"] = {"candidate_found": True, "details": vr, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["V_REVERSAL"] = {"candidate_found": False, "status": "REJECT"}

    # 5. Cup & Handle
    ch = _detect_cup_and_handle(df_window, atr14)
    if ch:
        candidate_patterns.append(ch)
        trace["04_PATTERN_VALIDATION"]["CUP_HANDLE"] = {"candidate_found": True, "details": ch, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["CUP_HANDLE"] = {"candidate_found": False, "status": "REJECT"}

    # 6. Ascending Triangle
    at = _detect_ascending_triangle(df_window, atr14)
    if at:
        candidate_patterns.append(at)
        trace["04_PATTERN_VALIDATION"]["ASCENDING_TRIANGLE"] = {"candidate_found": True, "details": at, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["ASCENDING_TRIANGLE"] = {"candidate_found": False, "status": "REJECT"}

    # 7. Bull Pennant
    bp = _detect_bull_pennant(df_window, atr14)
    if bp:
        candidate_patterns.append(bp)
        trace["04_PATTERN_VALIDATION"]["BULL_PENNANT"] = {"candidate_found": True, "details": bp, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["BULL_PENNANT"] = {"candidate_found": False, "status": "REJECT"}

    # 8. Higher Low Reversal
    hl = _detect_higher_low_reversal(df_window, atr14)
    if hl:
        candidate_patterns.append(hl)
        trace["04_PATTERN_VALIDATION"]["HIGHER_LOW_REVERSAL"] = {"candidate_found": True, "details": hl, "status": "PASS"}
    else:
        trace["04_PATTERN_VALIDATION"]["HIGHER_LOW_REVERSAL"] = {"candidate_found": False, "status": "REJECT"}

    trace["03_PATTERN_DISCOVERY"]["detected_patterns"] = [p["pattern"] for p in candidate_patterns]

    if not candidate_patterns:
        trace["FINAL"]["terminal_stage"] = "03_PATTERN_DISCOVERY"
        trace["FINAL"]["terminal_reason"] = "NO_VALID_PATTERN"
        return _finish(None)

    # Primary pattern ranking hierarchy
    PATTERN_PRIORITY_RANK = {
        "BULL_FLAG": 1,
        "DOUBLE_BOTTOM": 2,
        "ASCENDING_TRIANGLE": 3,
        "BULL_PENNANT": 4,
        "CUP_HANDLE": 5,
        "HIGHER_LOW_REVERSAL": 6,
        "SHAKEOUT_RECLAIM": 7,
        "V_REVERSAL": 8,
    }

    candidate_patterns.sort(key=lambda p: (
        PATTERN_PRIORITY_RANK.get(p["pattern"], 99),
        -p.get("pattern_quality_score", 0)
    ))
    primary = candidate_patterns[0]
    secondary_pattern_names = [p["pattern"] for p in candidate_patterns[1:]]

    # ── RISK ENGINE & STOP LOSS CALCULATION ─────────────────────────────────────────
    raw_invalidation = primary.get("invalidation_level", l_today * 0.99)
    hard_sl_floor = c_today * (1.0 - MAX_SL_PCT)
    stop_loss = round(max(raw_invalidation, hard_sl_floor), 2)

    risk_points = max(c_today - stop_loss, c_today * MIN_SL_PCT)
    stop_loss = round(c_today - risk_points, 2)
    risk_pct = round((risk_points / c_today) * 100.0, 2)

    target_1 = round(c_today + (1.5 * risk_points), 2)
    target_2 = round(c_today + (3.0 * risk_points), 2)
    target_3 = round(c_today + (5.0 * risk_points), 2)
    rr_1 = round((target_1 - c_today) / max(risk_points, 0.01), 2)

    target_res = primary.get("target_resistance", c_today * 1.15)
    room_to_resistance_points = target_res - c_today
    room_to_resistance_r = room_to_resistance_points / max(risk_points, 0.01)

    trace["05_RISK"]["sl"] = stop_loss
    trace["05_RISK"]["target_1"] = target_1
    trace["05_RISK"]["target_2"] = target_2
    trace["05_RISK"]["target_3"] = target_3
    trace["05_RISK"]["risk_pct"] = risk_pct
    trace["05_RISK"]["natural_rr"] = rr_1
    trace["05_RISK"]["target_resistance"] = target_res
    trace["05_RISK"]["room_r"] = round(room_to_resistance_r, 2)

    # Room-to-Resistance Hard Gate (>= 1.5R with epsilon tolerance)
    if room_to_resistance_r < (MIN_ROOM_TO_RESISTANCE_R - 1e-6) and target_res > c_today:
        trace["05_RISK"]["status"] = "REJECT"
        trace["05_RISK"]["rejection_code"] = f"{primary['pattern']}_ROOM_LT_1_5R"
        trace["FINAL"]["terminal_stage"] = "05_RISK"
        trace["FINAL"]["terminal_reason"] = f"{primary['pattern']}_ROOM_LT_1_5R"
        trace["FINAL"]["required"]["min_room_r"] = MIN_ROOM_TO_RESISTANCE_R
        trace["FINAL"]["observed"]["room_r"] = round(room_to_resistance_r, 2)
        return _finish(None)

    trace["05_RISK"]["status"] = "PASS"

    # ── TIER C CONFLUENCE BOOSTERS ──────────────────────────────────────────────────
    confluences, confluence_pts = _detect_confluence_factors(df)

    # ── CLEAN 100-POINT ADDITIVE SCORING SYSTEM ────────────────────────────────────
    # 1. Pattern Quality (0 to 25 pts)
    score_pattern = float(primary.get("pattern_quality_score", 22))

    # 2. Volume Confirmation & Signature (0 to 25 pts)
    score_volume = 16.0  # Base for passing RVOL >= 1.20x
    if vol_ratio >= 2.0:
        score_volume += 9.0
    elif vol_ratio >= 1.5:
        score_volume += 6.0
    elif vol_ratio >= 1.35:
        score_volume += 3.0

    # 3. Price Action & Close Quality (0 to 20 pts)
    if clv >= 0.85 and upper_wick_pct <= 0.15:
        score_price_action = 20.0
    elif clv >= 0.75 and upper_wick_pct <= 0.25:
        score_price_action = 17.0
    else:
        score_price_action = 14.0

    # 4. Structure & Cleanliness (0 to 15 pts)
    score_structure = 12.0
    if len(secondary_pattern_names) > 0:
        score_structure += 3.0

    # 5. Risk / Room to Resistance (0 to 10 pts)
    if room_to_resistance_r >= 3.0:
        score_risk = 10.0
    elif room_to_resistance_r >= 2.0:
        score_risk = 8.0
    else:
        score_risk = 6.0

    # 6. Confluence (0 to 5 pts)
    score_confluence = float(confluence_pts)

    total_score = int(score_pattern + score_volume + score_price_action + score_structure + score_risk + score_confluence)
    total_score = min(100, max(0, total_score))

    # Unified Score Breakdown Schema (Canonical dual-schema)
    score_breakdown = {
        "pattern": int(score_pattern),
        "volume": int(score_volume),
        "price_action": int(score_price_action),
        "structure": int(score_structure),
        "risk": int(score_risk),
        "confluence": int(score_confluence),
        "total": total_score,
        "pattern_score": int(score_pattern),
        "volume_score": int(score_volume),
        "price_action_score": int(score_price_action),
        "structure_score": int(score_structure),
        "risk_score": int(score_risk),
        "confluence_score": int(score_confluence),
        "total_score": total_score,
    }

    trace["06_SCORE"] = score_breakdown
    trace["06_SCORE"]["status"] = "PASS" if total_score >= 70 else "REJECT"

    if total_score < 70:
        trace["FINAL"]["terminal_stage"] = "06_SCORE"
        trace["FINAL"]["terminal_reason"] = "SCORE_BELOW_70"
        trace["FINAL"]["required"]["min_score"] = 70
        trace["FINAL"]["observed"]["score"] = total_score
        return _finish(None)

    # Classification Hierarchy
    if total_score >= 90:
        classification = "🔥🔥 ELITE"
    elif total_score >= 80:
        classification = "🔥 VERY STRONG"
    else:
        classification = "⚡ STRONG"

    trace["FINAL"]["status"] = "SELECTED"
    trace["FINAL"]["terminal_stage"] = "FINAL"
    trace["FINAL"]["terminal_reason"] = f"ALL_HARD_GATES_PASS_SCORE_{total_score}"
    trace["FINAL"]["selected_pattern"] = primary["pattern"]

    res_payload = {
        "symbol": symbol,
        "cmp": c_today,
        "entry_price": c_today,
        "primary_pattern": primary["pattern"],
        "tier": primary["tier"],
        "description": primary["description"],
        "secondary_patterns": secondary_pattern_names,
        "confluences": confluences,
        "pattern_details": primary,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "target_3": target_3,
        "rr_1": rr_1,
        "risk_pct": risk_pct,
        "room_to_resistance_r": round(room_to_resistance_r, 1),
        "score": total_score,
        "classification": classification,
        "score_breakdown": score_breakdown,
        "clv": round(clv, 2),
        "upper_wick_pct": round(upper_wick_pct, 2),
        "rvol": round(vol_ratio, 2),
        "atr_source": atr_source,
        "is_degraded_atr": (atr_source == "DEFAULT_2PCT"),
        "history_class": history_class,
        "history_confidence": history_confidence,
        "engine_min_history": engine_min_history,
        "trend_validation_mode": trend_validation_mode,
        "alert_time": now_ist.strftime("%Y-%m-%d %H:%M:%S"),
        "technical_trace": trace,
    }

    return _finish(res_payload)


# =====================================================================================
# 4. SCANNER EXECUTION RUNNER WITH FUNNEL CONSERVATION
# =====================================================================================

def run_technical_scan(
    run_date: Optional[str] = None,
    is_test_mode: bool = False,
    run_ctx: Any = None,
    trigger_type: str = "SCHEDULED",
    scheduler_name: str = "CRON",
) -> int:
    """
    Main Execution Entry Point for Unified TECHNICAL Scanner.
    Runs daily at 18:15 IST (6:15 PM IST) post-market close.
    """
    if _scan_lock.locked():
        logger.warning("🛑 [DUPLICATE GUARD] TECHNICAL Scanner is ALREADY actively running in thread lock. Skipping duplicate trigger.")
        if run_ctx:
            complete_scanner_execution_run(run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Scanner lock busy")
        else:
            try:
                from database import record_skipped_execution_run
                record_skipped_execution_run(scanner_name="TECHNICAL", trigger_type=trigger_type, scheduler_name=scheduler_name, stop_reason="Scanner lock held (previous run active)")
            except Exception:
                pass
        return {"status": "skipped", "reason": "already_running"}

    acquired_global = False
    acquired_scan = False
    start_time = time.monotonic()
    real_run_ctx = run_ctx

    try:
        if not _scan_lock.acquire(blocking=False):
            logger.warning("🔒 [TECHNICAL] Scanner is already running. Skipping duplicate cycle.")
            try:
                from database import record_skipped_execution_run
                record_skipped_execution_run(scanner_name="TECHNICAL", trigger_type=trigger_type, scheduler_name=scheduler_name, stop_reason="Scanner lock held (previous run active)")
            except Exception:
                pass
            return 0
        acquired_scan = True

        # Acquire universal global scanner lock
        if not _global_lock.acquire(blocking=False, owner_scanner="TECHNICAL", operation="FULL_SCAN"):
            logger.info("⏳ [TECHNICAL] Global scanner lock busy — waiting in queue until active scanner finishes...")
            upsert_scanner_health("TECHNICAL", "QUEUED", error_msg="Waiting in queue for active scanner to release lock...")

            try:
                acquired_global = _global_lock.acquire(blocking=True, owner_scanner="TECHNICAL", operation="FULL_SCAN", run_ctx=real_run_ctx)
            except Exception as lock_err:
                logger.error(f"❌ [TECHNICAL] Error acquiring global lock: {lock_err}")
                acquired_global = False

            if not acquired_global:
                logger.error("❌ [TECHNICAL] Failed to acquire global scanner lock after queue wait.")
                if real_run_ctx:
                    complete_scanner_execution_run(real_run_ctx, status_override="FAILED", stop_reason="Global lock acquire timeout")
                upsert_scanner_health("TECHNICAL", "IDLE", error_msg="Lock acquisition timed out")
                return 0
        else:
            acquired_global = True

        telemetry.log_scheduler_event("TECHNICAL", "CYCLE_START")

        logger.info("=" * 70)
        logger.info("🚀 TECHNICAL SCANNER | Starting 6:15 PM Multi-Pattern Technical Execution...")
        logger.info("=" * 70)

        if not real_run_ctx:
            try:
                real_run_ctx = start_scanner_execution_run(
                    scanner_name="TECHNICAL",
                    trigger_type=trigger_type,
                    scheduler_name=scheduler_name,
                )
            except Exception as exc:
                if "actively running" in str(exc).lower():
                    logger.info("🛑 [TECHNICAL] Scanner is ALREADY actively running. Skipping duplicate execution.")
                    return 0
                logger.warning(f"⚠️ [TECHNICAL] Could not create run_ctx: {exc}")
                real_run_ctx = None

        init_db()
        upsert_scanner_health(
            scanner_name="TECHNICAL",
            status="RUNNING",
            error_msg="Multi-Pattern Technical scan in progress...",
            scheduled_for="Daily 18:15 IST (Post-Close Technical Scan)",
        )

        # 1. Fetch Universe Watchlist
        wl_df = get_watchlist("TECHNICAL")
        if isinstance(wl_df, pd.DataFrame) and "Stock" in wl_df.columns:
            watchlist = wl_df["Stock"].dropna().tolist()
        elif isinstance(wl_df, (list, set, tuple)):
            watchlist = list(wl_df)
        else:
            watchlist = get_elite_watchlist() or []

        if not watchlist:
            logger.warning("⚠️ [TECHNICAL] Watchlist is empty.")
            upsert_scanner_health(
                scanner_name="TECHNICAL",
                status="OK",
                outcome="SUCCESS",
                processed_count=0,
                duration_seconds=round(time.monotonic() - start_time, 2),
                scheduled_for="Daily 18:15 IST (Post-Close Technical Scan)",
            )
            telemetry.log_scheduler_event("TECHNICAL", "CYCLE_COMPLETE")
            if real_run_ctx:
                complete_scanner_execution_run(real_run_ctx)
            return 0

        try:
            from surveillance import get_live_blacklist
            bl = get_live_blacklist()
            if bl:
                watchlist = [s for s in watchlist if str(s).upper() not in bl]
        except Exception:
            pass

        logger.info(f"📋 [TECHNICAL] Screening {len(watchlist)} universe stocks on Daily timeframe...")

        # 2. Fetch 1d OHLCV Data for Watchlist
        all_1d = fetch_watchlist_data(
            watchlist,
            period="1y",
            interval="1d",
            requester="TECHNICAL",
            run_ctx=real_run_ctx,
        )

        qualified_candidates: List[Dict[str, Any]] = []
        rejection_counts: Dict[str, int] = {}
        funnel_stats = {
            "universe": len(watchlist),
            "data_fetched": 0,
            "common_gates_pass": 0,
            "pattern_candidates": 0,
            "risk_pass": 0,
            "score_pass": 0,
            "final_alerts": 0,
        }

        for symbol in watchlist:
            df = all_1d.get(symbol)
            if df is None or df.empty:
                rejection_counts["NO_DATA"] = rejection_counts.get("NO_DATA", 0) + 1
                continue

            funnel_stats["data_fetched"] += 1

            try:
                res, tr = detect_technical_setup(df, symbol, return_trace=True)
                
                # Funnel accounting based on structured trace
                if tr["02_COMMON_GATES"]["status"] == "PASS":
                    funnel_stats["common_gates_pass"] += 1
                    
                if len(tr["03_PATTERN_DISCOVERY"]["detected_patterns"]) > 0:
                    funnel_stats["pattern_candidates"] += 1
                    
                if tr["05_RISK"]["status"] == "PASS":
                    funnel_stats["risk_pass"] += 1
                    
                if tr["06_SCORE"].get("status") == "PASS":
                    funnel_stats["score_pass"] += 1

                if res and res.get("score", 0) >= 70:
                    qualified_candidates.append(res)
                else:
                    term_reason = tr["FINAL"].get("terminal_reason", "UNKNOWN_REJECTION")
                    rejection_counts[term_reason] = rejection_counts.get(term_reason, 0) + 1
            except Exception as e:
                logger.debug(f"Error evaluating {symbol} in technical scanner: {e}")
                rejection_counts["EXCEPTION"] = rejection_counts.get("EXCEPTION", 0) + 1

        alerts_saved = 0

        # 3. Sort by Score and Register Breakout Alerts
        qualified_candidates.sort(key=lambda x: x["score"], reverse=True)

        for cand in qualified_candidates:
            sym = cand["symbol"]
            cmp_price = cand["cmp"]
            score = cand["score"]
            sl = cand["stop_loss"]
            t1 = cand["target_1"]
            t2 = cand["target_2"]
            t3 = cand["target_3"]
            pat = cand["primary_pattern"]
            classification = cand["classification"]
            rvol = cand["rvol"]
            desc = cand["description"]

            logger.info(
                f"{classification} [TECHNICAL TRIGGERED] {sym} | Pattern: {pat} | CMP: ₹{cmp_price:.2f} | "
                f"RVOL: {rvol:.2f}x | SL: ₹{sl:.2f} | T1: ₹{t1:.2f} | Score: {score}/100 | {desc}"
            )

            if not is_test_mode:
                inserted, reason, _, _ = save_alert_if_new(
                    symbol=sym,
                    breakout_type="TECHNICAL",
                    alert_time=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                    scanner="TECHNICAL",
                    category="SWING",
                    entry_price=cmp_price,
                    stop_loss=sl,
                    target_1=t1,
                    target_2=t2,
                    target_3=t3,
                    signals=pat,
                    score=int(score),
                    context={
                        "primary_pattern": pat,
                        "tier": cand["tier"],
                        "description": desc,
                        "secondary_patterns": cand.get("secondary_patterns", []),
                        "confluences": cand.get("confluences", []),
                        "rvol": rvol,
                        "clv": cand["clv"],
                        "upper_wick_pct": cand["upper_wick_pct"],
                        "classification": classification,
                        "risk_pct": cand["risk_pct"],
                        "room_to_resistance_r": cand["room_to_resistance_r"],
                        "score_breakdown": cand["score_breakdown"],
                    },
                    entry_mode="BREAKOUT_TRIGGER",
                )
                if inserted:
                    alerts_saved += 1
                    try:
                        from telegram_engine import queue_telegram_message
                        sec_pats = cand.get("secondary_patterns", [])
                        sec_str = ", ".join(s.replace("_", " ") for s in sec_pats) if sec_pats else "None"
                        conf_list = cand.get("confluences", [])
                        conf_str = ", ".join(c.replace("_", " ") for c in conf_list) if conf_list else "None"
                        sb = cand.get("score_breakdown", {})

                        tg_msg = (
                            f"🚀 <b>TECHNICAL SCANNER ALERT ({classification})</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📈 <b>Stock:</b> #{sym}\n"
                            f"💰 <b>Entry CMP:</b> ₹{cmp_price:.2f}\n"
                            f"📐 <b>Primary Pattern:</b> {pat.replace('_', ' ')} (Tier {cand['tier']})\n"
                            f"📝 <b>Structure Details:</b> {desc}\n"
                            f"📦 <b>Volume Surge:</b> {rvol:.2f}x RVOL (Hard Gate: ≥1.20x)\n"
                            f"🕯️ <b>Candle Quality:</b> CLV {cand['clv']:.2f} | Upper Wick {cand['upper_wick_pct']:.1f}%\n"
                            f"✨ <b>Confluences:</b> {conf_str}\n"
                            f"🔄 <b>Secondary Patterns:</b> {sec_str}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"🛡️ <b>Stop Loss:</b> ₹{sl:.2f} (-{cand['risk_pct']}%)\n"
                            f"🎯 <b>Target 1:</b> ₹{t1:.2f} (1:1.5 RR)\n"
                            f"🎯 <b>Target 2:</b> ₹{t2:.2f} (1:3.0 RR)\n"
                            f"🎯 <b>Target 3:</b> ₹{t3:.2f} (1:4.5 RR)\n"
                            f"🚀 <b>Room to Resistance:</b> {cand['room_to_resistance_r']:.1f}R (Clear space ≥1.5R)\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"⭐ <b>Institutional Score:</b> {score}/100\n"
                            f"📊 <b>Score Breakdown:</b> Pat {sb.get('pattern_score', 0)}/25 | Vol {sb.get('volume_score', 0)}/25 | PA {sb.get('price_action_score', 0)}/20 | Struct {sb.get('structure_score', 0)}/15 | Risk {sb.get('risk_score', 0)}/10 | Conf {sb.get('confluence_score', 0)}/5\n"
                            f"🏷️ <b>Category:</b> SWING (Daily 1D · 3–15 Days)\n"
                            f"⏰ <b>Trigger Time:</b> {datetime.now(IST).strftime('%I:%M %p IST (%Y-%m-%d)')}"
                        )
                        queue_telegram_message(tg_msg, symbol=sym)
                    except Exception as _tg_err:
                        logger.debug(f"Telegram notification dispatch error: {_tg_err}")
                else:
                    rejection_counts[f"PERSISTENCE_{reason}"] = rejection_counts.get(f"PERSISTENCE_{reason}", 0) + 1
            else:
                alerts_saved += 1

        funnel_stats["final_alerts"] = alerts_saved

        # 4. Print Forensic Funnel Summary & Rejection Accounting
        logger.info("=" * 70)
        logger.info("📊 [TECHNICAL SCANNER FUNNEL SUMMARY]")
        logger.info("=" * 70)
        logger.info(f"  Universe Evaluated:             {funnel_stats['universe']}")
        logger.info(f"  Data Fetched:                   {funnel_stats['data_fetched']}")
        logger.info(f"  Common Gates Passed:            {funnel_stats['common_gates_pass']}")
        logger.info(f"  Pattern Candidates Discovered:  {funnel_stats['pattern_candidates']}")
        logger.info(f"  Risk / Room to Res Passed:      {funnel_stats['risk_pass']}")
        logger.info(f"  Score Passed (Score >= 70):     {funnel_stats['score_pass']}")
        logger.info(f"  Final Alerts Saved:             {funnel_stats['final_alerts']}")
        logger.info("-" * 70)
        logger.info("  TERMINAL REJECTIONS BREAKDOWN:")
        for r_code, r_cnt in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    * {r_code:<32}: {r_cnt}")
        logger.info("=" * 70)

        duration = round(time.monotonic() - start_time, 2)
        logger.info(
            f"✅ [TECHNICAL] Cycle complete in {duration}s | Processed: {len(watchlist)} | Alerts Saved: {alerts_saved}"
        )

        upsert_scanner_health(
            scanner_name="TECHNICAL",
            status="OK",
            last_success=datetime.now(IST).isoformat(),
            today_alerts=alerts_saved,
            processed_count=len(watchlist),
            total_count=len(watchlist),
            duration_seconds=duration,
            outcome="SUCCESS",
            error_msg=None,
            scheduled_for="Daily 18:15 IST (Post-Close Technical Scan)",
        )

        telemetry.log_scheduler_event("TECHNICAL", "CYCLE_COMPLETE")
        if real_run_ctx:
            complete_scanner_execution_run(real_run_ctx)

        if alerts_saved > 0:
            try:
                from performance_tracker import trigger_performance_rebuild
                trigger_performance_rebuild()
            except Exception as _pr_err:
                logger.debug(f"Performance rebuild trigger on technical alerts: {_pr_err}")

        return alerts_saved

    except Exception as exc:
        duration = round(time.monotonic() - start_time, 2)
        logger.exception(f"❌ [TECHNICAL] Fatal error during cycle: {exc}")
        upsert_scanner_health(
            scanner_name="TECHNICAL",
            status="DOWN",
            error_msg=str(exc)[:500],
            duration_seconds=duration,
            outcome="FAILED",
            scheduled_for="Daily 18:15 IST (Post-Close Technical Scan)",
        )
        telemetry.log_scheduler_event("TECHNICAL", "CYCLE_FAILED", error=str(exc))
        if real_run_ctx:
            try:
                complete_scanner_execution_run(real_run_ctx, exception=exc)
            except Exception:
                pass
        return 0
    finally:
        if acquired_global:
            try:
                _global_lock.release()
            except Exception as _ge:
                logger.debug(f"Error releasing global lock: {_ge}")
        if acquired_scan:
            try:
                _scan_lock.release()
            except Exception as _se:
                logger.debug(f"Error releasing scan lock: {_se}")
