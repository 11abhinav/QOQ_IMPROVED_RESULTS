# =====================================================================================
# app/multitf/consolidation.py
# MULTI_TF V3 — Adaptive 15m Consolidation & Structural Discovery Engine
#
# Responsibility: Identifies high-quality, mature compressions on the 15m chart.
#
# Upgrades in V3:
#   - Multi-window candidate evaluation (6, 8, 10, 12, 16, 20, 24, 30, 35 bars).
#   - Duration-aware dynamic ATR & percentage width limits.
#   - Best-base selector: selects the highest composite quality base across competing windows.
#   - Active compression vs dormancy detection (penalizes dead/frozen stocks, rewards coils).
#   - Rewards proximity to breakout ceiling (time in top 25-30% of base).
#   - Operates strictly on CLOSED 15m candles.
# =====================================================================================

import os
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import numpy as np

logger = logging.getLogger("multitf.consolidation")


@dataclass
class ConsolidationResult:
    """The complete structural definition of a valid 15m consolidation."""
    symbol: str
    is_valid: bool
    box_id: str = ""

    # Window info
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    bars_count: int = 0
    sessions_count: int = 1
    winning_window_bars: int = 0

    # Geometry
    box_high: float = 0.0
    box_low: float = 0.0
    box_mid: float = 0.0
    box_value_center: float = 0.0
    hard_high: float = 0.0
    hard_low: float = 0.0
    box_width_pct: float = 0.0
    box_width_atr: float = 0.0
    box_occupancy: float = 0.0
    dynamic_width_limit_atr: float = 2.20

    # Structure
    resistance_test_count: int = 0
    last_confirmed_pivot_level: float = 0.0
    last_confirmed_pivot_ts: Optional[pd.Timestamp] = None

    # [V3] 15m BASE QUALITY ENGINE (0-100) — 7 Component Breakdown
    score_maturity: int = 0             # A. Max 15: Duration × quality interaction
    score_tightness: int = 0            # B. Max 20: Range/ATR (tighter = better)
    score_resistance_quality: int = 0   # C. Max 20: Ceiling std dev (sharper = better)
    score_repeated_tests: int = 0       # D. Max 15: Distinct touches (2=10, 3=13, 4+=15)
    score_compression: int = 0          # E. Max 15: Late-ATR / Early-ATR contraction
    score_higher_lows: int = 0          # F. Max 10: Rising lows = buyers getting aggressive
    score_support_integrity: int = 0    # G. Max 5:  Few floor touches = buyers well above support

    # [V3.1] Decoupled Scoring Architecture
    base_quality_score: int = 0         # Pure structural geometry (0-100), price position independent
    proximity_score: int = 0            # Distance of price to resistance (0-100)
    pressure_score: int = 0             # Active price migration / momentum toward resistance (0-100)
    setup_score: int = 0                # Composite opportunity score (0–100)
    lifecycle_stage: str = "FORMING"    # FORMING, QUALIFIED, STRONG, PRE_BREAKOUT, PRESSURE
    soft_width_penalty: int = 0         # Penalty if width slightly exceeded continuous limit

    # Structural insights exposed for downstream engines
    has_higher_lows: bool = False
    higher_lows_strength: float = 0.0   # late_low_min - early_low_min (in price terms)
    compression_ratio: float = 1.0      # late_range_avg / early_range_avg (<1 = contracting)
    base_rating_label: str = ""         # EXCEPTIONAL / SUPER / GOOD / WATCH / REJECT
    rejection_depth_declining: bool = False
    time_near_resistance_pct: float = 0.0
    failed_break_count: int = 0
    supply_absorption_label: str = "MODERATE"
    is_dormant: bool = False
    rejection_reason: str = ""

    # Legacy field aliases (backwards compat with scanner/state code)
    score_resistance_def: int = 0
    score_tight_range: int = 0
    score_resistance_tests: int = 0
    score_compression_vcp: int = 0
    score_prior_bullish: int = 0
    score_clean_action: int = 0
    score_liquidity: int = 0
    score_duration: int = 0
    score_atr: int = 0
    score_occupancy: int = 0
    score_tests: int = 0
    score_hl: int = 0
    score_vol: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "box_id": self.box_id,
            "bars_count": self.bars_count,
            "sessions_count": self.sessions_count,
            "winning_window_bars": self.winning_window_bars,
            "box_high": round(self.box_high, 2),
            "box_low": round(self.box_low, 2),
            "box_width_pct": round(self.box_width_pct, 4),
            "box_width_atr": round(self.box_width_atr, 2),
            "box_occupancy": round(self.box_occupancy, 2),
            "dynamic_width_limit_atr": round(self.dynamic_width_limit_atr, 2),
            "resistance_test_count": self.resistance_test_count,
            "base_quality_score": self.base_quality_score,
            "proximity_score": self.proximity_score,
            "pressure_score": self.pressure_score,
            "setup_score": self.setup_score,
            "lifecycle_stage": self.lifecycle_stage,
            "base_rating_label": self.base_rating_label,
            "has_higher_lows": self.has_higher_lows,
            "compression_ratio": round(self.compression_ratio, 3),
            "rejection_depth_declining": self.rejection_depth_declining,
            "time_near_resistance_pct": round(self.time_near_resistance_pct, 3),
            "failed_break_count": self.failed_break_count,
            "supply_absorption_label": self.supply_absorption_label,
            "is_dormant": self.is_dormant,
            "rejection_reason": self.rejection_reason,
            "score_breakdown": {
                "maturity": self.score_maturity,
                "tightness": self.score_tightness,
                "resistance_quality": self.score_resistance_quality,
                "repeated_tests": self.score_repeated_tests,
                "compression": self.score_compression,
                "higher_lows": self.score_higher_lows,
                "support_integrity": self.score_support_integrity,
            }
        }


def get_duration_width_limits(bars_count: int, config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns (max_atr, max_pct) allowed for a specific base duration using a smooth continuous curve.
    W_max(N) = 1.80 + 0.35 * sqrt(max(0, N - 5))
    P_max(N) = 0.030 + 0.006 * sqrt(max(0, N - 5))
    Eliminates discrete bucket jumps (e.g. 8 vs 9 bars, 14 vs 15 bars).
    """
    duration_limits = config.get("DURATION_ATR_WIDTH_LIMITS")
    if duration_limits and not config.get("USE_CONTINUOUS_WIDTH_SCALING", True):
        for limit in sorted(duration_limits, key=lambda x: x.get("max_bars", 999)):
            if bars_count <= limit.get("max_bars", 999):
                return float(limit.get("max_atr", 3.6)), float(limit.get("max_pct", 0.065))
    
    n_offset = max(0, bars_count - 5)
    max_atr = round(1.80 + 0.35 * float(np.sqrt(n_offset)), 2)
    max_pct = round(0.030 + 0.006 * float(np.sqrt(n_offset)), 4)
    return max_atr, max_pct


@dataclass(frozen=True)
class Prepared15mContext:
    """
    [RULE 67 CHANGE-RATIONALE: PREPARED_15M_CONTEXT_V1.1]
    Immutable, zero-copy, pre-extracted NumPy arrays and session metadata for a stock.
    Eliminates repeated DataFrame slicing, datetime conversions, and pandas allocations
    across 9 candidate window evaluations. Precomputes time-of-day baseline volume in O(1).
    """
    symbol: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    session_dates: np.ndarray
    minutes_of_session: np.ndarray
    gap_prefix: np.ndarray
    atr_15m: float
    volume_baseline: float
    tod_baseline: float
    recent_high: float
    recent_low: float
    timestamps: Any


def prepare_15m_context(
    df: Optional[pd.DataFrame],
    atr_15m: float,
    config: Dict[str, Any],
    symbol: Optional[str] = None
) -> Optional[Prepared15mContext]:
    """
    Builds an immutable Prepared15mContext once per stock.
    Fast single-pass C-level DatetimeIndex resolution and pre-extracted contiguous float64 arrays.
    """
    sym = symbol or (df.attrs.get("symbol") if df is not None else None) or "?"
    min_bars = config.get("MIN_CONSOLIDATION_BARS", 6)
    if df is None or not hasattr(df, "empty") or df.empty or len(df) < min_bars or atr_15m <= 0:
        return None

    try:
        high = np.ascontiguousarray(df["High"].to_numpy(dtype=np.float64))
        low = np.ascontiguousarray(df["Low"].to_numpy(dtype=np.float64))
        open_arr = np.ascontiguousarray(df["Open"].to_numpy(dtype=np.float64))
        close = np.ascontiguousarray(df["Close"].to_numpy(dtype=np.float64))
        if "Volume" in df.columns:
            volume = np.ascontiguousarray(df["Volume"].to_numpy(dtype=np.float64))
        else:
            volume = np.ones(len(df), dtype=np.float64)

        # [RULE 67 CHANGE-RATIONALE: FAST_DATETIME_INDEX_V1.0]
        # Resolve DatetimeIndex ONCE without repeated flexible string regex parsing per symbol.
        if isinstance(df.index, pd.DatetimeIndex):
            dt_idx = df.index
        elif "Datetime" in df.columns:
            col_dt = df["Datetime"]
            if pd.api.types.is_datetime64_any_dtype(col_dt):
                dt_idx = pd.DatetimeIndex(col_dt)
            else:
                dt_idx = pd.DatetimeIndex(pd.to_datetime(col_dt.values, errors='coerce'))
        elif "Date" in df.columns:
            col_d = df["Date"]
            if pd.api.types.is_datetime64_any_dtype(col_d):
                dt_idx = pd.DatetimeIndex(col_d)
            else:
                dt_idx = pd.DatetimeIndex(pd.to_datetime(col_d.values, errors='coerce'))
        else:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                dt_idx = pd.DatetimeIndex(df.index)
            else:
                dt_idx = pd.DatetimeIndex(pd.to_datetime(df.index.values, errors='coerce'))

        dates_arr = dt_idx.date
        minutes_arr = (dt_idx.hour * 60 + dt_idx.minute).values
        timestamps = dt_idx

        # Cumulative overnight gap prefix array for O(1) gap detection
        total_len = len(close)
        gap_pct_limit = config.get("GAP_PCT_THRESHOLD", 0.020)
        gap_atr_limit = config.get("GAP_ATR_MULT", 2.0) * atr_15m
        is_gap = np.zeros(total_len, dtype=np.int32)
        for i in range(1, total_len):
            if dates_arr[i] != dates_arr[i - 1]:
                open_px = open_arr[i]
                prev_close_px = close[i - 1]
                gap_abs = abs(open_px - prev_close_px)
                gap_pct = gap_abs / prev_close_px if prev_close_px > 0 else 0.0
                if gap_pct > gap_pct_limit or gap_abs > gap_atr_limit:
                    is_gap[i] = 1
        gap_prefix = np.cumsum(is_gap)

        vol_baseline = float(np.median(volume[-40:])) if len(volume) >= 40 else float(np.median(volume)) if len(volume) > 0 else 1.0

        # [RULE 67 CHANGE-RATIONALE: PRECOMPUTE_TOD_BASELINE_V1.0]
        # Precompute time-of-day baseline ONCE per stock to avoid 3,600+ repeated median computations in window loops.
        target_minute = int(minutes_arr[-1]) if len(minutes_arr) > 0 else 0
        time_mask = (np.abs(minutes_arr - target_minute) <= 30)
        same_time_vols = volume[time_mask]
        if len(same_time_vols) >= 5:
            tod_baseline = float(np.median(same_time_vols))
        else:
            tod_baseline = vol_baseline
        if tod_baseline <= 0:
            tod_baseline = 1.0

        recent_slice_len = min(35, total_len)
        rec_high = float(np.max(high[-recent_slice_len:]))
        rec_low = float(np.min(low[-recent_slice_len:]))

        return Prepared15mContext(
            symbol=sym,
            open=open_arr,
            high=high,
            low=low,
            close=close,
            volume=volume,
            session_dates=dates_arr,
            minutes_of_session=minutes_arr,
            gap_prefix=gap_prefix,
            atr_15m=atr_15m,
            volume_baseline=vol_baseline,
            tod_baseline=tod_baseline,
            recent_high=rec_high,
            recent_low=rec_low,
            timestamps=timestamps
        )
    except Exception as prep_exc:
        logger.warning("[%s] prepare_15m_context failed: %s", sym, prep_exc)
        return None


def _compute_structure_np(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_15m: float,
    res: ConsolidationResult,
    config: Dict[str, Any]
):
    """NumPy-native structure and resistance multi-touch detection."""
    tol_pct = config.get("RESISTANCE_TEST_TOL_PCT", 0.0015) * res.box_high
    tol_atr = config.get("RESISTANCE_TEST_TOL_ATR", 0.08) * atr_15m
    tol = max(tol_pct, tol_atr)
    test_zone_low = res.box_high - tol

    tests = 0
    in_test = False
    rejections = []
    current_test_idx = None
    false_breaks = 0
    n = len(highs)

    for i in range(n):
        h = highs[i]
        c = closes[i]
        l = lows[i]

        if h > res.box_high and c <= res.box_high:
            false_breaks += 1

        if h >= test_zone_low:
            if not in_test:
                tests += 1
                in_test = True
                current_test_idx = i
        else:
            if in_test and current_test_idx is not None:
                pullback_depth = res.box_high - l
                rejections.append(pullback_depth)
                in_test = False

    res.resistance_test_count = max(tests, 1)
    res.failed_break_count = false_breaks

    if len(rejections) >= 2:
        res.rejection_depth_declining = (rejections[-1] <= rejections[0] * 0.90)

    upper_threshold = res.box_low + 0.65 * (res.box_high - res.box_low)
    res.time_near_resistance_pct = float(np.mean(closes >= upper_threshold)) if n > 0 else 0.0

    if res.resistance_test_count >= 3 and res.has_higher_lows and res.rejection_depth_declining:
        res.supply_absorption_label = "EXCELLENT"
    elif res.resistance_test_count >= 2 and (res.has_higher_lows or res.time_near_resistance_pct >= 0.50):
        res.supply_absorption_label = "STRONG"
    elif res.resistance_test_count >= 2:
        res.supply_absorption_label = "MODERATE"
    else:
        res.supply_absorption_label = "EARLY"


def _evaluate_dormancy_np(
    closes: np.ndarray,
    volumes: np.ndarray,
    ctx: Prepared15mContext,
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    [RULE 67 CHANGE-RATIONALE: O1_DORMANCY_EVALUATION_V1.0]
    Evaluates dormancy in O(1) using precomputed time-of-day baseline from Prepared15mContext.
    Eliminates 3,600+ repeated boolean mask array allocations and median calculations per scan.
    """
    if len(volumes) == 0:
        return False, 1.0
    mean_base_vol = float(np.mean(volumes))
    if mean_base_vol == 0:
        return True, 0.0

    # 1. Turnover Sanity Check
    mean_turnover = float(np.mean(closes * volumes))
    if mean_turnover >= config.get("LIQUIDITY_MIN_TURNOVER_RS", 50000.0):
        return False, 1.0

    # 2. Time-of-Day Aware Baseline (Precomputed in Prepared15mContext)
    baseline_vol = ctx.tod_baseline
    if baseline_vol <= 0:
        return False, 1.0

    vol_ratio = mean_base_vol / baseline_vol
    min_vol_ratio = config.get("DORMANCY_MIN_VOL_RATIO", 0.15)

    if mean_turnover >= 25000.0 and vol_ratio >= 0.08:
        return False, round(vol_ratio, 3)

    is_dormant = (vol_ratio < min_vol_ratio)
    return is_dormant, round(vol_ratio, 3)


def _compute_scores_np(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    volumes: np.ndarray,
    atr_15m: float,
    res: ConsolidationResult,
    config: Dict[str, Any],
    max_atr_limit: Optional[float] = None
):
    """Vectorized scoring across orthogonal dimensions (Base Quality, Proximity, Pressure)."""
    n = len(highs)
    score = 0

    # A. MATURITY (15 pts)
    if n >= 16: s_mat = 15
    elif n >= 12: s_mat = 14
    elif n >= 10: s_mat = 13
    elif n >= 8: s_mat = 12
    elif n >= 6: s_mat = 9
    elif n >= 4: s_mat = 6
    else: s_mat = 0

    if res.box_width_atr > 1.25 and s_mat > 10:
        s_mat = 10
    res.score_maturity = min(s_mat, config.get("SCORE_MATURITY_MAX", 15))
    score += res.score_maturity

    # B. TIGHTNESS (20 pts)
    w = res.box_width_atr
    eff_max_atr = max_atr_limit or config.get("MAX_BOX_WIDTH_ATR", 2.20)
    if w <= 0.75: s_tight = 20
    elif w <= 1.00: s_tight = 17
    elif w <= 1.25: s_tight = 13
    elif w <= 1.50: s_tight = 8
    elif w <= eff_max_atr * 0.75: s_tight = 5
    elif w <= eff_max_atr: s_tight = 2
    else: s_tight = 0
    res.score_tightness = min(s_tight, config.get("SCORE_TIGHTNESS_MAX", 20))
    score += res.score_tightness

    # C. RESISTANCE QUALITY (20 pts)
    top_highs = highs[highs >= (res.box_high - (0.15 * atr_15m))]
    std_top = float(np.std(top_highs, ddof=1)) if len(top_highs) >= 2 else 0.0
    std_pct = std_top / res.box_high if res.box_high > 0 else 0.0
    if len(top_highs) >= 2 and std_pct <= 0.0010: s_rq = 20
    elif len(top_highs) >= 2 and std_pct <= 0.0020: s_rq = 16
    elif len(top_highs) >= 2 and std_pct <= 0.0035: s_rq = 12
    elif len(top_highs) >= 1: s_rq = 8
    else: s_rq = 4
    res.score_resistance_quality = min(s_rq, config.get("SCORE_RESISTANCE_QUALITY_MAX", 20))
    score += res.score_resistance_quality

    # D. REPEATED TESTS (15 pts)
    t = res.resistance_test_count
    if t >= 4: s_tests = 15
    elif t == 3: s_tests = 13
    elif t == 2: s_tests = 10
    elif t == 1: s_tests = 3
    else: s_tests = 0
    res.score_repeated_tests = min(s_tests, config.get("SCORE_REPEATED_TESTS_MAX", 15))
    score += res.score_repeated_tests

    # E. COMPRESSION / VCP (15 pts)
    s_comp = 8
    if n >= 4:
        half = n // 2
        early_ranges = highs[:half] - lows[:half]
        late_ranges = highs[half:] - lows[half:]
        mean_early = float(np.mean(early_ranges)) if len(early_ranges) > 0 else 0.0
        mean_late = float(np.mean(late_ranges)) if len(late_ranges) > 0 else 0.0
        if mean_early > 0:
            comp_ratio = mean_late / mean_early
            res.compression_ratio = round(comp_ratio, 3)
            if comp_ratio <= 0.60: s_comp = 15
            elif comp_ratio <= 0.75: s_comp = 12
            elif comp_ratio <= 0.90: s_comp = 8
            elif comp_ratio <= 1.00: s_comp = 4
            else: s_comp = 0
    res.score_compression = min(s_comp, config.get("SCORE_COMPRESSION_MAX", 15))
    score += res.score_compression

    # F. HIGHER LOWS (10 pts)
    s_hl = 0
    if n >= 4:
        half = n // 2
        early_low_min = float(np.min(lows[:half]))
        late_low_min = float(np.min(lows[half:]))
        hl_rise = late_low_min - early_low_min
        res.higher_lows_strength = round(hl_rise, 2)
        min_strong_rise = config.get("HIGHER_LOWS_MIN_RISE_ATR", 0.15) * atr_15m
        if hl_rise >= min_strong_rise:
            s_hl = 10
            res.has_higher_lows = True
        elif hl_rise >= 0:
            s_hl = 7
            res.has_higher_lows = True
        elif hl_rise >= -(0.10 * atr_15m):
            s_hl = 4
        else:
            s_hl = 0
    res.score_higher_lows = min(s_hl, config.get("SCORE_HIGHER_LOWS_MAX", 10))
    score += res.score_higher_lows

    # G. SUPPORT INTEGRITY (5 pts)
    s_si = 3
    if n >= 4 and atr_15m > 0:
        worst_excursion_below = max(float(res.box_low - np.min(lows)), 0.0)
        worst_excursion_atr = worst_excursion_below / atr_15m
        closed_below_floor = bool(np.any(closes < res.box_low))
        if closed_below_floor: s_si = 1
        elif worst_excursion_atr <= 0.08: s_si = 5
        elif worst_excursion_atr <= 0.20: s_si = 4
        else: s_si = 2
    res.score_support_integrity = min(s_si, config.get("SCORE_SUPPORT_INTEGRITY_MAX", 5))
    score += res.score_support_integrity

    # H. PENALTIES
    if getattr(res, "soft_width_penalty", 0) > 0:
        score -= res.soft_width_penalty
    if getattr(res, "is_dormant", False):
        score -= config.get("DORMANCY_PENALTY", 15)

    # 1. Base Quality Score
    res.base_quality_score = int(np.clip(score, 0, 100))

    # 2. Proximity Score
    if n > 0:
        last_close = float(closes[-1])
        box_height = max(res.box_high - res.box_low, 0.0001)
        rel_pos = (last_close - res.box_low) / box_height
        if rel_pos >= 0.85: res.proximity_score = 100
        elif rel_pos >= 0.70: res.proximity_score = 80
        elif rel_pos >= 0.55: res.proximity_score = 65
        elif rel_pos >= 0.40: res.proximity_score = 50
        elif rel_pos >= 0.25: res.proximity_score = 30
        else: res.proximity_score = 15
    else:
        res.proximity_score = 50

    # 3. Pressure Score
    s_press = 0
    if res.has_higher_lows:
        s_press += 35
    if n >= 3:
        recent_c = closes[-3:]
        recent_mid = (highs[-3:] + lows[-3:]) / 2.0
        up_bars = int(np.sum(recent_c >= recent_mid))
        s_press += int(35 * (up_bars / 3))
    else:
        s_press += 20

    if len(volumes) > 0:
        green_mask = (closes >= opens)
        green_vol = float(np.sum(volumes[green_mask]))
        tot_vol = float(np.sum(volumes))
        if tot_vol > 0 and (green_vol / tot_vol) >= 0.55:
            s_press += 30
        elif tot_vol > 0 and (green_vol / tot_vol) >= 0.45:
            s_press += 15
    else:
        s_press += 15
    res.pressure_score = min(s_press, 100)

    # 4. Opportunity Composite Score
    res.setup_score = int(np.clip(
        0.50 * res.base_quality_score +
        0.30 * res.proximity_score +
        0.20 * res.pressure_score,
        0, 100
    ))

    # 5. Lifecycle Stage
    if res.base_quality_score >= 65 and res.proximity_score >= 80 and res.pressure_score >= 60:
        res.lifecycle_stage = "PRESSURE"
    elif res.base_quality_score >= 65 and res.proximity_score >= 75:
        res.lifecycle_stage = "PRE_BREAKOUT"
    elif res.base_quality_score >= 75:
        res.lifecycle_stage = "STRONG"
    elif res.base_quality_score >= 60:
        res.lifecycle_stage = "QUALIFIED"
    else:
        res.lifecycle_stage = "FORMING"

    # Populate legacy aliases
    res.score_resistance_def   = res.score_resistance_quality
    res.score_tight_range      = res.score_tightness
    res.score_resistance_tests = res.score_repeated_tests
    res.score_compression_vcp  = res.score_compression
    res.score_hl               = res.score_higher_lows

    if res.base_quality_score >= 85:
        res.base_rating_label = "EXCEPTIONAL"
    elif res.base_quality_score >= 75:
        res.base_rating_label = "SUPER"
    elif res.base_quality_score >= 65:
        res.base_rating_label = "GOOD"
    elif res.base_quality_score >= 50:
        res.base_rating_label = "WATCH"
    else:
        res.base_rating_label = "REJECT"


def detect_15m_consolidation_from_context(
    ctx: Prepared15mContext,
    ist_now: datetime,
    config: Dict[str, Any]
) -> ConsolidationResult:
    """
    [RULE 67 CHANGE-RATIONALE: DETECT_FROM_CONTEXT_V1.0]
    Executes candidate window evaluation on pre-extracted zero-copy NumPy slices.
    Preserves all adaptive windows [6, 8, 10, 12, 16, 20, 24, 30, 35] without hard 35-bar veto.
    """
    candidate_bars = config.get("CANDIDATE_WINDOW_BARS", [6, 8, 10, 12, 16, 20, 24, 30, 35])
    min_bars = config.get("MIN_CONSOLIDATION_BARS", 6)
    q_high = config.get("BOX_HIGH_QUANTILE", 0.90)
    q_low = config.get("BOX_LOW_QUANTILE", 0.10)
    min_occupancy = config.get("MIN_BOX_OCCUPANCY", 0.60)
    min_base_score = config.get("MONITOR_SETUP_SCORE", 50)
    min_tests = config.get("MIN_RESISTANCE_TESTS", 1)

    total_len = len(ctx.close)
    if total_len < min_bars:
        return ConsolidationResult(symbol=ctx.symbol, is_valid=False, rejection_reason="INSUFFICIENT_DATA_OR_ATR_ZERO")

    evaluated_bases: List[ConsolidationResult] = []
    best_rejection_reason = "NO_CANDIDATE_QUALIFIED"
    seen_bar_counts = set()

    for k in sorted(candidate_bars):
        if k > total_len or k < min_bars:
            continue
        if k in seen_bar_counts:
            continue
        seen_bar_counts.add(k)

        window_start_idx = total_len - k
        # O(1) Overnight Gap Check via pre-computed prefix array
        if (ctx.gap_prefix[total_len - 1] - ctx.gap_prefix[window_start_idx]) > 0:
            continue

        cand_res = ConsolidationResult(symbol=ctx.symbol, is_valid=False)
        cand_res.bars_count = k

        # Zero-copy NumPy slices
        w_high = ctx.high[window_start_idx:]
        w_low = ctx.low[window_start_idx:]
        w_close = ctx.close[window_start_idx:]
        w_open = ctx.open[window_start_idx:]
        w_volume = ctx.volume[window_start_idx:]
        w_dates = ctx.session_dates[window_start_idx:]

        cand_res.sessions_count = 1 if w_dates[0] == w_dates[-1] else int(len(np.unique(w_dates)))
        cand_res.start_ts = ctx.timestamps[window_start_idx]
        cand_res.end_ts = ctx.timestamps[-1]

        # 1. Base Geometry
        cand_res.hard_high = float(np.max(w_high))
        cand_res.hard_low = float(np.min(w_low))

        # Fast Necessary-Condition Gate: if the raw extreme range already exceeds sanity cap,
        # quantile clipping can never shrink it enough to qualify. Skip percentiles and scoring.
        max_atr_limit, max_pct_limit = get_duration_width_limits(k, config)
        sanity_atr_cap = min(max_atr_limit * 1.30, 4.50)
        hard_width_atr = (cand_res.hard_high - cand_res.hard_low) / (ctx.atr_15m if ctx.atr_15m > 0 else 1.0)
        if hard_width_atr > sanity_atr_cap * 1.50:
            best_rejection_reason = f"HARD_WIDTH_ATR_EXCEEDED ({hard_width_atr:.2f} > {sanity_atr_cap * 1.50:.2f})"
            continue

        if k >= 6:
            cand_res.box_high = float(np.percentile(w_high, q_high * 100))
            cand_res.box_low = float(np.percentile(w_low, q_low * 100))
        else:
            cand_res.box_high = cand_res.hard_high
            cand_res.box_low = cand_res.hard_low

        med_close = float(np.median(w_close))
        if cand_res.box_high < med_close:
            cand_res.box_high = cand_res.hard_high
        if cand_res.box_low > med_close:
            cand_res.box_low = cand_res.hard_low

        cand_res.box_mid = (cand_res.box_high + cand_res.box_low) / 2.0
        cand_res.box_value_center = med_close

        eff_mid = max(cand_res.box_mid, 1.0)
        cand_res.box_width_pct = (cand_res.box_high - cand_res.box_low) / eff_mid
        cand_res.box_width_atr = (cand_res.box_high - cand_res.box_low) / (ctx.atr_15m if ctx.atr_15m > 0 else 1.0)

        tol = 0.10 * ctx.atr_15m
        inside_mask = (w_close >= (cand_res.box_low - tol)) & (w_close <= (cand_res.box_high + tol))
        cand_res.box_occupancy = float(np.mean(inside_mask)) if k > 0 else 1.0

        # 2. Continuous Dynamic Range Width & Soft Constraint Check
        max_atr_limit, max_pct_limit = get_duration_width_limits(k, config)
        cand_res.dynamic_width_limit_atr = max_atr_limit

        sanity_atr_cap = min(max_atr_limit * 1.30, 4.50)
        sanity_pct_cap = min(max_pct_limit * 1.30, 0.080)

        if cand_res.box_width_atr > sanity_atr_cap:
            best_rejection_reason = f"WIDTH_ATR_EXCEEDED ({cand_res.box_width_atr:.2f} > {sanity_atr_cap:.2f})"
            continue
        if cand_res.box_width_pct > sanity_pct_cap:
            best_rejection_reason = f"WIDTH_PCT_EXCEEDED ({cand_res.box_width_pct:.3f} > {sanity_pct_cap:.3f})"
            continue

        if cand_res.box_width_atr > max_atr_limit:
            excess_ratio = (cand_res.box_width_atr - max_atr_limit) / max(0.01, sanity_atr_cap - max_atr_limit)
            cand_res.soft_width_penalty = int(round(15.0 * excess_ratio))
        else:
            cand_res.soft_width_penalty = 0

        # 3. Minimum Occupancy
        if cand_res.box_occupancy < min_occupancy:
            best_rejection_reason = f"OCCUPANCY_TOO_LOW ({cand_res.box_occupancy:.2f} < {min_occupancy:.2f})"
            continue

        # 4. Box ID
        date_val = w_dates[-1]
        date_str = date_val.strftime("%Y%m%d") if hasattr(date_val, "strftime") else str(date_val)
        h_str = f"{cand_res.box_high:.2f}"
        l_str = f"{cand_res.box_low:.2f}"
        raw_box = f"{cand_res.symbol}_{date_str}_{h_str}_{l_str}"
        cand_res.box_id = hashlib.md5(raw_box.encode()).hexdigest()[:10]

        # 5. Structure & Resistance Multi-Touch Detection
        _compute_structure_np(w_high, w_low, w_close, ctx.atr_15m, cand_res, config)

        # 6. Dormancy Detection
        is_dormant, vol_ratio = _evaluate_dormancy_np(w_close, w_volume, ctx, config)
        cand_res.is_dormant = is_dormant

        # 7. Decoupled Structural Scoring
        _compute_scores_np(w_high, w_low, w_close, w_open, w_volume, ctx.atr_15m, cand_res, config, max_atr_limit)

        # 8. Qualification Check
        if cand_res.base_quality_score >= min_base_score and cand_res.resistance_test_count >= min_tests:
            cand_res.is_valid = True
            cand_res.winning_window_bars = k
            evaluated_bases.append(cand_res)
        else:
            if cand_res.base_quality_score < min_base_score:
                best_rejection_reason = f"SCORE_TOO_LOW ({cand_res.base_quality_score} < {min_base_score})"
            elif cand_res.resistance_test_count < min_tests:
                best_rejection_reason = f"TESTS_TOO_LOW ({cand_res.resistance_test_count} < {min_tests})"

    if not evaluated_bases:
        return ConsolidationResult(symbol=ctx.symbol, is_valid=False, rejection_reason=best_rejection_reason)

    # Deterministic Best-Base Selector:
    evaluated_bases.sort(
        key=lambda r: (
            r.base_quality_score,                 # 1. Primary: overall structural base quality
            r.score_resistance_quality,          # 2. Resistance ceiling precision
            r.score_compression,                 # 3. Active volatility contraction
            r.proximity_score,                   # 4. Price proximity to resistance
            r.bars_count                         # 5. Duration maturity
        ),
        reverse=True
    )
    winning_base = evaluated_bases[0]
    winning_base.is_valid = True
    return winning_base


def _detect_15m_consolidation_legacy(
    df_15m_closed: Optional[pd.DataFrame],
    atr_15m: float,
    ist_now: datetime,
    config: Dict[str, Any],
    symbol: Optional[str] = None
) -> ConsolidationResult:
    """Legacy Pandas-based detector preserved for differential benchmarking and zero-risk rollback."""
    sym = symbol or (df_15m_closed.attrs.get("symbol") if df_15m_closed is not None else None) or "?"
    min_bars = config.get("MIN_CONSOLIDATION_BARS", 6)
    if df_15m_closed is None or len(df_15m_closed) < min_bars or atr_15m <= 0:
        return ConsolidationResult(symbol=sym, is_valid=False, rejection_reason="INSUFFICIENT_DATA_OR_ATR_ZERO")

    try:
        candidate_windows = _generate_candidate_windows(df_15m_closed, atr_15m, config)
        if not candidate_windows:
            return ConsolidationResult(symbol=sym, is_valid=False, rejection_reason="NO_VALID_WINDOW_GAP_LIMIT")

        evaluated_bases: List[ConsolidationResult] = []
        best_rejection_reason = "NO_CANDIDATE_QUALIFIED"

        for window_df, sessions_count in candidate_windows:
            cand_res = ConsolidationResult(symbol=sym, is_valid=False)
            cand_res.bars_count = len(window_df)
            cand_res.sessions_count = sessions_count

            _build_geometry(window_df, atr_15m, cand_res, config)

            max_atr_limit, max_pct_limit = get_duration_width_limits(cand_res.bars_count, config)
            cand_res.dynamic_width_limit_atr = max_atr_limit

            sanity_atr_cap = min(max_atr_limit * 1.30, 4.50)
            sanity_pct_cap = min(max_pct_limit * 1.30, 0.080)

            if cand_res.box_width_atr > sanity_atr_cap:
                best_rejection_reason = f"WIDTH_ATR_EXCEEDED ({cand_res.box_width_atr:.2f} > {sanity_atr_cap:.2f})"
                continue
            if cand_res.box_width_pct > sanity_pct_cap:
                best_rejection_reason = f"WIDTH_PCT_EXCEEDED ({cand_res.box_width_pct:.3f} > {sanity_pct_cap:.3f})"
                continue

            if cand_res.box_width_atr > max_atr_limit:
                excess_ratio = (cand_res.box_width_atr - max_atr_limit) / max(0.01, sanity_atr_cap - max_atr_limit)
                cand_res.soft_width_penalty = int(round(15.0 * excess_ratio))
            else:
                cand_res.soft_width_penalty = 0

            min_occupancy = config.get("MIN_BOX_OCCUPANCY", 0.60)
            if cand_res.box_occupancy < min_occupancy:
                best_rejection_reason = f"OCCUPANCY_TOO_LOW ({cand_res.box_occupancy:.2f} < {min_occupancy:.2f})"
                continue

            _generate_box_id(window_df, cand_res)
            _compute_structure(window_df, atr_15m, cand_res, config)

            is_dormant, vol_ratio = _evaluate_dormancy(window_df, df_15m_closed, config)
            cand_res.is_dormant = is_dormant

            _compute_scores(window_df, df_15m_closed, atr_15m, cand_res, config, max_atr_limit=max_atr_limit)

            min_base_score = config.get("MONITOR_SETUP_SCORE", 50)
            min_tests = config.get("MIN_RESISTANCE_TESTS", 1)

            if cand_res.base_quality_score >= min_base_score and cand_res.resistance_test_count >= min_tests:
                cand_res.is_valid = True
                cand_res.winning_window_bars = cand_res.bars_count
                evaluated_bases.append(cand_res)
            else:
                if cand_res.base_quality_score < min_base_score:
                    best_rejection_reason = f"SCORE_TOO_LOW ({cand_res.base_quality_score} < {min_base_score})"
                elif cand_res.resistance_test_count < min_tests:
                    best_rejection_reason = f"TESTS_TOO_LOW ({cand_res.resistance_test_count} < {min_tests})"

        if not evaluated_bases:
            return ConsolidationResult(symbol=sym, is_valid=False, rejection_reason=best_rejection_reason)

        evaluated_bases.sort(
            key=lambda r: (
                r.base_quality_score,
                r.score_resistance_quality,
                r.score_compression,
                r.proximity_score,
                r.bars_count
            ),
            reverse=True
        )

        winning_base = evaluated_bases[0]
        winning_base.is_valid = True
        return winning_base
    except Exception as exc:
        logger.warning("[%s] detect_15m_consolidation failed: %s", symbol, exc)
        return ConsolidationResult(symbol=symbol or "?", is_valid=False, rejection_reason=f"EXCEPTION: {exc}")


def detect_15m_consolidation(
    df_15m_closed: Optional[pd.DataFrame],
    atr_15m: float,
    ist_now: datetime,
    config: Dict[str, Any],
    symbol: Optional[str] = None,
    precomputed_context: Optional[Prepared15mContext] = None
) -> ConsolidationResult:
    """
    Main entry point for 15m consolidation base detection (Adaptive V3).
    Supports dual-mode execution via MULTI_TF_CONSOLIDATION_ENGINE ('optimized' default, 'legacy' fallback).
    """
    engine_mode = os.getenv("MULTI_TF_CONSOLIDATION_ENGINE", "optimized").lower()
    if engine_mode == "legacy":
        return _detect_15m_consolidation_legacy(df_15m_closed, atr_15m, ist_now, config, symbol=symbol)

    ctx = precomputed_context
    if ctx is None:
        ctx = prepare_15m_context(df_15m_closed, atr_15m, config, symbol=symbol)

    if ctx is None:
        sym = symbol or (df_15m_closed.attrs.get("symbol") if df_15m_closed is not None else None) or "?"
        return ConsolidationResult(symbol=sym, is_valid=False, rejection_reason="INSUFFICIENT_DATA_OR_ATR_ZERO")

    return detect_15m_consolidation_from_context(ctx, ist_now, config)


def _generate_candidate_windows(df: pd.DataFrame, atr_15m: float, config: Dict[str, Any]) -> List[Tuple[pd.DataFrame, int]]:
    """
    Evaluates candidate window lengths (e.g. 6, 8, 10, 12, 16, 20, 24, 30, 35 bars).
    Enforces overnight gap policy per window. If an overnight gap broke the base,
    multi-day windows spanning that gap are discarded while intraday windows survive.
    Optimized: Single-pass vectorized session and overnight gap analysis without repeated DataFrame copying.
    """
    candidate_bars = config.get("CANDIDATE_WINDOW_BARS", [6, 8, 10, 12, 16, 20, 24, 30, 35])
    min_bars = config.get("MIN_CONSOLIDATION_BARS", 6)
    gap_pct_limit = config.get("GAP_PCT_THRESHOLD", 0.020)
    gap_atr_limit = config.get("GAP_ATR_MULT", 2.0) * (atr_15m if atr_15m > 0 else 1.0)
    total_len = len(df)
    if total_len < min_bars:
        return []

    # Pre-extract session dates efficiently
    if "session_date" in df.columns:
        dates_arr = df["session_date"].values
    elif "Date" in df.columns:
        dates_arr = pd.to_datetime(df["Date"]).dt.date.values
    elif isinstance(df.index, pd.DatetimeIndex):
        dates_arr = df.index.date
    else:
        dt_idx = pd.to_datetime(df.index)
        dates_arr = dt_idx.date if hasattr(dt_idx, "date") else np.array([d.date() for d in dt_idx])

    # Find disruptive overnight gap positions across the entire DataFrame
    # Day changes occur when dates_arr[i] != dates_arr[i-1]
    # For each day boundary, bar i is open of current day, bar i-1 is close of previous day
    open_vals = df["Open"].values.astype(float)
    close_vals = df["Close"].values.astype(float)

    # Any disruptive gap at index `i` invalidates any window spanning across it (window_start_idx < i)
    most_recent_disruptive_gap_idx = -1
    for i in range(1, total_len):
        if dates_arr[i] != dates_arr[i - 1]:
            open_px = open_vals[i]
            prev_close_px = close_vals[i - 1]
            gap_abs = abs(open_px - prev_close_px)
            gap_pct = gap_abs / prev_close_px if prev_close_px > 0 else 0.0
            if gap_pct > gap_pct_limit or gap_abs > gap_atr_limit:
                most_recent_disruptive_gap_idx = i

    valid_windows: List[Tuple[pd.DataFrame, int]] = []
    seen_bar_counts = set()

    for k in sorted(candidate_bars):
        if k > total_len or k < min_bars:
            continue

        window_start_idx = total_len - k
        if most_recent_disruptive_gap_idx != -1 and window_start_idx < most_recent_disruptive_gap_idx:
            # Spans across a disruptive overnight gap!
            continue

        slice_df = df.iloc[-k:]
        n_sessions = len(set(dates_arr[-k:]))

        if k not in seen_bar_counts:
            seen_bar_counts.add(k)
            if "session_date" not in slice_df.columns:
                slice_df = slice_df.copy()
                slice_df["session_date"] = dates_arr[-k:]
            valid_windows.append((slice_df, n_sessions))

    return valid_windows


def _find_valid_window(df: pd.DataFrame, atr_15m: float, config: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
    """
    Legacy helper maintained for backward compatibility.
    Scans backward from the most recent bar up to MAX_CONSOLIDATION_BARS.
    """
    max_bars = config.get("MAX_CONSOLIDATION_BARS", 35)
    windows = _generate_candidate_windows(df, atr_15m, config)
    if not windows:
        min_bars = config.get("MIN_CONSOLIDATION_BARS", 6)
        slice_df = df.iloc[-min_bars:].copy()
        return slice_df, 1
    # Pick longest valid window up to max_bars
    valid_up_to_max = [w for w in windows if w[0].shape[0] <= max_bars]
    if valid_up_to_max:
        return valid_up_to_max[-1]
    return windows[-1]


def _build_geometry(df: pd.DataFrame, atr_15m: float, res: ConsolidationResult, config: Dict[str, Any]):
    """Calculates adaptive base geometry, structural resistance, and occupancy."""
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)

    q_high = config.get("BOX_HIGH_QUANTILE", 0.90)
    q_low = config.get("BOX_LOW_QUANTILE", 0.10)

    res.hard_high = float(np.max(highs))
    res.hard_low = float(np.min(lows))

    # Structural resistance uses the top high or 90th percentile to avoid single wick distortion
    res.box_high = float(np.percentile(highs, q_high * 100)) if len(highs) >= 6 else res.hard_high
    res.box_low = float(np.percentile(lows, q_low * 100)) if len(lows) >= 6 else res.hard_low

    # Ensure box_high never falls below median close
    med_close = float(np.median(closes))
    if res.box_high < med_close:
        res.box_high = float(np.max(highs))
    if res.box_low > med_close:
        res.box_low = float(np.min(lows))

    res.box_mid = (res.box_high + res.box_low) / 2.0
    res.box_value_center = med_close

    eff_mid = max(res.box_mid, 1.0)
    res.box_width_pct = (res.box_high - res.box_low) / eff_mid
    res.box_width_atr = (res.box_high - res.box_low) / (atr_15m if atr_15m > 0 else 1.0)

    # Occupancy: % of closes inside the base (vectorized numpy mask)
    tol = 0.10 * atr_15m
    inside_mask = (closes >= (res.box_low - tol)) & (closes <= (res.box_high + tol))
    res.box_occupancy = float(np.mean(inside_mask)) if len(closes) > 0 else 1.0

    res.bars_count = len(df)
    res.sessions_count = df["session_date"].nunique() if "session_date" in df.columns else 1
    res.start_ts = df.index[0] if isinstance(df.index[0], pd.Timestamp) else pd.to_datetime(df.index[0])
    res.end_ts = df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.to_datetime(df.index[-1])


def _generate_box_id(df: pd.DataFrame, res: ConsolidationResult):
    """Generates a deterministic hash for this specific consolidation instance."""
    date_val = df.iloc[-1].get("session_date", "today")
    date_str = date_val.strftime("%Y%m%d") if hasattr(date_val, "strftime") else str(date_val)
    h_str = f"{res.box_high:.2f}"
    l_str = f"{res.box_low:.2f}"
    raw = f"{res.symbol}_{date_str}_{h_str}_{l_str}"
    res.box_id = hashlib.md5(raw.encode()).hexdigest()[:10]


def _compute_structure(df: pd.DataFrame, atr_15m: float, res: ConsolidationResult, config: Dict[str, Any]):
    """
    Counts distinct resistance touches, tracks rejection depth progression,
    and detects false-break history.
    """
    tol_pct = config.get("RESISTANCE_TEST_TOL_PCT", 0.0015) * res.box_high
    tol_atr = config.get("RESISTANCE_TEST_TOL_ATR", 0.08) * atr_15m
    tol = max(tol_pct, tol_atr)
    test_zone_low = res.box_high - tol

    tests = 0
    in_test = False
    rejections = []
    current_test_idx = None
    false_breaks = 0

    highs = df["High"].astype(float).values
    lows = df["Low"].astype(float).values
    closes = df["Close"].astype(float).values
    n = len(df)

    for i in range(n):
        h = highs[i]
        c = closes[i]
        l = lows[i]

        # False break check: bar traded above resistance but closed back below
        if h > res.box_high and c <= res.box_high:
            false_breaks += 1

        if h >= test_zone_low:
            if not in_test:
                tests += 1
                in_test = True
                current_test_idx = i
        else:
            if in_test and current_test_idx is not None:
                pullback_depth = res.box_high - l
                rejections.append(pullback_depth)
                in_test = False

    res.resistance_test_count = max(tests, 1)
    res.failed_break_count = false_breaks

    # Declining rejection depth check: are successive pullbacks becoming shallower?
    if len(rejections) >= 2:
        res.rejection_depth_declining = (rejections[-1] <= rejections[0] * 0.90)

    # Time near resistance: % of closes in top 35% of the base
    upper_threshold = res.box_low + 0.65 * (res.box_high - res.box_low)
    res.time_near_resistance_pct = float((closes >= upper_threshold).mean()) if n > 0 else 0.0

    # Supply absorption qualitative classification
    if res.resistance_test_count >= 3 and res.has_higher_lows and res.rejection_depth_declining:
        res.supply_absorption_label = "EXCELLENT"
    elif res.resistance_test_count >= 2 and (res.has_higher_lows or res.time_near_resistance_pct >= 0.50):
        res.supply_absorption_label = "STRONG"
    elif res.resistance_test_count >= 2:
        res.supply_absorption_label = "MODERATE"
    else:
        res.supply_absorption_label = "EARLY"


def _evaluate_dormancy(window_df: pd.DataFrame, full_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Checks if the base has genuine market activity vs a frozen/dead volume flatline.
    Uses time-of-day volume baseline and liquidity turnover checks.
    Differentiates healthy VCP volume contraction from true illiquid abandonment.
    """
    if "Volume" not in window_df.columns or "Volume" not in full_df.columns:
        return False, 1.0
    vol_base = window_df["Volume"].astype(float).values
    if len(vol_base) == 0:
        return False, 1.0
    mean_base_vol = float(np.mean(vol_base))
    if mean_base_vol == 0:
        return True, 0.0

    # 1. Turnover Sanity Check: If average 15m turnover >= 50,000 Rs, the stock is liquid
    if "Close" in window_df.columns:
        mean_turnover = float((window_df["Close"].astype(float) * window_df["Volume"].astype(float)).mean())
        if mean_turnover >= config.get("LIQUIDITY_MIN_TURNOVER_RS", 50000.0):
            return False, 1.0

    # 2. Time-of-Day Aware Baseline
    # Match the time of day of the window's latest bars against historical bars at the same time
    try:
        last_dt = window_df.index[-1] if isinstance(window_df.index[-1], pd.Timestamp) else pd.to_datetime(window_df.index[-1])
        target_minute = last_dt.hour * 60 + last_dt.minute
        
        full_dt_idx = pd.to_datetime(full_df.index) if not isinstance(full_df.index, pd.DatetimeIndex) else full_df.index
        minutes = full_dt_idx.hour * 60 + full_dt_idx.minute
        # Find bars within +/- 30 mins of the same time of day across previous sessions
        time_mask = (abs(minutes - target_minute) <= 30)
        same_time_vols = full_df["Volume"].loc[time_mask].astype(float)
        if len(same_time_vols) >= 5:
            baseline_vol = float(same_time_vols.median())
        else:
            baseline_vol = float(full_df["Volume"].astype(float).tail(40).median())
    except Exception:
        baseline_vol = float(full_df["Volume"].astype(float).tail(40).median())

    if baseline_vol <= 0:
        return False, 1.0

    vol_ratio = mean_base_vol / baseline_vol
    min_vol_ratio = config.get("DORMANCY_MIN_VOL_RATIO", 0.15)
    
    # Healthy VCP contraction: if volume is low because price range is also tightly compressed,
    # and turnover is reasonable (> 25k Rs), it's healthy dry-up, NOT dormancy
    if "Close" in window_df.columns:
        mean_turnover = float((window_df["Close"].astype(float) * window_df["Volume"].astype(float)).mean())
        if mean_turnover >= 25000.0 and vol_ratio >= 0.08:
            return False, round(vol_ratio, 3)

    is_dormant = (vol_ratio < min_vol_ratio)
    return is_dormant, round(vol_ratio, 3)


def _compute_scores(
    window_df: pd.DataFrame,
    full_df: pd.DataFrame,
    atr_15m: float,
    res: ConsolidationResult,
    config: Dict[str, Any],
    max_atr_limit: Optional[float] = None
):
    """
    [V3.1] DECOUPLED STRUCTURAL & OPPORTUNITY ENGINE:
    1. Base Quality (0-100): Pure geometry and structure (price location independent)
       - Maturity (15 pts)
       - Tightness (20 pts)
       - Resistance Quality (20 pts)
       - Repeated Tests (15 pts)
       - Compression/VCP (15 pts)
       - Higher Lows (10 pts)
       - Support Integrity (5 pts)
       - Soft Width Penalty (0-15 pts)
       - Dormancy Penalty (15 pts if dead)
    2. Proximity Score (0-100): Distance of latest price to resistance boundary
    3. Pressure Score (0-100): Active price migration and momentum
    4. Opportunity Composite: Weighted combination for downstream monitoring
    """
    n = len(window_df)
    score = 0

    # ── A. MATURITY (15 pts) — Duration × Quality Interaction ────────────────
    if n >= 16:
        s_mat = 15
    elif n >= 12:
        s_mat = 14
    elif n >= 10:
        s_mat = 13
    elif n >= 8:
        s_mat = 12
    elif n >= 6:
        s_mat = 9
    elif n >= 4:
        s_mat = 6
    else:
        s_mat = 0

    # Interaction: if base is wide, cap maturity at 10
    if res.box_width_atr > 1.25 and s_mat > 10:
        s_mat = 10
    res.score_maturity = min(s_mat, config.get("SCORE_MATURITY_MAX", 15))
    score += res.score_maturity

    # ── B. TIGHTNESS (20 pts) — Range/ATR ────────────────────────────────────
    w = res.box_width_atr
    eff_max_atr = max_atr_limit or config.get("MAX_BOX_WIDTH_ATR", 2.20)
    if w <= 0.75:
        s_tight = 20
    elif w <= 1.00:
        s_tight = 17
    elif w <= 1.25:
        s_tight = 13
    elif w <= 1.50:
        s_tight = 8
    elif w <= eff_max_atr * 0.75:
        s_tight = 5
    elif w <= eff_max_atr:
        s_tight = 2
    else:
        s_tight = 0
    res.score_tightness = min(s_tight, config.get("SCORE_TIGHTNESS_MAX", 20))
    score += res.score_tightness

    # ── C. RESISTANCE QUALITY (20 pts) — Ceiling Precision ───────────────────
    highs = window_df["High"].astype(float)
    top_highs = highs[highs >= res.box_high - (0.15 * atr_15m)]
    std_top = float(top_highs.std()) if len(top_highs) >= 2 else 0.0
    std_pct = std_top / res.box_high if res.box_high > 0 else 0.0
    if len(top_highs) >= 2 and std_pct <= 0.0010:
        s_rq = 20
    elif len(top_highs) >= 2 and std_pct <= 0.0020:
        s_rq = 16
    elif len(top_highs) >= 2 and std_pct <= 0.0035:
        s_rq = 12
    elif len(top_highs) >= 1:
        s_rq = 8
    else:
        s_rq = 4
    res.score_resistance_quality = min(s_rq, config.get("SCORE_RESISTANCE_QUALITY_MAX", 20))
    score += res.score_resistance_quality

    # ── D. REPEATED TESTS (15 pts) — Distinct Touches ────────────────────────
    t = res.resistance_test_count
    if t >= 4:
        s_tests = 15
    elif t == 3:
        s_tests = 13
    elif t == 2:
        s_tests = 10
    elif t == 1:
        s_tests = 3
    else:
        s_tests = 0
    res.score_repeated_tests = min(s_tests, config.get("SCORE_REPEATED_TESTS_MAX", 15))
    score += res.score_repeated_tests

    # ── E. COMPRESSION / VCP (15 pts) — Volatility Contracting ───────────────
    s_comp = 8
    if n >= 4:
        half = n // 2
        early_ranges = (window_df["High"].iloc[:half] - window_df["Low"].iloc[:half]).values.astype(float)
        late_ranges  = (window_df["High"].iloc[half:] - window_df["Low"].iloc[half:]).values.astype(float)
        mean_early = float(np.mean(early_ranges)) if len(early_ranges) > 0 else 0.0
        mean_late  = float(np.mean(late_ranges))  if len(late_ranges) > 0 else 0.0
        if mean_early > 0:
            comp_ratio = mean_late / mean_early
            res.compression_ratio = round(comp_ratio, 3)
            if comp_ratio <= 0.60:
                s_comp = 15
            elif comp_ratio <= 0.75:
                s_comp = 12
            elif comp_ratio <= 0.90:
                s_comp = 8
            elif comp_ratio <= 1.00:
                s_comp = 4
            else:
                s_comp = 0
    res.score_compression = min(s_comp, config.get("SCORE_COMPRESSION_MAX", 15))
    score += res.score_compression

    # ── F. HIGHER LOWS (10 pts) — Rising Lows = Buyers Getting Aggressive ────
    s_hl = 0
    if n >= 4:
        half = n // 2
        early_lows = window_df["Low"].iloc[:half].astype(float)
        late_lows  = window_df["Low"].iloc[half:].astype(float)
        early_low_min = float(early_lows.min())
        late_low_min  = float(late_lows.min())
        hl_rise = late_low_min - early_low_min
        res.higher_lows_strength = round(hl_rise, 2)
        min_strong_rise = config.get("HIGHER_LOWS_MIN_RISE_ATR", 0.15) * atr_15m
        if hl_rise >= min_strong_rise:
            s_hl = 10
            res.has_higher_lows = True
        elif hl_rise >= 0:
            s_hl = 7
            res.has_higher_lows = True
        elif hl_rise >= -(0.10 * atr_15m):
            s_hl = 4
        else:
            s_hl = 0
    res.score_higher_lows = min(s_hl, config.get("SCORE_HIGHER_LOWS_MAX", 10))
    score += res.score_higher_lows

    # ── G. SUPPORT INTEGRITY (5 pts) — Structural Defense Quality ───────────
    s_si = 3
    if n >= 4 and atr_15m > 0:
        lows = window_df["Low"].astype(float).values
        closes = window_df["Close"].astype(float).values
        worst_excursion_below = max(float(res.box_low - min(lows)), 0.0)
        worst_excursion_atr = worst_excursion_below / atr_15m
        closed_below_floor = any(c < res.box_low for c in closes)

        if closed_below_floor:
            s_si = 1
        elif worst_excursion_atr <= 0.08:
            s_si = 5
        elif worst_excursion_atr <= 0.20:
            s_si = 4
        else:
            s_si = 2
    res.score_support_integrity = min(s_si, config.get("SCORE_SUPPORT_INTEGRITY_MAX", 5))
    score += res.score_support_integrity

    # ── H. PENALTIES (Soft Width + Dormancy) ──────────────────────────────────
    if getattr(res, "soft_width_penalty", 0) > 0:
        score -= res.soft_width_penalty

    if getattr(res, "is_dormant", False):
        dormancy_penalty = config.get("DORMANCY_PENALTY", 15)
        score -= dormancy_penalty

    # ── 1. BASE QUALITY SCORE (0-100) ─────────────────────────────────────────
    # Pure geometry & structure, zero current price position bias!
    res.base_quality_score = int(np.clip(score, 0, 100))

    # ── 2. BREAKOUT PROXIMITY SCORE (0-100) ───────────────────────────────────
    if len(window_df) > 0:
        last_close = float(window_df["Close"].iloc[-1])
        box_height = max(res.box_high - res.box_low, 0.0001)
        rel_pos = (last_close - res.box_low) / box_height

        if rel_pos >= 0.85:
            res.proximity_score = 100
        elif rel_pos >= 0.70:
            res.proximity_score = 80
        elif rel_pos >= 0.55:
            res.proximity_score = 65
        elif rel_pos >= 0.40:
            res.proximity_score = 50
        elif rel_pos >= 0.25:
            res.proximity_score = 30
        else:
            res.proximity_score = 15
    else:
        res.proximity_score = 50

    # ── 3. PRESSURE / MOMENTUM SCORE (0-100) ──────────────────────────────────
    s_press = 0
    if res.has_higher_lows:
        s_press += 35
    if len(window_df) >= 3:
        recent_bars = window_df.iloc[-3:]
        up_bars = sum(
            1 for _, b in recent_bars.iterrows()
            if float(b["Close"]) >= (float(b["High"]) + float(b["Low"])) / 2.0
        )
        s_press += int(35 * (up_bars / 3))
    else:
        s_press += 20

    if "Volume" in window_df.columns:
        vols = window_df["Volume"].astype(float)
        closes = window_df["Close"].astype(float)
        opens = window_df["Open"].astype(float)
        green_vol = vols[closes >= opens].sum()
        total_vol = vols.sum()
        if total_vol > 0 and (green_vol / total_vol) >= 0.55:
            s_press += 30
        elif total_vol > 0 and (green_vol / total_vol) >= 0.45:
            s_press += 15
    else:
        s_press += 15
    res.pressure_score = min(s_press, 100)

    # ── 4. OPPORTUNITY COMPOSITE SCORE (0-100) ────────────────────────────────
    res.setup_score = int(np.clip(
        0.50 * res.base_quality_score +
        0.30 * res.proximity_score +
        0.20 * res.pressure_score,
        0, 100
    ))

    # ── 5. LIFECYCLE STAGE ────────────────────────────────────────────────────
    if res.base_quality_score >= 65 and res.proximity_score >= 80 and res.pressure_score >= 60:
        res.lifecycle_stage = "PRESSURE"
    elif res.base_quality_score >= 65 and res.proximity_score >= 75:
        res.lifecycle_stage = "PRE_BREAKOUT"
    elif res.base_quality_score >= 75:
        res.lifecycle_stage = "STRONG"
    elif res.base_quality_score >= 60:
        res.lifecycle_stage = "QUALIFIED"
    else:
        res.lifecycle_stage = "FORMING"

    # Populate legacy aliases for backward compatibility
    res.score_resistance_def   = res.score_resistance_quality
    res.score_tight_range      = res.score_tightness
    res.score_resistance_tests = res.score_repeated_tests
    res.score_compression_vcp  = res.score_compression
    res.score_hl               = res.score_higher_lows

    if res.base_quality_score >= 85:
        res.base_rating_label = "EXCEPTIONAL"
    elif res.base_quality_score >= 75:
        res.base_rating_label = "SUPER"
    elif res.base_quality_score >= 65:
        res.base_rating_label = "GOOD"
    elif res.base_quality_score >= 50:
        res.base_rating_label = "WATCH"
    else:
        res.base_rating_label = "REJECT"

