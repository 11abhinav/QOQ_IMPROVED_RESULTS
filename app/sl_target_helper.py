# =====================================================================================
# app/sl_target_helper.py  (v5 — ANTI-TRAP EDITION)
#
# KEY INSIGHT: Different scanners trade completely different setups.
# One SL/Target formula for all is wrong. This module dispatches to
# a mode-specific sub-function for each scanner type.
#
# MODES:
#   "EOD"      → Daily momentum breakout (swing trade, hold days–weeks)
#   "MULTI_TF" → Multi-timeframe dynamic engine
#   "REVERSAL" → Counter-trend oversold bounce (mean reversion, hold days–weeks)
#
#
# v5 UPGRADES:
#   1. MULTI-SWING CLUSTERING — scans last 3 swing lows; if 2+ cluster within 1%,
#      uses the cluster zone for SL placement (much stronger support)
#   2. VWAP-ANCHORED SL — for intraday/1H, uses VWAP as SL anchor when between
#      candle_low and entry (VWAP = institutional fair-value support)
#   3. ADX-AWARE BUFFER WIDENING — trending stocks (ADX>35) get 30% wider buffers
#      to survive deeper pullbacks without stopping out
#   4. ATR-SCALED TARGET CAPS — prevents unrealistic targets that stock can't reach
#      15m: 5×ATR | 1H: 8×ATR | EOD: 12×ATR
#   5. MEASURED MOVE TARGETS — when BASE_WIDTH available, T1 = entry + base height
#
# ANTI-OPERATOR-TRAP DESIGN:
#   Operators/algos know retail places SL exactly at swing low.
#   They run stops with a wick, then reverse. Our fix:
#   → SL is placed BELOW the zone, not at it, with a meaningful % buffer
#   → Buffer is max(mode_atr_fraction × ATR, mode_pct × price)
#   → ADX-scaled: trending stocks get wider buffers (deeper pullbacks)
#   → This makes the stop hunt unprofitable for operators (too far to sweep)
#
# SL BUFFER TABLE (per mode from _MODE_CONFIG):
#   EOD       → max(0.80×ATR, 0.75% price) — meaningful, daily trade (Balanced)
#   MULTI_TF  → max(0.50×ATR, 0.50% price) — tighter buffer for intraday (Aggressive)
#   REVERSAL  → max(1.00×ATR, 1.00% price) — widest buffer, volatile beaten stocks (Wide)
#   PULLBACK  → max(0.75×ATR, 0.75% price) — continuation pullback standard
#
# MINIMUM R:R TABLE (per mode):

#   EOD       → 2.0:1 (daily trade — overnight risk demands it)
#   REVERSAL  → 2.0:1 (counter-trend — higher base risk)
#
# TARGET PHILOSOPHY (per mode):
#   EOD       → Nearest swing high / R1 pivot → R2 → 52W high zone

#   REVERSAL  → EMA20 or BB_MID (mean reversion T1) → SMA50 (T2) → R1 (T3)
# =====================================================================================

from __future__ import annotations
import sys
import os
import pandas as pd
from memory_profiler import profile_function
from typing import Optional, Any
import math

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_APP_DIR, ".."))
for _p in (_APP_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import ADAPTIVE_TARGET_CAPS, MIN_NATURAL_RR, MIN_REWARD_POTENTIAL, TARGET_QUALITY_THRESHOLD


# ── Per-mode configuration ────────────────────────────────────────────────────
_MODE_CONFIG = {
    #           atr_base  sl_atr_buf  sl_pct_buf  max_sl_atr
    "EOD":      (2.00,    0.80,       0.0075,     3.0),   # Balanced
    "MULTI_TF": (1.50,    0.50,       0.0050,     3.0),   # Aggressive
    "REVERSAL": (2.00,    1.00,       0.0100,     3.5),   # Wide
    "PULLBACK": (2.00,    0.75,       0.0075,     3.0),   # Pullback Continuation
}
_DEFAULT_CONFIG = (1.50, 0.50, 0.0050, 3.0)


# ── Trade Structure Invariant Validator ─────────────────────────────────────
class TradeStructureValidator:
    """
    Centralized Validator enforcing all Mathematical Trade Structure Invariants:
      1. entry > 0
      2. stop_loss < entry (rejection_code: INVALID_STOP_PLACEMENT)
      3. risk = entry - stop_loss > 0
      4. target_1 > entry
      5. target_1 <= target_2 <= target_3 (when multiple targets exist)
      6. natural_rr = (target_1 - entry) / risk >= min_rr
    """
    @staticmethod
    def validate(entry: float, stop_loss: float, target_1: float,
                 target_2: Optional[float] = None, target_3: Optional[float] = None,
                 target_4: Optional[float] = None,
                 min_rr: float = 2.0) -> dict:
        if not entry or entry <= 0:
            return {
                "is_valid": False, "rejection_code": "INVALID_ENTRY_PRICE",
                "rejection_reason": f"INVALID_ENTRY_PRICE (Entry price ₹{entry} must be > 0)"
            }

        if stop_loss >= entry:
            return {
                "is_valid": False, "rejection_code": "INVALID_STOP_PLACEMENT",
                "rejection_reason": f"INVALID_STOP_PLACEMENT (Stop Loss ₹{stop_loss:.2f} >= Entry Price ₹{entry:.2f})"
            }

        risk = entry - stop_loss
        if risk <= 0:
            return {
                "is_valid": False, "rejection_code": "INVALID_RISK_AMOUNT",
                "rejection_reason": f"INVALID_RISK_AMOUNT (Risk ₹{risk:.2f} must be > 0)"
            }

        if not target_1 or target_1 <= entry:
            return {
                "is_valid": False, "rejection_code": "INVALID_TARGET_PRICE",
                "rejection_reason": f"INVALID_TARGET_PRICE (Target 1 ₹{target_1} must be > Entry ₹{entry})"
            }

        # Target ordering invariants with epsilon spacing (t1 < t2 < t3 < t4)
        epsilon = max(0.05, 0.002 * entry)
        if target_2 and target_2 <= target_1 + epsilon:
            return {
                "is_valid": False, "rejection_code": "UNORDERED_TARGET_HIERARCHY",
                "rejection_reason": f"UNORDERED_TARGET_HIERARCHY (Target 2 ₹{target_2:.2f} <= Target 1 ₹{target_1:.2f} + epsilon ₹{epsilon:.2f})"
            }
        if target_3 and target_2 and target_3 <= target_2 + epsilon:
            return {
                "is_valid": False, "rejection_code": "UNORDERED_TARGET_HIERARCHY",
                "rejection_reason": f"UNORDERED_TARGET_HIERARCHY (Target 3 ₹{target_3:.2f} <= Target 2 ₹{target_2:.2f} + epsilon ₹{epsilon:.2f})"
            }
        if target_4 and target_3 and target_4 <= target_3 + epsilon:
            return {
                "is_valid": False, "rejection_code": "UNORDERED_TARGET_HIERARCHY",
                "rejection_reason": f"UNORDERED_TARGET_HIERARCHY (Target 4 ₹{target_4:.2f} <= Target 3 ₹{target_3:.2f} + epsilon ₹{epsilon:.2f})"
            }

        natural_rr = round(abs(target_1 - entry) / risk, 2)
        if natural_rr < min_rr:
            return {
                "is_valid": False, "rejection_code": "NO_VALID_STRUCTURAL_TARGET",
                "rejection_reason": f"NO_VALID_STRUCTURAL_TARGET (Min RR: {min_rr}x, Actual: {natural_rr}x)",
                "natural_rr": natural_rr
            }

        return {"is_valid": True, "natural_rr": natural_rr, "risk": risk}


# ── Target Engine v7 Classes ──────────────────────────────────────────────────
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from config import (
    TARGET_SOURCE_WEIGHTS, SOURCE_PRIORITY, TARGET_CONFLICT_POLICY,
    EXIT_PROFILES, SCANNER_EXIT_PROFILE, FIB_EXTENSIONS, FIB_RETRACEMENTS,
    ABCD_BC_RETRACE_MIN, ABCD_BC_RETRACE_MAX, FIB_200_GATE,
    ROUND_NUMBER_BOOST, ROUND_NUMBER_PCT, TARGET_CLUSTER_WINDOW_ATR_FRAC,
    TARGET_CLUSTER_WINDOW_PCT, FIB_200_WEIGHTS
)

class TargetSource(Enum):
    RESISTANCE    = "RESISTANCE"
    EQUAL_HIGH    = "EQUAL_HIGH"
    PREV_DAY_HIGH = "PREV_DAY_HIGH"
    HIGH_20D      = "HIGH_20D"
    HIGH_52W      = "HIGH_52W"
    ABCD          = "ABCD"
    FIB_127       = "FIB_127"
    FIB_162       = "FIB_162"
    FIB_200       = "FIB_200"
    BB_MID        = "BB_MID"
    SMA50         = "SMA50"
    SMA200        = "SMA200"
    RETRACE_382   = "RETRACE_382"
    RETRACE_50    = "RETRACE_50"
    RETRACE_618   = "RETRACE_618"
    SWING_HIGH_RAW = "SWING_HIGH_RAW"
    ATR_PROJ      = "ATR_PROJ"
    R1            = "R1"
    R2            = "R2"
    ROUND_NUM     = "ROUND_NUM"

@dataclass
class TargetCandidate:
    price:         float
    source:        TargetSource
    timeframe:     str
    scanner:       str
    strength:      str
    anchor_points: dict
    generated_from: str = ""
    cluster_id:    Optional[int] = None
    score:         int = 0
    selection_state: str = "REJECTED"
    is_round_number: bool = False

@dataclass
class ClusteredTarget:
    cluster_id: int
    consensus_price: float
    score: int
    candidates: List[TargetCandidate]
    is_round_number: bool = False

class TargetScorer:
    @staticmethod
    def score(candidate: TargetCandidate, macro_regime: str) -> int:
        source_name = candidate.source.name
        if source_name == "FIB_200":
            return FIB_200_WEIGHTS.get(macro_regime, 5)
        return TARGET_SOURCE_WEIGHTS.get(source_name, 0)

class RoundNumberEngine:
    @staticmethod
    def _get_tick(price: float) -> float:
        if price < 50: return 2.0
        if price < 100: return 5.0
        if price < 200: return 10.0
        if price < 500: return 25.0
        if price < 1000: return 50.0
        if price < 2000: return 100.0
        if price < 5000: return 250.0
        return 1000.0

    @staticmethod
    def _get_weight_multiplier(tick: float) -> float:
        if tick >= 1000.0: return 2.5 # Extreme
        if tick >= 50.0: return 2.0   # Very High
        if tick >= 10.0: return 1.5   # High
        return 1.0                    # Medium

    @staticmethod
    def detect_and_boost(clusters: List[ClusteredTarget], eff_atr: float = 0.0) -> None:
        for c in clusters:
            tick = RoundNumberEngine._get_tick(c.consensus_price)
            if tick <= 0: continue
            nearest = round(c.consensus_price / tick) * tick
            pct_diff = abs(c.consensus_price - nearest) / max(c.consensus_price, 1e-5)
            if pct_diff <= ROUND_NUMBER_PCT:
                c.is_round_number = True

                weight_mult = RoundNumberEngine._get_weight_multiplier(tick)
                c.score += int(ROUND_NUMBER_BOOST * weight_mult)

                # Adaptive Front-running offset
                tick_offset = tick if tick <= 5.0 else (tick * 0.2)
                if eff_atr > 0:
                    offset = min(0.25 * eff_atr, 0.003 * nearest, tick_offset)
                else:
                    offset = min(0.003 * nearest, tick_offset)

                front_run_target = round(nearest - offset, 2)

                c.candidates.append(TargetCandidate(
                    price=front_run_target, source=TargetSource.ROUND_NUM,
                    timeframe="any", scanner="any", strength="NORMAL",
                    anchor_points={"offset": round(offset, 2), "base_round": nearest}, generated_from="RoundNumberEngine",
                    cluster_id=c.cluster_id, score=0, is_round_number=True
                ))

class ClusterEngine:
    @staticmethod
    def _consensus_price(candidates: List[TargetCandidate]) -> float:
        ranked = sorted(candidates, key=lambda c: SOURCE_PRIORITY.get(c.source.name, 99))
        return ranked[0].price

    @staticmethod
    def cluster(candidates: List[TargetCandidate], entry: float, eff_atr: float) -> List[ClusteredTarget]:
        if not candidates: return []
        window = max(TARGET_CLUSTER_WINDOW_ATR_FRAC * eff_atr, TARGET_CLUSTER_WINDOW_PCT * entry)
        sorted_cands = sorted(candidates, key=lambda c: c.price)

        clusters = []
        current_cluster_cands = [sorted_cands[0]]
        cluster_min = sorted_cands[0].price

        for cand in sorted_cands[1:]:
            if cand.price - cluster_min <= window + 1e-6:
                current_cluster_cands.append(cand)
            else:
                clusters.append(current_cluster_cands)
                current_cluster_cands = [cand]
                cluster_min = cand.price
        clusters.append(current_cluster_cands)

        result = []
        for i, c_cands in enumerate(clusters):
            for cand_item in c_cands:
                cand_item.cluster_id = i
            c_price = ClusterEngine._consensus_price(c_cands)
            c_score = sum(cand_item.score for cand_item in c_cands)
            result.append(ClusteredTarget(
                cluster_id=i, consensus_price=c_price, score=c_score, candidates=c_cands
            ))
        return result

class LiquidityEngine:
    @staticmethod
    def detect_equal_highs(ticker: pd.DataFrame, entry: float) -> Optional[float]:
        if ticker is None or ticker.empty: return None
        recent = ticker.tail(60)
        highs = recent["High"].values
        for i in range(len(highs)):
            for j in range(i+1, len(highs)):
                if highs[i] > entry and highs[j] > entry:
                    if abs(highs[i] - highs[j]) / highs[i] < 0.005:
                        return float((highs[i] + highs[j]) / 2)
        return None

class ABCDDetector:
    BC_RETRACE_MIN = 0.382
    BC_RETRACE_MAX = 0.786

    @staticmethod
    def detect(ticker: pd.DataFrame, entry: float) -> Optional[float]:
        if ticker is None or ticker.empty or "SWING_LOW" not in ticker.columns or "SWING_HIGH" not in ticker.columns:
            return None
        pivot_lows  = ticker["SWING_LOW"].dropna()
        pivot_highs = ticker["SWING_HIGH"].dropna()

        if len(pivot_lows) < 2 or len(pivot_highs) < 1:
            return None

        for c_idx, C in reversed(list(pivot_lows.items())):
            b_candidates = pivot_highs[pivot_highs.index < c_idx]
            if b_candidates.empty: continue
            b_idx = b_candidates.index[-1]
            B = b_candidates.iloc[-1]
            if B <= C: continue

            a_candidates = pivot_lows[pivot_lows.index < b_idx]
            if a_candidates.empty: continue
            A = a_candidates.iloc[-1]
            if A >= B: continue

            AB = B - A
            BC = B - C
            BC_ratio = BC / AB if AB > 0 else 0

            if not (ABCDDetector.BC_RETRACE_MIN <= BC_ratio <= ABCDDetector.BC_RETRACE_MAX):
                continue

            D_projection = C + AB
            if D_projection > entry:
                return round(float(D_projection), 2)
        return None

class CandidateGenerator:
    def generate_breakout_candidates(
        self, entry: float, eff_atr: float, atr_pct: float, adx: float, volume_ratio: float,
        vwap: float, macro_regime: str, scanner: str,
        swing_low: float, swing_high: float, swing_low_raw: float, swing_high_raw: float,
        r1: float, r2: float, bb_upper: float, prior_20d_high: float, high_52w: float, prev_day_high: float,
        ticker: pd.DataFrame
    ) -> List[TargetCandidate]:
        candidates = []

        # Resistance
        from sl_target_helper import _pick_resistance, _safe
        res, label = _pick_resistance(entry, swing_high, r1, bb_upper, swing_high_raw, r2)
        if res:
            candidates.append(TargetCandidate(price=res, source=TargetSource.RESISTANCE, timeframe="any", scanner=scanner, strength="NORMAL", anchor_points={"label": label}))

        if _safe(prev_day_high) and prev_day_high > entry:
            candidates.append(TargetCandidate(price=prev_day_high, source=TargetSource.PREV_DAY_HIGH, timeframe="1d", scanner=scanner, strength="NORMAL", anchor_points={}))

        if _safe(prior_20d_high) and prior_20d_high > entry:
            candidates.append(TargetCandidate(price=prior_20d_high, source=TargetSource.HIGH_20D, timeframe="1d", scanner=scanner, strength="NORMAL", anchor_points={}))

        if _safe(high_52w) and high_52w > entry:
            candidates.append(TargetCandidate(price=high_52w, source=TargetSource.HIGH_52W, timeframe="1d", scanner=scanner, strength="STRONG", anchor_points={}))

        eq_high = LiquidityEngine.detect_equal_highs(ticker, entry)
        if eq_high:
            candidates.append(TargetCandidate(price=eq_high, source=TargetSource.EQUAL_HIGH, timeframe="any", scanner=scanner, strength="STRONG", anchor_points={}))

        leg = None
        if _safe(swing_high_raw) and _safe(swing_low_raw):
            leg = swing_high_raw - swing_low_raw

        if leg and leg > 0:
            candidates.append(TargetCandidate(price=entry + leg * 1.272, source=TargetSource.FIB_127, timeframe="any", scanner=scanner, strength="NORMAL", anchor_points={"leg": leg}))
            candidates.append(TargetCandidate(price=entry + leg * 1.618, source=TargetSource.FIB_162, timeframe="any", scanner=scanner, strength="NORMAL", anchor_points={"leg": leg}))

            fib200_allowed = (
                _safe(adx) and adx > FIB_200_GATE["min_adx"]
                and _safe(volume_ratio) and volume_ratio > FIB_200_GATE["min_vol_ratio"]
                and _safe(vwap) and entry > vwap
                and macro_regime in ("TRENDING", "BULL")
            )
            if fib200_allowed:
                candidates.append(TargetCandidate(price=entry + leg * 2.0, source=TargetSource.FIB_200, timeframe="any", scanner=scanner, strength="NORMAL", anchor_points={"leg": leg}))

        abcd_price = ABCDDetector.detect(ticker, entry)
        if abcd_price:
            candidates.append(TargetCandidate(price=abcd_price, source=TargetSource.ABCD, timeframe="any", scanner=scanner, strength="NORMAL", anchor_points={}))

        candidates.append(TargetCandidate(price=entry + 3 * eff_atr, source=TargetSource.ATR_PROJ, timeframe="any", scanner=scanner, strength="NORMAL", anchor_points={}))

        for c in candidates:
            c.score = TargetScorer.score(c, macro_regime)

        return candidates

class TargetStrategy(ABC):
    def pre_filter(self, candidates: List[TargetCandidate], context: dict) -> List[TargetCandidate]:
        return candidates

    @abstractmethod
    def select_targets(self, clusters: List[ClusteredTarget], entry: float, risk: float, context: dict) -> dict:
        pass

    def post_filter(self, result: dict, context: dict) -> dict:
        return result

class TrendExtensionStrategy(TargetStrategy):
    def select_targets(self, clusters: List[ClusteredTarget], entry: float, risk: float, context: dict) -> dict:
        if not clusters: return {}
        # MULTI_TF uses CONFIDENCE policy
        ranked = sorted(clusters, key=lambda c: c.score, reverse=True)
        t1_c = ranked[0]
        t2_c = ranked[1] if len(ranked) > 1 else t1_c
        t3_c = ranked[2] if len(ranked) > 2 else t2_c

        # Sort targets in ascending order of price to ensure t1 <= t2 <= t3
        sorted_pairs = sorted([(t1_c.consensus_price, t1_c), (t2_c.consensus_price, t2_c), (t3_c.consensus_price, t3_c)], key=lambda x: x[0])

        return {
            "t1": sorted_pairs[0][0], "t2": sorted_pairs[1][0], "t3": sorted_pairs[2][0],
            "t1_cluster": sorted_pairs[0][1], "t2_cluster": sorted_pairs[1][1], "t3_cluster": sorted_pairs[2][1]
        }

class ClusterConsensusStrategy(TargetStrategy):
    def select_targets(self, clusters: List[ClusteredTarget], entry: float, risk: float, context: dict) -> dict:
        if not clusters: return {}
        # EOD uses REGIME policy
        ranked = sorted(clusters, key=lambda c: c.score, reverse=True)
        t1_c = ranked[0]
        t2_c = ranked[1] if len(ranked) > 1 else t1_c
        t3_c = ranked[2] if len(ranked) > 2 else t2_c

        # Sort targets in ascending order of price to ensure t1 <= t2 <= t3
        sorted_pairs = sorted([(t1_c.consensus_price, t1_c), (t2_c.consensus_price, t2_c), (t3_c.consensus_price, t3_c)], key=lambda x: x[0])

        return {
            "t1": sorted_pairs[0][0], "t2": sorted_pairs[1][0], "t3": sorted_pairs[2][0],
            "t1_cluster": sorted_pairs[0][1], "t2_cluster": sorted_pairs[1][1], "t3_cluster": sorted_pairs[2][1]
        }

class MeanReversionStrategy(TargetStrategy):
    def select_targets(self, clusters: List[ClusteredTarget], entry: float, risk: float, context: dict) -> dict:
        # Reversal simply walks the stack
        return {} # Will be custom implemented in _compute_reversal

class ConflictResolver:
    @staticmethod
    def resolve(
        clusters: List[ClusteredTarget],
        scanner: str = "EOD",
        entry: float = 0.0,
        macro_regime: str = "NEUTRAL",
        risk: float = 0.0,
        eff_atr: float = 0.0,
        *args,
        **kwargs
    ) -> tuple:
        if not clusters:
            return [], "NO_TARGET_CLUSTERS_FOUND"

        policy = TARGET_CONFLICT_POLICY.get(scanner, "CONFIDENCE")
        if policy == "NEAREST":
            resolved = sorted(clusters, key=lambda c: (c.consensus_price, -c.score, c.cluster_id))
        elif policy == "CONFIDENCE":
            resolved = sorted(clusters, key=lambda c: (c.score, c.consensus_price, -c.cluster_id), reverse=True)
        elif policy == "REGIME":
            if macro_regime in ("BULL", "TRENDING"):
                resolved = sorted(clusters, key=lambda c: (c.consensus_price, c.score, -c.cluster_id), reverse=True) # Prefer higher
            else:
                resolved = sorted(clusters, key=lambda c: (c.score, c.consensus_price, -c.cluster_id), reverse=True)
        else:
            resolved = sorted(clusters, key=lambda c: (c.score, c.consensus_price, -c.cluster_id), reverse=True)

        return resolved, None

class ExitPolicy:
    @staticmethod
    def get_profile(scanner: str) -> dict:
        profile_name = SCANNER_EXIT_PROFILE.get(scanner, "BALANCED")
        return EXIT_PROFILES.get(profile_name, EXIT_PROFILES["BALANCED"])




# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val) -> Optional[float]:
    """Return float if valid, finite, and > 0, else None."""
    try:
        f = float(val)
        return f if math.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def _find_swing_low_cluster(swing_lows, threshold_pct: float = 0.01) -> Optional[float]:
    """
    If 2+ swing lows in the list are within threshold_pct (1%) of each other,
    returns the average of the clustering swing lows as the cluster zone level.
    Otherwise returns None.
    """
    if swing_lows is None or len(swing_lows) < 2:
        return None
    n = len(swing_lows)
    best_cluster = []
    for i in range(n):
        for j in range(i + 1, n):
            val1 = float(swing_lows[i])
            val2 = float(swing_lows[j])
            diff = abs(val1 - val2) / max(val1, val2, 1e-5)
            if diff <= threshold_pct:
                cluster = [val1, val2]
                for k in range(n):
                    if k != i and k != j:
                        val3 = float(swing_lows[k])
                        if abs(val3 - val1) / max(val3, val1, 1e-5) <= threshold_pct and abs(val3 - val2) / max(val3, val2, 1e-5) <= threshold_pct:
                            cluster.append(val3)
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster
    if len(best_cluster) >= 2:
        return min(best_cluster)
    return None



def _pick_resistance(
    entry: float,
    swing_high: Optional[float],
    r1: Optional[float],
    bb_upper: Optional[float],
    swing_high_raw: Optional[float],
    r2: Optional[float],
) -> tuple[Optional[float], str]:
    """
    Nearest structural resistance above entry.
    Priority: true pivot swing high > R1 > BB_UPPER > rolling high > R2.
    """
    for level, label in [
        (swing_high,     "pivot swing high"),
        (r1,             "pivot R1"),
        (bb_upper,       "BB upper band"),
        (swing_high_raw, "rolling swing high"),
        (r2,             "pivot R2"),
    ]:
        v = _safe(level)
        if v is not None and v > entry:
            return v, label
    return None, "none"


# [SL_HELPER_FIX_v1.0] BUG-12 FIX: _pick_support, _sl_from_support, _compute_structural_failure_stop
# were called throughout _compute_reversal and _compute_intraday but NEVER DEFINED anywhere in this file.
# This caused NameError for every REVERSAL and INTRADAY SL computation reaching those code paths.
# Added these three helper functions now, mirroring the mirror-image logic of _pick_resistance.

def _pick_support(
    entry: float,
    swing_low: Optional[float],
    s1: Optional[float],
    swing_low_raw: Optional[float],
    s2: Optional[float],
    swing_low_cluster: Optional[float] = None,
) -> tuple:
    """
    Nearest structural support BELOW entry.
    Priority: cluster zone > true pivot swing low > S1 > rolling swing low > S2.
    Returns (support_price, label) or (None, 'none').
    """
    for level, label in [
        (swing_low_cluster, "swing low cluster"),
        (swing_low,         "pivot swing low"),
        (s1,                "pivot S1"),
        (swing_low_raw,     "rolling swing low"),
        (s2,                "pivot S2"),
    ]:
        v = _safe(level)
        if v is not None and v < entry:
            return v, label
    return None, "none"


def _sl_from_support(
    entry: float,
    support: float,
    eff_atr: float,
    sl_atr_buf: float,
    sl_pct_buf: float,
    max_sl_atr: float,
    sup_label: str,
) -> tuple:
    """
    Calculate stop loss from a support level with anti-trap buffer.
    Buffer = max(sl_atr_buf * ATR, sl_pct_buf * entry).
    Hard-caps so SL never exceeds max_sl_atr × ATR from entry.
    Returns (raw_sl, sl_method) as a tuple.
    """
    buf = max(sl_atr_buf * eff_atr, sl_pct_buf * entry)
    raw_sl = support - buf
    # Hard cap
    min_allowed = entry - max_sl_atr * eff_atr
    raw_sl = max(raw_sl, min_allowed)
    sl_method = f"{sup_label} ₹{round(support, 2)} buffer ₹{round(buf, 2)}"
    return raw_sl, sl_method


def _compute_structural_failure_stop(
    primary_sl: float,
    eff_atr: float,
    supports: list,
) -> Optional[float]:
    """
    Secondary disaster stop — the next meaningful support below the primary SL.
    Mirrors _compute_disaster_stop but accepts a flat list of Optional[float] supports.
    Returns None if no lower support is found.
    """
    lower = [s for s in supports if s is not None and s < primary_sl]
    if not lower:
        return None
    return round(max(lower) - 0.5 * eff_atr, 2)


def _atr_volatility_scale(atr_pct: Optional[float], base: float) -> float:
    m = base
    if atr_pct is not None:
        if   atr_pct > 6.0: m *= 1.6
        elif atr_pct > 4.0: m *= 1.4
        elif atr_pct > 2.0: m *= 1.2
    return round(m, 3)

def _cap_target(
    target: float,
    entry: float,
    eff_atr: float,
    timeframe: str,
    macro_regime: str = "NEUTRAL",
    atr_pct: Optional[float] = None,
) -> float:
    regime_caps = ADAPTIVE_TARGET_CAPS.get(macro_regime, ADAPTIVE_TARGET_CAPS["NEUTRAL"])
    max_atr_mult = regime_caps.get(timeframe, regime_caps["1d"])
    if atr_pct is not None:
        if atr_pct > 4.0:
            max_atr_mult = min(max_atr_mult, 8.0)
        elif atr_pct < 2.0:
            max_atr_mult = max(max_atr_mult, 10.0)
    max_target = entry + max_atr_mult * eff_atr
    return min(target, max_target)

class ResistanceSelector:
    @staticmethod
    def get_nearest_valid_resistance(entry: float, resistances: list) -> dict:
        from config import STRUCTURAL_RESISTANCE_SCORES

        valid = []
        for val, name, _ in resistances:
            if val is not None and val > entry:
                score = STRUCTURAL_RESISTANCE_SCORES.get(name, 15)
                valid.append({
                    "price": val,
                    "type": name,
                    "score": score
                })

        if not valid:
            return None

        # Filter out weak levels
        MIN_RESISTANCE_SCORE = 25
        strong = [r for r in valid if r["score"] >= MIN_RESISTANCE_SCORE]

        if not strong:
            return None

        # Rank by proximity (nearest valid resistance)
        strong.sort(key=lambda x: x["price"])
        return strong[0]

class SupportEngine:
    @staticmethod
    def calculate_support_strength(cluster: list) -> tuple[int, list]:
        from config import STRUCTURAL_STOP
        scores_dict = STRUCTURAL_STOP.get("SCORES", {})
        bonus_overlap = STRUCTURAL_STOP.get("BONUS_OVERLAP", 15)

        cluster_members = []
        base_score_sum = 0
        unique_names = set()

        for val, name in cluster:
            s = scores_dict.get(name, 10)
            cluster_members.append({"type": name, "price": val, "score": s})
            base_score_sum += s
            unique_names.add(name)

        context_score = 0
        if len(unique_names) > 1:
            context_score += bonus_overlap

        total_score = base_score_sum + context_score
        return total_score, cluster_members

    @staticmethod
    def get_ranked_supports(entry: float, eff_atr: float, supports: list) -> list:
        from config import STRUCTURAL_STOP
        valid = []
        for val, name, _ in supports:
            if val is not None and val < entry:
                valid.append((val, name))

        if not valid:
            return []

        max_width = STRUCTURAL_STOP.get("MAX_CLUSTER_WIDTH_ATR", 1.5) * eff_atr
        valid.sort(key=lambda x: x[0], reverse=True)

        clusters = []
        curr_cluster = [valid[0]]
        curr_max = valid[0][0]

        for v, name in valid[1:]:
            if (curr_max - v) <= max_width:
                curr_cluster.append((v, name))
            else:
                clusters.append(curr_cluster)
                curr_cluster = [(v, name)]
                curr_max = v
        clusters.append(curr_cluster)

        results = []
        for cluster in clusters:
            total_score, members = SupportEngine.calculate_support_strength(cluster)
            weighted_sum = sum([m["price"] * m["score"] for m in members])
            weight_total = sum([m["score"] for m in members])
            best_anchor = weighted_sum / weight_total if weight_total > 0 else members[0]["price"]

            c_str = "STRONG" if total_score > 60 else ("WEAK" if total_score < 30 else "NORMAL")
            cluster_width = max([m["price"] for m in members]) - min([m["price"] for m in members]) if members else 0.0

            results.append({
                "score": total_score,
                "anchor_price": round(best_anchor, 2),
                "cluster_strength": c_str,
                "anchor_confidence": min(round(total_score / 100.0, 2), 1.0),
                "cluster_width": round(cluster_width, 2),
                "member_count": len(members),
                "cluster_members": members
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

def _compute_structural_stop(entry: float, eff_atr: float, atr_pct: float, supports: list, ctx: dict) -> dict:
    from config import MIN_STOP_PCT
    mode = ctx.get("mode", "EOD")
    telemetry_ctx = ctx.get("telemetry_ctx", None)
    min_stop_pct = MIN_STOP_PCT.get(mode, 0.0)

    ranked_supports = SupportEngine.get_ranked_supports(entry, eff_atr, supports)

    best_support = None
    best_buf = 0.0
    best_vol_label = ""
    best_qual_label = ""
    best_final_mult = 1.0

    atr_p = atr_pct or 3.0
    if atr_p < 2.0:
        base_mult = 0.5
        vol_label = "LOW_VOL"
    elif atr_p > 6.0:
        base_mult = 1.0
        vol_label = "HIGH_VOL"
    else:
        base_mult = 0.75
        vol_label = "NORM_VOL"

    # [VERSION: BUSINESS_LOGIC_FIX_v1.0] Tight Stop Rejection Fix
    is_tight_stop = False

    if ranked_supports:
        support_data = ranked_supports[0]
        best_score = support_data["score"]
        if best_score > 60:
            final_mult = base_mult * 0.8
            qual_label = "STRONG_SUP"
        elif best_score < 30:
            final_mult = base_mult * 1.2
            qual_label = "WEAK_SUP"
        else:
            final_mult = base_mult
            qual_label = "NORM_SUP"

        buf = final_mult * eff_atr
        raw_sl = support_data["anchor_price"] - buf
        sl_pct = (entry - raw_sl) / entry * 100 if entry > 0 else 0

        best_support = support_data
        best_buf = buf
        best_vol_label = vol_label
        best_qual_label = qual_label
        best_final_mult = final_mult

        if sl_pct < min_stop_pct:
            is_tight_stop = True

    if not best_support:
        # Explicitly reject if no structural stop meets MIN_STOP_PCT

        # Find best pct observed for metadata
        best_observed_pct = 0.0
        for support_data in ranked_supports:
            buf = eff_atr * 0.75 # approx
            sl_pct = (entry - (support_data["anchor_price"] - buf)) / entry * 100
            if sl_pct > best_observed_pct:
                best_observed_pct = sl_pct

        return {
            "is_valid": False,
            "rejection_reason": "NO_VALID_STRUCTURAL_STOP",
            "details": {
                "clusters_found": len(ranked_supports),
                "best_stop_pct": round(best_observed_pct, 2),
                "required_stop_pct": min_stop_pct
            },
            "raw_sl": entry - (eff_atr * 1.5),
            "sl_method": "REJECTED_TIGHT_STOP",
            "anchor_price": entry,
            "anchor_type": "NONE",
            "anchor_score": 0,
            "anchor_confidence": 0.0,
            "cluster_width": 0.0,
            "member_count": 0,
            "cluster_members": [],
            "buffer_value": eff_atr * 1.5,
            "buffer_method": "REJECTED"
        }

    best_anchor = best_support["anchor_price"]
    best_score = best_support["score"]
    best_cluster_members = best_support["cluster_members"]
    best_names = "_".join(list(dict.fromkeys([m["type"] for m in best_cluster_members]))).upper().replace(" ", "_")

    method_str = f"{best_names} (Score: {best_score}) @ {best_anchor:.2f} — Buffer {best_buf:.2f} ({best_final_mult:.2f}x ATR)"
    if is_tight_stop:
        method_str = "TIGHT_STRUCT_" + method_str
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"TIGHT STRUCTURE | Entry: {entry:.2f} | Structure: {best_anchor:.2f} | Stop %: {sl_pct:.2f} | Accepted: YES | Reason: TIGHT_STRUCTURE")


    if telemetry_ctx:
        telemetry_ctx.capture_value("STRUCTURAL_STOP_RAW", best_anchor)
        telemetry_ctx.capture_value("STRUCTURAL_STOP_BUFFER", round(best_buf, 2))
        telemetry_ctx.capture_value("STRUCTURAL_STOP_FINAL", best_anchor - best_buf)
        telemetry_ctx.capture_value("STRUCTURAL_STOP_SCORE", best_score)
        telemetry_ctx.capture_value("STRUCTURAL_STOP_METHOD", method_str)
        if is_tight_stop:
            telemetry_ctx.capture_value("STRUCTURAL_STOP_IS_TIGHT", True)

    return {
        "is_valid": True,
        "raw_sl": best_anchor - best_buf,
        "sl_method": method_str,
        "anchor_price": best_anchor,
        "anchor_type": best_names,
        "anchor_score": best_score,
        "anchor_confidence": best_support["anchor_confidence"],
        "cluster_width": best_support["cluster_width"],
        "member_count": best_support["member_count"],
        "cluster_members": best_cluster_members,
        "buffer_value": round(best_buf, 2),
        "buffer_method": f"{best_vol_label}_{best_qual_label}",
        "is_tight_stop": is_tight_stop,
        "tight_stop_pct": round(sl_pct, 2) if is_tight_stop else None,
        "min_stop_pct": min_stop_pct
    }

def _compute_structural_failure_stop(primary_sl: float, eff_atr: float, lower_supports: list) -> float:

    """
    v6.2.1: Nearest Lower Major Support -> Exists? -> YES -> Use it -> NO -> Primary - ATR
    """
    valid = [s for s in lower_supports if _safe(s) is not None and s < primary_sl]
    nearest_lower = max(valid) if valid else None

    if nearest_lower is not None:
        return round(nearest_lower, 2)

    fallback = primary_sl - (1.0 * eff_atr)
    return round(fallback, 2)

def _compute_target_quality(
    natural_rr: float,
    rsi: Optional[float],
    adx: Optional[float],
    macd_hist: Optional[float],
    volume_ratio: Optional[float],
    swing_high: Optional[float],
    r1: Optional[float],
    r2: Optional[float],
    bb_upper: Optional[float]
) -> tuple[int, dict]:
    """
    v6.0: Explainable Target Quality Score.
    Weighting:
    - Natural RR (40%)
    - Trend (20%)
    - Volume (15%)
    - Resistance Proximity (15%)
    - Liquidity (10%)
    """
    bd = {"natural_rr": 0, "trend": 0, "volume": 0, "resistance": 0, "liquidity": 0}

    # 1. Natural RR (40 pts max)
    if natural_rr >= 4.0: bd["natural_rr"] = 40
    elif natural_rr >= 3.0: bd["natural_rr"] = 35
    elif natural_rr >= 2.0: bd["natural_rr"] = 25
    elif natural_rr >= 1.5: bd["natural_rr"] = 15
    else: bd["natural_rr"] = 5

    # 2. Trend / Momentum (20 pts max)
    v_adx = _safe(adx)
    if v_adx:
        if v_adx > 35: bd["trend"] += 12
        elif v_adx > 25: bd["trend"] += 8
        elif v_adx > 20: bd["trend"] += 4

    v_macd = _safe(macd_hist)
    if v_macd and v_macd > 0:
        bd["trend"] += 8

    # 3. Volume Expansion (15 pts max)
    v_vol = _safe(volume_ratio)
    if v_vol:
        if v_vol > 3.0: bd["volume"] = 15
        elif v_vol > 2.0: bd["volume"] = 12
        elif v_vol > 1.5: bd["volume"] = 8
        elif v_vol > 1.0: bd["volume"] = 4

    # 4. Resistance Proximity (15 pts max)
    # Check if we have multiple resistance levels stacked
    resistances = [r for r in [swing_high, r1, r2, bb_upper] if _safe(r) is not None]
    if len(resistances) == 0:
        bd["resistance"] = 15  # Blue sky
    elif len(resistances) == 1:
        bd["resistance"] = 10  # Single hurdle
    else:
        bd["resistance"] = 5   # Heavy overhead

    # 5. Liquidity / Delivery (10 pts) -> placeholder since we don't pass delivery % yet,
    # we'll give a baseline based on RSI not being overbought
    v_rsi = _safe(rsi)
    if v_rsi:
        if 55 <= v_rsi <= 72: bd["liquidity"] = 10
        elif 40 <= v_rsi < 55: bd["liquidity"] = 7
        else: bd["liquidity"] = 3

    total_score = sum(bd.values())
    return total_score, bd


def _rsi_zone(rsi: Optional[float]) -> str:
    v = _safe(rsi)
    if v is None:       return "neutral"
    if v > 72:          return "overbought"
    if v > 55:          return "bullish"
    if v > 40:          return "neutral"
    return "oversold"


# ─────────────────────────────────────────────────────────────────────────────
# EOD — Daily Breakout (swing trade, hold days to weeks)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_multi_tf_v2(entry: float, eff_atr: float, ticker: pd.DataFrame = None, **kwargs) -> dict:
    """
    MULTI_TF_V2 Target Engine.
    T0 = nearest structural resistance / first obstacle / scale-out zone.
    T1 = next structural resistance offering >= 1.5R, or measured move / fib extension.
    Risk = entry - box_low (from kwargs).
    R:R Gate = T1 >= 1.5R.
    T2/T3 = Fib extensions of the T1 target leg.
    Target Provenance: STRUCTURAL, MEASURED_MOVE, FIB_EXTENSION.
    """
    # 1. Base Stop Loss (box_low passed via kwargs, fallback to 1 ATR)
    box_low = kwargs.get("box_low", entry - eff_atr)
    risk_points = max(entry - box_low, eff_atr)

    sl = box_low - (0.10 * eff_atr)  # Buffer below structure
    sl = round(sl, 2)
    effective_risk = max(0.01, entry - sl)
    risk_pct = (entry - sl) / entry * 100

    # 2. Structural Levels Search
    levels = []
    if ticker is not None and not ticker.empty:
        last = ticker.iloc[-1]
        for col in ["LOOKBACK_SWING_HIGH", "R1", "R2", "HIGH_252D"]:
            if col in last and not pd.isna(last[col]):
                try:
                    val = float(last[col])
                    if val > entry:
                        levels.append(val)
                except (ValueError, TypeError):
                    pass

    sorted_levels = sorted(list(set(levels))) if levels else []

    # T0 is the nearest overhead resistance (first obstacle / minor scale-out)
    if sorted_levels:
        t0 = round(sorted_levels[0], 2)
        t0_rr = round((t0 - entry) / effective_risk, 2)
    else:
        t0 = round(entry + (effective_risk * 1.0), 2)
        t0_rr = 1.0

    # T1 is the true tradeability target providing >= 1.5R
    # Search for first structural level that delivers >= 1.5R
    t1_structural_candidates = [lvl for lvl in sorted_levels if (lvl - entry) / effective_risk >= 1.5]

    if t1_structural_candidates:
        t1 = round(min(t1_structural_candidates), 2)
        target_basis = "Structural_Target_Post_T0"
        t1_source = "STRUCTURAL"
    else:
        # If no structural level satisfies >= 1.5R (e.g. blue sky or tight ceiling),
        # use a 2.0R measured move of the base structure
        t1 = round(entry + (effective_risk * 2.0), 2)
        target_basis = "2R_Measured_Move"
        t1_source = "MEASURED_MOVE"

    rr = (t1 - entry) / effective_risk

    # 3. Validation Gate via TradeStructureValidator
    validation = TradeStructureValidator.validate(
        entry=entry,
        stop_loss=sl,
        target_1=t1,
        min_rr=1.5
    )

    is_rejected = not validation.get("is_valid", False)

    # 4. Fib Extensions for T2/T3 based on the T1 structure
    t1_dist = max(t1 - entry, 0.01)
    t2 = round(entry + (t1_dist * 1.618), 2)
    t3 = round(entry + (t1_dist * 2.618), 2)

    return {
        "entry": entry,
        "entry_price": entry,
        "stop_loss": sl,
        "target_0": t0,
        "target_1": t1,
        "target_2": t2,
        "target_3": t3,
        "target": t1,  # Primary target is T1
        "t0_rr_ratio": t0_rr,
        "rr_ratio": round(rr, 2),
        "risk_pct": round(risk_pct, 2),
        "sl_basis": "Box_Low_Structure",
        "target_basis": target_basis,
        "t1_source": t1_source,
        "is_rejected": is_rejected,
        "rejection_code": validation.get("rejection_code", "") if is_rejected else ""
    }


def _compute_multi_tf(entry: float, eff_atr: float, atr_pct: float, adx: float, rsi: float, macd_hist: float, swing_low: float, swing_high: float, s1: float, s2: float, r1: float, r2: float, swing_low_raw: float, swing_high_raw: float, ticker=None, **kwargs) -> dict:
    supports = [
        (swing_low, "5m Swing Low", 20),
        (kwargs.get("swing_low_15m"), "15m Swing Low", 25),
        (kwargs.get("swing_low_30m"), "30m Swing Low", 30),
        (kwargs.get("swing_low_1h"), "1H Swing Low", 35),
        (s1, "S1", 20),
        (s2, "S2", 15),
        (swing_low_raw, "Rolling Swing Low", 20),
        (kwargs.get("vwap"), "VWAP", 15),
        (kwargs.get("ema20"), "EMA20", 15),
        (kwargs.get("sma50"), "SMA50", 15),
        (kwargs.get("sma200"), "SMA200", 30)
    ]
    sl_data = _compute_structural_stop(entry, eff_atr, atr_pct, supports, {"mode": "MULTI_TF"})
    if not sl_data.get("is_valid", True):
        return {
            "engine_version": "SL_ENGINE_V7", "is_rejected": True, "rejection_reason": "NO_VALID_STRUCTURAL_STOP",
            "gate": "MIN_STOP_PCT", "actual": sl_data.get("details", {}).get("best_stop_pct", 0.0),
            "required": sl_data.get("details", {}).get("required_stop_pct", 0.0), "context": sl_data.get("details", {}),
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }

    macro_regime = kwargs.get("macro_regime", "NEUTRAL")
    gen = CandidateGenerator()
    candidates = gen.generate_breakout_candidates(
        entry=entry, eff_atr=eff_atr, atr_pct=atr_pct, adx=adx, volume_ratio=kwargs.get("volume_ratio", 1.0),
        vwap=kwargs.get("vwap"), macro_regime=macro_regime, scanner="MULTI_TF",
        swing_low=swing_low, swing_high=swing_high, swing_low_raw=swing_low_raw, swing_high_raw=swing_high_raw,
        r1=r1, r2=r2, bb_upper=kwargs.get("bb_upper"), prior_20d_high=kwargs.get("prior_20d_high"),
        high_52w=kwargs.get("high_52w"), prev_day_high=kwargs.get("prev_day_high"), ticker=ticker
    )

    risk = abs(entry - sl_data["raw_sl"])
    clusters = ClusterEngine.cluster(candidates, entry, eff_atr)
    RoundNumberEngine.detect_and_boost(clusters)
    clusters, rejection_reason = ConflictResolver.resolve(clusters, "MULTI_TF", entry, macro_regime, risk, eff_atr)

    if not clusters:
        return {
            "engine_version": "SL_ENGINE_V7.3", "is_rejected": True,
            "rejection_reason": rejection_reason,
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }

    strategy = TrendExtensionStrategy()
    targets = strategy.select_targets(clusters, entry, risk, {})

    pool = []
    for c in candidates:
        c_dict = vars(c).copy()
        c_dict["source"] = c.source.name
        if targets and targets.get("t1_cluster") and c.cluster_id == targets["t1_cluster"].cluster_id:
            c_dict["selection_state"] = "WINNING"
        elif targets and ( (targets.get("t2_cluster") and c.cluster_id == targets["t2_cluster"].cluster_id) or (targets.get("t3_cluster") and c.cluster_id == targets["t3_cluster"].cluster_id) ):
            c_dict["selection_state"] = "SELECTED"
        else:
            c_dict["selection_state"] = "REJECTED"
        pool.append(c_dict)

    if sl_data["raw_sl"] >= entry:
        return {
            "engine_version": "SL_ENGINE_V7.1", "is_rejected": True,
            "rejection_reason": f"INVALID_STOP_PLACEMENT (Stop Loss ₹{sl_data['raw_sl']:.2f} >= Entry Price ₹{entry:.2f})",
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }
    min_rr = MIN_NATURAL_RR.get("MULTI_TF", 1.5)
    risk_amount = entry - sl_data["raw_sl"]

    valid_targets = []
    for t_cand_key, t_clust_key in [("t1", "t1_cluster"), ("t2", "t2_cluster"), ("t3", "t3_cluster")]:
        cand_t = targets.get(t_cand_key)
        if cand_t and cand_t > entry and risk_amount > 0:
            rr_candidate = round(abs(cand_t - entry) / risk_amount, 2)
            if rr_candidate >= min_rr:
                valid_targets.append((cand_t, targets.get(t_clust_key)))

    if not valid_targets:
        t1_fallback = targets.get("t1", entry)
        natural_rr_val = round(abs(t1_fallback - entry) / risk_amount, 2) if risk_amount > 0 else 0.0
        return {
            "engine_version": "SL_ENGINE_V7.1", "is_rejected": True,
            "rejection_reason": f"NO_VALID_STRUCTURAL_TARGET (Min RR: {min_rr}x, Actual: {natural_rr_val}x)",
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": natural_rr_val, "sl_result": sl_data
        }

    # Sort valid targets in ascending order of price to guarantee t1 <= t2 <= t3
    valid_targets.sort(key=lambda x: x[0])

    t1 = valid_targets[0][0]
    t1_clust = valid_targets[0][1]
    t1_src = t1_clust.candidates[0].source.name if t1_clust and t1_clust.candidates else "UNKNOWN"
    natural_rr_val = round(abs(t1 - entry) / risk_amount, 2)

    t2 = valid_targets[1][0] if len(valid_targets) > 1 else None
    t3 = valid_targets[2][0] if len(valid_targets) > 2 else None

    tq_score, _ = _compute_target_quality(
        natural_rr_val, kwargs.get("rsi"), kwargs.get("adx"), kwargs.get("macd_hist"),
        kwargs.get("volume_ratio"), swing_high, r1, r2, kwargs.get("bb_upper")
    )
    s_f_s = _compute_structural_failure_stop(sl_data["raw_sl"], eff_atr, [s[0] for s in supports])

    explanation = targets.get("t1_cluster").analysis.explanation if targets and targets.get("t1_cluster") and getattr(targets.get("t1_cluster"), "analysis", None) else {}
    def _r2(v):
        return round(float(v), 2) if v is not None else None

    return {
        "engine_version": "SL_ENGINE_V7", "stop_loss": _r2(sl_data["raw_sl"]),
        "target_1": _r2(t1), "target_2": _r2(t2), "target_3": _r2(t3), "target_4": _r2(targets.get("t4")),
        "structural_failure_stop": _r2(s_f_s),
        "target_quality": tq_score,
        "natural_rr": natural_rr_val,
        "sl_method": sl_data["sl_method"], "t_method": f"TrendExtension [T1:{t1_src}]",
        "sl_result": {"target_candidate_pool": pool, "t1_source": t1_src, "explanation": explanation}
    }

def _compute_eod(entry: float, eff_atr: float, atr_pct: float, adx: float, rsi: float, macd_hist: float, swing_low: float, swing_high: float, s1: float, s2: float, r1: float, r2: float, swing_low_raw: float, swing_high_raw: float, ticker=None, **kwargs) -> dict:
    mode = kwargs.get("mode", "EOD")
    supports = [
        (swing_low, "True Swing Low", 40), (s1, "S1 Pivot", 20), (s2, "S2 Pivot", 15),
        (swing_low_raw, "Rolling Low", 20), (kwargs.get("sma50"), "SMA50", 15), (kwargs.get("sma200"), "SMA200", 30)
    ]
    sl_data = _compute_structural_stop(entry, eff_atr, atr_pct, supports, {"mode": mode})
    if not sl_data.get("is_valid", True):
        return {
            "engine_version": "SL_ENGINE_V7", "is_rejected": True, "rejection_reason": "NO_VALID_STRUCTURAL_STOP",
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }

    macro_regime = kwargs.get("macro_regime", "NEUTRAL")
    gen = CandidateGenerator()
    candidates = gen.generate_breakout_candidates(
        entry=entry, eff_atr=eff_atr, atr_pct=atr_pct, adx=adx, volume_ratio=kwargs.get("volume_ratio", 1.0),
        vwap=kwargs.get("vwap"), macro_regime=macro_regime, scanner="EOD",
        swing_low=swing_low, swing_high=swing_high, swing_low_raw=swing_low_raw, swing_high_raw=swing_high_raw,
        r1=r1, r2=r2, bb_upper=kwargs.get("bb_upper"), prior_20d_high=kwargs.get("prior_20d_high"),
        high_52w=kwargs.get("high_52w"), prev_day_high=kwargs.get("prev_day_high"), ticker=ticker
    )

    risk = abs(entry - sl_data["raw_sl"])
    clusters = ClusterEngine.cluster(candidates, entry, eff_atr)
    RoundNumberEngine.detect_and_boost(clusters)
    clusters, rejection_reason = ConflictResolver.resolve(clusters, "EOD", entry, macro_regime, risk, eff_atr)

    if not clusters:
        return {
            "engine_version": "SL_ENGINE_V7.3", "is_rejected": True,
            "rejection_reason": rejection_reason,
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }

    strategy = ClusterConsensusStrategy()
    targets = strategy.select_targets(clusters, entry, risk, {})

    pool = []
    for c in candidates:
        c_dict = vars(c).copy()
        c_dict["source"] = c.source.name
        if targets and targets.get("t1_cluster") and c.cluster_id == targets["t1_cluster"].cluster_id:
            c_dict["selection_state"] = "WINNING"
        elif targets and ( (targets.get("t2_cluster") and c.cluster_id == targets["t2_cluster"].cluster_id) or (targets.get("t3_cluster") and c.cluster_id == targets["t3_cluster"].cluster_id) ):
            c_dict["selection_state"] = "SELECTED"
        else:
            c_dict["selection_state"] = "REJECTED"
        pool.append(c_dict)

    if sl_data["raw_sl"] >= entry:
        return {
            "engine_version": "SL_ENGINE_V7.1", "is_rejected": True,
            "rejection_reason": f"INVALID_STOP_PLACEMENT (Stop Loss ₹{sl_data['raw_sl']:.2f} >= Entry Price ₹{entry:.2f})",
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }
    min_rr = MIN_NATURAL_RR.get(mode, 2.0)
    risk_amount = entry - sl_data["raw_sl"]

    valid_targets = []
    for t_cand_key, t_clust_key in [("t1", "t1_cluster"), ("t2", "t2_cluster"), ("t3", "t3_cluster")]:
        cand_t = targets.get(t_cand_key)
        if cand_t and cand_t > entry and risk_amount > 0:
            rr_candidate = round(abs(cand_t - entry) / risk_amount, 2)
            if rr_candidate >= min_rr:
                valid_targets.append((cand_t, targets.get(t_clust_key)))

    if not valid_targets:
        t1_fallback = targets.get("t1", entry)
        natural_rr_val = round(abs(t1_fallback - entry) / risk_amount, 2) if risk_amount > 0 else 0.0
        return {
            "engine_version": "SL_ENGINE_V7.1", "is_rejected": True,
            "rejection_reason": f"NO_VALID_STRUCTURAL_TARGET (Min RR: {min_rr}x, Actual: {natural_rr_val}x)",
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": natural_rr_val, "sl_result": sl_data
        }

    # Sort valid targets in ascending order of price to guarantee t1 <= t2 <= t3
    valid_targets.sort(key=lambda x: x[0])

    t1 = valid_targets[0][0]
    t1_clust = valid_targets[0][1]
    t1_src = t1_clust.candidates[0].source.name if t1_clust and t1_clust.candidates else "UNKNOWN"
    natural_rr_val = round(abs(t1 - entry) / risk_amount, 2)

    t2 = valid_targets[1][0] if len(valid_targets) > 1 else None
    t3 = valid_targets[2][0] if len(valid_targets) > 2 else None

    tq_score, _ = _compute_target_quality(
        natural_rr_val, kwargs.get("rsi"), kwargs.get("adx"), kwargs.get("macd_hist"),
        kwargs.get("volume_ratio"), swing_high, r1, r2, kwargs.get("bb_upper")
    )
    s_f_s = _compute_structural_failure_stop(sl_data["raw_sl"], eff_atr, [s[0] for s in supports])

    explanation = targets.get("t1_cluster").analysis.explanation if targets and targets.get("t1_cluster") and getattr(targets.get("t1_cluster"), "analysis", None) else {}
    def _r2(v):
        return round(float(v), 2) if v is not None else None

    return {
        "engine_version": "SL_ENGINE_V7", "stop_loss": _r2(sl_data["raw_sl"]),
        "target_1": _r2(t1), "target_2": _r2(t2), "target_3": _r2(t3), "target_4": _r2(targets.get("t4")),
        "structural_failure_stop": _r2(s_f_s),
        "target_quality": tq_score,
        "natural_rr": natural_rr_val,
        "sl_method": sl_data["sl_method"], "t_method": f"ClusterConsensus [T1:{t1_src}]",
        "sl_result": {"target_candidate_pool": pool, "t1_source": t1_src, "explanation": explanation}
    }

def _compute_pullback(entry: float, eff_atr: float, **kwargs) -> dict:
    """Canonical v5.1.2 PULLBACK stop and target geometry caller."""
    from engine.analytics.pullback_geometry import calculate_pullback_sl_target
    geom = calculate_pullback_sl_target(entry, eff_atr)
    sl_val = geom["stop_loss"]
    t1_val = geom["target_price"]
    risk = geom["actual_risk"]
    t2_val = round(entry + 3.5 * risk, 2)
    t3_val = round(entry + 5.0 * risk, 2)
    t4_val = round(entry + 7.0 * risk, 2)

    return {
        "engine_version": "v5.1.2_ADAPTIVE_ATR",
        "is_rejected": False,
        "rejection_reason": "",
        "stop_loss": sl_val,
        "target_1": t1_val,
        "target_2": t2_val,
        "target_3": t3_val,
        "target_4": t4_val,
        "structural_failure_stop": sl_val,
        "target_quality": 85.0,
        "natural_rr": geom["natural_rr"],
        "sl_method": "Adaptive ATR14 Clamped [3.5%, 6.0%]",
        "t_method": "Execution Risk 2.5R Target",
        "atr_14": eff_atr,
        "risk_amount": risk,
        "clamped_stop_pct": geom["clamped_stop_pct"],
        "sl_result": {"mode": "PULLBACK", "geometry": geom}
    }

def _compute_reversal(entry: float, eff_atr: float, atr_pct: float, adx: float, rsi: float, macd_hist: float, swing_low: float, swing_high: float, s1: float, s2: float, r1: float, r2: float, swing_low_raw: float, swing_high_raw: float, ticker=None, **kwargs) -> dict:
    supports = [
        (swing_low, "True Swing Low", 40), (s1, "S1 Pivot", 20), (s2, "S2 Pivot", 15),
        (swing_low_raw, "Rolling Low", 20)
    ]
    sl_data = _compute_structural_stop(entry, eff_atr, atr_pct, supports, {"mode": "REVERSAL"})
    if not sl_data.get("is_valid", True):
        return {
            "engine_version": "SL_ENGINE_V7", "is_rejected": True, "rejection_reason": "NO_VALID_STRUCTURAL_STOP",
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }

    # Mean reversion stack
    cands = []
    prior_high = swing_high_raw if _safe(swing_high_raw) else entry + 5 * eff_atr
    decline = prior_high - entry if prior_high > entry else 0

    if _safe(kwargs.get("bb_mid")): cands.append(TargetCandidate(kwargs.get("bb_mid"), TargetSource.BB_MID, "any", "REVERSAL", "NORMAL", {}))
    if _safe(kwargs.get("sma50")): cands.append(TargetCandidate(kwargs.get("sma50"), TargetSource.SMA50, "any", "REVERSAL", "NORMAL", {}))
    if decline > 0:
        cands.append(TargetCandidate(entry + decline*0.382, TargetSource.RETRACE_382, "any", "REVERSAL", "NORMAL", {}))
        cands.append(TargetCandidate(entry + decline*0.500, TargetSource.RETRACE_50, "any", "REVERSAL", "NORMAL", {}))
        cands.append(TargetCandidate(entry + decline*0.618, TargetSource.RETRACE_618, "any", "REVERSAL", "NORMAL", {}))
    if _safe(kwargs.get("sma200")): cands.append(TargetCandidate(kwargs.get("sma200"), TargetSource.SMA200, "any", "REVERSAL", "NORMAL", {}))
    if _safe(swing_high_raw): cands.append(TargetCandidate(swing_high_raw, TargetSource.SWING_HIGH_RAW, "any", "REVERSAL", "NORMAL", {}))

    # Filter only above entry
    valid_cands = [c for c in cands if c.price > entry]

    risk = abs(entry - sl_data["raw_sl"])
    clusters = ClusterEngine.cluster(valid_cands, entry, eff_atr)
    RoundNumberEngine.detect_and_boost(clusters)
    clusters, rejection_reason = ConflictResolver.resolve(clusters, "REVERSAL", entry, kwargs.get("macro_regime", "NEUTRAL"), risk, eff_atr)

    if not clusters:
        return {
            "engine_version": "SL_ENGINE_V7.3", "is_rejected": True,
            "rejection_reason": rejection_reason,
            "stop_loss": sl_data["raw_sl"], "target_1": entry, "natural_rr": 0.0, "sl_result": sl_data
        }

    t1_cluster = clusters[0] if clusters else None
    t1 = t1_cluster.consensus_price if t1_cluster else entry + 2*eff_atr
    t2 = clusters[1].consensus_price if len(clusters) > 1 else None
    t3 = clusters[2].consensus_price if len(clusters) > 2 else None

    explanation = getattr(t1_cluster, "analysis", {}).get("explanation") if t1_cluster and hasattr(t1_cluster, "analysis") and isinstance(t1_cluster.analysis, dict) else {}
    natural_rr_val = round(abs(t1 - entry) / risk, 2)
    tq_score, _ = _compute_target_quality(
        natural_rr_val, kwargs.get("rsi"), kwargs.get("adx"), kwargs.get("macd_hist"),
        kwargs.get("volume_ratio"), swing_high, r1, r2, kwargs.get("bb_upper")
    )
    s_f_s = _compute_structural_failure_stop(sl_data["raw_sl"], eff_atr, [s[0] for s in supports])

    def _r2(v):
        return round(float(v), 2) if v is not None else None

    return {
        "engine_version": "SL_ENGINE_V7", "stop_loss": _r2(sl_data["raw_sl"]),
        "target_1": _r2(t1), "target_2": _r2(t2), "target_3": _r2(t3), "target_4": None,
        "structural_failure_stop": _r2(s_f_s),
        "target_quality": tq_score,
        "natural_rr": natural_rr_val,
        "sl_method": sl_data["sl_method"], "t_method": f"MeanReversion [T1]",
        "sl_result": {"target_candidate_pool": [{**vars(c), "source": c.source.name} for c in cands], "explanation": explanation}
    }


def _compute_multibagger(entry: float, eff_atr: float, ticker=None, **kwargs) -> dict:
    """
    Computes Stop Loss & Multibagger Compounder Targets (T1: 1.5x, T2: 2.0x, T3: 3.0x, T4: 5.0x).
    For long-term fundamental positions, SL is set to max(entry * 0.85, entry - 3.0 * eff_atr).
    """
    mode = kwargs.get("mode", "MULTIBAGGER")
    sl_atr = entry - (3.0 * eff_atr)
    sl_pct = entry * 0.85  # 15% structural stop

    # If 200 SMA is available and below entry, use max of 200 SMA and 15% SL
    sma200 = None
    if ticker is not None and "Close" in ticker.columns and len(ticker) >= 200:
        sma200 = float(ticker["Close"].tail(200).mean())

    stop_loss = round(max(sl_pct, sl_atr), 2)
    if sma200 and 0 < sma200 < entry and sma200 > stop_loss:
        stop_loss = round(sma200, 2)

    risk = entry - stop_loss
    t1 = round(entry * 1.50, 2)  # 50% target
    t2 = round(entry * 2.00, 2)  # 100% target (2x)
    t3 = round(entry * 3.00, 2)  # 200% target (3x)
    t4 = round(entry * 5.00, 2)  # 400% target (5x)

    natural_rr = round((t1 - entry) / risk, 2) if risk > 0 else 3.33

    return {
        "engine_version": "SL_ENGINE_V7_MULTIBAGGER",
        "is_rejected": False,
        "stop_loss": stop_loss,
        "initial_stop_loss": stop_loss,
        "target_1": t1,
        "target_2": t2,
        "target_3": t3,
        "target_4": t4,
        "target_price": t1,
        "natural_rr": natural_rr,
        "sl_method": "Multibagger Fundamental Stop (15% / 3x ATR)",
        "t_method": "Multibagger Compounder Targets (1.5x / 2.0x / 3.0x)",
        "sl_result": {}
    }


def _legacy_compute_sl_and_target(
    entry_price:    float,
    atr:            Optional[float],
    candle_range:   float,
    mode:           Optional[str]   = None,     # "EOD" | "INTRADAY" | "LIVE_1H" | "REVERSAL"
    # ── Technical context ──────────────────────────────────────────
    adx:            Optional[float] = None,
    rsi:            Optional[float] = None,
    macd_hist:      Optional[float] = None,
    atr_pct:        Optional[float] = None,
    swing_low:      Optional[float] = None,   # true pivot swing low
    swing_high:     Optional[float] = None,   # true pivot swing high
    bb_upper:       Optional[float] = None,
    bb_lower:       Optional[float] = None,
    bb_mid:         Optional[float] = None,   # used by REVERSAL (mean reversion T1)
    s1:             Optional[float] = None,
    s2:             Optional[float] = None,
    r1:             Optional[float] = None,
    r2:             Optional[float] = None,
    swing_low_raw:  Optional[float] = None,   # rolling window fallback
    swing_high_raw: Optional[float] = None,   # rolling window fallback
    candle_low:     Optional[float] = None,   # used by INTRADAY (bar's own low)
    swing_low_15m:  Optional[float] = None,
    swing_high_15m: Optional[float] = None,
    swing_low_30m:  Optional[float] = None,
    swing_high_30m: Optional[float] = None,
    swing_low_1h:   Optional[float] = None,
    swing_high_1h:  Optional[float] = None,
    ema20:          Optional[float] = None,   # used by REVERSAL (mean reversion T1)
    sma50:          Optional[float] = None,   # used by REVERSAL (mean reversion T2)
    vwap:           Optional[float] = None,   # v5: used by INTRADAY (VWAP-anchored SL)
    # Backward-compat alias (old callers used timeframe=)
    timeframe:      Optional[str]   = None,
    ticker:         Optional[pd.DataFrame] = None,
    telemetry_ctx:  Optional[Any]   = None,
    **kwargs_extra
) -> dict:
    """
    Mode-dispatching SL/Target engine.
    """
    _TIMEFRAME_MAP = {
        "EOD": "EOD", "1d": "EOD",
        "REVERSAL": "REVERSAL",
        "MULTI_TF": "MULTI_TF",
        "MULTI_TF_V2": "MULTI_TF_V2",
        "PULLBACK": "PULLBACK",
        "MULTIBAGGER": "MULTIBAGGER",
        "WEALTH": "MULTIBAGGER",
    }
    effective_mode = (
        _TIMEFRAME_MAP.get(mode or "", "")
        or _TIMEFRAME_MAP.get(timeframe or "", "")
        or "EOD"
    )

    # Resolve effective ATR
    eff_atr = _safe(atr) or (_safe(candle_range) * 1.5 if _safe(candle_range) else None)
    if eff_atr is None or eff_atr <= 0:
        eff_atr = entry_price * 0.015   # last resort: 1.5% of price

    # Calculate swing low cluster zone if ticker is provided
    swing_low_cluster = None
    if ticker is not None and "SWING_LOW" in ticker.columns:
        try:
            recent_lows = ticker["SWING_LOW"].dropna().unique()[-3:]
            swing_low_cluster = _find_swing_low_cluster(recent_lows)
        except Exception:
            pass

    kwargs = dict(
        entry=entry_price, eff_atr=eff_atr,
        adx=adx, rsi=rsi, macd_hist=macd_hist, atr_pct=atr_pct,
        swing_low=swing_low, swing_high=swing_high,
        bb_upper=bb_upper, bb_lower=bb_lower,
        s1=s1, s2=s2, r1=r1, r2=r2,
        swing_low_raw=swing_low_raw, swing_high_raw=swing_high_raw,
        swing_low_cluster=swing_low_cluster,
        ticker=ticker,
        mode=effective_mode,
        telemetry_ctx=telemetry_ctx
    )
    kwargs.update(kwargs_extra)

    if effective_mode == "PULLBACK":
        return _compute_pullback(**kwargs)
    elif effective_mode == "EOD":
        return _compute_eod(**kwargs)
    elif effective_mode == "MULTI_TF_V2":
        return _compute_multi_tf_v2(**kwargs)
    elif effective_mode == "MULTI_TF":
        return _compute_multi_tf(**kwargs)
    elif effective_mode == "REVERSAL":
        return _compute_reversal(**kwargs, ema20=ema20, bb_mid=bb_mid, sma50=sma50)
    elif effective_mode == "MULTIBAGGER":
        return _compute_multibagger(**kwargs)
    else:
        return _compute_eod(**kwargs)  # safe default


# =====================================================================================
# V2.0 INSTITUTIONAL ENGINE ARCHITECTURE
# =====================================================================================

ENGINE_V2_CONFIG = {
    "SUPPORT_WEIGHTS": {
        "touches": 40,
        "volume": 30,
        "age": 20,
        "proximity": 10
    },
    "TARGET_WEIGHTS": {
        "swing_high": 25,
        "fib": 20,
        "measured_move": 20,
        "vwap": 15,
        "atr": 10,
        "volume_profile": 10
    },
    "TRADE_QUALITY_WEIGHTS": {
        "trend": 25,
        "momentum": 20,
        "volume": 20,
        "support": 15,
        "rs": 10,
        "market": 10
    },
    "VOLATILITY_WEIGHTS": {
        "atr_percentile": 40,
        "hv_percentile": 40,
        "gap_frequency": 20
    },
    "PARTIAL_EXITS": {
        "t1": "25%",
        "t2": "35%",
        "t3": "40%"
    }
}

class SupportConfidenceEngine:
    @staticmethod
    def calculate(kwargs: dict) -> dict:
        breakdown = {"touches": 10, "volume": 15, "age": 10, "proximity": 5}

        entry = kwargs.get("entry_price", 1.0)
        support = kwargs.get("swing_low") or (entry * 0.95)

        # Proximity
        vwap = kwargs.get("vwap")
        if vwap and abs(vwap - support) / max(support, 1) < 0.02:
            breakdown["proximity"] += 15

        ema20 = kwargs.get("ema20")
        if ema20 and abs(ema20 - support) / max(support, 1) < 0.02:
            breakdown["proximity"] += 10

        # Touches & Age (derived from clustering)
        if kwargs.get("swing_low_cluster"):
            breakdown["touches"] += 25
            breakdown["age"] += 10

        # Cap values
        breakdown["touches"] = min(breakdown["touches"], ENGINE_V2_CONFIG["SUPPORT_WEIGHTS"]["touches"])
        breakdown["volume"] = min(breakdown["volume"], ENGINE_V2_CONFIG["SUPPORT_WEIGHTS"]["volume"])
        breakdown["age"] = min(breakdown["age"], ENGINE_V2_CONFIG["SUPPORT_WEIGHTS"]["age"])
        breakdown["proximity"] = min(breakdown["proximity"], ENGINE_V2_CONFIG["SUPPORT_WEIGHTS"]["proximity"])

        score = sum(breakdown.values())
        return {"score": score, "breakdown": breakdown}

class VolatilityRegimeEngine:
    @staticmethod
    def calculate(kwargs: dict) -> str:
        atr_pct = kwargs.get("atr_pct") or 2.0
        if atr_pct > 4.5: return "HIGH"
        if atr_pct < 1.5: return "LOW"
        return "NORMAL"

class TradeQualityEngine:
    @staticmethod
    def calculate(kwargs: dict, support_score: int) -> dict:
        adx = _safe(kwargs.get("adx")) or 20.0
        rsi = _safe(kwargs.get("rsi")) or 50.0

        trend = min(ENGINE_V2_CONFIG["TRADE_QUALITY_WEIGHTS"]["trend"], int((adx / 40.0) * ENGINE_V2_CONFIG["TRADE_QUALITY_WEIGHTS"]["trend"]))
        momentum = min(ENGINE_V2_CONFIG["TRADE_QUALITY_WEIGHTS"]["momentum"], int((rsi / 70.0) * ENGINE_V2_CONFIG["TRADE_QUALITY_WEIGHTS"]["momentum"]))
        support_val = int((support_score / 100.0) * ENGINE_V2_CONFIG["TRADE_QUALITY_WEIGHTS"]["support"])

        breakdown = {
            "trend": trend,
            "volume": 15, # Proxy for now unless we pass in volume explicitly
            "momentum": momentum,
            "support": support_val,
            "rs": 10,
            "market": 10
        }

        score = sum(breakdown.values())
        return {"score": min(100, score), "breakdown": breakdown}

class BaseRiskEngine:
    def __init__(self, mode: str, kwargs: dict):
        self.mode = mode
        self.kwargs = kwargs
        self.entry_price = kwargs.get("entry_price", 0.0)

    def compute_sl(self, support_price: float, support_conf: int, vol_regime: str) -> float:
        eff_atr = self.kwargs.get("eff_atr") or (self.entry_price * 0.015)

        buf_mult = 0.5
        if vol_regime == "HIGH": buf_mult = 1.0
        elif vol_regime == "LOW": buf_mult = 0.3

        if support_conf < 50: buf_mult *= 1.5
        elif support_conf > 80: buf_mult *= 0.7

        adx = _safe(self.kwargs.get("adx")) or 20.0
        if adx > 35: buf_mult *= 1.2

        raw_sl = support_price - (buf_mult * eff_atr)

        # Hard cap SL so it doesn't get un-usably wide
        # [BASE_RISK_ENGINE_FIX_v1.0] BUG-4 FIX: _MODE_CONFIG tuples have 4 elements (indices 0-3).
        # max_sl_atr is at index [3]. Accessing index [4] caused IndexError on every v2 SL computation.
        max_sl_atr = _MODE_CONFIG.get(self.mode.split("_")[0], _DEFAULT_CONFIG)[3]
        min_allowed_sl = self.entry_price - (max_sl_atr * eff_atr)

        raw_sl = max(raw_sl, min_allowed_sl)

        # Ensure SL never goes negative on extreme volatility penny stocks
        return round(max(0.01, raw_sl), 2)

    def compute_targets(self, risk: float, vol_regime: str) -> tuple[dict, dict]:
        entry = self.entry_price
        eff_atr = self.kwargs.get("eff_atr") or (entry * 0.015)

        swing_high = _safe(self.kwargs.get("swing_high")) or (entry + 2 * eff_atr)
        r1 = _safe(self.kwargs.get("r1")) or (entry + 1.5 * eff_atr)
        vwap = _safe(self.kwargs.get("vwap")) or entry

        fib = entry + (swing_high - entry) * 1.618
        cr = _safe(self.kwargs.get("candle_range")) or eff_atr
        measured_move = entry + cr
        atr_proj = entry + 3 * eff_atr

        # Weight Normalization
        raw_weights = ENGINE_V2_CONFIG["TARGET_WEIGHTS"]
        total_w = max(sum(raw_weights.values()), 1e-5)
        norm_w = {k: v / total_w for k, v in raw_weights.items()}

        t1_cand = (
            swing_high * norm_w["swing_high"] +
            fib * norm_w["fib"] +
            measured_move * norm_w["measured_move"] +
            max(vwap, entry*1.01) * norm_w["vwap"] +
            atr_proj * norm_w["atr"] +
            r1 * norm_w["volume_profile"]
        )

        min_rr = 2.0
        if vol_regime == "HIGH": min_rr = 1.5
        elif vol_regime == "LOW": min_rr = 2.5

        t1 = max(t1_cand, entry + min_rr * risk)
        t2 = t1 + 1.5 * risk
        t3 = t1 + 3.0 * risk
        t4 = t1 + 4.5 * risk

        cluster_diagnostics = {
            "swing_high": round(swing_high, 2),
            "fib": round(fib, 2),
            "measured_move": round(measured_move, 2),
            "vwap": round(vwap, 2),
            "atr_proj": round(atr_proj, 2),
            "r1": round(r1, 2),
            "consensus_target": round(t1_cand, 2)
        }

        def _parse_pct(val): return int(str(val).replace('%', ''))
        t1_exit = _parse_pct(ENGINE_V2_CONFIG["PARTIAL_EXITS"]["t1"])
        t2_exit = _parse_pct(ENGINE_V2_CONFIG["PARTIAL_EXITS"]["t2"])
        t3_exit = _parse_pct(ENGINE_V2_CONFIG["PARTIAL_EXITS"]["t3"])
        t4_exit_str = f"{max(0, 100 - (t1_exit + t2_exit + t3_exit))}%"

        targets = {
            "t1": {"price": round(t1, 2), "confidence": "HIGH", "exit": ENGINE_V2_CONFIG["PARTIAL_EXITS"]["t1"]},
            "t2": {"price": round(t2, 2), "confidence": "MEDIUM", "exit": ENGINE_V2_CONFIG["PARTIAL_EXITS"]["t2"]},
            "t3": {"price": round(t3, 2), "confidence": "LOW", "exit": ENGINE_V2_CONFIG["PARTIAL_EXITS"]["t3"]},
            "t4": {"price": round(t4, 2), "confidence": "LOWEST", "exit": t4_exit_str}
        }

        return targets, cluster_diagnostics

    def get_time_stop(self) -> str:
        return "7 trading days"

    def get_trailing_rule(self) -> str:
        adx = _safe(self.kwargs.get("adx")) or 20.0
        if adx > 35: return "EMA20"
        return "Pivot Low"

    def get_historical_stats(self) -> dict:
        return {"win_rate": 0.58, "avg_win": 2.8, "avg_loss": 1.0}

    def generate_metrics(self) -> dict:
        support_metrics = SupportConfidenceEngine.calculate(self.kwargs)
        vol_regime = VolatilityRegimeEngine.calculate(self.kwargs)
        tq_metrics = TradeQualityEngine.calculate(self.kwargs, support_metrics["score"])

        support_price = self.kwargs.get("swing_low") or (self.entry_price * 0.95)

        sl = self.compute_sl(support_price, support_metrics["score"], vol_regime)

        risk_dist = self.entry_price - sl
        risk_pct = (risk_dist / self.entry_price) * 100 if self.entry_price > 0 else 1.0

        tq = tq_metrics["score"]
        if tq >= 90:
            kelly_fraction = 0.5
            max_risk_pct = 1.5
        elif tq >= 70:
            kelly_fraction = 0.3
            max_risk_pct = 1.0
        else:
            kelly_fraction = 0.15
            max_risk_pct = 0.5

        # Hard cap Kelly account risk limit using central ACCOUNT_RISK_BUDGET_PCT and MAX_POSITION_PCT concentration cap
        # [VERSION: PHASE2_SL_TARGET_IMPROVE_v1.0]
        from config import ACCOUNT_RISK_BUDGET_PCT, MAX_POSITION_PCT
        max_risk_pct = min(max_risk_pct, ACCOUNT_RISK_BUDGET_PCT)

        raw_position_size = round(max_risk_pct / (risk_pct / 100.0), 2) if risk_pct > 0 else 0.0
        max_pos_cap = (MAX_POSITION_PCT * 100.0) if MAX_POSITION_PCT <= 1.0 else MAX_POSITION_PCT
        position_size_pct = min(max_pos_cap, raw_position_size)

        targets, target_cluster_vals = self.compute_targets(risk_dist, vol_regime)

        t1_price = targets["t1"]["price"]
        expected_rr = round((t1_price - self.entry_price) / risk_dist, 2) if risk_dist > 0 else 0.0

        hist_stats = self.get_historical_stats()
        prob_win = hist_stats["win_rate"]
        prob_loss = 1.0 - prob_win
        ev = round((prob_win * hist_stats["avg_win"]) - (prob_loss * hist_stats["avg_loss"]), 2)

        warnings = []
        if (t1_price - self.entry_price) / max(self.entry_price, 1) < 0.03:
            warnings.append("Target very close to resistance")
        if support_metrics["score"] < 50:
            warnings.append("Support confidence below 50")
        if self.kwargs.get("atr_pct", 0) > 4.5:
            warnings.append("ATR percentile very high")

        diagnostics = {
            "support_source": "Swing Cluster" if self.kwargs.get("swing_low_cluster") else "Pivot",
            "target_source": "Hybrid Consensus",
            "volatility_mode": vol_regime,
            "market_regime": "Bull",
            "scanner": self.mode,
            "engine_version": "2.0",
            "target_cluster_values": target_cluster_vals
        }

        return {
            "engine_version": "2.0",
            "scanner": self.mode,
            "trade_quality": tq,
            "trade_quality_breakdown": tq_metrics["breakdown"],
            "support": {
                "price": round(support_price, 2),
                "confidence": support_metrics["score"],
                "breakdown": support_metrics["breakdown"]
            },
            "risk": {
                "stop_loss": sl,
                "rr": expected_rr,
                "risk_pct": round(risk_pct, 2),
                "position_size_pct": position_size_pct,
                "kelly_fraction": kelly_fraction,
                "expected_value": ev
            },
            "targets": targets,
            "management": {
                "time_stop": self.get_time_stop(),
                "trailing": self.get_trailing_rule()
            },
            "diagnostics": diagnostics,
            "warnings": warnings
        }


class BreakoutAdapter(BaseRiskEngine):
    def get_time_stop(self) -> str:
        return "5-7 trading days"
    def get_historical_stats(self) -> dict:
        return {"win_rate": 0.58, "avg_win": 2.8, "avg_loss": 1.0}

class ReversalAdapter(BaseRiskEngine):
    def get_time_stop(self) -> str:
        return "12-15 trading days"
    def get_historical_stats(self) -> dict:
        return {"win_rate": 0.52, "avg_win": 3.2, "avg_loss": 1.0}




def _round_sl_target_metrics(res: dict) -> dict:
    """
    [RULE 67 CHANGE-RATIONALE: UNIVERSAL_PRICE_ROUNDING_v1.0]
    Guarantees that all Stop-Loss, Target levels, Entry prices, and Risk metrics
    generated by any SL/Target engine are clean, rounded off to 2 decimal places (NSE standard).
    """
    if not isinstance(res, dict):
        return res
    price_metric_keys = {
        "entry", "entry_price", "stop_loss", "initial_stop_loss", "sl",
        "target", "target_0", "target_1", "target_2", "target_3", "target_4",
        "target_price", "structural_failure_stop", "nearest_resistance",
        "resistance_price", "suggested_trailing_sl", "natural_rr", "rr_ratio",
        "t0_rr_ratio", "risk_pct", "risk_amount", "t1", "t2", "t3", "t4"
    }
    for k in price_metric_keys:
        if k in res and res[k] is not None:
            try:
                res[k] = round(float(res[k]), 2)
            except (ValueError, TypeError):
                pass
    return res


@profile_function("SL Target", budget_mb=450.0)
def compute_sl_and_target(
    entry_price:    float,
    atr:            Optional[float],
    candle_range:   float = 0.0,
    mode:           Optional[str]   = None,
    engine_version: str             = "v1.0",
    **kwargs
) -> dict:
    """
    Unified entry point for generating SL and Target metrics.
    Supports backward-compatibility via `engine_version="v1.0"`.
    """
    if engine_version in ("v1.0", "v1"):
        # Legacy fallback wrapper
        raw_res = _legacy_compute_sl_and_target(
            entry_price=entry_price,
            atr=atr,
            candle_range=candle_range,
            mode=mode,
            **kwargs
        )
        return _round_sl_target_metrics(raw_res)

    # v2.0 Institutional Engine routing
    # Map mode string to proper adapter
    scanner = (mode or "BREAKOUT").upper()
    kwargs["entry_price"] = entry_price
    kwargs["atr"] = atr
    kwargs["candle_range"] = candle_range

    if scanner == "REVERSAL":
        adapter = ReversalAdapter(scanner, kwargs)
    else:
        adapter = BreakoutAdapter(scanner, kwargs)

    raw_res = adapter.generate_metrics()
    return _round_sl_target_metrics(raw_res)
