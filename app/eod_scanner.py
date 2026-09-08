from scanner_telemetry import ScannerDecisionLogger, global_telemetry
import time as _time
# =====================================================================================
# app/eod_scanner.py (SCHEDULER READY)
# EOD BREAKOUT SCANNER WITH CONSOLIDATED MAIL AUTOMATION
#
# RULE 67 MANDATORY CHANGE-RATIONALE:
# - Verified and enhanced GlobalScannerTelemetryEngine logging across all EOD gate evaluations.
# - Rationale: Ensures full per-symbol diagnostic visibility in scanner_telemetry.jsonl and console,
#   tracking exact raw inputs, indicator calculations, scoring breakdowns, and gate pass/fail reasons
#   without diagnostic gaps during production execution.
# =====================================================================================

import os
import time
import json
import math
import pandas as pd
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from technical_indicators import apply_indicators, hydrate_indicators
from memory_profiler import MemoryProfiler
from breakout_engine import detect_breakouts
from scoring_engine import calculate_score
from sector_rotation import get_sector_scores, SectorRotationResult
from surveillance import get_live_blacklist, force_refresh_blacklist
from trade_ranking_engine import TradeRankingEngine
from macro_utils import (
    MarketRegimeEngine, get_nifty_20d_return, get_macro_regime,
    compute_nifty_rs_rating, compute_sector_regime_rankings
)
from strategy_policy import StrategyPolicyEngine
from database import (
    init_db, save_alert_if_new, save_candidate, upsert_fetch_error,
    upsert_scanner_health, insert_notification,
    get_recent_alerts_for_scanner, verify_alerts_saved_today
)
from core_enums import ProviderResult
from core_models import ScanFailure
from delivery_data import fetch_delivery_data
from price_cache import fetch_watchlist_data
from sl_target_helper import compute_sl_and_target
from watchlist_cache import get_watchlist
import time
import database

# [VERSION: PERF_PROFILER_v1.0] Stage timing + filter rejection observability
# profile_timing logs duration + RSS delta for each EOD scanner run.
# FilterStats exports per-filter rejection CSV to artifacts/profiling/.
from perf_utils import profile_timing, FilterStats
from zero_alert_diagnostic import (
    SingleTerminalTracker,
    StageWaterfallTracker,
    classify_zero_alert_run,
    format_zero_alert_diagnostic_block,
)

from config import (
    EOD_CONFIG,
    EOD_ADVANCED_CONFIG,
    ACTIVE_ALGO_VERSION,
    ALERT_COOLDOWN_MINUTES,
    ADX_MIN_THRESHOLD,
    MIN_STOCK_PRICE,
    SCORE_THRESHOLDS,
    MIN_BREAKOUT_MARGIN,
    MIN_BREAKOUT_VOLUME_RATIO,
    BASE_TIGHTNESS_THRESHOLD,
    RS_BONUS,
    SECTOR_BONUS,
    MAX_MOMENTUM_BONUS,
)

logger = logging.getLogger(__name__)

from scanner_telemetry import DecisionContext, telemetry_engine

IST        = ZoneInfo("Asia/Kolkata")

MIN_SIGNALS             = EOD_CONFIG["MIN_SIGNALS"]
MIN_BODY_RATIO          = EOD_CONFIG["MIN_BODY_RATIO"]
MIN_CLOSE_POSITION      = EOD_CONFIG["MIN_CLOSE_POSITION"]
MAX_UPPER_WICK_RATIO    = EOD_CONFIG["MAX_UPPER_WICK"]
MIN_VOLUME_RATIO        = EOD_CONFIG["MIN_VOLUME_RATIO"]
MIN_AVG_VOLUME_SHARES   = EOD_CONFIG["MIN_VOLUME_AVG"]
MIN_RSI                 = EOD_CONFIG["MIN_RSI"]
MAX_RSI                 = EOD_CONFIG["MAX_RSI"]

# MIN_STOCK_PRICE imported from config (₹100)
MAX_DISTANCE_FROM_52W_HIGH_PCT = EOD_ADVANCED_CONFIG["MAX_DISTANCE_FROM_52W_HIGH_PCT"]

from lock_utils import ProcessLock
_scan_lock = ProcessLock("eod_scanner")
_global_lock = ProcessLock("global_scanner_lock")

def start(force: bool = False, session=None, run_ctx=None, trigger_type="SCHEDULED", scheduler_name="CRON", used_fallback_data: bool = False):
    from database import is_scanner_stopped, upsert_scanner_health, start_scanner_execution_run, complete_scanner_execution_run
    from lock_utils import print_scanner_start_banner, print_scanner_end_banner
    if is_scanner_stopped("EOD"):
        logger.info("🛑 EOD Scanner is STOPPED by Admin. Skipping execution.")
        if run_ctx:
            complete_scanner_execution_run(run_ctx, status_override="STOPPED", stop_reason="Scanner stopped by admin")
        return 0

    if _scan_lock.locked():
        logger.warning("🛑 [DUPLICATE GUARD] EOD Scanner is ALREADY actively running in thread lock. Skipping duplicate trigger.")
        if run_ctx:
            complete_scanner_execution_run(run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Scanner lock busy")
        return {"status": "skipped", "reason": "already_running"}

    acquired_global = False
    acquired_scan = False
    _scan_start = None

    try:
        queued_at = None
        if not _global_lock.acquire(blocking=False, owner_scanner="EOD", operation="FULL_SCAN"):
            queued_at = time.monotonic()
            logger.info("⏳ [EOD] Global scanner lock busy — waiting in queue until lock is released...")
            upsert_scanner_health("EOD", "QUEUED", error_msg="Waiting in queue for active scanner to release lock...")

            try:
                acquired_global = _global_lock.acquire(blocking=True, owner_scanner="EOD", operation="FULL_SCAN", run_ctx=run_ctx)
            except Exception as lock_err:
                logger.error(f"❌ [EOD] Error acquiring global lock: {lock_err}")
                acquired_global = False

            if not acquired_global:
                logger.error("❌ [EOD] Failed to acquire global scanner lock after queue wait.")
                if run_ctx:
                    complete_scanner_execution_run(run_ctx, status_override="FAILED", stop_reason="Global lock acquire timeout")
                upsert_scanner_health("EOD", "IDLE", error_msg="Lock acquisition timed out")
                return 0
        else:
            acquired_global = True

        if queued_at is not None:
            logger.info(f"✅ [EOD] Global lock acquired after {round(time.monotonic()-queued_at,1)}s wait. Starting scan...")

        upsert_scanner_health("EOD", "RUNNING", error_msg="EOD scan in progress...")

        if not _scan_lock.acquire(blocking=False):
            logger.warning("🛑 EOD Scanner is ALREADY actively running. Skipping duplicate execution.")
            if run_ctx:
                complete_scanner_execution_run(run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Scanner already actively running")
            else:
                try:
                    from database import record_skipped_execution_run
                    record_skipped_execution_run(scanner_name="EOD", trigger_type=trigger_type, scheduler_name=scheduler_name, stop_reason="Scanner lock held (previous run active)")
                except Exception:
                    pass
            upsert_scanner_health("EOD", "IDLE", error_msg="Duplicate trigger skipped")
            return 0
        acquired_scan = True

        # [RULE: HISTORY ENTRY AFTER LOCK ACQUIRED] Only create execution history entry once all locks are secured
        if run_ctx is None:
            try:
                from database import start_scanner_execution_run
                run_ctx = start_scanner_execution_run(scanner_name="EOD", trigger_type=trigger_type, scheduler_name=scheduler_name)
            except Exception as exc:
                if "actively running" in str(exc).lower():
                    logger.info("🛑 [EOD] Scanner is ALREADY actively running. Skipping duplicate execution.")
                    return 0
                logger.warning(f"⚠️ [EOD] Could not create run_ctx: {exc}")
        elif run_ctx:
            from database import update_scanner_run_lifecycle
            update_scanner_run_lifecycle(run_ctx.run_id, "RUNNING")

        _scan_start = print_scanner_start_banner("eod_scanner", queued_at=queued_at, run_id=run_ctx.run_id if run_ctx else None)
        total = _start_wrapper(force, session=session, run_ctx=run_ctx, used_fallback_data=used_fallback_data)
        if run_ctx and isinstance(total, int):
            run_ctx.add_alert(total)
        if run_ctx:
            complete_scanner_execution_run(run_ctx, status_override="COMPLETED")
        return total
    except Exception as e:
        logger.exception(f"❌ [EOD] Unhandled exception during scan: {e}")
        if run_ctx:
            try:
                complete_scanner_execution_run(run_ctx, status_override="FAILED", exception=e)
            except Exception: pass
        try:
            upsert_scanner_health("EOD", status="DOWN", error_msg=f"Scan crashed: {str(e)[:300]}", run_id=run_ctx.run_id if run_ctx else None)
            insert_notification("error", "🚨 EOD Scanner CRASHED", f"Error: {str(e)[:400]}")
        except Exception: pass
        raise e
    finally:
        if _scan_start is not None:
            print_scanner_end_banner("eod_scanner", _scan_start, run_id=run_ctx.run_id if run_ctx else None)

        if acquired_scan:
            try: _scan_lock.release()
            except Exception: pass
        if acquired_global:
            try: _global_lock.release()
            except Exception: pass


def _safe_float(val, default=0.0):
    try:
        import pandas as pd
        if val is None or pd.isna(val) or val == "":
            return default
        return float(val)
    except Exception:
        return default


def _check_eod_conditions(
    ticker: pd.DataFrame,
    latest: pd.Series,
    symbol: str,
    mode: str = "production",
    prior_high_source: str = "indicator",
    delivery_pct: float = None,
    nifty_ret: float = None,
    regime_ctx: dict = None,
    ctx = None,
) -> dict:
    """
    Shared EOD breakout condition checks for both UI and production paths.

    Args:
        ticker: full DataFrame with indicators applied
        latest: the last row of ticker
        symbol: stock symbol for logging
        mode: "ui" returns reasons list; "production" returns rejection key + logs
        prior_high_source: "indicator" uses PRIOR_20D_HIGH column, "raw" uses 20-bar max
        delivery_pct, nifty_ret, regime_ctx: passed through for scoring

    Returns dict:
        passed: bool
        reason: str or None
        candle_penalty: int
        body_ratio, close_pos, wick_ratio, rsi_val, volume_ratio, avg_volume, atr20
        candle_close, candle_open, candle_high, candle_low, candle_range, candle_body
        prior_high, atr_extension, gap_pct (may be None if not computed)
    """
    candle_high  = _safe_float(latest.get("High"))
    candle_low   = _safe_float(latest.get("Low"))
    candle_open  = _safe_float(latest.get("Open"))
    candle_close = _safe_float(latest.get("Close"))
    candle_range = candle_high - candle_low
    candle_body  = abs(candle_close - candle_open)
    upper_wick   = candle_high - max(candle_close, candle_open)

    if candle_range < 0:
        return {"passed": False, "reason": "Negative candle range (corrupt data)"}
    elif candle_range == 0:
        if _safe_float(latest.get("Volume", 0)) <= 0:
            return {"passed": False, "reason": "Zero range and zero volume (invalid candle)"}
        # Otherwise it's a valid circuit candle (range=0, volume>0)

    if candle_range > 0:
        body_ratio  = candle_body / candle_range
        close_pos   = (candle_close - candle_low) / candle_range
        wick_ratio  = upper_wick / candle_range
    else:
        body_ratio = 1.0
        close_pos = 1.0
        wick_ratio = 0.0

    if len(ticker) >= 22:
        avg_volume = float(ticker["Volume"].iloc[-21:-1].mean())
    else:
        avg_volume = float(ticker["Volume"].iloc[:-1].mean())

    if avg_volume <= 0:
        return {"passed": False, "reason": "Zero avg volume"}

    volume_ratio = _safe_float(latest.get("Volume")) / avg_volume
    rsi_val      = _safe_float(latest.get("RSI"), 50.0)
    atr20        = _safe_float(latest.get("ATR20"), _safe_float(latest.get("ATR"), candle_close * 0.025))

    if ctx:
        ctx.add_decision_input(name="Close", value=candle_close, source="Indicator", as_of="Live", freshness="LIVE", required=True, valid=True)
        ctx.add_decision_input(name="VolumeRatio", value=volume_ratio, source="Indicator", as_of="Live", freshness="LIVE", required=True, valid=True)
        ctx.add_decision_input(name="AvgVolume", value=avg_volume, source="Indicator", as_of="Live", freshness="LIVE", required=True, valid=True)
        ctx.add_decision_input(name="RSI", value=rsi_val, source="Indicator", as_of="Live", freshness="LIVE", required=True, valid=True)

    # ── Shared hard gates ──────────────────────────────────────────────────
    if volume_ratio < MIN_VOLUME_RATIO:
        return {"passed": False, "reason": f"Volume ratio {volume_ratio:.2f}x < {MIN_VOLUME_RATIO:.1f}x"}
    if avg_volume < MIN_AVG_VOLUME_SHARES:
        return {"passed": False, "reason": f"Avg volume {avg_volume:.0f} < {MIN_AVG_VOLUME_SHARES:.0f}"}
    if candle_close < MIN_STOCK_PRICE:
        return {"passed": False, "reason": f"Close ₹{candle_close:.2f} < ₹{MIN_STOCK_PRICE:.0f} floor"}
    if not (MIN_RSI <= rsi_val <= MAX_RSI):
        return {"passed": False, "reason": f"RSI {rsi_val:.1f} outside {MIN_RSI}-{MAX_RSI}"}

    # ── Prior high & breakout check ────────────────────────────────────────
    if prior_high_source == "indicator":
        if "PRIOR_20D_HIGH" not in ticker.columns or pd.isna(latest.get("PRIOR_20D_HIGH")):
            return {"passed": False, "reason": "Missing PRIOR_20D_HIGH"}
        prior_high = _safe_float(latest.get("PRIOR_20D_HIGH"))
        if prior_high <= 0:
            return {"passed": False, "reason": "Invalid PRIOR_20D_HIGH"}
        if candle_close <= prior_high:
            return {"passed": False, "reason": f"Close ₹{candle_close:.2f} <= Prior High ₹{prior_high:.2f}"}
    else:
        lookback = 20 if len(ticker) >= 21 else len(ticker) - 1
        prior_high = float(ticker['High'].iloc[-lookback-1:-1].max()) if lookback > 0 else float(ticker['High'].max())
        if candle_close <= prior_high:
            return {"passed": False, "reason": f"Close ₹{candle_close:.2f} <= 20D High ₹{prior_high:.2f}"}

    # ── ATR checks ─────────────────────────────────────────────────────────
    if "ATR20" in ticker.columns and not pd.isna(latest.get("ATR20")):
        atr20 = _safe_float(latest.get("ATR20"))
    if atr20 <= 0:
        return {"passed": False, "reason": "ATR20 <= 0"}

    atr_extension = (candle_close - prior_high) / atr20
    max_ext = EOD_ADVANCED_CONFIG.get("MAX_EXTENDED_BREAKOUT_ATR_MULT", 1.5)
    if atr_extension > max_ext:
        pass  # Soft penalty, not hard reject

    # ── ATR expansion ──────────────────────────────────────────────────────
    import circuit_helper
    is_circuit = circuit_helper.is_valid_circuit_candle(
        candle_range=candle_range,
        volume=_safe_float(latest.get("Volume")),
        close_price=candle_close
    )

    if is_circuit:
        atr_expansion = None
    else:
        atr_expansion = candle_range / atr20

    min_atr_expansion = EOD_ADVANCED_CONFIG.get("MIN_ATR_EXPANSION_RATIO", 0.9)
    if not is_circuit and atr_expansion is not None and atr_expansion < min_atr_expansion:
        return {"passed": False, "reason": f"ATR expansion {atr_expansion:.2f} < {min_atr_expansion}"}

    # ── Trend alignment ────────────────────────────────────────────────────
    if "EMA20" in ticker.columns and not pd.isna(latest.get("EMA20")):
        if candle_close < _safe_float(latest.get("EMA20")):
            return {"passed": False, "reason": f"Below EMA20"}
    if "SMA50" in ticker.columns and not pd.isna(latest.get("SMA50")):
        if candle_close < _safe_float(latest.get("SMA50")):
            return {"passed": False, "reason": "Below SMA50"}
    if "ADX" in ticker.columns and not pd.isna(latest.get("ADX")):
        if _safe_float(latest.get("ADX")) < ADX_MIN_THRESHOLD:
            return {"passed": False, "reason": f"ADX {_safe_float(latest.get('ADX')):.1f} < {ADX_MIN_THRESHOLD}"}

    # ── 52W high distance — TWO-MODE structural gate ─────────────────────
    # [FIX: TWO_MODE_52W] Mode A (High Breakout): within 5% of 52W high.
    # Mode B (Recovery Breakout): 5-15% below 52W high with stricter
    # secondary checks (volume, base compression, RS strength).
    _recovery_mode_b = False
    if "HIGH_52W" in ticker.columns and not pd.isna(latest.get("HIGH_52W")):
        high_52w = _safe_float(latest.get("HIGH_52W"))
        if high_52w > 0:
            pct_from_high = (high_52w - candle_close) / high_52w * 100
            if pct_from_high > MAX_DISTANCE_FROM_52W_HIGH_PCT:
                # Mode A failed. Try Mode B.
                _recovery_max = EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_MAX_DISTANCE_PCT", 15.0)
                _recovery_vol = EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_MIN_VOL_RATIO", 2.5)
                _recovery_bb  = EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_MAX_BB_WIDTH", 0.50)
                # [FIX: NO_MODE_B_RSI_CEILING] No separate RSI ceiling for Mode B.
                # Mode B is already constrained by: volume >= 2.5x, BB <= 0.50, RS >= 60,
                # plus the global RSI hard cap (92) and all structural/trend gates.
                # Adding a tighter RSI ceiling here recreates the same contradictory logic we're removing.
                # RSI is evaluated through the normal scoring and penalty model.
                # Check distance within Mode B range and higher volume requirement (>= 2.5x)
                _mode_b_ok = (pct_from_high <= _recovery_max) and (volume_ratio >= _recovery_vol)
                # Check BB tightness
                if "BB_WIDTH_PCTILE" in ticker.columns and len(ticker) >= 2:
                    bb_pctile_prev = _safe_float(ticker["BB_WIDTH_PCTILE"].iloc[-2])
                    _mode_b_ok = _mode_b_ok and (bb_pctile_prev <= _recovery_bb)
                if _mode_b_ok:
                    _recovery_mode_b = True
                else:
                    return {"passed": False, "reason": f"Too far from 52W high ({pct_from_high:.1f}%) — failed both Mode A (≤{MAX_DISTANCE_FROM_52W_HIGH_PCT}%) and Mode B (≤{_recovery_max}% with tighter filters)"}

    # ── Single-day move cap ────────────────────────────────────────────────
    if len(ticker) >= 2:
        prev_close = _safe_float(ticker["Close"].iloc[-2])
        if prev_close > 0:
            single_move_pct = abs(candle_close - prev_close) / prev_close * 100
            if single_move_pct > EOD_ADVANCED_CONFIG.get("MAX_SINGLE_DAY_MOVE_PCT", 15.0):
                return {"passed": False, "reason": f"Single-day move {single_move_pct:.1f}% > cap"}

    # ── Pre-breakout candle context ────────────────────────────────────────
    lookback_ctx = EOD_ADVANCED_CONFIG.get("PRE_BREAKOUT_LOOKBACK_BARS", 5)
    max_red = EOD_ADVANCED_CONFIG.get("MAX_PRE_BREAKOUT_RED_CANDLES", 3)
    tight_base_threshold = EOD_ADVANCED_CONFIG.get("TIGHT_BASE_BB_WIDTH_PCTILE", 0.50)
    if len(ticker) >= (lookback_ctx + 1):
        red_count = sum(
            1 for _ri in range(-(lookback_ctx + 1), -1)
            if _safe_float(ticker["Close"].iloc[_ri]) < _safe_float(ticker["Open"].iloc[_ri])
        )
        if red_count > max_red:
            is_tight_base = False
            if "BB_WIDTH_PCTILE" in ticker.columns and len(ticker) >= 2:
                if _safe_float(ticker["BB_WIDTH_PCTILE"].iloc[-2]) <= tight_base_threshold:
                    is_tight_base = True
            if not is_tight_base:
                return {"passed": False, "reason": f"Pre-breakout weak ({red_count}/{lookback_ctx} red candles)"}

    # ── Base width & 10-day Pre-Breakout Tightness [v5.3.0] ───────────────
    if "BB_WIDTH_PCTILE" in ticker.columns and len(ticker) >= 2:
        bb_width_pctile = _safe_float(ticker["BB_WIDTH_PCTILE"].iloc[-2])
        if bb_width_pctile > EOD_ADVANCED_CONFIG.get("MAX_BB_WIDTH_PCTILE", 0.80):
            return {"passed": False, "reason": f"Base too wide (BB Pctile {bb_width_pctile:.2f})"}

    # [RULE 67 CHANGE-RATIONALE: EOD_ATR_MODEL_A_V1]
    # Replaced binary 2.50% cliff with certified Model A tiered scoring:
    # <= 2.50%: 0 penalty (tight base consolidation)
    # 2.51% - 3.50%: -3 penalty (minor volatility penalty)
    # 3.51% - 4.50%: -7 penalty (moderate volatility penalty)
    # > 4.50%: Hard Reject (structural ceiling)
    base_atr_penalty = 0
    base_atr_pct = None
    if len(ticker) >= 12 and candle_close > 0:
        import numpy as _np
        highs_10 = ticker["High"].iloc[-11:-1]
        lows_10 = ticker["Low"].iloc[-11:-1]
        closes_10 = ticker["Close"].iloc[-12:-2]
        tr_10 = _np.maximum(highs_10 - lows_10, _np.maximum(_np.abs(highs_10 - closes_10), _np.abs(lows_10 - closes_10)))
        atr_10 = float(tr_10.mean())
        # Use pre-breakout close (ticker.Close[-2]) as normalization base; fall back to candle_close if unavailable
        _atr10_base_price = _safe_float(ticker["Close"].iloc[-2]) if len(ticker) >= 2 else candle_close
        if _atr10_base_price <= 0:
            _atr10_base_price = candle_close
        
        base_atr_pct = round((atr_10 / _atr10_base_price) * 100.0, 4)
        
        if base_atr_pct <= 2.5000 + 1e-7:
            base_atr_penalty = 0
        elif base_atr_pct <= 3.5000 + 1e-7:
            base_atr_penalty = 3
        elif base_atr_pct <= 4.5000 + 1e-7:
            base_atr_penalty = 7
        else:
            return {"passed": False, "reason": f"Base ATR10 ({base_atr_pct:.2f}% of prev-close) > 4.50% tightness ceiling"}

    # ── Candle quality penalties (soft, not hard) ──────────────────────────
    candle_penalty = 0
    if body_ratio < MIN_BODY_RATIO:
        shortfall = (MIN_BODY_RATIO - body_ratio) / MIN_BODY_RATIO
        candle_penalty += min(15, int(shortfall * 30))
    if candle_close <= candle_open:
        candle_penalty += 5
    if close_pos < MIN_CLOSE_POSITION:
        shortfall = (MIN_CLOSE_POSITION - close_pos) / MIN_CLOSE_POSITION
        candle_penalty += min(10, int(shortfall * 20))
    if wick_ratio > MAX_UPPER_WICK_RATIO:
        excess = (wick_ratio - MAX_UPPER_WICK_RATIO) / MAX_UPPER_WICK_RATIO
        candle_penalty += min(10, int(excess * 20))

    # ── Gap penalty (soft) ─────────────────────────────────────────────────
    gap_lookback_bars = EOD_ADVANCED_CONFIG.get("GAP_LOOKBACK_BARS", 10)
    max_gap_pct = EOD_ADVANCED_CONFIG.get("MAX_GAP_FROM_PRIOR_HIGH_PCT", 3.0)
    gap_pct = None
    if len(ticker) >= gap_lookback_bars + 1:
        gap_ref_high = float(ticker["High"].iloc[-(gap_lookback_bars + 1):-1].max())
        if gap_ref_high > 0:
            gap_pct = (candle_open - gap_ref_high) / gap_ref_high * 100

    return {
        "passed": True,
        "candle_penalty": candle_penalty,
        "base_atr_penalty": base_atr_penalty,
        "base_atr_pct": base_atr_pct,
        "body_ratio": body_ratio,
        "close_pos": close_pos,
        "wick_ratio": wick_ratio,
        "rsi_val": rsi_val,
        "volume_ratio": volume_ratio,
        "avg_volume": avg_volume,
        "atr20": atr20,
        "candle_close": candle_close,
        "candle_open": candle_open,
        "candle_high": candle_high,
        "candle_low": candle_low,
        "candle_range": candle_range,
        "candle_body": candle_body,
        "upper_wick": upper_wick,
        "prior_high": prior_high,
        "atr_extension": atr_extension,
        "gap_pct": gap_pct,
    }

def evaluate_eod_symbol(symbol: str, df: pd.DataFrame, fund_data: dict = None, regime_ctx: dict = None) -> dict:
    """
    Evaluates a single symbol against the production EOD breakout scanner rules.
    Runs full breakout detection, candle body/wick gates, ATR expansion, trend alignment, scoring engine, and target calculations without side effects.
    """
    if df is None or df.empty or len(df) < 50:
        return {
            "status": "NO",
            "reasons": [f"Insufficient historical price data ({len(df) if df is not None else 0} bars < 50 minimum)"],
            "score": 0.0,
            "qualified": False,
            "calibration_model_version": "EOD_ATR_MODEL_A_V1"
        }

    ticker = df.copy()
    if isinstance(ticker.columns, pd.MultiIndex):
        ticker.columns = ticker.columns.get_level_values(0)
    ticker = ticker.loc[:, ~ticker.columns.duplicated()]

    rename_map = {c: str(c).capitalize() for c in ticker.columns if str(c).lower() in ['open', 'high', 'low', 'close', 'volume']}
    if rename_map:
        ticker = ticker.rename(columns=rename_map)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in ticker.columns:
            return {"status": "NO", "reasons": [f"Missing required price column '{col}'"], "score": 0.0, "qualified": False, "calibration_model_version": "EOD_ATR_MODEL_A_V1"}
        ticker[col] = pd.Series(ticker[col]).astype(float)

    ticker = ticker.dropna(subset=required_cols)
    if ticker.empty or len(ticker) < 50:
        return {"status": "NO", "reasons": [f"Insufficient valid bars ({len(ticker)} < 50)"], "score": 0.0, "qualified": False, "calibration_model_version": "EOD_ATR_MODEL_A_V1"}

    ticker = apply_indicators(ticker, timeframe="1d")
    latest = ticker.iloc[-1]

    # [FIX P6-13] Use shared condition check for consistency with production path
    cond = _check_eod_conditions(
        ticker=ticker, latest=latest, symbol=symbol, mode="ui",
        prior_high_source="raw", ctx=None
    )
    if not cond.get("passed"):
        return {
            "status": "NO",
            "reasons": [cond.get("reason", "Condition check failed")],
            "score": 0.0,
            "qualified": False,
            "entry_price": _safe_float(latest.get("Close")),
            "atr_20": cond.get("atr20", 0),
            "base_atr_pct": cond.get("base_atr_pct"),
            "base_atr_penalty": cond.get("base_atr_penalty", 0),
            "calibration_model_version": "EOD_ATR_MODEL_A_V1"
        }

    candle_close = cond["candle_close"]
    candle_open  = cond["candle_open"]
    candle_low   = cond["candle_low"]
    candle_high  = cond["candle_high"]
    candle_range = cond["candle_range"]
    prior_high   = cond["prior_high"]
    rsi_val      = cond["rsi_val"]
    vol_ratio    = cond["volume_ratio"]
    atr20        = cond["atr20"]

    signals = detect_breakouts(ticker, timeframe="1d")
    score, _, _ = calculate_score(
        category=fund_data.get("Category", "EQUITY") if fund_data else "EQUITY",
        breakout_count=len(signals),
        rsi=rsi_val,
        volume_ratio=vol_ratio,
        breakout_signals=signals,
        ticker=ticker,
        latest=latest,
        symbol=symbol,
        timeframe="1d",
        atr_val=atr20,
        regime_ctx=regime_ctx
    )

    if score > 0:
        candle_pen = cond.get("candle_penalty", 0)
        base_atr_pen = cond.get("base_atr_penalty", 0)
        score = max(0, score - candle_pen - base_atr_pen)

        # Gap-and-go extension penalty
        atr_ext = (candle_close - prior_high) / atr20 if atr20 > 0 else 0
        max_ext = EOD_ADVANCED_CONFIG.get("MAX_EXTENDED_BREAKOUT_ATR_MULT", 1.5)
        if atr_ext > max_ext:
            pen_mult = EOD_ADVANCED_CONFIG.get("GAP_AND_GO_PENALTY_MULT", 10)
            max_pen = EOD_ADVANCED_CONFIG.get("GAP_AND_GO_MAX_PENALTY", 20)
            score = max(0, score - min(max_pen, (atr_ext - max_ext) * pen_mult))

        # Gap penalty (gapping > 3% above prior high)
        gap_pct = cond.get("gap_pct")
        if gap_pct is not None:
            max_gap_pct = EOD_ADVANCED_CONFIG.get("MAX_GAP_FROM_PRIOR_HIGH_PCT", 3.0)
            if gap_pct > max_gap_pct:
                excess = gap_pct - max_gap_pct
                score = max(0, score - min(20, int(excess * 3)))

        # OBV divergence penalty
        if "OBV_SLOPE" in ticker.columns and not pd.isna(latest.get("OBV_SLOPE")):
            if _safe_float(latest.get("OBV_SLOPE")) <= EOD_ADVANCED_CONFIG.get("MIN_OBV_SLOPE", 0.0):
                score = max(0, score - 5)

    score_threshold = SCORE_THRESHOLDS.get("1d", 82)
    is_qualified = (score >= score_threshold)

    sl_result = compute_sl_and_target(
        entry_price=candle_close,
        atr=atr20,
        candle_range=candle_range,
        mode="EOD",
        rsi=rsi_val,
        candle_low=candle_low,
        ticker=ticker
    )

    status_str = "CORE MET" if is_qualified else "NO"
    reasons = [f"Clean Breakout Close (₹{candle_close:.2f} > ₹{prior_high:.2f}) | Volume Surge {vol_ratio:.2f}x ≥ 1.8x | EOD Score {score:.1f}/100"] if is_qualified else [f"Score {score:.1f} < {score_threshold} minimum threshold"]

    # ── PER-STOCK TERMINAL TELEMETRY DUMP (Section 4 & 8) ──
    try:
        from scanner_telemetry import DecisionContext, telemetry_engine
        ctx = DecisionContext(symbol=symbol, scanner_name="EOD")
        ctx.capture_raw_market(
            open_p=_safe_float(latest.get("Open")),
            high_p=_safe_float(latest.get("High")),
            low_p=_safe_float(latest.get("Low")),
            close_p=_safe_float(latest.get("Close")),
            volume=_safe_float(latest.get("Volume")),
            prev_close=_safe_float(latest.get("Close_Prev"))
        )
        ctx.capture_indicators(
            rsi=rsi_val,
            sma20=_safe_float(latest.get("SMA20")),
            sma50=_safe_float(latest.get("SMA50")),
            sma200=_safe_float(latest.get("SMA200")),
            vol_ratio=vol_ratio,
            atr=atr20
        )
        ctx.capture_config("SCORE_THRESHOLD", score_threshold)
        ctx.capture_gate("ConditionCheck", cond.get("passed", False), actual_val=candle_close, operator_str=">", threshold_val=prior_high)
        ctx.capture_score("TOTAL", score, 100.0)
        ctx.capture_sl_target(candle_close, sl_result.get("stop_loss", 0.0), sl_result.get("target_1", 0.0))

        consumed_fields = {
            "Close": candle_close,
            "High": candle_high,
            "Low": candle_low,
            "Open": candle_open,
            "Volume Ratio": vol_ratio,
            "RSI": rsi_val,
            "ATR": atr20,
            "SMA20": _safe_float(latest.get("SMA20")),
            "SMA50": _safe_float(latest.get("SMA50")),
            "SMA200": _safe_float(latest.get("SMA200")),
            "Prior High": prior_high,
            "Body Ratio": _safe_float(latest.get("Body_Ratio")),
            "Close Position": _safe_float(latest.get("Close_Position"))
        }
        for k, v in consumed_fields.items():
            if v is not None and not pd.isna(v):
                ctx.add_decision_input(name=k, value=v, source="EODScanner", as_of="Live", freshness="LIVE", required=True, valid=True)

        ctx.finalize(decision="SELECTED" if is_qualified else "REJECTED", primary_reason=reasons[0])
        telemetry_engine.emit_terminal(ctx)
    except Exception as telemetry_err:
        logger.debug(f"Telemetry recording skipped: {telemetry_err}")

    return {
        "status": status_str,
        "reasons": reasons,
        "score": score,
        "qualified": is_qualified,
        "entry_price": candle_close,
        "stop_loss": sl_result.get("stop_loss"),
        "target_1": sl_result.get("target_1"),
        "target_2": sl_result.get("target_2"),
        "target_3": sl_result.get("target_3"),
        "target_4": sl_result.get("target_4"),
        "atr_20": atr20,
        "base_atr_pct": cond.get("base_atr_pct"),
        "base_atr_penalty": cond.get("base_atr_penalty", 0),
        "calibration_model_version": "EOD_ATR_MODEL_A_V1"
    }



# [VERSION: PERF_PROFILER_v1.0] Wrap the scan body so every EOD run reports
# wall-clock time, memory delta (RSS), and any top-level exception — all without
# changing any business logic or scanner decision paths.
@profile_timing("eod_scanner._start_wrapper", log_to_file=True)
def _start_wrapper(force: bool = False, session=None, run_ctx=None, used_fallback_data: bool = False):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
    start_time = datetime.now(IST)
    _last_hb = time.monotonic()
    total_alerts = 0
    total_fetched_count = 0
    duration_sec = 0.0

    is_test_mode = True  # Safe default
    init_db()
    # Initialize the fundamentals cache into the DatasetRegistry (DURABLE)
    from fundamentals_cache import init_fundamentals_registry
    init_fundamentals_registry()

    try:
        upsert_scanner_health("EOD", "RUNNING", error_msg="EOD Scan in progress...")
    except Exception:
        logger.warning("⚠️ Could not mark EOD as RUNNING")

    force_refresh_blacklist()

    nifty_ret_20d = get_nifty_20d_return()


    ist_now = datetime.now(IST)
    logger.info("\n" + "=" * 80)
    logger.info(f"🚀🚀🚀 [START] EOD SCANNER INIT | {ist_now.strftime('%Y-%m-%d %H:%M:%S')} 🚀🚀🚀")
    logger.info("=" * 80 + "\n")

    start_time = datetime.now(IST)
    from perf_utils import ScannerStageTracker
    stage_tracker = ScannerStageTracker("EOD_SCANNER")
    stage_tracker.start_stage(1, "Watchlist & Context Setup", "Loading fundamental watchlist and sector ratings")

    # Check if we are outside the valid EOD window (21:00 - 23:59:59)
    now_time = ist_now.time()
    scan_start = datetime.strptime("21:00", "%H:%M").time()
    scan_end = datetime.strptime("23:59:59", "%H:%M:%S").time()

    # [VERSION: ALL_ALERTS_PERSIST_v1.0] Dry-run mode disabled — all generated alerts persist to DB at all times.
    is_test_mode = False

    try:
        from memory_profiler import StageTimelineTracker
        with StageTimelineTracker("EOD", "1. Watchlist Universe Load"):
            try:
                watchlist = get_watchlist()
                stage_tracker.total_symbols = len(watchlist) if watchlist is not None else 0
                logger.info(f"🛡️ EOD Scanner running on full fundamental watchlist: {len(watchlist)} stocks")
            except Exception as e:
                logger.exception("❌ Failed to load watchlist")
                if not is_test_mode:
                    try:
                        upsert_scanner_health(scanner_name="EOD", status="DOWN", error_msg=f"Watchlist load failed: {str(e)[:200]}")
                    except Exception:
                        pass
                return 0

        if watchlist.empty:
            logger.info("🛡️ EOD Scanner | Universe is empty (no stocks passed Wealth Engine BUY signals). Exiting cleanly.")
            if not is_test_mode:
                try:
                    upsert_scanner_health("EOD", status="OK", last_success=datetime.now(IST).isoformat(), today_alerts=0, total_count=0)
                except Exception:
                    pass
            return 0

        # We do NOT purge here yet.
        # Purge only after upstream validation and fetch sufficiency checks succeed.
        # [VERSION: EOD_PATCH_v1.0] [BUG FIX 5] Compute today_str once here and reuse it throughout to avoid duplication
        today_str = ist_now.strftime("%Y-%m-%d")

        # [VERSION: SCANNER_DIAG_LOG_v1.0] Watchlist fingerprint for cross-run comparison
        import hashlib
        import uuid
        _wl_stocks = sorted(watchlist["Stock"].tolist())
        _wl_hash = hashlib.md5("|".join(_wl_stocks).encode()).hexdigest()[:12]
        scan_id = str(uuid.uuid4())

        all_universe_symbols = [str(s) for s in watchlist["Stock"].tolist() if s]
        terminal_tracker = SingleTerminalTracker(all_universe_symbols, scanner_name="EOD")
        terminal_tracker.map_gates_to_stage("FETCHED_DATA", [
            "BLACKLIST_EXCLUDED", "DATA_UNAVAILABLE", "DATA_EMPTY", "STALE_DATA",
            "MISSING_COL", "INSUFFICIENT_BARS", "INDICATOR_FAIL", "INDICATOR_NAN", "PROCESSING_ERROR"
        ])
        terminal_tracker.map_gates_to_stage("BREAKOUT_STRUCTURE", [
            "ZERO_AVG_VOLUME", "ZERO_CANDLE_RANGE", "LOW_VOLUME", "LOW_AVG_VOLUME", "PENNY_STOCK",
            "RSI_RANGE", "MISSING_STRUCTURE_INDICATOR", "NO_STRUCTURAL_BREAKOUT", "MISSING_ATR",
            "NO_ATR_EXPANSION", "BELOW_EMA20", "BELOW_SMA50", "WEAK_ADX", "FAR_FROM_52W_HIGH",
            "GAP_DAY", "BASE_TOO_WIDE", "WEAK_SIGNALS"
        ])
        terminal_tracker.map_gates_to_stage("QUALITY_AND_RISK", [
            "FORENSIC_REJECT", "LOW_SCORE", "COOLDOWN_ACTIVE", "RISK_REJECTED"
        ])
        terminal_tracker.map_gates_to_stage("FINAL_ALERTS", [
            "SUPPRESSED_FALLBACK_DATA", "SUPPRESSED_TOP_N", "DUPLICATE_ALERT", "ALERT_GENERATED"
        ])
        waterfall = StageWaterfallTracker(["UNIVERSE_WATCHLIST", "FETCHED_DATA", "BREAKOUT_STRUCTURE", "QUALITY_AND_RISK", "FINAL_ALERTS"])
        waterfall.set_stage_count("UNIVERSE_WATCHLIST", len(watchlist))
        # [RULE 67 CHANGE-RATIONALE: THREAD_SAFE_WATERFALL_COUNTS_V1.0]
        # Use mutable dictionary for waterfall counters rather than outer scalar integers.
        # RATIONALE: _process_row executes inside ThreadPoolExecutor. In Python, doing += 1 on outer
        # scalar integers raises UnboundLocalError. Mutating dictionary keys inside _batch_lock is
        # completely immune to UnboundLocalError and 100% thread-safe.
        waterfall_counts = {
            "structure_entered": 0,
            "quality_entered": 0,
            "near_misses": 0
        }

        stage_tracker.end_stage(f"Loaded watchlist ({len(watchlist)} stocks)")

        delivery_map: dict[str, float] = {}
        all_ticker_data = {}

        stage_tracker.start_stage(2, "Pre-Scan Context & Macro Setup", "Pledge, delivery data, sector rotation scores")
        with StageTimelineTracker("EOD", "2. Pre-Scan Data (Pledge, Delivery, Sectors)"):

            # [VERSION: MARKET_DATA_SESSION_v1.0] Load pledge & delivery from session when available.
            # Session already fetched these in parallel during build(); skip independent fetches.
            delivery_days_back = 0
            delivery_found = False
            symbols = [str(s) for s in watchlist["Stock"].tolist() if s]

            if session is not None:
                logger.info("📦 [EOD] Loading pledge & delivery from MarketDataSession (pre-fetched)")
                pledge_map = {
                    sym: session.get(sym).pledge_pct
                    for sym in symbols
                    if session.get(sym) is not None and session.get(sym).pledge_pct is not None
                }
                delivery_map = {
                    sym: session.get(sym).delivery_pct
                    for sym in symbols
                    if session.get(sym) is not None and session.get(sym).delivery_pct is not None
                }
                _delivery_stale = any(
                    session.get(sym).delivery_stale
                    for sym in symbols
                    if session.get(sym) is not None
                )
                delivery_found = bool(delivery_map)
                delivery_days_back = 1 if _delivery_stale else 0
                if _delivery_stale:
                    logger.info("⚠️ [EOD] Session delivery data is STALE (previous trading day's Bhavcopy)")
                logger.info(f"🛡️ Session pledge data: {len(pledge_map)} symbols | delivery: {len(delivery_map)} symbols")

                # Parallelize RS rating, Sector scores, and Sector regime rankings
                with ThreadPoolExecutor(max_workers=3, thread_name_prefix="EOD_PreScan") as executor:
                    f_rot = executor.submit(get_sector_scores)
                    f_rs = executor.submit(compute_nifty_rs_rating, symbols)
                    f_sre = executor.submit(compute_sector_regime_rankings)

                    try:
                        rotation_result = f_rot.result()
                    except Exception:
                        rotation_result = SectorRotationResult({}, set(), set(), "", datetime.now(IST).date(), 0.0)
                    try:
                        rs_dict = f_rs.result()
                    except Exception as _rse:
                        logger.warning(f"Failed to pre-compute RS ratings: {_rse}")
                        rs_dict = {}
                    try:
                        sector_rankings_dict = f_sre.result()
                    except Exception as _sre:
                        logger.warning(f"Failed to pre-compute sector regime rankings: {_sre}")
                        sector_rankings_dict = {}
            else:
                def _fetch_pledge():
                    try:
                        from database import get_pledge_map
                        p_map = get_pledge_map(symbols)
                        logger.info(f"🛡️ Fetched pledge data for {len(p_map)} symbols")
                        return p_map
                    except Exception as e:
                        logger.exception("Failed to fetch pledge map")
                        return {}

                def _fetch_delivery():
                    d_map = {}
                    d_days_back = 0
                    fallback_used = False
                    seen_dates = set()
                    for days_back in range(0, 5):
                        candidate = ist_now.date() - timedelta(days=days_back)
                        while candidate.weekday() >= 5:
                            candidate -= timedelta(days=1)
                        if candidate in seen_dates:
                            continue
                        seen_dates.add(candidate)
                        try:
                            d_map = fetch_delivery_data(candidate, skip_db_save=(days_back > 0))
                            if d_map:
                                d_days_back = (ist_now.date() - candidate).days
                                if d_days_back > 0:
                                    fallback_used = True
                                    logger.info(f"✅ EOD Scanner using FALLBACK Bhavcopy from: {candidate}")
                                    try:
                                        from push_service import send_push_to_all
                                        msg = f"EOD Scanner is using stale Bhavcopy (fallback from {candidate}) because today's data is not yet published."
                                        insert_notification("warning", "⚠️ Stale Bhavcopy Used", msg)
                                        send_push_to_all("⚠️ Stale Bhavcopy Used", msg, bypass_throttle=True)
                                    except Exception as ne:
                                        logger.error(f"Failed to send stale Bhavcopy notification: {ne}")
                                else:
                                    logger.info(f"✅ EOD Scanner using TODAY'S Bhavcopy from: {candidate}")
                                break
                        except Exception as e:
                            logger.debug(f"Delivery fetch failed for {candidate}: {e}")
                    return d_map, d_days_back, fallback_used

                with ThreadPoolExecutor(max_workers=5, thread_name_prefix="EOD_PreScan") as executor:
                    f_pledge = executor.submit(_fetch_pledge)
                    f_delivery = executor.submit(_fetch_delivery)
                    f_rot = executor.submit(get_sector_scores)
                    f_rs = executor.submit(compute_nifty_rs_rating, symbols)
                    f_sre = executor.submit(compute_sector_regime_rankings)

                    pledge_map = f_pledge.result()
                    delivery_map, delivery_days_back, _fb = f_delivery.result()
                    if _fb:
                        used_fallback_data = True
                    delivery_found = bool(delivery_map)
                    try:
                        rotation_result = f_rot.result()
                    except Exception:
                        rotation_result = SectorRotationResult({}, set(), set(), "", datetime.now(IST).date(), 0.0)
                    try:
                        rs_dict = f_rs.result()
                    except Exception as _rse:
                        logger.warning(f"Failed to pre-compute RS ratings: {_rse}")
                        rs_dict = {}
                    try:
                        sector_rankings_dict = f_sre.result()
                    except Exception as _sre:
                        logger.warning(f"Failed to pre-compute sector regime rankings: {_sre}")
                        sector_rankings_dict = {}

        total_alerts       = 0
        approved_candidates = []
        alerts_by_category = {}

        provider_stats_counts = {
            "SUCCESS": 0,
            "NOT_FOUND": 0,
            "RATE_LIMIT": 0,
            "NETWORK_ERROR": 0,
            "TIMEOUT": 0,
            "EMPTY_DATA": 0
        }
        scan_failures = []

        rejection_counts = {k: 0 for k in [
            "no_data", "missing_col", "indicator_nan", "insufficient_bars", "indicator_fail", "weak_signals",
            "weak_body", "bearish_candle", "weak_close_pos", "upper_wick", "low_volume",
            "low_avg_volume", "penny_stock", "rsi_range", "below_ema20",
            "below_sma50", "weak_adx", "far_from_52w_high",
            "gap_day", "extended_breakout", "gap_extended", "low_score", "duplicate", "stale_data",
            "prior_red_candles", "obv_divergence",
            "no_structural_breakout", "no_atr_expansion", "base_too_wide",
            "missing_atr", "zero_avg_volume", "zero_candle_range", "low_rr"
        ]}

        market_regime = get_macro_regime(nifty_ret_20d)
        telemetry_logger = ScannerDecisionLogger("EOD", scan_id, market_regime)
        logger.info(f"📊 Market Regime Classifier: {market_regime}")

        # [EOD_REGIME_CTX_FIX_v1.0] BUG-1 FIX: regime_ctx was never initialized in eod_scanner.
        # Only market_regime (a string) was built via get_macro_regime().
        # reversal_scanner and multi_tf_scanner both correctly build the full dict via
        # MarketRegimeEngine.get_regime_context(). Now aligned.
        try:
            regime_ctx = MarketRegimeEngine.get_regime_context(nifty_ret_20d)
            policy = StrategyPolicyEngine.get_policy(regime_ctx, "EOD")
            regime_ctx["policy"] = policy
        except Exception:
            logger.warning("⚠️ Could not build regime_ctx from MarketRegimeEngine — using neutral fallback")
            regime_ctx = {"trend": market_regime, "biases": {}}

        try:
            from database import get_latest_weights
            regime_str = regime_ctx.get("trend", "NEUTRAL")
            latest_db_weights = get_latest_weights(regime_str)
            if latest_db_weights:
                bayesian_weights = latest_db_weights.get("weights")
                bayesian_version = latest_db_weights.get("version", "v1")
            else:
                bayesian_weights = None
                bayesian_version = "v1"
        except Exception:
            bayesian_weights = None
            bayesian_version = "v1"

        # [BUG-1 FIX v1.5] Compute threshold BEFORE regime check to avoid NameError
        BASE_SCORE_THRESHOLD = SCORE_THRESHOLDS.get("1d", 75)
        global_min_score = BASE_SCORE_THRESHOLD
        regime_modifier = 0

        # Wire the threshold increase to read dynamically from the config.py regime modifiers
        try:
            from config import REGIME_POLICIES
            regime_modifier = REGIME_POLICIES.get(market_regime, {}).get("score_modifier", 0)
            if regime_modifier > 0:
                logger.info(f"🛑 {market_regime} regime detected — raising score threshold by +{regime_modifier}.")
                global_min_score += regime_modifier
        except Exception as e:
            logger.warning(f"Failed to fetch REGIME_POLICIES: {e}")

        # Cap the regime-adjusted threshold at 82 to prevent over-rejection in neutral/bear regimes
        global_min_score = min(global_min_score, 82)
        effective_global_min_score = global_min_score

        logger.info(
            f"📊 Score threshold: BASE_SCORE_THRESHOLD={BASE_SCORE_THRESHOLD} | "
            f"REGIME_STRICTNESS_PENALTY={regime_modifier:+d} ({market_regime} makes bar higher/stricter) => "
            f"EFFECTIVE_GLOBAL_MIN_SCORE={effective_global_min_score} (stricter qualification floor, capped <= 82)"
        )

        stage_tracker.end_stage(f"Pledge: {len(pledge_map)}, Delivery: {len(delivery_map)}")

        import gc
        BATCH_SIZE = int(os.environ.get("EOD_FETCH_BATCH_SIZE", "250"))

        from config import ALERT_COOLDOWN_MINUTES
        cooldown_alerts = get_recent_alerts_for_scanner("EOD", ALERT_COOLDOWN_MINUTES.get("EOD", 1440))

        total_fetched_count = 0
        stage_tracker.start_stage(3, "Price History Fetch & Symbol Evaluation Loop", f"Chunk size: {BATCH_SIZE}")
        # [VERSION: MARKET_DATA_SESSION_v1.0] Log whether session is available

        if session is not None:
            logger.info(f"📦 [EOD] Using MarketDataSession | {session.metadata.valid_symbols} symbols pre-fetched")
        else:
            logger.info(f"📥 Processing EOD phase in chunks of {BATCH_SIZE} (no session — independent fetch)...")

        from memory_profiler import chunk_iterable, BatchMemoryTracker
        total_batches = (len(watchlist) + BATCH_SIZE - 1) // BATCH_SIZE

        import psutil
        process = psutil.Process(os.getpid())

        total_fetched_count = 0

        with MemoryProfiler("Process Symbols"):
            for i in range(0, len(watchlist), BATCH_SIZE):
                batch_num = (i // BATCH_SIZE) + 1
                batch_start_time = time.time()
                chunk_df = watchlist.iloc[i:i + BATCH_SIZE]
                rss_before = process.memory_info().rss / 1024 / 1024

                for _batch_run in range(1):
                    with BatchMemoryTracker("EOD", batch_num, total_batches, len(chunk_df), collect_gc=False) as tracker:
                        import pandas as pd
                        _batch_start_t = time.perf_counter()
                        # [VERSION: MARKET_DATA_SESSION_v1.0] Serve from session when available;
                        # fall back to independent fetch otherwise.
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
                            # EOD Scanner requires SMA200 & 52W High (252 trading days = ~365 cal days max).
                            # Using period="1y" shares the exact same Parquet cache files with Wealth Engine & Reversal scanner,
                            # eliminating 50% data payload and preventing cache key fragmentation.
                            all_ticker_data = fetch_watchlist_data(chunk_df, interval="1d", period="1y", requester="EOD")

                        _fetch_dur = time.perf_counter() - _batch_start_t
                        if not all_ticker_data:
                            continue

                        valid_fetches = sum(1 for v in all_ticker_data.values() if isinstance(v, pd.DataFrame) and not v.empty)
                        total_fetched_count += valid_fetches
                        from core_enums import ProviderResult
                        rows_fetched = sum(len(df) for df in all_ticker_data.values() if isinstance(df, pd.DataFrame))
                        tracker.mark_fetch_complete(row_count=rows_fetched)

                        import threading
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        _batch_lock = threading.Lock()
                        def _process_row(idx, row_tuple, all_ticker_data=all_ticker_data):
                            symbol = "UNKNOWN"
                            _row_start_time = _time.perf_counter()
                            try:
                                if isinstance(row_tuple, dict):
                                    row = row_tuple
                                elif hasattr(row_tuple, '_asdict'):
                                    try:
                                        row = row_tuple._asdict()
                                    except Exception:
                                        row = {}
                                elif hasattr(row_tuple, '_fields'):
                                    try:
                                        row = dict(zip(row_tuple._fields, row_tuple))
                                    except Exception:
                                        row = {}
                                elif hasattr(row_tuple, 'to_dict'):
                                    try:
                                        row = row_tuple.to_dict()
                                    except Exception:
                                        row = {}
                                else:
                                    row = {}

                                if not isinstance(row, dict):
                                    row = {}

                                symbol   = row.get("Stock", "UNKNOWN")
                                category = row.get("Category", "MIDCAP")
                                sector   = row.get("Sector", None)

                                if symbol in get_live_blacklist():
                                    terminal_tracker.record_terminal(symbol, "BLACKLIST_EXCLUDED", "Surveillance live blacklist")
                                    return

                                # Robust symbol resolution across .NS / .BO suffixes
                                ticker_data = all_ticker_data.get(symbol)
                                if ticker_data is None:
                                    ticker_data = all_ticker_data.get(f"{symbol}.NS") or all_ticker_data.get(f"{symbol}.BO") or all_ticker_data.get(symbol.split('.')[0])

                                if ticker_data is None:
                                    with _batch_lock:
                                        rejection_counts["no_data"] += 1
                                        terminal_tracker.record_terminal(symbol, "DATA_UNAVAILABLE", "No price data returned by provider")
                                        telemetry_logger.record_reject(symbol, "DATA", "NO_DATA", None, None, start_time=_row_start_time)
                                    with _batch_lock:
                                        provider_stats_counts["EMPTY_DATA"] += 1
                                    with _batch_lock:
                                        scan_failures.append(ScanFailure(symbol=symbol, scanner_name="EOD", provider="unknown", failure_reason="missing data", scan_id=scan_id))
                                    return

                                if isinstance(ticker_data, ProviderResult):
                                    res = ticker_data
                                    with _batch_lock:
                                        provider_stats_counts[res.name] += 1
                                    if res != ProviderResult.SUCCESS:
                                        with _batch_lock:
                                            rejection_counts["no_data"] += 1
                                            terminal_tracker.record_terminal(symbol, "DATA_UNAVAILABLE", f"Provider error: {res.name}")
                                            telemetry_logger.record_reject(symbol, "DATA", "NO_DATA", None, None, start_time=_row_start_time)
                                        with _batch_lock:
                                            scan_failures.append(ScanFailure(symbol=symbol, scanner_name="EOD", provider="unknown", failure_reason=f"Provider error: {res.name}", scan_id=scan_id))
                                        return
                                else:
                                    with _batch_lock:
                                        provider_stats_counts["SUCCESS"] += 1

                                ticker = ticker_data.copy()

                                if ticker.empty:
                                    with _batch_lock:
                                        rejection_counts["no_data"] += 1
                                        terminal_tracker.record_terminal(symbol, "DATA_EMPTY", "Empty dataframe returned")
                                        telemetry_logger.record_reject(symbol, "DATA", "NO_DATA", None, None, start_time=_row_start_time)
                                    return

                                # If provider returned stale data (used as fallback during rate limits), skip EOD buy generation
                                if getattr(ticker, 'attrs', {}).get('is_stale'):
                                    with _batch_lock:
                                        rejection_counts["stale_data"] += 1
                                        terminal_tracker.record_terminal(symbol, "STALE_DATA", "Stale fallback data")
                                        telemetry_logger.record_reject(symbol, "DATA", "STALE_DATA", None, None, start_time=_row_start_time)
                                    return

                                if isinstance(ticker.columns, pd.MultiIndex):
                                    ticker.columns = ticker.columns.get_level_values(0)

                                ticker = ticker.loc[:, ~ticker.columns.duplicated()]

                                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                                missing_col   = False

                                for col_name in required_cols:
                                    if col_name not in ticker.columns:
                                        missing_col = True
                                        break
                                    if isinstance(ticker[col_name], pd.DataFrame):
                                        ticker[col_name] = ticker[col_name].iloc[:, 0]
                                    ticker[col_name] = pd.Series(ticker[col_name]).astype(float)

                                if missing_col:
                                    with _batch_lock:
                                        rejection_counts["missing_col"] += 1
                                        terminal_tracker.record_terminal(symbol, "MISSING_COL", "Required OHLCV columns missing")
                                        telemetry_logger.record_reject(symbol, "DATA", "MISSING_COL", None, None, start_time=_row_start_time)
                                    return

                                ticker = ticker.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

                                if ticker.empty:
                                    with _batch_lock:
                                        rejection_counts["no_data"] += 1
                                        terminal_tracker.record_terminal(symbol, "DATA_EMPTY", "Empty dataframe after dropna")
                                        telemetry_logger.record_reject(symbol, "DATA", "NO_DATA", None, None, start_time=_row_start_time)
                                    return

                                # [VERSION: EOD_BAR_LIMIT_FIX] Lowered bar minimum from 200 to 50 to allow IPOs/new listings to be evaluated
                                if len(ticker) < 50:
                                    with _batch_lock:
                                        rejection_counts["insufficient_bars"] += 1
                                        terminal_tracker.record_terminal(symbol, "INSUFFICIENT_BARS", f"History bars {len(ticker)} < 50")
                                        telemetry_logger.record_reject(symbol, "LIQUIDITY", "INSUFFICIENT_BARS", len(ticker) if "ticker" in locals() else 0, 50, start_time=_row_start_time)
                                    return

                                # [RULE 67 CHANGE-RATIONALE: MODULAR_TARGETED_HYDRATION_v1.0]
                                # ── STAGE 1: LIGHTWEIGHT STRUCTURAL HYDRATION ───────────────────
                                # Hydrate only lightweight structural columns (rolling highs, RSI, ATR, BASE_WIDTH, OBV)
                                # to perform fast breakout detection and near-miss checks without paying for MACD/ADX/SMAs/EMAs.
                                if "PRIOR_20D_HIGH" not in ticker.columns:
                                    ticker = hydrate_indicators(
                                        ticker,
                                        required={"PRIOR_20D_HIGH", "HIGH_52W", "HIGH_20D", "HIGH_50D", "HIGH_100D", "HIGH_252D", "RSI", "ATR", "ATR20", "BASE_WIDTH", "OBV"},
                                        timeframe="1d"
                                    )

                                if ticker is None or ticker.empty:
                                    with _batch_lock:
                                        rejection_counts["indicator_fail"] += 1
                                        terminal_tracker.record_terminal(symbol, "INDICATOR_FAIL", "Indicator calculation failed")
                                        telemetry_logger.record_reject(symbol, "DATA", "INDICATOR_FAIL", None, None, start_time=_row_start_time)
                                    return

                                latest = ticker.iloc[-1]
                                ctx = telemetry_logger.get_or_create_context(symbol)
                                ctx.capture_dataframe_row(latest, is_fallback=used_fallback_data)

                                signals = detect_breakouts(ticker, timeframe="1d")

                                if len(signals) < MIN_SIGNALS:
                                    cmp_val = _safe_float(latest.get("Close"), 0.0)
                                    prior_20d_high_val = _safe_float(latest.get("PRIOR_20D_HIGH"), 0.0)
                                    if prior_20d_high_val <= 0:
                                        if len(ticker) >= 21:
                                            prior_20d_high_val = float(ticker["High"].iloc[-21:-1].max())
                                        elif not ticker.empty:
                                            prior_20d_high_val = float(ticker["High"].max())
                                        else:
                                            prior_20d_high_val = cmp_val

                                    if len(ticker) >= 22:
                                        avg_vol = float(ticker["Volume"].iloc[-21:-1].mean())
                                    elif not ticker.empty:
                                        avg_vol = float(ticker["Volume"].iloc[:-1].mean())
                                    else:
                                        avg_vol = 1.0
                                    vol_ratio_val = (_safe_float(latest.get("Volume"), 0.0) / avg_vol) if avg_vol > 0 else 1.0
                                    rsi_val = _safe_float(latest.get("RSI"), 50.0)
                                    atr_val = _safe_float(latest.get("ATR"), 0.0)

                                    dist_pct = ((prior_20d_high_val - cmp_val) / prior_20d_high_val * 100.0) if prior_20d_high_val > 0 else 0.0
                                    sl_approx = round(cmp_val - 1.5 * atr_val, 2) if atr_val > 0 else round(cmp_val * 0.95, 2)
                                    tgt_approx = round(cmp_val + 3.0 * atr_val, 2) if atr_val > 0 else round(cmp_val * 1.10, 2)
                                    rr_approx = round((tgt_approx - cmp_val) / max(0.01, cmp_val - sl_approx), 2)

                                    # 1. Pre-Breakout / Compression Watchlist (Consolidating within 4.5% of resistance)
                                    if 0.0 <= dist_pct <= 4.5 and cmp_val > 0:
                                        logger.info(
                                            f"👁️ [EOD: PRE-BREAKOUT WATCH] {symbol} added to Watchlist (Base Consolidation | Dist: {dist_pct:.1f}%) | "
                                            f"CMP: ₹{cmp_val:.2f} | Pending Breakout Level: ₹{prior_20d_high_val:.2f} | RVOL: {vol_ratio_val:.2f}x | "
                                            f"RSI: {rsi_val:.1f} | SL: ₹{sl_approx:.2f} | RR: {rr_approx:.2f} — (Pending breakout trigger, not an active trade yet)"
                                        )

                                    # 2. Near-Miss Logging (Price tested/crossed resistance but volume fell short of ignition threshold)
                                    if cmp_val >= prior_20d_high_val and vol_ratio_val < 1.80 and cmp_val > 0:
                                        try:
                                            from near_miss_tracker import log_near_miss
                                            log_near_miss(
                                                symbol=symbol,
                                                scanner="EOD",
                                                breakout_type="20D_BREAKOUT",
                                                gate_name="insufficient_volume_surge",
                                                observed_value=vol_ratio_val,
                                                threshold_value=1.80,
                                                entry_price=cmp_val,
                                                stop_loss=sl_approx,
                                                target_1=tgt_approx
                                            )
                                        except Exception:
                                            pass

                                    # 3. Contextual Rejection Log
                                    logger.info(
                                        f"🚫 [EOD] {symbol} REJECTED — Insufficient breakout signals ({len(signals)} < {MIN_SIGNALS}) "
                                        f"(CMP: ₹{cmp_val:.2f} | 20D High: ₹{prior_20d_high_val:.2f} [dist: -{dist_pct:.1f}%] | RVOL: {vol_ratio_val:.2f}x | RSI: {rsi_val:.1f})"
                                    )
                                    with _batch_lock:
                                        rejection_counts["weak_signals"] += 1
                                        terminal_tracker.record_terminal(symbol, "WEAK_SIGNALS", f"Breakout signals {len(signals)} < {MIN_SIGNALS}")
                                        ctx.capture_gate(
                                            gate_name="WEAK_SIGNALS",
                                            passed=False,
                                            actual_val=len(signals),
                                            operator_str="<",
                                            threshold_val=MIN_SIGNALS if "MIN_SIGNALS" in locals() else 3,
                                            gate_type="COMPOSITE",
                                            reason="Insufficient breakout signals",
                                            components={"signals_count": len(signals), "detected_signals": signals},
                                            expression=f"len(signals) >= {MIN_SIGNALS}"
                                        )
                                        ctx.add_decision_input(name="signals_count", value=len(signals), source="BreakoutEngine", as_of="Live", freshness="LIVE", required=True, valid=True)
                                        ctx.finalize(decision="REJECTED", primary_reason="WEAK_SIGNALS")
                                        telemetry_engine.emit_terminal(ctx)
                                    return

                                if "RSI" not in ticker.columns or pd.isna(latest["RSI"]):
                                    logger.debug(f"[EOD] {symbol} rejected: latest RSI is missing or NaN")
                                    with _batch_lock:
                                        rejection_counts["indicator_nan"] += 1
                                        terminal_tracker.record_terminal(symbol, "INDICATOR_NAN", "Latest RSI is NaN or missing")
                                        telemetry_logger.record_reject(symbol, "DATA", "INDICATOR_NAN", None, None, start_time=_row_start_time)
                                    return

                                # [VERSION: EOD_PATCH_v1.1] [BUG FIX 8 REGRESSION FIX] Proper fallback to DatetimeIndex when Date/Datetime column is missing
                                # [FIX P0] Compare against the last bar's own date rather than ist_now.date().
                                # On weekends/holidays, ist_now.date() is a non-trading day and every symbol
                                # would be rejected as stale. Instead, we confirm the last bar is reasonably
                                # recent (within 4 calendar days to cover long weekends).
                                _last_ts = None
                                _stale_col = next((c for c in ["Date", "Datetime"] if c in ticker.columns), None)
                                if _stale_col:
                                    try:
                                        _last_ts = pd.to_datetime(latest[_stale_col])
                                    except Exception as e:
                                        logger.debug(f"⏭️ {symbol} timestamp parse error: {e}")
                                elif isinstance(ticker.index, pd.DatetimeIndex):
                                    try:
                                        _last_ts = pd.Timestamp(ticker.index[-1])
                                    except Exception as e:
                                        logger.debug(f"⏭️ {symbol} index timestamp parse error: {e}")

                                from market_utils import evaluate_data_staleness
                                _staleness = evaluate_data_staleness(_last_ts, ist_now)
                                if _staleness["is_stale"]:
                                    with _batch_lock:
                                        rejection_counts["stale_data"] += 1
                                        terminal_tracker.record_terminal(symbol, "STALE_DATA", f"Data stale till {_staleness['latest_available']}")
                                        telemetry_logger.record_reject(symbol, "DATA", "STALE_DATA", None, None, start_time=_row_start_time)
                                    logger.info(f"🚫 [EOD] {symbol} skipped — Data stale. Available till {_staleness['latest_available']} (Expected at least {_staleness['expected_date']})")
                                    return

                                # [VERSION: EOD_VOL_RATIO_FIX] Protect against newly listed stocks with <22 bars
                                if len(ticker) >= 22:
                                    avg_volume = float(ticker["Volume"].iloc[-21:-1].mean())
                                else:
                                    avg_volume = float(ticker["Volume"].iloc[:-1].mean())

                                if avg_volume <= 0:
                                    logger.debug(f"REJECTION: {symbol} (Phase: LIQUIDITY_FILTER, Reason: 20D average volume is zero)")
                                    with _batch_lock:
                                        rejection_counts["zero_avg_volume"] += 1
                                        terminal_tracker.record_terminal(symbol, "ZERO_AVG_VOLUME", "20D average volume is zero")
                                        telemetry_logger.record_reject(symbol, "LIQUIDITY", "ZERO_AVG_VOLUME", avg_volume if "avg_volume" in locals() else 0, 1, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                    return

                                volume_ratio = _safe_float(latest.get("Volume", 0)) / avg_volume

                                candle_high  = _safe_float(latest.get("High"))
                                candle_low   = _safe_float(latest.get("Low"))
                                candle_open  = _safe_float(latest.get("Open"))
                                candle_close = _safe_float(latest.get("Close"))
                                candle_range = candle_high - candle_low
                                candle_body  = abs(candle_close - candle_open)
                                upper_wick   = candle_high - max(candle_close, candle_open)

                                if candle_range < 0:
                                    with _batch_lock:
                                        rejection_counts["zero_candle_range"] += 1
                                        terminal_tracker.record_terminal(symbol, "ZERO_CANDLE_RANGE", "Negative candle range")
                                        telemetry_logger.record_reject(symbol, "STRUCTURE", "ZERO_CANDLE_RANGE", 0, 1, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                    return
                                elif candle_range == 0:
                                    if _safe_float(latest.get("Volume", 0)) <= 0:
                                        with _batch_lock:
                                            rejection_counts["zero_candle_range"] += 1
                                            terminal_tracker.record_terminal(symbol, "ZERO_CANDLE_RANGE", "Zero candle range and zero volume")
                                            telemetry_logger.record_reject(symbol, "STRUCTURE", "ZERO_CANDLE_RANGE", 0, 1, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                        return

                                if candle_range > 0:
                                    body_ratio     = candle_body / candle_range
                                    close_position = (candle_close - candle_low) / candle_range
                                    wick_ratio     = upper_wick / candle_range
                                else:
                                    body_ratio = 1.0
                                    close_position = 1.0
                                    wick_ratio = 0.0
                                rsi_val        = _safe_float(latest.get("RSI"))

                                # [FIX P1-3] Converted hard candle gates to scoring penalties.
                                # Previously these 4 conditions hard-rejected ~40% of valid breakouts.
                                # Now each applies a proportional penalty to the final score.
                                # candle_penalty is Bucket A in the three-bucket penalty model.
                                candle_penalty = 0
                                if body_ratio < MIN_BODY_RATIO:
                                    shortfall = (MIN_BODY_RATIO - body_ratio) / MIN_BODY_RATIO
                                    pen = min(15, int(shortfall * 30))
                                    candle_penalty += pen
                                    logger.debug(f"⚠️ {symbol} body_ratio penalty: -{pen} (ratio={body_ratio:.2f} < {MIN_BODY_RATIO})")
                                if candle_close <= candle_open:
                                    candle_penalty += 5
                                    logger.debug(f"⚠️ {symbol} bearish_candle penalty: -5")
                                if close_position < MIN_CLOSE_POSITION:
                                    shortfall = (MIN_CLOSE_POSITION - close_position) / MIN_CLOSE_POSITION
                                    pen = min(10, int(shortfall * 20))
                                    candle_penalty += pen
                                    logger.debug(f"⚠️ {symbol} close_position penalty: -{pen} (pos={close_position:.2f} < {MIN_CLOSE_POSITION})")
                                if wick_ratio > MAX_UPPER_WICK_RATIO:
                                    excess = (wick_ratio - MAX_UPPER_WICK_RATIO) / MAX_UPPER_WICK_RATIO
                                    pen = min(10, int(excess * 20))
                                    candle_penalty += pen
                                    logger.debug(f"⚠️ {symbol} upper_wick penalty: -{pen} (wick={wick_ratio:.2f} > {MAX_UPPER_WICK_RATIO})")
                                # Bucket A capped independently at -15
                                candle_penalty = min(15, candle_penalty)
                                if volume_ratio < MIN_VOLUME_RATIO:
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED — Volume ratio {volume_ratio:.2f}x < {MIN_VOLUME_RATIO:.2f}x (CMP: ₹{candle_close:.2f} | RSI: {rsi_val:.1f})")
                                    with _batch_lock:
                                        rejection_counts["low_volume"] += 1
                                        terminal_tracker.record_terminal(symbol, "LOW_VOLUME", f"Volume ratio {volume_ratio:.2f}x < {MIN_VOLUME_RATIO:.2f}x")
                                        telemetry_logger.record_reject(symbol, "VOLUME", "LOW_VOLUME", volume_ratio if "volume_ratio" in locals() else 0, MIN_VOLUME_RATIO if "MIN_VOLUME_RATIO" in locals() else 1.0, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                    return
                                if avg_volume < MIN_AVG_VOLUME_SHARES:
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED — Avg volume {avg_volume:.0f} < {MIN_AVG_VOLUME_SHARES:.0f} shares (CMP: ₹{candle_close:.2f})")
                                    with _batch_lock:
                                        rejection_counts["low_avg_volume"] += 1
                                        terminal_tracker.record_terminal(symbol, "LOW_AVG_VOLUME", f"Avg volume {avg_volume:.0f} < {MIN_AVG_VOLUME_SHARES:.0f}")
                                        telemetry_logger.record_reject(symbol, "LIQUIDITY", "LOW_AVG_VOLUME", avg_volume if "avg_volume" in locals() else 0, MIN_AVG_VOLUME_SHARES if "MIN_AVG_VOLUME_SHARES" in locals() else 50000, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                    return
                                if candle_close < MIN_STOCK_PRICE:
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED — Close ₹{candle_close:.2f} < ₹{MIN_STOCK_PRICE:.2f}")
                                    with _batch_lock:
                                        rejection_counts["penny_stock"] += 1
                                        terminal_tracker.record_terminal(symbol, "PENNY_STOCK", f"Close ₹{candle_close:.2f} < ₹{MIN_STOCK_PRICE:.2f}")
                                        telemetry_logger.record_reject(symbol, "PRICE", "PENNY_STOCK", candle_close if "candle_close" in locals() else 0, MIN_STOCK_PRICE if "MIN_STOCK_PRICE" in locals() else 20, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                    return
                                if not (MIN_RSI <= rsi_val <= MAX_RSI):
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED — RSI {rsi_val:.1f} outside [{MIN_RSI}, {MAX_RSI}] range (CMP: ₹{candle_close:.2f} | RVOL: {volume_ratio:.2f}x)")
                                    with _batch_lock:
                                        rejection_counts["rsi_range"] += 1
                                        terminal_tracker.record_terminal(symbol, "RSI_RANGE", f"RSI {rsi_val:.1f} outside [{MIN_RSI}, {MAX_RSI}]")
                                        telemetry_logger.record_reject(symbol, "RSI", "RSI_RANGE", rsi_val if "rsi_val" in locals() else 0, f"[{MIN_RSI}, {MAX_RSI}]", start_time=_row_start_time, operator="NOT_IN_RANGE", gate_type="THRESHOLD")
                                    return

                                # [FIX: RSI_CEILING_TO_PENALTY] RSI 88-92 is a graduated scoring
                                # penalty, not a hard reject. Stocks in genuine breakouts routinely
                                # hit RSI 88-95 on the ignition day. Hard rejection at 88 was
                                # contradictory to the structural breakout + 52W high requirements.
                                # 92 remains the hard ceiling (MAX_RSI); this penalty fires for [88,92).
                                _RSI_PENALTY_THRESHOLD = 88.0
                                rsi_penalty = 0
                                if rsi_val > _RSI_PENALTY_THRESHOLD:
                                    rsi_excess = rsi_val - _RSI_PENALTY_THRESHOLD
                                    rsi_penalty = min(10, int(rsi_excess * 2.5))
                                    # Actual penalty ladder (formula: int(excess * 2.5), cap 10):
                                    #   RSI 89 → excess 1.0 → int(2.5) = 2 pts
                                    #   RSI 90 → excess 2.0 → int(5.0) = 5 pts
                                    #   RSI 91 → excess 3.0 → int(7.5) = 7 pts
                                    #   RSI 92 → excess 4.0 → int(10)  = 10 pts (cap)
                                    #   RSI >92 → hard reject (MAX_RSI gate above)
                                    logger.debug(f"⚠️ {symbol} RSI overextension penalty: -{rsi_penalty} (RSI={rsi_val:.1f} > {_RSI_PENALTY_THRESHOLD})")

                                # ── v6: STRUCTURAL BREAKOUT FILTERS ─────────────────────────────
                                # [VERSION: EOD_PATCH_v1.0] [BUG FIX 2] Added explicit outer else rejection to avoid silent bypass of structural filters
                                if "PRIOR_20D_HIGH" not in ticker.columns or pd.isna(latest.get("PRIOR_20D_HIGH")):
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED — Missing PRIOR_20D_HIGH indicator")
                                    with _batch_lock:
                                        rejection_counts["missing_atr"] += 1
                                        terminal_tracker.record_terminal(symbol, "MISSING_STRUCTURE_INDICATOR", "Missing PRIOR_20D_HIGH indicator")
                                        telemetry_logger.record_reject(symbol, "STRUCTURE", "MISSING_ATR", None, None, start_time=_row_start_time)
                                    return

                                prior_high = _safe_float(latest.get("PRIOR_20D_HIGH"))
                                if prior_high <= 0:
                                    logger.debug(f"REJECTION: {symbol} (Phase: STRUCTURAL_BREAKOUT, Reason: Invalid prior 20D high ₹{prior_high:.2f})")
                                    with _batch_lock:
                                        rejection_counts["no_structural_breakout"] += 1
                                        terminal_tracker.record_terminal(symbol, "NO_STRUCTURAL_BREAKOUT", f"Invalid prior 20D high ₹{prior_high:.2f}")
                                        telemetry_logger.record_reject(symbol, "STRUCTURE", "NO_BREAKOUT", candle_close if "candle_close" in locals() else 0, prior_high if "prior_high" in locals() else 0, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                    return

                                if candle_close <= prior_high:
                                    logger.debug(f"REJECTION: {symbol} (Phase: STRUCTURAL_BREAKOUT, Reason: Close ₹{candle_close:.2f} <= Prior 20D High ₹{prior_high:.2f})")
                                    with _batch_lock:
                                        rejection_counts["no_structural_breakout"] += 1
                                        terminal_tracker.record_terminal(symbol, "NO_STRUCTURAL_BREAKOUT", f"Close ₹{candle_close:.2f} <= Prior High ₹{prior_high:.2f}")
                                        telemetry_logger.record_reject(symbol, "STRUCTURE", "NO_BREAKOUT", candle_close if "candle_close" in locals() else 0, prior_high if "prior_high" in locals() else 0, start_time=_row_start_time, operator="<=", gate_type="THRESHOLD")
                                    return

                                # Not Extended
                                if "ATR20" not in ticker.columns or pd.isna(latest.get("ATR20")):
                                    with _batch_lock:
                                        rejection_counts["missing_atr"] += 1
                                        terminal_tracker.record_terminal(symbol, "MISSING_ATR", "Missing ATR20 column")
                                        telemetry_logger.record_reject(symbol, "STRUCTURE", "MISSING_ATR", None, None, start_time=_row_start_time)
                                    return

                                atr20 = _safe_float(latest.get("ATR20"))
                                if atr20 <= 0:
                                    with _batch_lock:
                                        rejection_counts["missing_atr"] += 1
                                        terminal_tracker.record_terminal(symbol, "MISSING_ATR", "ATR20 <= 0")
                                        telemetry_logger.record_reject(symbol, "STRUCTURE", "MISSING_ATR", None, None, start_time=_row_start_time)
                                    return

                                # [VERSION: BUSINESS_LOGIC_FIX_v1.0] Gap-and-go penalty (Soft Gate)
                                technical_penalties = {}
                                atr_extension = (candle_close - prior_high) / atr20
                                max_ext = EOD_ADVANCED_CONFIG.get("MAX_EXTENDED_BREAKOUT_ATR_MULT", 1.5)
                                if atr_extension > max_ext:
                                    pen_mult = EOD_ADVANCED_CONFIG.get("GAP_AND_GO_PENALTY_MULT", 10)
                                    max_pen = EOD_ADVANCED_CONFIG.get("GAP_AND_GO_MAX_PENALTY", 20)
                                    technical_penalties["extended_breakout"] = min(max_pen, (atr_extension - max_ext) * pen_mult)

                                # ATR Expansion
                                import circuit_helper
                                is_circuit = circuit_helper.is_valid_circuit_candle(
                                    candle_range=candle_range,
                                    volume=_safe_float(latest.get("Volume")),
                                    close_price=candle_close
                                )

                                min_atr_expansion = EOD_ADVANCED_CONFIG.get("MIN_ATR_EXPANSION_RATIO", 0.9)

                                if is_circuit:
                                    atr_expansion = None
                                    # Bypass check, do not reject.
                                else:
                                    atr_expansion = candle_range / atr20
                                    if atr_expansion < min_atr_expansion:
                                        logger.debug(f"REJECTION: {symbol} (Phase: ATR_EXPANSION, Reason: Candle range / ATR20 ({atr_expansion:.2f}) < {min_atr_expansion:.1f})")
                                        with _batch_lock:
                                            rejection_counts["no_atr_expansion"] += 1
                                            terminal_tracker.record_terminal(symbol, "NO_ATR_EXPANSION", f"Candle range / ATR20 ({atr_expansion:.2f}) < {min_atr_expansion:.1f}")
                                            telemetry_logger.record_reject(symbol, "STRUCTURE", "NO_ATR_EXPANSION", atr_expansion, min_atr_expansion, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                        return

                                # ── STAGE 2: FULL INDICATOR HYDRATION FOR QUALIFIED CANDIDATES ──
                                # At this point, the symbol has cleared all Stage 1 structural, volume,
                                # and breakout gates. Hydrate the complete indicator suite (EMA/SMA/ADX/MACD/BB)
                                # before evaluating trend alignment and Bayesian composite score.
                                ticker = hydrate_indicators(ticker, required=None, timeframe="1d")
                                latest = ticker.iloc[-1]

                                ctx.capture_indicators(
                                    rsi=_safe_float(latest.get("RSI")),
                                    sma20=_safe_float(latest.get("SMA20")),
                                    sma50=_safe_float(latest.get("SMA50")),
                                    sma100=_safe_float(latest.get("SMA100")),
                                    sma200=_safe_float(latest.get("SMA200")),
                                    ema9=_safe_float(latest.get("EMA9")),
                                    ema15=_safe_float(latest.get("EMA15")),
                                    ema20=_safe_float(latest.get("EMA20")),
                                    ema50=_safe_float(latest.get("EMA50")),
                                    ema200=_safe_float(latest.get("EMA200")),
                                    macd=_safe_float(latest.get("MACD")),
                                    macd_signal=_safe_float(latest.get("MACD_SIGNAL")),
                                    macd_hist=_safe_float(latest.get("MACD_HIST")),
                                    atr=_safe_float(latest.get("ATR")),
                                    adx=_safe_float(latest.get("ADX")),
                                    obv=_safe_float(latest.get("OBV")),
                                    prior_20d_high=_safe_float(latest.get("PRIOR_20D_HIGH")),
                                    bb_width_pctile=_safe_float(latest.get("BB_WIDTH_PCTILE"))
                                )
                                with _batch_lock:
                                    waterfall_counts["structure_entered"] += 1

                                if "EMA20" in ticker.columns and not pd.isna(latest.get("EMA20")):
                                    if candle_close < _safe_float(latest.get("EMA20")):
                                        logger.debug(f"REJECTION: {symbol} (Phase: EMA20_TREND, Reason: Close ₹{candle_close:.2f} < EMA20 ₹{_safe_float(latest.get('EMA20')):.2f})")
                                        with _batch_lock:
                                            rejection_counts["below_ema20"] += 1
                                            terminal_tracker.record_terminal(symbol, "BELOW_EMA20", f"Close ₹{candle_close:.2f} < EMA20")
                                            telemetry_logger.record_reject(symbol, "TREND", "BELOW_EMA20", candle_close if "candle_close" in locals() else 0, _safe_float(latest.get("EMA20")) if "latest" in locals() else 0, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                        return

                                if "SMA50" in ticker.columns and not pd.isna(latest.get("SMA50")):
                                    if candle_close < _safe_float(latest.get("SMA50")):
                                        logger.debug(f"REJECTION: {symbol} (Phase: SMA50_TREND, Reason: Close ₹{candle_close:.2f} < SMA50 ₹{_safe_float(latest.get('SMA50')):.2f})")
                                        with _batch_lock:
                                            rejection_counts["below_sma50"] += 1
                                            terminal_tracker.record_terminal(symbol, "BELOW_SMA50", f"Close ₹{candle_close:.2f} < SMA50")
                                            telemetry_logger.record_reject(symbol, "TREND", "BELOW_SMA50", candle_close if "candle_close" in locals() else 0, _safe_float(latest.get("SMA50")) if "latest" in locals() else 0, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                        return

                                if "ADX" in ticker.columns and not pd.isna(latest.get("ADX")):
                                    if _safe_float(latest.get("ADX")) < ADX_MIN_THRESHOLD:
                                        logger.debug(f"REJECTION: {symbol} (Phase: ADX_GATE, Reason: ADX {_safe_float(latest.get('ADX')):.1f} < {ADX_MIN_THRESHOLD})")
                                        with _batch_lock:
                                            rejection_counts["weak_adx"] += 1
                                            terminal_tracker.record_terminal(symbol, "WEAK_ADX", f"ADX {_safe_float(latest.get('ADX')):.1f} < {ADX_MIN_THRESHOLD}")
                                            telemetry_logger.record_reject(symbol, "TREND", "WEAK_ADX", _safe_float(latest.get("ADX")) if "latest" in locals() else 0, ADX_MIN_THRESHOLD if "ADX_MIN_THRESHOLD" in locals() else 20, start_time=_row_start_time, operator="<", gate_type="THRESHOLD")
                                        return

                                # MACD is no longer mandatory, shifted to scoring engine

                                # [FIX: TWO_MODE_52W] Two-mode 52W high gate.
                                # Mode A (High Breakout): stock within 5% of 52W high — direct pass.
                                # Mode B (Recovery Breakout): 5-15% below 52W high with stricter
                                #   secondary checks: volume >= 2.5×, BB tightness <= 0.50,
                                #   RS percentile >= 60, RSI <= 88.
                                #   Mode B candidates receive a flat -5 score penalty.
                                _breakout_mode = "A"  # default
                                recovery_adjustment = 0
                                if "HIGH_52W" in ticker.columns and not pd.isna(latest.get("HIGH_52W")):
                                    high_52w = _safe_float(latest.get("HIGH_52W"))
                                    if high_52w > 0:
                                        pct_from_high = (high_52w - candle_close) / high_52w * 100
                                        if pct_from_high > MAX_DISTANCE_FROM_52W_HIGH_PCT:
                                            # Mode A failed. Evaluate Mode B.
                                            _rec_max_dist = EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_MAX_DISTANCE_PCT", 15.0)
                                            _rec_min_vol  = EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_MIN_VOL_RATIO", 2.5)
                                            _rec_max_bb   = EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_MAX_BB_WIDTH", 0.50)
                                            _rec_min_rs   = EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_MIN_RS_PCT", 60.0)
                                            # [FIX: NO_MODE_B_RSI_CEILING] No separate RSI ceiling for Mode B.
                                            # Mode B is constrained by: RVOL >= 2.5x, BB <= 0.50, RS >= 60,
                                            # and the global RSI hard cap (92) + all structural/trend gates.
                                            # A tighter RSI ceiling would recreate the same contradictory
                                            # logic we are removing from Mode A.

                                            _mode_b_distance_ok = pct_from_high <= _rec_max_dist
                                            _mode_b_volume_ok   = volume_ratio >= _rec_min_vol
                                            _mode_b_bb_ok       = True
                                            if "BB_WIDTH_PCTILE" in ticker.columns and len(ticker) >= 2:
                                                _mode_b_bb_ok = _safe_float(ticker["BB_WIDTH_PCTILE"].iloc[-2]) <= _rec_max_bb
                                            _mode_b_rs_ok  = float(rs_dict.get(symbol, 0.0)) >= _rec_min_rs

                                            if _mode_b_distance_ok and _mode_b_volume_ok and _mode_b_bb_ok and _mode_b_rs_ok:
                                                _breakout_mode = "B"
                                                # Mode B: record recovery adjustment (applied post-score calculation for full auditability)
                                                recovery_adjustment = -int(EOD_ADVANCED_CONFIG.get("RECOVERY_BREAKOUT_SCORE_PENALTY", 5))
                                                logger.debug(f"⚠️ {symbol} Mode B Recovery Breakout: dist={pct_from_high:.1f}%, vol={volume_ratio:.2f}x, rs={float(rs_dict.get(symbol, 0)):.0f}, recovery_adjustment={recovery_adjustment}")
                                            else:
                                                _fail_reasons = []
                                                if not _mode_b_distance_ok: _fail_reasons.append(f"dist {pct_from_high:.1f}%>{_rec_max_dist}%")
                                                if not _mode_b_volume_ok:   _fail_reasons.append(f"vol {volume_ratio:.2f}x<{_rec_min_vol}x")
                                                if not _mode_b_bb_ok:       _fail_reasons.append("base not tight")
                                                if not _mode_b_rs_ok:       _fail_reasons.append(f"RS {float(rs_dict.get(symbol,0)):.0f}<{_rec_min_rs}")
                                                with _batch_lock:
                                                    rejection_counts["far_from_52w_high"] += 1
                                                    terminal_tracker.record_terminal(symbol, "FAR_FROM_52W_HIGH", f"dist={pct_from_high:.1f}% failed Mode A(≤{MAX_DISTANCE_FROM_52W_HIGH_PCT}%) and Mode B({'; '.join(_fail_reasons)})")
                                                    telemetry_logger.record_reject(symbol, "STRUCTURE", "FAR_FROM_52W_HIGH", pct_from_high, MAX_DISTANCE_FROM_52W_HIGH_PCT, start_time=_row_start_time, operator=">", gate_type="THRESHOLD")
                                                return

                                if len(ticker) >= 2:
                                    prev_close = _safe_float(ticker["Close"].iloc[-2])
                                    if prev_close > 0:
                                        single_move_pct = abs(candle_close - prev_close) / prev_close * 100
                                        max_single_day_move_pct = EOD_ADVANCED_CONFIG.get("MAX_SINGLE_DAY_MOVE_PCT", 15.0)
                                        if single_move_pct > max_single_day_move_pct:
                                            with _batch_lock:
                                                rejection_counts["gap_day"] += 1
                                                terminal_tracker.record_terminal(symbol, "GAP_DAY", f"Move {single_move_pct:.1f}% > {max_single_day_move_pct}%")
                                                telemetry_logger.record_reject(symbol, "STRUCTURE", "GAP_DAY", single_move_pct if "single_move_pct" in locals() else 0, 15.0, start_time=_row_start_time, operator=">", gate_type="THRESHOLD")
                                            return

                                # [FIX P1-1] Removed hard 3% gap filter — gap penalty is now
                                # applied as a scoring penalty via technical_penalties below.
                                # Previously this hard-rejected valid breakout candidates that
                                # gapped up on strong institutional demand.

                                # [FIX P1-1] Gap penalty: proportional scoring penalty instead of hard reject.
                                # Stocks gapping up >3% on breakout day are penalized but not killed.
                                gap_lookback_bars = EOD_ADVANCED_CONFIG.get("GAP_LOOKBACK_BARS", 10)
                                max_gap_pct = EOD_ADVANCED_CONFIG.get("MAX_GAP_FROM_PRIOR_HIGH_PCT", 3.0)
                                if len(ticker) >= gap_lookback_bars + 1:
                                    gap_reference_high = float(ticker["High"].iloc[-(gap_lookback_bars + 1):-1].max())
                                    if gap_reference_high > 0:
                                        gap_pct = (candle_open - gap_reference_high) / gap_reference_high * 100
                                        if gap_pct > max_gap_pct:
                                            excess = gap_pct - max_gap_pct
                                            pen = min(20, int(excess * 3))
                                            technical_penalties["gap_extended"] = pen
                                            logger.debug(f"⚠️ {symbol} gap penalty: -{pen} (gap={gap_pct:.1f}%)")

                                delivery_pct = delivery_map.get(symbol, None)

                                # ── v5: PREVIOUS CANDLE CONTEXT FILTER ─────────────────────────────
                                lookback = EOD_ADVANCED_CONFIG.get("PRE_BREAKOUT_LOOKBACK_BARS", 5)
                                # [FIX: RED_CANDLE_DEFAULT_MISMATCH] Default was hardcoded to 2 here
                                # but config = 3 and _check_eod_conditions (UI path) defaulted to 3.
                                # Both paths now agree: default = 3, matching config intent.
                                max_red = EOD_ADVANCED_CONFIG.get("MAX_PRE_BREAKOUT_RED_CANDLES", 3)
                                # [FIX: TIGHT_BASE_THRESHOLD_MISMATCH] UI path defaulted to 0.50,
                                # production loop defaulted to 0.35. Both now default to 0.50 (config value).
                                tight_base_threshold = EOD_ADVANCED_CONFIG.get("TIGHT_BASE_BB_WIDTH_PCTILE", 0.50)

                                if len(ticker) >= (lookback + 1):
                                    red_count = 0
                                    for _ri in range(-(lookback + 1), -1):
                                        if _safe_float(ticker["Close"].iloc[_ri]) < _safe_float(ticker["Open"].iloc[_ri]):
                                            red_count += 1

                                    if red_count > max_red:
                                        # Too many red candles. Reject unless it's a very tight base (volatility compression)
                                        is_tight_base = False
                                        if "BB_WIDTH_PCTILE" in ticker.columns and len(ticker) >= 2:
                                            if _safe_float(ticker["BB_WIDTH_PCTILE"].iloc[-2]) <= tight_base_threshold:
                                                is_tight_base = True

                                        if not is_tight_base:
                                            pen = (red_count - max_red) * 2
                                            technical_penalties["too_many_red_candles"] = pen
                                            logger.debug(f"⚠️ {symbol} pre-breakout trend too red ({red_count}/{lookback}) — applying -{pen} penalty")

                                # ── v5: BASE TIGHTNESS FILTER ──────────────────────────────────────────
                                if "BB_WIDTH_PCTILE" in ticker.columns and len(ticker) >= 2:
                                    bb_width_pctile = _safe_float(ticker["BB_WIDTH_PCTILE"].iloc[-2])
                                    if bb_width_pctile > EOD_ADVANCED_CONFIG.get("MAX_BB_WIDTH_PCTILE", 0.80):
                                        logger.debug(f"  ⊘ {symbol} base too wide (BB Pctile {bb_width_pctile:.2f}) — skipping")
                                        with _batch_lock:
                                            rejection_counts["base_too_wide"] += 1
                                            terminal_tracker.record_terminal(symbol, "BASE_TOO_WIDE", f"BB width pctile {bb_width_pctile:.2f} > 0.80")
                                            telemetry_logger.record_reject(symbol, "STRUCTURE", "BASE_TOO_WIDE", bb_width_pctile if "bb_width_pctile" in locals() else 0, 0.80, start_time=_row_start_time)
                                        return

                                # ── [RULE 67: EOD_ATR_MODEL_A_V1] BASE ATR10 CALIBRATION GATE ───────────
                                # ATR10 / Close <= 2.50%: 0 penalty (optimal tightness)
                                # 2.51% - 3.50%: -3 penalty (minor volatility penalty)
                                # 3.51% - 4.50%: -7 penalty (moderate volatility penalty)
                                # > 4.50%: Hard Reject (structural ceiling)
                                base_atr_penalty = 0
                                base_atr_pct = None
                                if len(ticker) >= 12 and candle_close > 0:
                                    import numpy as _np
                                    highs_10 = ticker["High"].iloc[-11:-1]
                                    lows_10 = ticker["Low"].iloc[-11:-1]
                                    closes_10 = ticker["Close"].iloc[-12:-2]
                                    tr_10 = _np.maximum(highs_10 - lows_10, _np.maximum(_np.abs(highs_10 - closes_10), _np.abs(lows_10 - closes_10)))
                                    atr_10 = float(tr_10.mean())
                                    _atr10_base_price = _safe_float(ticker["Close"].iloc[-2]) if len(ticker) >= 2 else candle_close
                                    if _atr10_base_price <= 0:
                                        _atr10_base_price = candle_close
                                    base_atr_pct = round((atr_10 / _atr10_base_price) * 100.0, 4)

                                    with _batch_lock:
                                        waterfall_counts.setdefault("base_atr_list", []).append(base_atr_pct)
                                        if base_atr_pct <= 2.5000 + 1e-7:
                                            waterfall_counts["atr_le_250"] = waterfall_counts.get("atr_le_250", 0) + 1
                                            base_atr_penalty = 0
                                        elif base_atr_pct <= 3.5000 + 1e-7:
                                            waterfall_counts["atr_251_350"] = waterfall_counts.get("atr_251_350", 0) + 1
                                            waterfall_counts["recovered_from_cliff"] = waterfall_counts.get("recovered_from_cliff", 0) + 1
                                            base_atr_penalty = 3
                                        elif base_atr_pct <= 4.5000 + 1e-7:
                                            waterfall_counts["atr_351_450"] = waterfall_counts.get("atr_351_450", 0) + 1
                                            waterfall_counts["recovered_from_cliff"] = waterfall_counts.get("recovered_from_cliff", 0) + 1
                                            base_atr_penalty = 7
                                        else:
                                            waterfall_counts["atr_gt_450_rejected"] = waterfall_counts.get("atr_gt_450_rejected", 0) + 1
                                            logger.info(f"🚫 [EOD] {symbol} REJECTED — Base ATR10 ({base_atr_pct:.2f}%) > 4.50% tightness ceiling")
                                            rejection_counts["base_atr_too_wide"] = rejection_counts.get("base_atr_too_wide", 0) + 1
                                            terminal_tracker.record_terminal(symbol, "BASE_ATR_TOO_WIDE", f"Base ATR10 {base_atr_pct:.2f}% > 4.50%")
                                            telemetry_logger.record_reject(symbol, "STRUCTURE", "BASE_ATR_TOO_WIDE", base_atr_pct, 4.50, start_time=_row_start_time)
                                            return

                                # ── v6: OBV STRUCTURE — SCORING PENALTY (not hard reject) ──────────
                                # [FINDING-8 FIX] OBV_SLOPE is a 3-bar diff which is noisy on breakout
                                # days. Converted from hard reject to a -5 score penalty applied after
                                # scoring. The scoring engine already penalizes via BASE_WIDTH and
                                # unsustained volume checks.
                                obv_penalty = 0
                                if "OBV_SLOPE" in ticker.columns and not pd.isna(latest.get("OBV_SLOPE")):
                                    if _safe_float(latest.get("OBV_SLOPE")) <= EOD_ADVANCED_CONFIG.get("MIN_OBV_SLOPE", 0.0):
                                        obv_penalty = -5
                                        logger.debug(f"⚠️ {symbol} OBV divergence detected (slope <= 0), applying -5 penalty")

                                atr_val_eod = (
                                    _safe_float(latest.get("ATR"))
                                    if "ATR" in ticker.columns and not pd.isna(latest.get("ATR"))
                                    else None
                                )

                                with _batch_lock:
                                    waterfall_counts["quality_entered"] += 1

                                score, model_version, applied_bayesian_weights = calculate_score(
                                    category=category,
                                    breakout_count=len(signals),
                                    rsi=rsi_val,
                                    volume_ratio=volume_ratio,
                                    breakout_signals=signals,
                                    ticker=ticker,
                                    latest=latest,
                                    symbol=symbol,
                                    timeframe="1d",
                                    atr_val=atr_val_eod,
                                    delivery_pct=delivery_pct,
                                    promoter_pledge_pct=pledge_map.get(symbol),
                                    nifty_ret=nifty_ret_20d,
                                    regime_ctx=regime_ctx,
                                    bayesian_weights=bayesian_weights,
                                    bayesian_version=bayesian_version
                                )

                                # Default momentum values in case score <= 0 or gating fails
                                rs_pct_val = float(rs_dict.get(symbol, 50.0))
                                rs_bonus_val = 0
                                sector_bonus_val = 0
                                total_momentum_bonus = 0
                                base_score_val = int(score)

                                if score > 0:
                                    # [FIX: PENALTY_BUCKETS] Three-bucket penalty architecture replaces the
                                    # single combined -15 cap. Each bucket is independently capped so that
                                    # simultaneous weaknesses across all three dimensions accumulate properly.
                                    #
                                    # Bucket A — Candle quality (already capped at -15 above at line ~1283)
                                    _bucket_candle = candle_penalty  # already min(15, ...)

                                    # Bucket B — Gap & extension penalties (capped at -15 independently)
                                    _gap_pen   = technical_penalties.get("gap_extended", 0)
                                    _ext_pen   = technical_penalties.get("extended_breakout", 0)
                                    _red_pen   = technical_penalties.get("too_many_red_candles", 0)
                                    _bucket_gap = min(15, _gap_pen + _ext_pen)

                                    # Bucket C — OBV divergence (capped at -5)
                                    _bucket_obv = min(5, abs(obv_penalty))

                                    # RSI overextension penalty (added directly, cap already built into formula)
                                    # rsi_penalty was computed at line ~1317 above

                                    # Red candle and RSI penalties applied separately (not folded into gap bucket)
                                    _bucket_misc = min(10, _red_pen + rsi_penalty)

                                    total_deductions = _bucket_candle + _bucket_gap + _bucket_obv + _bucket_misc + base_atr_penalty

                                    # [FIX: TRIPLE_FAULT_VETO] If all three primary quality dimensions
                                    # simultaneously show serious weakness, reject regardless of score.
                                    # This catches setups that individually scrape by but collectively signal poor quality.
                                    _CANDLE_FAULT_THRESHOLD = 10  # candle bucket >= 10 = seriously bad candle
                                    _GAP_FAULT_THRESHOLD    = 10  # gap bucket >= 10 = oversized gap
                                    if (_bucket_candle >= _CANDLE_FAULT_THRESHOLD and
                                        _bucket_gap    >= _GAP_FAULT_THRESHOLD and
                                        _bucket_obv    > 0):
                                        logger.info(f"🚫 [EOD] {symbol} REJECTED — TRIPLE_FAULT_VETO (candle:{_bucket_candle} gap:{_bucket_gap} obv:{_bucket_obv})")
                                        with _batch_lock:
                                            rejection_counts["triple_fault_reject"] = rejection_counts.get("triple_fault_reject", 0) + 1
                                            terminal_tracker.record_terminal(symbol, "TRIPLE_FAULT_REJECT",
                                                f"Simultaneous fault: candle={_bucket_candle}pts, gap={_bucket_gap}pts, OBV divergence")
                                            telemetry_logger.record_reject(symbol, "QUALITY", "TRIPLE_FAULT_REJECT",
                                                total_deductions, _CANDLE_FAULT_THRESHOLD, start_time=_row_start_time)
                                        return

                                    score = max(0, score - total_deductions)

                                    base_score_val = int(score)

                                    # ── Feature F-03 & F-07: Momentum Bonus Injection (Prior to Score Gate) ──
                                    rs_bonus_val = RS_BONUS if rs_pct_val >= 80.0 else 0

                                    safe_sec_str = "Unknown" if (sector is None or (isinstance(sector, float) and pd.isna(sector))) else str(sector).strip()
                                    sector_info = sector_rankings_dict.get(safe_sec_str, {})
                                    sector_status = sector_info.get("effective_status", "NEUTRAL")
                                    sector_bonus_val = SECTOR_BONUS if sector_status == "TAILWIND" else 0

                                    total_momentum_bonus = min(MAX_MOMENTUM_BONUS, rs_bonus_val + sector_bonus_val)
                                    score = max(0, min(100, score + total_momentum_bonus))

                                    # [FIX: MODE_B_RECOVERY_ADJUSTMENT] Apply recovery adjustment after
                                    # the main score calculation and bonuses so base score components remain
                                    # uncorrupted and fully auditable.
                                    if recovery_adjustment != 0:
                                        score = max(0, score + recovery_adjustment)

                                # ── FORENSIC RISK TIER POLICY CHECK ──────────────────────────────────────
                                forensic_tier = row.get("Forensic_Risk_Tier", "UNKNOWN")
                                if forensic_tier == "REJECT":
                                    f_reason = "Forensic Risk Engine tier REJECT"
                                    f_details_raw = row.get("Forensic_Details")
                                    if f_details_raw:
                                        try:
                                            f_details = json.loads(f_details_raw) if isinstance(f_details_raw, str) else f_details_raw
                                            if isinstance(f_details, dict) and f_details.get("reason"):
                                                f_reason = f"Forensic tier REJECT ({f_details['reason']})"
                                        except Exception:
                                            pass
                                    with _batch_lock:
                                        rejection_counts["forensic_reject"] = rejection_counts.get("forensic_reject", 0) + 1
                                        terminal_tracker.record_terminal(symbol, "FORENSIC_REJECT", f_reason)
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED — {f_reason}")
                                    return

                                signal_str = ", ".join(signals.keys() if isinstance(signals, dict) else signals)

                                # ── REGIME-AWARE THRESHOLDS ──────────────────────────────────────
                                if score < global_min_score:
                                    with _batch_lock:
                                        rejection_counts["low_score"] += 1
                                        if (global_min_score - score) <= 5.0:
                                            waterfall_counts["near_misses"] += 1
                                        terminal_tracker.record_terminal(symbol, "LOW_SCORE", f"Score {score:.1f} < threshold {global_min_score}")
                                        telemetry_logger.record_reject(
                                            symbol, "SCORE", "LOW_SCORE",
                                            actual=score if "score" in locals() else 0,
                                            required=global_min_score if "global_min_score" in locals() else 0,
                                            start_time=_row_start_time,
                                            raw_market={
                                                "open_p": candle_open, "high_p": candle_high, "low_p": candle_low,
                                                "close_p": candle_close, "volume": _safe_float(latest.get("Volume")),
                                                "high_52w": _safe_float(latest.get("HIGH_52W"))
                                            },
                                            indicators={
                                                "rsi": rsi_val, "sma50": _safe_float(latest.get("SMA50")),
                                                "sma200": _safe_float(latest.get("SMA200")), "ema20": _safe_float(latest.get("EMA20")),
                                                "vol_ratio": volume_ratio, "atr": atr20, "adx": _safe_float(latest.get("ADX"))
                                            }
                                        )
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED — Score {score:.1f} < threshold {global_min_score}")
                                    try:
                                        from near_miss_tracker import log_near_miss
                                        entry_px = float(candle_close) if "candle_close" in locals() and candle_close else None
                                        sl_px = float(candle_low) if "candle_low" in locals() and candle_low else (entry_px * 0.95 if entry_px else None)
                                        tgt_px = float(prior_high) if "prior_high" in locals() and prior_high and prior_high > (entry_px or 0) else (entry_px * 1.10 if entry_px else None)
                                        log_near_miss(
                                            symbol=symbol,
                                            scanner="EOD",
                                            breakout_type=signal_str,
                                            gate_name="score_threshold",
                                            observed_value=float(score),
                                            threshold_value=float(global_min_score),
                                            score=int(score),
                                            entry_price=entry_px,
                                            stop_loss=sl_px,
                                            target_1=tgt_px
                                        )
                                    except Exception:
                                        pass
                                    return

                                logger.info(f"📍 PICKED [EOD: IN BETWEEN]: {symbol} @ ₹{candle_close:.2f} (Score: {score:.1f}, Prior High: ₹{prior_high:.2f})")

                                # [VERSION: EOD_DEDUP_FIX] Fixed dedup check to correctly match DB tuple schema (symbol, breakout_type)
                                ctx.capture_trace("STRATEGY_SELECTED", "PASS")
                                ctx.capture_trace("DUPLICATE_CHECK", "PENDING")
                                if (symbol, "EOD") in cooldown_alerts:
                                    logger.info(f"🚫 [EOD] {symbol} REJECTED after picking — Reason: COOLDOWN_ACTIVE (Already alerted in cooldown window)")
                                    ctx.capture_trace("DUPLICATE_CHECK", "FAIL")
                                    with _batch_lock:
                                        rejection_counts["duplicate"] += 1
                                        terminal_tracker.record_terminal(symbol, "COOLDOWN_ACTIVE", "Already alerted in cooldown window")
                                        # [RULE 67] Capture actual duplicate evidence for forensic audit.
                                        # User required: "capture actual evidence (e.g., existing_alert_id)
                                        # in addition to DUPLICATE=true". We include the cooldown window
                                        # boundary date so the audit can verify which prior alert caused this.
                                        _dup_window_minutes = ALERT_COOLDOWN_MINUTES.get("EOD", 1440)
                                        _dup_window_days = round(_dup_window_minutes / 1440, 1)
                                        from datetime import timedelta
                                        _dup_cutoff_date = (datetime.now(IST) - timedelta(minutes=_dup_window_minutes)).strftime("%Y-%m-%d %H:%M IST")
                                        telemetry_logger.record_reject(
                                            symbol, "SYSTEM", "DUPLICATE_REJECTED", True, False, start_time=_row_start_time, operator="!=", gate_type="BOOLEAN",
                                            indicators={
                                                "existing_alert": True,
                                                "existing_alert_date": _dup_cutoff_date,
                                                "duplicate_window_minutes": _dup_window_minutes,
                                                "duplicate_window_days": _dup_window_days
                                            }
                                        )
                                    return
                                ctx.capture_trace("DUPLICATE_CHECK", "PASS")

                                # ── Dynamic S/R and Indicator-based SL + Target (EOD mode) ───────
                                sl_result = compute_sl_and_target(
                                    entry_price=candle_close,
                                    atr=atr_val_eod,
                                    candle_range=candle_range,
                                    mode="EOD",
                                    adx=latest.get("ADX"),
                                    rsi=rsi_val,
                                    macd_hist=latest.get("MACD_HIST"),
                                    atr_pct=latest.get("ATR_PCT"),
                                    swing_low=latest.get("SWING_LOW"),
                                    swing_high=latest.get("SWING_HIGH"),
                                    bb_upper=latest.get("BB_UPPER"),
                                    bb_lower=latest.get("BB_LOWER"),
                                    bb_mid=latest.get("BB_MID"),
                                    s1=latest.get("S1"),
                                    s2=latest.get("S2"),
                                    r1=latest.get("R1"),
                                    r2=latest.get("R2"),
                                    swing_low_raw=latest.get("SWING_LOW_RAW"),
                                    swing_high_raw=latest.get("SWING_HIGH_RAW"),
                                    candle_low=candle_low,
                                    vwap=latest.get("VWAP"),
                                    ticker=ticker,
                                )

                                if sl_result.get("is_rejected"):
                                    logger.warning(f"🚫 [EOD] {symbol} REJECTED after picking — Reason: SL_RR_ENGINE_REJECT ({sl_result.get('rejection_reason')}, Natural RR={sl_result.get('natural_rr', 0):.2f})")
                                    with _batch_lock:
                                        rejection_counts["low_rr"] += 1
                                        terminal_tracker.record_terminal(symbol, "RISK_REJECTED", f"SL/RR reject: {sl_result.get('rejection_reason')}")
                                        telemetry_logger.record_reject(symbol, "RISK", "LOW_RR", sl_result.get("natural_rr", 0) if "sl_result" in locals() else 0, 2.0, start_time=_row_start_time)  # Reusing this counter for engine rejects
                                    from database import save_rejected_alert
                                    save_rejected_alert(
                                        symbol=symbol,
                                        scanner="EOD",
                                        rejection_reason=sl_result.get("rejection_reason", "V7 Engine Reject"),
                                        engine_version=sl_result.get("engine_version", "SL_ENGINE_V7.0"),
                                        context={"category": category, "score": score, "sl_result": sl_result}
                                    )
                                    return

                                suggested_stop = sl_result["stop_loss"]
                                target_price = sl_result["target_1"]

                                above_ema20  = bool(candle_close >= _safe_float(latest.get("EMA20"))) if "EMA20" in ticker.columns and not pd.isna(latest.get("EMA20")) else None
                                above_sma50  = bool(candle_close >= _safe_float(latest.get("SMA50"))) if "SMA50" in ticker.columns and not pd.isna(latest.get("SMA50")) else None
                                # [VERSION: EOD_PATCH_v1.0] [BUG FIX 6] Renamed golden_cross to above_golden_cross to accurately reflect it's a state check
                                above_golden_cross = bool(_safe_float(latest.get("SMA50")) >= _safe_float(latest.get("SMA200"))) if ("SMA50" in ticker.columns and "SMA200" in ticker.columns and not pd.isna(latest.get("SMA50")) and not pd.isna(latest.get("SMA200"))) else None

                                context = {
                                    "technicals": {
                                        "above_ema20":      above_ema20,
                                        "above_sma50":      above_sma50,
                                        "above_golden_cross":     above_golden_cross,
                                        "body_ratio":       round(body_ratio * 100, 2),
                                        "delivery_pct":     round(delivery_pct, 1) if delivery_pct is not None else None,
                                        "rsi":              round(rsi_val, 1),
                                        "volume_ratio":     round(volume_ratio, 2),
                                        "breakout_level":   round(_safe_float(latest.get("PRIOR_20D_HIGH")), 2) if "PRIOR_20D_HIGH" in latest else None,
                                        "atr20":            round(_safe_float(latest.get("ATR20")), 2) if "ATR20" in latest else None,
                                        "regime":           market_regime,
                                        "score":            score,
                                        # [FIX: TWO_MODE_52W] Breakout mode and recovery adjustment for forward-testing visibility:
                                        # "A" = High Breakout (within 5% of 52W high, standard path)
                                        # "B" = Recovery Breakout (5-15% below 52W high, stricter secondary checks)
                                        "breakout_mode":        _breakout_mode if "_breakout_mode" in locals() else "A",
                                        "recovery_adjustment":  recovery_adjustment if "recovery_adjustment" in locals() else 0,
                                    },
                                    "session": {
                                        "open":             round(candle_open, 2),
                                        "day_high":         round(candle_high, 2),
                                        "day_low":          round(candle_low, 2)
                                    },
                                    "fundamentals": {
                                        "peg":              row.get("PEG Ratio"),
                                        "yoy_rev":          row.get("YOY Revenue %"),
                                        "yoy_profit":       row.get("YOY Profit %"),
                                        "roe":              row.get("ROE %")
                                    },
                                    "execution": {
                                        "sl_method":        sl_result.get("sl_method"),
                                        "t_method":         sl_result.get("target_method")
                                    },
                                    "sl_result": sl_result
                                }

                                # Append configuration metadata for forward-testing and analytics
                                context["algo_version"] = ACTIVE_ALGO_VERSION
                                if delivery_found and delivery_days_back > 0:
                                    context["delivery_data_status"] = "missing_used_fallback"
                                elif not delivery_found:
                                    context["delivery_data_status"] = "unavailable"

                                context["algo_params"] = {
                                    **EOD_CONFIG,
                                    **EOD_ADVANCED_CONFIG,
                                    "MIN_BREAKOUT_MARGIN_1D": MIN_BREAKOUT_MARGIN.get("1d"),
                                    "MIN_BREAKOUT_VOLUME_RATIO": MIN_BREAKOUT_VOLUME_RATIO,
                                    "BASE_TIGHTNESS_THRESHOLD": BASE_TIGHTNESS_THRESHOLD
                                }

                                if not is_test_mode:
                                    _bayesian_regime = regime_ctx.get("trend", "BULL") if isinstance(regime_ctx, dict) else "BULL"
                                    _regime_score = float(regime_ctx.get("market_score", 80.0)) if isinstance(regime_ctx, dict) else 80.0

                                    safe_sec_str = "Unknown" if (sector is None or (isinstance(sector, float) and pd.isna(sector))) else str(sector).strip()
                                    sector_info = sector_rankings_dict.get(safe_sec_str, {})
                                    sector_name_val = sector_info.get("sector_name", sector or "")

                                    cand = {
                                        "symbol": symbol,
                                        "breakout_type": "EOD",
                                        "alert_time": ist_now.strftime("%Y-%m-%d %H:%M:%S+05:30"),
                                        "scanner": "EOD",
                                        "category": category,
                                        "entry_price": round(candle_close, 2),
                                        "signals": signal_str,
                                        "score": int(score),
                                        "rsi": round(rsi_val, 1),
                                        "volume_ratio": round(volume_ratio, 2),
                                        # [FIX: TWO_MODE_52W] breakout_mode and recovery_adjustment surfaced at top-level of alert
                                        "breakout_mode": _breakout_mode if "_breakout_mode" in locals() else "A",
                                        "recovery_adjustment": recovery_adjustment if "recovery_adjustment" in locals() else 0,
                                        "stop_loss": suggested_stop,
                                        "target_1": sl_result.get("target_1"),
                                        "target_2": sl_result.get("target_2"),
                                        "target_3": sl_result.get("target_3"),
                                        "target_price": target_price,
                                        "context": context,
                                        "model_version": model_version,
                                        "bayesian_regime": _bayesian_regime,
                                        "bayesian_weights": applied_bayesian_weights,
                                        "structural_failure_stop": sl_result.get("structural_failure_stop"),
                                        "target_quality_score": sl_result.get("target_quality"),
                                        "base_score": base_score_val,
                                        "rs_bonus": rs_bonus_val,
                                        "sector_bonus": sector_bonus_val,
                                        "rs_percentile": rs_pct_val,
                                        "sector_name": sector_name_val,
                                        "regime_score": _regime_score,
                                        # Extra data for logging and tracking
                                        "_candle_open": candle_open,
                                        "_candle_high": candle_high,
                                        "_candle_low": candle_low,
                                        "_body_ratio": body_ratio,
                                        "_close_position": close_position,
                                        "_above_ema20": above_ema20,
                                        "_above_sma50": above_sma50,
                                        "_above_golden_cross": above_golden_cross,
                                        "_sl_method": sl_result.get("sl_method"),
                                        "_target_method": sl_result.get("target_method"),
                                        "_natural_rr": sl_result.get("natural_rr"),
                                        "_delivery_pct": delivery_pct,
                                        "_peg": row.get("PEG Ratio"),
                                        "_yoy_rev": row.get("YOY Revenue %"),
                                        "_yoy_profit": row.get("YOY Profit %"),
                                        "_roe": row.get("ROE %"),
                                        "_ticker": ticker
                                    }
                                    with _batch_lock:
                                        approved_candidates.append(cand)
                                        telemetry_logger.record_pass(symbol, int(score), float(sl_result.get("natural_rr", 0)), {"VolumeRatio": round(volume_ratio, 2), "RSI": round(rsi_val, 1)}, _row_start_time)
                                else:
                                    pass

                            # [VERSION: EOD_PATCH_v1.0] [BUG FIX 4] Catch general Exceptions rather than specific errors to prevent ZeroDivisionError/AttributeError from crashing the entire scan loop
                            except Exception as e:
                                error_type = type(e).__name__
                                logger.exception(f"⚠️ Exception ({error_type}) processing {symbol}: {e}")
                                with _batch_lock:
                                    rejection_counts["indicator_fail"] = rejection_counts.get("indicator_fail", 0) + 1
                                    terminal_tracker.record_terminal(symbol, "PROCESSING_ERROR", f"{error_type}: {str(e)[:100]}")
                                if not is_test_mode:
                                    try:
                                        upsert_fetch_error('yfinance', 'EOD', symbol, '1d', f'processing_error_{error_type}', str(e)[:500])
                                    except Exception:
                                        logger.exception(f'Failed to upsert fetch error for {symbol}')
                                return



                        with ThreadPoolExecutor(max_workers=10, thread_name_prefix="EOD_Worker") as executor:
                            futures = [executor.submit(_process_row, idx, row) for idx, row in enumerate(chunk_df.itertuples(index=False), start=1)]
                            for f in as_completed(futures):
                                f.result() # Raise any exceptions caught in thread
                    if run_ctx and (time.monotonic() - _last_hb) >= 10.0:
                        try:
                            run_ctx.heartbeat()
                            _last_hb = time.monotonic()
                        except Exception: pass
                    logger.info(f"⏳ [EOD SCANNER] Evaluated Batch {batch_num}/{total_batches} ({min(batch_num * BATCH_SIZE, len(watchlist))}/{len(watchlist)} stocks) | Candidates found so far: {len(approved_candidates)}")

            # ── MAX ALERTS ENFORCEMENT & PERSISTENCE ──────────────────────────────────────────
            if approved_candidates:
                logger.info(f"📊 EOD Candidates Discovered: {len(approved_candidates)}")
                for cand in approved_candidates:
                    logger.info(f"  • 🟢 {cand['symbol']} @ ₹{cand['entry_price']:.2f} (Score: {cand['score']}, RSI: {cand['rsi']:.1f}, Vol Ratio: {cand['volume_ratio']:.2f}x)")
                approved_candidates.sort(key=lambda x: x["score"], reverse=True)
            else:
                logger.info("📊 EOD Candidates Discovered: 0")

            # [ADJUSTED TRADING SESSION NORMALIZATION]
            # Normalize source_trading_date so weekend runs (Saturday/Sunday) inherit the
            # Friday trading session, enabling clean deduplication across Friday -> Sat -> Sun.
            from market_utils import get_expected_latest_trading_date
            source_trading_date = get_expected_latest_trading_date(ist_now)

            if approved_candidates:
                from config import SCANNER_MAX_ALERTS
                max_alerts = SCANNER_MAX_ALERTS.get("EOD", 10)

                if len(approved_candidates) > max_alerts:
                    logger.info(f"Limiting EOD alerts from {len(approved_candidates)} to {max_alerts}")
                    rejected_cands = approved_candidates[max_alerts:]
                    approved_candidates = approved_candidates[:max_alerts]
                    from database import save_rejected_alert
                    for cand in rejected_cands:
                        rejection_counts["max_alerts_exceeded"] = rejection_counts.get("max_alerts_exceeded", 0) + 1
                        terminal_tracker.record_terminal(cand["symbol"], "SUPPRESSED_TOP_N", f"Score {cand['score']} exceeded top {max_alerts}")
                        logger.info(f"🚫 {cand['symbol']} alert SUPPRESSED: Exceeded MAX_ALERTS_PER_SCAN limit (Score: {cand['score']})")

                for cand in approved_candidates:
                    # Ensure post-market entry price matches today's live CMP
                    try:
                        from price_cache import get_cached_price
                        fast_p = get_cached_price(cand["symbol"])
                        if fast_p and float(fast_p) > 0:
                            cand["entry_price"] = round(float(fast_p), 2)
                    except Exception:
                        pass

                    c = dict(cand)
                    # Remove extra keys before saving
                    _candle_open = c.pop("_candle_open")
                    _candle_high = c.pop("_candle_high")
                    _candle_low = c.pop("_candle_low")
                    _body_ratio = c.pop("_body_ratio")
                    _close_position = c.pop("_close_position")
                    _above_ema20 = c.pop("_above_ema20")
                    _above_sma50 = c.pop("_above_sma50")
                    _above_golden_cross = c.pop("_above_golden_cross")
                    _sl_method = c.pop("_sl_method")
                    _target_method = c.pop("_target_method")
                    _natural_rr = c.pop("_natural_rr")
                    _delivery_pct = c.pop("_delivery_pct")
                    _peg = c.pop("_peg")
                    _yoy_rev = c.pop("_yoy_rev")
                    _yoy_profit = c.pop("_yoy_profit")
                    _roe = c.pop("_roe")
                    _ticker = c.pop("_ticker")

                    c["source_trading_date"] = source_trading_date
                    if not is_test_mode:
                        saved, reason, cap_alloc, shares = save_alert_if_new(**c)
                    else:
                        saved, reason, cap_alloc, shares = True, "", 0.0, 0

                    if not saved:
                        rejection_counts["duplicate"] += 1
                        terminal_tracker.record_terminal(c["symbol"], "DUPLICATE_ALERT", reason or "Already alerted within cooldown window")
                        try:
                            ctx_dup = telemetry_logger.get_or_create_context(c["symbol"])
                            ctx_dup.capture_trace("STRATEGY_SELECTED", "PASS")
                            ctx_dup.capture_trace("DUPLICATE_CHECK", "PENDING")
                            ctx_dup.capture_trace("DUPLICATE_CHECK", "FAIL")
                            ctx_dup.capture_gate(
                                gate_name="DUPLICATE_REJECTED", passed=False, actual_val=True, threshold_val=False,
                                operator_str="!=", gate_type="BOOLEAN", reason=reason or "Symbol already alerted within cooldown window"
                            )
                            ctx_dup.add_decision_input("existing_alert", True, "DatabaseGuard", "Live", "LIVE", required=True, valid=True)
                            ctx_dup.add_decision_input("duplicate_window_days", 1.0, "DatabaseGuard", "Live", "LIVE", required=True, valid=True)
                            ctx_dup.finalize(decision="REJECTED", primary_reason="DUPLICATE_REJECTED")
                            telemetry_engine.emit_terminal(ctx_dup)
                        except Exception:
                            telemetry_logger.record_reject(c["symbol"], "SYSTEM", "DUPLICATE_REJECTED", None, None)
                        continue

                    alerts_by_category.setdefault(c["category"], []).append({
                        "symbol":           c["symbol"],
                        "category":         c["category"],
                        "breakout_signals": [c["signals"]],
                        "price":            c["entry_price"],
                        "open":             round(_candle_open, 2),
                        "day_high":         round(_candle_high, 2),
                        "day_low":          round(_candle_low, 2),
                        "rsi":              c["rsi"],
                        "volume_ratio":     c["volume_ratio"],
                        "body_ratio":       round(_body_ratio * 100),
                        "close_position":   round(_close_position * 100),
                        "score":            c["score"],
                        "above_ema20":      _above_ema20,
                        "above_sma50":      _above_sma50,
                        "above_golden_cross":     _above_golden_cross,
                        "atr_stop":         c["stop_loss"],
                        "target_price":     c["target_price"],
                        "target_2":         c["target_2"],
                        "target_3":         c["target_3"],
                        "sl_method":        _sl_method,
                        "t_method":         _target_method,
                        "rr_ratio":         _natural_rr,
                        "delivery_pct":     round(_delivery_pct, 1) if _delivery_pct is not None else None,
                        "peg":              _peg,
                        "yoy_rev":          _yoy_rev,
                        "yoy_profit":       _yoy_profit,
                        "roe":              _roe,
                        "capital_allocated": cap_alloc,
                        "shares_bought":     shares
                    })
                    total_alerts += 1
                    terminal_tracker.record_terminal(c["symbol"], "ALERT_GENERATED", f"EOD alert saved (Score: {c['score']})")

                    _last_bar_date = "unknown"
                    try:
                        if isinstance(_ticker.index, pd.DatetimeIndex):
                            _last_bar_date = str(_ticker.index[-1])[:10]
                        elif "Date" in _ticker.columns:
                            _last_bar_date = str(_ticker["Date"].iloc[-1])[:10]
                    except Exception:
                        pass
                    logger.info(
                        f"🌟 [EOD: SELECTED] {c['symbol']} | "
                        f"score={c['score']} | vol_ratio={c['volume_ratio']:.2f} | rsi={c['rsi']:.1f} | "
                        f"entry=₹{c['entry_price']:.2f} | sl=₹{round(float(c['stop_loss'] or 0.0))} | t1=₹{round(float(c['target_price'] or 0.0))} | "
                        f"last_bar={_last_bar_date} | category={c['category']}"
                    )

            # Run single garbage collection pass after all batch evaluations complete
            try:
                gc.collect()
            except Exception:
                pass

            # ── VERIFICATION & STATUS ────────────────────────────────────────────────────
            stage_tracker.end_stage(f"Evaluated {len(watchlist)} stocks | Alerts found: {total_alerts}")
            stage_tracker.start_stage(4, "Pipeline Summary & Alert Persistence", f"Total alerts: {total_alerts}")
            fired = {k: v for k, v in rejection_counts.items() if v > 0}

            duration_sec = round((datetime.now(IST) - start_time).total_seconds(), 1)
            total_symbols = len(watchlist)
            stale_count = rejection_counts.get("stale_data", 0)
            no_data_count = rejection_counts.get("no_data", 0)
            fresh_count = max(0, total_fetched_count - stale_count)
            data_status = "OK"
            if used_fallback_data:
                # [RULE 67] DEGRADED_FALLBACK takes precedence over all other statuses.
                # User explicitly required this: "enforce DEGRADED_FALLBACK at the final health write level."
                # Previously only the upsert block at line ~1838 set DEGRADED_FALLBACK, but the
                # data_status shown in the pipeline SUMMARY still said "OK". Now both agree.
                data_status = "DEGRADED_FALLBACK (Historical Fallback Dataset)"
            elif (stale_count / max(total_symbols, 1)) > 0.30:
                data_status = "DEGRADED (Stale Data > 30%)"

            if run_ctx:
                run_ctx.set_total_stocks(total_symbols)
                run_ctx.fresh_count = fresh_count
                run_ctx.stale_count = stale_count
                run_ctx.incomplete_count = no_data_count

            # Ensure 100% mathematical conservation
            terminal_tracker.record_untracked_remainder("UNTRACKED_DROP")
            cons_summary = terminal_tracker.get_summary()

            # Record final stage into waterfall
            waterfall.set_stage_count("UNIVERSE_WATCHLIST", total_symbols)
            waterfall.set_stage_count("FETCHED_DATA", fresh_count)
            waterfall.set_stage_count("BREAKOUT_STRUCTURE", waterfall_counts["structure_entered"])
            waterfall.set_stage_count("QUALITY_AND_RISK", waterfall_counts["quality_entered"])
            waterfall.set_stage_count("FINAL_ALERTS", total_alerts)

            attrition_results = waterfall.compute_attrition()
            dominant_bottleneck = waterfall.get_dominant_bottleneck()

            classification_res = classify_zero_alert_run(
                scanner_name="EOD",
                universe_size=total_symbols,
                valid_data_count=fresh_count,
                initial_setups_count=waterfall_counts["structure_entered"],
                finalist_candidates_count=len(approved_candidates),
                alerts_generated=total_alerts,
                near_miss_count=waterfall_counts["near_misses"],
                regime=market_regime,
                execution_mode="EOD_SCAN",
                stage_waterfall=attrition_results
            )

            summary_lines = [
                "======================================================================",
                "=== [EOD SCANNER PIPELINE SUMMARY] ===",
                "======================================================================",
                "📊 DATA QUALITY SNAPSHOT:",
                f"  • Total Watchlist Requested : {total_symbols}",
                f"  • Fresh Data OK             : {fresh_count}",
                f"  • Stale Data                : {stale_count}",
                f"  • Missing / No Data         : {no_data_count}",
                f"  • Data Health Status        : {data_status}",
                "",
                "🎯 CRITERIA & FILTER BREAKDOWN:"
            ]
            for k, v in fired.items():
                summary_lines.append(f"  • {k:<27}: {v}")

            summary_lines.extend([
                "",
                "📐 CONSERVATION ACCOUNTING (Single Terminal Disposition):",
                f"  • Universe Requested        : {cons_summary['total_universe']}",
                f"  • Sum of Terminal Outcomes  : {cons_summary['sum_terminal']}",
                f"  • Discrepancy (Delta)       : {cons_summary['conservation_delta']} (Conservation: {'VALID' if cons_summary['is_conserved'] else 'VIOLATED'})",
                "",
                "  Terminal Outcome Counts:"
            ])
            for disp, cnt in cons_summary["terminal_counts"].items():
                summary_lines.append(f"    - {disp:<26}: {cnt}")

            summary_lines.extend([
                "",
                "📉 STAGE WATERFALL PROGRESSION & ATTRITION RATES:"
            ])
            for stg in attrition_results:
                summary_lines.append(
                    f"  • {stg['stage']:<22}: {stg['entered']:>4} entered → {stg['passed']:>4} passed (Loss: {stg['eliminated']:>4}, Attrition: {stg['attrition_pct']:>5.1f}%)"
                )
            if dominant_bottleneck:
                b_stg = dominant_bottleneck.get('stage', 'UNKNOWN')
                b_elim = dominant_bottleneck.get('eliminated', 0)
                b_ent = dominant_bottleneck.get('entered', 0)
                b_pct = dominant_bottleneck.get('attrition_pct', 0.0)
                summary_lines.append(f"  • Dominant Bottleneck Gate  : {b_stg} ({b_elim}/{b_ent} eliminated, Attrition: {b_pct:.1f}%)")

            # Compact calibration monitoring summary (EOD_ATR_MODEL_A_V1)
            import numpy as _np
            base_atrs = waterfall_counts.get("base_atr_list", [])
            mean_base_atr = float(_np.mean(base_atrs)) if base_atrs else 0.0
            median_base_atr = float(_np.median(base_atrs)) if base_atrs else 0.0

            summary_lines.extend([
                "",
                "🔬 EOD_ATR_MODEL_A_V1 CALIBRATION MONITORING:",
                "---------------------------------------------",
                f"  • Universe Evaluated        : {total_symbols}",
                f"  • ATR <= 2.50% (0 pen)      : {waterfall_counts.get('atr_le_250', 0)}",
                f"  • ATR 2.51–3.50% (-3 pen)   : {waterfall_counts.get('atr_251_350', 0)}",
                f"  • ATR 3.51–4.50% (-7 pen)   : {waterfall_counts.get('atr_351_450', 0)}",
                f"  • ATR > 4.50% rejected      : {waterfall_counts.get('atr_gt_450_rejected', 0)}",
                "",
                f"  • Recovered from old cliff  : {waterfall_counts.get('recovered_from_cliff', 0)}",
                f"  • Mean base ATR             : {mean_base_atr:.2f}%",
                f"  • Median base ATR           : {median_base_atr:.2f}%",
                "",
                "🏆 FINAL OUTCOME:",
                f"  • Alerts Generated          : {total_alerts}",
                f"  • Near Misses (<=5 pts)     : {waterfall_counts['near_misses']}",
                f"  • Total Execution Time      : {duration_sec}s",
            ])

            if total_alerts == 0:
                b_stg = dominant_bottleneck.get('stage', '') if dominant_bottleneck else ''
                b_breakdown = terminal_tracker.get_stage_terminal_breakdown(b_stg) if b_stg else None

                diag_block = format_zero_alert_diagnostic_block(
                    scanner_name="EOD",
                    execution_mode="EOD_SCAN",
                    regime=market_regime,
                    classification_result=classification_res,
                    dominant_bottleneck=dominant_bottleneck,
                    conservation_summary=cons_summary,
                    stage_waterfall=attrition_results,
                    near_miss_count=waterfall_counts["near_misses"],
                    extra_specs=[
                        f"BASE_SCORE_THRESHOLD       : {BASE_SCORE_THRESHOLD}",
                        f"REGIME_STRICTNESS_PENALTY  : {regime_modifier:+d} ({market_regime} makes bar higher/stricter)",
                        f"EFFECTIVE_GLOBAL_MIN_SCORE : {effective_global_min_score} (stricter qualification floor, capped <= 82)",
                    ],
                    bottleneck_terminal_breakdown=b_breakdown
                )
                summary_lines.extend(diag_block)

            summary_lines.append("======================================================================")
            logger.info("\n".join(summary_lines))
            try:
                stage_tracker.end_stage(f"Alerts generated: {total_alerts}")
                stage_tracker.print_summary(alerts_found=total_alerts)
            except Exception:
                pass

            # ── EOD Alert Report ─────────────────────────────────────────────────
            # [RULE 26 / ARCH §26] Print rich per-alert cards once per scan cycle
            # so every run has a clear, diagnostic human-readable record of what
            # fired and why.
            try:
                from eod_alert_builder import build_eod_scan_report
                _eod_report = build_eod_scan_report(alerts_by_category, market_regime, ist_now)
                logger.info(_eod_report)
            except Exception as _rpt_err:
                logger.warning(f"⚠️ EOD alert report generation failed (non-fatal): {_rpt_err}")

            # ✅ CRITICAL: Verify alerts were actually saved to database (2026-06-17)
            if total_alerts > 0 and not is_test_mode:
                if not verify_alerts_saved_today("EOD", total_alerts):
                    logger.critical(f"🚨 CRITICAL ERROR: EOD generated {total_alerts} alerts but save failed!")
                    upsert_scanner_health(
                        scanner_name="EOD",
                        status="DOWN",
                        error_msg=f"CRITICAL: {total_alerts} alerts failed to save to database"
                    )
                    raise RuntimeError("Alert save verification failed - database connectivity issue")

            status = "OK"
            error_msg = None

            # [VERSION: EOD_STALE_DEGRADE_FIX] Mark degraded if >30% stale
            # [AUDIT-E1 FIX] stale_count and total_symbols already set at summary block above — removed duplicate assignments
            if total_symbols > 0 and (stale_count / total_symbols) > 0.30:
                status = "DEGRADED"
                error_msg = f"High stale data: {stale_count}/{total_symbols} symbols rejected (likely due to fallback watchlist)"

            # [VERSION: EOD_PATCH_v1.3] Log active thread count to monitor potential ThreadPoolExecutor leaks
            active_threads = threading.active_count()
            logger.info(f"🧵 Final Active Thread Count: {active_threads}")

            if total_fetched_count < len(watchlist) * 0.95:
                status = "DEGRADED"
                error_msg = f"Partial Fetch: {total_fetched_count}/{len(watchlist)} symbols"

            duration_sec = (datetime.now(IST) - start_time).total_seconds()

            # [RULE 67 CHANGE-RATIONALE]: Retain price data reference for Phase 2B EOD V2 pipeline before deleting all_ticker_data.
            price_data_map = all_ticker_data if 'all_ticker_data' in locals() and all_ticker_data else {}
            del all_ticker_data
            locals().pop('ticker', None)

            # Check if we fetched enough data overall
            if total_fetched_count < len(watchlist) * 0.70:
                logger.warning(f"⚠️ EOD data fetch returned {total_fetched_count}/{len(watchlist)} symbols (70% minimum required). EOD results may be incomplete.")
            else:
                logger.info(f"✅ Successfully fetched {total_fetched_count} symbols for EOD phase")
        # Insert scan failures via batch
        if scan_failures and not is_test_mode:
            try:
                from database import get_connection
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        from psycopg2.extras import execute_values
                        execute_values(
                            cur,
                            """
                            INSERT INTO scan_failures (symbol, scanner_name, provider, failure_reason, failed_at, scan_id)
                            VALUES %s
                            """,
                            [(f.symbol, f.scanner_name, f.provider, f.failure_reason, f.failed_at, f.scan_id) for f in scan_failures]
                        )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to record {len(scan_failures)} scan failures: {e}")

        # Map overall outcome & status guard — Missing/unfetched data is a CRITICAL BLOCKER
        outcome = "SUCCESS"
        no_data_count = rejection_counts.get("no_data", 0)

        if no_data_count >= len(watchlist) * 0.25:
            status = "DOWN"
            outcome = "FAILED"
            error_msg = f"🚫 CRITICAL BLOCKER: {no_data_count}/{len(watchlist)} symbols unfetched (missing data)"
            logger.error(f"🚨 {error_msg}")
            try:
                from telegram_engine import send_telegram_message
                send_telegram_message(f"🚨 <b>CRITICAL BLOCKER: EOD SCANNER FAILED</b>\n{no_data_count}/{len(watchlist)} symbols were unfetched / missing data.")
            except Exception:
                pass

        elif total_fetched_count == 0:
            outcome = "FAILED"
            status = "DOWN"
            error_msg = f"🚫 CRITICAL BLOCKER: 0/{len(watchlist)} symbols fetched (missing data)"
        elif total_fetched_count < len(watchlist) * 0.70:
            outcome = "PARTIAL"
            status = "DEGRADED"
        elif used_fallback_data:
            status = "DEGRADED_FALLBACK"

        if not is_test_mode:
            try:
                provider_stats_counts["STALE"] = rejection_counts.get("stale_data", 0)
                upsert_scanner_health(
                    scanner_name="EOD",
                    status=status,
                    last_success=datetime.now(IST).isoformat(),
                    today_alerts=total_alerts,
                    processed_count=total_alerts,
                    total_count=len(watchlist),
                    error_msg=error_msg,
                    outcome=outcome,
                    provider_stats=provider_stats_counts,
                    duration_seconds=duration_sec
                )
            except Exception:
                logger.exception("❌ Failed to update scanner health for EOD")
            if status == "OK":
                pass
            elif status == "DEGRADED":
                try:
                    insert_notification("admin", f"⚠️ EOD Scanner finished with DEGRADED status", error_msg or f"Generated {total_alerts} alerts but data was degraded.")
                    from push_service import send_push_to_all
                    send_push_to_all("⚠️ EOD Scanner DEGRADED", error_msg or "Stale data exceeded limit.")
                except Exception:
                    pass

        # ── Phase 2B: Parallel EOD V2 Pipeline Execution ([INV-1] Isolated) ───────
        try:
            logger.info("🚀 [EOD_V2_PIPELINE] Starting parallel Phase 2B EOD V2 pipeline...")
            from eod_v2_schema import init_eod_v2_schemas
            from eod_v2_engine import process_eod_v2_pipeline
            init_eod_v2_schemas()

            v2_elite_df = None
            v2_nq_df = None
            if os.path.exists("data/elite_universe_v2.parquet"):
                v2_elite_df = pd.read_parquet("data/elite_universe_v2.parquet")
            if os.path.exists("data/near_qualified_v2.parquet"):
                v2_nq_df = pd.read_parquet("data/near_qualified_v2.parquet")

            v2_res = process_eod_v2_pipeline(
                elite_df=v2_elite_df,
                nq_df=v2_nq_df,
                price_data_map=price_data_map
            )
            logger.info(
                f"✅ [EOD_V2_PIPELINE] Complete | WATCH={len(v2_res.get('watch', []))} "
                f"CONFIRMED={len(v2_res.get('confirmed', []))} MISSED={len(v2_res.get('missed', []))} "
                f"NQ_OBS={len(v2_res.get('nq_obs', []))}"
            )
        except Exception as _eod_v2_err:
            logger.warning(f"⚠️ [EOD_V2_PIPELINE] EOD V2 execution failed (non-fatal): {_eod_v2_err}")

        try:
            from funnel_telemetry import log_funnel_metrics
            log_funnel_metrics("EOD", market_regime, len(watchlist), rejection_counts, total_alerts)
        except Exception as e:
            logger.warning(f"Failed to log funnel telemetry: {e}")

        elapsed_time = (datetime.now(IST) - start_time).total_seconds()
        logger.info("\n" + "=" * 80)
        logger.info(f"🛑🛑🛑 [COMPLETE] EOD SCANNER DONE | {elapsed_time:.2f}s | Alerts={total_alerts} | Status={status} 🛑🛑🛑")
        logger.info(f"📊 Provider Stats: {dict(provider_stats_counts)}")
        logger.info(f"📊 Final Rejections: {dict(rejection_counts)}")
        print("\n" + "="*40)
        print(" EOD SCANNER REJECTION TELEMETRY")
        print("="*40)
        for reason, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f" {reason.ljust(30)} : {count}")
        print("="*40 + "\n")
        logger.info("=" * 80 + "\n")

        if not is_test_mode:
            try:
                from database import upload_history_bundle_to_db, submit_background_upload
                submit_background_upload(lambda: upload_history_bundle_to_db("1d"))
                logger.info("💾 [EOD] Submitted background upload of 1d history bundle to Postgres DB.")
            except Exception as _up_err:
                logger.warning(f"⚠️ Failed to queue background DB bundle upload in EOD: {_up_err}")

        return total_alerts


    except Exception as e:
        logger.exception("❌ CRITICAL EOD SCAN ERROR")
        if not is_test_mode:
            try:
                upsert_scanner_health(scanner_name="EOD", status="DOWN", error_msg=str(e))
                insert_notification("admin", f"❌ EOD Scanner CRASHED (DOWN)", f"Error: {str(e)[:200]}")
                from push_service import send_push_to_all
                send_push_to_all("❌ EOD Scanner DOWN", f"Crash: {str(e)[:100]}")
            except Exception:
                pass
        raise  # re-raise so caller can send Telegram crash alert
