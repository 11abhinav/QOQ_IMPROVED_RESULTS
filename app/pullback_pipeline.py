from scanner_telemetry import ScannerDecisionLogger, global_telemetry
import time as _time
# =====================================================================================
# app/pullback_pipeline.py
# PULLBACK CONTINUATION SCANNER PIPELINE
#
# RULE 67 MANDATORY CHANGE-RATIONALE:
# - Verified and enhanced GlobalScannerTelemetryEngine logging across Pullback pipeline stages.
# - Rationale: Emits explicit telemetry records tracking swing pivot detection, impulse leg gain %,
#   retracement depth %, volume contraction ratio, and resumption trigger candle verification to ensure
#   zero diagnostic blind spots.
# =====================================================================================
import os
import time
import json
import math
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Set
import pandas as pd

from core_enums import CandidateState, RejectionReason
from core_models import PullbackCandidate, DataQualityError
from config import PULLBACK_CONFIG, PULLBACK_CONFIG as config, REGIME_POLICIES
import swing_utils
from sl_target_helper import compute_sl_and_target
from database import (
    init_db, save_alert_if_new, upsert_scanner_health, insert_notification,
    get_recent_alerts_for_scanner, save_funnel_telemetry
)
from memory_profiler import MemoryProfiler, BatchMemoryTracker, chunk_iterable
from watchlist_cache import get_watchlist
from price_cache import fetch_watchlist_data
from macro_utils import MarketRegimeEngine, get_nifty_20d_return, get_macro_regime
from lock_utils import ProcessLock
from zero_alert_diagnostic import (
    SingleTerminalTracker,
    StageWaterfallTracker,
    classify_zero_alert_run,
    format_zero_alert_diagnostic_block
)

logger = logging.getLogger("pullback_scanner")

def _safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return float(val)
    except Exception:
        return default

IST = ZoneInfo("Asia/Kolkata")
_scan_lock = ProcessLock("pullback_scanner")
_global_lock = ProcessLock("global_scanner_lock")

def compute_pullback_score(
    pullback_count_in_trend: int,
    volume_ratio: float,
    trigger_close_position: float,
    trigger_volume_mult: float,
    rs_percentile: float,
    sector_status: str,
    has_prior_eod: bool,
    has_prior_multi: bool,
    is_full_high_takeover: bool = False,
    is_bullish_engulfing: bool = False,
    depth_pct: float = 30.0,
    impulse_pct: float = 10.0,
    max_bonus: float = 5.0
) -> dict:
    """
    Computes pullback base_score and final_score additively.
    Returns a dictionary of the score breakdown.
    """
    base_score = 70.0
    
    # 1. Relative Strength (RS) Bonus (Graduated, Up to +5)
    if rs_percentile >= 90.0:
        rs_bonus = 5.0
    elif rs_percentile >= 80.0:
        rs_bonus = 4.0
    elif rs_percentile >= 70.0:
        rs_bonus = 2.0
    else:
        rs_bonus = 0.0

    # 2. Sector Tailwind Bonus (Up to +3)
    if sector_status == "TAILWIND":
        sector_bonus = 3.0
    elif sector_status == "MILD_TAILWIND":
        sector_bonus = 1.0
    else:
        sector_bonus = 0.0

    # 3. Volume Contraction Bonus (Up to +4)
    if volume_ratio <= 0.50:
        vol_bonus = 4.0
    elif volume_ratio <= 0.70:
        vol_bonus = 2.0
    else:
        vol_bonus = 0.0

    # 4. Trigger Strength & Pattern Bonus (Graduated, Up to +6)
    trigger_bonus = 0.0
    if is_full_high_takeover:
        trigger_bonus += 2.0  # Exceptional reclaim over entire upper wick
    if is_bullish_engulfing:
        trigger_bonus += 2.0  # Bullish engulfing structure
    if trigger_close_position >= 0.85 and trigger_volume_mult >= 1.50:
        trigger_bonus += 3.0
    elif trigger_close_position >= 0.75 and trigger_volume_mult >= 1.30:
        trigger_bonus += 2.0
    elif trigger_close_position >= 0.80:
        trigger_bonus += 1.0

    # 5. Trend maturity penalty
    maturity_penalties = {0: 0, 1: 0, 2: -3, 3: -6}
    penalty = maturity_penalties.get(pullback_count_in_trend, -10)
    
    
    # 6. Flag Depth Classification Bonus (High Tight Flags get bonus)
    depth_bonus = 0.0
    if 10.0 <= depth_pct < 23.6:
        depth_bonus = 5.0  # High Tight Flag / Shallow Flag
    elif 23.6 <= depth_pct <= 38.2:
        depth_bonus = 2.0  # Classic Pullback
    else:
        depth_bonus = 0.0  # Deep Pullback (no bonus, relies on trigger strength)
        
    # 7. Impulse Strength Bonus
    impulse_bonus = 0.0
    if impulse_pct >= 20.0:
        impulse_bonus = 5.0
    elif impulse_pct >= 12.0:
        impulse_bonus = 3.0
    elif impulse_pct >= 8.0:
        impulse_bonus = 1.0

    eod_bonus = 3.0 if has_prior_eod else 0.0
    final_score = base_score + rs_bonus + sector_bonus + vol_bonus + trigger_bonus + maturity_penalties.get(pullback_count_in_trend, -10) + depth_bonus + impulse_bonus + eod_bonus
    final_score = min(100.0, max(0.0, final_score))
    
    return {
        "base_score": base_score,
        "rs_bonus": rs_bonus,
        "sector_bonus": sector_bonus,
        "vol_bonus": vol_bonus,
        "trigger_bonus": trigger_bonus,
        "depth_bonus": depth_bonus,
        "impulse_bonus": impulse_bonus,
        "maturity_penalty": maturity_penalties.get(pullback_count_in_trend, -10),
        "catalyst_bonus": eod_bonus,
        "final_score": final_score
    }

def evaluate_pullback_symbol(symbol: str, df: pd.DataFrame, fund_data: dict = None, regime_ctx: dict = None) -> dict:
    """
    Evaluates a single symbol against the production Pullback Continuation scanner rules.
    Runs trend alignment (Close > SMA50 > SMA200), swing pivot detection, impulse wave selection (gain >= 8%), retracement depth (23.6%-61.8%), resumption trigger candle, scoring, and target calculations without side effects.
    """
    # [VERSION: IPO_SHORT_HISTORY_QUALIFICATION_v1.0]
    # RULE 90 MANDATORY RATIONALE:
    # - Data fetching ALWAYS requests full 1-year history (period="1y", interval="1d").
    # - For established stocks, this yields ~250 trading candles.
    # - For newly listed IPO stocks (e.g., listed 30 days ago), full history naturally yields all available candles (30 bars).
    # - Minimum qualification threshold lowered from hardcoded 200 bars down to 15 bars so newly listed IPO stocks
    #   can qualify for setup evaluation (SMA50/indicators adaptively calculate on available bars) rather than being discarded.
    if df is None or df.empty or len(df) < 15:
        return {
            "status": "NO",
            "reasons": [f"Insufficient historical price data ({len(df) if df is not None else 0} bars < 15 minimum)"],
            "score": 0.0,
            "qualified": False
        }

    historical_view = df.copy()
    historical_view.attrs['adjusted'] = True
    if isinstance(historical_view.columns, pd.MultiIndex):
        historical_view.columns = historical_view.columns.get_level_values(0)
    historical_view = historical_view.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if len(historical_view) < 15:
        return {"status": "NO", "reasons": [f"Insufficient valid bars ({len(historical_view)} < 15)"], "score": 0.0, "qualified": False}

    from indicator_manager import manager
    try:
        bundle = manager.compute_base_indicators(historical_view, symbol)
    except Exception as _ie:
        return {"status": "NO", "reasons": [f"Failed to compute technical indicators: {_ie}"], "score": 0.0, "qualified": False}

    last_bar = historical_view.iloc[-1]
    close_price = float(last_bar['Close'])
    n_bars = len(historical_view)

    # Adaptive History Classification & Trend Validation Mode
    if n_bars >= 200:
        history_class = "MATURE"
        trend_validation_mode = "SMA200"
    elif n_bars >= 50:
        history_class = "RECENT_LISTING"
        trend_validation_mode = "SMA50_EMA20"
    else:
        history_class = "FRESH_IPO"
        trend_validation_mode = "EMA20"

    sma50_val = float(bundle.sma_50.iloc[-1]) if hasattr(bundle, 'sma_50') and bundle.sma_50 is not None and not bundle.sma_50.empty and not pd.isna(bundle.sma_50.iloc[-1]) else None
    sma200_val = float(bundle.sma_200.iloc[-1]) if hasattr(bundle, 'sma_200') and bundle.sma_200 is not None and not bundle.sma_200.empty and not pd.isna(bundle.sma_200.iloc[-1]) else None
    ema20_val = float(bundle.ema_20.iloc[-1]) if hasattr(bundle, 'ema_20') and bundle.ema_20 is not None and not bundle.ema_20.empty and not pd.isna(bundle.ema_20.iloc[-1]) else None

    # History-Tiered Uptrend Validation
    if history_class == "MATURE":
        uptrend_ok = bool(sma50_val and sma200_val and sma50_val > sma200_val and close_price >= (sma50_val * 0.95))
        trend_fail_reason = f"Trend Failure (MATURE): Close ₹{close_price:.2f} is not aligned above SMA50 ₹{sma50_val if sma50_val else 0:.2f} > SMA200 ₹{sma200_val if sma200_val else 0:.2f}"
    elif history_class == "RECENT_LISTING":
        uptrend_ok = bool(sma50_val and close_price >= (sma50_val * 0.95) and (ema20_val >= (sma50_val * 0.98) if ema20_val else True))
        trend_fail_reason = f"Trend Failure (RECENT_LISTING): Close ₹{close_price:.2f} is not aligned with SMA50 ₹{sma50_val if sma50_val else 0:.2f} / EMA20 ₹{ema20_val if ema20_val else 0:.2f}"
    else:  # FRESH_IPO
        uptrend_ok = bool(ema20_val and close_price >= (ema20_val * 0.95))
        trend_fail_reason = f"Trend Failure (FRESH_IPO): Close ₹{close_price:.2f} is not aligned above EMA20 ₹{ema20_val if ema20_val else 0:.2f}"

    if not uptrend_ok:
        return {
            "status": "NO",
            "reasons": [trend_fail_reason],
            "score": 0.0,
            "qualified": False,
            "entry_price": close_price,
            "history_class": history_class,
            "trend_validation_mode": trend_validation_mode,
        }

    pivots = swing_utils.detect_confirmed_pivots(historical_view, PULLBACK_CONFIG["LOOKBACK"], PULLBACK_CONFIG["CONFIRM"])
    if not pivots:
        return {"status": "NO", "reasons": ["No confirmed swing high/low pivots found for pullback calculation"], "score": 0.0, "qualified": False, "entry_price": close_price, "history_class": history_class, "trend_validation_mode": trend_validation_mode}

    impulse = swing_utils.select_pullback_origin(pivots, historical_view, PULLBACK_CONFIG)
    if not impulse:
        return {"status": "NO", "reasons": [f"No valid impulse origin leg identified (requires impulse gain ≥{PULLBACK_CONFIG['MIN_IMPULSE_GAIN_PCT']:.1f}%)"], "score": 0.0, "qualified": False, "entry_price": close_price, "history_class": history_class, "trend_validation_mode": trend_validation_mode}

    ps = swing_utils.measure_pullback(historical_view, impulse, PULLBACK_CONFIG)
    if not ps.valid:
        return {"status": "NO", "reasons": [f"Invalid pullback structure (Retracement {ps.depth_pct:.1f}%, Vol Ratio {ps.volume_ratio:.2f}x outside {PULLBACK_CONFIG['MIN_DEPTH_PCT']}%–{PULLBACK_CONFIG['MAX_DEPTH_PCT']}% bounds)"], "score": 0.0, "qualified": False, "entry_price": close_price, "history_class": history_class, "trend_validation_mode": trend_validation_mode}

    trig = swing_utils.detect_resumption_trigger(historical_view, ps, PULLBACK_CONFIG)
    if not trig.valid:
        return {
            "status": "WATCHLIST",
            "reasons": [f"Valid Pullback Structure (Depth {ps.depth_pct:.1f}%) — Awaiting Bullish Resumption Trigger Candle"],
            "score": 65.0,
            "qualified": False,
            "entry_price": close_price,
            "history_class": history_class,
            "trend_validation_mode": trend_validation_mode
        }

    entry_val = float(trig.entry_price)
    
    rs_percentile = 50.0
    if fund_data and isinstance(fund_data, dict):
        rs_percentile = fund_data.get("rs_rating") or fund_data.get("rs_percentile") or 50.0
    try:
        rs_percentile = float(rs_percentile)
    except (ValueError, TypeError):
        rs_percentile = 50.0

    sector_status = "NEUTRAL"
    if fund_data and isinstance(fund_data, dict):
        sector_status = fund_data.get("sector_status", "NEUTRAL")

    vol_ratio = float(ps.volume_ratio) if hasattr(ps, 'volume_ratio') and ps.volume_ratio is not None else 1.0

    # Calculate trigger close position from trig
    close_position = getattr(trig, "close_position", 0.5)

    has_prior_eod = False
    has_prior_multi = False
    if fund_data and isinstance(fund_data, dict):
        has_prior_eod = bool(fund_data.get("prior_eod_alert") or fund_data.get("has_prior_eod"))
        has_prior_multi = bool(fund_data.get("prior_multi_alert") or fund_data.get("has_prior_multi"))

    score_breakdown = compute_pullback_score(
        pullback_count_in_trend=ps.pullback_count_in_trend,
        volume_ratio=vol_ratio,
        trigger_close_position=close_position,
        trigger_volume_mult=trig.volume_mult,
        rs_percentile=rs_percentile,
        sector_status=sector_status,
        has_prior_eod=has_prior_eod,
        has_prior_multi=has_prior_multi,
        is_full_high_takeover=getattr(trig, "is_full_high_takeover", False),
        is_bullish_engulfing=getattr(trig, "is_bullish_engulfing", False),
        depth_pct=ps.depth_pct,
        impulse_pct=ps.impulse.gain_pct
    )
    final_score = score_breakdown["final_score"]

    market_regime = "NEUTRAL"
    if regime_ctx and isinstance(regime_ctx, dict):
        market_regime = regime_ctx.get("regime_type") or regime_ctx.get("regime") or "NEUTRAL"

    regime_thresholds = {
        "STRONG_BULL": 74.0,
        "BULL": 74.0,
        "NEUTRAL": 76.0,
        "WEAK_BEAR": 80.0,
        "BEAR": 80.0,
    }
    required_threshold = regime_thresholds.get(market_regime, 76.0)

    # 1. Run risk validation first
    sl_result = compute_sl_and_target(
        entry_price=entry_val,
        atr=float(bundle.atr_14.iloc[-1]) if hasattr(bundle, 'atr_14') and bundle.atr_14 is not None and not bundle.atr_14.empty and not pd.isna(bundle.atr_14.iloc[-1]) else (entry_val * 0.025),
        mode="PULLBACK",
        swing_low=ps.pullback_low.price if hasattr(ps, 'pullback_low') and ps.pullback_low else None,
        swing_high=ps.impulse.end.price if hasattr(ps, 'impulse') and ps.impulse else None
    )

    # 2. Qualification requires both threshold score and valid risk engine output
    is_qualified = (final_score >= required_threshold and not sl_result.get("is_rejected", False))

    status_str = "CORE MET" if is_qualified else "NO"
    
    if is_qualified:
        reasons = [f"Resumption Trigger Confirmed @ ₹{entry_val:.2f} (Depth {ps.depth_pct:.1f}%, Vol {ps.volume_ratio:.2f}x) | Pullback Score {final_score:.1f}/100"]
    else:
        reasons = []
        if final_score < required_threshold:
            reasons.append(f"Pullback Score {final_score:.1f} < {required_threshold} threshold")
        if sl_result.get("is_rejected"):
            reasons.append(f"Risk Rejected: {sl_result.get('rejection_reason')}")

    # ── PER-STOCK TERMINAL TELEMETRY DUMP (Section 4 & 8) ──
    try:
        from scanner_telemetry import DecisionContext, telemetry_engine
        ctx = DecisionContext(symbol=symbol, scanner_name="PULLBACK")
        ctx.capture_raw_market(
            open_p=_safe_float(last_bar.get("Open")),
            high_p=_safe_float(last_bar.get("High")),
            low_p=_safe_float(last_bar.get("Low")),
            close_p=close_price,
            volume=_safe_float(last_bar.get("Volume"))
        )
        ctx.capture_indicators(
            sma50=sma50_val,
            sma200=sma200_val,
            vol_ratio=vol_ratio,
            atr=sl_result.get("atr_14", close_price * 0.025)
        )
        ctx.capture_score("TOTAL", final_score, 100.0)
        ctx.capture_sl_target(entry_val, sl_result.get("stop_loss", 0.0), sl_result.get("target_1", 0.0))
        
        consumed_fields = {
            "RS Percentile": rs_percentile,
            "Sector Status": sector_status,
            "Volume Ratio": vol_ratio,
            "Trigger Close Position": close_position,
            "Has Prior EOD": has_prior_eod,
            "Has Prior Multi": has_prior_multi,
            "Retracement Depth %": ps.depth_pct,
            "Impulse Gain %": ps.impulse.gain_pct if ps.impulse else 0.0,
            "SMA50": sma50_val,
            "SMA200": sma200_val,
            "EMA20": ema20_val,
            "Close": close_price,
            "History Class": history_class,
            "Trend Validation Mode": trend_validation_mode
        }
        for k, v in consumed_fields.items():
            if v is not None:
                ctx.add_decision_input(name=k, value=v, source="PullbackEngine", as_of="Live", freshness="LIVE", required=True, valid=True)

        # [RULE 67 CHANGE-RATIONALE]: Set alert_generated matching is_qualified so terminal audit displays 'Alert Generated = YES' for qualified candidates
        ctx.alert_generated = bool(is_qualified)
        ctx.finalize(decision="SELECTED" if is_qualified else "REJECTED", primary_reason=reasons[0] if reasons else "NO_QUALIFY")
        telemetry_engine.emit_terminal(ctx)
    except Exception as telemetry_err:
        logger.debug(f"Telemetry recording skipped: {telemetry_err}")

    return {
        "status": status_str,
        "reasons": reasons,
        "score": final_score,
        "qualified": is_qualified,
        "entry_price": entry_val,
        "history_class": history_class,
        "trend_validation_mode": trend_validation_mode,
        "stop_loss": sl_result.get("stop_loss"),
        "target_1": sl_result.get("target_1"),
        "target_2": sl_result.get("target_2"),
        "target_3": sl_result.get("target_3"),
        "target_4": sl_result.get("target_4"),
        "atr_14": float(bundle.atr_14.iloc[-1]) if hasattr(bundle, 'atr_14') and bundle.atr_14 is not None and not bundle.atr_14.empty and not pd.isna(bundle.atr_14.iloc[-1]) else float(entry_val * 0.025)
    }

detect_pullback_setup = evaluate_pullback_symbol

def start(force: bool = False, session=None, run_ctx=None, trigger_type="SCHEDULED", scheduler_name="CRON", used_fallback_data: bool = False):
    """
    Main entry point for Pullback Scanner. Acquires process lock and delegates to pipeline.
    """
    from database import is_scanner_stopped, upsert_scanner_health
    from lock_utils import print_scanner_start_banner, print_scanner_end_banner
    if is_scanner_stopped("PULLBACK"):
        logger.info("🛑 Pullback Scanner is STOPPED by Admin. Skipping execution.")
        if run_ctx:
            from database import complete_scanner_execution_run
            complete_scanner_execution_run(run_ctx, status_override="STOPPED", stop_reason="Scanner stopped by admin")
        return 0

    if _scan_lock.locked():
        logger.warning("🛑 [DUPLICATE GUARD] PULLBACK Scanner is ALREADY actively running in thread lock. Skipping duplicate trigger.")
        if run_ctx:
            from database import complete_scanner_execution_run
            complete_scanner_execution_run(run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Scanner lock busy")
        return {"status": "skipped", "reason": "already_running"}

    created_ctx = False
    acquired_global = False
    acquired_scan = False
    _scan_start = None

    try:
        queued_at = None
        if not _global_lock.acquire(blocking=False, owner_scanner="PULLBACK", operation="FULL_SCAN"):
            queued_at = time.monotonic()
            logger.info("⏳ [PULLBACK] Global scanner lock busy — waiting in queue until lock is released...")
            upsert_scanner_health("PULLBACK", "QUEUED", error_msg="Waiting in queue for active scanner to release lock...")
            
            try:
                acquired_global = _global_lock.acquire(blocking=True, owner_scanner="PULLBACK", operation="FULL_SCAN", run_ctx=run_ctx)
            except Exception as lock_err:
                logger.error(f"❌ [PULLBACK] Error acquiring global lock: {lock_err}")
                acquired_global = False

            if not acquired_global:
                logger.error("❌ [PULLBACK] Failed to acquire global scanner lock after queue wait.")
                if run_ctx:
                    from database import complete_scanner_execution_run
                    complete_scanner_execution_run(run_ctx, status_override="FAILED", stop_reason="Global lock acquire timeout")
                upsert_scanner_health("PULLBACK", "IDLE", error_msg="Lock acquisition timed out")
                return 0
        else:
            acquired_global = True

        if queued_at is not None:
            logger.info(f"✅ [PULLBACK] Global lock acquired after {round(time.monotonic()-queued_at,1)}s wait. Starting scan...")

        if not _scan_lock.acquire(blocking=False):
            logger.warning("🛑 PULLBACK Scanner is ALREADY actively running. Skipping duplicate execution.")
            if run_ctx:
                from database import complete_scanner_execution_run
                complete_scanner_execution_run(run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Scanner already actively running")
            else:
                try:
                    from database import record_skipped_execution_run
                    record_skipped_execution_run(scanner_name="PULLBACK", trigger_type=trigger_type, scheduler_name=scheduler_name, stop_reason="Scanner lock held (previous run active)")
                except Exception:
                    pass
            upsert_scanner_health("PULLBACK", "IDLE", error_msg="Duplicate trigger skipped")
            return 0
        acquired_scan = True

        # [RULE: HISTORY ENTRY AFTER LOCK ACQUIRED] Only create execution history entry once all locks are secured
        if run_ctx is None:
            try:
                from database import start_scanner_execution_run
                run_ctx = start_scanner_execution_run(scanner_name="PULLBACK", trigger_type=trigger_type, scheduler_name=scheduler_name)
            except Exception as exc:
                if "actively running" in str(exc).lower():
                    logger.info("🛑 [PULLBACK] Scanner is ALREADY actively running. Skipping duplicate execution.")
                    return 0
                logger.warning(f"⚠️ [PULLBACK] Could not create run_ctx: {exc}")
        elif run_ctx:
            from database import update_scanner_run_lifecycle
            update_scanner_run_lifecycle(run_ctx.run_id, "RUNNING")

        _scan_start = print_scanner_start_banner("pullback_scanner", queued_at=queued_at, run_id=run_ctx.run_id if run_ctx else None)
        total = run_pullback_pipeline(force=force, session=session, run_ctx=run_ctx, used_fallback_data=used_fallback_data)
        if run_ctx and isinstance(total, dict) and "total_count" in total:
            run_ctx.set_total_stocks(total["total_count"])
            run_ctx.fresh_count = total["processed_count"]
            if "today_alerts" in total:
                run_ctx.add_alert(total["today_alerts"])
        if run_ctx:
            from database import complete_scanner_execution_run
            complete_scanner_execution_run(run_ctx, status_override="COMPLETED")
        return total
    except Exception as e:
        logger.exception(f"❌ [PULLBACK] Unhandled exception during scan: {e}")
        if run_ctx:
            try:
                from database import complete_scanner_execution_run
                complete_scanner_execution_run(run_ctx, status_override="FAILED", exception=e)
            except Exception: pass
        try:
            upsert_scanner_health("PULLBACK", status="DOWN", error_msg=f"Scan crashed: {str(e)[:300]}", run_id=run_ctx.run_id if run_ctx else None)
            from database import insert_notification
            insert_notification("error", "🚨 PULLBACK Scanner CRASHED", f"Error: {str(e)[:400]}")
        except Exception: pass
        raise e
    finally:
        if _scan_start is not None:
            print_scanner_end_banner("pullback_scanner", _scan_start, run_id=run_ctx.run_id if run_ctx else None)

        if acquired_scan:
            try: _scan_lock.release()
            except Exception: pass
        if acquired_global:
            try: _global_lock.release()
            except Exception: pass

def _determine_dataset_date(sample_data: dict) -> Optional[str]:
    if not sample_data:
        return None
    dates = []
    for s_df in sample_data.values():
        if s_df is not None and not s_df.empty:
            try:
                last_dt = s_df.iloc[-1].name if isinstance(s_df.index, pd.DatetimeIndex) else s_df.iloc[-1].get("Date", s_df.iloc[-1].get("Datetime"))
                if last_dt:
                    dt_str = pd.to_datetime(last_dt).strftime("%Y-%m-%d")
                    dates.append(dt_str)
            except Exception:
                pass
    if not dates:
        return None
    from collections import Counter
    counts = Counter(dates)
    most_common_date, count = counts.most_common(1)[0]
    # Require at least 80% consensus across valid dates
    if count >= len(dates) * 0.8:
        return most_common_date
    return None

def run_pullback_pipeline(run_date: str = None, force: bool = False, session=None, run_ctx=None, used_fallback_data: bool = False) -> int:
    init_db()
    ist_now = datetime.now(IST)
    if not run_date:
        run_date = ist_now.strftime("%Y-%m-%d")
        
    logger.info("=" * 80)
    logger.info(f"🚀🚀🚀 [START] PULLBACK SCANNER PIPELINE INIT | {ist_now.strftime('%Y-%m-%d %H:%M:%S')} 🚀🚀🚀")
    logger.info("=" * 80)

    from perf_utils import ScannerStageTracker
    stage_tracker = ScannerStageTracker("PULLBACK_SCANNER")
    stage_tracker.start_stage(1, "Regime Check & Config Init", "Computing Nifty regime and loading effective config")
    
    try:
        upsert_scanner_health("PULLBACK", "RUNNING", error_msg="Pullback Scan in progress...")
    except Exception:
        logger.warning("⚠️ Could not mark PULLBACK as RUNNING")

    # Capture effective config snapshot at start of run (immutability)
    effective_config = dict(config)

    # ---------------- PRECONDITIONS & REGIME CHECK ----------------
    nifty_ret_20d = get_nifty_20d_return()
    market_regime = get_macro_regime(nifty_ret_20d)
    # central telemetry setup
    scan_id = run_ctx.run_id if (run_ctx and getattr(run_ctx, "run_id", None)) else f"run_{int(_time.time())}"
    telemetry_logger = ScannerDecisionLogger("PULLBACK", scan_id, market_regime)
    logger.info(f"📊 Market Regime: {market_regime}")

    if market_regime == "STRONG_BEAR":
        logger.info("🛑 STRONG_BEAR regime detected — Pullback scanner disabled entirely.")
        upsert_scanner_health("PULLBACK", status="OK", today_alerts=0, error_msg="Disabled in STRONG_BEAR regime")
        return 0

    regime_thresholds = {
        "STRONG_BULL": 74.0,
        "BULL": 74.0,
        "WEAK_BULL": 74.0,     # [FIX: EXPLICIT] Was missing — fell to default 76.0. Aligning with BULL tier.
        "NEUTRAL": 76.0,
        "SIDEWAYS": 76.0,      # [FIX: EXPLICIT] Was missing — fell to default 76.0. Now explicit.
        "RANGEBOUND": 76.0,    # [FIX: EXPLICIT] Was missing — fell to default 76.0. Now explicit.
        "WEAK_BEAR": 80.0,
        "BEAR": 80.0,
    }
    required_threshold = regime_thresholds.get(market_regime, 76.0)


    stage_tracker.end_stage(f"Regime={market_regime}")
    stage_tracker.start_stage(2, "Watchlist & Data Acquisition", "Loading fundamental watchlist and fetching historical price data")
    # ---------------- DATA READINESS & ACQUISITION ----------------
    try:
        watchlist = get_watchlist()
    except Exception as e:
        logger.exception("❌ Failed to load fundamental watchlist for Pullback Scanner")
        upsert_scanner_health("PULLBACK", status="DOWN", error_msg=f"Watchlist load failed: {str(e)[:200]}")
        return 0

    if watchlist.empty:
        logger.info("🛡️ Watchlist is empty. Exiting Pullback scan cleanly.")
        upsert_scanner_health("PULLBACK", status="OK", today_alerts=0, total_count=0, processed_count=0)
        return 0

    all_universe_symbols = [str(s) for s in watchlist["Stock"].tolist() if s]
    terminal_tracker = SingleTerminalTracker(all_universe_symbols, scanner_name="PULLBACK")
    terminal_tracker.map_gates_to_stage("FETCHED_DATA", [
        "COOLDOWN_ACTIVE", "STALE_DATA", "NO_DATA", "PROVIDER_ERROR", "INSUFFICIENT_BARS",
        "DATA_QUALITY_FAIL", "PROCESSING_ERROR"
    ])
    terminal_tracker.map_gates_to_stage("UPTREND_AND_STRUCTURE", [
        "NO_UPTREND", "NO_PIVOTS", "NO_IMPULSE", "PULLBACK_INVALID", "NO_TRIGGER"
    ])
    terminal_tracker.map_gates_to_stage("SCORE_THRESHOLD", [
        "SCORE_BELOW_THRESHOLD", "EOD_SUPPRESSED"
    ])
    terminal_tracker.map_gates_to_stage("RISK_ENGINE", [
        "RISK_REJECTED"
    ])
    terminal_tracker.map_gates_to_stage("FINAL_ALERTS", [
        "SUPPRESSED_FALLBACK_DATA", "SUPPRESSED_TOP_N", "PERSISTENCE_FAILED", "ALERT_GENERATED"
    ])
    waterfall = StageWaterfallTracker([
        "UNIVERSE_WATCHLIST",
        "FETCHED_DATA",
        "UPTREND_AND_STRUCTURE",
        "SCORE_THRESHOLD",
        "RISK_ENGINE",
        "FINAL_ALERTS"
    ])
    waterfall.set_stage_count("UNIVERSE_WATCHLIST", len(watchlist))
    near_miss_count = 0

    # Step 1: Check if today's dataset is already processed/available
    if session is not None:
        dataset_date = run_date
        sample_data = {}
    else:
        sample_chunk = watchlist.head(10)
        sample_data = fetch_watchlist_data(sample_chunk, "1y", "1d", requester="PULLBACK")
        dataset_date = _determine_dataset_date(sample_data)

    is_historical_fallback = False

    if dataset_date == run_date:
        logger.info(f"[PULLBACK] Using processed dataset for {dataset_date}")
    elif not force:
        # Step 2: Today's dataset is not available yet. Wait for Bhavcopy acquisition logic (scheduled mode).
        logger.info("[PULLBACK] Today's dataset unavailable, waiting for Bhavcopy...")
        try:
            from main import wait_for_bhavcopy_or_fallback
            wait_for_bhavcopy_or_fallback("PULLBACK")
        except Exception as bh_err:
            logger.warning(f"Could not execute Bhavcopy wait: {bh_err}")

        # Re-fetch sample after Bhavcopy wait
        sample_data = fetch_watchlist_data(sample_chunk, "1y", "1d", requester="PULLBACK")
        dataset_date = _determine_dataset_date(sample_data)

        if dataset_date == run_date:
            logger.info("[PULLBACK] Today's Bhavcopy processed successfully.")
        else:
            # Step 3: Today's Bhavcopy unavailable. Fallback to latest historical processed dataset (Read-Only)
            is_historical_fallback = True
            fallback_date = dataset_date or "HISTORICAL"
            logger.info(f"[PULLBACK] Admin mode using historical dataset from {fallback_date} (read-only fallback)")
    else:
        # Forced/Manual trigger mode: bypass blocking wait and execute using available dataset
        is_historical_fallback = True
        fallback_date = dataset_date or "HISTORICAL"
        logger.info(f"[PULLBACK] Forced/Manual trigger mode: bypassing Bhavcopy wait, using dataset from {fallback_date} (read-only fallback)")

    cooldown_alerts = get_recent_alerts_for_scanner("PULLBACK", PULLBACK_CONFIG.get("COOLDOWN_MINUTES", 1440))
    
    # 30-day prior alerts for evidence bonus calculation (+3 for EOD, +2 for MULTIBAGGER/MULTI_TF)
    prior_window_mins = effective_config.get("PRIOR_WINDOW", 30) * 1440
    prior_eod_symbols = {s for (s, _) in get_recent_alerts_for_scanner("EOD", prior_window_mins, only_active=True)}
    prior_multi_symbols = {s for (s, _) in get_recent_alerts_for_scanner("MULTIBAGGER", prior_window_mins, only_active=True)}.union(
        {s for (s, _) in get_recent_alerts_for_scanner("MULTI_TF", prior_window_mins, only_active=True)}
    )

    BATCH_SIZE = int(os.environ.get("PULLBACK_FETCH_BATCH_SIZE", "200"))
    total_batches = (len(watchlist) + BATCH_SIZE - 1) // BATCH_SIZE
    
    candidates: list[PullbackCandidate] = []

    rejected = {k: 0 for k in [
        "no_data", "provider_error", "insufficient_bars", "data_quality",
        "no_uptrend", "no_pivots", "no_impulse", "pullback_invalid",
        "no_trigger", "processing_error", "cooldown", "stale_data",
        "score_below_threshold", "risk_rejected", "eod_suppressed", "ranked_out", "persistence_failed"
    ]}
    provider_stats_counts = {
        "SUCCESS": 0, "NOT_FOUND": 0, "RATE_LIMIT": 0,
        "NETWORK_ERROR": 0, "TIMEOUT": 0, "EMPTY_DATA": 0
    }
    provider_resolved_symbols = set()
    fresh_valid_symbols = set()

    stage_tracker.end_stage(f"Watchlist={len(watchlist)} stocks loaded")
    stage_tracker.start_stage(3, "Symbol Evaluation Loop", f"Running pullback structure analysis on {len(watchlist)} stocks")
    stage_tracker.total_symbols = len(watchlist)
    # ---------------- ORCHESTRATION LOOP ----------------
    symbols_processed = 0
    if session is not None:
        logger.info(f"📦 [PULLBACK] Using MarketDataSession | {session.metadata.valid_symbols} symbols pre-fetched")
    _last_hb = time.monotonic()
    with MemoryProfiler("Pullback Scanner Process"):
        for batch_num, chunk_df in enumerate(chunk_iterable(watchlist, BATCH_SIZE), start=1):
            if run_ctx and (time.monotonic() - _last_hb) >= 10.0:
                try:
                    run_ctx.heartbeat()
                    _last_hb = time.monotonic()
                except Exception: pass
            with BatchMemoryTracker("PULLBACK", batch_num, total_batches, len(chunk_df), collect_gc=True) as tracker:
                _batch_start_t = time.perf_counter()
                # [VERSION: MARKET_DATA_SESSION_v1.0] Serve from session when available.
                all_ticker_data: dict = {}
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
                    # Pullback Pipeline requires SMA200 & 200-bar min history (200 trading days = ~280 cal days max).
                    # Using period="1y" shares the exact same Parquet cache files with Wealth Engine & EOD scanner,
                    # eliminating 50% data payload and preventing cache key fragmentation.
                    all_ticker_data = fetch_watchlist_data(chunk_df, interval="1d", period="1y", requester="PULLBACK")
                    
                _fetch_dur = time.perf_counter() - _batch_start_t
                _eval_start_t = time.perf_counter()
                if not all_ticker_data:
                    for _, row in chunk_df.iterrows():
                        symbols_processed += 1
                        rejected["provider_error"] += 1
                    continue


                from concurrent.futures import ThreadPoolExecutor, as_completed

                # [VERSION: IMPORT_HOIST_v1.0] Import shared modules ONCE outside the worker closure.
                # Per-symbol imports inside a ThreadPoolExecutor acquire sys.modules lock on every call.
                # Hoisting to the outer scope eliminates this per-worker lock contention.
                from core_enums import ProviderResult as _ProviderResult
                from indicator_manager import manager as _indicator_manager

                # Convert chunk_df to records to avoid iterrows overhead
                chunk_records = chunk_df.to_dict('records')
                
                # [RULE 67 CHANGE-RATIONALE]: Pass ticker_data_map explicitly to avoid closure capture issues
                # with del all_ticker_data and clarify worker data contract.
                def _evaluate_row(row_dict, ticker_data_map):
                    sym = row_dict.get("Stock", "UNKNOWN")
                    try:
                        category = row_dict.get("Category", "MIDCAP")
                        sector = row_dict.get("Sector", None)

                        ticker_data = ticker_data_map.get(sym)
                        if ticker_data is None: ticker_data = ticker_data_map.get(f"{sym}.NS")
                        if ticker_data is None: ticker_data = ticker_data_map.get(f"{sym}.BO")
                        if ticker_data is None: ticker_data = ticker_data_map.get(sym.split('.')[0])

                        if ticker_data is None:
                            return (None, "no_data", "EMPTY_DATA", None, sym)

                        if isinstance(ticker_data, _ProviderResult):
                            return (None, "provider_error", ticker_data.name, None, sym)

                        if (sym, "PULLBACK") in cooldown_alerts:
                            return (None, "cooldown", "SUCCESS", None, sym)

                        df = ticker_data.copy()
                        df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

                        if getattr(ticker_data, 'attrs', {}).get('is_stale'):
                            return (None, "stale_data", "SUCCESS", None, sym)
                            
                        _stale_col = next((c for c in ["Date", "Datetime"] if c in df.columns), None)
                        if is_historical_fallback and dataset_date:
                            try:
                                _target_val = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else (df.iloc[-1][_stale_col] if _stale_col else None)
                                if _target_val is None:
                                    return (None, "stale_data", "SUCCESS", None, sym)
                                _last_ts = pd.to_datetime(_target_val)
                                if _last_ts.date() != pd.to_datetime(dataset_date).date():
                                    return (None, "stale_data", "SUCCESS", None, sym)
                            except Exception as e:
                                return (None, "stale_data", "SUCCESS", None, sym)
                        else:
                            _target_val = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else (df.iloc[-1][_stale_col] if _stale_col else None)
                            if _target_val is None:
                                return (None, "stale_data", "SUCCESS", None, sym)
                            try:
                                from market_utils import evaluate_data_staleness
                                _last_ts = pd.to_datetime(_target_val)
                                _staleness = evaluate_data_staleness(_last_ts, ist_now)
                                if _staleness["is_stale"]:
                                    return (None, "stale_data", "SUCCESS", None, sym)
                            except Exception as e:
                                return (None, "stale_data", "SUCCESS", None, sym)

                        if df.empty or len(df) < 15:
                            return (None, "insufficient_bars", "SUCCESS", None, sym)

                        df.attrs['adjusted'] = True
                        df.attrs['symbol'] = sym
                        as_of_index = len(df) - 1
                        historical_view = df.iloc[:as_of_index + 1]

                        try:
                            swing_utils.check_data_quality(historical_view)
                        except DataQualityError as dqe:
                            return (None, "data_quality", "SUCCESS", None, sym)

                        fresh_val = sym

                        bundle = _indicator_manager.compute_base_indicators(historical_view, sym)
                        last_bar = historical_view.iloc[-1]
                        # Extract SMA and EMA values for trend validation
                        sma50_val = float(bundle.sma_50.iloc[-1]) if hasattr(bundle, 'sma_50') and bundle.sma_50 is not None and not bundle.sma_50.empty and not pd.isna(bundle.sma_50.iloc[-1]) else None
                        sma200_val = float(bundle.sma_200.iloc[-1]) if hasattr(bundle, 'sma_200') and bundle.sma_200 is not None and not bundle.sma_200.empty and not pd.isna(bundle.sma_200.iloc[-1]) else None
                        
                        ctx = telemetry_logger.get_or_create_context(sym)
                        ctx.capture_dataframe_row(last_bar, is_fallback=used_fallback_data)
                        
                        n_bars = len(historical_view)
                        if n_bars >= 200:
                            history_class = "MATURE"
                            trend_validation_mode = "SMA200"
                            uptrend_ok = bool(sma50_val and sma200_val and sma50_val > sma200_val and last_bar['Close'] >= (sma50_val * 0.95))
                            act_u = {"Close": round(float(last_bar['Close']), 2), "SMA50": round(float(sma50_val), 2) if sma50_val else None, "SMA200": round(float(sma200_val), 2) if sma200_val else None, "Tier": history_class}
                            req_u = {"Close >= 0.95*SMA50": round(float(sma50_val * 0.95), 2) if sma50_val else None, "SMA50 > SMA200": True}
                        elif n_bars >= 50:
                            history_class = "RECENT_LISTING"
                            trend_validation_mode = "SMA50_EMA20"
                            ema20_val = bundle.ema_20.iloc[-1] if bundle.ema_20 is not None and not bundle.ema_20.empty else None
                            uptrend_ok = bool(sma50_val and last_bar['Close'] >= (sma50_val * 0.95) and (ema20_val >= (sma50_val * 0.98) if ema20_val else True))
                            act_u = {"Close": round(float(last_bar['Close']), 2), "SMA50": round(float(sma50_val), 2) if sma50_val else None, "EMA20": round(float(ema20_val), 2) if ema20_val else None, "Tier": history_class}
                            req_u = {"Close >= 0.95*SMA50": round(float(sma50_val * 0.95), 2) if sma50_val else None, "EMA20 >= 0.98*SMA50": True}
                        else:  # FRESH_IPO
                            history_class = "FRESH_IPO"
                            trend_validation_mode = "EMA20"
                            ema20_val = bundle.ema_20.iloc[-1] if bundle.ema_20 is not None and not bundle.ema_20.empty else None
                            uptrend_ok = bool(ema20_val and last_bar['Close'] >= (ema20_val * 0.95))
                            act_u = {"Close": round(float(last_bar['Close']), 2), "EMA20": round(float(ema20_val), 2) if ema20_val else None, "Tier": history_class}
                            req_u = {"Close >= 0.95*EMA20": round(float(ema20_val * 0.95), 2) if ema20_val else None}

                        if not uptrend_ok:
                            return (None, "no_uptrend", "SUCCESS", fresh_val, sym, act_u, req_u)

                        pivots = swing_utils.detect_confirmed_pivots(historical_view, effective_config["LOOKBACK"], effective_config["CONFIRM"])
                        if not pivots:
                            return (None, "no_pivots", "SUCCESS", fresh_val, sym, {"Pivots": 0}, {"Min Pivots": 1})

                        impulse = swing_utils.select_pullback_origin(pivots, historical_view, effective_config)
                        if not impulse:
                            return (None, "no_impulse", "SUCCESS", fresh_val, sym, {"Impulse": "None"}, {"Valid Impulse": True})

                        ps = swing_utils.measure_pullback(historical_view, impulse, effective_config, debug=effective_config.get("DEBUG_SWINGS", False))
                        save_funnel_telemetry("PULLBACK", run_date, sym, ps.stage_results)
                    
                        if not ps.valid:
                            v_ratio = round(float(ps.volume_ratio), 2) if hasattr(ps, 'volume_ratio') and ps.volume_ratio is not None else None
                            rej_obj = getattr(ps, "rejection_reason", None)
                            if hasattr(rej_obj, "name"):
                                failed_reason = rej_obj.name
                            elif rej_obj is not None:
                                failed_reason = str(rej_obj)
                            else:
                                failed_reason = "UNKNOWN"

                            act_pb = {
                                "retracement_pct": round(float(ps.depth_pct), 1) if ps.depth_pct is not None else None,
                                "volume_ratio": v_ratio,
                                "duration_bars": getattr(ps, 'duration_bars', None),
                                "failed_gate": failed_reason
                            }
                            req_pb = {
                                "min_depth_pct": float(effective_config.get("MIN_DEPTH_PCT", 10.0)),
                                "max_depth_pct": float(effective_config.get("MAX_DEPTH_PCT", 78.6)),
                                "max_volume_ratio": float(effective_config.get("MAX_PB_VOLUME_RATIO", 1.25)),
                                "min_duration_bars": int(effective_config.get("MIN_DURATION", 3)),
                                "max_duration_bars": int(effective_config.get("MAX_DURATION", 20))
                            }
                            return (None, "pullback_invalid", "SUCCESS", fresh_val, sym, act_pb, req_pb)

                        trig = swing_utils.detect_resumption_trigger(historical_view, ps, effective_config)
                        if not trig.valid:
                            act_tr = {"Entry Price": round(float(trig.entry_price), 2), "Reason": getattr(trig, 'reason', 'No trigger condition met')}
                            req_tr = {"Valid Trigger": True}
                            return (None, "no_trigger", "SUCCESS", fresh_val, sym, act_tr, req_tr)

                        logger.info(f"📍 PICKED [PULLBACK: IN BETWEEN]: {sym} @ ₹{trig.entry_price:.2f} (Retracement: {ps.depth_pct:.1f}%, Vol Ratio: {ps.volume_ratio:.2f}x)")
                        
                        close_position = getattr(trig, "close_position", 0.5)
                        atr_val = float(bundle.atr_14.iloc[-1]) if hasattr(bundle, 'atr_14') and bundle.atr_14 is not None and not bundle.atr_14.empty and not pd.isna(bundle.atr_14.iloc[-1]) else float(trig.entry_price) * 0.025

                        cand = PullbackCandidate(
                            symbol=sym,
                            as_of_date=ist_now.date(),
                            structure=ps,
                            trigger=trig,
                            entry_price=trig.entry_price,
                            warnings=[],
                            config_version=effective_config.get("VERSION", "pb-1.0.0"),
                            sector=sector,
                            status=CandidateState.NEW
                        )
                        cand.trigger_close_position = close_position
                        cand.atr_val = atr_val
                        
                        return (cand, None, "SUCCESS", fresh_val, sym, None, None)
                    except Exception as sym_err:
                        logger.error(f"❌ Error processing symbol {sym} in Pullback Scanner: {sym_err}")
                        return (None, "processing_error", "SUCCESS", None, sym, {"Error": str(sym_err)[:100]}, None)

                try:
                    from config import SCAN_WORKER_THREADS
                except ImportError:
                    SCAN_WORKER_THREADS = 8
                
                workers = min(os.cpu_count() or 8, SCAN_WORKER_THREADS, len(chunk_records))
                workers = max(1, workers)
                
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="PullbackWorker") as executor:
                    future_to_sym = {
                        executor.submit(_evaluate_row, rec, all_ticker_data): rec.get("Stock", "UNKNOWN")
                        for rec in chunk_records
                    }
                    for future in as_completed(future_to_sym):
                        sym = future_to_sym[future]
                        symbols_processed += 1
                        try:
                            future_res = future.result()
                            cand = future_res[0]
                            rej_reason = future_res[1]
                            prov_stat = future_res[2]
                            fresh_val = future_res[3]
                            act_data = future_res[5] if len(future_res) > 5 else None
                            req_data = future_res[6] if len(future_res) > 6 else None

                            if prov_stat != "EMPTY_DATA" and prov_stat != "SUCCESS":
                                provider_stats_counts[prov_stat] = provider_stats_counts.get(prov_stat, 0) + 1
                                provider_resolved_symbols.add(sym)
                            elif prov_stat == "EMPTY_DATA":
                                provider_stats_counts["EMPTY_DATA"] += 1
                            else:
                                provider_stats_counts["SUCCESS"] += 1
                                provider_resolved_symbols.add(sym)
                                
                            if fresh_val:
                                fresh_valid_symbols.add(fresh_val)
                                
                            if rej_reason:
                                rejected[rej_reason] = rejected.get(rej_reason, 0) + 1
                                terminal_tracker.record_terminal(sym, rej_reason.upper(), f"Pre-check gate: {rej_reason}")
                                logger.info(
                                    f"🚫 [PULLBACK] {sym} REJECTED — Gate: {rej_reason.upper()} | "
                                    f"Actual: {act_data} | Required: {req_data}"
                                )
                                telemetry_logger.record_reject(
                                    symbol=sym,
                                    last_stage="PRE_CHECK",
                                    gate=rej_reason.upper(),
                                    actual=act_data,
                                    required=req_data,
                                    start_time=_batch_start_t
                                )
                            if cand:
                                candidates.append(cand)
                                logger.info(
                                    f"👁️ [PULLBACK: SETUP WATCH] {cand.symbol} added to Watchlist | "
                                    f"CMP: ₹{getattr(cand, 'entry_price', 0.0):.2f} | 20 EMA: ₹{getattr(cand, 'ema20', 0.0):.2f} — (In pullback zone, awaiting confirmation trigger)"
                                )
                        except Exception as e:
                            logger.error(f"Error in pullback thread processing: {e}")
                            rejected["processing_error"] = rejected.get("processing_error", 0) + 1
                            terminal_tracker.record_terminal(sym, "PROCESSING_ERROR", f"Thread exception: {str(e)[:100]}")
                            telemetry_logger.record_reject(
                                symbol=sym,
                                last_stage="PRE_CHECK",
                                gate="PROCESSING_ERROR",
                                actual=str(e)[:100],
                                required=None,
                                start_time=_batch_start_t
                            )

            del all_ticker_data
            import gc; gc.collect()
            
            _eval_dur = time.perf_counter() - _eval_start_t
            logger.info(
                f"⏱️ [PULLBACK SCANNER] Batch {batch_num}/{total_batches} Timing | "
                f"Fetch {len(chunk_df)} symbols: {_fetch_dur:.2f}s | "
                f"Evaluation: {_eval_dur:.2f}s | "
                f"Candidates found so far: {len(candidates)}"
            )
            
            logger.info(f"⏳ [PULLBACK SCANNER] Evaluated Batch {batch_num}/{total_batches} ({min(batch_num * BATCH_SIZE, len(watchlist))}/{len(watchlist)} stocks) | Candidates found so far: {len(candidates)}")

    logger.info(f"📊 Pullback Candidates Discovered: {len(candidates)}")
    
    # ── CRITICAL BLOCKER GUARD ──
    no_data_count = rejected.get("no_data", 0)
    stale_count = rejected.get("stale_data", 0)
    dirty_count = rejected.get("data_quality", 0)
    provider_errors_count = rejected.get("provider_error", 0)
    # Exclude dirty_count (data quality) as it represents stock-specific issues, not infrastructure outages
    total_failures = no_data_count + stale_count + provider_errors_count
    total_symbols = len(watchlist)
    total_fetched_count = len(provider_resolved_symbols)
    elapsed_time = round((datetime.now(IST) - ist_now).total_seconds(), 1)

    status_val = "OK"
    err_val = None

    if not is_historical_fallback and total_symbols > 0 and total_failures >= total_symbols * 0.25:
        status_val = "DOWN"
        err_val = f"🚫 CRITICAL BLOCKER: {total_failures}/{total_symbols} symbols missing/stale/error (≥25%)"
        logger.error(f"🚨 {err_val}")
        try:
            from telegram_engine import send_telegram_message
            send_telegram_message(f"🚨 <b>CRITICAL BLOCKER: PULLBACK SCANNER FAILED</b>\n{err_val}")
        except Exception:
            pass
            
        try:
            from push_service import send_push_to_all
            send_push_to_all(
                title="🚨 CRITICAL DATA OUTAGE",
                body=f"PULLBACK SCANNER Halted. {total_failures}/{total_symbols} symbols failed data quality checks.",
                bypass_throttle=True
            )
        except Exception as e:
            logger.error(f"Could not dispatch web push: {e}")
            
        upsert_scanner_health(
            "PULLBACK",
            status=status_val,
            last_success=ist_now.isoformat(),
            today_alerts=0,
            total_count=total_symbols,
            processed_count=symbols_processed,
            duration_seconds=elapsed_time,
            error_msg=err_val
        )
        logger.info(f"🚫 [HALTED] PULLBACK SCANNER terminated early due to critical data outage ({total_failures}/{total_symbols} symbols failed).")
        return {
            "total_count": total_symbols,
            "processed_count": symbols_processed,
            "today_alerts": 0
        }

    # Partial fetch degraded status check (warn but do not block)
    if not is_historical_fallback and total_symbols > 0 and total_fetched_count < total_symbols * 0.70:
        status_val = "DEGRADED"
        err_val = f"Partial Fetch: {total_fetched_count}/{total_symbols} symbols"

    # [RULE 67] DEGRADED_FALLBACK takes precedence over DEGRADED.
    # User explicitly required this at the final health write level:
    # "enforce DEGRADED_FALLBACK at the final health write level, not just local variables."
    # is_historical_fallback is set at lines 537/542 when Bhavcopy fallback dataset is used.
    if is_historical_fallback:
        status_val = "DEGRADED_FALLBACK"
        err_val = f"Historical fallback dataset used (fallback from current date)"

    stage_tracker.end_stage(f"Candidates={len(candidates)} pullback structures found out of {symbols_processed} processed")
    stage_tracker.start_stage(4, "Scoring & RS/Sector Modifiers", f"Computing pullback scores for {len(candidates)} candidates")
    # ---------------- SCORING & MODIFIERS ----------------
    from macro_utils import compute_nifty_rs_rating, compute_sector_regime_rankings

    rs_rankings = {}
    missing_rs = False
    if candidates:
        try:
            rs_rankings = compute_nifty_rs_rating([c.symbol for c in candidates])
        except Exception as rs_err:
            logger.warning(f"Failed to compute RS ratings for pullbacks: {rs_err}")
            missing_rs = True

    sector_rankings_dict = {}
    missing_sector = False
    try:
        sector_rankings_dict = compute_sector_regime_rankings()
    except Exception as sec_err:
        logger.warning(f"Failed to compute sector rankings for pullbacks: {sec_err}")
        missing_sector = True

    service_warnings = []
    if missing_rs:
        service_warnings.append("missing_rs")
    if missing_sector:
        service_warnings.append("missing_sector")

    if service_warnings:
        warn_str = f"Service failure ({', '.join(service_warnings)})"
        err_val = f"{err_val} | {warn_str}" if err_val else warn_str
        # Fail-safe threshold markup
        required_threshold += 3.0
        logger.warning(f"⚠️ Service failure detected. Required threshold raised to {required_threshold}")

    for c in candidates:
        rs_pct_val = float(rs_rankings.get(c.symbol, 50.0))
        
        sector_info = sector_rankings_dict.get(c.sector, {}) if c.sector else {}
        sector_status = sector_info.get("effective_status", "NEUTRAL")
        
        vol_ratio = float(c.structure.volume_ratio) if hasattr(c.structure, 'volume_ratio') and c.structure.volume_ratio is not None else 1.0
        
        close_position = getattr(c, "trigger_close_position", 0.5)
        volume_mult = c.trigger.volume_mult

        has_prior_eod = c.symbol in prior_eod_symbols
        has_prior_multi = c.symbol in prior_multi_symbols

        score_breakdown = compute_pullback_score(
            pullback_count_in_trend=c.structure.pullback_count_in_trend,
            volume_ratio=vol_ratio,
            trigger_close_position=close_position,
            trigger_volume_mult=volume_mult,
            rs_percentile=rs_pct_val,
            sector_status=sector_status,
            has_prior_eod=has_prior_eod,
            has_prior_multi=has_prior_multi,
            is_full_high_takeover=getattr(c.trigger, "is_full_high_takeover", False),
            is_bullish_engulfing=getattr(c.trigger, "is_bullish_engulfing", False)
        )
        c.base_score = score_breakdown["base_score"]
        c.final_score = score_breakdown["final_score"]
        c.score_breakdown = score_breakdown

    # Filter out scores below threshold
    scored_candidates = []
    for c in candidates:
        if c.final_score < required_threshold:
            logger.info(f"🚫 [PULLBACK] {c.symbol} REJECTED — Gate: SCORE_BELOW_THRESHOLD | Score: {c.final_score:.1f} < Required: {required_threshold}")
            rejected["score_below_threshold"] += 1
            if (required_threshold - c.final_score) <= 5.0:
                near_miss_count += 1
            terminal_tracker.record_terminal(c.symbol, "SCORE_BELOW_THRESHOLD", f"Score {c.final_score:.1f} < {required_threshold}")
            telemetry_logger.record_reject(
                symbol=c.symbol,
                last_stage="SCORE_GATE",
                gate="SCORE_BELOW_THRESHOLD",
                actual=float(c.final_score),
                required=float(required_threshold)
            )
            try:
                from near_miss_tracker import log_near_miss
                entry_px = float(c.entry_price) if hasattr(c, 'entry_price') and c.entry_price else None
                sl_px = None
                tgt_px = None
                if entry_px and entry_px > 0:
                    sl_calc = compute_sl_and_target(
                        entry_price=entry_px,
                        atr=getattr(c, "atr_val", entry_px * 0.025),
                        mode="PULLBACK",
                        swing_low=c.structure.pullback_low.price if hasattr(c, "structure") and hasattr(c.structure, "pullback_low") else None,
                        swing_high=c.structure.impulse.end.price if hasattr(c, "structure") and hasattr(c.structure, "impulse") else None,
                    )
                    sl_px = float(sl_calc.get("stop_loss", 0.0)) if sl_calc.get("stop_loss") else round(entry_px * 0.95, 2)
                    tgt_px = float(sl_calc.get("target_1", 0.0)) if sl_calc.get("target_1") else round(entry_px + 2.0 * (entry_px - sl_px), 2)
                log_near_miss(
                    symbol=c.symbol,
                    scanner="PULLBACK",
                    breakout_type="PULLBACK_SETUP",
                    gate_name="score_below_threshold",
                    observed_value=float(c.final_score),
                    threshold_value=float(required_threshold),
                    score=int(c.final_score),
                    entry_price=entry_px,
                    stop_loss=sl_px,
                    target_1=tgt_px,
                )
            except Exception as _nm_e:
                # [FIX-P4] Promote to WARNING so near-miss telemetry failures are visible in production logs.
                logger.warning(f"⚠️ [PULLBACK] Near-miss log failed for {c.symbol}: {_nm_e}")
        else:
            scored_candidates.append(c)

    # ---------------- SAME-NIGHT EOD SUPPRESSION ----------------
    tonight_eod_alerts = get_recent_alerts_for_scanner("EOD", 300)
    for c in scored_candidates:
        if (c.symbol, "EOD") in tonight_eod_alerts:
            c.status = CandidateState.SUPPRESSED
            c.suppressed_by = "EOD"
            rejected["eod_suppressed"] += 1
            terminal_tracker.record_terminal(c.symbol, "EOD_SUPPRESSED", "Primary EOD alert already generated tonight")
            logger.info(f"🚫 [PULLBACK] {c.symbol} REJECTED — Gate: EOD_SUPPRESSED | Reason: Primary EOD alert already generated tonight")
            telemetry_logger.record_reject(
                symbol=c.symbol,
                last_stage="EOD_SUPPRESSION",
                gate="EOD_SUPPRESSED",
                actual=None,
                required=None
            )

    survivors = [c for c in scored_candidates if c.status != CandidateState.SUPPRESSED]

    stage_tracker.end_stage(f"Scored={len(scored_candidates)} cleared threshold, {rejected.get('score_below_threshold',0)} rejected")
    stage_tracker.start_stage(5, "Risk Engine & Alert Persistence", f"Validating SL/target for {len(scored_candidates)} candidates and saving alerts")
    # ---------------- RISK ENGINE VALIDATION ----------------
    valid_risk_candidates = []
    for c in survivors:
        entry_val = float(c.entry_price)
        sl_result = compute_sl_and_target(
            entry_price=entry_val,
            atr=getattr(c, "atr_val", entry_val * 0.025),
            mode="PULLBACK",
            swing_low=c.structure.pullback_low.price,
            swing_high=c.structure.impulse.end.price,
        )

        if sl_result.get("is_rejected"):
            logger.info(f"🚫 [PULLBACK] {c.symbol} REJECTED — Gate: RISK_REJECTED | Reason: {sl_result.get('rejection_reason')} | RR: {sl_result.get('natural_rr', 0.0):.2f}")
            c.status = CandidateState.REJECTED
            rejected["risk_rejected"] += 1
            terminal_tracker.record_terminal(c.symbol, "RISK_REJECTED", f"Risk engine: {sl_result.get('rejection_reason')}")

            sl_val = float(sl_result.get("stop_loss", 0.0))
            t1_val = float(sl_result.get("target_1", 0.0))
            nat_rr = float(sl_result.get("natural_rr", 0.0))
            risk_pct = round(((entry_val - sl_val) / entry_val) * 100.0, 2) if entry_val > 0 and sl_val > 0 else 0.0
            reward_pct = round(((t1_val - entry_val) / entry_val) * 100.0, 2) if entry_val > 0 and t1_val > 0 else 0.0

            actual_risk_metrics = {
                "Entry": round(entry_val, 2),
                "Stop Loss": round(sl_val, 2),
                "Target 1": round(t1_val, 2),
                "Natural RR": round(nat_rr, 2),
                "Risk %": risk_pct,
                "Reward %": reward_pct,
                "SL Method": sl_result.get("sl_method", "UNKNOWN"),
                "Target Method": sl_result.get("target_method", "UNKNOWN")
            }
            required_risk_metrics = {
                "Min RR": sl_result.get("min_rr_threshold", 2.0),
                "Reason": sl_result.get("rejection_reason", "RISK_REJECTED")
            }

            telemetry_logger.record_reject(
                symbol=c.symbol,
                last_stage="RISK_ENGINE",
                gate="RISK_REJECTED",
                actual=actual_risk_metrics,
                required=required_risk_metrics
            )
            try:
                from near_miss_tracker import log_near_miss
                log_near_miss(
                    symbol=c.symbol,
                    scanner="PULLBACK",
                    breakout_type="PULLBACK_SETUP",
                    gate_name=str(sl_result.get("rejection_reason") or "risk_rejected"),
                    observed_value=float(nat_rr) if nat_rr > 0 else float(c.final_score),
                    threshold_value=float(sl_result.get("min_rr_threshold", 2.0)) if nat_rr > 0 else 80.0,
                    score=int(c.final_score),
                    entry_price=entry_val,
                    stop_loss=sl_val,
                    target_1=t1_val,
                )
            except Exception:
                pass
        else:
            c.sl_result = sl_result
            valid_risk_candidates.append(c)

    # ---------------- ALERT LIMITING & SORTING ----------------
    valid_risk_candidates.sort(key=lambda x: x.final_score, reverse=True)
    from config import SCANNER_MAX_ALERTS
    max_alerts = SCANNER_MAX_ALERTS.get("PULLBACK", 10)
    if len(valid_risk_candidates) > max_alerts:
        logger.info(f"Limiting PULLBACK alerts from {len(valid_risk_candidates)} to {max_alerts}")
        ranked_out = valid_risk_candidates[max_alerts:]
        from database import save_rejected_alert
        for c in ranked_out:
            c.status = CandidateState.SUPPRESSED
            rejected["ranked_out"] += 1
            terminal_tracker.record_terminal(c.symbol, "SUPPRESSED_TOP_N", f"Score {c.final_score:.1f} exceeded top {max_alerts}")
            logger.info(f"🚫 {c.symbol} alert SUPPRESSED: Exceeded MAX_ALERTS_PER_SCAN limit (Score: {c.final_score:.1f})")
            telemetry_logger.record_reject(
                symbol=c.symbol,
                last_stage="RANK_LIMIT",
                gate="RANKED_OUT",
                actual=float(c.final_score),
                required=None
            )
            try:
                save_rejected_alert(c.symbol, "PULLBACK", "RANKED_OUT", context={"score": c.final_score})
            except Exception:
                pass
    alertable = valid_risk_candidates[:max_alerts]

    # [ADJUSTED TRADING SESSION NORMALIZATION]
    # Normalize source_trading_date so weekend runs (Saturday/Sunday) inherit the
    # Friday trading session, enabling clean deduplication across Friday -> Sat -> Sun.
    from market_utils import get_expected_latest_trading_date
    source_trading_date = get_expected_latest_trading_date(ist_now)

    # ---------------- SIGNAL DISPATCH & PERSISTENCE ----------------
    alert_count = 0
    for c in alertable:
        entry_val = float(c.entry_price)
        # Ensure post-market entry price matches today's live CMP
        try:
            from price_cache import get_cached_price
            fast_p = get_cached_price(c.symbol)
            if fast_p and float(fast_p) > 0:
                entry_val = round(float(fast_p), 2)
        except Exception:
            pass

        sl_result = c.sl_result

        # [VERSION: ALL_ALERTS_PERSIST_v1.0] Dry-run mode disabled — all generated alerts persist to DB at all times.
        rs_pct_val = float(rs_rankings.get(c.symbol, 50.0))
            
        sector_info = sector_rankings_dict.get(c.sector, {}) if c.sector else {}
        sector_name_val = sector_info.get("sector_name", "")

        # Exact score breakdown reconstruction for DB persistence
        rs_bonus_val = round(c.score_breakdown.get("rs_bonus", 0.0), 1)
        sector_bonus_val = round(c.score_breakdown.get("sector_bonus", 0.0), 1)
        final_score_val = round(c.final_score, 1)
        base_score_val = round(c.base_score, 1)

        saved, reason, _, _ = save_alert_if_new(
            symbol=c.symbol,
            breakout_type="PULLBACK",
            alert_time=ist_now.strftime("%Y-%m-%d %H:%M:%S+05:30"),
            scanner="PULLBACK",
            category="PULLBACK",
            entry_price=entry_val,
            stop_loss=sl_result.get("stop_loss"),
            target_1=sl_result.get("target_1"),
            target_2=sl_result.get("target_2"),
            target_3=sl_result.get("target_3"),
            score=final_score_val,
            context={
                "config_version": c.config_version,
                "source_trading_date": str(source_trading_date),
                "structure": {
                    "depth_pct": c.structure.depth_pct,
                    "duration_bars": c.structure.duration_bars,
                    "volume_ratio": c.structure.volume_ratio
                },
                "score_breakdown": {
                    "base_score": base_score_val,
                    "rs_bonus": rs_bonus_val,
                    "sector_bonus": sector_bonus_val,
                    "vol_bonus": round(c.score_breakdown.get("vol_bonus", 0.0), 1),
                    "trigger_bonus": round(c.score_breakdown.get("trigger_bonus", 0.0), 1),
                    "eligible_bonus": round(c.score_breakdown.get("eligible_bonus", 0.0), 1),
                    "penalty": round(c.score_breakdown.get("penalty", 0.0), 1)
                }
            },
            base_score=base_score_val,
            rs_bonus=rs_bonus_val,
            sector_bonus=sector_bonus_val,
            rs_percentile=rs_pct_val,
            sector_name=sector_name_val,
            regime_score=float(MarketRegimeEngine.get_regime_context().get("market_score", 80.0)),
            entry_mode="LIMIT_PULLBACK",
            source_trading_date=source_trading_date
        )

        if saved:
            alert_count += 1
            c.status = CandidateState.ALERTED
            terminal_tracker.record_terminal(c.symbol, "ALERT_GENERATED", f"Pullback alert saved (Score: {final_score_val})")
            logger.info(
                f"🌟 [PULLBACK: SELECTED] {c.symbol} | "
                f"score={final_score_val} | entry=₹{entry_val:.2f} | "
                f"depth={c.structure.depth_pct:.1f}% | volume_ratio={c.structure.volume_ratio:.2f}x | "
                f"SL=₹{sl_result.get('stop_loss', 0):.2f} | T1=₹{sl_result.get('target_1', 0):.2f}"
            )
            logger.info(
                f"✅ [PULLBACK] PASSED ALL FILTERS: {c.symbol} | "
                f"score={final_score_val} | entry=₹{entry_val:.2f} | "
                f"depth={c.structure.depth_pct:.1f}% | volume_ratio={c.structure.volume_ratio:.2f} | "
                f"category=PULLBACK"
            )
            telemetry_logger.record_pass(
                symbol=c.symbol,
                score=final_score_val,
                rr=float(sl_result.get("natural_rr", 2.0)),
                metadata={
                    "depth_pct": c.structure.depth_pct,
                    "volume_ratio": c.structure.volume_ratio
                }
            )
            try:
                from telegram_engine import send_telegram_message
                msg = (
                    f"↪️ <b>PULLBACK CONTINUATION ALERT</b> ↪️\n\n"
                    f"📌 <b>Symbol:</b> #{c.symbol}\n"
                    f"💰 <b>Entry Price:</b> ₹{entry_val:.2f}\n"
                    f"🛑 <b>Stop Loss:</b> ₹{sl_result.get('stop_loss', 0):.2f}\n"
                    f"🎯 <b>Target 1:</b> ₹{sl_result.get('target_1', 0):.2f}\n"
                    f"🎯 <b>Target 2:</b> ₹{sl_result.get('target_2', 0):.2f}\n"
                    f"🎯 <b>Target 3:</b> ₹{sl_result.get('target_3', 0):.2f}\n"
                    f"📊 <b>Score:</b> {c.final_score:.1f}/100\n"
                    f"📉 <b>Pullback Retracement:</b> {c.structure.depth_pct:.1f}% of impulse wave ({c.structure.duration_bars} bars)\n"
                    f"🔊 <b>Volume Ratio:</b> {c.structure.volume_ratio:.2f}x\n"
                    f"⚡ <b>Mode:</b> LIVE PRODUCTION"
                )
                send_telegram_message(msg, scan_type="PULLBACK")
            except Exception as tg_err:
                logger.warning(f"⚠️ Could not dispatch Telegram message for {c.symbol}: {tg_err}")
        else:
            c.status = CandidateState.SUPPRESSED
            rejected["persistence_failed"] += 1
            term_reason = "DUPLICATE_ALERT" if "Duplicate" in str(reason) else "PERSISTENCE_FAILED"
            terminal_tracker.record_terminal(c.symbol, term_reason, reason or "Failed to save to database")
            logger.info(f"REJECTION: {c.symbol} (Phase: PERSISTENCE, Reason: {reason})")
            telemetry_logger.record_reject(
                symbol=c.symbol,
                last_stage="PERSISTENCE",
                gate=term_reason,
                actual=None,
                required=None
            )

    provider_stats_counts["STALE"] = rejected.get("stale_data", 0)
    upsert_scanner_health(
        "PULLBACK",
        status=status_val,
        last_success=ist_now.isoformat(),
        today_alerts=alert_count,
        total_count=total_symbols,
        processed_count=symbols_processed,
        duration_seconds=elapsed_time,
        error_msg=err_val,
        provider_stats=provider_stats_counts
    )
    if status_val == "OK" and alert_count > 0:
        # [RULE 67 CHANGE-RATIONALE]: Ensure newly persisted pullback alerts are immediately reflected in performance_data
        try:
            from performance_tracker import trigger_performance_rebuild
            trigger_performance_rebuild(force=True)
        except Exception:
            pass
        try:
            insert_notification("admin", f"🎯 Pullback Scanner ran successfully. Found {alert_count} pullback alerts.", f"Generated {alert_count} alerts from {total_symbols} scanned stocks. Outcome: SUCCESS")
            from push_service import send_push_to_all
            send_push_to_all("🎯 Pullback Scanner OK", f"Found {alert_count} pullback alerts.", bypass_throttle=True)
        except Exception:
            pass
    elif status_val == "DEGRADED":
        try:
            insert_notification("admin", f"⚠️ Pullback Scanner finished with DEGRADED status", err_val or f"Generated {alert_count} alerts but data was degraded.")
            from push_service import send_push_to_all
            send_push_to_all("⚠️ Pullback Scanner DEGRADED", err_val or "Stale data exceeded limit.")
        except Exception:
            pass
    fired_pb = {k: v for k, v in rejected.items() if v > 0}
    stale_count = rejected.get("stale_data", 0)
    no_data_count = rejected.get("no_data", 0)
    fresh_count = len(fresh_valid_symbols)
    # [RULE 67] DEGRADED_FALLBACK takes precedence over stale-based DEGRADED in the summary display.
    # Matches the same precedence logic applied at the health write and EOD scanner.
    data_status = "OK"
    if is_historical_fallback:
        data_status = "DEGRADED_FALLBACK (Historical Fallback Dataset)"
    elif (stale_count / max(total_symbols, 1)) > 0.20:
        data_status = "DEGRADED (Stale Data > 20%)"

    # Ensure 100% mathematical conservation
    terminal_tracker.record_untracked_remainder("UNTRACKED_DROP")
    cons_summary = terminal_tracker.get_summary()

    waterfall.set_stage_count("UNIVERSE_WATCHLIST", total_symbols)
    waterfall.set_stage_count("FETCHED_DATA", fresh_count)
    waterfall.set_stage_count("UPTREND_AND_STRUCTURE", len(candidates))
    waterfall.set_stage_count("SCORE_THRESHOLD", len(scored_candidates))
    waterfall.set_stage_count("RISK_ENGINE", len(valid_risk_candidates))
    waterfall.set_stage_count("FINAL_ALERTS", alert_count)

    attrition_results = waterfall.compute_attrition()
    dominant_bottleneck = waterfall.get_dominant_bottleneck()

    classification_res = classify_zero_alert_run(
        scanner_name="PULLBACK",
        universe_size=total_symbols,
        valid_data_count=fresh_count,
        initial_setups_count=len(candidates),
        finalist_candidates_count=len(valid_risk_candidates),
        alerts_generated=alert_count,
        near_miss_count=near_miss_count,
        regime=market_regime,
        execution_mode="LIVE",
        stage_waterfall=attrition_results
    )

    summary_lines = [
        "======================================================================",
        "=== [PULLBACK SCANNER PIPELINE SUMMARY] ===",
        "======================================================================",
        "📊 DATA QUALITY SNAPSHOT:",
        f"  • Total Watchlist Requested : {total_symbols}",
        f"  • Provider Resolved Symbols : {total_fetched_count}",
        f"  • Fresh Valid Data OK       : {fresh_count}",
        f"  • Stale Data                : {stale_count}",
        f"  • Missing / No Data         : {no_data_count}",
        f"  • Data Health Status        : {data_status}",
        "",
        "🎯 CRITERIA & FILTER BREAKDOWN:"
    ]
    for k, v in fired_pb.items():
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

    summary_lines.extend([
        "",
        "🏆 FINAL OUTCOME:",
        f"  • Alerts Generated          : {alert_count}",
        f"  • Near Misses (<=5 pts)     : {near_miss_count}",
        f"  • Total Execution Time      : {elapsed_time}s",
    ])

    if alert_count == 0:
        b_stg = dominant_bottleneck.get('stage', '') if dominant_bottleneck else ''
        b_breakdown = terminal_tracker.get_stage_terminal_breakdown(b_stg) if b_stg else None

        diag_block = format_zero_alert_diagnostic_block(
            scanner_name="PULLBACK",
            execution_mode="LIVE",
            regime=market_regime,
            classification_result=classification_res,
            dominant_bottleneck=dominant_bottleneck,
            conservation_summary=cons_summary,
            stage_waterfall=attrition_results,
            near_miss_count=near_miss_count,
            extra_specs=[
                f"REQUIRED_SCORE_THRESHOLD   : {required_threshold}",
                f"MARKET_REGIME              : {market_regime}",
            ],
            bottleneck_terminal_breakdown=b_breakdown
        )
        summary_lines.extend(diag_block)

    summary_lines.append("======================================================================")
    logger.info("\n".join(summary_lines))
    telemetry_logger.print_summary()
    global_telemetry.print_system_summary()

    try:
        stage_tracker.end_stage(f"Alerts={alert_count} persisted")
        stage_tracker.print_summary(alerts_found=alert_count)
    except Exception:
        pass

    if not is_historical_fallback:
        logger.info(f"✅ [COMPLETE] PULLBACK SCANNER DONE | {elapsed_time:.2f}s | Alerts={alert_count} | Status={status_val}")
    else:
        logger.info(f"✅ [COMPLETE] PULLBACK SCANNER DONE (historical fallback) | {elapsed_time:.2f}s | Candidates={len(candidates)} | Dataset={dataset_date}")

    if not is_historical_fallback:
        try:
            from database import upload_history_bundle_to_db, submit_background_upload
            submit_background_upload(lambda: upload_history_bundle_to_db("1d"))
            logger.info("💾 [PULLBACK] Submitted background upload of 1d history bundle to Postgres DB.")
        except Exception as _up_err:
            logger.warning(f"⚠️ Failed to queue background DB bundle upload in Pullback: {_up_err}")

    # Execute Phase 2E Pullback V2 Pipeline in parallel isolation
    try:
        _run_pullback_v2_pipeline()
    except Exception as v2_err:
        logger.warning(f"⚠️ Phase 2E Pullback V2 pipeline execution warning: {v2_err}")

    return {
        "total_count": total_symbols,
        "processed_count": symbols_processed,
        "today_alerts": alert_count
    }


def _run_pullback_v2_pipeline():
    """
    Executes Phase 2E Pullback V2 pipeline in parallel isolation.
    """
    logger.info("[V2_PIPELINE] Starting Phase 2E Pullback V2 pipeline...")
    from pullback_schema import init_pullback_v2_schema
    from pullback_engine import evaluate_pullback_v2_symbol

    init_pullback_v2_schema()

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
    logger.info("[V2_PIPELINE] Phase 2E Pullback V2 pipeline evaluation ready.")

