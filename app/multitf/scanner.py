# =====================================================================================
# app/multitf/scanner.py
# MULTI_TF V2 — Main Execution Orchestrator
#
# Responsibility: Main entry point for the V2 scanner.
# Loops through the watchlist, delegates to engines, and pushes CONFIRMED signals
# directly to the global OpportunityManager.
# =====================================================================================

import os
import logging
import time
import concurrent.futures
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

from config import MULTI_TF_V2_CONFIG
from database import get_elite_watchlist, get_multitf_universe, save_alert_if_new
from opportunity_manager import OpportunityManager
from lock_utils import ProcessLock, set_scanner_fetch_active
from database import upsert_scanner_health
from price_cache import fetch_watchlist_data

from multitf.data import load_multitf_data, strip_closed_candles
from multitf.context import evaluate_1h_context, evaluate_30m_context, evaluate_market_context
from multitf.consolidation import detect_15m_consolidation, prepare_15m_context, Prepared15mContext
from multitf.pressure import evaluate_5m_pressure, compute_ignition_score
from multitf.confluence import evaluate_breakout_confluence
from multitf.breakout_strength import (
    compute_breakout_strength,
    classify_alert_severity,
    evaluate_trade_eligibility,
    SEVERITY_EMOJI,
    SEVERITY_LABEL
)
from multitf.alert_builder import build_multitf_alert_message
from multitf.state import (
    load_state,
    find_active_box_for_symbol,
    apply_ttl_and_cooldown,
    handle_box_invalidation,
    invalidate_record,
    persist_new_watchlist_candidate,
    update_state_in_db,
    get_active_armed_candidates,
    get_armed_candidate_lifecycle_summary,
    MtfSubstate
)
from multitf.candidate import build_watchlist_candidate, build_confirmed_payload
from zero_alert_diagnostic import classify_zero_alert_run, format_zero_alert_diagnostic_block

from sl_target_helper import compute_sl_and_target

logger = logging.getLogger("multitf.scanner")
_scan_lock = ProcessLock("multi_tf_scanner")
_scan_lock_5m = ProcessLock("multi_tf_5m_monitor")
_global_lock = ProcessLock("global_scanner_lock")

# [RULE 67 CHANGE-RATIONALE: GEOMETRY_FEATURE_CACHE_V1.0]
# In-memory geometry cache keyed by (symbol, last_closed_bar_timestamp, bar_count).
# Reuses expensive 15m 9-window geometry across scanner runs when no new candle closed,
# dropping Stage 2.5 latency from ~100s to <1s.
_GEOMETRY_CACHE: Dict[Tuple[str, str, int], Any] = {}


def _get_rss_mb() -> float:
    """Returns current process RSS memory in MB cross-platform (Linux & macOS)."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource, sys
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(usage) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(usage) / 1024.0
    except Exception:
        return 0.0


def _get_atr(df, default: float = 0.0) -> float:
    """Extracts ATR from DataFrame checking 'ATR_14', 'ATR', or 'ATR20', with rolling TrueRange fallback."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return default
    for col in ("ATR_14", "ATR", "ATR20"):
        if col in df.columns and len(df[col]) > 0:
            val = float(df[col].iloc[-1])
            if val > 0:
                return val
    # Fallback: compute last 14-bar True Range average if OHLC columns exist
    try:
        if all(c in df.columns for c in ("High", "Low", "Close")) and len(df) >= 2:
            prev_c = df["Close"].shift(1)
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - prev_c).abs(),
                (df["Low"] - prev_c).abs()
            ], axis=1).max(axis=1)
            atr_calc = float(tr.tail(14).mean())
            if atr_calc > 0:
                return atr_calc
    except Exception:
        pass
    return default


def run_multitf_v2(regime_ctx: Dict[str, Any], ist_now: datetime, run_ctx: str = "SCHEDULED"):
    """
    Main Primary Intelligence Layer for MULTI_TF V2 (15-Minute Cadence).
    1. Pre-fetches 1d and 15m closed bars across universe.
    2. Detects 15m consolidation setups.
    3. Lazy-fetches 1h, 30m, 5m ONLY for shortlisted/armed candidate stocks.
    4. Evaluates setups, targets, and dispatches breakout alerts.
    """
    if _scan_lock.locked():
        logger.warning("🛑 [DUPLICATE GUARD] MULTI_TF Scanner is ALREADY actively running in thread lock. Skipping duplicate trigger.")
        return {"status": "skipped", "reason": "already_running"}

    acquired_global = False
    acquired_scan = False
    start_time = time.monotonic()
    real_run_ctx = None

    try:
        logger.info("[MULTI_TF] Acquiring lock: multi_tf_scanner")
        if not _scan_lock.acquire(blocking=False):
            logger.warning("🛑 [MULTI_TF] Lock 'multi_tf_scanner' is held by another MULTI_TF instance. Skipping duplicate cycle.")
            try:
                from database import record_skipped_execution_run
                record_skipped_execution_run(scanner_name="MULTI_TF", trigger_type="SCHEDULED", scheduler_name="CRON", stop_reason="Lock multi_tf_scanner held (previous run active)")
            except Exception:
                pass
            return {"status": "skipped", "reason": "already_running"}
        acquired_scan = True

        # Acquire universal global scanner lock
        if not _global_lock.acquire(blocking=False, owner_scanner="MULTI_TF", operation="FULL_SCAN"):
            logger.info("⏳ [MULTI_TF] Global scanner lock busy — waiting in queue until active scanner finishes...")
            upsert_scanner_health("MULTI_TF", "QUEUED", error_msg="Waiting in queue for active scanner to release lock...")

            try:
                acquired_global = _global_lock.acquire(blocking=True, owner_scanner="MULTI_TF", operation="FULL_SCAN")
            except Exception as lock_err:
                logger.error(f"❌ [MULTI_TF] Error acquiring global lock: {lock_err}")
                acquired_global = False

            if not acquired_global:
                logger.error("❌ [MULTI_TF] Failed to acquire global scanner lock after queue wait.")
                upsert_scanner_health("MULTI_TF", "IDLE", error_msg="Lock acquisition timed out")
                return
        else:
            acquired_global = True

        logger.info("=" * 70)
        logger.info("📊 MULTI_TF V2 ENGINE | Starting 15m execution cycle (Lazy Fetch)...")
        logger.info("=" * 70)

        from telemetry_manager import telemetry
        from perf_utils import ScannerStageTracker

        telemetry.log_scheduler_event("MULTI_TF", "CYCLE_START")
        stage_tracker = ScannerStageTracker("MULTI_TF_V2")

        # Create proper DB execution run context
        trigger_type = run_ctx if isinstance(run_ctx, str) else "SCHEDULED"
        from database import start_scanner_execution_run, complete_scanner_execution_run
        try:
            real_run_ctx = start_scanner_execution_run(scanner_name="MULTI_TF", trigger_type=trigger_type, scheduler_name="CRON")
        except Exception as exc:
            if "actively running" in str(exc).lower():
                logger.info("🛑 [MULTI_TF] Scanner is ALREADY actively running. Skipping duplicate execution.")
                return 0
            logger.warning(f"⚠️ [MULTI_TF] Could not create run_ctx: {exc}")
            real_run_ctx = None

        # [RULE 67 CHANGE-RATIONALE]: Explicitly record scheduled_for in scanner_health so
        # the dashboard always displays accurate, true schedule timings without relying on background seeders.
        _MTF_SCHEDULE = "Every 15m Scan / 5m Monitor (09:30 - 15:30 IST)"
        stage_tracker.start_stage(1, "Load Watchlist", "Fetching elite watchlist symbols from DB")
        upsert_scanner_health(
            scanner_name="MULTI_TF",
            status="RUNNING",
            error_msg="Scan execution in progress...",
            scheduled_for=_MTF_SCHEDULE
        )

        watchlist = get_multitf_universe()
        if not watchlist:
            logger.warning("[MULTI_TF] Watchlist empty.")
            upsert_scanner_health(
                scanner_name="MULTI_TF",
                status="OK",
                outcome="SUCCESS",
                processed_count=0,
                duration_seconds=round(time.monotonic() - start_time, 2),
                scheduled_for=_MTF_SCHEDULE
            )
            stage_tracker.end_stage("Watchlist empty")
            telemetry.log_scheduler_event("MULTI_TF", "CYCLE_COMPLETE")
            if real_run_ctx:
                complete_scanner_execution_run(real_run_ctx)
            return

        stage_tracker.end_stage(f"Loaded {len(watchlist)} symbols")

        # Stage 2: Intelligence Layer: Universe Pre-fetch (1d and 15m)
        stage_tracker.start_stage(2, "Fetch Setup Data (1d, 15m)", "Pre-fetching 1d and 15m closed bars across universe")
        logger.info("[MULTI_TF] Pre-fetching setup timeframes (1d, 15m) for %d universe symbols...", len(watchlist))
        t_fetch_start = time.monotonic()

        set_scanner_fetch_active(True)
        try:
            # [RULE 67 CHANGE-RATIONALE: CONCURRENT_PREFETCH_v1.0] Fetch 1d and 15m in parallel instead of serially
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pre_exec:
                f_1d  = pre_exec.submit(fetch_watchlist_data, watchlist, "1y", "1d", "MULTI_TF", real_run_ctx)
                f_15m = pre_exec.submit(fetch_watchlist_data, watchlist, "15d", "15m", "MULTI_TF", real_run_ctx)
                all_1d  = f_1d.result()
                all_15m = f_15m.result()
        finally:
            set_scanner_fetch_active(False)

        # Stage 2.5: Fast 15m Consolidation Screening across universe (Adaptive V3 with Geometry Caching)
        # [RULE 67 CHANGE-RATIONALE: OPTIMIZED_STAGE_2_5_V1.1]
        # 1. Geometry fingerprint caching: reuses 9-window calculations if the last closed candle is unchanged.
        # 2. Zero DataFrame slicing inside candidate window loops; uses Prepared15mContext.
        # 3. Preserves all 9 adaptive candidate windows [6, 8, 10, 12, 16, 20, 24, 30, 35] without 35-bar veto.
        # 4. Strict conservation-of-universe accounting: universe = fast_rejected + deep_screened (0 lost symbols).
        # 5. Periodic progress logging every 50 symbols and per-stage timing / latency / memory RSS profiling.
        shortlisted_symbols = []
        consolidation_map = {}
        total_symbols = len(watchlist)
        min_bars_config = MULTI_TF_V2_CONFIG.get("MIN_CONSOLIDATION_BARS", 6)

        rss_before = _get_rss_mb()
        rss_peak = rss_before
        t_stage25_start = time.monotonic()

        t_ctx_prep_total = 0.0
        t_fast_filter_total = 0.0
        t_deep_screen_total = 0.0
        geometry_cache_hits = 0
        symbol_latencies_ms: List[float] = []

        fast_rejected_breakdown = {
            "NO_DATA": 0,
            "INSUFFICIENT_BARS": 0,
            "ATR_ZERO_OR_NEG": 0,
            "FLATLINE_ZERO_RANGE": 0,
        }

        deep_screened_breakdown = {
            "QUALIFIED": 0,
            "PRESSURE": 0,
            "PRE_BREAKOUT": 0,
            "STRONG": 0,
            "FORMING": 0,
            "WIDTH_EXCEEDED": 0,
            "SCORE_TOO_LOW": 0,
            "TESTS_TOO_LOW": 0,
            "DORMANT": 0,
            "GAP_BROKEN": 0,
            "OTHER_REJECT": 0,
        }

        # Execute Stage 2.5 screening concurrently across a worker pool with geometry caching.
        def _screen_symbol_worker(sym_idx: int, sym: str):
            t_s_start = time.perf_counter()
            df_raw = all_15m.get(sym)
            if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                return sym_idx, sym, "NO_DATA", None, (time.perf_counter() - t_s_start) * 1000.0, 0.0, (time.perf_counter() - t_s_start), 0.0, False

            t_f_start = time.perf_counter()
            df_c = strip_closed_candles(df_raw, 15, ist_now)
            if df_c is None or df_c.empty or len(df_c) < min_bars_config:
                return sym_idx, sym, "INSUFFICIENT_BARS", None, (time.perf_counter() - t_s_start) * 1000.0, 0.0, (time.perf_counter() - t_f_start), 0.0, False

            # [GEOMETRY_CACHE] Check if closed bar fingerprint is identical to previous calculation
            last_bar_ts = str(df_c.index[-1]) if hasattr(df_c.index, "values") else ""
            cache_key = (sym, last_bar_ts, len(df_c))
            cached_res = _GEOMETRY_CACHE.get(cache_key)
            if cached_res is not None:
                return sym_idx, sym, None, cached_res, (time.perf_counter() - t_s_start) * 1000.0, 0.0, 0.0, 0.0, True

            atr_val = _get_atr(df_c)
            if atr_val <= 0:
                return sym_idx, sym, "ATR_ZERO_OR_NEG", None, (time.perf_counter() - t_s_start) * 1000.0, 0.0, (time.perf_counter() - t_f_start), 0.0, False
            t_ff = time.perf_counter() - t_f_start

            t_c_start = time.perf_counter()
            c_ctx = prepare_15m_context(df_c, atr_val, MULTI_TF_V2_CONFIG, symbol=sym)
            t_cp = time.perf_counter() - t_c_start

            if c_ctx is None:
                return sym_idx, sym, "INSUFFICIENT_BARS", None, (time.perf_counter() - t_s_start) * 1000.0, t_cp, t_ff, 0.0, False

            if c_ctx.recent_high <= c_ctx.recent_low:
                return sym_idx, sym, "FLATLINE_ZERO_RANGE", None, (time.perf_counter() - t_s_start) * 1000.0, t_cp, t_ff, 0.0, False

            t_d_start = time.perf_counter()
            c_res = detect_15m_consolidation(
                df_c, atr_val, ist_now, MULTI_TF_V2_CONFIG, symbol=sym, precomputed_context=c_ctx
            )
            t_ds = time.perf_counter() - t_d_start
            if c_res is not None and c_res.is_valid:
                _GEOMETRY_CACHE[cache_key] = c_res
            return sym_idx, sym, None, c_res, (time.perf_counter() - t_s_start) * 1000.0, t_cp, t_ff, t_ds, False

        max_workers = min(8, max(2, (os.cpu_count() or 4)))
        results_by_idx = {}
        completed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_screen_symbol_worker, idx, symbol)
                for idx, symbol in enumerate(watchlist)
            ]
            for fut in concurrent.futures.as_completed(futures):
                idx_res, sym_res, rej_code, cons_res, lat_ms, t_cp, t_ff, t_ds, was_hit = fut.result()
                results_by_idx[idx_res] = (sym_res, rej_code, cons_res, lat_ms, t_cp, t_ff, t_ds, was_hit)
                completed_count += 1
                if real_run_ctx and completed_count % 20 == 0:
                    try:
                        real_run_ctx.heartbeat()
                    except Exception:
                        pass
                if completed_count % 50 == 0 or completed_count == total_symbols:
                    cur_rss = _get_rss_mb()
                    if cur_rss > rss_peak:
                        rss_peak = cur_rss
                    logger.info(
                        "[MULTI_TF][2.5] Screening progress: %d/%d symbols processed (concurrent pool)",
                        completed_count, total_symbols
                    )

        # Assemble in strict deterministic index order
        for idx in range(total_symbols):
            symbol, rej_code, cons, lat_ms, t_cp, t_ff, t_ds, was_hit = results_by_idx[idx]
            t_ctx_prep_total += t_cp
            t_fast_filter_total += t_ff
            t_deep_screen_total += t_ds
            if was_hit:
                geometry_cache_hits += 1
            symbol_latencies_ms.append(lat_ms)

            if rej_code:
                fast_rejected_breakdown[rej_code] += 1
                continue

            if cons and cons.is_valid:
                shortlisted_symbols.append(symbol)
                consolidation_map[symbol] = cons
                stage = getattr(cons, "lifecycle_stage", "FORMING")
                if stage in deep_screened_breakdown:
                    deep_screened_breakdown[stage] += 1
                else:
                    deep_screened_breakdown["QUALIFIED"] += 1
            elif cons:
                reason = cons.rejection_reason or ""
                if "GAP" in reason:
                    deep_screened_breakdown["GAP_BROKEN"] += 1
                elif "WIDTH" in reason or "OCCUPANCY" in reason:
                    deep_screened_breakdown["WIDTH_EXCEEDED"] += 1
                elif "SCORE" in reason:
                    deep_screened_breakdown["SCORE_TOO_LOW"] += 1
                elif "TEST" in reason:
                    deep_screened_breakdown["TESTS_TOO_LOW"] += 1
                elif cons.is_dormant:
                    deep_screened_breakdown["DORMANT"] += 1
                else:
                    deep_screened_breakdown["OTHER_REJECT"] += 1
            else:
                fast_rejected_breakdown["NO_DATA"] += 1

        t_stage25_total = time.monotonic() - t_stage25_start
        rss_after = _get_rss_mb()
        if rss_after > rss_peak:
            rss_peak = rss_after

        fast_rejected_count = sum(fast_rejected_breakdown.values())
        deep_screened_count = total_symbols - fast_rejected_count
        qualified_count = len(shortlisted_symbols)
        invalid_screened_count = deep_screened_count - qualified_count

        import numpy as _np
        if symbol_latencies_ms:
            lat_arr = _np.array(symbol_latencies_ms)
            p50_ms = float(_np.percentile(lat_arr, 50))
            p95_ms = float(_np.percentile(lat_arr, 95))
            max_ms = float(_np.max(lat_arr))
        else:
            p50_ms, p95_ms, max_ms = 0.0, 0.0, 0.0

        logger.info(
            f"\n[MULTI_TF][2.5] COMPLETE\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Universe                  : {total_symbols}\n"
            f"Fast rejected             : {fast_rejected_count}\n"
            f"  ├── No Data             : {fast_rejected_breakdown['NO_DATA']}\n"
            f"  ├── Insufficient Bars   : {fast_rejected_breakdown['INSUFFICIENT_BARS']}\n"
            f"  ├── ATR <= 0            : {fast_rejected_breakdown['ATR_ZERO_OR_NEG']}\n"
            f"  └── Flatline Zero Range : {fast_rejected_breakdown['FLATLINE_ZERO_RANGE']}\n"
            f"Deep screened             : {deep_screened_count} (valid={qualified_count}, invalid={invalid_screened_count})\n"
            f"  ├── Qualified Setups    : {qualified_count}\n"
            f"  │     ├── PRESSURE      : {deep_screened_breakdown['PRESSURE']}\n"
            f"  │     ├── PRE-BREAKOUT  : {deep_screened_breakdown['PRE_BREAKOUT']}\n"
            f"  │     ├── STRONG        : {deep_screened_breakdown['STRONG']}\n"
            f"  │     └── FORMING       : {deep_screened_breakdown['FORMING']}\n"
            f"  └── Rejections (Deep)   :\n"
            f"        ├── Width Exceeded: {deep_screened_breakdown['WIDTH_EXCEEDED']}\n"
            f"        ├── Score Too Low : {deep_screened_breakdown['SCORE_TOO_LOW']}\n"
            f"        ├── Tests Too Low : {deep_screened_breakdown['TESTS_TOO_LOW']}\n"
            f"        ├── Dormant Vol   : {deep_screened_breakdown['DORMANT']}\n"
            f"        ├── Gap Broken    : {deep_screened_breakdown['GAP_BROKEN']}\n"
            f"        └── Other Reject  : {deep_screened_breakdown['OTHER_REJECT']}\n"
            f"\n"
            f"Conservation Accounting   : {fast_rejected_count} + {deep_screened_count} = {total_symbols} (delta={total_symbols - (fast_rejected_count + deep_screened_count)})\n"
            f"Geometry Cache Hits       : {geometry_cache_hits}/{total_symbols}\n"
            f"\n"
            f"Timing\n"
            f"  Context preparation     : {t_ctx_prep_total:.2f}s\n"
            f"  Fast funnel             : {t_fast_filter_total:.2f}s\n"
            f"  Deep geometry & scoring : {t_deep_screen_total:.2f}s\n"
            f"  Total Stage 2.5         : {t_stage25_total:.2f}s\n"
            f"\n"
            f"Per-symbol Latency\n"
            f"  p50                     : {p50_ms:.1f}ms\n"
            f"  p95                     : {p95_ms:.1f}ms\n"
            f"  max                     : {max_ms:.1f}ms\n"
            f"\n"
            f"Memory\n"
            f"  RSS before              : {rss_before:.1f}MB\n"
            f"  RSS peak                : {rss_peak:.1f}MB\n"
            f"  RSS after               : {rss_after:.1f}MB\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        if real_run_ctx:
            try:
                real_run_ctx.heartbeat(force=True)
            except Exception:
                pass

        # Also include any previously ARMED candidates from DB to ensure active setups continue tracking
        active_armed = get_active_armed_candidates()
        db_armed_symbols = set()
        for cand in active_armed:
            sym = cand.get("symbol")
            if sym:
                db_armed_symbols.add(sym)
                if sym not in shortlisted_symbols:
                    shortlisted_symbols.append(sym)

        # [RULE 67 CHANGE-RATIONALE: TIERED_CANDIDATE_GATING_V2.0]
        # Preserves core setup threshold (score >= 60, PRESSURE, PRE_BREAKOUT, STRONG, db_armed) without altering strategy recall.
        # Introduces a zero-cost Tier 2 pre-fetch gate using already-loaded 15m/1d geometry:
        # 1. Tier 1: All structural candidates meeting score >= 60 or active stages (~160-180 symbols).
        # 2. Tier 2: Immediately actionable candidates (~35-60 symbols) that are within striking distance
        #    of the base breakout ceiling (dist_to_ceiling <= 6.0% or upper 40% of consolidation box), or DB-armed,
        #    or in active urgency stages (PRESSURE, PRE_BREAKOUT, STRONG).
        # Forming bases deep at the floor (>6% below ceiling) are safely preserved in DB/state without wasting network time.
        tier1_candidates = []
        for sym in shortlisted_symbols:
            if sym in db_armed_symbols:
                tier1_candidates.append(sym)
                continue
            cons = consolidation_map.get(sym)
            if cons is not None:
                stage = getattr(cons, "lifecycle_stage", "FORMING")
                score = getattr(cons, "setup_score", 0)
                if stage in ("PRESSURE", "PRE_BREAKOUT", "STRONG") or score >= 60:
                    tier1_candidates.append(sym)

        tier1_candidates = list(dict.fromkeys(tier1_candidates))

        actionable_symbols = []
        tier2_deferred_count = 0
        for sym in tier1_candidates:
            if sym in db_armed_symbols:
                actionable_symbols.append(sym)
                continue
            cons = consolidation_map.get(sym)
            if cons is not None:
                stage = getattr(cons, "lifecycle_stage", "FORMING")
                if stage in ("PRESSURE", "PRE_BREAKOUT", "STRONG"):
                    actionable_symbols.append(sym)
                    continue

                box_high = getattr(cons, "box_high", 0.0)
                box_low = getattr(cons, "box_low", 0.0)
                df_c = all_15m.get(sym)
                last_close = 0.0
                if df_c is not None and not df_c.empty:
                    # RULE 67 RATIONALE: Support both 'Close' and 'close' column naming across data providers/DataFrames
                    c_col = 'Close' if 'Close' in df_c.columns else ('close' if 'close' in df_c.columns else None)
                    if c_col:
                        last_close = float(df_c[c_col].iloc[-1])

                is_near_ceiling = False
                if box_high > 0 and last_close > 0:
                    dist_to_ceiling_pct = (box_high - last_close) / last_close * 100.0
                    box_range = box_high - box_low
                    pos_in_box = (last_close - box_low) / box_range if box_range > 0 else 0.5
                    if dist_to_ceiling_pct <= 6.0 or pos_in_box >= 0.40:
                        is_near_ceiling = True

                if is_near_ceiling:
                    actionable_symbols.append(sym)
                else:
                    tier2_deferred_count += 1

        actionable_symbols = list(dict.fromkeys(actionable_symbols))

        all_1h = {}
        all_30m = {}
        all_5m = {}
        if actionable_symbols:
            if real_run_ctx:
                try:
                    real_run_ctx.heartbeat(force=True)
                except Exception:
                    pass
            logger.info(
                f"⚡ [MULTI_TF] Lazy-fetching (1h, 30m, 5m) concurrently for {len(actionable_symbols)} actionable candidates "
                f"(Tier 1: {len(tier1_candidates)}, Tier 2 Actionable: {len(actionable_symbols)}, Deferred: {tier2_deferred_count})..."
            )
            set_scanner_fetch_active(True)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as tf_executor:
                    f_1h  = tf_executor.submit(fetch_watchlist_data, actionable_symbols, "45d", "1h", "MULTI_TF", real_run_ctx)
                    f_30m = tf_executor.submit(fetch_watchlist_data, actionable_symbols, "20d", "30m", "MULTI_TF", real_run_ctx)
                    f_5m  = tf_executor.submit(fetch_watchlist_data, actionable_symbols, "5d",  "5m",  "MULTI_TF", real_run_ctx)

                    all_1h  = f_1h.result()
                    all_30m = f_30m.result()
                    all_5m  = f_5m.result()
            finally:
                set_scanner_fetch_active(False)


        t_fetch_dur = round(time.monotonic() - t_fetch_start, 2)
        logger.info("⚡ [MULTI_TF] Completed market data pre-fetch in %ss", t_fetch_dur)
        stage_tracker.end_stage(f"Fetched data in {t_fetch_dur}s")

        # Stage 3: Process Symbols
        stage_tracker.start_stage(3, "Process Symbols", "Evaluating compression and breakout models per symbol")
        logger.info("[MULTI_TF] Analyzing breakout signals for actionable symbols...")
        t_process_start = time.monotonic()
        opp_manager = OpportunityManager(policy=regime_ctx.get("policy", {}) if regime_ctx else {})

        # [RULE 67 CHANGE-RATIONALE]: Downstream rejection funnel tracking across all candidate evaluations
        # [RULE 67 CHANGE-RATIONALE: CONCURRENT_SYMBOL_EVALUATION_v1.0]
        # Parallelize symbol evaluation using ThreadPoolExecutor across available CPU cores.
        # Eliminates 250s+ single-threaded bottleneck down to <35s while safely updating DB and funnel.
        from collections import defaultdict
        import threading
        mtf_funnel = defaultdict(int)
        _eval_lock = threading.Lock()
        completed_evals = 0

        target_evaluation_symbols = actionable_symbols if actionable_symbols else []

        def _eval_symbol_worker(symbol):
            nonlocal completed_evals
            local_funnel = defaultdict(int)
            try:
                _process_symbol(
                    symbol=symbol,
                    ist_now=ist_now,
                    regime_ctx=regime_ctx,
                    opp_manager=opp_manager,
                    all_1d=all_1d,
                    all_1h=all_1h,
                    all_30m=all_30m,
                    all_15m=all_15m,
                    all_5m=all_5m,
                    config=MULTI_TF_V2_CONFIG,
                    precomputed_consolidation=consolidation_map.get(symbol),
                    funnel_counters=local_funnel
                )
            except Exception as loop_exc:
                logger.error("[MULTI_TF] Failed processing %s: %s", symbol, loop_exc)
                local_funnel["evaluation_exception"] += 1

            with _eval_lock:
                for k, v in local_funnel.items():
                    mtf_funnel[k] += v
                completed_evals += 1
                if real_run_ctx and completed_evals % 20 == 0:
                    try:
                        if hasattr(real_run_ctx, "heartbeat"):
                            real_run_ctx.heartbeat()
                    except Exception:
                        pass

        eval_workers = min(8, max(2, (os.cpu_count() or 4)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as eval_exec:
            list(eval_exec.map(_eval_symbol_worker, target_evaluation_symbols))

        t_process_dur = round(time.monotonic() - t_process_start, 2)
        logger.info("⚡ [MULTI_TF] Completed symbol evaluations in %ss", t_process_dur)
        stage_tracker.end_stage(f"Processed symbols in {t_process_dur}s")

        # Stage 4: Dispatch Opportunities
        stage_tracker.start_stage(4, "Dispatch Opportunities", "Filtering and executing OpportunityManager alerts")
        t_opp_start = time.monotonic()
        try:
            opp_manager.process()
        except Exception as e:
            logger.error("[MULTI_TF] OpportunityManager failed to process: %s", e)
        t_opp_dur = round(time.monotonic() - t_opp_start, 2)
        stage_tracker.end_stage(f"Dispatched in {t_opp_dur}s")

        duration = round(time.monotonic() - start_time, 2)

        from market_utils import is_market_open
        _is_market = is_market_open(ist_now)
        _exec_mode = "LIVE_ENTRY" if _is_market else "PREARM"
        _scan_log_suffix = (
            "Full execution cycle complete — live entry active"
            if _is_market
            else "Non-market / weekend scan — armed candidates & adjusted evaluation active"
        )

        alerts_generated = mtf_funnel.get("alert_triggered", 0)

        # Health status is strictly operational health (OK / DOWN / DEGRADED);
        # execution mode (LIVE_ENTRY vs PREARM) is captured in structured outcome.
        upsert_scanner_health(
            scanner_name="MULTI_TF",
            status="OK",
            outcome={"status": "SUCCESS", "mode": _exec_mode, "alerts": alerts_generated},
            today_alerts=alerts_generated,
            processed_count=len(watchlist),
            total_count=len(watchlist),
            duration_seconds=duration,
            scheduled_for=_MTF_SCHEDULE
        )

        lifecycle = get_armed_candidate_lifecycle_summary()

        fired_mtf = {k: v for k, v in sorted(mtf_funnel.items(), key=lambda x: x[1], reverse=True) if v > 0}
        summary_mtf_lines = [
            "======================================================================",
            "=== [MULTI_TF PIPELINE SUMMARY] ===",
            "======================================================================",
            f"  • Total Watchlist Requested : {len(watchlist)}",
            f"  • Actionable Evaluated      : {len(target_evaluation_symbols)}",
            f"  • Alerts Generated          : {alerts_generated}",
            f"  • Execution Mode            : {_exec_mode}",
            f"  • Alert Generation Active   : YES (All-Day Enabled)",
            "",
            "📦 ARMED CANDIDATE LIFECYCLE (OVERNIGHT PERSISTENCE):",
            f"  • Total in Watchlist Table  : {lifecycle['total_in_watchlist']}",
            f"  • Active Substates          : {lifecycle['active_substates']} (WATCHING / PRESSURE / ATTEMPT)",
            f"  • In Cooldown / TTL Expired : {lifecycle['in_cooldown']}",
            f"  • Invalidated Boxes         : {lifecycle['invalidated']}",
            f"  • Live Monitor Eligible     : {lifecycle['live_monitor_eligible']}",
            "",
            "🎯 GATE-BY-GATE REJECTION BREAKDOWN:"
        ]
        for k, v in fired_mtf.items():
            summary_mtf_lines.append(f"  • {k:<30}: {v}")

        if alerts_generated == 0:
            classification = classify_zero_alert_run(
                scanner_name="MULTI_TF",
                universe_size=len(watchlist),
                valid_data_count=len(watchlist),
                initial_setups_count=len(target_evaluation_symbols),
                finalist_candidates_count=mtf_funnel.get("confluence_approved", 0),
                alerts_generated=alerts_generated,
                near_miss_count=lifecycle.get("live_monitor_eligible", 0),
                regime="NEUTRAL",
                execution_mode=_exec_mode,
                persistence_failures_count=mtf_funnel.get("persistence_failed", 0),
                candidates_persisted_count=lifecycle.get("total_in_watchlist"),
                lifecycle_summary=lifecycle
            )
            dominant_mtf = next(iter(fired_mtf.items())) if fired_mtf else ("None", 0)
            summary_mtf_lines.extend([
                "",
                "⚠️ ZERO_ALERT_DIAGNOSTIC (MULTI_TF):",
                f"  • Classification            : {classification['classification']} [{classification['severity']}]",
                f"  • Finding                   : {classification['explanation']}",
                f"  • Execution Mode            : {_exec_mode}",
                f"  • Dominant Rejection Gate   : {dominant_mtf[0]} ({dominant_mtf[1]} occurrences)",
                f"  • Recommendation            : {classification['recommendation']}"
            ])

        summary_mtf_lines.append("======================================================================")
        logger.info("\n".join(summary_mtf_lines))

        if real_run_ctx:
            try:
                real_run_ctx.set_total_stocks(len(watchlist))
                real_run_ctx.record_fresh_data(len(watchlist))
                complete_scanner_execution_run(real_run_ctx, status_override="COMPLETED")
            except Exception as _c_err:
                logger.warning(f"⚠️ [MULTI_TF] Failed to complete execution run: {_c_err}")

        telemetry.log_scheduler_event("MULTI_TF", "CYCLE_COMPLETE")
        logger.info("✅ MULTI_TF V2 ENGINE | Execution cycle complete in %ss. %s", duration, _scan_log_suffix)

        # Telemetry distinguishing SCAN_SUCCESS and CACHE_PERSISTED vs CACHE_PERSIST_PENDING
        try:
            from database import get_interval_generation
            g_15m, up_15m = get_interval_generation("15m")
            g_1d, up_1d = get_interval_generation("1d")
            status_15m = f"15m:gen={g_15m}/up={up_15m}(" + ("PENDING" if up_15m < g_15m else "PERSISTED") + ")"
            status_1d = f"1d:gen={g_1d}/up={up_1d}(" + ("PENDING" if up_1d < g_1d else "PERSISTED") + ")"
            logger.info(f"💾 [MULTI_TF PERSISTENCE TELEMETRY] SCAN_SUCCESS | {status_15m} | {status_1d}")
        except Exception:
            pass

        # Background sync updated history bundles to PostgreSQL parquet_cache so restarts never re-fetch
        try:
            from database import upload_history_bundle_to_db, submit_background_upload
            submit_background_upload(lambda: upload_history_bundle_to_db("15m", min_interval_sec=300.0))
            submit_background_upload(lambda: upload_history_bundle_to_db("1d", min_interval_sec=300.0))
        except Exception as _sync_err:
            logger.debug(f"[MULTI_TF] History bundle background upload dispatch error: {_sync_err}")

        return {
            "total_count": len(watchlist),
            "processed_count": len(target_evaluation_symbols),
            "today_alerts": alerts_generated
        }

    except Exception as exc:
        duration = round(time.monotonic() - start_time, 2)
        logger.error("[MULTI_TF] Fatal error during cycle: %s", exc)
        upsert_scanner_health(
            scanner_name="MULTI_TF",
            status="DOWN",
            outcome="FAILED",
            error_msg=str(exc),
            duration_seconds=duration,
            scheduled_for=_MTF_SCHEDULE
        )
        if real_run_ctx:
            try:
                complete_scanner_execution_run(real_run_ctx, exception=exc)
            except Exception:
                pass
        telemetry.log_scheduler_event("MULTI_TF", "CYCLE_FAILED", error=str(exc))
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


def run_multitf_5m_monitor(regime_ctx: Optional[Dict[str, Any]] = None, ist_now: Optional[datetime] = None, run_ctx: Any = None):
    """
    Secondary Confirmation Layer: Runs every 5 minutes on closed 5m candles.
    Only checks currently ARMED candidates from mtf_v2_watchlist.
    Takes < 3 seconds to confirm 5m pressure/expansion and trigger alerts.
    """
    if ist_now is None:
        ist_now = datetime.now(ZoneInfo("Asia/Kolkata") if "ZoneInfo" in globals() else None)
    if regime_ctx is None:
        regime_ctx = {"status": "NORMAL"}

    trigger_type = run_ctx if isinstance(run_ctx, str) else "SCHEDULED"
    from database import start_scanner_execution_run, complete_scanner_execution_run, upsert_scanner_health
    _MTF_5M_SCHEDULE = "Every 5min Monitor (09:35 - 15:25 IST)"

    try:
        real_run_ctx = start_scanner_execution_run(scanner_name="MULTI_TF_5M", trigger_type=trigger_type, scheduler_name="CRON")
    except Exception as exc:
        if "actively running" in str(exc).lower():
            logger.info("🛑 [MULTI_TF_5M] Scanner is ALREADY actively running. Skipping duplicate execution.")
            return 0
        real_run_ctx = None

    active_candidates = get_active_armed_candidates()
    lifecycle_5m = get_armed_candidate_lifecycle_summary()
    logger.info(
        f"📦 [MULTI_TF_5M ARMED LIFECYCLE SNAPSHOT] "
        f"Watchlist Total: {lifecycle_5m['total_in_watchlist']} | "
        f"Active Substates: {lifecycle_5m['active_substates']} | "
        f"In Cooldown: {lifecycle_5m['in_cooldown']} | "
        f"Invalidated: {lifecycle_5m['invalidated']} | "
        f"Live Monitor Eligible: {lifecycle_5m['live_monitor_eligible']}"
    )

    if not active_candidates:
        classification_0 = classify_zero_alert_run(
            scanner_name="MULTI_TF_5M",
            universe_size=0,
            valid_data_count=0,
            initial_setups_count=0,
            finalist_candidates_count=0,
            alerts_generated=0,
            near_miss_count=0,
            regime="NEUTRAL",
            execution_mode="MONITOR",
            lifecycle_summary=lifecycle_5m
        )
        if classification_0["classification"] == "CRITICAL_ZERO":
            logger.error(
                f"🚨 [MULTI_TF_5M] LIFECYCLE MISMATCH: {classification_0['explanation']} "
                f"Recommendation: {classification_0['recommendation']}"
            )
            upsert_scanner_health(
                scanner_name="MULTI_TF_5M",
                status="DEGRADED",
                outcome={"status": "FAILURE", "mode": "MONITOR", "reason": "lifecycle_load_mismatch", "error": classification_0["explanation"]},
                processed_count=0,
                total_count=lifecycle_5m.get("live_monitor_eligible", 0),
                duration_seconds=0.05,
                scheduled_for=_MTF_5M_SCHEDULE,
                error_msg=classification_0["explanation"]
            )
            if real_run_ctx:
                try:
                    real_run_ctx.set_total_stocks(lifecycle_5m.get("live_monitor_eligible", 0))
                    complete_scanner_execution_run(real_run_ctx, status_override="FAILED", stop_reason=classification_0["explanation"])
                except Exception:
                    pass
            return 0
        else:
            logger.debug("[MULTI_TF_5M] No active armed candidates to monitor.")
            upsert_scanner_health(
                scanner_name="MULTI_TF_5M",
                status="OK",
                outcome={"status": "SUCCESS", "mode": "MONITOR", "alerts": 0},
                processed_count=0,
                total_count=0,
                duration_seconds=0.05,
                scheduled_for=_MTF_5M_SCHEDULE
            )
            if real_run_ctx:
                try:
                    real_run_ctx.set_total_stocks(0)
                    complete_scanner_execution_run(real_run_ctx, status_override="COMPLETED", stop_reason="No armed candidates to monitor")
                except Exception:
                    pass
            return 0

    logger.info("[MULTI_TF_5M] Acquiring lock: multi_tf_5m_monitor")
    if not _scan_lock_5m.acquire(blocking=False):
        logger.warning("🛑 [MULTI_TF_5M] Lock 'multi_tf_5m_monitor' is held by another MULTI_TF_5M instance. Skipping duplicate monitor cycle.")
        upsert_scanner_health(
            scanner_name="MULTI_TF_5M",
            status="OK",
            outcome={"status": "SKIPPED", "mode": "MONITOR", "reason": "lock_busy"},
            processed_count=0,
            total_count=len(active_candidates),
            duration_seconds=0.05,
            scheduled_for=_MTF_5M_SCHEDULE,
            error_msg="Lock multi_tf_5m_monitor held (previous monitor cycle running)"
        )
        if real_run_ctx:
            try:
                real_run_ctx.set_total_stocks(len(active_candidates))
                complete_scanner_execution_run(real_run_ctx, status_override="SKIPPED_DUPLICATE", stop_reason="Lock multi_tf_5m_monitor held (previous monitor cycle running)")
            except Exception:
                pass
        return 0

    start_time = time.monotonic()
    try:
        symbols = list({c["symbol"] for c in active_candidates if c.get("symbol")})
        logger.info(f"⚡ [MULTI_TF_5M] Monitoring {len(symbols)} ARMED candidates for 5m breakout: {symbols}")

        # [RULE 67 CHANGE-RATIONALE: CONCURRENT_TIMEFRAME_FETCH_v1.0]
        # Concurrently fetch all 5 timeframes in parallel instead of 5 serial blocking calls.
        set_scanner_fetch_active(True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                fut_1d  = executor.submit(fetch_watchlist_data, symbols, "1y",  "1d",  "MULTI_TF_5M", real_run_ctx)
                fut_1h  = executor.submit(fetch_watchlist_data, symbols, "45d", "1h",  "MULTI_TF_5M", real_run_ctx)
                fut_30m = executor.submit(fetch_watchlist_data, symbols, "20d", "30m", "MULTI_TF_5M", real_run_ctx)
                fut_15m = executor.submit(fetch_watchlist_data, symbols, "15d", "15m", "MULTI_TF_5M", real_run_ctx)
                fut_5m  = executor.submit(fetch_watchlist_data, symbols, "5d",  "5m",  "MULTI_TF_5M", real_run_ctx)

                all_1d  = fut_1d.result()
                all_1h  = fut_1h.result()
                all_30m = fut_30m.result()
                all_15m = fut_15m.result()
                all_5m  = fut_5m.result()
        finally:
            set_scanner_fetch_active(False)

        opp_manager = OpportunityManager(policy=regime_ctx.get("policy", {}) if regime_ctx else {})
        from collections import defaultdict
        import threading
        mtf_5m_funnel = defaultdict(int)
        _eval_lock_5m = threading.Lock()

        # [RULE 67 CHANGE-RATIONALE: CONCURRENT_5M_EVALUATION_v1.0]
        # Parallelize 5m candidate evaluation across available CPU cores
        def _eval_5m_worker(sym):
            local_funnel = defaultdict(int)
            try:
                _process_symbol(
                    symbol=sym,
                    ist_now=ist_now,
                    regime_ctx=regime_ctx,
                    opp_manager=opp_manager,
                    all_1d=all_1d,
                    all_1h=all_1h,
                    all_30m=all_30m,
                    all_15m=all_15m,
                    all_5m=all_5m,
                    config=MULTI_TF_V2_CONFIG,
                    funnel_counters=local_funnel
                )
            except Exception as e:
                logger.error(f"[MULTI_TF_5M] Error evaluating {sym}: {e}")
                local_funnel["evaluation_exception"] += 1

            with _eval_lock_5m:
                for k, v in local_funnel.items():
                    mtf_5m_funnel[k] += v

        eval_workers = min(8, max(2, (os.cpu_count() or 4)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as eval_exec:
            list(eval_exec.map(_eval_5m_worker, symbols))

        opp_manager.process()
        duration = round(time.monotonic() - start_time, 2)
        alerts_generated = mtf_5m_funnel.get("alert_triggered", 0)

        upsert_scanner_health(
            scanner_name="MULTI_TF_5M",
            status="OK",
            outcome="SUCCESS",
            today_alerts=alerts_generated,
            processed_count=len(symbols),
            total_count=len(symbols),
            duration_seconds=duration,
            scheduled_for=_MTF_5M_SCHEDULE
        )

        fired_5m = {k: v for k, v in sorted(mtf_5m_funnel.items(), key=lambda x: x[1], reverse=True) if v > 0}
        summary_5m_lines = [
            "======================================================================",
            "=== [MULTI_TF 5M MONITOR SUMMARY] ===",
            "======================================================================",
            f"  • Armed Candidates Monitored: {len(symbols)}",
            f"  • Alerts Generated          : {alerts_generated}",
            f"  • Duration                  : {duration}s",
            "",
            "🎯 GATE-BY-GATE BREAKDOWN:"
        ]
        for k, v in fired_5m.items():
            summary_5m_lines.append(f"  • {k:<30}: {v}")

        if alerts_generated == 0:
            classification_5m = classify_zero_alert_run(
                scanner_name="MULTI_TF_5M",
                universe_size=len(symbols),
                valid_data_count=len(symbols),
                initial_setups_count=len(symbols),
                finalist_candidates_count=mtf_5m_funnel.get("5m_confluence_passed", 0),
                alerts_generated=alerts_generated,
                near_miss_count=len(symbols),
                regime="NEUTRAL",
                execution_mode="MONITOR",
                persistence_failures_count=mtf_5m_funnel.get("persistence_failed", 0),
                candidates_persisted_count=lifecycle_5m.get("total_in_watchlist"),
                lifecycle_summary=lifecycle_5m
            )
            dominant_5m = next(iter(fired_5m.items())) if fired_5m else ("None", 0)
            summary_5m_lines.extend([
                "",
                "⚠️ ZERO_ALERT_DIAGNOSTIC (MULTI_TF_5M):",
                f"  • Classification            : {classification_5m['classification']} [{classification_5m['severity']}]",
                f"  • Finding                   : {classification_5m['explanation']}",
                f"  • Dominant Rejection Gate   : {dominant_5m[0]} ({dominant_5m[1]} occurrences)",
                f"  • Recommendation            : {classification_5m['recommendation']}"
            ])

        summary_5m_lines.append("======================================================================")
        logger.info("\n".join(summary_5m_lines))

        if real_run_ctx:
            try:
                real_run_ctx.set_total_stocks(len(symbols))
                real_run_ctx.record_fresh_data(len(symbols))
                complete_scanner_execution_run(real_run_ctx, status_override="COMPLETED")
            except Exception:
                pass
        logger.info(f"✅ [MULTI_TF_5M] 5m monitor cycle complete in {duration}s for {len(symbols)} candidates. Alerts: {alerts_generated}")
    except Exception as exc:
        duration = round(time.monotonic() - start_time, 2)
        logger.error(f"[MULTI_TF_5M] Error during 5m monitor: {exc}")
        upsert_scanner_health(
            scanner_name="MULTI_TF_5M",
            status="DOWN",
            outcome="FAILED",
            error_msg=str(exc),
            duration_seconds=duration,
            scheduled_for=_MTF_5M_SCHEDULE
        )
        if real_run_ctx:
            try:
                complete_scanner_execution_run(real_run_ctx, exception=exc)
            except Exception:
                pass
    finally:
        _scan_lock_5m.release()


def _process_symbol(
    symbol: str,
    ist_now: datetime,
    regime_ctx: Dict[str, Any],
    opp_manager: OpportunityManager,
    all_1d: Dict,
    all_1h: Dict,
    all_30m: Dict,
    all_15m: Dict,
    all_5m: Dict,
    config: Dict[str, Any],
    precomputed_consolidation=None,
    funnel_counters: Optional[Dict[str, int]] = None
):
    # 1. Load Data
    bundle = load_multitf_data(symbol, ist_now, all_1h, all_30m, all_15m, all_5m, all_1d)
    if not bundle.data_sufficient:
        if funnel_counters is not None:
            funnel_counters["DATA_INSUFFICIENT"] += 1
        return

    # Extract indicators
    atr_15m = _get_atr(bundle.df_15m_closed)
    atr_5m = _get_atr(bundle.df_5m_closed)
    if atr_15m <= 0 or atr_5m <= 0:
        if funnel_counters is not None:
            funnel_counters["INVALID_ATR"] += 1
        return

    c_col_5m = 'Close' if 'Close' in bundle.df_5m_closed.columns else ('close' if 'close' in bundle.df_5m_closed.columns else bundle.df_5m_closed.columns[0])
    current_price = float(bundle.df_5m_closed[c_col_5m].iloc[-1])

    # 2. Setup Detection (15m strictly closed)
    # [FIX: CONSOLIDATION_MAP_REUSE_v1.0] Use pre-computed result from pre-screen if available.
    # Armed-only symbols (pulled from DB) may not have a pre-computed consolidation.
    # For those, re-detect normally. If still invalid, exit — the DB record stays as-is.
    if precomputed_consolidation is not None and precomputed_consolidation.is_valid:
        consolidation = precomputed_consolidation
    else:
        consolidation = detect_15m_consolidation(bundle.df_15m_closed, atr_15m, ist_now, config)
    if not consolidation.is_valid:
        if funnel_counters is not None:
            if "TESTS_TOO_LOW" in getattr(consolidation, "rejection_reason", ""):
                funnel_counters["15M_REJECT_RESISTANCE"] += 1
            else:
                funnel_counters["15M_REJECT_BASE"] += 1
        return

    # 3. State Management & Stable Box Lineage
    active_record = find_active_box_for_symbol(symbol, consolidation.box_high, atr_15m)
    if active_record:
        # Re-use the existing box_id so the structure evolves without duplicate records
        consolidation.box_id = active_record.box_id
        state_record = active_record
        is_new = False
    else:
        state_record = load_state(symbol, consolidation.box_id)
        is_new = (state_record is None)

    # 4. Context Evaluation (lazy, only needed if valid setup exists)
    ctx_1h = evaluate_1h_context(bundle.df_1h, config)
    ctx_30m = evaluate_30m_context(bundle.df_30m, consolidation.box_high, config)
    market_ctx = evaluate_market_context(regime_ctx, symbol, bundle.df_5m_closed)

    if is_new:
        # First time seeing this box
        cand_dict = build_watchlist_candidate(bundle, consolidation, ctx_1h, ctx_30m, market_ctx, ist_now)
        try:
            persist_new_watchlist_candidate(cand_dict)
            logger.info(
                f"👁️ [MULTI_TF: CONSOLIDATION WATCH] {symbol} added to Watchlist (Setup Score: {consolidation.setup_score:.1f}/100) | "
                f"CMP: ₹{current_price:.2f} | Box High: ₹{consolidation.box_high:.2f} | Box Low: ₹{consolidation.box_low:.2f} | "
                f"Width: {consolidation.box_width_pct:.1f}% — (Pending 5M breakout trigger, not an active trade yet)"
            )
        except Exception as _p_err:
            logger.error("[MULTI_TF] Failed persisting candidate for %s: %s", symbol, _p_err)
            if funnel_counters is not None:
                funnel_counters["persistence_failed"] += 1
        state_record = load_state(symbol, consolidation.box_id) # Reload to get initialized record
        if not state_record:
            if funnel_counters is not None:
                funnel_counters["persistence_failed"] += 1
            return

    # [FIX: EARLY_EXIT_STAMP_v1.0]
    # Helper: builds the live-data dict so every early exit also refreshes box/score columns.
    def _live_data_updates():
        prov_1h  = bundle.prov_1h.to_dict()  if bundle.prov_1h  else {}
        prov_30m = bundle.prov_30m.to_dict() if bundle.prov_30m else {}
        prov_15m = bundle.prov_15m.to_dict() if bundle.prov_15m else {}
        prov_5m  = bundle.prov_5m.to_dict()  if bundle.prov_5m  else {}
        return {
            "box_high": consolidation.box_high, "box_low": consolidation.box_low,
            "box_mid": consolidation.box_mid, "box_value_center": consolidation.box_value_center,
            "hard_high": consolidation.hard_high, "hard_low": consolidation.hard_low,
            "box_width_pct": consolidation.box_width_pct, "box_width_atr": consolidation.box_width_atr,
            "box_occupancy": consolidation.box_occupancy,
            "consolidation_bars": consolidation.bars_count,
            "consolidation_sessions": consolidation.sessions_count,
            "consolidation_end_ts": consolidation.end_ts,
            "resistance_test_count": consolidation.resistance_test_count,
            "higher_low_score": consolidation.score_hl,
            "compression_score": consolidation.score_compression,
            "setup_score": consolidation.setup_score,
            "last_confirmed_pivot_level": consolidation.last_confirmed_pivot_level,
            "last_confirmed_pivot_ts": consolidation.last_confirmed_pivot_ts,
            "live_position_5m": round(current_price, 4) if current_price else None,
            "distance_to_box_high": round(consolidation.box_high - current_price, 4) if current_price else None,
            "data_source_1h": prov_1h.get("source", ""), "data_source_30m": prov_30m.get("source", ""),
            "data_source_15m": prov_15m.get("source", ""), "data_source_5m": prov_5m.get("source", ""),
            "candle_ts_1h": prov_1h.get("last_candle_ts"), "candle_ts_30m": prov_30m.get("last_candle_ts"),
            "candle_ts_15m": prov_15m.get("last_candle_ts"), "candle_ts_5m": prov_5m.get("last_candle_ts"),
        }

    # If already fully handled or invalid — still stamp last_evaluated_at so UI shows current time
    if state_record.mtf_substate in (MtfSubstate.INVALIDATED, MtfSubstate.BREAKOUT_CONFIRMED):
        if funnel_counters is not None:
            funnel_counters["ALREADY_HANDLED_OR_INVALIDATED"] += 1
        update_state_in_db(state_record, _live_data_updates())
        return

    # Check invalidation logic
    if handle_box_invalidation(state_record, current_price, consolidation.box_low, atr_15m, ist_now):
        if funnel_counters is not None:
            funnel_counters["BOX_BREAKDOWN"] += 1
        logger.info(f"🚫 [MULTI_TF] {symbol} REJECTED — Box breakdown below ₹{consolidation.box_low:.2f}")
        if not update_state_in_db(state_record, _live_data_updates()):
            return
        return

    # TTL checks
    current_5m_bars_count = len(bundle.df_5m_closed)
    if apply_ttl_and_cooldown(state_record, ist_now, current_5m_bars_count):
        if funnel_counters is not None:
            funnel_counters["TTL_OR_COOLDOWN_EXPIRED"] += 1
        logger.info(f"🚫 [MULTI_TF] {symbol} REJECTED — TTL expired or cooldown active")
        if not update_state_in_db(state_record, _live_data_updates()):
            return

    if state_record.mtf_substate == MtfSubstate.FAILED_ATTEMPT:
        if funnel_counters is not None:
            funnel_counters["FAILED_ATTEMPT_COOLDOWN"] += 1
        # Still stamp last_evaluated_at even while cooling down
        update_state_in_db(state_record, _live_data_updates())
        return

    # 5. Pressure / Expansion (5m Live + Closed)
    daily_atr_val = _get_atr(bundle.df_1d)
    pressure = evaluate_5m_pressure(
        live_candle=bundle.live_5m,
        df_5m_closed=bundle.df_5m_closed,
        box_high=consolidation.box_high,
        atr_5m=atr_5m,
        ist_now=ist_now,
        config=config,
        daily_atr=daily_atr_val,
        atr_15m=atr_15m
    )

    updates = {}

    # Session Timing & Cutoff Checks (IST)
    # Normal Cutoff: 14:15 IST
    # Late Session: 14:15 - 15:00 IST (applies stricter quality floor: Base >= 75, Breakout >= 75, RVOL >= 1.50, Confluence >= 82)
    # Hard Blackout: >= 15:00 IST (strictly blocks new trade initiation across all regimes)
    from datetime import time as dtime
    from market_utils import is_market_open
    _market_active = is_market_open(ist_now)
    is_late_session = False
    is_past_hard_cutoff = False
    if _market_active:
        now_time = ist_now.time()
        if now_time >= dtime(15, 0):
            is_past_hard_cutoff = True
        elif now_time >= dtime(14, 15):
            is_late_session = True

    if pressure.is_confirmed:
        # ── BREAKOUT PATH (EARLY_BREAKOUT Execution Alert) ──
        if state_record.mtf_substate == MtfSubstate.BREAKOUT_CONFIRMED:
            # [RULE: MODEL B IS TRADE EVOLUTION, NOT A DUPLICATE TRADE]
            if pressure.trigger_model == "MODEL_B_RETEST":
                logger.info(f"🛡️ [{symbol}] Breakout Retest Defended @ ₹{current_price:.2f} — Trade Evolution recorded.")
                updates["last_retest_ts"] = ist_now
            else:
                if funnel_counters is not None:
                    funnel_counters["DUPLICATE_SETUP"] += 1
            update_state_in_db(state_record, {**_live_data_updates(), **updates})
            return
        elif is_past_hard_cutoff:
            if funnel_counters is not None:
                funnel_counters["LATE_SESSION_HARD_CUTOFF"] += 1
            logger.info("⏳ [MULTI_TF] %s REJECTED — Pressure confirmed but past hard entry cutoff (15:00 IST). New trade initiation suppressed.", symbol)
            update_state_in_db(state_record, _live_data_updates())
            return
        else:
            # 6. Confluence Evaluation
            confluence = evaluate_breakout_confluence(
                consolidation=consolidation,
                pressure=pressure,
                ctx_1h=ctx_1h,
                ctx_30m=ctx_30m,
                market_ctx=market_ctx,
                config=config
            )

            # ── 5M EARLY BREAKOUT CONTRACT ──
            c_col_5m = 'Close' if 'Close' in bundle.df_5m_closed.columns else ('close' if 'close' in bundle.df_5m_closed.columns else bundle.df_5m_closed.columns[0])
            c_5m = float(bundle.df_5m_closed[c_col_5m].iloc[-1])
            buffer_atr = config.get("BREAKOUT_BUFFER_ATR_MULT", 0.10) * (atr_5m if atr_5m > 0 else 1.0)
            res_line = consolidation.box_high

            # Gate 1: 5M Close above resistance + buffer
            if c_5m < (res_line + buffer_atr):
                dist_res = ((res_line + buffer_atr - c_5m) / res_line * 100.0) if res_line > 0 else 0.0
                if funnel_counters is not None:
                    funnel_counters["BREAKOUT_CLOSE_FAIL"] += 1
                logger.info("🚫 [MULTI_TF] %s REJECTED — 5M Close (₹%.2f) failed to clear resistance buffer (₹%.2f [dist: -%.1f%%]) | RVOL: %.2fx", symbol, c_5m, res_line + buffer_atr, dist_res, pressure.volume_ratio)
                if c_5m >= res_line:
                    try:
                        from near_miss_tracker import log_near_miss
                        log_near_miss(symbol, "MULTI_TF", "5M_BREAKOUT", "resistance_buffer_clearance", c_5m, res_line + buffer_atr, entry_price=c_5m, stop_loss=consolidation.box_low)
                    except Exception:
                        pass
                update_state_in_db(state_record, _live_data_updates())
                return

            # Gate 2: Volume confirmation
            min_rvol = config.get("MIN_VOLUME_EXPANSION_CONFIRM", 1.25)
            if pressure.volume_ratio < min_rvol and pressure.trigger_model != "MODEL_B_RETEST":
                if funnel_counters is not None:
                    funnel_counters["BREAKOUT_RVOL_FAIL"] += 1
                logger.info("🚫 [MULTI_TF] %s REJECTED — 5M Volume ratio (%.2fx) below confirmation threshold (%.2fx) (CMP: ₹%.2f | Level: ₹%.2f)", symbol, pressure.volume_ratio, min_rvol, c_5m, res_line)
                try:
                    from near_miss_tracker import log_near_miss
                    log_near_miss(symbol, "MULTI_TF", "5M_BREAKOUT", "insufficient_volume_surge", pressure.volume_ratio, min_rvol, score=int(confluence.total_score if 'confluence' in locals() else 0), entry_price=c_5m, stop_loss=consolidation.box_low)
                except Exception:
                    pass
                update_state_in_db(state_record, _live_data_updates())
                return

            # Gate 3: Overextension / Velocity Exhaustion
            if pressure.is_overextended:
                if funnel_counters is not None:
                    funnel_counters["BREAKOUT_EXHAUSTION"] += 1
                logger.info("🚫 [MULTI_TF] %s REJECTED — Breakout candle overextended or abnormal velocity blow-off (CMP: ₹%.2f | Level: ₹%.2f)", symbol, c_5m, res_line)
                update_state_in_db(state_record, _live_data_updates())
                return

            # Gate 4: Confluence approval
            if not confluence.is_approved:
                if funnel_counters is not None:
                    funnel_counters["LOW_CONFLUENCE"] += 1
                logger.info("🚫 [MULTI_TF] %s REJECTED — Confluence not approved (Score: %.1f | CMP: ₹%.2f | Level: ₹%.2f | RVOL: %.2fx)", symbol, confluence.total_score, c_5m, res_line, pressure.volume_ratio)
                try:
                    from near_miss_tracker import log_near_miss
                    log_near_miss(symbol, "MULTI_TF", "5M_BREAKOUT", "confluence_score_below_threshold", confluence.total_score, 70.0, score=int(confluence.total_score), entry_price=c_5m, stop_loss=consolidation.box_low)
                except Exception:
                    pass
                update_state_in_db(state_record, _live_data_updates())
                return

            if funnel_counters is not None:
                funnel_counters["confluence_approved"] += 1

            # 7. R:R Target Generation & Pre-Validation Gate (T0 obstacle + T1 tradeability)
            sl_target = compute_sl_and_target(
                entry_price=c_5m,
                atr=atr_5m,
                ticker=bundle.df_1h,  # Pass 1H for structural targets
                mode="MULTI_TF_V2",
                box_low=consolidation.box_low
            )

            rr_actual = float(sl_target.get("rr_ratio", 0.0))
            if sl_target.get("is_rejected") or rr_actual < config.get("MIN_RR_RATIO", 1.5):
                if funnel_counters is not None:
                    funnel_counters["RR_T1_FAIL"] += 1
                logger.info("🚫 [MULTI_TF] %s REJECTED — R:R gate failed (%.2f < %.2f). NOT SAVING TO ALERTS.",
                            symbol, rr_actual, config.get("MIN_RR_RATIO", 1.5))
                invalidate_record(state_record, ist_now, "NOT_TRADEABLE")
                updates["invalidated_at"] = ist_now
                updates["invalidation_reason"] = "NOT_TRADEABLE"
                update_state_in_db(state_record, {**_live_data_updates(), **updates})
                try:
                    from near_miss_tracker import log_near_miss
                    log_near_miss(
                        symbol=symbol,
                        scanner="MULTI_TF",
                        breakout_type="MULTI_TF",
                        gate_name="rr_ratio_gate",
                        observed_value=rr_actual,
                        threshold_value=float(config.get("MIN_RR_RATIO", 1.5)),
                        score=int(confluence.total_score),
                        entry_price=float(sl_target.get("entry_price") or 0.0),
                        stop_loss=float(sl_target.get("stop_loss") or 0.0),
                        target_1=float(sl_target.get("target_1") or 0.0),
                    )
                except Exception:
                    pass
                return

            # 8. Breakout Strength Engine
            nifty_5m = bundle.__dict__.get("df_nifty_5m", None)
            brkout_strength = compute_breakout_strength(
                pressure_result=pressure,
                consolidation_result=consolidation,
                df_5m_closed=bundle.df_5m_closed,
                nifty_5m=nifty_5m,
                ist_now=ist_now,
                config=config
            )

            # 9. Alert Severity Classification (Descriptive)
            market_status = str(market_ctx.get("status", "NORMAL"))
            severity = classify_alert_severity(
                consolidation_result=consolidation,
                breakout_result=brkout_strength,
                config=config,
                market_status=market_status
            )

            # 10. Institutional Trade Eligibility Contract (Decoupled from Severity)
            is_eligible, reject_reason = evaluate_trade_eligibility(
                base_score=consolidation.setup_score,
                breakout_score=brkout_strength.breakout_score,
                volume_ratio=pressure.volume_ratio,
                confluence_score=int(confluence.total_score),
                rr_ratio=rr_actual,
                market_status=market_status,
                config=config,
                is_late_session=is_late_session
            )

            if not is_eligible:
                if funnel_counters is not None:
                    funnel_counters[reject_reason] += 1
                logger.info(
                    "🚫 [MULTI_TF] %s REJECTED — Failed trade eligibility contract: %s "
                    "(Base: %d, Brk: %d, RVOL: %.2f, Conf: %d, RR: %.2f, Mkt: %s, Late: %s)",
                    symbol, reject_reason, consolidation.setup_score, brkout_strength.breakout_score,
                    pressure.volume_ratio, int(confluence.total_score), rr_actual, market_status, is_late_session
                )
                update_state_in_db(state_record, _live_data_updates())
                return

            # 11. Canonical Alert Registration for High-Conviction EARLY_BREAKOUT
            idempotency_signals = f"BOX_ID={consolidation.box_id}"

            inserted, _, _, _ = save_alert_if_new(
                symbol=symbol,
                breakout_type="MULTI_TF",
                alert_time=ist_now.strftime("%Y-%m-%d %H:%M:%S"),
                scanner="MULTI_TF",
                category="INTRADAY",
                entry_price=sl_target.get("entry_price"),
                stop_loss=sl_target.get("stop_loss"),
                target_1=sl_target.get("target_1"),
                target_2=sl_target.get("target_2"),
                target_3=sl_target.get("target_3"),
                signals=idempotency_signals,
                score=int(consolidation.setup_score),
                volume_ratio=pressure.volume_ratio,
                context={
                    "box_id": consolidation.box_id,
                    "signal_status": "CONFIRMED",
                    "breakout_stage": "EARLY_BREAKOUT",
                    "trigger_model": pressure.trigger_model,
                    "tradeability_status": "TRADEABLE",
                    "tradeability_reason": "",
                    "rr_ratio": rr_actual,
                    "t0_obstacle": sl_target.get("target_0"),
                    "t0_rr_ratio": sl_target.get("t0_rr_ratio"),
                    "t1_target": sl_target.get("target_1"),
                    "t1_source": sl_target.get("t1_source", "STRUCTURAL"),
                    # Base Quality
                    "base_score": consolidation.setup_score,
                    "base_rating": consolidation.base_rating_label,
                    "has_higher_lows": consolidation.has_higher_lows,
                    "compression_ratio": consolidation.compression_ratio,
                    "resistance_tests": consolidation.resistance_test_count,
                    "supply_absorption": consolidation.supply_absorption_label,
                    "base_score_breakdown": {
                        "maturity": consolidation.score_maturity,
                        "tightness": consolidation.score_tightness,
                        "resistance_quality": consolidation.score_resistance_quality,
                        "repeated_tests": consolidation.score_repeated_tests,
                        "compression": consolidation.score_compression,
                        "higher_lows": consolidation.score_higher_lows,
                        "support_integrity": consolidation.score_support_integrity,
                    },
                    # Breakout Strength
                    "breakout_score": brkout_strength.breakout_score,
                    "breakout_rating": brkout_strength.breakout_rating_label,
                    "breakout_energy": brkout_strength.breakout_energy,
                    "breakout_energy_label": brkout_strength.breakout_energy_label,
                    "severity": severity,
                    "severity_label": SEVERITY_LABEL.get(severity, severity),
                    "rvol": round(pressure.volume_ratio, 2),
                    "rvol_label": brkout_strength.rvol_label,
                    "volume_acceleration": brkout_strength.volume_acceleration,
                    "base_relative_volume": brkout_strength.base_relative_volume,
                    "velocity_label": brkout_strength.velocity_label,
                    "penetration_atr": brkout_strength.penetration_atr,
                    "close_position": brkout_strength.close_position,
                    "breakout_score_breakdown": brkout_strength.to_dict().get("score_breakdown"),
                    "market_regime": market_status,
                    "is_late_session": is_late_session,
                }
            )

            # 12. Dispatch to OpportunityManager
            if inserted:
                if funnel_counters is not None:
                    funnel_counters["alert_triggered"] += 1
                rich_message = build_multitf_alert_message(
                    symbol=symbol,
                    exchange="NSE",
                    consolidation=consolidation,
                    pressure=pressure,
                    breakout_strength=brkout_strength,
                    severity=severity,
                    sl_levels={
                        "entry": float(sl_target.get("entry_price") or 0),
                        "stop":  float(sl_target.get("stop_loss") or 0),
                        "t0":    float(sl_target.get("target_0") or 0),
                        "t1":    float(sl_target.get("target_1") or 0),
                        "t2":    float(sl_target.get("target_2") or 0),
                        "t3":    float(sl_target.get("target_3") or 0),
                        "t1_source": sl_target.get("t1_source", "STRUCTURAL"),
                        "rr_ratio": rr_actual,
                        "extension_daily_atr": float(sl_target.get("extension_daily_atr") or 0),
                    },
                    ist_now=ist_now
                )
                logger.info("[%s] %s Early Breakout Alert (base=%d, brk=%d):\n%s",
                            symbol, SEVERITY_EMOJI.get(severity, "🟢"),
                            consolidation.setup_score, brkout_strength.breakout_score, rich_message)
                logger.info(
                    f"🌟 [MULTI_TF: SELECTED] {symbol} @ ₹{float(sl_target.get('entry_price') or 0):.2f} | "
                    f"Severity: {severity} | Base: {consolidation.setup_score} ({consolidation.base_rating_label}) | "
                    f"Breakout Score: {brkout_strength.breakout_score} ({brkout_strength.breakout_rating_label}) | "
                    f"RVOL: {pressure.volume_ratio:.2f}x | SL: ₹{float(sl_target.get('stop_loss') or 0):.2f} | "
                    f"T0: ₹{float(sl_target.get('target_0') or 0):.2f} | "
                    f"T1: ₹{float(sl_target.get('target_1') or 0):.2f} [{sl_target.get('t1_source')}] | RR: {rr_actual:.2f}"
                )

                payload = build_confirmed_payload(
                    bundle=bundle,
                    consolidation=consolidation,
                    pressure=pressure,
                    confluence=None,
                    sl_target=sl_target,
                    ist_now=ist_now,
                    alert_message=rich_message,
                    severity=severity,
                    breakout_strength=brkout_strength
                )
                opp_manager.add(payload)
            else:
                if funnel_counters is not None:
                    funnel_counters["DUPLICATE_SETUP"] += 1
                logger.info("🚫 [MULTI_TF] %s REJECTED — Alert already processed for box %s, skipping OpportunityManager.", symbol, consolidation.box_id)

            state_record.mtf_substate = MtfSubstate.BREAKOUT_CONFIRMED
            state_record.state = "CONFIRMED"
            updates["last_confirmation_ts"] = ist_now

    else:
        # ── PRE-BREAKOUT PATH (Coiling / Ignition Readiness Evaluation) ──
        # Evaluates high-quality bases near resistance before breakout confirmation.
        # Updates setup state to ARMED_PRE_BREAKOUT (Candidate in mtf_v2_watchlist) rather than emitting trade alert.
        dist_to_high = consolidation.box_high - current_price
        dist_atr = dist_to_high / atr_15m if atr_15m > 0 else 999.0
        min_prebreak_base = config.get("PRE_BREAKOUT_MIN_BASE_SCORE", 75)

        ign_res = compute_ignition_score(
            consolidation=consolidation,
            pressure=pressure,
            distance_to_box_high_atr=dist_atr,
            ctx_1h=ctx_1h,
            config=config
        )

        if (consolidation.setup_score >= min_prebreak_base
                and ign_res.get("is_ignition_ready")
                and state_record.mtf_substate in (MtfSubstate.WATCHING, MtfSubstate.PRESSURE_BUILDING, MtfSubstate.ARMED_PRE_BREAKOUT, MtfSubstate.ATTEMPT)):

            # Pre-Breakout Contract: Projected Tradeability Check
            planned_entry = consolidation.box_high + (0.05 * atr_5m)
            proj_sl_target = compute_sl_and_target(
                entry_price=planned_entry,
                atr=atr_5m,
                ticker=bundle.df_1h,
                mode="MULTI_TF_V2",
                box_low=consolidation.box_low
            )
            proj_rr = float(proj_sl_target.get("rr_ratio", 0.0))

            if proj_rr >= config.get("MIN_RR_RATIO", 1.5) and not proj_sl_target.get("is_rejected"):
                state_record.mtf_substate = MtfSubstate.ARMED_PRE_BREAKOUT
                state_record.state = "CANDIDATE"
                updates["pressure_state"] = "ARMED_PRE_BREAKOUT"
                if funnel_counters is not None:
                    funnel_counters["ARMED_PRE_BREAKOUT_ACTIVE"] += 1
                logger.info(
                    "🎯 [MULTI_TF: ARMED] %s ARMED_PRE_BREAKOUT | Base: %d | Ignition: %d | "
                    "Dist: %.2f ATR | Proj Entry: ₹%.2f | Proj SL: ₹%.2f | Proj T1: ₹%.2f (%s) | Proj RR: %.2f",
                    symbol, consolidation.setup_score, ign_res["ignition_score"], dist_atr,
                    planned_entry, float(proj_sl_target.get("stop_loss", 0)),
                    float(proj_sl_target.get("target_1", 0)), proj_sl_target.get("t1_source"), proj_rr
                )
            else:
                if funnel_counters is not None:
                    funnel_counters["PREBREAK_RR_FAIL"] += 1
                logger.info("🚫 [MULTI_TF] %s — Pre-breakout ignition ready but projected R:R (%.2f) < 1.5R", symbol, proj_rr)

        elif pressure.is_attempt and state_record.mtf_substate == MtfSubstate.WATCHING:
            if funnel_counters is not None:
                funnel_counters["ATTEMPT_REGISTERED"] += 1
            logger.info("👀 [MULTI_TF] %s — Breakout attempt detected, moving to ATTEMPT state.", symbol)
            state_record.mtf_substate = MtfSubstate.ATTEMPT
            state_record.state = "CANDIDATE"
            state_record.attempt_count += 1
            updates["attempt_started_ts"] = ist_now
            updates["last_attempt_ts"] = ist_now
            updates["attempt_bar_boundary"] = pressure.attempt_bar_boundary
        else:
            if dist_atr > config.get("PRE_BREAKOUT_MAX_DISTANCE_ATR", 0.40):
                if funnel_counters is not None:
                    funnel_counters["PREBREAK_DISTANCE_FAIL"] += 1
            elif consolidation.setup_score < min_prebreak_base:
                if funnel_counters is not None:
                    funnel_counters["PREBREAK_BASE_FAIL"] += 1
            elif not ign_res.get("is_ignition_ready"):
                if funnel_counters is not None:
                    funnel_counters["PREBREAK_IGNITION_FAIL"] += 1
            else:
                if funnel_counters is not None:
                    funnel_counters["NO_BREAKOUT_PRESSURE"] += 1

    # 10. Sync all live evaluation data to DB on every cycle
    # [FIX: LIVE_DATA_ALWAYS_REFRESH_v1.0]
    # Previously only state-transition fields were written. Box geometry, scores,
    # pressure metrics, context scores, candle timestamps were frozen at first insert.
    # Now every 15m cycle refreshes ALL columns so the UI always shows current data.
    prov_1h  = bundle.prov_1h.to_dict()  if bundle.prov_1h  else {}
    prov_30m = bundle.prov_30m.to_dict() if bundle.prov_30m else {}
    prov_15m = bundle.prov_15m.to_dict() if bundle.prov_15m else {}
    prov_5m  = bundle.prov_5m.to_dict()  if bundle.prov_5m  else {}

    updates.update({
        # Box geometry (refreshed each 15m — box can evolve as new bars close)
        "box_high":               consolidation.box_high,
        "box_low":                consolidation.box_low,
        "box_mid":                consolidation.box_mid,
        "box_value_center":       consolidation.box_value_center,
        "hard_high":              consolidation.hard_high,
        "hard_low":               consolidation.hard_low,
        "box_width_pct":          consolidation.box_width_pct,
        "box_width_atr":          consolidation.box_width_atr,
        "box_occupancy":          consolidation.box_occupancy,
        "consolidation_bars":     consolidation.bars_count,
        "consolidation_sessions": consolidation.sessions_count,
        "consolidation_end_ts":   consolidation.end_ts,
        # Base quality scores (recomputed each scan)
        "resistance_test_count":      consolidation.resistance_test_count,
        "higher_low_score":           consolidation.score_hl,
        "compression_score":          consolidation.score_compression,
        "setup_score":                consolidation.setup_score,
        "last_confirmed_pivot_level": consolidation.last_confirmed_pivot_level,
        "last_confirmed_pivot_ts":    consolidation.last_confirmed_pivot_ts,
        # Pressure metrics (live 5m state)
        "pressure_state":       pressure.label if hasattr(pressure, "label") else None,
        "volume_ratio_5m":      round(pressure.volume_ratio, 4) if pressure.volume_ratio else None,
        "range_ratio_5m":       round(pressure.range_ratio, 4) if hasattr(pressure, "range_ratio") and pressure.range_ratio else None,
        "distance_to_box_high": round(consolidation.box_high - current_price, 4) if current_price else None,
        "live_position_5m":     round(current_price, 4) if current_price else None,
        # Multi-TF context scores
        "context_1h_score":  ctx_1h.get("score",  0),
        "context_30m_score": ctx_30m.get("score", 0),
        "market_regime":     market_ctx.get("regime", "UNKNOWN"),
        # Data freshness provenance
        "data_source_1h":  prov_1h.get("source",  ""),
        "data_source_30m": prov_30m.get("source", ""),
        "data_source_15m": prov_15m.get("source", ""),
        "data_source_5m":  prov_5m.get("source",  ""),
        "candle_ts_1h":    prov_1h.get("last_candle_ts"),
        "candle_ts_30m":   prov_30m.get("last_candle_ts"),
        "candle_ts_15m":   prov_15m.get("last_candle_ts"),
        "candle_ts_5m":    prov_5m.get("last_candle_ts"),
    })

    update_state_in_db(state_record, updates)

