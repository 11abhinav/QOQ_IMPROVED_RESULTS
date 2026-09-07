import os
import time
import logging
import threading
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST_ZONE = ZoneInfo("Asia/Kolkata")
from constituent_service import fetch_constituents

def is_in_window(now: datetime = None) -> bool:
    """Check if current time is within active worker window: Whole Saturday & Sunday only."""
    if now is None:
        now = datetime.now(IST_ZONE)
    return now.weekday() >= 5  # Saturday & Sunday

def get_active_window_description(now: datetime = None) -> str:
    return "00:00 - 23:59 IST (Sat-Sun Only)"

# [VERSION: AI_WORKER_MANUAL_v1.0] Extract run_ai_worker_scan_once and protect with _scan_lock
_scan_lock = threading.Lock()

def run_ai_worker_scan_once() -> dict:
    """Run a single scan of watchlist and excluded stocks to analyze concalls.
    Protected by _scan_lock to prevent concurrent executions.
    """
    from database import is_scanner_stopped
    if is_scanner_stopped("AI Worker"):
        logger.info("⏭️ AI Worker is PAUSED by Admin. Skipping execution.")
        return {"total_count": 0, "processed_count": 0}

    if not _scan_lock.acquire(blocking=False):
        logger.warning("🤖 AI Worker Scan already running. Skipping execution.")
        raise RuntimeError("AI Worker is already actively running!")
        
    _fn_start = time.time()
    try:
        from config import WATCHLIST_PATH
        from database import get_recent_concall_analysis, upsert_scanner_health, get_total_cached_concalls, upsert_fetch_error, save_concall_analysis, has_valid_concall_cache, has_error_concall_cache_within_24h, start_scanner_execution_run, complete_scanner_execution_run
        from dashboard_server import fetch_and_analyze_concall
        
        try:
            run_ctx = start_scanner_execution_run(scanner_name="AI Worker", trigger_type="SCHEDULED", scheduler_name="WORKER")
        except Exception as _ai_err:
            if "actively running" in str(_ai_err).lower():
                logger.info("⏳ AI Worker is already actively running. Skipping duplicate pass.")
                return {"total_count": 0, "processed_count": 0}
            raise
        
        upsert_scanner_health("AI Worker", "RUNNING", error_msg="AI Worker Scan in progress...")
        now_ist = datetime.now(IST_ZONE)
        logger.debug("=" * 70)
        logger.debug(f"🤖 [AI WORKER] Starting concall analysis scan | {now_ist.strftime('%H:%M:%S IST')}")
        logger.debug("=" * 70)
        
        if not os.path.exists(WATCHLIST_PATH):
            logger.warning("Watchlist parquet file does not exist yet.")
            complete_scanner_execution_run(run_ctx, status_override="SKIPPED", stop_reason="No watchlist")
            return {"total_count": 0, "processed_count": 0}
            
        try:
            df = pd.read_parquet(WATCHLIST_PATH)
        except Exception as e:
            logger.exception(f"Failed to read parquet watchlist")
            complete_scanner_execution_run(run_ctx, exception=e)
            return {"total_count": 0, "processed_count": 0}
            
        pending_stocks = df["Stock"].tolist()
        
        # Read excluded stocks so they are pre-cached if they break out later
        excluded_csv_paths = [
            os.path.join(os.path.dirname(WATCHLIST_PATH), 'elite_fundamental_watchlist_excluded.csv'),
            os.path.join(os.path.dirname(WATCHLIST_PATH), 'elite_fundamental_watchlist-excluded.csv'),
            WATCHLIST_PATH.replace('.parquet', '_excluded.csv'),
        ]
        for excluded_csv_path in excluded_csv_paths:
            if os.path.exists(excluded_csv_path):
                try:
                    df_ex = pd.read_csv(excluded_csv_path)
                    if 'Stock' in df_ex.columns:
                        ex_stocks = df_ex['Stock'].dropna().tolist()
                        pending_stocks.extend(ex_stocks)
                        break
                except Exception as e:
                    logger.warning(f"Failed to load exclusion list {excluded_csv_path}: {e}")
                    
        from config import NON_EQUITY_BLOCKLIST
        pending_stocks = sorted([s for s in set(pending_stocks) if str(s).strip().upper() not in NON_EQUITY_BLOCKLIST])
        total_stocks = len(pending_stocks)
        
        # ── Pre-filter: only process stocks that genuinely need analysis ────────────────
        from database import get_bulk_concall_cache_status
        status_cache = get_bulk_concall_cache_status(pending_stocks)
        
        actual_pending = []
        for sym in pending_stocks:
            # PRIMARY CHECK: Does a valid (non-error) cache exist for this symbol?
            if sym in status_cache['valid']:
                continue  # Valid analysis exists → skip

            # SECONDARY CHECK: Was an error cached within the last 7 days?
            if sym in status_cache['recent_error']:
                continue  # Recent error → back off

            actual_pending.append(sym)
            
        db_processed_count = get_total_cached_concalls()
        if not actual_pending:
            elapsed = round(time.time() - _fn_start, 1)
            logger.debug(f"🤖 [AI WORKER] All {total_stocks} stocks already cached today. Nothing to do. ({elapsed}s)")
            complete_scanner_execution_run(run_ctx, status_override="SKIPPED", stop_reason="All cached")
            return {"total_count": total_stocks, "processed_count": db_processed_count}
            
        # Check Gemini API Key availability before scanning
        from gemini_key_manager import get_active_gemini_key
        if not get_active_gemini_key():
            # [RULE 67 - FIX RATIONALE]: Updated warning and health message to 1-day blacklist.
            logger.warning("🚨 [AI WORKER] All Gemini API keys are blacklisted/exhausted for the next 1 day (24h). Pausing AI Worker for 1 hour.")
            upsert_scanner_health("AI Worker", "EXHAUSTED", error_msg="All Gemini API keys exhausted (1-day blacklist) — Paused 1h")
            try:
                from database import insert_notification
                from push_service import send_push_to_all
                insert_notification("admin", "🚨 AI WORKER PAUSED", "All Gemini API keys are marked exhausted for the next 1 day. AI Worker paused for 1 hour to prevent per-stock errors.")
                send_push_to_all("🚨 AI WORKER PAUSED", "All Gemini API keys exhausted. AI Worker paused for 1 hour.")
            except Exception as notif_err:
                logger.warning(f"Failed to send AI key exhaustion notifications: {notif_err}")
            complete_scanner_execution_run(run_ctx, status_override="SKIPPED", stop_reason="All Gemini API keys exhausted")
            return {"total_count": total_stocks, "processed_count": db_processed_count}

        logger.info(f"📊 [AI WORKER] Pending symbols to fetch today: {len(actual_pending)} (out of {total_stocks} universe) | {total_stocks - len(actual_pending)} already cached in DB")
        
        max_retries = 3
        global_penalty_idx = 0
        final_failed_count = 0
        db_processed_count = total_stocks - len(actual_pending)
        
        for attempt in range(max_retries):
            attempt_start = time.time()
            logger.info(f"🤖 [AI WORKER] Batch attempt {attempt+1}/{max_retries} | Symbols to process: {len(actual_pending)}")
            failed_stocks = []
            import concurrent.futures
            import threading
            
            _db_lock = threading.Lock()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(fetch_and_analyze_concall, sym): sym for sym in actual_pending}
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    sym = futures[future]
                    sym_start = time.time()
                    try:
                        logger.info(f"🤖 [AI WORKER] [{i+1}/{len(actual_pending)}] Fetching live concall for {sym}...")
                        result = future.result()
                        
                        if result and "error" not in result:
                            sym_elapsed = round(time.time() - sym_start, 2)
                            conf = result.get("management_confidence", "N/A")
                            key_used = result.get("key_used", "Key 1")
                            logger.info(f"✅ [AI WORKER] {sym} ({i+1}/{len(actual_pending)}) ✔ Cached | Conf={conf} | {key_used} | {sym_elapsed}s")
                            with _db_lock:
                                db_processed_count += 1
                                upsert_scanner_health("AI Worker", "OK", last_success=datetime.now(IST_ZONE).isoformat(), today_alerts=db_processed_count, processed_count=db_processed_count, total_count=total_stocks, error_msg=f"Last: {sym} | Total: {total_stocks}")
                        else:
                            error_msg = result.get('error', 'Unknown Error') if result else 'No result returned'
                            logger.warning(f"⚠️ [AI WORKER] Failed to cache {sym}: {error_msg}")
                            try:
                                upsert_fetch_error('ai', 'AI Worker', sym, None, 'ai_concall', error_msg)
                            except Exception as inner_e:
                                logger.exception(f"Failed to upsert fetch_error for AI Worker: {inner_e}")
                            
                            with _db_lock:
                                upsert_scanner_health("AI Worker", "OK", last_success=datetime.now(IST_ZONE).isoformat(), today_alerts=db_processed_count, processed_count=db_processed_count, total_count=total_stocks, error_msg=f"Last: {sym} | Total: {total_stocks}")
    
                            is_rate_limit = "429" in error_msg or "All AI models" in error_msg
                            is_transient  = ("503" in error_msg or "502" in error_msg or "timeout" in error_msg.lower()) and "no recent concall" not in error_msg.lower()
    
                            if is_rate_limit or is_transient:
                                failed_stocks.append(sym)
                                logger.warning(f"⚠️ [AI WORKER] Transient/rate-limit error for {sym}. Adding to retry queue...")
                            else:
                                logger.warning(f"⚠️ [AI WORKER] Persistent/no-data result for {sym}: {error_msg}. Saving 24h negative cache to skip for today.")
                                save_concall_analysis(sym, f"NONE_{sym}", {"error": error_msg})
                                
                    except Exception as e:
                        logger.exception(f"❌ [AI WORKER] Error processing {sym}")
                        try:
                            upsert_fetch_error('ai', 'AI Worker', sym, None, 'ai_concall_failure', str(e))
                        except Exception as inner_e:
                            logger.exception(f"Failed to upsert fetch_error for {sym}: {inner_e}")
                        failed_stocks.append(sym)
                    
            attempt_elapsed = round(time.time() - attempt_start, 1)
            logger.info(f"🤖 [AI WORKER] Attempt {attempt+1} done in {attempt_elapsed}s | Processed={len(actual_pending)-len(failed_stocks)} | Failed={len(failed_stocks)}")
            if not failed_stocks:
                break
            actual_pending = failed_stocks
            if attempt < max_retries - 1:
                logger.info(f"🤖 [AI WORKER] {len(failed_stocks)} stocks failed. Retrying in 60s (Attempt {attempt+2}/{max_retries})...")
                time.sleep(60)
            else:
                logger.error(f"❌ [AI WORKER] Giving up on {len(failed_stocks)} stocks after {max_retries} attempts.")
                final_failed_count = len(failed_stocks)
                for fsym in failed_stocks:
                    try:
                        upsert_fetch_error('ai', 'AI Worker', fsym, None, 'ai_concall', 'Giving up after retries')
                        save_concall_analysis(fsym, f"NONE_{fsym}", {"error": "Giving up after retries"})
                    except Exception as inner_e:
                        logger.exception(f"Failed to upsert final fetch_error for {fsym}: {inner_e}")
                        
        total_elapsed = round(time.time() - _fn_start, 1)
        run_ctx.set_total_stocks(total_stocks)
        run_ctx.record_fresh_data(db_processed_count)
        run_ctx.record_stale_data(final_failed_count)
        logger.info("=" * 70)
        logger.info(f"🤖 [AI WORKER] Scan complete in {total_elapsed}s | Total={total_stocks} | Processed={db_processed_count} | Failed={final_failed_count}")
        logger.info("=" * 70)
        complete_scanner_execution_run(run_ctx)
        return {"total_count": total_stocks, "processed_count": db_processed_count}
    except Exception as outer_err:
        try:
            complete_scanner_execution_run(run_ctx, exception=outer_err)
        except:
            pass
        raise outer_err
        
    finally:
        _scan_lock.release()

def run_worker_loop():
    """Infinite loop that scans the watchlist CSV and fetches AI concall reports."""
    from database import upsert_scanner_health, get_ai_concall_stats
    from config import WATCHLIST_PATH
    
    logger.info("🤖 AI Worker Thread Started. Monitoring watchlist for missing caches...")
    
    # [VERSION: AI_WORKER_PROGRESS_v1.0] Calculate initial dynamic counts on boot
    try:
        symbols_set = set()
        if os.path.exists(WATCHLIST_PATH):
            df = pd.read_parquet(WATCHLIST_PATH)
            if "Stock" in df.columns:
                symbols_set.update(df["Stock"].dropna().unique().tolist())
                
        excluded_paths = [
            os.path.join(os.path.dirname(WATCHLIST_PATH), 'elite_fundamental_watchlist_excluded.csv'),
            os.path.join(os.path.dirname(WATCHLIST_PATH), 'elite_fundamental_watchlist-excluded.csv'),
            WATCHLIST_PATH.replace(".parquet", "_excluded.csv"),
        ]
        for f in excluded_paths:
            if os.path.exists(f):
                try:
                    dfw = pd.read_csv(f)
                    if 'Stock' in dfw.columns:
                        symbols_set.update(dfw['Stock'].dropna().tolist())
                        break
                except Exception:
                    pass
                    
        idx_symbols = fetch_constituents()
        if idx_symbols:
            symbols_set.update(idx_symbols)
            
        from config import NON_EQUITY_BLOCKLIST
        symbols = [s for s in list(symbols_set) if str(s).strip().upper() not in NON_EQUITY_BLOCKLIST]
        total_watch = len(symbols)
        stats = get_ai_concall_stats(symbols)
        processed_count = stats.get("total_cached", 0)
    except Exception as e:
        logger.warning(f"Failed to calculate boot progress stats: {e}")
        total_watch = 0
        processed_count = 0
        
    upsert_scanner_health("AI Worker", "IDLE", today_alerts=processed_count, processed_count=processed_count, total_count=total_watch, error_msg="Status: Booting up")
    
    while True:
        # Re-calculate on each loop iteration
        try:
            symbols_set = set()
            if os.path.exists(WATCHLIST_PATH):
                df = pd.read_parquet(WATCHLIST_PATH)
                if "Stock" in df.columns:
                    symbols_set.update(df["Stock"].dropna().unique().tolist())
                    
            for f in excluded_paths:
                if os.path.exists(f):
                    try:
                        dfw = pd.read_csv(f)
                        if 'Stock' in dfw.columns:
                            symbols_set.update(dfw['Stock'].dropna().tolist())
                            break
                    except Exception:
                        pass
                        
            idx_symbols = fetch_constituents()
            if idx_symbols:
                symbols_set.update(idx_symbols)
                
            symbols = list(symbols_set)
            total_watch = len(symbols)
            stats = get_ai_concall_stats(symbols)
            processed_count = stats.get("total_cached", 0)
        except Exception as e:
            logger.warning(f"Failed to calculate loop progress stats: {e}")
            total_watch = total_watch or 0
            processed_count = processed_count or 0
            
        from database import is_scanner_stopped
        if is_scanner_stopped("AI Worker"):
            upsert_scanner_health("AI Worker", "STOPPED", today_alerts=processed_count, processed_count=processed_count, total_count=total_watch, error_msg="Stopped by Admin")
            time.sleep(60)
            continue

        from gemini_key_manager import get_active_gemini_key
        if not get_active_gemini_key():
            # [RULE 67 - FIX RATIONALE]: Updated warning and health message to 1-day blacklist.
            logger.warning("🚨 [AI WORKER DOWN] All Gemini API keys are blacklisted/exhausted for the next 1 day (24h). Marking AI Worker DOWN and sleeping 1h.")
            upsert_scanner_health("AI Worker", "DOWN", today_alerts=processed_count, processed_count=processed_count, total_count=total_watch, error_msg="DOWN: All Gemini API keys exhausted (1-day blacklist)")
            try:
                from database import insert_notification
                from push_service import send_push_to_all
                insert_notification("admin", "❌ AI WORKER DOWN", "All Gemini API keys are marked exhausted for the next 1 day. AI Worker marked DOWN.")
                send_push_to_all("❌ AI WORKER DOWN", "All Gemini API keys exhausted. AI Worker marked DOWN.")
            except Exception as notif_err:
                logger.warning(f"Failed to send AI key exhaustion notifications: {notif_err}")
            time.sleep(3600)
            continue

        now_ist = datetime.now(IST_ZONE)
        if not is_in_window(now_ist):
            win_desc = get_active_window_description(now_ist)
            logger.info(f"🤖 [AI WORKER] Outside active window ({win_desc}). Sleeping 300s...")
            upsert_scanner_health("AI Worker", "IDLE", today_alerts=processed_count, processed_count=processed_count, total_count=total_watch, error_msg=f"Outside active window ({win_desc})")
            time.sleep(300)
            continue

        try:
            stats_scan = run_ai_worker_scan_once()
            status = "IDLE"
            error_msg = f"Last: Finished | Total: {stats_scan.get('total_count', 'N/A')}"
            
            # Recalculate after running scan
            try:
                stats = get_ai_concall_stats(symbols)
                processed_count = stats.get("total_cached", 0)
            except Exception:
                pass
                
            upsert_scanner_health("AI Worker", status, last_success=datetime.now(IST_ZONE).isoformat(), today_alerts=processed_count, processed_count=processed_count, total_count=total_watch, error_msg=error_msg)
        except RuntimeError:
            # Already running manually
            pass
        except Exception as e:
            logger.exception(f"❌ [AI WORKER] Main loop crashed")
            upsert_scanner_health("AI Worker", "DOWN", error_msg=str(e))
            try:
                from database import insert_notification
                from push_service import send_push_to_all
                insert_notification("admin", f"❌ AI WORKER CRASHED (DOWN)", f"Error: {str(e)[:200]}")
                send_push_to_all("❌ AI WORKER DOWN", f"Crash: {str(e)[:100]}")
            except Exception as outer_e:
                logger.exception(f"Failed to send crash notifications: {outer_e}")
            
        time.sleep(300)

def start_worker():
    """Starts the AI worker in a daemon thread."""
    thread = threading.Thread(target=run_worker_loop, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_worker_loop()
