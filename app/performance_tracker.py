# =====================================================================================
# app/performance_tracker.py
# Builds performance_data.json from the Postgres alerts table + live yfinance prices.
# Called every 5 minutes from main.py.
#
# SL / TARGET DETECTION LOGIC
# ────────────────────────────
# Both SL and Target are detected using intraday (1h) bars filtered to >= alert_time.
# This means:
#   • Any low printed BEFORE the alert on the same day is IGNORED for SL.
#   • Any high printed BEFORE the alert on the same day is IGNORED for Target.
#
# Priority:
#   1. SL hit first  → status = LOSS  (locked at stop_loss price)
#   2. Target hit first → status = WIN  (locked at target_price)
#   3. Neither hit   → mark-to-market vs current close
#
# To determine which hit first, we compare the timestamps of the first SL-breach
# candle and the first Target-breach candle.
# =====================================================================================

import os
import math
import json
import logging
import time
import pandas as pd
from typing import Union, Optional, Tuple
from datetime import datetime, date, timedelta, time as time_cls
from zoneinfo import ZoneInfo
from price_cache import fetch_watchlist_data



from config import MIN_STOCK_PRICE
from database import get_all_alerts, update_alert_outcome, upsert_scanner_health, save_system_state

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

try:
    from config import DATA_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

PERF_JSON_PATH = os.path.join(DATA_DIR, "performance_data.json")

# Time-based auto-exit is disabled; holdings are kept open until SL or Target is hit.
# HOLD_DAYS = 5


# =====================================================================================
# HELPERS
# =====================================================================================

from data_provider import get_fetcher

def _parse_dedup_key(breakout_type: str) -> tuple[str, str, str]:
    parts = breakout_type.split("|")
    if len(parts) >= 4:
        return parts[0].strip(), parts[1].strip(), parts[3].strip()
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), "UNKNOWN"
    return "Unknown", breakout_type, "UNKNOWN"


def _fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """Batch-fetch latest prices using Fyers with Yahoo fallback.

    [RULE 67 CHANGE-RATIONALE]: Added Tier-3 parquet disk cache fallback.
    When live providers (Fyers/Upstox) AND the DB CMP table both fail to return
    a price for a symbol (e.g. HEG stuck in the 30-min dead_symbols_cache after a
    transient provider failure), we read the last known Close from the daily 1D
    parquet cache file as a stale-price last resort.
    This prevents the recurring '🚨 No live price available' ERROR spam from firing
    for stocks that are correctly held open positions but temporarily unavailable
    from live quote APIs. The stale price is logged as WARNING (not ERROR) to clearly
    distinguish it from a true live market price.
    """
    if not symbols:
        return {}

    from live_prices import get_live_prices
    prices = get_live_prices(symbols)

    try:
        from data_fetch_status import mark_success
        mark_success('performance_tracker')
    except Exception:
        pass

    # ── Tier 3: Disk Parquet Last-Known-Close Fallback ───────────────────────
    # For any symbol still missing after live providers returned nothing,
    # try to read the last Close from the daily history parquet cache.
    missing_syms = [s for s in symbols if prices.get(s) is None]
    if missing_syms:
        try:
            from config import DATA_DIR
            history_dir = os.path.join(DATA_DIR, "history", "1d")
            for sym in missing_syms:
                clean = str(sym).upper().replace(".NS", "").replace(".BO", "")
                # Try symbol name variants matching price_cache.py resolution conventions
                candidates = [
                    clean,
                    clean.replace("&", "_"),
                    clean.replace("-", "_"),
                    clean.replace("&", "-"),
                ]
                for variant in candidates:
                    fpath = os.path.join(history_dir, f"{variant}.parquet")
                    if os.path.exists(fpath):
                        try:
                            df_cached = pd.read_parquet(fpath)
                            if not df_cached.empty and "Close" in df_cached.columns:
                                last_close = float(df_cached["Close"].dropna().iloc[-1])
                                if last_close > 0:
                                    prices[sym] = last_close
                                    logger.warning(
                                        f"⚠️ [PERF_TRACKER] {sym}: Live quote unavailable — "
                                        f"using last-known close ₹{last_close:.2f} from disk parquet cache. "
                                        f"Evaluation will proceed with stale price (safe — no false SL hit)."
                                    )
                                    break
                        except Exception as read_err:
                            logger.debug(f"[PERF_TRACKER] Parquet fallback read error for {sym}: {read_err}")
                        break  # found file path, stop searching variants
        except Exception as parquet_err:
            logger.debug(f"[PERF_TRACKER] Parquet disk fallback failed: {parquet_err}")

    return prices


def _fetch_post_alert_bars(symbol: str, alert_time_val: Union[str, datetime], prefetched_hist: pd.DataFrame = None) -> Optional[pd.DataFrame]:
    """
    Fetch 5m bars for *symbol* from the alert date to today using DataFetcher.
    Filters out any bars that occurred before the alert_time.
    """
    try:
        if isinstance(alert_time_val, datetime):
            # If it's timezone aware, convert to IST.
            if alert_time_val.tzinfo is not None:
                alert_dt_ist = alert_time_val.astimezone(IST)
            else:
                # Naive datetime from DB — treat as IST (our DB session is SET TIME ZONE 'Asia/Kolkata')
                alert_dt_ist = alert_time_val.replace(tzinfo=IST)
        else:
            alert_time_str = str(alert_time_val).replace("Z", "+00:00")
            alert_dt_naive = datetime.fromisoformat(alert_time_str)
            if alert_dt_naive.tzinfo is not None:
                alert_dt_ist = alert_dt_naive.astimezone(IST)
            else:
                alert_dt_ist = alert_dt_naive.replace(tzinfo=IST)

        alert_date = alert_dt_ist.date()

        # If the alert was recorded after market close (e.g. delayed run or EOD),
        # the entry is effectively the next trading day. We advance to the next day at 09:15
        # instead of replacing the time on the same day (which would incorrectly test that day's intraday dips).
        if alert_dt_ist.time() >= time_cls(15, 30):
            alert_dt_ist = (alert_dt_ist + timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)


        # Guard: if alert is from today and market hasn't opened yet (before 09:15 IST),
        # no 5m bars exist — return None immediately to avoid yfinance "delisted" noise.
        now_ist = datetime.now(IST)
        market_open_ist = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        if alert_date == now_ist.date() and now_ist < market_open_ist:
            logger.debug(f"⏳ {symbol} | Alert is from today but market not open yet — skipping bar fetch")
            return None

        # Determine period since alert
        days_since = (now_ist.date() - alert_date).days
        period_days = max(days_since + 2, 5)
        period_str = f"{period_days}d"

        # Yahoo Finance limits: 5m data is only available for the last 60 days.
        # We use 5m for maximum precision on SL/Target tracking if possible.
        if period_days <= 59:
            interval = "5m"
        elif period_days <= 720:
            interval = "1h"
        else:
            interval = "1d"

        if prefetched_hist is not None:
            if isinstance(prefetched_hist, pd.DataFrame) and not prefetched_hist.empty:
                hist = prefetched_hist.copy()
            else:
                return None
        else:
            # Route fallback through global cache
            df_request = pd.DataFrame({"Stock": [symbol]})
            raw_dict = fetch_watchlist_data(df_request, interval=interval, period=period_str, requester="performance_tracker")
            hist = raw_dict.get(symbol)

        from core_enums import ProviderResult
        if hist is None or isinstance(hist, ProviderResult) or hist.empty:
            return None

        if not {"High", "Low", "Close"}.issubset(hist.columns):
            return None

        # Find datetime column
        date_col = next((c for c in ["Datetime", "Date", "index"] if c in hist.columns), None)
        if date_col is None:
            if not isinstance(hist.index, pd.DatetimeIndex):
                return None
        else:
            hist[date_col] = pd.to_datetime(hist[date_col])
            hist = hist.set_index(date_col)

        # Localise index to IST
        idx = hist.index
        if idx.tz is None:
            idx = idx.tz_localize("Asia/Kolkata")
        else:
            idx = idx.tz_convert("Asia/Kolkata")
        hist.index = idx

        # Drop all candles that opened before the alert timestamp
        hist = hist[hist.index >= alert_dt_ist].copy()

        return hist if not hist.empty else None

    except Exception as e:
        logger.exception(f"⚠️ Could not fetch bars for {symbol} (alert={alert_time_val}): {e}")
        try:
            from data_fetch_status import mark_failure
            mark_failure('performance_tracker', f"{symbol} bars fetch failed: {str(e)}")
        except Exception:
            pass
        return None


import json

def process_trade_history(t: dict, hist: pd.DataFrame, cur_p: float):
    """
    State Machine Evaluator for Partial Exits (Full Replay Architecture).
    Walks forward through historical ticks from alert creation and executes trailing SLs and limits.
    Deduplicates database writes by checking existing exit_history.
    Adjusts cost basis and SL/targets for any stock splits or bonus corporate actions.
    """
    from database import update_partial_exit, update_alert_outcome
    from corporate_actions import adjust_trade_for_corporate_actions
    import json
    from datetime import datetime
    import pandas as pd

    # Adjust trade for stock splits / bonus issues prior to evaluating exits
    adjust_trade_for_corporate_actions(t)

    t1 = t.get("target_1")
    t2 = t.get("target_2")
    t3 = t.get("target_3")

    if not t1 and t.get("target_price"):
        # Legacy alert fallback
        t1 = t.get("target_price")
        t2 = t1 * 1.05
        t3 = t1 * 1.10

    if not t1: return  # Sanity check

    shares_bought = t.get("shares_bought", 0)
    if shares_bought == 0: return

    # Load existing DB events to avoid duplicate writes
    eh = t.get("exit_history")
    existing_hist = eh if isinstance(eh, list) else json.loads(eh or "[]")
    db_events = {e.get("type") for e in existing_hist}

    scanner = t.get("scanner", "LIVE_1H")
    symbol = t.get("symbol", "UNKNOWN")
    from config import EXIT_PROFILES, SCANNER_EXIT_PROFILE
    exit_profile_name = SCANNER_EXIT_PROFILE.get(scanner, "BALANCED")
    exit_config_dict = EXIT_PROFILES.get(exit_profile_name, EXIT_PROFILES["BALANCED"])
    exit_config = [exit_config_dict["t1"], exit_config_dict["t2"], exit_config_dict["t3"]]

    execution_state = t.get("execution_state")
    if not execution_state:
        execution_state = "OPEN"

    entry_mode = t.get("entry_mode", "LEGACY_UNKNOWN")
    actual_entry_price = t.get("actual_entry_price")

    structural_failure_stop = t.get("structural_failure_stop")

    # ── Initialize State ───────────────────────────────────────────────────────
    if hist is not None:
        # Full Replay Mode: Reset state to beginning of time
        initial_sl = t.get("initial_stop_loss")
        if not initial_sl or initial_sl == 0:
            t["stop_loss"] = t.get("stop_loss")
        else:
            t["stop_loss"] = initial_sl

        t["status"] = "OPEN" if execution_state != "PENDING_ENTRY" else "PENDING_ENTRY"
        t["remaining_shares"] = shares_bought
        hist_list = []
        t["exit_history"] = "[]"
    else:
        # Fast Mode: Preserve current DB state and history
        hist_list = existing_hist

    # Build sequence of all historical ticks (from alert creation) + live price
    ticks = []
    if hist is not None and not hist.empty:
        # [RULE 67 CHANGE-RATIONALE: CRITICAL WEEKEND CANDLE BAN]
        # Saturday and Sunday candles must NEVER be accepted, evaluated, or used for performance/P&L.
        from trading_calendar import enforce_trading_day_candles
        hist = enforce_trading_day_candles(hist, symbol)
        # Prevent Fyers API glitches from causing time-travel by ensuring chronological order and no duplicates
        hist = hist[~hist.index.duplicated(keep='first')].sort_index()
        for ts, row in hist.iterrows():
            ts_dt = pd.to_datetime(ts)
            if ts_dt.weekday() >= 5:
                continue
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            vol = float(row.get("Volume", 0.0))
            close_p = float(row.get("Close", float(row["High"])))
            ticks.append((ts_str, float(row["Open"]), float(row["Low"]), float(row["High"]), close_p, vol))
    if cur_p:
        now_dt = datetime.now(IST)
        # Strictly prohibit weekend candles from current price injection
        if now_dt.weekday() < 5:
            now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            ticks.append((now_str, cur_p, cur_p, cur_p, cur_p, 0.0))
    # ── State Validation & Fallback ──
    if execution_state == "OPEN" and actual_entry_price is None:
        if t.get("entry_price") is not None:
            actual_entry_price = t["entry_price"]
            t["actual_entry_price"] = actual_entry_price
            # Async backfill in DB to fix data integrity
            try:
                from database import get_connection
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE alerts SET actual_entry_price = %s WHERE id = %s AND actual_entry_price IS NULL", (actual_entry_price, t["id"]))
            except Exception:
                pass
        else:
            logger.error(f"❌ [PERF_TRACKER] DATA_INTEGRITY_ERROR: {symbol} is OPEN but missing actual_entry_price and entry_price. Safe-rejecting evaluation.")
            return
    if execution_state == "PENDING_ENTRY" and actual_entry_price is not None:
        logger.error(f"❌ [PERF_TRACKER] DATA_INTEGRITY_ERROR: {symbol} is PENDING_ENTRY but has actual_entry_price populated. Safe-rejecting evaluation.")
        return

    for ts_str, open_p, low, high, close_p, vol in ticks:
        # [RULE 67 CHANGE-RATIONALE: CRITICAL WEEKEND CANDLE BAN]
        # Never evaluate entry, exit, or calculate P&L from a Saturday or Sunday candle.
        t_tick = pd.to_datetime(ts_str)
        if t_tick.weekday() >= 5:
            continue

        if t["status"] in ("WIN", "LOSS", "CLOSED", "REJECTED"):
            break

        sl = t["stop_loss"]
        status = t["status"]
        # [VERSION: NULL_REM_SHARES_HOTFIX_v1.0] Defensive fallback for NULL remaining_shares
        rem_shares = t.get("remaining_shares") if t.get("remaining_shares") is not None else t.get("shares_bought", 0)

        # ── 1. ENTRY TRIGGER ──────────────────────────────────────────────────
        # CONSERVATIVE SAME-BAR COLLISION POLICY:
        # Daily OHLC cannot distinguish chronological order of high vs low.
        # If both entry trigger (high) and SL (low) are hit on the same day:
        # We process entry FIRST, then allow SL to evaluate on the SAME bar.
        # This guarantees we assume the worst-case scenario (taking the loss).
        if execution_state == "PENDING_ENTRY":
            is_triggered = False
            fill_price = None

            if entry_mode == "BREAKOUT_TRIGGER":
                if high >= t["entry_price"]:
                    is_triggered = True
                    fill_price = max(t["entry_price"], open_p)
            elif entry_mode == "LIMIT_PULLBACK":
                if low <= t["entry_price"]:
                    is_triggered = True
                    fill_price = min(t["entry_price"], open_p)
            elif entry_mode == "MARKET" or entry_mode == "LEGACY_UNKNOWN":
                logger.error(f"❌ [PERF_TRACKER] INVALID_ENTRY_STATE: {symbol} has entry_mode={entry_mode} but execution_state=PENDING_ENTRY. Safe-rejecting transition to OPEN.")
                # Flag as invalid and continue without transitioning
                continue

            if is_triggered and fill_price is not None:
                # The condition happened! Transition to OPEN and continue evaluation.
                execution_state = "OPEN"
                t["status"] = "OPEN"
                t["execution_state"] = "OPEN"
                t["actual_entry_price"] = fill_price
                actual_entry_price = fill_price
                if "ENTRY_TRIGGERED" not in db_events:
                    try:
                        from database import get_connection
                        with get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("UPDATE alerts SET execution_state = 'OPEN', status = 'OPEN', actual_entry_price = %s WHERE id = %s", (fill_price, t["id"]))
                    except Exception as e:
                        logger.error(f"❌ [PERF_TRACKER] DB UPDATE FAILED for ENTRY_TRIGGERED {symbol}: {e}")
                # Add an entry event
                event = {"type": "ENTRY_TRIGGERED", "price": fill_price, "shares": rem_shares, "pnl": 0.0, "time": ts_str}
                hist_list.append(event)
                t["exit_history"] = json.dumps(hist_list)
            else:
                # Still waiting for entry condition, skip all exit logic for this tick
                continue

        effective_entry = t.get("actual_entry_price") or t["entry_price"]

        # 0. Gap Risk Tracking
        if execution_state == "OPEN" and open_p < sl:
            exit_p = open_p
            pnl_rs_event = rem_shares * (exit_p - effective_entry)
            event = {"type": "GAP_LOSS", "price": exit_p, "shares": rem_shares, "pnl": round(pnl_rs_event, 2), "time": ts_str}

            hist_list.append(event)
            total_pnl_rs = sum(e["pnl"] for e in hist_list)
            cap = t.get("capital_allocated") or (float(effective_entry or 0.0) * float(t.get("shares_bought") or 0.0))
            total_pnl_pct = round((total_pnl_rs / cap) * 100, 2) if cap > 0 else 0.0

            execution_state = "GAP_LOSS"
            final_status = "LOSS"

            if "GAP_LOSS" not in db_events:
                update_partial_exit(t["id"], final_status, sl, rem_shares, 0, pnl_rs_event, event, execution_state)
                update_alert_outcome(t["id"], final_status, exit_p, total_pnl_pct, pnl_rs=total_pnl_rs, closed_at=ts_str, exit_signal="GAP_LOSS", execution_state=execution_state)

            t["status"] = final_status
            t["exit_price"] = exit_p
            t["exit_signal"] = "GAP_LOSS"
            t["exit_reason"] = "GAP_LOSS"
            t["remaining_shares"] = 0
            t["exit_history"] = json.dumps(hist_list)
            t["pnl_pct"] = total_pnl_pct
            t["pnl_rs"] = total_pnl_rs
            t["stopped_out"] = True
            t["closed_at"] = ts_str
            t["execution_state"] = execution_state
            continue

        # 1. Evaluate Stop Loss First (Protective)
        if low <= sl:


            exit_p = open_p if open_p < sl else sl
            pnl_rs_event = rem_shares * (exit_p - effective_entry)
            event = {"type": "SL_HIT", "price": exit_p, "shares": rem_shares, "pnl": round(pnl_rs_event, 2), "time": ts_str}

            final_status = "WIN" if "PARTIAL" in status else "LOSS"
            execution_state = "SL_HIT"

            hist_list.append(event)
            total_pnl_rs = sum(e["pnl"] for e in hist_list)
            cap = t.get("capital_allocated") or (float(effective_entry or 0.0) * float(t.get("shares_bought") or 0.0))
            total_pnl_pct = round((total_pnl_rs / cap) * 100, 2) if cap > 0 else 0.0

            if "SL_HIT" not in db_events:
                update_partial_exit(t["id"], final_status, sl, rem_shares, 0, pnl_rs_event, event, execution_state)
                update_alert_outcome(t["id"], final_status, exit_p, total_pnl_pct, pnl_rs=total_pnl_rs, closed_at=ts_str, exit_signal="STOP_LOSS", execution_state=execution_state)

            t["status"] = final_status
            t["exit_price"] = exit_p
            t["exit_signal"] = "STOP_LOSS"
            t["exit_reason"] = "STOP_LOSS"
            t["remaining_shares"] = 0
            t["exit_history"] = json.dumps(hist_list)
            t["pnl_pct"] = total_pnl_pct
            t["pnl_rs"] = total_pnl_rs
            t["stopped_out"] = True
            t["closed_at"] = ts_str
            t["execution_state"] = execution_state
            continue

        # 1.5 Evaluate Structural Failure Stop (Closing Basis Early Exit)
        if structural_failure_stop and close_p <= structural_failure_stop:
            exit_p = close_p
            pnl_rs_event = rem_shares * (exit_p - effective_entry)
            event = {"type": "STRUCT_FAIL", "price": exit_p, "shares": rem_shares, "pnl": round(pnl_rs_event, 2), "time": ts_str}

            final_status = "WIN" if "PARTIAL" in status else "LOSS"
            execution_state = "STRUCT_FAIL"

            hist_list.append(event)
            total_pnl_rs = sum(e["pnl"] for e in hist_list)
            cap = t.get("capital_allocated") or (float(effective_entry or 0.0) * float(t.get("shares_bought") or 0.0))
            total_pnl_pct = round((total_pnl_rs / cap) * 100, 2) if cap > 0 else 0.0

            if "STRUCT_FAIL" not in db_events:
                update_partial_exit(t["id"], final_status, sl, rem_shares, 0, pnl_rs_event, event, execution_state)
                update_alert_outcome(t["id"], final_status, exit_p, total_pnl_pct, pnl_rs=total_pnl_rs, closed_at=ts_str, exit_signal="STRUCTURAL_FAIL", execution_state=execution_state)

            t["status"] = final_status
            t["exit_price"] = exit_p
            t["exit_signal"] = "STRUCTURAL_FAIL"
            t["exit_reason"] = "STRUCTURAL_FAIL"
            t["remaining_shares"] = 0
            t["exit_history"] = json.dumps(hist_list)
            t["pnl_pct"] = total_pnl_pct
            t["pnl_rs"] = total_pnl_rs
            t["stopped_out"] = True
            t["closed_at"] = ts_str
            t["execution_state"] = execution_state
            continue

        status = t["status"]
        # 2. Evaluate T1
        if status == "OPEN" and high >= t1:
            exit_p = open_p if open_p > t1 else t1
            if not t.get("target_1") or not t2:
                # If there's no explicitly defined target array or if T2 is missing, sell everything at T1
                shares_to_sell = rem_shares
            else:
                shares_to_sell = int(shares_bought * (exit_config[0] / 100.0))
                if shares_to_sell == 0: shares_to_sell = rem_shares

            pnl_rs_event = shares_to_sell * (exit_p - effective_entry)
            event = {"type": "T1_HIT", "price": exit_p, "shares": shares_to_sell, "pnl": round(pnl_rs_event, 2), "time": ts_str}

            new_rem = rem_shares - shares_to_sell
            # Monotonic Ratchet Invariant: Stop Loss never loosens/moves backward
            new_sl = round(max(float(sl or 0.0), float(effective_entry * 1.003)), 2) if new_rem > 0 else sl  # Breakeven + 0.3% cost buffer
            new_status = "PARTIAL_WIN_1"
            execution_state = "PARTIAL_1_HIT"

            hist_list.append(event)
            t["exit_history"] = json.dumps(hist_list)
            t["status"] = new_status
            t["stop_loss"] = new_sl
            t["remaining_shares"] = new_rem
            t["execution_state"] = execution_state

            if "T1_HIT" not in db_events:
                update_partial_exit(t["id"], new_status, new_sl, shares_to_sell, new_rem, pnl_rs_event, event, execution_state)

            if new_rem <= 0:
                t["status"] = "WIN"
                execution_state = "WIN"
                total_pnl_rs = sum(e["pnl"] for e in hist_list)
                cap = t.get("capital_allocated") or (float(effective_entry or 0.0) * float(t.get("shares_bought") or 0.0))
                p_pct = round((total_pnl_rs / cap) * 100, 2) if cap > 0 else 0.0
                t["pnl_pct"] = p_pct
                t["pnl_rs"] = total_pnl_rs
                t["target_hit"] = True
                t["closed_at"] = ts_str
                t["exit_price"] = exit_p
                t["execution_state"] = execution_state
                if "T1_HIT" not in db_events:
                    update_alert_outcome(t["id"], "WIN", exit_p, p_pct, pnl_rs=total_pnl_rs, closed_at=ts_str, exit_signal="TARGET_HIT", execution_state=execution_state)
                continue

        status = t["status"]
        # 3. Evaluate T2
        if t2 and status == "PARTIAL_WIN_1" and high >= t2:
            exit_p = open_p if open_p > t2 else t2

            if not t3:
                # If there is no T3 (e.g. MF scanner), sell everything remaining at T2
                shares_to_sell = rem_shares
            else:
                shares_to_sell = int(shares_bought * (exit_config[1] / 100.0))
                if shares_to_sell > rem_shares: shares_to_sell = rem_shares
                if shares_to_sell == 0: shares_to_sell = rem_shares

            pnl_rs_event = shares_to_sell * (exit_p - effective_entry)
            event = {"type": "T2_HIT", "price": exit_p, "shares": shares_to_sell, "pnl": round(pnl_rs_event, 2), "time": ts_str}

            new_rem = rem_shares - shares_to_sell
            # Monotonic Ratchet Invariant: Stop Loss never loosens/moves backward
            new_sl = max(float(sl or 0.0), float(t1)) if new_rem > 0 else sl  # Raise to T1 only if remaining shares exist
            new_status = "PARTIAL_WIN_2"
            execution_state = "PARTIAL_2_HIT"

            hist_list.append(event)
            t["exit_history"] = json.dumps(hist_list)
            t["status"] = new_status
            t["stop_loss"] = new_sl
            t["remaining_shares"] = new_rem
            t["execution_state"] = execution_state

            if "T2_HIT" not in db_events:
                update_partial_exit(t["id"], new_status, new_sl, shares_to_sell, new_rem, pnl_rs_event, event, execution_state)

            if new_rem <= 0:
                t["status"] = "WIN"
                execution_state = "WIN"
                total_pnl_rs = sum(e["pnl"] for e in hist_list)
                cap = t.get("capital_allocated") or (float(effective_entry or 0.0) * float(t.get("shares_bought") or 0.0))
                p_pct = round((total_pnl_rs / cap) * 100, 2) if cap > 0 else 0.0
                t["pnl_pct"] = p_pct
                t["pnl_rs"] = total_pnl_rs
                t["target_hit"] = True
                t["closed_at"] = ts_str
                t["exit_price"] = exit_p
                t["execution_state"] = execution_state
                if "T2_HIT" not in db_events:
                    update_alert_outcome(t["id"], "WIN", exit_p, p_pct, pnl_rs=total_pnl_rs, closed_at=ts_str, exit_signal="TARGET_HIT", execution_state=execution_state)
                continue

        status = t["status"]
        # 4. Evaluate T3 (Final Target)
        if t3 and status == "PARTIAL_WIN_2" and high >= t3:
            exit_p = open_p if open_p > t3 else t3
            shares_to_sell = rem_shares

            pnl_rs_event = shares_to_sell * (exit_p - effective_entry)
            event = {"type": "T3_HIT", "price": exit_p, "shares": shares_to_sell, "pnl": round(pnl_rs_event, 2), "time": ts_str}

            hist_list.append(event)
            t["exit_history"] = json.dumps(hist_list)
            t["status"] = "WIN"
            execution_state = "WIN"
            t["remaining_shares"] = 0

            total_pnl_rs = sum(e["pnl"] for e in hist_list)
            cap = t.get("capital_allocated") or (float(effective_entry or 0.0) * float(t.get("shares_bought") or 0.0))
            p_pct = round((total_pnl_rs / cap) * 100, 2) if cap > 0 else 0.0
            t["pnl_pct"] = p_pct
            t["pnl_rs"] = total_pnl_rs
            t["target_hit"] = True
            t["closed_at"] = ts_str
            t["exit_price"] = exit_p
            t["execution_state"] = execution_state

            if "T3_HIT" not in db_events:
                update_partial_exit(t["id"], "WIN", t2, shares_to_sell, 0, pnl_rs_event, event, execution_state)
                update_alert_outcome(t["id"], "WIN", exit_p, p_pct, pnl_rs=total_pnl_rs, closed_at=ts_str, exit_signal="TARGET_HIT", execution_state=execution_state)
            continue

    # ── Calculate Alert Quality Metrics (Wave 1) ──
    # Excursion metrics must be computed strictly post-entry.
    entry_ts_val = None
    entry_idx = None
    exit_ts_val = None
    exit_idx = None

    for idx, (ts_str, open_p, low, high, close_p, vol) in enumerate(ticks):
        if entry_mode in ("BREAKOUT_TRIGGER", "LIMIT_PULLBACK") and execution_state != "PENDING_ENTRY":
            if (entry_mode == "BREAKOUT_TRIGGER" and high >= t.get("entry_price", 0)) or \
               (entry_mode == "LIMIT_PULLBACK" and low <= t.get("entry_price", 999999)):
                entry_ts_val = ts_str
                entry_idx = idx
                break
        elif entry_mode in ("MARKET", "LEGACY_UNKNOWN"):
            entry_ts_val = ticks[0][0]
            entry_idx = 0
            break

    if entry_idx is not None:
        exit_ts_val = t.get("closed_at")
        if exit_ts_val:
            for idx, (ts_str, open_p, low, high, close_p, vol) in enumerate(ticks):
                if ts_str == exit_ts_val:
                    exit_idx = idx
                    break
        if exit_idx is None:
            exit_idx = len(ticks) - 1
            exit_ts_val = ticks[-1][0]

        post_entry_ticks = ticks[entry_idx : exit_idx + 1]

        mfe_val = 0.0
        mae_val = 0.0

        entry_price_val = t.get("actual_entry_price") or t.get("entry_price", 1.0)
        sl_price_val = t.get("initial_stop_loss") or t.get("stop_loss", 0.0)
        risk_unit = abs(entry_price_val - sl_price_val)
        if risk_unit == 0:
            risk_unit = entry_price_val * 0.05

        for ts_str, open_p, low, high, close_p, vol in post_entry_ticks:
            # LONG position assumptions (can be adjusted for direction)
            fav_move = high - entry_price_val
            adv_move = entry_price_val - low

            mfe_val = max(mfe_val, fav_move)
            mae_val = max(mae_val, adv_move)

        mfe_pct = round((mfe_val / entry_price_val) * 100.0, 2)
        mae_pct = round((mae_val / entry_price_val) * 100.0, 2)
        mfe_r = round(mfe_val / risk_unit, 2)
        mae_r = round(mae_val / risk_unit, 2)

        t1_ts = None
        t2_ts = None
        t3_ts = None
        sl_ts = None

        for e in hist_list:
            e_type = e.get("type")
            e_time = e.get("time")
            if e_type == "T1_HIT":
                t1_ts = e_time
            elif e_type == "T2_HIT":
                t2_ts = e_time
            elif e_type == "T3_HIT":
                t3_ts = e_time
            elif e_type == "SL_HIT" or e_type == "GAP_LOSS":
                sl_ts = e_time

        bars_to_t1 = None
        bars_to_sl = None

        time_to_entry = None
        time_to_t1 = None
        time_to_t2 = None
        time_to_sl = None

        try:
            alert_dt = datetime.fromisoformat(t["alert_time"].replace(" IST", "").replace("Z", "+00:00")).replace(tzinfo=IST)
            if entry_ts_val:
                entry_dt = datetime.strptime(entry_ts_val, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                time_to_entry = (entry_dt - alert_dt).total_seconds()
            if t1_ts:
                t1_dt = datetime.strptime(t1_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                time_to_t1 = (t1_dt - alert_dt).total_seconds()
            if t2_ts:
                t2_dt = datetime.strptime(t2_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                time_to_t2 = (t2_dt - alert_dt).total_seconds()
            if sl_ts:
                sl_dt = datetime.strptime(sl_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                time_to_sl = (sl_dt - alert_dt).total_seconds()
        except Exception:
            pass

        if entry_idx is not None:
            if t1_ts:
                for idx, (ts_str, open_p, low, high, close_p, vol) in enumerate(ticks):
                    if ts_str == t1_ts:
                        bars_to_t1 = max(0, idx - entry_idx)
                        break
            if sl_ts:
                for idx, (ts_str, open_p, low, high, close_p, vol) in enumerate(ticks):
                    if ts_str == sl_ts:
                        bars_to_sl = max(0, idx - entry_idx)
                        break

        outcome_labels = {
            "A": bool(t1_ts is not None),
            "B": bool(t2_ts is not None),
            "C": bool(mfe_r >= 2.0 and mae_r < 1.0),
            "D": bool(t1_ts is not None and bars_to_t1 is not None and bars_to_t1 <= 5),
            "E": bool(sl_ts is not None and t1_ts is None)
        }

        weighted_exit_price = 0.0
        total_shares_sold = 0
        total_realized_pnl = 0.0

        for e in hist_list:
            if e.get("type") in ("T1_HIT", "T2_HIT", "T3_HIT", "SL_HIT", "GAP_LOSS", "STRUCT_FAIL"):
                sh_sold = e.get("shares", 0)
                exit_price_evt = e.get("price", 0.0)
                weighted_exit_price += exit_price_evt * sh_sold
                total_shares_sold += sh_sold
                total_realized_pnl += e.get("pnl", 0.0)

        if total_shares_sold > 0:
            weighted_exit_price = weighted_exit_price / total_shares_sold
            gross_realized_r = round((weighted_exit_price - entry_price_val) / risk_unit, 2)
            net_realized_r = round(gross_realized_r - (0.0015 * entry_price_val / risk_unit), 2)
            weighted_realized_r = gross_realized_r
        else:
            gross_realized_r = round((close_p - entry_price_val) / risk_unit, 2)
            net_realized_r = round(gross_realized_r - (0.0015 * entry_price_val / risk_unit), 2)
            weighted_realized_r = gross_realized_r

        quality_metrics = {
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "gross_realized_r": gross_realized_r,
            "net_realized_r": net_realized_r,
            "entry_timestamp": entry_ts_val,
            "exit_timestamp": exit_ts_val if t["status"] in ("WIN", "LOSS", "CLOSED", "GAP_LOSS") else None,
            "t1_timestamp": t1_ts,
            "t2_timestamp": t2_ts,
            "t3_timestamp": t3_ts,
            "sl_timestamp": sl_ts,
            "entry_bar_id": entry_idx,
            "exit_bar_id": exit_idx if t["status"] in ("WIN", "LOSS", "CLOSED", "GAP_LOSS") else None,
            "time_to_entry": time_to_entry,
            "time_to_t1": time_to_t1,
            "time_to_t2": time_to_t2,
            "time_to_sl": time_to_sl,
            "bars_to_t1": bars_to_t1,
            "bars_to_sl": bars_to_sl,
            "outcome_labels": outcome_labels,
            "weighted_realized_r": weighted_realized_r
        }

        try:
            from database import update_alert_quality_metrics
            update_alert_quality_metrics(t["id"], quality_metrics)
        except Exception as db_err:
            logger.error(f"❌ Failed to persist alert quality metrics: {db_err}")
def _days_held(alert_date_str: str) -> int:
    try:
        # [RULE 67 CHANGE-RATIONALE: CRITICAL WEEKEND CANDLE BAN]
        # Use official NSE trading calendar to compute trading days held, completely excluding weekends and holidays.
        from trading_calendar import default_trading_calendar
        return max(0, default_trading_calendar.days_between(date.fromisoformat(alert_date_str), datetime.now(IST).date()))
    except Exception:
        return 0


def _trade_status(
    pnl_pct: Optional[float],
    days: int,
    stopped_out: bool,
    target_hit: bool,
) -> str:
    if stopped_out:
        return "LOSS"
    if target_hit:
        return "WIN"
    # Keep positions open until SL or Target is hit (no time-based auto-exit)
    return "OPEN"


# =====================================================================================
# MAIN BUILD FUNCTION
# =====================================================================================

def build_performance_data(fast_mode=False, force_live_fetch=False, recalc_ids: list[int] = None, run_ctx=None):
    import time as _time
    _fn_start = _time.time()
    from perf_utils import ScannerStageTracker
    stage_tracker = ScannerStageTracker("PERFORMANCE_TRACKER")
    stage_tracker.start_stage(1, "Load Alerts from DB", "Fetching all alerts from database")
    trigger = "MANUAL_OVERRIDE" if force_live_fetch else "AUTO_BACKGROUND"
    logger.info("=" * 70)
    logger.info(f"📊 PERFORMANCE TRACKER | Building performance data... (Trigger: {trigger})")
    logger.info("=" * 70)

    # Re-evaluate all historical CLOSED / SELL_REVIEW trades across all scanners
    try:
        from revisit_closed_trades import audit_and_correct_closed_trades
        audit_and_correct_closed_trades()
    except Exception as _aud_err:
        logger.debug(f"Trade auditor warning: {_aud_err}")

    try:
        raw_alerts = get_all_alerts()
    except Exception:
        logger.exception("❌ Could not load alerts from database")
        _write_empty()
        return

    if not raw_alerts:
        logger.warning("⚠️ No alerts in database yet.")
        _write_empty()
        return

    logger.info(f"📋 {len(raw_alerts)} total alerts in database")
    stage_tracker.total_symbols = len(raw_alerts)
    if run_ctx:
        run_ctx.set_total_stocks(len(raw_alerts))
        run_ctx.record_fresh_data(len(raw_alerts))
    stage_tracker.end_stage(f"Loaded {len(raw_alerts)} alerts from DB")

    def _f(v):
        return float(v) if v is not None else None

    def _extract_exit_price(row_dict):
        ep = _f(row_dict.get("exit_price"))
        if ep is not None and ep > 0:
            return ep
        eh = row_dict.get("exit_history")
        if eh:
            try:
                eh_list = eh if isinstance(eh, list) else json.loads(eh)
                if eh_list and isinstance(eh_list, list):
                    last_evt = eh_list[-1]
                    if isinstance(last_evt, dict) and last_evt.get("price"):
                        return float(last_evt["price"])
            except Exception:
                pass
        return None

    def _extract_exit_reason(row_dict):
        sig = row_dict.get("exit_signal")
        if sig and not str(sig).startswith("⚠️ UNVERIFIED EARNINGS"):
            return str(sig)
        eh = row_dict.get("exit_history")
        if eh:
            try:
                eh_list = eh if isinstance(eh, list) else json.loads(eh)
                if eh_list and isinstance(eh_list, list):
                    last_evt = eh_list[-1]
                    if isinstance(last_evt, dict) and last_evt.get("type"):
                        return str(last_evt["type"])
            except Exception:
                pass
        return ""

    trades = []
    for row in raw_alerts:
        symbol      = row["symbol"]
        alert_time_raw = row.get("alert_time")
        if hasattr(alert_time_raw, "isoformat"):
            alert_time_str = alert_time_raw.isoformat()
        else:
            alert_time_str = str(alert_time_raw) if alert_time_raw else ""

        alert_date_raw = row.get("alert_date")
        if hasattr(alert_date_raw, "isoformat"):
            alert_date_str = alert_date_raw.isoformat()[:10]
        else:
            alert_date_str = str(alert_date_raw)[:10] if alert_date_raw else (alert_time_str[:10] if alert_time_str else "")

        alert_time  = alert_time_str or alert_time_raw
        alert_date  = alert_date_str

        entry_price = _f(row.get("entry_price"))

        cat_stored     = row.get("category")
        scanner_stored = row.get("scanner")
        sig_stored     = row.get("signals")

        category, signals, scanner = _parse_dedup_key(row["breakout_type"])
        if cat_stored:     category = cat_stored
        if scanner_stored: scanner  = scanner_stored
        if sig_stored:     signals  = sig_stored

        trades.append({
            "id":            row["id"],          # needed for write-back
            "symbol":        symbol,
            "scanner":       scanner,
            "category":      category,
            "signals":       signals,
            "entry_date":    alert_date,
            "alert_time":    alert_time,
            "entry_price":   entry_price,
            "stop_loss":     _f(row.get("stop_loss")),
            "initial_stop_loss": _f(row.get("initial_stop_loss")),
            "target_price":  _f(row.get("target_price")),
            "target_1":      _f(row.get("target_1")),
            "target_2":      _f(row.get("target_2")),
            "target_3":      _f(row.get("target_3")),
            "current_price": _f(row.get("current_price")),
            "exit_price":    _extract_exit_price(row),   # pre-filled if already closed
            "pnl_pct":       _f(row.get("pnl_pct")),      # pre-filled if already closed
            "stopped_out":   row.get("status") == "LOSS",
            "target_hit":    row.get("status") == "WIN",
            "days_held":     _days_held(alert_date),
            "status":        row.get("status") or "OPEN",
            "shares_bought": row.get("shares_bought", 0),
            "capital_allocated": _f(row.get("capital_allocated")),
            "pnl_rs":        _f(row.get("pnl_rs")),
            "score":         row.get("score"),
            "rsi":           _f(row.get("rsi")),
            "volume_ratio":  _f(row.get("volume_ratio")),
            "closed_at":     row.get("closed_at"),        # ISO timestamp when SL/Target locked
            "remaining_shares": row.get("remaining_shares") if row.get("remaining_shares") is not None else row.get("shares_bought", 0),
            "exit_history":  row.get("exit_history"),
            "context":       row.get("context"),          # Diagnostic filters and context
            "is_rejected":   row.get("is_rejected", False),
            "execution_state": row.get("execution_state"),
            "structural_failure_stop": _f(row.get("structural_failure_stop")),
            "exit_signal":   _extract_exit_reason(row),
            "exit_reason":   _extract_exit_reason(row),
            "trade_evolution_state": row.get("trade_evolution_state", "INITIAL"),
            "evidence_count": row.get("evidence_count", 1),
            "distinct_patterns_count": row.get("distinct_patterns_count", 1),
            "confirmation_quality": row.get("confirmation_quality", "INITIAL"),
            "last_event_type": row.get("last_event_type", "NEW_ENTRY"),
            "last_event_date": str(row.get("last_event_date") or ""),
            "_db_closed":    row.get("status") in ("WIN", "LOSS", "CLOSED"),  # internal flag
        })

    # ── 2. Fetch current prices ──────────────────────────────────────────────────────
    unique_symbols = list({t["symbol"] for t in trades})
    stage_tracker.start_stage(2, "Fetch Current Market Prices", f"Fetching live prices for {len(unique_symbols)} unique symbols")
    from market_utils import is_market_open
    is_open = is_market_open() or force_live_fetch

    # [RULE 67 CHANGE-RATIONALE]: Cooperative broker yielding. If a high-priority breakout scanner (e.g. MULTI_TF)
    # is actively executing market data downloads, briefly yield broker bandwidth to eliminate socket/rate contention.
    try:
        from lock_utils import is_scanner_fetch_active, wait_for_scanner_fetch_idle
        if is_scanner_fetch_active():
            logger.info("⏳ [PERF_TRACKER] Primary scanner fetch in progress. Cooperatively yielding broker bandwidth (up to 8s)...")
            wait_for_scanner_fetch_idle(timeout=8.0)
    except Exception:
        pass

    logger.info(f"📈 Fetching current/last-known prices for {len(unique_symbols)} symbols...")
    current_prices = _fetch_current_prices(unique_symbols) or {}

    # Fallback to stock_analysis_master CMP if live quote returns partial/empty dict
    if len(current_prices) < len(unique_symbols):
        try:
            from database import get_connection
            with get_connection() as _cmp_conn:
                with _cmp_conn.cursor() as _cmp_cur:
                    _cmp_cur.execute("SELECT symbol, cmp FROM stock_analysis_master WHERE symbol = ANY(%s) AND cmp IS NOT NULL", (unique_symbols,))
                    for row in _cmp_cur.fetchall():
                        if row[0] not in current_prices and row[1] is not None:
                            current_prices[row[0]] = float(row[1])
        except Exception as _db_cmp_err:
            logger.warning(f"Could not load fallback CMP from stock_analysis_master: {_db_cmp_err}")

    # [VERSION: CMP_MASTER_v1.0] Fetch CMP for ALL watchlist symbols and persist to stock_analysis_master.
    # This makes the master table the single source of truth for live prices across admin + user dashboards.
    if is_open:
        try:
            from database import get_connection, bulk_update_cmp
            watchlist_symbols = set()
            with get_connection() as _wconn:
                with _wconn.cursor() as _wcur:
                    _wcur.execute("""
                        SELECT DISTINCT symbol FROM user_watchlists WHERE symbol IS NOT NULL AND symbol != ''
                        UNION
                        SELECT DISTINCT symbol FROM breakout_watchlist WHERE is_active = TRUE AND symbol IS NOT NULL
                        UNION
                        SELECT DISTINCT symbol FROM daily_watchlist_v2 WHERE build_date = (SELECT MAX(build_date) FROM daily_watchlist_v2) AND symbol IS NOT NULL
                    """)
                    watchlist_symbols = {r[0] for r in _wcur.fetchall() if r[0]}
            # Symbols already in current_prices → reuse; new ones → fetch live
            extra_syms = list(watchlist_symbols - set(unique_symbols))
            if extra_syms:
                extra_prices = _fetch_current_prices(extra_syms)
                all_cmp = {**current_prices, **extra_prices}
            else:
                all_cmp = {sym: current_prices[sym] for sym in watchlist_symbols if sym in current_prices}
            if all_cmp:
                bulk_update_cmp(all_cmp)
                logger.info(f"💰 [CMP_MASTER] Updated CMP for {len(all_cmp)} universe/watchlist symbols in stock_analysis_master")
        except Exception as _cmp_err:
            logger.warning(f"⚠️ [CMP_MASTER] Could not update watchlist CMP: {_cmp_err}")


    stage_tracker.end_stage(f"Fetched prices for {len(current_prices)} symbols | market_open={is_open}")
    # ── 3. Per-trade SL + Target detection via post-alert intraday bars ─────────────
    stage_tracker.start_stage(3, "SL/Target Detection & Trade Processing", f"Processing {len(trades)} trades for SL/target hits")
    if is_open and not fast_mode:
        logger.info("📉 Checking SL / Target levels via post-alert intraday bars...")
    else:
        logger.info("⏭️ Skipping SL/Target intraday bar checks (fast_mode or market closed)")

    # [OPTIMIZATION] Pre-fetch all required intraday histories in big batches to avoid individual API hits
    fetch_groups = {}
    now_ist = datetime.now(IST)
    market_open_ist = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    for t in trades:
        if t["_db_closed"] or t["entry_price"] is None or not t["stop_loss"] or not (t.get("target_price") or t.get("target_1")) or not t["alert_time"]:
            continue

        if recalc_ids is not None and t["id"] not in recalc_ids:
            continue

        alert_time_val = t["alert_time"]
        if isinstance(alert_time_val, datetime):
            alert_dt_ist = alert_time_val.astimezone(IST) if alert_time_val.tzinfo else alert_time_val.replace(tzinfo=IST)
        else:
            alert_dt_naive = datetime.fromisoformat(str(alert_time_val).replace("Z", "+00:00"))
            alert_dt_ist = alert_dt_naive.astimezone(IST) if alert_dt_naive.tzinfo else alert_dt_naive.replace(tzinfo=IST)

        alert_date = alert_dt_ist.date()
        if alert_date == now_ist.date() and now_ist < market_open_ist:
            continue

        days_since = (now_ist.date() - alert_date).days
        period_days = max(days_since + 2, 5)
        period_str = f"{period_days}d"
        interval = "5m" if period_days <= 59 else ("1h" if period_days <= 720 else "1d")

        key = (interval, period_str)
        fetch_groups.setdefault(key, []).append(t["symbol"])

    prefetched_data = {}

    # USER DIRECTIVE: Tick-by-tick history fetch MUST ONLY happen for explicitly passed recalc_ids.
    # Normal performance loop just checks the current live price.
    do_tick_replay = (recalc_ids is not None and len(recalc_ids) > 0)

    if is_open and not do_tick_replay:
        logger.info(f"⚡ FAST EVALUATION: Processing open trades using live prices only (No historical replay).")

    if is_open and do_tick_replay and fetch_groups:
        for (interval, period_str), syms in fetch_groups.items():
            syms_preview = ",".join(syms[:5]) + ("..." if len(syms) > 5 else "")
            logger.info(f"📦 Pre-fetching batch history for {len(syms)} active trades [{syms_preview}] ({interval}/{period_str}) to prevent API spam...")
            df_request = pd.DataFrame({"Stock": syms})
            batch_res = fetch_watchlist_data(df_request, interval=interval, period=period_str, requester="performance_tracker")
            if batch_res:
                prefetched_data.update(batch_res)

    for t in trades:
        # [PERF] Yield the GIL so the Flask Dashboard server can handle incoming API requests
        # preventing 504 Gateway Timeouts during heavy performance tracking loops.
        time.sleep(0.01)

        from corporate_actions import adjust_trade_for_corporate_actions
        adjust_trade_for_corporate_actions(t)

        sym        = t["symbol"]
        ep         = t["entry_price"]
        sl         = t["stop_loss"]
        tp         = t["target_price"]
        alert_time = t["alert_time"]
        scanner    = t.get("scanner", "")

        # Attach real-time CMP for ALL trades (including MULTIBAGGER and WEALTH)
        cur_p = current_prices.get(sym)
        if cur_p is not None and cur_p > 0:
            t["current_price"] = round(cur_p, 2)
            if not t["_db_closed"]:
                # Calculate live unrealized P&L
                if ep and ep > 0:
                    live_pnl_pct = round(((cur_p - ep) / ep) * 100.0, 2)
                    t["pnl_pct"] = live_pnl_pct
                    sh_b = t.get("shares_bought") or 0
                    if sh_b > 0:
                        t["pnl_rs"] = round((cur_p - ep) * sh_b, 2)
                    elif t.get("capital_allocated"):
                        t["pnl_rs"] = round((live_pnl_pct / 100.0) * float(t["capital_allocated"]), 2)
                try:
                    from database import update_alert_current_price
                    update_alert_current_price(t["id"], cur_p)
                except Exception as e:
                    logger.warning(f"Failed to persist current_price for trade id {t['id']}: {e}")
        elif is_open and not t["_db_closed"]:
            # [VERSION: EXIT_MONITOR_MISSING_PRICE_FIX_v1.0]
            # Safely skip instead of triggering false SL hits if live price fails
            logger.error(f"🚨 [PERFORMANCE TRACKER] {sym}: No live price available. Skipping evaluation to prevent false exit.")
            try:
                from telegram_engine import queue_telegram_message
                msg = f"🚨 <b>Exit Monitor Error</b>\nUnable to fetch live price for {sym}. Skipping performance tracking evaluation to prevent false exit. Providers may be rate-limited."
                queue_telegram_message(msg, symbol=sym)
            except Exception:
                pass
            continue

        # ── Already closed in DB — no bar download needed ────────────────────────
        if t["_db_closed"]:
            # pnl_pct and exit_price already populated from DB above
            # Just refresh current_price for display; status stays locked
            logger.debug(f"⏭️  {sym} already closed ({t['status']}) — skipping bar fetch")
            continue

        # Long-term compounder positions (MULTIBAGGER, WEALTH) are managed exclusively
        # by their own fundamental exit monitors (e.g. evaluate_multibagger_exits).
        # Skip swing SL / target processing for them in performance_tracker.
        if scanner in ("MULTIBAGGER", "WEALTH", "Wealth Engine"):
            continue
            # pnl_pct and exit_price already populated from DB above
            # Just refresh current_price for display; status stays locked
            logger.debug(f"⏭️  {sym} already closed ({t['status']}) — skipping bar fetch")
            continue

        # ── Counterfactual Shadow Tracking for Rejected Trades ──────────────────────
        if t.get("is_rejected"):
            shadow_st = t.get("shadow_status") or "SHADOW_OPEN"
            if shadow_st not in ("SHADOW_WIN", "SHADOW_LOSS", "SHADOW_EXPIRED", "SHADOW_NEUTRAL") and cur_p and ep and sl:
                t1 = t.get("target_1") or t.get("target_price") or (ep * 1.05)
                sh_status = None
                sh_exit_p = None
                if cur_p <= sl:
                    sh_status = "SHADOW_LOSS"
                    sh_exit_p = sl
                elif cur_p >= t1:
                    sh_status = "SHADOW_WIN"
                    sh_exit_p = t1
                elif (t.get("days_held") or 0) >= 20:
                    sh_status = "SHADOW_EXPIRED"
                    sh_exit_p = cur_p

                if sh_status:
                    sh_pnl = round(((sh_exit_p - ep) / ep) * 100, 2)
                    t["shadow_status"] = sh_status
                    t["shadow_exit_price"] = sh_exit_p
                    t["shadow_pnl_pct"] = sh_pnl
                    from database import update_shadow_alert_outcome
                    update_shadow_alert_outcome(t["id"], sh_status, sh_exit_p, sh_pnl)
            continue

        # FIX: use `is None` (not falsy check) so ep=0.0 doesn't misfire.
        # When ep is None we cannot compute any P&L — mark status and move on.
        if ep is None:
            t["pnl_pct"] = None
            t["status"]  = _trade_status(None, t["days_held"], False, False)
            continue

        if sl and alert_time and (t.get("target_1") or t.get("target_price")):
            # ── V2 Multi-Stage Target & Trail Processing ─────────────────────────
            hist = None
            if is_open and do_tick_replay:
                if t["id"] in recalc_ids:
                    logger.info(f"🔄 Recalculating {sym} (Alert #{t['id']}) - Replaying historical ticks...")
                    pre_hist = prefetched_data.get(sym) if sym in prefetched_data else None
                    hist = _fetch_post_alert_bars(sym, alert_time, prefetched_hist=pre_hist)

            # If we are doing a historical replay (hist is populated), do NOT artificially append the current live price
            # as a tick. It can trigger trailing SLs at incorrect (current) timestamps.
            process_trade_history(t, hist, cur_p=None if (hist is not None) else cur_p)

        elif sl and alert_time:
            # SL only (no target stored — legacy or partial row)
            hist = None
            if do_tick_replay and (t["id"] in recalc_ids):
                hist = _fetch_post_alert_bars(sym, alert_time)
            if hist is not None and not hist.empty:
                lowest_low = float(hist["Low"].min())
                if lowest_low <= sl:
                    t["stopped_out"] = True
                    t["exit_price"]  = sl
                    t["exit_signal"] = "STOP_LOSS"
                    t["exit_reason"] = "STOP_LOSS"
                    t["pnl_pct"]     = round((sl - ep) / ep * 100, 2)
                    # Find the first candle that breached the Stop Loss
                    hit_row = hist[hist["Low"] <= sl]
                    hit_time = hit_row.index[0].strftime("%Y-%m-%d %H:%M:%S") if not hit_row.empty else None
                    t["pnl_rs"]      = t["shares_bought"] * (sl - ep) if t["shares_bought"] else 0.0
                    t["closed_at"]   = hit_time
                    update_alert_outcome(t["id"], "LOSS", sl, t["pnl_pct"], pnl_rs=t["pnl_rs"], closed_at=hit_time, exit_signal="STOP_LOSS")
                elif cur_p and cur_p <= sl:
                    t["stopped_out"] = True
                    t["exit_price"]  = sl
                    t["exit_signal"] = "STOP_LOSS"
                    t["exit_reason"] = "STOP_LOSS"
                    t["pnl_pct"]     = round((sl - ep) / ep * 100, 2)
                    t["pnl_rs"]      = t["shares_bought"] * (sl - ep) if t["shares_bought"] else 0.0
                    hit_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                    t["closed_at"]   = hit_time
                    logger.debug(f"🛑 {sym} SL HIT (LIVE) | entry={ep} sl={sl} pnl={t['pnl_pct']}%")
                    update_alert_outcome(t["id"], "LOSS", sl, t["pnl_pct"], pnl_rs=t["pnl_rs"], closed_at=hit_time, exit_signal="STOP_LOSS")
                elif cur_p:
                    t["pnl_pct"] = round((cur_p - ep) / ep * 100, 2)
            elif cur_p and cur_p <= sl:
                t["stopped_out"] = True
                t["exit_price"]  = sl
                t["exit_signal"] = "STOP_LOSS"
                t["exit_reason"] = "STOP_LOSS"
                t["pnl_pct"]     = round((sl - ep) / ep * 100, 2)
                t["pnl_rs"]      = t["shares_bought"] * (sl - ep) if t["shares_bought"] else 0.0
                hit_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                t["closed_at"]   = hit_time
                logger.debug(f"🛑 {sym} SL HIT (LIVE) | entry={ep} sl={sl} pnl={t['pnl_pct']}%")
                update_alert_outcome(t["id"], "LOSS", sl, t["pnl_pct"], pnl_rs=t["pnl_rs"], closed_at=hit_time, exit_signal="STOP_LOSS")
            elif cur_p:
                t["pnl_pct"] = round((cur_p - ep) / ep * 100, 2)

        elif cur_p:
            # Legacy alert — no SL/Target at all
            t["pnl_pct"] = round((cur_p - ep) / ep * 100, 2)

        t["status"] = _trade_status(
            t["pnl_pct"], t["days_held"], t["stopped_out"], t["target_hit"]
        )
        if t.get("is_rejected"):
            t["status"] = "REJECTED"

    stage_tracker.end_stage(f"Processed {len(trades)} trades")
    # ── 4. Summary stats ────────────────────────────────────────────────────────────
    stage_tracker.start_stage(4, "Summary Stats & Aggregation", "Computing win rate, P&L stats, scanner/category breakdowns")
    judged  = [t for t in trades if t["status"] in ("WIN", "LOSS", "NEUTRAL", "CLOSED")]
    winners = [t for t in judged if t["status"] == "WIN" or (t["status"] == "CLOSED" and (t.get("pnl_pct") or 0.0) > 0)]
    losers  = [t for t in judged if t["status"] == "LOSS" or (t["status"] == "CLOSED" and (t.get("pnl_pct") or 0.0) <= 0)]
    open_p  = [t for t in trades if t["status"] in ("OPEN", "SELL_REVIEW", "TRAILING")]

    pnls    = [t["pnl_pct"] for t in judged if t["pnl_pct"] is not None]
    win_pnl = [t["pnl_pct"] for t in winners if t["pnl_pct"] is not None]
    los_pnl = [t["pnl_pct"] for t in losers  if t["pnl_pct"] is not None]

    n_judged  = len(judged)
    wr        = round(len(winners) / n_judged * 100, 1) if n_judged else 0
    avg_ret   = round(sum(pnls) / len(pnls), 2)          if pnls     else 0
    avg_win   = round(sum(win_pnl) / len(win_pnl), 2)    if win_pnl  else 0
    avg_loss  = round(sum(los_pnl) / len(los_pnl), 2)    if los_pnl  else 0
    best      = round(max(pnls), 2)                       if pnls     else 0
    worst     = round(min(pnls), 2)                       if pnls     else 0
    expectancy = round((wr / 100) * avg_win + (1 - wr / 100) * avg_loss, 2)

    # SL vs Target breakdown
    sl_closed     = [t for t in judged if t["stopped_out"]]
    target_closed = [t for t in judged if t["target_hit"]]

    summary = {
        "total_alerts":      len(trades),
        "judged":            n_judged,
        "winners":           len(winners),
        "losers":            len(losers),
        "open_positions":    len(open_p),
        "sl_triggered":      len(sl_closed),
        "target_hit":        len(target_closed),
        "win_rate":          wr,
        "avg_return_pct":    avg_ret,
        "avg_win_pct":       avg_win,
        "avg_loss_pct":      avg_loss,
        "best_trade_pct":    best,
        "worst_trade_pct":   worst,
        "expectancy":        expectancy,
    }

    # ── 5. Equity curve ─────────────────────────────────────────────────────────────
    sorted_judged = sorted(judged, key=lambda t: t["entry_date"])
    cum = 0.0
    equity_curve = []
    for i, t in enumerate(sorted_judged):
        if t["pnl_pct"] is not None:
            cum += t["pnl_pct"]
            equity_curve.append({
                "date":              t["entry_date"],
                "symbol":            t["symbol"],
                "trade_return":      t["pnl_pct"],
                "cumulative_return": round(cum / (i + 1), 2),
                "close_reason":      "SL" if t["stopped_out"] else ("TARGET" if t["target_hit"] else "TIME"),
            })

    # ── 6. Monthly breakdown ────────────────────────────────────────────────────────
    mmap: dict[str, dict] = {}
    for t in judged:
        ed = t.get("entry_date")
        if isinstance(ed, (datetime, date)):
            m = ed.isoformat()[:7]
        else:
            m = str(ed)[:7]
        if m not in mmap:
            mmap[m] = {"alerts": 0, "wins": 0, "pnls": []}
        mmap[m]["alerts"] += 1
        if t["status"] == "WIN":
            mmap[m]["wins"] += 1
        if t["pnl_pct"] is not None:
            mmap[m]["pnls"].append(t["pnl_pct"])

    monthly = [
        {
            "month":    m,
            "alerts":   v["alerts"],
            "wins":     v["wins"],
            "win_rate": round(v["wins"] / v["alerts"] * 100, 1) if v["alerts"] else 0,
            "avg_return": round(sum(v["pnls"]) / len(v["pnls"]), 2) if v["pnls"] else 0,
        }
        for m in sorted(mmap)
        for v in [mmap[m]]
    ]

    # ── 7. By scanner ────────────────────────────────────────────────────────────────
    all_scanners = {t["scanner"] for t in trades}
    by_scanner = {}
    for sc in all_scanners:
        sc_judged = [t for t in judged  if t["scanner"] == sc]
        sc_wins   = [t for t in sc_judged if t["status"] == "WIN"]
        sc_pnls   = [t["pnl_pct"] for t in sc_judged if t["pnl_pct"] is not None]
        by_scanner[sc] = {
            "total":      len([t for t in trades if t["scanner"] == sc]),
            "judged":     len(sc_judged),
            "win_rate":   round(len(sc_wins) / len(sc_judged) * 100, 1) if sc_judged else 0,
            "avg_return": round(sum(sc_pnls) / len(sc_pnls), 2) if sc_pnls else 0,
        }

    # ── 8. By category ───────────────────────────────────────────────────────────────
    all_cats = {t["category"] for t in trades}
    by_category = {}
    for cat in all_cats:
        cat_judged = [t for t in judged  if t["category"] == cat]
        cat_wins   = [t for t in cat_judged if t["status"] == "WIN"]
        cat_pnls   = [t["pnl_pct"] for t in cat_judged if t["pnl_pct"] is not None]
        by_category[cat] = {
            "total":      len([t for t in trades if t["category"] == cat]),
            "judged":     len(cat_judged),
            "win_rate":   round(len(cat_wins) / len(cat_judged) * 100, 1) if cat_judged else 0,
            "avg_return": round(sum(cat_pnls) / len(cat_pnls), 2) if cat_pnls else 0,
        }

    # Strip internal tracking flag and massive exit_history before serialising
    for t in trades:
        t.pop("_db_closed", None)
        t.pop("exit_history", None)

    # ── 9. Write scanner health to Postgres (source of truth) ──────────────────
    today_str = datetime.now(IST).date().isoformat()
    for sc in all_scanners:
        sc_today = [t for t in trades if t["scanner"] == sc and t["entry_date"] == today_str]
        try:
            # We pass last_success=None so that the DB preserves the actual heartbeat
            # timestamps updated directly by the scanner loops.
            upsert_scanner_health(
                scanner_name  = sc,
                status        = None,
                last_success  = None,
                today_alerts  = len(sc_today),
                error_msg     = None,
            )
        except Exception:
            logger.warning(f"⚠️ Could not update scanner_health for {sc}")

    # ── 10. Write DB State ───────────────────────────────────────────────────────────
    try:
        from corporate_events import decorate_events
        trades = decorate_events(trades)
    except Exception as _ce_err:
        logger.debug(f"Corporate event decoration warning in performance tracker: {_ce_err}")

    payload = {
        "generated_at": datetime.now(IST).isoformat(),
        "summary":      summary,
        "trades":       sorted(trades, key=lambda t: t.get("entry_date", ""), reverse=True),
        "equity_curve": equity_curve,
        "monthly":      monthly,
        "by_scanner":   by_scanner,
        "by_category":  by_category,
    }

    try:
        import math
        def sanitize_nans(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            elif isinstance(obj, dict):
                return {k: sanitize_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_nans(item) for item in obj]
            return obj

        payload = sanitize_nans(payload)
        summary = sanitize_nans(summary)

        payload_str = json.dumps(payload, default=str)
        save_system_state("performance_summary", json.dumps(summary, default=str))
        save_system_state("performance_generated_at", json.dumps(payload["generated_at"]))
        save_system_state("performance_data", payload_str)
        try:
            from dashboard_server import invalidate_performance_cache
            invalidate_performance_cache()
        except Exception:
            pass
        logger.info("✅ PERFORMANCE TRACKER | Stored performance metrics in PostgreSQL")
    except Exception:
        logger.exception("❌ PERFORMANCE TRACKER | Failed to store performance metrics in DB")

    logger.info(
        f"✅ PERFORMANCE TRACKER | {len(trades)} alerts | "
        f"{len(winners)}W / {len(losers)}L / {len(open_p)} OPEN | "
        f"SL triggers={len(sl_closed)} | Target hits={len(target_closed)}"
    )
    try:
        stage_tracker.end_stage(f"Stats computed | WR={wr}% | AvgReturn={avg_ret}%")
        stage_tracker.print_summary(alerts_found=len(winners))
    except Exception:
        pass


# =====================================================================================
# EMPTY FALLBACK
# =====================================================================================

def _write_empty():
    payload = {
        "generated_at": datetime.now(IST).isoformat(),
        "trades":       [],
        "summary": {
            "total_alerts": 0, "judged": 0, "winners": 0, "losers": 0,
            "open_positions": 0, "sl_triggered": 0, "target_hit": 0,
            "win_rate": 0, "avg_return_pct": 0, "avg_win_pct": 0,
            "avg_loss_pct": 0, "best_trade_pct": 0, "worst_trade_pct": 0,
            "expectancy": 0,
        },
        "equity_curve": [],
        "monthly":      [],
        "by_scanner":   {},
        "by_category":  {},
        "scanner_stats": {},
    }
    try:
        payload_str = json.dumps(payload, default=str)
        save_system_state("performance_summary", json.dumps(payload["summary"], default=str))
        save_system_state("performance_generated_at", json.dumps(payload["generated_at"]))
        save_system_state("performance_data", payload_str)
        logger.info("✅ PERFORMANCE TRACKER | Stored empty performance metrics in PostgreSQL")
    except Exception:
        logger.exception("❌ PERFORMANCE TRACKER | Failed to store empty metrics in DB")


# =====================================================================================
# DEBUNCED ASYNCHRONOUS REBUILD
# =====================================================================================

# [RULE 67 CHANGE-RATIONALE]:
# Alerts must NEVER be lost or dropped by debounce cooldowns.
# When multiple alerts arrive in rapid succession (e.g., Pullback scanner saving REDINGTON, TFCILTD, MARKSANS),
# previous logic dropped calls inside the 45s cooldown on the floor with no trailing execution.
# This implementation adds a trailing-edge timer so that any alert batch is guaranteed to trigger
# a final rebuild as soon as the batch finishes, ensuring 100% of newly generated alerts are reflected.
import threading
_perf_rebuild_lock = threading.Lock()
_last_perf_rebuild_ts = 0.0
_perf_rebuild_cooldown = 15.0
_trailing_timer = None
_trailing_timer_lock = threading.Lock()

def trigger_performance_rebuild(recalc_ids: list[int] = None, force: bool = False):
    """
    Guaranteed execution debounced trigger for rebuilding performance data.
    Ensures that trailing alerts in a multi-stock batch are never dropped.
    """
    global _last_perf_rebuild_ts, _trailing_timer
    now = time.time()

    def _execute_rebuild():
        global _last_perf_rebuild_ts
        t_name = threading.current_thread().name
        if not _perf_rebuild_lock.acquire(blocking=False):
            logger.info("📈 PERFORMANCE TRACKER | Rebuild already running, scheduling trailing rebuild.")
            _schedule_trailing(delay=5.0)
            return
        try:
            _last_perf_rebuild_ts = time.time()
            logger.info(f"🚀 [BACKGROUND WORKER START] Worker='{t_name}' | Action='Rebuilding performance metrics & trade tracker'")
            _t_start = time.perf_counter()
            build_performance_data(force_live_fetch=True, recalc_ids=recalc_ids)
            dur_s = time.perf_counter() - _t_start
            logger.info(f"✅ [BACKGROUND WORKER COMPLETE] Worker='{t_name}' | Action='Performance metrics rebuild' | Duration={dur_s:.2f}s")
        except Exception as e:
            logger.exception(f"❌ [BACKGROUND WORKER FAIL] Worker='{t_name}' | Action='Performance rebuild failed' | Error={e}")
        finally:
            _perf_rebuild_lock.release()

    def _schedule_trailing(delay=None):
        global _trailing_timer
        with _trailing_timer_lock:
            if _trailing_timer is not None and _trailing_timer.is_alive():
                return
            wait_sec = delay if delay is not None else max(2.0, _perf_rebuild_cooldown - (time.time() - _last_perf_rebuild_ts))
            logger.info(f"📈 PERFORMANCE TRACKER | Scheduling trailing rebuild in {wait_sec:.1f}s to guarantee all batch alerts are included.")
            _trailing_timer = threading.Timer(wait_sec, _execute_rebuild)
            _trailing_timer.daemon = True
            _trailing_timer.start()

    if not force and recalc_ids is None and (now - _last_perf_rebuild_ts) < _perf_rebuild_cooldown:
        _schedule_trailing()
        return

    threading.Thread(target=_execute_rebuild, name="PerfRebuildThread", daemon=True).start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    build_performance_data()
