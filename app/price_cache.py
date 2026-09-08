# =====================================================================================
# app/price_cache.py (BULLETPROOF EDITION)
# =====================================================================================

import logging
import threading
import time
import random
import gc
from memory_profiler import profile_function
from datetime import time as dt_time
import pandas as pd
import re
from typing import Optional, Tuple, Any, Union, Set, Dict, List
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from database import upsert_fetch_error
from data_provider import get_fetcher
from config import BATCH_DOWNLOAD_SIZE, PRICE_CACHE_TTL_SECONDS, DATA_DIR
from core_enums import ProviderResult
from validation import ValidationEngine, MarketData, ValidationContext, registry as val_registry, DatasetType
from validation.result import ValidatedDataset, ValidationStatus
from validation.history import history_recorder
from config import SOURCE_RELIABILITY, MAX_HISTORY_SHRINK
import json
import os

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# [VERSION: V5_ACQUISITION_ROUTING_V1.0] Cache Schema & Metadata Invariants
CACHE_SCHEMA_VERSION = 3
INDICATOR_VERSION = "v5.2"

import hashlib

def compute_ohlcv_hash(df: pd.DataFrame) -> str:
    """Computes a fast deterministic hash of core OHLCV data for change detection."""
    if df is None or df.empty:
        return ""
    try:
        cols = [c for c in ["Date", "Datetime", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if not cols and not df.index.empty:
            sample_data = str(df.index.tolist()[:5]) + str(df.iloc[:5].to_dict())
        else:
            sample_data = f"{len(df)}_{df[cols].iloc[0].to_dict()}_{df[cols].iloc[-1].to_dict()}"
        return hashlib.sha256(sample_data.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""

def validate_ohlcv_structure(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validates structural OHLCV integrity:
    1. Timestamp monotonicity (strictly increasing timestamps).
    2. Price sanity (High >= Low, Open & Close within Low/High bounds, Volume >= 0).
    3. Non-empty DataFrame.
    """
    if df is None or df.empty:
        return False, "EMPTY_DATAFRAME"
        
    try:
        # [RULE 67 CHANGE-RATIONALE: CRITICAL WEEKEND CANDLE BAN]
        # Saturday and Sunday candles must NEVER be fetched, accepted, evaluated, or stored as valid market candles.
        time_col = 'Date' if 'Date' in df.columns else ('Datetime' if 'Datetime' in df.columns else None)
        if time_col:
            ts_series = pd.to_datetime(df[time_col], errors='coerce')
        elif isinstance(df.index, pd.DatetimeIndex):
            ts_series = df.index
        else:
            ts_series = pd.to_datetime(df.index, errors='coerce')

        if ts_series is not None and len(ts_series) > 0:
            is_weekend = (ts_series.dt.weekday >= 5) if hasattr(ts_series, 'dt') else (ts_series.weekday >= 5)
            if is_weekend.any():
                return False, "WEEKEND_CANDLES_PROHIBITED"

        # 1. Monotonicity
        if not ts_series.is_monotonic_increasing:
            return False, "NON_MONOTONIC_TIMESTAMPS"
            
        # 2. Price Sanity
        if "High" in df.columns and "Low" in df.columns:
            if (df["High"] < df["Low"]).any():
                return False, "HIGH_LESS_THAN_LOW"

        # 3. Corporate Action Envelope Auto-Sanitization
        if all(col in df.columns for col in ["High", "Low", "Open", "Close"]):
            # Sanitize envelope bounds for corporate action / bonus / split adjusted historical candles
            df["High"] = df[["High", "Open", "Close"]].max(axis=1)
            df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

        if "Close" in df.columns and "High" in df.columns and "Low" in df.columns:
            if (df["Close"] > df["High"] * 1.015).any() or (df["Close"] < df["Low"] * 0.985).any():
                return False, "CLOSE_OUT_OF_BOUNDS"
                
        if "Open" in df.columns and "High" in df.columns and "Low" in df.columns:
            if (df["Open"] > df["High"] * 1.015).any() or (df["Open"] < df["Low"] * 0.985).any():
                return False, "OPEN_OUT_OF_BOUNDS"
                
        if "Volume" in df.columns:
            if (df["Volume"] < 0).any():
                return False, "NEGATIVE_VOLUME"
                
        return True, "VALID"
    except Exception as e:
        return False, f"VALIDATION_EXCEPTION_{e}"

_cache: dict[tuple, dict] = {}
_lock = threading.Lock()
_fetch_lock = threading.Lock()  # Monolithic global fallback lock

# [PHASE4_ISOLATION_v1.0] Per-provider lock split for parallel fetch pipeline
_provider_locks = {
    "upstox": threading.Lock(),
    "fyers": threading.Lock(),
    "bse": threading.Lock(),
    "yahoo": threading.Lock(),
}

# [VERSION: PER_REQUESTER_LOCK_v1.0] Per-requester lock map for intraday multi-TF parallel fetching.
# MULTI_TF_30m, MULTI_TF_15m, and MULTI_TF_5m all resolve to the same 'fyers' provider lock,
# causing them to serialize instead of running in parallel. A per-requester lock gives each
# caller its own mutex, enabling genuine concurrent fetch execution across timeframes.
import collections
_requester_locks: dict = collections.defaultdict(threading.Lock)
_interval_locks: dict = collections.defaultdict(threading.Lock)

def get_provider_fetch_lock(requester: str = None) -> threading.Lock:
    """Returns a per-requester lock (when FEATURE_PROVIDER_LOCK_SPLIT_V1 is True) so that
    callers like MULTI_TF_30m, MULTI_TF_15m, and MULTI_TF_5m can fetch truly in parallel
    without being serialized by a shared provider lock. Falls back to provider-level lock
    for unknown/generic callers and monolithic lock when feature flag is off."""
    from config import FEATURE_PROVIDER_LOCK_SPLIT_V1, DATA_PROVIDER
    if not FEATURE_PROVIDER_LOCK_SPLIT_V1:
        return _fetch_lock
    # Named callers (MULTI_TF_30m, MULTI_TF_15m, MULTI_TF_5m, macro_intraday, etc.) get
    # their own dedicated lock so parallel fetches across timeframes are never serialized.
    if requester:
        return _requester_locks[requester]
    # Generic/unnamed callers fall back to provider-level lock (original behaviour)
    prov = str(DATA_PROVIDER or "fyers").lower()
    if "upstox" in prov: return _provider_locks["upstox"]
    if "fyers" in prov: return _provider_locks["fyers"]
    if "bse" in prov: return _provider_locks["bse"]
    return _provider_locks["yahoo"]

CACHE_TTL_SECONDS = PRICE_CACHE_TTL_SECONDS

# Cache metrics tracking
def _mark_cache_staleness(df):
    if df is None or getattr(df, 'empty', True):
        return
    from market_utils import evaluate_data_staleness
    c_col = 'Date' if 'Date' in df.columns else ('Datetime' if 'Datetime' in df.columns else None)
    if c_col:
        c_last_dt = pd.to_datetime(df[c_col].iloc[-1])
    else:
        c_last_dt = pd.to_datetime(df.index[-1]) if not df.index.empty else None
        
    is_stale = True
    if c_last_dt:
        stale_res = evaluate_data_staleness(c_last_dt)
        is_stale = stale_res.get("is_stale", True)
    df.attrs['is_stale'] = is_stale

_cache_hits = 0
_cache_misses = 0

def _log_cache_timeline():
    """Calculates and logs the current memory footprint of _cache."""
    with _lock:
        keys_count = len(_cache)
        if keys_count == 0:
            return
            
        total_dfs = 0
        total_mb = 0.0
        largest_key = None
        largest_key_mb = 0.0
        
        # [RULE 67 CHANGE RATIONALE - CACHE TIMELINE TRAVERSAL FIX]
        # _cache structure is _cache[(interval, period)][symbol] = {"data": df, "ts": ...}.
        # Previously `data = entry.get("data", {})` assumed top-level entry had a "data" key, returning {}
        # and falsely logging Total DFs: 0 and Memory: 0.0 MB.
        # Fixed by iterating over sym_map.items() and accessing item.get("data").
        for k, sym_map in _cache.items():
            key_mb = 0.0
            dfs_in_key = 0
            if isinstance(sym_map, dict):
                for sym, item in sym_map.items():
                    df = item.get("data") if isinstance(item, dict) else item
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        dfs_in_key += 1
                        try:
                            key_mb += df.memory_usage(deep=False).sum() / (1024 * 1024)
                        except Exception:
                            pass
            
            total_dfs += dfs_in_key
            total_mb += key_mb
            
            if key_mb > largest_key_mb:
                largest_key_mb = key_mb
                largest_key = k
                
        logger.info(
            f"[CACHE_TIMELINE] Keys: {keys_count} | Memory: {total_mb:.1f} MB | "
            f"Largest: {largest_key} ({largest_key_mb:.1f} MB) | Total DFs: {total_dfs} | "
            f"Hits: {_cache_hits} | Misses: {_cache_misses}"
        )

def _cache_timeline_worker():
    while True:
        time.sleep(1800)  # Log every 30 mins
        try:
            _log_cache_timeline()
        except Exception as e:
            logger.warning(f"Failed to log cache timeline: {e}")

threading.Thread(target=_cache_timeline_worker, name="CacheTimeline", daemon=True).start()

from market_utils import is_market_open

def _is_market_hours() -> bool:
    return is_market_open()

def get_dynamic_cadence(interval: str) -> int:
    """Calculates exact seconds until the next NSE candle boundary for any given interval."""
    now_dt = datetime.now(IST)
    market_open = now_dt.replace(hour=9, minute=15, second=0, microsecond=0)
    
    if not _is_market_hours():
        # If after 15:30, target tomorrow's open
        if now_dt.time() > dt_time(15, 30):
            next_open = (now_dt + timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            # It's before 9:15 AM today
            next_open = market_open
            
        # Fast-forward through weekends (Saturday=5, Sunday=6)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        
        secs_to_open = (next_open - now_dt).total_seconds()
        return max(3600, int(secs_to_open))  # Cache until market opens
        

    # If it's before market open, the next boundary is market open
    if now_dt < market_open:
        secs = (market_open - now_dt).total_seconds()
        return max(5, int(secs))
        
    interval_lower = interval.lower()

    # [VERSION: DAILY_CACHE_TTL_FIX_v1.0] Daily intervals must cache until end of trading day.
    # Previously fell through to CACHE_TTL_SECONDS (60s), causing the Wealth Engine to
    # re-download 1 year of daily OHLCV data every minute. Daily bars only change once at
    # market close, so the cache should survive the entire trading session.
    if interval_lower in ('1d', 'daily', '1wk', '1mo'):
        market_close = now_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        if now_dt < market_close:
            # Cache until today's market close
            secs = (market_close - now_dt).total_seconds()
            return max(300, int(secs))  # At least 5 min floor
        else:
            # After market close: cache for 12h (next run will be after next open)
            return 43200

    match = re.match(r'^(\d+)(m|h)$', interval_lower)
    if not match:
        return CACHE_TTL_SECONDS
        
    val = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'h':
        val = val * 60
        
    if val <= 0:
        return CACHE_TTL_SECONDS
        
    # Calculate minutes since market open (9:15 AM)
    minutes_since_open = (now_dt - market_open).total_seconds() / 60.0
    
    # Find the next multiple of the interval
    next_multiple = ((int(minutes_since_open) // val) + 1) * val
    
    # Calculate the exact timestamp of the next boundary
    next_boundary = market_open + timedelta(minutes=next_multiple)
    secs = (next_boundary - now_dt).total_seconds()
    
    # Add a small 5s buffer to allow broker data to settle on their end before fetching
    raw_cadence = max(5, int(secs) + 5)

    # [VERSION: CACHE_FLOOR_FIX_v1.0] Enforce a minimum cache floor per interval.
    # Problem: near a candle boundary (e.g. 11:14 AM for 1H candle at 11:15),
    # get_dynamic_cadence("1h") returned only ~60s. Any scanner run that started before
    # the boundary and checked the cache after would always get a miss, triggering a full
    # delta re-fetch for ALL symbols on EVERY run near that boundary.
    # Fix: floor = 50% of the interval's duration in seconds. Data within the same candle
    # period is always reused regardless of where in the cycle the scan falls.
    # Floors by interval: 5m→150s, 15m→450s, 30m→900s, 1h→1800s
    interval_floor_secs = int(val * 60 * 0.5)  # 50% of interval duration
    return max(raw_cadence, interval_floor_secs)


# [VERSION: MEMORY_RECALIBRATION_v1.0] Recalibrated profile budget from 350 MB to 500 MB to match steady-state process RSS.
@profile_function("Price Fetch", budget_mb=500.0)
def fetch_watchlist_data(watchlist: Any, period: str = "10d", interval: str = "15m", requester: str = None, run_ctx: Any = None, required_indicators: Optional[Set[str]] = None) -> dict[str, pd.DataFrame]:
    global _cache_hits, _cache_misses
    from telemetry_manager import telemetry
    
    # [RULE 67 CHANGE-RATIONALE]:
    # Defensive normalization: Support callers passing a list of symbol strings (e.g. from get_elite_watchlist()),
    # a pandas Series, or a DataFrame with either 'Stock' or 'symbol' column name.
    if isinstance(watchlist, (list, set, tuple)):
        watchlist = pd.DataFrame({"Stock": [str(s) for s in watchlist if s]})
    elif isinstance(watchlist, pd.Series):
        watchlist = pd.DataFrame({"Stock": [str(s) for s in watchlist.dropna().values if s]})
    elif isinstance(watchlist, pd.DataFrame):
        watchlist = watchlist.copy()
        if "Stock" not in watchlist.columns:
            if "symbol" in watchlist.columns:
                watchlist["Stock"] = watchlist["symbol"]
            elif "Symbol" in watchlist.columns:
                watchlist["Stock"] = watchlist["Symbol"]
            elif len(watchlist.columns) > 0:
                watchlist["Stock"] = watchlist.iloc[:, 0]
            else:
                watchlist["Stock"] = []
    else:
        watchlist = pd.DataFrame({"Stock": []})

    # [VERSION: NON_EQUITY_BLOCKLIST_v2.0] Filter non-equity trusts from all price fetching
    from config import NON_EQUITY_BLOCKLIST
    if not watchlist.empty and "Stock" in watchlist.columns:
        watchlist = watchlist[~watchlist["Stock"].astype(str).str.upper().isin(NON_EQUITY_BLOCKLIST)].copy()

    # [VERSION: UNIFIED_1Y_CACHE_v2.0] Standardize all 1d requests to "1y" so EOD, Reversal, Pullback,
    # Wealth Engine and Multibagger all share one single cache key ("1d", "1y").
    if interval == "1d" and period in ("6mo", "1mo", "10d", "3mo", "2y"):
        period = "1y"
    cache_key = (interval, period)
    cadence = get_dynamic_cadence(interval)
    now_mono = time.monotonic()

    with _lock:
        cache_dict = _cache.get(cache_key, {})
        cached_result = {}
        missing_symbols = []
        
        for s in watchlist["Stock"]:
            sym_entry = cache_dict.get(s)
            # [VERSION: UNIFIED_1Y_CACHE_v2.0] Cross-period RAM lookup: if requesting "1y" and not found, check "2y" RAM slot.
            if not sym_entry and interval == "1d":
                sym_entry = _cache.get(("1d", "2y"), {}).get(s)
            if isinstance(sym_entry, pd.DataFrame) and not sym_entry.empty:
                cached_result[s] = sym_entry
                continue
            elif isinstance(sym_entry, dict) and isinstance(sym_entry.get("data"), pd.DataFrame) and not sym_entry["data"].empty:
                # [RULE 67 CHANGE-RATIONALE]: Defensive timestamp resolution preventing KeyError: 'ts' if entry has 'timestamp' or missing ts
                entry_ts = sym_entry.get("ts")
                if entry_ts is not None:
                    age = now_mono - entry_ts
                elif sym_entry.get("timestamp") is not None:
                    age = time.time() - sym_entry["timestamp"]
                else:
                    age = cadence + 1  # Force refresh if missing

                if age < cadence:
                    # [VERSION: EOD_BOUNDARY_RAM_VALIDATION_v1.0]
                    # For 1d data, verify that if market has closed (>= 15:30 IST), the RAM cached DataFrame
                    # actually contains today's closed bar. If not, treat as cache miss to trigger EOD fetch.
                    if interval == "1d":
                        df_d = sym_entry["data"]
                        t_col = 'Date' if 'Date' in df_d.columns else ('Datetime' if 'Datetime' in df_d.columns else None)
                        last_bar_ts = df_d[t_col].iloc[-1] if t_col else (df_d.index[-1] if not df_d.index.empty else None)
                        if last_bar_ts is not None:
                            from market_utils import get_expected_latest_closed_daily_bar
                            if pd.to_datetime(last_bar_ts).date() < get_expected_latest_closed_daily_bar():
                                missing_symbols.append(s)
                                continue
                    cached_result[s] = sym_entry["data"]
                    continue
            missing_symbols.append(s)
            
        if not missing_symbols:
            _cache_hits += len(watchlist)
            logger.debug(f"📦 Price cache hit | {interval} | {period} | All {len(watchlist)} symbols fresh in RAM")
            return cached_result
        else:
            _cache_hits += len(cached_result)
            _cache_misses += len(missing_symbols)
            logger.debug(f"📦 Price cache partial/miss | {interval} | {period} | Cached: {len(cached_result)}, Fetching: {len(missing_symbols)}")

    # [VERSION: CONCURRENT_FETCH_COALESCING_v1.0] Serialize per (interval, period) key so concurrent callers
    # (e.g. PULLBACK, REVERSAL, MULTIBAGGER) wait and reuse freshly cached RAM data via double-check (lines 338-351).
    fetch_lock = _interval_locks[cache_key]
    logger.debug(f"🔒 Attempting to acquire fetch lock for {interval}|{period}...")
    with fetch_lock:
        logger.debug(f"🔓 Fetch lock acquired for {interval}|{period}")
        
        # Double-check cache for missing symbols in case concurrent thread populated them while waiting for lock
        with _lock:
            cache_dict = _cache.get(cache_key, {})
            still_missing = []
            now_mono = time.monotonic()
            for s in missing_symbols:
                sym_entry = cache_dict.get(s)
                if isinstance(sym_entry, pd.DataFrame) and not sym_entry.empty:
                    cached_result[s] = sym_entry
                    continue
                elif isinstance(sym_entry, dict) and isinstance(sym_entry.get("data"), pd.DataFrame) and not sym_entry["data"].empty:
                    entry_ts = sym_entry.get("ts")
                    if entry_ts is not None:
                        age = now_mono - entry_ts
                    elif sym_entry.get("timestamp") is not None:
                        age = time.time() - sym_entry["timestamp"]
                    else:
                        age = cadence + 1

                    if age < cadence:
                        if interval == "1d":
                            df_d = sym_entry["data"]
                            t_col = 'Date' if 'Date' in df_d.columns else ('Datetime' if 'Datetime' in df_d.columns else None)
                            last_bar_ts = df_d[t_col].iloc[-1] if t_col else (df_d.index[-1] if not df_d.index.empty else None)
                            if last_bar_ts is not None:
                                from market_utils import get_expected_latest_closed_daily_bar
                                if pd.to_datetime(last_bar_ts).date() < get_expected_latest_closed_daily_bar():
                                    still_missing.append(s)
                                    continue
                        cached_result[s] = sym_entry["data"]
                        continue
                still_missing.append(s)
                
            if not still_missing:
                logger.info(f"📦 Cache populated by concurrent thread for all requested symbols; reusing.")
                return {s: cached_result[s] for s in watchlist["Stock"] if s in cached_result}

        # Fetch only the missing symbols
        fetch_sub_watchlist = watchlist[watchlist["Stock"].isin(still_missing)].copy()
        if fetch_sub_watchlist.empty:
            return cached_result
            
        result = _download_all_robust(fetch_sub_watchlist, period=period, interval=interval, requester=requester, run_ctx=run_ctx)

    # Determine data_as_of timestamp from freshly fetched data
    data_as_of = None
    if result:
        timestamps = []
        for symbol, df in result.items():
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    ts = None
                    if "Datetime" in df.columns:
                        ts = df["Datetime"].iloc[-1]
                    elif "Date" in df.columns:
                        ts = df["Date"].iloc[-1]
                    else:
                        ts = df.index[-1]
                    ts = pd.to_datetime(ts)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize(IST)
                    else:
                        ts = ts.tz_convert(IST)
                    timestamps.append(ts)
                except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
                    logger.warning(f"Failed to parse timestamp for {symbol} in price_cache: {e}")
                    pass
                    
        if not timestamps and any(isinstance(df, pd.DataFrame) and not df.empty for df in result.values()):
            logger.error("DataFetchError: All dataframes returned malformed or missing timestamps. Aborting cache update.")
            raise ValueError("DataFetchError: Malformed timestamps across entire batch.")
            
        if timestamps:
            data_as_of = min(timestamps)
            if data_as_of.tzinfo is None:
                data_as_of = data_as_of.replace(tzinfo=IST)
            else:
                data_as_of = data_as_of.astimezone(IST)

    with _lock:
        if cache_key not in _cache:
            _cache[cache_key] = {}
        now_mono = time.monotonic()
        
        for symbol, df in result.items():
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                provider_name = getattr(df, 'attrs', {}).get('provider', 'unknown')
                is_stale = getattr(df, 'attrs', {}).get('is_stale', False)
                _cache[cache_key][symbol] = {
                    "data": df,
                    "ts": 0 if is_stale else now_mono,
                    "timestamp": time.time(),
                    "data_as_of": data_as_of,
                    "provider": provider_name,
                    "schema_version": "v8.4.0",
                    "fetch_interval": interval,
                    "fetch_period": period
                }
                # Ensure newly fetched DataFrames are populated into cached_result so final_res returns them to caller
                cached_result[symbol] = df
            else:
                cached_result[symbol] = None
    final_res = {s: (cached_result.get(s) if cached_result.get(s) is not None else (result.get(s) if result else None)) for s in watchlist["Stock"]}

    # [VERSION: POST_MARKET_CMP_ALIGNMENT_v1.0]
    # When fetching 1d data post-market (>= 15:30 IST on a weekday), verify that every symbol's
    # daily DataFrame contains today's closed bar. If Bhavcopy/EOD download is pending and DataFrame
    # ends at the previous session, overlay today's candle using live CMP so all scanners analyze
    # today's actual close (CMP), eliminating stale previous-day alert prices and phantom P&L.
    if interval == "1d" and not os.environ.get("PYTEST_CURRENT_TEST"):
        now_ist = datetime.now(IST)
        if now_ist.weekday() < 5 and now_ist.time() >= dt_time(15, 30):
            try:
                from live_prices import get_live_prices
                valid_syms = [s for s, df_val in final_res.items() if isinstance(df_val, pd.DataFrame) and not df_val.empty]
                if valid_syms:
                    live_prices_map = get_live_prices(valid_syms)
                    for s in valid_syms:
                        df_val = final_res[s]
                        lp = live_prices_map.get(s)
                        if lp and float(lp) > 0:
                            lp_float = round(float(lp), 2)
                            t_col = 'Date' if 'Date' in df_val.columns else ('Datetime' if 'Datetime' in df_val.columns else None)
                            last_val = df_val[t_col].iloc[-1] if t_col else df_val.index[-1]
                            last_dt = pd.to_datetime(last_val)
                            if last_dt.date() < now_ist.date():
                                # Append today's completed session candle with live CMP
                                today_date_val = now_ist.date() if t_col == 'Date' else pd.Timestamp(now_ist.date())
                                new_row = {col: None for col in df_val.columns}
                                for p_col in ("Open", "High", "Low", "Close"):
                                    if p_col in df_val.columns:
                                        c_dtype = df_val[p_col].dtype
                                        new_row[p_col] = c_dtype.type(lp_float) if hasattr(c_dtype, 'type') else lp_float
                                if "Volume" in df_val.columns:
                                    v_dtype = df_val["Volume"].dtype
                                    new_row["Volume"] = v_dtype.type(0) if hasattr(v_dtype, 'type') else 0
                                if t_col:
                                    new_row[t_col] = today_date_val
                                    new_df = pd.concat([df_val, pd.DataFrame([new_row])], ignore_index=True)
                                else:
                                    new_df = pd.concat([df_val, pd.DataFrame([new_row], index=[pd.Timestamp(now_ist.date())])])
                                new_df.attrs = df_val.attrs.copy()
                                final_res[s] = new_df
                                if cache_key in _cache and s in _cache[cache_key]:
                                    _cache[cache_key][s]["data"] = new_df
                            elif last_dt.date() == now_ist.date():
                                col_idx = df_val.columns.get_loc("Close")
                                close_dtype = df_val["Close"].dtype
                                df_val.iloc[-1, col_idx] = close_dtype.type(lp_float) if hasattr(close_dtype, 'type') else lp_float
            except Exception as _cmp_sync_err:
                logger.debug(f"Post-market 1d CMP overlay warning: {_cmp_sync_err}")

    # [RULE 67 CHANGE-RATIONALE: MODULAR_TARGETED_HYDRATION_v1.0]
    # If the caller explicitly requested a set of indicators, hydrate them on the in-memory results.
    if required_indicators is not None:
        from technical_indicators import hydrate_indicators
        for s, df_val in final_res.items():
            if df_val is not None and isinstance(df_val, pd.DataFrame) and not df_val.empty:
                final_res[s] = hydrate_indicators(df_val, required=required_indicators, timeframe=interval)

    return final_res


import os
from config import DATA_DIR
from datetime import timedelta

from enum import Enum

class CacheState(Enum):
    STALE = "STALE"
    FRESH_HISTORY = "FRESH_HISTORY"
    FRESH_WITH_LIVE_OVERLAY = "FRESH_WITH_LIVE_OVERLAY"
    EXPIRED = "EXPIRED"


class CacheFreshnessPolicy:
    """Base interface for timeframe-specific cache freshness policies."""
    def is_fresh(self, last_ts: pd.Timestamp, now_dt: datetime = None) -> bool:
        raise NotImplementedError


class DailyPolicy(CacheFreshnessPolicy):
    """
    Daily Cache Freshness Policy.
    Uses get_expected_latest_closed_daily_bar() to determine the expected closed daily bar:
    - Pre-market or active market hours (Mon-Fri 09:15-15:30): expected closed daily bar is previous trading day.
    - Post-market (Mon-Fri >= 15:30): expected closed daily bar is today's session.
    - Weekends/holidays: expected closed daily bar is the last completed trading day (Friday).
    """
    def is_fresh(self, last_ts: pd.Timestamp, now_dt: datetime = None) -> bool:
        if now_dt is None:
            now_dt = datetime.now(IST)
        from market_utils import get_expected_latest_closed_daily_bar
        expected_closed_bar = get_expected_latest_closed_daily_bar(now_dt)
        return last_ts.date() >= expected_closed_bar


# =====================================================================================
# RULE 67 MANDATORY CHANGE-RATIONALE:
# - Updated FifteenMinutePolicy & FiveMinutePolicy to treat same-day intraday cache as fresh during market hours.
# - Rationale: Prevents invalidating disk cache and triggering 5-7 minute redundant network downloads on every scan cycle
#   when live CMP ticks are already stitched into forming bars.
# =====================================================================================
class FifteenMinutePolicy(CacheFreshnessPolicy):
    """15-Minute Intraday Cache Freshness Policy."""
    def is_fresh(self, last_ts: pd.Timestamp, now_dt: datetime = None) -> bool:
        if now_dt is None:
            now_dt = datetime.now(IST)
        from market_utils import is_market_open
        if is_market_open(now_dt):
            # Same-day session or within 2 hours is considered fresh (live CMP is stitched into forming bar)
            if last_ts.date() == now_dt.date() or (now_dt - last_ts).total_seconds() <= (120 * 60):
                return True
            return (now_dt - last_ts).total_seconds() <= (20 * 60)
        return _is_cache_up_to_date_legacy(last_ts, "15m", now_dt)


class FiveMinutePolicy(CacheFreshnessPolicy):
    """5-Minute Intraday Cache Freshness Policy."""
    def is_fresh(self, last_ts: pd.Timestamp, now_dt: datetime = None) -> bool:
        if now_dt is None:
            now_dt = datetime.now(IST)
        from market_utils import is_market_open
        if is_market_open(now_dt):
            # Same-day session or within 2 hours is considered fresh (live CMP is stitched into forming bar)
            if last_ts.date() == now_dt.date() or (now_dt - last_ts).total_seconds() <= (120 * 60):
                return True
            return (now_dt - last_ts).total_seconds() <= (10 * 60)
        return _is_cache_up_to_date_legacy(last_ts, "5m", now_dt)


def _is_cache_up_to_date_legacy(last_ts: pd.Timestamp, interval: str, now_dt: datetime = None) -> bool:
    """Legacy boundary check for off-market hours."""
    if now_dt is None:
        now_dt = datetime.now(IST)
    market_open = now_dt.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_dt.replace(hour=15, minute=30, second=0, microsecond=0)
    is_weekend = now_dt.weekday() >= 5
    inv_lower = interval.lower()

    if is_weekend:
        last_close_date = (market_close - timedelta(days=now_dt.weekday() - 4)).date()
    elif now_dt < market_open:
        if now_dt.weekday() == 0:
            last_close_date = (now_dt - timedelta(days=3)).date()
        else:
            last_close_date = (now_dt - timedelta(days=1)).date()
    else:
        last_close_date = now_dt.date()

    if last_ts.date() < last_close_date:
        return False

    final_bar_cutoffs = {
        '1h':  (14, 15),
        '60m': (14, 15),
        '30m': (15, 0),
        '15m': (15, 15),
        '5m':  (15, 25),
        '1m':  (15, 29),
    }
    cutoff_h, cutoff_m = final_bar_cutoffs.get(inv_lower, (15, 0))
    cutoff_time = last_ts.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
    return last_ts >= cutoff_time


class OneHourPolicy(CacheFreshnessPolicy):
    """1-Hour Intraday Cache Freshness Policy."""
    def is_fresh(self, last_ts: pd.Timestamp, now_dt: datetime = None) -> bool:
        if now_dt is None:
            now_dt = datetime.now(IST)
        from market_utils import is_market_open
        if is_market_open(now_dt):
            if last_ts.date() == now_dt.date() or (now_dt - last_ts).total_seconds() <= (240 * 60):
                return True
            return (now_dt - last_ts).total_seconds() <= (90 * 60)
        return _is_cache_up_to_date_legacy(last_ts, "1h", now_dt)


class ThirtyMinutePolicy(CacheFreshnessPolicy):
    """30-Minute Intraday Cache Freshness Policy."""
    def is_fresh(self, last_ts: pd.Timestamp, now_dt: datetime = None) -> bool:
        if now_dt is None:
            now_dt = datetime.now(IST)
        from market_utils import is_market_open
        if is_market_open(now_dt):
            if last_ts.date() == now_dt.date() or (now_dt - last_ts).total_seconds() <= (180 * 60):
                return True
            return (now_dt - last_ts).total_seconds() <= (45 * 60)
        return _is_cache_up_to_date_legacy(last_ts, "30m", now_dt)


def _is_cache_up_to_date(last_ts: pd.Timestamp, interval: str, now_dt: datetime = None) -> bool:
    """
    Checks if the cached data contains up-to-date data for the given interval
    using timeframe-specific CacheFreshnessPolicy rules.
    """
    if now_dt is None:
        now_dt = datetime.now(IST)
    inv_lower = interval.lower()

    if inv_lower in ('1d', 'daily', '1wk', '1mo'):
        return DailyPolicy().is_fresh(last_ts, now_dt)
    elif inv_lower in ('1h', '60m'):
        return OneHourPolicy().is_fresh(last_ts, now_dt)
    elif inv_lower in ('30m', '30min'):
        return ThirtyMinutePolicy().is_fresh(last_ts, now_dt)
    elif inv_lower in ('15m', '15min'):
        return FifteenMinutePolicy().is_fresh(last_ts, now_dt)
    elif inv_lower in ('5m', '5min'):
        return FiveMinutePolicy().is_fresh(last_ts, now_dt)
    else:
        return _is_cache_up_to_date_legacy(last_ts, interval, now_dt)

def _is_cache_long_enough(cached_df: pd.DataFrame, period: str, sym: str = "", interval: str = "") -> bool:
    """Check if the cached dataframe has enough calendar days to satisfy the requested period."""
    if cached_df.empty:
        return False
    # Intraday / short-period optimization: For intraday (1h, 30m, 15m, 5m) or short requests (15d, 10d, 5d),
    # if cache already has >= 30 candles, allow DELTA updates without forcing full re-downloads.
    is_intraday_or_short = (
        interval.lower() in ("1h", "30m", "15m", "5m", "15min", "5min") or 
        period.lower() in ("15d", "10d", "5d", "1d")
    )
    if is_intraday_or_short and len(cached_df) >= 30:
        return True
        
    # Daily (1d) optimization: If cached daily dataframe already has >= 30 candles, allow DELTA updates without forcing full re-downloads
    if interval.lower() in ("1d", "daily") and len(cached_df) >= 30:
        return True
    try:
        if 'Date' in cached_df.columns:
            first_ts = pd.to_datetime(cached_df['Date'].iloc[0])
            last_ts = pd.to_datetime(cached_df['Date'].iloc[-1])
        elif 'Datetime' in cached_df.columns:
            first_ts = pd.to_datetime(cached_df['Datetime'].iloc[0])
            last_ts = pd.to_datetime(cached_df['Datetime'].iloc[-1])
        else:
            first_ts = pd.to_datetime(cached_df.index[0])
            last_ts = pd.to_datetime(cached_df.index[-1])
            
        days_diff = (last_ts - first_ts).days
        
        req = 0
        p = period.lower()
        if p == "10y": req = 3600
        elif p == "5y": req = 1800
        elif p == "2y": req = 700
        elif p == "1y": req = 300
        elif p == "6mo": req = 150
        elif p == "3mo": req = 75
        elif p == "1mo": req = 20
        elif p.endswith("d"):
            try: req = int(p[:-1]) - 1
            except Exception: pass
            
        if req > 0:
            # A requested period of N calendar days will have at least N * 0.65 calendar days diff
            # between the first and last candle. If days_diff is smaller, we are missing historical data.
            if days_diff < (req * 0.65):
                # Check if we already hit the beginning of history (IPO/recent listing)
                if len(cached_df) >= 10:
                    earliest_path = os.path.join(DATA_DIR, "earliest_dates.json")
                    if os.path.exists(earliest_path):
                        try:
                            with open(earliest_path, "r") as f:
                                earliest_dates = json.load(f)
                                clean_sym = sym.replace('.NS', '').replace('.BO', '').replace('BSE:', '').replace('NSE:', '').strip().upper()
                                raw_sym = sym.strip().upper()
                                target_earliest = (
                                    earliest_dates.get(clean_sym) or 
                                    earliest_dates.get(raw_sym) or 
                                    earliest_dates.get(f"NSE:{clean_sym}") or
                                    earliest_dates.get(f"BSE:{clean_sym}")
                                )
                                if target_earliest:
                                    return True
                        except Exception:
                            pass
                return False
            
        return True
    except Exception:
        return True

def _download_all_robust(watchlist: pd.DataFrame, period: str, interval: str, requester: str = None, run_ctx: Any = None) -> dict[str, pd.DataFrame]:
    symbols = watchlist["Stock"].tolist()
    all_data: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    if requester == "multibagger":
        batch_size = int(os.environ.get("MULTIBAGGER_INNER_BATCH_SIZE", "250"))
    else:
        batch_size = BATCH_DOWNLOAD_SIZE
    fetcher = get_fetcher()
    rate_limited = False

    history_dir = os.path.join(DATA_DIR, "history", interval)
    os.makedirs(history_dir, exist_ok=True)

    # Group symbols by what they need to fetch
    # Key: (range_from, range_to) or "FULL"
    # Value: list of (symbol, cached_df)
    fetch_groups = {}
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    fresh_count = 0
    
    def _has_parquet_file(s: str) -> bool:
        clean_s = str(s).split(":")[-1].strip()
        variants = [
            clean_s,
            clean_s.replace("&", "_"),
            clean_s.replace("-", "_"),
            clean_s.replace("&", "-"),
            clean_s.replace("-EQ", ""),
            clean_s.replace("_EQ", ""),
            f"{clean_s}.NS",
            f"{clean_s.replace('&', '_')}.NS"
        ]
        return any(os.path.exists(os.path.join(history_dir, f"{v}.parquet")) for v in variants)

    missing_any_disk = any(not _has_parquet_file(s) for s in symbols)
    if missing_any_disk:
        try:
            from database import restore_history_bundle_from_db
            restore_history_bundle_from_db(interval)
        except Exception as res_err:
            logger.debug(f"History bundle DB restore check: {res_err}")

    import concurrent.futures
    import threading
    local_lock = threading.Lock()
    fresh_count_container = [0]

    def _resolve_parquet_file_path(s: str) -> str:
        clean_s = str(s).split(":")[-1].strip()
        variants = [
            clean_s,
            clean_s.replace("&", "_"),
            clean_s.replace("-", "_"),
            clean_s.replace("&", "-"),
            clean_s.replace("-EQ", ""),
            clean_s.replace("_EQ", ""),
            f"{clean_s}.NS",
            f"{clean_s.replace('&', '_')}.NS"
        ]
        for v in variants:
            f_path = os.path.join(history_dir, f"{v}.parquet")
            if os.path.exists(f_path):
                return f_path
        clean_base = clean_s.replace("-EQ", "").replace("&", "_")
        return os.path.join(history_dir, f"{clean_base}.parquet")

    def process_symbol(sym):
        file_path = _resolve_parquet_file_path(sym)
        needs_full = True
        cached_df = None
        
        if os.path.exists(file_path):
            try:
                cached_df = pd.read_parquet(file_path)
                from trading_calendar import enforce_trading_day_candles
                cached_df = enforce_trading_day_candles(cached_df, sym)
                meta_path = file_path.replace('.parquet', '.meta.json')
                is_invalid = False
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        score = meta.get("validation_score", 100)
                        cached_df.attrs["quality_score"] = score
                        val_status = str(meta.get("validation_status", ""))
                        if "INVALID" in val_status or score < 50:
                            is_invalid = True
                    except Exception: pass
                    
                if not cached_df.empty:
                    # Find last timestamp
                    if 'Date' in cached_df.columns:
                        last_ts = pd.to_datetime(cached_df['Date'].iloc[-1])
                    elif 'Datetime' in cached_df.columns:
                        last_ts = pd.to_datetime(cached_df['Datetime'].iloc[-1])
                    else:
                        last_ts = pd.to_datetime(cached_df.index[-1])
                        
                    if pd.isna(last_ts):
                        raise ValueError("last_ts is NaT, cache file might be corrupt")
                        
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.tz_localize(IST)
                    else:
                        last_ts = last_ts.tz_convert(IST)
                        
                    # 🚀 OPTIMIZATION: If data is already up to the last market close, skip DELTA fetch completely!
                    is_up_to_date = _is_cache_up_to_date(last_ts, interval)
                    is_long_enough = _is_cache_long_enough(cached_df, period, sym, interval=interval)
                    
                    if is_invalid:
                        is_up_to_date = False
                        logger.warning(f"CACHE_POLICY | {sym} is marked INVALID (score={cached_df.attrs.get('quality_score')}). Forcing retry despite timestamp {last_ts}.")

                    # Determine maximum allowed delta gap per interval before forcing FULL re-fetch
                    max_delta_days = {
                        "1m": 2,
                        "5m": 3,
                        "15m": 5,
                        "30m": 7,
                        "1h": 12,
                        "60m": 12,
                        "1d": 20,  # 20 calendar days (~14 trading sessions)
                    }.get(interval.lower(), 7)

                    earliest_allowed_dt = (datetime.now(IST) - timedelta(days=max_delta_days))
                    is_recent_enough = (last_ts >= earliest_allowed_dt)
                    is_usable = (is_long_enough and is_recent_enough)
                    
                    if is_up_to_date and is_usable:
                        # [VERSION: V5_ACQUISITION_ROUTING_V1.0] Enforce Cache Invariants: schema_version, indicator_version, ohlcv_hash
                        meta_valid = False
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, "r") as f:
                                    meta = json.load(f)
                                if (meta.get("schema_version") == CACHE_SCHEMA_VERSION and 
                                    meta.get("indicator_version") == INDICATOR_VERSION and
                                    meta.get("ohlcv_hash") == compute_ohlcv_hash(cached_df)):
                                    meta_valid = True
                            except Exception:
                                pass

                        if not cached_df.empty and not meta_valid:
                            try:
                                new_meta = {
                                    "schema_version": CACHE_SCHEMA_VERSION,
                                    "indicator_version": INDICATOR_VERSION,
                                    "ohlcv_hash": compute_ohlcv_hash(cached_df),
                                    "generated_at": time.time(),
                                    "row_count": len(cached_df)
                                }
                                with open(meta_path, "w") as f:
                                    json.dump(new_meta, f)
                            except Exception as e:
                                logger.warning(f"Failed to resave meta for {sym}: {e}")
                                
                        with local_lock:
                            all_data[sym] = cached_df
                            fresh_count_container[0] += 1
                        needs_full = False
                        return
                    elif is_usable:
                        # Not up to date, but cache is long enough and recent enough -> DELTA fetch only
                        needs_full = False
                    else:
                        # Cache missing, has < 30 bars, or last_ts is older than max_delta_days -> FULL fetch required
                        needs_full = True
                            
                    if not needs_full:
                        # Back up 1 day to ensure we get overlapping candles to avoid gaps
                        range_from = (last_ts - timedelta(days=1)).strftime("%Y-%m-%d")
                        range_to = today_str
                        
                        group_key = (range_from, range_to)
                        with local_lock:
                            if group_key not in fetch_groups:
                                fetch_groups[group_key] = []
                            fetch_groups[group_key].append((sym, cached_df))
                        needs_full = False
            except Exception as e:
                logger.warning(f"Failed to read disk cache for {sym}: {e}")
                # [VERSION: CORRUPTED_CACHE_QUARANTINE_v2.0]
                # Quarantine corrupted/zero-byte parquet files to .corrupt.<timestamp> to preserve diagnostic evidence.
                try:
                    if os.path.exists(file_path):
                        corrupt_path = f"{file_path}.corrupt.{int(time.time())}"
                        os.rename(file_path, corrupt_path)
                        logger.info(f"☣️ [CACHE QUARANTINED] Preserved corrupted cache for {sym} to {corrupt_path}")
                    meta_path_clean = file_path.replace('.parquet', '.meta.json')
                    if os.path.exists(meta_path_clean):
                        os.remove(meta_path_clean)
                except Exception as _del_err:
                    logger.debug(f"Failed to quarantine corrupted cache for {sym}: {_del_err}")
                
        if needs_full:
            with local_lock:
                if "FULL" not in fetch_groups:
                    fetch_groups["FULL"] = []
                fetch_groups["FULL"].append((sym, cached_df))

    max_w = min(32, (os.cpu_count() or 4) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
        executor.map(process_symbol, symbols)
        
    fresh_count = fresh_count_container[0]

    # [VERSION: SCOPED_DELTA_COALESCING_V1.0] Coalesce fragmented DELTA date ranges for current (interval, period) scope
    # Merges multiple incremental DELTA date ranges (e.g. 2026-08-03 and 2026-08-04) into single min(range_from) pass.
    coalesced_groups = {}
    delta_items = []
    delta_range_froms = []

    for group_key, items in fetch_groups.items():
        if group_key == "FULL":
            coalesced_groups["FULL"] = items
        else:
            delta_items.extend(items)
            if isinstance(group_key, tuple) and len(group_key) >= 1:
                delta_range_froms.append(group_key[0])

    if delta_items:
        min_range_from = min(delta_range_froms) if delta_range_froms else today_str
        unified_group_key = (min_range_from, today_str)
        coalesced_groups[unified_group_key] = delta_items
        if len(delta_range_froms) > 1:
            logger.info(
                f"⚡ [SCOPED_DELTA_COALESCE] Coalesced {len(delta_range_froms)} DELTA date ranges "
                f"into single unified range [{min_range_from} to {today_str}] for {len(delta_items)} symbols [{interval}]"
            )

    # Process each coalesced group
    any_parquet_written = False
    t_broker_total = 0.0
    t_merge_total = 0.0
    t_write_total = 0.0

    for group_key, items in coalesced_groups.items():
        group_symbols = [item[0] for item in items]
        group_total = len(group_symbols)
        
        range_from, range_to = (None, None) if group_key == "FULL" else group_key
        desc = "FULL" if group_key == "FULL" else f"DELTA {range_from} to {range_to}"
        
        for i in range(0, group_total, batch_size):
            batch = group_symbols[i : i + batch_size]
            batch_end = min(i + batch_size, group_total)
            import random
            sample_str = ", ".join(random.sample(batch, min(3, len(batch))))
            logger.info(f"[{requester}] 📥 Fetching Batch {desc} ({i}–{batch_end}/{group_total}) [{interval}] (e.g., {sample_str})")
            
            _t_b0 = time.monotonic()
            batch_results = fetcher.get_batch_ohlcv(batch, interval=interval, period=period, retries=3, range_from=range_from, range_to=range_to, caller=requester)
            t_broker_total += (time.monotonic() - _t_b0)
            
            if batch_results is None:
                batch_results = {}

            # [RULE 67 CHANGE-RATIONALE: PARALLEL_BATCH_FALLBACK_v1.0]
            # When primary batch provider fails or returns incomplete results (e.g. broker rate limit, 
            # connection drop, or symbol timeout), fetching fallbacks sequentially took 15-20 minutes.
            # We identify all missing symbols in the batch upfront and fetch them concurrently using
            # a bounded ThreadPoolExecutor (max_workers=10), reducing fallback duration to ~15-30 seconds.
            missing_syms = []
            for sym in batch:
                md_check = batch_results.get(sym)
                if md_check is None:
                    md_check = batch_results.get(f"{sym}.NS") or batch_results.get(f"{sym}.BO") or batch_results.get(sym.split('.')[0])
                if md_check is None or getattr(md_check, 'dataframe', None) is None or getattr(md_check.dataframe, 'empty', True):
                    missing_syms.append(sym)

            if missing_syms:
                logger.warning(f"⚠️ Primary batch fetch missing/empty for {len(missing_syms)}/{len(batch)} symbols. Attempting parallel UnifiedFetcher fallback...")
                def _fetch_fallback_item(sym_to_fb):
                    try:
                        from data_providers.unified_fetcher import UnifiedFetcher
                        uf = UnifiedFetcher()
                        fb_df = uf.fetch_historical(sym_to_fb, interval, period, consumer="price_cache_fallback")
                        if fb_df is not None and not fb_df.empty:
                            from validation.report import MarketData, DataQualityReport
                            return sym_to_fb, MarketData(
                                dataframe=fb_df,
                                source=fb_df.attrs.get("provider", "fallback"),
                                quality_report=DataQualityReport(is_valid=True, quality_score=100.0, status="VALID", issues=[]),
                                stale=False,
                                used_fallback=True
                            )
                    except Exception as fb_err:
                        logger.error(f"❌ UnifiedFetcher fallback failed for {sym_to_fb}: {fb_err}")
                    return sym_to_fb, None

                fb_max_w = min(10, len(missing_syms))
                with concurrent.futures.ThreadPoolExecutor(max_workers=fb_max_w) as fb_exec:
                    fb_outcomes = list(fb_exec.map(_fetch_fallback_item, missing_syms))

                for sym_done, md_done in fb_outcomes:
                    if md_done is not None:
                        batch_results[sym_done] = md_done
                        logger.info(f"✅ Parallel fallback successful for {sym_done} using {md_done.source}")

            if batch_results:
                batch_validation_items = []
                batch_indicator_jobs = []
                batch_symbol_meta = {}
                batch_earliest_updates = {}

                for sym in batch:
                    # Ingestion Boundary Canonical Symbol Lookup: Try sym, sym.NS, sym.BO, and base symbol
                    md = batch_results.get(sym)
                    if md is None:
                        md = batch_results.get(f"{sym}.NS") or batch_results.get(f"{sym}.BO") or batch_results.get(sym.split('.')[0])
                    
                    cached_df = next((item[1] for item in items if item[0] == sym), None)
                    
                    if md is None or getattr(md, 'dataframe', None) is None or getattr(md.dataframe, 'empty', True):
                        if cached_df is not None and not cached_df.empty:
                            _mark_cache_staleness(cached_df)
                            all_data[sym] = cached_df
                        else:
                            all_data[sym] = None
                        continue
                        
                    new_df = md.dataframe
                    if new_df is not None:
                        from trading_calendar import enforce_trading_day_candles
                        new_df = enforce_trading_day_candles(new_df, sym)
                    new_report = md.quality_report
                    remote_source = md.source
                    
                    if new_report:
                        batch_validation_items.append(
                            ValidatedDataset(
                                data=new_df, 
                                result=new_report, # DataQualityReport is compatible enough for history_recorder (it has score, status, etc)
                                score=new_report.quality_score, 
                                status=new_report.status
                            )
                        )
                        
                    if new_df is None:
                        if cached_df is not None and not cached_df.empty:
                            _mark_cache_staleness(cached_df)
                            all_data[sym] = cached_df
                        else:
                            all_data[sym] = None
                        continue
                    
                    # Cache Decision Engine
                    if cached_df is not None and not cached_df.empty:
                        cached_score_raw = cached_df.attrs.get("quality_score", 100)
                        cached_row_count = len(cached_df)

                        remote_score = (new_report.quality_score if new_report else 0) * SOURCE_RELIABILITY.get(remote_source, 1.0)
                        cache_score = cached_score_raw * SOURCE_RELIABILITY.get("Cache", 0.95)

                        # Check if remote data contains newer candles than disk cache
                        cached_last_date = None
                        remote_last_date = None
                        c_col = 'Date' if 'Date' in cached_df.columns else ('Datetime' if 'Datetime' in cached_df.columns else None)
                        if c_col and not cached_df.empty:
                            cached_last_date = pd.to_datetime(cached_df[c_col].iloc[-1]).date()
                        elif not cached_df.empty and isinstance(cached_df.index, pd.DatetimeIndex):
                            cached_last_date = cached_df.index[-1].date()
                            
                        r_col = 'Date' if 'Date' in new_df.columns else ('Datetime' if 'Datetime' in new_df.columns else None)
                        if r_col and not new_df.empty:
                            remote_last_date = pd.to_datetime(new_df[r_col].iloc[-1]).date()
                        elif not new_df.empty and isinstance(new_df.index, pd.DatetimeIndex):
                            remote_last_date = new_df.index[-1].date()

                        has_newer_bars = (remote_last_date is not None and cached_last_date is not None and remote_last_date > cached_last_date)

                        # [VERSION: CACHE_DECISION_FRESHNESS_FIX_v1.0]
                        # If remote data is dated >= today's expected trading bar, it is genuinely current.
                        # Do NOT mark it stale just because its quality score is lower than the cached score.
                        # Surveillance stocks, circuit-hit stocks, and thin-volume stocks all produce lower
                        # quality scores (fewer bars, compressed OHLCV) but their data is still fresh and valid.
                        from market_utils import get_expected_latest_trading_date
                        expected_trading_date = get_expected_latest_trading_date()
                        remote_is_current = (remote_last_date is not None and remote_last_date >= expected_trading_date)

                        logger.debug(
                            f"CACHE_DECISION | Symbol={sym} | RemoteScore={remote_score:.1f} ({remote_source}) "
                            f"| CacheScore={cache_score:.1f} | HasNewerBars={has_newer_bars} "
                            f"| RemoteLastDate={remote_last_date} | CachedLastDate={cached_last_date} "
                            f"| RemoteIsCurrent={remote_is_current} | ExpectedDate={expected_trading_date}"
                        )

                        # 1. Critical Cache Validation
                        reject_reason = None
                        is_delta_fetch = bool(range_from)
                        is_full_fetch = (group_key == "FULL")
                        op_mode = "INCREMENTAL_MERGE" if (cached_df is not None and not cached_df.empty and not is_full_fetch) else "FULL_REPLACE"

                        if new_report:
                            q_score = getattr(new_report, 'quality_score', 100)
                            if op_mode == "FULL_REPLACE" and not remote_is_current and isinstance(q_score, (int, float)) and q_score < 50:
                                reject_reason = "POOR_QUALITY_SCORE"
                            elif getattr(new_report, 'is_valid', True) is False:
                                reject_reason = "INVALID_QUALITY_REPORT"
                            elif (op_mode == "FULL_REPLACE" and 
                                  interval.lower() in ("1d", "daily") and 
                                  getattr(new_report, 'row_count', 0) < cached_row_count * (1.0 - MAX_HISTORY_SHRINK)):
                                # [RULE 67 CHANGE-RATIONALE]:
                                # Restrict HISTORICAL_SHRINK checks strictly to daily data. 
                                # For intraday intervals, short full fetches (e.g. 5d window) naturally 
                                # return fewer rows than the cache capacity (e.g. 750), leading to false-positive 
                                # rejects and persistent stale data deadlock warnings.
                                reject_reason = "HISTORICAL_SHRINK"

                        if reject_reason:
                            logger.warning(f"Critical Cache Validation Failed for {sym} (Mode={op_mode}): {reject_reason}. REJECTING remote data to protect cache.")
                            logger.info(f"CACHE_DECISION | Action=KEEP_CACHE | Reason={reject_reason} | Symbol={sym} | Mode={op_mode} | ExistingRows={cached_row_count} | IncomingRows={getattr(new_report, 'row_count', 0)} | Threshold={MAX_HISTORY_SHRINK*100}%")
                            _mark_cache_staleness(cached_df)
                            all_data[sym] = cached_df
                            continue
                        elif has_newer_bars or remote_is_current or op_mode == "INCREMENTAL_MERGE" or remote_score >= cache_score or (not new_report and remote_score == cache_score):
                            # Accept and Merge:
                            # - Remote has newer candles, OR
                            # - Remote data is dated today (fresh), OR
                            # - Operation is INCREMENTAL_MERGE, OR
                            # - Remote quality >= cached quality
                            pass
                        else:
                            # Reject Remote Data (genuinely lower quality AND no new/current data)
                            logger.info(f"CACHE_DECISION | Action=KEEP_CACHE | Reason=REMOTE_LOWER_QUALITY | Symbol={sym} | CacheScore={cache_score} | RemoteScore={remote_score} | RemoteLastDate={remote_last_date} | ExpectedDate={expected_trading_date} — marking stale")
                            _mark_cache_staleness(cached_df)
                            all_data[sym] = cached_df
                            continue
                            
                    if new_df is not None and not new_df.empty:
                        _t_m0 = time.monotonic()
                        # [VERSION: TIMEZONE_FIX_v1.0] True timezone normalization at ingestion boundary
                        time_col = 'Date' if 'Date' in new_df.columns else ('Datetime' if 'Datetime' in new_df.columns else None)
                        if time_col:
                            new_df[time_col] = pd.to_datetime(new_df[time_col])
                            if new_df[time_col].dt.tz is None:
                                new_df[time_col] = new_df[time_col].dt.tz_localize('Asia/Kolkata')
                            else:
                                new_df[time_col] = new_df[time_col].dt.tz_convert('Asia/Kolkata')
                            
                            # [VERSION: TIME_COLUMN_MERGE_FIX] Standardize to 'Datetime' to prevent NaN gaps during concat
                            if time_col == 'Date':
                                new_df = new_df.rename(columns={'Date': 'Datetime'})
                                time_col = 'Datetime'
                                
                        elif not new_df.index.empty:
                            new_df.index = pd.to_datetime(new_df.index)
                            if new_df.index.tz is None:
                                new_df.index = new_df.index.tz_localize('Asia/Kolkata')
                            else:
                                new_df.index = new_df.index.tz_convert('Asia/Kolkata')
                                
                        fresh_count += 1
                        if cached_df is not None and not cached_df.empty:
                            # [VERSION: TIMEZONE_FIX_v1.1] Normalize cached_df timezone before concat
                            c_time_col = 'Date' if 'Date' in cached_df.columns else ('Datetime' if 'Datetime' in cached_df.columns else None)
                            if c_time_col:
                                cached_df[c_time_col] = pd.to_datetime(cached_df[c_time_col])
                                if cached_df[c_time_col].dt.tz is None:
                                    cached_df[c_time_col] = cached_df[c_time_col].dt.tz_localize('Asia/Kolkata')
                                else:
                                    cached_df[c_time_col] = cached_df[c_time_col].dt.tz_convert('Asia/Kolkata')
                                    
                                # [VERSION: TIME_COLUMN_MERGE_FIX] Standardize to 'Datetime' to prevent NaN gaps during concat
                                if c_time_col == 'Date':
                                    cached_df = cached_df.rename(columns={'Date': 'Datetime'})
                                    c_time_col = 'Datetime'
                                    
                            elif not cached_df.index.empty:
                                cached_df.index = pd.to_datetime(cached_df.index)
                                if cached_df.index.tz is None:
                                    cached_df.index = cached_df.index.tz_localize('Asia/Kolkata')
                                else:
                                    cached_df.index = cached_df.index.tz_convert('Asia/Kolkata')

                            # [VERSION: CACHE_MERGE_ALIGNMENT_FIX] Align structural mismatch (time in column vs index)
                            if time_col and not c_time_col:
                                cached_df = cached_df.reset_index()
                                if 'Date' in cached_df.columns:
                                    cached_df = cached_df.rename(columns={'Date': 'Datetime'})
                                elif 'index' in cached_df.columns:
                                    cached_df = cached_df.rename(columns={'index': 'Datetime'})
                                c_time_col = 'Datetime'
                            elif c_time_col and not time_col:
                                new_df = new_df.reset_index()
                                if 'Date' in new_df.columns:
                                    new_df = new_df.rename(columns={'Date': 'Datetime'})
                                elif 'index' in new_df.columns:
                                    new_df = new_df.rename(columns={'index': 'Datetime'})
                                time_col = 'Datetime'

                            # Merge them
                            combined = pd.concat([cached_df, new_df])
                            # Deduplicate based on timestamp
                            time_col_comb = 'Date' if 'Date' in combined.columns else ('Datetime' if 'Datetime' in combined.columns else None)
                            if time_col_comb:
                                combined = combined.drop_duplicates(subset=[time_col_comb], keep='last')
                            else:
                                combined = combined[~combined.index.duplicated(keep='last')]
                                
                            combined = combined.sort_index() if time_col_comb is None else combined.sort_values(time_col_comb)
                            
                            # [VERSION: CACHE_MAX_ROWS_OPTIMIZATION_v1.0]
                            # RATIONALE: Previously, intraday data was hardcoded to 5000 rows and daily data to 2000 rows.
                            # - PERFORMANCE FIX: 5000 rows for 30m/1h caused massive CPU bottlenecks during apply_indicators 
                            #   (taking ~80s per 14 symbols). We cap 15m/30m/1h at 800 rows which is enough for a 260-period 
                            #   indicator lookback + burn-in padding. 1m-5m stay high to ensure enough trading days.
                            # - MULTIBAGGER FIX: The old 2000 row limit truncated the 10-year Multibagger scan (which needs ~2520 days).
                            #   Daily limit is now raised to 3000 rows to ensure 10-year history remains cached intact.
                            if 'm' in interval:
                                if interval in ('1m', '2m', '3m'):
                                    max_rows = 1500
                                elif interval == '5m':
                                    max_rows = 750
                                else: # 10m, 15m, 30m, 45m, 60m
                                    max_rows = 400
                            elif 'h' in interval:
                                max_rows = 400 # 1h, 2h, 4h
                            else:
                                max_rows = 3000 # 1d, 1w, 1mo (supports 10-year daily history)
                                
                            combined = combined.tail(max_rows).copy()
                            
                            # [VERSION: CACHE_INDEX_FIX] If time is in a column, reset the index to prevent PyArrow
                            # from crashing on a mixed type index resulting from concat.
                            if time_col_comb:
                                combined = combined.reset_index(drop=True)
                            
                            all_data[sym] = combined
                        else:
                            # [VERSION: FRESH_DATA_SORT_FIX_v1.0] Deduplicate and sort fresh DataFrames by date before validation
                            if time_col:
                                new_df[time_col] = pd.to_datetime(new_df[time_col])
                                new_df = new_df.drop_duplicates(subset=[time_col], keep='last').sort_values(time_col).reset_index(drop=True)
                            elif not new_df.index.empty:
                                new_df.index = pd.to_datetime(new_df.index)
                                new_df = new_df[~new_df.index.duplicated(keep='last')].sort_index()
                            all_data[sym] = new_df
                            
                        # [VERSION: V5_ACQUISITION_ROUTING_V1.0] OHLCV Validation Stage before indicator calculation
                        if not all_data[sym].empty:
                            is_valid_struct, reason = validate_ohlcv_structure(all_data[sym])
                            if not is_valid_struct:
                                logger.warning(f"⚠️ OHLCV Structure Validation Failed for {sym}: {reason}. Reverting to stale cache if available.")
                                if cached_df is not None and not cached_df.empty:
                                    _mark_cache_staleness(cached_df)
                                    all_data[sym] = cached_df
                                else:
                                    all_data[sym] = None
                                continue

                            # [NEW] Bhavcopy Stitching Logic for Daily Timeframe
                            if interval.lower() in ("1d", "daily", "1y"):
                                try:
                                    _mark_cache_staleness(all_data[sym])
                                    if all_data[sym].attrs.get('is_stale', False):
                                        from datetime import time as dt_time
                                        from zoneinfo import ZoneInfo
                                        from trading_calendar import default_trading_calendar
                                        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
                                        expected_date = now_ist.date()
                                        # Strictly prohibit synthesizing or stitching candles on weekends or NSE exchange holidays
                                        if default_trading_calendar.is_trading_day(now_ist) and now_ist.time() >= dt_time(17, 30):
                                            from data_registry import registry
                                            full_bhavcopy_key = f"bhavcopy_full_{expected_date.isoformat()}"
                                            bhavcopy_df = registry.get(full_bhavcopy_key)
                                            if bhavcopy_df is not None and not bhavcopy_df.empty:
                                                if sym in bhavcopy_df['SYMBOL'].values:
                                                    df_target = all_data[sym]
                                                    last_dt = df_target.index[-1] if isinstance(df_target.index, pd.DatetimeIndex) else pd.to_datetime(df_target['Date'].iloc[-1] if 'Date' in df_target.columns else df_target['Datetime'].iloc[-1])
                                                    if last_dt.date() < expected_date:
                                                        row = bhavcopy_df[bhavcopy_df['SYMBOL'] == sym].iloc[0]
                                                        new_row = {
                                                            'Open': float(row['OPEN']),
                                                            'High': float(row['HIGH']),
                                                            'Low': float(row['LOW']),
                                                            'Close': float(row['CLOSE']),
                                                            'Volume': float(row['TOTTRDQTY'])
                                                        }
                                                        if isinstance(df_target.index, pd.DatetimeIndex):
                                                            # If timezone-aware index, make new_idx timezone-aware
                                                            new_idx = pd.to_datetime(expected_date)
                                                            if df_target.index.tz is not None:
                                                                new_idx = new_idx.tz_localize(df_target.index.tz)
                                                            df_target.loc[new_idx] = pd.Series(new_row)
                                                        else:
                                                            time_col = 'Date' if 'Date' in df_target.columns else 'Datetime'
                                                            new_row[time_col] = pd.to_datetime(expected_date)
                                                            df_target = pd.concat([df_target, pd.DataFrame([new_row])], ignore_index=True)
                                                            all_data[sym] = df_target
                                                        
                                                        all_data[sym].attrs['is_stale'] = False
                                                        logger.info(f"🧵 [BHAVCOPY STITCHING] Synthesized today's daily candle for {sym} from NSE Bhavcopy.")
                                except Exception as stitch_err:
                                    logger.warning(f"Failed to stitch Bhavcopy data for {sym}: {stitch_err}")

                        t_merge_total += (time.monotonic() - _t_m0)

                        # [RULE 67 CHANGE-RATIONALE: CANONICAL_RAW_SCHEMA_ENFORCEMENT_v1.0]
                        # Enforce canonical raw cache invariant: disk Parquets strictly store raw OHLCV + timestamps.
                        raw_allowed_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Datetime', 'timestamp'}
                        keep_cols = [c for c in all_data[sym].columns if c in raw_allowed_cols]
                        if keep_cols and len(keep_cols) < len(all_data[sym].columns):
                            all_data[sym] = all_data[sym][keep_cols].copy()

                        # [RULE 67 CHANGE-RATIONALE: DECOUPLE_RAW_CACHE_PERSISTENCE_v1.0]
                        # Save raw merged OHLCV DataFrame directly to Parquet on disk without blocking
                        # on full 35+ indicator calculations. Scanners consume raw data or hydrate on-demand.
                        _t_w0 = time.monotonic()
                        try:
                            file_path = os.path.join(history_dir, f"{sym.replace(':', '_')}.parquet")
                            if isinstance(all_data[sym].columns, pd.MultiIndex):
                                all_data[sym].columns = ['_'.join(map(str, col)).strip() for col in all_data[sym].columns.values]
                            all_data[sym].columns = all_data[sym].columns.astype(str)

                            time_cols = ['Date', 'Datetime']
                            for col in all_data[sym].columns:
                                if col not in time_cols and all_data[sym][col].dtype == 'object':
                                    all_data[sym][col] = pd.to_numeric(all_data[sym][col], errors='coerce')

                            if all_data[sym].index.name in time_cols or isinstance(all_data[sym].index, pd.DatetimeIndex):
                                all_data[sym].index = pd.to_datetime(all_data[sym].index, errors='coerce')
                            elif not isinstance(all_data[sym].index, pd.RangeIndex):
                                all_data[sym].index = all_data[sym].index.astype(str)

                            from trading_calendar import enforce_trading_day_candles
                            all_data[sym] = enforce_trading_day_candles(all_data[sym], sym)

                            import uuid
                            tmp_file_path = f"{file_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
                            all_data[sym].to_parquet(tmp_file_path, compression='snappy')
                            os.replace(tmp_file_path, file_path)
                            any_parquet_written = True
                            t_write_total += (time.monotonic() - _t_w0)

                            meta_path = file_path.replace('.parquet', '.meta.json')
                            val_score = getattr(new_report, 'quality_score', 100) if new_report else 100
                            if not isinstance(val_score, (int, float)): val_score = 100

                            val_status = getattr(new_report, 'status', 'ValidationStatus.VALID') if new_report else 'ValidationStatus.VALID'
                            if not isinstance(val_status, str): val_status = str(val_status)

                            val_name = getattr(new_report, 'validator_name', 'Unknown') if new_report else 'Unknown'
                            if not isinstance(val_name, str): val_name = str(val_name)

                            meta = {
                                "schema_version": CACHE_SCHEMA_VERSION,
                                "indicator_version": INDICATOR_VERSION,
                                "ohlcv_hash": compute_ohlcv_hash(all_data[sym]),
                                "generated_at": time.time(),
                                "row_count": len(all_data[sym]),
                                "validation_score": val_score,
                                "validation_status": val_status,
                                "validator_name": val_name
                            }
                            with open(meta_path, "w") as f:
                                json.dump(meta, f)
                        except OSError as oe:
                            if getattr(oe, 'errno', None) == 28 or 'No space left' in str(oe):
                                logger.warning(f"⚠️ Disk full — skipped disk cache write for {sym} (in-memory data preserved)")
                            else:
                                logger.warning(f"Disk write error for {sym}: {oe}")
                        except Exception as e:
                            logger.exception(f"Failed to write disk cache for {sym}")

                        # Record earliest date into batch dict (saved once after loop)
                        if group_key == "FULL" and new_df is not None and not new_df.empty and len(new_df) >= 10 and period.lower() in ("max", "10y", "5y", "2y", "1y", "ytd"):
                            try:
                                t_col = 'Date' if 'Date' in new_df.columns else ('Datetime' if 'Datetime' in new_df.columns else None)
                                earliest_ts = pd.to_datetime(new_df[t_col].iloc[0]) if t_col else pd.to_datetime(new_df.index[0])
                                earliest_dt_str = earliest_ts.date().isoformat() if hasattr(earliest_ts, 'date') else None
                                if earliest_dt_str:
                                    clean_sym = sym.replace('.NS', '').replace('.BO', '').replace('BSE:', '').replace('NSE:', '').strip().upper()
                                    batch_earliest_updates[sym] = earliest_dt_str
                                    batch_earliest_updates[clean_sym] = earliest_dt_str
                                    batch_earliest_updates[f"NSE:{clean_sym}"] = earliest_dt_str
                            except Exception as e:
                                logger.debug(f"Failed to record earliest date for {sym}: {e}")
                    else:
                        # Fallback to stale cached data if fresh fetch returned empty
                        if cached_df is not None and not cached_df.empty:
                            _mark_cache_staleness(cached_df)
                            all_data[sym] = cached_df

                # Batch write earliest_dates.json ONCE per sub-chunk instead of N times in loop
                if batch_earliest_updates:
                    try:
                        earliest_path = os.path.join(DATA_DIR, "earliest_dates.json")
                        earliest_dates = {}
                        if os.path.exists(earliest_path):
                            with open(earliest_path, "r") as f:
                                earliest_dates = json.load(f)
                        earliest_dates.update(batch_earliest_updates)
                        with open(earliest_path, "w") as f:
                            json.dump(earliest_dates, f)
                    except Exception as ed_err:
                        logger.debug(f"Failed to batch-write earliest_dates.json: {ed_err}")
                
                # Record batch validation history
                history_recorder.record_batch(DatasetType.PRICE, batch_validation_items)
            else:
                logger.error(f"❌ Batch {desc} failed or returned empty for {len(batch)} symbols.")
                rate_limited = True
                
                # Record empty batch history
                history_recorder.record_batch(DatasetType.PRICE, [], fallback_status=ValidationStatus.INVALID)
                
                # Fallback entire batch to stale cache
                for sym in batch:
                    cached_df = next((item[1] for item in items if item[0] == sym), None)
                    if cached_df is not None and not cached_df.empty:
                        _mark_cache_staleness(cached_df)
                        all_data[sym] = cached_df
                time.sleep(0.5)

            # [VERSION: BATCH_HEARTBEAT_PULSE_v1.0] Pulse heartbeat to DB so watchdog never marks long multi-batch runs as TIMEOUT_STALE
            # [RULE 67 CHANGE-RATIONALE]: Do not blindly call mark_fresh here; each scanner accurately classifies fresh, stale, and incomplete based on actual returned DataFrames.
            if run_ctx:
                try:
                    if hasattr(run_ctx, "heartbeat"):
                        run_ctx.heartbeat(force=True)
                except Exception as _hb_err:
                    logger.debug(f"Heartbeat pulse error during batch download: {_hb_err}")

            # [RULE 67 CHANGE-RATIONALE: REMOVE_INNER_LOOP_BUNDLE_SYNC_v1.0]
            # Inner-loop 45MB tar + DB upload removed to eliminate ~2.5min I/O thrashing between sub-batches.
            # Persistence is now decoupled and handled once per fetch cycle via generation coalescing.

    successful_syms = []
    failed_syms = []
    for sym in symbols:
        df = all_data.get(sym)
        if df is None or isinstance(df, ProviderResult) or not isinstance(df, pd.DataFrame) or df.empty:
            failed_syms.append(sym)
            try:
                import config
                active_providers = getattr(config, "PROVIDER_ROUTING_POLICY", {}).get(f"price_{interval}", ["fyers", "upstox"])
                provider_desc = "+".join(active_providers)
                upsert_fetch_error(
                    'UNIFIED_PIPELINE',
                    'PRICE_CACHE',
                    sym,
                    interval,
                    'no_data_after_fetch',
                    f"Exhausted active providers [{provider_desc}] for {sym} [{interval}]"
                )
            except Exception:
                pass
            all_data[sym] = None
        else:
            successful_syms.append(sym)

    logger.info(
        f"✅ Data secured for {len(successful_syms)}/{total} symbols [{interval}] | "
        f"Stage Timings: BrokerFetch={t_broker_total:.2f}s, RawMerge={t_merge_total:.2f}s, DiskWrite={t_write_total:.2f}s"
    )

    if successful_syms:
        try:
            from database import delete_fetch_errors_batch_on_success
            delete_fetch_errors_batch_on_success('UNIFIED_PIPELINE', 'PRICE_CACHE', successful_syms, interval, 'no_data_after_fetch')
            delete_fetch_errors_batch_on_success('fyers', 'PRICE_CACHE', successful_syms, interval, 'no_data_after_fetch')
        except Exception:
            pass

    try:
        from data_fetch_status import mark_success, mark_failure
        
        failed_fresh = total - fresh_count
        
        if total > 0:
            failure_rate = failed_fresh / total
            if failure_rate > 0.25:
                mark_failure(f"fyers:{interval}", f"Scanner failed: >25% stale/missing ({failed_fresh}/{total} records failed fresh fetch)")
            else:
                # >= 75% success is acceptable
                mark_success(f"fyers:{interval}")
        else:
            mark_failure(f"fyers:{interval}", "No data returned (completely empty)")
    except Exception:
        pass
        
    if any_parquet_written:
        try:
            from database import advance_interval_generation, upload_history_bundle_to_db, submit_background_upload
            # [RULE 67 CHANGE-RATIONALE: BATCH_GENERATION_ADVANCE_v1.0]
            # Advance mutation generation once per completed fetch cycle rather than 121 times per file write.
            new_gen = advance_interval_generation(interval)
            logger.info(f"⚡ [PRICE_CACHE] Mutated {interval} cache (gen={new_gen}). Scheduling background bundle persistence.")
            submit_background_upload(lambda _iv=interval: upload_history_bundle_to_db(_iv))
        except Exception as _hb_up_err:
            logger.debug(f"History bundle auto-upload submission: {_hb_up_err}")
    
    return all_data

@profile_function("Hist Fetch", budget_mb=400.0)
def fetch_unified_historical(symbols: Any, period: str = "1y", interval: str = "1d", requester: str = None) -> dict[str, pd.DataFrame]:
    """
    Unified data fetcher for wealth_engine, eod_scanner, and reversal_scanner.
    Uses unified cache key (interval, period) to allow cross-scanner reuse.
    
    OPTIMIZATION: All 1D data now shares cache key (1d, 1y) instead of
    having separate cache per module (price_fetcher vs price_cache).
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    elif not isinstance(symbols, list):
        symbols = list(symbols)
    watchlist_df = pd.DataFrame({"Stock": symbols})
    return fetch_watchlist_data(watchlist_df, period=period, interval=interval, requester=requester)


# -----------------------------------------------------------------------------
# MARKET-HOUR & INTRADAY SHARED SNAPSHOT
# -----------------------------------------------------------------------------

import os
from config import DATA_DIR
WEALTH_PATH = os.path.join(DATA_DIR, "elite_wealth_system.parquet")


# [BUG FIX 2026-06-24] _INTERVAL_CADENCE and _TTL_JITTER were used inside
# get_intraday_snapshot() but were never defined anywhere in this file.
# This was a latent NameError crash waiting to happen if any scanner called
# get_intraday_snapshot(). Defined here to match the dynamic cadence logic
# in get_dynamic_cadence() above.
_INTERVAL_CADENCE: dict[str, int] = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "1d":  86400,
}

# Jitter adds a small buffer after the candle closes to allow broker data to settle
_TTL_JITTER: dict[str, int] = {
    "1m":  5,
    "5m":  10,
    "15m": 15,
    "30m": 20,
    "1h":  30,
    "1d":  60,
}

# Tracks in-progress fetches so only one thread fetches per (interval, period)
_inflight_fetches: dict[tuple, threading.Event] = {}


def get_intraday_snapshot(symbols: list[str], interval: str = "5m", period: str = "5d", wait_timeout: int = 30, requester: str = None, cadence_override: int = None) -> dict[str, pd.DataFrame]:
    requester = requester or threading.current_thread().name or "Unknown"
    """
    Return cached intraday frames for (interval, period) for the provided symbols.
    If cache is stale or missing, a single thread will perform the fetch and others
    will wait up to `wait_timeout` seconds for the result. This guarantees only one
    fetch per cache key is in-flight at any time.

    cadence_override: If set, overrides the default per-interval cadence for the stale check.
    Use this when the caller can tolerate slightly older data to avoid unnecessary re-fetches.
    Example: Wealth Engine passes 900s (15 min) so it reuses cached data instead of
    triggering a full 10-minute re-fetch of 302 symbols on every scan cycle.

    Returns the raw mapping: { symbol: DataFrame }
    """
    cache_key = (interval, period)
    cadence = cadence_override if cadence_override is not None else _INTERVAL_CADENCE.get(interval, CACHE_TTL_SECONDS)
    jitter = _TTL_JITTER.get(interval, 0) if cadence_override is None else 0
    cadence_with_jitter = cadence + jitter

    # Quick cache check
    with _lock:
        cache_dict = _cache.get(cache_key)
        if cache_dict:
            now_mono = time.monotonic()
            res = {}
            all_hit = True
            for s in symbols:
                sym_entry = cache_dict.get(s)
                if sym_entry and isinstance(sym_entry.get("data"), pd.DataFrame) and not sym_entry["data"].empty:
                    age = now_mono - sym_entry.get("ts", 0)
                    if age < cadence_with_jitter:
                        res[s] = sym_entry["data"]
                        continue
                all_hit = False
                break
            if all_hit and len(res) == len(symbols):
                logger.debug(f"[{requester}] 📦 Intraday cache hit | {interval}|{period} | All {len(symbols)} symbols fresh")
                # [VERSION: DATA_FETCH_ACCELERATION_v1.0] Stitch 1-second live price tick into last candle
                from market_utils import is_market_open
                if is_market_open() and not os.environ.get("PYTEST_CURRENT_TEST"):
                    try:
                        from live_prices import get_live_prices
                        live_prices_map = get_live_prices(list(res.keys()))
                        for sym, df_item in res.items():
                            if isinstance(df_item, pd.DataFrame) and not df_item.empty and sym in live_prices_map:
                                lp = live_prices_map[sym]
                                if lp and float(lp) > 0:
                                    col_idx = df_item.columns.get_loc("Close")
                                    close_dtype = df_item["Close"].dtype
                                    df_item.iloc[-1, col_idx] = close_dtype.type(lp) if hasattr(close_dtype, 'type') else float(lp)
                    except Exception:
                        pass
                return res

        # If another thread is already fetching this key, wait for it to complete
        inflight = _inflight_fetches.get(cache_key)

    # If inflight exists, wait for completion then return cache (may still be missing)
    if inflight:
        inflight.wait(wait_timeout)
        with _lock:
            cache_dict = _cache.get(cache_key, {})
            return {s: cache_dict[s]["data"] if s in cache_dict and isinstance(cache_dict[s], dict) else None for s in symbols}

    # No inflight — attempt to become the fetcher
    evt = threading.Event()
    with _lock:
        # Double-check in case someone set it while creating event
        if cache_key in _inflight_fetches:
            inflight = _inflight_fetches[cache_key]
        else:
            _inflight_fetches[cache_key] = evt
            inflight = None

    if inflight:
        # Race lost — wait for the actual fetcher
        inflight.wait(wait_timeout)
        with _lock:
            cache_dict = _cache.get(cache_key, {})
            return {s: cache_dict[s]["data"] if s in cache_dict and isinstance(cache_dict[s], dict) else None for s in symbols}

    # This thread is responsible for fetching
    try:
        logger.info(f"[{requester}] 🔁 Performing single fetch for intraday key {cache_key} for {len(symbols)} symbols")
        watchlist_df = pd.DataFrame({"Stock": symbols})
        # Use existing serialized path which already uses a global fetch lock
        result = fetch_watchlist_data(watchlist_df, period=period, interval=interval)
        # Return subset for requested symbols
        return {s: result.get(s) for s in symbols} if result else {s: None for s in symbols}
    finally:
        # Signal completion so waiters can proceed
        try:
            evt.set()
        except Exception:
            pass
        with _lock:
            _inflight_fetches.pop(cache_key, None)


def fetch_market_hour_snapshot(symbols: list[str], recent_period: str = "5d", requester: str = None) -> dict:
    requester = requester or threading.current_thread().name or "Unknown"
    """
    Fetch a small, shared snapshot optimized for market-hours:
      - Recent daily OHLCV for `recent_period` (default 5d) via cached batch fetch
      - SMA200 lookup from persisted Wealth parquet (fast) when available
      - Compute SMA200 from recent data only if enough bars exist (avoid 1y re-fetch)

    Returns: {
      "daily": dict[str, pd.DataFrame],
      "sma_200": dict[str, Optional[float]],
      "data_as_of": datetime or None
    }
    """
    result = {
        "daily": {},
        "sma_200": {},
        "data_as_of": None,
    }

    if not symbols:
        return result

    # 1) Fetch recent daily bars using the unified cached path (this is serialized by fetch_watchlist_data)
    try:
        daily = fetch_unified_historical(symbols, period=recent_period, interval="1d", requester=requester)
    except Exception as e:
        logger.warning(f"[{requester}] Failed to fetch recent daily data for snapshot: {e}")
        daily = {}

    result["daily"] = daily or {}

    # Determine data_as_of (oldest/latest timestamp across fetched frames)
    timestamps = []
    for df in result["daily"].values():
        try:
            if df is None or df.empty:
                continue
            if "Datetime" in df.columns:
                ts = pd.to_datetime(df["Datetime"].iloc[-1])
            elif "Date" in df.columns:
                ts = pd.to_datetime(df["Date"].iloc[-1])
            else:
                ts = pd.to_datetime(df.index[-1])
            if ts.tzinfo is None:
                ts = ts.tz_localize(IST)
            else:
                ts = ts.tz_convert(IST)
            timestamps.append(ts)
        except Exception:
            continue

    if timestamps:
        result["data_as_of"] = min(timestamps)

    # 2) Load SMA200 values from persisted wealth parquet (fast lookup)
    sma_map = {}
    try:
        if os.path.exists(WEALTH_PATH):
            prev = pd.read_parquet(WEALTH_PATH)
            if "Stock" in prev.columns and "sma_200" in prev.columns:
                # [OPTIMIZATION] Replaced O(N) iterrows with vectorized set_index and to_dict
                sma_map = prev.set_index("Stock")["sma_200"].to_dict()
                # Handle NaNs
                sma_map = {k: (float(v) if pd.notna(v) else None) for k, v in sma_map.items()}
    except Exception as e:
        logger.warning(f"Could not read wealth parquet for SMA lookup: {e}")

    # 3) Fill missing SMA200 by computing from available recent daily frames only when possible
    for sym in symbols:
        if sym in sma_map and sma_map[sym] is not None:
            result["sma_200"][sym] = sma_map[sym]
            continue
        df = result["daily"].get(sym)
        try:
            if df is None or df.empty:
                result["sma_200"][sym] = None
                continue
            # If we have at least 200 bars in the recent fetch (unlikely for 5d), compute; else leave None
            if len(df) >= 200 and "Close" in df.columns:
                sma_val = float(df["Close"].tail(200).mean())
                result["sma_200"][sym] = sma_val
            else:
                result["sma_200"][sym] = None
        except Exception:
            result["sma_200"][sym] = None

    return result


def get_price_cache_stats() -> dict:
    """Calculates number of keys, total symbol dataframes, and estimated memory in MB."""
    with _lock:
        keys_count = len(_cache)
        total_dfs = 0
        total_bytes = 0
        for entry in _cache.values():
            if isinstance(entry, dict) and "data" in entry and isinstance(entry["data"], dict):
                total_dfs += len(entry["data"])
                for df in entry["data"].values():
                    if isinstance(df, pd.DataFrame):
                        try:
                            total_bytes += df.memory_usage(deep=False).sum()
                        except Exception:
                            pass
        return {
            "keys": keys_count,
            "entries": total_dfs,
            "memory_mb": round(total_bytes / (1024 * 1024), 2)
        }


def clear_price_cache():
    """Explicitly release all in-memory price dataframes and trim heap allocation."""
    stats_before = get_price_cache_stats()
    with _lock:
        for k, entry in list(_cache.items()):
            if isinstance(entry, dict):
                data = entry.get("data")
                if isinstance(data, dict):
                    data.clear()
                entry.clear()
        _cache.clear()
    gc.collect()
    try:
        import sys
        if sys.platform.startswith("linux"):
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    stats_after = get_price_cache_stats()
    logger.info(
        f"🧹 [PRICE_CACHE PURGE] Before: keys={stats_before['keys']} | entries={stats_before['entries']} | memory={stats_before['memory_mb']} MB → "
        f"After: keys={stats_after['keys']} | entries={stats_after['entries']} | memory={stats_after['memory_mb']} MB"
    )
    return stats_before, stats_after


_FAST_CMP_MEMO: dict = {}  # symbol -> (price, source, is_live, timestamp, cache_time)

def get_cached_df(symbol: str, interval: str = "1d", period: str = "1y") -> pd.DataFrame:
    """Retrieve a cached dataframe from RAM or disk parquet without making network requests."""
    key = (interval, period)
    with _lock:
        if key in _cache and isinstance(_cache[key], dict):
            entry = _cache[key].get(symbol)
            if entry and isinstance(entry, dict) and isinstance(entry.get("data"), pd.DataFrame):
                from trading_calendar import enforce_trading_day_candles
                return enforce_trading_day_candles(entry["data"], symbol)
            elif isinstance(_cache[key].get("data"), dict):
                df = _cache[key]["data"].get(symbol)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    from trading_calendar import enforce_trading_day_candles
                    return enforce_trading_day_candles(df, symbol)

    # Disk fallback (variant-aware)
    clean_s = str(symbol).split(":")[-1].strip()
    variants = [
        clean_s,
        clean_s.replace("&", "_"),
        clean_s.replace("-", "_"),
        clean_s.replace("&", "-"),
        clean_s.replace("-EQ", ""),
        clean_s.replace("_EQ", ""),
        f"{clean_s}.NS",
        f"{clean_s.replace('&', '_')}.NS"
    ]
    for v in variants:
        file_path = os.path.join(DATA_DIR, "history", interval, f"{v}.parquet")
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                if df is not None and not df.empty:
                    from trading_calendar import enforce_trading_day_candles
                    df = enforce_trading_day_candles(df, symbol)
                    
                    # Only populate RAM cache if data is fresh (meets expected latest closed daily bar)
                    is_fresh = True
                    if interval == "1d":
                        t_col = 'Date' if 'Date' in df.columns else ('Datetime' if 'Datetime' in df.columns else None)
                        last_bar_ts = df[t_col].iloc[-1] if t_col else (df.index[-1] if not df.index.empty else None)
                        if last_bar_ts is not None:
                            from market_utils import get_expected_latest_closed_daily_bar
                            if pd.to_datetime(last_bar_ts).date() < get_expected_latest_closed_daily_bar():
                                is_fresh = False

                    if is_fresh:
                        with _lock:
                            if key not in _cache or not isinstance(_cache[key], dict):
                                _cache[key] = {}
                            _cache[key][symbol] = {
                                "data": df,
                                "ts": time.monotonic(),
                                "timestamp": time.time()
                            }
                    return df
            except Exception:
                pass
    return None


def get_cached_price_details(symbol: str) -> Tuple[Optional[float], str, bool, Optional[str]]:
    """
    [VERSION: CMP_CACHE_PROVENANCE_v1.2] [RULE 67 CHANGE-RATIONALE]
    Resolves price details from RAM live cache, fast in-memory memoization, or daily Parquet cache.
    Uses RAM-first architecture to eliminate blocking synchronous disk scans during API request handling.
    """
    import time as _t_mod
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    clean_sym = str(symbol).split(":")[-1].strip().upper().replace(".NS", "").replace(".BO", "")
    now_mono = _t_mod.monotonic()

    # 1. Check live prices RAM cache first (non-blocking O(1))
    try:
        from live_prices import get_cached_live_price
        p = get_cached_live_price(symbol) or get_cached_live_price(clean_sym)
        if p is not None and float(p) > 0:
            now_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')
            res = (float(p), "LIVE_TICK", True, now_str)
            _FAST_CMP_MEMO[clean_sym] = (*res, now_mono)
            return res
    except Exception:
        pass

    # 2. Check fast RAM CMP memo (30s TTL)
    if clean_sym in _FAST_CMP_MEMO:
        cached_tuple = _FAST_CMP_MEMO[clean_sym]
        if (now_mono - cached_tuple[4]) < 30.0:
            return cached_tuple[0], cached_tuple[1], cached_tuple[2], cached_tuple[3]

    # 3. Fallback to daily close cache (memoized in RAM after first read)
    try:
        df = get_cached_df(symbol, interval="1d", period="10d")
        if df is not None and not df.empty and "Close" in df.columns:
            valid_df = df.dropna(subset=["Close"])
            if not valid_df.empty:
                last_row = valid_df.iloc[-1]
                dt = last_row.get("Datetime") or last_row.name
                dt_str = str(dt) if dt else None
                res = (float(last_row["Close"]), "DAILY_CACHE", False, dt_str)
                _FAST_CMP_MEMO[clean_sym] = (*res, now_mono)
                return res
    except Exception:
        pass

    res = (None, "UNAVAILABLE", False, None)
    _FAST_CMP_MEMO[clean_sym] = (*res, now_mono)
    return res


def get_cached_price(symbol: str) -> Optional[float]:
    """Fast in-memory or cached price lookup for a single symbol."""
    price, _, _, _ = get_cached_price_details(symbol)
    return price





