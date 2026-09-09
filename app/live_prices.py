import logging
from typing import Dict, List, Optional, Tuple, Any
import time
import os
from data_providers.unified_fetcher import fetcher

logger = logging.getLogger(__name__)
_dead_symbols_cache = {}
# [RULE 67 CHANGE-RATIONALE]: Reduced from 24h to 30 minutes.
# A 24h dead TTL caused HEG and similar stocks to be permanently blocked after a single
# transient quote failure (provider rate-limit / token expiry), producing repeated
# '🚨 No live price available' ERROR spam in performance_tracker every 5 minutes.
# 30 minutes allows the symbol to be retried on the next performance tracker cycle.
_DEAD_TTL = 30 * 60  # 30 minutes
_MAX_DEAD_CACHE_SIZE = 1000

_recent_quotes_cache = {}
_RECENT_TTL = 60  # 60 seconds shared TTL across all scanners & dashboard endpoints
import threading
_live_prices_lock = threading.Lock()

def _get_dead_symbols() -> dict:
    from session_context import get_session_cache_or_fallback
    return get_session_cache_or_fallback("dead_symbols", _dead_symbols_cache, logger)

def _cleanup_dead_symbols_cache():
    now = time.time()
    cache = _get_dead_symbols()
    # 1. Remove expired
    expired_keys = [k for k, v in cache.items() if now - v > _DEAD_TTL]
    for k in expired_keys:
        del cache[k]
        
    # 2. If still over limit, remove oldest (Python 3.7+ dicts preserve insertion order)
    if len(cache) > _MAX_DEAD_CACHE_SIZE:
        excess = len(cache) - _MAX_DEAD_CACHE_SIZE
        oldest_keys = list(cache.keys())[:excess]
        for k in oldest_keys:
            del cache[k]
        logger.info(f"🧹 Evicted {len(expired_keys)} expired and {len(oldest_keys)} oldest entries from _dead_symbols_cache.")

def _store_symbol_aliases_in_cache(sym: str, val: float, now_ts: float):
    """Store symbol and all canonical/exchange aliases in _recent_quotes_cache to prevent cache bypasses."""
    if val is None or float(val) <= 0:
        return
    clean_sym = str(sym).replace('.NS', '').replace('.BO', '').replace('BSE:', '').replace('NSE:', '').strip().upper()
    aliases = {sym, clean_sym, f"{clean_sym}.NS", f"NSE:{clean_sym}", f"BSE:{clean_sym}"}
    for alias in aliases:
        _recent_quotes_cache[alias] = {"price": float(val), "ts": now_ts}

def bulk_warmup_live_prices(symbols: List[str]) -> Dict[str, float]:
    """
    [STANDARD 6: SHARED LIVE PRICE LAYER]
    Pre-fetches live quotes for a list of symbols in ONE single bulk API call before scanner loops begin.
    Populates _recent_quotes_cache in RAM so all subsequent single-symbol lookups hit memory in 0ms.
    """
    if not symbols:
        return {}
    return get_live_prices(symbols)

def get_cached_live_price(symbol: str) -> Optional[float]:
    """
    [RAM-ONLY CMP LOOKUP]
    Checks _recent_quotes_cache for a fresh CMP quote (< 60s) without making network requests.
    Returns float price if found, otherwise None.
    """
    if not symbol:
        return None
    now = time.time()
    clean_s = str(symbol).replace('.NS', '').replace('.BO', '').replace('BSE:', '').replace('NSE:', '').strip().upper()
    with _live_prices_lock:
        cached_entry = (
            _recent_quotes_cache.get(symbol) or 
            _recent_quotes_cache.get(clean_s) or 
            _recent_quotes_cache.get(f"{clean_s}.NS") or 
            _recent_quotes_cache.get(f"NSE:{clean_s}")
        )
        if cached_entry and (now - cached_entry["ts"]) < _RECENT_TTL:
            return cached_entry["price"]
    return None

ALLOWED_SINGLE_SYMBOL_PURPOSES = {
    "TRADE_EXECUTION_VERIFY", 
    "MANUAL_USER_ANALYSIS", 
    "ALERT_PERSISTENCE"
}

_quote_access_metrics = {
    "single_symbol_external_attempts": 0,
    "single_symbol_external_blocked": 0,
    "single_symbol_external_allowed": 0,
    "bulk_external_requests": 0,
    "bulk_symbols_requested": 0,
}

def get_quote_access_audit_stats() -> dict:
    """Returns a snapshot of quote access circuit breaker metrics."""
    with _live_prices_lock:
        return dict(_quote_access_metrics)

def reset_quote_access_audit_stats():
    """Resets audit counters for a clean scanner test run."""
    with _live_prices_lock:
        for k in _quote_access_metrics:
            _quote_access_metrics[k] = 0

def get_live_prices(symbols: List[str], purpose: str = "UNSPECIFIED") -> Dict[str, float]:
    """
    Fetches real-time Last Traded Price (CMP) for a list of standard NSE symbols.
    Routes through UnifiedFetcher for provider enforcement and telemetry.
    Uses alias matching to maximize shared cache hits and eliminate 1-symbol network roundtrips.

    [INVARIANT 4 CIRCUIT BREAKER]: Single-symbol network requests are blocked unless
    explicitly authorized via ALLOWED_SINGLE_SYMBOL_PURPOSES.
    """
    if not symbols:
        return {}

    with _live_prices_lock:
        if len(symbols) == 1:
            _quote_access_metrics["single_symbol_external_attempts"] += 1
        else:
            _quote_access_metrics["bulk_external_requests"] += 1
            _quote_access_metrics["bulk_symbols_requested"] += len(symbols)

    # Invariant 4 Circuit Breaker Enforcement
    if len(symbols) == 1 and purpose not in ALLOWED_SINGLE_SYMBOL_PURPOSES:
        with _live_prices_lock:
            _quote_access_metrics["single_symbol_external_blocked"] += 1

        import inspect
        caller_frame = inspect.currentframe().f_back
        caller_info = f"{os.path.basename(caller_frame.f_code.co_filename)}:{caller_frame.f_lineno}->{caller_frame.f_code.co_name}" if caller_frame else "UNKNOWN"
        sym = symbols[0]
        ram_price = get_cached_live_price(sym)
        logger.warning(
            f"🚨 [QUOTE_ACCESS_VIOLATION_BLOCKED] Single-symbol external quote request attempted & blocked! "
            f"symbol='{sym}' purpose='{purpose}' caller='{caller_info}'. Returning RAM cache quote ({ram_price}) — network HTTP request BLOCKED."
        )
        return {sym: ram_price} if ram_price is not None else {}

    if len(symbols) == 1:
        with _live_prices_lock:
            _quote_access_metrics["single_symbol_external_allowed"] += 1

    now = time.time()
    cache = _get_dead_symbols()
    valid_symbols = []
    prices = {}
    
    with _live_prices_lock:
        # Check recent fast-cache first (with alias normalization)
        for s in symbols:
            if not s or not isinstance(s, str):
                continue
            clean_s = str(s).replace('.NS', '').replace('.BO', '').replace('BSE:', '').replace('NSE:', '').strip().upper()
            if not clean_s or clean_s in ('?', 'NONE', 'NAN', 'NULL', 'UNKNOWN') or not any(c.isalnum() for c in clean_s):
                continue
            cached_entry = _recent_quotes_cache.get(s) or _recent_quotes_cache.get(clean_s) or _recent_quotes_cache.get(f"{clean_s}.NS") or _recent_quotes_cache.get(f"NSE:{clean_s}")
            if cached_entry and (now - cached_entry["ts"]) < _RECENT_TTL:
                prices[s] = cached_entry["price"]
            else:
                if s not in cache or (now - cache.get(s, 0)) >= _DEAD_TTL:
                    valid_symbols.append(s)

    if not valid_symbols:
        return prices

    # Delegate complex fallback, mapping, and chunking to UnifiedFetcher
    results = fetcher.fetch_live_quotes(valid_symbols, consumer=f"live_prices:{purpose}")
    
    new_prices = {}
    for sym, quote in results.items():
        if "v" in quote and "cmd" in quote["v"]:
            try:
                val = float(quote["v"]["cmd"]["c"])
                new_prices[sym] = val
                prices[sym] = val
            except (ValueError, TypeError):
                pass
                
    with _live_prices_lock:
        for sym, val in new_prices.items():
            _store_symbol_aliases_in_cache(sym, val, now)
            
        # Clean up stale entries to prevent memory leak
        stale_keys = [k for k, v in _recent_quotes_cache.items() if (now - v["ts"]) > _RECENT_TTL * 2]
        for k in stale_keys:
            del _recent_quotes_cache[k]
                
    # Evaluate completely dead symbols (not returned by any provider)
    missing = set(valid_symbols) - set(new_prices.keys())
    if missing:
        _cleanup_dead_symbols_cache()
        for s in missing:
            cache[s] = time.time()
            logger.warning(f"🚫 Marking {s} as completely DEAD for 24h (failed across all configured providers).")

    return prices
