import os
import sys
import time
import logging
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from zoneinfo import ZoneInfo
from threading import Lock

# Ensure parent directory is in sys.path to access configurations and auth utilities
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_provider import DataFetcher
from validation import ValidationEngine, MarketData, ValidationContext, registry as val_registry, DatasetType

import fyers_auth
import config

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
_last_auth_notif_time = 0

# ── Process-lifetime permission error tracker ──────────────────────────────────
# Tracks consecutive -403 permission errors across symbols. When all stocks fail
# with -403 (Fyers Historical Data API permission not enabled), open the circuit
# breaker to route all traffic immediately to Yahoo Finance.
import threading as _threading
_perm_error_lock = _threading.Lock()
_perm_error_count = 0          # consecutive permission errors
_perm_error_window_start = 0.0 # monotonic timestamp of first error in current window
_PERM_ERROR_THRESHOLD = 5      # open circuit after this many -403s in 60s
_PERM_ERROR_WINDOW = 60.0      # seconds

def _record_permission_error():
    """Records a per-symbol -403 permission error. Opens circuit breaker if all stocks are blocked."""
    global _perm_error_count, _perm_error_window_start
    with _perm_error_lock:
        now = time.monotonic()
        if now - _perm_error_window_start > _PERM_ERROR_WINDOW:
            _perm_error_count = 0
            _perm_error_window_start = now
        _perm_error_count += 1
        if _perm_error_count >= _PERM_ERROR_THRESHOLD:
            # Trip circuit breaker — all Fyers historical data is permission-blocked
            if _fyers_circuit_breaker.is_available():
                for _ in range(_fyers_circuit_breaker.failure_threshold):
                    _fyers_circuit_breaker.record_failure()
                logger.error(
                    f"🚫 [FYERS CIRCUIT OPEN] {_perm_error_count} consecutive -403 permission errors detected. "
                    "Fyers Historical Data API permission is NOT enabled on your Fyers app. "
                    "Go to developer.fyers.in → App settings → enable 'Historical Data' → regenerate token. "
                    "Routing ALL traffic to Yahoo Finance fallback for 10 minutes."
                )

def _reset_permission_error_counter():
    global _perm_error_count, _perm_error_window_start
    with _perm_error_lock:
        _perm_error_count = 0
        _perm_error_window_start = 0.0

class RateLimiter:
    """Thread-safe rate limiter to space requests and prevent HTTP 429 rate limit errors."""
    def __init__(self, max_per_second: float):
        self.interval = 1.0 / max_per_second
        self.last_call = 0.0
        self.lock = Lock()
        
    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

# [VERSION: DATA_FETCH_ACCELERATION_v1.0] Rate limit increased to 3.0 req/sec with 6 parallel threads
_fyers_rate_limiter = RateLimiter(max_per_second=3.0)

# Circuit breaker for Fyers API to auto-fallback on repeated failures
class FyersCircuitBreaker:
    def __init__(self, failure_threshold: int = 10, reset_after_seconds: int = 300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_after_seconds = reset_after_seconds
        self.last_failure_time = 0
        self.is_open = False
        self.lock = Lock()

    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning(f"⚠️ Fyers API circuit breaker OPENED after {self.failure_count} failures. Falling back to YFinance.")

    def is_available(self) -> bool:
        with self.lock:
            if not self.is_open:
                return True
            # Check if enough time has passed to attempt recovery
            if time.time() - self.last_failure_time > self.reset_after_seconds:
                self.is_open = False
                self.failure_count = 0
                logger.info("✅ Fyers API circuit breaker CLOSED. Attempting recovery.")
                return True
            return False

    def reset(self):
        with self.lock:
            self.failure_count = 0
            self.is_open = False

_fyers_circuit_breaker = FyersCircuitBreaker(failure_threshold=15, reset_after_seconds=600)


class _FyersPermissionError(Exception):
    """
    Sentinel exception for Fyers code -403 (permission not enabled on app).
    Caught separately to:
    - Skip all retries immediately (retrying will never succeed)
    - Skip all candidate variants (all will get same -403)
    - Skip mark_fyers_invalid (symbol IS valid, it's an app permission issue)
    - Increment permission error counter for mass-403 circuit breaker detection
    """
    pass


class FyersFetcher(DataFetcher):
    def __init__(self):
        self.rate_limiter = _fyers_rate_limiter
        
        # Map standard intervals to Fyers V3 resolution parameters per official API spec
        self.INTERVAL_MAP = {
            "1m": "1",
            "1min": "1",
            "1minute": "1",
            "2m": "2",
            "2min": "2",
            "3m": "3",
            "3min": "3",
            "5m": "5",
            "5min": "5",
            "5minute": "5",
            "10m": "10",
            "10min": "10",
            "15m": "15",
            "15min": "15",
            "15minute": "15",
            "20m": "20",
            "20min": "20",
            "30m": "30",
            "30min": "30",
            "30minute": "30",
            "45m": "45",
            "45min": "45",
            "60m": "60",
            "60min": "60",
            "60minute": "60",
            "1h": "60",
            "1hour": "60",
            "120m": "120",
            "180m": "180",
            "240m": "240",
            "1d": "1D",
            "d": "1D",
            "daily": "1D",
            "day": "1D",
            "1w": "1W",
            "1wk": "1W",
            "weekly": "1W",
            "week": "1W",
            "1mo": "1M",
            "monthly": "1M",
            "month": "1M"
        }

    def _normalize_symbol(self, symbol: str) -> str:
        """Translates standard symbols (e.g. RELIANCE, FIVESTAR.NS, ^NSEI) to Fyers specific formats.
        Also trims whitespace and normalizes casing to avoid "Invalid input" caused by trailing spaces or stray newlines.
        """
        # [VERSION: NULL_POINTER_FIX_v1.0] Guard against missing symbols from upstream
        if not symbol:
            return ""
        # Trim invisible characters first
        symbol = str(symbol).strip()
        sym = symbol.upper()
        is_bse = sym.endswith(".BO") or sym.startswith("BSE:")

        # Check persistent learned mappings FIRST for all symbols (including BSE: prefixes)
        try:
            from data_providers.fyers_mapping_utils import load_fyers_mappings
            f_map = load_fyers_mappings()
            if symbol in f_map:
                return f_map[symbol]
            if sym in f_map:
                return f_map[sym]
        except Exception:
            pass

        # If already formatted with exchange prefix and no custom mapping exists, return as is
        if sym.startswith("NSE:") or sym.startswith("BSE:") or sym.startswith("MCX:"):
            if sym.startswith("BSE:") and not any(sym.endswith(sfx) for sfx in ("-EQ", "-BE", "-SM", "-ST", "-A", "-B", "-T", "-M", "-X", "-XC", "-XD", "-XT", "-INDEX")):
                return f"{sym}-EQ"
            return sym
        if sym.endswith(".NS") or sym.endswith(".BO"):
            sym = sym[:-3]
            
        # [VERSION: FYERS_PATCH_v1.0] Intercept ampersand symbols before blind replace
        # This fixes Fyers API warnings for M-M-EQ by enforcing M&M regardless of DB state
        _ampersand_map = {
            "M_M": "M&M", "M-M": "M&M",
            "M_MFIN": "M&MFIN", "M-MFIN": "M&MFIN",
            "J_KBANK": "J&KBANK", "J-KBANK": "J&KBANK",
            "GVT_D": "GVT&D", "GVT-D": "GVT&D",
            "L_TFH": "L&TFH", "L-TFH": "L&TFH",
            "T_IPOWER": "T&IPOWER", "T-IPOWER": "T&IPOWER",
        }
        if sym in _ampersand_map:
            sym = _ampersand_map[sym]
        else:
            sym = sym.replace("_", "-")
        
        # Map index symbols
        _FYERS_INDEX_MAP = {
            "^NSEI": "NSE:NIFTY50-INDEX", "NIFTY": "NSE:NIFTY50-INDEX", "NIFTY-50": "NSE:NIFTY50-INDEX", "NIFTY 50": "NSE:NIFTY50-INDEX", "NSEI": "NSE:NIFTY50-INDEX",
            "^NSEBANK": "NSE:NIFTYBANK-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX", "NIFTYBANK": "NSE:NIFTYBANK-INDEX", "NSEBANK": "NSE:NIFTYBANK-INDEX",
            "^BSESN": "BSE:SENSEX-INDEX", "SENSEX": "BSE:SENSEX-INDEX", "BSESN": "BSE:SENSEX-INDEX",
            "^CNXIT": "NSE:NIFTYIT-INDEX", "NIFTYIT": "NSE:NIFTYIT-INDEX",
            "^CNXAUTO": "NSE:NIFTYAUTO-INDEX", "NIFTYAUTO": "NSE:NIFTYAUTO-INDEX",
            "^CNXFMCG": "NSE:NIFTYFMCG-INDEX", "NIFTYFMCG": "NSE:NIFTYFMCG-INDEX",
            "^CNXPHARMA": "NSE:NIFTYPHARMA-INDEX", "NIFTYPHARMA": "NSE:NIFTYPHARMA-INDEX",
            "^CNXMETAL": "NSE:NIFTYMETAL-INDEX", "NIFTYMETAL": "NSE:NIFTYMETAL-INDEX",
            "^CNXREALTY": "NSE:NIFTYREALTY-INDEX", "NIFTYREALTY": "NSE:NIFTYREALTY-INDEX",
            "^CNXENERGY": "NSE:NIFTYENERGY-INDEX", "NIFTYENERGY": "NSE:NIFTYENERGY-INDEX",
            "^CNXINFRA": "NSE:NIFTYINFRA-INDEX", "NIFTYINFRA": "NSE:NIFTYINFRA-INDEX",
            "^CNXPSUBANK": "NSE:NIFTYPSUBANK-INDEX", "NIFTYPSUBANK": "NSE:NIFTYPSUBANK-INDEX",
            "^CNXFIN": "NSE:FINNIFTY-INDEX", "^CNXFINANCE": "NSE:FINNIFTY-INDEX", "NIFTYFINANCE": "NSE:FINNIFTY-INDEX",
            "^CNXCMDT": "NSE:NIFTYCOMMODITIES-INDEX", "^CNXCOMMODITIES": "NSE:NIFTYCOMMODITIES-INDEX", "NIFTYCOMMODITIES": "NSE:NIFTYCOMMODITIES-INDEX",
            "^NSMIDCP": "NSE:NIFTYMIDCAP100-INDEX", "NIFTYMIDCAP": "NSE:NIFTYMIDCAP100-INDEX",
            "^CNXSMALLCAP": "NSE:NIFTYSMALLCAP100-INDEX", "NIFTYSMALLCAP": "NSE:NIFTYSMALLCAP100-INDEX",
            "^NIFTYOILGAS": "NSE:NIFTYOILANDGAS-INDEX",
            "^NIFTYHEALTHCARE": "NSE:NIFTYHEALTHCARE-INDEX",
            "^NIFTYCONSRDURBL": "NSE:NIFTYCONSRDURBL-INDEX",
            "^CNXMEDIA": "NSE:NIFTYMEDIA-INDEX", "NIFTYMEDIA": "NSE:NIFTYMEDIA-INDEX", "NIFTY MEDIA": "NSE:NIFTYMEDIA-INDEX",
            "^INDIAVIX": "NSE:INDIAVIX-INDEX", "INDIAVIX": "NSE:INDIAVIX-INDEX", "INDIA VIX": "NSE:INDIAVIX-INDEX", "VIX": "NSE:INDIAVIX-INDEX",
        }
        if sym in _FYERS_INDEX_MAP:
            return _FYERS_INDEX_MAP[sym]
            
        if sym.startswith("^"):
            # Generic index format
            return f"NSE:{sym[1:]}-INDEX"
        
        # [VERSION: FYERS_SCRIP_OVERRIDE_v1.1] Static overrides for stocks where Fyers uses custom codes.
        _bse_scrip_overrides = {"NSDL": "BSE:NSDL-EQ"}
        if sym in _bse_scrip_overrides:
            return _bse_scrip_overrides[sym]
            
        # Use institutional SymbolResolutionService (O(1) memory hotpath + DB registry + auto-healing)
        try:
            from symbol_resolution_engine import get_symbol_resolver
            resolved = get_symbol_resolver().resolve(symbol, provider="fyers")
            if resolved and resolved.is_valid and resolved.mapped_symbol:
                return resolved.mapped_symbol
        except Exception as ex:
            logger.debug(f"SymbolResolutionService fallback for Fyers {symbol}: {ex}")

        # Standard stock format fallback
        prefix = "BSE:" if is_bse else "NSE:"
        return f"{prefix}{sym}-EQ"


    def _get_date_range(self, period: str) -> tuple[str, str]:
        """Calculates historical range_from and range_to date strings based on period string.
        For 'y' (year) requests, cap at 365 days to avoid Fyers 'Invalid input' on daily resolution.
        Uses zero-padded YYYY-MM-DD strings.
        """
        today = datetime.now(IST).date()
        days_back = 30
        p = (period or "").lower()

        if p.endswith("d"):
            try:
                days_back = int(p[:-1])
            except ValueError:
                days_back = 5
            buffer_days = max(3, int(days_back * 0.2))
        elif p.endswith("mo") or (p.endswith("m") and len(p) > 1):
            unit = p[:-2] if p.endswith("mo") else p[:-1]
            try:
                days_back = int(unit) * 30
            except ValueError:
                days_back = 30
            buffer_days = max(5, int(days_back * 0.25))
        elif p.endswith("y"):
            try:
                requested_years = int(p[:-1])
            except ValueError:
                requested_years = 1
            # Cap any yearly request to at most 365 days per single call
            days_back = min(requested_years * 365, 365)
            buffer_days = 0
        elif p == "max":
            days_back = 365 * 5  # keep as-is for non-daily resolutions
            buffer_days = int(days_back * 0.2)
        else:
            # default last 30 days
            days_back = 30
            buffer_days = max(3, int(days_back * 0.2))

        range_from = (today - timedelta(days=days_back + buffer_days)).strftime("%Y-%m-%d")
        range_to = today.strftime("%Y-%m-%d")
        return range_from, range_to

    def _generate_fyers_candidate_symbols(self, symbol: str) -> list[str]:
        """
        Generates an exhaustive, multi-exchange & multi-series candidate list for Fyers API.
        Supports: Mainboard (-EQ), Trade-to-Trade/ASM/GSM (-BE), SME (-SM, -ST),
        BSE Mainboard (BSE:SYMBOL-EQ, BSE:SYMBOL), and BSE Scrip Codes (BSE:5XXXXX-EQ).
        Prioritizes BSE series for BSE-preference symbols (.BO, BSE:, or BSE-mapped).
        """
        raw = str(symbol).strip().upper()
        norm = self._normalize_symbol(raw)
        if norm and norm.endswith("-INDEX"):
            return [norm]
        _ampersand_map = {
            "M_M": "M&M", "M-M": "M&M",
            "M_MFIN": "M&MFIN", "M-MFIN": "M&MFIN",
            "J_KBANK": "J&KBANK", "J-KBANK": "J&KBANK",
        }
        if raw in _ampersand_map:
            raw = _ampersand_map[raw]

        is_bse_pref = raw.startswith("BSE:") or raw.endswith(".BO")

        if raw.endswith(".NS") or raw.endswith(".BO"):
            raw = raw[:-3]
        if ":" in raw:
            raw = raw.split(":", 1)[1]

        # Strip standard series suffixes
        base = raw
        for suffix in ("-EQ", "-BE", "-SM", "-ST", "-A", "-B", "-T", "-M", "-X", "-XC", "-XD", "-XT"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break

        # Check if known in BSE mappings
        try:
            from bse_mapping_utils import load_bse_mappings
            if base in load_bse_mappings():
                is_bse_pref = True
        except Exception:
            pass

        candidates = []

        # 0. Check official Fyers Master Contract Resolver first (100% verified tradable symbols from Fyers public CDN)
        try:
            from data_providers.fyers_symbol_mapper import fyers_mapper
            verified_fyers = fyers_mapper.get_fyers_symbol(base)
            if verified_fyers:
                candidates.append(verified_fyers)
        except Exception as mapper_err:
            logger.debug(f"Fyers symbol mapper error: {mapper_err}")

        # If base is numeric (BSE Scrip Code), prioritize BSE:5XXXXX-EQ
        if base.isdigit():
            candidates.append(f"BSE:{base}-EQ")
        elif is_bse_pref:
            # Prioritize BSE:SYMBOL-EQ for BSE-preference stocks, then NSE:SYMBOL-EQ
            candidates.append(f"BSE:{base}-EQ")
            candidates.append(f"NSE:{base}-EQ")
        else:
            # Standard NSE:SYMBOL-EQ first, then BSE:SYMBOL-EQ, then SME series (-SM, -ST, -BE)
            candidates.append(f"NSE:{base}-EQ")
            candidates.append(f"BSE:{base}-EQ")
            candidates.append(f"NSE:{base}-SM")
            candidates.append(f"NSE:{base}-ST")
            candidates.append(f"NSE:{base}-BE")

            # Known BSE scrip code map for stocks with custom Fyers BSE tickers
            # CRITICAL: Always use BSE:CODE-EQ (with -EQ suffix). Bare BSE:CODE
            # is rejected by Fyers API v3 with code -403.
            _KNOWN_BSE_SCRIP_CODES = {
                "POONAWALLA": "524000",
                "PFC": "532648",
                "SENORES": "544256",
                "MRF": "500290",
                "TORNTPHARM": "500420",
                "HINDUNILVR": "500696",
                "HAL": "541154",
                "AADHARHFC": "544175",
                "MTARTECH": "543270",
                "STLTECH": "532374",
                "DIACABS": "532959",
            }
            if base in _KNOWN_BSE_SCRIP_CODES:
                bse_code = _KNOWN_BSE_SCRIP_CODES[base]
                candidates.append(f"BSE:{bse_code}-EQ")

            try:
                from bse_mapping_utils import load_bse_mappings
                bse_map = load_bse_mappings()
                if base in bse_map:
                    clean_code = str(bse_map[base]).upper().replace(".BO", "").replace("BSE:", "").strip()
                    if clean_code.isdigit():
                        candidates.append(f"BSE:{clean_code}-EQ")
            except Exception:
                pass

        # ── FORMAT GATE: validate every candidate against Fyers API v3 rules ──────
        # Strips -BE and bare BSE (auto-fixed) and logs any format violations.
        try:
            from symbol_format_validator import sanitize_fyers_candidate_list
            candidates = sanitize_fyers_candidate_list(candidates)
        except Exception:
            pass

        # Deduplicate preserving order
        seen = set()
        deduped = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped

    def get_ohlcv(self, symbol: str, interval: str, period: str, retries: int = 5, range_from: str = None, range_to: str = None) -> MarketData:
        """Fetch OHLCV data for a single symbol from Fyers."""
        # [VERSION: NULL_POINTER_FIX_v1.0]
        if not symbol:
            return MarketData(None, "UNKNOWN", None, False, False, "No symbol")
            
        # Check if Fyers circuit breaker is open (too many failures)
        if not _fyers_circuit_breaker.is_available():
            return MarketData(None, "Fyers", None, False, False, "Circuit Breaker Open")
        
        ns_symbol = self._normalize_symbol(symbol)
        
        orig_sym = symbol.strip().upper()
        if orig_sym.endswith(".NS"): orig_sym = orig_sym[:-3]
        if orig_sym.startswith("NSE:"): orig_sym = orig_sym[4:]
        if orig_sym.startswith("BSE:"): orig_sym = orig_sym[4:]
        if orig_sym.endswith("-EQ"): orig_sym = orig_sym[:-3]
        if orig_sym.endswith("-BE"): orig_sym = orig_sym[:-3]
        
        _ampersand_map = {
            "M_M": "M&M", "M-M": "M&M",
            "M_MFIN": "M&MFIN", "M-MFIN": "M&MFIN",
            "J_KBANK": "J&KBANK", "J-KBANK": "J&KBANK",
            "GVT_D": "GVT&D", "GVT-D": "GVT&D",
            "L_TFH": "L&TFH", "L-TFH": "L&TFH",
            "T_IPOWER": "T&IPOWER", "T-IPOWER": "T&IPOWER",
        }
        if orig_sym in _ampersand_map:
            orig_sym = _ampersand_map[orig_sym]
        else:
            orig_sym = orig_sym.replace("_", "-")
        # [VERSION: NON_EQUITY_BLOCKLIST_v2.0] Filter only known InvITs/REITs (never ASM/GSM equities)
        from config import NON_EQUITY_BLOCKLIST
        if orig_sym and orig_sym.upper() in NON_EQUITY_BLOCKLIST:
            return None
        try:

            from data_providers.fyers_mapping_utils import is_fyers_invalid
            # Skip the invalid check if this symbol has a known static scrip override
            _scrip_overrides_check = {"NSDL"}  # keep in sync with _normalize_symbol overrides
            if orig_sym not in _scrip_overrides_check and is_fyers_invalid(orig_sym):
                logger.debug(f"⚠️ Skipping known invalid Fyers symbol: {orig_sym}")
                return None
        except Exception:
            pass
            
        tried_suffixes = set()
        original_ns = ns_symbol

        # Determine if this is an incremental fetch
        if range_from and range_to:
            logger.debug(f"📥 Fetching incremental OHLCV for {symbol} ({interval}) from {range_from} to {range_to} via Fyers API...")
            calc_range_from, calc_range_to = range_from, range_to
        else:
            logger.debug(f"📥 Fetching OHLCV for {symbol} ({interval}, {period}) via Fyers API...")
            calc_range_from, calc_range_to = self._get_date_range(period)
        
        # Normalize interval key and map to Fyers resolution
        res = self.INTERVAL_MAP.get(interval.lower()) if isinstance(interval, str) else None
        if not res:
            logger.error(f"Unsupported interval for FyersFetcher: {interval}")
            return None

        # Compute date range and then enforce strict 365-day cap for daily resolution
        range_from, range_to = calc_range_from, calc_range_to
        try:
            start_date = datetime.strptime(range_from, "%Y-%m-%d").date()
            end_date = datetime.strptime(range_to, "%Y-%m-%d").date()
        except Exception:
            # Fall back to safe defaults
            end_date = datetime.now(IST).date()
            start_date = end_date - timedelta(days=30)
            range_from = start_date.strftime("%Y-%m-%d")
            range_to = end_date.strftime("%Y-%m-%d")

        if res in ("1D", "D", "1W", "1M"):
            span_days = (end_date - start_date).days
            if span_days > 365:
                # Cap span to 365 days per Fyers API limits (max 366 days for 1D, 1W, 1M)
                start_date = end_date - timedelta(days=365)
                range_from = start_date.strftime("%Y-%m-%d")
        else:
            # Fyers API enforces a strict 100-day cap for intraday resolutions (1m to 240m / 1h)
            span_days = (end_date - start_date).days
            if span_days > 99:
                start_date = end_date - timedelta(days=99)
                range_from = start_date.strftime("%Y-%m-%d")

        # Multi-series & multi-exchange candidate resolution loop
        candidates = self._generate_fyers_candidate_symbols(orig_sym)
        if ns_symbol not in candidates:
            candidates.insert(0, ns_symbol)

        for cand_symbol in candidates:
            if cand_symbol in tried_suffixes:
                continue

            data = {
                "symbol": cand_symbol,
                "resolution": res,
                "date_format": "1",
                "range_from": range_from,
                "range_to": range_to
            }
            if any(sfx in cand_symbol for sfx in ("-FUT", "-OPT")):
                data["cont_flag"] = "0"
            
            for attempt in range(retries):
                try:
                    client = fyers_auth.get_fyers_client()
                    if not client:
                        logger.error("Fyers API client is uninitialized. Generate a token via /fyers/login.")
                        from core_exceptions import ProviderError
                        raise ProviderError("Fyers Authentication Required")

                    self.rate_limiter.wait()
                    response = client.history(data=data)
                    
                    if not response:
                        raise ValueError("Received empty response from Fyers history API")
                        
                    if response.get("s") != "ok":
                        error_msg = str(response.get("message", "Unknown error"))
                        code = str(response.get("code", "NO_CODE"))
                        
                        # Only break candidate loop if Fyers explicitly returns non-existent symbol error
                        if "invalid symbol" in error_msg.lower() or "invalid input" in error_msg.lower():
                            logger.info(f"Fyers API symbol miss for {cand_symbol} ({error_msg}) - trying next candidate")
                            tried_suffixes.add(cand_symbol)
                            break
                        else:
                            logger.warning(f"Fyers API warning for {cand_symbol}: code={code}, message={error_msg}, full_response={response}")
                        
                        # ── PERMISSION ERROR (-403): Non-retryable. Break immediately. ───────────
                        # code -403 = Fyers Historical Data API permission not enabled on app.
                        # Retrying is futile — will never succeed until app permissions are fixed.
                        # Do NOT mark symbol as INVALID (it IS a valid symbol, just blocked globally).
                        if code in ("-403", "403") or any(k in error_msg.lower() for k in ("permission required", "regenerate access token")):
                            logger.error(f"🚫 Fyers auth/permission error for {cand_symbol} (code {code}): {error_msg}")
                            _record_permission_error()
                            # Raise a sentinel that the outer loop catches to skip mark_fyers_invalid
                            raise _FyersPermissionError(f"Fyers permission error for {cand_symbol} (code {code})")

                        if code in ["494", "-401", "401", "-16", "-15"] or "authenticate" in error_msg.lower():
                            logger.error(f"🚫 Fyers auth error for {cand_symbol} (code {code}): {error_msg}")
                            raise ValueError(f"Fyers auth/permission error for {cand_symbol} (code {code}): {error_msg}")
                            
                        raise ValueError(f"Fyers history API error (code {code}): {error_msg}")
                        
                    candles = response.get("candles", [])
                    if not candles:
                        return MarketData(None, "Fyers", None, False, False, "No data available in response")
                    
                    df = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
                    
                    # Convert Fyers Unix epoch timestamps (seconds) to IST Datetimes
                    timestamps = pd.to_datetime(df["Timestamp"], unit="s", utc=True).dt.tz_convert(IST)
                    
                    # Cast columns to appropriate float types
                    import numpy as np
                    df["Open"] = df["Open"].astype(np.float32)
                    df["High"] = df["High"].astype(np.float32)
                    df["Low"] = df["Low"].astype(np.float32)
                    df["Close"] = df["Close"].astype(np.float32)
                    df["Volume"] = df["Volume"].astype(np.float32)
                    
                    if str(interval).lower() in ("1d", "1w", "1m", "1wk", "1mo"):
                        df["Date"] = pd.to_datetime(timestamps.dt.date)
                        df = df.drop(columns=["Timestamp"], errors="ignore")
                    else:
                        df["Datetime"] = timestamps
                        df = df.drop(columns=["Timestamp"], errors="ignore")

                    # [RULE 67 CHANGE-RATIONALE: SYSTEM-WIDE WEEKEND CANDLE BAN]
                    # Purge any weekend mock or synthetic candles immediately at raw broker API parsing layer
                    from trading_calendar import enforce_trading_day_candles
                    df = enforce_trading_day_candles(df, cand_symbol)
                    
                    # ── Save confirmed mapping after a successful fetch ──────────────
                    if not cand_symbol.endswith("-INDEX"):
                        try:
                            from data_providers.fyers_mapping_utils import save_fyers_mapping
                            save_fyers_mapping(orig_sym, cand_symbol)
                            save_fyers_mapping(symbol.strip().upper(), cand_symbol)
                            from symbol_resolution_engine import get_symbol_resolver, ResolvedInstrument
                            res_obj = ResolvedInstrument(f"EQ:{orig_sym}", orig_sym, "fyers", cand_symbol, cand_symbol.split(":")[0], "EQ", 100, "LEARNED")
                            get_symbol_resolver()._cache_and_persist_mapping("fyers", orig_sym, res_obj)
                        except Exception:
                            pass
                        
                    pipeline = val_registry.get_pipeline(DatasetType.PRICE)
                    engine = ValidationEngine(pipeline.validator, pipeline.score_calculator)
                    r_from_str = range_from.strftime("%Y-%m-%d") if hasattr(range_from, "strftime") else (str(range_from) if range_from else None)
                    r_to_str = range_to.strftime("%Y-%m-%d") if hasattr(range_to, "strftime") else (str(range_to) if range_to else None)
                    ctx = ValidationContext(provider="Fyers", period=period, interval=interval, range_from=r_from_str, range_to=r_to_str, fetch_mode="DELTA" if range_from else "FULL")
                    report = engine.validate(df, ctx)
                    if not report.is_valid and not range_from:
                        return MarketData(None, "Fyers", report, False, False, "Quality Check Failed")
                    return MarketData(df, "Fyers", report, False, False, None if report.is_valid else "Quality Warning")

                    
                except _FyersPermissionError as perm_err:
                    # -403: non-retryable, break entire candidate loop, return None immediately
                    # Caller (price_cache.py) will route this symbol to Yahoo Finance fallback
                    logger.debug(f"⏭️ Fyers permission-blocked for {cand_symbol} — skipping all candidates, routing to Yahoo.")
                    return None

                except Exception as e:
                    error_str = str(e)
                    
                    # Record failure for circuit breaker
                    if "error" in error_str.lower() or "request" in error_str.lower():
                        if not any(k in error_str.lower() for k in ("invalid symbol", "invalid input", "additional permission required", "403")):
                            _fyers_circuit_breaker.record_failure()

                    if "Could not authenticate the user" in error_str:
                        return None
                        
                    # Do not retry for bad symbols; move immediately to the next candidate
                    if "Invalid symbol provided" in error_str:
                        tried_suffixes.add(cand_symbol)
                        logger.info(f"🔄 Fyers candidate {cand_symbol} is invalid. Trying next candidate...")
                        break  # Break inner network retries, try next candidate in outer loop
                        
                    if "Invalid input" in error_str:
                        logger.warning(f"⚠️ Skipping {cand_symbol} — non-retryable Fyers error: {e}")
                        break
                        
                    # Exponential backoff for network rate limits or server errors
                    import random
                    if "request limit reached" in error_str.lower() or "429" in error_str or "bad request" in error_str.lower():
                        backoff_time = random.uniform(1.0, 2.5)
                        logger.info(f"⏳ Fyers history API rate limit (429) for {cand_symbol}. Backing off for {backoff_time:.1f}s... (Attempt {attempt+1}/{retries})")
                        time.sleep(backoff_time)
                    else:
                        logger.warning(f"⚠️ Attempt {attempt+1}/{retries} failed for {cand_symbol}: {e}")
                        time.sleep((2 ** attempt) * 1.5 + random.uniform(0.5, 1.5))

        # [RULE 67 CHANGE-RATIONALE: PRESERVE_SYMBOL_MISS_ERROR_V1.0]
        # Return explicit MarketData error string rather than None when all Fyers symbol candidates fail.
        # RATIONALE: Returning None causes data_provider.py line 739 to fallback to 'No data returned',
        # which classify_error_code() maps to ProviderErrorCode.UNKNOWN instead of UNSUPPORTED_SYMBOL.
        # By passing the explicit 'Invalid symbol / symbol miss' error, symbol_router correctly classifies
        # the failure as UNSUPPORTED_SYMBOL and permanently routes the ticker to UPSTOX_ONLY across all intervals.
        logger.info(f"⚠️ All Fyers series candidates failed for {orig_sym} ({candidates}). Returning explicit symbol-miss error for Upstox fallback.")
        return MarketData(
            dataframe=None,
            source="Fyers",
            quality_report=None,
            is_fallback=False,
            is_stale=False,
            error=f"Invalid symbol: All Fyers series candidates failed for {orig_sym} ({candidates})"
        )

    def get_batch_ohlcv(self, symbols: list[str], interval: str, period: str, retries: int = 5, range_from: str = None, range_to: str = None, caller: str = None) -> dict[str, MarketData]:
        """Fetch OHLCV data for multiple symbols concurrently using ThreadPoolExecutor."""

        # Check if Fyers circuit breaker is open (too many failures)
        if not _fyers_circuit_breaker.is_available():
            logger.warning(f"🚫 Fyers Circuit Breaker is OPEN. Skipping Fyers batch fetch for {len(symbols)} symbols.")
            return {}

        prefix = f"[{caller}] " if caller else ""
        if range_from and range_to:
            logger.info(f"{prefix}📥 Fetching incremental batch OHLCV for {len(symbols)} symbols ({interval}, {range_from} to {range_to}) via Fyers API...")
        else:
            logger.info(f"{prefix}📥 Fetching batch OHLCV for {len(symbols)} symbols ({interval}, {period}) via Fyers API...")
            
        normalized_map = {}
        for s in symbols:
            # [VERSION: NULL_POINTER_FIX_v1.0] Prevent None leaks from batch dataframe extraction
            if not s:
                continue
            orig = s.strip() if isinstance(s, str) else str(s)
            ns_sym = self._normalize_symbol(orig)
            if not ns_sym:
                continue
            if ns_sym not in normalized_map:
                normalized_map[ns_sym] = []
            normalized_map[ns_sym].append(orig)
            
        ns_symbols = list(normalized_map.keys())
        results = {}
        
        # [VERSION: FYERS_CONCURRENCY_ACCELERATION_v3.0] Increased max workers to 8 for fast parallel historical fetches
        max_workers = min(8, len(ns_symbols) if ns_symbols else 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ns = {
                executor.submit(self.get_ohlcv, normalized_map[ns_sym][0], interval, period, retries, range_from, range_to): ns_sym
                for ns_sym in ns_symbols
            }
            
            completed = 0
            total = len(future_to_ns)
            try:
                # Dynamic timeout based on list size, max 1800s (30 mins)
                # 1.5 req/sec = ~666 ms per request. Add generous buffer for backoffs.
                calc_timeout = min(1800, max(300, len(ns_symbols) * 2))
                for future in concurrent.futures.as_completed(future_to_ns, timeout=calc_timeout):
                    ns_sym = future_to_ns[future]
                    completed += 1
                    if completed % 50 == 0 or completed == total:
                        logger.info(f"{prefix}⏳ Progress: Fetched {completed}/{total} symbols from Fyers...")
                        
                    try:
                        df = future.result()
                        # Map dataframe to all requested symbols mapping to this normalized symbol
                        for orig_sym in normalized_map[ns_sym]:
                            results[orig_sym] = df
                    except Exception as e:
                        logger.exception(f"Error fetching batch OHLCV for {ns_sym}")
                        for orig_sym in normalized_map[ns_sym]:
                            results[orig_sym] = MarketData(None, "Fyers", None, False, False, "Exception")
            except concurrent.futures.TimeoutError:
                logger.error(f"Fyers batch fetch timed out after {calc_timeout}s. Cancelling remaining fetches.")
                # Forcibly open the circuit breaker to prevent subsequent batches from hanging
                for _ in range(_fyers_circuit_breaker.failure_threshold):
                    _fyers_circuit_breaker.record_failure()
                pass
                        
        for s in symbols:
            if s not in results:
                results[s] = MarketData(None, "Fyers", None, False, False, "Missing")
                        
        return results

    def get_quote(self, symbol: str) -> dict:
        """Fetch current market quote for a single symbol from Fyers."""

        # Check if Fyers circuit breaker is open (too many failures)
        if not _fyers_circuit_breaker.is_available():
            return {}

        ns_symbol = self._normalize_symbol(symbol)
        
        orig_sym = symbol.strip().upper()
        if orig_sym.endswith(".NS"): orig_sym = orig_sym[:-3]
        
        _ampersand_map = {
            "M_M": "M&M", "M-M": "M&M",
            "M_MFIN": "M&MFIN", "M-MFIN": "M&MFIN",
            "J_KBANK": "J&KBANK", "J-KBANK": "J&KBANK",
            "GVT_D": "GVT&D", "GVT-D": "GVT&D",
            "L_TFH": "L&TFH", "L-TFH": "L&TFH",
            "T_IPOWER": "T&IPOWER", "T-IPOWER": "T&IPOWER",
        }
        if orig_sym in _ampersand_map:
            orig_sym = _ampersand_map[orig_sym]
        else:
            orig_sym = orig_sym.replace("_", "-")
        try:
            from data_providers.fyers_mapping_utils import is_fyers_invalid
            if is_fyers_invalid(orig_sym):
                logger.debug(f"⚠️ Skipping known invalid Fyers symbol for quotes: {orig_sym}")
                return {}
        except Exception:
            pass
        logger.info(f"📥 Fetching quote for {symbol} via Fyers API...")
        client = fyers_auth.get_fyers_client()
        if not client:
            logger.error("Fyers API client not initialized.")
            return {}
            
        data = {
            "symbols": ns_symbol
        }
        
        try:
            self.rate_limiter.wait()
            response = client.quotes(data=data)
            
            if response and response.get("s") == "ok" and response.get("d"):
                quote_data = response["d"][0]
                v = quote_data.get("v", {})
                
                # Mimic standard YFinance ticker info dictionary structure
                close_price = v.get("close", 0.0)
                net_change = v.get("ch", 0.0)
                prev_close = close_price - net_change
                
                return {
                    "regularMarketPrice": v.get("lp", close_price),
                    "currentPrice": v.get("lp", close_price),
                    "open": v.get("open"),
                    "dayHigh": v.get("high"),
                    "dayLow": v.get("low"),
                    "previousClose": prev_close,
                    "volume": v.get("volume"),
                    "symbol": symbol
                }
            else:
                error_msg = response.get("message", "Unknown error") if response else "Empty response"
                code = response.get("code", "NO_CODE") if response else "NO_CODE"
                logger.warning(f"Fyers quotes API returned warning for {ns_symbol}: {error_msg}, code={code}")
                
                if str(code) in ["494", "-401", "401", "-16", "-15"]:
                    logger.warning(f"Fyers quotes auth warning for {ns_symbol} (code {code}): {error_msg}")
                    # Trigger the 'Could not authenticate' handling below
                    raise ValueError("Could not authenticate the user")
                    
                if "invalid symbol" not in error_msg.lower():
                    _fyers_circuit_breaker.record_failure()
                    
                try:
                    from data_fetch_status import mark_failure
                    mark_failure('fyers', f"Quote API error for {symbol}: {error_msg}")
                except Exception:
                    pass
                return {}
        except Exception as e:
            error_str = str(e)
            # Record failure for circuit breaker
            if "error" in error_str.lower() or "request" in error_str.lower():
                if "invalid symbol" not in error_str.lower() and "invalid input" not in error_str.lower():
                    _fyers_circuit_breaker.record_failure()

            if "Could not authenticate the user" in error_str:
                logger.error("Fyers API authentication expired or invalid.")
                from core_exceptions import ProviderError
                raise ProviderError("Fyers Authentication Required")

            logger.exception(f"Failed to fetch quote for symbol {symbol}")
            try:
                from data_fetch_status import mark_failure
                mark_failure('fyers', f"Quote fetch exception for {symbol}: {str(e)}")
            except Exception:
                pass
            return {}

    def verify_historical_scope(self) -> bool:
        """
        Startup Gate:
        - First check DB if token exist.
        - If yes, try one sample quote and fetch historical data to test.
        - If fails (or if token doesn't exist initially), regenerate token using scraper and save in DB.
        - Check access again.
        - If it still doesn't work, notify admin and continue normal flow.
        """
        def _test_access(client):
            try:
                # 1. Test Sample Quote
                q_data = {"symbols": "NSE:TCS-EQ"}
                self.rate_limiter.wait()
                q_res = client.quotes(data=q_data)
                if not q_res or q_res.get("s") != "ok":
                    return False, f"Quote test failed: {q_res}"

                # 2. Test Historical Data
                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                range_from = (datetime.now(IST) - timedelta(days=5)).strftime("%Y-%m-%d")
                h_data = {
                    "symbol": "NSE:TCS-EQ",
                    "resolution": "1D",
                    "date_format": "1",
                    "range_from": range_from,
                    "range_to": today_str
                }
                self.rate_limiter.wait()
                h_res = client.history(data=h_data)
                
                if not h_res or h_res.get("s") != "ok":
                    error_msg = str(h_res.get("message", "Unknown response")) if h_res else "Empty response"
                    code = str(h_res.get("code", "NO_CODE")) if h_res else "NO_CODE"
                    return False, f"History test failed (code {code}): {error_msg}"
                
                return True, "OK"
            except Exception as e:
                return False, f"Exception during test: {e}"

        try:
            # Step 1: Get client (this implicitly checks DB, and if missing, does 1st scraper attempt via get_access_token)
            client = fyers_auth.get_fyers_client()
            if not client:
                logger.warning("⚠️ Fyers startup check: No token available initially. Will attempt generation.")
                success = False
            else:
                logger.info("Fyers token found/generated. Testing access (quote + history)...")
                success, msg = _test_access(client)
                if success:
                    logger.info("✅ [FYERS STARTUP GATE] Token VERIFIED successfully (Quote + History).")
                    self._update_db_health("OK", None)
                    return True
                else:
                    logger.warning(f"⚠️ Initial Fyers token test failed: {msg}")

            # Step 2: If we are here, initial token test failed OR token was completely missing.
            # Regenerate token using scraper.
            logger.info("🔄 Regenerating Fyers token using scraper...")
            fyers_auth.clear_token(force=True)
            
            # auto_login will generate via scraper and save to DB
            new_token = fyers_auth.auto_login()
            if not new_token:
                logger.error("❌ Token regeneration via scraper failed.")
                final_success = False
                final_msg = "Token regeneration failed."
            else:
                # Step 3: Check access again with the new token
                new_client = fyers_auth.get_fyers_client()
                if new_client:
                    final_success, final_msg = _test_access(new_client)
                else:
                    final_success, final_msg = False, "Could not initialize client with new token."

            if final_success:
                logger.info("✅ [FYERS STARTUP GATE] Regenerated Token VERIFIED successfully.")
                self._update_db_health("OK", None)
                return True
            else:
                # Step 4: If it still doesn't work, notify admin and continue normal flow
                _perm_error_msg = f"🚫 [FYERS STARTUP ERROR] Token test failed after regeneration: {final_msg}. Continuing normal flow (fallback order: Upstox -> Yahoo)."
                logger.error(_perm_error_msg)
                
                # Open circuit breaker immediately
                for _ in range(_fyers_circuit_breaker.failure_threshold):
                    _fyers_circuit_breaker.record_failure()

                # Persist PERMISSION_DENIED or ERROR status in Postgres DB
                self._update_db_health("ERROR", _perm_error_msg)

                # Send WebPush / Bell notification to notify admin
                try:
                    fyers_auth.dispatch_fyers_reauth_notification(
                        f"Fyers token test failed: {final_msg}. Check app permissions or credentials."
                    )
                except Exception:
                    pass
                
                # Continue normal flow (return False, but app shouldn't crash)
                return False

        except Exception as e:
            logger.warning(f"⚠️ Fyers startup verification exception: {e}")
            self._update_db_health("ERROR", str(e))
            return False

    def _update_db_health(self, status: str, error_msg: Optional[str]):
        """Persists Fyers provider health state into PostgreSQL scanner_health DB table."""
        try:
            from database import upsert_scanner_health
            upsert_scanner_health(
                scanner_name="FYERS_PROVIDER",
                status=status,
                error_msg=error_msg[:500] if error_msg else None,
                last_success=datetime.now(IST).isoformat() if status == "OK" else None,
                scheduled_for="Startup / Active Session"
            )
        except Exception as db_err:
            logger.debug(f"Failed to persist FYERS_PROVIDER health to DB: {db_err}")


def verify_fyers_startup_scope() -> bool:
    """Module-level helper to trigger Fyers historical scope check at process startup."""
    try:
        fetcher = FyersFetcher()
        return fetcher.verify_historical_scope()
    except Exception as e:
        logger.warning(f"Failed to run verify_fyers_startup_scope: {e}")
        return False

