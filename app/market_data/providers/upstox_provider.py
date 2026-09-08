import logging
import time
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..core.interfaces import ProviderInterface
from ..core.models import NormalizedMarketData, CapabilityMatrix, ProviderStatus, DataProvenance
from validation import MarketData

logger = logging.getLogger(__name__)

# [VERSION: UPSTOX_SESSION_POOL_v1.0]
# Module-level Session with connection pooling and automatic retry on transient errors.
# Replaces per-call requests.get() to reuse TCP connections, saving ~50ms per call
# and respecting Upstox connection limits. Retries 3 times on 502/503/504 only.
_upstox_retry = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=[502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_upstox_adapter = HTTPAdapter(
    pool_connections=50,
    pool_maxsize=100,
    max_retries=_upstox_retry,
)
_upstox_session = requests.Session()
_upstox_session.mount("https://", _upstox_adapter)
_upstox_session.mount("http://", _upstox_adapter)

# Module-level in-memory RAM cache for resolved Upstox instrument keys
_inst_key_cache = {}

# [PHASE1_DIAG] Module-level cached resolver references.
# Resolved once on first call; avoids repeated `from ... import ...` in the hot resolution path.
_cached_resolver = None
_cached_mapper_func = None

def _get_cached_resolver():
    """Returns the SymbolResolutionService instance, cached at module level."""
    global _cached_resolver
    if _cached_resolver is None:
        try:
            from symbol_resolution_engine import get_symbol_resolver
            _cached_resolver = get_symbol_resolver()
        except Exception as e:
            logger.warning(f"⚠️ [Upstox] SymbolResolutionEngine import failed — symbol resolution disabled: {e}")
            _cached_resolver = False  # sentinel: import failed
    return _cached_resolver if _cached_resolver is not False else None

def _get_cached_mapper_func():
    """Returns the upstox_instrument_mapper.get_upstox_instrument_key function, cached at module level."""
    global _cached_mapper_func
    if _cached_mapper_func is None:
        try:
            from market_data.providers.upstox_instrument_mapper import get_upstox_instrument_key
            _cached_mapper_func = get_upstox_instrument_key
        except Exception as e:
            logger.warning(f"⚠️ [Upstox] InstrumentMapper import failed — mapper disabled, will use resolver/fallback: {e}")
            _cached_mapper_func = False  # sentinel: import failed
    return _cached_mapper_func if _cached_mapper_func is not False else None

class UpstoxProvider(ProviderInterface):
    """
    Official Upstox API v2 Integration.
    Acts as the Primary Historical Data Provider to bypass WAF bans.
    """
    def __init__(self, auth_service=None):
        self.auth_service = auth_service
        self._capabilities = CapabilityMatrix(
            supports_1m=True,
            supports_5m=True,
            supports_15m=True,
            supports_1h=False,  # Resampled from 30m candles
            supports_1d=True,
            supports_corporate_actions=False,
            supports_oi=True
        )
        self._health_score = 100.0
        self._status = ProviderStatus.HEALTHY
        
    @property
    def provider_name(self) -> str:
        return "Upstox"
        
    @property
    def capabilities(self) -> CapabilityMatrix:
        return self._capabilities
        
    def get_health_score(self) -> float:
        return self._health_score
        
    def get_status(self) -> ProviderStatus:
        return self._status
        
    # [VERSION: UPSTOX_DATE_NORM_v1.0]
    # Normalise the DataFrame column naming so that Upstox output matches Fyers/Yahoo convention:
    #   - Daily intervals (1d, day) → 'Date' column (date-only, no time component)
    #   - Intraday intervals        → 'Datetime' column (full timestamp)
    # This is required because 14 downstream consumers branch on 'if "Date" in df.columns'
    # to detect daily candles and derive delta fetch timestamps.
    # Without this, price_cache.py, eod_scanner.py, request_planner.py etc. would treat
    # daily Upstox candles as intraday — causing wrong indicator windows and stale deltas.
    def _build_ohlcv_df(self, candles: list, timeframe: str) -> 'pd.DataFrame':
        """Build a normalized OHLCV DataFrame from Upstox candle list.
        Daily intervals emit a 'Date' column; intraday emits 'Datetime'.
        """
        df = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        
        # [VERSION: UPSTOX_TZ_FIX_v1.0] Convert UTC to IST before normalizing dates.
        # Upstox API v3 emits daily timestamps in UTC (e.g. 18:30 UTC = 00:00 IST of trading day).
        # Converting UTC to IST ensures August 27th 00:00 IST is not misclassified as August 26th stale data.
        try:
            if df["Datetime"].dt.tz is not None:
                df["Datetime"] = df["Datetime"].dt.tz_convert("Asia/Kolkata")
            else:
                df["Datetime"] = df["Datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        except Exception:
            pass

        is_daily = timeframe.lower() in ("1d", "day", "1day", "d")
        if is_daily:
            # Rename to 'Date' (date-only midnight Timestamp in IST)
            df["Date"] = df["Datetime"].dt.normalize().dt.tz_localize(None)
            df = df.drop(columns=["Datetime"])
            df = df.sort_values("Date").reset_index(drop=True)
        else:
            df = df.set_index("Datetime").sort_index()
        return df

    def _map_timeframe(self, timeframe: str) -> Tuple[str, str]:
        """Maps timeframe string to official Upstox V3 (unit, interval) path parameters per API spec."""
        tf_clean = str(timeframe).lower().strip()
        mapping = {
            "1m": ("minutes", "1"),
            "1min": ("minutes", "1"),
            "1minute": ("minutes", "1"),
            "2m": ("minutes", "2"),
            "3m": ("minutes", "3"),
            "5m": ("minutes", "5"),
            "5min": ("minutes", "5"),
            "5minute": ("minutes", "5"),
            "10m": ("minutes", "10"),
            "15m": ("minutes", "15"),
            "15min": ("minutes", "15"),
            "15minute": ("minutes", "15"),
            "30m": ("minutes", "30"),
            "30min": ("minutes", "30"),
            "30minute": ("minutes", "30"),
            "60m": ("hours", "1"),
            "60min": ("hours", "1"),
            "60minute": ("hours", "1"),
            "1h": ("hours", "1"),
            "1hour": ("hours", "1"),
            "1d": ("days", "1"),
            "d": ("days", "1"),
            "daily": ("days", "1"),
            "day": ("days", "1"),
            "1w": ("weeks", "1"),
            "w": ("weeks", "1"),
            "weekly": ("weeks", "1"),
            "week": ("weeks", "1"),
            "1mo": ("months", "1"),
            "mo": ("months", "1"),
            "monthly": ("months", "1"),
            "month": ("months", "1"),
        }
        return mapping.get(tf_clean, ("days", "1"))
        
    # Upstox index instrument key map — NSE_INDEX / BSE_INDEX segment, NOT NSE_EQ
    # Source: Upstox historical-candle API instrument key registry (verified format)
    _INDEX_KEY_MAP = {
        # ── Broad Market Indices ──────────────────────────────────────────────────
        "^NSEI":        "NSE_INDEX|Nifty 50",
        "NIFTY":        "NSE_INDEX|Nifty 50",
        "NIFTY50":      "NSE_INDEX|Nifty 50",
        "NIFTY-50":     "NSE_INDEX|Nifty 50",
        "NIFTY 50":     "NSE_INDEX|Nifty 50",
        "NSEI":         "NSE_INDEX|Nifty 50",

        "^NSEBANK":     "NSE_INDEX|Nifty Bank",
        "BANKNIFTY":    "NSE_INDEX|Nifty Bank",
        "NIFTYBANK":    "NSE_INDEX|Nifty Bank",
        "NSEBANK":      "NSE_INDEX|Nifty Bank",

        "^BSESN":       "BSE_INDEX|SENSEX",
        "SENSEX":       "BSE_INDEX|SENSEX",
        "BSE:SENSEX":   "BSE_INDEX|SENSEX",

        "^INDIAVIX":    "NSE_INDEX|India VIX",
        "INDIAVIX":     "NSE_INDEX|India VIX",
        "INDIA VIX":    "NSE_INDEX|India VIX",
        "VIX":          "NSE_INDEX|India VIX",

        # ── Midcap / Smallcap / Broad ─────────────────────────────────────────────
        "^NSMIDCP":         "NSE_INDEX|Nifty Midcap 100",
        "^NSMIDCP50":       "NSE_INDEX|Nifty Midcap 50",
        "^CNXSMALLCAP":     "NSE_INDEX|Nifty Smallcap 100",
        "^CNXSMALLCAP50":   "NSE_INDEX|Nifty Smallcap 50",
        "^CNXMICROCAP250":  "NSE_INDEX|Nifty Microcap 250",
        "^NIFTY200":        "NSE_INDEX|Nifty 200",
        "^NIFTY500":        "NSE_INDEX|Nifty 500",
        "^NIFTY100":        "NSE_INDEX|Nifty 100",
        "^NIFTYNEXT50":     "NSE_INDEX|Nifty Next 50",

        # ── Sectoral Indices ──────────────────────────────────────────────────────
        "^CNXIT":           "NSE_INDEX|Nifty IT",
        "NIFTYIT":          "NSE_INDEX|Nifty IT",
        "^CNXAUTO":         "NSE_INDEX|Nifty Auto",
        "NIFTYAUTO":        "NSE_INDEX|Nifty Auto",
        "^CNXFMCG":         "NSE_INDEX|Nifty FMCG",
        "NIFTYFMCG":        "NSE_INDEX|Nifty FMCG",
        "^CNXPHARMA":       "NSE_INDEX|Nifty Pharma",
        "NIFTYPHARMA":      "NSE_INDEX|Nifty Pharma",
        "^CNXMETAL":        "NSE_INDEX|Nifty Metal",
        "NIFTYMETAL":       "NSE_INDEX|Nifty Metal",
        "^CNXREALTY":       "NSE_INDEX|Nifty Realty",
        "NIFTYREALTY":      "NSE_INDEX|Nifty Realty",
        "^CNXENERGY":       "NSE_INDEX|Nifty Energy",
        "NIFTYENERGY":      "NSE_INDEX|Nifty Energy",
        "^CNXINFRA":        "NSE_INDEX|Nifty Infra",
        "NIFTYINFRA":       "NSE_INDEX|Nifty Infra",
        "^CNXPSUBANK":      "NSE_INDEX|Nifty PSU Bank",
        "NIFTYPSUBANK":     "NSE_INDEX|Nifty PSU Bank",
        "^CNXPSU":          "NSE_INDEX|Nifty PSE",
        "^CNXFIN":          "NSE_INDEX|Nifty Fin Service",
        "^CNXFINANCE":      "NSE_INDEX|Nifty Fin Service",
        "NIFTYFINANCE":     "NSE_INDEX|Nifty Fin Service",
        "^CNXCONSUMPTION":  "NSE_INDEX|Nifty India Consumption",
        "^CNXCMDT":         "NSE_INDEX|Nifty Commodities",
        "^CNXCOMMODITIES":  "NSE_INDEX|Nifty Commodities",
        "NIFTYCOMMODITIES": "NSE_INDEX|Nifty Commodities",
        "^NIFTYOILGAS":     "NSE_INDEX|Nifty Oil & Gas",
        "^NIFTYDEFENCE":    "NSE_INDEX|Nifty India Defence",
        "^CNXMNC":          "NSE_INDEX|Nifty MNC",
        "^CNXSERVICE":      "NSE_INDEX|Nifty Services Sector",
        "^CNXMEDIA":        "NSE_INDEX|Nifty Media",
        "NIFTYMEDIA":       "NSE_INDEX|Nifty Media",
        "NIFTY MEDIA":      "NSE_INDEX|Nifty Media",
        "^NIFTYHEALTHCARE": "NSE_INDEX|Nifty Healthcare Index",
    }

    def _get_instrument_key(self, symbol: str) -> str:
        """Maps symbol to official Upstox instrument key using O(1) RAM cache & instrument mapper first."""
        clean = str(symbol).strip().upper()
        if clean in _inst_key_cache:
            return _inst_key_cache[clean]

        for sfx in (".NS", ".BO", ".BSE"):
            if clean.endswith(sfx):
                clean = clean[:-len(sfx)]
                break

        # 1. Fast-path for indices: Always use exact case-sensitive Upstox index keys first
        if clean in self._INDEX_KEY_MAP:
            key = self._INDEX_KEY_MAP[clean]
            _inst_key_cache[clean] = key
            return key
        raw_clean = str(symbol).strip()
        if raw_clean in self._INDEX_KEY_MAP:
            key = self._INDEX_KEY_MAP[raw_clean]
            _inst_key_cache[clean] = key
            return key

        # 2. Upstox Instrument Mapper O(1) RAM dict lookup (41,221 pre-loaded keys in RAM)
        mapper_func = _get_cached_mapper_func()
        if mapper_func:
            try:
                mapped = mapper_func(symbol)
                if mapped and mapped != f"NSE_EQ|{clean.lstrip('^')}":
                    _inst_key_cache[clean] = mapped
                    return mapped
            except Exception as e:
                logger.warning(f"⚠️ [Upstox] Instrument mapper failed for '{symbol}': {e} — falling back to resolver")

        # 3. Dynamic symbol resolution service fallback (if mapper didn't match)
        resolver = _get_cached_resolver()
        if resolver:
            try:
                resolved = resolver.resolve(symbol, provider="upstox")
                if resolved and resolved.is_valid and resolved.mapped_symbol:
                    _inst_key_cache[clean] = resolved.mapped_symbol
                    return resolved.mapped_symbol
            except Exception as e:
                logger.warning(f"⚠️ [Upstox] Symbol resolver failed for '{symbol}': {e} — using NSE_EQ fallback")

        clean_bare = clean.lstrip("^")
        fallback_key = f"NSE_EQ|{clean_bare}"
        _inst_key_cache[clean] = fallback_key
        return fallback_key

    def fetch_ohlcv(self, symbol: str, timeframe: str, range_from: datetime, range_to: datetime) -> NormalizedMarketData:
        import config
        import urllib.parse
        from datetime import timedelta
        
        # [VERSION: NON_EQUITY_BLOCKLIST_v2.0] Filter only known InvITs/REITs (never ASM/GSM equities)
        non_equity_blocklist = getattr(config, "NON_EQUITY_BLOCKLIST", {"VERTIS", "HIGHWAYS", "POWERINVIT", "IRBINVIT", "INDIGRID", "EMBASSY", "MINDSPACE", "BROOKFIELD", "NEXUS"})
        if symbol and str(symbol).strip().upper() in non_equity_blocklist:
            start_time = datetime.now()
            prov = DataProvenance(self.provider_name, start_time, 0.0, 0)
            return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), prov, error="Blacklisted non-equity trust")


        token = getattr(config, "UPSTOX_ACCESS_TOKEN", None)
        raw_key = self._get_instrument_key(symbol)
        instrument_key = urllib.parse.quote(raw_key)
        unit, interval = self._map_timeframe(timeframe)
        
        # Upstox V3 API historical candles for indices: defer intraday index candles to Fyers fallback
        if (raw_key.startswith("NSE_INDEX|") or raw_key.startswith("BSE_INDEX|")) and unit in ("minutes", "hours"):
            logger.debug(f"Upstox API does not support intraday candles for index {symbol} ({raw_key}); deferring to fallback.")
            start_time = datetime.now()
            prov = DataProvenance(self.provider_name, start_time, 0.0, 0)
            return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), prov, error="Intraday index candles not supported by Upstox")
        
        if not token:
            self._status = ProviderStatus.AUTH_FAILED
            self._health_score -= 10
            raise PermissionError("UPSTOX_ACCESS_TOKEN is completely missing from config.")
            
        adjusted_range_to = range_to
        if range_to and hasattr(range_to, "weekday"):
            if range_to.weekday() == 5:
                adjusted_range_to = range_to - timedelta(days=1)
            elif range_to.weekday() == 6:
                adjusted_range_to = range_to - timedelta(days=2)
                
        # Upstox V3 API Historical Candle Endpoint
        url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{adjusted_range_to.strftime('%Y-%m-%d')}/{range_from.strftime('%Y-%m-%d')}"
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        start_time = datetime.now()
        
        try:
            import random
            backoff = 1.0
            response = None
            for attempt in range(4):
                response = _upstox_session.get(url, headers=headers, timeout=10)
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status_code == 429:
                    self._health_score = max(0, self._health_score - 2)
                    sleep_time = backoff + random.uniform(0.5, 2.0)
                    logger.warning(
                        f"🚦 [UPSTOX] HTTP 429 Rate Limit for {symbol} "
                        f"(attempt {attempt+1}/4) — sleeping {sleep_time:.2f}s before retry. "
                        f"health_score={self._health_score:.0f}"
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue
                break

            if response is not None and response.status_code == 429:
                logger.error(
                    f"❌ [UPSTOX] HTTP 429 Rate Limit EXHAUSTED for {symbol} after 4 attempts — "
                    f"giving up. Consider increasing stagger or reducing workers."
                )
                return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), DataProvenance(self.provider_name, start_time, latency, 0), error="429 Rate Limit Exhausted")
                
            if response.status_code == 401:
                self._status = ProviderStatus.AUTH_FAILED
                self._health_score -= 20
                logger.error(f"❌ [UPSTOX] HTTP 401 Auth EXPIRED for {symbol} — token is invalid or expired. Re-login required.")
                return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), DataProvenance(self.provider_name, start_time, latency, 0), error="401 Auth Expired")
                
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "success":
                api_error = data.get("errors") or data.get("message") or data.get("error") or "unknown"
                logger.error(f"❌ [UPSTOX] API returned non-success for {symbol}: status={data.get('status')!r} | error={api_error!r}")
                return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), DataProvenance(self.provider_name, start_time, latency, 0), error=f"API Failure: {api_error}")
                
            candles = data.get("data", {}).get("candles", [])
            if not candles:
                return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), DataProvenance(self.provider_name, start_time, latency, 100), error=None)
                
            df = self._build_ohlcv_df(candles, timeframe)
            
            prov = DataProvenance(self.provider_name, start_time, latency, 100.0)
            
            return NormalizedMarketData(
                symbol=symbol,
                timeframe=timeframe,
                dataframe=df,
                provenance=prov,
                is_complete_candle=True,
                error=None
            )
            
        except requests.HTTPError as e:
            self._health_score = max(0, self._health_score - 2)
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 400:
                # [RULE 67 CHANGE-RATIONALE]: Log HTTP 400 as warning instead of error since invalid/discontinued ticker requests should not trigger system error alerts
                logger.warning(f"⚠️ [UPSTOX HTTP 400] Stale or invalid instrument key for {symbol} ({e}). Invalidating cached key...")
                try:
                    from market_data.providers.upstox_instrument_mapper import mapper
                    clean_sym = str(symbol).strip().upper()
                    for sfx in (".NS", ".BO", ".BSE"):
                        if clean_sym.endswith(sfx):
                            clean_sym = clean_sym[:-len(sfx)]
                    mapper._symbol_map.pop(clean_sym, None)
                    mapper.trigger_background_download(force=True)
                except Exception as _inv_err:
                    logger.debug(f"Failed to invalidate stale key for {symbol}: {_inv_err}")
            else:
                logger.error(f"Upstox fetch HTTP error for {symbol}: {e}")
            latency = (datetime.now() - start_time).total_seconds() * 1000
            prov = DataProvenance(self.provider_name, start_time, latency, 0.0)
            return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), prov, error=f"HTTP {status_code}")
        except Exception as e:
            self._health_score = max(0, self._health_score - 2)
            logger.exception(f"Upstox fetch error for {symbol}: {e}")
            latency = (datetime.now() - start_time).total_seconds() * 1000
            prov = DataProvenance(self.provider_name, start_time, latency, 0.0)
            return NormalizedMarketData(symbol, timeframe, pd.DataFrame(), prov, error=str(e))

    def fetch_batch_ohlcv(self, symbols: List[str], timeframe: str, range_from: datetime, range_to: datetime) -> Dict[str, NormalizedMarketData]:
        """Fetches batch normalized market data concurrently for multiple symbols."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        if not symbols:
            return results
        max_workers = min(15, len(symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {
                executor.submit(self.fetch_ohlcv, sym, timeframe, range_from, range_to): sym
                for sym in symbols
            }
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    results[sym] = future.result()
                except Exception as e:
                    logger.error(f"Error fetching batch symbol {sym}: {e}")
        return results

    def get_quote(self, symbol: str) -> dict:
        """Fetches live market quote for a single symbol from Upstox v2 API."""
        quotes = self.get_quotes([symbol])
        return quotes.get(symbol, {})

    def fetch_live_quotes_batch(self, symbols: List[str]) -> Dict[str, dict]:
        """Fetches live market quotes for multiple symbols (alias for UnifiedFetcher callers)."""
        return self.get_quotes(symbols)

    def get_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        """Fetches live market quotes for multiple symbols from Upstox v2 API."""
        if not symbols:
            return {}

        import config
        import urllib.parse
        token = getattr(config, "UPSTOX_ACCESS_TOKEN", None)
        if not token:
            logger.error("❌ [UPSTOX] UPSTOX_ACCESS_TOKEN missing — cannot fetch live quotes. Returning empty.")
            return {}

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        }

        results = {}
        chunk_size = 200
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]

            # [PHASE1_DIAG] Stage A: Instrument Key Resolution
            _t_res = time.perf_counter()
            formatted_keys_list = [urllib.parse.quote(self._get_instrument_key(s)) for s in chunk if s and self._get_instrument_key(s)]
            resolution_ms = (time.perf_counter() - _t_res) * 1000

            if not formatted_keys_list:
                continue

            formatted_keys = ",".join(formatted_keys_list)
            url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={formatted_keys}"

            try:
                # [PHASE1_DIAG] Stage B: HTTP Network Round-Trip
                _t_http = time.perf_counter()
                res = _upstox_session.get(url, headers=headers, timeout=10)
                http_ms = (time.perf_counter() - _t_http) * 1000
                payload_kb = len(res.content) / 1024 if res.content else 0

                # [PHASE1_DIAG] Stage C: JSON Parsing
                _t_json = time.perf_counter()
                data = {}
                if res.status_code == 200 and res.content:
                    try:
                        import ujson
                        data = ujson.loads(res.content)
                    except Exception:
                        # ujson not available or failed — fall through to stdlib json (expected)
                        try:
                            import json
                            data = json.loads(res.content)
                        except Exception as json_err:
                            logger.warning(f"⚠️ [Upstox] JSON parse failed for live quote response: {json_err} — falling back to res.json()")
                            data = res.json()
                json_ms = (time.perf_counter() - _t_json) * 1000

                # [PHASE1_DIAG] Stage D: Quote Merge
                _t_merge = time.perf_counter()
                if res.status_code == 200:
                    quote_dict = data.get("data", {})
                    for key, quote in quote_dict.items():
                        results[key] = quote
                        results[key.replace(":", "|")] = quote
                        clean_sym = key.split(":")[-1].split("|")[-1]
                        results[clean_sym] = quote
                        if isinstance(quote, dict) and quote.get("symbol"):
                            results[str(quote["symbol"]).upper()] = quote
                    
                    for orig_sym in chunk:
                        inst_key = self._get_instrument_key(orig_sym)
                        matched_quote = quote_dict.get(inst_key) or quote_dict.get(inst_key.replace("|", ":"))
                        if matched_quote:
                            results[orig_sym] = matched_quote
                elif res.status_code == 400 and len(chunk) > 1:
                    # [RULE 67 CHANGE-RATIONALE] Upstox returns UDAPI1087 (400) for the entire batch if a single key is unlisted/invalid.
                    # Recover gracefully by requesting keys individually for this chunk so valid symbols succeed.
                    logger.debug(f"Upstox batch 400 on chunk of {len(chunk)} — falling back to individual symbol resolution")
                    for single_sym in chunk:
                        s_key = self._get_instrument_key(single_sym)
                        if not s_key:
                            continue
                        try:
                            s_url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={urllib.parse.quote(s_key)}"
                            s_res = _upstox_session.get(s_url, headers=headers, timeout=4)
                            if s_res.status_code == 200 and s_res.content:
                                try:
                                    import json
                                    s_data = json.loads(s_res.content)
                                    s_dict = s_data.get("data", {})
                                    for k, q in s_dict.items():
                                        results[k] = q
                                        results[single_sym] = q
                                except Exception:
                                    pass
                        except Exception:
                            pass
                else:
                    logger.debug(f"Live quote fetch returned status {res.status_code}: {res.text[:150] if res.text else ''}")
                merge_ms = (time.perf_counter() - _t_merge) * 1000

                # [PHASE1_DIAG] Redundant-call detection: warn if calls exceed requested symbols
                if len(formatted_keys_list) > len(chunk):
                    logger.warning(
                        f"[PHASE1_DIAG] Redundant resolution detected: "
                        f"keys={len(formatted_keys_list)} > requested={len(chunk)}"
                    )

                logger.debug(
                    f"\U0001f4ca [LIVE_QUOTE_PIPELINE] chunk={len(chunk)} | "
                    f"status={res.status_code} | payload={payload_kb:.1f}KB\n"
                    f"   \u251c\u2500\u2500 Resolution stage : {resolution_ms:.1f}ms (calls={len(chunk)})\n"
                    f"   \u251c\u2500\u2500 HTTP Network     : {http_ms:.1f}ms\n"
                    f"   \u251c\u2500\u2500 JSON Parsing     : {json_ms:.1f}ms\n"
                    f"   \u2514\u2500\u2500 Quote Merge      : {merge_ms:.1f}ms\n"
                    f"   Total pipeline   : {resolution_ms + http_ms + json_ms + merge_ms:.1f}ms"
                )

            except requests.RequestException as e:
                logger.error(f"Network error fetching live quote batch: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error fetching live quote batch: {e}", exc_info=True)

        return results

    def get_ohlcv(self, symbol: str, interval: str = "1d", period: str = "1y", retries: int = 3, range_from = None, range_to = None) -> MarketData:
        """
        Adapter method for legacy DataFetcher callers.
        Converts NormalizedMarketData to MarketData validation format.
        """
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        now = datetime.now(IST)
        
        if range_from and range_to:
            r_from = datetime.strptime(range_from, "%Y-%m-%d") if isinstance(range_from, str) else range_from
            r_to = datetime.strptime(range_to, "%Y-%m-%d") if isinstance(range_to, str) else range_to
        else:
            days = 365
            if period.endswith("y"):
                try: days = int(period[:-1]) * 365
                except Exception as e: logger.warning(f"⚠️ [Upstox] Failed to parse period '{period}' as years: {e} — defaulting to 365d"); days = 365
            elif period.endswith("mo"):
                try: days = int(period[:-2]) * 30
                except Exception as e: logger.warning(f"⚠️ [Upstox] Failed to parse period '{period}' as months: {e} — defaulting to 30d"); days = 30
            elif period.endswith("d"):
                try: days = int(period[:-1])
                except Exception as e: logger.warning(f"⚠️ [Upstox] Failed to parse period '{period}' as days: {e} — defaulting to 10d"); days = 10
            r_from = now - timedelta(days=days)
            r_to = now
            
        norm_data = self.fetch_ohlcv(symbol, timeframe=interval, range_from=r_from, range_to=r_to)
        
        from validation import ValidationEngine, MarketData, ValidationContext, registry as val_registry, DatasetType
            
        df = norm_data.dataframe
        if df is None or df.empty:
            return MarketData(dataframe=pd.DataFrame(), source="Upstox", quality_report=None, stale=False, used_fallback=False, error=norm_data.error or "Empty DataFrame")
            
        try:
            pipeline = val_registry.get_pipeline(DatasetType.PRICE)
            engine = ValidationEngine(pipeline.validator, pipeline.score_calculator)
            r_from_str = r_from.strftime("%Y-%m-%d") if hasattr(r_from, "strftime") else str(r_from)
            r_to_str = r_to.strftime("%Y-%m-%d") if hasattr(r_to, "strftime") else str(r_to)
            ctx = ValidationContext(provider="Upstox", period=period, interval=interval, range_from=r_from_str, range_to=r_to_str, fetch_mode="DELTA" if range_from else "FULL")
            report = engine.validate(df, ctx)
            
            if not report.is_valid and not range_from:
                return MarketData(dataframe=None, source="Upstox", quality_report=report, stale=False, used_fallback=False, error="Quality Check Failed")
            return MarketData(dataframe=df, source="Upstox", quality_report=report, stale=False, used_fallback=False, error=None if report.is_valid else "Quality Warning")

        except Exception as val_err:
            logger.warning(f"⚠️ [Upstox] ValidationEngine exception for {symbol}: {val_err} — returning raw dataframe without quality report", exc_info=True)
            return MarketData(dataframe=df, source="Upstox", quality_report=None, stale=False, used_fallback=False, error=None)

    def get_batch_ohlcv(self, symbols: List[str], interval: str = "1d", period: str = "1y", retries: int = 3, range_from = None, range_to = None, caller: str = None) -> Dict:
        """
        Concurrent batch fetch using ThreadPoolExecutor.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        if not symbols:
            return results

        prefix = f"[{caller}] " if caller else ""
        # Upstox API allows 10 req/s. 5 workers with micro-staggering (100ms) maximizes fetch throughput safely.
        max_workers = min(5, len(symbols))
        logger.info(f"{prefix}📥 Upstox: batch fetching {len(symbols)} symbols ({interval}, {period}) concurrently (workers={max_workers})...")

        import time
        t_batch_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {}
            for sym in symbols:
                future_to_sym[executor.submit(self.get_ohlcv, sym, interval, period, retries, range_from, range_to)] = sym
                time.sleep(0.1)  # 100ms micro-stagger for smooth API request pacing

            completed = 0
            errors = 0
            last_progress_log_t = time.monotonic()
            last_progress_log_n = 0
            total = len(symbols)

            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    result = future.result()
                    results[sym] = result
                    df = getattr(result, 'dataframe', None)
                    if df is None or (hasattr(df, 'empty') and df.empty):
                        errors += 1
                except Exception as e:
                    logger.error(f"Upstox batch fetch exception for {sym}: {e}")
                    errors += 1

                completed += 1
                now_t = time.monotonic()
                elapsed = now_t - t_batch_start
                # Log every 10 symbols OR every 30s, whichever comes first
                symbols_since_log = completed - last_progress_log_n
                secs_since_log = now_t - last_progress_log_t
                if symbols_since_log >= 10 or secs_since_log >= 30 or completed == total:
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta_s = int((total - completed) / rate) if rate > 0 else 0
                    logger.info(
                        f"{prefix}📡 Upstox batch progress: {completed}/{total} done | "
                        f"{errors} errors | Elapsed: {elapsed:.0f}s | Rate: {rate:.1f} sym/s | "
                        f"ETA: ~{eta_s}s"
                    )
                    last_progress_log_t = now_t
                    last_progress_log_n = completed

        fetched_count = sum(1 for v in results.values() if v and getattr(v, 'dataframe', None) is not None and not getattr(v.dataframe, 'empty', True))
        valid_count = sum(1 for v in results.values() if v and getattr(v, 'dataframe', None) is not None and getattr(getattr(v, 'quality_report', None), 'is_valid', False))
        logger.info(f"{prefix}📊 Upstox batch fetch complete: {fetched_count}/{len(symbols)} fetched | {valid_count}/{len(symbols)} validated OK")
        return results
