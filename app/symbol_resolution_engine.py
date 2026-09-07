"""
app/symbol_resolution_engine.py
=================================
Centralized, institutional-grade Symbol Resolution Engine for multi-broker operations.
Provides microsecond O(1) in-memory resolution, atomic double-buffered index swapping,
per-symbol single-flight concurrency control, strategy-pattern provider adapters,
and exponential backoff negative caching.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# ── ENUMS & VALUE OBJECTS ───────────────────────────────────────────────────────

class MappingSource(IntEnum):
    MASTER = 100
    MANUAL = 100
    LEARNED = 95
    PROBED = 80


class MappingStatus(str, Enum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    INVALID = "INVALID"


class ResolutionLevel(str, Enum):
    MEMORY = "MEMORY"
    DB_MAPPING = "DB_MAPPING"
    REGISTRY = "REGISTRY"
    BROKER_MASTER = "BROKER_MASTER"
    SMART_PROBE = "SMART_PROBE"
    NEGATIVE_CACHE = "NEGATIVE_CACHE"


@dataclass(frozen=True)
class InstrumentMetadata:
    instrument_id: str
    symbol: str
    company_name: str = ""
    primary_exchange: str = "NSE"
    series: str = "EQ"
    bse_scrip_code: Optional[str] = None


@dataclass(frozen=True)
class ResolvedInstrument:
    instrument_id: str
    symbol: str
    provider: str
    mapped_symbol: str
    exchange: str = "NSE"
    series: str = "EQ"
    confidence_score: int = 100
    source: str = "MASTER"
    is_valid: bool = True
    error_message: Optional[str] = None


# ── DOUBLE-BUFFERED IN-MEMORY INDEX STORE ─────────────────────────────────────

class MemoryIndexStore:
    """
    Immutable, double-buffered in-memory index container providing O(1) microsecond resolution.
    Updated via atomic reference swaps (self._active_indexes = new_store).
    """
    def __init__(self):
        self.idx_by_symbol: Dict[str, str] = {}                         # 'RELIANCE' -> instrument_id
        self.idx_by_id: Dict[str, InstrumentMetadata] = {}              # instrument_id -> InstrumentMetadata
        self.idx_provider_mapping: Dict[Tuple[str, str], ResolvedInstrument] = {}  # ('fyers', 'RELIANCE') -> ResolvedInstrument
        self.idx_provider_by_id: Dict[Tuple[str, str], str] = {}         # ('fyers', instrument_id) -> mapped_symbol
        self.negative_cache: Dict[Tuple[str, str], Tuple[datetime, int]] = {} # ('fyers', 'FAKE') -> (expire_dt, fail_count)


# ── PROVIDER ADAPTER STRATEGY PATTERN ──────────────────────────────────────────

class BaseProviderAdapter(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def lookup_master(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        pass

    @abstractmethod
    def probe_candidates(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        pass


class FyersAdapter(BaseProviderAdapter):
    @property
    def provider_name(self) -> str:
        return "fyers"

    def lookup_master(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        sym_raw = symbol.upper()
        force_bse = sym_raw.endswith(".BO") or sym_raw.startswith("BSE:")
        sym = sym_raw.replace(".NS", "").replace(".BO", "").replace("NSE:", "").replace("BSE:", "")
        
        # 1. Authoritative Instrument Registry Lookup
        try:
            from instrument_registry import get_instrument_registry
            reg_rec = get_instrument_registry().lookup(symbol)
            if reg_rec and reg_rec.fyers_symbol:
                inst_id = f"{reg_rec.asset_type}:{reg_rec.exchange}:{reg_rec.canonical_symbol}"
                return ResolvedInstrument(inst_id, sym, "fyers", reg_rec.fyers_symbol, reg_rec.exchange, reg_rec.asset_type, 100, "REGISTRY")
        except Exception as reg_err:
            logger.debug(f"Registry lookup error for {symbol}: {reg_err}")

        # 2. Authoritative Fyers Master Symbol Mapper (Includes -SM, -ST, -BE, -EQ)
        try:
            from data_providers.fyers_symbol_mapper import fyers_mapper
            mapped = fyers_mapper.get_fyers_symbol(sym)
            if mapped:
                inst_id = metadata.instrument_id if metadata else f"EQ:{sym}"
                exch = mapped.split(":")[0] if ":" in mapped else ("BSE" if force_bse else "NSE")
                srs = mapped.split("-")[-1] if "-" in mapped else "EQ"
                return ResolvedInstrument(inst_id, sym, "fyers", mapped, exch, srs, 100, "MASTER")
        except Exception as mapper_err:
            logger.debug(f"Fyers symbol mapper lookup error for {symbol}: {mapper_err}")

        # Standard equity fallback for clean symbols
        if not sym.startswith("UNKNOWN") and not str(symbol).startswith("^"):
            prefix = "BSE:" if force_bse else "NSE:"
            inst_id = metadata.instrument_id if metadata else f"EQ:{sym}"
            return ResolvedInstrument(inst_id, sym, "fyers", f"{prefix}{sym}-EQ", prefix.rstrip(":"), "EQ", 90, "MASTER")

        return None

    def probe_candidates(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        sym = symbol.upper()
        
        # Asset Type Guard: Index symbols MUST NEVER probe equity candidates!
        try:
            from instrument_registry import get_instrument_registry
            if get_instrument_registry().is_index(symbol):
                logger.debug(f"🛑 [FyersProbe] Skipping equity candidate probing for index symbol '{symbol}'")
                return None
        except Exception:
            pass

        # Build candidate probe list in priority order
        candidates = []
        target_series = metadata.series if metadata else "EQ"
        
        # Guided Probing: put target series first if specified
        if target_series and target_series != "EQ":
            candidates.append(f"NSE:{sym}-{target_series}")
            candidates.append(f"BSE:{sym}-{target_series}")

        candidates.extend([
            f"NSE:{sym}-EQ",
            f"BSE:{sym}-EQ",
            f"NSE:{sym}-BE",
            f"NSE:{sym}-BZ",
            f"NSE:{sym}-SM",
            f"NSE:{sym}-ST",
            f"BSE:{sym}-B",
            f"BSE:{sym}-Z",
            f"BSE:{sym}-T",
            f"BSE:{sym}-P",
            f"BSE:{sym}-M",
            f"BSE:{sym}-X",
            f"BSE:{sym}-XC",
            f"BSE:{sym}-XD",
            f"BSE:{sym}-A",
        ])

        if metadata and metadata.bse_scrip_code:
            candidates.append(f"BSE:{metadata.bse_scrip_code}-EQ")
            candidates.append(f"BSE:{metadata.bse_scrip_code}-B")

        # Dynamic BSE Scrip Code fallback lookup for surveillance/BE/BZ tickers like GVKPIL
        try:
            from bse_mapping_utils import load_bse_mappings
            bse_map = load_bse_mappings()
            bse_code = bse_map.get(sym) or bse_map.get(f"{sym}.NS") or bse_map.get(f"{sym}.BO")
            if bse_code:
                candidates.append(f"BSE:{bse_code}-EQ")
                candidates.append(f"BSE:{bse_code}-B")
        except Exception:
            pass

        candidates = list(dict.fromkeys(candidates))

        # Test candidate against Fyers historical or quote endpoint
        try:
            from data_providers.fyers_fetcher import FyersFetcher
            fetcher = FyersFetcher()
            for cand in candidates:
                try:
                    df = fetcher.get_ohlcv_single(cand, period="5d", interval="1d")
                    if df is not None and not df.empty:
                        exch = cand.split(":")[0]
                        srs = cand.split("-")[-1] if "-" in cand else "EQ"
                        inst_id = metadata.instrument_id if metadata else f"EQ:{sym}"
                        logger.info(f"🎯 [FyersProbe] Successfully resolved {sym} → {cand}")
                        return ResolvedInstrument(inst_id, sym, "fyers", cand, exch, srs, 80, "PROBED")
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Fyers probing error for {sym}: {e}")
        return None


class UpstoxAdapter(BaseProviderAdapter):
    @property
    def provider_name(self) -> str:
        return "upstox"

    def lookup_master(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        sym = symbol.upper()

        # 1. Authoritative Instrument Registry Lookup
        try:
            from instrument_registry import get_instrument_registry
            reg_rec = get_instrument_registry().lookup(symbol)
            if reg_rec and reg_rec.upstox_instrument_key:
                inst_id = f"{reg_rec.asset_type}:{reg_rec.exchange}:{reg_rec.canonical_symbol}"
                return ResolvedInstrument(inst_id, sym, "upstox", reg_rec.upstox_instrument_key, reg_rec.exchange, reg_rec.asset_type, 100, "REGISTRY")
        except Exception as reg_err:
            logger.debug(f"Registry lookup error for {symbol}: {reg_err}")

        # 2. Upstox ISIN / Instrument Key Mapper Lookup
        try:
            from market_data.providers.upstox_instrument_mapper import mapper
            key = mapper.get_instrument_key(sym, allow_fallback=not sym.startswith("UNKNOWN") and not sym.startswith("^"))
            if key:
                inst_id = metadata.instrument_id if metadata else f"EQ:{sym}"
                exch = "NSE" if "NSE" in key else "BSE"
                return ResolvedInstrument(inst_id, sym, "upstox", key, exch, "EQ", 100, "MASTER")
        except Exception:
            pass

        return None

    def probe_candidates(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        sym = symbol.upper()

        # Asset Type Guard: Index symbols MUST NEVER probe equity candidates!
        try:
            from instrument_registry import get_instrument_registry
            if get_instrument_registry().is_index(symbol):
                logger.debug(f"🛑 [UpstoxProbe] Skipping equity candidate probing for index symbol '{symbol}'")
                return None
        except Exception:
            pass

        candidates = [f"NSE_EQ|{sym}", f"BSE_EQ|{sym}", f"NSE_BE|{sym}", f"NSE_BZ|{sym}"]
        try:
            from bse_mapping_utils import load_bse_mappings
            bse_map = load_bse_mappings()
            bse_code = bse_map.get(sym) or bse_map.get(f"{sym}.NS") or bse_map.get(f"{sym}.BO")
            if bse_code:
                candidates.append(f"BSE_EQ|{bse_code}")
        except Exception:
            pass
        candidates = list(dict.fromkeys(candidates))
        try:
            from market_data.providers.upstox_provider import UpstoxProvider
            up = UpstoxProvider()
            for cand in candidates:
                try:
                    res = up.fetch_ohlcv(cand, timeframe="1d")
                    if res and res.dataframe is not None and not res.dataframe.empty:
                        exch = "NSE" if "NSE" in cand else "BSE"
                        inst_id = metadata.instrument_id if metadata else f"EQ:{sym}"
                        logger.info(f"🎯 [UpstoxProbe] Successfully resolved {sym} → {cand}")
                        return ResolvedInstrument(inst_id, sym, "upstox", cand, exch, "EQ", 80, "PROBED")
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Upstox probing error for {sym}: {e}")
        return None


class YahooAdapter(BaseProviderAdapter):
    @property
    def provider_name(self) -> str:
        return "yahoo"

    def lookup_master(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        sym = symbol.upper()

        # 1. Authoritative Instrument Registry Lookup
        try:
            from instrument_registry import get_instrument_registry
            reg_rec = get_instrument_registry().lookup(symbol)
            if reg_rec and reg_rec.yahoo_symbol:
                inst_id = f"{reg_rec.asset_type}:{reg_rec.exchange}:{reg_rec.canonical_symbol}"
                return ResolvedInstrument(inst_id, sym, "yahoo", reg_rec.yahoo_symbol, reg_rec.exchange, reg_rec.asset_type, 100, "REGISTRY")
        except Exception as reg_err:
            logger.debug(f"Registry lookup error for {symbol}: {reg_err}")

        # Standard Yahoo format `.NS`
        cand = f"{sym}.NS"
        inst_id = metadata.instrument_id if metadata else f"EQ:{sym}"
        return ResolvedInstrument(inst_id, sym, "yahoo", cand, "NSE", "EQ", 95, "MASTER")

    def probe_candidates(self, symbol: str, metadata: Optional[InstrumentMetadata]) -> Optional[ResolvedInstrument]:
        sym = symbol.upper()

        # Asset Type Guard: Index symbols MUST NEVER probe equity candidates!
        try:
            from instrument_registry import get_instrument_registry
            if get_instrument_registry().is_index(symbol):
                logger.debug(f"🛑 [YahooProbe] Skipping equity candidate probing for index symbol '{symbol}'")
                return None
        except Exception:
            pass
        candidates = [f"{sym}.NS", f"{sym}.BO"]
        try:
            from bse_mapping_utils import load_bse_mappings
            bse_map = load_bse_mappings()
            bse_code = bse_map.get(sym) or bse_map.get(f"{sym}.NS") or bse_map.get(f"{sym}.BO")
            if bse_code:
                candidates.append(f"{bse_code}.BO")
        except Exception:
            pass
        candidates = list(dict.fromkeys(candidates))

        for cand in candidates:
            try:
                import yfinance as yf
                t = yf.Ticker(cand)
                hist = t.history(period="5d")
                if hist is not None and not hist.empty:
                    exch = "NSE" if ".NS" in cand else "BSE"
                    inst_id = metadata.instrument_id if metadata else f"EQ:{sym}"
                    logger.info(f"🎯 [YahooProbe] Successfully resolved {sym} → {cand}")
                    return ResolvedInstrument(inst_id, sym, "yahoo", cand, exch, "EQ", 80, "PROBED")
            except Exception:
                continue
        return None


# ── CENTRALIZED SYMBOL RESOLUTION SERVICE SINGLETON ────────────────────────────

class SymbolResolutionService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._active_indexes = MemoryIndexStore()
        self._adapters: Dict[str, BaseProviderAdapter] = {
            "fyers": FyersAdapter(),
            "upstox": UpstoxAdapter(),
            "yahoo": YahooAdapter(),
        }
        self._single_flight_locks: Dict[Tuple[str, str], threading.Lock] = {}
        self._single_flight_mutex = threading.Lock()

        # Telemetry Aggregators
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "memory_hits": 0,
            "master_hits": 0,
            "probe_hits": 0,
            "negative_hits": 0,
            "total_requests": 0,
            "latencies_ms": []
        }
        import concurrent.futures
        self._async_probe_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="AsyncProbeWorker")

        # Initialize Memory Index Store from DB
        self.reload_memory_indexes()

    def _enqueue_async_probe(self, provider: str, symbol: str, adapter: BaseProviderAdapter, metadata: Optional[InstrumentMetadata]):
        """Dispatches non-blocking symbol probing to background worker thread."""
        def _async_probe_job():
            try:
                resolved = adapter.probe_candidates(symbol, metadata)
                if resolved and resolved.is_valid:
                    self._cache_and_persist_mapping(provider, symbol, resolved)
                    logger.info(f"✅ [AsyncProbeJob] Resolved {symbol} on {provider} → {resolved.mapped_symbol}")
            except Exception as e:
                logger.debug(f"Async probe job failed for {symbol}: {e}")
        self._async_probe_executor.submit(_async_probe_job)

    def reload_memory_indexes(self):
        """
        Loads all active mappings and instrument registry from PostgreSQL into 
        a new MemoryIndexStore object, then performs an ATOMIC REFERENCE SWAP.
        """
        logger.info("🔄 [SymbolResolver] Building new MemoryIndexStore from PostgreSQL...")
        t0 = time.perf_counter()
        new_store = MemoryIndexStore()
        
        try:
            from database import load_all_symbol_resolution_data
            data = load_all_symbol_resolution_data()

            for r in data.get("instrument_registry", []):
                meta = InstrumentMetadata(
                    instrument_id=r["instrument_id"],
                    symbol=r["symbol"].upper(),
                    company_name=r.get("company_name", ""),
                    primary_exchange=r.get("primary_exchange", "NSE"),
                    series=r.get("series", "EQ"),
                    bse_scrip_code=r.get("bse_scrip_code")
                )
                new_store.idx_by_symbol[meta.symbol] = meta.instrument_id
                new_store.idx_by_id[meta.instrument_id] = meta

            for r in data.get("symbol_mappings", []):
                if r.get("status") == MappingStatus.ACTIVE.value:
                    res = ResolvedInstrument(
                        instrument_id=r.get("instrument_id") or f"EQ:{r['original_symbol']}",
                        symbol=r["original_symbol"].upper(),
                        provider=r["provider"].lower(),
                        mapped_symbol=r["mapped_symbol"],
                        exchange=r.get("exchange", "NSE"),
                        series=r.get("series", "EQ"),
                        confidence_score=r.get("confidence_score", 100),
                        source=r.get("mapping_source", "MASTER"),
                        is_valid=True
                    )
                    new_store.idx_provider_mapping[(res.provider, res.symbol)] = res
                    if res.instrument_id:
                        new_store.idx_provider_by_id[(res.provider, res.instrument_id)] = res.mapped_symbol

            # ATOMIC REFERENCE SWAP (0 lock contention during runtime!)
            self._active_indexes = new_store
            dt_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"✅ [SymbolResolver] Loaded {len(new_store.idx_provider_mapping)} provider mappings into MemoryIndexStore ({dt_ms:.2f}ms).")
        except Exception as e:
            logger.error(f"❌ [SymbolResolver] Memory index load error: {e}")

    def _get_single_flight_lock(self, key: Tuple[str, str]) -> threading.Lock:
        with self._single_flight_mutex:
            if key not in self._single_flight_locks:
                self._single_flight_locks[key] = threading.Lock()
            return self._single_flight_locks[key]

    def resolve(self, symbol: str, provider: str = "fyers") -> ResolvedInstrument:
        """
        Main entry point for all symbol resolution requests across the system.
        Provides microsecond O(1) memory lookup, single-flight locking, master resolution,
        smart series probing, and negative caching with exponential backoff.
        """
        t0 = time.perf_counter()
        if not symbol or not isinstance(symbol, str):
            return ResolvedInstrument("INVALID", str(symbol), provider, "", is_valid=False, error_message="Empty symbol")

        sym_clean = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        prov = provider.lower()
        key = (prov, symbol.strip().upper())
        store = self._active_indexes

        with self._metrics_lock:
            self._metrics["total_requests"] += 1

        # ── LEVEL 1: In-Memory O(1) Hash Lookup (5–20 microseconds) ───────────────────
        if key in store.idx_provider_mapping:
            res = store.idx_provider_mapping[key]
            latency_ms = (time.perf_counter() - t0) * 1000
            self._record_telemetry("memory_hits", latency_ms)
            return res

        # ── LEVEL 1.1: Exponential Backoff Negative Cache Check ──────────────────────
        # Registered INDEX symbols and canonical registry entries bypass negative cache lockouts
        try:
            from instrument_registry import get_instrument_registry
            reg_check = get_instrument_registry()
            if reg_check.is_index(symbol) or reg_check.lookup(symbol) is not None:
                store.negative_cache.pop(key, None)
        except Exception:
            pass

        if key in store.negative_cache:
            expire_dt, fail_cnt, fail_reason = store.negative_cache[key]
            if datetime.now() < expire_dt:
                latency_ms = (time.perf_counter() - t0) * 1000
                self._record_telemetry("negative_hits", latency_ms)
                return ResolvedInstrument("INVALID", sym_clean, prov, "", is_valid=False, error_message=f"Negative cache active ({fail_reason}) until {expire_dt.strftime('%H:%M:%S')}")

        # ── LEVEL 2 & 3: Single-Flight Protected Master Lookup & Probing ─────────────
        flight_lock = self._get_single_flight_lock(key)
        with flight_lock:
            # Double-check memory index inside lock (another thread may have resolved it)
            if key in self._active_indexes.idx_provider_mapping:
                res = self._active_indexes.idx_provider_mapping[key]
                latency_ms = (time.perf_counter() - t0) * 1000
                self._record_telemetry("memory_hits", latency_ms)
                return res

            adapter = self._adapters.get(prov)
            if not adapter:
                # Fallback to standard symbol formatting if provider adapter is missing
                cand = f"NSE:{sym_clean}-EQ" if prov == "fyers" else f"{sym_clean}.NS"
                return ResolvedInstrument(f"EQ:{sym_clean}", sym_clean, prov, cand, "NSE", "EQ", 50, "FALLBACK")

            # Look up metadata from instrument_registry
            inst_id = store.idx_by_symbol.get(sym_clean)
            metadata = store.idx_by_id.get(inst_id) if inst_id else None

            # ── LEVEL 2: Broker Instrument Master Lookup ─────────────────────────────
            resolved = adapter.lookup_master(symbol, metadata)
            if resolved and resolved.is_valid:
                self._cache_and_persist_mapping(prov, symbol.strip().upper(), resolved)
                latency_ms = (time.perf_counter() - t0) * 1000
                self._record_telemetry("master_hits", latency_ms)
                return resolved

            # ── LEVEL 3: Smart Series Probing ─────────────────────────────────────────
            from config import FEATURE_ASYNC_SYMBOL_PROBING_V1
            if FEATURE_ASYNC_SYMBOL_PROBING_V1:
                self._enqueue_async_probe(prov, symbol.strip().upper(), adapter, metadata)
                fallback_cand = f"NSE:{sym_clean}-EQ" if prov == "fyers" else f"NSE_EQ|{sym_clean}"
                logger.info(f"🚀 [AsyncProbe] Dispatched non-blocking probe for {sym_clean} on {prov}. Using fallback {fallback_cand}")
                return ResolvedInstrument(f"EQ:{sym_clean}", sym_clean, prov, fallback_cand, "NSE", "EQ", 70, "PROBING_ASYNC")

            resolved = adapter.probe_candidates(symbol, metadata)
            if resolved and resolved.is_valid:
                self._cache_and_persist_mapping(prov, symbol.strip().upper(), resolved)
                latency_ms = (time.perf_counter() - t0) * 1000
                self._record_telemetry("probe_hits", latency_ms)
                
                # Log audit event to DB
                try:
                    from database import log_resolution_event_db
                    log_resolution_event_db(prov, sym_clean, resolved.mapped_symbol, "PROBE_SUCCESS", "SMART_PROBE", resolved.confidence_score, latency_ms)
                except Exception:
                    pass
                return resolved

            # ── LEVEL 4: All Levels Failed → Store Typed Negative Cache with Exponential Backoff
            fail_record = store.negative_cache.get(key, (None, 0, "NOT_FOUND"))
            fail_count = fail_record[1] + 1
            fail_reason = "MASTER_MISSING" if str(symbol).startswith("^") else "NOT_FOUND"
            hours_map = {1: 1, 2: 6, 3: 24, 4: 72}
            backoff_hours = hours_map.get(fail_count, 168)
            expire_dt = datetime.now() + timedelta(hours=backoff_hours)
            store.negative_cache[key] = (expire_dt, fail_count, fail_reason)

            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning(f"🚫 [SymbolResolver] Unresolved {sym_clean} on {prov}. Negative caching for {backoff_hours}h.")

            try:
                from database import save_symbol_mapping_db, log_resolution_event_db
                save_symbol_mapping_db(prov, sym_clean, "", status=MappingStatus.INVALID.value, retry_after=expire_dt)
                log_resolution_event_db(prov, sym_clean, "", "FAILURE", "EXHAUSTED", 0, latency_ms, "EXHAUSTED_ALL_PROBES")
            except Exception:
                pass

            return ResolvedInstrument("INVALID", sym_clean, prov, "", is_valid=False, error_message=f"Could not resolve {sym_clean} across available series")

    def _cache_and_persist_mapping(self, provider: str, original_sym: str, resolved: ResolvedInstrument):
        """Updates in-memory store and persists learned mapping to DB."""
        key = (provider, original_sym)
        self._active_indexes.idx_provider_mapping[key] = resolved
        self._active_indexes.negative_cache.pop(key, None)

        try:
            from database import save_symbol_mapping_db
            save_symbol_mapping_db(
                provider=provider,
                original_symbol=original_sym,
                mapped_symbol=resolved.mapped_symbol,
                instrument_id=resolved.instrument_id,
                exchange=resolved.exchange,
                series=resolved.series,
                confidence_score=resolved.confidence_score,
                mapping_source=resolved.source,
                status=MappingStatus.ACTIVE.value
            )
        except Exception as e:
            logger.error(f"Failed to persist learned mapping to DB for {original_sym}: {e}")

    def record_failure(self, provider: str, symbol: str):
        """
        Called when data fetch fails for a mapped symbol. Increments failure count,
        and marks status STALE if consecutive_failures >= 3 AND last_success > 30 days.
        """
        sym_clean = symbol.strip().upper()
        prov = provider.lower()
        try:
            from database import record_symbol_mapping_failure_db
            res = record_symbol_mapping_failure_db(prov, sym_clean)
            if res and res.get("status") == MappingStatus.STALE.value:
                logger.warning(f"🔄 [AutoHealing] Marked mapping {prov}/{sym_clean} as STALE. Evicting from MemoryIndexStore.")
                self._active_indexes.idx_provider_mapping.pop((prov, sym_clean), None)
        except Exception as e:
            logger.error(f"Error recording mapping failure for {prov}/{sym_clean}: {e}")

    def _record_telemetry(self, hit_type: str, latency_ms: float):
        with self._metrics_lock:
            self._metrics[hit_type] = self._metrics.get(hit_type, 0) + 1
            self._metrics["latencies_ms"].append(latency_ms)
            if len(self._metrics["latencies_ms"]) > 1000:
                self._metrics["latencies_ms"] = self._metrics["latencies_ms"][-1000:]

    def get_metrics_summary(self) -> dict:
        with self._metrics_lock:
            total = max(1, self._metrics["total_requests"])
            lats = sorted(self._metrics["latencies_ms"]) if self._metrics["latencies_ms"] else [0.0]
            p50 = lats[int(len(lats) * 0.50)]
            p95 = lats[int(len(lats) * 0.95)]
            p99 = lats[int(len(lats) * 0.99)]
            return {
                "total_requests": total,
                "memory_hit_ratio": f"{(self._metrics['memory_hits'] / total) * 100:.2f}%",
                "master_hit_ratio": f"{(self._metrics['master_hits'] / total) * 100:.2f}%",
                "probe_hit_ratio": f"{(self._metrics['probe_hits'] / total) * 100:.2f}%",
                "negative_hit_ratio": f"{(self._metrics['negative_hits'] / total) * 100:.2f}%",
                "latency_p50_ms": round(p50, 4),
                "latency_p95_ms": round(p95, 4),
                "latency_p99_ms": round(p99, 4),
            }


# Helper accessor function
def get_symbol_resolver() -> SymbolResolutionService:
    return SymbolResolutionService()
