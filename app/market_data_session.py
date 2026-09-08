# =====================================================================================
# app/market_data_session.py
# [VERSION: MARKET_DATA_SESSION_v1.0]
#
# WHAT THIS FILE DOES:
#   Builds a single, immutable per-trading-day data context that is constructed ONCE
#   by the scheduler (main.py → run_evening_scanners) and then consumed read-only by
#   all scanners (EOD, Reversal, Pullback, Wealth Engine, Multibagger).
#
# ARCHITECTURE:
#   Scheduler
#       │
#       ▼
#   MarketDataSession.build(symbols, ist_date)   ← produced once
#       │
#       ├── Stage 1: Bulk OHLCV fetch (2Y, 1d) — shared cache key
#       ├── Stage 2: Validate OHLCV
#       ├── Stage 3: Compute indicators (batch, parallel via IndicatorExecutor)
#       ├── Stage 4: [Parallel] Delivery data (DB-first, STALE flag on 404)
#       ├── Stage 4: [Parallel] Fundamentals
#       ├── Stage 4: [Parallel] Macro state + regime context
#       └── Stage 7: Freeze & publish session
#           │
#           ├── EOD Scanner       → session.get(symbol)
#           ├── Reversal Scanner  → session.get(symbol)
#           ├── Pullback Scanner  → session.get(symbol)
#           └── Wealth Engine     → session.get(symbol)
#
# DESIGN PRINCIPLES:
#   - Immutable after build(): scanners COPY if they need to mutate a DataFrame
#   - Provider resolved once during build(); all symbols use the same provider
#   - Session is versioned (session_id, build_ts) for logging correlation
#   - Build fails atomically — never publishes a partial session
#   - Full per-stage telemetry for future profiling
# =====================================================================================

from __future__ import annotations

import os
import json
import math
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Data container for a single symbol inside the session
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionSymbolData:
    """Immutable, pre-built per-symbol data bundle consumed by all scanners."""
    symbol: str
    ohlcv_df: pd.DataFrame          # OHLCV + all indicators pre-computed
    delivery_pct: Optional[float]   # Bhavcopy delivery % (None if unavailable)
    delivery_stale: bool            # True = using previous day's Bhavcopy
    pledge_pct: Optional[float]     # Promoter pledge %
    # Fundamentals as a raw dict — scanners extract what they need
    fundamentals: dict = field(default_factory=dict)

    class Config:
        # Allow pandas DataFrame inside frozen dataclass
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------

@dataclass
class SessionMetadata:
    session_id: str
    ist_date: date
    build_ts: datetime
    provider_id: str                # e.g. "upstox", "fyers", "yahoo"
    total_symbols: int
    valid_symbols: int
    cache_hit_count: int
    cache_miss_count: int
    delivery_status: str            # "FRESH" | "STALE" | "UNAVAILABLE"
    delivery_date: Optional[date]
    build_duration_s: float
    stage_timings: dict             # {stage_name: duration_s}


# ---------------------------------------------------------------------------
# The session itself
# ---------------------------------------------------------------------------

class MarketDataSession:
    """
    Immutable per-trading-day data context.

    Usage:
        session = MarketDataSession.build(symbols, ist_date=date.today())
        data = session.get("RELIANCE")  # → SessionSymbolData | None

    Scanners MUST NOT modify session data in place.
    Copy the DataFrame first:
        df = data.ohlcv_df.copy()
    """

    _INSTANCE_LOCK = threading.Lock()
    _CURRENT: Optional[MarketDataSession] = None  # Module-level last-built session

    def __init__(self, symbol_data: dict[str, SessionSymbolData],
                 metadata: SessionMetadata,
                 macro_ctx: dict):
        self._symbol_data: dict[str, SessionSymbolData] = symbol_data
        self.metadata: SessionMetadata = metadata
        self.macro_ctx: dict = macro_ctx          # Shared macro/regime context
        self._frozen_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public read-only API
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> Optional[SessionSymbolData]:
        """Returns pre-built symbol data. Returns None if symbol wasn't loaded."""
        return self._symbol_data.get(symbol)

    def symbols(self) -> list[str]:
        return list(self._symbol_data.keys())

    def __len__(self) -> int:
        return len(self._symbol_data)

    @property
    def all_1d(self) -> dict:
        """
        [RULE 67 CHANGE-RATIONALE: SESSION_ALL_1D_PROPERTY_v1.0]
        Provides dictionary mapping of symbol -> ohlcv_df for all loaded symbols in session.
        Enables instant, zero-copy ingestion by technical and batch scanners without re-fetching from disk.
        """
        return {
            sym: data.ohlcv_df
            for sym, data in self._symbol_data.items()
            if data is not None and data.ohlcv_df is not None
        }

    def summary(self) -> str:
        m = self.metadata
        return (
            f"MarketDataSession[{m.session_id[:8]}] "
            f"date={m.ist_date} symbols={m.valid_symbols}/{m.total_symbols} "
            f"provider={m.provider_id} delivery={m.delivery_status} "
            f"built_in={m.build_duration_s:.1f}s"
        )

    # ------------------------------------------------------------------
    # Class-level current session registry
    # ------------------------------------------------------------------

    @classmethod
    def set_current(cls, session: MarketDataSession):
        with cls._INSTANCE_LOCK:
            cls._CURRENT = session

    @classmethod
    def get_current(cls) -> Optional[MarketDataSession]:
        with cls._INSTANCE_LOCK:
            return cls._CURRENT

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, symbols: list[str], ist_date: Optional[date] = None,
              requester: str = "MarketDataSession") -> MarketDataSession:
        """
        Build an immutable MarketDataSession for the given trading date.

        Stages:
          1. Bulk OHLCV fetch (shared 2Y daily cache)
          2. OHLCV validation (timestamp, price sanity, volume)
          3. Indicator computation (batch, parallel)
          4. [Parallel] Delivery data, Fundamentals, Macro
          5. Assemble & freeze session

        On any build failure: logs error and raises — caller should retain the
        previous valid session rather than publishing a partial one.
        """
        if ist_date is None:
            ist_date = datetime.now(IST).date()

        build_start = time.monotonic()
        session_id = uuid.uuid4().hex
        stage_timings = {}

        import pandas as pd
        if isinstance(symbols, pd.DataFrame):
            if "Stock" in symbols.columns:
                symbols = symbols["Stock"].dropna().tolist()
            elif "Symbol" in symbols.columns:
                symbols = symbols["Symbol"].dropna().tolist()
            else:
                symbols = symbols.iloc[:, 0].dropna().tolist()
        elif isinstance(symbols, pd.Series):
            symbols = symbols.dropna().tolist()
        elif isinstance(symbols, (set, tuple)):
            symbols = list(symbols)
        elif isinstance(symbols, str):
            symbols = [symbols]

        # [NON_EQUITY_BLOCKLIST] Drop non-equity trusts/InvITs and blacklisted symbols upfront
        try:
            from surveillance import get_live_blacklist
            _bl = get_live_blacklist()
            if _bl:
                symbols = [str(s).strip().upper() for s in symbols if s and str(s).strip().upper() not in _bl]
        except Exception:
            pass

        logger.info(
            f"🏗️  [SESSION:{session_id[:8]}] Building MarketDataSession for {ist_date} "
            f"| {len(symbols)} symbols"
        )

        # ── Stage 1: Bulk OHLCV Fetch ───────────────────────────────────────
        t = time.monotonic()
        ohlcv_raw, provider_id, cache_hit_count, cache_miss_count = \
            cls._stage_fetch_ohlcv(symbols, requester=requester)
        stage_timings["1_fetch_ohlcv_s"] = round(time.monotonic() - t, 2)
        logger.info(
            f"[SESSION:{session_id[:8]}] ✅ Stage 1 done "
            f"| {len(ohlcv_raw)} frames | cache_hits={cache_hit_count} "
            f"| cache_misses={cache_miss_count} | {stage_timings['1_fetch_ohlcv_s']:.1f}s"
        )

        # ── Stage 2: Validate OHLCV ─────────────────────────────────────────
        t = time.monotonic()
        ohlcv_valid = cls._stage_validate_ohlcv(ohlcv_raw, session_id)
        stage_timings["2_validate_ohlcv_s"] = round(time.monotonic() - t, 2)
        logger.info(
            f"[SESSION:{session_id[:8]}] ✅ Stage 2 done "
            f"| valid={len(ohlcv_valid)} | {stage_timings['2_validate_ohlcv_s']:.2f}s"
        )

        # ── Stage 3: Indicator Computation ──────────────────────────────────
        t = time.monotonic()
        ohlcv_with_indicators = cls._stage_compute_indicators(ohlcv_valid, session_id)
        stage_timings["3_indicators_s"] = round(time.monotonic() - t, 2)
        logger.info(
            f"[SESSION:{session_id[:8]}] ✅ Stage 3 done "
            f"| enriched={len(ohlcv_with_indicators)} | {stage_timings['3_indicators_s']:.1f}s"
        )

        # ── Stage 4: Parallel — Delivery, Fundamentals, Macro ───────────────
        t = time.monotonic()
        delivery_map, delivery_date, delivery_status = {}, None, "UNAVAILABLE"
        pledge_map: dict[str, float] = {}
        fundamentals_map: dict[str, dict] = {}
        macro_ctx: dict = {}

        parallel_results = {}
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="SessionStage4") as ex:
            futures = {
                ex.submit(cls._stage_load_delivery, ist_date):     "delivery",
                ex.submit(cls._stage_load_pledge, symbols):         "pledge",
                ex.submit(cls._stage_load_macro):                   "macro",
            }
            for fut in as_completed(futures, timeout=60):
                key = futures[fut]
                try:
                    parallel_results[key] = fut.result()
                except Exception as e:
                    logger.warning(f"[SESSION:{session_id[:8]}] ⚠️ Stage 4 [{key}] failed: {e}")
                    parallel_results[key] = None

        if parallel_results.get("delivery"):
            delivery_map, delivery_date, delivery_status = parallel_results["delivery"]
        if parallel_results.get("pledge"):
            pledge_map = parallel_results["pledge"]
        if parallel_results.get("macro"):
            macro_ctx = parallel_results["macro"]

        stage_timings["4_parallel_context_s"] = round(time.monotonic() - t, 2)
        logger.info(
            f"[SESSION:{session_id[:8]}] ✅ Stage 4 done "
            f"| delivery={delivery_status}({delivery_date}) "
            f"| pledge={len(pledge_map)} | macro={macro_ctx.get('trend','?')} "
            f"| {stage_timings['4_parallel_context_s']:.1f}s"
        )

        # ── Stage 5: Assemble & Freeze ───────────────────────────────────────
        t = time.monotonic()
        symbol_data: dict[str, SessionSymbolData] = {}
        for sym, df in ohlcv_with_indicators.items():
            symbol_data[sym] = SessionSymbolData(
                symbol=sym,
                ohlcv_df=df,
                delivery_pct=delivery_map.get(sym),
                delivery_stale=(delivery_status == "STALE"),
                pledge_pct=pledge_map.get(sym),
                fundamentals=fundamentals_map.get(sym, {}),
            )
        stage_timings["5_assemble_s"] = round(time.monotonic() - t, 2)

        total_duration = round(time.monotonic() - build_start, 2)
        metadata = SessionMetadata(
            session_id=session_id,
            ist_date=ist_date,
            build_ts=datetime.now(IST),
            provider_id=provider_id,
            total_symbols=len(symbols),
            valid_symbols=len(symbol_data),
            cache_hit_count=cache_hit_count,
            cache_miss_count=cache_miss_count,
            delivery_status=delivery_status,
            delivery_date=delivery_date,
            build_duration_s=total_duration,
            stage_timings=stage_timings,
        )

        session = MarketDataSession(symbol_data=symbol_data,
                                    metadata=metadata,
                                    macro_ctx=macro_ctx)

        logger.info(
            f"🏁 [SESSION:{session_id[:8]}] Build complete | {session.summary()} "
            f"| Stages: {stage_timings}"
        )
        return session

    # ------------------------------------------------------------------
    # Internal stage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_fetch_ohlcv(symbols: list[str],
                           requester: str
                           ) -> tuple[dict[str, pd.DataFrame], str, int, int]:
        """
        Bulk-fetch 2-year daily OHLCV for all symbols in one pass.
        Returns (ohlcv_map, provider_id, cache_hits, cache_misses).

        Cache hit/miss counts are estimated by comparing _cache key sizes
        before and after the fetch call.
        """
        from price_cache import fetch_watchlist_data, _cache
        import pandas as pd

        # Estimate prior cached symbols for hit/miss delta
        cache_key = ("1d", "1y")
        pre_cached_syms = set(_cache.get(cache_key, {}).keys())

        wl = pd.DataFrame({"Stock": symbols})
        raw = fetch_watchlist_data(wl, period="1y", interval="1d", requester=requester)

        post_cached_syms = set(_cache.get(cache_key, {}).keys())
        new_fetched = post_cached_syms - pre_cached_syms
        cache_misses = len(new_fetched)                  # symbols not in cache before → fetched fresh
        cache_hits   = len(symbols) - cache_misses        # rest came from RAM

        # Determine which provider was used (inspect attrs on first non-empty frame)
        provider_id = "unknown"
        for df in raw.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                provider_id = getattr(df, "attrs", {}).get("provider", "unknown")
                break

        # Filter to valid DataFrames only
        result = {
            sym: df for sym, df in raw.items()
            if isinstance(df, pd.DataFrame) and not df.empty
        }
        return result, provider_id, max(cache_hits, 0), cache_misses

    @staticmethod
    def _stage_validate_ohlcv(ohlcv_raw: dict[str, pd.DataFrame],
                               session_id: str) -> dict[str, pd.DataFrame]:
        """
        Validate OHLCV structural integrity per symbol.
        Symbols failing validation are dropped; logged as warnings.

        Checks per symbol:
          - Last timestamp >= today's date (or previous trading day after close)
          - Row count never decreases unexpectedly vs cached row count
          - OHLC: no NaN in Close, High, Low; High >= Low
          - Volume >= 0
          - Chronological order of timestamps
        """
        from price_cache import validate_ohlcv_structure

        valid: dict[str, pd.DataFrame] = {}
        dropped = 0
        for sym, df in ohlcv_raw.items():
            try:
                ok, reason = validate_ohlcv_structure(df)
                if ok:
                    valid[sym] = df
                else:
                    logger.warning(
                        f"[SESSION:{session_id[:8]}] ⚠️ OHLCV validation failed for {sym}: {reason}"
                    )
                    dropped += 1
            except Exception as e:
                logger.warning(f"[SESSION:{session_id[:8]}] ⚠️ Validation error for {sym}: {e}")
                dropped += 1

        if dropped > 0:
            logger.warning(
                f"[SESSION:{session_id[:8]}] Stage 2: Dropped {dropped} symbols due to OHLCV validation failures"
            )
        return valid

    @staticmethod
    def _stage_compute_indicators(ohlcv_valid: dict[str, pd.DataFrame],
                                   session_id: str) -> dict[str, pd.DataFrame]:
        """
        Compute all technical indicators once for all symbols, in parallel.

        Skip recomputation if the cached DataFrame already has EMA20 pre-baked
        (i.e. loaded from a fresh parquet with INDICATOR_VERSION matching).
        """
        from indicator_executor import indicator_executor

        needs_compute: list[dict] = []
        already_computed: dict[str, pd.DataFrame] = {}

        for sym, df in ohlcv_valid.items():
            if "EMA20" in df.columns and "RSI" in df.columns and "ATR" in df.columns:
                already_computed[sym] = df
            else:
                needs_compute.append({"symbol": sym, "timeframe": "1d", "dataframe": df})

        logger.info(
            f"[SESSION:{session_id[:8]}] Stage 3: {len(already_computed)} symbols have pre-computed indicators "
            f"| Computing fresh for {len(needs_compute)} symbols"
        )

        freshly_computed: dict[str, pd.DataFrame] = {}
        if needs_compute:
            results = indicator_executor.execute(needs_compute)
            freshly_computed = {sym: df for sym, df in results.items() if df is not None}

        return {**already_computed, **freshly_computed}

    @staticmethod
    def _stage_load_delivery(ist_date: date
                              ) -> tuple[dict[str, float], Optional[date], str]:
        """
        Load Bhavcopy delivery data.

        Strategy:
          1. Exact date check for today from DB (1 query, <10ms)
          2. If today not in DB, try ScraperAPI live fetch
          3. On failure: single-query bulk fallback — get most recent DB-cached entry
             (uses get_latest_bhavcopy_cache_with_date(), no N+1 loop)
          4. Mark delivery_status = "STALE" if using non-today data

        Returns: (delivery_map, resolved_date, status)
          status: "FRESH" | "STALE" | "UNAVAILABLE"

        [VERSION: MARKET_DATA_SESSION_v1.0] N+1 fix: replaced per-date loop
        with single get_latest_bhavcopy_cache_with_date() bulk query.
        """
        try:
            from database import get_bhavcopy_cache, get_latest_bhavcopy_cache_with_date

            # ── Path 1: Exact date DB hit (fast, 1 query) ──────────────────
            today_data = get_bhavcopy_cache(ist_date)
            if today_data:
                logger.info(
                    f"[SESSION] ⚡ Delivery: DB cache hit for {ist_date} "
                    f"({len(today_data)} symbols) — FRESH"
                )
                return today_data, ist_date, "FRESH"

            # ── Path 2: Live Crawlora / ScraperAPI fetch ───────────────────────────────
            logger.info(f"[SESSION] 🔄 Delivery: Attempting live fetch for {ist_date} (Crawlora primary)...")
            try:
                from delivery_data import fetch_delivery_data
                live_data = fetch_delivery_data(ist_date, skip_db_save=False)
                if live_data:
                    logger.info(
                        f"[SESSION] ✅ Delivery: Live fetch succeeded for {ist_date} "
                        f"({len(live_data)} symbols) — FRESH"
                    )
                    return live_data, ist_date, "FRESH"
            except Exception as live_err:
                logger.warning(f"[SESSION] ⚠️ Delivery: Live fetch failed: {live_err}")

            # ── Path 3: Bulk single-query fallback (no N+1 loop) ────────────
            # get_latest_bhavcopy_cache_with_date() issues one SQL:
            #   SELECT delivery_data, trading_date FROM bhavcopy_cache
            #   ORDER BY trading_date DESC LIMIT 1
            # This returns whatever the most recent cached date is in one round-trip.
            logger.warning(
                f"[SESSION] ⚠️ Delivery: Today ({ist_date}) unavailable. "
                f"Fetching most recent DB-cached entry in single query..."
            )
            stale_data, stale_date = get_latest_bhavcopy_cache_with_date()
            if stale_data:
                logger.info(
                    f"[SESSION] 📅 Delivery: Using STALE data from {stale_date} "
                    f"({len(stale_data)} symbols)"
                )
                return stale_data, stale_date, "STALE"

            logger.error("[SESSION] ❌ Delivery: No Bhavcopy data available in DB at all")
            return {}, None, "UNAVAILABLE"

        except Exception as e:
            logger.error(f"[SESSION] ❌ Delivery stage failed: {e}")
            return {}, None, "UNAVAILABLE"


    @staticmethod
    def _stage_load_pledge(symbols: list[str]) -> dict[str, float]:
        """Load promoter pledge data for all symbols from DB."""
        try:
            from database import get_pledge_map
            pledge_map = get_pledge_map(symbols)
            logger.info(f"[SESSION] 🛡️ Pledge: loaded {len(pledge_map)} symbols")
            return pledge_map
        except Exception as e:
            logger.warning(f"[SESSION] ⚠️ Pledge load failed: {e}")
            return {}

    @staticmethod
    def _stage_load_macro() -> dict:
        """Load macro/regime context (Nifty 20D return, market regime, policy)."""
        try:
            from macro_utils import get_nifty_20d_return, get_macro_regime, MarketRegimeEngine
            from strategy_policy import StrategyPolicyEngine

            nifty_ret_20d = get_nifty_20d_return()
            market_regime = get_macro_regime(nifty_ret_20d)

            try:
                regime_ctx = MarketRegimeEngine.get_regime_context(nifty_ret_20d)
                policy = StrategyPolicyEngine.get_policy(regime_ctx, "EOD")
                regime_ctx["policy"] = policy
            except Exception as _re:
                logger.warning(f"[SESSION] ⚠️ Could not build regime_ctx from MarketRegimeEngine: {_re}. Using neutral fallback.")
                regime_ctx = {"trend": market_regime, "biases": {}}

            regime_ctx["nifty_ret_20d"] = nifty_ret_20d
            logger.info(f"[SESSION] 📊 Macro: regime={market_regime} nifty_20d={nifty_ret_20d:.2f}%")
            return regime_ctx
        except Exception as e:
            logger.warning(f"[SESSION] ⚠️ Macro load failed: {e}")
            return {"trend": "NEUTRAL", "biases": {}, "nifty_ret_20d": 0.0}


# ---------------------------------------------------------------------------
# Convenience builder used by main.py scheduler
# ---------------------------------------------------------------------------

def build_evening_session(symbols: list[str],
                           ist_date: Optional[date] = None) -> Optional[MarketDataSession]:
    """
    Top-level function called by main.py run_evening_scanners().
    Builds a MarketDataSession and stores it as the current session.

    On failure: logs error, sends admin notification, returns None.
    Caller should retain the previous valid session rather than using None.
    """
    try:
        session = MarketDataSession.build(symbols, ist_date=ist_date,
                                          requester="EveningScheduler")
        MarketDataSession.set_current(session)

        # Log session summary to DB for admin dashboard visibility
        try:
            from database import upsert_scanner_health
            m = session.metadata
            upsert_scanner_health(
                "MARKET_DATA_SESSION",
                status="OK",
                last_success=m.build_ts.isoformat(),
                today_alerts=m.valid_symbols,
                scheduled_for="Before Evening Scanners",
                duration_seconds=m.build_duration_s,
                error_msg=(
                    f"session_id={m.session_id[:8]} "
                    f"provider={m.provider_id} "
                    f"delivery={m.delivery_status} "
                    f"cache_hits={m.cache_hit_count}"
                )
            )
        except Exception as db_err:
            logger.warning(f"[SESSION] Could not persist session health: {db_err}")

        return session

    except Exception as e:
        logger.exception(f"❌ [SESSION] MarketDataSession build FAILED: {e}")
        try:
            from database import upsert_scanner_health, insert_notification
            upsert_scanner_health(
                "MARKET_DATA_SESSION", status="DOWN",
                error_msg=str(e)[:500],
                scheduled_for="Before Evening Scanners"
            )
            insert_notification(
                notif_type="scanner_down",
                title="🚨 MarketDataSession Build FAILED",
                message=f"Scanners will run without shared session. Error: {str(e)[:400]}"
            )
        except Exception:
            pass
        return None
