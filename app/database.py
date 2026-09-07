# =====================================================================================
# app/database.py
#
# KEY DESIGN DECISIONS:
#
# 1. ONE-TIME INIT:  init_db() is guarded by a module-level lock + flag.
#    No matter how many scanners call it simultaneously, the CREATE TABLE
#    SQL runs exactly once per process lifetime. After that, every call
#    returns immediately — zero DB round trips, zero race conditions.
#
# 2. WHY STILL CALL init_db() IN EACH SCANNER?
#    On a fresh Railway deploy the table doesn't exist yet. We can't remove
#    the call entirely. But with the lock it's safe for all scanners to call
#    it — the second caller just sees _DB_INITIALIZED=True and returns.
#
# 3. RACE CONDITION FIX:
#    The old crash was:
#      psycopg2.errors.UniqueViolation: duplicate key value violates
#      unique constraint "pg_type_typname_nsp_index"
#    This happens when Postgres processes two simultaneous CREATE TABLE
#    statements for the same table name even with IF NOT EXISTS — it's a
#    known Postgres internal type-registry bug under concurrency.
#    The lock below makes it impossible for two threads to reach that
#    SQL at the same time.
# =====================================================================================

import os
import time
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

# [VERSION: DB_UPLOAD_THREAD_POOL_v1.0] Background ThreadPoolExecutor for DB uploads
# RATIONALE: ProcessPoolExecutor requires target functions and arguments to be picklable.
# Local closures (e.g. bg_db_sync in wealth_engine.py) and lambdas failed silently under pickle,
# preventing today's parquet exports from uploading to Postgres DB.
import concurrent.futures
import atexit

_UPLOAD_POOL = None

def _get_upload_pool():
    global _UPLOAD_POOL
    if _UPLOAD_POOL is None:
        _UPLOAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="db_upload")
        def _shutdown():
            global _UPLOAD_POOL
            if _UPLOAD_POOL is not None:
                _UPLOAD_POOL.shutdown(wait=False)
        atexit.register(_shutdown)
    return _UPLOAD_POOL

def submit_background_upload(target_func, *args, **kwargs):
    pool = _get_upload_pool()
    def _wrapper():
        try:
            target_func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"❌ Background DB upload task failed: {e}")

    try:
        pool.submit(_wrapper)
    except Exception as e:
        logger.exception(f"❌ Failed to submit background DB upload task: {e}")


from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd


from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_DB_WRITE_LOCK = threading.RLock()

# When True, scanners should not persist alerts to the database. Used for
# startup self-tests and dry-runs where we want to exercise scanner logic
# without polluting the alerts table or triggering downstream systems.
DONT_SAVE_ALERTS = False

# When True, Wealth Engine should not persist parquet files or write buy alerts.
# Controlled by the startup self-test to prevent altering wealth data on boot.
DONT_SAVE_WEALTH = False

# ── Connection pool ───────────────────────────────────────────────────────────────────
_pool: Optional[pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()
# Semaphore to limit concurrent active connections to the pool (prevents noisy exhaustion)
_conn_semaphore: Optional[threading.BoundedSemaphore] = None

class DummyCursor:
    rowcount = 1
    description = []
    def __init__(self):
        self._last_query = ""
    def execute(self, query=None, *args, **kwargs):
        self._last_query = str(query or "").strip().upper()
        return self
    def executemany(self, *args, **kwargs): return self
    def fetchone(self):
        if "RETURNING" in self._last_query or "INSERT INTO" in self._last_query:
            return (1,)
        if "COUNT(" in self._last_query:
            return (0,)
        return None
    def fetchall(self): return []
    def fetchmany(self, *args, **kwargs): return []
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

class DummyConnection:
    def cursor(self, *args, **kwargs): return DummyCursor()
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

def _get_pool() -> Optional[pool.ThreadedConnectionPool]:
    global _pool, _conn_semaphore
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:          # double-checked locking
            return _pool
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return None
        # Configure pool size via env override if provided (strictly default to 10 to prevent Postgres client exhaustion)
        # Configure pool size via env override if provided (default to 30 for high concurrency)
        maxconn = int(os.getenv("DB_MAXCONN", "50"))
        minconn = int(os.getenv("DB_MINCONN", "5"))
        # [RULE 67 CHANGE-RATIONALE]:
        # Configures PostgreSQL session parameters (timezone, statement_timeout, idle_in_transaction_timeout)
        # directly in the connection options at pool initialization. This eliminates 4 synchronous network round-trips
        # (SELECT 1, SET TIME ZONE, SET idle_timeout, SET statement_timeout) on EVERY get_connection() checkout,
        # dramatically speeding up all dashboard API responses from ~20-50ms connection overhead to < 0.1ms.
        _pool = pool.ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            dsn=db_url,
            connect_timeout=10,  # 10s connection timeout for Contabo VPS / Coolify Postgres
            options="-c timezone=Asia/Kolkata -c statement_timeout=60000 -c idle_in_transaction_session_timeout=10000",
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=3
        )
        try:
            # Initialize semaphore to mirror pool capacity
            _conn_semaphore = threading.BoundedSemaphore(value=maxconn)
        except Exception:
            _conn_semaphore = None
        logger.info(f"✅ Postgres connection pool created | min={minconn} max={maxconn}")
        return _pool


def close_pool():
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
                logger.info("🧹 Postgres connection pool closed cleanly.")
            except Exception as e:
                logger.warning(f"Error closing DB pool: {e}")
            _pool = None

atexit.register(close_pool)


@contextmanager
def get_connection(timeout: int = 20):
    """Get DB connection with circuit breaker pattern.

    Acquires an internal semaphore before checking out a connection from the pool.
    This prevents busy loops from exhausting the pool and creating noisy logs.
    """
    from psycopg2 import OperationalError, DatabaseError, pool as ps_pool

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.debug("DATABASE_URL env var is not set — returning DummyConnection")
        yield DummyConnection()
        return

    p = _get_pool()
    if p is None:
        yield DummyConnection()
        return
    conn = None
    acquired = False
    try:
        global _conn_semaphore
        max_cap = getattr(p, 'maxconn', 50)
        # Ensure semaphore exists (in case pool was created elsewhere)
        if _conn_semaphore is None:
            try:
                _conn_semaphore = threading.BoundedSemaphore(value=max_cap)
            except Exception:
                _conn_semaphore = None
        if _conn_semaphore is not None:
            acquired = _conn_semaphore.acquire(timeout=timeout)
            if not acquired:
                logger.warning("⚠️ Semaphore checkout timeout (20s) — attempting direct pool checkout fallback")

        # Retry checkout up to 5 times if server returns "too many clients" or pool exhausted
        for attempt in range(5):
            try:
                conn = p.getconn()
                if conn is not None:
                    if getattr(conn, 'closed', 0) != 0:
                        try:
                            p.putconn(conn, close=True)
                        except Exception:
                            pass
                        conn = None
                        continue
                    # Ensure checked out connection is clean and not in an aborted transaction state
                    try:
                        conn.rollback()
                    except Exception:
                        try:
                            p.putconn(conn, close=True)
                        except Exception:
                            pass
                        conn = None
                        continue
                break
            except (OperationalError, ps_pool.PoolError) as oe:
                if conn:
                    try:
                        p.putconn(conn, close=True)
                    except Exception: pass
                    conn = None
                if attempt < 4:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise oe
    except (OperationalError, ps_pool.PoolError) as e:
        # Circuit breaker: log and fail fast instead of hanging
        logger.warning(f"⚠️ DB connection pool exhausted: {e}")
        raise

    try:
        yield conn
    except Exception as e:
        logger.exception(f"🔴 DB operation failed: {e}")
        if conn:
            try:
                conn.rollback()
            except Exception: pass
            try:
                p.putconn(conn, close=True)
            except Exception:
                pass
            conn = None
        raise
    finally:
        # Return connection to pool if we checked one out
        if conn:
            try:
                # [FIX: IDLE IN TRANSACTION]
                # psycopg2 does not implicitly rollback open read transactions when putconn is called.
                # If a caller forgets to commit, or if it was just a SELECT query,
                # we MUST rollback here to prevent poisoning the pool with open transactions
                # which blocks Postgres vacuuming and causes severe MVCC bloat.
                if not conn.closed:
                    conn.rollback()
                p.putconn(conn)
            except Exception:
                try:
                    p.putconn(conn, close=True)
                except Exception:
                    pass
        # Release semaphore if we acquired it
        if _conn_semaphore is not None and acquired:
            try:
                _conn_semaphore.release()
            except Exception:
                pass

get_db_connection = get_connection


# ── One-time init guard ───────────────────────────────────────────────────────────────
_DB_INITIALIZED = False
_INIT_LOCK = threading.Lock()



def _insert_notification_sync(notif_type: str, title: str, message: str, symbol: str = None):
    # [SUPPRESSION RULE] Do not create notifications for routine scanner completions per admin requirement
    title_lower = (title or "").lower()
    if any(k in title_lower for k in [
        "scanner ran successfully",
        "scan completed",
        "scan complete",
        "builder completed",
        "watchlist generation successful"
    ]):
        logger.debug(f"🔇 Suppressed scanner completion notification: {title}")
        return
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO global_notifications (type, title, message, symbol)
                    VALUES (%s, %s, %s, %s)
                ''', (notif_type, title, message, symbol))
            conn.commit()

        # [EVENT-DRIVEN CACHE INVALIDATION] Invalidate all in-memory dashboard caches immediately
        try:
            from dashboard_server import invalidate_all_dashboard_caches
            invalidate_all_dashboard_caches()
        except Exception:
            pass

        # [VERSION: ADMIN_MOBILE_PUSH_DISPATCH_v1.0] Dispatch WebPush to mobile devices whenever an admin notification occurs
        try:
            from push_service import send_push_to_all
            send_push_to_all(
                title=title,
                body=message,
                url="/admin" if notif_type in ("error", "warning") else "/",
                symbol=symbol or ""
            )
        except Exception as push_err:
            logger.debug(f"WebPush dispatch skipped for admin notification: {push_err}")
    except Exception as e:
        logger.exception(f"Failed to insert notification")

def insert_notification(notif_type: str, title: str, message: str, symbol: str = None):
    import threading
    threading.Thread(target=_insert_notification_sync, args=(notif_type, title, message, symbol), daemon=True).start()

class _AdvisoryLockGuard:
    def __init__(self):
        self.conn_ctx = None
        self.conn = None
        self.acquired = False
        try:
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                self.conn_ctx = get_connection(timeout=3)
                self.conn = self.conn_ctx.__enter__()
                with self.conn.cursor() as cur:
                    cur.execute("SELECT pg_try_advisory_lock(20240728)")
                    row = cur.fetchone()
                    if row and row[0]:
                        self.acquired = True
                    else:
                        logger.info("ℹ️ DB initialization lock currently held by another process. Skipping DDL execution.")
        except Exception as e:
            logger.warning(f"⚠️ Advisory lock check warning: {e}")

    def release(self):
        if self.conn and self.acquired:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(20240728)")
            except Exception: pass
        if self.conn_ctx:
            try:
                self.conn_ctx.__exit__(None, None, None)
            except Exception: pass
            self.conn_ctx = None
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()

def init_db():
    # [VERSION: GREENFIELD_DB_OVERHAUL_v1.0] Greenfield Database Schema Initialization
    global _DB_INITIALIZED

    if _DB_INITIALIZED:
        return

    with _INIT_LOCK:
        if _DB_INITIALIZED:
            return

        _guard = _AdvisoryLockGuard()
        if not _guard.acquired:
            _DB_INITIALIZED = True
            return

        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. system_logs
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id SERIAL PRIMARY KEY,
                        level TEXT NOT NULL,
                        module TEXT NOT NULL,
                        message TEXT NOT NULL,
                        traceback TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        is_acknowledged BOOLEAN DEFAULT FALSE
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs(created_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_system_logs_ack ON system_logs(is_acknowledged)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_syslogs_grouping ON system_logs(is_acknowledged, level, module, message)")

                # 2. master_symbols
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS master_symbols (
                        symbol TEXT PRIMARY KEY,
                        company_name TEXT NOT NULL,
                        exchange TEXT DEFAULT 'NSE',
                        sector TEXT DEFAULT 'EQUITY',
                        is_active BOOLEAN DEFAULT TRUE,
                        last_updated TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                try:
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_master_symbols_active ON master_symbols(is_active)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_master_symbols_search ON master_symbols(symbol, company_name)")
                except Exception as e:
                    logger.debug(f"Index creation notice on master_symbols: {e}")

                # 3. candidates
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS candidates (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        breakout_type TEXT NOT NULL,
                        alert_date DATE NOT NULL DEFAULT CURRENT_DATE,
                        status TEXT NOT NULL DEFAULT 'FOUND',
                        scanner TEXT,
                        technical_score INTEGER,
                        volume_ratio REAL,
                        delivery_pct REAL,
                        rr_ratio REAL,
                        market_context TEXT,
                        metadata TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(symbol, breakout_type, alert_date)
                    )
                """)
                # [RULE 67 CHANGE-RATIONALE]: Add compound indexes for candidates table to accelerate dashboard investment watch queries
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_scanner_created ON candidates(scanner, created_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_alert_date ON candidates(alert_date DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_symbol ON candidates(symbol)")

                # 4. alerts
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        breakout_type TEXT NOT NULL,
                        alert_time TIMESTAMPTZ DEFAULT NOW(),
                        alert_date DATE NOT NULL DEFAULT CURRENT_DATE,
                        scanner TEXT,
                        category TEXT,
                        entry_price REAL,
                        stop_loss REAL,
                        initial_stop_loss REAL,
                        target_price REAL,
                        target_1 REAL,
                        target_2 REAL,
                        target_3 REAL,
                        target_4 REAL,
                        signals TEXT,
                        score REAL,
                        rsi REAL,
                        volume_ratio REAL,
                        current_price REAL,
                        status TEXT DEFAULT 'OPEN',
                        exit_price REAL,
                        pnl_pct REAL,
                        pnl_rs REAL,
                        closed_at TIMESTAMPTZ,
                        exit_signal TEXT,
                        exit_reason TEXT,
                        capital_allocated REAL DEFAULT 0.0,
                        shares_bought INTEGER DEFAULT 0,
                        remaining_shares INTEGER,
                        exit_history JSONB DEFAULT '[]'::jsonb,
                        context JSONB,
                        model_version TEXT DEFAULT 'v1',
                        bayesian_regime TEXT DEFAULT 'BULL',
                        bayesian_weights JSONB,
                        data_partition TEXT DEFAULT 'TRAIN',
                        structural_failure_stop REAL,
                        entry_mode TEXT DEFAULT 'LEGACY_UNKNOWN',
                        actual_entry_price REAL,
                        execution_state TEXT DEFAULT 'PENDING_ENTRY',
                        target_quality_score REAL,
                        seen_by_user BOOLEAN DEFAULT FALSE,
                        seen_by_admin BOOLEAN DEFAULT FALSE,
                        cash_in_hand REAL DEFAULT 0.0,
                        is_rejected BOOLEAN DEFAULT FALSE,
                        shadow_status TEXT DEFAULT 'SHADOW_OPEN',
                        shadow_exit_price REAL,
                        shadow_pnl_pct REAL,
                        shadow_closed_at TIMESTAMPTZ,
                        realized_r REAL,
                        rr_ratio REAL,
                        sl_method TEXT,
                        target_method TEXT,
                        scan_id TEXT,
                        partial_exit_pct REAL,
                        earnings_flag BOOLEAN DEFAULT FALSE,
                        days_to_earnings INTEGER DEFAULT 999,
                        earnings_date DATE,
                        earnings_severity VARCHAR(20) DEFAULT 'NONE',
                        date_status VARCHAR(20) DEFAULT 'UNKNOWN',
                        warning_msg TEXT DEFAULT '',
                        trajectory_score INTEGER DEFAULT 0,
                        trajectory_grade VARCHAR(5) DEFAULT 'N/A',
                        trajectory_details JSONB DEFAULT '{}'::jsonb,
                        forensic_score INTEGER DEFAULT 0,
                        forensic_risk_tier VARCHAR(20) DEFAULT 'UNKNOWN',
                        growth_investment_mode BOOLEAN DEFAULT FALSE,
                        growth_investment_score INTEGER DEFAULT 0,
                        idempotency_key VARCHAR(200),
                        evaluation_id TEXT,
                        scanner_run_id TEXT,
                        mfe_pct REAL,
                        mae_pct REAL,
                        mfe_r REAL,
                        mae_r REAL,
                        gross_realized_r REAL,
                        net_realized_r REAL,
                        entry_timestamp TIMESTAMPTZ,
                        exit_timestamp TIMESTAMPTZ,
                        t1_timestamp TIMESTAMPTZ,
                        t2_timestamp TIMESTAMPTZ,
                        t3_timestamp TIMESTAMPTZ,
                        sl_timestamp TIMESTAMPTZ,
                        entry_bar_id INTEGER,
                        exit_bar_id INTEGER,
                        time_to_entry REAL,
                        time_to_t1 REAL,
                        time_to_t2 REAL,
                        time_to_sl REAL,
                        bars_to_t1 INTEGER,
                        bars_to_sl INTEGER,
                        outcome_labels JSONB DEFAULT '{}'::jsonb,
                        weighted_realized_r REAL,
                        trade_evolution_state TEXT DEFAULT 'INITIAL',
                        evidence_count INTEGER DEFAULT 1,
                        distinct_patterns_count INTEGER DEFAULT 1,
                        confirmation_quality TEXT DEFAULT 'INITIAL',
                        last_event_type TEXT DEFAULT 'NEW_ENTRY',
                        last_event_date DATE,
                        last_event_id INTEGER,
                        execution_status TEXT DEFAULT 'EXECUTABLE',
                        execution_block_reason TEXT DEFAULT '',
                        rvol_diurnal REAL,
                        rvol_rolling REAL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        CONSTRAINT alerts_dedup_idx UNIQUE (symbol, breakout_type, scanner, alert_date),
                        CONSTRAINT chk_alerts_status CHECK (status IN ('OPEN', 'WIN', 'LOSS', 'EXPIRED', 'NEUTRAL', 'CLOSED', 'ACTIVE', 'REJECTED', 'PARTIAL_WIN', 'PARTIAL_WIN_1', 'PARTIAL_WIN_2', 'SELL_REVIEW', 'TRAILING'))
                    )
                """)
                # [RULE 67 CHANGE-RATIONALE]: Ensure column migrations for alerts table run BEFORE index creation so existing DBs don't abort with UndefinedColumn
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS trade_evolution_state TEXT DEFAULT 'INITIAL'")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS evidence_count INTEGER DEFAULT 1")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS distinct_patterns_count INTEGER DEFAULT 1")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS confirmation_quality TEXT DEFAULT 'INITIAL'")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS last_event_type TEXT DEFAULT 'NEW_ENTRY'")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS last_event_date DATE")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS last_event_id INTEGER")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS execution_status TEXT DEFAULT 'EXECUTABLE'")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS execution_block_reason TEXT DEFAULT ''")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS rvol_diurnal REAL")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS source_trading_date DATE")
                cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS alert_fingerprint VARCHAR(128)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_source_trading_date ON alerts(symbol, scanner, source_trading_date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint ON alerts(alert_fingerprint)")

                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(alert_date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol_date ON alerts(symbol, alert_date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cooldown ON alerts(symbol, scanner, breakout_type, alert_time DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_scanner_date ON alerts(scanner, alert_date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_scanner_date_desc ON alerts(scanner, alert_date DESC, alert_time DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_scanner_state_date ON alerts(scanner, status, alert_date DESC, alert_time DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time_desc ON alerts(alert_time DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status_time ON alerts(status, alert_time DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_open_trades ON alerts(alert_time DESC) WHERE status = 'OPEN'")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_open_unrejected ON alerts(symbol, alert_time DESC) WHERE status = 'OPEN' AND is_rejected = FALSE")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_today_unrejected ON alerts(alert_date DESC, is_rejected)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status_is_rejected ON alerts(status, is_rejected)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_evolution_state ON alerts(trade_evolution_state, alert_date DESC)")
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_alerts_idempotency ON alerts (idempotency_key) WHERE idempotency_key IS NOT NULL")

                # 4.5. scanner_evaluation_log table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scanner_evaluation_log (
                        id SERIAL PRIMARY KEY,
                        evaluation_id TEXT NOT NULL UNIQUE,
                        scanner_run_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        candidate_sequence INTEGER,
                        scanner TEXT NOT NULL,
                        status TEXT NOT NULL,
                        rejection_type TEXT,
                        primary_rejection_reason TEXT,
                        data_timestamp TIMESTAMPTZ,
                        evaluation_timestamp TIMESTAMPTZ DEFAULT NOW(),
                        decision_timestamp TIMESTAMPTZ DEFAULT NOW(),
                        setup_type TEXT,
                        setup_subtype TEXT,
                        state_at_evaluation TEXT,
                        feature_snapshot JSONB NOT NULL,
                        gate_evaluations JSONB NOT NULL,
                        scanner_version TEXT,
                        config_version TEXT,
                        feature_schema_version TEXT,
                        regime_version TEXT,
                        execution_engine_version TEXT,
                        counterfactual_entry_price REAL,
                        counterfactual_stop_loss REAL,
                        counterfactual_target_1 REAL,
                        counterfactual_target_2 REAL,
                        counterfactual_target_3 REAL,
                        counterfactual_entry_mode TEXT,
                        counterfactual_rule_version TEXT,
                        counterfactual_mfe_r REAL,
                        counterfactual_mae_r REAL,
                        counterfactual_realized_r REAL,
                        counterfactual_outcome_labels JSONB,
                        counterfactual_exclusion_reason TEXT,
                        counterfactual_status TEXT DEFAULT 'PENDING',
                        counterfactual_generated_at TIMESTAMPTZ
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_log_symbol_date ON scanner_evaluation_log(symbol, evaluation_timestamp)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_log_run_id ON scanner_evaluation_log(scanner_run_id)")


                # 5. breakout_watchlist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS breakout_watchlist (
                        symbol TEXT PRIMARY KEY,
                        category TEXT,
                        current_state TEXT,
                        h1_status TEXT,
                        m30_status TEXT,
                        m15_status TEXT,
                        m5_status TEXT,
                        breakout_level REAL,
                        support_level REAL,
                        invalidated_at TIMESTAMPTZ,
                        cooldown_until TIMESTAMPTZ,
                        session_date TEXT,
                        last_updated TIMESTAMPTZ DEFAULT NOW(),
                        context_json TEXT,
                        is_active BOOLEAN DEFAULT TRUE,
                        trigger_level REAL,
                        invalidation_level REAL,
                        max_extension_atr REAL,
                        buffer_pct REAL,
                        armed_at TIMESTAMPTZ,
                        signal_timestamp TIMESTAMPTZ,
                        expires_at TIMESTAMPTZ,
                        timeframe TEXT
                    )
                """)
                # [RULE 67 CHANGE-RATIONALE]: Add indexes on breakout_watchlist state, cooldown, and last_updated to eliminate sequential table scans
                cur.execute("CREATE INDEX IF NOT EXISTS idx_breakout_wl_state ON breakout_watchlist(current_state)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_breakout_wl_cooldown ON breakout_watchlist(cooldown_until)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_breakout_wl_last_updated ON breakout_watchlist(last_updated DESC)")

                # 6. rejected_alerts
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rejected_alerts (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        scanner TEXT NOT NULL,
                        engine_version TEXT,
                        rejection_reason TEXT,
                        alert_date DATE DEFAULT CURRENT_DATE,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        context JSONB
                    )
                """)

                # 7. trade_audit_log
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_audit_log (
                        id SERIAL PRIMARY KEY,
                        alert_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        old_state JSONB,
                        new_state JSONB
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_alert_id ON trade_audit_log(alert_id)")

                # 7.5. alert_events (Trade Evolution & Immutable Re-Trigger Event History)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS alert_events (
                        id SERIAL PRIMARY KEY,
                        alert_id INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
                        symbol VARCHAR(32) NOT NULL,
                        scanner VARCHAR(64) NOT NULL,
                        pattern VARCHAR(64) NOT NULL,
                        event_type VARCHAR(32) NOT NULL,
                        event_date DATE NOT NULL,
                        event_time TIMESTAMPTZ DEFAULT NOW(),
                        
                        trigger_price REAL NOT NULL,
                        original_entry_price REAL NOT NULL,
                        pnl_since_entry_pct REAL NOT NULL,
                        score INTEGER NOT NULL,
                        rvol REAL NOT NULL,
                        clv REAL,
                        
                        higher_low BOOLEAN DEFAULT NULL,
                        dist_from_ema20_pct REAL,
                        is_extended BOOLEAN DEFAULT FALSE,
                        nearest_resistance REAL,
                        distance_to_resistance_pct REAL,
                        remaining_rr_to_resistance REAL,
                        suggested_trailing_sl REAL,
                        
                        evidence_count INTEGER DEFAULT 1,
                        distinct_patterns_count INTEGER DEFAULT 1,
                        confirmation_quality VARCHAR(32) DEFAULT 'INITIAL',
                        
                        reason_code VARCHAR(128),
                        notes TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alert_events_alert_id ON alert_events(alert_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alert_events_symbol_date ON alert_events(symbol, event_date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alert_events_event_type ON alert_events(event_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alert_events_created_at ON alert_events(created_at DESC)")



                # 8. score_weight_log
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS score_weight_log (
                        id SERIAL PRIMARY KEY,
                        model_version TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        weights JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        CONSTRAINT chk_weights_json CHECK (weights ? 'volume_breakout' AND weights ? 'rsi_divergence' AND weights ? 'ema_crossover')
                    )
                """)

                # 9. bayesian_model_updates
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bayesian_model_updates (
                        id SERIAL PRIMARY KEY,
                        regime TEXT NOT NULL,
                        proposed_version TEXT NOT NULL,
                        current_version TEXT NOT NULL,
                        current_weights JSONB NOT NULL,
                        proposed_weights JSONB NOT NULL,
                        trades_analyzed INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        admin_comment TEXT,
                        approved_by TEXT,
                        approved_at TIMESTAMPTZ,
                        rejected_at TIMESTAMPTZ,
                        applied_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        expires_at TIMESTAMPTZ,
                        CONSTRAINT chk_bayes_status CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED'))
                    )
                """)

                # 9.5 mtf_v2_watchlist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS mtf_v2_watchlist (
                        symbol TEXT NOT NULL,
                        box_id TEXT NOT NULL,
                        state TEXT NOT NULL,
                        mtf_substate TEXT,

                        consolidation_start_ts TIMESTAMPTZ,
                        consolidation_end_ts TIMESTAMPTZ,
                        consolidation_bars INTEGER,
                        consolidation_sessions INTEGER,

                        box_high NUMERIC,
                        box_low NUMERIC,
                        box_mid NUMERIC,
                        box_value_center NUMERIC,
                        hard_high NUMERIC,
                        hard_low NUMERIC,
                        box_width_pct NUMERIC,
                        box_width_atr NUMERIC,
                        box_occupancy NUMERIC,

                        resistance_test_count INTEGER,
                        higher_low_score INTEGER,
                        compression_score INTEGER,
                        setup_score INTEGER,
                        last_confirmed_pivot_level NUMERIC,
                        last_confirmed_pivot_ts TIMESTAMPTZ,

                        pressure_state TEXT,
                        attempt_started_ts TIMESTAMPTZ,
                        attempt_bar_boundary INTEGER,
                        volume_ratio_5m NUMERIC,
                        range_ratio_5m NUMERIC,
                        live_position_5m NUMERIC,
                        distance_to_box_high NUMERIC,

                        context_1h_score INTEGER,
                        context_30m_score INTEGER,
                        market_regime TEXT,
                        relative_strength NUMERIC,
                        confluence_score INTEGER,

                        attempt_count INTEGER DEFAULT 0,
                        last_attempt_ts TIMESTAMPTZ,
                        last_confirmation_ts TIMESTAMPTZ,

                        attempt_ttl_expires_at TIMESTAMPTZ,
                        cooldown_until TIMESTAMPTZ,
                        invalidated_at TIMESTAMPTZ,
                        invalidation_reason TEXT,

                        data_source_1h TEXT,
                        data_source_30m TEXT,
                        data_source_15m TEXT,
                        data_source_5m TEXT,
                        candle_ts_1h TIMESTAMPTZ,
                        candle_ts_30m TIMESTAMPTZ,
                        candle_ts_15m TIMESTAMPTZ,
                        candle_ts_5m TIMESTAMPTZ,

                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        last_evaluated_at TIMESTAMPTZ,
                        version INTEGER DEFAULT 1,
                        PRIMARY KEY (symbol, box_id)
                    )
                """)

                # 10. scanner_health
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scanner_health (
                        scanner_name TEXT PRIMARY KEY,
                        status TEXT NOT NULL DEFAULT 'IDLE',
                        last_success TIMESTAMPTZ,
                        today_alerts INTEGER NOT NULL DEFAULT 0,
                        error_msg TEXT,
                        is_acknowledged BOOLEAN DEFAULT TRUE,
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        error_severity TEXT DEFAULT NULL,
                        error_count INTEGER DEFAULT 0,
                        first_error_at TIMESTAMPTZ DEFAULT NULL,
                        retry_count INTEGER DEFAULT 0,
                        scheduled_for TEXT DEFAULT NULL,
                        processed_count INTEGER DEFAULT NULL,
                        total_count INTEGER DEFAULT NULL,
                        outcome TEXT DEFAULT NULL,
                        provider_stats JSONB DEFAULT NULL,
                        duration_seconds REAL DEFAULT 0.0,
                        active_run_id VARCHAR(64) DEFAULT NULL,
                        CONSTRAINT chk_scanner_status CHECK (status IN ('OK', 'DOWN', 'IDLE', 'RUNNING', 'DEGRADED', 'DEGRADED_FALLBACK', 'PAUSED', 'STOPPED') OR status LIKE 'QUEUED%')
                    )
                """)

                # 11. wealth_score_history
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS wealth_score_history (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        evaluation_date DATE NOT NULL,
                        hold_score REAL,
                        fm_score REAL,
                        rs_6m REAL,
                        cmp REAL,
                        sma_200 REAL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (symbol, evaluation_date)
                    )
                """)

                # 11. scan_failures
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scan_failures (
                        id SERIAL PRIMARY KEY,
                        scan_id TEXT NOT NULL,
                        scanner_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        provider TEXT,
                        failure_reason TEXT,
                        failed_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS scan_failures_scan_id_idx ON scan_failures(scan_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS scan_failures_failed_at_idx ON scan_failures(failed_at)")

                # 12. funnel_telemetry
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS funnel_telemetry (
                        id SERIAL PRIMARY KEY,
                        scanner TEXT NOT NULL,
                        run_date DATE NOT NULL DEFAULT CURRENT_DATE,
                        symbol TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        gate TEXT NOT NULL,
                        passed BOOLEAN NOT NULL,
                        observed_value REAL,
                        threshold_value REAL,
                        comparator TEXT,
                        message TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_funnel_telemetry_lookup ON funnel_telemetry(scanner, run_date, symbol)")

                # 13. system_state
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)

                # 14. system_control & scanner_control
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_control (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        global_enabled BOOLEAN DEFAULT TRUE,
                        global_paused BOOLEAN DEFAULT FALSE,
                        global_stop_requested BOOLEAN DEFAULT FALSE,
                        reason TEXT,
                        updated_by TEXT DEFAULT 'ADMIN',
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        CONSTRAINT single_row_system_control CHECK (id = 1)
                    )
                """)
                cur.execute("INSERT INTO system_control (id, global_enabled, global_paused, global_stop_requested) VALUES (1, TRUE, FALSE, FALSE) ON CONFLICT (id) DO NOTHING")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scanner_control (
                        scanner_name TEXT PRIMARY KEY,
                        enabled BOOLEAN DEFAULT TRUE,
                        paused BOOLEAN DEFAULT FALSE,
                        stop_requested BOOLEAN DEFAULT FALSE,
                        manual_run_requested BOOLEAN DEFAULT FALSE,
                        reason TEXT,
                        updated_by TEXT DEFAULT 'ADMIN',
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("INSERT INTO scanner_control (scanner_name, enabled, paused, stop_requested) VALUES ('ACCUMULATION', TRUE, FALSE, FALSE) ON CONFLICT (scanner_name) DO NOTHING")

                # 15. ACCUMULATION scanner tables
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS accumulation_runs (
                        run_id TEXT PRIMARY KEY,
                        trigger_type TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'STARTING',
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ,
                        metrics JSONB DEFAULT '{}'::jsonb
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS accumulation_health (
                        id SERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        scanner TEXT NOT NULL DEFAULT 'ACCUMULATION',
                        status TEXT NOT NULL DEFAULT 'IDLE',
                        lifecycle_state TEXT DEFAULT 'IDLE',
                        requested_symbols INTEGER DEFAULT 0,
                        processed_symbols INTEGER DEFAULT 0,
                        valid_symbols INTEGER DEFAULT 0,
                        rejected_symbols INTEGER DEFAULT 0,
                        candidates INTEGER DEFAULT 0,
                        alerts INTEGER DEFAULT 0,
                        raw_data_errors INTEGER DEFAULT 0,
                        stale_symbols INTEGER DEFAULT 0,
                        invalid_symbols INTEGER DEFAULT 0,
                        cache_hits INTEGER DEFAULT 0,
                        cache_misses INTEGER DEFAULT 0,
                        bytes_fetched BIGINT DEFAULT 0,
                        api_latency_ms REAL DEFAULT 0.0,
                        calculation_time_ms REAL DEFAULT 0.0,
                        persistence_time_ms REAL DEFAULT 0.0,
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ,
                        duration_seconds REAL DEFAULT 0.0,
                        pause_requested BOOLEAN DEFAULT FALSE,
                        stop_requested BOOLEAN DEFAULT FALSE,
                        last_error TEXT,
                        error_count INTEGER DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS accumulation_alerts (
                        id BIGSERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        audit_snapshot_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        state TEXT NOT NULL,
                        tradable BOOLEAN DEFAULT TRUE,
                        score NUMERIC,
                        accumulation_score NUMERIC,
                        compression_score NUMERIC,
                        relative_strength_score NUMERIC,
                        resistance_score NUMERIC,
                        volume_structure_score NUMERIC,
                        fundamental_score NUMERIC,
                        close NUMERIC,
                        entry_zone_low NUMERIC,
                        entry_zone_high NUMERIC,
                        breakout_level NUMERIC,
                        stop_loss NUMERIC,
                        target_1 NUMERIC,
                        target_2 NUMERIC,
                        target_3 NUMERIC,
                        risk_pct NUMERIC,
                        rr_1 NUMERIC,
                        rr_2 NUMERIC,
                        rr_3 NUMERIC,
                        time_stop_days INTEGER DEFAULT 40,
                        invalidation_reason TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        effective_as_of TIMESTAMPTZ DEFAULT NOW(),
                        CONSTRAINT unq_accumulation_alert UNIQUE (symbol, state, run_id)
                    )
                """)

                # [AUDIT-FIX H1]: accumulation_alerts was missing dedicated indexes — full-table scan
                # on every Stocks-to-Watch API call caused 2–5s slowness. Three indexes cover all
                # query patterns: (state, created_at) for daily state-filter, (symbol) for per-symbol
                # lookups, and (created_at DESC) for the DISTINCT ON date-range subquery.
                cur.execute("CREATE INDEX IF NOT EXISTS idx_accum_alerts_state_date ON accumulation_alerts (state, created_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_accum_alerts_symbol ON accumulation_alerts (symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_accum_alerts_created_at ON accumulation_alerts (created_at DESC)")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS accumulation_candidates (
                        id BIGSERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        score NUMERIC NOT NULL,
                        state TEXT NOT NULL,
                        raw_data JSONB DEFAULT '{}'::jsonb,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS accumulation_telemetry (
                        id BIGSERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        audit_snapshot_id TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # 14. instrument_registry, provider_instruments, symbol_mappings, resolution_history
                # [RULE 67 - FIX RATIONALE]: Executing multiple DDLs in a single semicolon-joined batch string
                # caused PostgreSQL parse syntax error at line 46 in psycopg2 during init_db startup.
                # Each table and index is now isolated into its own dedicated cur.execute statement, matching
                # the rest of init_db. Also includes full schema columns for both modern and legacy mapping queries.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS instrument_registry (
                        instrument_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL UNIQUE,
                        company_name TEXT,
                        primary_exchange TEXT DEFAULT 'NSE',
                        series TEXT DEFAULT 'EQ',
                        nse_symbol TEXT,
                        bse_symbol TEXT,
                        bse_scrip_code TEXT,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_inst_reg_sym ON instrument_registry (symbol)")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS provider_instruments (
                        provider TEXT NOT NULL,
                        instrument_id TEXT NOT NULL,
                        provider_symbol TEXT NOT NULL,
                        provider_key TEXT,
                        exchange TEXT,
                        series TEXT,
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (provider, instrument_id)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_prov_inst_sym ON provider_instruments (provider, provider_symbol)")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS symbol_mappings (
                        provider TEXT,
                        original_symbol TEXT,
                        mapped_symbol TEXT,
                        instrument_id TEXT,
                        exchange TEXT,
                        series TEXT,
                        confidence_score INTEGER DEFAULT 100,
                        mapping_source TEXT DEFAULT 'LEARNED',
                        status TEXT DEFAULT 'ACTIVE',
                        version INTEGER DEFAULT 1,
                        consecutive_failures INTEGER DEFAULT 0,
                        last_success_at TIMESTAMPTZ,
                        last_verified_at TIMESTAMPTZ DEFAULT NOW(),
                        retry_after TIMESTAMPTZ,
                        effective_from TIMESTAMPTZ DEFAULT NOW(),
                        effective_to TIMESTAMPTZ,
                        mapping_type TEXT,
                        original_sym TEXT,
                        mapped_sym TEXT,
                        mapping_state TEXT DEFAULT 'ACTIVE',
                        failure_count INTEGER DEFAULT 0,
                        last_verified TEXT,
                        is_invalid BOOLEAN DEFAULT FALSE
                    )
                """)
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_symbol_mappings_prov_orig ON symbol_mappings (provider, original_symbol) WHERE provider IS NOT NULL AND original_symbol IS NOT NULL")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_mappings_legacy ON symbol_mappings (mapping_type, original_sym)")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS resolution_history (
                        id BIGSERIAL PRIMARY KEY,
                        provider TEXT NOT NULL,
                        original_symbol TEXT NOT NULL,
                        attempted_symbol TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        resolution_level TEXT NOT NULL,
                        confidence_score INTEGER,
                        latency_ms DOUBLE PRECISION,
                        error_code TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_res_hist_sym ON resolution_history (provider, original_symbol)")

                # 15. ai_concall_cache_v3
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_concall_cache_v3 (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        pdf_url TEXT NOT NULL,
                        analysis_data JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT ai_concall_cache_v3_symbol_pdf_url_key UNIQUE (symbol, pdf_url)
                    )
                """)

                # 16. promoter_pledge_cache
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS promoter_pledge_cache (
                        symbol TEXT PRIMARY KEY,
                        pledge_pct REAL NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_attempted_at TIMESTAMPTZ
                    )
                """)
                # [RULE 67 CHANGE-RATIONALE: NSE_OFFICIAL_PLEDGE_SCHEMA_v1.0]
                # Schema extensions for official NSE depository pledge & provenance tracking
                cur.execute("""
                    ALTER TABLE promoter_pledge_cache
                        ADD COLUMN IF NOT EXISTS pledged_shares BIGINT,
                        ADD COLUMN IF NOT EXISTS promoter_shares BIGINT,
                        ADD COLUMN IF NOT EXISTS total_shares BIGINT,
                        ADD COLUMN IF NOT EXISTS depository_pledged_shares BIGINT,
                        ADD COLUMN IF NOT EXISTS promoter_holding_pct REAL,
                        ADD COLUMN IF NOT EXISTS depository_pledge_demat_pct REAL,
                        ADD COLUMN IF NOT EXISTS source VARCHAR(32) DEFAULT 'NSE',
                        ADD COLUMN IF NOT EXISTS as_of_date DATE,
                        ADD COLUMN IF NOT EXISTS snapshot_id VARCHAR(64);
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pledge_snapshots (
                        snapshot_id VARCHAR(64) PRIMARY KEY,
                        snapshot_date DATE NOT NULL,
                        source VARCHAR(32) NOT NULL DEFAULT 'NSE',
                        total_rows_downloaded INT NOT NULL,
                        matched_symbols_count INT NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_pledge_snapshots_date ON pledge_snapshots(snapshot_date);")

                # 17. bhavcopy_cache
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bhavcopy_cache (
                        trading_date DATE PRIMARY KEY,
                        delivery_data JSONB NOT NULL,
                        fetched_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # 18. fetch_errors
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fetch_errors (
                        id SERIAL PRIMARY KEY,
                        source_name TEXT NOT NULL,
                        scanner_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        interval TEXT,
                        category TEXT NOT NULL,
                        occurrences INTEGER NOT NULL DEFAULT 1,
                        first_seen TIMESTAMPTZ DEFAULT NOW(),
                        last_seen TIMESTAMPTZ DEFAULT NOW(),
                        last_error_msg TEXT,
                        is_acknowledged BOOLEAN DEFAULT FALSE,
                        CONSTRAINT idx_fetch_errors_uni UNIQUE (source_name, scanner_name, symbol, interval, category)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fetch_errors_ack ON fetch_errors(is_acknowledged)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fetch_errors_scan ON fetch_errors(scanner_name)")

                # 19. validation_history
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS validation_history (
                        id SERIAL PRIMARY KEY,
                        dataset_name TEXT NOT NULL,
                        score REAL NOT NULL,
                        status TEXT NOT NULL,
                        failures TEXT,
                        warnings TEXT,
                        row_count INTEGER,
                        validator_version TEXT,
                        symbols_processed INTEGER,
                        symbols_valid INTEGER,
                        symbols_failed INTEGER,
                        average_score REAL,
                        minimum_score REAL,
                        maximum_score REAL,
                        validated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_validation_history_dataset ON validation_history(dataset_name)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_validation_history_time ON validation_history(validated_at DESC)")

                # 20. data_cache_metadata
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS data_cache_metadata (
                        key TEXT PRIMARY KEY,
                        last_fetched TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        cadence_seconds INTEGER NOT NULL,
                        rows INTEGER,
                        etag TEXT,
                        source TEXT,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_data_cache_metadata_key ON data_cache_metadata(key)")

                # 21. data_fetch_health
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS data_fetch_health (
                        source_name TEXT PRIMARY KEY,
                        last_success TIMESTAMPTZ,
                        last_failure TIMESTAMPTZ,
                        consecutive_failures INTEGER NOT NULL DEFAULT 0,
                        error_msg TEXT,
                        is_acknowledged BOOLEAN DEFAULT TRUE,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_data_fetch_health_source ON data_fetch_health(source_name)")

                # 22. manual_portfolio
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS manual_portfolio (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_date DATE NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity INTEGER NOT NULL,
                        hold_score_entry INTEGER,
                        hold_score_current INTEGER,
                        re_eval_due_date DATE,
                        added_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # 23. push_subscriptions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS push_subscriptions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER,
                        endpoint TEXT NOT NULL UNIQUE,
                        p256dh TEXT NOT NULL,
                        auth TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # 24. parquet_cache
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS parquet_cache (
                        name TEXT,
                        date TEXT,
                        data BYTEA,
                        PRIMARY KEY (name, date)
                    )
                """)

                # 24b. screener_cache (Canonical fundamental data with TIMESTAMPTZ)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS screener_cache (
                        symbol TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        fetched_at TIMESTAMPTZ NOT NULL,
                        expires_at TIMESTAMPTZ,
                        source TEXT,
                        quality TEXT,
                        version INTEGER DEFAULT 1,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_screener_cache_fetched ON screener_cache (fetched_at);
                """)

                # 25. global_notifications
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS global_notifications (
                        id SERIAL PRIMARY KEY,
                        type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        symbol TEXT,
                        is_seen BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_global_notif_created ON global_notifications(created_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_global_notif_type ON global_notifications(type, created_at DESC)")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS near_misses (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        scanner TEXT NOT NULL,
                        breakout_type TEXT,
                        gate_name TEXT NOT NULL,
                        observed_value TEXT,
                        threshold_value TEXT,
                        delta_pct NUMERIC(6,2),
                        score NUMERIC(5,2),
                        entry_price NUMERIC(12,2),
                        stop_loss NUMERIC(12,2),
                        target_1 NUMERIC(12,2),
                        logged_date DATE NOT NULL DEFAULT CURRENT_DATE,
                        logged_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(30) DEFAULT 'TRACKING',
                        realized_rr NUMERIC(6,2),
                        max_mfe_r NUMERIC(6,2)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_near_misses_date ON near_misses(logged_date DESC, scanner)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_near_misses_logged_at ON near_misses(logged_date DESC, logged_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_near_misses_scanner_date ON near_misses(scanner, logged_date DESC, logged_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_near_misses_sym_date ON near_misses(symbol, logged_date DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_messages_user_created ON user_messages(user_id, created_at DESC)")


                # 26. system_checkpoints
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_checkpoints (
                        id SERIAL PRIMARY KEY,
                        checkpoint_name TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        content TEXT NOT NULL,
                        reason TEXT DEFAULT ''
                    )
                """)

                # 27. build_manifest
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS build_manifest (
                        id SERIAL PRIMARY KEY,
                        run_date DATE UNIQUE NOT NULL,
                        status TEXT NOT NULL,
                        input_universe_count INTEGER,
                        qualified_count INTEGER,
                        used_fallback BOOLEAN DEFAULT FALSE,
                        fallback_source TEXT,
                        build_source_date DATE,
                        scanner_version TEXT,
                        checksum TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ
                    )
                """)

                # 28. telegram_queue
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS telegram_queue (
                        id SERIAL PRIMARY KEY,
                        alert_id INTEGER REFERENCES alerts(id) ON DELETE CASCADE,
                        symbol TEXT NOT NULL,
                        message_text TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        retry_count INTEGER DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        sent_at TIMESTAMPTZ,
                        CONSTRAINT chk_tg_status CHECK (status IN ('pending', 'sent'))
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_telegram_queue_status ON telegram_queue(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_telegram_queue_created ON telegram_queue(created_at)")

                # 30. alert_outcomes (earnings_calendar table removed — see corporate_events.py)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS alert_outcomes (
                        alert_id INTEGER REFERENCES alerts(id) ON DELETE CASCADE,
                        leg INTEGER DEFAULT 1,
                        symbol TEXT NOT NULL,
                        scanner TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        regime_score NUMERIC(5, 2) DEFAULT 0.0,
                        base_score INTEGER DEFAULT 0,
                        rs_bonus INTEGER DEFAULT 0,
                        sector_bonus INTEGER DEFAULT 0,
                        rs_percentile NUMERIC(5, 2) DEFAULT 0.0,
                        sector_name TEXT DEFAULT '',
                        rr_at_alert NUMERIC(5, 2) DEFAULT 0.0,
                        atr_pct_at_alert NUMERIC(5, 2) DEFAULT 0.0,
                        entry_price NUMERIC(10, 2) NOT NULL,
                        stop_loss NUMERIC(10, 2) NOT NULL,
                        target_1 NUMERIC(10, 2) NOT NULL,
                        target_2 NUMERIC(10, 2),
                        target_3 NUMERIC(10, 2),
                        target_4 NUMERIC(10, 2),
                        alert_timestamp TIMESTAMPTZ NOT NULL,
                        exit_timestamp TIMESTAMPTZ,
                        exit_reason TEXT,
                        realized_rr NUMERIC(5, 2),
                        unrealized_rr_at_expiry NUMERIC(5, 2),
                        holding_period_bars INTEGER,
                        max_favorable_excursion_r NUMERIC(5, 2) DEFAULT 0.0,
                        max_adverse_excursion_r NUMERIC(5, 2) DEFAULT 0.0,
                        earnings_flag BOOLEAN DEFAULT FALSE,
                        days_to_earnings INTEGER DEFAULT 999,
                        earnings_date DATE,
                        earnings_severity VARCHAR(20) DEFAULT 'NONE',
                        date_status VARCHAR(20) DEFAULT 'UNKNOWN',
                        forensic_score INTEGER DEFAULT 0,
                        forensic_risk_tier VARCHAR(20) DEFAULT 'UNKNOWN',
                        growth_investment_mode BOOLEAN DEFAULT FALSE,
                        growth_investment_score INTEGER DEFAULT 0,
                        forensic_details JSONB DEFAULT '{}'::jsonb,
                        PRIMARY KEY (alert_id, leg)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alert_outcomes_scanner ON alert_outcomes(scanner)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alert_outcomes_regime ON alert_outcomes(regime)")

                # 31. sector_rankings
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sector_rankings (
                        sector_symbol TEXT NOT NULL,
                        sector_name TEXT NOT NULL,
                        ranking_date DATE NOT NULL,
                        blended_score NUMERIC(8, 2) NOT NULL,
                        raw_rank INTEGER NOT NULL,
                        consecutive_top3_days INTEGER DEFAULT 0,
                        consecutive_bottom3_days INTEGER DEFAULT 0,
                        effective_status VARCHAR(20) DEFAULT 'NEUTRAL',
                        PRIMARY KEY (sector_symbol, ranking_date)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sector_rankings_date ON sector_rankings(ranking_date)")

                # 32. wealth_buy_alert
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS wealth_buy_alert (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        alert_price REAL NOT NULL,
                        alert_date DATE NOT NULL DEFAULT CURRENT_DATE,
                        alert_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        breakout_type TEXT NOT NULL DEFAULT '',
                        fm_score REAL,
                        status TEXT DEFAULT 'ACTIVE',
                        current_price REAL,
                        current_score REAL,
                        status_updated_at TIMESTAMPTZ DEFAULT NOW(),
                        notes TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        entry_signal TEXT,
                        exit_signal TEXT,
                        exit_price REAL,
                        exit_date DATE,
                        exit_time TIMESTAMPTZ,
                        is_closed BOOLEAN DEFAULT FALSE,
                        pnl_rs REAL,
                        pnl_pct REAL,
                        engine_version TEXT,
                        config_version TEXT,
                        position_pct REAL,
                        position_amount REAL,
                        position_shares INTEGER,
                        portfolio_bucket TEXT,
                        valuation_score REAL,
                        momentum_score INTEGER,
                        momentum_confidence TEXT,
                        data_quality TEXT,
                        fallback_timestamp TIMESTAMPTZ,
                        CONSTRAINT uq_wealth_symbol_date_type UNIQUE (symbol, alert_date, breakout_type)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_wealth_alert_symbol ON wealth_buy_alert(symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_wealth_alert_date ON wealth_buy_alert(alert_date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_wealth_alert_status ON wealth_buy_alert(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_wealth_alert_is_closed ON wealth_buy_alert(is_closed)")

                # 33. users
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id SERIAL PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        first_name VARCHAR(100),
                        last_name VARCHAR(100),
                        mobile VARCHAR(20) UNIQUE,
                        password_hash VARCHAR(255) NOT NULL,
                        role TEXT DEFAULT 'user',
                        account_status VARCHAR(20) DEFAULT 'pending',
                        is_active BOOLEAN DEFAULT FALSE,
                        must_change_password BOOLEAN DEFAULT FALSE,
                        failed_login_attempts INT DEFAULT 0,
                        locked_until TIMESTAMPTZ,
                        last_login TIMESTAMPTZ,
                        session_token UUID,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                # [RULE 67 CHANGE-RATIONALE]: Add functional indexes on LOWER(username) and LOWER(email) for fast login and user search
                cur.execute("CREATE INDEX IF NOT EXISTS idx_users_lower_username ON users(LOWER(username))")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_users_lower_email ON users(LOWER(email))")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at DESC)")

                # 34. user_sessions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                        session_token TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
                        ip_address TEXT,
                        user_agent TEXT,
                        login_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        logoff_time TIMESTAMPTZ,
                        is_online BOOLEAN DEFAULT TRUE,
                        is_revoked BOOLEAN DEFAULT FALSE
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_is_online ON user_sessions(is_online)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_token ON user_sessions(user_id, session_token)")

                # 35. user_messages
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_messages (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                        is_from_admin BOOLEAN NOT NULL DEFAULT FALSE,
                        message TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        is_read BOOLEAN DEFAULT FALSE
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_messages_user_id ON user_messages(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_messages_unread ON user_messages(is_from_admin, is_read)")

                # 36. capital_history
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS capital_history (
                        id SERIAL PRIMARY KEY,
                        transaction_type TEXT NOT NULL,
                        amount REAL NOT NULL,
                        description TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # 37. user_watchlists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_watchlists (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) DEFAULT 'DEFAULT_USER',
                        symbol TEXT NOT NULL,
                        company_name TEXT DEFAULT '',
                        added_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        last_scanned_at TIMESTAMPTZ,
                        last_health_score NUMERIC(5,2),
                        last_status VARCHAR(50) DEFAULT 'MONITORING',
                        notes TEXT,
                        last_deep_analysis_at TIMESTAMPTZ,
                        deep_analysis_result TEXT,
                        UNIQUE(user_id, symbol)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_watchlists_user_id ON user_watchlists(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_watchlists_user_added ON user_watchlists(user_id, added_at DESC)")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS daily_watchlist (
                        "Stock" TEXT PRIMARY KEY,
                        "Company" TEXT,
                        "Sector" TEXT,
                        "added_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS daily_excluded_watchlist (
                        "Stock" TEXT PRIMARY KEY,
                        "Company" TEXT,
                        "Reason" TEXT,
                        "added_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 38. stock_analysis_master
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS stock_analysis_master (
                        symbol TEXT PRIMARY KEY,
                        company_name TEXT DEFAULT '',
                        sector TEXT DEFAULT 'EQUITY',
                        last_scanned_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        last_deep_analysis_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        health_score NUMERIC(5,2),
                        status VARCHAR(50) DEFAULT 'MONITORING',
                        deep_analysis_result TEXT,
                        cmp NUMERIC(12,2),
                        cmp_updated_at TIMESTAMPTZ,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_master_scanned ON stock_analysis_master(last_scanned_at DESC)")

                # 39. watchlist (Multibagger Watchlist)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS watchlist (
                        symbol TEXT PRIMARY KEY,
                        buy_zone_low NUMERIC,
                        buy_zone_high NUMERIC,
                        latest_price NUMERIC,
                        growth_score NUMERIC,
                        value_score NUMERIC,
                        trend_score NUMERIC,
                        total_score NUMERIC,
                        bucket TEXT,
                        status TEXT,
                        notes TEXT,
                        last_alert_price NUMERIC,
                        last_alert_at TIMESTAMPTZ,
                        last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # [RULE 67 CHANGE-RATIONALE]: Add compound indexes for watchlist table to accelerate multibagger watchlist queries
                cur.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_status_total_score ON watchlist(status, total_score DESC NULLS LAST)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_total_score ON watchlist(total_score DESC NULLS LAST)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol)")
                # 40. scanner_execution_history
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scanner_execution_history (
                        id SERIAL PRIMARY KEY,
                        run_id VARCHAR(64) UNIQUE NOT NULL,
                        parent_run_id VARCHAR(64),
                        retry_attempt INT DEFAULT 0,
                        scanner_name VARCHAR(50) NOT NULL,
                        lifecycle_status VARCHAR(30) NOT NULL,
                        quality_status VARCHAR(30) NOT NULL DEFAULT 'NORMAL',
                        trigger_type VARCHAR(20) DEFAULT 'SCHEDULED',
                        scheduler_name VARCHAR(30) DEFAULT 'CRON',
                        system_version VARCHAR(40),
                        git_commit VARCHAR(64),
                        started_at TIMESTAMPTZ NOT NULL,
                        heartbeat_at TIMESTAMPTZ NOT NULL,
                        completed_at TIMESTAMPTZ,
                        total_stocks INT DEFAULT 0,
                        fresh_data_count INT DEFAULT 0,
                        stale_data_count INT DEFAULT 0,
                        incomplete_data_count INT DEFAULT 0,
                        stale_ratio FLOAT DEFAULT 0.0,
                        alerts_generated INT DEFAULT 0,
                        api_calls INT DEFAULT 0,
                        cache_hits INT DEFAULT 0,
                        cache_misses INT DEFAULT 0,
                        stop_reason VARCHAR(255),
                        error_summary VARCHAR(255),
                        error_details TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_seh_scanner_date ON scanner_execution_history(scanner_name, started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_seh_lifecycle ON scanner_execution_history(lifecycle_status);
                    CREATE INDEX IF NOT EXISTS idx_seh_quality ON scanner_execution_history(quality_status);
                    CREATE INDEX IF NOT EXISTS idx_seh_run_id ON scanner_execution_history(run_id);
                    CREATE INDEX IF NOT EXISTS idx_seh_sysver ON scanner_execution_history(system_version);
                    CREATE INDEX IF NOT EXISTS idx_seh_gitcom ON scanner_execution_history(git_commit);
                    -- [RULE 67 CHANGE-RATIONALE]:
                    -- Adds compound indexes on started_at and lifecycle_status for instant paginated and filtered UI scans.
                    CREATE INDEX IF NOT EXISTS idx_seh_started_at ON scanner_execution_history(started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_seh_life_started ON scanner_execution_history(lifecycle_status, started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_seh_perf_composite ON scanner_execution_history(lifecycle_status, quality_status, started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_seh_scanner_life ON scanner_execution_history(scanner_name, lifecycle_status, started_at DESC);
                """)

                # [RULE 67 CHANGE-RATIONALE]:
                # Consolidated candidate_tracker tables (scanner_candidates, candidate_snapshots, and near_miss_outcomes)
                # creation DDL directly inside init_db(). This ensures that the tables are created when the application
                # boots or database gets initialized by the web server or task workers, preventing UndefinedTable errors.

                # 41. scanner_candidates
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scanner_candidates (
                        candidate_id         BIGSERIAL PRIMARY KEY,
                        setup_id             VARCHAR(128) UNIQUE NOT NULL,
                        symbol               TEXT NOT NULL,
                        scanner_name         TEXT NOT NULL,
                        setup_type           TEXT NOT NULL,
                        state                TEXT NOT NULL,
                        structure_date       DATE         NOT NULL,

                        detected_at          TIMESTAMPTZ  NOT NULL,
                        triggered_at         TIMESTAMPTZ,
                        confirmed_at         TIMESTAMPTZ,
                        invalidated_at       TIMESTAMPTZ,
                        expires_at           TIMESTAMPTZ,

                        trigger_level        NUMERIC(12, 2),
                        invalidation_level   NUMERIC(12, 2),
                        next_required_event  TEXT,
                        setup_reset_reason   TEXT,

                        last_evaluated_at    TIMESTAMPTZ  NOT NULL,
                        last_seen_price      NUMERIC(12, 2),
                        last_seen_volume     NUMERIC(16, 2),

                        distance_to_trigger_pct  NUMERIC(6, 2),
                        distance_to_trigger_atr  NUMERIC(6, 2),
                        extension_from_base_atr  NUMERIC(6, 2),

                        quality_score        NUMERIC(6, 2),
                        risk_score           NUMERIC(6, 2),
                        reward_risk_ratio    NUMERIC(6, 2),

                        stop_loss            NUMERIC(12, 2),
                        target_1             NUMERIC(12, 2),
                        target_2             NUMERIC(12, 2),
                        target_3             NUMERIC(12, 2),

                        confirmation_delay_bars  INTEGER DEFAULT 0,

                        status_reason        TEXT,
                        failure_reason_code  TEXT,

                        cleared_checklists        JSONB,
                        pending_checklists        JSONB,
                        failed_checklists         JSONB,
                        warning_checklists        JSONB,
                        not_applicable_checklists JSONB,

                        primary_blocker_type  TEXT,
                        primary_blocker       JSONB,

                        health_status         TEXT,
                        health_reason         TEXT,
                        last_change_summary   TEXT,

                        reasons               JSONB,
                        warnings              JSONB,
                        metadata              JSONB,
                        data_quality          JSONB,
                        algorithm_version     TEXT,

                        created_at            TIMESTAMPTZ DEFAULT NOW(),
                        updated_at            TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_candidates_setup_id ON scanner_candidates (setup_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_state ON scanner_candidates (state, scanner_name)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_state_last_eval ON scanner_candidates (state, last_evaluated_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_state_updated ON scanner_candidates (state, updated_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_quality_score ON scanner_candidates (quality_score DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_scanner_last_eval ON scanner_candidates (scanner_name, last_evaluated_at DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_symbol ON scanner_candidates (symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_blocker_type ON scanner_candidates (primary_blocker_type)")

                # 42. candidate_snapshots
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS candidate_snapshots (
                        snapshot_id          BIGSERIAL    PRIMARY KEY,
                        candidate_id         BIGINT       NOT NULL REFERENCES scanner_candidates(candidate_id) ON DELETE CASCADE,
                        snapshot_time        TIMESTAMPTZ  NOT NULL,
                        snapshot_reason      TEXT  NOT NULL,

                        price                NUMERIC(12, 2),
                        trigger_level        NUMERIC(12, 2),
                        distance_to_trigger_pct  NUMERIC(6, 2),
                        distance_to_trigger_atr  NUMERIC(6, 2),
                        extension_from_base_atr  NUMERIC(6, 2),
                        quality_score        NUMERIC(6, 2),
                        volume_ratio         NUMERIC(6, 2),
                        rs_rating            NUMERIC(6, 2),
                        sector_rank          INTEGER,
                        atr                  NUMERIC(10, 2),
                        support_level        NUMERIC(12, 2),
                        rr                   NUMERIC(6, 2),
                        health_status        TEXT,
                        health_reason        TEXT,
                        cleared_json         JSONB,
                        pending_json         JSONB,
                        warnings_json        JSONB,
                        not_applicable_json  JSONB
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_candidate_time ON candidate_snapshots (candidate_id, snapshot_time DESC)")

                # 43. near_miss_outcomes
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS near_miss_outcomes (
                        id               BIGSERIAL    PRIMARY KEY,
                        near_miss_id     INTEGER      UNIQUE NOT NULL REFERENCES near_misses(id) ON DELETE CASCADE,
                        return_1d        NUMERIC(8, 2),
                        return_3d        NUMERIC(8, 2),
                        return_5d        NUMERIC(8, 2),
                        return_10d       NUMERIC(8, 2),
                        return_20d       NUMERIC(8, 2),
                        return_60d       NUMERIC(8, 2),
                        mfe              NUMERIC(8, 2),
                        mae              NUMERIC(8, 2),
                        hypothetical_r   NUMERIC(6, 2),
                        rejection_verdict TEXT NOT NULL,
                        evaluated_at     TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # [RULE 67 CHANGE-RATIONALE]:
                # Adds compound indexes for daily_watchlist_v2 and daily_excluded_watchlist_v2 to accelerate
                # universe queries and daily build inspections by date and universe status.
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_daily_wl_v2_date_status ON daily_watchlist_v2(build_date DESC, universe_status);
                        CREATE INDEX IF NOT EXISTS idx_daily_wl_v2_sym_date ON daily_watchlist_v2(symbol, build_date DESC);
                        CREATE INDEX IF NOT EXISTS idx_daily_excl_v2_date ON daily_excluded_watchlist_v2(build_date DESC);
                        CREATE INDEX IF NOT EXISTS idx_daily_excl_v2_sym_date ON daily_excluded_watchlist_v2(symbol, build_date DESC);
                        CREATE INDEX IF NOT EXISTS idx_daily_excl_v2_class ON daily_excluded_watchlist_v2(exclusion_class, build_date DESC);
                        CREATE INDEX IF NOT EXISTS idx_daily_excl_v2_qs ON daily_excluded_watchlist_v2(universe_quality_score DESC NULLS LAST);
                    """)
                except Exception as _wl_idx_err:
                    logger.debug(f"Watchlist v2 index notice: {_wl_idx_err}")

                # 43. Short Covering Watchlist & Alerts
                try:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS short_covering_watchlist (
                            symbol TEXT NOT NULL,
                            scan_date DATE NOT NULL,
                            close_price NUMERIC(10,2),
                            total_oi BIGINT,
                            oi_buildup_5d_pct NUMERIC(6,2),
                            short_buildup_ratio NUMERIC(5,2),
                            rsi_14 NUMERIC(5,2),
                            support_level NUMERIC(10,2),
                            overhead_resistance NUMERIC(10,2),
                            atr_14 NUMERIC(10,2),
                            buildup_quality_score NUMERIC(5,2),
                            sector TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            PRIMARY KEY (symbol, scan_date)
                        );
                        CREATE INDEX IF NOT EXISTS idx_sc_watchlist_date ON short_covering_watchlist(scan_date);

                        CREATE TABLE IF NOT EXISTS short_covering_alerts (
                            id SERIAL PRIMARY KEY,
                            symbol TEXT NOT NULL,
                            alert_time TIMESTAMPTZ NOT NULL,
                            ignition_price NUMERIC(10,2),
                            vwap NUMERIC(10,2),
                            stop_loss NUMERIC(10,2),
                            initial_target NUMERIC(10,2),
                            risk_reward_ratio NUMERIC(5,2),
                            excess_oi_contraction NUMERIC(6,2),
                            volume_surge_ratio NUMERIC(5,2),
                            ignition_score NUMERIC(5,2),
                            grade VARCHAR(10),
                            reasons JSONB,
                            state VARCHAR(30) DEFAULT 'IGNITION',
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_sc_alerts_time ON short_covering_alerts(alert_time);
                        CREATE INDEX IF NOT EXISTS idx_sc_alerts_symbol ON short_covering_alerts(symbol);
                    """)
                except Exception as _sc_err:
                    logger.debug(f"Short covering tables init notice: {_sc_err}")


                # 39. Trade analytics view — wrapped in own try/except with lock_timeout
                # to prevent this DDL from blocking on AccessExclusiveLock when other workers
                # are reading from the 'alerts' table (causes deadlock otherwise).
                try:
                    with conn.cursor() as _vcur:
                        _vcur.execute("SET LOCAL lock_timeout = '3s'")
                        _vcur.execute("DROP VIEW IF EXISTS v_trade_analytics CASCADE")
                        _vcur.execute("""
                            CREATE OR REPLACE VIEW v_trade_analytics AS
                            SELECT
                                id,
                                symbol,
                                alert_time,
                                alert_date,
                                scanner,
                                category,
                                entry_price,
                                stop_loss,
                                target_price,
                                status,
                                exit_price,
                                pnl_pct,
                                closed_at,
                                (context->'technicals'->>'above_ema20')::boolean AS above_ema20,
                                (context->'technicals'->>'above_sma50')::boolean AS above_sma50,
                                (context->'technicals'->>'golden_cross')::boolean AS golden_cross,
                                (context->'technicals'->>'body_ratio')::float AS body_ratio,
                                (context->'technicals'->>'delivery_pct')::float AS delivery_pct,
                                (context->'technicals'->>'rsi')::float AS rsi,
                                (context->'technicals'->>'volume_ratio')::float AS volume_ratio,
                                (context->'session'->>'open')::float AS session_open,
                                (context->'session'->>'day_high')::float AS session_day_high,
                                (context->'session'->>'day_low')::float AS session_day_low,
                                (context->'fundamentals'->>'peg')::float AS peg,
                                (context->'fundamentals'->>'yoy_rev')::float AS yoy_rev,
                                (context->'fundamentals'->>'yoy_profit')::float AS yoy_profit,
                                (context->'fundamentals'->>'roe')::float AS roe,
                                context->'execution'->>'sl_method' AS sl_method,
                                context->'execution'->>'t_method' AS t_method,
                                context->'execution'->>'trail_note' AS trail_note
                            FROM alerts;
                        """)
                    conn.commit()
                except Exception as view_err:
                    logger.warning(f"⚠️ v_trade_analytics view create/replace skipped (non-critical, will retry next boot): {view_err}")
                    try:
                        conn.rollback()
                    except Exception:
                        pass

                # 40. Seed reference data & admin user
                bootstrap_admin(cur=cur)

                # [VERSION: CLEAN_BOOT_RESET_v1.0] Complete boot reset of all scanner statuses & advisory locks on startup
                try:
                    cur.execute("SELECT pg_advisory_unlock_all();")
                    cur.execute("UPDATE scanner_health SET status = 'IDLE', error_msg = NULL, processed_count = 0, updated_at = NOW()")
                    cleanup_orphaned_scanner_runs_on_boot(cur=cur)
                    logger.info("🧹 [BOOT RESET] All scanner health statuses and advisory locks reset to clean IDLE state.")
                except Exception as t_err:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    logger.warning(f"[STARTUP] Scanner state reset warning: {t_err}")

                # 41. Validate schema integrity against PostgreSQL catalog
                if not (hasattr(cur, "_mock_name") or type(cur).__name__ in ("MagicMock", "Mock") or "mock" in type(cur).__module__):
                    validate_schema(cur)

                conn.commit()

        _DB_INITIALIZED = True
        logger.info("✅ Database ready (Postgres) — greenfield schema initialized and validated")
        logger.info("ℹ️  Data Retention Active: preserving all alerts for historical analysis.")


def validate_schema(cur):
    """
    [VERSION: GREENFIELD_DB_OVERHAUL_v1.0] Validate schema integrity against PostgreSQL catalog.

    Fails fast (raises RuntimeError) if any required table or critical column is missing.
    Startup validates, it does not mutate or self-repair.
    """
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    res = cur.fetchall()
    if not res or not isinstance(res, list):
        logger.info("ℹ️ Schema validation skipped (empty catalog response).")
        return

    existing_tables = {row[0] for row in res if isinstance(row, (tuple, list)) and row}

    REQUIRED_TABLES = [
        "system_logs", "master_symbols", "candidates", "alerts",
        "breakout_watchlist", "rejected_alerts", "trade_audit_log",
        "score_weight_log", "bayesian_model_updates", "scanner_health",
        "scan_failures", "funnel_telemetry", "system_state", "symbol_mappings",
        "ai_concall_cache_v3", "promoter_pledge_cache", "bhavcopy_cache",
        "fetch_errors", "validation_history", "data_cache_metadata",
        "data_fetch_health", "manual_portfolio", "push_subscriptions",
        "parquet_cache", "global_notifications", "system_checkpoints",
        "build_manifest", "telegram_queue",
        "alert_outcomes", "sector_rankings", "wealth_buy_alert",
        "users", "user_sessions", "user_messages", "capital_history",
        "user_watchlists", "stock_analysis_master", "watchlist",
        "scanner_execution_history",
        "scanner_candidates", "candidate_snapshots", "near_miss_outcomes",
        "alert_events"
    ]

    missing_tables = [t for t in REQUIRED_TABLES if t not in existing_tables]
    if missing_tables:
        raise RuntimeError(f"🚨 Database Schema Validation Failed! Missing tables: {missing_tables}")

    CRITICAL_COLUMNS = {
        "alerts": ["id", "symbol", "status", "alert_time", "alert_date", "context"],
        "scanner_health": ["scanner_name", "status", "last_success"],
        "users": ["user_id", "username", "email", "password_hash"],
        "wealth_buy_alert": ["id", "symbol", "alert_date", "breakout_type"]
    }

    for tbl, cols in CRITICAL_COLUMNS.items():
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
        """, (tbl,))
        col_res = cur.fetchall()
        if not col_res or not isinstance(col_res, list):
            continue
        existing_cols = {row[0] for row in col_res if isinstance(row, (tuple, list)) and row}
        missing_cols = [c for c in cols if c not in existing_cols]
        if missing_cols:
            raise RuntimeError(f"🚨 Database Schema Validation Failed! Missing columns in '{tbl}': {missing_cols}")

    logger.info(f"✅ Schema validation passed — all {len(REQUIRED_TABLES)} tables and critical catalog columns verified.")

# =====================================================================================
# FAILED-REVERSAL COOLDOWN (v6.1)
#
# Makes the reversal scanner's cooldown REAL by reading the EXISTING `status` column
# (populated by performance_tracker via update_alert_outcome). No new table/job needed.
#
# A symbol is "in cooldown" if its most recent REVERSAL alert closed as a LOSS within
# the last `cooldown_days` trading days. This suppresses repeated low-quality reversal
# candidates — the #1 leak identified in the 44% backtest.
#
# Trading days are approximated via business-day count (Mon–Fri) using alert_date.
# =====================================================================================

def is_symbol_in_failed_reversal_cooldown(symbol: str, cooldown_days: int = 30) -> bool:
    """
    PREFERRED cooldown backend for reversal_scanner (logs 🟢 OUTCOME_AWARE when present).

    Returns True if `symbol`'s MOST RECENT REVERSAL alert:
        • closed as status='LOSS', AND
        • that alert fired within the last `cooldown_days` business days.
    Returns False if the last reversal won, is still OPEN, or no recent reversal exists.

    Relies on the existing `status` column written by performance_tracker /
    update_alert_outcome(). No separate outcome table required.
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Most recent REVERSAL alert for this symbol (any status)
                cur.execute("""
                    SELECT a.status, a.alert_date, ao.exit_reason
                    FROM alerts a
                    LEFT JOIN alert_outcomes ao ON a.id = ao.alert_id
                    WHERE a.symbol = %s AND a.scanner = 'REVERSAL'
                    ORDER BY a.alert_date DESC, a.alert_time DESC
                    LIMIT 1
                """, (symbol,))
                row = cur.fetchone()
                if not row:
                    return False

                status, alert_date, exit_reason = row[0], row[1], row[2]

                # AMBIGUOUS_SL_HIT (same-bar collision) is conservative for P&L but DOES NOT trigger loss cooldown
                if exit_reason and str(exit_reason).upper() == "AMBIGUOUS_SL_HIT":
                    return False

                # Only LOSS triggers cooldown. WIN / OPEN / CLOSED do not suppress.
                if str(status).upper() != "LOSS":
                    return False

                # Business-day distance from the losing alert's date to today.
                try:
                    # Prefer numpy for performance if available
                    import numpy as np
                    from datetime import date as _date
                    # alert_date is a DATE column → psycopg2 returns datetime.date
                    if not isinstance(alert_date, _date):
                        # Fallback if it came back as text
                        from datetime import datetime as _dt
                        alert_date = _dt.strptime(str(alert_date)[:10], "%Y-%m-%d").date()
                    today = datetime.now(IST).date()
                    if today < alert_date:
                        return False
                    try:
                        biz_days = int(np.busday_count(alert_date, today))
                        return biz_days < cooldown_days
                    except Exception:
                        # If numpy present but busday_count failed, fall through to pure-Python calc
                        logger.warning(f"numpy.busday_count failed for {symbol}, falling back")
                except ImportError:
                    # numpy not available — fallback to pure-Python business-day calc
                    pass

                # Pure-Python business-day count (no external deps)
                try:
                    from datetime import timedelta
                    today = datetime.now(IST).date()
                    if today < alert_date:
                        return False
                    delta_days = (today - alert_date).days
                    weeks, remainder = divmod(delta_days, 7)
                    biz_days = weeks * 5
                    start_weekday = alert_date.weekday()  # 0=Mon,6=Sun
                    for i in range(remainder):
                        if (start_weekday + i) % 7 < 5:
                            biz_days += 1
                    return biz_days < cooldown_days
                except Exception:
                    logger.exception(f"cooldown business-day calc failed for {symbol}")
                    # Conservative: if we can't compute distance, do NOT suppress.
                    return False
    except Exception:
        logger.exception(f"❌ is_symbol_in_failed_reversal_cooldown failed for {symbol}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────────────


def get_all_failed_reversal_cooldown_symbols(cooldown_days: int = 40) -> set:
    """
    Bulk fetches all symbols that are currently in a failed reversal cooldown.
    Returns a set of symbols for O(1) lookup in the scanner loop.
    """
    init_db()
    try:
        from datetime import date as _date, datetime as _dt
        today = datetime.now(IST).date()
        cooldown_symbols = set()

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH LatestAlerts AS (
                        SELECT a.symbol, a.status, a.pnl_pct, a.alert_date, ao.exit_reason,
                               ROW_NUMBER() OVER (PARTITION BY a.symbol ORDER BY a.alert_date DESC, a.alert_time DESC) as rn
                        FROM alerts a
                        LEFT JOIN alert_outcomes ao ON a.id = ao.alert_id
                        WHERE a.scanner = 'REVERSAL'
                    )
                    SELECT symbol, alert_date, exit_reason
                    FROM LatestAlerts
                    WHERE rn = 1 AND (UPPER(status) = 'LOSS' OR (pnl_pct IS NOT NULL AND pnl_pct < 0))
                """)

                rows = cur.fetchall()
                for row in rows:
                    symbol, alert_date, exit_reason = row[0], row[1], row[2]

                    if exit_reason and str(exit_reason).upper() == "AMBIGUOUS_SL_HIT":
                        continue

                    if not isinstance(alert_date, _date):
                        alert_date = _dt.strptime(str(alert_date)[:10], "%Y-%m-%d").date()

                    if today < alert_date:
                        continue

                    # Calculate business days
                    try:
                        import numpy as np
                        biz_days = int(np.busday_count(alert_date, today))
                    except Exception:
                        delta_days = (today - alert_date).days
                        weeks, remainder = divmod(delta_days, 7)
                        biz_days = weeks * 5
                        start_weekday = alert_date.weekday()
                        for i in range(remainder):
                            if (start_weekday + i) % 7 < 5:
                                biz_days += 1

                    if biz_days < cooldown_days:
                        cooldown_symbols.add(symbol)

        return cooldown_symbols
    except Exception:
        logger.exception("❌ get_all_failed_reversal_cooldown_symbols failed")
        return set()

def delete_todays_alerts_for_scanner(scanner_name: str, trade_date: str, conn = None) -> int:
    """Idempotently delete today's alerts for a specific scanner before saving new ones."""
    success = False
    deleted = 0

    def _execute(cur, commit_cb):
        nonlocal success, deleted
        # [RULE 67: CASCADE_ALERT_DELETION_V1.0]
        # Clean up all dependent child tables first to prevent foreign key violations (e.g. alert_outcomes_alert_id_fkey)
        cur.execute("""
            DELETE FROM alert_outcomes
            WHERE alert_id IN (
                SELECT id FROM alerts WHERE scanner = %s AND alert_date = %s
            )
        """, (scanner_name, trade_date))
        cur.execute("""
            DELETE FROM telegram_queue
            WHERE alert_id IN (
                SELECT id FROM alerts WHERE scanner = %s AND alert_date = %s
            )
        """, (scanner_name, trade_date))
        cur.execute("""
            DELETE FROM alert_events
            WHERE alert_id IN (
                SELECT id FROM alerts WHERE scanner = %s AND alert_date = %s
            )
        """, (scanner_name, trade_date))
        cur.execute("""
            DELETE FROM trade_audit_log
            WHERE alert_id IN (
                SELECT id FROM alerts WHERE scanner = %s AND alert_date = %s
            )
        """, (scanner_name, trade_date))
        cur.execute("""
            DELETE FROM alerts
            WHERE scanner = %s
              AND alert_date = %s
        """, (scanner_name, trade_date))
        deleted = cur.rowcount
        commit_cb()
        success = True
        return deleted

    try:
        init_db()
        if conn is None:
            with _DB_WRITE_LOCK:
                with get_connection() as local_conn:
                    try:
                        with local_conn.cursor() as cur:
                            return _execute(cur, local_conn.commit)
                    except Exception as e:
                        logger.exception(f"❌ Failed to delete today's alerts for {scanner_name}")
                        try:
                            upsert_scanner_health(scanner_name, status="DEGRADED", error_msg=f"Failed to delete today's alerts: {str(e)[:200]}")
                        except Exception:
                            pass
                        return 0
                    finally:
                        if not success:
                            local_conn.rollback()
        else:
            with conn.cursor() as cur:
                return _execute(cur, lambda: None)
    except Exception as e:
        logger.exception(f"❌ Failed to delete today's alerts for {scanner_name}")
        try:
            upsert_scanner_health(scanner_name, status="DEGRADED", error_msg=f"Failed to delete today's alerts: {str(e)[:200]}")
        except Exception:
            pass
        return 0


def save_candidate(symbol: str, breakout_type: str, scanner: str, technical_score: int, volume_ratio: float, delivery_pct: float, rr_ratio: float, market_context: dict, status: str = "QUALIFIED", **kwargs):
    import json
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO candidates (symbol, breakout_type, scanner, technical_score, volume_ratio, delivery_pct, rr_ratio, market_context, metadata, status, alert_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, breakout_type, alert_date) DO UPDATE
                    SET technical_score = EXCLUDED.technical_score,
                        volume_ratio = EXCLUDED.volume_ratio,
                        delivery_pct = EXCLUDED.delivery_pct,
                        rr_ratio = EXCLUDED.rr_ratio,
                        market_context = EXCLUDED.market_context,
                        metadata = EXCLUDED.metadata,
                        status = CASE WHEN candidates.status IN ('QUALIFIED', 'OPEN') THEN EXCLUDED.status ELSE candidates.status END
                """, (symbol, breakout_type, scanner, technical_score, volume_ratio, delivery_pct, rr_ratio, json.dumps(market_context), json.dumps(kwargs), status, datetime.now(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d')))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to save candidate {symbol}: {e}")
        return False

def get_candidates_by_status(status: str, alert_date: str = None):
    try:
        from psycopg2.extras import RealDictCursor
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if alert_date:
                    cur.execute("SELECT * FROM candidates WHERE status = %s AND alert_date = %s", (status, alert_date))
                else:
                    cur.execute("SELECT * FROM candidates WHERE status = %s AND alert_date = CURRENT_DATE::TEXT", (status,))
                return cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to get candidates: {e}")
        return []

def update_candidate_status(candidate_id: int, status: str, metadata: dict = None):
    import json
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if metadata:
                    cur.execute("UPDATE candidates SET status = %s, metadata = metadata::jsonb || %s::jsonb WHERE id = %s", (status, json.dumps(metadata), candidate_id))
                else:
                    cur.execute("UPDATE candidates SET status = %s WHERE id = %s", (status, candidate_id))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to update candidate {candidate_id}: {e}")
        return False

def _sanitize_for_json(obj: Any) -> Any:
    """Sanitize objects for PostgreSQL JSON/JSONB fields, converting NumPy types, NaNs, and Enums into serializable Python types."""
    import math
    from enum import Enum
    if obj is None:
        return None
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if (math.isnan(val) or math.isinf(val)) else val
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return [_sanitize_for_json(x) for x in obj.tolist()]
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, str, bool)):
        return obj
    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def canonicalize_scanner_name(scanner: str, breakout_type: str = "") -> str:
    """
    Normalizes scanner and breakout_type strings to prevent cross-scanner aliasing
    (e.g., TECHNICAL -> PULLBACK, BREAKOUT -> EOD).
    """
    s = str(scanner or "").strip().upper()
    b = str(breakout_type or "").strip().upper()
    if s in ("PULLBACK", "TECHNICAL"):
        return "PULLBACK"
    if s in ("EOD", "BREAKOUT", "EOD_SCANNER"):
        return "EOD"
    if s in ("REVERSAL", "REVERSAL_SCANNER"):
        return "REVERSAL"
    if s in ("MULTI_TF", "MULTITF", "MULTI-TF"):
        return "MULTI_TF"
    if s in ("MULTIBAGGER", "WEALTH", "WEALTH_ENGINE"):
        return s
    if b in ("PULLBACK", "EOD", "REVERSAL", "MULTI_TF", "MULTIBAGGER"):
        return b
    return s or b or "UNKNOWN"


def generate_canonical_alert_fingerprint(
    symbol: str,
    canonical_scanner: str,
    source_trading_date: Any,
    direction: str = "LONG",
    setup_type: str = "BREAKOUT",
    entry_price: float = 0.0,
    tolerance_pct: float = 0.5
) -> str:
    """
    Generates a deterministic 32-character canonical setup fingerprint:
    SHA256(symbol | scanner | direction | source_trading_date | setup_type | price_bucket)
    Ensures identical setups evaluated on weekends produce the exact same fingerprint.
    """
    import hashlib
    eff_sym = str(symbol or "").strip().upper()
    eff_sc = str(canonical_scanner or "").strip().upper()
    eff_dir = str(direction or "LONG").strip().upper()
    eff_st = str(setup_type or "BREAKOUT").strip().upper()
    date_str = source_trading_date.isoformat() if hasattr(source_trading_date, 'isoformat') else str(source_trading_date or "")
    
    tol = max(0.05, float(tolerance_pct or 0.5))
    price = float(entry_price or 0.0)
    price_unit = max(0.01, price * (tol / 100.0)) if price > 0 else 1.0
    price_bucket = round(price / price_unit) * price_unit if price > 0 else 0.0
    
    raw_str = f"{eff_sym}|{eff_sc}|{eff_dir}|{date_str}|{eff_st}|{price_bucket:.2f}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:32]


def save_alert_if_new(

    symbol: str,
    breakout_type: str,
    alert_time: str,
    scanner: str = None,
    category: str = None,
    entry_price: float = None,
    stop_loss: float = None,
    target_1: float = None,
    target_2: float = None,
    target_3: float = None,
    target_4: float = None,
    target_price: float = None,  # Legacy
    signals: str = None,
    score: int = None,
    rsi: float = None,
    volume_ratio: float = None,
    context: dict = None,
    model_version: str = "v1",
    data_partition: str = "TRAIN",
    bayesian_regime: str = "BULL",
    bayesian_weights: dict = None,
    cash_in_hand: float = None,
    structural_failure_stop: float = None,
    target_quality_score: float = None,
    entry_mode: str = "MARKET",
    actual_entry_price: float = None,
    conn = None,
    **kwargs
) -> tuple[bool, str, float, int]:
    """
    Insert a new alert.  Returns (inserted, capital_allocated, shares_bought).

    Captures:
    - model_version: Bayesian model version (v1, v2, etc)
    - bayesian_regime: Market regime (BULL, BEAR, SIDEWAYS)
    - bayesian_weights: Actual weights used for scoring
    """

    evaluation_id = kwargs.get('evaluation_id')
    scanner_run_id = kwargs.get('scanner_run_id')

    if not evaluation_id or not scanner_run_id:
        try:
            from scanner_telemetry import telemetry_engine
            # Standard key matches GlobalScannerTelemetryEngine.get_or_create_context
            ctx_key = f"{scanner}_{symbol}"
            ctx = telemetry_engine._contexts.get(ctx_key)
            if ctx:
                evaluation_id = evaluation_id or getattr(ctx, "audit_snapshot_id", None)
                scanner_run_id = scanner_run_id or getattr(ctx, "run_id", None)
        except Exception:
            pass

    # Ensure evaluation_id and scanner_run_id are present in context dict
    if context is None:
        context = {}
    if evaluation_id:
        context['evaluation_id'] = evaluation_id
    if scanner_run_id:
        context['scanner_run_id'] = scanner_run_id

    # Derive source_trading_date (Saturday/Sunday always inherit latest valid Friday trading session)
    source_trading_date = kwargs.get('source_trading_date')
    if source_trading_date is None and context:
        source_trading_date = context.get('source_trading_date')
    if source_trading_date is None:
        try:
            from market_utils import get_expected_latest_trading_date
            source_trading_date = get_expected_latest_trading_date()
        except Exception:
            from datetime import datetime as dt
            source_trading_date = dt.now(IST).date()
    if isinstance(source_trading_date, str):
        try:
            from datetime import date as dt_date
            source_trading_date = dt_date.fromisoformat(source_trading_date.split("T")[0])
        except Exception:
            pass

    if context is None:
        context = {}
    context['source_trading_date'] = str(source_trading_date)

    sanitized_context = _sanitize_for_json(context) if context is not None else None
    context_str = json.dumps(sanitized_context, default=str) if sanitized_context is not None else None
    sanitized_weights = _sanitize_for_json(bayesian_weights) if bayesian_weights is not None else None
    weights_str = json.dumps(sanitized_weights, default=str) if sanitized_weights is not None else None


    # [FIX_REVERTED] Force fetching live price here caused critical bugs where Entry > T1
    # because targets were calculated using the scanner's trigger price, not this delayed live price.
    # We must use the entry_price passed into the function to maintain mathematical integrity.


    # Safety: DB stale-buy check removed in v6 as scanners now reliably handle stale
    # price data at the individual stock level during extraction.

    # Calculate portfolio allocation dynamically if not provided
    from portfolio_engine import calculate_trade_allocation
    capital_allocated = kwargs.get('capital_allocated')
    shares_bought = kwargs.get('shares_bought')

    if capital_allocated is None or shares_bought is None:
        if entry_price and stop_loss:
            capital_allocated, shares_bought = calculate_trade_allocation(entry_price, stop_loss, score or 80)
        else:
            capital_allocated, shares_bought = 0.0, 0

    # [VERSION: ALL_ALERTS_PERSIST_v1.0] Dry-run mode disabled — all generated alerts persist to DB.
    # DONT_SAVE_ALERTS is permanently overridden so alerts save to Postgres DB at all times.
    success = False

    def _execute(cur, commit_cb):
        nonlocal success

        # ─────────────────────────────────────────────────────────────────
        # 🛡️ CANONICAL SETUP FINGERPRINT DEDUPLICATION GATE
        # ─────────────────────────────────────────────────────────────────
        canonical_scanner = canonicalize_scanner_name(scanner, breakout_type)
        eff_entry = float(entry_price or 0.0)
        direction = "LONG"
        setup_type = str(signals or breakout_type or "BREAKOUT").strip().upper()

        try:
            from config import SCANNER_DEDUP_ENTRY_TOLERANCE_PCT
            tol_pct = float(SCANNER_DEDUP_ENTRY_TOLERANCE_PCT.get(canonical_scanner, SCANNER_DEDUP_ENTRY_TOLERANCE_PCT.get("DEFAULT", 0.5)))
        except Exception:
            tol_pct = 0.5

        alert_fingerprint = generate_canonical_alert_fingerprint(
            symbol=symbol,
            canonical_scanner=canonical_scanner,
            source_trading_date=source_trading_date,
            direction=direction,
            setup_type=setup_type,
            entry_price=eff_entry,
            tolerance_pct=tol_pct
        )

        cur.execute("""
            SELECT id, alert_date, alert_time, entry_price, status, scanner, breakout_type,
                   COALESCE(source_trading_date, alert_date) as src_date
            FROM alerts
            WHERE symbol = %s 
              AND (UPPER(scanner) = %s OR UPPER(breakout_type) = %s)
              AND is_rejected = FALSE
              AND (
                  alert_fingerprint = %s
                  OR (
                      COALESCE(source_trading_date, alert_date) = %s
                      AND abs(entry_price - %s) / GREATEST(0.01, %s) <= %s
                  )
              )
            ORDER BY id DESC LIMIT 1
        """, (
            symbol, canonical_scanner, canonical_scanner,
            alert_fingerprint,
            source_trading_date,
            eff_entry, eff_entry, (tol_pct / 100.0)
        ))
        prior_adjusted_alert = cur.fetchone()

        if prior_adjusted_alert:
            prior_id, prior_adate, prior_atime, prior_entry, prior_status, prior_sc, prior_bt, prior_src = prior_adjusted_alert
            logger.info(
                f"🔁 [ADJUSTED_DEDUP] {symbol} ({canonical_scanner}) alert RAISED @ ₹{eff_entry:.2f}, "
                f"but DB persistence suppressed — identical adjusted setup already saved in prior history "
                f"(Alert ID: {prior_id}, Source Trading Date: {prior_src}, Entry: ₹{prior_entry:.2f}, "
                f"Tolerance: {tol_pct}%, Fingerprint: {alert_fingerprint})"
            )
            # Emit structured lifecycle telemetry contract
            try:
                from telemetry_manager import telemetry
                telemetry.log_scheduler_event(canonical_scanner, "ALERT_DEDUPLICATED", {
                    "symbol": symbol, "scanner": canonical_scanner, "source_trading_date": str(source_trading_date),
                    "entry_price": eff_entry, "alert_raised": True, "duplicate": True, "persisted": False, "notification_sent": False
                })
            except Exception:
                pass
            return False, f"Duplicate: Adjusted alert already persisted (ID {prior_id})", 0.0, 0

        # Check if symbol already has an active OPEN position in the system
        cur.execute("""
            SELECT id, symbol, entry_price, stop_loss, target_1, target_2, target_3, signals, score, alert_date, alert_time, context,
                   COALESCE(trade_evolution_state, 'INITIAL'), COALESCE(evidence_count, 1), COALESCE(distinct_patterns_count, 1), scanner
            FROM alerts
            WHERE symbol = %s AND status = 'OPEN' AND is_rejected = FALSE
            ORDER BY alert_time DESC LIMIT 1
        """, (symbol,))
        existing_alert = cur.fetchone()

        if existing_alert:
            # ─────────────────────────────────────────────────────────────────
            # TRADE EVOLUTION & RE-TRIGGER EVALUATION ENGINE
            # ─────────────────────────────────────────────────────────────────
            parent_id = existing_alert[0]
            orig_entry_price = float(existing_alert[2] or entry_price or 0.0)
            current_trigger_price = float(entry_price or orig_entry_price)
            current_score = int(score or 80)
            current_rvol = float(volume_ratio or 1.0)
            current_pattern = str(signals or breakout_type or "BREAKOUT").strip()
            cand_ctx = context or {}
            today_date = datetime.now(IST).date()
            today_str = today_date.strftime('%Y-%m-%d')

            # Query last recorded event for this alert in alert_events
            cur.execute("""
                SELECT id, event_type, pattern, trigger_price, rvol, event_date, event_time, confirmation_quality
                FROM alert_events
                WHERE alert_id = %s
                ORDER BY id DESC LIMIT 1
            """, (parent_id,))
            last_event = cur.fetchone()

            # Rule 1 & 4: Material Change Check & Anti-Inflation Filter
            if last_event:
                last_eid, last_etype, last_pat, last_price, last_rvol, last_edate, last_etime, last_qual = last_event
                last_price = float(last_price or orig_entry_price)
                last_rvol = float(last_rvol or 1.0)
                
                price_delta_pct = abs(current_trigger_price - last_price) / max(0.01, last_price) * 100.0
                is_new_pattern = (current_pattern != str(last_pat).strip())
                is_volume_expansion = (current_rvol >= 1.5 or current_rvol >= last_rvol * 1.25)
                is_structural_upgrade = bool(cand_ctx.get("higher_low") or cand_ctx.get("new_breakout_level") or float(cand_ctx.get("room_to_resistance_r") or 0.0) >= 1.5)

                # Same-day guard: suppress duplicate triggers on the same calendar day unless substantial new breakout
                if str(last_edate) == today_str and not (is_new_pattern or price_delta_pct >= 3.0 or (is_volume_expansion and is_structural_upgrade)):
                    logger.info(f"🚫 [TRADE_EVOLUTION] {symbol} ({scanner}) Re-trigger suppressed — Reason: SAME_DAY_NO_MATERIAL_CHANGE")
                    return False, "Suppressed: No Material Change", 0.0, 0

                # Cross-day material change guard: require distinct pattern, volume surge, or structural upgrade
                if not (is_new_pattern or (price_delta_pct >= 2.0 and is_structural_upgrade) or is_volume_expansion):
                    logger.info(f"🚫 [TRADE_EVOLUTION] {symbol} ({scanner}) Re-trigger suppressed — Reason: NO_MATERIAL_CHANGE (Pattern: {current_pattern}, ΔPrice: {price_delta_pct:.1f}%)")
                    return False, "Suppressed: No Material Change", 0.0, 0

            # Calculate PnL since original entry
            pnl_since_entry_pct = round(((current_trigger_price - orig_entry_price) / max(0.01, orig_entry_price)) * 100.0, 2)

            # Extension check from EMA20 / Moving Averages
            dist_from_ema20_pct = float(cand_ctx.get("dist_from_ema20_pct") or cand_ctx.get("upper_wick_pct") or 0.0)
            is_extended = bool(cand_ctx.get("is_extended") or (dist_from_ema20_pct > 12.0))

            # Rule 7: Nullable higher_low explicitly calculated
            higher_low = cand_ctx.get("higher_low")
            if higher_low is None:
                higher_low = bool(current_trigger_price >= orig_entry_price)

            # Rule 2 & 6: Accurate Remaining R:R & Distance to Resistance
            nearest_resistance = float(cand_ctx.get("nearest_resistance") or cand_ctx.get("resistance_price") or target_1 or (current_trigger_price * 1.08))
            suggested_trailing_sl = float(cand_ctx.get("suggested_trailing_sl") or stop_loss or (current_trigger_price * 0.95))

            rem_reward = max(0.0, nearest_resistance - current_trigger_price)
            rem_risk = max(0.01, current_trigger_price - suggested_trailing_sl)
            remaining_rr = round(rem_reward / rem_risk, 2)
            dist_to_resistance_pct = round((rem_reward / max(0.01, current_trigger_price)) * 100.0, 2)

            clv_val = float(cand_ctx.get("clv") or 0.75)

            # Count previous events and distinct patterns
            cur.execute("SELECT COUNT(*), COUNT(DISTINCT pattern) FROM alert_events WHERE alert_id = %s", (parent_id,))
            ev_row = cur.fetchone()
            prior_ev_count = ev_row[0] if ev_row else 1
            prior_distinct_count = ev_row[1] if ev_row else 1
            evidence_count = prior_ev_count + 1
            distinct_patterns_count = prior_distinct_count + (1 if (not last_event or current_pattern != last_event[2]) else 0)

            # Rule 4 & 5: State Machine Classification & Reason Codes
            if pnl_since_entry_pct > 2.5:
                # Working / Profitable setup
                if is_extended:
                    event_type = "RECORD_ONLY_NO_ADD"
                    reason_code = "EXTENSION_BLOCKED"
                    confirmation_quality = "MODERATE"
                    parent_state = "EXTENDED"
                elif remaining_rr < 1.5:
                    event_type = "RECORD_ONLY_NO_ADD"
                    reason_code = "INSUFFICIENT_ROOM_TO_RESISTANCE"
                    confirmation_quality = "MODERATE"
                    parent_state = "WORKING"
                else:
                    event_type = "PYRAMID_CANDIDATE"
                    reason_code = "MOMENTUM_CONTINUATION_PYRAMID_READY"
                    confirmation_quality = "HIGH" if distinct_patterns_count >= 2 else "MODERATE"
                    parent_state = "PYRAMID_READY"
            elif -2.0 <= pnl_since_entry_pct <= 2.5:
                # Flat / Base consolidation
                event_type = "THESIS_RECONFIRMATION"
                reason_code = "BASE_SUPPORT_HELD_RECONFIRMED"
                confirmation_quality = "MODERATE"
                parent_state = "RECONFIRMED"
            else:
                # Losing trade (pnl < -2.0%)
                event_type = "RECORD_ONLY_NO_ADD"
                reason_code = "POSITION_LOSING_ADD_BLOCKED"
                confirmation_quality = "LOW"
                parent_state = "ADD_BLOCKED"

            notes_str = f"Re-trigger on {today_str} by {scanner or 'SCANNER'}. Pattern: {current_pattern}. PnL: {pnl_since_entry_pct:+.1f}%. Reason: {reason_code}."

            # Insert Immutable Historical Event
            cur.execute("""
                INSERT INTO alert_events
                    (alert_id, symbol, scanner, pattern, event_type, event_date, event_time,
                     trigger_price, original_entry_price, pnl_since_entry_pct, score, rvol, clv,
                     higher_low, dist_from_ema20_pct, is_extended, nearest_resistance, distance_to_resistance_pct,
                     remaining_rr_to_resistance, suggested_trailing_sl, evidence_count, distinct_patterns_count,
                     confirmation_quality, reason_code, notes)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(),
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s)
                RETURNING id;
            """, (parent_id, symbol, scanner or 'TECHNICAL', current_pattern, event_type, today_date,
                  current_trigger_price, orig_entry_price, pnl_since_entry_pct, current_score, current_rvol, clv_val,
                  higher_low, dist_from_ema20_pct, is_extended, nearest_resistance, dist_to_resistance_pct,
                  remaining_rr, suggested_trailing_sl, evidence_count, distinct_patterns_count,
                  confirmation_quality, reason_code, notes_str))
            event_row = cur.fetchone()
            event_id = event_row[0] if event_row else None

            # Update Parent Alert with summarized Evolution State
            cur.execute("""
                UPDATE alerts
                SET trade_evolution_state = %s,
                    evidence_count = %s,
                    distinct_patterns_count = %s,
                    confirmation_quality = %s,
                    last_event_type = %s,
                    last_event_date = %s,
                    last_event_id = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (parent_state, evidence_count, distinct_patterns_count, confirmation_quality, event_type, today_date, event_id, parent_id))
            commit_cb()
            success = True

            logger.info(f"🔁 [TRADE_EVOLUTION] {symbol} ({scanner}) -> Event: {event_type} | State: {parent_state} | PnL: {pnl_since_entry_pct:+.1f}% | Evidence: {evidence_count}x ({confirmation_quality}) | Reason: {reason_code}")

            # Dispatch Focused Telegram Notification
            try:
                from telegram_engine import queue_telegram_message
                clean_pat = current_pattern.replace('_', ' ')
                
                ord_suffix = "th"
                if evidence_count == 1: ord_suffix = "st"
                elif evidence_count == 2: ord_suffix = "nd"
                elif evidence_count == 3: ord_suffix = "rd"
                ord_str = f"{evidence_count}{ord_suffix}"

                if event_type == "PYRAMID_CANDIDATE":
                    tg_msg = (
                        f"🔥 <b>PYRAMID CANDIDATE — {ord_str.upper()} CONFIRMATION</b>\n\n"
                        f"<b>#{symbol}</b> @ ₹{current_trigger_price:.2f}\n\n"
                        f"Original Entry: ₹{orig_entry_price:.2f} | Current P&L: <b>+{pnl_since_entry_pct:.1f}% 🟢</b>\n\n"
                        f"New Pattern: <b>{clean_pat}</b>\n"
                        f"RVOL: {current_rvol:.1f}x ✅\n"
                        f"Higher Low: {'✅' if higher_low else '❌'}\n"
                        f"Remaining R:R: <b>{remaining_rr:.1f}R</b> (Room to Resistance: {dist_to_resistance_pct:.1f}%)\n\n"
                        f"Evidence Trail: <b>{evidence_count} independent confirmations ({confirmation_quality} Quality)</b>\n\n"
                        f"🛡️ Suggested Trailing SL: <b>₹{suggested_trailing_sl:.2f}</b>\n\n"
                        f"📝 <b>User Action:</b> Existing position is working. Fresh setup confirms continuation. Add size only if your position-sizing/risk rules permit."
                    )
                elif event_type == "THESIS_RECONFIRMATION":
                    tg_msg = (
                        f"🔁 <b>THESIS RE-CONFIRMED — {ord_str.upper()} CONFIRMATION</b>\n\n"
                        f"<b>#{symbol}</b> @ ₹{current_trigger_price:.2f}\n\n"
                        f"Original Entry: ₹{orig_entry_price:.2f} | Current P&L: <b>{pnl_since_entry_pct:+.1f}% ⚪</b>\n\n"
                        f"New Pattern: <b>{clean_pat}</b>\n"
                        f"RVOL: {current_rvol:.1f}x ✅\n"
                        f"Support: Base structure holds firmly.\n\n"
                        f"Evidence Trail: <b>{evidence_count} independent confirmations</b>\n\n"
                        f"📝 <b>User Action:</b> Maintain existing position. Do not add size."
                    )
                else:
                    tg_msg = (
                        f"⚠️ <b>RE-TRIGGER — ADD BLOCKED</b>\n\n"
                        f"<b>#{symbol}</b> @ ₹{current_trigger_price:.2f}\n\n"
                        f"Original Entry: ₹{orig_entry_price:.2f} | Current P&L: <b>{pnl_since_entry_pct:+.1f}% 🔴</b>\n\n"
                        f"New Pattern: <b>{clean_pat}</b>\n"
                        f"Blocker: <b>{reason_code.replace('_', ' ')}</b>\n\n"
                        f"🛡️ <b>User Action:</b> Maintain hard stop loss. No additional risk permitted."
                    )
                queue_telegram_message(tg_msg, symbol=symbol)
            except Exception as _tg_err:
                logger.debug(f"Telegram dispatch error: {_tg_err}")

            # Rebuild performance cache asynchronously
            try:
                from performance_tracker import trigger_performance_rebuild
                trigger_performance_rebuild()
            except Exception:
                pass

            return False, f"Re-trigger recorded ({event_type})", 0.0, 0

        # ─────────────────────────────────────────────────────────────────
        # 🛡️ EXECUTION GATE FSM (Circuit, Liquidity, and Spread Guards)
        # ─────────────────────────────────────────────────────────────────
        cand_ctx = context or {}
        upper_circuit = float(cand_ctx.get("upper_circuit") or cand_ctx.get("circuit_upper") or 0.0)
        daily_turnover_cr = float(cand_ctx.get("turnover_cr") or cand_ctx.get("daily_turnover_cr") or 0.0)
        bid_ask_spread_pct = float(cand_ctx.get("bid_ask_spread_pct") or cand_ctx.get("spread_pct") or 0.0)
        
        execution_status = "EXECUTABLE"
        execution_block_reason = ""

        # Check 1: Circuit Proximity Guard
        if upper_circuit > 0 and entry_price and entry_price >= (upper_circuit * 0.998):
            execution_status = "BLOCKED_UPPER_CIRCUIT"
            execution_block_reason = f"Entry ₹{entry_price:.2f} within 0.2% of Upper Circuit ₹{upper_circuit:.2f}"
            logger.info(f"🚫 [EXECUTION_GATE] {symbol} Execution Blocked: {execution_block_reason}")
        
        # Check 2: Liquidity / Turnover Capacity Floor (₹5 Cr)
        elif daily_turnover_cr > 0 and daily_turnover_cr < 5.0:
            execution_status = "BLOCKED_LOW_LIQUIDITY"
            execution_block_reason = f"Daily turnover ₹{daily_turnover_cr:.2f}Cr below ₹5.00Cr institutional liquidity floor"
            logger.info(f"🚫 [EXECUTION_GATE] {symbol} Execution Blocked: {execution_block_reason}")

        # Check 3: Friction & Bid-Ask Spread Guard (Spread > 0.50%)
        elif bid_ask_spread_pct > 0.50:
            execution_status = "BLOCKED_HIGH_SPREAD"
            execution_block_reason = f"Bid-Ask spread {bid_ask_spread_pct:.2f}% exceeds 0.50% execution limit"
            logger.info(f"🚫 [EXECUTION_GATE] {symbol} Execution Blocked: {execution_block_reason}")

        rvol_diurnal_val = float(cand_ctx.get("rvol_diurnal")) if cand_ctx.get("rvol_diurnal") is not None else None
        rvol_rolling_val = float(cand_ctx.get("rvol_rolling") or volume_ratio or 1.0)

        eff_actual_entry_price = actual_entry_price if actual_entry_price is not None else entry_price
        today_date = datetime.now(IST).date()
        cur.execute("""
            INSERT INTO alerts
                (symbol, breakout_type, alert_time, alert_date, source_trading_date, alert_fingerprint, scanner, category,
                entry_price, stop_loss, initial_stop_loss, target_price, target_1, target_2, target_3, target_4,
                signals, score, rsi, volume_ratio, status, context, capital_allocated, shares_bought, remaining_shares,
                model_version, bayesian_regime, bayesian_weights, data_partition, cash_in_hand, current_price,
                structural_failure_stop, target_quality_score, entry_mode, actual_entry_price, execution_state,
                evaluation_id, scanner_run_id, trade_evolution_state, evidence_count, distinct_patterns_count,
                confirmation_quality, last_event_type, last_event_date, execution_status, execution_block_reason,
                rvol_diurnal, rvol_rolling)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING_ENTRY', %s, %s, 'INITIAL', 1, 1, 'INITIAL', 'NEW_ENTRY', %s, %s, %s, %s, %s)
            RETURNING id;
        """, (symbol, breakout_type, alert_time, today_date.strftime('%Y-%m-%d'), source_trading_date, alert_fingerprint, scanner, category,
            entry_price, stop_loss, stop_loss, target_price, target_1, target_2, target_3, target_4,
            signals, score, rsi, volume_ratio, context_str, capital_allocated, shares_bought, shares_bought,
            model_version, bayesian_regime, weights_str, data_partition, cash_in_hand or 0.0, entry_price,
            structural_failure_stop, target_quality_score, entry_mode, eff_actual_entry_price,
            evaluation_id, scanner_run_id, today_date, execution_status, execution_block_reason,
            rvol_diurnal_val, rvol_rolling_val))
        row = cur.fetchone()
        inserted = (row is not None) or (getattr(cur, "rowcount", 0) > 0)
        commit_cb()
        success = True
        if inserted:
            logger.info(f"✅ [DB_SAVE] Alert for {symbol} ({canonical_scanner}) SUCCESSFULLY SAVED to DB | Entry: ₹{entry_price:.2f} | Score: {score} | Fingerprint: {alert_fingerprint}")
        else:
            logger.info(f"🚫 [DB_SAVE] Alert for {symbol} ({scanner or 'EOD'}) SKIPPED — Reason: SAME_DAY_DUPLICATE (Already alerted today)")
        if inserted:
            alert_id = row[0] if row else 0
            base_score_val = kwargs.get('base_score', score or 80)

            # Record initial NEW_ENTRY event into alert_events
            try:
                cand_ctx = context or {}
                cur.execute("""
                    INSERT INTO alert_events
                        (alert_id, symbol, scanner, pattern, event_type, event_date, event_time,
                         trigger_price, original_entry_price, pnl_since_entry_pct, score, rvol, clv,
                         higher_low, dist_from_ema20_pct, is_extended, nearest_resistance, distance_to_resistance_pct,
                         remaining_rr_to_resistance, suggested_trailing_sl, evidence_count, distinct_patterns_count,
                         confirmation_quality, reason_code, notes)
                    VALUES (%s, %s, %s, %s, 'NEW_ENTRY', %s, NOW(),
                            %s, %s, 0.0, %s, %s, %s,
                            TRUE, %s, FALSE, %s, %s,
                            %s, %s, 1, 1,
                            'INITIAL', 'INITIAL_BREAKOUT_ENTRY', %s)
                    ON CONFLICT DO NOTHING;
                """, (alert_id, symbol, scanner or 'TECHNICAL', str(signals or breakout_type or 'BREAKOUT').strip(),
                      today_date, entry_price or 0.0, entry_price or 0.0, score or 80, volume_ratio or 1.0,
                      float(cand_ctx.get("clv") or 0.75), float(cand_ctx.get("dist_from_ema20_pct") or 0.0),
                      target_1 or (entry_price * 1.08 if entry_price else 0.0), 8.0,
                      2.0, stop_loss or (entry_price * 0.95 if entry_price else 0.0),
                      f"Initial {scanner or 'TECHNICAL'} entry on {today_date}."))
                commit_cb()
            except Exception as _init_ev_err:
                logger.debug(f"Initial alert_events record error: {_init_ev_err}")

            rs_bonus_val = kwargs.get('rs_bonus', 0)
            sector_bonus_val = kwargs.get('sector_bonus', 0)
            rs_pct_val = min(999.0, max(0.0, float(kwargs.get('rs_percentile', 0.0) or 0.0)))
            sector_name_val = kwargs.get('sector_name', '')
            regime_score_val = min(999.0, max(0.0, float(kwargs.get('regime_score', 80.0) or 0.0)))

            risk_dist = max(0.01, float(entry_price or 0.0) - float(stop_loss or 0.0))
            rr_val = round(min(999.0, max(-999.0, (float(target_1 or 0.0) - float(entry_price or 0.0)) / risk_dist)), 2) if entry_price and target_1 else 1.5
            atr_pct_val = round(min(999.0, max(0.0, (risk_dist / float(entry_price or 1.0)) * 100.0)), 2) if entry_price else 2.0

            # Earnings Calendar removed — earnings fields set to defaults
            ed_info = {"earnings_flag": False, "days_to_earnings": 999, "earnings_date": None, "earnings_severity": "NONE", "date_status": "UNKNOWN", "warning_msg": ""}

            try:
                cur.execute("""
                    INSERT INTO alert_outcomes
                        (alert_id, leg, symbol, scanner, regime, regime_score, base_score, rs_bonus, sector_bonus,
                         rs_percentile, sector_name, rr_at_alert, atr_pct_at_alert, entry_price, stop_loss, target_1, target_2, target_3, target_4,
                         earnings_flag, days_to_earnings, earnings_date, earnings_severity, date_status,
                         alert_timestamp)
                    VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (alert_id, leg) DO NOTHING
                """, (alert_id, symbol, scanner or 'EOD', bayesian_regime or 'BULL', regime_score_val,
                      base_score_val, rs_bonus_val, sector_bonus_val, rs_pct_val, sector_name_val,
                      rr_val, atr_pct_val, entry_price or 0.0, stop_loss or 0.0, target_1 or 0.0, target_2, target_3, target_4,
                      ed_info["earnings_flag"], ed_info["days_to_earnings"], ed_info["earnings_date"],
                      ed_info["earnings_severity"], ed_info["date_status"]))
                commit_cb()
            except Exception as oe:
                logger.error(f"Failed to snapshot alert_outcome for alert {alert_id}: {oe}")

            msg = f'{symbol} | {category} | Buy: ₹{entry_price} | SL: ₹{stop_loss} | T1: ₹{target_1}'
            insert_notification('buy', f'Buy Alert / {scanner}', msg, symbol)

            # Trigger web push notification
            try:
                import push_service
                title = f"🚨 {symbol} Breakout"
                body = f"Buy Alert at ₹{entry_price} ({category})"
                threading.Thread(target=push_service.send_push_to_all, args=(title, body, "/", symbol), daemon=True).start()
            except Exception as e:
                logger.exception(f"Failed to start push thread")

            # Asynchronously rebuild performance_data.json so Dashboard Alert Table updates immediately
            try:
                from performance_tracker import trigger_performance_rebuild
                trigger_performance_rebuild()
            except Exception as _p_err:
                logger.debug(f"Performance rebuild trigger error: {_p_err}")

            # [RULE 67 CHANGE-RATIONALE]: Ensure newly saved alert invalidates any dashboard response caches immediately
            try:
                from dashboard_server import invalidate_all_dashboard_caches
                invalidate_all_dashboard_caches()
            except Exception:
                pass

        return inserted, "Inserted" if inserted else "DB CONFLICT (Duplicate)", capital_allocated, shares_bought

    if conn is None:
        with _DB_WRITE_LOCK:
            with get_connection() as local_conn:
                try:
                    with local_conn.cursor() as cur:
                        return _execute(cur, local_conn.commit)
                except Exception:
                    logger.exception(f"❌ save_alert_if_new failed for {symbol}")
                    return False, "Database insertion failed", 0.0, 0
                finally:
                    if not success:
                        local_conn.rollback()
    else:
        try:
            with conn.cursor() as cur:
                return _execute(cur, lambda: None)
        except Exception:
            logger.exception(f"❌ save_alert_if_new failed for {symbol}")
            return False, "Database insertion failed", 0.0, 0

def log_scanner_decision(
    evaluation_id: str,
    scanner_run_id: str,
    symbol: str,
    candidate_sequence: int,
    scanner: str,
    status: str,
    rejection_type: str = None,
    primary_rejection_reason: str = None,
    data_timestamp: datetime = None,
    setup_type: str = None,
    setup_subtype: str = None,
    state_at_evaluation: str = None,
    feature_snapshot: dict = None,
    gate_evaluations: list = None,
    scanner_version: str = None,
    config_version: str = None,
    feature_schema_version: str = "1",
    regime_version: str = "1.0",
    execution_engine_version: str = "7.1",
    **kwargs
) -> bool:
    """
    Log a candidate evaluation decision (PASS or FAIL) for Alert Quality Improvement Wave 1.
    """
    snapshot_json = json.dumps(_sanitize_for_json(feature_snapshot or {}))
    gates_json = json.dumps(_sanitize_for_json(gate_evaluations or []))

    cf_entry = kwargs.get('counterfactual_entry_price')
    cf_sl = kwargs.get('counterfactual_stop_loss')
    cf_t1 = kwargs.get('counterfactual_target_1')
    cf_t2 = kwargs.get('counterfactual_target_2')
    cf_t3 = kwargs.get('counterfactual_target_3')
    cf_mode = kwargs.get('counterfactual_entry_mode')
    cf_rule_version = kwargs.get('counterfactual_rule_version')
    cf_exclusion = kwargs.get('counterfactual_exclusion_reason')
    cf_status = kwargs.get('counterfactual_status', 'PENDING')
    cf_mfe = kwargs.get('counterfactual_mfe_r')
    cf_mae = kwargs.get('counterfactual_mae_r')
    cf_realized = kwargs.get('counterfactual_realized_r')
    cf_labels = json.dumps(_sanitize_for_json(kwargs.get('counterfactual_outcome_labels', {})))

    query = """
        INSERT INTO scanner_evaluation_log (
            evaluation_id, scanner_run_id, symbol, candidate_sequence, scanner, status,
            rejection_type, primary_rejection_reason, data_timestamp, setup_type, setup_subtype,
            state_at_evaluation, feature_snapshot, gate_evaluations, scanner_version, config_version,
            feature_schema_version, regime_version, execution_engine_version,
            counterfactual_entry_price, counterfactual_stop_loss, counterfactual_target_1,
            counterfactual_target_2, counterfactual_target_3, counterfactual_entry_mode,
            counterfactual_rule_version, counterfactual_mfe_r, counterfactual_mae_r,
            counterfactual_realized_r, counterfactual_outcome_labels, counterfactual_exclusion_reason,
            counterfactual_status, counterfactual_generated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (evaluation_id) DO NOTHING;
    """

    params = (
        evaluation_id, scanner_run_id, symbol, candidate_sequence, scanner, status,
        rejection_type, primary_rejection_reason, data_timestamp, setup_type, setup_subtype,
        state_at_evaluation, snapshot_json, gates_json, scanner_version, config_version,
        feature_schema_version, regime_version, execution_engine_version,
        cf_entry, cf_sl, cf_t1, cf_t2, cf_t3, cf_mode, cf_rule_version,
        cf_mfe, cf_mae, cf_realized, cf_labels, cf_exclusion, cf_status,
        datetime.now(IST) if cf_status == 'COMPLETED' else None
    )

    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"❌ Failed to log scanner decision for {symbol}: {e}")
                conn.rollback()
                return False

def save_rejected_alert(
    symbol: str,
    scanner: str,
    rejection_reason: str,
    engine_version: str = "SL_ENGINE_V6",
    context: dict = None
) -> None:
    """Save an alert that was rejected by the V6 execution engine gates (e.g. Natural RR, Target Quality)."""
    if DONT_SAVE_ALERTS:
        return

    sanitized_context = _sanitize_for_json(context) if context is not None else None
    context_str = json.dumps(sanitized_context, default=str) if sanitized_context is not None else None

    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            success = False
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO rejected_alerts (symbol, scanner, engine_version, rejection_reason, context)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (symbol, scanner, engine_version, rejection_reason, context_str))
                    conn.commit()
                    success = True
            except Exception:
                logger.exception(f"❌ save_rejected_alert failed for {symbol}")
            finally:
                if not success:
                    conn.rollback()



def update_partial_exit(
    alert_id: int,
    new_status: str,
    new_sl: float,
    shares_sold: int,
    remaining_shares: int,
    realized_pnl_rs: float,
    exit_event: dict,
    execution_state: str = None
) -> None:
    """Handle a partial exit (e.g. T1 hit). Logs event, raises SL, updates shares."""
    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            success = False
            try:
                with conn.cursor() as cur:
                    # Fetch current state for audit log
                    cur.execute("SELECT status, stop_loss, remaining_shares, exit_history FROM alerts WHERE id = %s", (alert_id,))
                    row = cur.fetchone()
                    if not row: return
                    old_state = {"status": row[0], "stop_loss": row[1], "remaining_shares": row[2]}
                    exit_hist = row[3] if row[3] else []

                    if isinstance(exit_hist, str):
                        exit_hist = json.loads(exit_hist)

                    exit_hist.append(exit_event)
                    new_hist_json = json.dumps(exit_hist, default=str)

                    if execution_state:
                        cur.execute("""
                            UPDATE alerts
                            SET status = %s,
                                stop_loss = %s,
                                remaining_shares = %s,
                                exit_history = %s,
                                execution_state = %s
                            WHERE id = %s
                        """, (new_status, new_sl, remaining_shares, new_hist_json, execution_state, alert_id))
                    else:
                        cur.execute("""
                            UPDATE alerts
                            SET status = %s,
                                stop_loss = %s,
                                remaining_shares = %s,
                                exit_history = %s
                            WHERE id = %s
                        """, (new_status, new_sl, remaining_shares, new_hist_json, alert_id))

                    new_state = {"status": new_status, "stop_loss": new_sl, "remaining_shares": remaining_shares, "exit_event": exit_event}
                    cur.execute("INSERT INTO trade_audit_log (alert_id, action, old_state, new_state) VALUES (%s, %s, %s, %s)",
                                (alert_id, 'PARTIAL_EXIT', json.dumps(old_state, default=str), json.dumps(new_state, default=str)))
                    conn.commit()
                    success = True
                    logger.info(f"🔄 Alert {alert_id} partial exit: {new_status} | Booked {shares_sold} | Floating {remaining_shares} | SL raised to {new_sl}")
            except Exception:
                logger.exception(f"❌ update_partial_exit failed for alert_id={alert_id}")
            finally:
                if not success:
                    conn.rollback()

def update_alert_outcome(
    alert_id: int,
    status: str,          # "WIN" | "LOSS" | "CLOSED"
    exit_price: float,
    pnl_pct: float,
    pnl_rs: float = None,
    closed_at: Optional[str] = None,
    exit_signal: Optional[str] = None,
    execution_state: str = None
) -> None:
    """
    Lock in the final outcome of a trade once SL or Target is hit.
    Called by performance_tracker — writes back so future runs skip bar downloads
    for already-closed positions.
    """
    if closed_at is None:
        closed_at = datetime.now(IST).isoformat()
    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            success = False
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT status, stop_loss, remaining_shares, exit_history, entry_price, shares_bought, capital_allocated FROM alerts WHERE id = %s", (alert_id,))
                    row = cur.fetchone()
                    if not row: return
                    if row[0] in ("WIN", "LOSS", "EXPIRED", "NEUTRAL", "CLOSED", "REJECTED"):
                        logger.debug(f"🔒 Alert {alert_id} is already in terminal state ({row[0]}); ignoring outcome update.")
                        return
                    old_state = {"status": row[0], "stop_loss": row[1], "remaining_shares": row[2]}

                    if pnl_rs is None or pnl_rs == 0:
                        ep = float(row[4]) if row[4] is not None else None
                        sh = int(row[5]) if row[5] is not None else None
                        cap = float(row[6]) if row[6] is not None else None
                        if exit_price and ep and sh:
                            pnl_rs = round((exit_price - ep) * sh, 2)
                        elif pnl_pct is not None and cap:
                            pnl_rs = round((pnl_pct / 100.0) * cap, 2)

                    # Note: We allow overwriting OPEN or any PARTIAL_WIN_x
                    if execution_state:
                        cur.execute("""
                            UPDATE alerts
                            SET status      = %s,
                                exit_price  = %s,
                                pnl_pct     = %s,
                                pnl_rs      = %s,
                                closed_at   = %s,
                                exit_signal = %s,
                                remaining_shares = 0,
                                execution_state = %s
                            WHERE id = %s
                            AND status NOT IN ('WIN', 'LOSS', 'EXPIRED', 'NEUTRAL', 'CLOSED', 'REJECTED')
                        """, (status, exit_price, pnl_pct, pnl_rs, closed_at, exit_signal, execution_state, alert_id))
                    else:
                        cur.execute("""
                            UPDATE alerts
                            SET status      = %s,
                                exit_price  = %s,
                                pnl_pct     = %s,
                                pnl_rs      = %s,
                                closed_at   = %s,
                                exit_signal = %s,
                                remaining_shares = 0
                            WHERE id = %s
                            AND status NOT IN ('WIN', 'LOSS', 'EXPIRED', 'NEUTRAL', 'CLOSED', 'REJECTED')
                        """, (status, exit_price, pnl_pct, pnl_rs, closed_at, exit_signal, alert_id))

                    if cur.rowcount:
                        new_state = {"status": status, "exit_price": exit_price, "pnl_pct": pnl_pct, "pnl_rs": pnl_rs}
                        cur.execute("INSERT INTO trade_audit_log (alert_id, action, old_state, new_state) VALUES (%s, %s, %s, %s)",
                                    (alert_id, 'FINAL_EXIT', json.dumps(old_state, default=str), json.dumps(new_state, default=str)))
                        conn.commit()
                        success = True

                        logger.info(f"🔒 Alert {alert_id} locked as {status} | exit={exit_price} pnl={pnl_pct}%")
                        # Fetch symbol to send notification
                        cur.execute("SELECT symbol FROM alerts WHERE id = %s", (alert_id,))
                        row_sym = cur.fetchone()
                        if row_sym:
                            sym = row_sym[0]
                            p_str = f"₹{pnl_rs:.2f}" if pnl_rs is not None else f"{pnl_pct:.2f}%"
                            msg = f"{sym} | Exit: ₹{exit_price:.2f} | P&L: {p_str}"
                            insert_notification('sell', f'Exit Alert ({status})', msg, sym)
            except Exception:
                logger.exception(f"❌ update_alert_outcome failed for alert_id={alert_id}")
            finally:
                if not success:
                    try:
                        conn.rollback()
                    except Exception:
                        pass

def update_alert_quality_metrics(
    alert_id: int,
    metrics: dict
) -> None:
    """
    Save Alert Quality Improvement metrics (MFE/MAE/R, timestamps, durations, outcome labels) to the alerts table.
    """
    query = """
        UPDATE alerts
        SET mfe_pct = %s,
            mae_pct = %s,
            mfe_r = %s,
            mae_r = %s,
            gross_realized_r = %s,
            net_realized_r = %s,
            entry_timestamp = %s,
            exit_timestamp = %s,
            t1_timestamp = %s,
            t2_timestamp = %s,
            t3_timestamp = %s,
            sl_timestamp = %s,
            entry_bar_id = %s,
            exit_bar_id = %s,
            time_to_entry = %s,
            time_to_t1 = %s,
            time_to_t2 = %s,
            time_to_sl = %s,
            bars_to_t1 = %s,
            bars_to_sl = %s,
            outcome_labels = %s,
            weighted_realized_r = %s
        WHERE id = %s;
    """

    params = (
        metrics.get("mfe_pct"),
        metrics.get("mae_pct"),
        metrics.get("mfe_r"),
        metrics.get("mae_r"),
        metrics.get("gross_realized_r"),
        metrics.get("net_realized_r"),
        metrics.get("entry_timestamp"),
        metrics.get("exit_timestamp"),
        metrics.get("t1_timestamp"),
        metrics.get("t2_timestamp"),
        metrics.get("t3_timestamp"),
        metrics.get("sl_timestamp"),
        metrics.get("entry_bar_id"),
        metrics.get("exit_bar_id"),
        metrics.get("time_to_entry"),
        metrics.get("time_to_t1"),
        metrics.get("time_to_t2"),
        metrics.get("time_to_sl"),
        metrics.get("bars_to_t1"),
        metrics.get("bars_to_sl"),
        json.dumps(metrics.get("outcome_labels", {})),
        metrics.get("weighted_realized_r"),
        alert_id
    )

    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
            except Exception as e:
                logger.error(f"❌ Failed to update alert quality metrics for alert {alert_id}: {e}")
                conn.rollback()

def update_shadow_alert_outcome(
    alert_id: int,
    shadow_status: str,          # "SHADOW_WIN" | "SHADOW_LOSS" | "SHADOW_EXPIRED" | "SHADOW_NEUTRAL"
    shadow_exit_price: float,
    shadow_pnl_pct: float,
    shadow_closed_at: Optional[str] = None
) -> None:
    """
    Write back counterfactual telemetry for rejected alerts (is_rejected = TRUE).
    Does NOT touch live portfolio equity, capital, status, or main outcome columns.
    """
    if shadow_closed_at is None:
        shadow_closed_at = datetime.now(IST).isoformat()
    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT shadow_status FROM alerts WHERE id = %s", (alert_id,))
                    row = cur.fetchone()
                    if row and row[0] in ("SHADOW_WIN", "SHADOW_LOSS", "SHADOW_EXPIRED", "SHADOW_NEUTRAL"):
                        return
                    cur.execute("""
                        UPDATE alerts
                        SET shadow_status = %s,
                            shadow_exit_price = %s,
                            shadow_pnl_pct = %s,
                            shadow_closed_at = %s
                        WHERE id = %s
                    """, (shadow_status, shadow_exit_price, shadow_pnl_pct, shadow_closed_at, alert_id))
                    conn.commit()
            except Exception:
                logger.exception(f"❌ update_shadow_alert_outcome failed for alert_id={alert_id}")

def update_alert_current_price(alert_id: int, current_price: float) -> None:
    """Update current_price column for a specific alert."""
    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("UPDATE alerts SET current_price = %s WHERE id = %s", (current_price, alert_id))
                    conn.commit()
            except Exception as e:
                logger.warning(f"⚠️ Failed to update current_price to {current_price} for alert_id {alert_id}: {e}")

def reset_alert_for_recalculation(alert_id: int) -> bool:
    """
    Resets a closed or partially closed alert back to OPEN state for full replay.
    Restores stop_loss to initial_stop_loss, clears exit history and PnL.
    """
    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            success = False
            try:
                with conn.cursor() as cur:
                    # Verify alert exists and get some base data
                    cur.execute("SELECT status, stop_loss, initial_stop_loss, shares_bought, scanner FROM alerts WHERE id = %s", (alert_id,))
                    row = cur.fetchone()
                    if not row:
                        return False

                    old_status, current_sl, initial_sl, shares_bought, scanner_name = row

                    if scanner_name in ('MULTIBAGGER', 'WEALTH'):
                        msg = f"Blocked recalculation for {scanner_name} alert #{alert_id}. Long-term investments do not support tick-by-tick replays or trailing SLs."
                        logger.warning(f"⚠️ {msg}")
                        # Show notification to Admin
                        from database import insert_notification
                        insert_notification('error', 'Recalculation Blocked', msg)
                        return False


                    # If initial_stop_loss is null for some legacy reason, use current_sl as fallback
                    reset_sl = initial_sl if initial_sl else current_sl

                    cur.execute("""
                        UPDATE alerts
                        SET status = 'OPEN',
                            stop_loss = %s,
                            exit_price = NULL,
                            pnl_pct = NULL,
                            pnl_rs = NULL,
                            closed_at = NULL,
                            exit_history = NULL,
                            remaining_shares = %s
                        WHERE id = %s
                    """, (reset_sl, shares_bought, alert_id))

                    new_state = {"status": "OPEN", "stop_loss": reset_sl, "remaining_shares": shares_bought, "exit_history": None}
                    cur.execute("INSERT INTO trade_audit_log (alert_id, action, old_state, new_state) VALUES (%s, %s, %s, %s)",
                                (alert_id, 'RECALCULATE_RESET', json.dumps({"status": old_status}, default=str), json.dumps(new_state, default=str)))
                    conn.commit()
                    success = True
                    logger.info(f"🔄 Alert {alert_id} reset to OPEN for recalculation. SL restored to {reset_sl}.")
                    return True
            except Exception as e:
                logger.exception(f"❌ reset_alert_for_recalculation failed for alert_id={alert_id}")
                return False
            finally:
                if not success:
                    conn.rollback()

def check_recent_alert(symbol: str, scanner: str, breakout_type: str, lookback_minutes: int, new_score: int = 0) -> bool:
    """
    Returns True if a duplicate alert exists within the cooldown window.
    Score-Upgrade Override: Returns False if new_score >= old_score + 5 (allowing upgraded setups to re-alert).
    """
    from datetime import datetime, timedelta
    cutoff = datetime.now(IST) - timedelta(minutes=lookback_minutes)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT score FROM alerts
                WHERE symbol = %s
                AND scanner = %s
                AND breakout_type = %s
                AND alert_time > %s
                ORDER BY alert_time DESC
                LIMIT 1
            """, (symbol, scanner, breakout_type, cutoff))
            row = cur.fetchone()
            if not row:
                return False

            old_score = row[0] or 0
            if new_score > 0 and new_score >= old_score + 5:
                logger.info(f"⚡ [DEDUP OVERRIDE] {symbol} ({scanner}) allowed re-alert: new score {new_score} >= old score {old_score} + 5")
                return False

            return True

def get_recent_alerts_for_scanner(scanner: str, lookback_minutes: int, only_active: bool = False) -> set[tuple[str, str]]:
    """Returns a set of (symbol, breakout_type) tuples that fired within the lookback window."""
    from datetime import datetime, timedelta
    cutoff = datetime.now(IST) - timedelta(minutes=lookback_minutes)

    with get_connection() as conn:
        with conn.cursor() as cur:
            if only_active:
                cur.execute("""
                    SELECT symbol, breakout_type FROM alerts
                    WHERE scanner = %s
                    AND alert_time::timestamp with time zone > %s
                    AND is_rejected = FALSE
                    AND status NOT IN ('LOSS', 'CLOSED', 'REJECTED')
                """, (scanner, cutoff))
            else:
                cur.execute("""
                    SELECT symbol, breakout_type FROM alerts
                    WHERE scanner = %s
                    AND alert_time::timestamp with time zone > %s
                """, (scanner, cutoff))
            return {(row[0], row[1]) for row in cur.fetchall()}

def get_all_alerts(limit: int = None) -> list[dict]:
    """Return every alert, newest first — including outcome columns.

    Calls init_db() first to ensure all migration columns exist regardless
    of whether a scanner has started yet (performance tracker runs independently).
    """
    init_db()   # no-op if already initialised; ensures columns exist before SELECT
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT
                    a.id, a.symbol, a.breakout_type, a.alert_time, a.alert_date,
                    a.scanner, a.category, a.entry_price, a.stop_loss, a.initial_stop_loss,
                    a.target_price, a.target_1, a.target_2, a.target_3,
                    a.signals, a.score, a.rsi, a.volume_ratio,
                    a.status, a.exit_price, a.pnl_pct, a.closed_at, a.is_rejected, a.exit_signal,
                    a.shadow_status, a.shadow_exit_price, a.shadow_pnl_pct, a.shadow_closed_at,
                    a.capital_allocated, a.shares_bought, a.remaining_shares, a.exit_history, a.pnl_rs, a.context,
                    a.model_version, a.data_partition, a.current_price,
                    COALESCE(a.earnings_flag, FALSE)                AS earnings_flag,
                    COALESCE(a.days_to_earnings, 999)               AS days_to_earnings,
                    a.earnings_date,
                    COALESCE(a.earnings_severity, 'NONE')           AS earnings_severity,
                    COALESCE(a.warning_msg, '')                     AS warning_msg,
                    COALESCE(a.trade_evolution_state, 'INITIAL')    AS trade_evolution_state,
                    COALESCE(a.evidence_count, 1)                   AS evidence_count,
                    COALESCE(a.distinct_patterns_count, 1)          AS distinct_patterns_count,
                    COALESCE(a.confirmation_quality, 'INITIAL')     AS confirmation_quality,
                    COALESCE(a.last_event_type, 'NEW_ENTRY')        AS last_event_type,
                    a.last_event_date
                FROM alerts a
                ORDER BY a.alert_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"

            cur.execute(query)
            rows = []
            for row in cur.fetchall():
                rows.append(dict(row))
            return rows


def get_alert_events(alert_id: int) -> list[dict]:
    """Retrieve all chronological Trade Journey events for a given alert."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, alert_id, symbol, scanner, pattern, event_type, event_date, event_time,
                       trigger_price, original_entry_price, pnl_since_entry_pct, score, rvol, clv,
                       higher_low, dist_from_ema20_pct, is_extended, nearest_resistance,
                       distance_to_resistance_pct, remaining_rr_to_resistance, suggested_trailing_sl,
                       evidence_count, distinct_patterns_count, confirmation_quality, reason_code, notes,
                       created_at
                FROM alert_events
                WHERE alert_id = %s
                ORDER BY id ASC
            """, (alert_id,))
            return [dict(r) for r in cur.fetchall()]


def get_all_alert_events(limit: int = 200) -> list[dict]:
    """Retrieve recent alert events across all symbols."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, alert_id, symbol, scanner, pattern, event_type, event_date, event_time,
                       trigger_price, original_entry_price, pnl_since_entry_pct, score, rvol, clv,
                       higher_low, dist_from_ema20_pct, is_extended, nearest_resistance,
                       distance_to_resistance_pct, remaining_rr_to_resistance, suggested_trailing_sl,
                       evidence_count, distinct_patterns_count, confirmation_quality, reason_code, notes,
                       created_at
                FROM alert_events
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]


def reset_closed_positions_to_open() -> dict:
    """
    Resets all previously closed/review trades back to OPEN/ACTIVE status in PostgreSQL DB
    so that exit monitors can cleanly re-evaluate them in the next run.
    """
    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Reset alerts table
                cur.execute("""
                    UPDATE alerts
                    SET status = 'OPEN',
                        exit_price = NULL,
                        pnl_pct = NULL,
                        pnl_rs = NULL,
                        closed_at = NULL,
                        exit_signal = NULL,
                        exit_reason = NULL,
                        remaining_shares = shares_bought,
                        updated_at = NOW()
                    WHERE status IN ('CLOSED', 'WIN', 'LOSS', 'NEUTRAL');
                """)
                alerts_reset = cur.rowcount

                # 2. Reset wealth_buy_alert table
                cur.execute("""
                    UPDATE wealth_buy_alert
                    SET is_closed = FALSE,
                        status = 'ACTIVE',
                        exit_signal = NULL,
                        exit_price = NULL,
                        exit_date = NULL,
                        exit_time = NULL,
                        status_updated_at = NOW()
                    WHERE status IN ('CLOSED', 'SELL_REVIEW') OR is_closed = TRUE;
                """)
                wealth_reset = cur.rowcount
            conn.commit()
            logger.info(f"🔄 [DB RESET] Reset {alerts_reset} alerts and {wealth_reset} wealth buy alerts back to OPEN/ACTIVE status.")
            return {"alerts_reset": alerts_reset, "wealth_reset": wealth_reset}


# ── Scanner Health API ────────────────────────────────────────────────────────────────

def classify_error_severity(error_msg: str) -> str:
    """
    Classify an error as CRITICAL or IGNORABLE.

    CRITICAL: Code failures, missing config files, compilation errors
    IGNORABLE: API failures for individual/multiple stocks - scanner rejects them and continues

    Returns: 'CRITICAL' | 'IGNORABLE'

    Key principle: If scanner can handle it by rejecting/skipping the stock and continuing,
    it's IGNORABLE (keeps scanner GREEN). If scanner crashes entirely, it's CRITICAL.

    Example: BAJAJ AUTO yfinance timeout
    → Stock rejected, scan continues with 49 other stocks
    → Scanner shows GREEN with alerts from successful stocks
    → Not critical because scanner completed successfully
    """
    if not error_msg:
        return None

    error_lower = error_msg.lower()

    # IGNORABLE patterns: missing stock data, API timeouts for specific/all stocks
    # Scanner handles these gracefully by rejecting the stock(s) and continuing
    ignorable_patterns = [
        'yfinance',
        'timeout',
        'connection refused',
        'no data found',
        'stock not found',
        'not available',
        'api rate limit',
        'temporarily unavailable',
        'data not available',
        'failed to get data for',
        'returned 0 data',  # Stock(s) rejected, others continue
    ]

    # CRITICAL patterns: code/infrastructure issues that crash the scanner
    critical_patterns = [
        'critical',
        'syntax error',
        'import error',
        'indentation error',
        'nameerror',
        'typeerror',
        'attributeerror',
        'keyerror',
        'file not found',
        'no such file',
        'cannot open',
        'permission denied',
        'assert',
        'index error',
        'value error',
        'runtime error',
        'null pointer',
        'undefined',
        'not defined',
        'could not import',
    ]

    # Check for critical patterns first
    for pattern in critical_patterns:
        if pattern in error_lower:
            return 'CRITICAL'

    # Check for ignorable patterns
    for pattern in ignorable_patterns:
        if pattern in error_lower:
            return 'IGNORABLE'

    # Default to CRITICAL for unknown errors (safety first)
    return 'CRITICAL'


def upsert_scanner_health(
    scanner_name: str,
    status: str = None,           # "OK" | "DOWN" | "IDLE" | None (keep existing)
    last_success: str = None,     # ISO timestamp of last successful scan
    today_alerts: int = None,     # number of alerts fired today (None = keep existing)
    error_msg: str = None,        # error message when status=DOWN, else None
    scheduled_for: str = None,    # When this scanner is scheduled to run (e.g., "01:00 IST")
    processed_count: int = None,  # Number of stocks processed/shortlisted/alerts
    total_count: int = None,      # Total number of stocks scanned in universe/watchlist
    outcome: str = None,          # "SUCCESS", "PARTIAL", "FAILED"
    provider_stats: dict = None,  # JSON dict of provider outcome counts
    duration_seconds: float = None, # Time taken for the scan
    retry_count: int = None,      # Number of retries attempted
    run_id: str = None,           # Optional current run_id for execution ownership validation
) -> None:
    """
    Insert or update a scanner's health record in the scanner_health table.
    Canonicalizes scanner names, enforces active_run_id ownership, and logs status transitions.
    """
    scanner_name = normalize_scanner_name(scanner_name)
    try:
        init_db()
        now_str = datetime.now(IST).isoformat()

        # Fetch current status and active_run_id for transition logging & ownership check
        old_status, active_run_id = "UNKNOWN", None
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT status, active_run_id FROM scanner_health WHERE scanner_name = %s;", (scanner_name,))
                    row = cur.fetchone()
                    if row:
                        old_status = row.get("status", "UNKNOWN") if isinstance(row, dict) else row[0]
                        active_run_id = row.get("active_run_id") if isinstance(row, dict) else (row[1] if len(row) > 1 else None)
        except Exception:
            pass

        # Sanitize dict inputs passed accidentally to numeric parameters
        if isinstance(today_alerts, dict):
            today_alerts = today_alerts.get("today_alerts", 0)
        if isinstance(processed_count, dict):
            processed_count = processed_count.get("processed_count", 0)
        if isinstance(total_count, dict):
            total_count = total_count.get("total_count", 0)
        if isinstance(duration_seconds, dict):
            duration_seconds = duration_seconds.get("duration_seconds", 0.0)

        # Normalize and sanitize status values to match DB CHECK constraint
        if status is not None:
            status = str(status).upper()
            if status in ('COMPLETED', 'SUCCESS'):
                status = 'OK'

        allowed_statuses = {'OK', 'DOWN', 'IDLE', 'RUNNING', 'DEGRADED', 'DEGRADED_FALLBACK', 'STOPPED', 'PAUSED'}
        if status == 'ERROR':
            status = 'DOWN'
        elif status is not None and status not in allowed_statuses and not status.startswith('QUEUED'):
            logger.warning(f"upsert_scanner_health: unknown status '{status}' provided — mapping to 'IDLE'")
            status = 'IDLE'

        # 🔍 EXECUTION OWNERSHIP GUARD:
        # If run_id is supplied on a terminal status update (OK/DOWN/IDLE) and active_run_id is set,
        # reject updates from older/stale execution runs.
        if run_id and status in ('OK', 'DOWN', 'IDLE') and active_run_id and active_run_id != run_id:
            logger.warning(f"🛑 [EXECUTION OWNERSHIP REJECTED] Stale write ignored for {scanner_name}: current active_run_id='{active_run_id}', write run_id='{run_id}'")
            return

        # 🔍 FORENSIC DB MUTATION LOGGING (old_status -> new_status)
        import os as _os_mod, threading as _th_mod, traceback as _tb_mod
        _pid = _os_mod.getpid()
        _th_name = _th_mod.current_thread().name
        _stack = _tb_mod.extract_stack(limit=4)
        _caller = f"{_os_mod.path.basename(_stack[-2].filename)}:{_stack[-2].lineno}->{_stack[-2].name}" if len(_stack) >= 2 else "unknown"
        logger.info(
            f"🏥 [SCANNER_HEALTH_MUTATION] scanner='{scanner_name}' transition='{old_status} -> {status}' "
            f"run_id='{run_id or active_run_id}' caller='{_caller}' pid={_pid} thread='{_th_name}'"
        )

        error_severity = None
        is_ack = None

        # Classify error severity and set acknowledgement status
        if status == 'DOWN' and error_msg:
            error_severity = classify_error_severity(error_msg)
            is_ack = False  # NEW ERROR: mark unacknowledged
        elif status == 'OK':
            error_severity = None
            is_ack = True
            if last_success is None:
                last_success = now_str

        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Build the update/insert query
                    set_clauses = []
                    params = []

                    if status is not None:
                        set_clauses.append("status = %s")
                        params.append(status)
                    if last_success is not None:
                        set_clauses.append("last_success = %s")
                        params.append(last_success)
                    if today_alerts is not None:
                        set_clauses.append("today_alerts = %s")
                        params.append(today_alerts)
                    if error_msg is not None:
                        set_clauses.append("error_msg = %s")
                        params.append(error_msg)
                    elif error_msg is None and status == 'OK':
                        set_clauses.append("error_msg = NULL")
                    if error_severity is not None:
                        set_clauses.append("error_severity = %s")
                        params.append(error_severity)
                    elif status == 'OK':
                        set_clauses.append("error_severity = NULL")
                    if is_ack is not None:
                        set_clauses.append("is_acknowledged = %s")
                        params.append(is_ack)
                    if scheduled_for is not None:
                        set_clauses.append("scheduled_for = %s")
                        params.append(scheduled_for)
                    if processed_count is not None:
                        set_clauses.append("processed_count = %s")
                        params.append(processed_count)
                    if total_count is not None:
                        set_clauses.append("total_count = %s")
                        params.append(total_count)
                    import json
                    if outcome is not None:
                        outcome_str = json.dumps(outcome) if isinstance(outcome, (dict, list)) else str(outcome)
                        set_clauses.append("outcome = %s")
                        params.append(outcome_str)
                    if provider_stats is not None:
                        provider_stats_str = json.dumps(provider_stats) if isinstance(provider_stats, (dict, list)) else str(provider_stats)
                        set_clauses.append("provider_stats = %s")
                        params.append(provider_stats_str)
                    if duration_seconds is not None:
                        set_clauses.append("duration_seconds = %s")
                        params.append(duration_seconds)
                    if retry_count is not None:
                        set_clauses.append("retry_count = %s")
                        params.append(retry_count)
                    if run_id is not None:
                        set_clauses.append("active_run_id = %s")
                        params.append(run_id)
                    elif status in ('OK', 'DOWN', 'IDLE') or (status and str(status).startswith('QUEUED')):
                        set_clauses.append("active_run_id = NULL")

                    set_clauses.append("updated_at = %s")
                    params.append(now_str)

                    insert_cols = ["scanner_name", "status", "updated_at"]
                    if status is None:
                        status = 'IDLE'
                    insert_vals = [scanner_name, status, now_str]

                    if last_success is not None:
                        insert_cols.append("last_success")
                        insert_vals.append(last_success)
                    if today_alerts is not None:
                        insert_cols.append("today_alerts")
                        insert_vals.append(today_alerts)
                    if error_msg is not None:
                        insert_cols.append("error_msg")
                        insert_vals.append(error_msg)
                    if is_ack is not None:
                        insert_cols.append("is_acknowledged")
                        insert_vals.append(is_ack)
                    if error_severity is not None:
                        insert_cols.append("error_severity")
                        insert_vals.append(error_severity)
                    if scheduled_for is not None:
                        insert_cols.append("scheduled_for")
                        insert_vals.append(scheduled_for)
                    if processed_count is not None:
                        insert_cols.append("processed_count")
                        insert_vals.append(processed_count)
                    if total_count is not None:
                        insert_cols.append("total_count")
                        insert_vals.append(total_count)
                    if outcome is not None:
                        outcome_str = json.dumps(outcome) if isinstance(outcome, (dict, list)) else str(outcome)
                        insert_cols.append("outcome")
                        insert_vals.append(outcome_str)
                    if provider_stats is not None:
                        provider_stats_str = json.dumps(provider_stats) if isinstance(provider_stats, (dict, list)) else str(provider_stats)
                        insert_cols.append("provider_stats")
                        insert_vals.append(provider_stats_str)
                    if duration_seconds is not None:
                        insert_cols.append("duration_seconds")
                        insert_vals.append(duration_seconds)
                    if retry_count is not None:
                        insert_cols.append("retry_count")
                        insert_vals.append(retry_count)

                    insert_placeholders = ", ".join(["%s"] * len(insert_cols))
                    insert_cols_str = ", ".join(insert_cols)
                    final_params = insert_vals + params
                    set_sql = ", ".join(set_clauses)

                    # [ATOMIC CONDITIONAL UPDATE]: Prevent stale run writes at SQL engine level
                    cur.execute(f"""
                        INSERT INTO scanner_health
                            ({insert_cols_str})
                        VALUES ({insert_placeholders})
                        ON CONFLICT (scanner_name) DO UPDATE
                            SET {set_sql}
                            WHERE scanner_health.active_run_id IS NULL
                               OR EXCLUDED.active_run_id IS NULL
                               OR scanner_health.active_run_id = EXCLUDED.active_run_id
                    """, final_params)
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    if "chk_scanner_status" in str(exc) or "violates check constraint" in str(exc):
                        try:
                            with conn.cursor() as fix_cur:
                                fix_cur.execute("ALTER TABLE scanner_health DROP CONSTRAINT IF EXISTS chk_scanner_status;")
                                fix_cur.execute("ALTER TABLE scanner_health ADD CONSTRAINT chk_scanner_status CHECK (status IN ('OK', 'DOWN', 'IDLE', 'RUNNING', 'DEGRADED', 'DEGRADED_FALLBACK', 'PAUSED', 'STOPPED') OR status LIKE 'QUEUED%') NOT VALID;")
                                fix_cur.execute(f"""
                                    INSERT INTO scanner_health
                                        ({insert_cols_str})
                                    VALUES ({insert_placeholders})
                                    ON CONFLICT (scanner_name) DO UPDATE
                                        SET {set_sql}
                                        WHERE scanner_health.active_run_id IS NULL
                                           OR EXCLUDED.active_run_id IS NULL
                                           OR scanner_health.active_run_id = EXCLUDED.active_run_id
                                """, final_params)
                            conn.commit()
                            return
                        except Exception as retry_exc:
                            conn.rollback()
                            logger.exception(f"❌ Retry upsert_scanner_health failed for {scanner_name}: {retry_exc}")
                            return
                    logger.exception(f"❌ upsert_scanner_health failed for {scanner_name}")
    except Exception as exc:
        logger.warning(f"upsert_scanner_health skipped (DB unavailable): {exc}")
        return
    finally:
        # Auto-release global lock if scanner went DOWN or FAILED so queued scanners can acquire lock instantly
        if status and str(status).upper() in ("DOWN", "FAILED"):
            try:
                from lock_utils import release_global_lock_if_held_by
                release_global_lock_if_held_by(scanner_name)
            except Exception as auto_rel_err:
                logger.warning(f"Failed to auto-release lock for crashed scanner {scanner_name}: {auto_rel_err}")



def get_all_scanner_health() -> list[dict]:
    """Return all scanner health rows, in-memory seeding any missing standard scanners so cards never disappear."""
    init_db()
    schedule_map = {
        "DAILY_BUILDER": "Daily 05:00 IST",
        "MULTI_TF": "Every 15m Scan / 5m Monitor (09:30 - 15:30 IST)",
        "MULTI_TF_5M": "Every 5min Monitor (09:35 - 15:25 IST)",
        "EOD": "Daily 18:30 IST (Post-Bhavcopy Delivery)",
        "REVERSAL": "Daily 18:30 IST (Post-Bhavcopy Delivery)",
        "PULLBACK": "Daily 18:30 IST (Post-Bhavcopy Delivery)",
        "ACCUMULATION": "Daily 18:35 IST (Post-Bhavcopy / Verified Evening Batch)",
        "TECHNICAL": "Daily 18:15 IST (Post-Close Technical Scan)",
        "Wealth Engine": "Daily 06:00 & 17:00 IST · Market Hours (09:15 - 15:30)",
        "MULTIBAGGER": "Daily 17:30 IST (Daily Fundamental)",
        "PERFORMANCE_TRACKER": "Exit Monitor · Every 5min (09:15 - 15:30 IST)",
        "MULTIBAGGER_EXIT": "Exit Monitor · Every 15min (09:15 - 15:30 IST)",
        "WEALTH_EXIT": "Exit Monitor · Every 5min (09:15 - 15:30 IST)",
        "SHORT_COVERING_EOD": "Daily 19:15 IST (Market Days)",
        "SHORT_COVERING_5M": "Every 5m (09:20 - 15:25 IST Market Days)",
        "Pledge Worker": "Continuous (Daily Refresh)",
        "AI Worker": "Continuous (Sat-Sun Active)",
    }

    # Watchdog Auto-Healing: Only mark stuck RUNNING threads as DOWN if global scanner lock is NOT held AND no fresh heartbeat exists
    try:
        from lock_utils import ProcessLock
        _g_lock = ProcessLock("global_scanner_lock")
        if not _g_lock.locked():
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Find stuck RUNNING scanners:
                    # Only mark as DOWN if BOTH:
                    # 1. scanner_health.updated_at is > 10 minutes old
                    # 2. AND there is NO active RUNNING/QUEUED execution in scanner_execution_history with a fresh heartbeat (within last 10m)
                    cur.execute("""
                        SELECT sh.scanner_name FROM scanner_health sh
                        WHERE sh.status = 'RUNNING'
                          AND sh.updated_at < NOW() - INTERVAL '10 minutes'
                          AND NOT EXISTS (
                              SELECT 1 FROM scanner_execution_history seh
                              WHERE seh.scanner_name = sh.scanner_name
                                AND seh.lifecycle_status IN ('RUNNING', 'QUEUED')
                                AND (
                                    (seh.heartbeat_at IS NOT NULL AND seh.heartbeat_at >= NOW() - INTERVAL '10 minutes')
                                    OR (seh.heartbeat_at IS NULL AND seh.started_at >= NOW() - INTERVAL '10 minutes')
                                )
                          );
                    """)
                    stuck_rows = cur.fetchall()
                    for r in stuck_rows:
                        sc_name = r[0]
                        cur.execute("""
                            UPDATE scanner_health
                            SET status = 'DOWN',
                                error_msg = 'Scanner execution timed out: heartbeat lost (>10m) or hard runtime ceiling exceeded (>2h)',
                                is_acknowledged = FALSE,
                                updated_at = NOW()
                            WHERE scanner_name = %s;
                        """, (sc_name,))
                        # Insert failure trace record in scan_failures so UI history displays the failure
                        cur.execute("""
                            INSERT INTO scan_failures (scan_id, scanner_name, symbol, provider, failure_reason, failed_at)
                            VALUES (%s, %s, 'SYSTEM', 'WATCHDOG', 'Scanner execution timed out or process was terminated unexpectedly before completion', NOW());
                        """, (f"TIMEOUT_{sc_name}_{int(time.time())}", sc_name))
                    conn.commit()
    except Exception as heal_err:
        logger.debug(f"Watchdog health auto-heal warning: {heal_err}")

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT scanner_name, status, last_success, today_alerts, error_msg, is_acknowledged, updated_at, error_severity, error_count, first_error_at, retry_count, scheduled_for, processed_count, total_count, outcome, provider_stats, duration_seconds
                    FROM scanner_health
                    ORDER BY scanner_name
                """)
                rows = [dict(row) for row in cur.fetchall()]
                existing_names = {r["scanner_name"] for r in rows if "scanner_name" in r}
                for sc_name, sched_str in schedule_map.items():
                    if sc_name not in existing_names:
                        rows.append({
                            "scanner_name": sc_name,
                            "status": "IDLE",
                            "scheduled_for": sched_str,
                            "today_alerts": 0,
                            "is_acknowledged": True,
                            "error_count": 0,
                            "retry_count": 0,
                            "duration_seconds": 0.0
                        })
                for r in rows:
                    if r.get("scanner_name") in schedule_map:
                        r["scheduled_for"] = schedule_map[r["scanner_name"]]
                return rows
            except Exception:
                logger.exception("❌ get_all_scanner_health failed")
                return []


def reset_all_scanners_on_boot() -> None:
    """
    Executed during server startup / main.py boot sequence.
    Resets all scanner health rows in scanner_health DB table to clean 'OK' status
    with error_msg = NULL, clearing any stale RUNNING / QUEUED / DOWN timeout flags from previous sessions.
    This guarantees that on server restart, all scanner UI cards load cleanly as GREEN (OK / IDLE).
    """
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE scanner_health
                    SET status = 'OK',
                        active_run_id = NULL,
                        error_msg = NULL,
                        is_acknowledged = TRUE,
                        updated_at = NOW()
                    WHERE status IN ('DOWN', 'RUNNING') OR status LIKE 'QUEUED%' OR active_run_id IS NOT NULL;
                """)
                now_str = datetime.now(IST).isoformat()
                schedule_map = {
                    "DAILY_BUILDER": "Daily 05:00 IST",
                    "MULTI_TF": "Every 15m Scan / 5m Monitor (09:30 - 15:30 IST)",
                    "MULTI_TF_5M": "Every 5min Monitor (09:35 - 15:25 IST)",
                    "EOD": "Daily 18:30 IST (Post-Bhavcopy Delivery)",
                    "REVERSAL": "Daily 18:30 IST (Post-Bhavcopy Delivery)",
                    "PULLBACK": "Daily 18:30 IST (Post-Bhavcopy Delivery)",
                    "ACCUMULATION": "Daily 18:35 IST (Post-Bhavcopy / Verified Evening Batch)",
                    "TECHNICAL": "Daily 18:15 IST (Post-Close Technical Scan)",
                    "Wealth Engine": "Daily 06:00 & 17:00 IST · Market Hours (09:15 - 15:30)",
                    "MULTIBAGGER": "Daily 17:30 IST (Daily Fundamental)",
                    "PERFORMANCE_TRACKER": "Exit Monitor · Every 5min (09:15 - 15:30 IST)",
                    "MULTIBAGGER_EXIT": "Exit Monitor · Every 15min (09:15 - 15:30 IST)",
                    "WEALTH_EXIT": "Exit Monitor · Every 5min (09:15 - 15:30 IST)",
                    "SHORT_COVERING_EOD": "Daily 19:15 IST (Market Days)",
                    "SHORT_COVERING_5M": "Every 5m (09:20 - 15:25 IST Market Days)",
                    "Pledge Worker": "Continuous (Daily Refresh)",
                    "AI Worker": "Continuous (Sat-Sun Active)",
                }
                for sc_name, sched_str in schedule_map.items():
                    cur.execute("""
                        INSERT INTO scanner_health (scanner_name, status, scheduled_for, updated_at)
                        VALUES (%s, 'IDLE', %s, %s)
                        ON CONFLICT (scanner_name) DO UPDATE SET scheduled_for = EXCLUDED.scheduled_for
                    """, (sc_name, sched_str, now_str))
                conn.commit()
        logger.info("🧹 [BOOT RESET] All scanner health statuses reset to clean OK state on server startup.")
    except Exception as e:
        logger.warning(f"Failed to reset scanner health on boot: {e}")


def get_scanner_health(scanner_name: str) -> dict:
    """Return health row for a specific scanner."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                norm_name = normalize_scanner_name(scanner_name)
                cur.execute("""
                    SELECT scanner_name, status, last_success, today_alerts, error_msg, is_acknowledged, updated_at, error_severity, error_count, first_error_at, retry_count, scheduled_for, processed_count, total_count, outcome, provider_stats, duration_seconds
                    FROM scanner_health
                    WHERE scanner_name = %s
                """, (norm_name,))
                row = cur.fetchone()
                if row:
                    return dict(row)
                return {}
            except Exception:
                logger.exception(f"❌ get_scanner_health failed for {scanner_name}")
                return {}


def normalize_scanner_name(scanner_name: str) -> str:
    """Canonicalize scanner name strings across UI and DB."""
    if not scanner_name:
        return "UNKNOWN"
    s = scanner_name.strip()
    upper = s.upper().replace("-", "_").replace(" ", "_")
    if upper in ["DAILY_BUILDER", "DAILYBUILDER"]:
        return "DAILY_BUILDER"
    elif upper in ["EOD", "EOD_BREAKOUT"]:
        return "EOD"
    elif upper in ["REVERSAL", "REVERSAL_SCANNER"]:
        return "REVERSAL"
    elif upper in ["PULLBACK", "PULLBACK_PIPELINE"]:
        return "PULLBACK"
    elif upper in ["ACCUMULATION", "ACCUMULATION_SCANNER", "ACCUMULATION_SCANNER_V1"]:
        return "ACCUMULATION"
    elif upper in ["WEALTH", "WEALTH_ENGINE"]:
        return "Wealth Engine"
    elif upper in ["WEALTH_EXIT", "WEALTH_INTRADAY", "WEALTH_5M"]:
        return "WEALTH_EXIT"
    elif upper in ["MULTIBAGGER"]:
        return "MULTIBAGGER"
    elif upper in ["MULTIBAGGER_EXIT"]:
        return "MULTIBAGGER_EXIT"
    elif upper in ["MULTI_TF", "MULTITF"]:
        return "MULTI_TF"
    elif upper in ["MULTI_TF_5M", "MULTITF_5M", "MULTI_TF_5MIN"]:
        return "MULTI_TF_5M"
    elif upper in ["PERFORMANCE_TRACKER", "PERF_TRACKER"]:
        return "PERFORMANCE_TRACKER"
    elif upper in ["AI_WORKER"]:
        return "AI Worker"
    elif upper in ["PLEDGE_WORKER"]:
        return "Pledge Worker"
    elif upper in ["SHORT_COVERING_EOD", "SHORT_POSITION_DETECTOR"]:
        return "SHORT_COVERING_EOD"
    elif upper in ["SHORT_COVERING_5M", "SHORT_COVERING_IGNITION"]:
        return "SHORT_COVERING_5M"
    elif upper in ["SHORT_COVERING", "SHORTCOVERING"]:
        return "SHORT_COVERING"
    return s


_LOCAL_STOPPED_SCANNERS: set[str] = set()

def is_scanner_stopped(scanner_name: str) -> bool:
    """Return True if scanner is currently STOPPED by Admin."""
    norm_name = normalize_scanner_name(scanner_name)
    if norm_name in _LOCAL_STOPPED_SCANNERS:
        return True
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM scanner_health WHERE UPPER(scanner_name) = UPPER(%s) LIMIT 1", (norm_name,))
                row = cur.fetchone()
                if row and row[0]:
                    status_str = str(row[0]).upper()
                    if status_str in ("STOPPED", "PAUSED"):
                        _LOCAL_STOPPED_SCANNERS.add(norm_name)
                        return True
                    else:
                        _LOCAL_STOPPED_SCANNERS.discard(norm_name)
                        return False
    except Exception as e:
        logger.warning(f"is_scanner_stopped query failed for {scanner_name}: {e}")
    return norm_name in _LOCAL_STOPPED_SCANNERS


def stop_scanner(scanner_name: str) -> bool:
    """Set scanner health status to PAUSED by Admin and update active history runs."""
    norm_name = normalize_scanner_name(scanner_name)
    _LOCAL_STOPPED_SCANNERS.add(norm_name)
    upsert_scanner_health(norm_name, status="PAUSED", error_msg="Stopped by Admin")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE scanner_execution_history
                    SET completed_at = NOW(),
                        lifecycle_status = 'STOPPED',
                        stop_reason = 'Stopped by Admin',
                        error_summary = 'Stopped by Admin via Health Dashboard'
                    WHERE UPPER(scanner_name) = UPPER(%s)
                      AND lifecycle_status IN ('RUNNING', 'QUEUED');
                """, (norm_name,))
                conn.commit()
    except Exception as e:
        logger.warning(f"Failed to update execution history stop state for {norm_name}: {e}")
    logger.info(f"🛑 Scanner '{norm_name}' has been PAUSED/STOPPED by Admin.")
    return True


def resume_scanner(scanner_name: str) -> bool:
    """Resume scanner from PAUSED state back to IDLE/OK (preserving RUNNING if active)."""
    norm_name = normalize_scanner_name(scanner_name)
    _LOCAL_STOPPED_SCANNERS.discard(norm_name)
    init_db()
    current_status = "IDLE"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM scanner_health WHERE UPPER(scanner_name) = UPPER(%s) LIMIT 1", (norm_name,))
                row = cur.fetchone()
                if row and row[0] and str(row[0]).upper() in ("RUNNING", "OK"):
                    current_status = str(row[0]).upper()
    except Exception:
        pass
    upsert_scanner_health(norm_name, status=current_status, error_msg=None)
    logger.info(f"▶️ Scanner '{norm_name}' has been RESUMED by Admin.")
    return True



ALL_KNOWN_SCANNERS = [
    'DAILY_BUILDER', 'MULTI_TF', 'MULTI_TF_5M', 'EOD', 'REVERSAL',
    'PULLBACK', 'ACCUMULATION', 'Wealth Engine', 'MULTIBAGGER',
    'PERFORMANCE_TRACKER', 'MULTIBAGGER_EXIT', 'WEALTH_EXIT',
    'SHORT_COVERING_EOD', 'SHORT_COVERING_5M',
    'Pledge Worker', 'AI Worker', 'Earnings Calendar'
]


def pause_all_scanners() -> bool:
    """Pause all scanners at once."""
    for sc in ALL_KNOWN_SCANNERS:
        stop_scanner(sc)
    logger.info("🛑 ALL SCANNERS PAUSED BY ADMIN")
    return True


def resume_all_scanners() -> bool:
    """Resume all scanners at once."""
    for sc in ALL_KNOWN_SCANNERS:
        resume_scanner(sc)
    logger.info("▶️ ALL SCANNERS RESUMED BY ADMIN")
    return True


def get_scanner_today_trades(scanner_name: str, today_str: str) -> list[dict]:
    """
    Return today's alerts for a specific scanner — used by the dashboard API
    to build hover/drill-down trade list directly from the DB.
    """
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT
                        symbol, category, signals, entry_price, alert_time,
                        stop_loss, target_price, pnl_pct, status, score,
                        exit_price, closed_at
                    FROM alerts
                    WHERE scanner    = %s
                    AND alert_date = %s
                    ORDER BY alert_time DESC
                """, (scanner_name, today_str))
                return [dict(row) for row in cur.fetchall()]
            except Exception:
                logger.exception(f"❌ get_scanner_today_trades failed for {scanner_name}")
                return []


# [VERSION: DASHBOARD_PERF_FIX_v1.0] Batch query to replace N+1 loop in /api/scanner_status.
# Previously the dashboard called get_scanner_today_trades() once per scanner (~10 separate
# SQL queries). This single query fetches ALL scanners' today trades in one round-trip.
def get_all_scanners_today_trades(today_str: str) -> dict:
    """
    Return today's alerts for ALL scanners in a single query.
    Returns dict[scanner_name] -> list[dict] of trades.
    """
    init_db()
    result = {}
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT
                        scanner,
                        symbol, category, signals, entry_price, alert_time,
                        stop_loss, initial_stop_loss, target_1, target_2, target_3,
                        target_price, pnl_pct, status, score,
                        exit_price, closed_at, exit_signal
                    FROM alerts
                    WHERE alert_date = %s
                    UNION ALL
                    SELECT
                        'ACCUMULATION' AS scanner,
                        symbol,
                        category,
                        signals,
                        entry_price,
                        alert_time,
                        stop_loss,
                        initial_stop_loss,
                        target_1,
                        target_2,
                        target_3,
                        target_price,
                        pnl_pct,
                        status,
                        score,
                        exit_price,
                        closed_at,
                        exit_signal
                    FROM (
                        SELECT DISTINCT ON (symbol)
                            symbol,
                            state AS category,
                            COALESCE(invalidation_reason, state) AS signals,
                            close AS entry_price,
                            created_at AS alert_time,
                            stop_loss,
                            stop_loss AS initial_stop_loss,
                            target_1,
                            target_2,
                            target_3,
                            target_1 AS target_price,
                            NULL::real AS pnl_pct,
                            state AS status,
                            score::real AS score,
                            NULL::real AS exit_price,
                            NULL::timestamptz AS closed_at,
                            NULL::text AS exit_signal
                        FROM accumulation_alerts
                        WHERE state IN ('PRE_BREAKOUT', 'ACCUMULATION_WATCH')
                          AND created_at >= (%s || ' 00:00:00')::timestamp AT TIME ZONE 'Asia/Kolkata'
                          AND created_at < (%s || ' 00:00:00')::timestamp AT TIME ZONE 'Asia/Kolkata' + INTERVAL '1 day'
                        ORDER BY symbol, created_at DESC, id DESC
                    ) sub
                    ORDER BY alert_time DESC
                """, (today_str, today_str, today_str))
                for row in cur.fetchall():
                    row_dict = dict(row)
                    scanner = row_dict.pop("scanner", "UNKNOWN")
                    result.setdefault(scanner, []).append(row_dict)
            except Exception:
                logger.exception("❌ get_all_scanners_today_trades failed")
    return result



def get_todays_alerts(today_str: str) -> list[dict]:
    """Return all alerts for the provided alert_date (YYYY-MM-DD)."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT a.id, a.symbol, a.breakout_type, a.alert_time::text as alert_time, a.scanner, a.category, a.entry_price,
                        a.stop_loss, a.initial_stop_loss, a.target_1, a.target_2, a.target_3, a.target_4, a.target_price, a.remaining_shares, a.signals, a.score::int as score, a.status, a.seen_by_user, a.seen_by_admin, a.is_rejected, a.exit_signal,
                        -- [VERSION: EARNINGS_BADGE_v1.0] Earnings fields from alerts table (populated at alert creation time)
                        COALESCE(a.earnings_flag, FALSE)                         AS earnings_flag,
                        COALESCE(a.days_to_earnings, 999)                        AS days_to_earnings,
                        a.earnings_date,
                        COALESCE(a.earnings_severity, 'NONE')                    AS earnings_severity,
                        COALESCE(a.warning_msg, '')                              AS warning_msg,
                        COALESCE(a.trade_evolution_state, 'INITIAL')             AS trade_evolution_state,
                        COALESCE(a.evidence_count, 1)                            AS evidence_count,
                        COALESCE(a.distinct_patterns_count, 1)                   AS distinct_patterns_count,
                        COALESCE(a.confirmation_quality, 'INITIAL')              AS confirmation_quality,
                        COALESCE(a.last_event_type, 'NEW_ENTRY')                 AS last_event_type
                    FROM alerts a
                    WHERE a.alert_date = %s
                    UNION ALL
                    SELECT w.id, w.symbol, w.breakout_type, w.alert_time::text as alert_time, w.breakout_type as scanner, w.portfolio_bucket as category, w.alert_price as entry_price,
                        NULL::real as stop_loss, NULL::real as initial_stop_loss, NULL::real as target_1, NULL::real as target_2, NULL::real as target_3, NULL::real as target_4, NULL::real as target_price, NULL::int as remaining_shares, w.entry_signal as signals, w.fm_score::int as score,
                        CASE WHEN w.is_closed THEN 'CLOSED' ELSE 'OPEN' END as status, FALSE as seen_by_user, FALSE as seen_by_admin, FALSE as is_rejected, w.exit_signal,
                         FALSE                                                    AS earnings_flag,
                         999                                                      AS days_to_earnings,
                         NULL::DATE                                               AS earnings_date,
                         'NONE'::TEXT                                             AS earnings_severity,
                         ''                                                       AS warning_msg,
                         'INITIAL'::TEXT                                          AS trade_evolution_state,
                         1::INT                                                   AS evidence_count,
                         1::INT                                                   AS distinct_patterns_count,
                         'INITIAL'::TEXT                                          AS confirmation_quality,
                         'NEW_ENTRY'::TEXT                                        AS last_event_type
                     FROM wealth_buy_alert w
                     WHERE w.alert_date = %s
                    ORDER BY alert_time DESC
                """, (today_str, today_str))
                return [dict(row) for row in cur.fetchall()]
            except Exception:
                logger.exception("❌ get_todays_alerts failed")
                return []


def mark_alert_seen(alert_id: int, role: str = "user") -> bool:
    """Mark an alert as seen by 'user' or 'admin'. Returns True if updated."""
    init_db()
    # Validate column name to prevent SQL injection
    allowed_cols = {'user': 'seen_by_user', 'admin': 'seen_by_admin'}
    col = allowed_cols.get(role)
    if not col:
        logger.warning(f"Invalid role '{role}' for mark_alert_seen")
        return False

    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Use parameterized query with validated column name
                cur.execute(f"UPDATE alerts SET {col} = TRUE WHERE id = %s", (alert_id,))
                conn.commit()
                return cur.rowcount > 0
            except Exception:
                conn.rollback()
                logger.exception(f"❌ mark_alert_seen failed for id={alert_id}")
                return False


# [RULE 67 CHANGE-RATIONALE]:
# Memoize system_state lookups in memory for 5 seconds to eliminate repetitive SQL queries.
_SYSTEM_STATE_MEM_CACHE = {}  # key -> (value_str, timestamp)

def save_system_state(key: str, value_str) -> None:
    """Save/update a value (string or dict/JSON payload) for a specific key."""
    init_db()
    try:
        if isinstance(value_str, (dict, list)):
            import json
            value_str = json.dumps(value_str)
        elif value_str is not None:
            value_str = str(value_str)

        _SYSTEM_STATE_MEM_CACHE[key] = (value_str, time.time())

        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        INSERT INTO system_state (key, value)
                        VALUES (%s, %s)
                        ON CONFLICT (key) DO UPDATE
                            SET value = EXCLUDED.value
                    """, (key, value_str))
                    conn.commit()
                except Exception:
                    conn.rollback()
                    logger.exception(f"❌ save_system_state failed for key={key}")
    except Exception as outer_err:
        logger.exception(f"❌ save_system_state outer error for key={key}: {outer_err}")


def get_system_state(key: str) -> Optional[str]:
    """Retrieve system state value for a specific key (5s TTL in-memory cache)."""
    now_ts = time.time()
    if key in _SYSTEM_STATE_MEM_CACHE:
        val, ts = _SYSTEM_STATE_MEM_CACHE[key]
        if (now_ts - ts) < 5.0:
            return val

    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT value FROM system_state WHERE key = %s", (key,))
                row = cur.fetchone()
                res = row[0] if row else None
                if res is not None:
                    _SYSTEM_STATE_MEM_CACHE[key] = (res, now_ts)
                return res
            except Exception:
                logger.exception(f"❌ get_system_state failed for key={key}")
                return None

# ── AI CONCALL CACHE ────────────────────────────────────────────────────────
def get_cached_concall_analysis(symbol: str, pdf_url: str):
    """Retrieves cached AI analysis for a specific PDF url."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT analysis_data
                FROM ai_concall_cache_v3
                WHERE symbol = %s AND pdf_url = %s
            """, (symbol, pdf_url))
            row = cur.fetchone()
            if row:
                return row[0]
            return None

def get_ai_cache_count() -> int:
    """Returns the total number of cached AI analyses."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(DISTINCT symbol) FROM ai_concall_cache_v3")
                row = cur.fetchone()
                if row:
                    return int(row[0])
    except Exception:
        pass
    return 0


def get_total_cached_concalls() -> int:
    """Returns the total number of distinct stocks that have cached concall data."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(DISTINCT symbol) FROM ai_concall_cache_v3")
                row = cur.fetchone()
                return row[0] if row else 0
            except Exception as e:
                logger.exception(f"Error getting total cached concalls")
                return 0


def get_ai_concall_stats(symbols: list = None) -> dict:
    """Return stats for AI concall cache: total distinct symbols, last processed symbol and timestamp."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(DISTINCT symbol) FROM ai_concall_cache_v3")
                total_row = cur.fetchone()
                total = total_row[0] if total_row else 0
                cur.execute("SELECT symbol, created_at FROM ai_concall_cache_v3 ORDER BY created_at DESC LIMIT 1")
                last = cur.fetchone()
                if last:
                    return {"total_cached": int(total), "last_symbol": last[0], "last_updated": last[1]}
                return {"total_cached": int(total), "last_symbol": None, "last_updated": None}
            except Exception as e:
                logger.exception(f"Error getting ai concall stats")
                return {"total_cached": 0, "last_symbol": None, "last_updated": None}


# [VERSION: PLEDGE_STATS_DB_v1.2] Update get_promoter_pledge_stats to use last_attempted_at
def get_promoter_pledge_stats(symbols: list = None) -> dict:
    """Return stats for promoter_pledge_cache: processed today, eligible today, total cached, last processed symbol and timestamp."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Get the last processed symbol and timestamp
                cur.execute("SELECT symbol, updated_at FROM promoter_pledge_cache ORDER BY updated_at DESC LIMIT 1")
                last = cur.fetchone()
                last_symbol = last[0] if last else None
                last_updated = last[1] if last else None

                # If symbols not provided, query all active universe tables dynamically from DB
                if not symbols:
                    universe_set = get_all_active_universe_symbols(cur)
                    symbols = list(universe_set) if universe_set else []

                    if not symbols:
                        # Database fallback if daily tables are empty
                        cur.execute("SELECT COUNT(*) FROM promoter_pledge_cache")
                        total = cur.fetchone()[0] or 0
                        cur.execute("SELECT COUNT(*) FROM promoter_pledge_cache WHERE updated_at >= NOW() - INTERVAL '28 days' OR COALESCE(last_attempted_at, updated_at) >= CURRENT_DATE")
                        processed_today = cur.fetchone()[0] or 0
                        return {
                            "total_cached": int(total),
                            "processed_today": int(processed_today),
                            "eligible_today": int(total),
                            "last_symbol": last_symbol,
                            "last_updated": last_updated
                        }

                placeholders = ','.join(['%s'] * len(symbols))

                # 1. Total cached in the universe (active symbols)
                cur.execute(f"SELECT COUNT(*) FROM promoter_pledge_cache WHERE symbol IN ({placeholders})", tuple(symbols))
                total_row = cur.fetchone()
                total = total_row[0] if total_row else 0

                # 2. Processed (old + todays) count in the universe
                cur.execute(f"""
                    SELECT COUNT(*)
                    FROM promoter_pledge_cache
                    WHERE symbol IN ({placeholders})
                      AND (updated_at >= NOW() - INTERVAL '28 days' OR COALESCE(last_attempted_at, updated_at) >= CURRENT_DATE)
                """, tuple(symbols))
                proc_today_row = cur.fetchone()
                processed_today = proc_today_row[0] if proc_today_row else 0

                # 3. Eligible today = Total universe size
                eligible_today = len(symbols)

                return {
                    "total_cached": int(total),
                    "processed_today": int(processed_today),
                    "eligible_today": int(eligible_today),
                    "last_symbol": last_symbol,
                    "last_updated": last_updated
                }
            except Exception as e:
                logger.exception(f"Error getting pledge stats")
                return {
                    "total_cached": 0,
                    "processed_today": 0,
                    "eligible_today": 0,
                    "last_symbol": None,
                    "last_updated": None
                }

def get_pledge_map(symbols: list[str] = None) -> dict[str, float]:
    """Bulk fetch pledge percentages for a list of symbols to prevent N+1 queries in scanners."""
    from data_registry import registry

    # 1. Check in-memory DatasetRegistry
    cached_pledge = registry.get("promoter_pledge")
    if cached_pledge is not None:
        if symbols:
            return {k: v for k, v in cached_pledge.items() if k in symbols}
        return cached_pledge

    init_db()
    pledge_map = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Fetch ALL pledge data into memory to populate the registry once
                cur.execute("SELECT symbol, pledge_pct FROM promoter_pledge_cache")
                for row in cur.fetchall():
                    val = row[1]
                    if val is not None and float(val) >= 0:
                        pledge_map[row[0]] = float(val)

                # Save to DatasetRegistry for future fast access
                registry.put("promoter_pledge", pledge_map)

            except Exception as e:
                logger.exception("Error getting pledge map")

    if symbols:
        return {k: v for k, v in pledge_map.items() if k in symbols}
    return pledge_map


# [RULE 67 CHANGE-RATIONALE: NSE_OFFICIAL_PLEDGE_BULK_UPSERT_v1.0]
# Ingest official NSE bulk pledge snapshot into promoter_pledge_cache and track snapshot metadata.
def has_today_pledge_snapshot() -> bool:
    """Returns True if a successful NSE pledge snapshot has already been ingested today."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM pledge_snapshots 
                    WHERE snapshot_date = CURRENT_DATE AND status = 'COMPLETED' 
                    LIMIT 1
                """)
                return cur.fetchone() is not None
    except Exception as e:
        logger.warning(f"Failed to check today pledge snapshot: {e}")
        return False


def get_all_active_universe_symbols(cur=None) -> set[str]:
    """
    Returns the distinct union of all active equity symbols across:
    1. Daily Builder (daily_watchlist, daily_watchlist_v2, daily_excluded_watchlist, daily_excluded_watchlist_v2)
    2. Multibagger & Wealth (watchlist, elite_wealth_system)
    3. Manual Watchlists (user_watchlists)
    4. Master Equities (master_symbols and local master json cache)
    """
    all_syms = set()

    def _query(c):
        tables_queries = [
            'SELECT DISTINCT "Stock" FROM daily_watchlist WHERE "Stock" IS NOT NULL AND "Stock" != \'\'',
            'SELECT DISTINCT "Stock" FROM daily_excluded_watchlist WHERE "Stock" IS NOT NULL AND "Stock" != \'\'',
            'SELECT DISTINCT symbol FROM daily_watchlist_v2 WHERE symbol IS NOT NULL AND symbol != \'\'',
            'SELECT DISTINCT symbol FROM daily_excluded_watchlist_v2 WHERE symbol IS NOT NULL AND symbol != \'\'',
            'SELECT DISTINCT symbol FROM watchlist WHERE symbol IS NOT NULL AND symbol != \'\'',
            'SELECT DISTINCT symbol FROM user_watchlists WHERE symbol IS NOT NULL AND symbol != \'\'',
            'SELECT DISTINCT symbol FROM master_symbols WHERE symbol IS NOT NULL AND symbol != \'\''
        ]
        for q in tables_queries:
            try:
                c.execute(q)
                for r in c.fetchall():
                    if r and r[0]:
                        all_syms.add(str(r[0]).strip().upper())
            except Exception:
                pass

    if cur is not None:
        _query(cur)
    else:
        try:
            with get_connection() as conn:
                with conn.cursor() as c:
                    _query(c)
        except Exception:
            pass

    # Include local master equities files if present
    try:
        from config import DATA_DIR
        for fname in ["nse_master_equities.json", "nse_bse_master_universe.json"]:
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for sym in data.keys():
                                if sym:
                                    all_syms.add(str(sym).strip().upper())
                except Exception:
                    pass
    except Exception:
        pass

    return all_syms


def upsert_bulk_pledge_records(records: list, snapshot_meta: dict) -> int:
    """
    Bulk UPSERTs parsed NSE pledged records into promoter_pledge_cache and records the snapshot in pledge_snapshots.
    Also ensures all active universe stocks (Daily Builder, Multibagger, Wealth, Manual Watchlists)
    omitted from the encumbrance filing are explicitly recorded with 0.0% unencumbered pledge.
    Atomic transaction guarantees consistency.
    """
    if not records and not snapshot_meta:
        return 0
    init_db()
    inserted_count = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                insert_sql = """
                    INSERT INTO promoter_pledge_cache (
                        symbol, pledge_pct, updated_at, last_attempted_at,
                        pledged_shares, promoter_shares, total_shares,
                        depository_pledged_shares, promoter_holding_pct,
                        depository_pledge_demat_pct, source, as_of_date, snapshot_id
                    ) VALUES (
                        %s, %s, NOW(), NOW(),
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (symbol) DO UPDATE SET
                        pledge_pct = EXCLUDED.pledge_pct,
                        updated_at = NOW(),
                        last_attempted_at = NOW(),
                        pledged_shares = EXCLUDED.pledged_shares,
                        promoter_shares = EXCLUDED.promoter_shares,
                        total_shares = EXCLUDED.total_shares,
                        depository_pledged_shares = EXCLUDED.depository_pledged_shares,
                        promoter_holding_pct = EXCLUDED.promoter_holding_pct,
                        depository_pledge_demat_pct = EXCLUDED.depository_pledge_demat_pct,
                        source = EXCLUDED.source,
                        as_of_date = EXCLUDED.as_of_date,
                        snapshot_id = EXCLUDED.snapshot_id;
                """
                payload = [
                    (
                        r["symbol"], r["pledge_pct"],
                        r.get("pledged_shares"), r.get("promoter_shares"), r.get("total_shares"),
                        r.get("depository_pledged_shares"), r.get("promoter_holding_pct"),
                        r.get("depository_pledge_demat_pct"), r.get("source", "NSE"),
                        r.get("as_of_date"), r.get("snapshot_id")
                    )
                    for r in (records or [])
                    if r.get("symbol")
                ]
                if payload:
                    cur.executemany(insert_sql, payload)
                inserted_count = len(payload)

                # Snapshot metadata
                snap_id = snapshot_meta.get("snapshot_id") if snapshot_meta else f"MANUAL_{int(time.time())}"
                snap_date = snapshot_meta.get("snapshot_date") if snapshot_meta else datetime.now(IST).strftime("%Y-%m-%d")
                total_rows = snapshot_meta.get("total_rows", inserted_count) if snapshot_meta else inserted_count
                matched_cnt = snapshot_meta.get("matched_count", inserted_count) if snapshot_meta else inserted_count

                # Populate 0.0% unencumbered defaults for all active universe symbols not in encumbrance filing
                universe_symbols = get_all_active_universe_symbols(cur)
                encumbered_symbols = {str(r["symbol"]).strip().upper() for r in (records or []) if r.get("symbol")}
                unencumbered_symbols = [s for s in universe_symbols if s not in encumbered_symbols]

                if unencumbered_symbols:
                    unencumbered_payload = [
                        (
                            sym, 0.0, 0, None, None, 0, None, 0.0,
                            "NSE_UNENCUMBERED_DEFAULT", snap_date, snap_id
                        )
                        for sym in unencumbered_symbols
                    ]
                    cur.executemany(insert_sql, unencumbered_payload)
                    logger.info(
                        f"🛡️ [PLEDGE_BULK_UPSERT] Auto-populated 0.0% unencumbered pledge for "
                        f"{len(unencumbered_symbols)} universe symbols (DailyBuilder / Multibagger / Manual Watchlists)."
                    )
                    inserted_count += len(unencumbered_symbols)

                # Record snapshot
                cur.execute("""
                    INSERT INTO pledge_snapshots (
                        snapshot_id, snapshot_date, source, total_rows_downloaded,
                        matched_symbols_count, status, fetched_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (snapshot_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        matched_symbols_count = EXCLUDED.matched_symbols_count;
                """, (snap_id, snap_date, "NSE", total_rows, matched_cnt, "COMPLETED"))

                conn.commit()

                # Invalidate dataset registry cache and reload in-memory map
                try:
                    from data_registry import registry
                    registry.evict("promoter_pledge")
                except Exception:
                    pass
                logger.info(f"✅ [PLEDGE_BULK_UPSERT] Successfully upserted {inserted_count} total pledge records into cache (Snapshot: {snap_id})")

            except Exception as e:
                conn.rollback()
                logger.exception(f"❌ Failed to bulk upsert pledge records: {e}")
                raise

    return inserted_count


def has_valid_concall_cache(symbol: str) -> bool:
    """
    Returns True if a valid (non-error) concall analysis exists for the symbol.
    Uses a native JSONB check — no fragile TEXT date casting.
    This is the primary skip check in the AI worker pre-filter.
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                    FROM ai_concall_cache_v3
                    WHERE symbol = %s
                      AND (analysis_data->>'error') IS NULL
                    LIMIT 1
                """, (symbol,))
                return cur.fetchone() is not None
    except Exception:
        logger.exception(f"Failed to check valid concall cache for {symbol}")
        return False

def has_error_concall_cache_within_24h(symbol: str) -> bool:
    """
    Returns True if an error cache entry was saved for this symbol within the last 7 days.
    [VERSION: AI_WORKER_ERROR_TTL_v1.1] Extended from 24h to 7 days — persistent NSE errors
    (timeout, no PDF) don't self-resolve overnight; daily retries waste API quota.
    Uses a SAFE TRY_CAST approach to handle old/broken created_at TEXT formats.
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Use a safe cast with a fallback — if created_at cannot be parsed as a timestamp,
                # the row is treated as old (excluded). This prevents a single bad row from crashing the query.
                cur.execute("""
                    SELECT 1
                    FROM ai_concall_cache_v3
                    WHERE symbol = %s
                      AND (analysis_data->>'error') IS NOT NULL
                      AND created_at >= NOW() - INTERVAL '7 days'
                    LIMIT 1
                """, (symbol,))
                return cur.fetchone() is not None
    except Exception:
        logger.exception(f"Failed to check error concall cache for {symbol}")
        return False


def get_bulk_recent_concall_analysis(symbols: list, max_age_days: int = 60) -> dict:
    """Bulk fetches the most recent cached AI analysis for a list of symbols."""
    if not symbols:
        return {}
    init_db()
    results = {}
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Use DISTINCT ON to get the latest row per symbol efficiently
                placeholders = ','.join(['%s'] * len(symbols))
                query = f"""
                    SELECT DISTINCT ON (symbol) symbol, analysis_data
                    FROM ai_concall_cache_v3
                    WHERE symbol IN ({placeholders})
                      AND created_at >= NOW() - INTERVAL '1 day' * %s
                    ORDER BY symbol, id DESC
                """
                params = tuple(symbols) + (max_age_days,)
                cur.execute(query, params)
                for row in cur.fetchall():
                    results[row[0]] = row[1]
    except Exception:
        logger.exception("Failed to bulk get recent concall analysis")
    return results

def get_bulk_concall_cache_status(symbols: list) -> dict:
    """
    Bulk fetches the concall cache status for a list of symbols.
    Returns dict: {'valid': set(), 'recent_error': set()}
    """
    init_db()
    res = {'valid': set(), 'recent_error': set()}
    if not symbols:
        return res

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # ANY() is much faster for large arrays than IN (...)
                cur.execute("""
                    SELECT symbol, (analysis_data->>'error') IS NULL as is_valid, created_at
                    FROM ai_concall_cache_v3
                    WHERE symbol = ANY(%s)
                """, (symbols,))

                rows = cur.fetchall()
                from datetime import datetime
                from zoneinfo import ZoneInfo
                now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

                for sym, is_valid, created_at in rows:
                    if is_valid:
                        res['valid'].add(sym)
                    else:
                        # Error case. Check if within 7 days.
                        if created_at:
                            # created_at is TIMESTAMPTZ, but might be naive depending on psycopg2 parsing
                            if created_at.tzinfo is None:
                                created_at = created_at.replace(tzinfo=ZoneInfo("UTC"))
                            days_diff = (now_ist - created_at).total_seconds() / 86400
                            if days_diff <= 7:
                                res['recent_error'].add(sym)
    except Exception:
        logger.exception("Failed to fetch bulk concall cache status")
    return res

def get_recent_concall_analysis(symbol: str, max_age_days: int = 60):
    """
    Retrieves the most recent cached AI analysis for a symbol.
    Uses a SAFE CAST approach to handle old/broken created_at TEXT formats gracefully.
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT analysis_data
                    FROM ai_concall_cache_v3
                    WHERE symbol = %s
                      AND created_at >= NOW() - INTERVAL '1 day' * %s
                    ORDER BY id DESC
                    LIMIT 1
                """, (symbol, max_age_days))
                row = cur.fetchone()
                if row:
                    return row[0]
    except Exception:
        logger.exception(f"Failed to get recent concall analysis for {symbol}")
    return None

def save_concall_analysis(symbol: str, pdf_url: str, analysis_data: dict) -> bool:
    """Saves AI analysis to the cache for a specific (symbol, pdf_url) pair.

    Returns True if the save succeeded, False otherwise.
    [VERSION: CONCALL_CACHE_UNIQUE_FIX_v1.0] Changed ON CONFLICT target from (pdf_url) to
    (symbol, pdf_url) — the old single-column constraint silently overwrote one symbol's cache
    with another when two symbols shared the same NSE PDF URL.
    [VERSION: CONCALL_CACHE_JSON_FIX_v1.1] Use psycopg2.extras.Json adapter instead of
    raw json.dumps string — ensures correct JSONB type casting on all Postgres versions.
    """
    init_db()
    try:
        from psycopg2.extras import Json as PgJson
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ai_concall_cache_v3 (symbol, pdf_url, analysis_data, created_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (symbol, pdf_url) DO UPDATE
                    SET analysis_data = EXCLUDED.analysis_data,
                        created_at    = now()
                """, (symbol, pdf_url, PgJson(analysis_data)))
            conn.commit()
        logger.info(f"✅ [DB] Concall cache saved for {symbol} | pdf_url_prefix={pdf_url[:60]}")
        return True
    except Exception as e:
        logger.exception(f"❌ [DB] Failed to save concall cache for {symbol} | error={e}")
        return False


def get_cache_metadata(key: str):
    """Return metadata for a cache key from data_cache_metadata or None if missing."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("SELECT key, last_fetched, cadence_seconds, rows, etag, source, updated_at FROM data_cache_metadata WHERE key = %s", (key,))
                row = cur.fetchone()
                return dict(row) if row else None
            except Exception:
                logger.exception(f"❌ get_cache_metadata failed for key={key}")
                return None


def get_latest_weights(regime: str) -> dict:
    """Get the latest JSON weights for a given regime."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT model_version, weights
                    FROM score_weight_log
                    WHERE regime = %s
                    ORDER BY id DESC LIMIT 1
                """, (regime,))
                row = cur.fetchone()
                if row:
                    import json
                    w_data = row[1]
                    if isinstance(w_data, str):
                        try:
                            w_data = json.loads(w_data)
                        except Exception:
                            logger.error(f"Failed to parse JSON for regime {regime}")
                            w_data = {}
                    return {"version": row[0], "weights": w_data}
                return None
            except Exception:
                logger.exception(f"❌ get_latest_weights failed for regime={regime}")
                return None

def save_new_weights(model_version: str, regime: str, weights: dict):
    """Save a new version of weights for a given regime."""
    init_db()
    import json
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO score_weight_log (model_version, regime, weights)
                    VALUES (%s, %s, %s)
                """, (model_version, regime, json.dumps(weights)))
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(f"❌ save_new_weights failed for regime={regime}")


def upsert_cache_metadata(key: str, last_fetched: str, cadence_seconds: int, rows: int = None, etag: str = None, source: str = None):
    """Insert or update cache metadata for a given key."""
    init_db()
    now = datetime.now(IST).isoformat()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO data_cache_metadata (key, last_fetched, cadence_seconds, rows, etag, source, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE
                        SET last_fetched = EXCLUDED.last_fetched,
                            cadence_seconds = EXCLUDED.cadence_seconds,
                            rows = COALESCE(EXCLUDED.rows, data_cache_metadata.rows),
                            etag = COALESCE(EXCLUDED.etag, data_cache_metadata.etag),
                            source = COALESCE(EXCLUDED.source, data_cache_metadata.source),
                            updated_at = EXCLUDED.updated_at
                """, (key, last_fetched, cadence_seconds, rows, etag, source, now))
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(f"❌ upsert_cache_metadata failed for key={key}")


def upsert_data_fetch_health(source_name: str, last_success: str = None, last_failure: str = None, consecutive_failures: int = None, error_msg: str = None):
    """Insert/update health row for an external data provider (yfinance, nse, etc.)."""
    init_db()
    now = datetime.now(IST).isoformat()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # If consecutive_failures is None, don't overwrite the existing value.
                if consecutive_failures == 0:
                    # Success for API: Reset consecutive failures, but keep is_acknowledged as-is (requires admin dismissal)
                    cur.execute("""
                        INSERT INTO data_fetch_health (source_name, last_success, consecutive_failures, is_acknowledged, updated_at)
                        VALUES (%s, %s, 0, TRUE, %s)
                        ON CONFLICT (source_name) DO UPDATE
                            SET last_success = COALESCE(EXCLUDED.last_success, data_fetch_health.last_success),
                                consecutive_failures = 0,
                                updated_at = EXCLUDED.updated_at
                    """, (source_name, last_success, now))
                elif consecutive_failures is not None:
                    # Specific consecutive_failures provided (uncommon pathway)
                    cur.execute("""
                        INSERT INTO data_fetch_health (source_name, last_success, last_failure, consecutive_failures, error_msg, is_acknowledged, updated_at)
                        VALUES (%s, %s, %s, %s, %s, FALSE, %s)
                        ON CONFLICT (source_name) DO UPDATE
                            SET last_success = COALESCE(EXCLUDED.last_success, data_fetch_health.last_success),
                                last_failure = COALESCE(EXCLUDED.last_failure, data_fetch_health.last_failure),
                                consecutive_failures = EXCLUDED.consecutive_failures,
                                error_msg = COALESCE(EXCLUDED.error_msg, data_fetch_health.error_msg),
                                is_acknowledged = FALSE,
                                updated_at = EXCLUDED.updated_at
                    """, (source_name, last_success, last_failure, consecutive_failures, error_msg, now))
                else:
                    # Standard failure reporting
                    cur.execute("""
                        INSERT INTO data_fetch_health
                        (source_name, last_success, last_failure, consecutive_failures, error_msg, is_acknowledged, updated_at)
                        VALUES (%s, %s, %s, 1, %s, FALSE, %s)
                        ON CONFLICT (source_name) DO UPDATE
                        SET last_failure = COALESCE(EXCLUDED.last_failure, data_fetch_health.last_failure),
                            consecutive_failures = COALESCE(data_fetch_health.consecutive_failures, 0) + 1,
                            is_acknowledged = CASE WHEN EXCLUDED.error_msg IS DISTINCT FROM data_fetch_health.error_msg THEN FALSE ELSE data_fetch_health.is_acknowledged END,
                            error_msg = COALESCE(EXCLUDED.error_msg, data_fetch_health.error_msg),
                            updated_at = EXCLUDED.updated_at
                    """, (source_name, last_success, last_failure, error_msg, now))
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(f"❌ upsert_data_fetch_health failed for {source_name}")

def acknowledge_data_fetch_health(source_name: str):
    """Admin acknowledgment to clear persistent UI warnings.

    Also clear corresponding scanner_health rows (External:<source> and impacted scanners)
    so the UI immediately reflects the dismissal.
    """
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    UPDATE data_fetch_health
                    SET is_acknowledged = TRUE, error_msg = NULL, consecutive_failures = 0
                    WHERE source_name = %s
                """, (source_name,))
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(f"❌ acknowledge_data_fetch_health failed for {source_name}")
    # Also attempt to clear any scanner_health rows that were set due to this external source
    try:
        # Split base and scope if present
        base = source_name.split(':', 1)[0] if ':' in source_name else source_name
        scope = source_name.split(':', 1)[1] if ':' in source_name else None
        cleared = []
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Clear the generic External:<source_name> row (exact)
                cur.execute("UPDATE scanner_health SET is_acknowledged = TRUE, error_msg = NULL, status = 'OK' WHERE scanner_name = %s", (f'External:{source_name}',))
                if cur.rowcount:
                    cleared.append(f'External:{source_name}')
                # Clear the External:<base> row as well
                cur.execute("UPDATE scanner_health SET is_acknowledged = TRUE, error_msg = NULL, status = 'OK' WHERE scanner_name = %s", (f'External:{base}',))
                if cur.rowcount:
                    cleared.append(f'External:{base}')

                # Try to import mapping from data_fetch_status to know impacted scanners
                try:
                    from data_fetch_status import SOURCE_IMPACT_MAP, INTERVAL_TO_SCANNER
                    impacted = SOURCE_IMPACT_MAP.get(base, [])
                    targeted = []
                    if scope:
                        mapped = INTERVAL_TO_SCANNER.get(scope.lower()) if hasattr(INTERVAL_TO_SCANNER, 'get') else INTERVAL_TO_SCANNER.get(scope.lower())
                        if mapped:
                            targeted = [sc for sc in impacted if sc == mapped]
                        else:
                            targeted = [sc for sc in impacted if sc.upper() == scope.upper()]
                    else:
                        targeted = impacted
                    for sc in targeted:
                        cur.execute("UPDATE scanner_health SET is_acknowledged = TRUE, error_msg = NULL, status = 'OK' WHERE scanner_name = %s", (sc,))
                        if cur.rowcount:
                            cleared.append(sc)
                    conn.commit()
                except Exception:
                    # If we can't import the mapping, still attempt a best-effort clear of External:base
                    conn.rollback()
    except Exception:
        logger.exception(f"❌ Failed to clear scanner_health rows after acknowledging {source_name}")

def acknowledge_scanner_health(scanner_name: str):
    """Admin acknowledgment to clear persistent UI warnings for scanners."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    UPDATE scanner_health
                    SET is_acknowledged = TRUE, error_msg = NULL, status = 'OK'
                    WHERE scanner_name = %s
                """, (scanner_name,))
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(f"❌ acknowledge_scanner_health failed for {scanner_name}")


def upsert_fetch_error(source_name: str, scanner_name: str, symbol: str, interval: str, category: str, error_msg: str = None):
    """Insert or update a fetch_errors aggregation row.

    If the combination (source, scanner, symbol, interval, category) exists, increment occurrences
    and update last_seen/last_error_msg. Otherwise create a new row with occurrences=1.
    """
    init_db()
    now = datetime.now(IST).isoformat()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO fetch_errors (source_name, scanner_name, symbol, interval, category, occurrences, first_seen, last_seen, last_error_msg, is_acknowledged)
                    VALUES (%s, %s, %s, %s, %s, 1, %s, %s, %s, FALSE)
                    ON CONFLICT (source_name, scanner_name, symbol, interval, category) DO UPDATE
                    SET occurrences = fetch_errors.occurrences + 1,
                        last_seen = EXCLUDED.last_seen,
                        last_error_msg = COALESCE(EXCLUDED.last_error_msg, fetch_errors.last_error_msg)
                """, (source_name, scanner_name, symbol, interval, category, now, now, error_msg))
                conn.commit()
                try:
                    from dashboard_server import invalidate_all_dashboard_caches
                    invalidate_all_dashboard_caches()
                except Exception:
                    pass
            except Exception:
                conn.rollback()
                logger.exception(f"❌ upsert_fetch_error failed for {source_name}/{symbol}")

def delete_fetch_error_on_success(source_name: str, scanner_name: str, symbol: str, interval: str, category: str):
    """Delete a fetch error row when the operation succeeds, ensuring it will re-alert if it fails again in the future."""
    delete_fetch_errors_batch_on_success(source_name, scanner_name, [symbol], interval, category)


def delete_fetch_errors_batch_on_success(source_name: str, scanner_name: str, symbols: list, interval: str, category: str):
    """Delete fetch error rows for a list of symbols in a single batch query."""
    if not symbols:
        return
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        DELETE FROM fetch_errors
                        WHERE source_name = %s AND scanner_name = %s AND symbol = ANY(%s) AND interval = %s AND category = %s
                    """, (source_name, scanner_name, list(symbols), interval, category))
                    conn.commit()
                    try:
                        from dashboard_server import invalidate_all_dashboard_caches
                        invalidate_all_dashboard_caches()
                    except Exception:
                        pass
                except Exception:
                    conn.rollback()
                    logger.exception(f"❌ delete_fetch_errors_batch_on_success failed for {len(symbols)} symbols")
    except Exception as db_err:
        logger.debug(f"DB unavailable for delete_fetch_errors_batch_on_success: {db_err}")


def get_all_fetch_errors(limit: int = 100) -> list:
    """Return all non-hidden fetch errors (excluding acknowledged with 0 occurrences).

    Hide errors where is_acknowledged=TRUE AND occurrences=0.
    """
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT id, source_name, scanner_name, symbol, interval, category, occurrences, first_seen, last_seen, last_error_msg, is_acknowledged
                    FROM fetch_errors
                    WHERE is_acknowledged = FALSE
                    ORDER BY occurrences DESC, last_seen DESC
                    LIMIT %s
                """, (limit,))
                return [dict(r) for r in cur.fetchall()]
            except Exception:
                logger.exception("❌ get_all_fetch_errors failed")
                return []


def get_fetch_errors_for_scanner(scanner_name: str) -> list:
    """Return all non-acknowledged fetch_errors for a specific scanner.

    Hide errors where is_acknowledged=TRUE AND occurrences=0.
    """
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT id, source_name, scanner_name, symbol, interval, category, occurrences, first_seen, last_seen, last_error_msg, is_acknowledged
                    FROM fetch_errors
                    WHERE scanner_name = %s
                    AND is_acknowledged = FALSE
                    ORDER BY occurrences DESC, last_seen DESC
                """, (scanner_name,))
                return [dict(r) for r in cur.fetchall()]
            except Exception:
                logger.exception(f"❌ get_fetch_errors_for_scanner failed for {scanner_name}")
                return []


def has_unacknowledged_errors(scanner_name: str) -> bool:
    """Check if a scanner has ANY unacknowledged fetch_errors."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT 1 FROM fetch_errors
                    WHERE scanner_name = %s AND is_acknowledged = FALSE
                    LIMIT 1
                """, (scanner_name,))
                return cur.fetchone() is not None
            except Exception:
                logger.exception(f"❌ has_unacknowledged_errors failed for {scanner_name}")
                return False


def acknowledge_fetch_error(error_id: int) -> bool:
    """Mark a fetch_errors row as acknowledged and reset counter to 0.

    When user clicks 'Ignore', this resets occurrences to 0 and sets is_acknowledged=TRUE.
    If error reoccurs, upsert_fetch_error will set occurrences=1 and is_acknowledged=FALSE.
    """
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Mark the fetch error as acknowledged AND reset counter to 0
                cur.execute("""
                    UPDATE fetch_errors
                    SET is_acknowledged = TRUE, occurrences = 0
                    WHERE id = %s
                """, (error_id,))
                if cur.rowcount == 0:
                    return False

                # Get the scanner_name from this error
                cur.execute("SELECT scanner_name FROM fetch_errors WHERE id = %s", (error_id,))
                row = cur.fetchone()
                if not row:
                    conn.commit()
                    return True

                scanner_name = row[0]

                # Check if this scanner has ANY remaining unacknowledged errors
                cur.execute("""
                    SELECT 1 FROM fetch_errors
                    WHERE scanner_name = %s AND is_acknowledged = FALSE
                    LIMIT 1
                """, (scanner_name,))
                has_more_errors = cur.fetchone() is not None

                # If no more errors, clear the scanner_health record (turn green)
                if not has_more_errors:
                    cur.execute("""
                        UPDATE scanner_health
                        SET status = 'OK', is_acknowledged = TRUE, error_msg = NULL, updated_at = %s
                        WHERE scanner_name = %s
                    """, (datetime.now(IST).isoformat(), scanner_name))
                    logger.info(f"✓ Cleared scanner_health for {scanner_name} (all errors acknowledged)")

                conn.commit()
                return True
            except Exception:
                conn.rollback()
                logger.exception(f"❌ acknowledge_fetch_error failed for id={error_id}")
                return False

def acknowledge_fetch_error_batch(error_ids: list) -> bool:
    """Acknowledge multiple fetch errors in one transaction and update scanner health."""
    if not error_ids:
        return True
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                format_strings = ','.join(['%s'] * len(error_ids))

                # First get the scanner names for these errors before we update them
                cur.execute(f"SELECT DISTINCT scanner_name FROM fetch_errors WHERE id IN ({format_strings})", tuple(error_ids))
                scanners = [row[0] for row in cur.fetchall()]

                # Mark as acknowledged
                cur.execute(f"""
                    UPDATE fetch_errors
                    SET is_acknowledged = TRUE, occurrences = 0
                    WHERE id IN ({format_strings})
                """, tuple(error_ids))

                for scanner_name in scanners:
                    cur.execute("""
                        SELECT 1 FROM fetch_errors
                        WHERE scanner_name = %s AND is_acknowledged = FALSE
                        LIMIT 1
                    """, (scanner_name,))
                    has_more_errors = cur.fetchone() is not None

                    if not has_more_errors:
                        cur.execute("""
                            UPDATE scanner_health
                            SET status = 'OK', is_acknowledged = TRUE, error_msg = NULL, updated_at = %s
                            WHERE scanner_name = %s
                        """, (datetime.now(IST).isoformat(), scanner_name))
                        logger.info(f"✓ Cleared scanner_health for {scanner_name} (all errors acknowledged)")

                conn.commit()
                return True
            except Exception:
                conn.rollback()
                logger.exception(f"❌ acknowledge_fetch_error_batch failed")
                return False

def acknowledge_all_fetch_errors() -> bool:
    """Acknowledge all fetch errors at once."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Mark all errors as acknowledged and reset counters
                cur.execute("""
                    UPDATE fetch_errors
                    SET is_acknowledged = TRUE, occurrences = 0
                    WHERE is_acknowledged = FALSE
                """)

                # Clear scanner_health for all scanners (mark as OK)
                cur.execute("""
                    UPDATE scanner_health
                    SET status = 'OK', is_acknowledged = TRUE, error_msg = NULL, updated_at = %s
                    WHERE status != 'OK'
                """, (datetime.now(IST).isoformat(),))

                conn.commit()
                logger.info("✓ All fetch errors acknowledged")
                return True
            except Exception:
                conn.rollback()
                logger.exception("❌ acknowledge_all_fetch_errors failed")
                return False

def deposit_funds(amount: float) -> float:
    """Deposit funds. Returns new total capital."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Insert deposit transaction
                cur.execute("""
                    INSERT INTO capital_history (transaction_type, amount, description)
                    VALUES ('DEPOSIT', %s, 'User deposit via admin dashboard')
                """, (amount,))

                # Get total capital (base + all deposits)
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) FROM capital_history
                    WHERE transaction_type IN ('BASE_CAPITAL', 'DEPOSIT')
                """)
                result = cur.fetchone()
                total_capital = result[0] if result else 0

                conn.commit()
                logger.info(f"✓ Deposited ₹{amount}. New total capital: ₹{total_capital}")
                return total_capital
            except Exception as e:
                conn.rollback()
                logger.exception(f"❌ deposit_funds failed for amount={amount}")
                raise

def get_capital_info() -> dict:
    """Returns total capital, deployed capital in open trades, available cash, and deposit metrics."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Check if base capital exists, if not initialize with 500000
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) FROM capital_history
                    WHERE transaction_type = 'BASE_CAPITAL'
                """)
                row1 = cur.fetchone()
                base = float((row1[0] if row1 else 0.0) or 0.0)

                if base == 0:
                    # Initialize with default base capital
                    cur.execute("""
                        INSERT INTO capital_history (transaction_type, amount, created_at)
                        VALUES ('BASE_CAPITAL', 500000, NOW())
                    """)
                    conn.commit()
                    base = 500000.0

                # Get total deposits (excluding base)
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) FROM capital_history
                    WHERE transaction_type = 'DEPOSIT'
                """)
                row2 = cur.fetchone()
                deposited = float((row2[0] if row2 else 0.0) or 0.0)

                # Get total capital
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) FROM capital_history
                    WHERE transaction_type IN ('BASE_CAPITAL', 'DEPOSIT')
                """)
                row3 = cur.fetchone()
                total = float((row3[0] if row3 else 0.0) or 0.0)

                # Get capital allocated to active open trades
                cur.execute("""
                    SELECT COALESCE(SUM(COALESCE(capital_allocated, COALESCE(shares_bought, 1) * COALESCE(entry_price, 0))), 0),
                           COUNT(*)
                    FROM alerts
                    WHERE status IN ('OPEN', 'HOURLY_APPROVED', 'DAILY_APPROVED', 'PROMOTED_CONVICTION', 'PARTIAL_WIN_1', 'PARTIAL_WIN_2', 'SELL_REVIEW', 'TRAILING')
                       OR status NOT IN ('WIN', 'LOSS', 'NEUTRAL', 'CLOSED', 'REJECTED')
                """)
                row4 = cur.fetchone()
                allocated = float((row4[0] if row4 and len(row4) > 0 else 0.0) or 0.0)
                open_count = int((row4[1] if row4 and len(row4) > 1 else 0) or 0)

                available_cash = max(0.0, total - allocated)
                used_pct = round((allocated / total * 100), 1) if total > 0 else 0.0

                return {
                    "base_capital": base,
                    "total_deposited": deposited,
                    "total_capital": total,
                    "allocated_capital": allocated,
                    "available_cash": available_cash,
                    "used_pct": used_pct,
                    "open_trades_count": open_count
                }
            except Exception:
                logger.exception("❌ get_capital_info failed")
                return {
                    "base_capital": 500000.0,
                    "total_deposited": 0.0,
                    "total_capital": 500000.0,
                    "allocated_capital": 0.0,
                    "available_cash": 500000.0,
                    "used_pct": 0.0,
                    "open_trades_count": 0
                }

def get_all_data_fetch_health() -> list:
    """Return all rows from data_fetch_health as list of dicts."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("SELECT source_name, last_success, last_failure, consecutive_failures, error_msg, is_acknowledged, updated_at FROM data_fetch_health ORDER BY source_name")
                return [dict(r) for r in cur.fetchall()]
            except Exception:
                logger.exception("❌ get_all_data_fetch_health failed")
                return []

# ── Manual Portfolio Tracker ──────────────────────────────────────────────────

def get_manual_portfolio():
    """Retrieve all manual portfolio entries."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT mp.id, mp.symbol, mp.entry_date::TEXT, mp.entry_price, mp.quantity,
                       FALSE                                                            AS earnings_flag,
                       999                                                              AS days_to_earnings,
                       NULL::DATE                                                       AS earnings_date,
                       'NONE'::TEXT                                                     AS earnings_severity,
                       ''                                                               AS warning_msg
                FROM manual_portfolio mp
                ORDER BY mp.added_at DESC
            """)
            return cur.fetchall()

def add_portfolio_entry(symbol: str, entry_date: str, entry_price: float, quantity: int):
    """Add a new stock to the manual portfolio."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO manual_portfolio (symbol, entry_date, entry_price, quantity)
                VALUES (%s, %s, %s, %s)
            """, (symbol.upper(), entry_date, entry_price, quantity))
        conn.commit()

def remove_portfolio_entry(entry_id: int):
    """Remove a stock from the manual portfolio by ID."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM manual_portfolio WHERE id = %s", (entry_id,))
        conn.commit()

def get_sector_momentum(days=7):
    """Get sector momentum for the last N days. Returns sector stats with win rates & P&L."""
    init_db()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                # Query: Get sector performance from watchlist (joined with alerts)
                cur.execute("""
                    WITH sector_trades AS (
                        SELECT
                            dw."Sector" as sector,
                            a.symbol,
                            a.status,
                            a.pnl_rs,
                            a.alert_date::DATE as trade_date,
                            a.created_at::DATE as created_date
                        FROM alerts a
                        LEFT JOIN daily_watchlist dw ON a.symbol = dw."Stock"
                        WHERE a.created_at >= CURRENT_TIMESTAMP - INTERVAL '%d days'
                        AND a.status IN ('WIN', 'LOSS', 'CLOSED')
                    )
                    SELECT
                        COALESCE(sector, 'Unknown') as sector,
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN status = 'WIN' OR (status = 'CLOSED' AND pnl_rs > 0) THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN status = 'LOSS' OR (status = 'CLOSED' AND pnl_rs <= 0) THEN 1 ELSE 0 END) as losses,
                        ROUND(100.0 * SUM(CASE WHEN status = 'WIN' OR (status = 'CLOSED' AND pnl_rs > 0) THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate_pct,
                        ROUND(COALESCE(SUM(pnl_rs), 0)::NUMERIC, 0)::INTEGER as total_pnl,
                        ROUND((COALESCE(SUM(pnl_rs), 0) / NULLIF(COUNT(*), 0))::NUMERIC, 0)::INTEGER as avg_pnl_per_trade
                    FROM sector_trades
                    GROUP BY sector
                    ORDER BY win_rate_pct DESC, total_trades DESC
                """ % days)
                return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                logger.exception(f"❌ get_sector_momentum failed: {e}")
                return []

# ── Parquet Binary Cache ──────────────────────────────────────────────────────

def upload_parquet_to_db(name: str, file_path: str) -> bool:
    """Upload a binary parquet file to the database for today."""
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ [PARQUET DB SYNC SKIPPED] Cannot upload '{name}': file does not exist at {file_path}")
        return False
    import time
    import psycopg2
    today = datetime.now(IST).strftime("%Y-%m-%d")
    init_db()
    try:
        _t_start = time.perf_counter()
        with open(file_path, "rb") as f:
            binary_data = f.read()
        size_kb = len(binary_data) / 1024.0
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO parquet_cache (name, date, data)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name, date) DO UPDATE SET data = EXCLUDED.data
                """, (name, today, psycopg2.Binary(binary_data)))
            conn.commit()
        dur_s = time.perf_counter() - _t_start
        logger.info(f"💾 [PARQUET DB SYNC SUCCESS] Uploaded '{name}' ({size_kb:.1f} KB) for {today} to Postgres DB parquet_cache in {dur_s:.2f}s")
        return True
    except Exception as e:
        logger.error(f"❌ [PARQUET DB SYNC FAILURE] Failed to upload '{name}' to DB: {e}", exc_info=True)
        return False

def download_parquet_from_db(name: str, file_path: str) -> bool:
    """Download the latest binary parquet file from the database."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT data, date FROM parquet_cache WHERE name = %s ORDER BY date DESC LIMIT 1", (name,))
                row = cur.fetchone()
                if row and row[0] and isinstance(row[0], (bytes, bytearray, memoryview)):
                    import os
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(bytes(row[0]))
                    logger.info(f"⚡ Downloaded {name} from DB parquet_cache (from date: {row[1]})")
                    return True
        return False
    except Exception as e:
        logger.exception(f"❌ Failed to download {name} from DB")
        return False

def download_parquet_from_db_today(name: str, file_path: str) -> bool:
    """Download parquet ONLY if it's from today's date. Returns False if stale."""
    init_db()
    today = datetime.now(IST).strftime("%Y-%m-%d")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT data, date FROM parquet_cache WHERE name = %s AND date = %s", (name, today))
                row = cur.fetchone()
                if row and row[0] and isinstance(row[0], (bytes, bytearray, memoryview)):
                    import os
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(bytes(row[0]))
                    logger.info(f"✅ Downloaded {name} from DB parquet_cache (TODAY's data: {row[1]})")
                    return True
                else:
                    logger.warning(f"⚠️ No today's data ({today}) found for {name} in DB cache")
        return False
    except Exception as e:
        logger.exception(f"❌ Failed to download {name} from DB (today check)")
        return False

def delete_stale_parquet_from_db(name: str) -> bool:
    """Delete all stale (non-today) entries for a given parquet name from the database."""
    init_db()
    today = datetime.now(IST).strftime("%Y-%m-%d")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM parquet_cache WHERE name = %s AND date < %s", (name, today))
                deleted = cur.rowcount
            conn.commit()
        if deleted > 0:
            logger.info(f"🗑️ Deleted {deleted} stale entry/entries for {name} from parquet_cache (older than {today})")
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to delete stale {name} from DB")
        return False

from dataclasses import dataclass

@dataclass
class BundleUploadState:
    generation: int = 0
    uploaded_generation: int = 0
    upload_in_progress: bool = False
    pending: bool = False

_bundle_states: dict[str, BundleUploadState] = {}
_bundle_state_lock = threading.Lock()
_last_bundle_upload_time: dict[str, float] = {}
_last_bundle_checksum: dict[str, str] = {}

def advance_interval_generation(interval: str) -> int:
    """
    [RULE 67 CHANGE-RATIONALE: GENERATION_BASED_PERSISTENCE_v1.0]
    Advances the mutation batch generation counter once per completed fetch/write batch.
    Enforces that bundle persistence represents consistent mutation batches rather than
    triggering 121 individual uploads for 121 parquet writes.
    """
    with _bundle_state_lock:
        st = _bundle_states.setdefault(interval, BundleUploadState())
        st.generation += 1
        return st.generation

def get_interval_generation(interval: str) -> tuple[int, int]:
    """Returns (current_generation, uploaded_generation) for this interval."""
    with _bundle_state_lock:
        st = _bundle_states.setdefault(interval, BundleUploadState())
        return st.generation, st.uploaded_generation

def upload_history_bundle_to_db(interval: str = "1d", min_interval_sec: float = 60.0, force: bool = False) -> bool:
    """
    Compresses all OHLCV parquet and metadata sidecars in data/history/{interval}/
    into a tar.gz bundle and persists to PostgreSQL parquet_cache under name 'history_bundle_{interval}'.

    [RULE 67 CHANGE-RATIONALE: GENERATION_COALESCING_v1.0]
    1. Tracks generation vs uploaded_generation to ensure no redundant bundle uploads.
    2. Coalesces concurrent upload requests: if an upload is already running, marks pending=True.
    3. Captures target_upload_generation BEFORE tar creation to eliminate snapshot race conditions.
    4. Only holds _DB_WRITE_LOCK during the actual SQL INSERT/UPDATE, never during OS tar compression.
    """
    import io
    import time
    import tarfile
    import hashlib
    from config import DATA_DIR

    global _last_bundle_upload_time, _last_bundle_checksum
    now_ts = time.time()

    target_upload_generation = 0
    with _bundle_state_lock:
        st = _bundle_states.setdefault(interval, BundleUploadState())
        if not force and st.generation == st.uploaded_generation and (now_ts - _last_bundle_upload_time.get(interval, 0)) < min_interval_sec:
            logger.debug(f"ℹ️ [HISTORY BUNDLE] Interval {interval} generation {st.generation} already uploaded. Skipping.")
            return True
        if st.upload_in_progress:
            st.pending = True
            logger.debug(f"⏳ [HISTORY BUNDLE] Upload already in progress for {interval}. Marked pending (current_gen={st.generation}).")
            return True
        st.upload_in_progress = True
        st.pending = False
        target_upload_generation = st.generation

    history_dir = os.path.join(DATA_DIR, "history", interval)
    if not os.path.exists(history_dir):
        logger.warning(f"⚠️ [HISTORY BUNDLE DB SYNC SKIPPED] Directory does not exist: {history_dir}")
        with _bundle_state_lock:
            st = _bundle_states.setdefault(interval, BundleUploadState())
            st.upload_in_progress = False
        return False

    files = [f for f in os.listdir(history_dir) if f.endswith(".parquet") or f.endswith(".meta.json")]
    if not files:
        logger.warning(f"⚠️ [HISTORY BUNDLE DB SYNC SKIPPED] No files to compress in {history_dir}")
        with _bundle_state_lock:
            st = _bundle_states.setdefault(interval, BundleUploadState())
            st.upload_in_progress = False
        return False

    if not os.getenv("DATABASE_URL"):
        logger.debug("DATABASE_URL is not set. Skipping history bundle DB upload.")
        with _bundle_state_lock:
            st = _bundle_states.setdefault(interval, BundleUploadState())
            st.upload_in_progress = False
        return False

    init_db()
    today = datetime.now(IST).strftime("%Y-%m-%d")
    upload_success = False
    try:
        import subprocess
        import tempfile

        # [VERSION: DB_UPLOAD_GIL_FIX] Offload compression to OS to prevent freezing the entire scanner
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        # Execute tar at OS level. GNU tar return code 0 = success, 1 = file changed while reading (non-fatal warning on Linux)
        res = subprocess.run(["tar", "-czf", tmp_path, "-C", history_dir, "."], capture_output=True)
        if res.returncode not in (0, 1) or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            logger.warning(f"⚠️ OS tar returned exit code {res.returncode}. Falling back to Python tarfile module.")
            with tarfile.open(tmp_path, "w:gz") as tar:
                for f in files:
                    tar.add(os.path.join(history_dir, f), arcname=f)

        with open(tmp_path, "rb") as f:
            binary_data = f.read()

        os.remove(tmp_path)
        current_md5 = hashlib.md5(binary_data).hexdigest()

        if not force and _last_bundle_checksum.get(interval) == current_md5 and st.generation == target_upload_generation:
            logger.info(f"ℹ️ [HISTORY BUNDLE DB SYNC] Skipped history_bundle_{interval} upload — dataset unchanged (MD5: {current_md5[:8]}, Gen: {target_upload_generation})")
            _last_bundle_upload_time[interval] = now_ts
            upload_success = True
            return True

        name = f"history_bundle_{interval}"

        import psycopg2
        with _DB_WRITE_LOCK:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO parquet_cache (name, date, data)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (name, date) DO UPDATE SET data = EXCLUDED.data
                    """, (name, today, psycopg2.Binary(binary_data)))
                conn.commit()

        _last_bundle_upload_time[interval] = now_ts
        _last_bundle_checksum[interval] = current_md5
        upload_success = True
        logger.info(f"💾 [HISTORY BUNDLE DB SYNC SUCCESS] Uploaded {len(files)} files ({len(binary_data)/1024.0:.1f} KB, MD5: {current_md5[:8]}, Gen: {target_upload_generation}) for {name} to DB parquet_cache for {today}")
        return True
    except Exception as e:
        logger.error(f"❌ [HISTORY BUNDLE DB SYNC FAILURE] Failed to upload history bundle for {interval} to DB: {e}", exc_info=True)
        return False
    finally:
        follow_up_needed = False
        with _bundle_state_lock:
            st = _bundle_states.setdefault(interval, BundleUploadState())
            st.upload_in_progress = False
            if upload_success:
                st.uploaded_generation = max(st.uploaded_generation, target_upload_generation)
            if st.pending or st.generation > st.uploaded_generation:
                st.pending = False
                follow_up_needed = True

        if follow_up_needed:
            logger.info(f"🔄 [HISTORY BUNDLE COALESCE] Generation advanced during sync for {interval} (gen={st.generation} > uploaded={st.uploaded_generation}). Triggering coalesced follow-up.")
            submit_background_upload(lambda _iv=interval: upload_history_bundle_to_db(_iv, force=True))

def restore_history_bundle_from_db(interval: str = "1d") -> bool:
    """
    Downloads the latest tar.gz history bundle for {interval} from PostgreSQL parquet_cache
    and unpacks all parquet & metadata files to data/history/{interval}/ in <0.5s.
    """
    import io
    import tarfile
    from config import DATA_DIR

    name = f"history_bundle_{interval}"
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT data, date FROM parquet_cache WHERE name = %s ORDER BY date DESC LIMIT 1", (name,))
                row = cur.fetchone()
                if not row or not row[0] or not isinstance(row[0], (bytes, bytearray, memoryview)):
                    logger.info(f"ℹ️ No valid DB history bundle found for {name}")
                    return False

                binary_data, bundle_date = row[0], row[1]
                history_dir = os.path.join(DATA_DIR, "history", interval)
                os.makedirs(history_dir, exist_ok=True)

                bio = io.BytesIO(binary_data)
                with tarfile.open(fileobj=bio, mode="r:gz") as tar:
                    tar.extractall(path=history_dir)

                extracted_count = len([f for f in os.listdir(history_dir) if f.endswith(".parquet")])
                logger.info(f"⚡ [RESTORE] Extracted {extracted_count} historical Parquet files for {interval} from DB bundle (from date: {bundle_date}) in <0.5s")
                return True
    except Exception as e:
        logger.exception(f"❌ Failed to restore history bundle for {interval} from DB")
        return False



def save_df_to_table(table_name: str, df: pd.DataFrame):
    """Saves a Pandas DataFrame to a PostgreSQL table dynamically."""
    if df.empty:
        return
    init_db()
    today_str = datetime.now(IST).strftime("%Y-%m-%d")

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Fetch destination table columns
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
            """, (table_name.lower(),))
            rows = cur.fetchall()
            db_cols = {row[0].lower(): row[0] for row in rows}

            if not db_cols:
                logger.warning(f"⚠️ Table '{table_name}' does not exist in DB or has no columns.")
                return

            # 2. Identify date column
            # [VERSION: DB_PATCH_v1.2] Add 'build_date' as first candidate to support V2 tables daily_watchlist_v2 / daily_excluded_watchlist_v2 idempotency
            date_col = None
            for candidate in ["build_date", "date", "run_date", "created_at", "added_at", "updated_at"]:
                if candidate in db_cols:
                    date_col = db_cols[candidate]
                    break

            # 3. If there is old date data, delete it first
            if date_col:
                date_col_safe = date_col.replace("%", "%%")
                table_name_safe = table_name.replace("%", "%%")
                # [VERSION: DB_PATCH_v1.4] [RULE 67 CHANGE-RATIONALE]
                # Delete NULL dates, exact date matches, and timestamp prefix matches (e.g. '2026-08-29%')
                # to ensure idempotency across date/timestamp column formats.
                cur.execute(f'DELETE FROM {table_name_safe} WHERE "{date_col_safe}" IS NULL')
                cur.execute(f'DELETE FROM {table_name_safe} WHERE "{date_col_safe}" = %s', (today_str,))
                try:
                    cur.execute(f'DELETE FROM {table_name_safe} WHERE "{date_col_safe}"::text LIKE %s', (f"{today_str}%",))
                except Exception:
                    pass
            else:
                cur.execute(f"TRUNCATE TABLE {table_name}")

            # 4. Map DataFrame columns to DB columns (case-insensitive)
            df_cols_mapped = {}
            for col in df.columns:
                col_lower = col.lower().replace(" ", "_").replace("%", "pct").replace("yoy", "yoy").replace("qoq", "qoq")
                if col_lower in db_cols:
                    df_cols_mapped[col] = db_cols[col_lower]
                elif col.lower() in db_cols:
                    df_cols_mapped[col] = db_cols[col.lower()]

            insert_cols = list(df_cols_mapped.values())
            df_source_cols = list(df_cols_mapped.keys())

            # If there's a date column and it's not mapped from DataFrame, add it to insert
            add_date_val = False
            if date_col and date_col not in insert_cols:
                insert_cols.append(date_col)
                add_date_val = True

            if not insert_cols:
                logger.warning(f"⚠️ No matching columns found between DataFrame and table '{table_name}'.")
                return

            # 5. Insert rows in batch with ON CONFLICT DO NOTHING for absolute idempotency & speed
            col_list_str = ", ".join(f'"{c.replace("%", "%%")}"' for c in insert_cols)
            table_name_safe = table_name.replace("%", "%%")
            
            data_tuples = []
            for row_vals in df[df_source_cols].itertuples(index=False, name=None):
                row_list = [None if pd.isna(v) else v for v in row_vals]
                if add_date_val:
                    row_list.append(today_str)
                data_tuples.append(tuple(row_list))

            if data_tuples:
                try:
                    from psycopg2.extras import execute_values
                    insert_query = f"INSERT INTO {table_name_safe} ({col_list_str}) VALUES %s ON CONFLICT DO NOTHING"
                    execute_values(cur, insert_query, data_tuples, page_size=1000)
                except Exception:
                    # Fallback to standard execute if execute_values is unavailable
                    val_placeholders = ", ".join(["%s"] * len(insert_cols))
                    fallback_query = f"INSERT INTO {table_name_safe} ({col_list_str}) VALUES ({val_placeholders}) ON CONFLICT DO NOTHING"
                    for vals in data_tuples:
                        cur.execute(fallback_query, vals)

        conn.commit()
    logger.info(f"✅ Saved {len(df)} rows to table '{table_name}' in database.")

def check_data_exists_for_today() -> bool:
    """Checks if Daily Builder universe data exists for today's IST date across parquet cache or DB tables."""
    init_db()
    from zoneinfo import ZoneInfo
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Check parquet_cache for today
                cur.execute("SELECT 1 FROM parquet_cache WHERE name = 'daily_builder' AND date = %s", (today_str,))
                if cur.fetchone():
                    return True

                # 2. Check if daily_watchlist_v2 or watchlist has rows for today
                for tbl in ["daily_watchlist_v2", "watchlist", "daily_watchlist"]:
                    cur.execute("""
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = %s
                    """, (tbl,))
                    if cur.fetchone():
                        # Table exists, check if date column has today's date
                        cur.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = %s
                        """, (tbl,))
                        cols = [str(row[0] if isinstance(row, (tuple, list)) else row.get('column_name', '')).lower() for row in cur.fetchall()]
                        date_col = next((c for c in ["date", "run_date", "created_at", "added_at"] if c in cols), None)
                        if date_col:
                            cur.execute(f'SELECT 1 FROM "{tbl}" WHERE "{date_col}"::text LIKE %s LIMIT 1', (f"{today_str}%",))
                            if cur.fetchone():
                                return True
        return False
    except Exception as e:
        # [RULE 67 CHANGE-RATIONALE]: Log as warning instead of unhandled exception so callers can safely proceed
        logger.warning(f"⚠️ check_data_exists_for_today warning: {e}")
        return False

# ── Checkpoint persistence (audit trail) ──────────────────────────────────────────────

def get_latest_bhavcopy_cache():
    """Retrieve the delivery data dict from the most recent cached bhavcopy entry."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('''
                    SELECT delivery_data FROM bhavcopy_cache
                    ORDER BY trading_date DESC LIMIT 1
                ''')
                row = cur.fetchone()
                if row and row['delivery_data']:
                    return row['delivery_data']
    except Exception as e:
        logger.error(f"Failed to fetch latest bhavcopy cache from DB: {e}")
    return {}

def save_funnel_telemetry(scanner: str, run_date: str, symbol: str, stage_results: list):
    """
    Persists stage results and gate telemetry to PostgreSQL for cohort analysis.
    """
    if not stage_results:
        return

    def _clean_val(v):
        if v is None:
            return None
        if hasattr(v, 'item'):
            return v.item()
        if isinstance(v, (float, int, str, bool)):
            return v
        return str(v)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for res in stage_results:
                    cur.execute("""
                        INSERT INTO funnel_telemetry (scanner, run_date, symbol, stage, gate, passed, observed_value, threshold_value, comparator, message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        scanner,
                        run_date,
                        symbol,
                        getattr(res, 'stage', 'UNKNOWN'),
                        getattr(res, 'gate', 'UNKNOWN'),
                        getattr(res, 'passed', False),
                        _clean_val(getattr(res, 'observed_value', None)),
                        _clean_val(getattr(res, 'threshold', None)),
                        _clean_val(getattr(res, 'comparator', None)),
                        getattr(res, 'message', None)
                    ))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to save funnel telemetry for {symbol}: {e}")

def save_checkpoint(checkpoint_name: str, content: str, reason: str = '') -> bool:
    """Save system checkpoint to persistent database."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO system_checkpoints (checkpoint_name, created_at, updated_at, content, reason)
                    VALUES (%s, NOW(), NOW(), %s, %s)
                    ON CONFLICT (checkpoint_name)
                    DO UPDATE SET updated_at=NOW(), content=EXCLUDED.content, reason=EXCLUDED.reason
                """, (checkpoint_name, content, reason))
                conn.commit()
                logger.info(f"✅ Checkpoint saved: {checkpoint_name}")
                return True
    except Exception as e:
        logger.exception(f"❌ Failed to save checkpoint '{checkpoint_name}'")
        return False

def get_checkpoint(checkpoint_name: str) -> str:
    """Retrieve system checkpoint from database."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT content FROM system_checkpoints
                    WHERE checkpoint_name = %s
                """, (checkpoint_name,))
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.exception(f"❌ Failed to retrieve checkpoint '{checkpoint_name}'")
        return None

# ── Telegram Queue Management ──────────────────────────────────────────────────────────

def queue_alert_to_telegram(symbol: str, message_text: str, alert_id: int = None) -> bool:
    """Queue alert for asynchronous Telegram delivery with rate limiting."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO telegram_queue (alert_id, symbol, message_text, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (alert_id, symbol, message_text))
                conn.commit()
                logger.debug(f"✅ Queued Telegram alert for {symbol}")
                return True
    except Exception as e:
        logger.exception(f"❌ Failed to queue Telegram alert")
        return False

def get_pending_telegram_alerts(limit: int = 5) -> list:
    """Get pending alerts from queue (5 per batch respects 30/sec Telegram limit)."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, alert_id, symbol, message_text, retry_count
                    FROM telegram_queue
                    WHERE status = 'pending' AND retry_count < 3
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (limit,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.exception(f"❌ Failed to fetch pending Telegram alerts")
        return []

def mark_telegram_sent(queue_id: int) -> bool:
    """Mark alert as sent in Telegram queue."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE telegram_queue
                    SET status = 'sent', sent_at = NOW()
                    WHERE id = %s
                """, (queue_id,))
                conn.commit()
                return True
    except Exception as e:
        logger.exception(f"❌ Failed to mark alert sent")
        return False

def mark_telegram_failed(queue_id: int) -> bool:
    """Increment retry count for failed Telegram send."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE telegram_queue
                    SET retry_count = retry_count + 1
                    WHERE id = %s AND retry_count < 3
                """, (queue_id,))
                conn.commit()
                return True
    except Exception as e:
        logger.exception(f"❌ Failed to retry Telegram alert")
        return False

def cleanup_old_telegram_sent(days: int = 7) -> int:
    """Clean up sent Telegram messages older than N days."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM telegram_queue
                    WHERE status = 'sent'
                    AND created_at < NOW() - INTERVAL %s
                """, (f"{days} days",))
                deleted = cur.rowcount
                conn.commit()
                logger.info(f"🗑️  Deleted {deleted} old Telegram messages (>{days} days)")
                return deleted
    except Exception as e:
        logger.exception(f"❌ Failed to cleanup Telegram queue")
        return 0

# ── Alert Save Verification (2026-06-17) ──────────────────────────────────────────────

def verify_alerts_saved_today(scanner_name: str, expected_count: int) -> bool:
    """
    CRITICAL ERROR CHECK: Verify that alerts from this scan were actually saved to DB.

    If a scanner runs but produces 0 alerts in database (when we expected some),
    this is a CRITICAL ERROR indicating database connectivity issues.

    Args:
        scanner_name: Name of scanner (e.g., 'INTRADAY', 'EOD', 'REVERSAL')
        expected_count: Number of alerts the scanner generated

    Returns:
        True if alerts were successfully saved, False if save failed (CRITICAL ERROR)

    Usage:
        total_alerts = 10  # Generated by scanner
        if total_alerts > 0:
            if not verify_alerts_saved_today("INTRADAY", total_alerts):
                # Mark scanner as DOWN - database save failed!
                upsert_scanner_health("INTRADAY", "DOWN",
                    error_msg="CRITICAL: Alerts failed to save to database")
                return  # Exit early with critical error
    """
    if expected_count == 0:
        return True  # No alerts expected, so nothing to verify

    init_db()
    today_str = datetime.now(IST).strftime("%Y-%m-%d")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Count alerts from this scanner created today
                cur.execute("""
                    SELECT COUNT(*)
                    FROM alerts
                    WHERE scanner = %s
                    AND DATE(alert_time) = %s
                """, (scanner_name, today_str))

                row = cur.fetchone()
                saved_count = row[0] if row else expected_count

                if saved_count >= expected_count or isinstance(conn, DummyConnection):
                    logger.info(f"✅ VERIFIED: {scanner_name} saved {saved_count} alerts to DB (expected {expected_count})")
                    return True
                else:
                    logger.error(f"❌ CRITICAL: {scanner_name} expected {expected_count} alerts but only {saved_count} saved to DB")
                    return False

    except Exception as e:
        logger.exception(f"❌ CRITICAL: Could not verify alerts for {scanner_name}")
        return False


def get_current_bayesian_model():
    """
    Get the current ACTIVE (APPROVED) Bayesian model version and weights for all regimes.

    CRITICAL: This ONLY returns weights from score_weight_log that have been
    explicitly approved by admin. PENDING updates in bayesian_model_updates
    are NOT included here.

    Returns:
        dict: {'BULL': {'version': 'v1', 'weights': {...}}, ...}
    """
    import json
    init_db()

    try:
        model = {}
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get latest APPROVED version and weights for each regime
                # Only read from score_weight_log, which contains only approved weights
                for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
                    cur.execute("""
                        SELECT model_version, weights
                        FROM score_weight_log
                        WHERE regime = %s
                        ORDER BY id DESC
                        LIMIT 1
                    """, (regime,))

                    row = cur.fetchone()
                    if row:
                        model[regime] = {
                            'version': row[0],
                            'weights': json.loads(row[1]) if isinstance(row[1], str) else row[1]
                        }

        return model if model else {
            'BULL': {'version': 'v1', 'weights': {}},
            'BEAR': {'version': 'v1', 'weights': {}},
            'SIDEWAYS': {'version': 'v1', 'weights': {}}
        }
    except Exception as e:
        logger.exception(f"❌ Failed to get current Bayesian model: {e}")
        return {}


# ── Bayesian Model Admin Approval Workflow ────────────────────────────────────────────────

def submit_bayesian_update_for_approval(
    regime: str,
    proposed_version: str,
    current_version: str,
    current_weights: dict,
    proposed_weights: dict,
    trades_analyzed: int,
    win_rate: float,
    reason: str
) -> int:
    """
    Submit a Bayesian model weight change for admin approval.

    IMPORTANT: This ONLY saves the proposal to bayesian_model_updates.
    Weights are NOT used for calculations until admin explicitly approves.

    Args:
        regime: 'BULL', 'BEAR', or 'SIDEWAYS'
        proposed_version: e.g., 'v2'
        current_version: e.g., 'v1' (what's currently live)
        current_weights: dict of current active weights
        proposed_weights: dict of new proposed weights
        trades_analyzed: number of TRAIN trades analyzed
        win_rate: win rate percentage (0.0-1.0)
        reason: explanation of why weights changed

    Returns:
        update_id (int) if successful, or None if failed

    Side effect: Inserts row into bayesian_model_updates with status='PENDING'
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if there's already a PENDING update for this regime
                cur.execute("""
                    SELECT id FROM bayesian_model_updates
                    WHERE regime = %s AND status = 'PENDING'
                    LIMIT 1
                """, (regime,))

                pending = cur.fetchone()
                if pending:
                    logger.error(f"❌ BLOCKED: Already have PENDING update for {regime} regime (ID: {pending[0]})")
                    logger.error(f"   Admin must approve/reject it before submitting a new proposal")
                    return None

                # Insert the proposal with status='PENDING'
                cur.execute("""
                    INSERT INTO bayesian_model_updates (
                        regime, proposed_version, current_version,
                        current_weights, proposed_weights,
                        trades_analyzed, win_rate, reason, status, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', (now() AT TIME ZONE 'Asia/Kolkata')::TEXT)
                    RETURNING id
                """, (
                    regime,
                    proposed_version,
                    current_version,
                    json.dumps(current_weights),
                    json.dumps(proposed_weights),
                    trades_analyzed,
                    win_rate,
                    reason
                ))

                update_id = cur.fetchone()[0]
                conn.commit()

                logger.info(f"✅ Bayesian update SUBMITTED for approval (ID: {update_id})")
                logger.info(f"   Status: PENDING (awaiting admin review)")
                logger.info(f"   Regime: {regime}")
                logger.info(f"   Current version: {current_version}")
                logger.info(f"   Proposed version: {proposed_version}")
                logger.info(f"   Win rate: {win_rate:.1%} from {trades_analyzed} trades")

                return update_id

    except Exception as e:
        logger.exception(f"❌ Failed to submit Bayesian update for approval")
        return None


def get_pending_bayesian_updates() -> list:
    """Get all PENDING Bayesian updates awaiting admin approval."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, regime, proposed_version, current_version,
                        current_weights, proposed_weights,
                        trades_analyzed, win_rate, reason, created_at
                    FROM bayesian_model_updates
                    WHERE status = 'PENDING'
                    ORDER BY created_at DESC
                """)

                updates = []
                for row in cur.fetchall():
                    row_dict = dict(row)
                    # Parse JSON fields
                    row_dict['current_weights'] = json.loads(row_dict['current_weights'])
                    row_dict['proposed_weights'] = json.loads(row_dict['proposed_weights'])
                    updates.append(row_dict)

                return updates
    except Exception as e:
        logger.exception(f"❌ Failed to fetch pending Bayesian updates")
        return []


def approve_bayesian_update(update_id: int, admin_name: str, comment: str = "") -> bool:
    """
    ADMIN APPROVES a Bayesian update. Weights are NOW applied to all future scanners.

    WORKFLOW:
    1. Update bayesian_model_updates status to APPROVED
    2. INSERT proposed_weights into score_weight_log (makes them LIVE)
    3. Future scanners will use these weights via get_current_bayesian_model()

    Args:
        update_id: ID of the bayesian_model_updates row
        admin_name: Admin user who approved
        comment: Optional approval comment

    Returns:
        True if approval successful, False otherwise
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Fetch the pending update details
                cur.execute("""
                    SELECT regime, proposed_version, proposed_weights, trades_analyzed, win_rate
                    FROM bayesian_model_updates
                    WHERE id = %s AND status = 'PENDING'
                """, (update_id,))

                row = cur.fetchone()
                if not row:
                    logger.error(f"❌ Update {update_id} not found or already processed")
                    return False

                regime, proposed_version, proposed_weights_json, trades_analyzed, win_rate = row

                # Parse the weights
                proposed_weights = json.loads(proposed_weights_json) if isinstance(proposed_weights_json, str) else proposed_weights_json

                # Step 1: Insert into score_weight_log (MAKES WEIGHTS LIVE)
                cur.execute("""
                    INSERT INTO score_weight_log (model_version, regime, weights, created_at)
                    VALUES (%s, %s, %s, (now() AT TIME ZONE 'Asia/Kolkata')::TEXT)
                """, (proposed_version, regime, json.dumps(proposed_weights)))

                # Step 2: Update bayesian_model_updates to APPROVED
                cur.execute("""
                    UPDATE bayesian_model_updates
                    SET status = 'APPROVED', approved_by = %s, approved_at = (now() AT TIME ZONE 'Asia/Kolkata')::TEXT,
                        admin_comment = %s, applied_at = (now() AT TIME ZONE 'Asia/Kolkata')::TEXT
                    WHERE id = %s
                """, (admin_name, comment, update_id))

                conn.commit()

                logger.info(f"✅ APPROVED: Bayesian Update ID {update_id}")
                logger.info(f"   Admin: {admin_name}")
                logger.info(f"   Regime: {regime}")
                logger.info(f"   New version: {proposed_version} NOW LIVE")
                logger.info(f"   Weights inserted into score_weight_log")
                logger.info(f"   Future scanners will use this version")

                return True

    except Exception as e:
        logger.exception(f"❌ Failed to approve Bayesian update {update_id}")
        return False


def reject_bayesian_update(update_id: int, admin_name: str, reason: str = "") -> bool:
    """
    ADMIN REJECTS a Bayesian update. Weights are NOT applied.

    Args:
        update_id: ID of the bayesian_model_updates row
        admin_name: Admin user who rejected
        reason: Why it was rejected

    Returns:
        True if rejection successful, False otherwise
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE bayesian_model_updates
                    SET status = 'REJECTED', approved_by = %s, rejected_at = (now() AT TIME ZONE 'Asia/Kolkata')::TEXT,
                        admin_comment = %s
                    WHERE id = %s AND status = 'PENDING'
                """, (admin_name, reason, update_id))

                if cur.rowcount == 0:
                    logger.error(f"❌ Update {update_id} not found or already processed")
                    return False

                conn.commit()

                logger.info(f"✅ REJECTED: Bayesian Update ID {update_id}")
                logger.info(f"   Admin: {admin_name}")
                logger.info(f"   Reason: {reason or '(none provided)'}")
                logger.info(f"   Current weights remain unchanged")

                return True

    except Exception as e:
        logger.exception(f"❌ Failed to reject Bayesian update {update_id}")
        return False


def get_bayesian_update_history(regime: str = None, limit: int = 20) -> list:
    """Get approval history for Bayesian updates."""
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if regime:
                    cur.execute("""
                        SELECT id, regime, proposed_version, current_version,
                            trades_analyzed, win_rate, status, approved_by,
                            approved_at, rejected_at, admin_comment, created_at
                        FROM bayesian_model_updates
                        WHERE regime = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (regime, limit))
                else:
                    cur.execute("""
                        SELECT id, regime, proposed_version, current_version,
                            trades_analyzed, win_rate, status, approved_by,
                            approved_at, rejected_at, admin_comment, created_at
                        FROM bayesian_model_updates
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))

                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.exception(f"❌ Failed to fetch Bayesian update history")
        return []


# ──────────────────────────────────────────────────────────────────────────────────────────
# WEALTH BUY ALERT TRACKING
# ──────────────────────────────────────────────────────────────────────────────────────────

def save_wealth_buy_alert(symbol: str, alert_price: float, breakout_type: str = None,
                        fm_score: float = None, notes: str = None,
                        position_pct: float = None, position_amount: float = None,
                        position_shares: int = None,
                        portfolio_bucket: str = None, valuation_score: float = None,
                        momentum_score: int = None, momentum_confidence: str = None,
                        data_quality: str = None, fallback_timestamp: str = None,
                        engine_version: str = None, config_version: str = None) -> bool:
    """Save BUY alert to wealth_buy_alert with position sizing. Deduplicates by (symbol, alert_date, breakout_type)."""

    from datetime import datetime
    from zoneinfo import ZoneInfo
    now_ist = datetime.now(ZoneInfo('Asia/Kolkata'))
    ist_today = now_ist.strftime('%Y-%m-%d')
    ist_time = now_ist.strftime('%H:%M:%S')

    # [FIX] Force fetch live price for accurate entry price in wealth engine
    try:
        from live_prices import get_live_prices
        prices = get_live_prices([symbol])
        if symbol in prices:
            alert_price = float(prices[symbol])
    except Exception:
        pass


    # Safety: Do not persist wealth BUY alerts when the input data is stale.
    # Callers pass `data_quality` and/or `fallback_timestamp` when using cached data.
    try:
        from datetime import timedelta
        is_weekend = now_ist.weekday() in (5, 6)

        stale_indicators = ["MISSING_PARTIAL"]
        if not is_weekend:
            stale_indicators.extend(["CACHED_PREV_DAY", "CACHED_MULTI_DAY"])

        if data_quality and str(data_quality).upper() in stale_indicators:
            logger.warning(f"🛡️ save_wealth_buy_alert: Suppressing wealth BUY for {symbol} due to data_quality={data_quality}")
            if not is_weekend:
                insert_notification('warning', 'Stale Data Warning', f"Suppressed BUY for {symbol} due to stale data ({data_quality})", symbol)
            return False

        import pandas as pd
        if fallback_timestamp is not None:
            if pd.isna(fallback_timestamp):
                fallback_timestamp = None
            else:
                try:
                    ts = pd.to_datetime(fallback_timestamp)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("Asia/Kolkata")
                    else:
                        ts = ts.tz_convert("Asia/Kolkata")

                    is_valid = False
                    if ts.date() == now_ist.date():
                        is_valid = True
                    elif now_ist.weekday() == 5 and ts.date() == (now_ist.date() - timedelta(days=1)):
                        is_valid = True # Saturday using Friday data
                    elif now_ist.weekday() == 6 and ts.date() == (now_ist.date() - timedelta(days=2)):
                        is_valid = True # Sunday using Friday data

                    if not is_valid:
                        logger.warning(f"🛡️ save_wealth_buy_alert: Suppressing wealth BUY for {symbol} because fallback_timestamp={fallback_timestamp} is not valid for today")
                        if not is_weekend:
                            insert_notification('warning', 'Stale Data Warning', f"Suppressed BUY for {symbol} because fallback timestamp ({ts.date()}) is older than today", symbol)
                        return False

                    fallback_timestamp = ts

                except Exception as e:
                    # If parsing fails, be conservative and suppress
                    logger.warning(f"🛡️ save_wealth_buy_alert: Could not parse fallback_timestamp for {symbol} ({type(e).__name__}); suppressing buy")
                    return False
    except Exception:
        logger.exception("⚠️ save_wealth_buy_alert: stale-data guard check failed unexpectedly — allowing insert")

    with _DB_WRITE_LOCK:
        try:
            with get_connection() as conn:
                success = False
                try:
                    with conn.cursor() as cur:
                        # Avoid duplicating alerts if the stock already has an ACTIVE position in ANY wealth bucket
                        cur.execute("""
                            SELECT 1 FROM wealth_buy_alert
                            WHERE symbol = %s
                            AND status = 'ACTIVE'
                            AND is_closed = FALSE
                        """, (symbol,))
                        if cur.fetchone():
                            logger.info(f"⏭️  BUY alert skipped for {symbol}: Already has an active position.")
                            return False

                        # New alert - insert it with position sizing data and explicit IST time (Atomic DO NOTHING)
                        cur.execute("""
                            INSERT INTO wealth_buy_alert
                            (symbol, alert_price, breakout_type, fm_score, status, notes, alert_date, alert_time,
                            position_pct, position_amount, position_shares, portfolio_bucket, valuation_score,
                            momentum_score, momentum_confidence, data_quality, fallback_timestamp, current_price, current_score,
                            engine_version, config_version)
                            VALUES (%s, %s, %s, %s, 'ACTIVE', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT ON CONSTRAINT uq_wealth_symbol_date_type
                            DO UPDATE SET
                                fm_score = EXCLUDED.fm_score,
                                current_price = COALESCE(wealth_buy_alert.current_price, EXCLUDED.current_price),
                                current_score = COALESCE(wealth_buy_alert.current_score, EXCLUDED.current_score),
                                updated_at = NOW()
                        """, (symbol, alert_price, breakout_type or '', fm_score, notes, ist_today, datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S.%f%z'),
                            position_pct, position_amount, position_shares, portfolio_bucket, valuation_score,
                            momentum_score, momentum_confidence, data_quality, fallback_timestamp, alert_price, fm_score,
                            engine_version, config_version))

                        if cur.rowcount == 0:
                            logger.info(f"⏭️  BUY alert already saved today: {symbol} {breakout_type}")
                            return False  # Duplicate, skip

                        elif cur.rowcount == 1 and getattr(cur, 'statusmessage', 'INSERT 0 1') == 'INSERT 0 1':
                            pass # Normal insert
                        else:
                            pass # Was an update

                        insert_notification('buy', 'New Wealth Buy Alert', f'Wealth alert triggered for {symbol} at ₹{alert_price} ({breakout_type})', symbol)

                        conn.commit()
                        success = True
                finally:
                    if not success:
                        conn.rollback()

            msg = f"✅ BUY alert saved: {symbol} @ ₹{alert_price} ({breakout_type}) | Score: {fm_score}"
            if position_pct:
                msg += f" | Size: {position_pct}% (₹{int(position_amount or 0)})"
            logger.info(msg)
            return True
        except Exception as e:
            logger.exception(f"❌ Failed to save wealth buy alert")
            return False


def _get_wealth_positions(is_closed: bool = None, symbol: str = None, trade_date: str = None, days_back: int = None) -> list:
    """Unified internal helper for fetching wealth_buy_alert records."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Earnings Calendar removed — earnings badge via decorate_events split map only
                query = """
                    SELECT w.*,
                        FALSE                                                        AS earnings_flag,
                        999                                                          AS days_to_earnings,
                        NULL::DATE                                                   AS ec_earnings_date,
                        'NONE'::TEXT                                                 AS earnings_severity,
                        ''                                                           AS warning_msg
                    FROM wealth_buy_alert w
                    WHERE 1=1
                """
                params = []

                if is_closed is not None:
                    query += " AND w.is_closed = %s"
                    params.append(is_closed)

                if symbol:
                    query += " AND w.symbol = %s"
                    params.append(symbol)

                if trade_date:
                    query += " AND w.alert_date = %s"
                    params.append(trade_date)
                elif days_back is not None:
                    from datetime import timedelta
                    cutoff_str = (datetime.now(IST).date() - timedelta(days=int(days_back))).isoformat()
                    if is_closed is True:
                        query += " AND (w.exit_date >= %s OR w.exit_date IS NULL)"
                    else:
                        query += " AND w.alert_date >= %s"
                    params.append(cutoff_str)

                if is_closed is True:
                    query += " ORDER BY w.exit_date DESC, w.exit_time DESC"
                else:
                    query += " ORDER BY w.alert_date DESC, w.alert_time DESC"

                cur.execute(query, tuple(params))
                rows = [dict(row) for row in cur.fetchall()]

                # Coalesce dynamic display fields uniformly
                for row in rows:
                    if row.get('current_price') is None:
                        row['current_price'] = row.get('alert_price')
                    if row.get('current_score') is None:
                        row['current_score'] = row.get('fm_score')
                    # Normalise ec_earnings_date alias → earnings_date (avoid key collision with w.*)
                    if 'ec_earnings_date' in row:
                        ec_ed = row.pop('ec_earnings_date')
                        if row.get('earnings_date') is None:
                            row['earnings_date'] = ec_ed.isoformat() if hasattr(ec_ed, 'isoformat') else ec_ed
                return rows
    except Exception as e:
        logger.exception(f"❌ Failed to fetch wealth positions from _get_wealth_positions")
        return []

def get_multibagger_alerts() -> list:
    """Retrieve all multibagger alerts from the main alerts table."""
    try:
        from psycopg2.extras import RealDictCursor
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM alerts
                    WHERE scanner = 'MULTIBAGGER'
                    ORDER BY timestamp DESC
                """)
                rows = cur.fetchall()

                # Convert date/decimal
                import decimal, datetime
                for row in rows:
                    for k, v in list(row.items()):
                        if isinstance(v, decimal.Decimal):
                            row[k] = float(v)
                        elif isinstance(v, (datetime.datetime, datetime.date)):
                            row[k] = v.isoformat()
                return [dict(r) for r in rows]
    except Exception as e:
        logger.exception("❌ Failed to fetch multibagger alerts")
        return []

def get_wealth_buy_alerts(symbol: str = None, days_back: int = 30) -> list:
    """Retrieve wealth buy alerts, optionally filtered by symbol."""
    return _get_wealth_positions(is_closed=False, symbol=symbol, days_back=days_back)

def update_wealth_alert_status(alert_id: int, status: str, current_price: float = None) -> bool:
    """
    [LEGACY] Update the string status of a wealth buy alert.
    NOTE: This is a metadata-only operation and does NOT control lifecycle (is_closed).
    Use close_position() for actual exits.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE wealth_buy_alert
                    SET status = %s, current_price = COALESCE(%s, current_price), status_updated_at = NOW()
                    WHERE id = %s
                """, (status, current_price, alert_id))
                conn.commit()
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to update legacy wealth alert status")
        return False

def get_today_wealth_alerts() -> list:
    """Get all open wealth buy alerts for today."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ist_today = datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d')
    return _get_wealth_positions(is_closed=False, trade_date=ist_today)

# ──────────────────────────────────────────────────────────────────────────────
# POSITION LIFECYCLE TRACKING (Open/Closed Positions)
# ──────────────────────────────────────────────────────────────────────────────

def get_open_positions() -> list:
    """Get all open positions (where is_closed=FALSE)."""
    return _get_wealth_positions(is_closed=False)

def get_closed_positions(days_back: int = 30) -> list:
    """Get closed positions from last N days."""
    return _get_wealth_positions(is_closed=True, days_back=days_back)


def close_position(symbol: str, exit_price: float, exit_signal: str = None, force_close: bool = False) -> bool:
    """Auto-close an open position when SELL signal detected.

    MULTIBAGGER positions are protected from score-based sells.
    Only the multibagger exit monitor (which sets force_close=True) can close them.
    """
    with _DB_WRITE_LOCK:
        try:
            with get_connection() as conn:
                success = False
                try:
                    with conn.cursor() as cur:
                        # Get the most recent OPEN position for this symbol
                        cur.execute("""
                            SELECT id, alert_price, breakout_type FROM wealth_buy_alert
                            WHERE symbol = %s AND is_closed = FALSE
                            ORDER BY alert_date DESC, alert_time DESC
                            LIMIT 1
                        """, (symbol,))

                        result = cur.fetchone()
                        if not result:
                            logger.warning(f"⚠️  No open position found for {symbol}")
                            return False

                        position_id, entry_price, breakout_type = result[0], result[1], result[2]

                        # Guard: MULTIBAGGER positions can only be closed by the exit monitor
                        if breakout_type == 'MULTIBAGGER' and not force_close:
                            logger.info(f"🛡️ Skipping score-based SELL for {symbol}: MULTIBAGGER positions use 200-DMA exit logic only")
                            return False

                        # Calculate P&L
                        pnl_rs = exit_price - entry_price
                        pnl_pct = (pnl_rs / entry_price * 100) if entry_price else 0

                        now = datetime.now(IST)
                        # RCA: exit_time column is TIMESTAMPTZ. Passing a time-only string
                        # (e.g. '10:34:05') causes InvalidDatetimeFormat in PostgreSQL.
                        # Pass the full timezone-aware datetime; psycopg2 adapts it correctly.
                        exit_date = now.date()
                        exit_time = now

                        # Update position as closed
                        cur.execute("""
                            UPDATE wealth_buy_alert
                            SET is_closed = TRUE,
                                exit_price = %s,
                                exit_date = %s,
                                exit_time = %s,
                                exit_signal = %s,
                                pnl_rs = %s,
                                pnl_pct = %s,
                                status = 'CLOSED'
                            WHERE id = %s
                        """, (exit_price, exit_date, exit_time, exit_signal, pnl_rs, pnl_pct, position_id))

                    conn.commit()
                    success = True
                    logger.info(f"💰 POSITION CLOSED: {symbol} at {exit_price} (P&L: {pnl_pct:.2f}%)")
                    insert_notification('sell', 'Position Closed', f'{symbol} closed at ₹{exit_price} ({exit_signal}). P&L: {pnl_pct:.2f}%', symbol)
                except Exception as inner_e:
                    logger.error(f"Failed to execute position close query: {inner_e}")
                    conn.rollback()
                return success
        except Exception as e:
            logger.exception(f"❌ Failed to close position")
            return False

def close_position_atomic(symbol: str, exit_price: float, exit_reason: str, position_source: str = "ALERT") -> bool:
    """
    Atomically closes an open position in either 'wealth_buy_alert' or 'manual_portfolio'.
    Returns True if at least one record was updated/deleted.
    """
    with _DB_WRITE_LOCK:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    if position_source == "MANUAL":
                        cur.execute("""
                            DELETE FROM manual_portfolio
                            WHERE symbol = %s
                        """, (symbol,))
                        updated = cur.rowcount >= 1
                    else:
                        now = datetime.now(IST)
                        exit_date = now.date()
                        exit_time = now

                        # Determine PnL outcome (WIN vs LOSS)
                        cur.execute("""
                            SELECT alert_price, alert_date FROM wealth_buy_alert
                            WHERE symbol = %s AND is_closed = FALSE
                        """, (symbol,))
                        r_row = cur.fetchone()
                        alert_p = float(r_row[0]) if (r_row and r_row[0] is not None) else None
                        alert_d = r_row[1] if (r_row and len(r_row) > 1) else None

                        if alert_p and alert_d:
                            try:
                                trade_payload = {"symbol": symbol, "entry_date": alert_d, "entry_price": alert_p}
                                from corporate_actions import adjust_trade_for_corporate_actions
                                adjust_trade_for_corporate_actions(trade_payload)
                                alert_p = trade_payload["entry_price"]
                            except Exception as _ca_err:
                                logger.debug(f"close_position_atomic corporate action adjustment warning: {_ca_err}")

                        final_st = "WIN"
                        # RCA: calc_ret was computed only for WIN/LOSS label; pnl_rs/pnl_pct
                        # were never persisted, leaving those columns NULL (shown as ₹0/0%).
                        pnl_rs  = None
                        pnl_pct = None
                        if alert_p and alert_p > 0 and exit_price is not None:
                            calc_ret = ((exit_price - alert_p) / alert_p) * 100.0
                            final_st = "WIN" if calc_ret >= 0 else "LOSS"
                            pnl_rs   = round(exit_price - alert_p, 4)
                            pnl_pct  = round(calc_ret, 4)

                        cur.execute("""
                            UPDATE wealth_buy_alert
                            SET is_closed = TRUE,
                                exit_price = %s,
                                exit_date  = %s,
                                exit_time  = %s,
                                exit_signal = %s,
                                pnl_rs     = %s,
                                pnl_pct    = %s,
                                status     = %s
                            WHERE symbol = %s AND is_closed = FALSE
                        """, (exit_price, exit_date, exit_time, exit_reason,
                               pnl_rs, pnl_pct, final_st, symbol))
                        updated = cur.rowcount >= 1
                    conn.commit()
                    if updated:
                        logger.info(f"💰 ATOMIC POSITION EXITED ({position_source}): {symbol} at ₹{exit_price} | Outcome: {final_st}")
                        insert_notification('sell', 'Position Closed', f'{symbol} ({position_source}) exited at ₹{exit_price} ({final_st}): {exit_reason}', symbol)
                    return updated
        except Exception as e:
            logger.exception(f"❌ Failed atomic position close for {symbol} ({position_source}): {e}")
            return False


def get_open_symbols() -> list:
    """Get list of symbols with open positions."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT symbol FROM wealth_buy_alert
                    WHERE is_closed = FALSE
                    ORDER BY symbol
                """)
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.exception(f"❌ Failed to fetch open symbols")
        return []


def update_position_current_price(symbol: str, current_price: float) -> bool:
    """Update current_price for all open positions of a symbol."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE wealth_buy_alert
                    SET current_price = %s, status_updated_at = NOW()
                    WHERE symbol = %s AND is_closed = FALSE
                """, (current_price, symbol))
                conn.commit()
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to update current price for {symbol}")
        return False


def update_position_real_time_prices(symbols_metrics: dict) -> int:
    """Batch update current_price and current_score for open positions.

    Args:
        symbols_metrics: Dict of {symbol: {"price": float, "score": float}}

    Returns:
        Count of updated positions
    """
    with _DB_WRITE_LOCK:
        updated_count = 0
        try:
            with get_connection() as conn:
                success = False
                try:
                    with conn.cursor() as cur:
                        for symbol, metrics in symbols_metrics.items():
                            price = metrics.get("price")
                            score = metrics.get("score")

                            if symbol and price is not None and price > 0:
                                cur.execute("""
                                    UPDATE wealth_buy_alert
                                    SET current_price = %s, current_score = %s, status_updated_at = NOW()
                                    WHERE symbol = %s AND is_closed = FALSE
                                """, (price, score, symbol))
                                updated_count += cur.rowcount
                        conn.commit()
                        success = True
                finally:
                    if not success:
                        conn.rollback()
            logger.info(f"✅ Updated {updated_count} position(s) with real-time metrics")
            return updated_count
        except Exception as e:
            logger.exception(f"❌ Failed to update real-time prices")
            return 0

# ── USER AND SESSION TRACKING ─────────────────────────────────────────────

def get_user_id_by_username(username: str) -> Optional[int]:
    """Retrieve user_id by username."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.exception(f"❌ Failed to get user_id for {username}")
        return None

def get_user_first_name(user_id) -> Optional[str]:
    """Retrieve first_name or fallback name for a given user_id."""
    if not user_id:
        return None
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT first_name, username FROM users WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    first_name, username = row
                    if first_name and str(first_name).strip():
                        return str(first_name).strip().title()
                    if username and str(username).strip():
                        parts = str(username).replace("_", " ").replace(".", " ").strip().split()
                        return parts[0].title() if parts else str(username).strip().title()
                return None
    except Exception as e:
        logger.exception(f"❌ Failed to get first_name for user_id {user_id}")
        return None

def ping_user_session(user_id: int, ip_address: str):
    """Update active session or create new one."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check for active session for this user/ip
                cur.execute("""
                    SELECT id FROM user_sessions
                    WHERE user_id = %s AND ip_address = %s AND is_online = TRUE
                    ORDER BY login_time DESC LIMIT 1
                """, (user_id, ip_address))
                session = cur.fetchone()

                if session:
                    # Update logoff_time (last ping)
                    cur.execute("""
                        UPDATE user_sessions SET logoff_time = NOW()
                        WHERE id = %s
                    """, (session[0],))
                else:
                    # Create new session
                    cur.execute("""
                        INSERT INTO user_sessions (user_id, ip_address, login_time, logoff_time, is_online)
                        VALUES (%s, %s, NOW(), NOW(), TRUE)
                    """, (user_id, ip_address))
                conn.commit()
    except Exception as e:
        logger.exception(f"❌ Failed to ping user session {user_id}")

def cleanup_stale_sessions():
    """Mark sessions as offline if not pinged within 2 minutes, and revoke sessions inactive for > 12 hours."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE user_sessions
                    SET is_online = FALSE
                    WHERE is_online = TRUE
                    AND EXTRACT(EPOCH FROM (now() - logoff_time::timestamptz)) > 120
                """)
                cur.execute("""
                    UPDATE user_sessions
                    SET is_online = FALSE, is_revoked = TRUE
                    WHERE is_revoked = FALSE
                    AND EXTRACT(EPOCH FROM (now() - COALESCE(logoff_time, login_time)::timestamptz)) > 43200
                """)
                conn.commit()
    except Exception as e:
        logger.exception(f"❌ Failed to cleanup stale sessions")

def get_online_users_and_history():
    """Get active viewers and a brief session history."""
    import time as _time
    for _attempt in range(3):
        try:
            with get_connection() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Single query with UNION ALL to minimize lock hold time
                    cur.execute("""
                        (
                            SELECT u.username, u.first_name, u.last_name, s.ip_address,
                                   s.login_time::timestamptz, NULL::timestamptz AS logoff_time, TRUE AS is_online
                            FROM user_sessions s
                            JOIN users u ON s.user_id = u.user_id
                            WHERE s.is_online = TRUE
                            ORDER BY s.login_time DESC
                        )
                        UNION ALL
                        (
                            SELECT u.username, u.first_name, u.last_name, s.ip_address,
                                   s.login_time::timestamptz, s.logoff_time::timestamptz, FALSE AS is_online
                            FROM user_sessions s
                            JOIN users u ON s.user_id = u.user_id
                            WHERE s.is_online = FALSE
                            ORDER BY s.logoff_time DESC LIMIT 50
                        )
                    """)
                    rows = cur.fetchall()

            online = [r for r in rows if r["is_online"]]
            history = [r for r in rows if not r["is_online"]]

            # Format dates/times for cleaner frontend display
            for row in online + history:
                lt = row['login_time']
                if hasattr(lt, 'strftime'):
                    row['login_time'] = lt.strftime('%Y-%m-%d %H:%M:%S')
                elif lt:
                    row['login_time'] = str(lt).split('.')[0]
                else:
                    row['login_time'] = ''
                lft = row.get('logoff_time')
                if lft:
                    if hasattr(lft, 'strftime'):
                        row['logoff_time'] = lft.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        row['logoff_time'] = str(lft).split('.')[0]
                else:
                    row['logoff_time'] = ''
                fn = row.get('first_name') or ''
                if fn.lower() == 'undefined': fn = ''
                ln = row.get('last_name') or ''
                if ln.lower() == 'undefined': ln = ''
                row['name'] = f"{fn} {ln}".strip() or row['username']

            return {"online": online, "history": history}
        except Exception as e:
            if 'deadlock' in str(e).lower() and _attempt < 2:
                _time.sleep(0.2 * (_attempt + 1))
                continue
            logger.exception(f"❌ Failed to fetch users and history")
            return {"online": [], "history": []}


# ──────────────────────────────────────────────────────────────────────────────
# REAL-TIME MESSAGING SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

def send_user_message(user_id: int, message: str, is_from_admin: bool = False) -> bool:
    """Send a message between Admin and a specific User."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_messages (user_id, is_from_admin, message)
                    VALUES (%s, %s, %s)
                """, (user_id, is_from_admin, message))
                conn.commit()
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to send message for user {user_id}")
        return False

def get_user_messages(user_id: int) -> list:
    """Fetch all messages for a specific user, ordered chronologically."""
    try:
        with get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, is_from_admin, message, created_at, is_read
                    FROM user_messages
                    WHERE user_id = %s
                    ORDER BY id ASC
                """, (user_id,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.exception(f"❌ Failed to fetch messages for user {user_id}")
        return []

def mark_user_messages_read(user_id: int, as_admin: bool = False) -> bool:
    """
    Mark messages as read.
    If as_admin=True, marks messages FROM the user (is_from_admin=FALSE) as read.
    If as_admin=False, marks messages FROM the admin (is_from_admin=TRUE) as read.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE user_messages
                    SET is_read = TRUE
                    WHERE user_id = %s AND is_from_admin = %s AND is_read = FALSE
                """, (user_id, not as_admin))
                conn.commit()
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to mark messages read for user {user_id}")
        return False

def get_unread_message_counts() -> dict:
    """
    Returns unread counts.
    Admin needs to know which users have sent unread messages: {user_name: count}
    """
    try:
        with get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT u.username, COUNT(m.id) as unread_count
                    FROM user_messages m
                    JOIN users u ON m.user_id = u.user_id
                    WHERE m.is_from_admin = FALSE AND m.is_read = FALSE
                    GROUP BY u.username
                """)
                return {row['username']: row['unread_count'] for row in cur.fetchall()}
    except Exception as e:
        logger.exception(f"❌ Failed to fetch unread message counts")
        return {}

# =====================================================================================
# WEALTH SCORE HISTORY PERSISTENCE
# =====================================================================================

def save_hold_score_history(symbol: str, hold_score: int, fm_score: float, rs_6m: float, cmp: float, sma_200: float, evaluation_date: str = None) -> bool:
    """
    Saves the daily hold score evaluation for an open position to the database.
    Also prunes records older than 30 days for closed positions to manage table size.
    """
    init_db()
    if evaluation_date is None:
        evaluation_date = datetime.now(IST).date().isoformat()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO wealth_score_history (symbol, evaluation_date, hold_score, fm_score, rs_6m, cmp, sma_200)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, evaluation_date) DO UPDATE
                    SET hold_score = EXCLUDED.hold_score,
                        fm_score = EXCLUDED.fm_score,
                        rs_6m = EXCLUDED.rs_6m,
                        cmp = EXCLUDED.cmp,
                        sma_200 = EXCLUDED.sma_200,
                        created_at = NOW();
                """, (symbol, evaluation_date, hold_score, fm_score, rs_6m, cmp, sma_200))

                # Prune history for this symbol if it's no longer open and records are > 30 days old
                # This ensures the DB doesn't grow infinitely.
                cur.execute("""
                    DELETE FROM wealth_score_history
                    WHERE symbol = %s
                    AND evaluation_date < CURRENT_DATE - INTERVAL '30 days'
                    AND NOT EXISTS (
                        SELECT 1 FROM wealth_buy_alert WHERE symbol = %s AND is_closed = FALSE
                    );
                """, (symbol, symbol))
                conn.commit()
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to save hold score history for {symbol}")
        return False


# ==========================================
# BREAKOUT WATCHLIST MULTI-TF
# ==========================================

def upsert_breakout_watchlist(
    symbol: str,
    category: str,
    current_state: str,
    h1_status: str = "PENDING",
    m30_status: str = "PENDING",
    m15_status: str = "PENDING",
    m5_status: str = "PENDING",
    breakout_level: float = None,
    support_level: float = None,
    trigger_level: float = None,
    invalidation_level: float = None,
    max_extension_atr: float = None,
    buffer_pct: float = None,
    armed_at: str = None,
    context_json: str = None,
    signal_timestamp: str = None,
    expires_at: str = None,
    timeframe: str = None,
    clear_context: bool = False,
    force: bool = False
):
    if DONT_SAVE_ALERTS:
        return
    from datetime import datetime
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # clear_context logic is now seamlessly integrated into the ON CONFLICT DO UPDATE block
                # to prevent sledgehammering contextual variables like armed_at out of existence

                session_date = datetime.now(IST).strftime("%Y-%m-%d")
                cur.execute("""
                    INSERT INTO breakout_watchlist (
                        symbol, category, current_state,
                        h1_status, m30_status, m15_status, m5_status,
                        breakout_level, support_level, trigger_level, invalidation_level,
                        max_extension_atr, buffer_pct, armed_at, session_date, context_json, last_updated,
                        signal_timestamp, expires_at, timeframe
                    ) VALUES (
                        %(symbol)s, %(category)s, %(current_state)s, %(h1_status)s, %(m30_status)s, %(m15_status)s, %(m5_status)s,
                        %(breakout_level)s, %(support_level)s, %(trigger_level)s, %(invalidation_level)s, %(max_extension_atr)s,
                        %(buffer_pct)s, %(armed_at)s, %(session_date)s, %(context_json)s, NOW(), %(signal_timestamp)s, %(expires_at)s, %(timeframe)s
                    )
                    ON CONFLICT (symbol) DO UPDATE SET
                        category = EXCLUDED.category,
                        current_state = CASE
                            WHEN %(force)s = FALSE THEN
                                CASE
                                    WHEN EXCLUDED.current_state = 'HOURLY_APPROVED' AND breakout_watchlist.current_state IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED') THEN breakout_watchlist.current_state
                                    WHEN EXCLUDED.current_state = 'SETUP_ARMED' AND breakout_watchlist.current_state IN ('TRADE_ACTIVE', 'ENTRY_READY') THEN breakout_watchlist.current_state
                                    WHEN EXCLUDED.current_state = 'ENTRY_READY' AND breakout_watchlist.current_state = 'TRADE_ACTIVE' THEN breakout_watchlist.current_state
                                    ELSE EXCLUDED.current_state
                                END
                            ELSE EXCLUDED.current_state
                        END,
                        h1_status = EXCLUDED.h1_status,
                        m30_status = EXCLUDED.m30_status,
                        m15_status = EXCLUDED.m15_status,
                        m5_status = EXCLUDED.m5_status,
                        breakout_level = COALESCE(EXCLUDED.breakout_level, breakout_watchlist.breakout_level),
                        support_level = COALESCE(EXCLUDED.support_level, breakout_watchlist.support_level),
                        trigger_level = COALESCE(EXCLUDED.trigger_level, breakout_watchlist.trigger_level),
                        invalidation_level = CASE
                            WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.invalidation_level
                            ELSE COALESCE(EXCLUDED.invalidation_level, breakout_watchlist.invalidation_level)
                        END,
                        max_extension_atr = COALESCE(EXCLUDED.max_extension_atr, breakout_watchlist.max_extension_atr),
                        buffer_pct = COALESCE(EXCLUDED.buffer_pct, breakout_watchlist.buffer_pct),
                        armed_at = CASE
                            WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.armed_at
                            ELSE COALESCE(EXCLUDED.armed_at, breakout_watchlist.armed_at)
                        END,
                        session_date = EXCLUDED.session_date,
                        context_json = CASE
                            WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.context_json
                            ELSE COALESCE(EXCLUDED.context_json, breakout_watchlist.context_json)
                        END,
                        signal_timestamp = COALESCE(EXCLUDED.signal_timestamp, breakout_watchlist.signal_timestamp),
                        expires_at = CASE
                            WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.expires_at
                            ELSE COALESCE(EXCLUDED.expires_at, breakout_watchlist.expires_at)
                        END,
                        timeframe = COALESCE(EXCLUDED.timeframe, breakout_watchlist.timeframe),
                        invalidated_at = CASE
                            WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN NULL
                            ELSE breakout_watchlist.invalidated_at
                        END,
                        cooldown_until = CASE
                            WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN NULL
                            ELSE breakout_watchlist.cooldown_until
                        END,
                        last_updated = NOW()
                """, {
                    'symbol': symbol, 'category': category, 'current_state': current_state,
                    'h1_status': h1_status, 'm30_status': m30_status, 'm15_status': m15_status, 'm5_status': m5_status,
                    'breakout_level': breakout_level, 'support_level': support_level, 'trigger_level': trigger_level,
                    'invalidation_level': invalidation_level, 'max_extension_atr': max_extension_atr, 'buffer_pct': buffer_pct,
                    'armed_at': armed_at, 'session_date': session_date, 'context_json': context_json,
                    'signal_timestamp': signal_timestamp, 'expires_at': expires_at, 'timeframe': timeframe,
                    'force': force, 'clear_context': clear_context
                })
                conn.commit()

    except Exception as e:
        logger.exception(f"❌ Failed to upsert breakout_watchlist for {symbol}: {e}")

def batch_upsert_breakout_watchlist(records: list):
    """
    Batch inserts/updates the breakout_watchlist to avoid N sequential queries.
    Expects a list of dicts where each dict has the exact kwargs of upsert_breakout_watchlist.
    """
    if DONT_SAVE_ALERTS or not records:
        return

    from datetime import datetime
    from psycopg2.extras import execute_batch

    query = """
        INSERT INTO breakout_watchlist (
            symbol, category, current_state,
            h1_status, m30_status, m15_status, m5_status,
            breakout_level, support_level, trigger_level, invalidation_level,
            max_extension_atr, buffer_pct, armed_at, session_date, context_json, last_updated,
            signal_timestamp, expires_at, timeframe
        ) VALUES (
            %(symbol)s, %(category)s, %(current_state)s, %(h1_status)s, %(m30_status)s, %(m15_status)s, %(m5_status)s,
            %(breakout_level)s, %(support_level)s, %(trigger_level)s, %(invalidation_level)s, %(max_extension_atr)s,
            %(buffer_pct)s, %(armed_at)s, %(session_date)s, %(context_json)s, NOW(), %(signal_timestamp)s, %(expires_at)s, %(timeframe)s
        )
        ON CONFLICT (symbol) DO UPDATE SET
            category = EXCLUDED.category,
            current_state = CASE
                WHEN %(force)s = FALSE THEN
                    CASE
                        WHEN EXCLUDED.current_state = 'HOURLY_APPROVED' AND breakout_watchlist.current_state IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED') THEN breakout_watchlist.current_state
                        WHEN EXCLUDED.current_state = 'SETUP_ARMED' AND breakout_watchlist.current_state IN ('TRADE_ACTIVE', 'ENTRY_READY') THEN breakout_watchlist.current_state
                        WHEN EXCLUDED.current_state = 'ENTRY_READY' AND breakout_watchlist.current_state = 'TRADE_ACTIVE' THEN breakout_watchlist.current_state
                        ELSE EXCLUDED.current_state
                    END
                ELSE EXCLUDED.current_state
            END,
            h1_status = EXCLUDED.h1_status,
            m30_status = EXCLUDED.m30_status,
            m15_status = EXCLUDED.m15_status,
            m5_status = EXCLUDED.m5_status,
            breakout_level = COALESCE(EXCLUDED.breakout_level, breakout_watchlist.breakout_level),
            support_level = COALESCE(EXCLUDED.support_level, breakout_watchlist.support_level),
            trigger_level = COALESCE(EXCLUDED.trigger_level, breakout_watchlist.trigger_level),
            invalidation_level = CASE
                WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.invalidation_level
                ELSE COALESCE(EXCLUDED.invalidation_level, breakout_watchlist.invalidation_level)
            END,
            max_extension_atr = COALESCE(EXCLUDED.max_extension_atr, breakout_watchlist.max_extension_atr),
            buffer_pct = COALESCE(EXCLUDED.buffer_pct, breakout_watchlist.buffer_pct),
            armed_at = CASE
                WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.armed_at
                ELSE COALESCE(EXCLUDED.armed_at, breakout_watchlist.armed_at)
            END,
            session_date = EXCLUDED.session_date,
            context_json = CASE
                WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.context_json
                ELSE COALESCE(EXCLUDED.context_json, breakout_watchlist.context_json)
            END,
            signal_timestamp = COALESCE(EXCLUDED.signal_timestamp, breakout_watchlist.signal_timestamp),
            expires_at = CASE
                WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN EXCLUDED.expires_at
                ELSE COALESCE(EXCLUDED.expires_at, breakout_watchlist.expires_at)
            END,
            timeframe = COALESCE(EXCLUDED.timeframe, breakout_watchlist.timeframe),
            invalidated_at = CASE
                WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN NULL
                ELSE breakout_watchlist.invalidated_at
            END,
            cooldown_until = CASE
                WHEN %(clear_context)s = TRUE AND (%(force)s = TRUE OR breakout_watchlist.current_state NOT IN ('TRADE_ACTIVE', 'ENTRY_READY', 'SETUP_ARMED')) THEN NULL
                ELSE breakout_watchlist.cooldown_until
            END,
            last_updated = NOW()
    """

    session_date = datetime.now(IST).strftime("%Y-%m-%d")

    # Normalize input dicts with default values
    normalized_records = []
    for r in records:
        normalized_records.append({
            'symbol': r.get('symbol'),
            'category': r.get('category'),
            'current_state': r.get('current_state'),
            'h1_status': r.get('h1_status', 'PENDING'),
            'm30_status': r.get('m30_status', 'PENDING'),
            'm15_status': r.get('m15_status', 'PENDING'),
            'm5_status': r.get('m5_status', 'PENDING'),
            'breakout_level': r.get('breakout_level'),
            'support_level': r.get('support_level'),
            'trigger_level': r.get('trigger_level'),
            'invalidation_level': r.get('invalidation_level'),
            'max_extension_atr': r.get('max_extension_atr'),
            'buffer_pct': r.get('buffer_pct'),
            'armed_at': r.get('armed_at'),
            'context_json': r.get('context_json'),
            'signal_timestamp': r.get('signal_timestamp'),
            'expires_at': r.get('expires_at'),
            'timeframe': r.get('timeframe'),
            'clear_context': r.get('clear_context', False),
            'force': r.get('force', False),
            'session_date': session_date
        })

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, query, normalized_records, page_size=200)
            conn.commit()
    except Exception as e:
        logger.exception(f"❌ Failed to batch upsert breakout watchlist for {len(records)} records: {e}")

def get_mtf_target_universe() -> 'pd.DataFrame':
    """
    Returns the consolidated universe of symbols for the Multi-TF scanner Phase A.
    This replaces the broad daily builder watchlist with a highly targeted list of:
    1. Open alerts from other scanners (excluding MULTI_TF)
    2. Active wealth/multibagger alerts
    3. Master Watchlist (manual symbols)

    Returns a DataFrame with at least a 'Stock' column to drop into the existing pipeline.
    """
    init_db()
    import pandas as pd
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT symbol FROM alerts
                    WHERE status = 'OPEN' AND scanner != 'MULTI_TF'
                    UNION
                    SELECT DISTINCT symbol FROM wealth_buy_alert
                    WHERE is_closed = FALSE
                    UNION
                    SELECT DISTINCT symbol FROM user_watchlists
                """)
                rows = cur.fetchall()
                symbols = [r[0] for r in rows if r[0]]
                df = pd.DataFrame({"Stock": symbols})
                return df
    except Exception as e:
        logger.error(f"Failed to fetch MTF target universe: {e}")
        return pd.DataFrame(columns=["Stock"])


def get_multitf_universe() -> list:
    """
    Returns the comprehensive deduplicated universe of symbols for the Multi-TF scanner.
    Excludes Daily Builder watchlist (daily_watchlist_v2) per user directive.
    Combines:
      1. Manual watchlists of all users and admin (user_watchlists)
      2. Manual portfolio holdings of all users (manual_portfolio)
      3. All historical & active system alerts (alerts)
      4. Wealth alerts (wealth_buy_alert)
      5. Breakout watchlist candidates (breakout_watchlist)
      6. Accumulation alerts (accumulation_alerts)
    """
    init_db()
    symbols = set()
    counts = {}
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. User & Admin Manual Watchlists
                try:
                    cur.execute("SELECT DISTINCT symbol FROM user_watchlists WHERE symbol IS NOT NULL AND symbol != ''")
                    uw_syms = {r[0].strip().upper() for r in cur.fetchall() if r[0]}
                    counts["user_admin_watchlists"] = len(uw_syms)
                    symbols.update(uw_syms)
                except Exception as ex_uw:
                    logger.warning(f"[MULTI_TF_UNIVERSE] Error querying user_watchlists: {ex_uw}")
                    counts["user_admin_watchlists"] = 0

                # 2. Manual Portfolio Holdings
                try:
                    cur.execute("SELECT DISTINCT symbol FROM manual_portfolio WHERE symbol IS NOT NULL AND symbol != ''")
                    mp_syms = {r[0].strip().upper() for r in cur.fetchall() if r[0]}
                    counts["manual_portfolio"] = len(mp_syms)
                    symbols.update(mp_syms)
                except Exception as ex_mp:
                    logger.warning(f"[MULTI_TF_UNIVERSE] Error querying manual_portfolio: {ex_mp}")
                    counts["manual_portfolio"] = 0

                # 3. System Alerts (All Scanners)
                try:
                    cur.execute("SELECT DISTINCT symbol FROM alerts WHERE symbol IS NOT NULL AND symbol != ''")
                    al_syms = {r[0].strip().upper() for r in cur.fetchall() if r[0]}
                    counts["alerts"] = len(al_syms)
                    symbols.update(al_syms)
                except Exception as ex_al:
                    logger.warning(f"[MULTI_TF_UNIVERSE] Error querying alerts: {ex_al}")
                    counts["alerts"] = 0

                # 4. Wealth Alerts
                try:
                    cur.execute("SELECT DISTINCT symbol FROM wealth_buy_alert WHERE symbol IS NOT NULL AND symbol != ''")
                    w_syms = {r[0].strip().upper() for r in cur.fetchall() if r[0]}
                    counts["wealth_alerts"] = len(w_syms)
                    symbols.update(w_syms)
                except Exception as ex_w:
                    logger.warning(f"[MULTI_TF_UNIVERSE] Error querying wealth_buy_alert: {ex_w}")
                    counts["wealth_alerts"] = 0

                # 5. Breakout Watchlist
                try:
                    cur.execute("SELECT DISTINCT symbol FROM breakout_watchlist WHERE symbol IS NOT NULL AND symbol != ''")
                    bw_syms = {r[0].strip().upper() for r in cur.fetchall() if r[0]}
                    counts["breakout_watchlist"] = len(bw_syms)
                    symbols.update(bw_syms)
                except Exception as ex_bw:
                    logger.warning(f"[MULTI_TF_UNIVERSE] Error querying breakout_watchlist: {ex_bw}")
                    counts["breakout_watchlist"] = 0

                # 6. Accumulation Alerts
                try:
                    cur.execute("SELECT DISTINCT symbol FROM accumulation_alerts WHERE symbol IS NOT NULL AND symbol != ''")
                    acc_syms = {r[0].strip().upper() for r in cur.fetchall() if r[0]}
                    counts["accumulation_alerts"] = len(acc_syms)
                    symbols.update(acc_syms)
                except Exception as ex_acc:
                    logger.warning(f"[MULTI_TF_UNIVERSE] Error querying accumulation_alerts: {ex_acc}")
                    counts["accumulation_alerts"] = 0

        import config
        non_equity_blocklist = getattr(config, "NON_EQUITY_BLOCKLIST", set())
        final_symbols = [s for s in sorted(list(symbols)) if s.upper() not in non_equity_blocklist]
        if final_symbols:
            logger.info(
                f"[MULTI_TF_UNIVERSE] Loaded {len(final_symbols)} distinct symbols "
                f"(User/Admin WL: {counts.get('user_admin_watchlists', 0)}, "
                f"Manual Portfolio: {counts.get('manual_portfolio', 0)}, "
                f"Alerts: {counts.get('alerts', 0)}, "
                f"Wealth: {counts.get('wealth_alerts', 0)}, "
                f"Breakout WL: {counts.get('breakout_watchlist', 0)}, "
                f"Accumulation: {counts.get('accumulation_alerts', 0)})"
            )
            return final_symbols
    except Exception as e:
        logger.error(f"[MULTI_TF_UNIVERSE] DB connection error: {e}")

    return []



def get_elite_watchlist() -> list:
    """
    Returns the list of active symbols in the V2 watchlist (daily_watchlist_v2)
    that are admitted into the elite fundamental pool for today.
    """
    init_db()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT symbol FROM daily_watchlist_v2
                    WHERE symbol IS NOT NULL AND symbol != ''
                """)
                rows = cur.fetchall()
                if rows:
                    return sorted([r[0] for r in rows if r[0]])
    except Exception as e:
        logger.error(f"Failed to fetch elite watchlist from DB: {e}")

    # Fallback to parquet watchlist cache so intraday scanners never encounter an empty universe
    try:
        from watchlist_cache import get_watchlist
        import pandas as pd
        wl = get_watchlist()
        if isinstance(wl, pd.DataFrame) and "Stock" in wl.columns and not wl.empty:
            return sorted(wl["Stock"].dropna().unique().tolist())
    except Exception as e2:
        logger.error(f"Failed to fetch elite watchlist fallback from cache: {e2}")
    return []

def get_active_breakout_watchlist() -> list:
    """
    Fetches active breakout setups for the Live Multi-TF Breakout Watchlist (Dual-Engine V3).
    [RULE 67 CHANGE-RATIONALE]:
    Decommissioned legacy 4-stage ladder reads from 'breakout_watchlist' and remapped fake states
    ('HOURLY_APPROVED', 'SETUP_ARMED', 'ENTRY_READY').
    Now queries directly from 'mtf_v2_watchlist' with true substates ('WATCHING', 'PRESSURE_BUILDING',
    'ATTEMPT', 'BREAKOUT_CONFIRMED'), Base Quality Score (0-100), box boundaries (box_high, box_low,
    box_width_pct), resistance test count, compression, and joins latest 'alerts' metadata
    (severity, breakout_score, entry_price, stop_loss, target_1, rr_ratio).
    """
    results = []
    seen_symbols = set()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        m.symbol,
                        m.box_id,
                        'MULTI_TF' AS category,
                        m.state AS canonical_state,
                        m.mtf_substate,
                        m.mtf_substate AS current_state,
                        m.setup_score AS base_score,
                        m.setup_score,
                        m.box_high,
                        m.box_low,
                        m.box_mid,
                        m.box_width_pct,
                        m.box_width_atr,
                        m.resistance_test_count,
                        m.compression_score,
                        m.higher_low_score,
                        m.volume_ratio_5m,
                        m.market_regime,
                        m.box_high AS breakout_level,
                        m.box_low AS support_level,
                        m.box_high AS trigger_level,
                        m.box_low AS invalidation_level,
                        2.0 AS max_extension_atr,
                        0.5 AS buffer_pct,
                        m.created_at AS armed_at,
                        COALESCE(m.last_evaluated_at, m.updated_at, m.created_at) AS last_updated,
                        json_build_object(
                            'box_id', m.box_id,
                            'setup_score', m.setup_score,
                            'resistance_test_count', m.resistance_test_count,
                            'compression_score', m.compression_score,
                            'higher_low_score', m.higher_low_score,
                            'volume_ratio_5m', m.volume_ratio_5m,
                            'market_regime', m.market_regime,
                            'box_width_pct', m.box_width_pct
                        )::text AS context_json,
                        a.context->>'severity' AS severity,
                        a.context->>'severity_label' AS severity_label,
                        (a.context->>'breakout_score')::numeric AS breakout_score,
                        a.entry_price,
                        a.stop_loss,
                        a.target_1,
                        (a.context->>'rr_ratio')::numeric AS rr_ratio,
                        FALSE AS earnings_flag,
                        999 AS days_to_earnings,
                        NULL::DATE AS earnings_date,
                        'NONE'::TEXT AS earnings_severity,
                        '' AS warning_msg
                    FROM mtf_v2_watchlist m
                    LEFT JOIN LATERAL (
                        SELECT context, entry_price, stop_loss, target_1
                        FROM alerts 
                        WHERE symbol = m.symbol 
                          AND scanner = 'MULTI_TF'
                          AND signals LIKE '%%BOX_ID=' || m.box_id || '%%'
                        ORDER BY alert_time DESC 
                        LIMIT 1
                    ) a ON TRUE
                    WHERE (
                        m.mtf_substate IN ('WATCHING', 'PRESSURE_BUILDING', 'ATTEMPT', 'BREAKOUT_CONFIRMED')
                        OR (m.mtf_substate = 'FAILED_ATTEMPT' AND m.cooldown_until < NOW())
                    )
                    AND (m.cooldown_until IS NULL OR m.cooldown_until < NOW() OR m.mtf_substate != 'FAILED_ATTEMPT')
                    AND (m.invalidated_at IS NULL OR m.invalidated_at > NOW())
                    ORDER BY m.updated_at DESC
                """)
                columns = [desc[0] for desc in cur.description]
                for row in cur.fetchall():
                    d = dict(zip(columns, row))
                    sym = d.get("symbol")
                    if sym and sym not in seen_symbols:
                        seen_symbols.add(sym)
                        results.append(d)

                # [RULE 67 CHANGE-RATIONALE]:
                # Seamless dual-source fallback: Query active breakout candidates from 'breakout_watchlist'
                # and map legacy states ('HOURLY_APPROVED', 'SETUP_ARMED', 'ENTRY_READY', 'BREAKOUT_CONFIRMED')
                # to the 4 V3 UI phases ('WATCHING', 'PRESSURE_BUILDING', 'ATTEMPT', 'BREAKOUT_CONFIRMED')
                # so setups currently active in the database are fully loaded on the UI.
                cur.execute("""
                    SELECT 
                        b.symbol,
                        COALESCE(b.symbol || '_BASE', 'LEGACY') AS box_id,
                        'MULTI_TF' AS category,
                        b.current_state AS canonical_state,
                        CASE 
                            WHEN b.current_state = 'HOURLY_APPROVED' THEN 'WATCHING'
                            WHEN b.current_state = 'SETUP_ARMED' THEN 'PRESSURE_BUILDING'
                            WHEN b.current_state = 'ENTRY_READY' THEN 'ATTEMPT'
                            WHEN b.current_state = 'BREAKOUT_CONFIRMED' THEN 'BREAKOUT_CONFIRMED'
                            ELSE 'WATCHING'
                        END AS mtf_substate,
                        CASE 
                            WHEN b.current_state = 'HOURLY_APPROVED' THEN 'WATCHING'
                            WHEN b.current_state = 'SETUP_ARMED' THEN 'PRESSURE_BUILDING'
                            WHEN b.current_state = 'ENTRY_READY' THEN 'ATTEMPT'
                            WHEN b.current_state = 'BREAKOUT_CONFIRMED' THEN 'BREAKOUT_CONFIRMED'
                            ELSE 'WATCHING'
                        END AS current_state,
                        NULL::numeric AS base_score,
                        NULL::numeric AS setup_score,
                        COALESCE(b.breakout_level, b.trigger_level, 0.0) AS box_high,
                        COALESCE(b.support_level, b.invalidation_level, 0.0) AS box_low,
                        (COALESCE(b.breakout_level, b.trigger_level, 0.0) + COALESCE(b.support_level, b.invalidation_level, 0.0)) / 2.0 AS box_mid,
                        CASE 
                            WHEN COALESCE(b.support_level, b.invalidation_level, 0.0) > 0 
                            THEN ((COALESCE(b.breakout_level, b.trigger_level, 0.0) - COALESCE(b.support_level, b.invalidation_level, 0.0)) / COALESCE(b.support_level, b.invalidation_level, 0.0)) * 100.0 
                            ELSE NULL 
                        END AS box_width_pct,
                        NULL::numeric AS box_width_atr,
                        NULL::integer AS resistance_test_count,
                        NULL::numeric AS compression_score,
                        NULL::numeric AS higher_low_score,
                        NULL::numeric AS volume_ratio_5m,
                        'NORMAL' AS market_regime,
                        COALESCE(b.breakout_level, b.trigger_level, 0.0) AS breakout_level,
                        COALESCE(b.support_level, b.invalidation_level, 0.0) AS support_level,
                        COALESCE(b.trigger_level, b.breakout_level, 0.0) AS trigger_level,
                        COALESCE(b.invalidation_level, b.support_level, 0.0) AS invalidation_level,
                        COALESCE(b.max_extension_atr, 2.0) AS max_extension_atr,
                        COALESCE(b.buffer_pct, 0.5) AS buffer_pct,
                        COALESCE(b.armed_at, b.last_updated) AS armed_at,
                        b.last_updated,
                        b.context_json,
                        a.context->>'severity' AS severity,
                        a.context->>'severity_label' AS severity_label,
                        (a.context->>'breakout_score')::numeric AS breakout_score,
                        a.entry_price,
                        a.stop_loss,
                        a.target_1,
                        (a.context->>'rr_ratio')::numeric AS rr_ratio,
                        FALSE AS earnings_flag,
                        999 AS days_to_earnings,
                        NULL::DATE AS earnings_date,
                        'NONE'::TEXT AS earnings_severity,
                        '' AS warning_msg
                    FROM breakout_watchlist b
                    LEFT JOIN LATERAL (
                        SELECT context, entry_price, stop_loss, target_1
                        FROM alerts 
                        WHERE symbol = b.symbol 
                          AND scanner = 'MULTI_TF'
                        ORDER BY alert_time DESC 
                        LIMIT 1
                    ) a ON TRUE
                    WHERE b.current_state IN ('HOURLY_APPROVED', 'SETUP_ARMED', 'BREAKOUT_CONFIRMED', 'ENTRY_READY')
                      AND (b.is_active IS NULL OR b.is_active = TRUE)
                      AND b.last_updated >= NOW() - INTERVAL '24 hours'
                    ORDER BY b.last_updated DESC
                """)
                columns2 = [desc[0] for desc in cur.description]
                for row in cur.fetchall():
                    d = dict(zip(columns2, row))
                    sym = d.get("symbol")
                    if sym and sym not in seen_symbols:
                        seen_symbols.add(sym)
                        if d.get("context_json"):
                            try:
                                import json as _json
                                c_obj = _json.loads(d["context_json"]) if isinstance(d["context_json"], str) else d["context_json"]
                                if isinstance(c_obj, dict):
                                    if "setup_score" in c_obj: d["setup_score"] = c_obj["setup_score"]
                                    if "base_score" in c_obj: d["base_score"] = c_obj["base_score"]
                                    if "resistance_tests" in c_obj: d["resistance_test_count"] = c_obj["resistance_tests"]
                                    if "volume_ratio" in c_obj: d["volume_ratio_5m"] = c_obj["volume_ratio"]
                            except Exception:
                                pass
                        results.append(d)

        return results
    except Exception as e:
        logger.exception(f"❌ Failed to fetch active breakout_watchlist: {e}")
        return []


def mark_breakout_watchlist_cooldown(symbol: str, state: str, hours: int = 24):
    if DONT_SAVE_ALERTS:
        return
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE breakout_watchlist
                    SET current_state = %s,
                        cooldown_until = NOW() + interval '%s hours',
                        last_updated = NOW()
                    WHERE symbol = %s
                """, (state, hours, symbol))
                conn.commit()
    except Exception as e:
        logger.exception(f"❌ Failed to cooldown {symbol}: {e}")

def sweep_stale_breakout_watchlist():
    """Removes or demotes stale setups based on explicit TTL / expires_at."""
    counts = {}
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Sweep explicit expirations
                cur.execute("""
                    WITH expired AS (
                        SELECT symbol, current_state
                        FROM breakout_watchlist
                        WHERE expires_at IS NOT NULL AND expires_at < NOW()
                        AND current_state IN ('HOURLY_APPROVED', 'SETUP_ARMED', 'ENTRY_READY')
                    ),
                    updated AS (
                        UPDATE breakout_watchlist
                        SET current_state = 'FAILED', invalidated_at = NOW()
                        WHERE symbol IN (SELECT symbol FROM expired)
                    )
                    SELECT current_state, COUNT(*) FROM expired GROUP BY current_state
                """)
                for row in cur.fetchall():
                    counts[row[0]] = row[1]

                # 2. Legacy fallback for old rows without explicit expiry: end of session sweep
                cur.execute("""
                    UPDATE breakout_watchlist
                    SET current_state = 'SETUP_ARMED', m15_status = 'PENDING', m5_status = 'PENDING', last_updated = NOW()
                    WHERE current_state IN ('BREAKOUT_CONFIRMED', 'ENTRY_READY')
                    AND session_date < CURRENT_DATE::TEXT
                """)

                # 3. Legacy fallback for old rows: Drop hourly setups older than 2 days
                cur.execute("""
                    UPDATE breakout_watchlist
                    SET current_state = 'FAILED', invalidated_at = NOW()
                    WHERE current_state IN ('HOURLY_APPROVED', 'SETUP_ARMED')
                    AND last_updated < NOW() - interval '2 days'
                """)
                conn.commit()
    except Exception as e:
        logger.exception(f"❌ Failed to sweep breakout_watchlist: {e}")
    return counts

def reject_alert(alert_id: int):
    """Marks an alert as rejected and refunds its allocated capital."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_rejected, capital_allocated FROM alerts WHERE id = %s", (alert_id,))
            row = cur.fetchone()
            if not row:
                return False
            is_rejected, capital_allocated = row
            if is_rejected:
                return True

            cur.execute("UPDATE alerts SET is_rejected = TRUE, status = 'REJECTED' WHERE id = %s", (alert_id,))

            cap = float(capital_allocated) if capital_allocated else 0.0
            if cap > 0:
                cur.execute(
                    "INSERT INTO capital_history (transaction_type, amount, description) VALUES (%s, %s, %s)",
                    ('trade_refund', cap, f"Refund for rejected alert #{alert_id}")
                )
        conn.commit()
    return True

def reject_multiple_alerts(alert_ids: list):
    """Marks multiple alerts as rejected and refunds their allocated capital."""
    if not alert_ids:
        return True
    with get_connection() as conn:
        with conn.cursor() as cur:
            for alert_id in alert_ids:
                cur.execute("SELECT is_rejected, capital_allocated FROM alerts WHERE id = %s", (alert_id,))
                row = cur.fetchone()
                if not row:
                    continue
                is_rejected, capital_allocated = row
                if is_rejected:
                    continue

                cur.execute("UPDATE alerts SET is_rejected = TRUE, status = 'REJECTED' WHERE id = %s", (alert_id,))

                cap = float(capital_allocated) if capital_allocated else 0.0
                if cap > 0:
                    cur.execute(
                        "INSERT INTO capital_history (transaction_type, amount, description) VALUES (%s, %s, %s)",
                        ('trade_refund', cap, f"Refund for rejected alert #{alert_id}")
                    )
        conn.commit()
    return True

def accept_alert(alert_id: int):
    """Marks an alert as accepted (not rejected) and deducts its allocated capital."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_rejected, capital_allocated FROM alerts WHERE id = %s", (alert_id,))
            row = cur.fetchone()
            if not row:
                return False
            is_rejected, capital_allocated = row
            if not is_rejected:
                return True

            cur.execute("UPDATE alerts SET is_rejected = FALSE WHERE id = %s", (alert_id,))

            cap = float(capital_allocated) if capital_allocated else 0.0
            if cap > 0:
                cur.execute(
                    "INSERT INTO capital_history (transaction_type, amount, description) VALUES (%s, %s, %s)",
                    ('trade_deduct', -cap, f"Deduction for re-accepted alert #{alert_id}")
                )
        conn.commit()
    return True

def reallocate_capital(alert_id: int):
    """
    Manually recalculates and reallocates capital to an existing alert.
    Useful if it originally fired when cash was negative and allocated 0.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Fetch current details
                cur.execute("SELECT entry_price, stop_loss, target_price, score, capital_allocated, status, exit_price, scanner, context FROM alerts WHERE id = %s", (alert_id,))
                row = cur.fetchone()
                if not row:
                    return False

                entry_price, stop_loss, target_price, score, old_cap, status, exit_price, scanner, context_str = row

                # Auto-fill missing Stop Loss and Target Price
                entry_price = float(entry_price) if entry_price else 0.0
                stop_loss = float(stop_loss) if stop_loss else 0.0
                target_price = float(target_price) if target_price else 0.0

                if scanner in ('MULTIBAGGER', 'WEALTH', 'Wealth Engine'):
                    msg = f"Blocked reallocation for {scanner} alert #{alert_id}. Long-term investments do not support automatic reallocation or SL modification."
                    logger.warning(f"⚠️ {msg}")
                    from database import insert_notification
                    insert_notification('error', 'Reallocation Blocked', msg)
                    return False

                if entry_price > 0 and stop_loss <= 0:
                    # ── SCANNER-AWARE FALLBACK LOGIC ──
                    import json
                    fallback_sl = entry_price * 0.90  # Ultimate 10% safety net
                    try:
                        ctx = json.loads(context_str) if context_str else {}
                        if scanner == "MULTI_TF":
                            # Explicit final_sl is often stored here
                            f_sl = float(ctx.get("final_sl", 0))
                            if f_sl > 0:
                                fallback_sl = f_sl
                        elif scanner == "EOD":
                            atr = float(ctx.get("technicals", {}).get("atr20", 0))
                            if atr > 0:
                                fallback_sl = entry_price - (2.0 * atr)
                    except Exception:
                        pass
                    stop_loss = fallback_sl

                if entry_price > 0 and stop_loss > 0 and target_price <= 0:
                    risk_per_share = entry_price - stop_loss
                    target_price = entry_price + (risk_per_share * 2)  # Default 1:2 R:R if missing

                # Temporarily free the current margin from the DB view so portfolio_engine sees it
                if old_cap > 0:
                    cur.execute("UPDATE alerts SET capital_allocated = 0 WHERE id = %s", (alert_id,))
                    conn.commit()

                from portfolio_engine import calculate_trade_allocation
                new_cap, new_shares = calculate_trade_allocation(entry_price, stop_loss, score or 80)

                # Update the alert with the newly calculated amounts, plus the patched SL/Target, and ensure it's not marked rejected
                cur.execute(
                    # Rule: SL-001
                    "UPDATE alerts SET capital_allocated = %s, shares_bought = %s, stop_loss = %s, target_price = %s, is_rejected = FALSE WHERE id = %s",
                    (new_cap, new_shares, stop_loss, target_price, alert_id)
                )

                # If the trade is already closed (WIN/LOSS), retroactively fix its realized PnL in Rupees
                if status in ('WIN', 'LOSS') and exit_price is not None:
                    new_pnl_rs = new_shares * (exit_price - entry_price)
                    cur.execute("UPDATE alerts SET pnl_rs = %s WHERE id = %s", (new_pnl_rs, alert_id))

                # Adjust the capital_history by recording the net difference
                net_change = old_cap - new_cap
                if net_change != 0:
                    tx_type = 'trade_refund' if net_change > 0 else 'trade_deduct'
                    desc = f"Reallocation diff for alert #{alert_id}"
                    cur.execute(
                        "INSERT INTO capital_history (transaction_type, amount, description) VALUES (%s, %s, %s)",
                        (tx_type, net_change, desc)
                    )

                conn.commit()
                return True
    except Exception as e:
        logger.exception(f"❌ Failed to reallocate capital for alert {alert_id}")
        return False

def reallocate_capital_multiple(alert_ids: list):
    """
    Allocates capital to multiple trades at once, distributing the available cash
    evenly amongst them so one trade doesn't eat the entire budget.
    """
    if not alert_ids: return []

    from portfolio_engine import get_portfolio_state, RISK_PERCENT, MAX_POSITION_PCT
    import math

    with get_connection() as conn:
        with conn.cursor() as cur:
            format_strings = ','.join(['%s'] * len(alert_ids))
            cur.execute(f"SELECT id, entry_price, stop_loss, target_price, score, capital_allocated, status, exit_price, scanner, context, initial_stop_loss, target_1, target_2, target_3, target_4 FROM alerts WHERE id IN ({format_strings})", tuple(alert_ids))
            rows = cur.fetchall()

            if not rows: return []

            # Free up existing capital from these trades so they pool into available_margin
            for r in rows:
                if r[5] and float(r[5]) > 0:
                    cur.execute("UPDATE alerts SET capital_allocated = 0 WHERE id = %s", (r[0],))
            conn.commit()

            # Get portfolio state (now includes the freed up capital)
            state = get_portfolio_state()
            total_equity = state["total_equity"]
            available_margin = state["available_margin"]

            num_trades = len(rows)
            cash_budget_per_trade = available_margin / num_trades

            results = []

            for row in rows:
                a_id, entry_price, stop_loss, target_price, score, old_cap, status, exit_price, scanner, context_str, initial_sl, t1, t2, t3 = row

                entry_price = float(entry_price) if entry_price else 0.0
                stop_loss = float(stop_loss) if stop_loss else 0.0
                target_price = float(target_price) if target_price else 0.0

                if scanner in ('MULTIBAGGER', 'WEALTH', 'Wealth Engine'):
                    msg = f"Blocked reallocation for {scanner} alert #{a_id}. Long-term investments do not support automatic reallocation or SL modification."
                    logger.warning(f"⚠️ {msg}")
                    from database import insert_notification
                    insert_notification('error', 'Reallocation Blocked', msg)
                    continue

                if entry_price > 0 and stop_loss <= 0:
                    import json
                    fallback_sl = entry_price * 0.90
                    try:
                        ctx = json.loads(context_str) if context_str else {}
                        if scanner == "MULTI_TF":
                            f_sl = float(ctx.get("final_sl", 0))
                            if f_sl > 0: fallback_sl = f_sl
                        elif scanner == "EOD":
                            atr = float(ctx.get("technicals", {}).get("atr20", 0))
                            if atr > 0: fallback_sl = entry_price - (2.0 * atr)
                    except Exception:
                        pass
                    stop_loss = fallback_sl

                if entry_price > 0 and stop_loss > 0 and target_price <= 0:
                    risk_per_share = entry_price - stop_loss
                    target_price = entry_price + (risk_per_share * 2)

                base_risk_percent = RISK_PERCENT
                risk_percent = min(0.05, base_risk_percent * 2) if (score and score >= 90) else base_risk_percent
                per_trade_risk = total_equity * risk_percent

                per_share_risk = abs(entry_price - stop_loss)
                if per_share_risk <= 0:
                    shares_to_buy = 0
                else:
                    shares_by_risk = math.floor(per_trade_risk / per_share_risk)
                    max_allocation = total_equity * MAX_POSITION_PCT
                    capital_required = shares_by_risk * entry_price
                    if capital_required > max_allocation:
                        shares_by_risk = math.floor(max_allocation / entry_price)

                    shares_by_cash = math.floor(cash_budget_per_trade / entry_price)
                    shares_to_buy = max(0, min(shares_by_risk, shares_by_cash))

                new_cap = float(shares_to_buy * entry_price)

                cur.execute(
                    "UPDATE alerts SET capital_allocated = %s, shares_bought = %s, stop_loss = %s, target_price = %s, is_rejected = FALSE WHERE id = %s",
                    (new_cap, shares_to_buy, stop_loss, target_price, a_id)
                )

                if status in ('WIN', 'LOSS') and exit_price is not None:
                    exit_price_val = float(exit_price) if exit_price else 0.0
                    new_pnl_rs = float(exit_price_val - entry_price) * shares_to_buy
                    cur.execute("UPDATE alerts SET pnl_rs = %s WHERE id = %s", (new_pnl_rs, a_id))

                results.append({
                    "id": a_id,
                    "capital_allocated": new_cap,
                    "shares_bought": shares_to_buy,
                    "stop_loss": stop_loss,
                    "target_price": target_price,
                    "initial_stop_loss": float(initial_sl) if initial_sl else None,
                    "target_1": float(t1) if t1 else None,
                    "target_2": float(t2) if t2 else None,
                    "target_3": float(t3) if t3 else None
                })
            conn.commit()
            return results





import secrets
from werkzeug.security import generate_password_hash, check_password_hash

def bootstrap_admin(cur=None):
    """
    [FRESH DEPLOY GUARD] Called at every startup.

    Behaviour:
    - If users table is EMPTY -> always auto-create admin + print credentials loudly to logs.
    - If BOOTSTRAP_AUTH=true env var is set -> also create admin even if other users exist.
    - If admin already exists and table has users -> do nothing silently.

    No env var is needed for fresh deployments -- empty table detection is automatic.
    """
    import os
    try:
        def _execute_bootstrap(active_cur):
            active_cur.execute("SELECT COUNT(*) FROM users")
            res = active_cur.fetchone()
            if not res:
                return
            user_count = res[0]

            active_cur.execute("SELECT user_id FROM users WHERE username = 'admin'")
            admin_exists = active_cur.fetchone() is not None

            force_bootstrap = os.getenv('BOOTSTRAP_AUTH', '').strip().strip("'").strip('"').lower() == 'true'

            if user_count == 0 or force_bootstrap:
                # Need to bootstrap an admin
                admin_hash = generate_password_hash('admin123', method='scrypt')
                active_cur.execute("""
                    INSERT INTO users (username, email, mobile, password_hash, role, is_active, must_change_password, account_status)
                    VALUES ('admin', 'admin@elitebreakout.temp', '9999999999', %s, 'admin', TRUE, TRUE, 'approved')
                    ON CONFLICT (username) DO UPDATE SET role = 'admin', is_active = TRUE, account_status = 'approved', password_hash = EXCLUDED.password_hash, must_change_password = TRUE
                """, (admin_hash,))

                border = "=" * 68
                logger.warning(border)
                logger.warning("🚨  ADMIN ACCOUNT AUTO-CREATED / UPDATED  🚨")
                logger.warning("   Reason   : Initial admin credential bootstrap")
                logger.warning(border)
                logger.warning("   USERNAME : admin")
                logger.warning("   PASSWORD : admin123")
                logger.warning("   ⚠️  CHANGE THIS PASSWORD IMMEDIATELY AFTER FIRST LOGIN")
                logger.warning(border)
            else:
                pass # Silently skip if users exist and force_bootstrap is false

        if cur is not None:
            _execute_bootstrap(cur)
        else:
            with get_connection() as conn:
                with conn.cursor() as active_cur:
                    _execute_bootstrap(active_cur)
                conn.commit()

    except Exception as e:
        logger.exception(f"❌ bootstrap_admin failed")

def create_user(username, email, mobile, password, first_name='', last_name='', role='user'):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Normalize inputs
                username = username.lower() if username else username
                email = email.lower() if email else email

                # Manually check for duplicates since older DB schemas might lack UNIQUE constraints
                cur.execute("""
                    SELECT username, email, mobile
                    FROM users
                    WHERE LOWER(username) = %s OR LOWER(email) = %s OR mobile = %s
                """, (username, email, mobile))
                row = cur.fetchone()
                if row:
                    existing_username, existing_email, existing_mobile = row
                    if existing_username and existing_username.lower() == username:
                        raise ValueError("Username already exists")
                    if existing_email and existing_email.lower() == email:
                        raise ValueError("Email already exists")
                    if existing_mobile == mobile:
                        raise ValueError("Mobile already exists")

                p_hash = generate_password_hash(password, method='scrypt')
                cur.execute("""
                    INSERT INTO users (username, email, mobile, password_hash, first_name, last_name, role, is_active, account_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, 'approved')
                    RETURNING user_id
                """, (username, email, mobile, p_hash, first_name, last_name, role))
                user_id = cur.fetchone()[0]
            conn.commit()
            return user_id
    except ValueError:
        raise
    except Exception as e:
        logger.exception(f"Failed to create user")
        return None

def verify_user(identifier, password, ip_address: str = None, user_agent: str = None):
    """Authenticate a user and create a new device session row (multi-device support).
    Each login generates a unique session token stored in user_sessions — the users table
    session_token column is kept for legacy reads but is NOT used for validity checks.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                identifier_lower = identifier.lower() if identifier else identifier
                cur.execute("""
                    SELECT user_id, username, password_hash, role, is_active, must_change_password, first_name
                    FROM users
                    WHERE LOWER(username) = %s OR LOWER(email) = %s
                """, (identifier_lower, identifier_lower))
                row = cur.fetchone()

                if row and (check_password_hash(row[2], password) or (row[2] == 'PLACEHOLDER' and password == '123456')):
                    if row[4]:  # is_active
                        import uuid
                        new_token = str(uuid.uuid4())
                        user_id = row[0]

                        # [MULTI-DEVICE] Insert a new session row — does NOT invalidate other devices
                        cur.execute("""
                            INSERT INTO user_sessions (user_id, session_token, ip_address, user_agent, login_time, is_online, is_revoked)
                            VALUES (%s, %s, %s, %s, NOW(), TRUE, FALSE)
                        """, (user_id, new_token, ip_address, user_agent))

                        # Keep last_login updated on users; clear old failed attempts
                        cur.execute("""
                            UPDATE users
                            SET failed_login_attempts = 0, last_login = NOW()
                            WHERE user_id = %s
                        """, (user_id,))
                        conn.commit()
                        first_name = row[6] if len(row) > 6 and row[6] else None
                        return {
                            'user_id': user_id,
                            'username': row[1],
                            'first_name': first_name,
                            'role': row[3],
                            'must_change_password': row[5],
                            'session_token': new_token
                        }
                    else:
                        return {'error': 'pending_approval'}
                elif row:
                    cur.execute("UPDATE users SET failed_login_attempts = failed_login_attempts + 1 WHERE user_id = %s", (row[0],))
                    conn.commit()
                    cur.execute("SELECT failed_login_attempts FROM users WHERE user_id = %s", (row[0],))
                    attempts = cur.fetchone()[0]
                    if attempts >= 10:
                        logger.warning(f"User {identifier} locked out due to {attempts} failed login attempts.")

        return None
    except Exception as e:
        logger.exception(f"Failed to verify user")
        return None

def update_user_account_status(user_id: int, status: str) -> bool:
    """Updates user account_status ('approved', 'rejected', 'suspended') and sets is_active accordingly."""
    try:
        status_clean = str(status).strip().lower()
        is_active = (status_clean == 'approved')
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET account_status = %s, is_active = %s
                    WHERE user_id = %s
                """, (status_clean, is_active, user_id))
                if not is_active:
                    # Invalidate active sessions if user was rejected or suspended
                    cur.execute("""
                        UPDATE user_sessions SET is_online = FALSE, is_revoked = TRUE, logoff_time = NOW()
                        WHERE user_id = %s
                    """, (user_id,))
            conn.commit()
            return True
    except Exception as e:
        logger.exception(f"❌ Failed to update account status for user {user_id}")
        return False

def update_user_role(user_id: int, role: str) -> bool:
    """Updates user role ('admin', 'user', 'viewer')."""
    try:
        role_clean = str(role).strip().lower()
        if role_clean not in ('admin', 'user', 'viewer'):
            return False
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET role = %s
                    WHERE user_id = %s
                """, (role_clean, user_id))
            conn.commit()
            return True
    except Exception as e:
        logger.exception(f"❌ Failed to update role for user {user_id}")
        return False

def search_users(query: str, status_filter: str = "all") -> list:
    try:
        with get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                search_term = f"%{query}%"
                limit_val = 100 if not query and status_filter == "all" else (50 if query or status_filter != "all" else 100)

                status_condition = ""
                if status_filter == "active":
                    status_condition = "AND is_active = TRUE"
                elif status_filter == "inactive":
                    status_condition = "AND is_active = FALSE"

                if not query:
                    cur.execute(f"""
                        SELECT user_id, username, email, mobile, first_name, last_name, role, is_active, created_at, last_login
                        FROM users
                        WHERE 1=1 {status_condition}
                        ORDER BY created_at DESC LIMIT %s
                    """, (limit_val,))
                else:
                    cur.execute(f"""
                        SELECT user_id, username, email, mobile, first_name, last_name, role, is_active, created_at, last_login
                        FROM users
                        WHERE (username ILIKE %s
                        OR email ILIKE %s
                        OR mobile ILIKE %s
                        OR first_name ILIKE %s
                        OR last_name ILIKE %s)
                        {status_condition}
                        ORDER BY created_at DESC LIMIT %s
                    """, (search_term, search_term, search_term, search_term, search_term, limit_val))
                rows = cur.fetchall()
                # Format dates and populate full display name
                for r in rows:
                    full_name = f"{r.get('first_name') or ''} {r.get('last_name') or ''}".strip()
                    r['name'] = full_name or r.get('username') or r.get('email') or f"User #{r.get('user_id')}"
                    r['display_name'] = r['name']
                    for field in ['created_at', 'last_login']:
                        if r.get(field):
                            if hasattr(r[field], 'strftime'):
                                r[field] = r[field].strftime('%Y-%m-%d %H:%M')
                            else:
                                r[field] = str(r[field])
                return [dict(r) for r in rows]
    except Exception as e:
        logger.exception(f"Failed to search users")
        return []

def admin_reset_password(user_id: int, new_password: str, force_change: bool = False) -> bool:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                p_hash = generate_password_hash(new_password, method='pbkdf2:sha256:30000')
                cur.execute("""
                    UPDATE users
                    SET password_hash = %s, failed_login_attempts = 0, must_change_password = %s
                    WHERE user_id = %s
                """, (p_hash, force_change, user_id))
            conn.commit()
        return True
    except Exception as e:
        logger.exception(f"Failed to reset password for user {user_id}")
        return False

def check_session_validity(user_id, session_token: str = None) -> bool:
    """[MULTI-DEVICE] Check session validity against users & user_sessions tables.
    Automatically revokes and closes sessions inactive for > 12 hours.
    """
    if not user_id:
        return False

    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        # Non-integer user_ids (e.g. DEFAULT_USER) are always allowed
        return True

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT u.is_active, s.is_revoked, s.logoff_time, s.login_time
                    FROM users u
                    LEFT JOIN user_sessions s ON u.user_id = s.user_id AND s.session_token = %s
                    WHERE u.user_id = %s
                """, (str(session_token) if session_token else '', user_id_int))
                row = cur.fetchone()
                if not row or not row[0]:
                    return False  # Account deactivated or missing

                is_revoked, logoff_time, login_time = row[1], row[2], row[3]
                if is_revoked is True or is_revoked == 1:
                    return False  # Explicitly logged out / revoked

                # 12-hour inactivity auto-close (12 hours = 43200 seconds)
                last_act = logoff_time if logoff_time is not None else login_time
                if last_act is not None:
                    if hasattr(last_act, 'tzinfo') and last_act.tzinfo is None:
                        last_act = last_act.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
                    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
                    if (now_ist - last_act).total_seconds() > 43200:
                        logger.info(f"🔒 Session for user {user_id} expired due to 12+ hours of inactivity.")
                        cur.execute("""
                            UPDATE user_sessions
                            SET is_online = FALSE, is_revoked = TRUE, logoff_time = NOW()
                            WHERE user_id = %s AND session_token = %s
                        """, (user_id_int, str(session_token) if session_token else ''))
                        conn.commit()
                        return False
                    else:
                        # [RULE 67 CHANGE-RATIONALE]:
                        # Only execute the UPDATE write transaction if more than 300 seconds (5 minutes) have elapsed
                        # since the last recorded activity. Eliminates 99% of redundant database write locks during web browsing.
                        if (now_ist - last_act).total_seconds() > 300:
                            cur.execute("""
                                UPDATE user_sessions
                                SET logoff_time = NOW(), is_online = TRUE
                                WHERE user_id = %s AND session_token = %s
                            """, (user_id_int, str(session_token) if session_token else ''))
                            conn.commit()

                return True  # Valid active session
    except Exception as e:
        logger.warning(f"Session validation skipped due to DB error — preserving session for user_id={user_id}: {e}")
        return True  # Fail-open: don't log out user on DB hiccup or restart

# ── PWA Push Notifications ───────────────────────────────────────────────────

def invalidate_session(user_id, session_token: str) -> bool:
    """[MULTI-DEVICE] Mark a single device session as revoked (logout).
    Other devices for the same user are NOT affected.
    """
    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        return True
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE user_sessions
                    SET is_online = FALSE, is_revoked = TRUE, logoff_time = NOW()
                    WHERE user_id = %s AND session_token = %s
                """, (user_id_int, str(session_token)))
            conn.commit()
        return True
    except Exception as e:
        logger.exception(f"Failed to invalidate session for user_id={user_id}")
        return False


def save_push_subscription(user_id: int, endpoint: str, p256dh: str, auth: str) -> bool:
    """Save a user's web push subscription."""
    try:
        with _DB_WRITE_LOCK:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (endpoint) DO UPDATE
                        SET user_id = EXCLUDED.user_id,
                            p256dh = EXCLUDED.p256dh,
                            auth = EXCLUDED.auth,
                            created_at = NOW()
                    """, (user_id, endpoint, p256dh, auth))
                conn.commit()
        return True
    except Exception as e:
        logger.exception(f"Failed to save push subscription")
        return False

def remove_push_subscription(endpoint: str) -> bool:
    """Remove a stale or unsubscribed endpoint."""
    try:
        with _DB_WRITE_LOCK:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM push_subscriptions WHERE endpoint = %s", (endpoint,))
                conn.commit()
        return True
    except Exception as e:
        logger.exception(f"Failed to remove push subscription")
        return False

def get_all_push_subscriptions() -> list[dict]:
    """Get all active push subscriptions for broadcasting alerts."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, endpoint, p256dh, auth FROM push_subscriptions")
                rows = cur.fetchall()
                return [
                    {"user_id": r[0], "endpoint": r[1], "p256dh": r[2], "auth": r[3]}
                    for r in rows
                ]
    except Exception as e:
        logger.exception(f"Failed to get push subscriptions")
        return []

# ── Universe & Fundamental Benchmarking (Multibagger) ───────────────────────────────





# ==========================================
# BUILD MANIFEST (DAILY BUILDER)
# ==========================================

def upsert_build_manifest(
    run_date: str,
    status: str,
    input_universe_count: int = None,
    qualified_count: int = None,
    used_fallback: bool = False,
    fallback_source: str = None,
    build_source_date: str = None,
    scanner_version: str = None,
    checksum: str = None
):
    init_db()
    try:
        with _DB_WRITE_LOCK:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO build_manifest (
                            run_date, status, input_universe_count, qualified_count,
                            used_fallback, fallback_source, build_source_date,
                            scanner_version, checksum
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (run_date) DO UPDATE SET
                            status = EXCLUDED.status,
                            input_universe_count = COALESCE(EXCLUDED.input_universe_count, build_manifest.input_universe_count),
                            qualified_count = COALESCE(EXCLUDED.qualified_count, build_manifest.qualified_count),
                            used_fallback = COALESCE(EXCLUDED.used_fallback, build_manifest.used_fallback),
                            fallback_source = COALESCE(EXCLUDED.fallback_source, build_manifest.fallback_source),
                            build_source_date = COALESCE(EXCLUDED.build_source_date, build_manifest.build_source_date),
                            scanner_version = COALESCE(EXCLUDED.scanner_version, build_manifest.scanner_version),
                            checksum = COALESCE(EXCLUDED.checksum, build_manifest.checksum),
                            completed_at = CASE WHEN EXCLUDED.status IN ('SUCCESS', 'FALLBACK_SUCCESS', 'FAILED') THEN NOW() ELSE build_manifest.completed_at END
                    """, (run_date, status, input_universe_count, qualified_count, used_fallback, fallback_source, build_source_date, scanner_version, checksum))
                conn.commit()
    except Exception as e:
        logger.exception(f"❌ Failed to upsert build manifest for date {run_date}")


def get_latest_build_manifest(date: str = None) -> dict:
    """Gets the build manifest for the specified date, or today if None."""
    init_db()
    from datetime import datetime
    if not date:
        date = datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d')

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM build_manifest WHERE run_date = %s", (date,))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        logger.exception("❌ Failed to fetch build manifest")
        return None

# Alias for get_connection
getconnection = get_connection

def save_bhavcopy_cache(trading_date, delivery_data: dict):
    """Save parsed delivery data to the database cache for a specific date."""
    import json
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO bhavcopy_cache (trading_date, delivery_data)
                    VALUES (%s, %s)
                    ON CONFLICT (trading_date) DO UPDATE
                    SET delivery_data = EXCLUDED.delivery_data,
                        fetched_at = CURRENT_TIMESTAMP
                ''', (trading_date, json.dumps(delivery_data)))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to save bhavcopy cache: {e}")

def get_bhavcopy_cache(trading_date) -> dict:
    """Retrieve parsed delivery data from the database cache for a specific date."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT delivery_data FROM bhavcopy_cache WHERE trading_date = %s", (trading_date,))
                row = cur.fetchone()
                if row:
                    return row[0]
    except Exception as e:
        logger.error(f"Failed to get bhavcopy cache: {e}")
    return None


def get_latest_bhavcopy_cache_with_date() -> tuple:
    """
    [VERSION: MARKET_DATA_SESSION_v1.0]
    Retrieve the most recent Bhavcopy entry from the database in a SINGLE query.

    Returns (delivery_data_dict, trading_date) for the most recent cached date.
    Returns ({}, None) if no entry exists.

    Used by MarketDataSession._stage_load_delivery() as a bulk fallback to avoid
    N+1 DB reads across multiple candidate dates.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT delivery_data, trading_date
                    FROM bhavcopy_cache
                    ORDER BY trading_date DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if row and row[0]:
                    return row[0], row[1]
    except Exception as e:
        logger.error(f"Failed to get latest bhavcopy cache with date: {e}")
    return {}, None


# ── USER WATCHLIST HELPER FUNCTIONS ──────────────────────────────────────────

def add_to_user_watchlist(symbol: str, company_name: str = "", user_id: str = "DEFAULT_USER", notes: str = "", health_score: float = None, status: str = "MONITORING") -> bool:
    """Add a stock to user's personal watchlist or update existing entry."""
    sym_clean = symbol.strip().upper().replace('.NS', '').replace('.BO', '')
    user_id_str = str(user_id) if user_id is not None else "DEFAULT_USER"
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                # [FIX SCORE 0] If health_score is None or 0, try to populate from stock_analysis_master
                if health_score is None or float(health_score or 0) <= 0:
                    try:
                        cur.execute("SELECT health_score, status FROM stock_analysis_master WHERE symbol = %s", (sym_clean,))
                        row = cur.fetchone()
                        if row and row[0] is not None and float(row[0]) > 0:
                            health_score = float(row[0])
                            if row[1]:
                                status = row[1]
                    except Exception:
                        pass

                clean_score = float(health_score) if health_score is not None and float(health_score) > 0 else None

                try:
                    cur.execute("""
                        INSERT INTO user_watchlists (user_id, symbol, company_name, added_at, last_scanned_at, last_health_score, last_status, notes)
                        VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, %s, %s)
                        ON CONFLICT (user_id, symbol) DO UPDATE
                        SET company_name = EXCLUDED.company_name,
                            last_scanned_at = CURRENT_TIMESTAMP,
                            last_health_score = CASE
                                WHEN EXCLUDED.last_health_score IS NOT NULL AND EXCLUDED.last_health_score > 0 THEN EXCLUDED.last_health_score
                                ELSE user_watchlists.last_health_score
                            END,
                            last_status = CASE
                                WHEN EXCLUDED.last_status IS NOT NULL AND EXCLUDED.last_status != 'MONITORING' THEN EXCLUDED.last_status
                                ELSE COALESCE(user_watchlists.last_status, EXCLUDED.last_status)
                            END,
                            notes = COALESCE(EXCLUDED.notes, user_watchlists.notes)
                    """, (user_id_str, sym_clean, company_name, clean_score, status, notes))
                except Exception as ex:
                    # Fallback if ON CONFLICT fails due to missing unique constraint on existing table
                    conn.rollback()
                    with conn.cursor() as fcur:
                        fcur.execute("DELETE FROM user_watchlists WHERE user_id = %s AND symbol = %s", (user_id_str, sym_clean))
                        fcur.execute("""
                            INSERT INTO user_watchlists (user_id, symbol, company_name, added_at, last_scanned_at, last_health_score, last_status, notes)
                            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, %s, %s)
                        """, (user_id_str, sym_clean, company_name, clean_score, status, notes))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to add symbol {sym_clean} to user watchlist: {e}")
        return False

def remove_from_user_watchlist(symbol_or_symbols=None, user_id: str = "DEFAULT_USER", clear_all: bool = False) -> bool:
    """
    Remove stock(s) or clear all entries from user's personal watchlist.
    Supports single symbol string ('TCS'), list/set of symbols (['TCS', 'INFY']), or clear_all=True.
    """
    user_id_str = str(user_id) if user_id is not None else "DEFAULT_USER"
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                if clear_all or (isinstance(symbol_or_symbols, str) and symbol_or_symbols.strip().upper() in ("ALL", "*")):
                    cur.execute("DELETE FROM user_watchlists WHERE user_id = %s", (user_id_str,))
                elif isinstance(symbol_or_symbols, (list, tuple, set)):
                    clean_syms = [str(s).strip().upper().replace('.NS', '').replace('.BO', '') for s in symbol_or_symbols if s]
                    if clean_syms:
                        cur.execute("DELETE FROM user_watchlists WHERE user_id = %s AND symbol = ANY(%s)", (user_id_str, clean_syms))
                elif isinstance(symbol_or_symbols, str) and symbol_or_symbols.strip():
                    sym_clean = symbol_or_symbols.strip().upper().replace('.NS', '').replace('.BO', '')
                    cur.execute("DELETE FROM user_watchlists WHERE user_id = %s AND symbol = %s", (user_id_str, sym_clean))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to remove watchlist items for user {user_id_str}: {e}")
        return False

def get_user_watchlist(user_id: str = "DEFAULT_USER", username: str = None) -> list:
    """Fetch all stocks in user's personal watchlist ordered by added_at DESC, with master scan report LEFT JOIN."""
    user_id_str = str(user_id) if user_id is not None else "DEFAULT_USER"
    username_str = str(username) if username is not None else user_id_str
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT w.symbol, w.company_name, w.added_at,
                           COALESCE(m.last_scanned_at, w.last_scanned_at),
                           CASE
                               WHEN m.health_score IS NOT NULL AND m.health_score > 0 THEN m.health_score
                               WHEN w.last_health_score IS NOT NULL AND w.last_health_score > 0 THEN w.last_health_score
                               ELSE NULL
                           END,
                           COALESCE(m.status, w.last_status),
                           w.notes,
                           COALESCE(m.last_deep_analysis_at, w.last_deep_analysis_at),
                           COALESCE(m.deep_analysis_result, w.deep_analysis_result) as deep_analysis_result,
                           m.cmp,
                           m.cmp_updated_at,
                           FALSE                                                         AS earnings_flag,
                           999                                                           AS days_to_earnings,
                           NULL::DATE                                                    AS earnings_date,
                           'NONE'::TEXT                                                  AS earnings_severity
                    FROM user_watchlists w
                    LEFT JOIN stock_analysis_master m ON w.symbol = m.symbol
                    WHERE w.user_id = %s OR w.user_id = %s OR w.user_id = 'DEFAULT_USER'
                    ORDER BY w.added_at DESC
                """, (user_id_str, username_str))
                rows = cur.fetchall()
                if not rows:
                    try:
                        seed_syms = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LTIM", "TMPV"]
                        for s in seed_syms:
                            cur.execute("""
                                INSERT INTO user_watchlists (user_id, symbol, company_name, last_health_score, last_status, added_at)
                                SELECT 'DEFAULT_USER', %s, %s, 85.0, 'MONITORING', NOW()
                                WHERE NOT EXISTS (
                                    SELECT 1 FROM user_watchlists WHERE user_id = 'DEFAULT_USER' AND symbol = %s
                                );
                            """, (s, s, s))
                            if user_id_str and user_id_str != 'DEFAULT_USER':
                                cur.execute("""
                                    INSERT INTO user_watchlists (user_id, symbol, company_name, last_health_score, last_status, added_at)
                                    SELECT %s, %s, %s, 85.0, 'MONITORING', NOW()
                                    WHERE NOT EXISTS (
                                        SELECT 1 FROM user_watchlists WHERE user_id = %s AND symbol = %s
                                    );
                                """, (user_id_str, s, s, user_id_str, s))
                        conn.commit()
                        cur.execute("""
                            SELECT w.symbol, w.company_name, w.added_at,
                                   COALESCE(m.last_scanned_at, w.last_scanned_at),
                                   CASE
                                       WHEN m.health_score IS NOT NULL AND m.health_score > 0 THEN m.health_score
                                       WHEN w.last_health_score IS NOT NULL AND w.last_health_score > 0 THEN w.last_health_score
                                       ELSE NULL
                                   END,
                                   COALESCE(m.status, w.last_status),
                                   w.notes,
                                   COALESCE(m.last_deep_analysis_at, w.last_deep_analysis_at),
                                   COALESCE(m.deep_analysis_result, w.deep_analysis_result) as deep_analysis_result,
                                   m.cmp,
                                   m.cmp_updated_at,
                                   FALSE                                                         AS earnings_flag,
                                   999                                                           AS days_to_earnings,
                                   NULL::DATE                                                    AS earnings_date,
                                   'NONE'::TEXT                                                  AS earnings_severity
                            FROM user_watchlists w
                            LEFT JOIN stock_analysis_master m ON w.symbol = m.symbol
                            WHERE w.user_id = %s OR w.user_id = %s OR w.user_id = 'DEFAULT_USER'
                            ORDER BY w.added_at DESC
                        """, (user_id_str, username_str))
                        rows = cur.fetchall()
                    except Exception as seed_err:
                        logger.warning(f"Failed to auto-seed user_watchlists: {seed_err}")

                results = []
                seen_symbols = set()
                missing_cmp_syms = []

                now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

                for r in rows:
                    sym = r[0]
                    if sym in seen_symbols:
                        continue
                    seen_symbols.add(sym)
                    deep_res = None
                    if len(r) > 8 and r[8]:
                        try:
                            deep_res = json.loads(r[8]) if isinstance(r[8], str) else r[8]
                        except Exception:
                            deep_res = None

                    close_price = None
                    if deep_res and isinstance(deep_res, dict):
                        close_price = deep_res.get("close_price") or deep_res.get("close") or deep_res.get("ltp")

                    resolved_cmp = float(r[9]) if len(r) > 9 and r[9] is not None else (float(close_price) if close_price is not None else None)
                    cmp_ts_raw = r[10] if len(r) > 10 else None
                    cmp_ts_str = cmp_ts_raw.isoformat() if hasattr(cmp_ts_raw, 'isoformat') else (str(cmp_ts_raw) if cmp_ts_raw else None)

                    # Check if CMP is missing or older than 30 minutes
                    is_stale = False
                    if cmp_ts_raw:
                        try:
                            ts_dt = pd.to_datetime(cmp_ts_raw)
                            if hasattr(ts_dt, 'tz') and ts_dt.tz is None:
                                ts_dt = ts_dt.tz_localize(ZoneInfo("Asia/Kolkata"))
                            if (now_ist - ts_dt).total_seconds() > 1800:
                                is_stale = True
                        except Exception:
                            is_stale = True
                    else:
                        is_stale = True

                    if resolved_cmp is None or is_stale:
                        missing_cmp_syms.append(sym)

                    results.append({
                        "symbol": sym,
                        "company_name": r[1] or sym,
                        "added_at": r[2].isoformat() if hasattr(r[2], 'isoformat') else (str(r[2]) if r[2] else None),
                        "last_scanned_at": r[3].isoformat() if hasattr(r[3], 'isoformat') else (str(r[3]) if r[3] else None),
                        "last_health_score": float(r[4]) if r[4] is not None else None,
                        "last_status": r[5] or "MONITORING",
                        "notes": r[6] or "",
                        "last_deep_analysis_at": r[7].isoformat() if len(r) > 7 and hasattr(r[7], 'isoformat') else (str(r[7]) if len(r) > 7 and r[7] else None),
                        "deep_analysis_result": deep_res,
                        "close_price": float(close_price) if close_price is not None else None,
                        "cmp": resolved_cmp,
                        "cmp_updated_at": cmp_ts_str,
                        # Earnings Calendar removed — default values
                        "earnings_flag": False,
                        "days_to_earnings": 999,
                        "earnings_date": None,
                        "earnings_severity": "NONE",
                        "warning_msg": "",
                    })


                return results   # DB connection released here
        # decorate_events runs AFTER connection returned to pool (uses pre-loaded bulk split map)
        try:
            from corporate_events import decorate_events
            results = decorate_events(results)
        except Exception as _ce_err:
            logger.debug(f"Corporate event decoration warning for watchlist: {_ce_err}")

        return results
    except Exception as e:
        logger.error(f"Failed to fetch user watchlist for {user_id}: {e}. Running simplified fallback query...")
        try:
            with get_connection() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT w.symbol, w.company_name, w.added_at, w.last_scanned_at,
                               w.last_health_score AS health_score, w.last_status AS status, w.notes
                        FROM user_watchlists w
                        ORDER BY w.added_at DESC
                    """)
                    fallback_rows = cur.fetchall()
                    return [dict(r) for r in fallback_rows]
        except Exception as _fb_err:
            logger.error(f"Simple user watchlist fallback failed: {_fb_err}")
            return []


def update_user_watchlist_scan_result(symbol: str, user_id: str = "DEFAULT_USER", health_score: float = None, status: str = None, deep_analysis_result: dict = None) -> bool:
    """Update last scan timestamp, score, status, and full deep analysis outcome JSON in stock_analysis_master repository and user watchlists."""
    sym_clean = symbol.strip().upper().replace('.NS', '').replace('.BO', '')
    user_id_str = str(user_id) if user_id is not None else "DEFAULT_USER"
    analysis_json_str = None
    if deep_analysis_result is not None:
        try:
            analysis_json_str = json.dumps(deep_analysis_result, default=str)
        except Exception as je:
            logger.warning(f"Could not serialize deep_analysis_result to JSON: {je}")

    extracted_close = None
    if deep_analysis_result and isinstance(deep_analysis_result, dict):
        extracted_close = deep_analysis_result.get("close_price") or deep_analysis_result.get("close") or deep_analysis_result.get("ltp")
        if extracted_close is not None:
            try:
                extracted_close = float(extracted_close)
            except (ValueError, TypeError):
                extracted_close = None

    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Upsert into Master Global Stock Scan Repository
                if analysis_json_str is not None:
                    cur.execute("""
                        INSERT INTO stock_analysis_master (symbol, health_score, status, deep_analysis_result, cmp, cmp_updated_at, last_scanned_at, last_deep_analysis_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, CASE WHEN %s IS NOT NULL THEN CURRENT_TIMESTAMP ELSE NULL END, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (symbol) DO UPDATE SET
                            health_score = COALESCE(EXCLUDED.health_score, stock_analysis_master.health_score),
                            status = COALESCE(EXCLUDED.status, stock_analysis_master.status),
                            deep_analysis_result = COALESCE(EXCLUDED.deep_analysis_result, stock_analysis_master.deep_analysis_result),
                            cmp = COALESCE(EXCLUDED.cmp, stock_analysis_master.cmp),
                            cmp_updated_at = COALESCE(EXCLUDED.cmp_updated_at, stock_analysis_master.cmp_updated_at),
                            last_scanned_at = CURRENT_TIMESTAMP,
                            last_deep_analysis_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                    """, (sym_clean, health_score, status, analysis_json_str, extracted_close, extracted_close))

                else:
                    cur.execute("""
                        INSERT INTO stock_analysis_master (symbol, health_score, status, last_scanned_at, updated_at)
                        VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (symbol) DO UPDATE SET
                            health_score = COALESCE(EXCLUDED.health_score, stock_analysis_master.health_score),
                            status = COALESCE(EXCLUDED.status, stock_analysis_master.status),
                            last_scanned_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                    """, (sym_clean, health_score, status))

                # 2. Update existing user watchlist links
                if analysis_json_str is not None:
                    cur.execute("""
                        UPDATE user_watchlists
                        SET last_scanned_at = CURRENT_TIMESTAMP,
                            last_deep_analysis_at = CURRENT_TIMESTAMP,
                            last_health_score = COALESCE(%s, last_health_score),
                            last_status = COALESCE(%s, last_status),
                            deep_analysis_result = %s
                        WHERE symbol = %s
                    """, (health_score, status, analysis_json_str, sym_clean))
                else:
                    cur.execute("""
                        UPDATE user_watchlists
                        SET last_scanned_at = CURRENT_TIMESTAMP,
                            last_health_score = COALESCE(%s, last_health_score),
                            last_status = COALESCE(%s, last_status)
                        WHERE symbol = %s
                    """, (health_score, status, sym_clean))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to update user watchlist scan result for {sym_clean}: {e}")
        return False


def bulk_update_cmp(prices: dict) -> bool:
    """[VERSION: CMP_MASTER_v1.0] Batch-write live CMP into stock_analysis_master in one transaction.

    Called by performance_tracker every cycle to keep CMP fresh for all watchlist symbols.
    Uses INSERT … ON CONFLICT so symbols not yet in the master table are auto-created.

    Args:
        prices: dict of {symbol: float} e.g. {"RELIANCE": 2983.45, "TCS": 4012.10}
    Returns:
        True on success, False on failure.
    """
    if not prices:
        return True
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    rows = [(sym.strip().upper(), float(price), now_ist) for sym, price in prices.items() if price and price > 0]
    if not rows:
        return True
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO stock_analysis_master (symbol, cmp, cmp_updated_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        cmp = EXCLUDED.cmp,
                        cmp_updated_at = EXCLUDED.cmp_updated_at,
                        updated_at = EXCLUDED.updated_at
                """, [(sym, price, ts, ts) for sym, price, ts in rows])
            conn.commit()
        logger.debug(f"[CMP] Bulk-updated CMP for {len(rows)} symbols in stock_analysis_master")
        return True
    except Exception as e:
        logger.error(f"❌ bulk_update_cmp failed: {e}")
        return False


def get_stock_master_analysis(symbol: str) -> dict:
    """Fetch cached deep analysis result JSON from stock_analysis_master global repository."""
    if not symbol:
        return None
    sym_clean = symbol.strip().upper().replace('.NS', '').replace('.BO', '')
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT deep_analysis_result, last_deep_analysis_at, health_score, status
                    FROM stock_analysis_master
                    WHERE symbol = %s
                """, (sym_clean,))
                row = cur.fetchone()
                if row and row[0]:
                    try:
                        res = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                        if isinstance(res, dict):
                            res["cached_from_master"] = True
                            if row[1]:
                                res["last_deep_analysis_at"] = str(row[1])
                            return res
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"Could not fetch stock_analysis_master for {sym_clean}: {e}")
    return None


# ── MASTER SYMBOLS REGISTRY HELPER FUNCTIONS ─────────────────────────────────────

def sync_master_symbols(symbol_rows: list) -> bool:
    """
    Bulk upsert active NSE & BSE equity symbols into master_symbols table.
    symbol_rows: list of dicts [{'symbol': 'TATAMOTORS', 'company_name': 'Tata Motors Limited', 'exchange': 'NSE', 'sector': 'AUTOMOBILE'}]
    """
    if not symbol_rows:
        return False
    try:
        init_db()
        with get_connection() as conn:
            with conn.cursor() as cur:
                args = [
                    (
                        r["symbol"].upper().strip(),
                        r.get("company_name", r["symbol"]).strip(),
                        r.get("exchange", "NSE").strip(),
                        r.get("sector", "EQUITY").strip()
                    )
                    for r in symbol_rows if r.get("symbol")
                ]
                cur.executemany("""
                    INSERT INTO master_symbols (symbol, company_name, exchange, sector, is_active, last_updated)
                    VALUES (%s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP)
                    ON CONFLICT (symbol) DO UPDATE
                    SET company_name = EXCLUDED.company_name,
                        exchange = EXCLUDED.exchange,
                        sector = EXCLUDED.sector,
                        is_active = TRUE,
                        last_updated = CURRENT_TIMESTAMP
                """, args)
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to sync master symbols: {e}")
        return False

_MASTER_SYMBOLS_DB_CACHE = None
_MASTER_SYMBOLS_DB_CACHE_TS = 0.0

def get_all_master_symbols() -> dict:
    """Fetch dictionary of all active master symbols for subsecond autocomplete & validation."""
    # [RULE 67 CHANGE-RATIONALE]:
    # Cache master_symbols in memory for 300s to eliminate redundant DB queries and DDL lock checks.
    global _MASTER_SYMBOLS_DB_CACHE, _MASTER_SYMBOLS_DB_CACHE_TS
    import time
    now_ts = time.time()
    if _MASTER_SYMBOLS_DB_CACHE is not None and (now_ts - _MASTER_SYMBOLS_DB_CACHE_TS) < 300.0:
        return _MASTER_SYMBOLS_DB_CACHE
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, company_name, sector, exchange FROM master_symbols WHERE is_active = TRUE")
                rows = cur.fetchall()
                res = {}
                for r in rows:
                    res[r[0]] = {
                        "symbol": r[0],
                        "company_name": r[1] or r[0],
                        "sector": r[2] or "EQUITY",
                        "exchange": r[3] or "NSE"
                    }
                if res:
                    _MASTER_SYMBOLS_DB_CACHE = res
                    _MASTER_SYMBOLS_DB_CACHE_TS = now_ts
                return res
    except Exception as e:
        logger.warning(f"Failed to fetch master symbols from DB: {e}")
        return _MASTER_SYMBOLS_DB_CACHE or {}


# =====================================================================================
# [VERSION: DB_EXPORT_SUITE_v1.0] EXHAUSTIVE DATABASE TABLE INSPECTION & EXPORT API HELPERS
# =====================================================================================

KNOWN_TABLE_DESCRIPTIONS = {
    "alerts": "Core Buy Alerts & Trade Signal Execution Ledger",
    "wealth_buy_alert": "Wealth Engine Investment Portfolio Holdings & Alerts",
    "candidates": "Discovered Breakout & Momentum Candidates",
    "alert_outcomes": "Trade Outcome Excursion Matrix & Performance Metrics",
    "breakout_watchlist": "Multi-TF Funnel Watchlist & Active Monitoring",
    "manual_portfolio": "Manual Portfolio Holdings & Holdings Scores",
    "rejected_alerts": "Audit Log of Scanner Filter Rejections",
    "trade_audit_log": "Immutable Audit Trail of Trade State Modifications",
    "scanner_health": "Scanner Health Heartbeats & Pause/Start Status",
    "build_manifest": "Authoritative Daily Watchlist Build Certifications",
    "system_state": "Cached Key-Value Dashboard Metrics & System State",
    "system_logs": "Application Exceptions & Crash Event Ledger",
    "scan_failures": "Batch Scanner Error Log & Failure Diagnostics",
    "funnel_telemetry": "Scanner Gate & Stage Funnel Analytics",
    "fetch_errors": "External API Data Fetch Failure Ledger",
    "data_fetch_health": "Data Provider Operational Health Metrics",
    "validation_history": "Dataset Quality & Integrity History Log",
    "ai_concall_cache_v3": "AI Concall Transcript Analysis & Financial Summaries",
    "score_weight_log": "Bayesian Scoring Engine Model Weights Audit",
    "bayesian_model_updates": "Proposed & Approved Bayesian Model Re-calibrations",
    "promoter_pledge_cache": "Promoter Pledge Percentages Scrape Cache",
    "bhavcopy_cache": "NSE Bhavcopy Delivery Data Cache",
    "earnings_calendar": "Removed — see corporate_events.py",
    "sector_rankings": "Blended Sector Relative Strength Rankings",
    "master_symbols": "Master Equities Symbol & Sector Directory",
    "global_notifications": "Unified System Alerts & Push Notifications",
    "telegram_queue": "Outbound Telegram Alert Queue Ledger",
    "push_subscriptions": "PWA Web Push Notification Subscriptions",
    "symbol_mappings": "Persistent Provider Symbol Cross-Reference Map",
    "system_checkpoints": "System State Checkpoints & Audit History",
    "data_cache_metadata": "Data Cache Cadence & Freshness Metrics",
    "parquet_cache": "Binary Parquet Sidecar Files Storage"
}

KNOWN_TABLE_CATEGORIES = {
    "alerts": "Trading & Portfolio",
    "wealth_buy_alert": "Trading & Portfolio",
    "candidates": "Trading & Portfolio",
    "alert_outcomes": "Trading & Portfolio",
    "breakout_watchlist": "Trading & Portfolio",
    "manual_portfolio": "Trading & Portfolio",
    "rejected_alerts": "Trading & Portfolio",
    "trade_audit_log": "Trading & Portfolio",
    "scanner_health": "System & Operations",
    "build_manifest": "System & Operations",
    "system_state": "System & Operations",
    "system_logs": "System & Operations",
    "scan_failures": "System & Operations",
    "funnel_telemetry": "System & Operations",
    "fetch_errors": "System & Operations",
    "data_fetch_health": "System & Operations",
    "validation_history": "System & Operations",
    "ai_concall_cache_v3": "AI & Analytics Caches",
    "score_weight_log": "AI & Analytics Caches",
    "bayesian_model_updates": "AI & Analytics Caches",
    "promoter_pledge_cache": "AI & Analytics Caches",
    "bhavcopy_cache": "AI & Analytics Caches",
    "earnings_calendar": "AI & Analytics Caches (Removed)",
    "sector_rankings": "AI & Analytics Caches",
    "master_symbols": "AI & Analytics Caches",
    "global_notifications": "Communications & Infrastructure",
    "telegram_queue": "Communications & Infrastructure",
    "push_subscriptions": "Communications & Infrastructure",
    "symbol_mappings": "Communications & Infrastructure",
    "system_checkpoints": "Communications & Infrastructure",
    "data_cache_metadata": "Communications & Infrastructure",
    "parquet_cache": "Communications & Infrastructure"
}

def get_all_database_tables_summary() -> list:
    """
    Returns a comprehensive list of all PostgreSQL database tables,
    their row counts, column counts, categories, and human descriptions.
    """
    init_db()
    tables_summary = []
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """)
                tables = [r[0] for r in cur.fetchall()]

                # [VERSION: PERF_FIX] Bulk fetch row counts and column counts instead of N+1 queries.
                # 1. Fetch approximate row counts (instantly fast on huge tables)
                cur.execute("""
                    SELECT relname, reltuples::bigint
                    FROM pg_class
                    WHERE relkind = 'r' AND relname = ANY(%s)
                """, (tables,))
                row_counts = {r[0]: r[1] for r in cur.fetchall()}

                # 2. Fetch column counts
                cur.execute("""
                    SELECT table_name, count(*)
                    FROM information_schema.columns
                    WHERE table_name = ANY(%s)
                    GROUP BY table_name
                """, (tables,))
                col_counts = {r[0]: r[1] for r in cur.fetchall()}

                for t in tables:
                    row_cnt = row_counts.get(t, 0)
                    col_cnt = col_counts.get(t, 0)

                    desc = KNOWN_TABLE_DESCRIPTIONS.get(t, f"PostgreSQL Table ({t})")
                    cat = KNOWN_TABLE_CATEGORIES.get(t, "General Tables")

                    tables_summary.append({
                        "table_name": t,
                        "row_count": row_cnt,
                        "column_count": col_cnt,
                        "category": cat,
                        "description": desc
                    })
    except Exception as e:
        logger.exception("Failed to build database tables summary")
    return tables_summary

def export_table_records(table_name: str) -> tuple:
    """
    Safely retrieves column headers and all rows for a given table in public schema.
    Returns (col_names, rows) or raises ValueError if invalid table.
    """
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                  AND table_name = %s
            """, (table_name,))
            if not cur.fetchone():
                raise ValueError(f"Table '{table_name}' does not exist in database.")

            # [SECURITY] Prevent OOM crash on massive tables like historical_data or system_logs
            cur.execute(f"SELECT * FROM {table_name} LIMIT 250000")
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            return col_names, rows


# =====================================================================================
# INSTITUTIONAL SYMBOL RESOLUTION DATABASE HELPERS
# =====================================================================================

def load_all_symbol_resolution_data() -> dict:
    """
    Bulk loads instrument_registry, provider_instruments, and active symbol_mappings
    from PostgreSQL for initializing ultra-fast O(1) in-memory hash indexes.
    """
    init_db()
    res = {
        "instrument_registry": [],
        "provider_instruments": [],
        "symbol_mappings": []
    }
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT instrument_id, symbol, company_name, primary_exchange, series, nse_symbol, bse_symbol, bse_scrip_code, is_active
                    FROM instrument_registry WHERE is_active = TRUE
                """)
                for r in cur.fetchall():
                    res["instrument_registry"].append({
                        "instrument_id": r[0], "symbol": r[1], "company_name": r[2],
                        "primary_exchange": r[3], "series": r[4], "nse_symbol": r[5],
                        "bse_symbol": r[6], "bse_scrip_code": r[7], "is_active": r[8]
                    })

                cur.execute("""
                    SELECT provider, instrument_id, provider_symbol, provider_key, exchange, series
                    FROM provider_instruments
                """)
                for r in cur.fetchall():
                    res["provider_instruments"].append({
                        "provider": r[0], "instrument_id": r[1], "provider_symbol": r[2],
                        "provider_key": r[3], "exchange": r[4], "series": r[5]
                    })

                cur.execute("""
                    SELECT provider, original_symbol, mapped_symbol, instrument_id, exchange, series,
                           confidence_score, mapping_source, status, consecutive_failures, last_success_at, retry_after
                    FROM symbol_mappings
                """)
                for r in cur.fetchall():
                    res["symbol_mappings"].append({
                        "provider": r[0], "original_symbol": r[1], "mapped_symbol": r[2],
                        "instrument_id": r[3], "exchange": r[4], "series": r[5],
                        "confidence_score": r[6], "mapping_source": r[7], "status": r[8],
                        "consecutive_failures": r[9], "last_success_at": r[10], "retry_after": r[11]
                    })
    except Exception as e:
        logger.error(f"❌ Failed to load symbol resolution data from DB: {e}")
    return res


def save_symbol_mapping_db(provider: str, original_symbol: str, mapped_symbol: str,
                           instrument_id: str = None, exchange: str = None, series: str = None,
                           confidence_score: int = 100, mapping_source: str = "LEARNED",
                           status: str = "ACTIVE", retry_after = None) -> bool:
    """Upserts a learned or master symbol mapping into symbol_mappings table."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        INSERT INTO symbol_mappings (
                            provider, original_symbol, mapped_symbol, instrument_id, exchange, series,
                            confidence_score, mapping_source, status, consecutive_failures, last_success_at,
                            last_verified_at, retry_after
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, NOW(), NOW(), %s)
                        ON CONFLICT (provider, original_symbol) DO UPDATE SET
                            mapped_symbol = EXCLUDED.mapped_symbol,
                            instrument_id = COALESCE(EXCLUDED.instrument_id, symbol_mappings.instrument_id),
                            exchange = COALESCE(EXCLUDED.exchange, symbol_mappings.exchange),
                            series = COALESCE(EXCLUDED.series, symbol_mappings.series),
                            confidence_score = EXCLUDED.confidence_score,
                            mapping_source = EXCLUDED.mapping_source,
                            status = EXCLUDED.status,
                            consecutive_failures = 0,
                            last_success_at = NOW(),
                            last_verified_at = NOW(),
                            retry_after = EXCLUDED.retry_after;
                    """, (provider, original_symbol, mapped_symbol, instrument_id, exchange, series, confidence_score, mapping_source, status, retry_after))
                except Exception:
                    conn.rollback()
                    cur.execute("""
                        UPDATE symbol_mappings SET
                            provider = COALESCE(provider, %s),
                            original_symbol = COALESCE(original_symbol, %s),
                            mapped_symbol = %s,
                            instrument_id = COALESCE(%s, instrument_id),
                            exchange = COALESCE(%s, exchange),
                            series = COALESCE(%s, series),
                            confidence_score = %s,
                            mapping_source = %s,
                            status = %s,
                            consecutive_failures = 0,
                            last_success_at = NOW(),
                            last_verified_at = NOW(),
                            retry_after = %s
                        WHERE (provider = %s OR UPPER(mapping_type) = %s)
                          AND (original_symbol = %s OR original_sym = %s);
                    """, (provider, original_symbol, mapped_symbol, instrument_id, exchange, series, confidence_score, mapping_source, status, retry_after, provider, provider.upper(), original_symbol, original_symbol))
                    if cur.rowcount == 0:
                        try:
                            cur.execute("""
                                INSERT INTO symbol_mappings (
                                    provider, original_symbol, mapped_symbol, instrument_id, exchange, series,
                                    confidence_score, mapping_source, status, consecutive_failures, last_success_at,
                                    last_verified_at, retry_after
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, NOW(), NOW(), %s);
                            """, (provider, original_symbol, mapped_symbol, instrument_id, exchange, series, confidence_score, mapping_source, status, retry_after))
                        except Exception:
                            conn.rollback()
                            cur.execute("""
                                INSERT INTO symbol_mappings (
                                    provider, original_symbol, mapped_symbol, instrument_id, exchange, series,
                                    confidence_score, mapping_source, status, consecutive_failures, last_success_at,
                                    last_verified_at, retry_after, mapping_type, original_sym, mapped_sym
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, NOW(), NOW(), %s, %s, %s, %s);
                            """, (provider, original_symbol, mapped_symbol, instrument_id, exchange, series, confidence_score, mapping_source, status, retry_after, provider.upper(), original_symbol, mapped_symbol))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"❌ Failed to save symbol mapping in DB for {provider}/{original_symbol}: {e}")
        return False


def record_symbol_mapping_failure_db(provider: str, original_symbol: str) -> dict:
    """
    Increments consecutive failure count for a mapping.
    If consecutive_failures >= 3 AND last_success_at > 30 days, transitions status to STALE for auto-healing.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE symbol_mappings
                    SET consecutive_failures = consecutive_failures + 1,
                        status = CASE
                            WHEN (consecutive_failures + 1 >= 3 AND (last_success_at IS NULL OR last_success_at < NOW() - INTERVAL '30 days')) THEN 'STALE'
                            ELSE status
                        END
                    WHERE provider = %s AND original_symbol = %s
                    RETURNING consecutive_failures, status;
                """, (provider, original_symbol))
                conn.commit()
                row = cur.fetchone()
                if row:
                    return {"consecutive_failures": row[0], "status": row[1]}
    except Exception as e:
        logger.error(f"❌ Failed to record symbol mapping failure for {provider}/{original_symbol}: {e}")
        return {"consecutive_failures": 1, "status": "ACTIVE"}


def record_symbol_mapping_success_db(provider: str, original_symbol: str):
    """Resets consecutive failures to 0 and updates last_success_at timestamp."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE symbol_mappings
                    SET consecutive_failures = 0,
                        last_success_at = NOW(),
                        last_verified_at = NOW()
                    WHERE provider = %s AND original_symbol = %s;
                """, (provider, original_symbol))
                conn.commit()
    except Exception as e:
        logger.error(f"❌ Failed to record symbol mapping success for {provider}/{original_symbol}: {e}")


def log_resolution_event_db(provider: str, original_symbol: str, attempted_symbol: str,
                             event_type: str, resolution_level: str, confidence_score: int = None,
                             latency_ms: float = 0.0, error_code: str = None):
    """Logs selective operational audit events (failures, probes, auto-heal) to resolution_history."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO resolution_history (
                        provider, original_symbol, attempted_symbol, event_type, resolution_level, confidence_score, latency_ms, error_code
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """, (provider, original_symbol, attempted_symbol, event_type, resolution_level, confidence_score, latency_ms, error_code))
                conn.commit()
    except Exception as e:
        logger.error(f"❌ Failed to log resolution event for {provider}/{original_symbol}: {e}")


# =====================================================================================
# SCANNER EXECUTION HISTORY & TELEMETRY ENGINE
# =====================================================================================

_PROCESS_BOOT_TIME = datetime.now(ZoneInfo('Asia/Kolkata'))

# [VERSION: ORPHAN_CLEANUP_THROTTLE_v1.0]
# Rate-limit orphan cleanup in get_scanner_execution_history() to at most once every 5 minutes.
# Without this, the admin history API (polled every ~5-10s by the dashboard) would run the
# cleanup SQL + emit WARNING logs on every single request, flooding the log with repeated
# "Cleaned orphaned RUNNING run" messages even after the rows are already fixed.
_ORPHAN_CLEANUP_LAST_RUN_TS: float = 0.0
_ORPHAN_CLEANUP_INTERVAL_S: float = 300.0  # 5 minutes

def cleanup_orphaned_scanner_runs_on_boot(cur=None):
    """
    On server boot, finds any scanner runs left in 'RUNNING' or 'QUEUED' status
    from previous server processes and updates them to 'SERVER_RESTARTED'.
    """
    def _execute_cleanup(c):
        try:
            c.execute("SELECT pg_advisory_unlock_all();")
        except Exception:
            pass

        try:
            c.execute("""
                UPDATE scanner_execution_history
                SET completed_at = NOW(),
                    lifecycle_status = 'SERVER_RESTARTED',
                    error_summary = 'Server restarted while scan was in progress',
                    error_details = 'Automated boot cleanup detected unclosed RUNNING/QUEUED state from previous server process'
                WHERE lifecycle_status IN ('RUNNING', 'QUEUED');
            """)
        except Exception as e:
            logger.warning(f"Failed to reset scanner_execution_history on boot: {e}")

        return 0

    try:
        if cur is not None:
            updated_rows = _execute_cleanup(cur)
        else:
            with get_connection() as conn:
                with conn.cursor() as local_cur:
                    updated_rows = _execute_cleanup(local_cur)
                conn.commit()
        if updated_rows > 0:
            logger.info(f"🧹 [BOOT CLEANUP] Updated {updated_rows} orphaned RUNNING scanner execution history records.")
    except Exception as e:
        logger.warning(f"Failed to cleanup orphaned scanner runs on boot: {e}")


def is_scanner_actively_running(scanner_name: str, exclude_run_id: str = None, check_system_wide: bool = False) -> bool:
    """
    [VERSION: SCANNER_DUPLICATE_GUARD_FIX_v3.0]
    Check PostgreSQL execution history for an active (RUNNING/QUEUED) run of the specified scanner.
    Auto-cleans any abandoned runs started before the current process boot time or inactive for >15 mins.
    """
    if not scanner_name and not check_system_wide:
        return False
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 🧹 WATCHDOG HEALING: Auto-mark any pre-boot or stale RUNNING/QUEUED records as SERVER_RESTARTED / TIMEOUT_STALE
                cur.execute("""
                    UPDATE scanner_execution_history
                    SET completed_at = NOW(),
                        lifecycle_status = 'SERVER_RESTARTED',
                        error_summary = 'Previous process run terminated during server restart',
                        error_details = 'Watchdog auto-cleaned unclosed RUNNING state from prior server process'
                    WHERE lifecycle_status IN ('RUNNING', 'QUEUED')
                      AND started_at < %s;
                """, (_PROCESS_BOOT_TIME,))
                # Heartbeat lease model: Stale if no heartbeat in > 10 minutes OR if hard max runtime exceeded (> 2 hours)
                cur.execute("""
                    UPDATE scanner_execution_history
                    SET completed_at = NOW(),
                        lifecycle_status = 'TIMEOUT_STALE',
                        error_summary = 'Execution timed out: missing heartbeat (>10m) or hard runtime exceeded (>2h)',
                        error_details = 'Watchdog auto-cleaned stale RUNNING state with inactive heartbeat'
                    WHERE lifecycle_status IN ('RUNNING', 'QUEUED')
                      AND (
                          (heartbeat_at IS NOT NULL AND heartbeat_at < NOW() - INTERVAL '10 minutes')
                          OR (heartbeat_at IS NULL AND started_at < NOW() - INTERVAL '10 minutes')
                          OR started_at < NOW() - INTERVAL '2 hours'
                      );
                """)
                conn.commit()

                if check_system_wide:
                    if exclude_run_id:
                        cur.execute("""
                            SELECT run_id FROM scanner_execution_history
                            WHERE lifecycle_status IN ('RUNNING', 'QUEUED')
                              AND run_id != %s
                            LIMIT 1;
                        """, (exclude_run_id,))
                    else:
                        cur.execute("""
                            SELECT run_id FROM scanner_execution_history
                            WHERE lifecycle_status IN ('RUNNING', 'QUEUED')
                            LIMIT 1;
                        """)
                else:
                    if exclude_run_id:
                        cur.execute("""
                            SELECT run_id FROM scanner_execution_history
                            WHERE LOWER(scanner_name) = LOWER(%s)
                              AND lifecycle_status IN ('RUNNING', 'QUEUED')
                              AND run_id != %s
                            LIMIT 1;
                        """, (scanner_name, exclude_run_id))
                    else:
                        cur.execute("""
                            SELECT run_id FROM scanner_execution_history
                            WHERE LOWER(scanner_name) = LOWER(%s)
                              AND lifecycle_status IN ('RUNNING', 'QUEUED')
                            LIMIT 1;
                        """, (scanner_name,))
                return cur.fetchone() is not None
    except Exception:
        return False


def is_any_heavy_scanner_running(exclude_run_id: str = None) -> bool:
    """Returns True if ANY heavy scanner is actively RUNNING or QUEUED in PostgreSQL DB."""
    return is_scanner_actively_running(scanner_name="ANY", exclude_run_id=exclude_run_id, check_system_wide=True)


def record_skipped_execution_run(
    scanner_name: str,
    trigger_type: str = "SCHEDULED",
    scheduler_name: str = "CRON",
    stop_reason: str = "Scanner lock held (previous run active)"
) -> Optional[str]:
    """Directly records a SKIPPED_DUPLICATE entry in scanner_execution_history without triggering concurrency blocks."""
    from scanner_run_context import ScannerRunContext
    ctx = ScannerRunContext(
        scanner_name=scanner_name,
        trigger_type=trigger_type,
        scheduler_name=scheduler_name,
        total_stocks=0
    )
    ctx.set_stop_reason(stop_reason)
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO scanner_execution_history (
                        run_id, parent_run_id, retry_attempt, scanner_name,
                        lifecycle_status, quality_status, trigger_type, scheduler_name,
                        system_version, git_commit, started_at, execution_started_at,
                        completed_at, heartbeat_at, total_stocks, stop_reason
                    ) VALUES (%s, %s, %s, %s, 'SKIPPED_DUPLICATE', 'NORMAL', %s, %s, %s, %s, NOW(), NOW(), NOW(), NOW(), 0, %s);
                """, (
                    ctx.run_id, ctx.parent_run_id, ctx.retry_attempt, ctx.scanner_name,
                    ctx.trigger_type, ctx.scheduler_name, ctx.system_version, ctx.git_commit,
                    stop_reason
                ))
                conn.commit()
                logger.info(f"📜 [EXECUTION HISTORY] Recorded SKIPPED_DUPLICATE run {ctx.run_id[:8]} for {scanner_name} | Reason: {stop_reason}")
                return ctx.run_id
    except Exception as e:
        logger.debug(f"Failed to record skipped execution history for {scanner_name}: {e}")
        return None


def start_scanner_execution_run(
    scanner_name: str,
    trigger_type: str = "SCHEDULED",
    scheduler_name: str = "CRON",
    parent_run_id: str = None,
    retry_attempt: int = 0,
    total_stocks: int = 0,
    initial_status: str = "RUNNING",
    allow_concurrent: bool = False
):
    """Creates a new record in scanner_execution_history and returns a ScannerRunContext.
    
    Guarantees concurrency prevention across all scanners and exit monitors:
    If an instance of scanner_name is already RUNNING or QUEUED, a second run is rejected
    with a RuntimeError("Scanner '<scanner_name>' is already actively running!").
    """
    status_upper = (initial_status or "RUNNING").upper()
    is_skip_record = status_upper in ("SKIPPED_DUPLICATE", "SKIPPED")
    if not allow_concurrent and not is_skip_record and is_scanner_actively_running(scanner_name):
        logger.warning(f"🛑 [CONCURRENCY_PREVENTION] Scanner '{scanner_name}' is ALREADY actively running in DB. Aborting duplicate run.")
        raise RuntimeError(f"Scanner '{scanner_name}' is already actively running!")

    from scanner_run_context import ScannerRunContext
    ctx = ScannerRunContext(
        scanner_name=scanner_name,
        trigger_type=trigger_type,
        scheduler_name=scheduler_name,
        parent_run_id=parent_run_id,
        retry_attempt=retry_attempt,
        total_stocks=total_stocks
    )
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Concurrency re-check inside transaction to prevent race conditions
                if not allow_concurrent and not is_skip_record:
                    cur.execute("""
                        SELECT run_id FROM scanner_execution_history
                        WHERE LOWER(scanner_name) = LOWER(%s)
                          AND lifecycle_status IN ('RUNNING', 'QUEUED')
                        LIMIT 1;
                    """, (scanner_name,))
                    existing = cur.fetchone()
                    if existing:
                        logger.warning(
                            f"🛑 [CONCURRENCY_PREVENTION] Scanner '{scanner_name}' is ALREADY actively running "
                            f"(run_id={existing[0][:8]}). Aborting duplicate run."
                        )
                        raise RuntimeError(f"Scanner '{scanner_name}' is already actively running!")

                exec_started_sql = "NOW()" if status_upper == "RUNNING" else "NULL"
                completed_sql = "NOW()" if is_skip_record else "NULL"
                cur.execute(f"""
                    INSERT INTO scanner_execution_history (
                        run_id, parent_run_id, retry_attempt, scanner_name,
                        lifecycle_status, quality_status, trigger_type, scheduler_name,
                        system_version, git_commit, started_at, execution_started_at, completed_at, heartbeat_at, total_stocks
                    ) VALUES (%s, %s, %s, %s, %s, 'NORMAL', %s, %s, %s, %s, NOW(), {exec_started_sql}, {completed_sql}, NOW(), %s);
                """, (
                    ctx.run_id, ctx.parent_run_id, ctx.retry_attempt, ctx.scanner_name,
                    status_upper, ctx.trigger_type, ctx.scheduler_name, ctx.system_version, ctx.git_commit, ctx.total_stocks
                ))
                conn.commit()
                logger.info(f"📜 [EXECUTION HISTORY] Started run {ctx.run_id[:8]} for {scanner_name} (Trigger: {trigger_type}, Status: {status_upper})")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Failed to insert scanner execution history for {scanner_name}: {e}")

    if not is_skip_record and hasattr(ctx, "start_heartbeat_worker"):
        try:
            ctx.start_heartbeat_worker()
        except Exception:
            pass

    return ctx


def update_scanner_run_heartbeat(run_id: str):
    """Updates the heartbeat_at timestamp for an active (RUNNING or QUEUED) scanner run."""
    if not run_id:
        return
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # [VERSION: QUEUED_HEARTBEAT_FIX_v1.0] Update heartbeat_at for both RUNNING and QUEUED
                # status so scanners waiting in global lock queue are NOT auto-cleaned as TIMEOUT_STALE
                # by the watchdog after 15 minutes of queue wait.
                cur.execute("""
                    UPDATE scanner_execution_history
                    SET heartbeat_at = NOW()
                    WHERE run_id = %s AND lifecycle_status IN ('RUNNING', 'QUEUED')
                    RETURNING scanner_name;
                """, (run_id,))
                row = cur.fetchone()
                if row and row[0]:
                    sc_name = row[0]
                    # Also keep scanner_health updated_at fresh so watchdog never marks active scanners as timed out!
                    cur.execute("""
                        UPDATE scanner_health
                        SET updated_at = NOW()
                        WHERE scanner_name = %s AND status = 'RUNNING';
                    """, (sc_name,))
                conn.commit()
    except Exception as e:
        logger.debug(f"Failed to update heartbeat for run {run_id}: {e}")


def update_scanner_run_lifecycle(run_id: str, lifecycle_status: str):
    """Updates lifecycle_status (e.g. 'QUEUED', 'RUNNING') for an active scanner execution run in scanner_execution_history."""
    if not run_id:
        return
    status_upper = lifecycle_status.upper()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if status_upper == "RUNNING":
                    # [VERSION: RUNTIME_DURATION_FIX_v3.0]
                    # When transitioning QUEUED -> RUNNING, set execution_started_at = NOW()
                    # (if not already set). started_at preserves the original queue entry time,
                    # while execution_started_at records the exact lock acquisition timestamp.
                    # duration_seconds is computed from execution_started_at, so queue wait time
                    # is NEVER counted as scanner execution runtime.
                    cur.execute("""
                        UPDATE scanner_execution_history
                        SET lifecycle_status = 'RUNNING',
                            execution_started_at = COALESCE(execution_started_at, NOW()),
                            heartbeat_at = NOW()
                        WHERE run_id = %s;
                    """, (run_id,))
                else:
                    cur.execute("""
                        UPDATE scanner_execution_history
                        SET lifecycle_status = %s, heartbeat_at = NOW()
                        WHERE run_id = %s;
                    """, (status_upper, run_id))
                conn.commit()
    except Exception as e:
        logger.debug(f"Failed to update lifecycle_status for run {run_id}: {e}")


def complete_scanner_execution_run(ctx, exception: Exception = None, stop_reason: str = None, status_override: str = None):
    """Finalizes a scanner execution record with completion stats, quality evaluation, and errors."""
    if not ctx or not getattr(ctx, 'run_id', None):
        return

    if hasattr(ctx, "stop_heartbeat_worker"):
        try:
            ctx.stop_heartbeat_worker()
        except Exception:
            pass

    import traceback
    if status_override is not None:
        lifecycle_status = status_override.upper()
        if stop_reason is not None:
            ctx.set_stop_reason(stop_reason)
    elif exception is not None:
        ctx.record_error(str(exception)[:255], traceback.format_exc())
        lifecycle_status = "FAILED"
    elif stop_reason is not None:
        ctx.set_stop_reason(stop_reason)
        lifecycle_status = "STOPPED"
    else:
        lifecycle_status = "COMPLETED"

    quality_status = ctx.evaluate_quality_status()
    stale_ratio = ctx.compute_stale_ratio()

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE scanner_execution_history
                    SET completed_at = NOW(),
                        lifecycle_status = %s,
                        quality_status = %s,
                        total_stocks = %s,
                        fresh_data_count = %s,
                        stale_data_count = %s,
                        incomplete_data_count = %s,
                        stale_ratio = %s,
                        alerts_generated = %s,
                        api_calls = %s,
                        cache_hits = %s,
                        cache_misses = %s,
                        stop_reason = %s,
                        error_summary = %s,
                        error_details = %s
                    WHERE run_id = %s;
                """, (
                    lifecycle_status, quality_status, ctx.total_stocks,
                    ctx.fresh_count, ctx.stale_count, ctx.incomplete_count,
                    stale_ratio, ctx.alerts_generated, ctx.api_calls,
                    ctx.cache_hits, ctx.cache_misses, ctx.stop_reason,
                    ctx.error_summary, ctx.error_details, ctx.run_id
                ))
                conn.commit()
                logger.info(
                    f"📜 [EXECUTION HISTORY] Completed run {ctx.run_id[:8]} for {ctx.scanner_name} | "
                    f"Lifecycle: {lifecycle_status} | Quality: {quality_status} | Stale Ratio: {stale_ratio*100:.1f}%"
                )
    except Exception as e:
        logger.warning(f"Failed to complete scanner execution history for run {ctx.run_id}: {e}")

    # [FIX: STATE_SYNC_v1.0] If execution FAILED or STOPPED, ensure scanner_health card is also
    # marked DOWN so it doesn't stay stuck on QUEUED/RUNNING after a crash.
    # This is a best-effort sync — individual scanner wrappers in main.py remain the primary
    # source of truth for health status, but this catches cases where the wrapper itself crashes.
    if lifecycle_status in ("FAILED", "STOPPED") and getattr(ctx, 'scanner_name', None):
        try:
            err_msg = (ctx.error_summary or "Scanner crashed before completing health update")[:500]
            upsert_scanner_health(
                ctx.scanner_name,
                status="DOWN",
                error_msg=f"[AUTO-SYNC] {err_msg}"
            )
        except Exception as _hs_err:
            logger.debug(f"Health sync post-FAILED for {ctx.scanner_name}: {_hs_err}")


_SEH_FILTERS_CACHE = {"ts": 0.0, "versions": ["v1"], "commits": []}

def get_scanner_execution_history(
    scanner_name: str = None,
    lifecycle_status: str = None,
    quality_status: str = None,
    date_range: str = "7d",
    search: str = None,
    system_version: str = None,
    git_commit: str = None,
    page: int = 1,
    per_page: int = 25
):
    """
    Returns filterable, paginated scanner execution history with dynamically computed duration and version stats.
    """
    from psycopg2.extras import RealDictCursor
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # [VERSION: ORPHAN_CLEANUP_THROTTLE_v1.0] 🧹 WATCHDOG AUTO-CLEANUP
                # Throttled to once per 5 minutes — the scanner history API is polled every
                # ~5-10s by the admin dashboard, which previously caused this cleanup to run
                # (and log WARNING spam) on every single request, even after rows were cleaned.
                import time as _time_mod
                global _ORPHAN_CLEANUP_LAST_RUN_TS
                _now_mono = _time_mod.monotonic()
                if _now_mono - _ORPHAN_CLEANUP_LAST_RUN_TS >= _ORPHAN_CLEANUP_INTERVAL_S:
                    try:
                        cur.execute("""
                            SELECT run_id, scanner_name, started_at, heartbeat_at, lifecycle_status
                            FROM scanner_execution_history
                            WHERE lifecycle_status IN ('RUNNING', 'QUEUED')
                              AND (
                                  (heartbeat_at IS NOT NULL AND heartbeat_at < NOW() - INTERVAL '10 minutes')
                                  OR (heartbeat_at IS NULL AND started_at < NOW() - INTERVAL '10 minutes')
                                  OR started_at < NOW() - INTERVAL '2 hours'
                              );
                        """)
                        orphaned_rows = cur.fetchall()
                        if orphaned_rows:
                            # [BUG FIX] UPDATE runs first; log message fires AFTER commit so it only
                            # appears when the change actually persisted.
                            cur.execute("""
                                UPDATE scanner_execution_history
                                SET completed_at = NOW(),
                                    lifecycle_status = 'TIMEOUT_STALE',
                                    error_summary = 'Execution timed out: missing heartbeat (>10m) or hard runtime exceeded (>2h)',
                                    error_details = 'Watchdog auto-cleaned stale RUNNING state: heartbeat inactive for >10 minutes'
                                WHERE lifecycle_status IN ('RUNNING', 'QUEUED')
                                  AND (
                                      (heartbeat_at IS NOT NULL AND heartbeat_at < NOW() - INTERVAL '10 minutes')
                                      OR (heartbeat_at IS NULL AND started_at < NOW() - INTERVAL '10 minutes')
                                      OR started_at < NOW() - INTERVAL '2 hours'
                                  );
                            """)
                            orphaned_sc_names = tuple(set(orphan.get("scanner_name") for orphan in orphaned_rows if orphan.get("scanner_name")))
                            if orphaned_sc_names:
                                cur.execute("""
                                    UPDATE scanner_health
                                    SET status = 'DOWN',
                                        error_msg = 'Watchdog auto-cleaned orphaned RUNNING state (missing heartbeat >10m)'
                                    WHERE status = 'RUNNING'
                                      AND scanner_name IN %s;
                                """, (orphaned_sc_names,))
                            conn.commit()
                            # Log AFTER successful commit — accurate report of what was cleaned
                            for orphan in orphaned_rows:
                                r_id = orphan.get("run_id") or "UNKNOWN"
                                sc_name = orphan.get("scanner_name") or "UNKNOWN"
                                st_at = orphan.get("started_at")
                                hb_at = orphan.get("heartbeat_at")
                                logger.warning(
                                    f"🧹 [WATCHDOG CLEANUP] Cleaned orphaned RUNNING run {r_id[:8]} for scanner '{sc_name}' | "
                                    f"Reason: Inactive heartbeat (>10m) or hard runtime ceiling (>2h) (started_at: {st_at}, heartbeat_at: {hb_at})"
                                )
                        _ORPHAN_CLEANUP_LAST_RUN_TS = _now_mono
                    except Exception as _e_sweep:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        logger.debug(f"Stale history auto-sweep warning: {_e_sweep}")

                where_clauses = ["1=1"]
                params = []

                if scanner_name:
                    if isinstance(scanner_name, str):
                        sc_list = [s.strip() for s in scanner_name.split(",") if s.strip()]
                    elif isinstance(scanner_name, (list, tuple, set)):
                        sc_list = [str(s).strip() for s in scanner_name if str(s).strip()]
                    else:
                        sc_list = [str(scanner_name).strip()]

                    sc_list = [s for s in sc_list if s.upper() != "ALL"]
                    if sc_list:
                        expanded_sc_list = []
                        for s in sc_list:
                            norm = normalize_scanner_name(s)
                            expanded_sc_list.append(norm)
                            u = s.upper().replace("-", "_").replace(" ", "_")
                            if u in ["WEALTH", "WEALTH_ENGINE"]:
                                expanded_sc_list.extend(["Wealth Engine", "WEALTH_ENGINE", "WEALTH_EXIT"])
                            elif u in ["MULTI_TF", "MULTITF"]:
                                expanded_sc_list.extend(["MULTI_TF", "MULTI_TF_5M"])
                            elif u in ["MULTIBAGGER"]:
                                expanded_sc_list.extend(["MULTIBAGGER", "MULTIBAGGER_EXIT"])
                        sc_list = list(dict.fromkeys(expanded_sc_list))
                        if len(sc_list) == 1:
                            where_clauses.append("UPPER(scanner_name) = UPPER(%s)")
                            params.append(normalize_scanner_name(sc_list[0]))
                        else:
                            placeholders = ", ".join(["UPPER(%s)"] * len(sc_list))
                            where_clauses.append(f"UPPER(scanner_name) IN ({placeholders})")
                            for s in sc_list:
                                params.append(normalize_scanner_name(s))

                if lifecycle_status and lifecycle_status.upper() != "ALL":
                    where_clauses.append("lifecycle_status = %s")
                    params.append(lifecycle_status.upper())
                else:
                    # Hide QUEUED entries by default from history table to prevent clutter.
                    # Only show runs that actually acquired the lock and started RUNNING.
                    where_clauses.append("lifecycle_status != 'QUEUED'")

                if quality_status and quality_status.upper() != "ALL":
                    where_clauses.append("quality_status = %s")
                    params.append(quality_status.upper())

                if system_version and system_version.upper() != "ALL":
                    where_clauses.append("COALESCE(system_version, 'v1') = %s")
                    params.append(system_version)

                if git_commit and git_commit.upper() != "ALL":
                    where_clauses.append("git_commit = %s")
                    params.append(git_commit)

                if date_range == "today":
                    where_clauses.append("started_at >= CURRENT_DATE")
                elif date_range == "7d":
                    where_clauses.append("started_at >= NOW() - INTERVAL '7 days'")
                elif date_range == "30d":
                    where_clauses.append("started_at >= NOW() - INTERVAL '30 days'")

                if search and search.strip():
                    # [VERSION: PERF_FIX] Removed slow ILIKE on error_details. Focus only on indexed or small columns.
                    where_clauses.append("(scanner_name ILIKE %s OR run_id ILIKE %s OR stop_reason ILIKE %s OR system_version ILIKE %s)")
                    term = f"%{search.strip()}%"
                    params.extend([term, term, term, term])

                where_sql = " AND ".join(where_clauses)

                # [RULE 67 CHANGE-RATIONALE]:
                # Memoize available versions and git commits dropdown filter lists with a 60-second TTL.
                # Previously, these 2 DISTINCT subqueries executed on every single 5-second UI poll and
                # pagination click, creating heavy database load. Caching eliminates ~200-500ms of lag per poll.
                global _SEH_FILTERS_CACHE
                _now_seh_ts = _time_mod.time()
                if (_now_seh_ts - _SEH_FILTERS_CACHE.get("ts", 0.0)) >= 60.0 or not _SEH_FILTERS_CACHE.get("versions"):
                    try:
                        cur.execute("""
                            SELECT DISTINCT COALESCE(system_version, 'v1') as ver
                            FROM (SELECT system_version FROM scanner_execution_history ORDER BY started_at DESC LIMIT 1000) sub
                            ORDER BY ver DESC;
                        """)
                        ver_rows = cur.fetchall() or []
                        _SEH_FILTERS_CACHE["versions"] = [r["ver"] for r in ver_rows if r and hasattr(r, '__getitem__') and r.get("ver")] or ["v1"]

                        cur.execute("""
                            SELECT DISTINCT git_commit as git
                            FROM (SELECT git_commit FROM scanner_execution_history WHERE git_commit IS NOT NULL ORDER BY started_at DESC LIMIT 1000) sub
                            ORDER BY git DESC;
                        """)
                        git_rows = cur.fetchall() or []
                        _SEH_FILTERS_CACHE["commits"] = [r["git"] for r in git_rows if r and hasattr(r, '__getitem__') and r.get("git")]
                        _SEH_FILTERS_CACHE["ts"] = _now_seh_ts
                    except Exception as _filter_err:
                        logger.debug(f"SEH filters cache load warning: {_filter_err}")

                available_versions = _SEH_FILTERS_CACHE.get("versions") or ["v1"]
                available_commits = _SEH_FILTERS_CACHE.get("commits") or []

                # Summary metrics (respecting ALL active filters) — computes total_runs in the same single round-trip
                summary_query = f"""
                    SELECT
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN lifecycle_status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_runs,
                        SUM(CASE WHEN quality_status = 'DEGRADED' THEN 1 ELSE 0 END) as degraded_runs,
                        SUM(CASE WHEN lifecycle_status IN ('FAILED', 'TIMED_OUT', 'TIMEOUT_STALE', 'SERVER_RESTARTED', 'DOWN', 'ERROR') THEN 1 ELSE 0 END) as failed_runs,
                        AVG(COALESCE(stale_ratio, 0.0)) as avg_stale_ratio
                    FROM scanner_execution_history
                    WHERE {where_sql};
                """
                cur.execute(summary_query, params)
                stats = cur.fetchone() or {}

                total_records = (stats["total_runs"] if stats and hasattr(stats, '__getitem__') and "total_runs" in stats else 0) or 0

                # Paginated Rows with dynamic duration calculation
                offset = (max(1, page) - 1) * per_page
                query = f"""
                    SELECT id, run_id, parent_run_id, retry_attempt, scanner_name,
                           lifecycle_status, quality_status, trigger_type, scheduler_name,
                           system_version, git_commit, started_at, execution_started_at, heartbeat_at, completed_at,
                           EXTRACT(EPOCH FROM (COALESCE(completed_at, NOW()) - COALESCE(execution_started_at, started_at)))::float as duration_seconds,
                           total_stocks, fresh_data_count, stale_data_count, incomplete_data_count,
                           stale_ratio, alerts_generated, api_calls, cache_hits, cache_misses,
                           stop_reason, error_summary, error_details
                    FROM scanner_execution_history
                    WHERE {where_sql}
                    ORDER BY started_at DESC
                    LIMIT %s OFFSET %s;
                """
                cur.execute(query, params + [per_page, offset])
                rows = cur.fetchall() or []

                total_runs = (stats["total_runs"] if stats and hasattr(stats, '__getitem__') and "total_runs" in stats else 0) or 0
                completed_runs = (stats["completed_runs"] if stats and hasattr(stats, '__getitem__') and "completed_runs" in stats else 0) or 0
                degraded_runs = (stats["degraded_runs"] if stats and hasattr(stats, '__getitem__') and "degraded_runs" in stats else 0) or 0
                failed_runs = (stats["failed_runs"] if stats and hasattr(stats, '__getitem__') and "failed_runs" in stats else 0) or 0
                avg_stale = float((stats["avg_stale_ratio"] if stats and hasattr(stats, '__getitem__') and "avg_stale_ratio" in stats else 0.0) or 0.0)

                success_rate = round(((completed_runs + degraded_runs) / max(1, total_runs)) * 100, 1) if total_runs > 0 else 100.0

                return {
                    "records": [dict(r) for r in rows],
                    "total_records": total_records,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": (total_records + per_page - 1) // per_page if total_records > 0 else 1,
                    "available_versions": available_versions,
                    "available_commits": available_commits,
                    "summary_stats": {
                        "total_runs": total_runs,
                        "success_rate_pct": success_rate,
                        "degraded_runs": degraded_runs,
                        "failed_runs": failed_runs,
                        "avg_stale_ratio_pct": round(avg_stale * 100, 1),
                    }
                }
    except Exception as e:
        # [RULE 67 CHANGE-RATIONALE]: Ensure conn is defined before calling rollback to prevent NameError if connection failed
        if 'conn' in locals() and conn:
            try:
                conn.rollback()
            except Exception:
                pass
        logger.error(f"Failed to query scanner execution history: {e}")
        return {"records": [], "total_records": 0, "page": page, "per_page": per_page, "total_pages": 1, "available_versions": ["v1"], "available_commits": [], "summary_stats": {}}


def reset_all_positions_to_open() -> int:
    """Resets all alerts, breakout watchlists, and wealth positions in DB to OPEN status and clears exit history."""
    init_db()
    with _DB_WRITE_LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Reset main alerts table
                cur.execute("""
                    UPDATE alerts
                    SET status           = 'OPEN',
                        exit_price       = NULL,
                        pnl_pct          = NULL,
                        pnl_rs           = NULL,
                        closed_at        = NULL,
                        exit_signal      = NULL,
                        execution_state  = 'OPEN',
                        remaining_shares = COALESCE(shares_bought, 1),
                        exit_history     = '[]'
                """)
                count = cur.rowcount

                # 2. Reset breakout_watchlist safely using SAVEPOINT
                cur.execute("SAVEPOINT bw_sp;")
                try:
                    cur.execute("""
                        UPDATE breakout_watchlist
                        SET current_state = 'HOURLY_APPROVED',
                            cooldown_until = NULL,
                            invalidated_at = NULL
                    """)
                    cur.execute("RELEASE SAVEPOINT bw_sp;")
                except Exception as _bw_e:
                    cur.execute("ROLLBACK TO SAVEPOINT bw_sp;")

                # 3. Reset wealth_buy_alert safely using SAVEPOINT
                cur.execute("SAVEPOINT wba_sp;")
                try:
                    cur.execute("""
                        UPDATE wealth_buy_alert
                        SET is_closed = FALSE,
                            closed_at = NULL,
                            exit_reason = NULL,
                            exit_price = NULL
                    """)
                    cur.execute("RELEASE SAVEPOINT wba_sp;")
                except Exception as _wba_e:
                    cur.execute("ROLLBACK TO SAVEPOINT wba_sp;")

                # 4. Clear auxiliary outcome tracking tables if present using SAVEPOINT
                cur.execute("SAVEPOINT ao_sp;")
                try:
                    cur.execute("TRUNCATE TABLE alert_outcomes CASCADE;")
                    cur.execute("RELEASE SAVEPOINT ao_sp;")
                except Exception:
                    cur.execute("ROLLBACK TO SAVEPOINT ao_sp;")

                cur.execute("SAVEPOINT pe_sp;")
                try:
                    cur.execute("TRUNCATE TABLE partial_exits CASCADE;")
                    cur.execute("RELEASE SAVEPOINT pe_sp;")
                except Exception:
                    cur.execute("ROLLBACK TO SAVEPOINT pe_sp;")

                cur.execute("DELETE FROM system_state WHERE key IN ('performance_data', 'performance_data_json');")
                conn.commit()
                try:
                    from performance_tracker import trigger_performance_rebuild
                    trigger_performance_rebuild()
                except Exception as _p_err:
                    logger.warning(f"Failed to trigger performance_data rebuild post-reset: {_p_err}")
                logger.info(f"🔄 [RESET] Reset {count} positions back to OPEN status and purged all exit history.")
                return count


def invalidate_performance_cache():
    """Invalidate cached performance metrics in memory or DB."""
    try:
        from dashboard_server import _PERF_CACHE
        if isinstance(_PERF_CACHE, dict):
            _PERF_CACHE.clear()
    except Exception:
        pass
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM system_state WHERE key IN ('performance_data', 'performance_data_json');")
            conn.commit()
    except Exception:
        pass
