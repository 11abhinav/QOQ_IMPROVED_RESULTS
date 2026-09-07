import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import fcntl
import logging
import time

logger = logging.getLogger(__name__)

import threading
import psycopg2
import zlib

from zoneinfo import ZoneInfo
from datetime import datetime
IST = ZoneInfo("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────────────────
# SCANNER IDENTITY CONFIG  — unique emoji + display name per scanner
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# SCANNER IDENTITY CONFIG  — unique emoji + display name per scanner
# ─────────────────────────────────────────────────────────────────────────────
SCANNER_CONFIG = {
    "wealth_engine":      {"emoji": "💰", "display": "WEALTH ENGINE",      "db_name": "Wealth Engine"},
    "WEALTH_ENGINE":      {"emoji": "💰", "display": "WEALTH ENGINE",      "db_name": "Wealth Engine"},
    "multi_tf_scanner":   {"emoji": "📊", "display": "MULTI-TF SCANNER",    "db_name": "MULTI_TF"},
    "MULTI_TF":           {"emoji": "📊", "display": "MULTI-TF SCANNER",    "db_name": "MULTI_TF"},
    "eod_scanner":        {"emoji": "🌙", "display": "EOD SCANNER",          "db_name": "EOD"},
    "EOD":                {"emoji": "🌙", "display": "EOD SCANNER",          "db_name": "EOD"},
    "reversal_scanner":   {"emoji": "🔄", "display": "REVERSAL SCANNER",     "db_name": "REVERSAL"},
    "REVERSAL":           {"emoji": "🔄", "display": "REVERSAL SCANNER",     "db_name": "REVERSAL"},
    "pullback_scanner":   {"emoji": "📉", "display": "PULLBACK SCANNER",     "db_name": "PULLBACK"},
    "PULLBACK":           {"emoji": "📉", "display": "PULLBACK SCANNER",     "db_name": "PULLBACK"},
    "multibagger":        {"emoji": "🚀", "display": "MULTIBAGGER SCANNER",   "db_name": "MULTIBAGGER"},
    "MULTIBAGGER":        {"emoji": "🚀", "display": "MULTIBAGGER SCANNER",   "db_name": "MULTIBAGGER"},
    "accumulation":       {"emoji": "📦", "display": "ACCUMULATION SCANNER", "db_name": "ACCUMULATION"},
    "ACCUMULATION":       {"emoji": "📦", "display": "ACCUMULATION SCANNER", "db_name": "ACCUMULATION"},
    "daily_builder":      {"emoji": "🏗️", "display": "DAILY BUILDER",       "db_name": "DAILY_BUILDER"},
    "DAILY_BUILDER":      {"emoji": "🏗️", "display": "DAILY BUILDER",       "db_name": "DAILY_BUILDER"},
    "short_covering":     {"emoji": "⚡", "display": "SHORT COVERING",        "db_name": "SHORT_COVERING"},
    "SHORT_COVERING":     {"emoji": "⚡", "display": "SHORT COVERING",        "db_name": "SHORT_COVERING"},
    "short_covering_eod": {"emoji": "⚡", "display": "SHORT COVERING EOD",    "db_name": "SHORT_COVERING_EOD"},
    "SHORT_COVERING_EOD": {"emoji": "⚡", "display": "SHORT COVERING EOD",    "db_name": "SHORT_COVERING_EOD"},
    "short_covering_5m":  {"emoji": "⚡", "display": "SHORT COVERING 5M",     "db_name": "SHORT_COVERING_5M"},
    "SHORT_COVERING_5M":  {"emoji": "⚡", "display": "SHORT COVERING 5M",     "db_name": "SHORT_COVERING_5M"},
}

_BAR_LEN = 30


def _resolve_scanner_identity(scanner_key: str):
    """Canonicalize scanner key and return (emoji, display_name, canonical_db_name). Fail fast if unmapped."""
    from database import normalize_scanner_name
    canonical_db_name = normalize_scanner_name(scanner_key)
    cfg = SCANNER_CONFIG.get(scanner_key) or SCANNER_CONFIG.get(canonical_db_name)
    if not cfg:
        # Fallback to normalized name if valid string
        if canonical_db_name and canonical_db_name != "UNKNOWN":
            cfg = {"emoji": "⚙️", "display": canonical_db_name, "db_name": canonical_db_name}
        else:
            raise RuntimeError(f"🚨 [FAIL-FAST] Unknown scanner lifecycle key: '{scanner_key}'")
    emoji = cfg.get("emoji", "⚙️")
    display = cfg.get("display", canonical_db_name)
    db_name = cfg.get("db_name") or canonical_db_name
    return emoji, display, db_name


def print_scanner_start_banner(scanner_key: str, queued_at: float = None, run_id: str = None) -> float:
    """
    Print a vivid START banner and immediately mark scanner as RUNNING in DB.
    Transitions status QUEUED → RUNNING the instant the global lock is acquired.
    """
    emoji, display, db_name = _resolve_scanner_identity(scanner_key)
    bar = emoji * _BAR_LEN
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    
    queue_wait_str = ""
    if queued_at is not None:
        queue_wait_secs = round(time.monotonic() - queued_at, 1)
        queue_wait_str = f" | Queue Wait: {queue_wait_secs}s"
    
    logger.info(bar)
    logger.info(f"&&&&& {display} STARTED — {ts}{queue_wait_str} &&&&&")
    logger.info(bar)
    
    # ✅ Immediately transition QUEUED → RUNNING in DB
    try:
        from database import upsert_scanner_health
        upsert_scanner_health(db_name, "RUNNING", error_msg="Scan in progress...", run_id=run_id)
        logger.info(f"🟢 [{display}] Status updated: RUNNING (was QUEUED{queue_wait_str})")
    except Exception as _e:
        logger.warning(f"⚠️ Could not update scanner status to RUNNING: {_e}")
    
    return time.monotonic()


def print_scanner_end_banner(scanner_key: str, start_mono: float, run_id: str = None, status: str = "OK", error_msg: str = None) -> None:
    """
    Print a vivid END banner for the given scanner and update scanner_health to OK/DOWN in DB.
    Must be called BEFORE releasing any locks so log order is guaranteed.
    """
    emoji, display, db_name = _resolve_scanner_identity(scanner_key)
    bar = emoji * _BAR_LEN
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    runtime = time.monotonic() - start_mono
    logger.info(bar)
    logger.info(f"##### {display} ENDED — {ts} | Runtime: {runtime:.0f}s #####")
    logger.info(bar)

    try:
        from database import upsert_scanner_health
        upsert_scanner_health(db_name, status=status, error_msg=error_msg, duration_seconds=runtime, run_id=run_id)
        logger.info(f"✅ [{display}] Status updated: {status} (Completed in {runtime:.0f}s)")
    except Exception as _e:
        logger.warning(f"⚠️ Could not update scanner status to {status}: {_e}")

_process_locks = {}
_process_locks_guard = threading.Lock()

def ProcessLock(lock_name: str):
    """Factory returning a reentrant Singleton ProcessLock per lock_name."""
    with _process_locks_guard:
        if lock_name not in _process_locks:
            _process_locks[lock_name] = ProcessLockImpl(lock_name)
        return _process_locks[lock_name]


class ProcessLockImpl:
    """
    True Reentrant Distributed Lock using PostgreSQL Advisory Locks + local threading.RLock.
    Protects against BOTH multiple threads AND multiple distributed containers on Railway.
    """
    def __init__(self, lock_name: str):
        self.lock_name = lock_name
        self.lock_file = f"data/{lock_name}.lock"
        self.lock_fd = None
        self.thread_lock = threading.RLock()
        self.db_conn = None
        # Generate a stable 32-bit integer for the Postgres lock key based on the name
        self.lock_key = zlib.crc32(lock_name.encode('utf-8'))
        self.is_acquired = False
        self._owner_thread = None
        self._recursion_depth = 0
        self._internal_lock = threading.Lock()
        
        # Telemetry fields
        self.lock_owner_scanner = "UNKNOWN"
        self.lock_owner_operation = "UNKNOWN"
        self._wait_start = 0.0
        self._acquire_time = 0.0

    def locked(self) -> bool:
        """Check if the lock is held by any thread."""
        return self.is_acquired or self._recursion_depth > 0

    def acquire(self, blocking: bool = False, timeout: float = -1, owner_scanner: str = "UNKNOWN", operation: str = "UNKNOWN", **kwargs) -> bool:
        current_thread = threading.current_thread().name
        wait_start_mono = time.monotonic()
        with self._internal_lock:
            if self._owner_thread == current_thread and self._recursion_depth > 0:
                self._recursion_depth += 1
                return True

        # [VERSION: SEQUENTIAL_LOCK_FIRST_EXECUTION_v1.0]
        # Never time out arbitrarily when blocking=True. Keep waiting indefinitely until lock is released.
        timeout_val = float(timeout) if timeout is not None and float(timeout) > 0 else -1.0

        # 1. Acquire local Python RLock with heartbeat logging and UI health updates when waiting
        if blocking:
            last_logged_s = 0
            _is_unknown = owner_scanner == "UNKNOWN"
            _log_interval = 60 if _is_unknown else 180
            while True:
                acquired_thread_lock = self.thread_lock.acquire(blocking=True, timeout=2.0)
                if acquired_thread_lock:
                    break
                elapsed_wait = time.monotonic() - wait_start_mono
                if timeout_val > 0 and elapsed_wait >= timeout_val:
                    logger.warning(f"⚠️ [{self.lock_name.upper()}] Thread lock wait timed out ({elapsed_wait:.1f}s >= {timeout_val}s) for {owner_scanner}.")
                    return False
                if int(elapsed_wait) >= last_logged_s + _log_interval:
                    last_logged_s = int(elapsed_wait)
                    active_owner = getattr(self, "lock_owner_scanner", "ACTIVE_SCANNER")
                    _msg = f"⏳ [{self.lock_name.upper()}] Lock held by {active_owner} — {owner_scanner} waiting in queue... (wait time: {last_logged_s}s)"
                    if _is_unknown:
                        logger.debug(_msg)
                    else:
                        logger.info(_msg)
                        try:
                            from database import upsert_scanner_health
                            upsert_scanner_health(owner_scanner, "QUEUED", error_msg=f"Waiting in queue for active scanner lock ({last_logged_s}s)...")
                        except Exception:
                            pass
                
                # 30-minute lock wait admin notification
                if elapsed_wait >= 1800.0 and not getattr(self, "_30m_wait_notified", False):
                    self._30m_wait_notified = True
                    try:
                        from database import insert_notification
                        insert_notification(
                            "warning",
                            f"⚠️ Lock Wait Warning: {owner_scanner}",
                            f"Scanner '{owner_scanner}' has been waiting in queue for lock '{self.lock_name}' for over 30 minutes. Please review system logs."
                        )
                        from push_service import send_push_to_all
                        send_push_to_all(
                            title=f"⚠️ Lock Wait Warning: {owner_scanner}",
                            body=f"Scanner '{owner_scanner}' waiting for lock >30 min. Please review logs.",
                            bypass_throttle=True
                        )
                    except Exception as _notif_err:
                        logger.warning(f"Failed to dispatch 30m lock wait notification: {_notif_err}")

                run_ctx_obj = kwargs.get("run_ctx")
                if run_ctx_obj:
                    try:
                        run_ctx_obj.heartbeat(force=True)
                    except Exception:
                        pass
        else:
            if not self.thread_lock.acquire(blocking=False):
                return False

        # Reset wait timestamp AFTER acquiring thread lock before entering Postgres advisory lock loop
        pg_wait_start_mono = time.monotonic()

        try:
            # 2. Fallback local file lock for non-distributed edge cases
            os.makedirs("data", exist_ok=True)
            if self.lock_fd is None:
                self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_RDWR)
                
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
                
            fcntl.flock(self.lock_fd, flags)

            # 3. True distributed PostgreSQL lock via controlled connection pool
            db_url = os.environ.get("DATABASE_URL")
            if db_url:
                if self.db_conn is None:
                    try:
                        from database import _get_pool, _conn_semaphore
                        p = _get_pool()
                        if p:
                            if _conn_semaphore is not None:
                                try:
                                    _conn_semaphore.acquire(timeout=5.0)
                                except Exception:
                                    pass
                            self.db_conn = p.getconn()
                            self._pool_ref = p
                            self.db_conn.autocommit = True
                    except Exception as pool_err:
                        logger.warning(f"Could not checkout lock connection from pool: {pool_err}")
                        self.db_conn = None

                if self.db_conn is not None:
                    _lock_conn_retries = 0
                    _lock_conn_max_retries = 5
                    _lock_cursor_ok = True
                    while _lock_cursor_ok:
                        try:
                            with self.db_conn.cursor() as cur:
                                if blocking:
                                    last_logged_s = 0
                                    while True:
                                        cur.execute("SELECT pg_try_advisory_lock(%s)", (self.lock_key,))
                                        locked = cur.fetchone()[0]
                                        elapsed = time.monotonic() - pg_wait_start_mono
                                        if locked:
                                            total_wait = time.monotonic() - wait_start_mono
                                            if total_wait > 1.0:
                                                logger.info(f"✅ [{self.lock_name.upper()}] Acquired Postgres lock for {owner_scanner} after {total_wait:.1f}s total wait")
                                            break
                                        if timeout_val > 0 and elapsed >= timeout_val:
                                            logger.warning(f"⚠️ [{self.lock_name.upper()}] Advisory lock wait timed out ({elapsed:.1f}s >= {timeout_val}s) for {owner_scanner}.")
                                            locked = False
                                            break
                                        _pg_log_interval = 60 if owner_scanner == "UNKNOWN" else 180
                                        if int(elapsed) >= last_logged_s + _pg_log_interval:
                                            last_logged_s = int(elapsed)
                                            _msg = f"⏳ [{self.lock_name.upper()}] Postgres advisory lock busy — {owner_scanner} waiting... (elapsed: {last_logged_s}s)"
                                            if owner_scanner == "UNKNOWN":
                                                logger.debug(_msg)
                                            else:
                                                logger.info(_msg)
                                                try:
                                                    from database import upsert_scanner_health
                                                    upsert_scanner_health(owner_scanner, "QUEUED", error_msg=f"Waiting in queue for active scanner lock ({last_logged_s}s)...")
                                                except Exception:
                                                    pass

                                        # 30-minute advisory lock wait admin notification
                                        total_elapsed_wait = time.monotonic() - wait_start_mono
                                        if total_elapsed_wait >= 1800.0 and not getattr(self, "_30m_wait_notified", False):
                                            self._30m_wait_notified = True
                                            try:
                                                from database import insert_notification
                                                insert_notification(
                                                    "warning",
                                                    f"⚠️ Lock Wait Warning: {owner_scanner}",
                                                    f"Scanner '{owner_scanner}' has been waiting in queue for Postgres advisory lock '{self.lock_name}' for over 30 minutes. Please review system logs."
                                                )
                                                from push_service import send_push_to_all
                                                send_push_to_all(
                                                    title=f"⚠️ Lock Wait Warning: {owner_scanner}",
                                                    body=f"Scanner '{owner_scanner}' waiting for lock >30 min. Please review logs.",
                                                    bypass_throttle=True
                                                )
                                            except Exception as _notif_err:
                                                logger.warning(f"Failed to dispatch 30m lock wait notification: {_notif_err}")

                                        run_ctx_obj = kwargs.get("run_ctx")
                                        if run_ctx_obj and int(elapsed) % 15 == 0:
                                            try:
                                                run_ctx_obj.heartbeat(force=True)
                                            except Exception:
                                                pass
                                        time.sleep(1.0)
                                else:
                                    cur.execute("SELECT pg_try_advisory_lock(%s)", (self.lock_key,))
                                    locked = cur.fetchone()[0]

                                if not locked:
                                    raise BlockingIOError("Could not acquire Postgres distributed lock")

                            _lock_cursor_ok = False  # Exited cursor block cleanly — done

                        except BlockingIOError:
                            raise  # Propagate intentional lock failures

                        except Exception as _conn_err:
                            # Dead connection: discard and get a fresh one from the pool
                            _lock_conn_retries += 1
                            if _lock_conn_retries > _lock_conn_max_retries:
                                logger.error(f"❌ [{self.lock_name.upper()}] Lock connection failed {_lock_conn_retries} times for {owner_scanner}. Giving up.")
                                raise
                            logger.warning(f"⚠️ [{self.lock_name.upper()}] Lock connection dropped mid-wait for {owner_scanner} (attempt {_lock_conn_retries}/{_lock_conn_max_retries}): {_conn_err}. Reconnecting...")
                            try:
                                if getattr(self, '_pool_ref', None):
                                    self._pool_ref.putconn(self.db_conn, close=True)
                                else:
                                    self.db_conn.close()
                            except Exception:
                                pass
                            self.db_conn = None
                            # Re-acquire a fresh connection from pool
                            try:
                                from database import _get_pool, _conn_semaphore
                                p = _get_pool()
                                if p:
                                    if _conn_semaphore is not None:
                                        try:
                                            _conn_semaphore.acquire(timeout=5.0)
                                        except Exception:
                                            pass
                                    self.db_conn = p.getconn()
                                    self._pool_ref = p
                                    self.db_conn.autocommit = True
                                    logger.info(f"🔄 [{self.lock_name.upper()}] Lock connection refreshed for {owner_scanner}. Resuming queue wait...")
                                else:
                                    raise RuntimeError("Could not get pool for lock reconnect")
                            except Exception as _reconnect_err:
                                logger.error(f"❌ [{self.lock_name.upper()}] Failed to reconnect lock connection for {owner_scanner}: {_reconnect_err}")
                                raise

            with self._internal_lock:
                self.is_acquired = True
                self._owner_thread = current_thread
                self._recursion_depth = 1
                self.lock_owner_scanner = owner_scanner
                self.lock_owner_operation = operation
                self._wait_start = wait_start_mono
                self._acquire_time = time.monotonic()
                wait_time = self._acquire_time - self._wait_start
                logger.info(f"🔒 [LOCK ACQUIRED] {self.lock_name} | Scanner: {owner_scanner} | Op: {operation} | Wait Time: {wait_time:.2f}s")
            return True
        except (BlockingIOError, IOError):
            self._cleanup_db_conn(close=True)
            try:
                self.thread_lock.release()
            except Exception:
                pass
            return False
        except Exception as e:
            # [VERSION: PROCESS_LOCK_EXC_FIX_v1.0] On DB or system exception, release thread lock and return False
            logger.error(f"Error acquiring distributed lock {self.lock_name}: {e}")
            self._cleanup_db_conn(close=True)
            try:
                self.thread_lock.release()
            except Exception:
                pass
            return False

    def _cleanup_db_conn(self, close=False):
        if self.db_conn is not None:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (self.lock_key,))
            except Exception: pass
            try:
                if getattr(self, '_pool_ref', None):
                    self._pool_ref.putconn(self.db_conn, close=close)
                else:
                    self.db_conn.close()
            except Exception: pass
            self.db_conn = None
            self._pool_ref = None
            try:
                from database import _conn_semaphore
                if _conn_semaphore is not None:
                    _conn_semaphore.release()
            except Exception: pass

    def release(self, force: bool = False):
        with self._internal_lock:
            current_thread = threading.current_thread().name
            if not force and self._owner_thread is not None and self._owner_thread != current_thread:
                logger.warning(f"⚠️ [{self.lock_name.upper()}] Lock release invoked by thread '{current_thread}', but lock owner is '{self._owner_thread}'. Forcing release for scanner '{self.lock_owner_scanner}'.")
            
            self._recursion_depth -= 1
            if self._recursion_depth > 0 and not force:
                return

            self.is_acquired = False
            self._owner_thread = None
            held_time = (time.monotonic() - self._acquire_time) if getattr(self, "_acquire_time", None) else 0.0
            logger.info(f"🔓 [LOCK RELEASED] {self.lock_name} | Scanner: {self.lock_owner_scanner} | Op: {self.lock_owner_operation} | Held Time: {held_time:.2f}s")
            self.lock_owner_scanner = "UNKNOWN"
            self.lock_owner_operation = "UNKNOWN"

        # 1. Unlock and return connection to pool cleanly
        self._cleanup_db_conn()

        # 2. Release local file lock
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                self.lock_fd = None
            except Exception:
                pass

        try:
            self.thread_lock.release()
        except Exception:
            pass

    def locked(self) -> bool:
        """Returns True if the lock is currently held."""
        return bool(self.is_acquired)

    def is_locked(self) -> bool:
        """Alias for locked()."""
        return bool(self.is_acquired)


def release_global_lock_if_held_by(scanner_name: str):
    """Safely force-release the global scanner lock if held by the specified scanner that went DOWN."""
    try:
        lock = ProcessLock("global_scanner_lock")
        if lock.is_acquired and str(getattr(lock, "lock_owner_scanner", "")).upper() == scanner_name.upper():
            logger.warning(f"🚨 [FAIL-SAFE AUTO-RELEASE] Force releasing global scanner lock held by crashed/down scanner: {scanner_name}")
            lock.release(force=True)
    except Exception as e:
        logger.warning(f"Could not auto-release global lock for {scanner_name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BROKER RESOURCE ISOLATION & PRIORITY BANDWIDTH RESERVATION
# ─────────────────────────────────────────────────────────────────────────────
_SCANNER_FETCH_ACTIVE = threading.Event()

def set_scanner_fetch_active(active: bool = True):
    """Signal that a primary high-priority scanner (e.g. MULTI_TF) is actively downloading market data."""
    if active:
        _SCANNER_FETCH_ACTIVE.set()
    else:
        _SCANNER_FETCH_ACTIVE.clear()

def is_scanner_fetch_active() -> bool:
    """Check if a primary scanner fetch burst is currently active."""
    return _SCANNER_FETCH_ACTIVE.is_set()

def wait_for_scanner_fetch_idle(timeout: float = 10.0):
    """If a scanner is actively fetching data, cooperatively yield broker bandwidth."""
    if _SCANNER_FETCH_ACTIVE.is_set():
        _SCANNER_FETCH_ACTIVE.wait(timeout=timeout)

