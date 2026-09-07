"""
app/multibagger_state_machine.py

Institutional Persistent State Machine for Multibagger Scanner.
Decouples Setup Qualification (BUY_ZONE -> ARMED_BUY_ZONE) from
Execution Confirmation (TRIGGER_CANDIDATE -> ALERT_TRIGGERED) with
auditable invalidation exits and strict idempotency.

Lifecycle:
QUALIFIED_SETUP / BUY_ZONE
          │
          ▼
   ARMED_BUY_ZONE ───────────────┬───> INVALIDATED_SUPPORT
          │                      ├───> INVALIDATED_AGE
          ▼                      ├───> INVALIDATED_FUNDAMENTALS
  TRIGGER_CANDIDATE              └───> INVALIDATED_SETUP
          │
          ▼
   ALERT_TRIGGERED
"""

import math
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from zoneinfo import ZoneInfo

from database import get_connection

logger = logging.getLogger("multibagger_state_machine")
IST = ZoneInfo("Asia/Kolkata")

MAX_ARMED_SESSIONS = 15
SETUP_VERSION = "v5.0"

_state_lock = threading.Lock()


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def ensure_multibagger_state_table() -> None:
    """Creates the multibagger_state table if it does not exist."""
    try:
        with get_connection() as conn:
            if hasattr(conn, "is_dummy") and getattr(conn, "is_dummy", False):
                return
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS multibagger_state (
                        setup_id VARCHAR(64) PRIMARY KEY,
                        symbol VARCHAR(32) NOT NULL,
                        state VARCHAR(32) NOT NULL,
                        setup_version VARCHAR(16) NOT NULL DEFAULT 'v5.0',
                        armed_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        armed_trade_date VARCHAR(16) NOT NULL,
                        armed_price NUMERIC NOT NULL,
                        armed_sma50 NUMERIC NOT NULL,
                        armed_atr NUMERIC NOT NULL,
                        buy_zone_low NUMERIC NOT NULL,
                        buy_zone_high NUMERIC NOT NULL,
                        cqs NUMERIC NOT NULL,
                        pas NUMERIC NOT NULL,
                        total_score NUMERIC NOT NULL,
                        conviction_tier VARCHAR(32) NOT NULL,
                        last_evaluated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        last_evaluated_trade_date VARCHAR(16),
                        age_sessions INT NOT NULL DEFAULT 0,
                        invalidation_reason TEXT,
                        triggered_at TIMESTAMP WITH TIME ZONE,
                        trigger_price NUMERIC,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_mb_state_active ON multibagger_state (symbol, state);
                """)
            conn.commit()
    except Exception as e:
        logger.warning(f"Could not verify/create multibagger_state table: {e}")


def get_active_armed_setups(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches all active ARMED_BUY_ZONE setups from database."""
    results = []
    try:
        with get_connection() as conn:
            if hasattr(conn, "is_dummy") and getattr(conn, "is_dummy", False):
                return []
            with conn.cursor() as cur:
                if symbol:
                    cur.execute("""
                        SELECT setup_id, symbol, state, setup_version, armed_at, armed_trade_date,
                               armed_price, armed_sma50, armed_atr, buy_zone_low, buy_zone_high,
                               cqs, pas, total_score, conviction_tier, last_evaluated_at,
                               last_evaluated_trade_date, age_sessions, invalidation_reason,
                               triggered_at, trigger_price
                        FROM multibagger_state
                        WHERE symbol = %s AND state = 'ARMED_BUY_ZONE'
                        ORDER BY armed_at DESC
                    """, (symbol.upper(),))
                else:
                    cur.execute("""
                        SELECT setup_id, symbol, state, setup_version, armed_at, armed_trade_date,
                               armed_price, armed_sma50, armed_atr, buy_zone_low, buy_zone_high,
                               cqs, pas, total_score, conviction_tier, last_evaluated_at,
                               last_evaluated_trade_date, age_sessions, invalidation_reason,
                               triggered_at, trigger_price
                        FROM multibagger_state
                        WHERE state = 'ARMED_BUY_ZONE'
                        ORDER BY armed_at DESC
                    """)
                rows = cur.fetchall()
                cols = [
                    "setup_id", "symbol", "state", "setup_version", "armed_at", "armed_trade_date",
                    "armed_price", "armed_sma50", "armed_atr", "buy_zone_low", "buy_zone_high",
                    "cqs", "pas", "total_score", "conviction_tier", "last_evaluated_at",
                    "last_evaluated_trade_date", "age_sessions", "invalidation_reason",
                    "triggered_at", "trigger_price"
                ]
                for r in rows:
                    results.append(dict(zip(cols, r)))
    except Exception as e:
        logger.warning(f"Could not load active armed multibagger setups: {e}")
    return results


def arm_setup(
    symbol: str,
    price: float,
    sma_50: float,
    atr_14: float,
    buy_zone_low: float,
    buy_zone_high: float,
    cqs: float,
    pas: float,
    total_score: float,
    conviction_tier: str,
    trade_date: Optional[str] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    [RULE 67 CHANGE-RATIONALE: PHASE_3_PERSISTENT_STATE_MACHINE_V1.0]
    Arms a qualified Multibagger setup in the 50 SMA buy zone.
    Strictly idempotent:
    - If already ARMED on the same trade date/setup -> refreshes last_evaluated_at without duplicating.
    - If no active armed setup exists -> creates a new setup_id and persists state as ARMED_BUY_ZONE.
    """
    sym = symbol.upper()
    now_ist = datetime.now(IST)
    cur_trade_date = trade_date or now_ist.strftime("%Y-%m-%d")

    with _state_lock:
        active_arms = get_active_armed_setups(sym)
        if active_arms:
            existing = active_arms[0]
            setup_id = existing["setup_id"]
            # Update evaluation timestamp
            try:
                with get_connection() as conn:
                    if not getattr(conn, "is_dummy", False):
                        with conn.cursor() as cur:
                            cur.execute("""
                                UPDATE multibagger_state
                                SET last_evaluated_at = %s,
                                    last_evaluated_trade_date = %s,
                                    updated_at = %s
                                WHERE setup_id = %s
                            """, (now_ist, cur_trade_date, now_ist, setup_id))
                        conn.commit()
            except Exception as e:
                logger.warning(f"Error updating existing armed state for {sym}: {e}")
            return False, "ALREADY_ARMED", existing

        # Create new armed setup instance
        setup_id = f"mb_{sym}_{cur_trade_date}_{SETUP_VERSION}"
        new_record = {
            "setup_id": setup_id,
            "symbol": sym,
            "state": "ARMED_BUY_ZONE",
            "setup_version": SETUP_VERSION,
            "armed_at": now_ist,
            "armed_trade_date": cur_trade_date,
            "armed_price": round(price, 2),
            "armed_sma50": round(sma_50, 2),
            "armed_atr": round(atr_14, 2),
            "buy_zone_low": round(buy_zone_low, 2),
            "buy_zone_high": round(buy_zone_high, 2),
            "cqs": round(cqs, 1),
            "pas": round(pas, 1),
            "total_score": round(total_score, 1),
            "conviction_tier": conviction_tier,
            "last_evaluated_at": now_ist,
            "last_evaluated_trade_date": cur_trade_date,
            "age_sessions": 0,
            "invalidation_reason": None,
            "triggered_at": None,
            "trigger_price": None
        }

        try:
            with get_connection() as conn:
                if not getattr(conn, "is_dummy", False):
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO multibagger_state (
                                setup_id, symbol, state, setup_version, armed_at, armed_trade_date,
                                armed_price, armed_sma50, armed_atr, buy_zone_low, buy_zone_high,
                                cqs, pas, total_score, conviction_tier, last_evaluated_at,
                                last_evaluated_trade_date, age_sessions, invalidation_reason,
                                triggered_at, trigger_price
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s
                            ) ON CONFLICT (setup_id) DO UPDATE SET
                                last_evaluated_at = EXCLUDED.last_evaluated_at,
                                last_evaluated_trade_date = EXCLUDED.last_evaluated_trade_date,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            setup_id, sym, "ARMED_BUY_ZONE", SETUP_VERSION, now_ist, cur_trade_date,
                            round(price, 2), round(sma_50, 2), round(atr_14, 2), round(buy_zone_low, 2),
                            round(buy_zone_high, 2), round(cqs, 1), round(pas, 1), round(total_score, 1),
                            conviction_tier, now_ist, cur_trade_date, 0, None, None, None
                        ))
                    conn.commit()
            logger.info(f"🎯 [STATE MACHINE] {sym}: ARMED_BUY_ZONE setup persisted ({setup_id}) @ ₹{price:.2f} [Zone: ₹{buy_zone_low:.2f} - ₹{buy_zone_high:.2f}]")
        except Exception as e:
            logger.warning(f"Could not persist new armed state for {sym}: {e}")

        return True, "ARMED_SUCCESS", new_record


def evaluate_armed_trigger(
    armed_setup: Dict[str, Any],
    price_data: Any,
    raw_fundamentals: Optional[Dict[str, Any]] = None,
    entry_confirmed_fn: Optional[Any] = None,
    current_trade_date: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    [RULE 67 CHANGE-RATIONALE: PHASE_3_PERSISTENT_STATE_MACHINE_V1.0]
    Evaluates an active ARMED_BUY_ZONE setup for entry trigger or terminal invalidation.
    Returns: (new_state, reason, updated_record)
    """
    setup_id = armed_setup["setup_id"]
    sym = armed_setup["symbol"]
    now_ist = datetime.now(IST)
    trade_date = current_trade_date or getattr(price_data, 'last_trade_date', '') or now_ist.strftime("%Y-%m-%d")

    cmp = _safe_float(getattr(price_data, 'price', 0.0))
    sma_50 = _safe_float(getattr(price_data, 'sma_50', 0.0), armed_setup["armed_sma50"])
    sma_200 = _safe_float(getattr(price_data, 'sma_200', 0.0))
    atr_14 = _safe_float(getattr(price_data, 'atr_14', 0.0), armed_setup["armed_atr"])
    buy_low = _safe_float(armed_setup.get("buy_zone_low", 0.0))
    buy_high = _safe_float(armed_setup.get("buy_zone_high", 0.0))

    # Calculate session age progression (only increments if trade date changed)
    prev_trade_date = armed_setup.get("last_evaluated_trade_date")
    current_age = int(armed_setup.get("age_sessions", 0))
    if prev_trade_date and trade_date and prev_trade_date != trade_date:
        current_age += 1

    # 1. Invalidation Check: Structural Support Break
    support_floor = min(buy_low, (sma_50 - (0.5 * atr_14)) if sma_50 > 0 and atr_14 > 0 else (cmp * 0.90))
    if sma_200 > 0 and cmp < (sma_200 * 0.96):
        reason = f"Price ₹{cmp:.2f} broke below SMA200 (₹{sma_200:.2f} * 0.96)"
        _persist_state_transition(setup_id, "INVALIDATED_SUPPORT", reason=reason, age_sessions=current_age, last_date=trade_date)
        return "INVALIDATED_SUPPORT", reason, armed_setup

    if cmp < (support_floor * 0.97):
        reason = f"Price ₹{cmp:.2f} broke below buy-zone support floor ₹{support_floor:.2f}"
        _persist_state_transition(setup_id, "INVALIDATED_SUPPORT", reason=reason, age_sessions=current_age, last_date=trade_date)
        return "INVALIDATED_SUPPORT", reason, armed_setup

    # 2. Invalidation Check: Setup Aging
    if current_age > MAX_ARMED_SESSIONS:
        reason = f"Armed setup exceeded {MAX_ARMED_SESSIONS} trading sessions ({current_age} sessions active)"
        _persist_state_transition(setup_id, "INVALIDATED_AGE", reason=reason, age_sessions=current_age, last_date=trade_date)
        return "INVALIDATED_AGE", reason, armed_setup

    # 3. Invalidation Check: Fundamental Degradation
    if raw_fundamentals:
        f_score = raw_fundamentals.get("piotroski_f_score", raw_fundamentals.get("f_score"))
        if f_score is not None:
            try:
                if int(f_score) < 7:
                    reason = f"Piotroski F-Score degraded to {f_score}/9 (< 7)"
                    _persist_state_transition(setup_id, "INVALIDATED_FUNDAMENTALS", reason=reason, age_sessions=current_age, last_date=trade_date)
                    return "INVALIDATED_FUNDAMENTALS", reason, armed_setup
            except Exception:
                pass
        pledge_pct = raw_fundamentals.get("promoter_pledge_pct")
        if pledge_pct is not None:
            try:
                p_val = float(pledge_pct)
                if p_val > 0.15:
                    reason = f"Promoter pledge increased to {p_val*100:.1f}% (> 15%)"
                    _persist_state_transition(setup_id, "INVALIDATED_FUNDAMENTALS", reason=reason, age_sessions=current_age, last_date=trade_date)
                    return "INVALIDATED_FUNDAMENTALS", reason, armed_setup
            except Exception:
                pass

    # 4. Invalidation Check: Price moved far above buy zone without trigger
    if buy_high > 0 and cmp > (buy_high + (1.5 * atr_14)):
        reason = f"Price ₹{cmp:.2f} escaped far above buy zone [₹{buy_low:.2f} - ₹{buy_high:.2f}] without confirmation"
        _persist_state_transition(setup_id, "INVALIDATED_SETUP", reason=reason, age_sessions=current_age, last_date=trade_date)
        return "INVALIDATED_SETUP", reason, armed_setup

    # 5. Execution Confirmation Trigger Check
    if entry_confirmed_fn:
        ec_ok, ec_reason = entry_confirmed_fn(price_data)
        if ec_ok:
            _persist_state_transition(setup_id, "TRIGGER_CANDIDATE", reason="Bullish price action + 2.0x volume expansion confirmed", age_sessions=current_age, last_date=trade_date)
            return "TRIGGER_CANDIDATE", "CONFIRMED", armed_setup

    # Setup remains armed and waiting
    _persist_state_transition(setup_id, "ARMED_BUY_ZONE", reason=None, age_sessions=current_age, last_date=trade_date)
    return "ARMED_BUY_ZONE", "WAITING_CONFIRMATION", armed_setup


def mark_alert_triggered(setup_id: str, trigger_price: float, triggered_at: Optional[datetime] = None) -> bool:
    """
    [RULE 67 CHANGE-RATIONALE: PHASE_3_PERSISTENT_STATE_MACHINE_V1.0]
    Marks a TRIGGER_CANDIDATE as ALERT_TRIGGERED. Strictly idempotent.
    """
    now_ist = triggered_at or datetime.now(IST)
    try:
        with get_connection() as conn:
            if getattr(conn, "is_dummy", False):
                return True
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE multibagger_state
                    SET state = 'ALERT_TRIGGERED',
                        triggered_at = %s,
                        trigger_price = %s,
                        updated_at = %s
                    WHERE setup_id = %s AND state IN ('ARMED_BUY_ZONE', 'TRIGGER_CANDIDATE')
                """, (now_ist, round(trigger_price, 2), now_ist, setup_id))
            conn.commit()
            return True
    except Exception as e:
        logger.warning(f"Could not mark alert triggered for setup {setup_id}: {e}")
        return False


def _persist_state_transition(
    setup_id: str,
    new_state: str,
    reason: Optional[str] = None,
    age_sessions: int = 0,
    last_date: Optional[str] = None
) -> None:
    now_ist = datetime.now(IST)
    try:
        with get_connection() as conn:
            if getattr(conn, "is_dummy", False):
                return
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE multibagger_state
                    SET state = %s,
                        invalidation_reason = %s,
                        age_sessions = %s,
                        last_evaluated_at = %s,
                        last_evaluated_trade_date = %s,
                        updated_at = %s
                    WHERE setup_id = %s
                """, (new_state, reason, age_sessions, now_ist, last_date, now_ist, setup_id))
            conn.commit()
    except Exception as e:
        logger.warning(f"Error persisting state transition {new_state} for {setup_id}: {e}")
