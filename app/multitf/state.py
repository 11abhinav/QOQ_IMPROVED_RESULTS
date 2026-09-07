# =====================================================================================
# app/multitf/state.py
# MULTI_TF V2 — State Machine & Persistence
#
# Responsibility: Manages the lifecycle of a consolidation box, maps internal states
# to canonical signal_contract states, and handles database persistence to mtf_v2_watchlist.
#
# Internal States: WATCHING, PRESSURE_BUILDING, ATTEMPT, FAILED_ATTEMPT, BREAKOUT_CONFIRMED, INVALIDATED
# Canonical Maps: WATCH, WATCH, CANDIDATE, WATCH, CONFIRMED, REJECTED
# =====================================================================================

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import psycopg2.extras
from zoneinfo import ZoneInfo

from database import get_connection
from signal_contract import assert_valid_transition

logger = logging.getLogger("multitf.state")
IST = ZoneInfo("Asia/Kolkata")


class MtfSubstate:
    WATCHING = "WATCHING"                     # Good 15m box found, no 5m pressure yet
    PRESSURE_BUILDING = "PRESSURE_BUILDING"   # 5m price near box ceiling, but lacks momentum
    ARMED_PRE_BREAKOUT = "ARMED_PRE_BREAKOUT" # High-quality base coiling at resistance, ready for ignition
    ATTEMPT = "ATTEMPT"                       # 5m live expansion triggering
    FAILED_ATTEMPT = "FAILED_ATTEMPT"         # ATTEMPT failed to close strong
    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED" # Closed 5m bar confirmed breakout
    INVALIDATED = "INVALIDATED"               # Box broken or aged out


def to_canonical(substate: str) -> str:
    """Maps internal MULTI_TF substate to canonical global state."""
    _map = {
        MtfSubstate.WATCHING:           "WATCH",
        MtfSubstate.PRESSURE_BUILDING:  "WATCH",
        MtfSubstate.ARMED_PRE_BREAKOUT: "CANDIDATE",
        MtfSubstate.ATTEMPT:            "CANDIDATE",
        MtfSubstate.FAILED_ATTEMPT:     "WATCH",
        MtfSubstate.BREAKOUT_CONFIRMED: "CONFIRMED",
        MtfSubstate.INVALIDATED:        "REJECTED"
    }
    return _map.get(substate, "UNKNOWN")


@dataclass
class MtfStateRecord:
    symbol: str
    box_id: str
    state: str = "WATCH"
    mtf_substate: str = MtfSubstate.WATCHING
    attempt_count: int = 0
    last_attempt_ts: Optional[datetime] = None
    attempt_started_ts: Optional[datetime] = None
    attempt_bar_boundary: int = 0
    attempt_ttl_expires_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    invalidated_at: Optional[datetime] = None
    invalidation_reason: str = ""
    version: int = 1


def load_state(symbol: str, box_id: str) -> Optional[MtfStateRecord]:
    """Loads the current state for a specific box instance."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT state, mtf_substate, attempt_count, last_attempt_ts,
                           attempt_started_ts, attempt_bar_boundary, attempt_ttl_expires_at,
                           cooldown_until, invalidated_at, invalidation_reason, version
                    FROM mtf_v2_watchlist
                    WHERE symbol = %s AND box_id = %s
                """, (symbol, box_id))
                row = cur.fetchone()
                if row:
                    return MtfStateRecord(
                        symbol=symbol,
                        box_id=box_id,
                        state=row["state"],
                        mtf_substate=row["mtf_substate"],
                        attempt_count=row["attempt_count"],
                        last_attempt_ts=row["last_attempt_ts"],
                        attempt_started_ts=row["attempt_started_ts"],
                        attempt_bar_boundary=row["attempt_bar_boundary"],
                        attempt_ttl_expires_at=row["attempt_ttl_expires_at"],
                        cooldown_until=row["cooldown_until"],
                        invalidated_at=row["invalidated_at"],
                        invalidation_reason=row["invalidation_reason"],
                        version=row.get("version", 1)
                    )
                return None
    except Exception as exc:
        logger.error("[%s] load_state failed: %s", symbol, exc)
        return None


def find_active_box_for_symbol(
    symbol: str,
    box_high: float,
    atr_15m: float,
    tol_pct: float = 0.010,
    tol_atr: float = 0.50
) -> Optional[MtfStateRecord]:
    """
    Finds an existing active (unconfirmed, non-invalidated) box record for this symbol
    whose ceiling level is within tolerance of the newly detected box_high.
    Allows the same underlying structure to evolve smoothly across expanding windows
    (e.g. 8-bar coil -> 12-bar base -> 16-bar shelf) without creating fragmented duplicate rows.
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT box_id, box_high, box_low, state, mtf_substate, attempt_count,
                           last_attempt_ts, attempt_started_ts, attempt_bar_boundary,
                           attempt_ttl_expires_at, cooldown_until, invalidated_at,
                           invalidation_reason, version
                    FROM mtf_v2_watchlist
                    WHERE symbol = %s
                      AND invalidated_at IS NULL
                      AND mtf_substate NOT IN ('CONFIRMED', 'INVALIDATED')
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (symbol,))
                row = cur.fetchone()
                if row:
                    prev_high = float(row["box_high"]) if row["box_high"] is not None else 0.0
                    allowed_delta = max(box_high * tol_pct, atr_15m * tol_atr)
                    if abs(prev_high - box_high) <= allowed_delta:
                        return MtfStateRecord(
                            symbol=symbol,
                            box_id=row["box_id"],
                            state=row["state"],
                            mtf_substate=row["mtf_substate"],
                            attempt_count=row["attempt_count"],
                            last_attempt_ts=row["last_attempt_ts"],
                            attempt_started_ts=row["attempt_started_ts"],
                            attempt_bar_boundary=row["attempt_bar_boundary"],
                            attempt_ttl_expires_at=row["attempt_ttl_expires_at"],
                            cooldown_until=row["cooldown_until"],
                            invalidated_at=row["invalidated_at"],
                            invalidation_reason=row["invalidation_reason"],
                            version=row.get("version", 1)
                        )
    except Exception as exc:
        logger.error("[%s] find_active_box_for_symbol failed: %s", symbol, exc)
    return None



def apply_ttl_and_cooldown(record: MtfStateRecord, ist_now: datetime, current_5m_bars: int) -> bool:
    """
    Evaluates time-to-live for ATTEMPTs and expiry for FAILED_ATTEMPT cooldowns.
    Returns True if the state was mutated.
    """
    mutated = False
    
    # 1. ATTEMPT TTL Check (expires after N completed bars)
    if record.mtf_substate == MtfSubstate.ATTEMPT:
        # If we have advanced 3 full 5m bars since the attempt started without confirming...
        if current_5m_bars >= record.attempt_bar_boundary + 3:
            logger.info("[%s] ATTEMPT TTL expired. Transitioning to FAILED_ATTEMPT.", record.symbol)
            _set_substate(record, MtfSubstate.FAILED_ATTEMPT, ist_now)
            record.cooldown_until = ist_now + timedelta(minutes=30)
            mutated = True
            
    # 2. Cooldown Expiry Check
    if record.mtf_substate == MtfSubstate.FAILED_ATTEMPT and record.cooldown_until:
        if ist_now >= record.cooldown_until:
            logger.info("[%s] FAILED_ATTEMPT cooldown expired. Re-arming to WATCHING.", record.symbol)
            _set_substate(record, MtfSubstate.WATCHING, ist_now)
            record.cooldown_until = None
            mutated = True
            
    return mutated


def invalidate_record(record: MtfStateRecord, ist_now: datetime, reason: str = "STRUCTURAL_BREAK") -> None:
    """Transitions a record to INVALIDATED substate and REJECTED canonical state."""
    _set_substate(record, MtfSubstate.INVALIDATED, ist_now)
    record.invalidated_at = ist_now
    record.invalidation_reason = reason


def handle_box_invalidation(record: MtfStateRecord, c_price: float, box_low: float, atr: float, ist_now: datetime) -> bool:
    """
    Marks the setup INVALIDATED if price breaks significantly below the box structure.
    Returns True if invalidated.
    """
    if record.mtf_substate == MtfSubstate.INVALIDATED:
        return False
        
    break_level = box_low - (0.50 * atr)
    if c_price < break_level:
        logger.info("[%s] Price (%.2f) broke structural support (%.2f). Invalidating box %s.", 
                    record.symbol, c_price, break_level, record.box_id)
        invalidate_record(record, ist_now, "STRUCTURAL_BREAK")
        return True
        
    return False


def _set_substate(record: MtfStateRecord, new_substate: str, ist_now: datetime):
    """Updates substate and ensures canonical state mapping passes validation."""
    old_canonical = record.state
    new_canonical = to_canonical(new_substate)
    
    if old_canonical != new_canonical:
        assert_valid_transition(old_canonical, new_canonical, "MULTI_TF")
        record.state = new_canonical
        
    record.mtf_substate = new_substate


def persist_new_watchlist_candidate(
    candidate_dict: Dict[str, Any]
):
    """
    Inserts a newly discovered 15m consolidation box.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cols = list(candidate_dict.keys())
                vals = [candidate_dict[c] for c in cols]
                placeholders = ",".join(["%s"] * len(cols))
                
                query = f"""
                    INSERT INTO mtf_v2_watchlist ({",".join(cols)})
                    VALUES ({placeholders})
                    ON CONFLICT (symbol, box_id) DO NOTHING
                """
                cur.execute(query, vals)
                conn.commit()
    except Exception as exc:
        logger.error("[%s] persist_new_watchlist_candidate failed: %s", candidate_dict.get("symbol"), exc)


def update_state_in_db(record: MtfStateRecord, updates: Dict[str, Any]) -> bool:
    """
    Applies incremental updates to an existing box record using Optimistic Concurrency Control (CAS).
    Returns True if the update succeeded, False if a concurrent modification occurred.
    """
    updates["state"] = record.state
    updates["mtf_substate"] = record.mtf_substate
    updates["attempt_count"] = record.attempt_count
    updates["last_attempt_ts"] = record.last_attempt_ts
    updates["attempt_started_ts"] = record.attempt_started_ts
    updates["attempt_bar_boundary"] = record.attempt_bar_boundary
    updates["attempt_ttl_expires_at"] = record.attempt_ttl_expires_at
    updates["cooldown_until"] = record.cooldown_until
    updates["invalidated_at"] = record.invalidated_at
    updates["invalidation_reason"] = record.invalidation_reason
    _now_ist = datetime.now(IST)
    updates["updated_at"] = _now_ist
    # [FIX: LAST_EVALUATED_AT_ALWAYS_STAMP_v1.0]
    # last_evaluated_at was only set on first insert (candidate creation), never on re-evaluation.
    # UI shows COALESCE(last_evaluated_at, updated_at, created_at) as 'last_updated'.
    # Without this, stocks stuck in WATCHING with no state change kept showing stale creation timestamps.
    updates["last_evaluated_at"] = _now_ist
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                set_clause = ", ".join([f"{k} = %s" for k in updates.keys()]) + ", version = version + 1"
                vals = list(updates.values())
                vals.extend([record.symbol, record.box_id, record.version])
                
                cur.execute(f"""
                    UPDATE mtf_v2_watchlist
                    SET {set_clause}
                    WHERE symbol = %s AND box_id = %s AND version = %s
                """, vals)
                conn.commit()
                
                if cur.rowcount == 0:
                    logger.warning("[%s] Concurrent update detected for box %s. Transition aborted.", record.symbol, record.box_id)
                    return False
                
                record.version += 1
                return True
    except Exception as exc:
        logger.error("[%s] update_state_in_db failed: %s", record.symbol, exc)
        return False


def get_active_armed_candidates() -> List[Dict[str, Any]]:
    """
    Returns all active, non-invalidated, non-executed candidates from mtf_v2_watchlist
    for lightweight 5-minute monitoring.
    [RULE 67 CHANGE-RATIONALE: ARMED_SUBSTATE_FILTER_v2.0]
    Strictly filters to active urgency substates (PRESSURE_BUILDING, ARMED_PRE_BREAKOUT, ATTEMPT)
    updated within the last 5 days. Excludes passive WATCHING formations to prevent inflating
    5m monitor and Tier 2 lazy-fetch cycles from ~30 candidates to 200+ historical rows.
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT *
                    FROM mtf_v2_watchlist
                    WHERE mtf_substate IN ('PRESSURE_BUILDING', 'ARMED_PRE_BREAKOUT', 'ATTEMPT')
                      AND (cooldown_until IS NULL OR cooldown_until <= NOW())
                      AND invalidated_at IS NULL
                      AND updated_at >= NOW() - INTERVAL '5 days'
                    ORDER BY updated_at DESC;
                """)
                rows = cur.fetchall()
                return [dict(r) for r in rows] if rows else []
    except Exception as exc:
        logger.error("get_active_armed_candidates failed: %s", exc)
        return []


def get_armed_candidate_lifecycle_summary() -> Dict[str, Any]:
    """
    Returns an institutional lifecycle breakdown of all candidates in mtf_v2_watchlist.
    Validates overnight survival and tracking:
    - total_in_watchlist
    - active_substates (PRESSURE_BUILDING, ARMED_PRE_BREAKOUT, ATTEMPT)
    - in_cooldown
    - invalidated
    - live_monitor_eligible
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT(*) FILTER (WHERE mtf_substate IN ('PRESSURE_BUILDING', 'ARMED_PRE_BREAKOUT', 'ATTEMPT') AND invalidated_at IS NULL AND updated_at >= NOW() - INTERVAL '5 days') as active_substates,
                        COUNT(*) FILTER (WHERE cooldown_until IS NOT NULL AND cooldown_until > NOW() AND invalidated_at IS NULL) as in_cooldown,
                        COUNT(*) FILTER (WHERE invalidated_at IS NOT NULL OR updated_at < NOW() - INTERVAL '5 days') as invalidated,
                        COUNT(*) FILTER (WHERE mtf_substate IN ('PRESSURE_BUILDING', 'ARMED_PRE_BREAKOUT', 'ATTEMPT')
                                           AND (cooldown_until IS NULL OR cooldown_until <= NOW())
                                           AND invalidated_at IS NULL
                                           AND updated_at >= NOW() - INTERVAL '5 days') as live_eligible
                    FROM mtf_v2_watchlist;
                """)
                row = cur.fetchone()
                if row:
                    return {
                        "total_in_watchlist": row[0] or 0,
                        "active_substates": row[1] or 0,
                        "in_cooldown": row[2] or 0,
                        "invalidated": row[3] or 0,
                        "live_monitor_eligible": row[4] or 0,
                    }
                return {
                    "total_in_watchlist": 0,
                    "active_substates": 0,
                    "in_cooldown": 0,
                    "invalidated": 0,
                    "live_monitor_eligible": 0,
                }
    except Exception as exc:
        logger.error("get_armed_candidate_lifecycle_summary failed: %s", exc)
        return {
            "total_in_watchlist": 0,
            "active_substates": 0,
            "in_cooldown": 0,
            "invalidated": 0,
            "live_monitor_eligible": 0,
        }


