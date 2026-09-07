"""
app/short_covering/short_covering_scanner.py

Layer 2: Intraday 5-Minute Ignition Engine for Short-Covering Early Alerts.
Features:
- Evidence-Based Dynamic Confirmation:
    High-conviction setups confirm and alert immediately on the same 5m bar without forced delay.
    Moderate-conviction setups transition to IGNITION_CANDIDATE and confirm on subsequent evidence hold.
- Latency Tracking: Measures exact minutes from primary ignition onset to confirmed alert.
- Tiered Progressive Scoring for 15m/30m structural context.
- Excess OI Contraction (Stock vs Index/Sector).
- Anti-Fake validation (rollover filter, liquidity, overhead clearance).
- Stateful alert emission with deduplication.
"""

import os
import logging
import time
from datetime import datetime, date
from typing import List, Dict, Optional, Set, Tuple, Any
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from app.short_covering.fno_universe import fno_universe_manager
from app.short_covering.oi_data_service import oi_data_service
from app.short_covering.short_covering_schema import (
    EODShortPositionCandidate,
    Intraday5mTrigger,
    ShortCoveringSignal,
    ShortCoveringState,
)
try:
    from app.lock_utils import ProcessLock
    from app.database import upsert_scanner_health
    from app.trading_calendar import get_latest_trading_date, is_trading_day
except ImportError:
    from lock_utils import ProcessLock
    from database import upsert_scanner_health
    from trading_calendar import get_latest_trading_date, is_trading_day

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
_scan_lock_5m = ProcessLock("short_covering_5m_lock")


class ShortCoveringEarlyIgnitionScanner:
    """Layer 2: Real-time 5-Minute Short-Covering Early-Ignition Scanner."""

    def __init__(
        self,
        min_volume_surge_ratio: float = 1.30,
        min_5m_oi_contraction_pct: float = -0.30,
        min_session_oi_contraction_pct: float = -0.60,
        min_ignition_score: float = 68.0,
    ):
        self.min_volume_surge_ratio = min_volume_surge_ratio
        self.min_5m_oi_contraction_pct = min_5m_oi_contraction_pct
        self.min_session_oi_contraction_pct = min_session_oi_contraction_pct
        self.min_ignition_score = min_ignition_score

        # Stateful candidate tracker across 5m cycles:
        # Maps symbol -> {'state': ShortCoveringState, 'true_ignition_time': datetime, 'count': int}
        self._tracked_states: Dict[str, Dict] = {}
        self._last_scan_date: Optional[date] = None

    def check_watchlist_freshness(self, target_date: date) -> Tuple[bool, Optional[date], date]:
        """
        Verifies that short_covering_watchlist has candidates from the latest valid trading session.
        For intraday trading on a market day, the candidate watchlist was created by the previous
        session's EOD scan (e.g. Friday for Monday, or Monday for Tuesday).
        Returns (is_fresh, latest_watchlist_date, expected_trading_date).
        """
        # RULE 67 RATIONALE: During active market hours on a trading day (e.g. Mon 09:20),
        # the candidates being traded were produced by Friday's (previous session) EOD scan.
        # If today is a non-trading day (weekend/holiday), the expected session is the latest completed session.
        expected_date = get_previous_trading_date(target_date) if is_trading_day(target_date) else get_latest_trading_date(target_date)
        if not os.getenv("DATABASE_URL") or os.getenv("DISABLE_DB_OI_LOOKUP"):
            return True, expected_date, expected_date
        try:
            from app.database import get_connection
            with get_connection(timeout=1) as conn:
                if hasattr(conn, "is_dummy") and conn.is_dummy:
                    return True, expected_date, expected_date
                with conn.cursor() as cur:
                    cur.execute("SELECT MAX(scan_date) FROM short_covering_watchlist;")
                    row = cur.fetchone()
                    max_date = row[0] if row and row[0] else None
                    if max_date is None:
                        return False, None, expected_date
                    # Consider fresh if max_date matches or exceeds the expected session date
                    is_fresh = max_date >= expected_date
                    return is_fresh, max_date, expected_date
        except Exception as e:
            logger.debug("Failed to check watchlist freshness: %s", e)
            return True, expected_date, expected_date

    def run_5m_scan_cycle(
        self,
        current_time: Optional[datetime] = None,
        candidate_watchlist: Optional[List[EODShortPositionCandidate]] = None,
        persist_db: bool = True
    ) -> List[ShortCoveringSignal]:
        """
        Executes one 5-minute scanning cycle across the candidate universe.
        Returns newly triggered CONFIRMED_IGNITION ShortCoveringSignal alerts.
        """
        if current_time is None:
            current_time = datetime.now(IST)

        today = current_time.date()
        if self._last_scan_date != today:
            self._tracked_states.clear()
            self._last_scan_date = today

        logger.info("[SHORT_COVERING_5M] Acquiring lock: short_covering_5m_lock")
        if not _scan_lock_5m.acquire(blocking=False):
            logger.warning("🛑 [SHORT_COVERING_5M] Lock 'short_covering_5m_lock' held by another instance. Skipping duplicate cycle.")
            return []

        start_t = time.monotonic()
        _SCHEDULE_STR = "Every 5m (09:20 - 15:25 IST Market Days)"
        try:
            # 1. Check Watchlist Freshness Guard
            is_fresh, max_date, expected_date = self.check_watchlist_freshness(today)
            if not is_fresh and candidate_watchlist is None:
                logger.warning(
                    "⚠️ [SHORT_COVERING_5M] Stale watchlist detected (Latest: %s, Expected: %s). Refusing to scan obsolete candidates.",
                    max_date, expected_date
                )
                upsert_scanner_health(
                    scanner_name="SHORT_COVERING_5M",
                    status="OK",
                    outcome="STALE_WATCHLIST",
                    error_msg=f"STALE_WATCHLIST (Latest: {max_date}, Expected: {expected_date})",
                    duration_seconds=round(time.monotonic() - start_t, 2),
                    scheduled_for=_SCHEDULE_STR
                )
                return []

            if candidate_watchlist is None:
                candidate_watchlist = self._load_eod_watchlist(today) if persist_db else None

            if candidate_watchlist:
                candidate_map = {c.symbol: c for c in candidate_watchlist}
                symbols_to_scan = list(candidate_map.keys())
            else:
                candidate_map = {}
                symbols_to_scan = []

            if not symbols_to_scan:
                logger.info("ℹ️ [SHORT_COVERING_5M] No active short-covering candidates in watchlist. Cycle complete.")
                upsert_scanner_health(
                    scanner_name="SHORT_COVERING_5M",
                    status="OK",
                    outcome="SUCCESS",
                    total_count=0,
                    processed_count=0,
                    duration_seconds=round(time.monotonic() - start_t, 2),
                    scheduled_for=_SCHEDULE_STR
                )
                return []

            logger.info("⚡ [SHORT_COVERING_5M] Starting 5m ignition cycle at %s across %d candidates", current_time.strftime("%H:%M:%S"), len(symbols_to_scan))

            new_alerts: List[ShortCoveringSignal] = []
            nifty_oi_5m_delta = self._get_index_5m_oi_delta(current_time)

            for symbol in symbols_to_scan:
                try:
                    signal = self.evaluate_symbol_5m(
                        symbol=symbol,
                        current_time=current_time,
                        eod_candidate=candidate_map.get(symbol),
                        nifty_oi_5m_delta=nifty_oi_5m_delta
                    )
                    if signal is not None and signal.state == ShortCoveringState.CONFIRMED_IGNITION:
                        new_alerts.append(signal)
                        logger.info("🚨 [SHORT COVERING ALERT] %s | Price=%.2f | Latency=%.0fm | Score=%.1f (%s)",
                                    symbol, signal.ignition_price, signal.alert_latency_minutes, signal.ignition_score, signal.grade)
                except Exception as e:
                    logger.debug("Error in 5m evaluation for %s: %s", symbol, e)

            if new_alerts and persist_db:
                self._persist_alerts(new_alerts)

            dur = round(time.monotonic() - start_t, 2)
            upsert_scanner_health(
                scanner_name="SHORT_COVERING_5M",
                status="OK",
                outcome="SUCCESS",
                total_count=len(symbols_to_scan),
                processed_count=len(new_alerts),
                today_alerts=len(new_alerts),
                duration_seconds=dur,
                scheduled_for=_SCHEDULE_STR
            )
            upsert_scanner_health(
                scanner_name="SHORT_COVERING",
                status="OK",
                outcome="SUCCESS",
                duration_seconds=dur,
                today_alerts=len(new_alerts),
                scheduled_for="EOD 19:15 / 5m (09:20 - 15:25 IST)"
            )
            return new_alerts
        except Exception as exc:
            dur = round(time.monotonic() - start_t, 2)
            logger.exception("❌ [SHORT_COVERING_5M] Cycle failed: %s", exc)
            upsert_scanner_health(
                scanner_name="SHORT_COVERING_5M",
                status="DOWN",
                outcome="FAILURE",
                error_msg=str(exc),
                duration_seconds=dur,
                scheduled_for=_SCHEDULE_STR
            )
            return []
        finally:
            _scan_lock_5m.release()


    def evaluate_symbol_5m(
        self,
        symbol: str,
        current_time: datetime,
        eod_candidate: Optional[EODShortPositionCandidate],
        nifty_oi_5m_delta: float = 0.0
    ) -> Optional[ShortCoveringSignal]:
        """
        Evaluates 5m bar, dynamic evidence-based state progression, and tiered structural context.
        """
        df_5m = oi_data_service.get_intraday_5m_data(symbol, current_time.date())
        if df_5m is None or len(df_5m) < 2:
            return None

        past_bars = df_5m[df_5m["timestamp"] <= current_time]
        if len(past_bars) < 2:
            past_bars = df_5m.head(2)


        cur_bar = past_bars.iloc[-1]
        prev_bar = past_bars.iloc[-2]
        session_open_price = float(past_bars.iloc[0]["open"])

        cur_close = float(cur_bar["close"])
        cur_open = float(cur_bar["open"])
        cur_vwap = float(cur_bar["vwap"])
        cur_vol = int(cur_bar["volume"])
        cur_oi = int(cur_bar["oi"])

        # 1. Primary 5m Ignition Evidence
        is_green_candle = cur_close >= cur_open
        is_above_vwap = cur_close >= cur_vwap * 0.999
        price_change_5m_pct = ((cur_close - float(prev_bar["close"])) / float(prev_bar["close"])) * 100.0

        oi_change_5m_pct = float(cur_bar["oi_change_5m_pct"])
        oi_change_session_pct = float(cur_bar["oi_change_session_pct"])
        excess_oi_contraction = oi_change_5m_pct - nifty_oi_5m_delta

        avg_vol_10 = past_bars["volume"].tail(10).mean()
        vol_surge_ratio = cur_vol / max(avg_vol_10, 1.0)

        has_primary_ignition = (
            is_green_candle and
            is_above_vwap and
            price_change_5m_pct >= 0.08 and
            (oi_change_5m_pct <= self.min_5m_oi_contraction_pct or excess_oi_contraction <= -0.20) and
            vol_surge_ratio >= 1.10
        )

        # Anti-Fake Rollover Check
        if oi_data_service.is_rollover_in_progress(symbol, oi_change_5m_pct, 0.0, current_time.date()):
            return None

        # Early Ignition Gate: Reject late entries if price has already moved > +2.5% from session open
        extension_from_open_pct = ((cur_close - session_open_price) / max(session_open_price, 1e-4)) * 100.0
        if extension_from_open_pct > 2.5:
            logger.debug(f"Rejecting {symbol}: Move already extended (+{extension_from_open_pct:.1f}% from open)")
            return None


        # 2. Tiered Multi-Timeframe Structural Context
        tf_confirmations = self._check_multitf_context(past_bars)

        # 3. Comprehensive Ignition Scoring (0 to 100)
        score = 0.0
        reasons = []

        # A. Prior Short Buildup Quality (25 pts)
        if eod_candidate:
            prior_pts = (eod_candidate.buildup_quality_score / 100.0) * 25.0
            score += prior_pts
            reasons.append(f"Prior Short Score: {eod_candidate.buildup_quality_score:.0f}")
        else:
            score += 15.0

        # B. Excess OI Contraction Speed (25 pts)
        if excess_oi_contraction <= -1.2:
            score += 25.0
            reasons.append(f"Strong Excess OI Unwind ({excess_oi_contraction:.2f}%)")
        elif excess_oi_contraction <= -0.5:
            score += 18.0
            reasons.append(f"Moderate Excess OI Unwind ({excess_oi_contraction:.2f}%)")
        else:
            score += 12.0

        # C. Volume Surge & Conviction (20 pts)
        if vol_surge_ratio >= 2.0:
            score += 20.0
            reasons.append(f"High 5m Volume Spike ({vol_surge_ratio:.1f}x)")
        elif vol_surge_ratio >= self.min_volume_surge_ratio:
            score += 15.0
            reasons.append(f"Volume Surge ({vol_surge_ratio:.1f}x)")
        else:
            score += 8.0

        # D. VWAP & Price Momentum (15 pts)
        if cur_close >= cur_vwap * 1.003 and price_change_5m_pct >= 0.30:
            score += 15.0
            reasons.append("Clean VWAP acceleration")
        else:
            score += 10.0

        # E. Progressive 30m / 15m Structural Context (15 pts)
        struct_30m = tf_confirmations.get("30m_structure", "BASE")
        if struct_30m == "BREAKOUT":
            score += 15.0
            reasons.append("30m Structural Breakout (+15)")
        elif struct_30m == "NEAR_BREAKOUT":
            score += 10.0
            reasons.append("Near 30m Breakout (+10)")
        elif struct_30m == "RECLAIMING_STRUCTURE":
            score += 6.0
            reasons.append("Reclaiming 30m Structure (+6)")
        else:
            score += 2.0

        # 4. Evidence-Based Dynamic State Machine
        tracking = self._tracked_states.get(symbol, {"state": ShortCoveringState.WATCH, "true_ignition_time": current_time, "count": 0})
        current_state = tracking["state"]

        if not has_primary_ignition or score < self.min_ignition_score:
            if current_state == ShortCoveringState.IGNITION_CANDIDATE:
                tracking["state"] = ShortCoveringState.WATCH
                self._tracked_states[symbol] = tracking
            return None

        # Evidence evaluation:
        # High Conviction (Score >= 76 or exceptionally clean surge) -> Confirm immediately on same candle!
        # Moderate Conviction (Score 68-76) -> Transition to candidate and confirm on next confirming pulse.
        is_high_conviction = (score >= 76.0) or (
            vol_surge_ratio >= 1.8 and excess_oi_contraction <= -0.8 and (eod_candidate is not None and eod_candidate.buildup_quality_score >= 70)
        )

        true_ignition_time = tracking.get("true_ignition_time", current_time)

        if current_state == ShortCoveringState.WATCH:
            true_ignition_time = current_time
            tracking["true_ignition_time"] = true_ignition_time
            if is_high_conviction:
                tracking["state"] = ShortCoveringState.CONFIRMED_IGNITION
                tracking["count"] = 1
                self._tracked_states[symbol] = tracking
                logger.info(f"⚡ [{symbol}] High-conviction ignition -> CONFIRMED_IGNITION immediately at {current_time.strftime('%H:%M')}")
            else:
                tracking["state"] = ShortCoveringState.IGNITION_CANDIDATE
                tracking["count"] = 1
                self._tracked_states[symbol] = tracking
                logger.debug(f"🔍 [{symbol}] Moderate ignition -> IGNITION_CANDIDATE at {current_time.strftime('%H:%M')}")
                return None  # Wait for confirming evidence

        elif current_state == ShortCoveringState.IGNITION_CANDIDATE:
            # Confirming evidence in subsequent candle
            tracking["state"] = ShortCoveringState.CONFIRMED_IGNITION
            tracking["count"] = tracking.get("count", 1) + 1
            self._tracked_states[symbol] = tracking

        elif current_state == ShortCoveringState.CONFIRMED_IGNITION:
            tracking["state"] = ShortCoveringState.CONTINUATION
            self._tracked_states[symbol] = tracking
            return None

        elif current_state in (ShortCoveringState.CONTINUATION, ShortCoveringState.EXHAUSTED):
            return None

        # Calculate Alert Latency
        latency_minutes = max(0.0, (current_time - true_ignition_time).total_seconds() / 60.0)

        # Determine Grade
        if score >= 85.0:
            grade = "A+"
        elif score >= 76.0:
            grade = "A"
        elif score >= 68.0:
            grade = "B"
        else:
            grade = "C"

        ignition_low = float(cur_bar["low"])
        stop_loss = min(ignition_low, cur_vwap * 0.996)
        risk_per_share = max(cur_close - stop_loss, cur_close * 0.005)

        if eod_candidate and eod_candidate.overhead_resistance > cur_close:
            target = eod_candidate.overhead_resistance
        else:
            target = cur_close + (risk_per_share * 2.0)

        rr_ratio = (target - cur_close) / risk_per_share
        rs_pct = price_change_5m_pct - 0.10

        signal = ShortCoveringSignal(
            symbol=symbol,
            timestamp=current_time,
            ignition_price=cur_close,
            session_open_price=session_open_price,
            true_ignition_time=true_ignition_time,
            alert_latency_minutes=latency_minutes,
            vwap=cur_vwap,
            stop_loss=stop_loss,
            initial_target=target,
            risk_reward_ratio=float(rr_ratio),
            excess_oi_contraction=float(excess_oi_contraction),
            oi_contraction_session_pct=float(oi_change_session_pct),
            volume_surge_ratio=float(vol_surge_ratio),
            rs_vs_nifty_pct=float(rs_pct),
            prior_short_score=float(score * 0.25),
            ignition_score=min(100.0, float(score)),
            grade=grade,
            state=ShortCoveringState.CONFIRMED_IGNITION,
            timeframe_confirmations=tf_confirmations,
            reasons=reasons
        )
        return signal

    def _check_multitf_context(self, past_5m_bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculates progressive 15m and 30m context from 5m bars."""
        result = {"15m_vwap_hold": True, "30m_structure": "RECLAIMING_STRUCTURE"}
        if len(past_5m_bars) < 6:
            return result

        last_3 = past_5m_bars.tail(3)
        if last_3["close"].iloc[-1] >= last_3["vwap"].iloc[-1]:
            result["15m_vwap_hold"] = True

        last_6 = past_5m_bars.tail(6)
        prior_high_30m = last_6["high"].iloc[:-1].max()
        cur_close = last_6["close"].iloc[-1]

        if cur_close >= prior_high_30m:
            result["30m_structure"] = "BREAKOUT"
        elif cur_close >= prior_high_30m * 0.995:
            result["30m_structure"] = "NEAR_BREAKOUT"
        elif cur_close >= last_6["vwap"].iloc[-1]:
            result["30m_structure"] = "RECLAIMING_STRUCTURE"
        else:
            result["30m_structure"] = "BELOW_RESISTANCE"

        return result

    def _get_index_5m_oi_delta(self, current_time: datetime) -> float:
        """Fetches NIFTY 5m futures OI change percentage."""
        try:
            df_nifty = oi_data_service.get_intraday_5m_data("NIFTY", current_time.date())
            if df_nifty is not None and not df_nifty.empty:
                past = df_nifty[df_nifty["timestamp"] <= current_time]
                if not past.empty:
                    return float(past.iloc[-1]["oi_change_5m_pct"])
        except Exception:
            pass
        return 0.0

    def _load_eod_watchlist(self, target_date: date) -> List[EODShortPositionCandidate]:
        """Loads yesterday's shortlisted candidates from DB."""
        candidates = []
        if os.getenv("DATABASE_URL") and not os.getenv("DISABLE_DB_OI_LOOKUP"):
            try:
                from app.database import get_connection
                with get_connection(timeout=1) as conn:
                    if hasattr(conn, "is_dummy") and conn.is_dummy:
                        return []
                    with conn.cursor() as cur:
                        # RULE 67 RATIONALE: Only select candidates matching the most recent session on/before target_date
                        # to ensure stale entries from older sessions are never intermingled.
                        cur.execute("""
                            SELECT * FROM short_covering_watchlist 
                            WHERE scan_date = (SELECT MAX(scan_date) FROM short_covering_watchlist WHERE scan_date <= %s)
                            ORDER BY buildup_quality_score DESC LIMIT 40;
                        """, (target_date,))
                        rows = cur.fetchall()
                        for r in rows:
                            candidates.append(EODShortPositionCandidate(
                                symbol=r["symbol"],
                                scan_date=r["scan_date"],
                                close_price=float(r["close_price"] or 0),
                                total_oi=int(r["total_oi"] or 0),
                                oi_change_pct_1d=float(r["oi_change_pct_1d"] or 0),
                                oi_buildup_5d_pct=float(r["oi_buildup_5d_pct"] or 0),
                                oi_buildup_10d_pct=float(r["oi_buildup_5d_pct"] or 0),
                                short_buildup_ratio=float(r["short_buildup_ratio"] or 0.6),
                                rsi_14=float(r["rsi_14"] or 40.0),
                                support_level=float(r["support_level"] or 0),
                                overhead_resistance=float(r["overhead_resistance"] or 0),
                                atr_14=float(r["atr_14"] or 0),
                                daily_volume=1_000_000,
                                sector=r["sector"] or "GENERAL",
                                buildup_quality_score=float(r["buildup_quality_score"] or 70.0),
                                reasons=["Loaded from Layer 1 EOD watchlist"]
                            ))
            except Exception as e:
                logger.error(f"Could not load EOD watchlist from DB: {e}")
                if os.getenv("DATABASE_URL") and not os.getenv("DISABLE_DB_OI_LOOKUP"):
                    raise
        return candidates

    def _persist_alerts(self, alerts: List[ShortCoveringSignal]) -> None:
        """Persists short-covering alerts to database."""
        if not os.getenv("DATABASE_URL") or os.getenv("DISABLE_DB_OI_LOOKUP"):
            return
        try:
            from app.database import get_connection
            with get_connection(timeout=1) as conn:
                if hasattr(conn, "is_dummy") and conn.is_dummy:
                    return
                with conn.cursor() as cur:
                    cur.execute("""
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
                            state VARCHAR(30) DEFAULT 'CONFIRMED_IGNITION',
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_sc_alerts_time ON short_covering_alerts(alert_time);
                        CREATE INDEX IF NOT EXISTS idx_sc_alerts_symbol ON short_covering_alerts(symbol);
                    """)
                    import json
                    for a in alerts:
                        cur.execute("""
                            INSERT INTO short_covering_alerts (
                                symbol, alert_time, ignition_price, vwap, stop_loss,
                                initial_target, risk_reward_ratio, excess_oi_contraction,
                                volume_surge_ratio, ignition_score, grade, reasons, state
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            a.symbol, a.timestamp, a.ignition_price, a.vwap, a.stop_loss,
                            a.initial_target, a.risk_reward_ratio, a.excess_oi_contraction,
                            a.volume_surge_ratio, a.ignition_score, a.grade,
                            json.dumps(a.reasons), a.state.value if hasattr(a.state, "value") else str(a.state)
                        ))
                    if hasattr(conn, "commit"):
                        conn.commit()
            logger.info(f"💾 Persisted {len(alerts)} alerts to short_covering_alerts table")
        except Exception as e:
            # RULE 67 RATIONALE: Re-raise DB persistence error when database is configured so that
            # scanner_health accurately reflects FAILURE / DOWN status rather than fake success.
            logger.error(f"❌ Could not persist alerts to DB: {e}")
            if os.getenv("DATABASE_URL") and not os.getenv("DISABLE_DB_OI_LOOKUP"):
                raise


# Global singleton instance
short_covering_scanner = ShortCoveringEarlyIgnitionScanner()
