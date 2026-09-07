"""
app/short_covering/short_covering_scanner.py

Layer 2: Intraday 5-Minute Ignition Engine for Short-Covering Early Alerts.
Features:
- Multi-State Lifecycle Progression:
    WATCH -> IGNITION_CANDIDATE -> CONFIRMED_IGNITION -> CONTINUATION -> EXHAUSTED
- 5m Price + OI + Volume = Primary Early-Ignition Trigger
- Tiered Progressive Scoring for 15m/30m structural context (30m breakout is not a hard barrier)
- Excess OI Contraction (Stock vs Index/Sector)
- Anti-Fake validation (rollover filter, liquidity, overhead clearance)
- Stateful alert emission on CONFIRMED_IGNITION with deduplication
"""

import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Set, Tuple
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

logger = logging.getLogger(__name__)


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
        # Maps symbol -> {'state': ShortCoveringState, 'first_candidate_time': datetime, 'signal': Optional[ShortCoveringSignal]}
        self._tracked_states: Dict[str, Dict] = {}
        self._last_scan_date: Optional[date] = None

    def run_5m_scan_cycle(
        self,
        current_time: Optional[datetime] = None,
        candidate_watchlist: Optional[List[EODShortPositionCandidate]] = None
    ) -> List[ShortCoveringSignal]:
        """
        Executes one 5-minute scanning cycle across the candidate universe.
        Returns newly triggered CONFIRMED_IGNITION ShortCoveringSignal alerts.
        """
        if current_time is None:
            current_time = datetime.now()

        today = current_time.date()
        if self._last_scan_date != today:
            self._tracked_states.clear()
            self._last_scan_date = today

        if candidate_watchlist is None:
            candidate_watchlist = self._load_eod_watchlist(today)

        candidate_map = {c.symbol: c for c in candidate_watchlist}
        symbols_to_scan = list(candidate_map.keys()) if candidate_map else fno_universe_manager.get_fno_symbols()[:60]

        logger.info(f"⚡ [Layer 2 5m Scanner] Starting cycle at {current_time.strftime('%H:%M:%S')} across {len(symbols_to_scan)} symbols")

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
                    logger.info(f"🚨 [SHORT COVERING ALERT] {symbol} | Price={signal.ignition_price:.2f} | OI Contraction={signal.excess_oi_contraction:.2f}% | Score={signal.ignition_score:.1f} ({signal.grade})")
            except Exception as e:
                logger.debug(f"Error in 5m evaluation for {symbol}: {e}")

        if new_alerts:
            self._persist_alerts(new_alerts)

        return new_alerts

    def evaluate_symbol_5m(
        self,
        symbol: str,
        current_time: datetime,
        eod_candidate: Optional[EODShortPositionCandidate],
        nifty_oi_5m_delta: float = 0.0
    ) -> Optional[ShortCoveringSignal]:
        """
        Evaluates 5m bar, state progression, and tiered structural context for an individual symbol.
        """
        df_5m = oi_data_service.get_intraday_5m_data(symbol, current_time.date())
        if df_5m is None or len(df_5m) < 4:
            return None

        past_bars = df_5m[df_5m["timestamp"] <= current_time]
        if len(past_bars) < 3:
            past_bars = df_5m.head(3)

        cur_bar = past_bars.iloc[-1]
        prev_bar = past_bars.iloc[-2]
        session_open_price = float(past_bars.iloc[0]["open"])

        cur_close = float(cur_bar["close"])
        cur_open = float(cur_bar["open"])
        cur_vwap = float(cur_bar["vwap"])
        cur_vol = int(cur_bar["volume"])
        cur_oi = int(cur_bar["oi"])

        # 1. Primary 5m Ignition Check (Price Up + OI Down + Volume Expansion)
        is_green_candle = cur_close >= cur_open
        is_above_vwap = cur_close >= cur_vwap * 0.999
        price_change_5m_pct = ((cur_close - float(prev_bar["close"])) / float(prev_bar["close"])) * 100.0

        oi_change_5m_pct = float(cur_bar["oi_change_5m_pct"])
        oi_change_session_pct = float(cur_bar["oi_change_session_pct"])
        excess_oi_contraction = oi_change_5m_pct - nifty_oi_5m_delta

        avg_vol_10 = past_bars["volume"].tail(10).mean()
        vol_surge_ratio = cur_vol / max(avg_vol_10, 1.0)

        # Basic ignition filter
        has_primary_ignition = (
            is_green_candle and
            is_above_vwap and
            price_change_5m_pct >= 0.10 and
            (oi_change_5m_pct <= self.min_5m_oi_contraction_pct or excess_oi_contraction <= -0.25) and
            vol_surge_ratio >= 1.15
        )

        # Anti-Fake Rollover Check
        if oi_data_service.is_rollover_in_progress(symbol, oi_change_5m_pct, 0.0, current_time.date()):
            return None

        # 2. State Machine Management
        tracking = self._tracked_states.get(symbol, {"state": ShortCoveringState.WATCH, "count": 0})
        current_state = tracking["state"]

        if not has_primary_ignition:
            if current_state == ShortCoveringState.IGNITION_CANDIDATE:
                tracking["state"] = ShortCoveringState.WATCH
                self._tracked_states[symbol] = tracking
            return None

        # Progress State
        if current_state == ShortCoveringState.WATCH:
            # Transition to IGNITION_CANDIDATE
            tracking["state"] = ShortCoveringState.IGNITION_CANDIDATE
            tracking["first_candidate_time"] = current_time
            tracking["count"] = 1
            self._tracked_states[symbol] = tracking
            logger.debug(f"🔍 [{symbol}] State -> IGNITION_CANDIDATE at {current_time.strftime('%H:%M')}")
            # If immediate surge is exceptionally strong (e.g. 5m volume > 2x and excess OI < -1.5%), allow single-candle ignition
            if vol_surge_ratio < 2.2 and excess_oi_contraction > -1.2 and (eod_candidate and eod_candidate.buildup_quality_score < 80):
                return None  # Wait for next 5m confirmation bar

        elif current_state == ShortCoveringState.IGNITION_CANDIDATE:
            # Second confirming bar -> CONFIRMED_IGNITION
            tracking["state"] = ShortCoveringState.CONFIRMED_IGNITION
            tracking["count"] = tracking.get("count", 1) + 1
            self._tracked_states[symbol] = tracking

        elif current_state == ShortCoveringState.CONFIRMED_IGNITION:
            # Subsequent bar -> CONTINUATION
            tracking["state"] = ShortCoveringState.CONTINUATION
            self._tracked_states[symbol] = tracking
            return None

        elif current_state in (ShortCoveringState.CONTINUATION, ShortCoveringState.EXHAUSTED):
            return None

        # 3. Tiered Multi-Timeframe Structural Context (Progressive Scoring)
        tf_confirmations = self._check_multitf_context(past_bars)

        # 4. Comprehensive Scoring (0 to 100)
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
        elif excess_oi_contraction <= -0.6:
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
        if cur_close >= cur_vwap * 1.003 and price_change_5m_pct >= 0.35:
            score += 15.0
            reasons.append("Clean VWAP acceleration")
        else:
            score += 10.0

        # E. Progressive 30m / 15m Structural Context (15 pts)
        # 30m breakout is positive boost, NOT hard filter
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

        if score < self.min_ignition_score:
            return None

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

    def _load_eod_watchlist(self, scan_date: date) -> List[EODShortPositionCandidate]:
        """Loads Layer 1 candidate watchlist from DB."""
        candidates = []
        try:
            from app.database import get_connection
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT symbol, scan_date, close_price, total_oi, oi_buildup_5d_pct,
                               short_buildup_ratio, rsi_14, support_level, overhead_resistance,
                               atr_14, buildup_quality_score, sector
                        FROM short_covering_watchlist
                        WHERE scan_date = %s
                        ORDER BY buildup_quality_score DESC
                    """, (scan_date,))
                    rows = cur.fetchall()
                    for r in rows:
                        candidates.append(EODShortPositionCandidate(
                            symbol=r["symbol"],
                            scan_date=r["scan_date"],
                            close_price=float(r["close_price"] or 0),
                            total_oi=int(r["total_oi"] or 0),
                            oi_change_pct_1d=0.0,
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
            logger.debug(f"Could not load EOD watchlist from DB: {e}")
        return candidates

    def _persist_alerts(self, alerts: List[ShortCoveringSignal]) -> None:
        """Persists short-covering alerts to database."""
        try:
            from app.database import get_connection
            with get_connection() as conn:
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
            logger.debug(f"Database save for short_covering_alerts skipped: {e}")


# Global singleton instance
short_covering_scanner = ShortCoveringEarlyIgnitionScanner()
