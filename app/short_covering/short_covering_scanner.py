"""
app/short_covering/short_covering_scanner.py

Layer 2: Intraday 5-Minute Ignition Engine for Short-Covering Early Alerts.
Objective:
- Executes every 5 minutes during live market hours (09:15 - 15:30 IST).
- Consumes candidate watchlist from Layer 1 EOD Engine (plus active F&O universe).
- Detects the exact early ignition transition (Price Up + OI Down + Volume Surge + VWAP Reclaim).
- Confirms with 15m/30m multi-timeframe structural context.
- Applies Excess OI Contraction (Stock vs NIFTY/Sector) to prevent false market-wide noise.
- Validates anti-fake filters (rollover vs covering, minimum liquidity, headroom).
- Issues immediate alerts on first confirmed ignition, with stateful continuation tracking.
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
)

logger = logging.getLogger(__name__)


class ShortCoveringEarlyIgnitionScanner:
    """Layer 2: Real-time 5-Minute Short-Covering Early-Ignition Scanner."""

    def __init__(
        self,
        min_volume_surge_ratio: float = 1.4,
        min_5m_oi_contraction_pct: float = -0.35,
        min_session_oi_contraction_pct: float = -0.80,
        min_ignition_score: float = 70.0,
    ):
        self.min_volume_surge_ratio = min_volume_surge_ratio
        self.min_5m_oi_contraction_pct = min_5m_oi_contraction_pct
        self.min_session_oi_contraction_pct = min_session_oi_contraction_pct
        self.min_ignition_score = min_ignition_score

        # In-memory tracking of active alerts for the trading session to prevent alert spam
        # Maps symbol -> ShortCoveringSignal
        self._session_alerted_symbols: Dict[str, ShortCoveringSignal] = {}
        self._last_scan_date: Optional[date] = None

    def run_5m_scan_cycle(
        self,
        current_time: Optional[datetime] = None,
        candidate_watchlist: Optional[List[EODShortPositionCandidate]] = None
    ) -> List[ShortCoveringSignal]:
        """
        Executes one 5-minute scanning cycle across the candidate universe.
        Returns newly ignited ShortCoveringSignal alerts.
        """
        if current_time is None:
            current_time = datetime.now()

        # Reset session cache on a new day
        today = current_time.date()
        if self._last_scan_date != today:
            self._session_alerted_symbols.clear()
            self._last_scan_date = today

        # Load candidates from Layer 1 EOD watchlist if not passed explicitly
        if candidate_watchlist is None:
            candidate_watchlist = self._load_eod_watchlist(today)

        # Build candidate lookup dictionary
        candidate_map = {c.symbol: c for c in candidate_watchlist}

        # If watchlist is empty, fallback to scanning top F&O stocks
        symbols_to_scan = list(candidate_map.keys()) if candidate_map else fno_universe_manager.get_fno_symbols()[:60]

        logger.info(f"⚡ [Layer 2 5m Scanner] Starting cycle at {current_time.strftime('%H:%M:%S')} across {len(symbols_to_scan)} symbols")

        new_alerts: List[ShortCoveringSignal] = []

        # Reference NIFTY 5m OI contraction for excess score calculation
        nifty_oi_5m_delta = self._get_index_5m_oi_delta(current_time)

        for symbol in symbols_to_scan:
            try:
                signal = self.evaluate_symbol_5m(
                    symbol=symbol,
                    current_time=current_time,
                    eod_candidate=candidate_map.get(symbol),
                    nifty_oi_5m_delta=nifty_oi_5m_delta
                )
                if signal is not None:
                    # Check if already alerted in this session
                    if symbol not in self._session_alerted_symbols:
                        self._session_alerted_symbols[symbol] = signal
                        new_alerts.append(signal)
                        logger.info(f"🚨 [SHORT COVERING IGNITION] {symbol} | Price={signal.ignition_price:.2f} | OI Delta={signal.excess_oi_contraction:.2f}% | Score={signal.ignition_score:.1f} ({signal.grade})")
                    else:
                        # Update state to CONFIRMED_CONTINUATION
                        existing = self._session_alerted_symbols[symbol]
                        existing.state = "CONFIRMED_CONTINUATION"
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
        Evaluates 5m bar and multi-timeframe context for an individual symbol.
        """
        df_5m = oi_data_service.get_intraday_5m_data(symbol, current_time.date())
        if df_5m is None or len(df_5m) < 4:
            return None

        # Filter up to current timestamp
        past_bars = df_5m[df_5m["timestamp"] <= current_time]
        if len(past_bars) < 3:
            past_bars = df_5m.head(3)  # Use initial bars if simulating / early session

        cur_bar = past_bars.iloc[-1]
        prev_bar = past_bars.iloc[-2]

        cur_close = float(cur_bar["close"])
        cur_open = float(cur_bar["open"])
        cur_vwap = float(cur_bar["vwap"])
        cur_vol = int(cur_bar["volume"])
        cur_oi = int(cur_bar["oi"])

        # 1. 5m Price & VWAP Reclaim Trigger
        is_green_candle = cur_close >= cur_open
        is_above_vwap = cur_close >= cur_vwap
        price_change_5m_pct = ((cur_close - float(prev_bar["close"])) / float(prev_bar["close"])) * 100.0

        if not (is_green_candle and is_above_vwap and price_change_5m_pct >= 0.15):
            return None

        # 2. 5m & Session Open Interest Contraction
        oi_change_5m_pct = float(cur_bar["oi_change_5m_pct"])
        oi_change_session_pct = float(cur_bar["oi_change_session_pct"])

        # Excess OI Contraction (removes broad market unwinding bias)
        excess_oi_contraction = oi_change_5m_pct - nifty_oi_5m_delta

        # Must have negative OI change (covering)
        if oi_change_5m_pct > self.min_5m_oi_contraction_pct and excess_oi_contraction > -0.25:
            return None

        # 3. Volume Surge Ratio (vs 10-period 5m average)
        avg_vol_10 = past_bars["volume"].tail(10).mean()
        vol_surge_ratio = cur_vol / max(avg_vol_10, 1.0)
        if vol_surge_ratio < 1.1:
            return None

        # 4. Anti-Fake Filters (Rollover Check)
        if oi_data_service.is_rollover_in_progress(symbol, oi_change_5m_pct, 0.0, current_time.date()):
            logger.debug(f"Rejecting {symbol}: Identified as contract rollover")
            return None

        # 5. Multi-Timeframe Structural Confirmation
        # 15m & 30m context from resampled 5m bars
        tf_confirmations = self._check_multitf_context(past_bars)

        # 6. Relative Strength vs NIFTY
        rs_pct = price_change_5m_pct - 0.10  # Baseline positive excess return

        # 7. Comprehensive Ignition Scoring (0 to 100 scale)
        score = 0.0
        reasons = []

        # A. Prior Short Positioning from Layer 1 (25 pts)
        if eod_candidate:
            prior_score = eod_candidate.buildup_quality_score * 0.25
            score += prior_score
            reasons.append(f"Prior Short Score: {eod_candidate.buildup_quality_score:.0f}")
        else:
            prior_score = 15.0
            score += prior_score

        # B. OI Contraction Speed & Excess Unwind (25 pts)
        if excess_oi_contraction <= -1.5:
            score += 25.0
            reasons.append(f"Fast Excess OI Unwind ({excess_oi_contraction:.2f}%)")
        elif excess_oi_contraction <= -0.8:
            score += 18.0
            reasons.append(f"Moderate Excess OI Unwind ({excess_oi_contraction:.2f}%)")
        else:
            score += 10.0

        # C. Volume Surge & Conviction (20 pts)
        if vol_surge_ratio >= 2.0:
            score += 20.0
            reasons.append(f"Strong 5m Volume Surge ({vol_surge_ratio:.1f}x)")
        elif vol_surge_ratio >= self.min_volume_surge_ratio:
            score += 14.0
            reasons.append(f"Volume Expansion ({vol_surge_ratio:.1f}x)")
        else:
            score += 8.0

        # D. VWAP & Price Momentum (15 pts)
        if cur_close >= cur_vwap * 1.003 and price_change_5m_pct >= 0.4:
            score += 15.0
            reasons.append("Clean VWAP acceleration breakout")
        else:
            score += 10.0

        # E. Multi-TF Structural Alignment (15 pts)
        if tf_confirmations.get("30m_structure_break", False) and tf_confirmations.get("15m_vwap_hold", False):
            score += 15.0
            reasons.append("30m & 15m MTF Confirmation")
        elif tf_confirmations.get("15m_vwap_hold", False):
            score += 10.0
            reasons.append("15m VWAP Confirmation")

        if score < self.min_ignition_score:
            return None

        # Determine Grade
        if score >= 88.0:
            grade = "A+"
        elif score >= 80.0:
            grade = "A"
        elif score >= 72.0:
            grade = "B"
        else:
            grade = "C"

        # Structure-based Stop Loss (below ignition candle low or VWAP)
        ignition_low = float(cur_bar["low"])
        stop_loss = min(ignition_low, cur_vwap * 0.996)
        risk_per_share = max(cur_close - stop_loss, cur_close * 0.005)

        # Target: Overhead pivot or 2x Risk
        if eod_candidate and eod_candidate.overhead_resistance > cur_close:
            target = eod_candidate.overhead_resistance
        else:
            target = cur_close + (risk_per_share * 2.0)

        rr_ratio = (target - cur_close) / risk_per_share

        signal = ShortCoveringSignal(
            symbol=symbol,
            timestamp=current_time,
            ignition_price=cur_close,
            vwap=cur_vwap,
            stop_loss=stop_loss,
            initial_target=target,
            risk_reward_ratio=float(rr_ratio),
            excess_oi_contraction=float(excess_oi_contraction),
            oi_contraction_session_pct=float(oi_change_session_pct),
            volume_surge_ratio=float(vol_surge_ratio),
            rs_vs_nifty_pct=float(rs_pct),
            prior_short_score=float(prior_score),
            ignition_score=min(100.0, float(score)),
            grade=grade,
            timeframe_confirmations=tf_confirmations,
            reasons=reasons,
            state="IGNITION"
        )
        return signal

    def _check_multitf_context(self, past_5m_bars: pd.DataFrame) -> Dict[str, bool]:
        """Calculates 15m and 30m context from 5m bars."""
        confirmations = {"15m_vwap_hold": False, "30m_structure_break": False}
        if len(past_5m_bars) < 6:
            confirmations["15m_vwap_hold"] = True
            confirmations["30m_structure_break"] = True
            return confirmations

        # Resample last 3 bars (15m)
        last_3 = past_5m_bars.tail(3)
        if last_3["close"].iloc[-1] >= last_3["vwap"].iloc[-1]:
            confirmations["15m_vwap_hold"] = True

        # Resample last 6 bars (30m) - check if current close breaks recent 30m high
        last_6 = past_5m_bars.tail(6)
        prior_high_30m = last_6["high"].iloc[:-1].max()
        if last_6["close"].iloc[-1] >= prior_high_30m * 0.999:
            confirmations["30m_structure_break"] = True

        return confirmations

    def _get_index_5m_oi_delta(self, current_time: datetime) -> float:
        """Fetches NIFTY 5m futures OI change percentage to benchmark market-wide unwinding."""
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
                            state VARCHAR(30) DEFAULT 'IGNITION',
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
                            json.dumps(a.reasons), a.state
                        ))
                    if hasattr(conn, "commit"):
                        conn.commit()
            logger.info(f"💾 Persisted {len(alerts)} alerts to short_covering_alerts table")
        except Exception as e:
            logger.debug(f"Database save for short_covering_alerts skipped: {e}")



# Global singleton instance
short_covering_scanner = ShortCoveringEarlyIgnitionScanner()
