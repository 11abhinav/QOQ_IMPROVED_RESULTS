"""
app/short_covering/short_covering_backtester.py

Intraday Point-in-Time Replay Backtester for Short-Covering Early-Ignition Scanner.
Objective:
- Replays historical days step-by-step strictly using information available at each 5-minute candle.
- Point-in-Time candidate generation at T-1 EOD.
- Measures forward performance (+15m, +30m, +60m, +120m, EOD close), Session MFE, Session MAE,
  and False-Covering rates.
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np

from app.short_covering.fno_universe import fno_universe_manager
from app.short_covering.oi_data_service import oi_data_service
from app.short_covering.short_position_detector import short_position_detector
from app.short_covering.short_covering_scanner import ShortCoveringEarlyIgnitionScanner
from app.short_covering.short_covering_schema import (
    EODShortPositionCandidate,
    ShortCoveringSignal,
    IntradayReplayMetrics,
)

logger = logging.getLogger(__name__)


class ShortCoveringBacktester:
    """Replays historical 5-minute intraday sessions point-in-time."""

    def __init__(self):
        self.metrics: List[IntradayReplayMetrics] = []

    def run_replay(
        self,
        start_date: date,
        end_date: date,
        sample_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes point-in-time replay over a date range.
        """
        symbols = sample_symbols or fno_universe_manager.get_fno_symbols()[:25]
        self.metrics = []

        cur_date = start_date
        while cur_date <= end_date:
            if cur_date.weekday() < 5:  # Monday to Friday
                self._replay_single_day(cur_date, symbols)
            cur_date += timedelta(days=1)

        summary = self._compute_summary_analytics()
        return summary

    def _replay_single_day(self, trade_date: date, symbols: List[str]) -> None:
        """
        Replays a single trading day step-by-step.
        """
        prev_date = trade_date - timedelta(days=1)
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)

        # 1. Point-in-Time Layer 1 EOD Scan as of T-1
        candidates = short_position_detector.scan_eod_universe(as_of=prev_date, custom_symbols=symbols)
        if not candidates:
            # Fallback: create mock candidates from symbols
            candidates = [
                short_position_detector.evaluate_symbol(s, prev_date)
                for s in symbols
            ]
            candidates = [c for c in candidates if c is not None]

        # 2. Initialize Layer 2 Scanner instance for this day
        scanner = ShortCoveringEarlyIgnitionScanner()

        # 3. Step through 5-minute bars from 09:15 to 15:30
        timestamps = [
            datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15) + timedelta(minutes=5 * i)
            for i in range(75)
        ]

        for ts in timestamps:
            alerts = scanner.run_5m_scan_cycle(current_time=ts, candidate_watchlist=candidates)
            for alert in alerts:
                metric = self._evaluate_alert_forward_path(alert, trade_date)
                if metric:
                    self.metrics.append(metric)

    def _evaluate_alert_forward_path(
        self,
        alert: ShortCoveringSignal,
        trade_date: date
    ) -> Optional[IntradayReplayMetrics]:
        """
        Evaluates forward return path from alert timestamp to session close.
        """
        df_5m = oi_data_service.get_intraday_5m_data(alert.symbol, trade_date)
        if df_5m is None or df_5m.empty:
            return None

        # Slice bars from alert time onward
        fwd_bars = df_5m[df_5m["timestamp"] >= alert.timestamp].reset_index(drop=True)
        if len(fwd_bars) < 2:
            return None

        entry_price = alert.ignition_price
        highs = fwd_bars["high"].values
        lows = fwd_bars["low"].values
        closes = fwd_bars["close"].values

        # Maximum Favorable & Adverse Excursion in the session
        mfe_pct = ((np.max(highs) - entry_price) / entry_price) * 100.0
        mae_pct = ((np.min(lows) - entry_price) / entry_price) * 100.0

        # Fixed forward returns
        ret_15m = ((closes[min(3, len(closes)-1)] - entry_price) / entry_price) * 100.0
        ret_30m = ((closes[min(6, len(closes)-1)] - entry_price) / entry_price) * 100.0
        ret_60m = ((closes[min(12, len(closes)-1)] - entry_price) / entry_price) * 100.0
        ret_120m = ((closes[min(24, len(closes)-1)] - entry_price) / entry_price) * 100.0
        eod_ret = ((closes[-1] - entry_price) / entry_price) * 100.0

        # False covering check: stopped out within first 6 bars (30 min)
        low_30m = np.min(lows[:min(6, len(lows))])
        is_false = low_30m <= alert.stop_loss

        # Next day continuation check (simulate T+1 close)
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        df_next = oi_data_service.get_intraday_5m_data(alert.symbol, next_day)
        next_day_cont = bool(df_next["close"].iloc[-1] > entry_price) if df_next is not None and not df_next.empty else (eod_ret > 0)

        return IntradayReplayMetrics(
            symbol=alert.symbol,
            alert_time=alert.timestamp,
            alert_price=entry_price,
            fwd_return_15m_pct=float(ret_15m),
            fwd_return_30m_pct=float(ret_30m),
            fwd_return_60m_pct=float(ret_60m),
            fwd_return_120m_pct=float(ret_120m),
            eod_return_pct=float(eod_ret),
            mfe_session_pct=float(mfe_pct),
            mae_session_pct=float(mae_pct),
            is_false_covering=bool(is_false),
            next_day_continuation=bool(next_day_cont)
        )

    def _compute_summary_analytics(self) -> Dict[str, Any]:
        """Calculates aggregated performance statistics over all replay alerts."""
        if not self.metrics:
            return {"total_alerts": 0, "message": "No alerts triggered during replay"}

        df = pd.DataFrame([m.dict() for m in self.metrics])

        total = len(df)
        win_count = (df["eod_return_pct"] > 0).sum()
        false_count = df["is_false_covering"].sum()

        summary = {
            "total_alerts": int(total),
            "win_rate_eod_pct": float((win_count / total) * 100.0),
            "false_covering_rate_pct": float((false_count / total) * 100.0),
            "next_day_continuation_rate_pct": float((df["next_day_continuation"].sum() / total) * 100.0),
            "avg_fwd_return_15m_pct": float(df["fwd_return_15m_pct"].mean()),
            "avg_fwd_return_30m_pct": float(df["fwd_return_30m_pct"].mean()),
            "avg_fwd_return_60m_pct": float(df["fwd_return_60m_pct"].mean()),
            "avg_fwd_return_120m_pct": float(df["fwd_return_120m_pct"].mean()),
            "avg_eod_return_pct": float(df["eod_return_pct"].mean()),
            "avg_session_mfe_pct": float(df["mfe_session_pct"].mean()),
            "avg_session_mae_pct": float(df["mae_session_pct"].mean()),
            "profit_factor": float(
                df[df["eod_return_pct"] > 0]["eod_return_pct"].sum() /
                max(abs(df[df["eod_return_pct"] < 0]["eod_return_pct"].sum()), 1e-6)
            ),
        }
        return summary


# Global singleton instance
short_covering_backtester = ShortCoveringBacktester()
