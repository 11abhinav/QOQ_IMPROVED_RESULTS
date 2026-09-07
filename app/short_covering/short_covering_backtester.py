"""
app/short_covering/short_covering_backtester.py

Intraday Point-in-Time Replay Backtester & Comparative Benchmark Engine for Short-Covering Scanner.
Objective:
- Replays historical sessions step-by-step strictly using point-in-time past information.
- Measures Move Consumed at Alert ('Earlyness Metric') and Move Captured After Alert.
- Evaluates forward metrics (+15m, +30m, +60m, +120m, EOD close, MFE, MAE, false-covering rate, next-day continuation).
- Executes comparative benchmark:
    PROPOSED SHORT-COVERING SCANNER
    vs
    BASELINE A: Simple Price ↑ + OI ↓
    vs
    BASELINE B: Price ↑ + OI ↓ + Volume Surge
- Generates formatted Markdown Certification Report.
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
    StrategyComparativeBenchmark,
    ShortCoveringState,
)

logger = logging.getLogger(__name__)


class ShortCoveringBacktester:
    """Intraday Point-in-Time Replay Backtester & Comparative Benchmark Engine."""

    def __init__(self):
        self.proposed_metrics: List[IntradayReplayMetrics] = []
        self.baseline_a_metrics: List[IntradayReplayMetrics] = []
        self.baseline_b_metrics: List[IntradayReplayMetrics] = []

    def run_replay(
        self,
        start_date: date,
        end_date: date,
        sample_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes point-in-time replay over a date range for the proposed strategy.
        """
        symbols = sample_symbols or fno_universe_manager.get_fno_symbols()[:30]
        self.proposed_metrics = []

        cur_date = start_date
        while cur_date <= end_date:
            if cur_date.weekday() < 5:
                self._replay_single_day(cur_date, symbols)
            cur_date += timedelta(days=1)

        summary = self._compute_summary_analytics(self.proposed_metrics, "PROPOSED_SHORT_COVERING")
        return summary

    def run_comparative_benchmark(
        self,
        start_date: date,
        end_date: date,
        sample_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive 3-way comparative benchmark:
        1. Proposed 2-Layer Early-Ignition Scanner
        2. Baseline A (Price ↑ + OI ↓)
        3. Baseline B (Price ↑ + OI ↓ + Volume Surge)
        """
        symbols = sample_symbols or fno_universe_manager.get_fno_symbols()[:30]
        self.proposed_metrics = []
        self.baseline_a_metrics = []
        self.baseline_b_metrics = []

        cur_date = start_date
        while cur_date <= end_date:
            if cur_date.weekday() < 5:
                self._replay_single_day(cur_date, symbols)
                self._replay_baselines_single_day(cur_date, symbols)
            cur_date += timedelta(days=1)

        proposed_summary = self._compute_summary_analytics(self.proposed_metrics, "PROPOSED_SHORT_COVERING")
        base_a_summary = self._compute_summary_analytics(self.baseline_a_metrics, "BASELINE_A (Price↑ + OI↓)")
        base_b_summary = self._compute_summary_analytics(self.baseline_b_metrics, "BASELINE_B (Price↑ + OI↓ + Vol)")

        report_markdown = self._generate_markdown_report(
            start_date, end_date, len(symbols), proposed_summary, base_a_summary, base_b_summary
        )

        return {
            "proposed_strategy": proposed_summary,
            "baseline_a": base_a_summary,
            "baseline_b": base_b_summary,
            "report_markdown": report_markdown,
        }

    def _replay_single_day(self, trade_date: date, symbols: List[str]) -> None:
        """Replays single day for proposed 2-Layer strategy."""
        prev_date = trade_date - timedelta(days=1)
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)

        # 1. Point-in-Time Layer 1 EOD Scan as of T-1
        candidates = short_position_detector.scan_eod_universe(as_of=prev_date, custom_symbols=symbols)
        if not candidates:
            candidates = [
                short_position_detector.evaluate_symbol(s, prev_date)
                for s in symbols
            ]
            candidates = [c for c in candidates if c is not None]

        # 2. Layer 2 Scanner with stateful progression
        scanner = ShortCoveringEarlyIgnitionScanner()
        timestamps = [
            datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15) + timedelta(minutes=5 * i)
            for i in range(75)
        ]

        for ts in timestamps:
            alerts = scanner.run_5m_scan_cycle(current_time=ts, candidate_watchlist=candidates)
            for alert in alerts:
                metric = self._evaluate_alert_forward_path(alert, trade_date)
                if metric:
                    self.proposed_metrics.append(metric)

    def _replay_baselines_single_day(self, trade_date: date, symbols: List[str]) -> None:
        """Replays simple baselines without Layer 1 prior short filter or state machine."""
        timestamps = [
            datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15) + timedelta(minutes=5 * i)
            for i in range(75)
        ]

        alerted_a: set = set()
        alerted_b: set = set()

        for ts in timestamps:
            for sym in symbols:
                df = oi_data_service.get_intraday_5m_data(sym, trade_date)
                if df is None or len(df) < 5:
                    continue

                past = df[df["timestamp"] <= ts]
                if len(past) < 3:
                    continue

                cur_bar = past.iloc[-1]
                prev_bar = past.iloc[-2]
                close = float(cur_bar["close"])
                open_p = float(cur_bar["open"])
                oi_delta = float(cur_bar["oi_change_5m_pct"])
                vol = int(cur_bar["volume"])
                avg_vol = past["volume"].tail(10).mean()

                # Baseline A: Simple Price ↑ + OI ↓
                if sym not in alerted_a and close > open_p and oi_delta < -0.2:
                    alerted_a.add(sym)
                    sig_a = ShortCoveringSignal(
                        symbol=sym,
                        timestamp=ts,
                        ignition_price=close,
                        session_open_price=float(past.iloc[0]["open"]),
                        vwap=float(cur_bar["vwap"]),
                        stop_loss=float(cur_bar["low"]),
                        initial_target=close * 1.02,
                        risk_reward_ratio=2.0,
                        excess_oi_contraction=oi_delta,
                        oi_contraction_session_pct=float(cur_bar["oi_change_session_pct"]),
                        volume_surge_ratio=float(vol / max(avg_vol, 1)),
                        rs_vs_nifty_pct=0.0,
                        prior_short_score=0.0,
                        ignition_score=50.0,
                        grade="BASELINE_A"
                    )
                    m = self._evaluate_alert_forward_path(sig_a, trade_date)
                    if m:
                        self.baseline_a_metrics.append(m)

                # Baseline B: Price ↑ + OI ↓ + Volume Surge
                if sym not in alerted_b and close > open_p and oi_delta < -0.2 and (vol / max(avg_vol, 1)) >= 1.5:
                    alerted_b.add(sym)
                    sig_b = ShortCoveringSignal(
                        symbol=sym,
                        timestamp=ts,
                        ignition_price=close,
                        session_open_price=float(past.iloc[0]["open"]),
                        vwap=float(cur_bar["vwap"]),
                        stop_loss=float(cur_bar["low"]),
                        initial_target=close * 1.02,
                        risk_reward_ratio=2.0,
                        excess_oi_contraction=oi_delta,
                        oi_contraction_session_pct=float(cur_bar["oi_change_session_pct"]),
                        volume_surge_ratio=float(vol / max(avg_vol, 1)),
                        rs_vs_nifty_pct=0.0,
                        prior_short_score=0.0,
                        ignition_score=60.0,
                        grade="BASELINE_B"
                    )
                    m = self._evaluate_alert_forward_path(sig_b, trade_date)
                    if m:
                        self.baseline_b_metrics.append(m)

    def _evaluate_alert_forward_path(
        self,
        alert: ShortCoveringSignal,
        trade_date: date
    ) -> Optional[IntradayReplayMetrics]:
        """Evaluates forward return path from alert timestamp to session close."""
        df_5m = oi_data_service.get_intraday_5m_data(alert.symbol, trade_date)
        if df_5m is None or df_5m.empty:
            return None

        fwd_bars = df_5m[df_5m["timestamp"] >= alert.timestamp].reset_index(drop=True)
        if len(fwd_bars) < 2:
            return None

        entry_price = alert.ignition_price
        session_open = float(df_5m["open"].iloc[0])
        session_high = float(df_5m["high"].max())
        session_low = float(df_5m["low"].min())

        # Move Consumed at Alert: Percentage of the session range already consumed when alert triggered
        total_session_range = max(session_high - session_low, 1e-4)
        move_consumed_pct = ((entry_price - session_low) / total_session_range) * 100.0

        # Move Captured After Alert: Remaining upside from alert price to session high
        move_captured_pct = ((session_high - entry_price) / entry_price) * 100.0

        highs = fwd_bars["high"].values
        lows = fwd_bars["low"].values
        closes = fwd_bars["close"].values

        mfe_pct = ((np.max(highs) - entry_price) / entry_price) * 100.0
        mae_pct = ((np.min(lows) - entry_price) / entry_price) * 100.0

        ret_15m = ((closes[min(3, len(closes)-1)] - entry_price) / entry_price) * 100.0
        ret_30m = ((closes[min(6, len(closes)-1)] - entry_price) / entry_price) * 100.0
        ret_60m = ((closes[min(12, len(closes)-1)] - entry_price) / entry_price) * 100.0
        ret_120m = ((closes[min(24, len(closes)-1)] - entry_price) / entry_price) * 100.0
        eod_ret = ((closes[-1] - entry_price) / entry_price) * 100.0

        low_30m = np.min(lows[:min(6, len(lows))])
        is_false = low_30m <= alert.stop_loss

        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        df_next = oi_data_service.get_intraday_5m_data(alert.symbol, next_day)
        next_day_cont = bool(df_next["close"].iloc[-1] > entry_price) if df_next is not None and not df_next.empty else (eod_ret > 0)

        return IntradayReplayMetrics(
            symbol=alert.symbol,
            alert_time=alert.timestamp,
            alert_price=entry_price,
            session_open=session_open,
            session_high=session_high,
            session_low=session_low,
            move_consumed_at_alert_pct=float(np.clip(move_consumed_pct, 0.0, 100.0)),
            move_captured_after_alert_pct=float(max(0.0, move_captured_pct)),
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

    def _compute_summary_analytics(self, metrics: List[IntradayReplayMetrics], strategy_name: str) -> Dict[str, Any]:
        """Calculates aggregated performance statistics."""
        if not metrics:
            return {
                "strategy_name": strategy_name,
                "total_alerts": 0,
                "win_rate_eod_pct": 0.0,
                "false_covering_rate_pct": 0.0,
                "next_day_continuation_rate_pct": 0.0,
                "avg_fwd_return_15m_pct": 0.0,
                "avg_fwd_return_30m_pct": 0.0,
                "avg_fwd_return_60m_pct": 0.0,
                "avg_fwd_return_120m_pct": 0.0,
                "avg_eod_return_pct": 0.0,
                "avg_session_mfe_pct": 0.0,
                "avg_session_mae_pct": 0.0,
                "median_move_consumed_pct": 0.0,
                "median_move_captured_pct": 0.0,
                "profit_factor": 0.0,
            }

        df = pd.DataFrame([m.dict() for m in metrics])
        total = len(df)
        win_count = (df["eod_return_pct"] > 0).sum()
        false_count = df["is_false_covering"].sum()

        pos_sum = df[df["eod_return_pct"] > 0]["eod_return_pct"].sum()
        neg_sum = abs(df[df["eod_return_pct"] < 0]["eod_return_pct"].sum())
        pf = float(pos_sum / max(neg_sum, 1e-6)) if neg_sum > 0 else 5.0

        return {
            "strategy_name": strategy_name,
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
            "median_move_consumed_pct": float(df["move_consumed_at_alert_pct"].median()),
            "median_move_captured_pct": float(df["move_captured_after_alert_pct"].median()),
            "profit_factor": pf,
        }

    def _generate_markdown_report(
        self,
        start_date: date,
        end_date: date,
        symbols_count: int,
        proposed: Dict[str, Any],
        base_a: Dict[str, Any],
        base_b: Dict[str, Any],
    ) -> str:
        """Generates markdown table comparing Proposed Scanner vs Baselines."""
        md = f"""# Empirical Validation & Comparative Certification Report
**Period**: {start_date.isoformat()} to {end_date.isoformat()} | **F&O Universe**: {symbols_count} Stocks

| Metric | Proposed 2-Layer Early Ignition | Baseline A (Price↑ + OI↓) | Baseline B (Price↑ + OI↓ + Vol) |
|---|---|---|---|
| **Total Alerts** | {proposed['total_alerts']} | {base_a['total_alerts']} | {base_b['total_alerts']} |
| **Win Rate (EOD Close > Alert)** | **{proposed['win_rate_eod_pct']:.1f}%** | {base_a['win_rate_eod_pct']:.1f}% | {base_b['win_rate_eod_pct']:.1f}% |
| **False-Covering Rate (Stopped out in 30m)** | **{proposed['false_covering_rate_pct']:.1f}%** | {base_a['false_covering_rate_pct']:.1f}% | {base_b['false_covering_rate_pct']:.1f}% |
| **Median Move Consumed at Alert** | **{proposed['median_move_consumed_pct']:.1f}%** *(Early Ignition)* | {base_a['median_move_consumed_pct']:.1f}% | {base_b['median_move_consumed_pct']:.1f}% |
| **Median Move Captured After Alert** | **+{proposed['median_move_captured_pct']:.2f}%** | +{base_a['median_move_captured_pct']:.2f}% | +{base_b['median_move_captured_pct']:.2f}% |
| **Avg Forward Return (+15m)** | **+{proposed['avg_fwd_return_15m_pct']:.2f}%** | +{base_a['avg_fwd_return_15m_pct']:.2f}% | +{base_b['avg_fwd_return_15m_pct']:.2f}% |
| **Avg Forward Return (+30m)** | **+{proposed['avg_fwd_return_30m_pct']:.2f}%** | +{base_a['avg_fwd_return_30m_pct']:.2f}% | +{base_b['avg_fwd_return_30m_pct']:.2f}% |
| **Avg Forward Return (+60m)** | **+{proposed['avg_fwd_return_60m_pct']:.2f}%** | +{base_a['avg_fwd_return_60m_pct']:.2f}% | +{base_b['avg_fwd_return_60m_pct']:.2f}% |
| **Avg Forward Return (+120m)** | **+{proposed['avg_fwd_return_120m_pct']:.2f}%** | +{base_a['avg_fwd_return_120m_pct']:.2f}% | +{base_b['avg_fwd_return_120m_pct']:.2f}% |
| **Avg Session MFE** | **+{proposed['avg_session_mfe_pct']:.2f}%** | +{base_a['avg_session_mfe_pct']:.2f}% | +{base_b['avg_session_mfe_pct']:.2f}% |
| **Avg Session MAE** | **{proposed['avg_session_mae_pct']:.2f}%** | {base_a['avg_session_mae_pct']:.2f}% | {base_b['avg_session_mae_pct']:.2f}% |
| **Next-Day Continuation Rate** | **{proposed['next_day_continuation_rate_pct']:.1f}%** | {base_a['next_day_continuation_rate_pct']:.1f}% | {base_b['next_day_continuation_rate_pct']:.1f}% |
| **Profit Factor** | **{proposed['profit_factor']:.2f}** | {base_a['profit_factor']:.2f} | {base_b['profit_factor']:.2f} |
"""
        return md


# Global singleton instance
short_covering_backtester = ShortCoveringBacktester()
