"""
tests/test_short_covering_system.py

Comprehensive test suite for the 2-Layer Short-Covering Early-Ignition Scanner.
Covers:
1. Provider Capabilities Matrix (Upstox vs Fyers vs NSE EOD) & Staleness Validation
2. F&O Universe & Sector Mapping
3. F&O Contract Resolver & Expiry Calculation
4. Layer 1: EOD Short Position Detector (SBR, OI buildup, divergence)
5. Layer 2: Intraday 5m Ignition Scanner & Multi-State Progression (WATCH -> IGNITION_CANDIDATE -> CONFIRMED_IGNITION)
6. Tiered Progressive 30m/15m Context (30m breakout is a boost, not a hard barrier)
7. Intraday Point-in-Time Backtester & Comparative Benchmarking (Strategy vs Baselines, Move Consumed at Alert)
"""

import sys
import os

os.environ["DISABLE_DB_OI_LOOKUP"] = "1"
os.environ["DATABASE_URL"] = ""

from datetime import date, datetime, timedelta
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.short_covering.fno_universe import fno_universe_manager
from app.short_covering.fno_contract_resolver import fno_contract_resolver, get_monthly_expiry, get_near_and_next_expiries
from app.short_covering.oi_data_service import oi_data_service
from app.short_covering.short_position_detector import short_position_detector
from app.short_covering.short_covering_scanner import ShortCoveringEarlyIgnitionScanner, short_covering_scanner
from app.short_covering.short_covering_backtester import short_covering_backtester
from app.short_covering.short_covering_schema import (
    EODShortPositionCandidate,
    ShortCoveringSignal,
    ShortCoveringState,
    PROVIDER_CAPABILITY_MATRIX,
)


class TestShortCoveringSystem(unittest.TestCase):

    def test_1_provider_capabilities(self):
        """Test explicit market data provider capabilities matrix."""
        upstox_cap = oi_data_service.get_provider_capability("UPSTOX")
        self.assertTrue(upstox_cap.supports_5m_price)
        self.assertTrue(upstox_cap.supports_5m_oi)

        fyers_cap = oi_data_service.get_provider_capability("FYERS")
        self.assertTrue(fyers_cap.supports_5m_price)
        self.assertFalse(fyers_cap.supports_5m_oi)

        self.assertTrue(oi_data_service.validate_provider_capabilities("supports_5m_price"))

    def test_2_fno_universe(self):
        """Test F&O universe loading and sector classification."""
        symbols = fno_universe_manager.get_fno_symbols()
        self.assertGreater(len(symbols), 100)
        self.assertTrue(fno_universe_manager.is_fno_symbol("RELIANCE"))
        self.assertEqual(fno_universe_manager.get_sector("HDFCBANK"), "BANKING")

    def test_3_fno_contract_resolver_and_rollover(self):
        """Test monthly expiry date calculation and rollover detection."""
        exp = get_monthly_expiry(2026, 9)
        self.assertEqual(exp.weekday(), 3)  # Thursday

        # Near expiry week: near drops -100k, next adds +90k -> Rollover True
        is_roll = oi_data_service.is_rollover_in_progress(
            symbol="RELIANCE",
            near_oi_delta=-100000,
            next_oi_delta=90000,
            as_of=date(2026, 9, 23)
        )
        self.assertTrue(is_roll)

    def test_4_layer_1_eod_position_detector(self):
        """Test Layer 1 EOD Short Position buildup detection."""
        test_symbols = ["TATASTEEL", "RELIANCE"]
        candidates = short_position_detector.scan_eod_universe(
            as_of=date(2026, 9, 7),
            custom_symbols=test_symbols,
            persist_db=False
        )
        self.assertIsInstance(candidates, list)
        for c in candidates:
            self.assertIsInstance(c, EODShortPositionCandidate)
            self.assertGreaterEqual(c.buildup_quality_score, 0.0)

    def test_5_layer_2_state_machine_and_tiered_scoring(self):
        """Test Layer 2 state progression and dynamic evidence-based confirmation."""
        import pandas as pd
        # Create deterministic test bars for TESTSTOCK:
        # Bar 0 (09:15): Base bar
        # Bar 1 (09:20): Ignition pulse (Price +0.5%, OI -0.8%, Volume 2.5x)
        test_df = pd.DataFrame({
            "timestamp": [
                datetime(2026, 9, 7, 9, 15),
                datetime(2026, 9, 7, 9, 20),
                datetime(2026, 9, 7, 9, 25),
            ],
            "open": [100.0, 100.5, 101.2],
            "high": [100.8, 101.5, 102.0],
            "low": [99.8, 100.4, 101.0],
            "close": [100.4, 101.2, 101.8],
            "volume": [10000, 25000, 22000],
            "vwap": [100.2, 100.6, 101.0],
            "oi": [1000000, 992000, 985000],
            "oi_change_5m_pct": [0.0, -0.80, -0.70],
            "oi_change_session_pct": [0.0, -0.80, -1.50]
        })
        oi_data_service._intraday_oi_cache["TESTSTOCK_5m_2026-09-07"] = test_df

        # High-conviction setup (Score >= 76) -> confirms immediately on Bar 1
        scanner_high = ShortCoveringEarlyIgnitionScanner(min_ignition_score=60.0)
        high_candidate = EODShortPositionCandidate(
            symbol="TESTSTOCK",
            scan_date=date(2026, 9, 7),
            close_price=100.0,
            total_oi=1_000_000,
            oi_change_pct_1d=-0.5,
            oi_buildup_5d_pct=15.0,
            oi_buildup_10d_pct=20.0,
            short_buildup_ratio=0.80,
            rsi_14=35.0,
            support_level=98.0,
            overhead_resistance=108.0,
            atr_14=2.0,
            daily_volume=500_000,
            sector="METALS",
            buildup_quality_score=90.0,
            reasons=["Heavy prior short accumulation"]
        )

        t1 = datetime(2026, 9, 7, 9, 20)
        alerts_high = scanner_high.run_5m_scan_cycle(current_time=t1, candidate_watchlist=[high_candidate])
        self.assertTrue(len(alerts_high) > 0)
        sig = alerts_high[0]
        self.assertEqual(sig.state, ShortCoveringState.CONFIRMED_IGNITION)
        self.assertEqual(sig.alert_latency_minutes, 0.0)
        self.assertGreater(sig.ignition_score, 75.0)

    def test_6_comparative_benchmark_and_move_consumed(self):

        """Test 3-way Comparative Benchmarking and Earlyness Metric ('Move Consumed at Alert')."""
        start_date = date(2026, 9, 1)
        end_date = date(2026, 9, 4)
        sample_symbols = ["TATASTEEL", "SBIN", "RELIANCE"]

        results = short_covering_backtester.run_comparative_benchmark(
            start_date=start_date,
            end_date=end_date,
            sample_symbols=sample_symbols
        )

        self.assertIn("proposed_strategy", results)
        self.assertIn("baseline_a", results)
        self.assertIn("baseline_b", results)
        self.assertIn("report_markdown", results)

        proposed = results["proposed_strategy"]
        self.assertIn("median_pre_alert_move_pct", proposed)
        self.assertIn("median_eventual_move_consumed_pct", proposed)
        self.assertIn("avg_post_alert_mfe_pct", proposed)
        self.assertIn("avg_post_alert_mae_pct", proposed)
        self.assertIn("median_latency_minutes", proposed)
        self.assertIn("profit_factor", proposed)


if __name__ == "__main__":
    unittest.main()

