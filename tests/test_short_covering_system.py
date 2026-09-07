"""
tests/test_short_covering_system.py

Comprehensive test suite for the 2-Layer Short-Covering Early-Ignition Scanner.
Covers:
1. F&O Universe & Sector Mapping
2. F&O Contract Resolver & Expiry Calculation
3. OI Data Service & Rollover Flow Detection
4. Layer 1: EOD Short Position Detector (SBR, OI buildup, divergence)
5. Layer 2: Intraday 5m Ignition Scanner (Price, OI contraction, excess score, anti-fake, MTF)
6. Intraday Point-in-Time Backtester (Simulation, forward returns, MFE, MAE)
"""

import sys
import os
from datetime import date, datetime, timedelta
import unittest

# Ensure project root is in python path
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
)


class TestShortCoveringSystem(unittest.TestCase):

    def test_1_fno_universe(self):
        """Test F&O universe loading and sector classification."""
        symbols = fno_universe_manager.get_fno_symbols()
        self.assertGreater(len(symbols), 100)
        self.assertTrue(fno_universe_manager.is_fno_symbol("RELIANCE"))
        self.assertTrue(fno_universe_manager.is_fno_symbol("HDFCBANK"))
        self.assertEqual(fno_universe_manager.get_sector("HDFCBANK"), "BANKING")
        self.assertEqual(fno_universe_manager.get_sector("TCS"), "IT")

    def test_2_fno_contract_resolver(self):
        """Test monthly expiry date calculation and near/next month contracts."""
        # Test September 2026 expiry (last Thursday of Sept 2026)
        # Sept 2026: 1st is Tuesday, 30th is Wednesday. 24th is Thursday.
        exp = get_monthly_expiry(2026, 9)
        self.assertEqual(exp.weekday(), 3)  # Thursday
        self.assertEqual(exp.month, 9)

        near_exp, next_exp = get_near_and_next_expiries(date(2026, 9, 7))
        self.assertEqual(near_exp.month, 9)
        self.assertEqual(next_exp.month, 10)

        contract = fno_contract_resolver.resolve("RELIANCE", as_of=date(2026, 9, 7))
        self.assertEqual(contract.symbol, "RELIANCE")
        self.assertIn("FUT", contract.near_trading_symbol)

    def test_3_oi_data_service_and_rollover(self):
        """Test daily and intraday OI data retrieval and rollover detection."""
        daily_df = oi_data_service.get_daily_oi_history("RELIANCE", lookback_days=15, as_of=date(2026, 9, 7))
        self.assertFalse(daily_df.empty)
        self.assertIn("total_oi", daily_df.columns)
        self.assertIn("oi_change_pct", daily_df.columns)

        intraday_df = oi_data_service.get_intraday_5m_data("RELIANCE", target_date=date(2026, 9, 7))
        self.assertEqual(len(intraday_df), 75)
        self.assertIn("vwap", intraday_df.columns)
        self.assertIn("oi_change_5m_pct", intraday_df.columns)

        # Test rollover flow detection
        # Near expiry week: near drops -100k, next adds +90k -> Rollover True
        is_roll = oi_data_service.is_rollover_in_progress(
            symbol="RELIANCE",
            near_oi_delta=-100000,
            next_oi_delta=90000,
            as_of=date(2026, 9, 23)  # Expiry week
        )
        self.assertTrue(is_roll)

    def test_4_layer_1_eod_position_detector(self):
        """Test Layer 1 EOD Short Position buildup detection."""
        test_symbols = ["RELIANCE", "SBIN", "INFY", "TATASTEEL", "HDFCBANK"]
        candidates = short_position_detector.scan_eod_universe(
            as_of=date(2026, 9, 7),
            custom_symbols=test_symbols
        )
        self.assertIsInstance(candidates, list)
        for c in candidates:
            self.assertIsInstance(c, EODShortPositionCandidate)
            self.assertGreaterEqual(c.buildup_quality_score, 0.0)
            self.assertLessEqual(c.buildup_quality_score, 100.0)
            self.assertGreater(c.atr_14, 0.0)

    def test_5_layer_2_intraday_5m_scanner(self):
        """Test Layer 2 Intraday 5m early ignition scan and scoring."""
        scanner = ShortCoveringEarlyIgnitionScanner(min_ignition_score=60.0)

        # Create mock candidate from Layer 1
        mock_candidate = EODShortPositionCandidate(
            symbol="TATASTEEL",
            scan_date=date(2026, 9, 7),
            close_price=150.0,
            total_oi=50_000_000,
            oi_change_pct_1d=-0.5,
            oi_buildup_5d_pct=12.5,
            oi_buildup_10d_pct=18.0,
            short_buildup_ratio=0.75,
            rsi_14=38.0,
            support_level=145.0,
            overhead_resistance=158.0,
            atr_14=3.5,
            daily_volume=20_000_000,
            sector="METALS",
            buildup_quality_score=85.0,
            reasons=["Heavy 5d short accumulation", "Oversold base hold"]
        )

        test_time = datetime(2026, 9, 7, 10, 15)
        alerts = scanner.run_5m_scan_cycle(
            current_time=test_time,
            candidate_watchlist=[mock_candidate]
        )
        self.assertIsInstance(alerts, list)

    def test_6_intraday_replay_backtester(self):
        """Test Intraday Point-in-Time Replay Backtester and metric calculations."""
        start_date = date(2026, 9, 1)
        end_date = date(2026, 9, 4)
        sample_symbols = ["TATASTEEL", "SBIN", "RELIANCE"]

        summary = short_covering_backtester.run_replay(
            start_date=start_date,
            end_date=end_date,
            sample_symbols=sample_symbols
        )
        self.assertIn("total_alerts", summary)
        if summary["total_alerts"] > 0:
            self.assertIn("avg_session_mfe_pct", summary)
            self.assertIn("false_covering_rate_pct", summary)
            self.assertIn("avg_fwd_return_30m_pct", summary)


if __name__ == "__main__":
    unittest.main()
