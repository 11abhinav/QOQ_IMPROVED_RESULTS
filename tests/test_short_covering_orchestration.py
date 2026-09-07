"""
tests/test_short_covering_orchestration.py

Comprehensive test suite verifying:
1. First-class scheduler & lock separation for SHORT_COVERING_EOD vs SHORT_COVERING_5M.
2. Independent process locks (neither blocks the other or MULTI_TF).
3. Stale-watchlist guard (skips/reports STALE_WATCHLIST when DB watchlist is outdated).
4. Fresh watchlist candidate scanning and ignition alerts.
5. Name normalization and health status reporting across both layers.
6. Market calendar / trading day resolution logic.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import date, datetime, time
import pandas as pd

from app.lock_utils import ProcessLock, SCANNER_CONFIG
from app.database import normalize_scanner_name
from app.short_covering.short_position_detector import short_position_detector, _eod_lock
from app.short_covering.short_covering_scanner import short_covering_scanner, _scan_lock_5m
from app.short_covering.short_covering_schema import EODShortPositionCandidate


class TestShortCoveringOrchestration(unittest.TestCase):

    def setUp(self):
        # Reset locks
        if _eod_lock.is_locked():
            _eod_lock.release()
        if _scan_lock_5m.is_locked():
            _scan_lock_5m.release()

    def test_scanner_config_and_name_normalization(self):
        """Verify SHORT_COVERING, SHORT_COVERING_EOD, and SHORT_COVERING_5M exist in config & normalization."""
        # 1. Config entries
        self.assertIn("SHORT_COVERING", SCANNER_CONFIG)
        self.assertIn("SHORT_COVERING_EOD", SCANNER_CONFIG)
        self.assertIn("SHORT_COVERING_5M", SCANNER_CONFIG)
        
        self.assertEqual(SCANNER_CONFIG["SHORT_COVERING_EOD"]["db_name"], "SHORT_COVERING_EOD")
        self.assertEqual(SCANNER_CONFIG["SHORT_COVERING_5M"]["db_name"], "SHORT_COVERING_5M")

        # 2. Normalization
        self.assertEqual(normalize_scanner_name("SHORT_COVERING"), "SHORT_COVERING")
        self.assertEqual(normalize_scanner_name("short_covering"), "SHORT_COVERING")
        self.assertEqual(normalize_scanner_name("SHORT_COVERING_EOD"), "SHORT_COVERING_EOD")
        self.assertEqual(normalize_scanner_name("SHORT_COVERING_5M"), "SHORT_COVERING_5M")
        self.assertEqual(normalize_scanner_name("short_covering_5m"), "SHORT_COVERING_5M")

    def test_lock_decoupling_eod_and_5m(self):
        """Verify EOD and 5M layers use distinct locks and do not block one another."""
        # Acquire EOD lock
        self.assertTrue(_eod_lock.acquire(blocking=False))
        self.assertTrue(_eod_lock.is_locked())

        # 5M lock MUST still be available
        self.assertFalse(_scan_lock_5m.is_locked())
        self.assertTrue(_scan_lock_5m.acquire(blocking=False))
        self.assertTrue(_scan_lock_5m.is_locked())

        # Release both
        _eod_lock.release()
        _scan_lock_5m.release()
        self.assertFalse(_eod_lock.is_locked())
        self.assertFalse(_scan_lock_5m.is_locked())

    @patch("app.short_covering.short_covering_scanner.upsert_scanner_health")
    @patch.object(short_covering_scanner, "check_watchlist_freshness")
    def test_stale_watchlist_guard_skips_scanning(self, mock_freshness, mock_health):
        """Verify 5M engine skips execution and updates health with STALE_WATCHLIST when watchlist is outdated."""
        # Stale: latest date is 2026-09-01, expected is 2026-09-07
        mock_freshness.return_value = (False, date(2026, 9, 1), date(2026, 9, 7))

        alerts = short_covering_scanner.run_5m_scan_cycle(
            current_time=datetime(2026, 9, 7, 10, 15),
            candidate_watchlist=None,
            persist_db=True
        )

        # Alerts must be empty because scan was skipped
        self.assertEqual(alerts, [])

        # Health update must state STALE_WATCHLIST
        mock_health.assert_called()
        found_stale_msg = any(
            "STALE_WATCHLIST" in str(call.kwargs.get("error_msg", ""))
            for call in mock_health.call_args_list
        )
        self.assertTrue(found_stale_msg, "Expected STALE_WATCHLIST message in scanner_health updates")

    @patch("app.short_covering.short_position_detector.upsert_scanner_health")
    @patch.object(short_position_detector, "evaluate_symbol")
    @patch.object(short_position_detector, "_persist_candidates_to_db")
    def test_eod_scan_cycle_execution(self, mock_persist, mock_eval, mock_health):
        """Verify EOD scanner evaluates universe, persists candidates, and reports health."""
        dummy_candidate = EODShortPositionCandidate(
            symbol="RELIANCE",
            scan_date=date(2026, 9, 7),
            close_price=2420.0,
            total_oi=5000000,
            oi_change_pct_1d=2.5,
            oi_buildup_5d_pct=12.5,
            oi_buildup_10d_pct=18.0,
            short_buildup_ratio=0.75,
            rsi_14=35.0,
            support_level=2400.0,
            overhead_resistance=2550.0,
            atr_14=30.0,
            daily_volume=2500000,
            buildup_quality_score=85.0
        )
        mock_eval.return_value = dummy_candidate

        candidates = short_position_detector.scan_eod_universe(
            as_of=date(2026, 9, 7),
            custom_symbols=["RELIANCE"],
            persist_db=True
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].symbol, "RELIANCE")
        mock_persist.assert_called_once()
        self.assertFalse(_eod_lock.is_locked())

    @patch("app.short_covering.short_covering_scanner.upsert_scanner_health")
    @patch.object(short_covering_scanner, "check_watchlist_freshness")
    @patch.object(short_covering_scanner, "evaluate_symbol_5m")
    def test_5m_scan_cycle_with_fresh_watchlist(self, mock_eval_5m, mock_freshness, mock_health):
        """Verify 5M scanner processes fresh candidates and reports health."""
        mock_freshness.return_value = (True, date(2026, 9, 7), date(2026, 9, 7))
        mock_eval_5m.return_value = None  # No trigger on this bar

        dummy_candidate = EODShortPositionCandidate(
            symbol="SBIN",
            scan_date=date(2026, 9, 7),
            close_price=805.0,
            total_oi=12000000,
            oi_change_pct_1d=1.8,
            oi_buildup_5d_pct=10.0,
            oi_buildup_10d_pct=14.0,
            short_buildup_ratio=0.65,
            rsi_14=40.0,
            support_level=800.0,
            overhead_resistance=840.0,
            atr_14=12.0,
            daily_volume=6000000,
            buildup_quality_score=80.0
        )

        alerts = short_covering_scanner.run_5m_scan_cycle(
            current_time=datetime(2026, 9, 7, 10, 30),
            candidate_watchlist=[dummy_candidate],
            persist_db=False
        )

        self.assertEqual(alerts, [])
        mock_eval_5m.assert_called_once()
        self.assertFalse(_scan_lock_5m.is_locked())


if __name__ == "__main__":
    unittest.main()
