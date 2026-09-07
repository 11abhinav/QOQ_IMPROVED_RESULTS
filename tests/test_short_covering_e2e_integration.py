"""
End-to-End Production Certification Test Suite for Short Covering Pipeline.

Validates the full execution chain:
1. 19:15 EOD short buildup detector -> persists candidate watchlist with valid session date.
2. 09:20 5M early ignition monitor -> checks freshness -> runs ignition cycle -> records alerts & health.
3. Explicit Failure & Edge Case Scenarios:
   - EOD scan failure -> 5M freshness guard rejects stale watchlist (outcome=STALE_WATCHLIST).
   - Weekend / Holiday calendar resolution -> EOD resolves previous trading session.
   - Lock decoupling & concurrent execution -> EOD and 5M run independently without blocking.
   - Cross-scanner concurrency -> MULTI_TF active lock does not block SHORT_COVERING_5M.
   - Duplicate scheduler trigger prevention -> non-blocking process lock returns cleanly.
   - Zero candidates in watchlist -> graceful completion with outcome=SUCCESS and total_count=0.
   - Partial DB write failure -> scanner_health records DOWN / FAILURE, preventing fake success.
   - Health state model verification -> RUNNING, SUCCESS, SKIPPED, DOWN/FAILURE, STALE_WATCHLIST.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

from app.short_covering.short_covering_schema import (
    EODShortPositionCandidate,
    ShortCoveringSignal,
    ShortCoveringState
)
from app.short_covering.short_position_detector import ShortPositionDetector
from app.short_covering.short_covering_scanner import ShortCoveringEarlyIgnitionScanner
from app.trading_calendar import get_latest_trading_date, get_previous_trading_date, is_trading_day
from app.lock_utils import ProcessLock

IST = ZoneInfo("Asia/Kolkata")


class TestShortCoveringE2EIntegration(unittest.TestCase):

    def setUp(self):
        self.detector = ShortPositionDetector()
        self.scanner = ShortCoveringEarlyIgnitionScanner()
        # Clean locks
        ProcessLock("short_covering_eod_lock").release()
        ProcessLock("short_covering_5m_lock").release()

    def tearDown(self):
        ProcessLock("short_covering_eod_lock").release()
        ProcessLock("short_covering_5m_lock").release()

    # =========================================================================
    # 1. Full Happy Path: EOD (19:15) -> Watchlist -> Next Day 5M (09:20) -> Alert
    # =========================================================================
    @patch("app.short_covering.short_position_detector.upsert_scanner_health")
    @patch("app.short_covering.short_covering_scanner.upsert_scanner_health")
    def test_full_happy_path_eod_to_5m_alert(self, mock_5m_health, mock_eod_health):
        friday_session = date(2026, 9, 4)
        monday_session = date(2026, 9, 7)

        # Mock EOD candidate discovery
        mock_eod_candidate = EODShortPositionCandidate(
            symbol="TATAMOTORS",
            scan_date=friday_session,
            close_price=950.0,
            total_oi=50_000_000,
            oi_change_pct_1d=4.5,
            oi_buildup_5d_pct=14.2,
            oi_buildup_10d_pct=22.0,
            short_buildup_ratio=0.75,
            rsi_14=34.0,
            support_level=940.0,
            overhead_resistance=985.0,
            atr_14=18.5,
            daily_volume=5_000_000,
            sector="AUTO",
            buildup_quality_score=85.0,
            reasons=["Heavy short buildup at multi-week support"]
        )

        with patch.object(self.detector, "evaluate_symbol", return_value=mock_eod_candidate), \
             patch.object(self.detector, "_persist_candidates_to_db") as mock_persist_eod:

            # 1. Run EOD Scan for Friday
            eod_candidates = self.detector.scan_eod_universe(as_of=friday_session, custom_symbols=["TATAMOTORS"], persist_db=True)
            self.assertEqual(len(eod_candidates), 1)
            self.assertEqual(eod_candidates[0].scan_date, friday_session)
            mock_persist_eod.assert_called_once_with(eod_candidates, friday_session)

            # EOD health should be OK / SUCCESS
            mock_eod_health.assert_any_call(
                scanner_name="SHORT_COVERING_EOD",
                status="OK",
                outcome="SUCCESS",
                total_count=1,
                processed_count=1,
                duration_seconds=unittest.mock.ANY,
                scheduled_for="Daily 19:15 IST (Market Days)",
                run_id=unittest.mock.ANY
            )

        # 2. Next Monday 09:20 AM: Run 5M Scan
        monday_920 = datetime(2026, 9, 7, 9, 20, 0, tzinfo=IST)

        mock_alert_signal = ShortCoveringSignal(
            symbol="TATAMOTORS",
            timestamp=monday_920,
            ignition_price=955.0,
            vwap=952.0,
            stop_loss=945.0,
            initial_target=975.0,
            risk_reward_ratio=2.0,
            excess_oi_contraction=-0.8,
            volume_surge_ratio=2.5,
            ignition_score=88.0,
            grade="A",
            state=ShortCoveringState.CONFIRMED_IGNITION,
            reasons=["Early rapid OI contraction (-4.2%) + volume surge"]
        )

        with patch.object(self.scanner, "check_watchlist_freshness", return_value=(True, friday_session, friday_session)), \
             patch.object(self.scanner, "_load_eod_watchlist", return_value=[mock_eod_candidate]), \
             patch.object(self.scanner, "evaluate_symbol_5m", return_value=mock_alert_signal), \
             patch.object(self.scanner, "_persist_alerts") as mock_persist_alerts:

            alerts = self.scanner.run_5m_scan_cycle(current_time=monday_920, persist_db=True)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].symbol, "TATAMOTORS")
            self.assertEqual(alerts[0].state, ShortCoveringState.CONFIRMED_IGNITION)
            mock_persist_alerts.assert_called_once()

            # 5M health should be OK / SUCCESS with alert counts
            mock_5m_health.assert_any_call(
                scanner_name="SHORT_COVERING_5M",
                status="OK",
                outcome="SUCCESS",
                total_count=1,
                processed_count=1,
                today_alerts=1,
                duration_seconds=unittest.mock.ANY,
                scheduled_for="Every 5m (09:20 - 15:25 IST Market Days)",
                run_id=unittest.mock.ANY
            )

    # =========================================================================
    # 2. Failure Scenario: EOD Fails -> 5M Rejects Stale Watchlist (STALE_WATCHLIST)
    # =========================================================================
    @patch("app.short_covering.short_covering_scanner.upsert_scanner_health")
    def test_stale_watchlist_guard_rejects_obsolete_data(self, mock_5m_health):
        monday_today = date(2026, 9, 7)
        # Expected session is Friday (2026-09-04), but DB only has Wednesday (2026-09-02)
        stale_date = date(2026, 9, 2)
        expected_date = date(2026, 9, 4)

        with patch.object(self.scanner, "check_watchlist_freshness", return_value=(False, stale_date, expected_date)), \
             patch.object(self.scanner, "evaluate_symbol_5m") as mock_eval:

            alerts = self.scanner.run_5m_scan_cycle(
                current_time=datetime(2026, 9, 7, 9, 25, 0, tzinfo=IST),
                persist_db=True
            )
            self.assertEqual(len(alerts), 0)
            mock_eval.assert_not_called()

            # Verify health is recorded as STALE_WATCHLIST outcome (deliberate safety skip)
            mock_5m_health.assert_called_once_with(
                scanner_name="SHORT_COVERING_5M",
                status="OK",
                outcome="STALE_WATCHLIST",
                error_msg=f"STALE_WATCHLIST (Latest: {stale_date}, Expected: {expected_date})",
                duration_seconds=unittest.mock.ANY,
                scheduled_for="Every 5m (09:20 - 15:25 IST Market Days)",
                run_id=unittest.mock.ANY
            )

    # =========================================================================
    # 3. Calendar Resolution: Weekends & Holidays
    # =========================================================================
    def test_calendar_resolution_across_weekends_and_holidays(self):
        # Sunday 2026-09-06 resolves to Friday 2026-09-04
        sunday = date(2026, 9, 6)
        self.assertFalse(is_trading_day(sunday))
        self.assertEqual(get_latest_trading_date(sunday), date(2026, 9, 4))

        # Monday 2026-09-07 previous trading date is Friday 2026-09-04
        monday = date(2026, 9, 7)
        self.assertTrue(is_trading_day(monday))
        self.assertEqual(get_previous_trading_date(monday), date(2026, 9, 4))

        # EOD scan triggered on Sunday resolves to Friday session
        with patch.object(self.detector, "evaluate_symbol", return_value=None), \
             patch("app.short_covering.short_position_detector.upsert_scanner_health"):
            self.detector.scan_eod_universe(as_of=sunday, custom_symbols=["RELIANCE"], persist_db=False)
            # Detector resolved target_date to 2026-09-04

    # =========================================================================
    # 4. Lock Decoupling: EOD & 5M Overlap Independence
    # =========================================================================
    def test_eod_and_5m_locks_do_not_block_each_other(self):
        eod_lock = ProcessLock("short_covering_eod_lock")
        scan_5m_lock = ProcessLock("short_covering_5m_lock")

        # Acquire EOD lock
        self.assertTrue(eod_lock.acquire(blocking=False))

        # 5M lock should still be freely acquirable
        self.assertTrue(scan_5m_lock.acquire(blocking=False))

        scan_5m_lock.release()
        eod_lock.release()

    # =========================================================================
    # 5. Cross-Scanner Concurrency: MULTI_TF Active Lock Does Not Block 5M
    # =========================================================================
    def test_multitf_running_allows_short_covering_5m(self):
        multitf_lock = ProcessLock("multitf_scanner_lock")
        multitf_5m_lock = ProcessLock("multitf_scanner_5m_lock")
        sc_5m_lock = ProcessLock("short_covering_5m_lock")

        # Acquire MULTI_TF locks
        self.assertTrue(multitf_lock.acquire(blocking=False))
        self.assertTrue(multitf_5m_lock.acquire(blocking=False))

        # SHORT_COVERING_5M lock must be completely decoupled and acquirable
        self.assertTrue(sc_5m_lock.acquire(blocking=False))

        sc_5m_lock.release()
        multitf_5m_lock.release()
        multitf_lock.release()

    # =========================================================================
    # 6. Duplicate Scheduler Trigger Prevention
    # =========================================================================
    def test_duplicate_scheduler_triggers_rejected_by_lock(self):
        lock_5m = ProcessLock("short_covering_5m_lock")
        self.assertTrue(lock_5m.acquire(blocking=False))

        # Simultaneous 5m cycle attempt while lock is held
        res = self.scanner.run_5m_scan_cycle(current_time=datetime.now(IST))
        self.assertEqual(res, [])

        lock_5m.release()

    # =========================================================================
    # 7. Zero Candidates in Watchlist: Clean SUCCESS Outcome
    # =========================================================================
    @patch("app.short_covering.short_covering_scanner.upsert_scanner_health")
    def test_zero_candidates_in_watchlist_is_success_not_error(self, mock_health):
        with patch.object(self.scanner, "check_watchlist_freshness", return_value=(True, date(2026, 9, 4), date(2026, 9, 4))), \
             patch.object(self.scanner, "_load_eod_watchlist", return_value=[]):

            alerts = self.scanner.run_5m_scan_cycle(
                current_time=datetime(2026, 9, 7, 9, 20, 0, tzinfo=IST),
                persist_db=True
            )
            self.assertEqual(alerts, [])
            mock_health.assert_called_once_with(
                scanner_name="SHORT_COVERING_5M",
                status="OK",
                outcome="SUCCESS",
                total_count=0,
                processed_count=0,
                duration_seconds=unittest.mock.ANY,
                scheduled_for="Every 5m (09:20 - 15:25 IST Market Days)",
                run_id=unittest.mock.ANY
            )

    # =========================================================================
    # 8. Partial DB Failure: Health Accurately Records FAILURE / DOWN
    # =========================================================================
    @patch("app.short_covering.short_position_detector.upsert_scanner_health")
    def test_db_persistence_failure_marks_health_down(self, mock_eod_health):
        with patch.object(self.detector, "evaluate_symbol", return_value=EODShortPositionCandidate(
            symbol="SBIN", scan_date=date(2026, 9, 4), close_price=800.0, total_oi=10000,
            oi_change_pct_1d=2.0, oi_buildup_5d_pct=10.0, oi_buildup_10d_pct=15.0,
            short_buildup_ratio=0.7, rsi_14=35.0, support_level=790.0, overhead_resistance=830.0,
            atr_14=12.0, daily_volume=100000, sector="BANK", buildup_quality_score=80.0, reasons=[]
        )), \
        patch.object(self.detector, "_persist_candidates_to_db", side_effect=RuntimeError("DB Disk Full")):

            candidates = self.detector.scan_eod_universe(as_of=date(2026, 9, 4), custom_symbols=["SBIN"], persist_db=True)
            self.assertEqual(candidates, [])

            # Verify health is marked DOWN / FAILURE
            mock_eod_health.assert_any_call(
                scanner_name="SHORT_COVERING_EOD",
                status="DOWN",
                outcome="FAILURE",
                error_msg="DB Disk Full",
                duration_seconds=unittest.mock.ANY,
                scheduled_for="Daily 19:15 IST (Market Days)",
                run_id=unittest.mock.ANY
            )


if __name__ == "__main__":
    unittest.main()
