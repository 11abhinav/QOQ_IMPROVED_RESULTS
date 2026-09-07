import unittest
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from lock_utils import ProcessLock
from multitf.scanner import _scan_lock, _scan_lock_5m
from multitf.state import MtfStateRecord, update_state_in_db, MtfSubstate
from multitf.data import strip_closed_candles
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


class TestMultiTfConcurrency(unittest.TestCase):
    """Verifies that MULTI_TF and MULTI_TF_5M have completely decoupled locks."""

    def tearDown(self):
        # Ensure locks are always released after each test
        if _scan_lock.locked():
            try:
                _scan_lock.release()
            except Exception:
                pass
        if _scan_lock_5m.locked():
            try:
                _scan_lock_5m.release()
            except Exception:
                pass

    def test_scenario_a_multitf_running_allows_multitf_5m(self):
        """Test A: When MULTI_TF holds multi_tf_scanner lock, MULTI_TF_5M can acquire multi_tf_5m_monitor."""
        # Acquire MULTI_TF lock
        acquired_15m = _scan_lock.acquire(blocking=False)
        self.assertTrue(acquired_15m, "Failed to acquire MULTI_TF lock")
        self.assertTrue(_scan_lock.locked())

        # MULTI_TF_5M must be able to acquire its own lock independently
        acquired_5m = _scan_lock_5m.acquire(blocking=False)
        self.assertTrue(acquired_5m, "MULTI_TF_5M was blocked by MULTI_TF lock! Concurrency decoupling failed.")
        self.assertTrue(_scan_lock_5m.locked())

        # Cleanup
        _scan_lock_5m.release()
        _scan_lock.release()

    def test_scenario_b_multitf_5m_running_allows_multitf(self):
        """Test B: When MULTI_TF_5M holds multi_tf_5m_monitor lock, MULTI_TF can acquire multi_tf_scanner."""
        # Acquire MULTI_TF_5M lock
        acquired_5m = _scan_lock_5m.acquire(blocking=False)
        self.assertTrue(acquired_5m, "Failed to acquire MULTI_TF_5M lock")
        self.assertTrue(_scan_lock_5m.locked())

        # MULTI_TF must be able to acquire its own lock independently
        acquired_15m = _scan_lock.acquire(blocking=False)
        self.assertTrue(acquired_15m, "MULTI_TF was blocked by MULTI_TF_5M lock! Concurrency decoupling failed.")
        self.assertTrue(_scan_lock.locked())

        # Cleanup
        _scan_lock.release()
        _scan_lock_5m.release()

    def test_scenario_c_two_multitf_processes_reject_duplicate(self):
        """Test C: Two simultaneous MULTI_TF instances cannot run concurrently."""
        acquired_1 = _scan_lock.acquire(blocking=False)
        self.assertTrue(acquired_1)

        acquired_2 = _scan_lock.acquire(blocking=False)
        self.assertFalse(acquired_2, "Duplicate MULTI_TF instance was erroneously allowed to acquire lock!")

        _scan_lock.release()

    def test_scenario_d_two_multitf_5m_processes_reject_duplicate(self):
        """Test D: Two simultaneous MULTI_TF_5M instances cannot run concurrently."""
        acquired_1 = _scan_lock_5m.acquire(blocking=False)
        self.assertTrue(acquired_1)

        acquired_2 = _scan_lock_5m.acquire(blocking=False)
        self.assertFalse(acquired_2, "Duplicate MULTI_TF_5M instance was erroneously allowed to acquire lock!")

        _scan_lock_5m.release()

    def test_scenario_e_optimistic_concurrency_control_prevents_race(self):
        """Test E: Optimistic Concurrency Control (CAS) prevents stale state updates."""
        record_v1 = MtfStateRecord(
            symbol="TESTSYM",
            box_id="BOX_001",
            state="WATCH",
            mtf_substate=MtfSubstate.ARMED_PRE_BREAKOUT,
            version=1
        )

        with patch("app.multitf.state.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cur

            # Simulate successful update when version matches (rowcount == 1)
            mock_cur.rowcount = 1
            success = update_state_in_db(record_v1, {"box_high": 105.0})
            self.assertTrue(success)
            self.assertEqual(record_v1.version, 2)

            # Simulate failed CAS when record was modified concurrently by another thread (rowcount == 0)
            mock_cur.rowcount = 0
            stale_record = MtfStateRecord(symbol="TESTSYM", box_id="BOX_001", version=1)
            success_stale = update_state_in_db(stale_record, {"box_high": 110.0})
            self.assertFalse(success_stale, "Stale CAS update should have failed!")

    def test_scenario_f_weekend_candles_banned_in_strip_closed_candles(self):
        """Test F: Weekend (Saturday/Sunday) candles are strictly stripped before screening."""
        dates = [
            pd.Timestamp("2026-09-04 15:15:00", tz=IST), # Friday valid
            pd.Timestamp("2026-09-05 10:00:00", tz=IST), # Saturday invalid
            pd.Timestamp("2026-09-06 10:00:00", tz=IST), # Sunday invalid
            pd.Timestamp("2026-09-07 10:00:00", tz=IST), # Monday valid
        ]
        df = pd.DataFrame({
            "open": [100, 101, 102, 103],
            "high": [105, 106, 107, 108],
            "low": [99, 100, 101, 102],
            "close": [104, 105, 106, 107],
            "volume": [1000, 1000, 1000, 1000],
        }, index=dates)

        now_monday = datetime(2026, 9, 7, 10, 30, tzinfo=IST)
        stripped = strip_closed_candles(df, interval_minutes=15, ist_now=now_monday)

        # Verify Saturday (2026-09-05) and Sunday (2026-09-06) are not in stripped index
        for idx in stripped.index:
            self.assertNotIn(idx.weekday(), [5, 6], f"Weekend candle on day {idx.weekday()} was not stripped!")


if __name__ == "__main__":
    unittest.main()
