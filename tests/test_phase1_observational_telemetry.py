import unittest
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from zero_alert_diagnostic import SingleTerminalTracker, StageWaterfallTracker
from accumulation_scanner import AccumulationScanner

class TestPhase1ObservationalTelemetry(unittest.TestCase):
    """
    Validates Phase 1 Acceptance Criteria:
    1. Zero threshold changes
    2. Exact Stage Waterfall reconciliation
    3. Input == Passed + Rejected + Skipped + Error (Delta == 0)
    4. Side-effect free telemetry
    """

    def test_single_terminal_tracker_conservation_all_categories(self):
        symbols = [f"SYM_{i}" for i in range(10)]
        tracker = SingleTerminalTracker(symbols, scanner_name="UNIT_TEST")
        
        # Record 2 PASSED
        tracker.record_terminal("SYM_0", "ALERT_GENERATED", "Alert triggered")
        tracker.record_terminal("SYM_1", "ALERT_GENERATED", "Alert triggered")
        
        # Record 4 REJECTED
        tracker.record_terminal("SYM_2", "SCORE_FAIL", "Score below 55")
        tracker.record_terminal("SYM_3", "FUNDAMENTAL_FAIL", "Piotroski below 7")
        tracker.record_terminal("SYM_4", "NOT_IN_BUY_ZONE", "Price outside buy zone")
        tracker.record_terminal("SYM_5", "ATR_FAIL", "ATR too wide")
        
        # Record 2 SKIPPED
        tracker.record_terminal("SYM_6", "DUPLICATE", "Already alerted within 5 days")
        tracker.record_terminal("SYM_7", "COOLDOWN_ACTIVE", "Symbol in active cooldown")
        
        # Record 2 ERROR
        tracker.record_terminal("SYM_8", "PIPELINE_FAILED", "Data pipeline exception")
        tracker.record_terminal("SYM_9", "INSERT_FAILED", "DB constraint violation")
        
        tracker.record_untracked_remainder("UNTRACKED_DROP")
        summary = tracker.get_summary()

        self.assertEqual(summary["total_universe"], 10)
        self.assertEqual(summary["sum_terminal"], 10)
        self.assertEqual(summary["conservation_delta"], 0)
        self.assertEqual(summary["untracked_drop"], 0)
        self.assertEqual(summary["passed_count"], 2)
        self.assertEqual(summary["rejected_count"], 4)
        self.assertEqual(summary["skipped_count"], 2)
        self.assertEqual(summary["error_count"], 2)
        self.assertTrue(summary["is_conserved"])

    def test_stage_waterfall_dominant_bottleneck(self):
        waterfall = StageWaterfallTracker(["1_UNIVERSE", "2_DATA", "3_QUALITY", "4_BUY_ZONE", "5_ALERTS"])
        waterfall.set_stage_count("1_UNIVERSE", 100)
        waterfall.set_stage_count("2_DATA", 95)
        waterfall.set_stage_count("3_QUALITY", 30)  # Loss: 65 (68.4% attrition)
        waterfall.set_stage_count("4_BUY_ZONE", 5)   # Loss: 25 (83.3% attrition) -> Dominant!
        waterfall.set_stage_count("5_ALERTS", 0)     # Loss: 5 (100% attrition) -> Dominant!

        attrition = waterfall.compute_attrition()
        self.assertEqual(len(attrition), 4)
        
        dominant = waterfall.get_dominant_bottleneck()
        self.assertIsNotNone(dominant)
        self.assertEqual(dominant["stage"], "4_BUY_ZONE")
        self.assertEqual(dominant["attrition_pct"], 100.0)

    def test_accumulation_scanner_telemetry_integration(self):
        symbols = ["TCS", "INFY", "INVALID_STOCK"]
        tracker = SingleTerminalTracker(symbols, scanner_name="ACCUMULATION")
        scanner = AccumulationScanner()

        # Valid mock dataframe
        dates = pd.date_range(end=datetime.now(ZoneInfo("Asia/Kolkata")), periods=80, freq="D")
        np.random.seed(42)
        prices = 1000.0 + np.cumsum(np.random.randn(80) * 5.0)
        df_valid = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(50000, 200000, size=80)
        }, index=dates)

        res_tcs = scanner.evaluate_symbol("TCS", df_valid, fund_data=None, nifty_20d_ret=0.0)
        tracker.record_terminal("TCS", "WATCHLIST_ONLY" if res_tcs.get("status") == "QUALIFIED" else "SCORE_FAIL", "Test run")

        res_infy = scanner.evaluate_symbol("INFY", df_valid, fund_data=None, nifty_20d_ret=0.0)
        tracker.record_terminal("INFY", "WATCHLIST_ONLY" if res_infy.get("status") == "QUALIFIED" else "SCORE_FAIL", "Test run")

        tracker.record_terminal("INVALID_STOCK", "DATA_MISSING", "No dataframe")

        tracker.record_untracked_remainder("UNTRACKED_DROP")
        summary = tracker.get_summary()

        self.assertEqual(summary["total_universe"], 3)
        self.assertEqual(summary["sum_terminal"], 3)
        self.assertEqual(summary["conservation_delta"], 0)
        self.assertTrue(summary["is_conserved"])

if __name__ == "__main__":
    unittest.main()
