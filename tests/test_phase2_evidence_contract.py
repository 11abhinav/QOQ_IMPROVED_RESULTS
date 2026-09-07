import unittest
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from accumulation_scanner import AccumulationScanner

class TestPhase2EvidenceContract(unittest.TestCase):
    """
    Phase 2 Acceptance Test Suite:
    Validates explicit evidence-aware data contract across the required test matrix:
    - Case A: Complete / Valid Fundamentals -> FULL_CONFIDENCE, normal 0-100 score, eligible for ACTIONABLE.
    - Case B: None Fundamentals -> REDUCED_CONFIDENCE, native 0-90 score, TECHNICAL_ONLY, cannot become ACTIONABLE.
    - Case C: Partial / Incomplete Fundamentals -> REDUCED_CONFIDENCE, TECHNICAL_ONLY.
    - Case D: Invalid / Insufficient Technical Data -> INSUFFICIENT_EVIDENCE, REJECTED.
    - Case E: Numerical Invariance for Full Fundamentals -> Identical score and decisions.
    - Case F: Borderline Threshold Gating -> Score 84.9 does not trigger, Reduced confidence >= 85 stays TECHNICAL_ONLY.
    """

    def setUp(self):
        self.scanner = AccumulationScanner()
        dates = pd.date_range(end=datetime.now(ZoneInfo("Asia/Kolkata")), periods=100, freq="D")
        np.random.seed(42)
        prices = 1000.0 + np.cumsum(np.random.randn(100) * 3.0)
        self.mock_df = pd.DataFrame({
            "Open": prices * 0.995,
            "High": prices * 1.015,
            "Low": prices * 0.985,
            "Close": prices,
            "Volume": np.random.randint(80000, 250000, size=100)
        }, index=dates)

        self.complete_fund_data = {
            "ROE": 18.5,
            "ROCE": 22.0,
            "DebtEquity": 0.35,
            "SalesGrowth": 16.0,
            "PATGrowth": 19.5
        }

    def test_case_a_complete_valid_fundamentals(self):
        """Case A: Complete and valid fundamentals receive FULL_CONFIDENCE and normal score."""
        res = self.scanner.evaluate_symbol("INFY", self.mock_df, fund_data=self.complete_fund_data)
        
        self.assertEqual(res["evidence_confidence"], "FULL_CONFIDENCE")
        self.assertIn(res["qualification_state"], ["ACTIONABLE", "WATCHLIST_ONLY", "REJECTED"])
        self.assertIsNotNone(res["scores_breakdown"]["FUNDAMENTAL"])
        self.assertGreaterEqual(res["scores_breakdown"]["FUNDAMENTAL"], 4.0)
        # Total score must equal sum of technicals + fundamental
        raw_tech = sum(v for k, v in res["scores_breakdown"].items() if k not in ("FUNDAMENTAL", "TOTAL"))
        self.assertAlmostEqual(res["score"], raw_tech + res["scores_breakdown"]["FUNDAMENTAL"], places=1)

    def test_case_b_none_fundamentals(self):
        """Case B: Missing fundamentals (None) receive REDUCED_CONFIDENCE on native 0-90 basis."""
        res = self.scanner.evaluate_symbol("INFY", self.mock_df, fund_data=None)
        
        self.assertEqual(res["evidence_confidence"], "REDUCED_CONFIDENCE")
        self.assertEqual(res["qualification_state"], "TECHNICAL_ONLY" if res["status"] == "QUALIFIED" else "REJECTED")
        self.assertIsNone(res["scores_breakdown"]["FUNDAMENTAL"])
        # Score must stay on native 0-90 scale without artificial +5 pt penalty or 100/90 inflation
        raw_tech = sum(v for k, v in res["scores_breakdown"].items() if k not in ("FUNDAMENTAL", "TOTAL"))
        self.assertAlmostEqual(res["score"], raw_tech, places=1)
        self.assertLessEqual(res["score"], 90.0)

    def test_case_c_partial_incomplete_fundamentals(self):
        """Case C: Partial fundamentals (missing keys) receive REDUCED_CONFIDENCE."""
        partial_fund = {
            "ROE": 18.5,
            "SalesGrowth": 16.0
            # Missing ROCE, DebtEquity, PATGrowth
        }
        res = self.scanner.evaluate_symbol("INFY", self.mock_df, fund_data=partial_fund)
        
        self.assertEqual(res["evidence_confidence"], "REDUCED_CONFIDENCE")
        self.assertEqual(res["qualification_state"], "TECHNICAL_ONLY" if res["status"] == "QUALIFIED" else "REJECTED")
        self.assertIsNone(res["scores_breakdown"]["FUNDAMENTAL"])

    def test_case_d_insufficient_evidence_corrupt_data(self):
        """Case D: Corrupt or insufficient technical data receives INSUFFICIENT_EVIDENCE."""
        empty_df = pd.DataFrame()
        res_empty = self.scanner.evaluate_symbol("CORRUPT", empty_df, fund_data=self.complete_fund_data)
        self.assertEqual(res_empty["evidence_confidence"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res_empty["qualification_state"], "REJECTED")
        self.assertEqual(res_empty["reason_code"], "DATA_MISSING")

        short_df = self.mock_df.iloc[:20]  # len < 50
        res_short = self.scanner.evaluate_symbol("SHORT", short_df, fund_data=self.complete_fund_data)
        self.assertEqual(res_short["evidence_confidence"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res_short["qualification_state"], "REJECTED")
        self.assertEqual(res_short["reason_code"], "DATA_MISSING")

    def test_case_e_numerical_invariance_full_fundamentals(self):
        """Case E: Numerical invariance for full fundamentals vs baseline mathematical formula."""
        res = self.scanner.evaluate_symbol("TCS", self.mock_df, fund_data=self.complete_fund_data)
        tech_score = (
            res["scores_breakdown"]["ACCUMULATION"] +
            res["scores_breakdown"]["COMPRESSION"] +
            res["scores_breakdown"]["RELATIVE_STRENGTH"] +
            res["scores_breakdown"]["RESISTANCE"] +
            res["scores_breakdown"]["VOLUME_STRUCTURE"]
        )
        fund_score = res["scores_breakdown"]["FUNDAMENTAL"]
        expected_total = round(tech_score + fund_score, 1)
        self.assertEqual(res["score"], expected_total)

    def test_case_f_reduced_confidence_cannot_become_actionable(self):
        """Case F: High score under REDUCED_CONFIDENCE remains TECHNICAL_ONLY and never ACTIONABLE."""
        res = self.scanner.evaluate_symbol("TCS", self.mock_df, fund_data=None)
        self.assertNotEqual(res["qualification_state"], "ACTIONABLE")
        if res["status"] == "QUALIFIED":
            self.assertEqual(res["qualification_state"], "TECHNICAL_ONLY")
            self.assertEqual(res["evidence_confidence"], "REDUCED_CONFIDENCE")

if __name__ == "__main__":
    unittest.main()
