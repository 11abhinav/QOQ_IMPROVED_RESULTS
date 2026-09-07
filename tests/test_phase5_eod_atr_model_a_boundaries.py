"""
tests/test_phase5_eod_atr_model_a_boundaries.py

Mandatory Boundary Condition & Production Verification Suite for EOD ATR Model A:
- 2.4999% -> pass, penalty 0
- 2.5000% -> pass, penalty 0
- 2.5001% -> pass, penalty 3
- 3.4999% -> pass, penalty 3
- 3.5000% -> pass, penalty 3
- 3.5001% -> pass, penalty 7
- 4.4999% -> pass, penalty 7
- 4.5000% -> pass, penalty 7
- 4.5001% -> reject (> 4.50% ceiling)
- Pre-breakout base ATR denominator verification (ticker.Close[-2])
- Metadata tag calibration_model_version = "EOD_ATR_MODEL_A_V1"
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from eod_scanner import _check_eod_conditions, evaluate_eod_symbol

class TestEODATRModelABoundaries(unittest.TestCase):

    def _create_synthetic_df(self, target_base_atr_pct: float, base_price: float = 1000.0) -> pd.DataFrame:
        """
        Creates a 60-bar synthetic dataframe where the 10-day base ATR
        relative to Close[-2] exactly equals target_base_atr_pct.
        """
        n_bars = 60
        # Target absolute ATR10
        target_atr = (target_base_atr_pct / 100.0) * base_price

        # Build prices with slight oscillation so RSI is natural (~60)
        closes = [base_price + ((-1)**j * 2.0) for j in range(n_bars)]
        closes[-2] = base_price
        closes[-1] = base_price * 1.03 # 3% breakout close
        opens = [base_price] * n_bars
        opens[-1] = base_price * 1.01

        # Highs and lows to deliver exact target_atr
        highs = [base_price + (target_atr / 2.0)] * n_bars
        highs[-1] = closes[-1] * 1.01
        lows = [base_price - (target_atr / 2.0)] * n_bars
        lows[-1] = opens[-1] * 0.99

        volumes = [100000] * n_bars
        volumes[-1] = 250000 # 2.5x volume surge

        dates = pd.date_range("2026-08-01", periods=n_bars, freq="B")
        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
            "SMA20": [base_price * 0.98] * n_bars,
            "SMA50": [base_price * 0.95] * n_bars,
            "SMA200": [base_price * 0.90] * n_bars,
            "EMA20": [base_price * 0.98] * n_bars,
            "RSI": [65.0] * n_bars,
            "ATR": [target_atr] * n_bars,
            "ATR20": [target_atr] * n_bars,
            "PRIOR_20D_HIGH": [base_price * 1.01] * n_bars,
            "HIGH_52W": [base_price * 1.02] * n_bars,
            "VOLUME_SMA20": [100000] * n_bars,
            "BB_WIDTH_PCTILE": [0.35] * n_bars
        }, index=dates)

        return df

    def test_boundary_2_4999_pct(self):
        df = self._create_synthetic_df(2.4999)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 0)
        self.assertAlmostEqual(res["base_atr_pct"], 2.4999, places=3)

    def test_boundary_2_5000_pct(self):
        df = self._create_synthetic_df(2.5000)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 0)
        self.assertAlmostEqual(res["base_atr_pct"], 2.5000, places=3)

    def test_boundary_2_5001_pct(self):
        df = self._create_synthetic_df(2.5001)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 3)
        self.assertAlmostEqual(res["base_atr_pct"], 2.5001, places=3)

    def test_boundary_3_4999_pct(self):
        df = self._create_synthetic_df(3.4999)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 3)
        self.assertAlmostEqual(res["base_atr_pct"], 3.4999, places=3)

    def test_boundary_3_5000_pct(self):
        df = self._create_synthetic_df(3.5000)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 3)
        self.assertAlmostEqual(res["base_atr_pct"], 3.5000, places=3)

    def test_boundary_3_5001_pct(self):
        df = self._create_synthetic_df(3.5001)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 7)
        self.assertAlmostEqual(res["base_atr_pct"], 3.5001, places=3)

    def test_boundary_4_4999_pct(self):
        df = self._create_synthetic_df(4.4999)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 7)
        self.assertAlmostEqual(res["base_atr_pct"], 4.4999, places=3)

    def test_boundary_4_5000_pct(self):
        df = self._create_synthetic_df(4.5000)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertTrue(res["passed"])
        self.assertEqual(res["base_atr_penalty"], 7)
        self.assertAlmostEqual(res["base_atr_pct"], 4.5000, places=3)

    def test_boundary_4_5001_pct(self):
        df = self._create_synthetic_df(4.5001)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertFalse(res["passed"])
        self.assertIn("tightness ceiling", res["reason"])

    def test_metadata_model_version_tagging(self):
        df = self._create_synthetic_df(3.0)
        latest = df.iloc[-1]
        res = _check_eod_conditions(df, latest, "TEST_SYM")
        self.assertEqual(res["base_atr_penalty"], 3)
        self.assertAlmostEqual(res["base_atr_pct"], 3.0, places=2)

        eval_res = evaluate_eod_symbol("TEST_SYM", df)
        self.assertEqual(eval_res["calibration_model_version"], "EOD_ATR_MODEL_A_V1")

if __name__ == "__main__":
    unittest.main()
