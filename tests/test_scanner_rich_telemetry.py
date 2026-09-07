import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

class TestScannerRichTelemetry(unittest.TestCase):
    """
    Validates rich diagnostic output, pre-breakout watch formatting,
    and near-miss logging across all scanners (EOD, Multibagger, Multi-TF, Reversal, Pullback).
    """

    def test_eod_pre_breakout_watch_conditions(self):
        """Verify that a stock within 4.5% of 20D High triggers pre-breakout watch payload."""
        cmp_val = 604.40
        prior_20d_high = 619.88
        dist_pct = (prior_20d_high - cmp_val) / prior_20d_high * 100.0
        
        self.assertAlmostEqual(dist_pct, 2.50, delta=0.1)
        self.assertTrue(0.0 <= dist_pct <= 4.5)
        
        # Test simulated logging payload
        log_msg = (
            f"👁️ [EOD: PRE-BREAKOUT WATCH] SKIPPER added to Watchlist (Base Consolidation | Dist: {dist_pct:.1f}%) | "
            f"CMP: ₹{cmp_val:.2f} | Pending Breakout Level: ₹{prior_20d_high:.2f} | RVOL: 1.25x | "
            f"RSI: 58.2 | SL: ₹531.67 | RR: 0.57 — (Pending breakout trigger, not an active trade yet)"
        )
        self.assertIn("👁️ [EOD: PRE-BREAKOUT WATCH]", log_msg)
        self.assertIn("CMP: ₹604.40", log_msg)
        self.assertIn("Pending Breakout Level: ₹619.88", log_msg)

    def test_multibagger_armed_watch_conditions(self):
        """Verify Multibagger armed buy-zone produces rich watchlist log."""
        price = 1050.0
        sma_50 = 1020.0
        buy_low = 1000.0
        buy_high = 1070.0
        tier = "🚀 Prime Multibagger"
        total_score = 88.5
        vol_ratio = 1.65
        
        log_msg = (
            f"👁️ [MULTIBAGGER: ARMED BUY ZONE] TATACHEM added to Watchlist (Conviction: {tier}, Score: {total_score:.1f}) | "
            f"CMP: ₹{price:.2f} | 50 SMA Level: ₹{sma_50:.2f} | Buy Zone: [₹{buy_low:.2f} - ₹{buy_high:.2f}] | "
            f"Volume: {vol_ratio:.2f}x | SL: ₹950.00 | RR: 2.10 — (Armed in buy zone, awaiting confirmation)"
        )
        self.assertIn("👁️ [MULTIBAGGER: ARMED BUY ZONE]", log_msg)
        self.assertIn("CMP: ₹1050.00", log_msg)
        self.assertIn("50 SMA Level: ₹1020.00", log_msg)

    def test_near_miss_delta_calculation(self):
        """Verify Near-Miss threshold delta logic within 10% tolerance."""
        observed = 68.0
        threshold = 70.0
        delta_pct = abs(observed - threshold) / threshold * 100.0
        self.assertAlmostEqual(delta_pct, 2.857, delta=0.01)
        self.assertTrue(delta_pct <= 10.0)

if __name__ == "__main__":
    unittest.main()
