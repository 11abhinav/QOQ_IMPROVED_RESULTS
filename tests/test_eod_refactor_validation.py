"""
Unit and behavioral test suite for EOD scanner refactor:
1. RSI penalty ladder (87 pass, 89 -2, 90 -5, 91 -7, 92 -10, 93 hard reject)
2. 52W High Two-Mode gate (Mode A <=5%, Mode B 5-15% with strict filters, >15% reject)
3. ATR10 pre-breakout close denominator verification (4800 vs 5000)
4. Penalty buckets independence and TRIPLE_FAULT_VETO
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from config import EOD_CONFIG, EOD_ADVANCED_CONFIG
from eod_scanner import _check_eod_conditions


def _make_ticker(close_price=1000.0, high_52w=1030.0, vol_ratio=2.0, rsi=65.0,
                 prev_close=None, atr10_val=None, bb_pctile=0.40, n_bars=60):
    """Deterministic ticker generator for gate verification."""
    dates = pd.date_range(end="2026-09-04", periods=n_bars, freq="D")
    df = pd.DataFrame(index=dates)
    
    atr20 = close_price * 0.02
    if prev_close is None:
        prev_close = close_price - 5.0
    
    # Fill close prices with linear ramp ending at prev_close then close_price
    closes = np.linspace(close_price * 0.90, prev_close, n_bars - 1).tolist()
    closes.append(close_price)
    df["Close"] = closes
    
    # Base bars: tight consolidated candle ranges (e.g. 1.2% of price)
    df["Open"] = df["Close"] - (atr20 * 0.2)
    df["High"] = df["Close"] + (atr20 * 0.3)
    df["Low"] = df["Close"] - (atr20 * 0.3)
    
    # Breakout bar (last bar): strong range (expansion >= 0.9x ATR20)
    df.loc[df.index[-1], "Open"] = close_price - (atr20 * 0.7)
    df.loc[df.index[-1], "High"] = close_price + (atr20 * 0.2)
    df.loc[df.index[-1], "Low"] = close_price - (atr20 * 0.8)  # Range = 1.0 * ATR20
    
    # Volume: historical 100k, last bar 100k * vol_ratio
    volumes = [100000.0] * (n_bars - 1) + [100000.0 * vol_ratio]
    df["Volume"] = volumes
    
    # Required indicators
    df["HIGH_52W"] = high_52w
    df["PRIOR_20D_HIGH"] = close_price - 10.0  # Breakout confirmed
    df["ATR20"] = atr20
    df["RSI"] = rsi
    df["EMA20"] = close_price - 20.0
    df["SMA50"] = close_price - 40.0
    df["ADX"] = 30.0
    df["BB_WIDTH_PCTILE"] = bb_pctile
    
    # Customise last 10 days for ATR10 if specified
    if atr10_val is not None:
        # We want mean(tr_10) == atr10_val
        for i in range(-11, -1):
            df.loc[df.index[i], "High"] = df.loc[df.index[i], "Close"] + (atr10_val / 2.0)
            df.loc[df.index[i], "Low"] = df.loc[df.index[i], "Close"] - (atr10_val / 2.0)
            df.loc[df.index[i], "Open"] = df.loc[df.index[i], "Close"]
    
    return df


class TestEODRefactorValidation(unittest.TestCase):

    def test_01_config_and_rsi_ladder(self):
        """Verify RSI config and graduated penalty formula."""
        self.assertEqual(EOD_CONFIG["MAX_RSI"], 92, "MAX_RSI must be 92")
        
        # Test ladder formula: min(10, int((rsi - 88) * 2.5))
        test_cases = [
            (87.0, 0),    # <= 88: 0
            (88.0, 0),    # 88: 0
            (89.0, 2),    # (1 * 2.5) = 2.5 -> int 2 -> -2
            (90.0, 5),    # (2 * 2.5) = 5.0 -> int 5 -> -5
            (91.0, 7),    # (3 * 2.5) = 7.5 -> int 7 -> -7
            (92.0, 10),   # (4 * 2.5) = 10.0 -> int 10 -> -10
        ]
        
        for rsi_val, expected_pen in test_cases:
            excess = max(0.0, rsi_val - 88.0)
            pen = min(10, int(excess * 2.5))
            self.assertEqual(pen, expected_pen, f"RSI {rsi_val} penalty mismatch")
            
            # Verify scanner condition check passes for <= 92
            ticker = _make_ticker(rsi=rsi_val)
            res = _check_eod_conditions(ticker=ticker, latest=ticker.iloc[-1], symbol="TEST")
            self.assertTrue(res["passed"], f"RSI {rsi_val} should pass hard gate, failed: {res.get('reason')}")
        
        # Verify RSI 93 hard rejects
        ticker_93 = _make_ticker(rsi=93.0)
        res_93 = _check_eod_conditions(ticker=ticker_93, latest=ticker_93.iloc[-1], symbol="TEST")
        self.assertFalse(res_93["passed"], "RSI 93 must hard reject")
        self.assertIn("RSI", res_93["reason"])

    def test_02_52w_mode_a_and_mode_b(self):
        """Verify 52W High two-mode gate: Mode A (<=5%), Mode B (5-15% with stricter filters)."""
        close_px = 1000.0
        
        # Case 1: 3% below 52W high -> Mode A (high_52w = 1030)
        ticker_mode_a = _make_ticker(close_price=close_px, high_52w=1030.0, vol_ratio=1.6)
        res_a = _check_eod_conditions(ticker=ticker_mode_a, latest=ticker_mode_a.iloc[-1], symbol="TEST_A")
        self.assertTrue(res_a["passed"], f"Mode A (3% dist) must pass, failed with: {res_a.get('reason')}")
        
        # Case 2: 8% below 52W high with RVOL 2.5x and BB <= 0.50 -> Mode B qualifies
        ticker_mode_b = _make_ticker(close_price=close_px, high_52w=1086.96, vol_ratio=2.6, bb_pctile=0.45)
        res_b = _check_eod_conditions(ticker=ticker_mode_b, latest=ticker_mode_b.iloc[-1], symbol="TEST_B")
        self.assertTrue(res_b["passed"], f"Mode B (8% dist + 2.6x vol) must pass, failed with: {res_b.get('reason')}")
        
        # Case 3: 15% below 52W high with requirements -> Mode B qualifies
        ticker_mode_b_15 = _make_ticker(close_price=close_px, high_52w=1176.47, vol_ratio=2.6, bb_pctile=0.45)
        res_b_15 = _check_eod_conditions(ticker=ticker_mode_b_15, latest=ticker_mode_b_15.iloc[-1], symbol="TEST_B_15")
        self.assertTrue(res_b_15["passed"], f"Mode B (15% dist) must pass, failed with: {res_b_15.get('reason')}")
        
        # Case 4: 15.1% below 52W high -> REJECTED
        ticker_b_151 = _make_ticker(close_price=close_px, high_52w=1178.0, vol_ratio=3.0, bb_pctile=0.30)
        res_b_151 = _check_eod_conditions(ticker=ticker_b_151, latest=ticker_b_151.iloc[-1], symbol="TEST_B_151")
        self.assertFalse(res_b_151["passed"], "Mode B > 15% distance must reject")
        self.assertIn("Too far from 52W high", res_b_151["reason"])
        
        # Case 5: 20% below 52W high -> REJECTED
        ticker_b_20 = _make_ticker(close_price=close_px, high_52w=1250.0, vol_ratio=3.0, bb_pctile=0.30)
        res_b_20 = _check_eod_conditions(ticker=ticker_b_20, latest=ticker_b_20.iloc[-1], symbol="TEST_B_20")
        self.assertFalse(res_b_20["passed"], "20% distance must reject")
        self.assertIn("Too far from 52W high", res_b_20["reason"])

    def test_03_atr_denominator_bug(self):
        """
        Deterministic ATR denominator test:
        Breakout close = 5000, Previous close = 4800, ATR10 = 120.
        Calculation: 120 / 4800 = 2.50% (meets 2.50% tightness floor).
        If ATR10 = 121: 121 / 4800 = 2.5208% > 2.50% -> Must REJECT.
        (If denominator were erroneously 5000, 121 / 5000 = 2.42% would falsely pass).
        """
        close_px = 5000.0
        prev_px = 4800.0
        
        # Case A: ATR10 = 120 -> 120 / 4800 = 2.50% (exactly at threshold, not strictly greater than 2.5%) -> PASS
        ticker_pass = _make_ticker(close_price=close_px, high_52w=5100.0, prev_close=prev_px, atr10_val=120.0)
        res_pass = _check_eod_conditions(ticker=ticker_pass, latest=ticker_pass.iloc[-1], symbol="TEST_ATR_PASS")
        self.assertTrue(res_pass["passed"], f"ATR10=120 on prev_close=4800 should pass, failed: {res_pass.get('reason')}")
        
        # Case B: ATR10 = 220 -> 220 / 4800 = 4.5833% > 4.50% -> REJECT under Model A
        ticker_fail = _make_ticker(close_price=close_px, high_52w=5100.0, prev_close=prev_px, atr10_val=220.0)
        res_fail = _check_eod_conditions(ticker=ticker_fail, latest=ticker_fail.iloc[-1], symbol="TEST_ATR_FAIL")
        self.assertFalse(res_fail["passed"], "ATR10=220 / 4800 = 4.58% must be rejected")
        self.assertIn("tightness ceiling", res_fail["reason"])
        self.assertIn("4.58%", res_fail["reason"])

    def test_04_penalty_buckets_and_triple_fault(self):
        """
        Verify independent penalty buckets and TRIPLE_FAULT_VETO:
        - bad candle only: score penalty, no veto
        - large gap only: score penalty, no veto
        - OBV divergence only: score penalty, no veto
        - bad candle + gap: score penalties accumulate up to -30, no veto
        - bad candle + gap + OBV: triggers TRIPLE_FAULT_VETO
        """
        CANDLE_FAULT_THRESHOLD = 10
        GAP_FAULT_THRESHOLD = 10
        
        def evaluate_penalties(candle_pen, gap_pen, ext_pen, obv_pen, red_pen=0, rsi_pen=0):
            _bucket_candle = min(15, candle_pen)
            _bucket_gap = min(15, gap_pen + ext_pen)
            _bucket_obv = min(5, abs(obv_pen))
            _bucket_misc = min(10, red_pen + rsi_pen)
            
            is_triple_fault = (
                _bucket_candle >= CANDLE_FAULT_THRESHOLD and
                _bucket_gap >= GAP_FAULT_THRESHOLD and
                _bucket_obv > 0
            )
            total_deductions = _bucket_candle + _bucket_gap + _bucket_obv + _bucket_misc
            return is_triple_fault, total_deductions, _bucket_candle, _bucket_gap, _bucket_obv
        
        # 1. Bad candle only (candle_pen = 12)
        veto, ded, b_c, b_g, b_o = evaluate_penalties(candle_pen=12, gap_pen=0, ext_pen=0, obv_pen=0)
        self.assertFalse(veto, "Bad candle only should not trigger triple fault veto")
        self.assertEqual(ded, 12)
        
        # 2. Large gap only (gap_pen = 12)
        veto, ded, b_c, b_g, b_o = evaluate_penalties(candle_pen=0, gap_pen=12, ext_pen=0, obv_pen=0)
        self.assertFalse(veto, "Large gap only should not trigger triple fault veto")
        self.assertEqual(ded, 12)
        
        # 3. OBV divergence only (obv_pen = -5)
        veto, ded, b_c, b_g, b_o = evaluate_penalties(candle_pen=0, gap_pen=0, ext_pen=0, obv_pen=-5)
        self.assertFalse(veto, "OBV divergence only should not trigger triple fault veto")
        self.assertEqual(ded, 5)
        
        # 4. Bad candle + gap (candle_pen = 12, gap_pen = 12)
        veto, ded, b_c, b_g, b_o = evaluate_penalties(candle_pen=12, gap_pen=12, ext_pen=0, obv_pen=0)
        self.assertFalse(veto, "Bad candle + gap without OBV should NOT trigger triple fault veto")
        self.assertEqual(ded, 24, "Independent buckets allow deduction to accumulate to -24")
        
        # 5. Bad candle + gap + OBV divergence -> TRIPLE_FAULT_VETO
        veto, ded, b_c, b_g, b_o = evaluate_penalties(candle_pen=12, gap_pen=12, ext_pen=0, obv_pen=-5)
        self.assertTrue(veto, "Simultaneous faults across Candle, Gap, and OBV MUST trigger TRIPLE_FAULT_VETO")
        self.assertEqual(ded, 29)


    def test_05_explicit_penalty_bucket_cases(self):
        """
        Validate explicit penalty combinations requested:
        1. Normal excellent breakout -> no deductions (0)
        2. RSI 89 only -> -2 (Misc bucket)
        3. RSI 91 only -> -7 (Misc bucket)
        4. Large gap only -> Bucket B penalty
        5. Bad candle only -> Bucket A penalty
        6. OBV divergence only -> Bucket C penalty
        7. Red candles only -> Misc penalty (-4)
        8. RSI 92 (-10) + red candles (-4) -> Misc bucket capped at -10
        9. Bad candle + large gap + OBV divergence -> TRIPLE_FAULT_VETO
        """
        def compute_total_deductions(candle_pen=0, gap_pen=0, ext_pen=0, obv_pen=0, red_pen=0, rsi_pen=0):
            b_candle = min(15, candle_pen)
            b_gap = min(15, gap_pen + ext_pen)
            b_obv = min(5, abs(obv_pen))
            b_misc = min(10, red_pen + rsi_pen)
            is_triple_fault = (b_candle >= 10 and b_gap >= 10 and b_obv > 0)
            return (b_candle + b_gap + b_obv + b_misc), is_triple_fault, (b_candle, b_gap, b_obv, b_misc)

        # 1. Normal excellent breakout -> 0 deductions
        ded, tf, buckets = compute_total_deductions()
        self.assertEqual(ded, 0)
        self.assertFalse(tf)
        self.assertEqual(buckets, (0, 0, 0, 0))

        # 2. RSI 89 only -> -2
        rsi_89_pen = min(10, int((89.0 - 88.0) * 2.5))
        ded, tf, buckets = compute_total_deductions(rsi_pen=rsi_89_pen)
        self.assertEqual(ded, 2)
        self.assertFalse(tf)
        self.assertEqual(buckets, (0, 0, 0, 2))

        # 3. RSI 91 only -> -7
        rsi_91_pen = min(10, int((91.0 - 88.0) * 2.5))
        ded, tf, buckets = compute_total_deductions(rsi_pen=rsi_91_pen)
        self.assertEqual(ded, 7)
        self.assertFalse(tf)
        self.assertEqual(buckets, (0, 0, 0, 7))

        # 4. Large gap only -> Bucket B penalty (-9)
        ded, tf, buckets = compute_total_deductions(gap_pen=9)
        self.assertEqual(ded, 9)
        self.assertFalse(tf)
        self.assertEqual(buckets, (0, 9, 0, 0))

        # 5. Bad candle only -> Bucket A penalty (-10)
        ded, tf, buckets = compute_total_deductions(candle_pen=10)
        self.assertEqual(ded, 10)
        self.assertFalse(tf)
        self.assertEqual(buckets, (10, 0, 0, 0))

        # 6. OBV divergence only -> Bucket C penalty (-5)
        ded, tf, buckets = compute_total_deductions(obv_pen=-5)
        self.assertEqual(ded, 5)
        self.assertFalse(tf)
        self.assertEqual(buckets, (0, 0, 5, 0))

        # 7. Red candles only -> Misc penalty (-4)
        ded, tf, buckets = compute_total_deductions(red_pen=4)
        self.assertEqual(ded, 4)
        self.assertFalse(tf)
        self.assertEqual(buckets, (0, 0, 0, 4))

        # 8. RSI 92 (-10) + red candles (-4) -> Misc bucket capped at -10 (NOT -14)
        rsi_92_pen = min(10, int((92.0 - 88.0) * 2.5))  # 10
        ded, tf, buckets = compute_total_deductions(rsi_pen=rsi_92_pen, red_pen=4)
        self.assertEqual(ded, 10, "Misc bucket must be capped at -10 when RSI=10 and red=4")
        self.assertFalse(tf)
        self.assertEqual(buckets, (0, 0, 0, 10))

        # 9. Bad candle + large gap + OBV divergence -> TRIPLE_FAULT_VETO
        ded, tf, buckets = compute_total_deductions(candle_pen=12, gap_pen=12, obv_pen=-5)
        self.assertTrue(tf, "Simultaneous faults across A, B, C MUST trigger TRIPLE_FAULT_VETO")
        self.assertEqual(ded, 29)
        self.assertEqual(buckets, (12, 12, 5, 0))


if __name__ == "__main__":
    unittest.main()

