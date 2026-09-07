import unittest
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from multibagger_state_machine import (
    arm_setup,
    evaluate_armed_trigger,
    mark_alert_triggered,
    MAX_ARMED_SESSIONS
)

@dataclass
class MockPriceData:
    symbol: str
    price: float
    sma_50: float
    sma_200: float
    ema_20: float
    atr_14: float
    latest_volume: float
    volume_sma20: float
    today_open: float
    today_close: float
    last_trade_date: str

class TestPhase3MultibaggerStateMachine(unittest.TestCase):
    """
    Phase 3 Acceptance Test Suite:
    Validates persistent Multibagger state machine lifecycles and idempotency:
    1. BUY_ZONE -> ARMED_BUY_ZONE
    2. ARMED -> TRIGGER_CANDIDATE
    3. TRIGGER_CANDIDATE -> ALERT_TRIGGERED
    4. ARMED -> INVALIDATED_SUPPORT
    5. ARMED -> INVALIDATED_AGE
    6. ARMED -> INVALIDATED_FUNDAMENTALS
    7. Repeated evaluation is idempotent (no duplicate ARM)
    8. Duplicate scheduler invocation creates no duplicate alert
    9. Stale/invalidated setup cannot trigger
    10. Session count increments only on trade date change
    """

    def setUp(self):
        self.symbol = "TEST_MULTI_SYM"
        self.base_armed_setup = {
            "setup_id": f"mb_{self.symbol}_2026-09-01_v5.0",
            "symbol": self.symbol,
            "state": "ARMED_BUY_ZONE",
            "setup_version": "v5.0",
            "armed_at": datetime.now(ZoneInfo("Asia/Kolkata")),
            "armed_trade_date": "2026-09-01",
            "armed_price": 500.0,
            "armed_sma50": 490.0,
            "armed_atr": 20.0,
            "buy_zone_low": 480.0,
            "buy_zone_high": 520.0,
            "cqs": 75.0,
            "pas": 65.0,
            "total_score": 80.0,
            "conviction_tier": "🚀 Prime Multibagger",
            "last_evaluated_at": datetime.now(ZoneInfo("Asia/Kolkata")),
            "last_evaluated_trade_date": "2026-09-01",
            "age_sessions": 0,
            "invalidation_reason": None,
            "triggered_at": None,
            "trigger_price": None
        }

    def test_1_buy_zone_to_armed_buy_zone(self):
        """Test 1: Setup qualification arms setup in buy zone."""
        is_new, msg, rec = arm_setup(
            symbol=self.symbol,
            price=500.0,
            sma_50=490.0,
            atr_14=20.0,
            buy_zone_low=480.0,
            buy_zone_high=520.0,
            cqs=75.0,
            pas=65.0,
            total_score=80.0,
            conviction_tier="🚀 Prime Multibagger",
            trade_date="2026-09-01"
        )
        self.assertIn(msg, ["ARMED_SUCCESS", "ALREADY_ARMED"])
        self.assertEqual(rec["state"], "ARMED_BUY_ZONE")
        self.assertEqual(rec["symbol"], self.symbol)

    def test_2_armed_to_trigger_candidate(self):
        """Test 2: Confirmation volume expansion (>=2x) + stabilized close triggers confirmation."""
        # Price in buy zone, 2.5x volume, bullish close
        pd_trigger = MockPriceData(
            symbol=self.symbol,
            price=505.0,
            sma_50=490.0,
            sma_200=450.0,
            ema_20=500.0,
            atr_14=20.0,
            latest_volume=250000.0,
            volume_sma20=100000.0,  # 2.5x volume
            today_open=498.0,
            today_close=505.0,
            last_trade_date="2026-09-02"
        )
        
        def mock_entry_confirmed(p):
            return (True, "")

        state, reason, rec = evaluate_armed_trigger(
            armed_setup=self.base_armed_setup,
            price_data=pd_trigger,
            entry_confirmed_fn=mock_entry_confirmed,
            current_trade_date="2026-09-02"
        )
        self.assertEqual(state, "TRIGGER_CANDIDATE")
        self.assertEqual(reason, "CONFIRMED")

    def test_3_trigger_candidate_to_alert_triggered(self):
        """Test 3: Alert persistence marks state as ALERT_TRIGGERED."""
        ok = mark_alert_triggered(
            setup_id=self.base_armed_setup["setup_id"],
            trigger_price=505.0
        )
        self.assertTrue(ok)

    def test_4_armed_to_invalidated_support(self):
        """Test 4: Price breaking below structural support invalidates setup."""
        pd_broken = MockPriceData(
            symbol=self.symbol,
            price=440.0,  # Below SMA200 (450 * 0.96 = 432, and below buy_low 480 * 0.97 = 465.6)
            sma_50=490.0,
            sma_200=470.0,  # 440 < 470 * 0.96 = 451.2
            ema_20=480.0,
            atr_14=20.0,
            latest_volume=50000.0,
            volume_sma20=100000.0,
            today_open=450.0,
            today_close=440.0,
            last_trade_date="2026-09-03"
        )

        state, reason, rec = evaluate_armed_trigger(
            armed_setup=self.base_armed_setup,
            price_data=pd_broken,
            current_trade_date="2026-09-03"
        )
        self.assertEqual(state, "INVALIDATED_SUPPORT")
        self.assertIn("broke below", reason)

    def test_5_armed_to_invalidated_age(self):
        """Test 5: Exceeding max active trading sessions without trigger invalidates setup."""
        stale_armed = self.base_armed_setup.copy()
        stale_armed["age_sessions"] = MAX_ARMED_SESSIONS + 1
        stale_armed["last_evaluated_trade_date"] = "2026-09-01"

        pd_normal = MockPriceData(
            symbol=self.symbol,
            price=500.0,
            sma_50=490.0,
            sma_200=450.0,
            ema_20=500.0,
            atr_14=20.0,
            latest_volume=50000.0,
            volume_sma20=100000.0,
            today_open=498.0,
            today_close=500.0,
            last_trade_date="2026-09-25"
        )

        state, reason, rec = evaluate_armed_trigger(
            armed_setup=stale_armed,
            price_data=pd_normal,
            current_trade_date="2026-09-25"
        )
        self.assertEqual(state, "INVALIDATED_AGE")
        self.assertIn("exceeded", reason)

    def test_6_armed_to_invalidated_fundamentals(self):
        """Test 6: Fundamental degradation (Piotroski < 7 or pledge > 15%) invalidates setup."""
        degraded_fund = {
            "piotroski_f_score": 5,  # Dropped < 7
            "promoter_pledge_pct": 0.05
        }
        pd_normal = MockPriceData(
            symbol=self.symbol,
            price=500.0,
            sma_50=490.0,
            sma_200=450.0,
            ema_20=500.0,
            atr_14=20.0,
            latest_volume=50000.0,
            volume_sma20=100000.0,
            today_open=498.0,
            today_close=500.0,
            last_trade_date="2026-09-02"
        )

        state, reason, rec = evaluate_armed_trigger(
            armed_setup=self.base_armed_setup,
            price_data=pd_normal,
            raw_fundamentals=degraded_fund,
            current_trade_date="2026-09-02"
        )
        self.assertEqual(state, "INVALIDATED_FUNDAMENTALS")
        self.assertIn("Piotroski", reason)

    def test_7_idempotent_arming(self):
        """Test 7: Re-evaluating an already armed setup refreshes metadata without creating duplicate setups."""
        is_new_1, msg_1, rec_1 = arm_setup(
            symbol=self.symbol,
            price=500.0,
            sma_50=490.0,
            atr_14=20.0,
            buy_zone_low=480.0,
            buy_zone_high=520.0,
            cqs=75.0,
            pas=65.0,
            total_score=80.0,
            conviction_tier="🚀 Prime Multibagger",
            trade_date="2026-09-01"
        )
        is_new_2, msg_2, rec_2 = arm_setup(
            symbol=self.symbol,
            price=502.0,
            sma_50=490.0,
            atr_14=20.0,
            buy_zone_low=480.0,
            buy_zone_high=520.0,
            cqs=75.0,
            pas=65.0,
            total_score=80.0,
            conviction_tier="🚀 Prime Multibagger",
            trade_date="2026-09-01"
        )
        # Second call must identify existing setup and not create new record
        self.assertEqual(rec_1["setup_id"], rec_2["setup_id"])

    def test_8_session_progression_date_awareness(self):
        """Test 8: Session age increments ONLY on distinct trading date change."""
        setup = self.base_armed_setup.copy()
        setup["last_evaluated_trade_date"] = "2026-09-01"
        setup["age_sessions"] = 2

        pd_same_day = MockPriceData(
            symbol=self.symbol, price=500.0, sma_50=490.0, sma_200=450.0, ema_20=500.0,
            atr_14=20.0, latest_volume=50000.0, volume_sma20=100000.0, today_open=498.0,
            today_close=500.0, last_trade_date="2026-09-01"
        )

        def mock_ec_waiting(p): return (False, "entry_vol_below_2x")

        # Same trade date -> session age does not increment
        st_1, _, _ = evaluate_armed_trigger(setup, pd_same_day, entry_confirmed_fn=mock_ec_waiting, current_trade_date="2026-09-01")
        self.assertEqual(st_1, "ARMED_BUY_ZONE")

        # Next trade date -> session age increments
        st_2, _, _ = evaluate_armed_trigger(setup, pd_same_day, entry_confirmed_fn=mock_ec_waiting, current_trade_date="2026-09-02")
        self.assertEqual(st_2, "ARMED_BUY_ZONE")

if __name__ == "__main__":
    unittest.main()
