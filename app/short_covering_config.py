"""
app/short_covering_config.py

Configuration constants for Short-Covering Early-Ignition Scanner (2-Layer Architecture).
"""

from typing import Dict, Any

SHORT_COVERING_SCANNER_NAME = "SHORT_COVERING"
SHORT_COVERING_VERSION = "v2.0_EARLY_IGNITION"

# ── LAYER 1: EOD POSITIONING ENGINE SETTINGS ────────────────────────────────
EOD_POSITIONING_CONFIG: Dict[str, Any] = {
    "MIN_5D_OI_BUILDUP_PCT": 6.0,       # Minimum 5-day OI expansion percentage for short buildup
    "MIN_10D_OI_BUILDUP_PCT": 10.0,     # Minimum 10-day OI expansion percentage
    "MIN_SHORT_BUILDUP_RATIO": 0.55,    # Days with Price Down and OI Up / Total Days
    "MAX_RSI_14": 50.0,                 # RSI threshold for oversold/basing condition
    "MIN_LOOKBACK_DAYS": 10,            # Days of history required
}

# ── LAYER 2: INTRADAY 5M IGNITION SCANNER SETTINGS ──────────────────────────
INTRADAY_IGNITION_CONFIG: Dict[str, Any] = {
    "SCAN_INTERVAL_MINUTES": 5,         # Run every 5 minutes during 09:15 - 15:30
    "MIN_5M_OI_CONTRACTION_PCT": -0.35, # Minimum 5m OI contraction percentage
    "MIN_SESSION_OI_CONTRACTION_PCT": -0.80, # Session cumulative OI contraction
    "MIN_5M_VOLUME_SURGE_RATIO": 1.40,  # 5m volume vs 10-period 5m average volume
    "MIN_5M_PRICE_CHANGE_PCT": 0.15,    # 5m price rate of change
    "REQUIRE_ABOVE_VWAP": True,         # 5m close must be >= intraday VWAP
    "ROLLOVER_EXCLUSION_RATIO": 0.70,   # If next month OI absorbs >= 70% of near drop in expiry week
    "MIN_IGNITION_SCORE": 70.0,         # Minimum score to issue early alert
}

# ── SCORING WEIGHT DISTRIBUTION (Total = 100) ───────────────────────────────
IGNITION_SCORING_WEIGHTS: Dict[str, float] = {
    "PRIOR_SHORT_INTENSITY": 25.0,      # Score from Layer 1 EOD buildup
    "OI_CONTRACTION_SPEED": 25.0,       # 5m and session excess OI contraction
    "VOLUME_SURGE_CONVICTION": 20.0,    # 5m volume spike vs average
    "VWAP_PRICE_MOMENTUM": 15.0,        # Clean price acceleration above VWAP
    "MULTITF_RS_CONFIRMATION": 15.0,    # 15m/30m structure + RS vs NIFTY
}

# ── MASTER AGGREGATED CONFIG ────────────────────────────────────────────────
SHORT_COVERING_CONFIG: Dict[str, Any] = {
    "NAME": SHORT_COVERING_SCANNER_NAME,
    "VERSION": SHORT_COVERING_VERSION,
    "EOD": EOD_POSITIONING_CONFIG,
    "INTRADAY": INTRADAY_IGNITION_CONFIG,
    "WEIGHTS": IGNITION_SCORING_WEIGHTS,
}
