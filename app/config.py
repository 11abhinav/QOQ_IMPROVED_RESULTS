# =====================================================================================
# app/config.py
# Centralized configuration for all scanners
# =====================================================================================

import os
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# =====================================================================================
# BASE DIRECTORY
# =====================================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================================================================
# TELEGRAM CONFIG (DYNAMIC ENVIRONMENT READ)
# =====================================================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

_thread_eod      = os.getenv("THREAD_EOD")
_thread_multi_tf = os.getenv("THREAD_MULTI_TF")
_thread_1h       = os.getenv("THREAD_1H")
_thread_reversal = os.getenv("THREAD_REVERSAL")

THREAD_EOD      = int(_thread_eod)      if _thread_eod      else None
THREAD_MULTI_TF = int(_thread_multi_tf) if _thread_multi_tf else None
THREAD_1H       = int(_thread_1h)       if _thread_1h       else None
THREAD_REVERSAL = int(_thread_reversal) if _thread_reversal else None

# =====================================================================================
# DATA DIRECTORY & PATHS
# =====================================================================================

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

WATCHLIST_PATH = os.path.join(DATA_DIR, "elite_fundamental_watchlist.parquet")
DB_PATH = os.path.join(DATA_DIR, "alerts.db")

# Path for NSE constituent disk cache (Fix RCA-MB-2)
CONSTITUENT_CACHE_PATH = os.path.join(DATA_DIR, "constituent_cache.json")

# [VERSION: NON_EQUITY_BLOCKLIST_v2.0] Authoritative blocklist for non-equity trusts (InvITs / REITs)
# These instruments do not have standard equity breakout patterns and must not enter equity scanners.
NON_EQUITY_BLOCKLIST = {
    "VERTIS", "HIGHWAYS", "POWERINVIT", "IRBINVIT", "INDIGRID",
    "EMBASSY", "MINDSPACE", "BROOKFIELD", "NEXUS"
}

# =====================================================================================
# RESILIENCE / FALLBACK CONFIGURATION
# =====================================================================================

# [VERSION: WEALTH_CB_FALLBACK_v1.0] Maximum age (hours) of the saved wealth parquet
# that is still acceptable for the circuit-breaker fallback path.
# When YFinance/Fyers circuit breakers fire, Wealth Engine will load the last-saved
# parquet (suppressing new BUY signals but keeping exit monitor running) if the
# parquet was written within this many hours. 12h covers the overnight 2AM scan
# through end of trading day. Beyond 12h the system aborts as before.
WEALTH_CB_FALLBACK_MAX_AGE_HOURS = int(os.environ.get("WEALTH_CB_FALLBACK_MAX_AGE_HOURS", "12"))

# [VERSION: CONSTITUENT_DISK_CACHE_v1.0] Maximum age (days) of the on-disk NSE
# constituent cache file before it is considered too stale to use as a last resort.
# NSE index rebalancing happens quarterly, so 7 days is always safe.
CONSTITUENT_DISK_CACHE_MAX_DAYS = int(os.environ.get("CONSTITUENT_DISK_CACHE_MAX_DAYS", "7"))

# =====================================================================================
# SYSTEM & PROFILING CONFIGURATION
# =====================================================================================

MEMORY_PROFILER_CONFIG = {
    "DEEP_DIAGNOSTIC_RSS_MB": 5.0,
    "MIN_DF_DELTA_MB": 1.0,
    "MAX_TRACEMALLOC_PEAK_MB": 20.0,
    "CONSECUTIVE_TRIGGER_COUNT": 3,
    "RATE_LIMIT_MINUTES": 30
}

# =====================================================================================
# API / FETCH CONFIGURATION
# =====================================================================================

DISABLE_NSE_SURVEILLANCE_FETCH = False  # Set to True in validation environments to avoid WAF/tarpit timeouts
CRAWLORA_API_KEY = os.getenv("CRAWLORA_API_KEY")

# =====================================================================================
# SCORE THRESHOLDS & AI
# =====================================================================================

ENABLE_AI_SENTIMENT_SCORE = True  # Set False to disable experimental AI sentiment scoring for audit/backtest runs

SCORE_THRESHOLDS = {
    "15m": 75,
    "1h":  75,
    "1d":  75,
}

# =====================================================================================
# SCAN CONFIGURATION (Algorithm Parameters)
# =====================================================================================
ACTIVE_ALGO_VERSION = "SL_ENGINE_V7.1"  # Updated: Target Engine v7 Pipeline, Institutional S/R Clustering, Parallel Orchestration + Combined Audit Fixes

def get_system_version() -> str:
    """Dynamically resolves deployment version incorporating git commit hash."""
    env_ver = os.getenv("DEPLOYMENT_VERSION") or os.getenv("SYSTEM_DEPLOYMENT_VERSION")
    if env_ver:
        return env_ver

    base_ver = "v1"
    commit_sha = ""

    # Check local version.json if generated during build/deployment
    import json
    ver_file = os.path.join(BASE_DIR, "app", "version.json")
    if os.path.exists(ver_file):
        try:
            with open(ver_file, "r") as f:
                data = json.load(f)
                if data.get("version"):
                    return data["version"]
                commit_sha = data.get("commit", "")
        except Exception:
            pass

    if not commit_sha:
        try:
            import subprocess
            res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout:
                commit_sha = res.stdout.strip()
        except Exception:
            pass

    if commit_sha:
        return f"{base_ver}-{commit_sha}"
    return base_ver

SYSTEM_DEPLOYMENT_VERSION = get_system_version()

# =====================================================================================
# MOMENTUM BONUS CONSTANTS & RULE 10 RATIONALE
# =====================================================================================
# RS_BONUS (10 pts): Awarded if stock's 63-day RS rating is >= 80th percentile vs Nifty 50 over active scan universe.
# SECTOR_BONUS (8 pts): Awarded if stock belongs to a Top-3 RS sector holding 3-session hysteresis.
# MAX_MOMENTUM_BONUS (15 pts): Hard cap on combined momentum bonuses so RS (+10) and Sector (+8) co-exist (10+5=15) without clipping Sector to zero.
RS_BONUS = 10
SECTOR_BONUS = 8
MAX_MOMENTUM_BONUS = 15



MULTI_TF_CONFIG = {
    "MIN_SIGNALS":        2,
    "MIN_BODY_RATIO":     0.60,
    "MIN_CLOSE_POSITION": 0.65,
    "MAX_UPPER_WICK":     0.35,
    "MIN_VOLUME_RATIO":   1.25,
    "MIN_VOLUME_AVG":     150_000,
    "MIN_RSI":            52,
    "MAX_RSI":            87,
    # [VERSION: PHASE_D_PULLBACK_REFINEMENT_v1.0] Set default pullback mode to PREVIOUS_CLOSE
    "PULLBACK_TRIGGER_MODE": "PREVIOUS_CLOSE", # Alternatives: PREVIOUS_CLOSE, PREVIOUS_BODY, PREVIOUS_HIGH, PREVIOUS_OPEN, INSIDE_BAR
}

MULTI_TF_SCHEDULE_METADATA = "15m Universe Scan / 5m Armed Monitor (09:30–15:30 IST)"

MULTI_TF_V2_CONFIG = {
    # ── CONTEXT (1H / 30m) ──
    "H1_BULLISH_SCORE":              10,
    "H1_NEUTRAL_SCORE":              0,
    "H1_BEARISH_SCORE":             -10,
    "M30_ROOM_THRESHOLD_PCT":       0.02,
    "M30_ROOM_SCORE":               10,

    # ── BASE GEOMETRY (ADAPTIVE V3 REDESIGN) ──
    # Multi-window candidate evaluation (6–35 candles) avoids monolithic 35-bar over-penalization.
    # Evaluates tight intraday coils (6-12 bars) up to mature multi-day shelves (24-35 bars).
    "MIN_CONSOLIDATION_BARS":       6,       # Minimum base window (1.5 hours)
    "MAX_CONSOLIDATION_BARS":       35,      # Maximum multi-day base lookback (up to 1.5 sessions)
    "CANDIDATE_WINDOW_BARS":        [6, 8, 10, 12, 16, 20, 24, 30, 35],
    "MIN_BOX_OCCUPANCY":            0.60,
    "MAX_BOX_WIDTH_PCT":            0.045,   # Fallback width % cap
    "MAX_BOX_WIDTH_ATR":            3.60,   # Absolute upper ceiling for longest bases
    # Duration-aware ATR & PCT width limits: (min_bars, max_bars) -> {max_atr, max_pct}
    "DURATION_ATR_WIDTH_LIMITS": [
        {"max_bars": 8,  "max_atr": 2.00, "max_pct": 0.035},
        {"max_bars": 14, "max_atr": 2.50, "max_pct": 0.045},
        {"max_bars": 22, "max_atr": 3.00, "max_pct": 0.055},
        {"max_bars": 35, "max_atr": 3.60, "max_pct": 0.065},
    ],
    "MIN_RESISTANCE_TESTS":         1,       # At least 1 touch required for initial watching base
    "GAP_PCT_THRESHOLD":            0.020,   # 2.0% gap limit to avoid truncating normal multi-day bases
    "GAP_ATR_MULT":                 2.0,
    "BOX_HIGH_QUANTILE":            0.90,
    "BOX_LOW_QUANTILE":             0.10,
    "RESISTANCE_TEST_TOL_PCT":      0.0015,  # 0.15% of price for touch detection
    "RESISTANCE_TEST_TOL_ATR":      0.08,
    "PIVOT_CONFIRM_ATR_MULT":       0.20,
    "PIVOT_CONFIRM_BOX_MULT":       0.15,
    "DORMANCY_MIN_VOL_RATIO":       0.15,   # Bases with volume < 15% of 20-period median are dormant
    "DORMANCY_PENALTY":             15,     # Penalty deducted for dormant/frozen price action

    # ── V3: BASE QUALITY ENGINE (0-100) — 7 Components ──
    # A. Maturity (15 pts): duration × quality interaction
    "SCORE_MATURITY_MAX":           15,
    "MATURITY_TIGHTNESS_THRESHOLD": 8,      # If tightness score < 8, cap maturity at 10 pts
    # B. Tightness (20 pts): range / 15m ATR
    "SCORE_TIGHTNESS_MAX":          20,
    # C. Resistance Quality (20 pts): std dev of top highs & ceiling stability
    "SCORE_RESISTANCE_QUALITY_MAX": 20,
    # D. Repeated Tests (15 pts): distinct touches
    "SCORE_REPEATED_TESTS_MAX":     15,
    # E. Compression/VCP (15 pts): early-ATR / late-ATR
    "SCORE_COMPRESSION_MAX":        15,
    # F. Higher Lows (10 pts): rising lows = buyers getting aggressive
    "SCORE_HIGHER_LOWS_MAX":        10,
    "HIGHER_LOWS_MIN_RISE_ATR":     0.15,   # Strong HL: late_low >= early_low + 0.15× ATR
    # G. Support Integrity (5 pts): rapid defense & structural integrity
    "SCORE_SUPPORT_INTEGRITY_MAX":  5,
    "SUPPORT_ZONE_ATR_MULT":        0.20,   # Lower zone = box_low + 0.20× ATR
    "SUPPORT_INTEGRITY_LOW_PCT":    0.20,   # < 20% of bars touch floor → clean support

    # Quality tier thresholds (Base Score)
    "MIN_SETUP_SCORE":              70,      # >= 70 → 15M_BREAKOUT_WATCH
    "MONITOR_SETUP_SCORE":          50,      # >= 50 → WATCHING base in watchlist
    "STRONG_SETUP_SCORE":           80,      # SUPER BASE tier
    "PREMIUM_SETUP_SCORE":          90,      # EXCEPTIONAL BASE tier

    # ── V3: 5M BREAKOUT STRENGTH ENGINE (0-100) — 7 Orthogonal Components ──
    # A. Volume Expansion / RVOL (25 pts)
    "SCORE_RVOL_MAX":               25,
    "RVOL_EXCEPTIONAL":             3.0,    # > 3.0× → 25 pts
    "RVOL_VERY_STRONG":             2.0,    # 2.0–3.0× → 22 pts
    "RVOL_STRONG":                  1.5,    # 1.5–2.0× → 18 pts
    "RVOL_CONFIRMED":               1.25,   # 1.25–1.5× → 12 pts
    "RVOL_NORMAL":                  1.0,    # 1.0–1.25× → 6 pts
    # B. Volume Acceleration (10 pts): vs previous 5m bar
    "SCORE_VOL_ACCEL_MAX":          10,
    # C. Base-Relative Volume (10 pts): vs 15m consolidation median bar volume
    "SCORE_BASE_REL_VOL_MAX":       10,
    # D. Breakout Penetration (20 pts): Cross-validated ATR distance & % price expansion (NO double-counting)
    "SCORE_PENETRATION_MAX":        20,
    "MAGNITUDE_IDEAL_MIN_ATR":      0.25,   # Below this → weaker penetration
    "MAGNITUDE_IDEAL_MAX_ATR":      0.70,   # Above this → possible extension
    # E. Candle Quality (15 pts): close position + range expansion
    "SCORE_CANDLE_QUALITY_MAX":     15,
    # F. Bar Breakout Velocity (10 pts): ATR/min
    "SCORE_VELOCITY_MAX":           10,
    "VELOCITY_EXPLOSIVE_ATR_MIN":   0.15,   # >= 0.15 ATR/min → EXPLOSIVE
    "VELOCITY_VERY_FAST_ATR_MIN":   0.08,   # >= 0.08 ATR/min → VERY FAST
    "VELOCITY_FAST_ATR_MIN":        0.04,   # >= 0.04 ATR/min → FAST
    # G. Market/Sector Relative Strength (10 pts): stock vs NIFTY (omitted from denom if unavailable)
    "SCORE_MARKET_RS_MAX":          10,
    "MARKET_RS_STRONG_LEAD":        0.005,  # stock > NIFTY + 0.5% → full points

    # Breakout quality tier thresholds (Breakout Score)
    "MIN_BREAKOUT_SCORE":           65,      # < 65 → WEAK, DB-only (no push)
    "STRONG_BREAKOUT_SCORE":        80,      # SUPER tier
    "EXPLOSIVE_BREAKOUT_SCORE":     90,      # EXPLOSIVE tier

    # ── V3: ALERT SEVERITY CLASSIFICATION ──
    "SEVERITY_APLUS_BASE":          90,      # A+ SETUP: base >= 90
    "SEVERITY_APLUS_BREAKOUT":      85,      # AND breakout >= 85
    "SEVERITY_APLUS_RVOL":          2.0,     # AND RVOL >= 2×
    "SEVERITY_EXPLOSIVE_BASE":      85,      # EXPLOSIVE: base >= 85
    "SEVERITY_EXPLOSIVE_BREAKOUT":  80,      # AND breakout >= 80
    "SEVERITY_EXPLOSIVE_RVOL":      1.75,    # AND RVOL >= 1.75×
    "SEVERITY_SUPER_BASE":          80,      # SUPER: base >= 80
    "SEVERITY_SUPER_BREAKOUT":      75,      # AND breakout >= 75
    "SEVERITY_SUPER_RVOL":          1.40,    # AND RVOL >= 1.40×
    "SEVERITY_GOOD_BASE":           70,      # GOOD: base >= 70
    "SEVERITY_GOOD_BREAKOUT":       65,      # AND breakout >= 65
    "SEVERITY_GOOD_RVOL":           1.25,    # AND RVOL >= 1.25×

    # ── 5M LIVE TRIGGER GATES ──
    "MIN_RANGE_EXPANSION":          1.15,
    "MIN_VOLUME_EXPANSION_ATTEMPT": 1.20,
    "MIN_VOLUME_EXPANSION_CONFIRM": 1.25,
    "MIN_LIVE_POSITION_ATTEMPT":    0.60,
    "MIN_CLOSE_POSITION_CONFIRMED": 0.60,
    "HEALTHY_PENETRATION_MAX_ATR":  1.20,    # Up to 1.20 ATR penetration is healthy expansion
    "EXHAUSTION_RVOL_MIN":          1.75,    # Penetration > 1.20 ATR requires RVOL >= 1.75 to prove thrust
    "EXHAUSTION_CLOSE_POS_MIN":     0.75,    # Penetration > 1.20 ATR requires close in top 25%
    "VELOCITY_THRUST_ENVELOPE_MAX": 0.25,    # Velocity cap preventing parabolic blow-off wick
    "MAX_EXTENSION_15M_ATR":        1.20,    # Local cap: (Close−Res)/15m ATR <= 1.20
    "MAX_EXTENSION_DAILY_ATR":      0.75,    # Daily cap: (Close−Res)/Daily ATR <= 0.75
    "PULLBACK_RETEST_TOL_ATR":      0.15,
    "ENTRY_CUTOFF_TIME":            "14:15", # Normal session entry cutoff
    "LATE_ENTRY_CUTOFF_TIME":       "15:00", # Strict late-session hard stop
    "LATE_SESSION_MIN_BASE":        75,      # Stricter base in 14:15-15:00
    "LATE_SESSION_MIN_BREAKOUT":    75,      # Stricter breakout in 14:15-15:00
    "LATE_SESSION_MIN_RVOL":        1.50,    # Stricter volume in 14:15-15:00
    "LATE_SESSION_MIN_CONFLUENCE":  82,      # Stricter confluence in 14:15-15:00

    # ── PRE-BREAKOUT / IGNITION CONFIG ──
    "PRE_BREAKOUT_MAX_DISTANCE_ATR": 0.40,   # Within 0.40 ATR of resistance
    "PRE_BREAKOUT_MIN_BASE_SCORE":   75,     # High-quality base required for pre-breakout
    "PRE_BREAKOUT_MIN_IGNITION_SCORE": 75,   # Minimum ignition score (0-100)

    # ── VOLUME BASELINE ──
    "MIN_VOLUME_PROJECTION_FRAC":   0.25,
    "FIRST_CANDLE_SLOT":            "09:15",
    "FIRST_CANDLE_VOLUME_MULT":     0.80,
    "APPROACH_ATR_MULT":            0.10,
    "BREAKOUT_BUFFER_ATR_MULT":     0.10,
    "SLOT_BASELINE_SESSIONS":       10,

    # ── STATE MACHINE ──
    "MAX_ATTEMPT_BARS":             3,
    "ATTEMPT_RESET_ATR_MULT":       0.50,
    "FAILED_BREAKOUT_COOLDOWN_MIN": 30,

    # ── SOFT MARKET REGIME SHIELD ──
    "BEAR_MIN_TOTAL_SCORE":         80,      # In BEAR/CRASH: confluence score must be >= 80 (or core technical >= 70)
    "BEAR_MIN_BASE_SCORE":          75,      # In BEAR/CRASH: base score must be >= 75
    "BEAR_MIN_BREAKOUT_SCORE":      68,      # In BEAR/CRASH: breakout score must be >= 68
    "BEAR_MIN_RVOL":                1.30,    # In BEAR/CRASH: RVOL must be >= 1.30×
    "BEAR_MIN_CORE_TECHNICAL_SCORE": 70,     # Structure + Momentum + Volume quality floor (when 1H >= 0)

    # ── TRADE QUALITY ──
    "MIN_RR_RATIO":                 1.5,
}

LIVE_1H_CONFIG = {
    "MIN_SIGNALS":        3,
    "MIN_BODY_RATIO":     0.55,
    "MIN_CLOSE_POSITION": 0.65,
    "MAX_UPPER_WICK":     0.25,
    "MIN_VOLUME_RATIO":   2.0,
    "MIN_VOLUME_AVG":     100_000,
    "MIN_RSI":            55,
    "MAX_RSI":            86,
}

EOD_CONFIG = {
    "MIN_SIGNALS":        1,
    "MIN_BODY_RATIO":     0.40,
    "MIN_CLOSE_POSITION": 0.55,
    "MAX_UPPER_WICK":     0.35,
    "MIN_VOLUME_RATIO":   1.5,   # [v5.3.0 UPGRADE]: Breakout Volume >= 1.5x SMA20
    "MIN_VOLUME_AVG":     50_000,
    "MIN_RSI":            50,
    "MAX_RSI":            92,    # [FIX: RSI_CEILING_TO_PENALTY] Raised from 88→92. RSI 88-92 is now a graduated scoring penalty (-2.5 pts/unit), not a hard reject. Genuine breakout stocks routinely hit RSI 88-95 on the ignition day.
}

EOD_ADVANCED_CONFIG = {
    "MAX_DISTANCE_FROM_52W_HIGH_PCT": 5.0,  # [v5.3.0 UPGRADE]: Within 5.0% of 52W High
    "MAX_BASE_ATR10_PCT": 2.5,              # [v5.3.0 UPGRADE]: 10-day ATR <= 2.5% of Price (Base Tightness)
    "MAX_SINGLE_DAY_MOVE_PCT": 15.0,
    "MAX_GAP_FROM_PRIOR_HIGH_PCT": 3.0,
    "GAP_LOOKBACK_BARS": 10,

    # ── Sustainability & Breakout Conviction ──
    "MAX_EXTENDED_BREAKOUT_ATR_MULT": 1.5,
    "GAP_AND_GO_PENALTY_MULT": 10,
    "GAP_AND_GO_MAX_PENALTY": 20,
    "MIN_ATR_EXPANSION_RATIO": 0.8,
    "MIN_OBV_SLOPE": 0.0,

    # ── Prior Context & Tight Bases ──
    "PRE_BREAKOUT_LOOKBACK_BARS": 5,
    "MAX_PRE_BREAKOUT_RED_CANDLES": 3,
    "TIGHT_BASE_BB_WIDTH_PCTILE": 0.50,

    # ── [FIX] Structural Breakout Constraint Relaxation ──
    "MAX_BB_WIDTH_PCTILE": 0.80,

    # ── [FIX: TWO_MODE_52W] Mode B — Recovery Breakout path ──────────────
    # Stocks 5-15% below 52W high can qualify if they demonstrate stronger conviction:
    # higher volume, tighter base, and strong relative strength.
    "RECOVERY_BREAKOUT_MAX_DISTANCE_PCT": 15.0,    # outer limit for Mode B (5%-15%)
    "RECOVERY_BREAKOUT_MIN_VOL_RATIO":    2.5,     # must match MIN_BREAKOUT_VOLUME_RATIO
    "RECOVERY_BREAKOUT_MAX_BB_WIDTH":     0.50,    # tighter base required vs Mode A's 0.80
    "RECOVERY_BREAKOUT_MIN_RS_PCT":       60.0,    # RS percentile floor for recovery setups
    # Note: Mode B has no separate RSI ceiling. RSI is evaluated through the normal
    # penalty model (88-92 graduated, >92 hard reject). Adding a tighter Mode B RSI
    # ceiling would recreate the same contradictory logic removed from Mode A.
    "RECOVERY_BREAKOUT_SCORE_PENALTY":    5,       # flat score penalty applied to Mode B candidates
}

# [RULE 67 CHANGE-RATIONALE]:
# REVERSAL_CONFIG refactored to eliminate zero-alert starvation bottlenecks:
# 1. MIN_VOLUME_RATIO: Lowered from rigid 2.0x to 1.35x base floor (1.50x in STRONG_BEAR). 2.0x+ is rewarded via score bonus rather than serving as a binary kill-switch.
# 2. DEEP_VALUE_MIN_ROE: Set to 12.0% for Deep Value reversals (stocks below SMA200) to ensure fundamental solvency while allowing Quality reversals (above SMA200) at 5.0%.
# 3. REVERSAL_RSI_LOOKBACK & REVERSAL_MAX_TROUGH_AGE: Calibrated from 15 to 25 trading bars to allow legitimate consolidation bases to mature without premature trough expiry.
REVERSAL_CONFIG = {
    "MIN_DROP_FROM_52W_HIGH": 20.0,
    "MAX_DROP_FROM_52W_HIGH": 45.0,
    "RSI_CURL_MIN": 38.0,
    "RSI_OVERSOLD_THRESHOLD": 35.0,
    "MIN_RSI_RECOVERY": 8.0,
    "MIN_VOLUME_RATIO": 1.35,
    "STRONG_BEAR_MIN_VOLUME_RATIO": 1.50,
    "MIN_AVG_DAILY_VOLUME": 50_000,
    "MIN_STOCK_PRICE": 100.0,
    "MIN_ROE": 5.0,
    "DEEP_VALUE_MIN_ROE": 12.0,
    "MIN_YOY_REVENUE_GROWTH_FLOOR": -15.0,
    "MAX_DROP_BELOW_SMA200": 20.0,
    "REVERSAL_COOLDOWN_TRADING_DAYS": 40,
    "QUALITY_CAT_MIN_DROP": 15.0,
}

ALERT_COOLDOWN_MINUTES = {
    "WEALTH": 1440,       # 24 hours
    "MULTI_TF": 240,      # 4 hours
    "EOD": 1440,          # 24 hours
    "REVERSAL": 10080,    # 7 days
    "PULLBACK": 10080,    # 7 days
    "MULTIBAGGER": 43200  # 30 days
}

SCANNER_MAX_ALERTS = {
    "WEALTH": 40,    # = sum of bucket caps: Core(15) + Growth(10) + Opportunistic(10) + QOS(5)
    "MULTI_TF": 15,
    "EOD": 10,
    "REVERSAL": 10,
    "PULLBACK": 10,
    "MULTIBAGGER": 10,
}

# =====================================================================================
# SCANNER LOOKBACK & THRESHOLD CONSTANTS
# =====================================================================================

REVERSAL_RSI_LOOKBACK = 25
REVERSAL_MAX_TROUGH_AGE = 25

BB_WIDTH_PCTILE_LOOKBACK = 60

MULTI_TF_FETCH_BATCH_SIZE = 100

# =====================================================================================
# POSITION SIZING & RISK BUDGETING CONFIGURATION
# =====================================================================================
MAX_SL_DISTANCE_PCT = 8.0         # Max allowed stop loss distance % from entry
ACCOUNT_RISK_BUDGET_PCT = 1.0     # Max portfolio equity risk % per trade (Kelly / risk budget)
MAX_POSITION_PCT = 0.25

PULLBACK_CONFIG = {
    "VERSION": "pb-1.0.0",
    "LOOKBACK": 10, "CONFIRM": 2,
    "MIN_IMPULSE_GAIN_PCT": 5.0, "MIN_IMPULSE_ATR": 3.0, "MAX_IMPULSE_BARS": 20,
    "MIN_DEPTH_PCT": 10.0, "MAX_DEPTH_PCT": 78.6,
    "MIN_DURATION": 3, "MAX_DURATION": 20,
    "MAX_INTERNAL_SWINGS": 3, "MAX_PB_VOLUME_RATIO": 1.25,
    "TRIGGER_VOL_MULT": 1.1,
    "MIN_CLOSE_LOCATION": 0.55,
    "MIN_BODY_ATR": 0.35,
    "MAX_UPPER_WICK": 0.35, "MAX_ENTRY_GAP_PCT": 3.0,
    "MAX_BONUS": 5, "PRIOR_WINDOW": 30,
    "OUTAGE_THRESHOLD_BUMP": 3,
    "MIN_HISTORY": 180,
    "MODE": "LIVE", "DEBUG_SWINGS": False,
}

# Configurable Entry Deduplication Tolerance (% delta) for Canonical Alert Fingerprinting
SCANNER_DEDUP_ENTRY_TOLERANCE_PCT = {
    "PULLBACK": 0.5,       # 0.5% tolerance on swing pullback entries
    "EOD": 0.5,            # 0.5% tolerance on daily breakout entries
    "REVERSAL": 0.75,      # 0.75% tolerance on mean-reversion bounces
    "MULTI_TF": 0.3,       # 0.3% tighter tolerance on 5m intraday entries
    "MULTIBAGGER": 1.0,    # 1.0% wider tolerance on fundamental buy zones
    "DEFAULT": 0.5
}

# ── Data Quality Framework (V8.0) ──
QUALITY_VALIDATOR_VERSION = "V8.0"

QUALITY_SCORE_WEIGHTS = {
    "row_completeness": 40,
    "missing": 20,
    "price_sanity": 20,
    "continuity": 10,
    "freshness": 10,
}


# Configurable Score Bands for Advanced Outcome Analytics (Feature F-13)
SCORE_BANDS = [
    (70, 75),
    (75, 80),
    (80, 85),
    (85, 90),
    (90, 999),
]


# Maximum percentage of row loss accepted before logging a regression warning
MAX_HISTORY_SHRINK = 0.30


# Source reliability multipliers (0.0 to 1.0). Used for fallback evaluation.
SOURCE_RELIABILITY = {
    "NSE": 1.0,
    "Fyers": 1.0,
    "Cache": 0.95,
    "BSE": 0.70
}


# [FINDING-F FIX] Lowered ADX from 25 to 18. ADX 25+ indicates a trend that has
# already moved significantly. ADX 18-24 captures the accumulation/developing phase
# exactly where breakouts occur, while still filtering out choppy (ADX < 18) stocks.
ADX_MIN_THRESHOLD = 15
MIN_STOCK_PRICE = 100.0    # No penny stocks — matches daily_builder MIN_PRICE

# LIQUIDITY THRESHOLDS (in Rupees)
MIN_DAILY_LIQUIDITY_RUPEES_WATCHLIST = 150_000_000  # ₹15 Cr/day for raw watchlist
MIN_DAILY_LIQUIDITY_RUPEES_WEALTH    = 10_000_000   # ₹1 Cr/day for long-term wealth engine

DELIVERY_CONVICTION_THRESHOLDS = {
    "institutional": 60,
    "positional":    40,
    "moderate":      25,
    "intraday_churn": 0,
}

# [VERSION: RATE_LIMIT_PROTECTION_v1.0] Lower batch size from 150 to 40 to prevent YFinance HTTP 429 rate limit block
BATCH_DOWNLOAD_SIZE = 40
YAHOO_TIMEOUT = 30
PRICE_CACHE_TTL_SECONDS = 60  # Changed from 180s: Intraday runs every 5min (need fresh cache hit)


TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1002341999976")

# ─── MARKET DATA PLATFORM MIGRATION (FEATURE FLAGS) ─────────────────────────
# When True, routes fetches through the new HistoricalDataService (Upstox + Fyers)
# When False, uses legacy data_provider.py and price_provider.py (Yahoo Finance)
USE_MARKET_DATA_PLATFORM = os.environ.get("USE_MARKET_DATA_PLATFORM", "False").lower() == "true"
USE_UPSTOX_PROVIDER = os.environ.get("USE_UPSTOX_PROVIDER", "True").lower() == "true"
USE_FYERS_PROVIDER = os.environ.get("USE_FYERS_PROVIDER", "True").lower() == "true"

# ── PERFORMANCE ENGINEERING V1 ROADMAP FEATURE FLAGS ──
# Active by default across all environments (No environment variable dependency)
FEATURE_PARALLEL_SCANNERS_V1 = True
FEATURE_ASYNC_SYMBOL_PROBING_V1 = True
FEATURE_PROVIDER_LOCK_SPLIT_V1 = True
# [VERSION: PERF_THREAD_TUNE_v1.0] Raised from 4 → 8 to match server vCPU count.
# Wealth Engine, Daily Builder, and Pullback all cap to min(cpu_count, this) — 4 was leaving 4 cores idle.
SCAN_WORKER_THREADS = 8

UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN")
TELEGRAM_TIMEOUT = 10
LOG_LEVEL = "INFO"

# =====================================================================================
# ANTI-FAKE-BREAKOUT PARAMETERS
# =====================================================================================

# Minimum % above prior high for a valid breakout (timeframe-aware)
MIN_BREAKOUT_MARGIN = {
    "15m": 0.003,   # 0.3% above prior high
    "1h":  0.005,   # 0.5%
    "1d":  0.007,   # 0.7%
}

# Breakout candle volume must be at least this multiple of 20-bar avg
MIN_BREAKOUT_VOLUME_RATIO = 2.5

# Reject if N prior candles are ALL bearish (no momentum build-up)
# Moved to EOD_ADVANCED_CONFIG["MAX_PRE_BREAKOUT_RED_CANDLES"]

# BASE_WIDTH below this = tight consolidation = bonus-worthy setup
BASE_TIGHTNESS_THRESHOLD = 1.5

# BASE_WIDTH above this = volatile/choppy = penalize
BASE_VOLATILITY_THRESHOLD = 3.0

# =====================================================================================
# ANTI-OPERATOR-TRAP PARAMETERS
# =====================================================================================

# Bars to look back for climax top volume pattern
CLIMAX_VOLUME_LOOKBACK = 20

# Bars to look back for lower-high pattern (failed breakout retest)
LOWER_HIGH_LOOKBACK = 6

# Minimum candle range as % of price (below this = thin spread trap)
MIN_CANDLE_RANGE_PCT = 0.003   # 0.3%

# =====================================================================================
# SL/TARGET ATR CAPS (max target distance from entry, per timeframe)
# =====================================================================================

ADAPTIVE_TARGET_CAPS = {
    "STRONG_BULL": {"15m": 10.0, "1h": 12.0, "1d": 15.0},
    "WEAK_BULL":   {"15m": 7.0,  "1h": 9.0,  "1d": 11.0},
    "BULL":        {"15m": 8.0,  "1h": 10.0, "1d": 12.0},
    "BEAR":        {"15m": 4.0,  "1h": 6.0,  "1d": 8.0},
    "WEAK_BEAR":   {"15m": 5.0,  "1h": 7.0,  "1d": 9.0},
    "STRONG_BEAR": {"15m": 3.0,  "1h": 4.0,  "1d": 6.0},
    "SIDEWAYS":    {"15m": 5.0,  "1h": 7.0,  "1d": 9.0},
    "RANGEBOUND":  {"15m": 5.0,  "1h": 7.0,  "1d": 9.0},
    "NEUTRAL":     {"15m": 6.0,  "1h": 8.0,  "1d": 10.0}
}

# =====================================================================================
# V6.0 INSTITUTIONAL CONFIGURATION
# =====================================================================================

MIN_NATURAL_RR = {
    "MULTI_TF": 1.5,
    "EOD": 2.0,
    "REVERSAL": 2.0,
    "PULLBACK": 2.0,
}

# =====================================================================================
# LOCK CONTENTION TELEMETRY CONFIGURATION
# =====================================================================================
LOCK_WAIT_WARNING_SECONDS = float(os.environ.get("LOCK_WAIT_WARNING_SECONDS", "10.0"))
LOCK_HOLD_WARNING_SECONDS = float(os.environ.get("LOCK_HOLD_WARNING_SECONDS", "120.0"))

MAX_REASONABLE_RR = {
    "MULTI_TF": 6.0,
    "EOD": 8.0,
    "REVERSAL": 4.0,
    "PULLBACK": 8.0,
}

MIN_TARGET_CONFIDENCE = 40
TARGET_CONFIDENCE_BASELINE = {
    "version": "2026_Q3",
    "percentile": 95,
    "sample_size": 18000,
    "value": 85
}

# [FIX: DUPLICATE_CONFIG] This was a duplicate of the SCORE_THRESHOLDS defined at the top of this file.
# Removed to avoid confusion. The authoritative definition is at the top of config.py.

MIN_NATURAL_RR = {
    "MULTI_TF": 1.5,
    "EOD": 2.5,          # [v5.3.0 UPGRADE]: 2.5R Risk Multiple Target
    "REVERSAL": 2.0,
    "PULLBACK": 2.0,
}

MIN_REWARD_POTENTIAL = {
    "MULTI_TF": 1.5,
    "EOD": 2.5,          # [v5.3.0 UPGRADE]: 2.5R Target Multiple
    "REVERSAL": 1.8,
    "PULLBACK": 2.0,
}

MIN_STOP_PCT = {
    "MULTI_TF": 0.6,
    "EOD": 1.5,
    "REVERSAL": 2.0,
    "PULLBACK": 1.5,
}



TARGET_QUALITY_THRESHOLD = {
    "MULTI_TF": 55,
    "EOD":      55,
    "REVERSAL": 50,
    "PULLBACK": 55,
}



STRUCTURAL_RESISTANCE_SCORES = {
    "1H Swing High": 35,
    "30m Swing High": 30,
    "15m Swing High": 25,
    "Major Swing High": 40,
    "Swing High": 30,
    "Rolling Swing High": 20,
    "5m Swing High": 20,
    "R2": 20,
    "R1": 15,
}

STRUCTURAL_STOP = {
    "MAX_CLUSTER_WIDTH_ATR": 1.5,
    "DISASTER_BUFFER_PCT": 1.5,
    "SCORES": {
        "1H Swing Low": 35,
        "30m Swing Low": 30,
        "15m Swing Low": 25,
        "Swing Low Cluster": 40,
        "Swing Low": 30,
        "Rolling Swing Low": 25,
        "S1 (Discovery)": 20,
        "S1": 20,
        "SMA200": 30,
        "EMA20": 15,
        "SMA50": 15,
        "VWAP": 15,
        "Intraday Candle Low": 20
    },
    "BONUS_OVERLAP": 15,
    "USE_SUPPORT_CLUSTER": True
}

# =====================================================================================
# FALLBACK PRICE PROVIDER (when YFinance rate-limited)
# =====================================================================================

# ── DATA PROVIDER SETTINGS ──────────────────────────────────────────────────────────
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "fyers")  # fyers, yfinance, or kite

# [VERSION: V5_ACQUISITION_ROUTING_V1.0] Provider routing policy and capabilities configuration
ROUTING_POLICY_VERSION = 2

PROVIDER_ROUTING_POLICY = {
    "price_1d":  ["fyers", "upstox"],
    "price_1wk": ["fyers", "upstox"],
    "price_1mo": ["fyers", "upstox"],

    "price_1h":  ["fyers", "upstox"],
    "price_30m": ["fyers", "upstox"],
    "price_15m": ["fyers", "upstox"],
    "price_5m":  ["fyers", "upstox"],
    "price_1m":  ["fyers", "upstox"],

    # Fyers & Upstox for live quotes
    "live_quotes": ["fyers", "upstox"],

    "bhavcopy_delivery": ["nse_bhavcopy", "bse_bhavcopy"],
    "promoter_pledge":   ["bse_corporate", "nse_corporate"],
    "default": ["fyers", "upstox"]
}

PROVIDER_CAPABILITIES = {
    "yahoo": {
        "bulk": True,
        "live": False,
        "intraday": True,
        "historical": True
    },
    "fyers": {
        "bulk": False,
        "live": True,
        "intraday": True,
        "historical": True
    },
    "bse": {
        "bulk": True,
        "live": False,
        "intraday": False,
        "historical": True
    }
}

STAGE_PERFORMANCE_BUDGETS = {
    "download_seconds": 5.0,
    "fallback_seconds": 3.0,
    "validation_seconds": 2.0,
    "indicators_seconds": 15.0,
    "parquet_write_seconds": 2.0,
    "scanner_seconds": 10.0,
    "database_seconds": 2.0,
    "cleanup_seconds": 1.0,
    "total_scan_seconds": 60.0
}

# ── FYERS CONFIGURATION ──────────────────────────────────────────────────────────
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
FYERS_SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
FYERS_REDIRECT_URL = os.getenv("FYERS_REDIRECT_URL", os.getenv("FYERS_REDIRECT_URI", os.getenv("APP_URL", "https://elitebreakout.duckdns.org").rstrip("/") + "/fyers/callback"))
FYERS_TOKEN_PATH = os.path.join(DATA_DIR, "fyers_token.txt")


REGIME_POLICIES = {
    "STRONG_BULL": {
        "score_modifier": 0,
        "allow_mean_reversion": False,
        "max_new_positions_per_day": 5,
        "min_target_quality_override": 60,
        "min_reward_potential_mult": 1.5,
        "capital_allocation_mult": 1.0
    },
    "WEAK_BULL": {
        "score_modifier": 0,
        "allow_mean_reversion": True,
        "max_new_positions_per_day": 3,
        "min_target_quality_override": 65,
        "min_reward_potential_mult": 1.0,
        "capital_allocation_mult": 1.0
    },

    "BULL": {
        "score_modifier": 0,
        "allow_mean_reversion": True,
        "max_new_positions_per_day": 3,
        "min_target_quality_override": 65,
        "min_reward_potential_mult": 1.0,
        "capital_allocation_mult": 1.0
    },
    "BEAR": {
        "score_modifier": 5,  # [FIX: ALERT_GATE] Was 5 — correct, threshold → 80
        "allow_mean_reversion": True,
        "max_new_positions_per_day": 1,
        "min_target_quality_override": 80,
        "min_reward_potential_mult": 0.8,
        "capital_allocation_mult": 0.5
    },
    "SIDEWAYS": {
        "score_modifier": 3,  # [FIX: ALERT_GATE] Was +8 (threshold→83, near-impossible). Reduced to +3 (threshold→78). SIDEWAYS should still allow quality breakouts.
        "allow_mean_reversion": True,
        "max_new_positions_per_day": 2,
        "min_target_quality_override": 75,
        "min_reward_potential_mult": 0.8,
        "capital_allocation_mult": 0.5
    },
    "RANGEBOUND": {
        "score_modifier": 3,  # [FIX: ALERT_GATE] Was +8 (threshold→83). Reduced to +3 (threshold→78). Same reasoning as SIDEWAYS.
        "allow_mean_reversion": True,
        "max_new_positions_per_day": 2,
        "min_target_quality_override": 75,
        "min_reward_potential_mult": 0.8,
        "capital_allocation_mult": 0.5
    },
    "WEAK_BEAR": {
        "score_modifier": 5,  # [FIX: ALERT_GATE] Was +10 (threshold→85, effectively impossible). Reduced to +5 (threshold→80). Same as BEAR — proportionate.
        "allow_mean_reversion": True,
        "max_new_positions_per_day": 1,
        "min_target_quality_override": 80,
        "min_reward_potential_mult": 0.8,
        "capital_allocation_mult": 0.5
    },
    "STRONG_BEAR": {
        "score_modifier": 10,
        "allow_mean_reversion": False,
        "max_new_positions_per_day": 0,
        "min_target_quality_override": 100,
        "min_reward_potential_mult": 0.5,
        "capital_allocation_mult": 0.0
    },
    "NEUTRAL": {
        "score_modifier": 0,
        "allow_mean_reversion": True,
        "max_new_positions_per_day": 3,
        "min_target_quality_override": 65,
        "min_reward_potential_mult": 1.0,
        "capital_allocation_mult": 1.0
    }
}

# ── Target Engine v7 — FINAL FROZEN ──────────────────────────────────────────

# For Enum typing, though Enum is defined in sl_target_helper.
# We will use string representations here to avoid circular imports,
# or just redefine them if we need them, but it's better to keep strings in config
# and map them to enums in the helper.
# Actually, the spec says "TARGET_SOURCE_WEIGHTS = { TargetSource.EQUAL_HIGH: 10 ... }"
# To do this cleanly without circular import, we can define the enum here or in a separate file.
# The spec puts the Enum in sl_target_helper.py. So we'll use strings in config and the engine will map/handle.
# Let's use the string names matching the enum keys.

TARGET_SOURCE_WEIGHTS = {
    "EQUAL_HIGH":     10,
    "RESISTANCE":     10,
    "HIGH_20D":        9,
    "PREV_DAY_HIGH":   9,
    "HIGH_52W":        8,
    "ABCD":            9,
    "RETRACE_50":      8,
    "RETRACE_618":     7,
    "RETRACE_382":     6,
    "FIB_127":         7,
    "FIB_162":         6,
    "SMA200":          8,
    "BB_MID":          7,
    "SMA50":           6,
    "FIB_200":         5,
    "ATR_PROJ":        4,
    "R1":              5,
    "R2":              4,
    "ROUND_NUM":       0,
}

FIB_200_WEIGHTS = {
    "STRONG_BULL": 8, "WEAK_BULL": 6, "BULL": 7, "TRENDING": 7,
    "BEAR": 2, "WEAK_BEAR": 3, "STRONG_BEAR": 1,
    "SIDEWAYS": 4, "RANGEBOUND": 4, "NEUTRAL": 5
}

SOURCE_PRIORITY = {
    "EQUAL_HIGH":     1,
    "RESISTANCE":     2,
    "HIGH_20D":       3,
    "PREV_DAY_HIGH":  4,
    "HIGH_52W":       5,
    "ABCD":           6,
    "RETRACE_618":    7,
    "RETRACE_50":     8,
    "RETRACE_382":    9,
    "FIB_127":        10,
    "FIB_162":        11,
    "SMA200":         12,
    "SMA50":          13,
    "BB_MID":         14,
    "FIB_200":        15,
    "ATR_PROJ":       16,
    "R1":             17,
    "R2":             18,
    "ROUND_NUM":      99,
}

TARGET_CONFLICT_POLICY = {
    "EOD":      "REGIME",
    "MULTI_TF": "CONFIDENCE",
    "REVERSAL": "SECOND_NEAREST",
    "PULLBACK": "REGIME",
}

EXIT_PROFILES = {
    "CONSERVATIVE": {"t1": 25, "t2": 50, "t3": 25},
    "BALANCED":     {"t1": 30, "t2": 40, "t3": 30},
    "AGGRESSIVE":   {"t1": 20, "t2": 30, "t3": 50},
}

SCANNER_EXIT_PROFILE = {
    "EOD":      "BALANCED",
    "MULTI_TF": "AGGRESSIVE",
    "REVERSAL": "CONSERVATIVE",
    "PULLBACK": "BALANCED",
}

FIB_EXTENSIONS   = [1.272, 1.618, 2.0]
FIB_RETRACEMENTS = [0.382, 0.500, 0.618]
ABCD_BC_RETRACE_MIN = 0.382
ABCD_BC_RETRACE_MAX = 0.786
FIB_200_GATE     = {"min_adx": 30, "min_vol_ratio": 2.0, "require_above_vwap": True}

ROUND_NUMBER_BOOST      = 8
ROUND_NUMBER_PCT        = 0.005
TARGET_CLUSTER_WINDOW_ATR_FRAC = 0.5
TARGET_CLUSTER_WINDOW_PCT      = 0.0075

#           atr_base  sl_atr_buf  sl_pct_buf  max_sl_atr
_MODE_CONFIG = {
    "EOD":      (2.00,    0.80,       0.0075,     3.0),
    "MULTI_TF": (1.50,    0.50,       0.0050,     3.0),
    "REVERSAL": (2.00,    1.00,       0.0100,     3.5),
    "PULLBACK": (2.00,    0.75,       0.0075,     3.0),   # Pullback Continuation
    "MULTIBAGGER": (2.00, 1.00,       0.0100,     3.5),
}


SCANNER_MULTI_TF = "MULTI_TF"

# Scanner Telemetry
SCANNER_DECISION_LOGGING = True

# [RULE 67 CHANGE-RATIONALE]:
# Exposes get_regime_state() helper in config.py for backward compatibility.
# Delegates to MarketRegimeEngine.get_regime_context() / regime_engine with defensive fallback,
# resolving any 'cannot import name get_regime_state from config' errors cleanly.
def get_regime_state() -> dict:
    try:
        from macro_utils import MarketRegimeEngine
        raw_regime = MarketRegimeEngine.get_regime_context()
        trend = raw_regime.get("trend", "NORMAL") if isinstance(raw_regime, dict) else "NORMAL"
        return {"status": trend, **(raw_regime if isinstance(raw_regime, dict) else {})}
    except Exception:
        try:
            from regime_engine import get_market_regime
            raw_regime = get_market_regime()
            return {"status": raw_regime.get("market_regime", "NORMAL"), **(raw_regime if isinstance(raw_regime, dict) else {})}
        except Exception:
            return {"status": "NORMAL"}
