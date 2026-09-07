# =====================================================================================
# app/technical_indicators.py  (UPGRADED v4 - MODULAR HYDRATION ENGINE)
#
# CHANGES FROM v3:
#   1. Modularized into clean, self-contained sub-builders with zero changes to math.
#   2. Introduced `hydrate_indicators(df, required=None, ...)` to support targeted/
#      on-demand column calculation for scanners.
#   3. Maintained 100% mathematical signal-equivalence for `apply_indicators()`.
#   4. All weekend filtering, swing detection, pivot formulas, and rolling highs preserved.
# =====================================================================================

import pandas as pd
import ta
import numpy as np
import warnings
import logging
from typing import Optional, Set, Dict, Any

logger = logging.getLogger(__name__)

# Suppress annoying numpy nanmean warnings for empty slices in rolling windows
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

import gc
from memory_profiler import profile_function

try:
    import pandas_ta as ta
except ImportError:
    pass


def _find_swing_lows(low: pd.Series, n: int = 3) -> pd.Series:
    """
    Detect pivot swing lows: a bar where the low is lower than the `n` bars
    on either side. Returns the swing low price at those pivots, NaN elsewhere.
    Then forward-fills so every bar knows the most recent confirmed swing low.
    """
    window_min = low.rolling(window=2*n + 1, center=True, min_periods=2*n + 1).min()
    result = low.where(low == window_min, float("nan"))
    return result.ffill()


def _find_swing_highs(high: pd.Series, n: int = 3) -> pd.Series:
    """
    Detect pivot swing highs: a bar where the high is higher than the `n` bars
    on either side. Forward-fills the most recent confirmed swing high.
    """
    window_max = high.rolling(window=2*n + 1, center=True, min_periods=2*n + 1).max()
    result = high.where(high == window_max, float("nan"))
    return result.ffill()


# ─────────────────────────────────────────────────────────────────────────────
# MODULAR INDICATOR BUILDERS (Pure Functions / Dictionaries)
# ─────────────────────────────────────────────────────────────────────────────

def _build_trend_columns(close: pd.Series) -> Dict[str, pd.Series]:
    cols = {}
    cols["EMA9"]   = ta.trend.ema_indicator(close, window=9)
    cols["EMA20"]  = ta.trend.ema_indicator(close, window=20)
    cols["EMA50"]  = ta.trend.ema_indicator(close, window=50)
    cols["SMA50"]  = ta.trend.sma_indicator(close, window=50)
    cols["SMA200"] = ta.trend.sma_indicator(close, window=200)
    return cols


def _build_momentum_rsi(close: pd.Series) -> Dict[str, pd.Series]:
    return {"RSI": ta.momentum.rsi(close, window=14)}


def _build_volatility_columns(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
    cols = {}
    cols["ATR"]     = ta.volatility.average_true_range(high, low, close, window=14)
    cols["ATR20"]   = ta.volatility.average_true_range(high, low, close, window=20)
    cols["ATR_PCT"] = (cols["ATR"] / close * 100).round(2)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    cols["BB_UPPER"] = bb.bollinger_hband()
    cols["BB_LOWER"] = bb.bollinger_lband()
    cols["BB_MID"]   = bb.bollinger_mavg()
    cols["BB_WIDTH"] = (cols["BB_UPPER"] - cols["BB_LOWER"]) / close
    cols["BB_WIDTH_PCTILE"] = cols["BB_WIDTH"].rolling(window=100, min_periods=50).rank(pct=True)
    return cols


def _build_adx_columns(df_index: pd.Index, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
    cols = {}
    try:
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        cols["ADX"] = adx_ind.adx()
    except Exception:
        cols["ADX"] = pd.Series(np.nan, index=df_index)
    return cols


def _build_macd_columns(df_index: pd.Index, close: pd.Series) -> Dict[str, pd.Series]:
    cols = {}
    try:
        macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        cols["MACD"]        = macd_ind.macd()
        cols["MACD_SIGNAL"] = macd_ind.macd_signal()
        cols["MACD_HIST"]   = macd_ind.macd_diff()
    except Exception:
        cols["MACD"]        = pd.Series(np.nan, index=df_index)
        cols["MACD_SIGNAL"] = pd.Series(np.nan, index=df_index)
        cols["MACD_HIST"]   = pd.Series(np.nan, index=df_index)
    return cols


def _build_swing_columns(high: pd.Series, low: pd.Series, timeframe: str) -> Dict[str, pd.Series]:
    cols = {}
    pivot_n = {"1d": 5, "1h": 4, "30m": 4, "15m": 3, "5m": 3}.get(timeframe, 5)
    cols["SWING_LOW"]  = _find_swing_lows(low,  n=pivot_n)
    cols["SWING_HIGH"] = _find_swing_highs(high, n=pivot_n)

    swing_window = {"1d": 20, "1h": 14, "30m": 12, "15m": 10, "5m": 10}.get(timeframe, 20)
    cols["SWING_LOW_RAW"]  = low.rolling(window=swing_window,  min_periods=swing_window // 2).min()
    cols["SWING_HIGH_RAW"] = high.rolling(window=swing_window, min_periods=swing_window // 2).max()
    return cols


def _build_pivot_columns(high: pd.Series, low: pd.Series, close: pd.Series, timeframe: str, daily_ohlc: Optional[pd.DataFrame]) -> Dict[str, Any]:
    cols = {}
    if timeframe in ("1h", "30m", "15m", "5m") and daily_ohlc is not None and len(daily_ohlc) >= 2:
        last_daily = daily_ohlc.iloc[-1]
        d_high  = float(last_daily["High"])
        d_low   = float(last_daily["Low"])
        d_close = float(last_daily["Close"])

        pp = round((d_high + d_low + d_close) / 3, 2)
        cols["PP"] = pp
        cols["R1"] = round(2 * pp - d_low, 2)
        cols["R2"] = round(pp + (d_high - d_low), 2)
        cols["R3"] = round(d_high + 2 * (pp - d_low), 2)
        cols["S1"] = round(2 * pp - d_high, 2)
        cols["S2"] = round(pp - (d_high - d_low), 2)
        cols["S3"] = round(d_low - 2 * (d_high - pp), 2)
    else:
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)
        prev_close = close.shift(1)

        cols["PP"] = ((prev_high + prev_low + prev_close) / 3).round(2)
        cols["R1"] = (2 * cols["PP"] - prev_low).round(2)
        cols["R2"] = (cols["PP"] + (prev_high - prev_low)).round(2)
        cols["R3"] = (prev_high + 2 * (cols["PP"] - prev_low)).round(2)
        cols["S1"] = (2 * cols["PP"] - prev_high).round(2)
        cols["S2"] = (cols["PP"] - (prev_high - prev_low)).round(2)
        cols["S3"] = (prev_low - 2 * (prev_high - cols["PP"])).round(2)
    return cols


def _build_rolling_high_columns(df: pd.DataFrame, high: pd.Series, timeframe: str) -> Dict[str, pd.Series]:
    cols = {}
    n = len(df)

    if timeframe == "1d":
        cols["HIGH_20D"]  = high.rolling(window=20,  min_periods=15).max()
        cols["PREV_DAY_HIGH"] = high.shift(1)
        cols["PRIOR_20D_HIGH"] = cols["HIGH_20D"].shift(1)
        cols["HIGH_50D"]  = high.rolling(window=50,  min_periods=40).max()
        cols["HIGH_100D"] = high.rolling(window=100, min_periods=80).max()
        cols["HIGH_252D"] = high.rolling(window=252, min_periods=200).max()

    elif timeframe == "1h":
        try:
            raw_ts = df.index
            if not isinstance(raw_ts, pd.DatetimeIndex):
                datetime_col = next((c for c in ["Datetime", "Date", "index"] if c in df.columns), None)
                if datetime_col is not None:
                    raw_ts = pd.to_datetime(df[datetime_col])
                else:
                    raw_ts = pd.date_range(end=pd.Timestamp.now(tz="Asia/Kolkata"), periods=len(df), freq="1h")

            if not isinstance(raw_ts, pd.DatetimeIndex):
                raw_ts = pd.DatetimeIndex(raw_ts)

            if raw_ts.tz is None:
                ist_index = raw_ts.tz_localize('UTC').tz_convert('Asia/Kolkata')
            elif str(raw_ts.tz) != 'Asia/Kolkata':
                ist_index = raw_ts.tz_convert('Asia/Kolkata')
            else:
                ist_index = raw_ts

            from zoneinfo import ZoneInfo
            import datetime
            ist_now = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
            today_date = ist_now.date()

            past_mask = (ist_index.date < today_date)
            if past_mask.any():
                session_dates = ist_index[past_mask].date
                df_past = df[past_mask]
                session_counts = df_past.groupby(session_dates).size()

                daily_max_ts = pd.Series(ist_index[past_mask], index=df_past.index).groupby(session_dates).max()
                session_complete = daily_max_ts.apply(lambda ts: ts.hour > 14 or (ts.hour == 14 and ts.minute >= 15))

                valid_sessions = session_counts[(session_counts >= 4) & session_complete].index
                valid_mask = past_mask & np.isin(ist_index.date, valid_sessions)

                if valid_mask.any():
                    daily_highs = df.loc[valid_mask].groupby(ist_index[valid_mask].date)["High"].max().sort_index()
                    rolling_20d_high = daily_highs.rolling(window=20, min_periods=min(20, len(daily_highs))).max()
                    mapped_20d_high = pd.Series(ist_index.date, index=df.index).map(rolling_20d_high).values
                else:
                    mapped_20d_high = high.rolling(window=130, min_periods=20).max().values
            else:
                mapped_20d_high = high.rolling(window=130, min_periods=20).max().values
        except Exception as exc:
            logger.debug("Failed 1h session-aware 20d high mapping, using rolling fallback: %s", exc)
            mapped_20d_high = high.rolling(window=130, min_periods=20).max().values

        cols["HIGH_6H"]   = high.rolling(window=6,   min_periods=5).max()
        cols["HIGH_26H"]  = high.rolling(window=26,  min_periods=20).max()
        cols["HIGH_125H"] = high.rolling(window=125, min_periods=80).max()
        cols["HIGH_130H"] = high.rolling(window=130, min_periods=100).max()
        cols["HIGH_260H"] = high.rolling(window=260, min_periods=200).max()

        cols["HIGH_20D"]  = mapped_20d_high
        cols["PRIOR_20D_HIGH"] = mapped_20d_high

        cols["HIGH_50D"]  = cols["HIGH_130H"]
        cols["HIGH_100D"] = cols["HIGH_130H"]
        cols["HIGH_252D"] = cols["HIGH_260H"]

    else:  # 15m / 5m / other
        cols["HIGH_26_15M"]  = high.rolling(window=26,  min_periods=20).max()
        cols["HIGH_52_15M"]  = high.rolling(window=52,  min_periods=40).max()
        cols["HIGH_104_15M"] = high.rolling(window=104, min_periods=80).max()
        cols["HIGH_20D"]  = cols["HIGH_26_15M"]
        cols["PRIOR_20D_HIGH"] = cols["HIGH_20D"].shift(1)
        cols["HIGH_50D"]  = cols["HIGH_104_15M"]
        cols["HIGH_100D"] = cols["HIGH_104_15M"]
        cols["HIGH_252D"] = high.rolling(window=n, min_periods=n // 2).max()

    # 52-week high — timeframe-aware
    if timeframe == "1d":
        window52, min52 = 252, 200
    elif timeframe == "1h":
        window52, min52 = n, max(n // 2, 50)
    else:
        window52, min52 = n, max(n // 2, 20)

    cols["HIGH_52W"] = high.rolling(window=window52, min_periods=min52).max()
    return cols


def _build_vwap_columns(df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series, timeframe: str) -> Dict[str, pd.Series]:
    cols = {}
    if "Volume" in df.columns:
        typical_price = (high + low + close) / 3
        df_index = df.index
        if not isinstance(df_index, pd.DatetimeIndex):
            datetime_col = next((c for c in ["Datetime", "Date", "index"] if c in df.columns), None)
            if datetime_col is not None:
                df_index = pd.to_datetime(df[datetime_col])

        if timeframe in ("15m", "1h") and hasattr(df_index, 'date'):
            date_groups = df_index.date
            cum_tp_vol  = (typical_price * df["Volume"]).groupby(date_groups).cumsum()
            cum_vol     = df["Volume"].groupby(date_groups).cumsum()
            cols["VWAP"]  = (cum_tp_vol / cum_vol).where(cum_vol > 0)
        else:
            cum_tp_vol  = (typical_price * df["Volume"]).cumsum()
            cum_vol     = df["Volume"].cumsum()
            cols["VWAP"]  = (cum_tp_vol / cum_vol).where(cum_vol > 0)
    else:
        cols["VWAP"] = float("nan")
    return cols


def _build_obv_columns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    cols = {}
    if "Volume" in df.columns and len(df) >= 50:
        obv_direction = np.sign(df["Close"].diff())
        obv_direction.iloc[0] = 0

        raw_obv = (obv_direction * df["Volume"]).cumsum()
        cols["OBV_20MA"] = raw_obv.rolling(window=20, min_periods=20).mean()
        cols["OBV"] = raw_obv
        cols["OBV_SLOPE"] = raw_obv.diff(3)

        rolling_obv = (obv_direction * df["Volume"]).rolling(50, min_periods=50).sum()
        obv_slope = rolling_obv.diff(5)

        conds = [obv_slope > 0, obv_slope < 0]
        choices = [1, -1]
        cols["OBV_TREND"] = pd.Series(np.select(conds, choices, default=0), index=df.index)
    else:
        cols["OBV_TREND"] = 0
        cols["OBV_20MA"] = float("nan")
        cols["OBV"] = float("nan")
    return cols


def _build_pattern_columns(df: pd.DataFrame, high: pd.Series, low: pd.Series, atr_series: Optional[pd.Series], atr_pct_series: Optional[pd.Series]) -> Dict[str, Any]:
    cols = {}
    base_window = 10
    if len(df) >= base_window and atr_series is not None:
        rolling_range = (
            high.rolling(window=base_window, min_periods=base_window).max()
            - low.rolling(window=base_window, min_periods=base_window).min()
        )
        cols["BASE_WIDTH"] = (rolling_range / atr_series).where(atr_series > 0)
    else:
        cols["BASE_WIDTH"] = float("nan")

    if len(df) >= 60 and atr_pct_series is not None:
        vol_20d = atr_pct_series.rolling(window=20).mean()
        vol_60d = atr_pct_series.rolling(window=60).mean()
        cols["VCP_TIGHTENING"] = vol_20d < vol_60d
    else:
        cols["VCP_TIGHTENING"] = False
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# CORE API: hydrate_indicators & apply_indicators
# ─────────────────────────────────────────────────────────────────────────────

def hydrate_indicators(
    df: pd.DataFrame, 
    required: Optional[Set[str]] = None, 
    timeframe: str = "1d", 
    daily_ohlc: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Applies requested technical indicators to the DataFrame.
    If `required` is None or empty, computes the full 35+ indicator suite.
    If `required` is a set of column names, computes ONLY the needed sub-builders.
    
    [RULE 67 CHANGE-RATIONALE: MODULAR_TARGETED_HYDRATION_v1.0]
    Decouples heavy indicator calculation so scanners only compute columns they consume,
    drastically reducing CPU starvation on cold-cache restarts while preserving 100%
    mathematical signal equivalence.
    """
    from trading_calendar import enforce_trading_day_candles
    df = enforce_trading_day_candles(df)
    if daily_ohlc is not None:
        daily_ohlc = enforce_trading_day_candles(daily_ohlc)

    if df is None or df.empty or len(df) < 20:
        return df

    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    new_cols: Dict[str, Any] = {}

    req_all = (required is None or len(required) == 0)

    # 1. Trend Filters (EMA9, EMA20, EMA50, SMA50, SMA200)
    if req_all or required.intersection({"EMA9", "EMA20", "EMA50", "SMA50", "SMA200"}):
        new_cols.update(_build_trend_columns(close))

    # 2. Momentum RSI (RSI)
    if req_all or "RSI" in required:
        new_cols.update(_build_momentum_rsi(close))

    # 3. Volatility (ATR, ATR20, ATR_PCT, Bollinger Bands)
    vol_needed = (
        req_all 
        or required.intersection({"ATR", "ATR20", "ATR_PCT", "BB_UPPER", "BB_LOWER", "BB_MID", "BB_WIDTH", "BB_WIDTH_PCTILE"})
        or "BASE_WIDTH" in required
        or "VCP_TIGHTENING" in required
    )
    if vol_needed:
        new_cols.update(_build_volatility_columns(high, low, close))

    # 4. ADX Directional
    if req_all or "ADX" in required:
        new_cols.update(_build_adx_columns(df.index, high, low, close))

    # 5. MACD Momentum
    if req_all or required.intersection({"MACD", "MACD_SIGNAL", "MACD_HIST"}):
        new_cols.update(_build_macd_columns(df.index, close))

    # 6. True Pivot Swing Points
    if req_all or required.intersection({"SWING_LOW", "SWING_HIGH", "SWING_LOW_RAW", "SWING_HIGH_RAW"}):
        new_cols.update(_build_swing_columns(high, low, timeframe))

    # 7. Classic Pivot Points
    if req_all or required.intersection({"PP", "R1", "R2", "R3", "S1", "S2", "S3"}):
        new_cols.update(_build_pivot_columns(high, low, close, timeframe, daily_ohlc))

    # 8. Rolling Window Highs
    rolling_needed = (
        req_all
        or required.intersection({
            "HIGH_20D", "PREV_DAY_HIGH", "PRIOR_20D_HIGH", "HIGH_50D", "HIGH_100D", "HIGH_252D", "HIGH_52W",
            "HIGH_6H", "HIGH_26H", "HIGH_125H", "HIGH_130H", "HIGH_260H",
            "HIGH_26_15M", "HIGH_52_15M", "HIGH_104_15M"
        })
    )
    if rolling_needed:
        new_cols.update(_build_rolling_high_columns(df, high, timeframe))

    # 9. VWAP
    if req_all or "VWAP" in required:
        new_cols.update(_build_vwap_columns(df, high, low, close, timeframe))

    # 10. OBV Volume Dynamics
    if req_all or required.intersection({"OBV", "OBV_20MA", "OBV_SLOPE", "OBV_TREND"}):
        new_cols.update(_build_obv_columns(df))

    # 11. Pattern Features (BASE_WIDTH, VCP_TIGHTENING)
    pattern_needed = req_all or required.intersection({"BASE_WIDTH", "VCP_TIGHTENING"})
    if pattern_needed:
        atr_s = new_cols.get("ATR") if "ATR" in new_cols else df.get("ATR")
        atr_pct_s = new_cols.get("ATR_PCT") if "ATR_PCT" in new_cols else df.get("ATR_PCT")
        new_cols.update(_build_pattern_columns(df, high, low, atr_s, atr_pct_s))

    if new_cols:
        df = df.assign(**new_cols)
    return df


def apply_indicators(df: pd.DataFrame, timeframe: str = "1d", daily_ohlc: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies all technical indicators and returns the enriched DataFrame.
    100% backward-compatible wrapper calling `hydrate_indicators(df, required=None)`.
    """
    return hydrate_indicators(df, required=None, timeframe=timeframe, daily_ohlc=daily_ohlc)
