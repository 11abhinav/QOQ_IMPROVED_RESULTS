# =====================================================================================
# tests/test_indicator_equivalence.py
# RIGOROUS DIFFERENTIAL EQUIVALENCE TESTS FOR TECHNICAL INDICATORS
# =====================================================================================

import pytest
import numpy as np
import pandas as pd
from app.technical_indicators import apply_indicators, hydrate_indicators


def generate_ohlcv(rows: int = 200, freq: str = "15min") -> pd.DataFrame:
    """Generates synthetic trading day OHLCV data."""
    dates = pd.date_range("2026-01-05 09:15:00", periods=rows, freq=freq, tz="Asia/Kolkata")
    # Filter to weekday market hours
    dates = dates[dates.dayofweek < 5]
    if len(dates) < rows:
        dates = pd.date_range("2026-01-05 09:15:00", periods=rows * 2, freq=freq, tz="Asia/Kolkata")
        dates = dates[dates.dayofweek < 5][:rows]

    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    high = close + np.abs(np.random.randn(len(dates)) * 1.2) + 0.1
    low = close - np.abs(np.random.randn(len(dates)) * 1.2) - 0.1
    open_p = close + np.random.randn(len(dates)) * 0.3
    vol = np.random.randint(1000, 50000, size=len(dates))

    return pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol
    }, index=dates)


@pytest.mark.parametrize("timeframe,rows", [
    ("15m", 50),
    ("15m", 200),
    ("15m", 800),
    ("1d", 100),
    ("1d", 500),
    ("1d", 1500),
    ("1h", 150),
    ("5m", 150),
])
def test_full_hydration_equivalence(timeframe, rows):
    """Verifies that apply_indicators and hydrate_indicators(required=None) are bit-for-bit identical."""
    df = generate_ohlcv(rows=rows, freq="1D" if timeframe == "1d" else ("1h" if timeframe == "1h" else "15min"))
    
    df_old = apply_indicators(df.copy(), timeframe=timeframe)
    df_new = hydrate_indicators(df.copy(), required=None, timeframe=timeframe)

    assert set(df_old.columns) == set(df_new.columns), f"Column mismatch: {set(df_old.columns) ^ set(df_new.columns)}"

    for col in df_old.columns:
        s_old = df_old[col].values
        s_new = df_new[col].values

        if np.issubdtype(df_old[col].dtype, np.number):
            # Assert mathematical numerical equivalence with small float tolerance
            np.testing.assert_allclose(
                s_old.astype(float), 
                s_new.astype(float), 
                rtol=1e-5, 
                atol=1e-5, 
                equal_nan=True,
                err_msg=f"Numerical mismatch on column {col} for {timeframe} ({rows} rows)"
            )
        elif df_old[col].dtype == bool:
            assert np.array_equal(s_old, s_new), f"Boolean mismatch on column {col}"


@pytest.mark.parametrize("required_cols", [
    {"EMA9", "EMA20"},
    {"RSI", "ATR"},
    {"SWING_LOW", "SWING_HIGH"},
    {"HIGH_20D", "PRIOR_20D_HIGH"},
    {"BASE_WIDTH", "VCP_TIGHTENING"},
])
def test_targeted_hydration_subset(required_cols):
    """Verifies that hydrate_indicators(required={...}) produces exactly the required columns with identical values."""
    df = generate_ohlcv(rows=300, freq="15min")

    df_full = apply_indicators(df.copy(), timeframe="15m")
    df_subset = hydrate_indicators(df.copy(), required=required_cols, timeframe="15m")

    for col in required_cols:
        assert col in df_subset.columns, f"Requested column {col} missing in hydrated subset"
        np.testing.assert_allclose(
            df_full[col].values.astype(float),
            df_subset[col].values.astype(float),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
            err_msg=f"Targeted hydration value mismatch on column {col}"
        )

    # Verify that unrequested unrelated columns (e.g. MACD if not requested) are NOT computed
    unrelated_cols = {"MACD", "MACD_SIGNAL", "MACD_HIST", "PP", "R1", "S1"} - required_cols
    for u_col in unrelated_cols:
        assert u_col not in df_subset.columns, f"Unrequested column {u_col} was unnecessarily computed!"
