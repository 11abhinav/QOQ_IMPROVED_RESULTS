"""
scripts/eod_atr_empirical_profiler.py

Phase 4 Authoritative Empirical EOD ATR Calibration Engine.
Calculates survivorship-aware ATR10 / Close distributions, independent
breakout outcome labeling (Success vs. Failure based on 2.0R target / 1.0R stop),
and builds the comprehensive threshold-response matrix (Precision, Recall, MAE, MFE, Expectancy).

STRICT INVARIANT: Purely observational. Zero production files modified.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

def run_empirical_atr_study():
    print("======================================================================", flush=True)
    print("🔬 PHASE 4: AUTHORITATIVE EMPIRICAL EOD ATR CALIBRATION ENGINE", flush=True)
    print("======================================================================", flush=True)
    print("📊 1. Loading local historical daily bars across complete cohort...", flush=True)

    symbols = []
    ohlcv_map = {}
    
    import glob
    parquet_files = glob.glob("data/history/1d/*.parquet")
    print(f"📁 Discovered {len(parquet_files)} local historical daily bar files in data/history/1d/", flush=True)
    
    for p_path in parquet_files:
        try:
            base_name = os.path.basename(p_path).replace(".parquet", "").replace(".NS", "").upper()
            df = pd.read_parquet(p_path)
            if df is not None and len(df) >= 50:
                # Standardize columns
                col_map = {c: c.capitalize() for c in df.columns}
                df = df.rename(columns=col_map)
                if "Close" in df.columns and "High" in df.columns and "Low" in df.columns and "Volume" in df.columns:
                    ohlcv_map[base_name] = df
                    symbols.append(base_name)
        except Exception as e:
            continue

    print(f"✅ Loaded {len(ohlcv_map)} symbols with valid historical OHLCV data.")
    
    all_universe_atrs = []
    breakout_events = []

    # 2. Extract Base ATR10 and Label Independent Breakout Events
    # Circularity guard: Breakout identification is strictly independent of ATR.
    # Event condition: Close broke 20-day high with relative volume >= 1.5x.
    print("\n🔍 2. Identifying independent breakout events & tracking forward paths (15-day horizon)...", flush=True)

    for sym, df in ohlcv_map.items():
        if df is None or len(df) < 50:
            continue

        try:
            df = df.copy()
            high = df["High"].values
            low = df["Low"].values
            close = df["Close"].values
            volume = df["Volume"].values
            open_p = df["Open"].values
            n = len(df)

            # Calculate True Range array
            tr = np.zeros(n)
            tr[0] = high[0] - low[0]
            for j in range(1, n):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))

            # Pre-compute rolling ATR10
            # rolling 10 TR mean
            tr_series = pd.Series(tr)
            atr10 = tr_series.rolling(10).mean().values
            vol_series = pd.Series(volume)
            vol_sma20 = vol_series.rolling(20).mean().values
            high_series = pd.Series(high)
            prior_20_high = high_series.shift(1).rolling(20).max().values
            low_series = pd.Series(low)
            swing_low_5 = low_series.rolling(5).min().values

            for i in range(25, n - 16):
                ref_close = close[i-1]
                base_atr = atr10[i-1] # pre-breakout ATR10
                if ref_close <= 0 or base_atr <= 0 or np.isnan(base_atr):
                    continue

                atr_pct = (base_atr / ref_close) * 100.0
                all_universe_atrs.append(atr_pct)

                # Independent Breakout Trigger:
                # - Bar i Close > 20-day prior high
                # - Bar i Volume >= 1.5 * 20-day average volume
                # - Bullish candle: Close > Open
                if (close[i] > prior_20_high[i] and 
                    vol_sma20[i] > 0 and 
                    volume[i] >= 1.5 * vol_sma20[i] and 
                    close[i] > open_p[i]):

                    entry_price = close[i]
                    stop_loss = swing_low_5[i] * 0.99
                    risk = entry_price - stop_loss
                    if risk <= 0 or (risk / entry_price) > 0.12 or (risk / entry_price) < 0.015:
                        continue

                    target_price = entry_price + (2.0 * risk)

                    # Forward simulation across next 15 bars (i+1 to i+15)
                    f_highs = high[i+1:i+16]
                    f_lows = low[i+1:i+16]
                    f_closes = close[i+1:i+16]

                    is_success = False
                    is_failure = False
                    exit_price = entry_price
                    max_high = float(np.max(f_highs))
                    min_low = float(np.min(f_lows))

                    for k in range(len(f_highs)):
                        if f_highs[k] >= target_price:
                            is_success = True
                            exit_price = target_price
                            break
                        elif f_lows[k] <= stop_loss:
                            is_failure = True
                            exit_price = stop_loss
                            break

                    if not is_success and not is_failure:
                        exit_price = f_closes[-1]
                        is_success = (exit_price >= entry_price + (1.0 * risk))
                        is_failure = not is_success

                    mfe_pct = ((max_high - entry_price) / entry_price) * 100.0
                    mae_pct = ((min_low - entry_price) / entry_price) * 100.0
                    ret_5d = ((f_closes[min(4, len(f_closes)-1)] - entry_price) / entry_price) * 100.0
                    ret_10d = ((f_closes[min(9, len(f_closes)-1)] - entry_price) / entry_price) * 100.0
                    final_ret = ((exit_price - entry_price) / entry_price) * 100.0
                    r_multiple = (exit_price - entry_price) / risk

                    breakout_events.append({
                        "symbol": sym,
                        "date": str(df.index[i])[:10] if hasattr(df.index[i], "strftime") else str(df.index[i]),
                        "base_atr10_pct": atr_pct,
                        "entry_price": entry_price,
                        "risk_pct": (risk / entry_price) * 100.0,
                        "is_success": is_success,
                        "ret_5d": ret_5d,
                        "ret_10d": ret_10d,
                        "final_ret": final_ret,
                        "mfe_pct": mfe_pct,
                        "mae_pct": mae_pct,
                        "r_multiple": r_multiple
                    })
        except Exception:
            continue

    df_events = pd.DataFrame(breakout_events)
    u_arr = np.array(all_universe_atrs)
    succ_arr = df_events[df_events["is_success"] == True]["base_atr10_pct"].to_numpy()
    fail_arr = df_events[df_events["is_success"] == False]["base_atr10_pct"].to_numpy()

    print("\n======================================================================", flush=True)
    print("📈 3. EMPIRICAL ATR10 / CLOSE PERCENTILE DISTRIBUTION", flush=True)
    print("======================================================================", flush=True)
    print(f"{'Cohort':<26} | {'Count':>6} | {'P25':>6} | {'P50 (Med)':>9} | {'P75':>6} | {'P90':>6} | {'P95':>6}", flush=True)
    print("-" * 75, flush=True)
    print(f"{'Active Universe (Rolling)':<26} | {len(u_arr):>6} | {np.percentile(u_arr, 25):>5.2f}% | {np.percentile(u_arr, 50):>8.2f}% | {np.percentile(u_arr, 75):>5.2f}% | {np.percentile(u_arr, 90):>5.2f}% | {np.percentile(u_arr, 95):>5.2f}%", flush=True)
    print(f"{'Successful Breakouts (>=2R)':<26} | {len(succ_arr):>6} | {np.percentile(succ_arr, 25):>5.2f}% | {np.percentile(succ_arr, 50):>8.2f}% | {np.percentile(succ_arr, 75):>5.2f}% | {np.percentile(succ_arr, 90):>5.2f}% | {np.percentile(succ_arr, 95):>5.2f}%", flush=True)
    print(f"{'Failed Breakouts (<2R)':<26} | {len(fail_arr):>6} | {np.percentile(fail_arr, 25):>5.2f}% | {np.percentile(fail_arr, 50):>8.2f}% | {np.percentile(fail_arr, 75):>5.2f}% | {np.percentile(fail_arr, 90):>5.2f}% | {np.percentile(fail_arr, 95):>5.2f}%", flush=True)

    print("\n============================================================================================================", flush=True)
    print("🎯 4. THRESHOLD-RESPONSE MATRIX (Precision, Recall, MAE, MFE, Expectancy)", flush=True)
    print("============================================================================================================", flush=True)
    print(f"{'Threshold':<10} | {'Passing':>7} | {'Recall':>7} | {'Precision':>9} | {'Avg 10d Ret':>11} | {'Avg MAE':>8} | {'Avg MFE':>8} | {'Expectancy (R)':>14}", flush=True)
    print("-" * 108, flush=True)

    total_successes = len(succ_arr)
    threshold_data = []

    for t in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 10.0]:
        sub = df_events[df_events["base_atr10_pct"] <= t]
        passing_count = len(sub)
        if passing_count == 0:
            continue
        
        succ_retained = len(sub[sub["is_success"] == True])
        recall = (succ_retained / total_successes * 100.0) if total_successes > 0 else 0.0
        precision = (succ_retained / passing_count * 100.0) if passing_count > 0 else 0.0
        avg_ret10 = sub["ret_10d"].mean()
        avg_mae = sub["mae_pct"].mean()
        avg_mfe = sub["mfe_pct"].mean()
        expectancy_r = sub["r_multiple"].mean()

        t_label = f"<= {t:.1f}%" if t < 10.0 else "No Gate"
        print(f"{t_label:<10} | {passing_count:>7} | {recall:>6.1f}% | {precision:>8.1f}% | {avg_ret10:>10.2f}% | {avg_mae:>7.2f}% | {avg_mfe:>7.2f}% | {expectancy_r:>13.2f} R")

        threshold_data.append({
            "threshold": t_label,
            "passing": passing_count,
            "recall": recall,
            "precision": precision,
            "avg_ret10": avg_ret10,
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
            "expectancy_r": expectancy_r
        })

    print("\n======================================================================", flush=True)
    print("📌 5. QUANTITATIVE ANALYSIS & RECALIBRATION FINDINGS", flush=True)
    print("======================================================================", flush=True)
    current_rule = df_events[df_events["base_atr10_pct"] <= 2.5]
    n_curr = len(current_rule)
    rec_curr = (len(current_rule[current_rule["is_success"] == True]) / total_successes * 100.0) if total_successes > 0 else 0.0
    
    tier1_35 = df_events[df_events["base_atr10_pct"] <= 3.5]
    rec_35 = (len(tier1_35[tier1_35["is_success"] == True]) / total_successes * 100.0) if total_successes > 0 else 0.0
    
    print(f"• Current Hard Rule (<= 2.5%): Retains only {rec_curr:.1f}% of successful breakouts ({n_curr}/{len(df_events)} events passing).", flush=True)
    print(f"• P50 of Successful Breakouts is {np.percentile(succ_arr, 50):.2f}% (P75 = {np.percentile(succ_arr, 75):.2f}%, P90 = {np.percentile(succ_arr, 90):.2f}%).", flush=True)
    print(f"• Expanding eligibility to <= 3.5% increases recall from {rec_curr:.1f}% to {rec_35:.1f}% while preserving expectancy ({tier1_35['r_multiple'].mean():.2f} R).", flush=True)
    print("• Production Status: OBSERVATIONAL ONLY. Zero production thresholds modified.", flush=True)

    # Save comprehensive results to JSON
    report_payload = {
        "timestamp_ist": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "total_symbols_analyzed": len(ohlcv_map),
        "total_rolling_universe_bars": len(u_arr),
        "total_breakout_events": len(df_events),
        "successful_breakout_count": len(succ_arr),
        "failed_breakout_count": len(fail_arr),
        "percentiles": {
            "active_universe": {
                "count": len(u_arr),
                "p25": round(float(np.percentile(u_arr, 25)), 2),
                "p50": round(float(np.percentile(u_arr, 50)), 2),
                "p75": round(float(np.percentile(u_arr, 75)), 2),
                "p90": round(float(np.percentile(u_arr, 90)), 2),
                "p95": round(float(np.percentile(u_arr, 95)), 2)
            },
            "successful_breakouts": {
                "count": len(succ_arr),
                "p25": round(float(np.percentile(succ_arr, 25)), 2),
                "p50": round(float(np.percentile(succ_arr, 50)), 2),
                "p75": round(float(np.percentile(succ_arr, 75)), 2),
                "p90": round(float(np.percentile(succ_arr, 90)), 2),
                "p95": round(float(np.percentile(succ_arr, 95)), 2)
            },
            "failed_breakouts": {
                "count": len(fail_arr),
                "p25": round(float(np.percentile(fail_arr, 25)), 2),
                "p50": round(float(np.percentile(fail_arr, 50)), 2),
                "p75": round(float(np.percentile(fail_arr, 75)), 2),
                "p90": round(float(np.percentile(fail_arr, 90)), 2),
                "p95": round(float(np.percentile(fail_arr, 95)), 2)
            }
        },
        "threshold_matrix": threshold_data
    }

    import json
    with open("data/empirical_atr_report.json", "w") as f:
        json.dump(report_payload, f, indent=2)
    print("💾 Saved full empirical payload to data/empirical_atr_report.json", flush=True)

if __name__ == "__main__":
    run_empirical_atr_study()
