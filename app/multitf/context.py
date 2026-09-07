# =====================================================================================
# app/multitf/context.py
# MULTI_TF V2 — Higher Timeframe Context Evaluation
#
# Responsibility: Grades the environment the setup is forming in.
# Context does NOT veto a setup; it only scales the final Confluence Score.
#
# Layers:
#   1. 1H Regime: Bullish/Neutral/Bearish alignment of EMA9, EMA20, SMA50, SMA200.
#   2. 30m Structure: Is there clear "room to move" above the consolidation box?
#   3. Market: Relative strength vs Nifty, general market regime.
# =====================================================================================

import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger("multitf.context")

def evaluate_1h_context(df_1h: Optional[pd.DataFrame], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates 1H directional alignment. Returns score and label.
    Expected to be extremely fast as indicators are already computed in technical_indicators.py.
    """
    if df_1h is None or df_1h.empty:
        return {"score": 0, "label": "NO_DATA"}

    try:
        # [RULE 67 CHANGE-RATIONALE: TARGETED_1H_CONTEXT_HYDRATION_v1.0]
        # Hydrate trend indicators on-demand if missing in raw 1h OHLCV DataFrame
        if "EMA9" not in df_1h.columns and "EMA_9" not in df_1h.columns and len(df_1h) >= 20:
            from technical_indicators import hydrate_indicators
            df_1h = hydrate_indicators(df_1h, required={"EMA9", "EMA20", "SMA50", "SMA200"}, timeframe="1h")

        last = df_1h.iloc[-1]
        c = last["Close"]
        e9 = last.get("EMA9") if last.get("EMA9") is not None else last.get("EMA_9")
        e20 = last.get("EMA20") if last.get("EMA20") is not None else last.get("EMA_20")
        s50 = last.get("SMA50") if last.get("SMA50") is not None else last.get("SMA_50")
        s200 = last.get("SMA200") if last.get("SMA200") is not None else last.get("SMA_200")

        # Fallbacks if indicators are missing
        if any(x is None or pd.isna(x) for x in (e9, e20, s50)):
            return {"score": 0, "label": "NEUTRAL"}

        is_bullish = (e9 > e20) and (e20 > s50)
        above_200 = (s200 is not None) and (not pd.isna(s200)) and (c > s200)
        above_50 = (c > s50)
        below_20 = (c < e20)

        if is_bullish and above_200:
            return {"score": config.get("H1_BULLISH_SCORE", 10), "label": "BULLISH"}
        elif below_20 or not above_50:
            return {"score": config.get("H1_BEARISH_SCORE", -10), "label": "BEARISH"}
        else:
            return {"score": config.get("H1_NEUTRAL_SCORE", 0), "label": "NEUTRAL"}

    except Exception as exc:
        logger.warning("[1h_context] Evaluation failed: %s", exc)
        return {"score": 0, "label": "ERROR"}


def evaluate_30m_context(df_30m: Optional[pd.DataFrame], box_high: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates immediate overhead structure on 30m.
    Checks if there is clear room before the next major resistance.
    """
    if df_30m is None or df_30m.empty or not box_high:
        return {"score": 0, "room_pct": 0.0, "has_room": False}

    try:
        last = df_30m.iloc[-1]
        
        # Look for the nearest overhead resistance level (Swing Highs, Daily Pivots, 52W High)
        levels = []
        for col in ["LOOKBACK_SWING_HIGH", "R1", "R2", "HIGH_252D"]:
            if col in last and not pd.isna(last[col]) and last[col] > box_high:
                levels.append(last[col])

        if not levels:
            # Blue sky
            return {"score": config.get("M30_ROOM_SCORE", 10), "room_pct": 999.9, "has_room": True}

        nearest_res = min(levels)
        room_pct = (nearest_res - box_high) / box_high
        threshold = config.get("M30_ROOM_THRESHOLD_PCT", 0.02)

        has_room = room_pct >= threshold
        score = config.get("M30_ROOM_SCORE", 10) if has_room else 0

        return {
            "score": score,
            "room_pct": round(room_pct, 4),
            "has_room": has_room,
            "nearest_res": nearest_res
        }

    except Exception as exc:
        logger.warning("[30m_context] Evaluation failed: %s", exc)
        return {"score": 0, "room_pct": 0.0, "has_room": False}


def evaluate_market_context(
    regime_ctx: Dict[str, Any],
    symbol: str,
    df_5m: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Injects global market regime (Risk-On/Risk-Off) and symbol Relative Strength.
    """
    # For MVP V2, we pass through the global regime state without heavy RS math.
    # Confluence engine will apply weight based on regime string.
    regime = regime_ctx.get("status", "NORMAL")
    if regime == "BEARISH":
        score = -10
    elif regime == "BULLISH":
        score = 10
    else:
        score = 0

    return {
        "regime": regime,
        "score": score
    }
