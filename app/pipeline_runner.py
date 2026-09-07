import pandas as pd
from typing import Optional, Dict, Any

from core.events import (
    EventPublisher,
    ValidationCompleted,
    IndicatorsCalculated,
    ScannerCompleted,
    ScoresCalculated,
    CandidateSelected,
    SLTargetComputed,
    AlertCreated,
    PipelineCompleted
)

from technical_indicators import apply_indicators
from breakout_engine import detect_breakouts
from scoring_engine import calculate_score
from sl_target_helper import compute_sl_and_target
from config import (
    EOD_CONFIG, EOD_ADVANCED_CONFIG, MIN_STOCK_PRICE, SCORE_THRESHOLDS,
    MIN_BREAKOUT_MARGIN, MIN_BREAKOUT_VOLUME_RATIO, BASE_TIGHTNESS_THRESHOLD,
    ADX_MIN_THRESHOLD, ACTIVE_ALGO_VERSION
)

def _safe_float(val, default=0.0):
    try:
        import pandas as pd
        if val is None or pd.isna(val) or val == "":
            return default
        return float(val)
    except Exception:
        return default

class PipelineRunner:
    """
    Orchestrates the core business logic of the ELITE Breakout System in a purely deterministic way.
    It has no knowledge of infrastructure, databases, scheduling, or snapshot testing.
    It simply executes the pipeline stages and emits domain events.
    """
    
    @classmethod
    def execute(
        cls,
        symbol: str,
        category: str,
        sector: str,
        ticker: pd.DataFrame,
        delivery_pct: Optional[float],
        pledge_pct: Optional[float],
        nifty_ret_20d: float,
        regime_ctx: dict,
        bayesian_weights: Optional[dict],
        bayesian_version: str,
        publisher: EventPublisher
    ) -> None:
        from memory_profiler import MemoryProfiler
        status = "REJECTED"
        try:
            with MemoryProfiler(f"Pipeline: {symbol}"):
                status = cls._execute_internal(
                    symbol, category, sector, ticker, delivery_pct, pledge_pct,
                    nifty_ret_20d, regime_ctx, bayesian_weights, bayesian_version, publisher
                )
        finally:
            publisher.publish(PipelineCompleted({
                "symbol": symbol,
                "status": status
            }))

    @classmethod
    def _execute_internal(
        cls,
        symbol: str,
        category: str,
        sector: str,
        ticker: pd.DataFrame,
        delivery_pct: Optional[float],
        pledge_pct: Optional[float],
        nifty_ret_20d: float,
        regime_ctx: dict,
        bayesian_weights: Optional[dict],
        bayesian_version: str,
        publisher: EventPublisher
    ) -> str:
        
        # ──────────────────────────────────────────────────────────
        # 1. Validation Stage
        # ──────────────────────────────────────────────────────────
        rejection_reason = None
        
        if ticker is None or ticker.empty:
            rejection_reason = "no_data"
        elif len(ticker) < 50:
            rejection_reason = "insufficient_bars"
        else:
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col_name in required_cols:
                if col_name not in ticker.columns:
                    rejection_reason = "missing_col"
                    break
                    
        publisher.publish(ValidationCompleted({
            "symbol": symbol,
            "status": "REJECTED" if rejection_reason else "PASSED",
            "rejection_reason": rejection_reason,
            "bars": len(ticker) if ticker is not None else 0
        }))
        
        if rejection_reason:
            return "REJECTED"

        # [RULE 67 CHANGE-RATIONALE: MODULAR_TARGETED_HYDRATION_v1.0]
        # Hydrate indicators on-demand for deterministic pipeline execution.
        if "RSI" not in ticker.columns or "PRIOR_20D_HIGH" not in ticker.columns:
            ticker = apply_indicators(ticker, timeframe="1d")
        
        if ticker is None or ticker.empty:
            rejection_reason = "indicator_fail"
        
        if rejection_reason:
            publisher.publish(IndicatorsCalculated({
                "symbol": symbol,
                "status": "REJECTED",
                "rejection_reason": rejection_reason
            }))
            return "REJECTED"
            
        latest = ticker.iloc[-1]
        
        publisher.publish(IndicatorsCalculated({
            "symbol": symbol,
            "status": "PASSED",
            "latest_indicators": {k: _safe_float(latest.get(k)) for k in ["RSI", "ADX", "EMA20", "SMA50", "SMA200", "ATR20", "PRIOR_20D_HIGH", "BB_UPPER", "BB_LOWER"]}
        }))

        # ──────────────────────────────────────────────────────────
        # 3. Scanner Stage (Breakout + Hard Rejections)
        # ──────────────────────────────────────────────────────────
        signals = detect_breakouts(ticker, timeframe="1d")
        
        if len(signals) < EOD_CONFIG["MIN_SIGNALS"]:
            rejection_reason = "weak_signals"
            
        # Apply strict technical filters
        candle_high  = _safe_float(latest.get("High"))
        candle_low   = _safe_float(latest.get("Low"))
        candle_open  = _safe_float(latest.get("Open"))
        candle_close = _safe_float(latest.get("Close"))
        candle_range = candle_high - candle_low
        candle_body  = abs(candle_close - candle_open)
        upper_wick   = candle_high - candle_close
        
        avg_volume = float(ticker["Volume"].iloc[-21:-1].mean()) if len(ticker) >= 22 else float(ticker["Volume"].iloc[:-1].mean())
        volume_ratio = _safe_float(latest.get("Volume")) / avg_volume if avg_volume > 0 else 0
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        close_position = (candle_close - candle_low) / candle_range if candle_range > 0 else 0
        wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        rsi_val = _safe_float(latest.get("RSI"))
        
        import circuit_helper
        is_circuit = circuit_helper.is_valid_circuit_candle(
            candle_range=candle_range,
            volume=_safe_float(latest.get("Volume")),
            close_price=candle_close
        )
        min_atr_expansion = EOD_ADVANCED_CONFIG.get("MIN_ATR_EXPANSION_RATIO", 1.2)
        atr_expansion = candle_range / _safe_float(latest.get("ATR20")) if _safe_float(latest.get("ATR20")) > 0 else 0
        
        # Scanner logical checks from eod_scanner.py
        if not rejection_reason:
            if avg_volume <= 0: rejection_reason = "zero_avg_volume"
            elif candle_range <= 0 and not is_circuit: rejection_reason = "zero_candle_range"
            elif body_ratio < EOD_CONFIG["MIN_BODY_RATIO"] and not is_circuit: rejection_reason = "weak_body"
            elif candle_close <= candle_open and not is_circuit: rejection_reason = "bearish_candle"
            elif close_position < EOD_CONFIG["MIN_CLOSE_POSITION"] and not is_circuit: rejection_reason = "weak_close_pos"
            elif wick_ratio > EOD_CONFIG["MAX_UPPER_WICK"] and not is_circuit: rejection_reason = "upper_wick"
            elif volume_ratio < MIN_BREAKOUT_VOLUME_RATIO: rejection_reason = "low_volume"
            elif avg_volume < EOD_CONFIG["MIN_VOLUME_AVG"]: rejection_reason = "low_avg_volume"
            elif candle_close < MIN_STOCK_PRICE: rejection_reason = "penny_stock"
            elif not (EOD_CONFIG["MIN_RSI"] <= rsi_val <= EOD_CONFIG["MAX_RSI"]): rejection_reason = "rsi_range"
            elif _safe_float(latest.get("PRIOR_20D_HIGH")) <= 0 or candle_close <= (_safe_float(latest.get("PRIOR_20D_HIGH")) * (1.0 + (MIN_BREAKOUT_MARGIN.get("EOD", 0.0) / 100.0))): rejection_reason = "no_structural_breakout"
            elif _safe_float(latest.get("ATR20")) <= 0: rejection_reason = "missing_atr"
            elif not is_circuit and atr_expansion < min_atr_expansion: rejection_reason = "no_atr_expansion"
            elif _safe_float(latest.get("BB_WIDTH_PCTILE")) > EOD_ADVANCED_CONFIG.get("MAX_BB_WIDTH_PCTILE", 0.80): rejection_reason = "base_too_wide"
            elif not (candle_close > _safe_float(latest.get("SMA50")) > _safe_float(latest.get("SMA200"))): rejection_reason = "bad_trend_alignment"
            elif candle_close < _safe_float(latest.get("EMA20")): rejection_reason = "below_ema20"
            elif _safe_float(latest.get("EMA20")) > 0 and (candle_close - _safe_float(latest.get("EMA20"))) / _safe_float(latest.get("ATR20")) > EOD_ADVANCED_CONFIG.get("MAX_EXTENDED_BREAKOUT_ATR_MULT", 1.5): rejection_reason = "overextended_breakout"
            elif _safe_float(latest.get("ADX")) < ADX_MIN_THRESHOLD: rejection_reason = "weak_adx"
            
        publisher.publish(ScannerCompleted({
            "symbol": symbol,
            "status": "REJECTED" if rejection_reason else "PASSED",
            "rejection_reason": rejection_reason,
            "signals": signals
        }))
        
        if rejection_reason:
            return "REJECTED"

        # ──────────────────────────────────────────────────────────
        # 4. Scoring Stage
        # ──────────────────────────────────────────────────────────
        atr_val = _safe_float(latest.get("ATR"))
        
        score, model_version, applied_bayesian_weights = calculate_score(
            category=category,
            breakout_count=len(signals),
            rsi=rsi_val,
            volume_ratio=volume_ratio,
            breakout_signals=signals,
            ticker=ticker,
            latest=latest,
            symbol=symbol,
            timeframe="1d",
            atr_val=atr_val,
            delivery_pct=delivery_pct,
            promoter_pledge_pct=pledge_pct,
            nifty_ret=nifty_ret_20d,
            regime_ctx=regime_ctx,
            bayesian_weights=bayesian_weights,
            bayesian_version=bayesian_version
        )
        
        # Penalties
        technical_penalties = {}
        prior_high = _safe_float(latest.get("PRIOR_20D_HIGH"))
        atr20 = _safe_float(latest.get("ATR20"))
        atr_extension = (candle_close - prior_high) / atr20 if atr20 > 0 else 0
        max_ext = EOD_ADVANCED_CONFIG.get("MAX_EXTENDED_BREAKOUT_ATR_MULT", 1.5)
        if atr_extension > max_ext:
            pen_mult = EOD_ADVANCED_CONFIG.get("GAP_AND_GO_PENALTY_MULT", 10)
            max_pen = EOD_ADVANCED_CONFIG.get("GAP_AND_GO_MAX_PENALTY", 20)
            technical_penalties["extended_breakout"] = min(max_pen, (atr_extension - max_ext) * pen_mult)
            
        obv_penalty = 0
        if _safe_float(latest.get("OBV_SLOPE")) <= EOD_ADVANCED_CONFIG.get("MIN_OBV_SLOPE", 0.0):
            obv_penalty = -5
            
        if score > 0:
            for pen_val in technical_penalties.values():
                score -= pen_val
            score = max(0, score + obv_penalty)
            
        publisher.publish(ScoresCalculated({
            "symbol": symbol,
            "raw_score": score,
            "model_version": model_version,
            "technical_penalties": technical_penalties,
            "obv_penalty": obv_penalty,
            "final_score": score
        }))

        # ──────────────────────────────────────────────────────────
        # 5. Candidate Selection Stage
        # ──────────────────────────────────────────────────────────
        BASE_SCORE_THRESHOLD = SCORE_THRESHOLDS.get("1d", 82)
        global_min_score = BASE_SCORE_THRESHOLD
        
        try:
            from config import REGIME_POLICIES
            market_regime = regime_ctx.get("trend", "NEUTRAL")
            modifier = REGIME_POLICIES.get(market_regime, {}).get("score_modifier", 0)
            if modifier > 0:
                global_min_score += modifier
        except Exception:
            pass
            
        if score < global_min_score:
            rejection_reason = "low_score"
            
        publisher.publish(CandidateSelected({
            "symbol": symbol,
            "status": "REJECTED" if rejection_reason else "PASSED",
            "rejection_reason": rejection_reason,
            "threshold_required": global_min_score
        }))
        
        if rejection_reason:
            return "REJECTED"

        # ──────────────────────────────────────────────────────────
        # 6. SL / Target Engine Stage
        # ──────────────────────────────────────────────────────────
        sl_result = compute_sl_and_target(
            entry_price=candle_close,
            atr=atr_val,
            candle_range=candle_range,
            mode="EOD",
            adx=latest.get("ADX"),
            rsi=rsi_val,
            macd_hist=latest.get("MACD_HIST"),
            atr_pct=latest.get("ATR_PCT"),
            swing_low=latest.get("SWING_LOW"),
            swing_high=latest.get("SWING_HIGH"),
            bb_upper=latest.get("BB_UPPER"),
            bb_lower=latest.get("BB_LOWER"),
            bb_mid=latest.get("BB_MID"),
            s1=latest.get("S1"),
            s2=latest.get("S2"),
            r1=latest.get("R1"),
            r2=latest.get("R2"),
            swing_low_raw=latest.get("SWING_LOW_RAW"),
            swing_high_raw=latest.get("SWING_HIGH_RAW"),
            candle_low=candle_low,
            vwap=latest.get("VWAP"),
            ticker=ticker,
        )
        
        if sl_result.get("is_rejected"):
            rejection_reason = "low_rr"
            
        publisher.publish(SLTargetComputed({
            "symbol": symbol,
            "status": "REJECTED" if rejection_reason else "PASSED",
            "rejection_reason": rejection_reason,
            "sl_result": sl_result
        }))
        
        if rejection_reason:
            return "REJECTED"

        # ──────────────────────────────────────────────────────────
        # 7. Alert Creation Stage
        # ──────────────────────────────────────────────────────────
        above_ema20  = bool(candle_close >= _safe_float(latest.get("EMA20")))
        above_sma50  = bool(candle_close >= _safe_float(latest.get("SMA50")))
        above_golden_cross = bool(_safe_float(latest.get("SMA50")) >= _safe_float(latest.get("SMA200")))
        
        context = {
            "technicals": {
                "above_ema20":      above_ema20,
                "above_sma50":      above_sma50,
                "above_golden_cross":     above_golden_cross,
                "body_ratio":       round(body_ratio * 100, 2),
                "delivery_pct":     round(delivery_pct, 1) if delivery_pct is not None else None,
                "rsi":              round(rsi_val, 1),
                "volume_ratio":     round(volume_ratio, 2),
                "breakout_level":   round(_safe_float(latest.get("PRIOR_20D_HIGH")), 2),
                "atr20":            round(_safe_float(latest.get("ATR20")), 2),
                "regime":           regime_ctx.get("trend"),
                "score":            score
            },
            "session": {
                "open":             round(candle_open, 2),
                "day_high":         round(candle_high, 2),
                "day_low":          round(candle_low, 2)
            },
            "execution": {
                "sl_method":        sl_result.get("sl_method"),
                "t_method":         sl_result.get("target_method")
            },
            "sl_result": sl_result,
            "algo_version": ACTIVE_ALGO_VERSION
        }
        
        publisher.publish(AlertCreated({
            "symbol": symbol,
            "category": category,
            "entry_price": round(candle_close, 2),
            "signals": list(signals.keys()),
            "score": score,
            "stop_loss": sl_result.get("stop_loss"),
            "target_1": sl_result.get("target_1"),
            "target_2": sl_result.get("target_2"),
            "context": context
        }))
        
        return "SUCCESS"
