import time
import logging
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
from database import get_connection

# [VERSION: MACRO_CACHE_V2.0] Session-Aware Single-Flight Macro Cache with 15m TTL
# RATIONALE:
#   - TTL set to 900s (15 minutes) to eliminate redundant network fetching across 5m scanner cycles.
#   - Stores session_date (IST) in cache entry to prevent yesterday's data from crossing session boundaries.
#   - Employs single-flight lock pattern: releases cache lock during external network API calls so
#     unrelated scanner threads are not blocked while macro calculations execute.
MACRO_CACHE_TTL_SECONDS = 900  # 15 minutes

class MacroCache:
    def __init__(self):
        self.lock = Lock()
        
        # Structure: {"computed_at": float, "session_date": str, "data": pd.DataFrame}
        self.daily_entry = None
        self.daily_in_flight = False

        self.intraday_entry = None
        self.intraday_in_flight = False

_cache = MacroCache()

def _get_daily_nifty() -> pd.DataFrame:
    """Fetch 1-year daily NIFTY data with single-flight session-aware 15-minute caching."""
    now_mono = time.monotonic()
    today_ist = datetime.now(IST).strftime("%Y-%m-%d")

    # 1. Fast Cache Check under Lock
    with _cache.lock:
        entry = _cache.daily_entry
        if entry is not None:
            if entry.get("session_date") == today_ist and (now_mono - entry.get("computed_at", 0)) < MACRO_CACHE_TTL_SECONDS:
                return entry.get("data")
            
        # Avoid duplicate parallel fetches
        if _cache.daily_in_flight and entry is not None:
            return entry.get("data")
            
        _cache.daily_in_flight = True

    # 2. Network API Fetch OUTSIDE of Cache Lock (Single-Flight Pattern)
    try:
        from price_cache import fetch_unified_historical
        fetched = fetch_unified_historical(["NIFTY 50"], period="1y", interval="1d", requester="macro_daily")
        df = fetched.get("NIFTY 50")
        from core_enums import ProviderResult
        if df is not None and not isinstance(df, ProviderResult) and not df.empty:
            with _cache.lock:
                _cache.daily_entry = {
                    "computed_at": now_mono,
                    "session_date": today_ist,
                    "data": df
                }
                _cache.daily_in_flight = False
            return df
    except Exception:
        logger.exception("Failed to fetch Nifty daily macro data")
    finally:
        with _cache.lock:
            _cache.daily_in_flight = False
        
    with _cache.lock:
        return _cache.daily_entry.get("data") if _cache.daily_entry else None

def _get_intraday_nifty() -> pd.DataFrame:
    """Fetch 5-day 15-minute NIFTY data with single-flight session-aware 15-minute caching."""
    now_mono = time.monotonic()
    today_ist = datetime.now(IST).strftime("%Y-%m-%d")

    # 1. Fast Cache Check under Lock
    with _cache.lock:
        entry = _cache.intraday_entry
        if entry is not None:
            if entry.get("session_date") == today_ist and (now_mono - entry.get("computed_at", 0)) < MACRO_CACHE_TTL_SECONDS:
                return entry.get("data")
                
        if _cache.intraday_in_flight and entry is not None:
            return entry.get("data")
            
        _cache.intraday_in_flight = True

    # 2. Network API Fetch OUTSIDE of Cache Lock (Single-Flight Pattern)
    try:
        from price_cache import fetch_unified_historical
        fetched = fetch_unified_historical(["NIFTY 50"], period="5d", interval="15m", requester="macro_intraday")
        df = fetched.get("NIFTY 50")
        from core_enums import ProviderResult
        if df is not None and not isinstance(df, ProviderResult) and not df.empty:
            with _cache.lock:
                _cache.intraday_entry = {
                    "computed_at": now_mono,
                    "session_date": today_ist,
                    "data": df
                }
                _cache.intraday_in_flight = False
            return df
    except Exception:
        logger.exception("Failed to fetch Nifty intraday macro data")
    finally:
        with _cache.lock:
            _cache.intraday_in_flight = False
        
    with _cache.lock:
        return _cache.intraday_entry.get("data") if _cache.intraday_entry else None

class MarketRegimeEngine:
    @staticmethod
    def _compute_state_for_row(df, idx):
        import pandas as pd
        
        try:
            price = float(df["Close"].iloc[idx])
            end_idx = len(df) + idx + 1 if idx < 0 else idx + 1
            hist_slice = df["Close"].iloc[:end_idx]
            sma20 = float(hist_slice.tail(20).mean())
            sma50 = float(hist_slice.tail(50).mean())
            sma200 = float(hist_slice.tail(200).mean())
            
            nifty_ago = float(df["Close"].iloc[idx - 20])
            n_ret = ((price - nifty_ago) / nifty_ago) * 100.0 if nifty_ago > 0 else 0.0
            
            try:
                import ta
                adx_series = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
                adx_val = float(adx_series.iloc[idx]) if pd.notna(adx_series.iloc[idx]) else 0.0
            except Exception:
                adx_val = 0.0
                
            trend = "NEUTRAL"
            if n_ret > 2.0: trend = "BULL"
            elif n_ret < -2.0: trend = "BEAR"
                
            strength = "WEAK"
            if adx_val >= 25.0: strength = "STRONG"
            elif adx_val >= 15.0: strength = "MODERATE"
            
            bull_signals = sum([
                1 if price > sma20 else 0,
                1 if price > sma50 else 0,
                1 if price > sma200 else 0,
                1 if strength == "STRONG" and trend == "BULL" else 0
            ])
            bear_signals = sum([
                1 if price < sma20 else 0,
                1 if price < sma50 else 0,
                1 if price < sma200 else 0,
                1 if strength == "STRONG" and trend == "BEAR" else 0
            ])
            
            total_signals = 4
            agreement_count = bull_signals if trend == "BULL" else (bear_signals if trend == "BEAR" else total_signals - abs(bull_signals - bear_signals))
            conf_pct = max(0, min(100, int((agreement_count / total_signals) * 100)))
            
            trend_score = 100 if trend == "BULL" else (0 if trend == "BEAR" else 50)
            strength_score = 100 if strength == "STRONG" else (50 if strength == "MODERATE" else 0)
            if trend == "BEAR" and strength == "STRONG": strength_score = 0
            elif trend == "BEAR" and strength == "MODERATE": strength_score = 25
            
            conf_score = conf_pct if trend == "BULL" else (100 - conf_pct if trend == "BEAR" else 50)
            
            # Simplified score for history tracking (Volatility excluded for historical row)
            score = (trend_score * 0.40) + (strength_score * 0.30) + (conf_score * 0.20)
            
            return {
                "score": score,
                "price": price,
                "sma20": sma20,
                "sma50": sma50,
                "sma200": sma200,
                "n_ret": n_ret,
                "adx_val": adx_val,
                "trend": trend,
                "strength": strength,
                "conf_pct": conf_pct,
                "agreement_count": agreement_count,
                "total_signals": total_signals
            }
        except Exception:
            return None

    @staticmethod
    def get_regime_context(nifty_ret: float = None) -> dict:
        import logging
        logger = logging.getLogger(__name__)

        try:
            from macro_utils import _get_daily_nifty, get_nifty_intraday_drop
            df = _get_daily_nifty()
            
            if df is not None and not df.empty and len(df) >= 200:
                state_today = MarketRegimeEngine._compute_state_for_row(df, -1)
                state_yesterday = MarketRegimeEngine._compute_state_for_row(df, -2)
                
                if state_today:
                    price = state_today["price"]
                    sma20 = state_today["sma20"]
                    sma50 = state_today["sma50"]
                    sma200 = state_today["sma200"]
                    n_ret = state_today["n_ret"]
                    adx_val = state_today["adx_val"]
                    trend = state_today["trend"]
                    strength = state_today["strength"]
                    conf_pct = state_today["conf_pct"]
                    agreement_count = state_today["agreement_count"]
                    base_total = state_today["total_signals"]
                    
                    intraday_drop = get_nifty_intraday_drop()
                    volatility = "NORMAL"
                    if intraday_drop >= 1.5: volatility = "HIGH"
                    elif intraday_drop <= 0.5: volatility = "LOW"
                    
                    if volatility == "LOW": 
                        if trend == "BULL": agreement_count += 1
                        elif trend == "BEAR": agreement_count -= 1
                    elif volatility == "HIGH":
                        if trend == "BULL": agreement_count -= 1
                        elif trend == "BEAR": agreement_count += 1
                        
                    total_signals = base_total + 1
                    agreement_count = max(0, min(total_signals, agreement_count))
                    conf_pct = int((agreement_count / total_signals) * 100)
                    
                    trend_score = 100 if trend == "BULL" else (0 if trend == "BEAR" else 50)
                    strength_score = 100 if strength == "STRONG" else (50 if strength == "MODERATE" else 0)
                    if trend == "BEAR" and strength == "STRONG": strength_score = 0
                    elif trend == "BEAR" and strength == "MODERATE": strength_score = 25
                    vol_score = 100 if volatility == "LOW" else (50 if volatility == "NORMAL" else 0)
                    conf_score = conf_pct if trend == "BULL" else (100 - conf_pct if trend == "BEAR" else 50)
                    
                    market_score = (trend_score * 0.40) + (strength_score * 0.30) + (conf_score * 0.20) + (vol_score * 0.10)
                    market_score = max(0, min(100, int(market_score)))
                    
                    trend_direction = "STABLE"
                    if state_yesterday:
                        y_score = state_yesterday["score"] + (vol_score * 0.10) # Assume same vol for approximation
                        if market_score > y_score + 2:
                            trend_direction = "IMPROVING"
                        elif market_score < y_score - 2:
                            trend_direction = "WEAKENING"
                            
                    phase = "CONSOLIDATION"
                    if trend == "BULL":
                        if price > sma20 and sma20 > sma50 and sma50 > sma200: phase = "EXPANSION"
                        elif price < sma20 and price > sma50: phase = "PULLBACK"
                    elif trend == "BEAR":
                        if price < sma20 and sma20 < sma50 and sma50 < sma200: phase = "CAPITULATION"
                        elif price < sma20 and price > sma200: phase = "DISTRIBUTION"

                    return {
                        "engine_version": "MARKET_CONTEXT_V1",
                        "trend": trend,
                        "strength": strength,
                        "volatility": volatility,
                        "market_phase": phase,
                        "trend_direction": trend_direction,
                        "market_score": market_score,
                        "confidence": {
                            "agreement": agreement_count,
                            "signals": total_signals,
                            "score": conf_pct
                        },
                        "metrics": {
                            "return20d": round(n_ret, 2),
                            "adx": round(adx_val, 2),
                            "atr_pct": round(intraday_drop, 2), # Approximating ATR pct with drop
                            "price_vs_20dma": round(((price - sma20)/sma20)*100, 2) if sma20 > 0 else 0,
                            "price_vs_50dma": round(((price - sma50)/sma50)*100, 2) if sma50 > 0 else 0,
                            "price_vs_200dma": round(((price - sma200)/sma200)*100, 2) if sma200 > 0 else 0
                        }
                    }
        except Exception as e:
            logger.warning(f"Failed to compute context inputs: {e}")
            
        return {
            "engine_version": "MARKET_CONTEXT_V1",
            "trend": "NEUTRAL",
            "strength": "WEAK",
            "volatility": "NORMAL",
            "market_phase": "CONSOLIDATION",
            "trend_direction": "STABLE",
            "market_score": 50,
            "confidence": {"agreement": 0, "signals": 5, "score": 0},
            "metrics": {}
        }


def get_macro_regime(nifty_ret: Optional[float] = None) -> str:
    ctx = MarketRegimeEngine.get_regime_context(nifty_ret=nifty_ret)
    trend = ctx.get("trend", "NEUTRAL")
    strength = ctx.get("strength", "WEAK")
    volatility = ctx.get("volatility", "NORMAL")
    
    # Priority Decision Tree for 9 Verbatim Regimes
    if trend == "BEAR":
        if strength == "STRONG":
            return "STRONG_BEAR"
        elif strength == "WEAK":
            return "WEAK_BEAR"
        return "BEAR"
    elif trend == "BULL":
        if strength == "STRONG":
            return "STRONG_BULL"
        elif strength == "WEAK":
            return "WEAK_BULL"
        return "BULL"
    else:  # NEUTRAL
        if volatility == "HIGH":
            return "RANGEBOUND"
        elif volatility == "LOW":
            return "SIDEWAYS"
        return "NEUTRAL"


def get_nifty_20d_return() -> float:
    """Returns the 20-day percentage return of Nifty. Defaults to 0.0% if unavailable."""
    try:
        df = _get_daily_nifty()
        if df is not None and not df.empty and len(df) >= 20:
            val_now = df["Close"].iloc[-1]
            nifty_now = float(val_now.iloc[0]) if hasattr(val_now, 'iloc') else float(val_now)
            val_ago = df["Close"].iloc[-20]
            nifty_ago = float(val_ago.iloc[0]) if hasattr(val_ago, 'iloc') else float(val_ago)
            if nifty_ago > 0:
                return (nifty_now - nifty_ago) / nifty_ago * 100.0
    except Exception as e:
        logger.warning(f"Failed to compute Nifty 20d return: {e}")
    return 0.0  # Fallback assumption

def get_nifty_6m_state() -> tuple[Optional[float], Optional[float]]:
    """
    Returns (ret_6m, dist_52w) for Nifty.
    Returns (None, None) if data is unavailable.
    """
    try:
        df = _get_daily_nifty()
        if df is not None and not df.empty and len(df) >= 2:
            hist_6m = df.tail(126) # Approx 6 months
            if len(hist_6m) >= 2:
                start_price = float(hist_6m['Close'].iloc[0])
                end_price = float(hist_6m['Close'].iloc[-1])
                ret_6m = ((end_price - start_price) / start_price) * 100.0 if start_price > 0 else 0.0
            else:
                ret_6m = None
                
            high_52w = float(df['High'].max())
            end_price_1y = float(df['Close'].iloc[-1])
            dist_52w = ((high_52w - end_price_1y) / high_52w) * 100.0 if high_52w > 0 else 0.0
            
            return ret_6m, dist_52w
    except Exception as e:
        logger.warning(f"Failed to compute Nifty 6m state: {e}")
    return None, None

def get_nifty_intraday_drop() -> float:
    """
    Returns the percentage drop from today's open to current price.
    If the market is up or data is unavailable, returns 0.0.
    """
    try:
        df = _get_intraday_nifty()
        if df is not None and not df.empty:
            today_str = datetime.now(IST).strftime('%Y-%m-%d')
            
            df_safe = df.copy()
            # Normalize index to avoid RangeIndex date coercion issues
            datetime_col = next((c for c in ["Datetime", "Date", "index"] if c in df_safe.columns), None)
            if datetime_col:
                df_safe = df_safe.set_index(datetime_col)
                
            df_safe.index = pd.to_datetime(df_safe.index, errors="coerce")
            today_data = df_safe[df_safe.index.notna() & (df_safe.index.strftime("%Y-%m-%d") == today_str)]
            
            if not today_data.empty:
                nifty_open = float(today_data['Open'].iloc[0])
                nifty_current = float(today_data['Close'].iloc[-1])
                if nifty_open > 0:
                    drop = ((nifty_open - nifty_current) / nifty_open) * 100.0
                    return drop if drop > 0 else 0.0
    except Exception as e:
        logger.warning(f"Failed to compute Nifty intraday drop: {e}")
    return 0.0

# =====================================================================================
# FEATURE F-03: RELATIVE STRENGTH (RS) VS NIFTY 50 (ACTIVE SCAN UNIVERSE)
# =====================================================================================

def compute_nifty_rs_rating(symbols: list = None) -> dict:
    """
    Computes 63-trading-day RS rating for candidate symbols relative to Nifty 50 (^NSEI)
    over the active scan universe (~500-700 liquid equities).

    Returns a dict mapping {symbol: rs_percentile_rank} where percentile is 0.0 to 100.0.
    """
    if not symbols:
        try:
            import os
            from config import WATCHLIST_PATH
            if os.path.exists(WATCHLIST_PATH):
                df_wl = pd.read_parquet(WATCHLIST_PATH)
                symbols = df_wl['symbol'].tolist()
        except Exception:
            pass

    if not symbols:
        return {}

    try:
        from price_cache import fetch_unified_historical
        fetch_list = list(set(symbols + ["NIFTY 50"]))
        historical_dict = fetch_unified_historical(fetch_list, period="1y", interval="1d", requester="rs_rating")

        nifty_df = historical_dict.get("NIFTY 50")
        if nifty_df is None or nifty_df.empty or len(nifty_df) < 20:
            return {s: 50.0 for s in symbols}

        nifty_start = float(nifty_df["Close"].iloc[max(0, len(nifty_df) - 63)])
        nifty_end = float(nifty_df["Close"].iloc[-1])
        nifty_ret = ((nifty_end - nifty_start) / nifty_start) * 100.0 if nifty_start > 0 else 0.0

        raw_rs_scores = {}
        for sym in symbols:
            df_sym = historical_dict.get(sym)
            if df_sym is not None and not df_sym.empty and len(df_sym) >= 20:
                s_start = float(df_sym["Close"].iloc[max(0, len(df_sym) - 63)])
                s_end = float(df_sym["Close"].iloc[-1])
                s_ret = ((s_end - s_start) / s_start) * 100.0 if s_start > 0 else 0.0
                raw_rs_scores[sym] = s_ret - nifty_ret
            else:
                raw_rs_scores[sym] = 0.0

        # Compute percentile ranks across the active scan universe
        series_rs = pd.Series(raw_rs_scores)
        percentiles = (series_rs.rank(pct=True) * 100.0).round(2).to_dict()
        return percentiles
    except Exception as e:
        logger.warning(f"Failed to compute Nifty RS ratings: {e}")
        return {s: 50.0 for s in (symbols or [])}

def compute_nifty_rs_rating_with_hysteresis(symbols: list = None) -> dict:
    """
    Computes RS ratings and applies a 3-Session Hysteresis Rule (3-day rolling average smoothing)
    to prevent RS 79% <-> 81% flickering across daily scans.
    """
    current_rs = compute_nifty_rs_rating(symbols)
    if not current_rs:
        return {}

    try:
        from datetime import datetime
        today_str = datetime.now(IST).strftime('%Y-%m-%d')

        # Persist daily RS percentiles into PostgreSQL table for rolling hysteresis
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS daily_rs_history (
                        symbol VARCHAR(20) NOT NULL,
                        rs_date DATE NOT NULL,
                        rs_percentile NUMERIC(5, 2) NOT NULL,
                        PRIMARY KEY (symbol, rs_date)
                    )
                """)
                for sym, rs_val in current_rs.items():
                    cur.execute("""
                        INSERT INTO daily_rs_history (symbol, rs_date, rs_percentile)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (symbol, rs_date) DO UPDATE
                        SET rs_percentile = EXCLUDED.rs_percentile
                    """, (sym, today_str, rs_val))
                conn.commit()

                # Fetch 3-session rolling average
                smoothed_rs = {}
                cur.execute("""
                    SELECT symbol, AVG(rs_percentile)::numeric(5, 2)
                    FROM (
                        SELECT symbol, rs_percentile,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY rs_date DESC) as rn
                        FROM daily_rs_history
                    ) t
                    WHERE rn <= 3
                    GROUP BY symbol
                """)
                for row in cur.fetchall():
                    smoothed_rs[row[0]] = float(row[1])

                # Return smoothed RS for symbols requested
                return {sym: smoothed_rs.get(sym, current_rs.get(sym, 50.0)) for sym in (symbols or current_rs.keys())}
    except Exception as e:
        logger.warning(f"RS Hysteresis calculation fallback to raw RS: {e}")
        return current_rs

# =====================================================================================
# FEATURE F-07: SECTOR & INDUSTRY REGIME ENGINE (BLENDED + 3-SESSION HYSTERESIS)
# =====================================================================================

SECTOR_MAP = {
    "^CNXAUTO": "NIFTY AUTO",
    "^NSEBANK": "NIFTY BANK",
    "^CNXIT": "NIFTY IT",
    "^CNXREALTY": "NIFTY REALTY",
    "^CNXPHARMA": "NIFTY PHARMA",
    "^CNXMETAL": "NIFTY METAL",
    "^CNXENERGY": "NIFTY ENERGY",
    "^CNXFMCG": "NIFTY FMCG",
    "^CNXINFRA": "NIFTY INFRA",
    "^CNXFIN": "NIFTY FINANCIAL SERVICES",
    "^CNXMEDIA": "NIFTY MEDIA",
    "^CNXPSUBANK": "NIFTY PSU BANK",
    "^CNXCMDT": "NIFTY COMMODITIES"
}

def compute_sector_regime_rankings() -> dict:
    """
    Downloads 14 NSE Sector Indices, computes blended 63d (70%) + 21d (30%) return,
    and applies a 3-Session Hysteresis rule:
      - Default counters start at 0.
      - Stock sector must hold Top-3 for 3 consecutive days to earn 'TAILWIND'.
      - Stock sector must hold Bottom-3 for 3 consecutive days for 'HEADWIND'.
    """
    try:
        from price_cache import fetch_unified_historical
        from database import get_connection, IST
        from datetime import datetime

        today_str = datetime.now(IST).strftime('%Y-%m-%d')
        sector_symbols = list(SECTOR_MAP.keys())

        fetched = fetch_unified_historical(sector_symbols, period="6mo", interval="1d", requester="sector_regime")
        blended_returns = {}

        for sym, name in SECTOR_MAP.items():
            df_sec = fetched.get(sym)
            if df_sec is not None and not df_sec.empty and len(df_sec) >= 21:
                p_current = float(df_sec["Close"].iloc[-1])
                p_21d = float(df_sec["Close"].iloc[max(0, len(df_sec) - 21)])
                p_63d = float(df_sec["Close"].iloc[max(0, len(df_sec) - 63)])

                ret_21d = ((p_current - p_21d) / p_21d) * 100.0 if p_21d > 0 else 0.0
                ret_63d = ((p_current - p_63d) / p_63d) * 100.0 if p_63d > 0 else 0.0

                blended = (0.7 * ret_63d) + (0.3 * ret_21d)
                blended_returns[sym] = round(blended, 2)
            else:
                blended_returns[sym] = 0.0

        # Rank sectors 1 to 14 (descending by blended score)
        sorted_sectors = sorted(blended_returns.items(), key=lambda x: x[1], reverse=True)
        raw_ranks = {sym: rank + 1 for rank, (sym, _) in enumerate(sorted_sectors)}

        # Fetch previous day's hysteresis counters from DB
        prev_counters = {}
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT sector_symbol, consecutive_top3_days, consecutive_bottom3_days
                        FROM sector_rankings
                        WHERE ranking_date = (SELECT MAX(ranking_date) FROM sector_rankings WHERE ranking_date < %s)
                    """, (today_str,))
                    for row in cur.fetchall():
                        prev_counters[row[0]] = (row[1] or 0, row[2] or 0)
        except Exception:
            pass

        results = {}
        db_rows = []

        for sym, name in SECTOR_MAP.items():
            rank = raw_ranks.get(sym, 7)
            b_score = blended_returns.get(sym, 0.0)
            prev_top3, prev_bottom3 = prev_counters.get(sym, (0, 0))

            if rank <= 3:
                consec_top3 = prev_top3 + 1
                consec_bottom3 = 0
            elif rank >= len(SECTOR_MAP) - 2:
                consec_top3 = 0
                consec_bottom3 = prev_bottom3 + 1
            else:
                consec_top3 = 0
                consec_bottom3 = 0

            # Grant status strictly on 3-session hysteresis rule
            if consec_top3 >= 3:
                status = 'TAILWIND'
            elif consec_bottom3 >= 3:
                status = 'HEADWIND'
            else:
                status = 'NEUTRAL'

            results[sym] = {
                "sector_name": name,
                "blended_score": b_score,
                "raw_rank": rank,
                "consecutive_top3_days": consec_top3,
                "consecutive_bottom3_days": consec_bottom3,
                "effective_status": status
            }
            db_rows.append((sym, name, today_str, b_score, rank, consec_top3, consec_bottom3, status))

        # Persist rankings to DB
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for row in db_rows:
                        cur.execute("""
                            INSERT INTO sector_rankings
                                (sector_symbol, sector_name, ranking_date, blended_score, raw_rank,
                                 consecutive_top3_days, consecutive_bottom3_days, effective_status)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (sector_symbol, ranking_date) DO UPDATE
                            SET blended_score = EXCLUDED.blended_score,
                                raw_rank = EXCLUDED.raw_rank,
                                consecutive_top3_days = EXCLUDED.consecutive_top3_days,
                                consecutive_bottom3_days = EXCLUDED.consecutive_bottom3_days,
                                effective_status = EXCLUDED.effective_status
                        """, row)
                    conn.commit()
        except Exception as dbe:
            logger.warning(f"Failed to persist sector_rankings: {dbe}")

        return results
    except Exception as e:
        logger.warning(f"Failed to compute sector regime rankings: {e}")
        return {}

