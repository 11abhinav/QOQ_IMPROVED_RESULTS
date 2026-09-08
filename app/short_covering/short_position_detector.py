"""
app/short_covering/short_position_detector.py

Layer 1: EOD Positioning Engine for Short-Covering Scanner.
Objective:
- Analyzes daily price action and open interest over the prior 5-10 trading sessions.
- Detects heavy short positioning buildup (SBR >= 0.60, OI expansion >= +8%, falling price into support).
- Identifies early stabilization / divergence (RSI oversold rebound, bottom wicks).
- Generates the Next-Day Short-Covering Candidate Watchlist and persists to DB.
"""

import os
import logging
import time
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from app.short_covering.fno_universe import fno_universe_manager
from app.short_covering.oi_data_service import oi_data_service
from app.short_covering.short_covering_schema import EODShortPositionCandidate
try:
    from app.lock_utils import ProcessLock
    from app.database import get_connection, upsert_scanner_health, start_scanner_execution_run, complete_scanner_execution_run
    from app.trading_calendar import get_latest_trading_date, is_trading_day
except ImportError:
    from lock_utils import ProcessLock
    from database import get_connection, upsert_scanner_health, start_scanner_execution_run, complete_scanner_execution_run
    from trading_calendar import get_latest_trading_date, is_trading_day

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
_eod_lock = ProcessLock("short_covering_eod_lock")


class ShortPositionDetector:
    """
    Layer 1 EOD Engine: Identifies stocks with accumulated short positions.
    Scans the F&O universe at 19:15 IST daily.
    """

    def __init__(
        self,
        min_oi_buildup_5d_pct: float = 6.0,
        min_short_buildup_ratio: float = 0.55,
        max_rsi: float = 50.0,
        min_quality_score: float = 50.0,
        max_watchlist_size: int = 35
    ):
        self.min_oi_buildup_5d_pct = min_oi_buildup_5d_pct
        self.min_short_buildup_ratio = min_short_buildup_ratio
        self.max_rsi = max_rsi
        self.min_quality_score = min_quality_score
        self.max_watchlist_size = max_watchlist_size

    def scan_eod_universe(
        self,
        as_of: Optional[date] = None,
        custom_symbols: Optional[List[str]] = None,
        persist_db: bool = True,
        trigger_type: str = "SCHEDULED",
        scheduler_name: str = "CRON"
    ) -> List[EODShortPositionCandidate]:
        """
        Scans the F&O universe at EOD to identify stocks with accumulated short positions.
        Returns a list of high-quality candidates sorted by buildup quality score.
        """
        target_date = as_of or datetime.now(IST).date()
        # Resolve to latest valid trading day (e.g. Monday resolves to today or Friday; weekends resolve to Friday)
        valid_trading_date = get_latest_trading_date(target_date) if not is_trading_day(target_date) else target_date

        logger.info("[SHORT_COVERING_EOD] Acquiring lock: short_covering_eod_lock")
        if not _eod_lock.acquire(blocking=False):
            logger.warning("🛑 [SHORT_COVERING_EOD] Lock 'short_covering_eod_lock' held by another instance. Skipping duplicate run.")
            return []

        symbols = custom_symbols or fno_universe_manager.get_fno_symbols()
        run_ctx = None
        try:
            run_ctx = start_scanner_execution_run(
                scanner_name="SHORT_COVERING_EOD",
                trigger_type=trigger_type,
                scheduler_name=scheduler_name,
                total_stocks=len(symbols)
            )
        except Exception as ctx_err:
            if "already actively running" in str(ctx_err).lower():
                logger.warning("🛑 [SHORT_COVERING_EOD] Already actively running in DB history. Skipping duplicate run.")
                _eod_lock.release()
                return []
            run_ctx = None

        start_t = time.monotonic()
        try:
            upsert_scanner_health(
                scanner_name="SHORT_COVERING_EOD",
                status="RUNNING",
                error_msg="EOD positioning analysis in progress...",
                scheduled_for="Daily 19:15 IST (Market Days)",
                run_id=run_ctx.run_id if run_ctx else None
            )

            candidates: List[EODShortPositionCandidate] = []

            logger.info("🔍 [SHORT_COVERING_EOD] Scanning %d F&O symbols for trading date: %s", len(symbols), valid_trading_date)

            for symbol in symbols:
                try:
                    candidate = self.evaluate_symbol(symbol, valid_trading_date)
                    if candidate is not None:
                        candidates.append(candidate)
                except Exception as e:
                    logger.debug("Error evaluating EOD candidate for %s: %s", symbol, e)

            # Sort descending by buildup quality score
            candidates.sort(key=lambda c: c.buildup_quality_score, reverse=True)
            logger.info("✅ [SHORT_COVERING_EOD] Identified %d short-buildup candidates for next-day watchlist", len(candidates))

            # Persist to database if requested
            if persist_db:
                self._persist_candidates_to_db(candidates, valid_trading_date)

            dur = round(time.monotonic() - start_t, 2)
            if run_ctx:
                run_ctx.set_total_stocks(len(symbols))
                run_ctx.record_fresh_data(len(candidates))
                complete_scanner_execution_run(run_ctx)

            upsert_scanner_health(
                scanner_name="SHORT_COVERING_EOD",
                status="OK",
                outcome="SUCCESS",
                total_count=len(symbols),
                processed_count=len(candidates),
                duration_seconds=dur,
                scheduled_for="Daily 19:15 IST (Market Days)",
                run_id=run_ctx.run_id if run_ctx else None
            )
            return candidates
        except Exception as exc:
            dur = round(time.monotonic() - start_t, 2)
            logger.exception("❌ [SHORT_COVERING_EOD] Scan failed: %s", exc)
            if run_ctx:
                complete_scanner_execution_run(run_ctx, exception=exc)
            upsert_scanner_health(
                scanner_name="SHORT_COVERING_EOD",
                status="DOWN",
                outcome="FAILURE",
                error_msg=str(exc),
                duration_seconds=dur,
                scheduled_for="Daily 19:15 IST (Market Days)",
                run_id=run_ctx.run_id if run_ctx else None
            )
            return []
        finally:
            _eod_lock.release()


    def evaluate_symbol(
        self,
        symbol: str,
        as_of: date
    ) -> Optional[EODShortPositionCandidate]:
        """
        Evaluates a single stock for prior short buildup over a 10-day lookback.
        """
        df = oi_data_service.get_daily_oi_history(symbol, lookback_days=15, as_of=as_of)
        if df is None or len(df) < 8:
            return None

        # Sort chronologically
        df = df.sort_values("date").reset_index(drop=True)

        closes = df["close"].values
        total_ois = df["total_oi"].values
        volumes = df["volume"].values

        # 1. Open Interest changes
        cur_oi = total_ois[-1]
        oi_5d_ago = total_ois[-6] if len(total_ois) >= 6 else total_ois[0]
        oi_10d_ago = total_ois[-11] if len(total_ois) >= 11 else total_ois[0]

        oi_5d_pct = ((cur_oi - oi_5d_ago) / max(oi_5d_ago, 1)) * 100.0
        oi_10d_pct = ((cur_oi - oi_10d_ago) / max(oi_10d_ago, 1)) * 100.0
        oi_1d_pct = df["oi_change_pct"].iloc[-1]

        # 2. Price changes over 5 and 10 days
        cur_price = closes[-1]
        price_5d_ago = closes[-6] if len(closes) >= 6 else closes[0]
        price_5d_pct = ((cur_price - price_5d_ago) / price_5d_ago) * 100.0

        # 3. Short Buildup Ratio (SBR) over last 6-10 days
        # Days where price fell and OI rose
        lookback_slice = df.tail(8)
        price_diffs = lookback_slice["close"].diff().dropna()
        oi_diffs = lookback_slice["total_oi"].diff().dropna()

        short_buildup_days = sum(1 for p, o in zip(price_diffs, oi_diffs) if p < 0 and o > 0)
        total_days = len(price_diffs)
        sbr = short_buildup_days / max(total_days, 1)

        # 4. Technical Indicators (RSI, ATR, Key Levels)
        rsi_14 = self._calculate_rsi(closes)
        atr_14 = self._calculate_atr(df)
        support_level = float(np.min(df["low"].tail(10)))
        overhead_resistance = float(np.max(df["high"].tail(10)))

        reasons = []
        score = 0.0

        # Criteria checks:
        # A. Prior short accumulation
        if oi_5d_pct >= self.min_oi_buildup_5d_pct or oi_10d_pct >= 8.0:
            score += 35.0
            reasons.append(f"Strong 5d/10d OI expansion (+{oi_5d_pct:.1f}% / +{oi_10d_pct:.1f}%)")
        elif oi_5d_pct >= 3.0 or oi_10d_pct >= 5.0:
            score += 20.0
            reasons.append(f"Moderate 5d/10d OI expansion (+{oi_5d_pct:.1f}%)")

        # B. Short buildup regime (SBR)
        if sbr >= self.min_short_buildup_ratio:
            score += 25.0
            reasons.append(f"High Short Buildup Ratio ({sbr:.2f})")
        elif sbr >= 0.35 or price_5d_pct < -1.5:
            score += 15.0
            reasons.append(f"Moderate Short Buildup ({sbr:.2f}) with price drop ({price_5d_pct:.1f}%)")

        # C. Price in oversold or support-forming zone
        if rsi_14 <= self.max_rsi:
            score += 20.0
            reasons.append(f"RSI in deep base/oversold zone ({rsi_14:.1f})")
        elif rsi_14 <= 58.0:
            score += 10.0
            reasons.append(f"Price stabilizing near base (RSI {rsi_14:.1f})")

        # D. Early signs of short fatigue (OI expansion plateaued or minor 1d dip with green candle)
        if oi_1d_pct <= 0.5 and closes[-1] >= df["open"].iloc[-1]:
            score += 20.0
            reasons.append("1d OI stall/reduction with bullish lower wick")

        # RULE 67 RATIONALE: Enforce self.min_quality_score (50.0+) rather than an arbitrary loose
        # 35.0 threshold to prevent low-conviction or noisy synthetic candidates from entering the watchlist.
        if score < self.min_quality_score:
            return None

        sector = fno_universe_manager.get_sector(symbol)

        candidate = EODShortPositionCandidate(
            symbol=symbol,
            scan_date=as_of,
            close_price=float(cur_price),
            total_oi=int(cur_oi),
            oi_change_pct_1d=float(oi_1d_pct),
            oi_buildup_5d_pct=float(oi_5d_pct),
            oi_buildup_10d_pct=float(oi_10d_pct),
            short_buildup_ratio=float(sbr),
            rsi_14=float(rsi_14),
            support_level=support_level,
            overhead_resistance=overhead_resistance,
            atr_14=float(atr_14),
            daily_volume=int(volumes[-1]),
            sector=sector,
            buildup_quality_score=min(100.0, score),
            reasons=reasons
        )
        return candidate

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculates RSI over closing prices."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / max(down, 1e-9)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period, len(deltas)):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / max(down, 1e-9)
            rsi = 100.0 - 100.0 / (1.0 + rs)
        return float(rsi)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates 14-period Average True Range."""
        if len(df) < 2:
            return float(df["close"].iloc[-1] * 0.02)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        if len(tr) < period:
            return float(np.mean(tr))
        return float(np.mean(tr[-period:]))

    def _persist_candidates_to_db(self, candidates: List[EODShortPositionCandidate], as_of: date) -> None:
        """Persists next-day short-covering watchlist to database with clean daily refresh and Top-N cap."""
        if not os.getenv("DATABASE_URL") or os.getenv("DISABLE_DB_OI_LOOKUP"):
            return
        
        # RULE 67 RATIONALE: Truncate candidate list to top self.max_watchlist_size (default 35)
        # to ensure intraday Layer 2 monitors only the highest-conviction institutional setups.
        top_candidates = candidates[:self.max_watchlist_size]

        try:
            try:
                from app.database import get_connection
            except ImportError:
                from database import get_connection
            with get_connection(timeout=1) as conn:
                if hasattr(conn, "is_dummy") and conn.is_dummy:
                    return
                with conn.cursor() as cur:
                    # Ensure table exists
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS short_covering_watchlist (
                            symbol TEXT NOT NULL,
                            scan_date DATE NOT NULL,
                            close_price NUMERIC(10,2),
                            total_oi BIGINT,
                            oi_buildup_5d_pct NUMERIC(6,2),
                            short_buildup_ratio NUMERIC(5,2),
                            rsi_14 NUMERIC(5,2),
                            support_level NUMERIC(10,2),
                            overhead_resistance NUMERIC(10,2),
                            atr_14 NUMERIC(10,2),
                            buildup_quality_score NUMERIC(5,2),
                            sector TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            PRIMARY KEY (symbol, scan_date)
                        );
                        CREATE INDEX IF NOT EXISTS idx_sc_watchlist_date ON short_covering_watchlist(scan_date);
                    """)

                    # RULE 67 RATIONALE: Cleanly delete all existing watchlist records for this session date
                    # before inserting the fresh Top-N candidates. This guarantees full idempotency and prevents
                    # stale, rejected, or obsolete stocks from lingering across re-runs.
                    cur.execute("DELETE FROM short_covering_watchlist WHERE scan_date = %s;", (as_of,))

                    for c in top_candidates:
                        cur.execute("""
                            INSERT INTO short_covering_watchlist (
                                symbol, scan_date, close_price, total_oi, oi_buildup_5d_pct,
                                short_buildup_ratio, rsi_14, support_level, overhead_resistance,
                                atr_14, buildup_quality_score, sector
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol, scan_date) DO UPDATE SET
                                close_price = EXCLUDED.close_price,
                                total_oi = EXCLUDED.total_oi,
                                oi_buildup_5d_pct = EXCLUDED.oi_buildup_5d_pct,
                                short_buildup_ratio = EXCLUDED.short_buildup_ratio,
                                rsi_14 = EXCLUDED.rsi_14,
                                support_level = EXCLUDED.support_level,
                                overhead_resistance = EXCLUDED.overhead_resistance,
                                atr_14 = EXCLUDED.atr_14,
                                buildup_quality_score = EXCLUDED.buildup_quality_score,
                                sector = EXCLUDED.sector;
                        """, (
                            c.symbol, c.scan_date, c.close_price, c.total_oi, c.oi_buildup_5d_pct,
                            c.short_buildup_ratio, c.rsi_14, c.support_level, c.overhead_resistance,
                            c.atr_14, c.buildup_quality_score, c.sector
                        ))
                    if hasattr(conn, "commit"):
                        conn.commit()
            logger.info(f"💾 Persisted {len(top_candidates)} top candidates to short_covering_watchlist table (wiped prior records for {as_of})")
        except Exception as e:
            # RULE 67 RATIONALE: Re-raise DB write failure when database is configured so that
            # scanner_health records DOWN / FAILURE instead of a misleading healthy/success state.
            logger.error(f"❌ Database save for short_covering_watchlist failed: {e}")
            if os.getenv("DATABASE_URL") and not os.getenv("DISABLE_DB_OI_LOOKUP"):
                raise



# Global singleton instance
short_position_detector = ShortPositionDetector()
