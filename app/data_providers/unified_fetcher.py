import logging
import pandas as pd
from typing import Optional, Dict
from .fyers_fetcher import FyersFetcher
from .provider_selector import selector
from data_registry import registry
from yf_rate_limiter import acquire as yf_acquire, release as yf_release

import threading

logger = logging.getLogger(__name__)

# Resource-specific lock for all external provider fetches
network_fetch_lock = threading.Lock()

class UnifiedFetcher:
    """
    Data requested through this unified fetcher is logged in DatasetRegistry.
    Provider fallback policy is governed by ProviderSelector.
    """
    def __init__(self):
        self.registry = registry
        self.selector = selector
        self.fyers = FyersFetcher()

    def fetch_historical(self, symbol: str, interval: str, period: str, consumer: str) -> Optional[pd.DataFrame]:
        from config import NON_EQUITY_BLOCKLIST
        clean_sym = str(symbol).replace('.NS', '').replace('.BO', '').replace('BSE:', '').replace('NSE:', '').strip().upper()
        if clean_sym in NON_EQUITY_BLOCKLIST:
            logger.info(f"ℹ️ Skipping non-equity trust {symbol} in UnifiedFetcher")
            return pd.DataFrame()

        logger.info(f"[{consumer}] Fetching {symbol} ({interval} / {period}) via UnifiedFetcher")
        
        dataset_id = f"price_{interval}"
        if self.registry.get_entry(dataset_id):
            self.registry.register_consumer(dataset_id, consumer)
            
        providers = self.selector.get_providers(dataset_id, fetch_type="historical")
        
        for provider in providers:
            if provider == "fyers":
                try:
                    md = self.fyers.get_ohlcv(symbol, interval=interval, period=period)
                    if md is not None and md.df is not None and not md.df.empty:
                        logger.info(f"✅ [Fyers] Successfully fetched historical {symbol}")
                        entry = self.registry.get_entry(dataset_id)
                        if entry:
                            entry.provider_used = "fyers"
                            is_fallback = entry.preferred_provider and provider != entry.preferred_provider
                            from datetime import datetime
                            md.df.attrs = {
                                "dataset": dataset_id,
                                "provider": provider,
                                "preferred_provider": entry.preferred_provider,
                                "fallback_used": bool(is_fallback),
                                "fetch_timestamp": datetime.now().isoformat()
                            }
                        from trading_calendar import enforce_trading_day_candles
                        return enforce_trading_day_candles(md.df, symbol)
                except Exception as e:
                    logger.warning(f"⚠️ [Fyers] Failed to fetch historical {symbol}: {e}")
            
            elif provider == "upstox":
                try:
                    from market_data.providers.upstox_provider import UpstoxProvider
                    upstox_fetcher = UpstoxProvider(auth_service=None)
                    md = upstox_fetcher.get_ohlcv(symbol, interval=interval, period=period)
                    if md is not None and getattr(md, 'dataframe', None) is not None and not md.dataframe.empty:
                        logger.info(f"✅ [Upstox] Successfully fetched historical {symbol}")
                        entry = self.registry.get_entry(dataset_id)
                        if entry:
                            entry.provider_used = "upstox"
                            is_fallback = entry.preferred_provider and provider != entry.preferred_provider
                            from datetime import datetime
                            md.dataframe.attrs = {
                                "dataset": dataset_id,
                                "provider": provider,
                                "preferred_provider": entry.preferred_provider,
                                "fallback_used": bool(is_fallback),
                                "fetch_timestamp": datetime.now().isoformat()
                            }
                        from trading_calendar import enforce_trading_day_candles
                        return enforce_trading_day_candles(md.dataframe, symbol)
                except Exception as e:
                    logger.warning(f"⚠️ [Upstox] Failed to fetch historical {symbol}: {e}")


        logger.error(f"❌ Exhausted all providers for historical {symbol}")
        return pd.DataFrame()

    def fetch_live_quotes(self, symbols: list[str], consumer: str) -> dict[str, dict]:
        """
        Fetches live snapshot data (quotes) for a list of symbols.
        Uses ProviderSelector for routing.
        Returns a dict of symbol -> quote data mapping.
        """
        valid_symbols = []
        for s in symbols:
            if not s or not isinstance(s, str):
                continue
            clean = str(s).replace('.NS', '').replace('.BO', '').replace('BSE:', '').replace('NSE:', '').strip().upper()
            if clean and clean not in ('?', 'NONE', 'NAN', 'NULL', 'UNKNOWN') and any(c.isalnum() for c in clean):
                valid_symbols.append(s)

        if not valid_symbols:
            return {}
        symbols = valid_symbols

        with network_fetch_lock:
            logger.info(f"[{consumer}] Fetching live quotes for {len(symbols)} symbols via UnifiedFetcher")
            dataset_id = "live_quotes"
            if self.registry.get_entry(dataset_id):
                self.registry.register_consumer(dataset_id, consumer)
                
            providers = self.selector.get_providers(dataset_id, fetch_type="live_quotes")
            results: Dict[str, dict] = {}
            pending = set(symbols)
            results_lock = threading.Lock()
            
            for provider in providers:
                if not pending:
                    break
                    
                if provider == "fyers":
                    logger.info(f"🔄 [Fyers] Fetching live quotes for {len(pending)} symbols...")
                    import concurrent.futures

                    def fetch_fyers_chunk(chunk):
                        fyers_map = {}
                        for orig in chunk:
                            norm = self.fyers._normalize_symbol(orig)
                            if norm and (norm.startswith("NSE:") or norm.startswith("BSE:") or norm.startswith("MCX:")):
                                fyers_map[norm] = orig

                        if not fyers_map:
                            return

                        try:
                            from fyers_auth import get_fyers_client
                            fyers_client = get_fyers_client()
                            if fyers_client:
                                fyers_symbols_str = ",".join(fyers_map.keys())
                                resp = fyers_client.quotes({"symbols": fyers_symbols_str})

                                if resp and isinstance(resp, dict) and resp.get("s") == "ok":
                                    success_count = 0
                                    for item in resp.get("d", []):
                                        if item.get("s") == "ok" and "v" in item and "lp" in item["v"]:
                                            sym_name = item.get("n")
                                            orig = fyers_map.get(sym_name)
                                            if orig:
                                                val = item["v"]["lp"]
                                                clean_orig = orig.replace(".NS", "").replace(".BO", "")
                                                with results_lock:
                                                    results[orig] = {"v": {"cmd": {"c": val}}}
                                                    results[clean_orig] = {"v": {"cmd": {"c": val}}}
                                                    results[clean_orig + ".NS"] = {"v": {"cmd": {"c": val}}}
                                                    pending.discard(orig)
                                                    pending.discard(clean_orig)
                                                    pending.discard(clean_orig + ".NS")
                                                logger.debug(f"✅ [Fyers] Successfully fetched live quote for {orig} ({sym_name}): ₹{val:.2f}")
                                                success_count += 1
                                    if success_count > 0:
                                        logger.info(f"✅ [Fyers] Fetched {success_count}/{len(fyers_map)} quotes successfully.")
                                else:
                                    logger.warning(f"⚠️ [Fyers] Quote batch response not ok for {len(fyers_map)} symbols: {resp}")
                                    code = resp.get("code") if isinstance(resp, dict) else None
                                    msg = str(resp.get("message", "")).lower() if isinstance(resp, dict) else ""
                                    if str(code) in ["-15", "-16", "401", "-401", "494"] or "valid token" in msg or "authenticate" in msg:
                                        logger.error("🚫 Fyers token invalid/expired during live quotes batch. Triggering auto-login...")
                                        from fyers_auth import clear_token, auto_login, get_fyers_client
                                        clear_token(force=True)
                                        if auto_login():
                                            new_client = get_fyers_client()
                                            if new_client:
                                                resp2 = new_client.quotes({"symbols": fyers_symbols_str})
                                                if resp2 and isinstance(resp2, dict) and resp2.get("s") == "ok":
                                                    success_count = 0
                                                    for item in resp2.get("d", []):
                                                        if item.get("s") == "ok" and "v" in item and "lp" in item["v"]:
                                                            sym_name = item.get("n")
                                                            orig = fyers_map.get(sym_name)
                                                            if orig:
                                                                val = item["v"]["lp"]
                                                                clean_orig = orig.replace(".NS", "").replace(".BO", "")
                                                                with results_lock:
                                                                    results[orig] = {"v": {"cmd": {"c": val}}}
                                                                    results[clean_orig] = {"v": {"cmd": {"c": val}}}
                                                                    results[clean_orig + ".NS"] = {"v": {"cmd": {"c": val}}}
                                                                    pending.discard(orig)
                                                                    pending.discard(clean_orig)
                                                                    pending.discard(clean_orig + ".NS")
                                                                logger.debug(f"✅ [Fyers] Successfully fetched live quote for {orig} ({sym_name}) on RETRY: ₹{val:.2f}")
                                                                success_count += 1
                                                    if success_count > 0:
                                                        logger.info(f"✅ [Fyers] Fetched {success_count}/{len(fyers_map)} quotes successfully on RETRY.")
                        except Exception as e:
                            logger.warning(f"⚠️ [Fyers] Batch quote fetch failed: {e}")

                    pending_list = list(pending)
                    chunk_size = 50
                    chunks = [pending_list[i:i+chunk_size] for i in range(0, len(pending_list), chunk_size)]
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(chunks) or 1)) as executor:
                        executor.map(fetch_fyers_chunk, chunks)
                        
                elif provider == "upstox":
                    logger.info(f"🔄 [Upstox] Fetching live quotes for {len(pending)} symbols...")
                    try:
                        from market_data.providers.upstox_provider import UpstoxProvider
                        upstox_fetcher = UpstoxProvider(auth_service=None)
                        pending_list = list(pending)
                        
                        chunk_size = 500
                        chunks = [pending_list[i:i+chunk_size] for i in range(0, len(pending_list), chunk_size)]
                        
                        for chunk in chunks:
                            resp = upstox_fetcher.fetch_live_quotes_batch(chunk)
                            if resp:
                                success_count = 0
                                for orig in chunk:
                                    try:
                                        raw_key = upstox_fetcher._get_instrument_key(orig)
                                        clean_sym = raw_key.split(":")[-1].split("|")[-1]
                                        clean_orig = orig.replace(".NS", "").replace(".BO", "")
                                        quote_data = (resp.get(orig) or resp.get(clean_orig) or resp.get(clean_orig + ".NS") or
                                                      resp.get(clean_sym) or resp.get(raw_key) or resp.get(raw_key.replace("|", ":")))
                                        
                                        if quote_data and isinstance(quote_data, dict):
                                            val = quote_data.get("last_price") or quote_data.get("lp") or quote_data.get("cp")
                                            if (val is None or float(val or 0) <= 0) and "ohlc" in quote_data and isinstance(quote_data["ohlc"], dict):
                                                val = quote_data["ohlc"].get("close") or quote_data["ohlc"].get("open")
                                            if val is not None and float(val) > 0:
                                                val_flt = float(val)
                                                results[orig] = {"v": {"cmd": {"c": val_flt}}}
                                                results[clean_orig] = {"v": {"cmd": {"c": val_flt}}}
                                                results[clean_orig + ".NS"] = {"v": {"cmd": {"c": val_flt}}}
                                                logger.debug(f"✅ [Upstox] Successfully fetched live quote for {orig}: ₹{val_flt:.2f}")
                                                pending.discard(orig)
                                                pending.discard(clean_orig)
                                                pending.discard(clean_orig + ".NS")
                                                success_count += 1
                                    except Exception as item_err:
                                        logger.error(f"❌ [Upstox] Quote parsing error for symbol {orig}: {item_err}", exc_info=True)
                                if success_count > 0:
                                    logger.info(f"✅ [Upstox] Fetched {success_count}/{len(chunk)} quotes successfully.")
                    except Exception as e:
                        logger.error(f"❌ [Upstox] Batch quote fetch failed: {e}", exc_info=True)

                elif provider == "yahoo":
                    # ── DB CMP FALLBACK BEFORE YAHOO ─────────────────────────────
                    # Try resolving pending stock symbols from Postgres DB master table & bhavcopy first
                    # so web scraping Yahoo Finance is strictly a last resort for index tickers.
                    if pending:
                        try:
                            from database import get_connection
                            with get_connection() as conn:
                                with conn.cursor() as cur:
                                    for orig in list(pending):
                                        if orig not in ("NIFTY 50", "BANKNIFTY", "SENSEX", "^NSEI", "^NSEBANK", "^BSESN"):
                                            clean_orig = orig.replace(".NS", "").replace(".BO", "")
                                            cur.execute("""
                                                SELECT val FROM (
                                                    SELECT cmp AS val, 1 AS prio FROM stock_analysis_master WHERE (symbol = %s OR symbol = %s) AND cmp IS NOT NULL AND cmp > 0
                                                    UNION ALL
                                                    SELECT latest_price AS val, 2 AS prio FROM watchlist WHERE (symbol = %s OR symbol = %s) AND latest_price IS NOT NULL AND latest_price > 0
                                                    UNION ALL
                                                    SELECT COALESCE(current_price, entry_price) AS val, 3 AS prio FROM alerts WHERE (symbol = %s OR symbol = %s) AND (current_price > 0 OR entry_price > 0)
                                                ) sub ORDER BY prio LIMIT 1;
                                            """, (orig, clean_orig, orig, clean_orig, orig, clean_orig))
                                            row = cur.fetchone()
                                            if row and row[0]:
                                                val_flt = float(row[0])
                                                results[orig] = {"v": {"cmd": {"c": val_flt}}}
                                                results[clean_orig] = {"v": {"cmd": {"c": val_flt}}}
                                                results[clean_orig + ".NS"] = {"v": {"cmd": {"c": val_flt}}}
                                                pending.discard(orig)
                                                pending.discard(clean_orig)
                                                pending.discard(clean_orig + ".NS")
                                                logger.info(f"⚡ [DB CMP FALLBACK] Resolved quote for {orig} from PostgreSQL master: ₹{val_flt:.2f}")
                        except Exception as db_err:
                            logger.warning(f"⚠️ DB CMP Fallback prior to Yahoo failed: {db_err}")

                    # [VERSION: ZERO_YAHOO_LIVE_QUOTES_v1.0] Yahoo/BSE live price fetching disabled to prevent YFRateLimitError & network delays.
                    # All remaining symbols are resolved directly from Postgres DB master table.
                    pass
                        
                elif provider == "bse":
                    pass

        if pending:
            try:
                from database import get_connection
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        for orig in list(pending):
                            clean_orig = orig.replace(".NS", "").replace(".BO", "")
                            cur.execute("""
                                SELECT val FROM (
                                    SELECT cmp AS val, 1 AS prio FROM stock_analysis_master WHERE (symbol = %s OR symbol = %s) AND cmp IS NOT NULL AND cmp > 0
                                    UNION ALL
                                    SELECT latest_price AS val, 2 AS prio FROM watchlist WHERE (symbol = %s OR symbol = %s) AND latest_price IS NOT NULL AND latest_price > 0
                                    UNION ALL
                                    SELECT COALESCE(current_price, entry_price) AS val, 3 AS prio FROM alerts WHERE (symbol = %s OR symbol = %s) AND (current_price > 0 OR entry_price > 0)
                                ) sub ORDER BY prio LIMIT 1;
                            """, (orig, clean_orig, orig, clean_orig, orig, clean_orig))
                            row = cur.fetchone()
                            if row and row[0]:
                                val_flt = float(row[0])
                                results[orig] = {"v": {"cmd": {"c": val_flt}}}
                                results[clean_orig] = {"v": {"cmd": {"c": val_flt}}}
                                results[clean_orig + ".NS"] = {"v": {"cmd": {"c": val_flt}}}
                                pending.discard(orig)
                                pending.discard(clean_orig)
                                pending.discard(clean_orig + ".NS")
                                logger.info(f"⚡ [DB CMP FALLBACK] Resolved fallback quote for {orig} from stock_analysis_master: ₹{val_flt:.2f}")
            except Exception as db_err:
                logger.warning(f"⚠️ DB CMP Fallback failed: {db_err}")

        if pending:
            logger.warning(f"⚠️ Live quotes unavailable from providers for ({len(pending)}): {sorted(list(pending))}")
            for p_sym in list(pending):
                clean_p = p_sym.replace(".NS", "").replace(".BO", "")
                results[p_sym] = {"v": {"cmd": {"c": None}}}
                results[clean_p] = {"v": {"cmd": {"c": None}}}
            
        if self.registry.get_entry(dataset_id):
            self.registry.get_entry(dataset_id).provider_used = "live_batch"
            
        return results

# Global instance
fetcher = UnifiedFetcher()
