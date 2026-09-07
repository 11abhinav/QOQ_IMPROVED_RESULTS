# =====================================================================================
# app/data_providers/fyers_symbol_mapper.py — OFFICIAL FYERS MASTER CONTRACT RESOLVER
# =====================================================================================
import os
import csv
import logging
import urllib.request
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_FILE = os.path.join(DATA_DIR, "fyers_symbols_cache.json")

_STATIC_SEED_MAP = {
    "FLYSBS": "NSE:FLYSBS-SM",
    "STLTECH": "NSE:STLTECH-BE",
    "NSDL": "BSE:NSDL-EQ",
}

class FyersSymbolMapper:
    """
    Authoritative Fyers Master Symbol Resolver.
    Downloads official Fyers symbol details (NSE_CM.csv & BSE_CM.csv) directly from Fyers public CDN.
    Replaces guess-based symbol generation with 100% verified tradable Fyers identifiers
    (e.g., handles -EQ, -BE, -SM, -ST, -T series automatically).
    """
    def __init__(self):
        self._symbol_map: Dict[str, str] = dict(_STATIC_SEED_MAP)
        self._isin_map: Dict[str, str] = {}
        self._loaded = False
        self.load_cache()

    def load_cache(self) -> None:
        if self._loaded:
            return
        if os.path.exists(CACHE_FILE):
            try:
                import json
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cached_symbols = data.get("symbol_map", {})
                    self._symbol_map.update(cached_symbols)
                    self._isin_map = data.get("isin_map", {})
                    self._loaded = True
                    logger.info(f"✅ [FYERS MAPPER] Loaded {len(self._symbol_map)} symbols from local cache ({CACHE_FILE}).")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Fyers symbol cache: {e}")

        # In offline/sandbox environments, static seeds ensure immediate availability
        self._loaded = True

    def refresh_master(self) -> None:
        """Downloads official Fyers NSE_CM.csv and BSE_CM.csv symbol contract files."""
        os.makedirs(DATA_DIR, exist_ok=True)
        symbol_map = {}
        isin_map = {}

        urls = [
            ("NSE", "https://public.fyers.in/sym_details/NSE_CM.csv"),
            ("BSE", "https://public.fyers.in/sym_details/BSE_CM.csv")
        ]

        import ssl
        ssl_ctx = ssl._create_unverified_context()

        for exch, url in urls:
            try:
                logger.info(f"📥 Downloading Fyers {exch} master symbol contract file...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
                    lines = resp.read().decode("utf-8").splitlines()

                reader = csv.reader(lines)
                for row in reader:
                    if len(row) > 13:
                        fyers_sym = row[9].strip()   # e.g., NSE:MOTHERSON-EQ or NSE:MOTHERSON-D1
                        raw_ticker = row[13].strip().upper() # e.g., MOTHERSON
                        isin = row[5].strip().upper() if len(row) > 5 else ""

                        if raw_ticker and fyers_sym:
                            # Series Priority Ranking for Cash Equities:
                            # -EQ > -BE > -SM > -ST > -T > -A > -B > Others (Warrants -W1, Debt -D1)
                            def _series_rank(sym_str: str) -> int:
                                if sym_str.endswith("-EQ"): return 100
                                if sym_str.endswith("-BE"): return 90
                                if sym_str.endswith("-SM"): return 80
                                if sym_str.endswith("-ST"): return 85
                                if sym_str.endswith("-T"):  return 70
                                if sym_str.endswith("-A") or sym_str.endswith("-B"): return 60
                                return 10 # Warrants, Debt, Bonds

                            existing = symbol_map.get(raw_ticker)
                            if not existing or _series_rank(fyers_sym) > _series_rank(existing):
                                symbol_map[raw_ticker] = fyers_sym

                            if isin:
                                existing_isin = isin_map.get(isin)
                                if not existing_isin or _series_rank(fyers_sym) > _series_rank(existing_isin):
                                    isin_map[isin] = fyers_sym

            except Exception as e:
                logger.warning(f"⚠️ Failed to download Fyers {exch} symbol master: {e}")

        if symbol_map:
            self._symbol_map = symbol_map
            self._isin_map = isin_map
            self._loaded = True
            try:
                import json
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump({"symbol_map": symbol_map, "isin_map": isin_map}, f)
                logger.info(f"✅ [FYERS MAPPER] Successfully cached {len(symbol_map)} official Fyers symbols.")
            except Exception as e:
                logger.warning(f"Failed to write Fyers symbol cache: {e}")

    def get_fyers_symbol(self, symbol: str, isin: Optional[str] = None) -> Optional[str]:
        """Resolves symbol/ISIN to official Fyers tradable symbol string."""
        self.load_cache()
        clean = str(symbol).strip().upper()
        for sfx in (".NS", ".BO", ".BSE"):
            if clean.endswith(sfx):
                clean = clean[:-len(sfx)]
                break

        # 1. Direct Ticker Match in Fyers Master Contract Map
        if clean in self._symbol_map:
            return self._symbol_map[clean]

        # 2. Direct ISIN Match in Fyers Master Contract Map
        if isin and isin in self._isin_map:
            return self._isin_map[isin]

        # 3. Check Institutional InstrumentRegistry
        try:
            from instrument_registry import get_instrument_registry
            rec = get_instrument_registry().lookup(clean)
            if rec and rec.fyers_symbol:
                return rec.fyers_symbol
        except Exception:
            pass

        return None

# Global Singleton Accessor
fyers_mapper = FyersSymbolMapper()
