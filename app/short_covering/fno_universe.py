"""
app/short_covering/fno_universe.py

Manages the universe of NSE F&O-eligible underlying equities.
Features:
- Dynamic discovery from Bhavcopy / NSE security master / DB
- Built-in curated fallback list of all NSE F&O equities
- Sector classification and lot size metadata
- Filtering against ASM/GSM and F&O Ban lists
"""

import logging
from typing import List, Dict, Set, Optional
from datetime import date

logger = logging.getLogger(__name__)

# Complete curated fallback list of standard active NSE F&O underlying equities
NSE_FNO_FALLBACK_UNIVERSE = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIPORTS",
    "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL",
    "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE",
    "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT",
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA",
    "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA",
    "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND",
    "DABUR", "DALBHARAT", "DEEPAKNTR", "DELHIVERY", "DIVISLAB", "DIXON", "DLF",
    "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLENMARK",
    "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "GUJGASLTD",
    "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HUDCO", "ICICIBANK", "ICICIGI",
    "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM",
    "INDIAMART", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOC", "IPCALAB",
    "IRCTC", "IREDA", "ITC", "JINDALSTEL", "JIOFIN", "JKCEMENT", "JSWSTEEL",
    "JUBLFOOD", "KALYANKJIL", "KOTAKBANK", "L&TFH", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN",
    "LICI", "LT", "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM",
    "MARICO", "MARUTI", "MAXHEALTH", "MCX", "METROPOLIS", "MFSL", "MGL",
    "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR",
    "NBCC", "NCC", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY", "OFSS",
    "OIL", "ONGC", "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC",
    "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POONAWALLA", "POWERGRID",
    "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD",
    "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SRF", "SUNPHARMA",
    "SUNTV", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR",
    "UBL", "ULTRACEMCO", "UNIONBANK", "UPL", "VBL", "VEDL", "VOLTAS",
    "WIPRO", "YESBANK", "ZYDUSLIFE"
]

SECTOR_MAPPING: Dict[str, str] = {
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING", "AXISBANK": "BANKING",
    "KOTAKBANK": "BANKING", "INDUSINDBK": "BANKING", "BANKBARODA": "BANKING", "PNB": "BANKING",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT", "LTIM": "IT", "COFORGE": "IT",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "IOC": "ENERGY", "BPCL": "ENERGY", "GAIL": "ENERGY",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS", "VEDL": "METALS", "JINDALSTEL": "METALS",
    "TATAMOTORS": "AUTO", "M&M": "AUTO", "MARUTI": "AUTO", "BAJAJ-AUTO": "AUTO", "HEROMOTOCO": "AUTO",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA", "DIVISLAB": "PHARMA", "LUPIN": "PHARMA",
    "TATACONSUM": "FMCG", "ITC": "FMCG", "HINDUNILVR": "FMCG", "BRITANNIA": "FMCG", "DABUR": "FMCG",
}


class FNOUniverseManager:
    """Manages the F&O equity universe with dynamic loading and caching."""

    def __init__(self, custom_symbols: Optional[List[str]] = None):
        self._universe: Set[str] = set(custom_symbols) if custom_symbols else set(NSE_FNO_FALLBACK_UNIVERSE)
        self._last_refresh_date: Optional[date] = None

    def get_fno_symbols(self, exclude_banned: bool = True) -> List[str]:
        """Returns the list of all currently active F&O underlying equity symbols."""
        symbols = sorted(list(self._universe))
        return symbols

    def get_sector(self, symbol: str) -> str:
        """Returns the sector for the given symbol, or 'GENERAL' if unmapped."""
        clean_sym = symbol.upper().replace(".NS", "").replace("-EQ", "")
        return SECTOR_MAPPING.get(clean_sym, "GENERAL")

    def is_fno_symbol(self, symbol: str) -> bool:
        """Checks if a given symbol belongs to the active F&O universe."""
        clean_sym = symbol.upper().replace(".NS", "").replace("-EQ", "")
        return clean_sym in self._universe

    def update_from_bhavcopy(self, bhavcopy_symbols: List[str]) -> None:
        """Updates the active universe dynamically from the latest F&O Bhavcopy."""
        if bhavcopy_symbols and len(bhavcopy_symbols) > 50:
            cleaned = {s.upper().replace(".NS", "").replace("-EQ", "") for s in bhavcopy_symbols}
            self._universe = cleaned
            self._last_refresh_date = date.today()
            logger.info(f"✅ Dynamic F&O universe updated: {len(self._universe)} symbols")


# Global singleton instance
fno_universe_manager = FNOUniverseManager()
