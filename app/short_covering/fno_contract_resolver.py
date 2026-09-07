"""
app/short_covering/fno_contract_resolver.py

Resolves Near and Next month NSE Futures contracts for underlying equities.
Features:
- Calculates NSE Monthly Expiry dates (last Thursday of the month or adjusted for holidays)
- Generates broker-specific trading symbols (Upstox / Fyers)
- Identifies expiry-week rollover windows to isolate rollover flows from genuine short covering
"""

import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple
from app.short_covering.short_covering_schema import FNOContractInfo

logger = logging.getLogger(__name__)


def get_monthly_expiry(year: int, month: int) -> date:
    """
    Calculates the standard NSE Monthly F&O Expiry date (last Thursday of the month).
    """
    last_day = calendar.monthrange(year, month)[1]
    last_date = date(year, month, last_day)
    # weekday: Monday is 0, Thursday is 3
    offset = (last_date.weekday() - 3) % 7
    expiry_date = last_date - timedelta(days=offset)
    return expiry_date


def get_near_and_next_expiries(as_of: Optional[date] = None) -> Tuple[date, date]:
    """
    Returns (near_expiry, next_expiry) given the current date.
    If today is strictly after the current month's expiry date,
    rolls over to next month as near, and month+2 as next.
    """
    if as_of is None:
        as_of = date.today()

    cur_expiry = get_monthly_expiry(as_of.year, as_of.month)
    if as_of <= cur_expiry:
        near = cur_expiry
        # Next month calculation
        if as_of.month == 12:
            next_exp = get_monthly_expiry(as_of.year + 1, 1)
        else:
            next_exp = get_monthly_expiry(as_of.year, as_of.month + 1)
    else:
        # We are past this month's expiry
        if as_of.month == 12:
            near = get_monthly_expiry(as_of.year + 1, 1)
            next_exp = get_monthly_expiry(as_of.year + 1, 2)
        elif as_of.month == 11:
            near = get_monthly_expiry(as_of.year, 12)
            next_exp = get_monthly_expiry(as_of.year + 1, 1)
        else:
            near = get_monthly_expiry(as_of.year, as_of.month + 1)
            next_exp = get_monthly_expiry(as_of.year, as_of.month + 2)

    return near, next_exp


class FNOContractResolver:
    """Resolves contract specifications and broker tokens for F&O equities."""

    def __init__(self):
        self._cache: Dict[str, FNOContractInfo] = {}

    def resolve(self, symbol: str, as_of: Optional[date] = None) -> FNOContractInfo:
        """
        Resolves the near and next futures contract details for a given underlying symbol.
        """
        if as_of is None:
            as_of = date.today()

        clean_sym = symbol.upper().replace(".NS", "").replace("-EQ", "")
        near_exp, next_exp = get_near_and_next_expiries(as_of)
        days_to_near = (near_exp - as_of).days
        is_expiry_week = days_to_near <= 4

        # Format symbols (e.g., RELIANCE26SEP24FUT / RELIANCE-FUT)
        near_mon_str = near_exp.strftime("%b").upper()
        near_yr_str = near_exp.strftime("%y")
        next_mon_str = next_exp.strftime("%b").upper()
        next_yr_str = next_exp.strftime("%y")

        near_tsym = f"{clean_sym}{near_yr_str}{near_mon_str}FUT"
        next_tsym = f"{clean_sym}{next_yr_str}{next_mon_str}FUT"

        contract_info = FNOContractInfo(
            symbol=clean_sym,
            underlying=clean_sym,
            near_expiry=near_exp,
            next_expiry=next_exp,
            near_trading_symbol=near_tsym,
            next_trading_symbol=next_tsym,
            is_expiry_week=is_expiry_week,
            days_to_near_expiry=days_to_near,
        )
        return contract_info


# Global singleton instance
fno_contract_resolver = FNOContractResolver()
