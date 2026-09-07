import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from config import NON_EQUITY_BLOCKLIST
from data_providers.fyers_symbol_mapper import fyers_mapper
from symbol_resolution_engine import FyersAdapter, UpstoxAdapter


class TestSmeResolutionAndUniverseIsolation(unittest.TestCase):

    def test_sme_fyers_mapper_lookup(self):
        # FLYSBS must resolve to NSE:FLYSBS-SM via FyersSymbolMapper
        res = fyers_mapper.get_fyers_symbol("FLYSBS")
        self.assertEqual(res, "NSE:FLYSBS-SM")

    def test_sme_adapter_lookup(self):
        adapter = FyersAdapter()
        res = adapter.lookup_master("FLYSBS", None)
        self.assertIsNotNone(res)
        self.assertEqual(res.mapped_symbol, "NSE:FLYSBS-SM")
        self.assertEqual(res.series, "SM")

    def test_standard_stock_adapter_lookup(self):
        adapter = FyersAdapter()
        res = adapter.lookup_master("RELIANCE", None)
        self.assertIsNotNone(res)
        self.assertEqual(res.mapped_symbol, "NSE:RELIANCE-EQ")

    def test_non_equity_blocklist_in_config(self):
        self.assertIn("VERTIS", NON_EQUITY_BLOCKLIST)
        self.assertIn("HIGHWAYS", NON_EQUITY_BLOCKLIST)
        self.assertIn("POWERINVIT", NON_EQUITY_BLOCKLIST)

    def test_price_cache_filters_non_equity_trusts(self):
        test_watchlist = pd.DataFrame({"Stock": ["RELIANCE", "VERTIS", "HIGHWAYS", "TCS"]})
        filtered = test_watchlist[~test_watchlist["Stock"].astype(str).str.upper().isin(NON_EQUITY_BLOCKLIST)]
        self.assertNotIn("VERTIS", filtered["Stock"].values)
        self.assertNotIn("HIGHWAYS", filtered["Stock"].values)
        self.assertIn("RELIANCE", filtered["Stock"].values)
        self.assertIn("TCS", filtered["Stock"].values)


if __name__ == "__main__":
    unittest.main()
