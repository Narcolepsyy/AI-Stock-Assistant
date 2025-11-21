import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import stock_service

class TestStockServiceRefactor(unittest.TestCase):
    def setUp(self):
        self.mock_ticker = MagicMock()
        self.mock_ticker.history.return_value = None
        self.mock_ticker.info = {"currentPrice": 150.0, "currency": "USD"}
        self.mock_ticker.fast_info = {"currency": "USD"}
        
    @patch("app.services.stock.providers.yfinance.yf.Ticker")
    def test_get_stock_quote(self, mock_yf):
        mock_yf.return_value = self.mock_ticker
        # Mock history data
        import pandas as pd
        data = {
            "Open": [148.0], "High": [152.0], "Low": [147.0], "Close": [150.0], "Volume": [1000000]
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex(["2023-01-01"]))
        self.mock_ticker.history.return_value = df
        
        result = stock_service.get_stock_quote("AAPL")
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["price"], 150.0)
        self.assertEqual(result["source"], "yfinance")

    @patch("app.services.stock.providers.yfinance.yf.Ticker")
    def test_get_company_profile(self, mock_yf):
        mock_yf.return_value = self.mock_ticker
        self.mock_ticker.get_info.return_value = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "currency": "USD"
        }
        
        result = stock_service.get_company_profile("AAPL")
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["longName"], "Apple Inc.")

    def test_normalization(self):
        from app.services.stock.normalization import normalize_symbol
        self.assertEqual(normalize_symbol("aapl"), "AAPL")
        self.assertEqual(normalize_symbol("8306"), "8306.T")

if __name__ == "__main__":
    unittest.main()
