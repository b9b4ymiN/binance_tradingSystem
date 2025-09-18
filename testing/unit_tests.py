import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.trading_config import TradingConfig
from core.database import DatabaseManager


class TradingBotTests(unittest.TestCase):
    """Comprehensive test suite for trading bot"""

    def setUp(self):
        """Set up test environment"""
        # Test configuration
        self.config = TradingConfig(
            api_key="test_key",
            api_secret="test_secret",
            use_testnet=True
        )

        # Mock database
        self.db_manager = DatabaseManager(":memory:")

    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion position sizing"""
        from core.risk_manager import RiskManager

        risk_manager = RiskManager(self.config, self.db_manager)
        position_size = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            entry_price=50000,
            stop_loss=49000,
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.01
        )

        self.assertGreater(position_size, 0)
        self.assertIsInstance(position_size, float)

    def test_rsi_calculation(self):
        """Test RSI indicator calculation"""
        from analysis.technical_analysis import TechnicalAnalysis

        prices = [50, 51, 52, 51, 50, 49, 50, 51, 52, 53, 52, 51, 50, 49, 48]
        rsi = TechnicalAnalysis.calculate_rsi(prices, 14)

        self.assertIsInstance(rsi, (int, float))
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        from analysis.technical_analysis import TechnicalAnalysis

        prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105] * 3
        upper, middle, lower = TechnicalAnalysis.calculate_bollinger_bands(prices, 20, 2.0)

        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)
        self.assertIsInstance(upper, (int, float))

    @patch('requests.Session.get')
    def test_api_request_handling(self, mock_get):
        """Test API request handling with mocked response"""
        from core.binance_api import BinanceAPI

        api = BinanceAPI(self.config)
        mock_response = Mock()
        mock_response.json.return_value = {'symbol': 'BTCUSDT', 'price': '50000.00'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api._make_request('GET', '/api/v3/ticker/price', {'symbol': 'BTCUSDT'})

        self.assertEqual(result['symbol'], 'BTCUSDT')
        self.assertEqual(result['price'], '50000.00')

    def test_risk_limits_enforcement(self):
        """Test risk limits enforcement"""
        from core.risk_manager import RiskManager

        risk_manager = RiskManager(self.config, self.db_manager)

        # Test with high risk position
        result = risk_manager.check_risk_limits(0.1)  # 10% risk
        self.assertFalse(result)  # Should be rejected

        # Test with acceptable risk
        result = risk_manager.check_risk_limits(0.01)  # 1% risk
        self.assertTrue(result)  # Should be accepted

    def test_database_operations(self):
        """Test database operations"""
        # Test trade logging
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, side, quantity, price, order_id, strategy, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', ('BTCUSDT', 'BUY', 0.001, 50000, 'test_order', 'test_strategy', 'filled'))
            conn.commit()

            # Verify insertion
            cursor.execute('SELECT COUNT(*) FROM trades')
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

    def test_webhook_signature_validation(self):
        """Test webhook signature validation"""
        from webhook.handler import WebhookHandler

        webhook_handler = WebhookHandler(self.config, None)

        # Test signature generation
        payload = b'{"test": "data"}'
        signature = webhook_handler._generate_webhook_signature(payload)

        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 64)  # SHA256 hex digest length


def run_unit_tests():
    """Run all unit tests"""
    unittest.main(module=__name__, exit=False, verbosity=2)