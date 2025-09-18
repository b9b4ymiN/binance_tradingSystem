import hmac
import hashlib
import requests
import time
import json
import logging
from typing import Dict, List
from config.trading_config import TradingConfig

logger = logging.getLogger(__name__)


class BinanceAPI:
    """Comprehensive Binance API integration with advanced features"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.base_url = config.testnet_url if config.use_testnet else config.base_url
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': config.api_key,
            'Content-Type': 'application/json'
        })

    def _generate_signature(self, params: str) -> str:
        """Generate HMAC SHA256 signature for API requests"""
        return hmac.new(
            self.config.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make authenticated API request with comprehensive error handling"""
        if params is None:
            params = {}

        url = f"{self.base_url}{endpoint}"

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 60000

            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, params=params, timeout=30)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            #logger.info(f"response : {response.url}")
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise

    def get_account_info(self) -> Dict:
        """Get comprehensive account information"""
        return self._make_request('GET', '/api/v3/account', signed=True)

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed symbol information including filters"""
        exchange_info = self._make_request('GET', '/api/v3/exchangeInfo')

        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info

        raise ValueError(f"Symbol {symbol} not found")

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        response = self._make_request(
            'GET', '/api/v3/ticker/price', {'symbol': symbol})
        return float(response['price'])

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        """Get historical kline data for technical analysis"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return self._make_request('GET', '/api/v3/klines', params)

    def place_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: float = None,
                    stop_price: float = None, time_in_force: str = 'GTC') -> Dict:
        """Place comprehensive order with all order types"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': round(quantity, 5),
            'timeInForce': time_in_force
        }
        if order_type.upper() == 'MARKET':
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': round(quantity, 5)
            }

        if price:
            params['price'] = price
        if stop_price:
            params['stopPrice'] = stop_price

        logger.info(f"Placing {side} order for {quantity} {symbol} at {price}")
        return self._make_request('POST', '/api/v3/order', params, signed=True)

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel existing order"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('DELETE', '/api/v3/order', params, signed=True)

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/api/v3/openOrders', params, signed=True)
