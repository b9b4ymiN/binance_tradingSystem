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

    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False, retries: int = 3) -> Dict:
        """Make authenticated API request with comprehensive error handling and retry logic"""
        if params is None:
            params = {}

        url = f"{self.base_url}{endpoint}"

        for attempt in range(retries):
            try:
                # Create fresh params for each attempt to avoid signature conflicts
                request_params = params.copy() if params else {}

                if signed:
                    request_params['timestamp'] = int(time.time() * 1000)
                    request_params['recvWindow'] = 60000

                    query_string = '&'.join([f"{k}={v}" for k, v in request_params.items()])
                    request_params['signature'] = self._generate_signature(query_string)

                if method.upper() == 'GET':
                    response = self.session.get(url, params=request_params, timeout=30)
                elif method.upper() == 'POST':
                    response = self.session.post(url, params=request_params, timeout=30)
                elif method.upper() == 'DELETE':
                    response = self.session.delete(url, params=request_params, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if not response.ok:
                    error_data = response.text
                    logger.error(f"API error {response.status_code}: {error_data}")

                    # Handle specific error codes
                    if response.status_code == 429:  # Rate limit
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{retries}")
                        time.sleep(wait_time)
                        continue
                    elif response.status_code == 400:  # Bad Request - check for specific errors
                        try:
                            error_json = response.json()
                            error_code = error_json.get('code')

                            if error_code == -2010:  # MAX_NUM_ALGO_ORDERS
                                logger.error("Maximum number of algo orders reached - cannot place more stop loss orders")
                                # Don't retry for this error - it's a limit issue
                                break
                            elif error_code == -1022:  # Invalid signature
                                if attempt < retries - 1:
                                    logger.warning(f"Invalid signature, retrying with fresh timestamp (attempt {attempt + 1}/{retries})")
                                    time.sleep(0.5)  # Short delay for signature retry
                                    continue
                        except:
                            pass  # If we can't parse the error, continue with normal flow

                    elif response.status_code in [500, 502, 503, 504]:  # Server errors
                        if attempt < retries - 1:
                            wait_time = 1 * (attempt + 1)
                            logger.warning(f"Server error, retrying in {wait_time}s (attempt {attempt + 1}/{retries})")
                            time.sleep(wait_time)
                            continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as e:
                if attempt < retries - 1:
                    logger.warning(f"Request timeout, retrying (attempt {attempt + 1}/{retries})")
                    time.sleep(1)
                    continue
                logger.error(f"API request timed out after {retries} attempts: {e}")
                raise
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    logger.warning(f"Request failed, retrying (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(1)
                    continue
                logger.error(f"API request failed after {retries} attempts: {e}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {e}")
                raise

        # This should never be reached, but just in case
        raise Exception(f"Request failed after {retries} attempts")

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
        order_type_upper = order_type.upper()
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type_upper,
            'quantity': round(quantity, 5)
        }

        if order_type_upper in {'LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT', 'LIMIT_MAKER'}:
            if price is None:
                raise ValueError(f"Price is required for {order_type_upper} orders")
            params['price'] = price

        if order_type_upper in {'LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT'}:
            params['timeInForce'] = time_in_force

        if order_type_upper in {'STOP_LOSS', 'TAKE_PROFIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT'}:
            if stop_price is None:
                raise ValueError(f"stop_price is required for {order_type_upper} orders")
            params['stopPrice'] = stop_price

        if order_type_upper == 'MARKET':
            params.pop('timeInForce', None)
            params.pop('price', None)

        logger.info(f"Placing {side} order for {quantity} {symbol} at {price}")
        return self._make_request('POST', '/api/v3/order', params, signed=True)

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an existing order"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('DELETE', '/api/v3/order', params, signed=True)

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get currently open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/api/v3/openOrders', params, signed=True)

    def cleanup_old_algo_orders(self, symbol: str = None, max_orders: int = 3) -> int:
        """Clean up old algorithmic orders to stay under the limit"""
        try:
            open_orders = self.get_open_orders(symbol)

            # Filter for algorithmic order types (STOP_LOSS, STOP_LOSS_LIMIT, etc.)
            algo_orders = [
                order for order in open_orders
                if order.get('type') in ['STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']
            ]

            logger.info(f"Found {len(algo_orders)} algorithmic orders")

            if len(algo_orders) >= max_orders:
                # Sort by creation time (oldest first)
                algo_orders.sort(key=lambda x: x.get('time', 0))

                # Cancel oldest orders to make room
                orders_to_cancel = len(algo_orders) - max_orders + 1
                cancelled_count = 0

                for order in algo_orders[:orders_to_cancel]:
                    try:
                        self.cancel_order(order['symbol'], order['orderId'])
                        logger.info(f"Cancelled old algo order {order['orderId']} ({order['type']})")
                        cancelled_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order['orderId']}: {e}")

                return cancelled_count

            return 0

        except Exception as e:
            logger.error(f"Failed to cleanup algo orders: {e}")
            return 0

    def place_order_with_cleanup(self, symbol: str, side: str, order_type: str,
                                quantity: float, price: float = None,
                                stop_price: float = None, time_in_force: str = None) -> Dict:
        """Place order with automatic cleanup of old algo orders if needed"""

        # If this is an algo order, try cleanup first
        if order_type in ['STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']:
            try:
                self.cleanup_old_algo_orders(symbol, max_orders=3)
            except Exception as e:
                logger.warning(f"Cleanup failed, continuing with order placement: {e}")

        # Place the order
        return self.place_order(symbol, side, order_type, quantity, price, stop_price, time_in_force)


