from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from datetime import datetime, timedelta
from typing import Dict, List
import sqlite3
import logging
from .cache_manager import cache, cached

logger = logging.getLogger(__name__)


class DashboardAPI:
    """API endpoints for dashboard data"""

    def __init__(self, trading_engine, db_manager):
        self.trading_engine = trading_engine
        self.db_manager = db_manager
        self.lockless_storage = trading_engine.lockless_storage  # Access lockless storage
        self.app = Flask(__name__)
        CORS(self.app)
        self._register_routes()

    def _register_routes(self):
        """Register all API routes"""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'trading-dashboard-api'
            })

        @self.app.route('/api/dashboard/system-health', methods=['GET'])
        def get_system_health():
            """Get system health metrics"""
            try:
                return jsonify(self._get_system_health())
            except Exception as e:
                logger.error(f"Error getting system health: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/overview', methods=['GET'])
        def get_dashboard_overview():
            """Get dashboard overview data"""
            try:
                # Use cached data where possible for overview
                cache_key = "dashboard_overview"
                cached_overview = cache.get(cache_key)

                if cached_overview:
                    return jsonify(cached_overview)

                overview_data = {
                    'performance': self._get_performance_metrics(),
                    'positions': self._get_current_positions(),
                    'system_health': self._get_system_health(),
                    'recent_trades': self._get_recent_trades(limit=10),
                    'account_balance': self._get_account_balance()
                }

                # Cache the complete overview for 15 seconds
                cache.set(cache_key, overview_data, 15)

                return jsonify(overview_data)
            except Exception as e:
                logger.error(f"Error getting dashboard overview: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/performance', methods=['GET'])
        def get_performance_metrics():
            """Get detailed performance metrics"""
            try:
                return jsonify(self._get_performance_metrics())
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/positions', methods=['GET'])
        def get_positions():
            """Get current positions"""
            try:
                return jsonify(self._get_current_positions())
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/positions/binance', methods=['GET'])
        def get_binance_positions():
            """Get real-time positions from Binance"""
            try:
                return jsonify(self._get_binance_positions())
            except Exception as e:
                logger.error(f"Error getting Binance positions: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/positions/enhanced', methods=['GET'])
        def get_enhanced_positions():
            """Get enhanced positions with both database and Binance data"""
            try:
                return jsonify(self._get_enhanced_positions())
            except Exception as e:
                logger.error(f"Error getting enhanced positions: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/open-orders', methods=['GET'])
        def get_open_orders():
            """Get current open orders"""
            try:
                return jsonify(self._get_open_orders())
            except Exception as e:
                logger.error(f"Error getting open orders: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/account-balance', methods=['GET'])
        def get_account_balance():
            """Get account balance information"""
            try:
                return jsonify(self._get_account_balance())
            except Exception as e:
                logger.error(f"Error getting account balance: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/trades', methods=['GET'])
        def get_trades():
            """Get trade history"""
            try:
                limit = request.args.get('limit', 50, type=int)
                symbol = request.args.get('symbol')
                return jsonify(self._get_recent_trades(limit, symbol))
            except Exception as e:
                logger.error(f"Error getting trades: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/strategies', methods=['GET'])
        def get_strategy_performance():
            """Get strategy performance data"""
            try:
                return jsonify(self._get_strategy_performance())
            except Exception as e:
                logger.error(f"Error getting strategy performance: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/charts/pnl', methods=['GET'])
        def get_pnl_chart_data():
            """Get P&L chart data"""
            try:
                days = request.args.get('days', 30, type=int)
                return jsonify(self._get_pnl_chart_data(days))
            except Exception as e:
                logger.error(f"Error getting P&L chart data: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/charts/portfolio', methods=['GET'])
        def get_portfolio_chart_data():
            """Get portfolio value chart data"""
            try:
                days = request.args.get('days', 30, type=int)
                return jsonify(self._get_portfolio_chart_data(days))
            except Exception as e:
                logger.error(f"Error getting portfolio chart data: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/alerts', methods=['GET'])
        def get_alerts():
            """Get system alerts"""
            try:
                return jsonify(self._get_system_alerts())
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/cache/status', methods=['GET'])
        def get_cache_status():
            """Get cache statistics"""
            try:
                return jsonify(cache.stats())
            except Exception as e:
                logger.error(f"Error getting cache status: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/cache/clear', methods=['POST'])
        def clear_cache():
            """Clear all cache entries"""
            try:
                cache.clear()
                return jsonify({'message': 'Cache cleared successfully'})
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/cache/invalidate', methods=['POST'])
        def invalidate_cache():
            """Invalidate specific cache patterns"""
            try:
                data = request.get_json() or {}
                pattern = data.get('pattern', '')

                if pattern:
                    cache.invalidate_pattern(pattern)
                    return jsonify({'message': f'Cache pattern "{pattern}" invalidated successfully'})
                else:
                    return jsonify({'error': 'Pattern parameter required'}), 400
            except Exception as e:
                logger.error(f"Error invalidating cache: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/refresh', methods=['POST'])
        def force_refresh():
            """Force refresh all dashboard data"""
            try:
                # Clear relevant cache entries
                cache.invalidate_pattern("perf_")
                cache.invalidate_pattern("pos_")
                cache.delete("dashboard_overview")

                # Force update performance and portfolio
                self.trading_engine._update_daily_performance()
                self.trading_engine._sync_portfolio_to_database()

                return jsonify({'message': 'Dashboard data refreshed successfully'})
            except Exception as e:
                logger.error(f"Error refreshing dashboard: {e}")
                return jsonify({'error': str(e)}), 500

    @cached(ttl_seconds=30, key_prefix="perf_")
    def _get_performance_metrics(self) -> Dict:
        """Calculate performance metrics using database performance table and current data"""
        try:
            # Get performance metrics from database table (faster than calculating from trades)
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get overall performance from database
                cursor.execute("""
                    SELECT
                        SUM(total_pnl) as total_pnl,
                        SUM(total_trades) as total_trades,
                        AVG(win_rate) as avg_win_rate,
                        MIN(max_drawdown) as max_drawdown
                    FROM performance
                """)
                perf_result = cursor.fetchone()

                if perf_result and perf_result[0] is not None:
                    total_pnl, total_trades, avg_win_rate, max_drawdown = perf_result
                else:
                    # Fallback to lockless calculation if no performance data
                    lockless_metrics = self.lockless_storage.calculate_performance_metrics()
                    total_pnl = lockless_metrics.get('total_pnl', 0)
                    total_trades = lockless_metrics.get('total_trades', 0)
                    avg_win_rate = lockless_metrics.get('win_rate', 0)
                    max_drawdown = 0

                # Get today's performance
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT total_pnl, total_trades, win_rate
                    FROM performance
                    WHERE date = ?
                """, (today,))
                today_result = cursor.fetchone()

                if today_result:
                    daily_pnl, daily_trades, daily_win_rate = today_result
                else:
                    # Calculate from recent trades if no daily performance record
                    cursor.execute("""
                        SELECT
                            SUM(realized_pnl) as daily_pnl,
                            COUNT(*) as daily_trades,
                            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                        FROM trades
                        WHERE DATE(timestamp) = ? AND status = 'FILLED'
                    """, (today,))
                    result = cursor.fetchone()
                    daily_pnl = result[0] or 0
                    daily_trades = result[1] or 0
                    winning_today = result[2] or 0
                    daily_win_rate = (winning_today / daily_trades * 100) if daily_trades > 0 else 0

                # Calculate winning/losing trades
                cursor.execute("""
                    SELECT
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                        AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
                        AVG(CASE WHEN realized_pnl <= 0 THEN realized_pnl END) as avg_loss
                    FROM trades
                    WHERE status = 'FILLED' AND realized_pnl != 0
                """)
                trade_stats = cursor.fetchone()
                winning_trades = trade_stats[0] or 0
                losing_trades = trade_stats[1] or 0
                avg_win = trade_stats[2] or 0
                avg_loss = abs(trade_stats[3]) if trade_stats[3] else 0

            # Get current positions for market value calculation from database
            positions = self._get_current_positions_from_db()
            total_market_value = sum(abs(pos['quantity']) * pos['avg_price'] for pos in positions)

            # Calculate portfolio value (simplified - starting balance + P&L)
            portfolio_value = 10000 + total_pnl  # Assuming $10k starting balance
            available_balance = portfolio_value * 0.8  # 80% available for trading

            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = 0
            if max_drawdown and max_drawdown != 0:
                sharpe_ratio = total_pnl / abs(max_drawdown)

            return {
                'total_pnl': round(total_pnl, 2),
                'daily_pnl': round(daily_pnl, 2),
                'total_trades': int(total_trades or 0),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'win_rate': round(avg_win_rate or 0, 1),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'max_drawdown': round(abs(max_drawdown) if max_drawdown else 0, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'portfolio_value': round(portfolio_value, 2),
                'available_balance': round(available_balance, 2),
                'total_market_value': round(total_market_value, 2),
                'positions_count': len(positions),
                'daily_trades': int(daily_trades or 0)
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            # Fallback to lockless storage
            try:
                lockless_metrics = self.lockless_storage.calculate_performance_metrics()
                return {
                    'total_pnl': round(lockless_metrics.get('total_pnl', 0), 2),
                    'daily_pnl': 0,
                    'total_trades': lockless_metrics.get('total_trades', 0),
                    'winning_trades': lockless_metrics.get('winning_trades', 0),
                    'losing_trades': lockless_metrics.get('losing_trades', 0),
                    'win_rate': round(lockless_metrics.get('win_rate', 0), 1),
                    'avg_win': 0,
                    'avg_loss': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'portfolio_value': 10000,
                    'available_balance': 8000,
                    'total_market_value': 0,
                    'positions_count': 0,
                    'daily_trades': 0
                }
            except:
                return {}

    def _get_current_positions_from_db(self) -> List[Dict]:
        """Get current positions from database table with enhanced market data"""
        try:
            # First, update all positions with current market prices
            self._update_positions_with_current_prices()

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        symbol, quantity, avg_price, entry_price, current_price,
                        market_value, unrealized_pnl, unrealized_pnl_percent,
                        position_size_usd, last_updated, last_price_update
                    FROM portfolio
                    WHERE ABS(quantity) > 0.000001
                    ORDER BY ABS(COALESCE(position_size_usd, quantity * avg_price)) DESC
                """)

                positions = []
                for row in cursor.fetchall():
                    (symbol, quantity, avg_price, entry_price, current_price,
                     market_value, unrealized_pnl, unrealized_pnl_percent,
                     position_size_usd, last_updated, last_price_update) = row

                    # Fallback calculations if database values are None
                    if current_price is None:
                        try:
                            current_price = self.trading_engine.binance_api.get_current_price(symbol)
                        except:
                            current_price = avg_price

                    if entry_price is None:
                        entry_price = avg_price

                    if market_value is None:
                        market_value = abs(quantity) * (current_price or avg_price)

                    if unrealized_pnl is None:
                        if quantity > 0:  # LONG
                            unrealized_pnl = quantity * ((current_price or avg_price) - entry_price)
                        else:  # SHORT
                            unrealized_pnl = abs(quantity) * (entry_price - (current_price or avg_price))

                    if unrealized_pnl_percent is None and entry_price and entry_price > 0:
                        unrealized_pnl_percent = (unrealized_pnl / (abs(quantity) * entry_price)) * 100

                    positions.append({
                        'symbol': symbol,
                        'quantity': round(quantity, 6),
                        'avg_price': round(avg_price, 4),
                        'entry_price': round(entry_price or avg_price, 4),
                        'current_price': round(current_price or avg_price, 4),
                        'market_value': round(market_value or 0, 2),
                        'position_size_usd': round(position_size_usd or market_value or 0, 2),
                        'unrealized_pnl': round(unrealized_pnl or 0, 2),
                        'unrealized_pnl_percent': round(unrealized_pnl_percent or 0, 2),
                        'side': 'LONG' if quantity > 0 else 'SHORT',
                        'last_updated': last_updated,
                        'last_price_update': last_price_update,
                        'price_change': round(((current_price or avg_price) - (entry_price or avg_price)) / (entry_price or avg_price) * 100, 2) if entry_price and entry_price > 0 else 0
                    })

                return positions

        except Exception as e:
            logger.error(f"Error getting enhanced positions from database: {e}")
            # Fallback to basic positions
            return self._get_current_positions_fallback()

    def _update_positions_with_current_prices(self):
        """Update all positions with current market prices"""
        try:
            def get_price(symbol):
                try:
                    return self.trading_engine.binance_api.get_current_price(symbol)
                except:
                    # Fallback prices for testing
                    fallback_prices = {
                        'BTCUSDT': 43500.0,
                        'ETHUSDT': 2420.0,
                        'ADAUSDT': 0.42,
                        'DOTUSDT': 5.80,
                        'BNBUSDT': 310.0,
                        'SOLUSDT': 98.0
                    }
                    return fallback_prices.get(symbol, 100.0)

            self.db_manager.update_all_positions_market_data(get_price)

        except Exception as e:
            logger.warning(f"Failed to update positions with current prices: {e}")

    def _get_binance_positions(self) -> List[Dict]:
        """Get real-time positions from Binance API"""
        try:
            account_info = self.trading_engine.binance_api.get_account_info()

            if not account_info or 'balances' not in account_info:
                return []

            positions = []

            # Get significant balances (excluding dust)
            for balance in account_info['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked

                # Only include positions with significant value
                if total > 0.001 and asset != 'USDT':  # Exclude USDT (base currency)
                    try:
                        # Get current price for the asset
                        symbol = f"{asset}USDT"
                        current_price = self.trading_engine.binance_api.get_current_price(symbol)

                        if current_price:
                            market_value = total * current_price

                            # Only include if market value > $1
                            if market_value > 1.0:
                                positions.append({
                                    'symbol': symbol,
                                    'asset': asset,
                                    'quantity': round(total, 6),
                                    'free': round(free, 6),
                                    'locked': round(locked, 6),
                                    'current_price': round(current_price, 4),
                                    'market_value': round(market_value, 2),
                                    'side': 'LONG',  # Spot positions are always long
                                    'source': 'binance_real_time',
                                    'last_updated': datetime.now().isoformat()
                                })
                    except Exception as e:
                        logger.debug(f"Could not get price for {asset}: {e}")

            # Sort by market value descending
            positions.sort(key=lambda x: x['market_value'], reverse=True)

            return positions

        except Exception as e:
            logger.error(f"Error fetching Binance positions: {e}")
            return []

    def _get_enhanced_positions(self) -> Dict:
        """Get comprehensive position data combining database and Binance"""
        try:
            # Get positions from both sources
            db_positions = self._get_current_positions_from_db()
            binance_positions = self._get_binance_positions()

            # Combine and reconcile positions
            combined_positions = []
            symbols_processed = set()

            # Process database positions (our tracked positions)
            for pos in db_positions:
                symbols_processed.add(pos['symbol'])

                # Try to find matching Binance position for comparison
                binance_match = None
                for b_pos in binance_positions:
                    if b_pos['symbol'] == pos['symbol']:
                        binance_match = b_pos
                        break

                enhanced_pos = {
                    **pos,
                    'source': 'database_tracked',
                    'binance_quantity': binance_match['quantity'] if binance_match else 0,
                    'binance_market_value': binance_match['market_value'] if binance_match else 0,
                    'quantity_difference': abs(pos['quantity']) - (binance_match['quantity'] if binance_match else 0) if binance_match else abs(pos['quantity']),
                    'is_synchronized': abs(abs(pos['quantity']) - (binance_match['quantity'] if binance_match else 0)) < 0.001 if binance_match else False
                }

                combined_positions.append(enhanced_pos)

            # Add Binance-only positions (not tracked in database)
            for b_pos in binance_positions:
                if b_pos['symbol'] not in symbols_processed:
                    enhanced_pos = {
                        **b_pos,
                        'source': 'binance_only',
                        'avg_price': b_pos['current_price'],  # Use current price as entry for untracked positions
                        'entry_price': b_pos['current_price'],
                        'unrealized_pnl': 0,  # No P&L calculation for untracked positions
                        'unrealized_pnl_percent': 0,
                        'is_synchronized': False,
                        'quantity_difference': b_pos['quantity']
                    }
                    combined_positions.append(enhanced_pos)

            # Calculate summary statistics
            total_market_value = sum(pos.get('market_value', 0) for pos in combined_positions)
            total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in combined_positions)
            synchronized_positions = sum(1 for pos in combined_positions if pos.get('is_synchronized', False))

            return {
                'positions': combined_positions,
                'summary': {
                    'total_positions': len(combined_positions),
                    'database_tracked': len(db_positions),
                    'binance_only': len(binance_positions) - len([p for p in binance_positions if p['symbol'] in symbols_processed]),
                    'synchronized_positions': synchronized_positions,
                    'total_market_value': round(total_market_value, 2),
                    'total_unrealized_pnl': round(total_unrealized_pnl, 2),
                    'last_updated': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error getting enhanced positions: {e}")
            return {
                'positions': [],
                'summary': {
                    'total_positions': 0,
                    'error': str(e)
                }
            }

    def _get_current_positions_fallback(self) -> List[Dict]:
        """Fallback to lockless storage for positions"""
        try:
            positions = self.lockless_storage.get_positions()

            formatted_positions = []
            for position in positions:
                market_value = abs(position['quantity']) * position['avg_price']

                formatted_positions.append({
                    'symbol': position['symbol'],
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'market_value': round(market_value, 2),
                    'unrealized_pnl': 0,
                    'last_updated': position.get('last_updated', ''),
                    'side': 'LONG' if position['quantity'] > 0 else 'SHORT'
                })

            return formatted_positions

        except Exception as e:
            logger.error(f"Error getting positions from lockless storage: {e}")
            return []

    @cached(ttl_seconds=20, key_prefix="pos_")
    def _get_current_positions(self) -> List[Dict]:
        """Get current open positions from database table with fallback"""
        return self._get_current_positions_from_db()

    def _get_open_orders(self) -> List[Dict]:
        """Get current open orders from Binance"""
        try:
            # Get open orders from Binance API
            open_orders = self.trading_engine.binance_api.get_open_orders()

            formatted_orders = []
            for order in open_orders:
                formatted_orders.append({
                    'order_id': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'quantity': float(order['origQty']),
                    'filled_quantity': float(order['executedQty']),
                    'price': float(order['price']) if order['price'] != '0.00000000' else None,
                    'stop_price': float(order['stopPrice']) if order.get('stopPrice') and order['stopPrice'] != '0.00000000' else None,
                    'status': order['status'],
                    'time_in_force': order['timeInForce'],
                    'created_time': order['time'],
                    'updated_time': order['updateTime']
                })

            return formatted_orders
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    @cached(ttl_seconds=60, key_prefix="balance_")
    def _get_account_balance(self) -> Dict:
        """Get account balance from Binance"""
        try:
            # Get account info from Binance API
            account_info = self.trading_engine.binance_api.get_account_info()

            balances = []
            total_value_usdt = 0

            # Process each balance
            for balance in account_info.get('balances', []):
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked

                if total > 0:  # Only show non-zero balances
                    # Get USD value (simplified estimation)
                    usd_value = 0
                    if asset == 'USDT':
                        usd_value = total
                    elif asset == 'BTC':
                        try:
                            btc_price = self.trading_engine.binance_api.get_current_price('BTCUSDT')
                            usd_value = total * btc_price
                        except:
                            usd_value = total * 62000  # Fallback BTC price
                    elif asset == 'ETH':
                        try:
                            eth_price = self.trading_engine.binance_api.get_current_price('ETHUSDT')
                            usd_value = total * eth_price
                        except:
                            usd_value = total * 2500  # Fallback ETH price
                    elif asset == 'ADA':
                        try:
                            ada_price = self.trading_engine.binance_api.get_current_price('ADAUSDT')
                            usd_value = total * ada_price
                        except:
                            usd_value = total * 0.35  # Fallback ADA price
                    # Add more assets as needed

                    total_value_usdt += usd_value

                    balances.append({
                        'asset': asset,
                        'free': round(free, 8),
                        'locked': round(locked, 8),
                        'total': round(total, 8),
                        'usd_value': round(usd_value, 2)
                    })

            # Sort by USD value descending
            balances.sort(key=lambda x: x['usd_value'], reverse=True)

            return {
                'balances': balances,
                'total_value_usdt': round(total_value_usdt, 2),
                'account_type': account_info.get('accountType', 'SPOT'),
                'can_trade': account_info.get('canTrade', False),
                'can_withdraw': account_info.get('canWithdraw', False),
                'can_deposit': account_info.get('canDeposit', False),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {
                'balances': [],
                'total_value_usdt': 0,
                'error': str(e)
            }

    def _get_recent_trades(self, limit: int = 50, symbol: str = None) -> List[Dict]:
        """Get recent trades from lockless storage"""
        try:
            # Get trades from lockless storage
            if symbol:
                trades = self.lockless_storage.get_symbol_trades(symbol, limit)
            else:
                trades = self.lockless_storage.get_all_trades(limit)

            # Format trades for dashboard
            formatted_trades = []
            for trade in trades:
                formatted_trades.append({
                    'id': trade['id'],
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'commission': trade['commission'],
                    'timestamp': trade['timestamp'],
                    'strategy': trade['strategy'],
                    'realized_pnl': trade['realized_pnl'],
                    'status': trade['status'],
                    'order_id': trade['order_id']
                })

            return formatted_trades

        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []

    def _get_strategy_performance(self) -> List[Dict]:
        """Get strategy performance data"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        strategy,
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(realized_pnl) as total_pnl,
                        AVG(realized_pnl) as avg_pnl
                    FROM trades
                    WHERE status = 'FILLED' AND strategy IS NOT NULL
                    GROUP BY strategy
                """)

                strategies = []
                for row in cursor.fetchall():
                    strategy, total_trades, winning_trades, total_pnl, avg_pnl = row
                    win_rate = (winning_trades / total_trades *
                                100) if total_trades > 0 else 0

                    strategies.append({
                        'name': strategy,
                        'total_trades': total_trades,
                        'win_rate': round(win_rate, 1),
                        'total_pnl': round(total_pnl or 0, 2),
                        'avg_trade_duration': 0,  # TODO: Calculate from trade data
                        'max_drawdown': 0,  # TODO: Calculate drawdown
                        'active': True  # TODO: Get from engine status
                    })

                return strategies
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return []

    def _get_system_health(self) -> Dict:
        """Get system health metrics"""
        import psutil
        import time

        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            # Get last trade time from database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT MAX(timestamp) FROM trades WHERE status = 'FILLED'")
                last_trade_result = cursor.fetchone()
                last_trade_time = last_trade_result[0] if last_trade_result[0] else datetime.now(
                ).isoformat()

            # Calculate uptime (simplified - should track actual start time)
            uptime = 3600  # Default 1 hour

            # Check trading engine status
            is_running = getattr(self.trading_engine, 'is_running', False)

            # Determine overall status
            status = 'HEALTHY'
            if cpu_percent > 90 or memory.percent > 90:
                status = 'WARNING'
            if not is_running:
                status = 'ERROR'

            return {
                'status': status,
                'uptime': uptime,
                'api_latency': 120,  # Mock API latency
                'error_rate': 0.01,  # Mock error rate
                'last_trade_time': last_trade_time,
                'memory_usage': round(memory.percent, 1),
                'cpu_usage': round(cpu_percent, 1),
                'is_trading_active': is_running,
                'total_memory_gb': round(memory.total / (1024**3), 2),
                'available_memory_gb': round(memory.available / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'status': 'ERROR',
                'uptime': 0,
                'api_latency': 0,
                'error_rate': 1.0,
                'last_trade_time': datetime.now().isoformat(),
                'memory_usage': 0,
                'cpu_usage': 0,
                'is_trading_active': False,
                'error': str(e)
            }

    def _get_pnl_chart_data(self, days: int) -> List[Dict]:
        """Get P&L chart data using performance table (faster) with trades fallback"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Try to get data from performance table first (much faster)
                cursor.execute("""
                    SELECT date, total_pnl
                    FROM performance
                    WHERE date >= date('now', '-{} days')
                    ORDER BY date
                """.format(days))

                perf_data = cursor.fetchall()

                if len(perf_data) > 1:
                    # Use performance table data
                    data = []
                    cumulative_pnl = 0

                    for i, (date, total_pnl) in enumerate(perf_data):
                        if i == 0:
                            # First entry - use total_pnl as cumulative
                            cumulative_pnl = total_pnl or 0
                            daily_pnl = total_pnl or 0
                        else:
                            # Calculate daily change
                            prev_cumulative = data[-1]['value'] if data else 0
                            cumulative_pnl = total_pnl or 0
                            daily_pnl = cumulative_pnl - prev_cumulative

                        data.append({
                            'timestamp': date,
                            'value': round(cumulative_pnl, 2),
                            'daily': round(daily_pnl, 2)
                        })

                    return data

                else:
                    # Fallback to trades table calculation
                    cursor.execute("""
                        SELECT
                            DATE(timestamp) as date,
                            SUM(realized_pnl) as daily_pnl
                        FROM trades
                        WHERE status = 'FILLED'
                            AND timestamp >= datetime('now', '-{} days')
                        GROUP BY DATE(timestamp)
                        ORDER BY date
                    """.format(days))

                    data = []
                    cumulative_pnl = 0
                    for row in cursor.fetchall():
                        date, daily_pnl = row
                        cumulative_pnl += daily_pnl or 0
                        data.append({
                            'timestamp': date,
                            'value': round(cumulative_pnl, 2),
                            'daily': round(daily_pnl or 0, 2)
                        })

                    return data

        except Exception as e:
            logger.error(f"Error getting P&L chart data: {e}")
            return []

    def _get_portfolio_chart_data(self, days: int) -> List[Dict]:
        """Get portfolio value chart data"""
        # This is a simplified version - in reality you'd track portfolio value over time
        base_value = 10000
        pnl_data = self._get_pnl_chart_data(days)

        return [{
            'timestamp': item['timestamp'],
            'value': base_value + item['value']
        } for item in pnl_data]

    def _get_system_alerts(self) -> List[Dict]:
        """Get system alerts"""
        # Mock alerts for now
        return [
            {
                'id': '1',
                'type': 'WARNING',
                'message': 'High volatility detected in BTCUSDT',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False
            },
            {
                'id': '2',
                'type': 'SUCCESS',
                'message': 'RSI Bollinger strategy executed profitable trade',
                'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
                'acknowledged': True
            }
        ]

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask app"""
        self.app.run(host=host, port=port, debug=debug)
