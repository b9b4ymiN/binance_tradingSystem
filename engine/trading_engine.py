import time
import logging
import math
import sqlite3
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone, timedelta
from threading import Thread
from typing import Dict
from config.trading_config import TradingConfig
from core.database import DatabaseManager
from core.robust_database import get_robust_db_manager
from core.lockless_storage import get_lockless_storage
from core.binance_api import BinanceAPI
from core.risk_manager import RiskManager
from strategies.rsi_bollinger import RSIBollingerStrategy
from strategies.breakout_swing import BreakoutSwingStrategy
from webhook.handler import WebhookHandler
from api.dashboard_api import DashboardAPI

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.robust_db = get_robust_db_manager(config.db_path)
        self.lockless_storage = get_lockless_storage("trading_data")
        self.binance_api = BinanceAPI(config)
        self.risk_manager = RiskManager(config, self.db_manager)
        self.risk_manager.binance_api = self.binance_api  # Set API reference
        
        # Initialize strategies
        self.rsi_bollinger_strategy = RSIBollingerStrategy(config, self.binance_api, self.risk_manager)
        self.breakout_swing_strategy = BreakoutSwingStrategy(config, self.binance_api, self.risk_manager)
        
        self.webhook_handler = WebhookHandler(config, self)
        self.dashboard_api = DashboardAPI(self, self.db_manager)
        self.is_running = False
        
    def process_signal(self, signal_data: Dict) -> Dict:
        """Process incoming trading signal from TradingView"""
        
        try:
            symbol = signal_data['symbol']
            action = signal_data['action'].lower()
            price_value = signal_data.get('price', signal_data.get('entry_price'))
            if price_value is None:
                return {'error': 'Price missing from signal'}
            price = float(price_value)
            strategy_name = signal_data.get('strategy', 'manual')
            
            logger.info(f"Processing {action} signal for {symbol} at {price}")
            
            # Validate symbol
            try:
                symbol_info = self.binance_api.get_symbol_info(symbol)
            except ValueError:
                return {'error': f'Invalid symbol: {symbol}'}
            
            filters = symbol_info.get('filters', [])
            lot_step = self._get_filter_value(filters, 'LOT_SIZE', 'stepSize', 0.000001)
            min_qty = self._get_filter_value(filters, 'LOT_SIZE', 'minQty', 0.0)
            min_notional = self._get_filter_value(filters, 'MIN_NOTIONAL', 'minNotional', 0.0)

            price_tick = self._get_filter_value(filters, 'PRICE_FILTER', 'tickSize', 0.0)
            signal_data['_price_tick'] = price_tick

            # Generate strategy signal if not provided
            if 'stop_loss' not in signal_data or 'take_profit' not in signal_data:
                if strategy_name == 'rsi_bollinger_scalping':
                    strategy_signal = self.rsi_bollinger_strategy.generate_signal(symbol)
                elif strategy_name == 'breakout_swing':
                    strategy_signal = self.breakout_swing_strategy.generate_signal(symbol)
                else:
                    strategy_signal = None
                
                if strategy_signal:
                    signal_data.update(strategy_signal)
            
            # Calculate position size
            stop_loss = signal_data.get('stop_loss', price * 0.98 if action == 'buy' else price * 1.02)
            position_size = self.risk_manager.calculate_position_size(
                symbol, price, stop_loss
            )
            position_size = self._normalize_quantity(position_size, lot_step)

            if position_size <= 0:
                return {'error': 'Position size too small or invalid'}

            if min_qty > 0 and position_size < min_qty:
                return {'error': f'Position size below minimum lot size {min_qty}'}

            notional_value = position_size * price
            if min_notional > 0 and notional_value < min_notional:
                return {'error': f'Order notional {notional_value:.8f} below minimum {min_notional}'}

            # Check risk limits
            position_risk = abs(price - stop_loss) / price * position_size * price
            logger.info(f"Position risk: ${position_risk:.2f}, Size: {position_size}")
            if not self.risk_manager.check_risk_limits(position_risk):
                return {'error': 'Position exceeds risk limits'}

            # Check account balance before placing order
            balance_check = self._check_sufficient_balance(symbol, action, position_size, price)
            if not balance_check['sufficient']:
                return {'error': f"Insufficient balance: {balance_check['message']}"}
            
            # Execute trade
            order_result = self._execute_trade(
                symbol, action, position_size, price, signal_data
            )

            return order_result
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _get_filter_value(filters, filter_type, key, default):
        for filter_data in (filters or []):
            if filter_data.get('filterType') == filter_type:
                try:
                    return float(filter_data.get(key, default))
                except (TypeError, ValueError):
                    return float(default)
        return float(default)

    @staticmethod
    def _normalize_quantity(quantity, step):
        if step <= 0:
            return float(f"{max(quantity, 0.0):.8f}")
        steps = math.floor(max(quantity, 0.0) / step)
        normalized = steps * step
        return float(f"{normalized:.8f}")


    @staticmethod
    def _normalize_price(value, tick_size):
        if tick_size is None or tick_size <= 0:
            return float(f"{value:.8f}")
        quantized_tick = Decimal(str(tick_size))
        price_decimal = Decimal(str(value))
        steps = (price_decimal / quantized_tick).to_integral_value(rounding=ROUND_DOWN)
        normalized = steps * quantized_tick
        return float(normalized)

    def _execute_trade(self, symbol: str, action: str, quantity: float, 
                      price: float, signal_data: Dict) -> Dict:
        """Execute the actual trade on Binance"""
        
        try:
            # Place main order
            order_result = self.binance_api.place_order(
                symbol=symbol,
                side='BUY' if action == 'buy' else 'SELL',
                order_type='MARKET',
                quantity=quantity
            )
            
            # Log trade to database
            self._log_trade(order_result, signal_data)

            # Invalidate relevant cache entries after successful trade
            try:
                from api.cache_manager import cache
                cache.invalidate_pattern("perf_")  # Performance metrics
                cache.invalidate_pattern("pos_")   # Positions
                cache.delete("dashboard_overview") # Overview
                logger.info("Cache invalidated after successful trade")
                if status == 'FILLED':
                    self.db_manager.update_daily_performance(trade_date)
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")
            
            # Place stop loss and take profit orders
            price_tick = signal_data.get('_price_tick')

            if 'stop_loss' in signal_data:
                self._place_stop_loss(symbol, action, quantity, signal_data['stop_loss'], price_tick)

            if 'take_profit' in signal_data:
                self._place_take_profit(symbol, action, quantity, signal_data['take_profit'], price_tick)

            logger.info(f"Trade executed successfully: {order_result}")
            return {'success': True, 'order': order_result}
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'error': f'Trade execution failed: {str(e)}'}
    

    def _place_stop_loss(self, symbol: str, action: str, quantity: float, stop_price: float, tick_size: float = None):
        """Place stop loss order"""
        try:
            # For stop loss: if we bought, we need to sell to exit (and vice versa)
            stop_side = 'SELL' if action.upper() == 'BUY' else 'BUY'
            normalized_stop = self._normalize_price(stop_price, tick_size)

            # Use STOP_LOSS_LIMIT for better control with automatic cleanup
            result = self.binance_api.place_order_with_cleanup(
                symbol=symbol,
                side=stop_side,
                order_type='STOP_LOSS_LIMIT',
                quantity=quantity,
                price=normalized_stop,  # Limit price
                stop_price=normalized_stop,  # Stop trigger price
                time_in_force='GTC'
            )
            logger.info(f"Stop loss ({stop_side}) placed at {normalized_stop} for {quantity} {symbol}")
            return result
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
            return None

    def _place_take_profit(self, symbol: str, action: str, quantity: float, target_price: float, tick_size: float = None):
        """Place take profit order"""
        try:
            # For take profit: if we bought, we need to sell to take profit (and vice versa)
            tp_side = 'SELL' if action.upper() == 'BUY' else 'BUY'
            normalized_target = self._normalize_price(target_price, tick_size)

            result = self.binance_api.place_order_with_cleanup(
                symbol=symbol,
                side=tp_side,
                order_type='LIMIT',
                quantity=quantity,
                price=normalized_target,
                time_in_force='GTC'
            )
            logger.info(f"Take profit ({tp_side}) placed at {normalized_target} for {quantity} {symbol}")
            return result
        except Exception as e:
            logger.error(f"Failed to place take profit: {e}")
            return None
    def _log_trade(self, order_result: Dict, signal_data: Dict):
        """Log trade using lockless file storage (eliminates database locks)"""
        try:
            # Use lockless storage - no database locks possible!
            trade_id = self.lockless_storage.log_trade(order_result, signal_data)
            logger.info(f"Trade logged successfully to lockless storage: {trade_id}")

            # Also try to log to database for backward compatibility (non-blocking)
            try:
                self._log_trade_to_database_async(order_result, signal_data)
            except Exception as db_error:
                logger.warning(f"Database logging failed (continuing with lockless storage): {db_error}")

        except Exception as e:
            logger.error(f"Failed to log trade to lockless storage: {e}")
            raise

    def _log_trade_to_database_async(self, order_result: Dict, signal_data: Dict):
        """Async database logging (non-blocking) for backward compatibility"""
        def log_to_db():
            try:
                status = (order_result.get('status') or '').upper()
                trade_date = None
                if status == 'FILLED':
                    transact_time = order_result.get('transactTime')
                    if transact_time:
                        try:
                            trade_date = datetime.fromtimestamp(int(transact_time) / 1000).strftime('%Y-%m-%d')
                        except Exception:
                            trade_date = None
                    if trade_date is None:
                        trade_date = datetime.now().strftime('%Y-%m-%d')
                # Calculate average fill price from fills
                fills = order_result.get('fills', [])
                if fills:
                    total_qty = sum(float(fill['qty']) for fill in fills)
                    total_value = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                    avg_price = total_value / total_qty if total_qty > 0 else 0
                    commission = sum(float(fill.get('commission', 0)) for fill in fills)
                else:
                    executed_qty = float(order_result.get('executedQty', 0))
                    cumulative_quote_qty = float(order_result.get('cummulativeQuoteQty', 0))
                    if executed_qty > 0 and cumulative_quote_qty > 0:
                        avg_price = cumulative_quote_qty / executed_qty
                    else:
                        avg_price = float(order_result.get('price', 0))
                    commission = 0

                symbol = order_result.get('symbol')
                side = order_result.get('side')
                quantity = float(order_result.get('executedQty', 0))

                # Calculate realized P&L
                realized_pnl = self._calculate_realized_pnl(symbol, side, quantity, avg_price)

                # Quick database insert (with short timeout)
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO trades (symbol, side, quantity, price, order_id, strategy, status, commission, realized_pnl)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        side,
                        quantity,
                        avg_price,
                        str(order_result.get('orderId')),
                        signal_data.get('strategy', 'manual'),
                        order_result.get('status', 'PENDING').upper(),
                        commission,
                        realized_pnl
                    ))
                    conn.commit()
                    logger.debug(f"Trade logged to database with P&L: ${realized_pnl:.4f}")

                    # Update portfolio position in database
                    if symbol and side and quantity > 0:
                        success = self.db_manager.update_portfolio_position(symbol, side, quantity, avg_price)
                        if success:
                            logger.debug(f"Portfolio position updated for {symbol}")
                        else:
                            logger.warning(f"Failed to update portfolio position for {symbol}")

                if status == 'FILLED':
                    self.db_manager.update_daily_performance(trade_date)
            except Exception as e:
                logger.debug(f"Background database logging failed: {e}")

        # Run in background thread (non-blocking)
        Thread(target=log_to_db, daemon=True).start()

    def _calculate_realized_pnl(self, symbol: str, side: str, quantity: float, price: float) -> float:
        """Calculate realized P&L for a trade based on current position"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get current position
                cursor.execute('SELECT quantity, avg_price FROM portfolio WHERE symbol = ?', (symbol,))
                result = cursor.fetchone()

                if result:
                    current_qty, current_avg_price = result
                else:
                    current_qty, current_avg_price = 0, 0

                realized_pnl = 0.0

                if side.upper() == 'BUY':
                    if current_qty < 0:
                        # Closing short position (partial or full)
                        close_qty = min(quantity, abs(current_qty))
                        realized_pnl = close_qty * (current_avg_price - price)  # Profit when price drops
                elif side.upper() == 'SELL':
                    if current_qty > 0:
                        # Closing long position (partial or full)
                        close_qty = min(quantity, current_qty)
                        realized_pnl = close_qty * (price - current_avg_price)  # Profit when price rises

                logger.debug(f"P&L calculation: {symbol} {side} {quantity} @ {price} -> ${realized_pnl:.4f}")
                return realized_pnl

        except Exception as e:
            logger.error(f"Failed to calculate realized P&L: {e}")
            return 0.0

    def _update_portfolio_position(self, order_result: Dict, avg_price: float):
        """Portfolio position is automatically updated by lockless storage"""
        # Note: Portfolio updates are now handled automatically by lockless_storage.log_trade()
        # This method is kept for compatibility but is no longer needed
        pass

    def _check_sufficient_balance(self, symbol: str, action: str, quantity: float, price: float) -> Dict:
        """Check if account has sufficient balance for the trade"""
        try:
            account_info = self.binance_api.get_account_info()
            balances = {bal['asset']: float(bal['free']) for bal in account_info['balances']}

            if action.upper() == 'BUY':
                # For BUY orders, need sufficient quote asset (USDT)
                quote_asset = 'USDT'  # Assuming all pairs are vs USDT
                required_amount = quantity * price
                available = balances.get(quote_asset, 0)

                if available >= required_amount:
                    return {'sufficient': True, 'message': 'Sufficient balance'}
                else:
                    return {
                        'sufficient': False,
                        'message': f'Need {required_amount:.2f} {quote_asset}, have {available:.2f}'
                    }

            else:  # SELL orders
                # For SELL orders, need sufficient base asset
                base_asset = symbol.replace('USDT', '')  # Extract base asset
                available = balances.get(base_asset, 0)

                if available >= quantity:
                    return {'sufficient': True, 'message': 'Sufficient balance'}
                else:
                    return {
                        'sufficient': False,
                        'message': f'Need {quantity:.6f} {base_asset}, have {available:.6f}'
                    }

        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return {'sufficient': True, 'message': 'Balance check failed, proceeding'}

    def start_auto_trading(self):
        """Start automated trading loop"""
        self.is_running = True

        def trading_loop():
            last_performance_update = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            while self.is_running:
                try:
                    # Monitor positions and adjust stops
                    self._monitor_positions()

                    # Generate automated signals if enabled
                    self._generate_auto_signals()

                    # Update daily performance metrics (once per day)
                    current_time = datetime.now()
                    if current_time.date() > last_performance_update.date():
                        self._update_daily_performance()
                        self._sync_portfolio_to_database()
                        last_performance_update = current_time

                    time.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(60)  # Wait longer on error

        trading_thread = Thread(target=trading_loop)
        trading_thread.daemon = True
        trading_thread.start()

        logger.info("Automated trading started")
    
    def _monitor_positions(self):
        """Monitor open positions and manage risk"""
        open_orders = self.binance_api.get_open_orders()
        # Implementation for position monitoring
        pass
    
    def _generate_auto_signals(self):
        """Generate automated trading signals"""
        # Popular crypto pairs for automated scanning
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
        
        for symbol in symbols:
            try:
                # Check RSI-Bollinger strategy
                signal = self.rsi_bollinger_strategy.generate_signal(symbol)
                if signal and signal.get('confidence', 0) > 0.7:
                    self.process_signal(signal)
                
                # Check breakout strategy
                signal = self.breakout_swing_strategy.generate_signal(symbol)
                if signal and signal.get('confidence', 0) > 0.6:
                    self.process_signal(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
    
    def stop_trading(self):
        """Stop all trading operations"""
        self.is_running = False
        logger.info("Trading stopped")
    
    def _update_daily_performance(self):
        """Update daily performance metrics for yesterday and today"""
        try:
            # Update yesterday's performance
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            success_yesterday = self.db_manager.update_daily_performance(yesterday)

            # Update today's performance
            today = datetime.now().strftime('%Y-%m-%d')
            success_today = self.db_manager.update_daily_performance(today)

            if success_yesterday or success_today:
                logger.info("Daily performance metrics updated successfully")
            else:
                logger.debug("No new trades to update performance metrics")

        except Exception as e:
            logger.error(f"Failed to update daily performance: {e}")

    def _sync_portfolio_to_database(self):
        """Sync portfolio table with lockless storage"""
        try:
            success = self.db_manager.sync_portfolio_from_lockless(self.lockless_storage)
            if success:
                logger.info("Portfolio table synced with lockless storage")
            else:
                logger.warning("Failed to sync portfolio table")
        except Exception as e:
            logger.error(f"Error syncing portfolio to database: {e}")

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'FILLED'")
            total_trades = cursor.fetchone()[0]

            # Win rate calculation
            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE status = 'FILLED' AND realized_pnl > 0
            """)
            winning_trades = cursor.fetchone()[0]
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Total PnL
            cursor.execute("SELECT SUM(realized_pnl) FROM trades WHERE status = 'FILLED'")
            total_pnl = cursor.fetchone()[0] or 0

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
