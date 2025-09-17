import time
import logging
import math
from datetime import datetime, timezone
from threading import Thread
from typing import Dict
from config.trading_config import TradingConfig
from core.database import DatabaseManager
from core.binance_api import BinanceAPI
from core.risk_manager import RiskManager
from strategies.rsi_bollinger import RSIBollingerStrategy
from strategies.breakout_swing import BreakoutSwingStrategy
from webhook.handler import WebhookHandler

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.binance_api = BinanceAPI(config)
        self.risk_manager = RiskManager(config, self.db_manager)
        
        # Initialize strategies
        self.rsi_bollinger_strategy = RSIBollingerStrategy(config, self.binance_api, self.risk_manager)
        self.breakout_swing_strategy = BreakoutSwingStrategy(config, self.binance_api, self.risk_manager)
        
        self.webhook_handler = WebhookHandler(config, self)
        self.is_running = False
        
    def process_signal(self, signal_data: Dict) -> Dict:
        """Process incoming trading signal from TradingView"""
        
        try:
            symbol = signal_data['symbol']
            action = signal_data['action'].lower()
            price = float(signal_data['price'])
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
            logger.info(f"Position risk: {position_risk}, Size: {position_size}")
            if not self.risk_manager.check_risk_limits(position_risk):
                return {'error': 'Position exceeds risk limits'}
            
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
            
            # Place stop loss and take profit orders
            if 'stop_loss' in signal_data:
                self._place_stop_loss(symbol, action, quantity, signal_data['stop_loss'])
            
            if 'take_profit' in signal_data:
                self._place_take_profit(symbol, action, quantity, signal_data['take_profit'])
            
            logger.info(f"Trade executed successfully: {order_result}")
            return {'success': True, 'order': order_result}
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'error': f'Trade execution failed: {str(e)}'}
    
    def _place_stop_loss(self, symbol: str, action: str, quantity: float, stop_price: float):
        """Place stop loss order"""
        try:
            opposite_side = 'SELL' if action == 'buy' else 'BUY'
            self.binance_api.place_order(
                symbol=symbol,
                side=opposite_side,
                order_type='STOP_MARKET',
                quantity=quantity,
                stop_price=stop_price
            )
            logger.info(f"Stop loss placed at {stop_price}")
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
    
    def _place_take_profit(self, symbol: str, action: str, quantity: float, target_price: float):
        """Place take profit order"""
        try:
            opposite_side = 'SELL' if action == 'buy' else 'BUY'
            self.binance_api.place_order(
                symbol=symbol,
                side=opposite_side,
                order_type='LIMIT',
                quantity=quantity,
                price=target_price,
                time_in_force='GTC'
            )
            logger.info(f"Take profit placed at {target_price}")
        except Exception as e:
            logger.error(f"Failed to place take profit: {e}")
    
    def _log_trade(self, order_result: Dict, signal_data: Dict):
        """Log trade to database"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, side, quantity, price, order_id, strategy, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                order_result.get('symbol'),
                order_result.get('side'),
                float(order_result.get('executedQty', 0)),
                float(order_result.get('price', 0)),
                str(order_result.get('orderId')),
                signal_data.get('strategy', 'manual'),
                order_result.get('status', 'pending')
            ))
            conn.commit()
    
    def start_auto_trading(self):
        """Start automated trading loop"""
        self.is_running = True
        
        def trading_loop():
            while self.is_running:
                try:
                    # Monitor positions and adjust stops
                    self._monitor_positions()
                    
                    # Generate automated signals if enabled
                    self._generate_auto_signals()
                    
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
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'filled'")
            total_trades = cursor.fetchone()[0]
            
            # Win rate calculation
            cursor.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE status = 'filled' AND pnl > 0
            """)
            winning_trades = cursor.fetchone()[0]
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Total PnL
            cursor.execute("SELECT SUM(pnl) FROM trades WHERE status = 'filled'")
            total_pnl = cursor.fetchone()[0] or 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }