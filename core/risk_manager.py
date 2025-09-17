import logging
from datetime import datetime
from threading import Lock
from typing import Dict, List
from config.trading_config import TradingConfig
from core.database import DatabaseManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Advanced risk management implementing Kelly Criterion and position sizing"""
    
    def __init__(self, config: TradingConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.position_lock = Lock()
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, win_rate: float = 0.6, 
                              avg_win: float = 0.02, avg_loss: float = 0.01) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        
        # Kelly Criterion: f = (bp - q) / b
        # where b = odds received on the wager (avg_win/avg_loss)
        # p = probability of winning (win_rate)
        # q = probability of losing (1 - win_rate)
        
        b = avg_win / avg_loss  # Risk-reward ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative scaling
        scaled_kelly = kelly_fraction * self.config.kelly_fraction
        
        # Ensure within risk limits
        max_risk_per_trade = self.config.max_position_risk
        
        # Calculate position size based on stop loss distance
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        
        if risk_per_unit == 0:
            return 0
        
        # Get account balance
        account_balance = self._get_account_balance()
        
        position_value = min(
            account_balance * scaled_kelly,
            account_balance * max_risk_per_trade / risk_per_unit
        )
        
        position_size = position_value / entry_price
        
        logger.info(f"Kelly fraction: {kelly_fraction:.4f}, Scaled: {scaled_kelly:.4f}")
        logger.info(f"Position size calculated: {position_size:.6f} {symbol}")
        
        return max(0, position_size)
    
    def _get_account_balance(self) -> float:
        """Get total account balance in USDT equivalent"""
        # This would integrate with BinanceAPI to get real balance
        # For now, return a default value
        return 1000.0  # Default 1000 USDT
    
    def check_risk_limits(self, new_position_risk: float) -> bool:
        """Check if new position exceeds risk limits"""
        with self.position_lock:
            current_positions = self._get_current_positions()
            
            if len(current_positions) >= self.config.max_concurrent_positions:
                logger.warning("Maximum concurrent positions reached")
                return False
            
            total_risk = sum(pos['risk'] for pos in current_positions) + new_position_risk
            
            if total_risk > self.config.max_position_risk * self.config.max_concurrent_positions:
                logger.warning("Total portfolio risk exceeds limits")
                return False
            
            daily_trades = self._get_daily_trade_count()
            if daily_trades >= self.config.max_daily_trades:
                logger.warning("Daily trade limit reached")
                return False
            
            return True
    
    def _get_current_positions(self) -> List[Dict]:
        """Get current open positions"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, SUM(quantity) as total_quantity, AVG(price) as avg_price
                FROM trades 
                WHERE status = 'filled' 
                GROUP BY symbol 
                HAVING total_quantity != 0
            ''')
            
            positions = []
            for row in cursor.fetchall():
                symbol, quantity, avg_price = row
                risk = abs(quantity * avg_price) * 0.02  # Assume 2% risk per position
                positions.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'risk': risk
                })
            
            return positions
    
    def _get_daily_trade_count(self) -> int:
        """Get number of trades executed today"""
        today = datetime.now().date()
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE DATE(timestamp) = ? AND status = 'filled'
            ''', (today,))
            return cursor.fetchone()[0]