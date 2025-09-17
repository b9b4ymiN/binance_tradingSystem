from typing import Optional, Dict
import logging
from strategies.base_strategy import BaseStrategy
from analysis.technical_analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)


class RSIBollingerStrategy(BaseStrategy):
    """RSI-Bollinger Bands scalping strategy (69-80% win rate from research)"""
    
    @property
    def strategy_name(self) -> str:
        return "rsi_bollinger_scalping"
    
    @property
    def expected_win_rate(self) -> float:
        return 0.75
    
    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate RSI-Bollinger Bands scalping signal"""
        
        # Get recent price data
        klines = self.api.get_klines(symbol, '1m', 100)
        closes = [float(kline[4]) for kline in klines]
        highs = [float(kline[2]) for kline in klines]
        lows = [float(kline[3]) for kline in klines]
        
        if len(closes) < 50:
            return None
        
        # Calculate indicators
        rsi = TechnicalAnalysis.calculate_rsi(closes, 14)
        upper_bb, middle_bb, lower_bb = TechnicalAnalysis.calculate_bollinger_bands(closes, 20, 2.0)
        current_price = closes[-1]
        atr = TechnicalAnalysis.calculate_atr(highs, lows, closes, 14)
        
        # Entry conditions
        if rsi < self.config.rsi_oversold and current_price <= lower_bb:
            # Long signal
            entry_price = current_price
            stop_loss = current_price - (2 * atr)
            take_profit = middle_bb
            
            return {
                'action': 'buy',
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': self.strategy_name,
                'confidence': 0.75
            }
        
        elif rsi > self.config.rsi_overbought and current_price >= upper_bb:
            # Short signal
            entry_price = current_price
            stop_loss = current_price + (2 * atr)
            take_profit = middle_bb
            
            return {
                'action': 'sell',
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': self.strategy_name,
                'confidence': 0.75
            }
        
        return None