from typing import Optional, Dict
import numpy as np
import logging
from strategies.base_strategy import BaseStrategy
from analysis.technical_analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)


class BreakoutSwingStrategy(BaseStrategy):
    """Breakout swing trading strategy (55-65% win rate from research)"""
    
    @property
    def strategy_name(self) -> str:
        return "breakout_swing"
    
    @property
    def expected_win_rate(self) -> float:
        return 0.65
    
    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate breakout swing trading signal"""
        
        # Get 4-hour data for swing trading
        klines = self.api.get_klines(symbol, '4h', 100)
        closes = [float(kline[4]) for kline in klines]
        highs = [float(kline[2]) for kline in klines]
        lows = [float(kline[3]) for kline in klines]
        volumes = [float(kline[5]) for kline in klines]
        
        if len(closes) < 50:
            return None
        
        # Calculate indicators
        ema_21 = TechnicalAnalysis.calculate_ema(closes, 21)
        ema_50 = TechnicalAnalysis.calculate_ema(closes, 50)
        rsi = TechnicalAnalysis.calculate_rsi(closes, 14)
        current_price = closes[-1]
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        
        # Resistance/Support levels (simplified)
        recent_highs = max(highs[-20:])
        recent_lows = min(lows[-20:])
        
        # Breakout conditions
        if (current_price > recent_highs and 
            current_volume > 1.5 * avg_volume and
            ema_21 > ema_50 and
            40 <= rsi <= 60):
            
            entry_price = current_price
            stop_loss = recent_highs * 0.98  # 2% below breakout level
            take_profit = current_price * 1.15  # 15% target
            
            return {
                'action': 'buy',
                'symbol': symbol,
                'price': entry_price,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': self.strategy_name,
                'confidence': 0.65
            }
        
        return None