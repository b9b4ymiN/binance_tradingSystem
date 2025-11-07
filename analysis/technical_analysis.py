import numpy as np
from typing import List, Tuple


class TechnicalAnalysis:
    """Advanced technical analysis implementation based on research strategies"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, 
                                std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def calculate_atr(high: List[float], low: List[float],
                     close: List[float], period: int = 14) -> float:
        """Calculate Average True Range for volatility-based position sizing"""
        if len(high) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))

        return np.mean(true_ranges[-period:]) if len(true_ranges) >= period else np.mean(true_ranges)

    @staticmethod
    def calculate_vwap(high: List[float], low: List[float],
                      close: List[float], volume: List[float], period: int = None) -> float:
        """
        Calculate Volume Weighted Average Price (VWAP)

        Args:
            high: List of high prices
            low: List of low prices
            close: List of close prices
            volume: List of volumes
            period: Number of periods to calculate VWAP (None = cumulative from start)

        Returns:
            VWAP value
        """
        if len(close) < 1 or len(volume) < 1:
            return close[-1] if close else 0.0

        # Typical price = (high + low + close) / 3
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]

        # If period specified, use rolling window
        if period and len(typical_prices) > period:
            typical_prices = typical_prices[-period:]
            volume = volume[-period:]

        # VWAP = Sum(Typical Price * Volume) / Sum(Volume)
        total_pv = sum(tp * v for tp, v in zip(typical_prices, volume))
        total_volume = sum(volume)

        if total_volume == 0:
            return close[-1]

        return total_pv / total_volume

    @staticmethod
    def calculate_vwap_bands(high: List[float], low: List[float],
                            close: List[float], volume: List[float],
                            period: int = 100, std_multiplier: float = 1.0) -> Tuple[float, float, float]:
        """
        Calculate VWAP with standard deviation bands

        Args:
            high, low, close, volume: Price and volume data
            period: VWAP calculation period
            std_multiplier: Standard deviation multiplier for bands

        Returns:
            (upper_band, vwap, lower_band)
        """
        vwap = TechnicalAnalysis.calculate_vwap(high, low, close, volume, period)

        if len(close) < 2:
            return vwap, vwap, vwap

        # Calculate standard deviation of price from VWAP
        recent_period = period if period and len(close) > period else len(close)
        recent_closes = close[-recent_period:]

        # Price deviation from VWAP
        deviations = [(price - vwap) ** 2 for price in recent_closes]
        variance = sum(deviations) / len(deviations)
        std_dev = variance ** 0.5

        upper_band = vwap + (std_multiplier * std_dev)
        lower_band = vwap - (std_multiplier * std_dev)

        return upper_band, vwap, lower_band

    @staticmethod
    def calculate_vwap_deviation(current_price: float, vwap: float) -> float:
        """
        Calculate percentage deviation from VWAP

        Args:
            current_price: Current market price
            vwap: VWAP value

        Returns:
            Percentage deviation (positive = above VWAP, negative = below VWAP)
        """
        if vwap == 0:
            return 0.0
        return (current_price - vwap) / vwap

    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0

        return np.mean(prices[-period:])