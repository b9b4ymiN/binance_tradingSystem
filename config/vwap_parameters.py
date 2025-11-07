"""
VWAP Mean Reversion Strategy Parameters
These parameters can be optimized weekly for best performance
"""

from dataclasses import dataclass, asdict
from typing import Dict
import json


@dataclass
class VWAPParameters:
    """Dynamic parameters for VWAP Mean Reversion Strategy"""

    # VWAP Calculation
    vwap_period: int = 100  # Number of candles for VWAP calculation

    # Entry Conditions
    entry_threshold: float = 0.007  # 0.7% deviation from VWAP for entry
    min_volume_multiplier: float = 1.5  # Volume must be > 1.5x average

    # Exit Conditions
    exit_threshold: float = 0.002  # 0.2% return to mean for exit
    profit_target_atr_mult: float = 1.5  # TP = VWAP + (1.5 * ATR)

    # Risk Management
    stop_loss_atr_mult: float = 2.0  # SL = entry - (2.0 * ATR)
    max_holding_periods: int = 60  # Exit after 60 candles (60 mins on 1m)

    # Volatility Filters
    atr_period: int = 14
    min_atr_filter: float = 0.003  # Don't trade if ATR < 0.3% (too quiet)
    max_atr_filter: float = 0.035  # Don't trade if ATR > 3.5% (too volatile)

    # Confidence Scoring
    base_confidence: float = 0.70  # Base confidence for VWAP signals

    # Timeframe
    timeframe: str = '1m'  # Trading timeframe
    lookback_candles: int = 200  # Candles to fetch for analysis

    def to_dict(self) -> Dict:
        """Convert parameters to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert parameters to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, params: Dict) -> 'VWAPParameters':
        """Create VWAPParameters from dictionary"""
        return cls(**params)

    @classmethod
    def from_json(cls, json_str: str) -> 'VWAPParameters':
        """Create VWAPParameters from JSON string"""
        params = json.loads(json_str)
        return cls.from_dict(params)

    def validate(self) -> bool:
        """Validate parameter values are within reasonable ranges"""
        try:
            assert 20 <= self.vwap_period <= 500, "vwap_period must be 20-500"
            assert 0.001 <= self.entry_threshold <= 0.05, "entry_threshold must be 0.1%-5%"
            assert 0.0001 <= self.exit_threshold <= 0.01, "exit_threshold must be 0.01%-1%"
            assert 1.0 <= self.min_volume_multiplier <= 3.0, "volume multiplier must be 1.0-3.0"
            assert 0.5 <= self.stop_loss_atr_mult <= 5.0, "stop_loss_atr_mult must be 0.5-5.0"
            assert 0.5 <= self.profit_target_atr_mult <= 5.0, "profit_target_atr_mult must be 0.5-5.0"
            assert 5 <= self.max_holding_periods <= 500, "max_holding_periods must be 5-500"
            assert 0.0001 <= self.min_atr_filter <= 0.01, "min_atr_filter must be 0.01%-1%"
            assert 0.01 <= self.max_atr_filter <= 0.10, "max_atr_filter must be 1%-10%"
            return True
        except AssertionError as e:
            raise ValueError(f"Invalid parameter: {e}")


# Default optimized parameters (will be updated by weekly optimizer)
DEFAULT_VWAP_PARAMS = VWAPParameters()

# Conservative parameters for initial testing
CONSERVATIVE_VWAP_PARAMS = VWAPParameters(
    vwap_period=150,
    entry_threshold=0.010,  # 1.0% - wider entry
    exit_threshold=0.003,   # 0.3% - wider exit
    min_volume_multiplier=2.0,  # Higher volume required
    stop_loss_atr_mult=2.5,  # Wider stop loss
    profit_target_atr_mult=1.0,  # Tighter profit target
    max_holding_periods=30,  # Exit faster
    min_atr_filter=0.005,  # Only trade higher volatility
    max_atr_filter=0.025,  # Avoid extreme volatility
)

# Aggressive parameters for high-frequency trading
AGGRESSIVE_VWAP_PARAMS = VWAPParameters(
    vwap_period=50,
    entry_threshold=0.004,  # 0.4% - tighter entry
    exit_threshold=0.001,   # 0.1% - tighter exit
    min_volume_multiplier=1.2,  # Lower volume requirement
    stop_loss_atr_mult=1.5,  # Tighter stop loss
    profit_target_atr_mult=2.0,  # Wider profit target
    max_holding_periods=120,  # Hold longer
    min_atr_filter=0.002,  # Trade in quieter markets
    max_atr_filter=0.04,  # Tolerate higher volatility
)


# Grid search parameter ranges for optimization
VWAP_OPTIMIZATION_GRID = {
    'vwap_period': [50, 75, 100, 150, 200],
    'entry_threshold': [0.003, 0.005, 0.007, 0.010, 0.012, 0.015],
    'exit_threshold': [0.001, 0.002, 0.003, 0.005],
    'min_volume_multiplier': [1.2, 1.5, 2.0, 2.5],
    'stop_loss_atr_mult': [1.5, 2.0, 2.5, 3.0],
    'profit_target_atr_mult': [1.0, 1.5, 2.0, 2.5],
    'max_holding_periods': [30, 60, 90, 120],
    'min_atr_filter': [0.002, 0.003, 0.005],
    'max_atr_filter': [0.025, 0.030, 0.035, 0.040],
}
