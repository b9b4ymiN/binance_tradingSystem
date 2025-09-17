from abc import ABC, abstractmethod
from typing import Optional, Dict
from config.trading_config import TradingConfig
from core.binance_api import BinanceAPI
from core.risk_manager import RiskManager


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config: TradingConfig, binance_api: BinanceAPI, risk_manager: RiskManager):
        self.config = config
        self.api = binance_api
        self.risk_manager = risk_manager
    
    @abstractmethod
    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signal for given symbol"""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy name"""
        pass
    
    @property
    @abstractmethod
    def expected_win_rate(self) -> float:
        """Return expected win rate for this strategy"""
        pass