from dataclasses import dataclass
from typing import List


@dataclass
class TradingConfig:
    """Complete trading configuration based on research findings"""
    # Binance API Configuration
    api_key: str
    api_secret: str
    base_url: str = "https://api.binance.com"
    testnet_url: str = "https://testnet.binance.vision"
    use_testnet: bool = True
    
    # Risk Management (Kelly Criterion Implementation)
    max_position_risk: float = 0.05  # 2% max risk per trade
    kelly_fraction: float = 0.50     # Conservative Kelly scaling
    max_concurrent_positions: int = 5
    max_daily_trades: int = 100
    
    # Strategy Parameters (From research)
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Webhook Security
    webhook_secret: str = "your_webhook_secret_here"
    allowed_ips: List[str] = None
    
    # Database Configuration
    db_path: str = "trading_data.db"
    
    def __post_init__(self):
        if self.allowed_ips is None:
            # TradingView webhook IPs from research
            self.allowed_ips = [
                "52.89.214.238", "34.212.75.30", 
                "54.218.53.128", "52.32.178.7",
                "127.0.0.1"  # For local testing
            ]