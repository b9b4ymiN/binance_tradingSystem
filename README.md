# Advanced Binance Trading System

A comprehensive, modular cryptocurrency trading system designed specifically for crypto traders, featuring TradingView integration, advanced risk management, and research-backed trading strategies.

## 🎯 Project Goals

### Primary Objectives
- **Automated Trading**: Execute trades automatically based on TradingView signals and internal strategies
- **Risk Management**: Implement Kelly Criterion-based position sizing with strict risk controls
- **High Win Rates**: Achieve 69-80% win rate using RSI-Bollinger Bands scalping strategy
- **Professional Grade**: Production-ready system with comprehensive logging and error handling
- **Market Focus**: Optimized for cryptocurrency traders with localized considerations

### Key Features
- **Multiple Trading Strategies**: RSI-Bollinger Bands scalping and Breakout swing trading
- **Advanced Risk Management**: Kelly Criterion position sizing, daily limits, portfolio risk controls
- **TradingView Integration**: Secure webhook handling for automated signal processing
- **Real-time Monitoring**: Live position tracking and performance analytics
- **Modular Architecture**: Clean, maintainable code structure for easy extension
- **Database Logging**: Comprehensive trade and performance tracking

## 🏗️ Architecture

### Modular Structure
```
trading_system/
├── config/           # Configuration and settings
├── core/            # Core trading components
├── strategies/      # Trading strategy implementations
├── analysis/        # Technical analysis tools
├── webhook/         # TradingView webhook handler
├── engine/          # Main trading engine
├── utils/           # Utilities and helpers
├── docs/            # Documentation
└── main.py          # Application entry point
```

### Core Components
- **TradingEngine**: Orchestrates all trading operations
- **BinanceAPI**: Secure API integration with comprehensive error handling
- **RiskManager**: Kelly Criterion-based position sizing and risk controls
- **DatabaseManager**: Trade logging and performance tracking
- **WebhookHandler**: Secure TradingView signal processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Binance account (testnet recommended for testing)
- TradingView account (for signal generation)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_SECRET_KEY="your_secret_key"
   export WEBHOOK_SECRET="your_webhook_secret"
   ```

4. Run the trading system:
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Trading Parameters
- **Max Position Risk**: 2% per trade (configurable)
- **Kelly Fraction**: 25% conservative scaling
- **Max Concurrent Positions**: 3 positions
- **Daily Trade Limit**: 10 trades per day

### Strategy Settings
- **RSI Oversold/Overbought**: 30/70 levels
- **Bollinger Bands**: 20-period, 2.0 standard deviation
- **Timeframes**: 1-minute for scalping, 4-hour for swing trading

## 📊 Trading Strategies

### 1. RSI-Bollinger Bands Scalping
- **Expected Win Rate**: 69-80%
- **Timeframe**: 1-minute charts
- **Entry**: RSI oversold/overbought + price at Bollinger Band extremes
- **Exit**: Mean reversion to middle Bollinger Band

### 2. Breakout Swing Trading
- **Expected Win Rate**: 55-65%
- **Timeframe**: 4-hour charts
- **Entry**: Volume-confirmed breakouts above resistance
- **Exit**: 15% profit target or 2% stop loss

## 🛡️ Risk Management

### Kelly Criterion Implementation
- Dynamic position sizing based on win rate and risk-reward ratio
- Conservative 25% Kelly fraction scaling for stability
- Real-time risk monitoring and position limits

### Safety Features
- Daily trade limits to prevent overtrading
- Maximum concurrent position limits
- Portfolio-wide risk exposure controls
- Automatic stop-loss and take-profit orders

## 🔗 TradingView Integration

### Webhook Setup
1. Create alerts in TradingView using provided Pine Script
2. Configure webhook URL: `http://your-server:5000/webhook`
3. Set webhook secret for security validation

### Signal Format
```json
{
  "action": "buy",
  "symbol": "BTCUSDT",
  "price": 45000,
  "strategy": "rsi_bollinger_scalping",
  "stop_loss": 44100,
  "take_profit": 45900
}
```

## 📈 Performance Tracking

### Metrics Available
- Total trades and win rate
- Profit/Loss tracking
- Strategy-specific performance
- Risk metrics and drawdown analysis

### Database Schema
- Trade history with full order details
- Portfolio tracking and position management
- Performance metrics by date
- Risk exposure monitoring

## 🔒 Security Features

### API Security
- HMAC SHA256 signature validation
- Secure environment variable configuration
- IP whitelist for webhook access
- Rate limiting and error handling

### Best Practices
- Testnet mode for development
- Comprehensive logging for audit trails
- Error recovery and failsafe mechanisms
- Secure credential management

## 📚 Documentation

Detailed documentation available in `/docs/`:
- **API Reference**: Complete API documentation
- **Strategy Guide**: In-depth strategy explanations
- **Setup Guide**: Detailed installation and configuration

## ⚠️ Important Notes

### Risk Disclaimer
- Cryptocurrency trading involves substantial risk
- Past performance does not guarantee future results
- Use testnet mode for testing and development
- Start with small position sizes in live trading

### Compliance
- Designed for educational and research purposes
- Users responsible for compliance with local regulations
- Market considerations included but not guaranteed

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines and submit pull requests for:
- Strategy improvements
- Risk management enhancements
- Bug fixes and optimizations
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Built for crypto traders seeking professional-grade automated trading solutions.**