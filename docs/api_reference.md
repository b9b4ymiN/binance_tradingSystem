# API Reference

## Core Components

### TradingConfig
**Location**: `config/trading_config.py`

Configuration class containing all trading parameters and settings.

```python
@dataclass
class TradingConfig:
    api_key: str                    # Binance API key
    api_secret: str                 # Binance API secret
    base_url: str                   # Binance API base URL
    testnet_url: str               # Testnet URL for testing
    use_testnet: bool              # Use testnet flag
    max_position_risk: float       # Maximum risk per trade (default: 2%)
    kelly_fraction: float          # Kelly Criterion scaling (default: 25%)
    max_concurrent_positions: int  # Maximum open positions (default: 3)
    max_daily_trades: int         # Daily trade limit (default: 10)
    rsi_oversold: int             # RSI oversold level (default: 30)
    rsi_overbought: int           # RSI overbought level (default: 70)
    bollinger_period: int         # Bollinger Bands period (default: 20)
    bollinger_std: float          # Bollinger Bands standard deviation (default: 2.0)
    webhook_secret: str           # Webhook security secret
    allowed_ips: List[str]        # Whitelisted IP addresses
    db_path: str                  # Database file path
```

### BinanceAPI
**Location**: `core/binance_api.py`

Handles all Binance API interactions with comprehensive error handling.

#### Methods

**`__init__(config: TradingConfig)`**
- Initializes API client with configuration
- Sets up session headers and base URL

**`get_account_info() -> Dict`**
- Returns comprehensive account information
- Requires API permissions

**`get_symbol_info(symbol: str) -> Dict`**
- Returns detailed symbol information including filters
- Raises ValueError if symbol not found

**`get_current_price(symbol: str) -> float`**
- Returns current market price for symbol
- Fast execution for real-time data

**`get_klines(symbol: str, interval: str, limit: int = 100) -> List[List]`**
- Returns historical candlestick data
- Used for technical analysis calculations

**`place_order(symbol: str, side: str, order_type: str, quantity: float, price: float = None, stop_price: float = None, time_in_force: str = 'GTC') -> Dict`**
- Places orders on Binance
- Supports MARKET, LIMIT, STOP_MARKET order types
- Returns order details including order ID

**`cancel_order(symbol: str, order_id: str) -> Dict`**
- Cancels existing orders
- Returns cancellation confirmation

**`get_open_orders(symbol: str = None) -> List[Dict]`**
- Returns all open orders for account or specific symbol

### DatabaseManager
**Location**: `core/database.py`

Manages SQLite database operations for trade logging and performance tracking.

#### Methods

**`__init__(db_path: str)`**
- Initializes database connection
- Creates tables if they don't exist

**`init_database()`**
- Creates all required database tables:
  - `trades`: Trade history and details
  - `portfolio`: Position tracking
  - `performance`: Daily performance metrics
  - `risk_metrics`: Risk exposure tracking

**`get_connection()`**
- Context manager for database connections
- Ensures proper connection cleanup

### RiskManager
**Location**: `core/risk_manager.py`

Implements Kelly Criterion-based position sizing and risk controls.

#### Methods

**`__init__(config: TradingConfig, db_manager: DatabaseManager)`**
- Initializes risk manager with configuration
- Sets up position lock for thread safety

**`calculate_position_size(symbol: str, entry_price: float, stop_loss: float, win_rate: float = 0.6, avg_win: float = 0.02, avg_loss: float = 0.01) -> float`**
- Calculates optimal position size using Kelly Criterion
- Applies conservative scaling and risk limits
- Returns position size in base currency

**`check_risk_limits(new_position_risk: float) -> bool`**
- Validates new position against risk limits
- Checks concurrent positions, total risk exposure, daily trade limits
- Returns True if position is within limits

## Strategy Components

### BaseStrategy
**Location**: `strategies/base_strategy.py`

Abstract base class for all trading strategies.

#### Abstract Methods

**`generate_signal(symbol: str) -> Optional[Dict]`**
- Must be implemented by all strategies
- Returns trading signal or None

**`strategy_name -> str`**
- Property returning strategy identifier

**`expected_win_rate -> float`**
- Property returning expected win rate (0.0 to 1.0)

### RSIBollingerStrategy
**Location**: `strategies/rsi_bollinger.py`

RSI-Bollinger Bands scalping strategy with 69-80% expected win rate.

#### Configuration
- **Timeframe**: 1-minute charts
- **RSI Period**: 14
- **Bollinger Bands**: 20-period, 2.0 standard deviation
- **Entry Conditions**: RSI oversold/overbought + price at BB extremes

### BreakoutSwingStrategy
**Location**: `strategies/breakout_swing.py`

Volume-confirmed breakout strategy with 55-65% expected win rate.

#### Configuration
- **Timeframe**: 4-hour charts
- **Volume Threshold**: 1.5x average volume
- **EMA**: 21 and 50 periods
- **Risk-Reward**: 15% target, 2% stop loss

## Technical Analysis

### TechnicalAnalysis
**Location**: `analysis/technical_analysis.py`

Static methods for calculating technical indicators.

#### Methods

**`calculate_rsi(prices: List[float], period: int = 14) -> float`**
- Calculates Relative Strength Index
- Returns value between 0-100

**`calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]`**
- Returns (upper_band, middle_band, lower_band)
- Uses simple moving average and standard deviation

**`calculate_ema(prices: List[float], period: int) -> float`**
- Calculates Exponential Moving Average
- More responsive than simple moving average

**`calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float`**
- Calculates Average True Range
- Used for volatility-based position sizing

## Webhook Integration

### WebhookHandler
**Location**: `webhook/handler.py`

Secure Flask server for handling TradingView webhooks.

#### Methods

**`__init__(config: TradingConfig, trading_engine)`**
- Initializes Flask app with security settings
- Sets up routes for webhook and health check

**`setup_routes()`**
- Configures `/webhook` POST endpoint
- Configures `/health` GET endpoint for monitoring

#### Webhook Endpoints

**`POST /webhook`**
- Processes TradingView signals
- Validates IP whitelist and signatures
- Returns JSON response with processing result

**`GET /health`**
- Health check endpoint
- Returns system status and timestamp

#### Security Features
- IP address whitelisting
- HMAC signature validation
- Request timeout protection
- Comprehensive error handling

## Trading Engine

### TradingEngine
**Location**: `engine/trading_engine.py`

Main orchestrator coordinating all trading operations.

#### Methods

**`__init__(config: TradingConfig)`**
- Initializes all components
- Sets up strategy instances
- Configures webhook handler

**`process_signal(signal_data: Dict) -> Dict`**
- Main signal processing method
- Validates symbols and calculates position sizes
- Executes trades and manages orders

**`start_auto_trading()`**
- Starts automated trading loop in background thread
- Monitors positions and generates signals
- Runs every 30 seconds

**`stop_trading()`**
- Stops all trading operations
- Graceful shutdown of background processes

**`get_performance_report() -> Dict`**
- Generates comprehensive performance metrics
- Includes win rate, total PnL, trade counts

#### Signal Processing Flow
1. Validate incoming signal data
2. Check symbol validity
3. Generate strategy-specific parameters if missing
4. Calculate optimal position size
5. Validate against risk limits
6. Execute trade on Binance
7. Place stop-loss and take-profit orders
8. Log trade to database

## Error Handling

### Exception Types
- `RequestException`: Network and API errors
- `ValueError`: Invalid symbols or parameters
- `JSONDecodeError`: Malformed webhook data

### Logging Levels
- `INFO`: Normal operations and trade executions
- `WARNING`: Risk limit violations and validation failures
- `ERROR`: API failures and critical errors

### Recovery Mechanisms
- Automatic retry on network errors
- Graceful degradation on API failures
- Database transaction rollback on errors
- Position monitoring for orphaned orders