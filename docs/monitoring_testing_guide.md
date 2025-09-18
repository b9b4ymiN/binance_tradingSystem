# Trading System Monitoring & Testing Guide

## Overview

This guide covers the comprehensive monitoring and testing systems integrated into the trading bot. The original `monitoring_testing_system.py` has been restructured into modular packages for better organization and maintainability.

## Project Structure After Integration

```
trading_system/
├── monitoring/           # Performance monitoring package
│   ├── __init__.py
│   ├── metrics_collector.py    # Prometheus metrics collection
│   └── performance_monitor.py  # Real-time performance monitoring
├── testing/             # Comprehensive testing package
│   ├── __init__.py
│   ├── unit_tests.py         # Unit tests for components
│   ├── backtesting_engine.py # Strategy backtesting
│   ├── stress_tester.py      # System stress testing
│   └── test_runner.py        # Test execution orchestrator
└── docs/
    └── monitoring_testing_guide.md  # This documentation
```

## Monitoring System

### Features

- **Real-time Metrics Collection**: Using Prometheus for comprehensive metrics
- **Performance Monitoring**: System resources, trading performance, API latency
- **Alert System**: Configurable alerts for risk management
- **Dashboard Support**: Prometheus metrics compatible with Grafana

### Usage

#### Basic Monitoring Setup

```python
from engine.trading_engine import TradingEngine
from monitoring import PerformanceMonitor
from config.trading_config import TradingConfig

# Initialize trading engine
config = TradingConfig(...)
engine = TradingEngine(config)

# Start performance monitoring
monitor = PerformanceMonitor(engine.db_manager, config)
monitor.start_monitoring()

# Monitoring will run on port 9999
# Access metrics at: http://localhost:9999/metrics
```

#### Manual Metrics Collection

```python
from monitoring import MetricsCollector

# Initialize metrics collector
metrics = MetricsCollector()

# Record trading events
metrics.record_trade("rsi_bollinger", "BUY", "filled", 1.25)
metrics.record_api_latency("place_order", 0.25)
metrics.record_webhook("success")

# Update system metrics
metrics.update_system_metrics()
```

### Available Metrics

#### Trading Metrics
- `trades_total`: Total trades by strategy, side, and status
- `trade_duration_seconds`: Trade execution duration
- `current_pnl`: Current profit/loss
- `open_positions`: Number of open positions
- `win_rate_percent`: Current win rate

#### System Metrics
- `system_cpu_percent`: CPU usage
- `system_memory_percent`: Memory usage
- `system_disk_percent`: Disk usage
- `api_request_duration_seconds`: API latency by endpoint

#### Risk Metrics
- `portfolio_risk_percent`: Current portfolio risk
- `max_drawdown_percent`: Maximum drawdown
- `daily_pnl`: Daily profit/loss

### Alert Configuration

```python
# Default alert thresholds (configurable)
alert_thresholds = {
    'max_drawdown': 0.10,        # 10% max drawdown
    'win_rate_min': 0.45,        # Minimum 45% win rate
    'daily_loss_limit': 0.05,    # 5% daily loss limit
    'api_latency_max': 5.0,      # 5 second max API latency
    'error_rate_max': 0.05       # 5% max error rate
}
```

## Testing System

### Testing Components

#### 1. Unit Tests

Comprehensive unit tests for all trading system components:

```python
from testing import TradingBotTests
import unittest

# Run unit tests
unittest.main(module='testing.unit_tests', verbosity=2)
```

**Test Coverage:**
- Kelly Criterion position sizing
- RSI calculation accuracy
- Bollinger Bands calculation
- API request handling
- Risk limits enforcement
- Database operations
- Webhook signature validation

#### 2. Backtesting Engine

Strategy validation using historical data:

```python
from testing import BacktestingEngine

# Define strategy function
def my_strategy(historical_data, symbol):
    # Your strategy logic here
    if len(historical_data) < 20:
        return None

    # Example: Simple RSI strategy
    current_rsi = calculate_rsi(historical_data)
    if current_rsi < 30:
        return {
            'symbol': symbol,
            'action': 'buy',
            'stop_loss': historical_data[-1]['close'] * 0.98,
            'take_profit': historical_data[-1]['close'] * 1.04
        }
    return None

# Run backtest
backtester = BacktestingEngine(my_strategy, initial_balance=10000)
results = backtester.run_backtest(historical_data, "BTCUSDT")

print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

**Backtest Report Includes:**
- Total trades and win rate
- Profit/Loss metrics
- Maximum drawdown
- Average win/loss
- Profit factor
- Equity curve
- Individual trade details

#### 3. Stress Testing

System resilience and performance testing:

```python
from testing import StressTester

# Initialize with trading engine
stress_tester = StressTester(trading_engine)
results = stress_tester.run_stress_tests()

print("Stress Test Results:")
for test_name, result in results.items():
    print(f"{test_name}: {result}")
```

**Stress Tests Include:**
- High-frequency signal processing
- Memory usage under sustained load
- Concurrent trade processing
- API failure simulation

### Running All Tests

#### Option 1: Quick Test Runner

```python
from testing.test_runner import run_all_tests, run_performance_tests

# Run unit tests
run_all_tests()

# Run performance tests (requires trading engine instance)
monitor = run_performance_tests(trading_engine)
```

#### Option 2: Individual Test Execution

```bash
# Unit tests only
python -m testing.unit_tests

# Full test runner
python testing/test_runner.py
```

## Integration with Main Application

### Adding Monitoring to Existing System

1. **Update main.py**:

```python
from monitoring import PerformanceMonitor

def main():
    config = TradingConfig(...)
    trading_engine = TradingEngine(config)

    # Start monitoring
    monitor = PerformanceMonitor(trading_engine.db_manager, config)
    monitor.start_monitoring()

    # Start trading
    trading_engine.start_auto_trading()

    # Start webhook server
    try:
        trading_engine.webhook_handler.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        trading_engine.stop_trading()
```

2. **Enhanced Error Handling**:

```python
from monitoring import MetricsCollector

class TradingEngine:
    def __init__(self, config):
        # ... existing code ...
        self.metrics = MetricsCollector()

    def process_signal(self, signal_data):
        start_time = time.time()
        try:
            # ... existing processing ...
            duration = time.time() - start_time
            self.metrics.record_trade(
                strategy=signal_data.get('strategy', 'manual'),
                side=action.upper(),
                status='success',
                duration=duration
            )
            return result
        except Exception as e:
            self.metrics.record_error('signal_processing')
            # ... error handling ...
```

## Best Practices

### Monitoring
1. **Resource Monitoring**: Always monitor system resources in production
2. **Alert Configuration**: Set appropriate alert thresholds for your risk tolerance
3. **Metrics Retention**: Configure appropriate retention policies for metrics data
4. **Dashboard Setup**: Use Grafana for visual monitoring dashboards

### Testing
1. **Regular Testing**: Run unit tests before each deployment
2. **Strategy Validation**: Always backtest strategies before live trading
3. **Stress Testing**: Regularly perform stress tests to ensure system reliability
4. **Test Data**: Use representative historical data for backtesting

### Performance Optimization
1. **Metric Collection**: Balance metric detail with performance impact
2. **Alert Frequency**: Avoid alert spam with appropriate thresholds
3. **Resource Monitoring**: Keep monitoring overhead minimal
4. **Database Optimization**: Regular database maintenance for performance metrics

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Default monitoring port is 9999
   - Solution: Configure alternative port in monitor initialization

2. **Memory Usage**: High memory usage during stress tests
   - Solution: Normal behavior, monitor for memory leaks over time

3. **Test Failures**: Import errors in unit tests
   - Solution: Ensure all dependencies are installed and PYTHONPATH is correct

4. **Metrics Not Updating**: Prometheus metrics server not accessible
   - Solution: Check firewall settings and port configuration

### Performance Considerations

- Monitoring adds ~2-5% system overhead
- Prometheus metrics are lightweight but accumulate over time
- Stress tests are resource-intensive, run during low-activity periods
- Database queries for metrics calculation may impact performance on large datasets

## Configuration

### Environment Variables

```bash
# Optional monitoring configuration
export MONITORING_PORT=9999
export METRICS_UPDATE_INTERVAL=30
export ALERT_WEBHOOK_URL="https://your-alert-endpoint.com"
```

### Database Tables

The monitoring system creates additional database tables:
- `alerts`: Alert history and notifications
- Performance metrics tables for historical tracking

## Dependencies

Required packages (added to requirements.txt):
```
psutil>=5.9.0          # System resource monitoring
prometheus_client>=0.18.0  # Metrics collection and exposition
```

## Conclusion

The integrated monitoring and testing system provides comprehensive observability and validation for the trading system. Use these tools to:

- Monitor system performance in real-time
- Validate trading strategies before deployment
- Ensure system reliability under load
- Track trading performance and risk metrics
- Receive alerts for critical conditions

For questions or issues, refer to the main project documentation or create an issue in the project repository.