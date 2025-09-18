import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from .ml_trading_strategy import MLTradingStrategy

logger = logging.getLogger(__name__)

def create_ml_training_pipeline():
    """Create a complete ML training pipeline"""

    def train_crypto_ml_model(symbol: str, timeframe: str = '1h', days: int = 365):
        """Train ML model for a specific crypto symbol"""

        print(f"🤖 Starting ML training pipeline for {symbol}")
        print(f"📊 Training data: {days} days of {timeframe} data")

        # Initialize components
        ml_strategy = MLTradingStrategy(config={})

        # Generate realistic sample data
        df = _generate_sample_data(symbol, days, timeframe)

        print("📊 Training ML models...")
        try:
            ml_strategy.train_strategy(df)

            print("🔍 Generating sample prediction...")
            latest_signal = ml_strategy.generate_signal(df.tail(100), symbol)

            if latest_signal:
                print(f"✅ Sample signal generated:")
                print(f"   Action: {latest_signal['action']}")
                print(f"   Confidence: {latest_signal['confidence']:.2%}")
                print(f"   Kelly Fraction: {latest_signal['kelly_fraction']:.3f}")
                print(f"   ML Probability: {latest_signal['signal_components']['ml_probability']:.2%}")
                print(f"   Sentiment Score: {latest_signal['sentiment_data']['combined_sentiment']:.3f}")

                # Validate signal quality
                quality_check = ml_strategy.validate_signal_quality(latest_signal)
                print(f"   Signal Quality: {quality_check['grade']} ({quality_check['quality_score']}/100)")

            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)

            # Save models
            model_path = f"models/{symbol.lower()}"
            ml_strategy.save_strategy(model_path)
            print(f"💾 Models saved to {model_path}/")

            # Performance summary
            performance = ml_strategy.get_model_performance_summary()
            print(f"📈 Model Performance:")
            print(f"   Direction Accuracy: {performance.get('direction_accuracy', 0):.1%}")
            print(f"   Features Used: {performance.get('feature_count', 0)}")

            return ml_strategy

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            raise

    return train_crypto_ml_model

def _generate_sample_data(symbol: str, days: int, timeframe: str) -> pd.DataFrame:
    """Generate realistic sample cryptocurrency data"""

    # Determine frequency based on timeframe
    freq_map = {
        '1m': 'T',
        '5m': '5T',
        '15m': '15T',
        '1h': 'H',
        '4h': '4H',
        '1d': 'D'
    }

    freq = freq_map.get(timeframe, 'H')
    periods = days * 24 if timeframe == '1h' else days * 24 * 60  # Adjust for timeframe

    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Limit to reasonable number of periods
    dates = dates[:min(len(dates), periods)]

    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol

    # Base prices for different symbols
    base_prices = {
        'BTCUSDT': 45000,
        'ETHUSDT': 2800,
        'ADAUSDT': 0.45,
        'DOTUSDT': 8.5,
        'LINKUSDT': 12.0,
        'BNBUSDT': 320
    }

    starting_price = base_prices.get(symbol, 1000)

    # Generate realistic price movements
    volatility = 0.02  # 2% volatility
    trend = np.random.uniform(-0.0005, 0.0005)  # Small trend component

    returns = np.random.normal(trend, volatility, len(dates))

    # Add some autocorrelation for more realistic price movements
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]

    # Generate prices using geometric Brownian motion
    prices = [starting_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))  # Ensure positive prices

    prices = prices[1:]  # Remove the initial price

    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility_factor = np.random.uniform(0.005, 0.02)

        high = close * (1 + volatility_factor * np.random.uniform(0, 1))
        low = close * (1 - volatility_factor * np.random.uniform(0, 1))

        if i == 0:
            open_price = close * np.random.uniform(0.99, 1.01)
        else:
            open_price = data[i-1]['close'] * np.random.uniform(0.995, 1.005)

        # Ensure OHLC relationships are valid
        high = max(high, close, open_price)
        low = min(low, close, open_price)

        # Generate volume with some correlation to price movement
        base_volume = np.random.uniform(100, 1000)
        price_change = abs((close - open_price) / open_price)
        volume = base_volume * (1 + price_change * 10)  # Higher volume with bigger moves

        data.append({
            'timestamp': date,
            'open': round(open_price, 6),
            'high': round(high, 6),
            'low': round(low, 6),
            'close': round(close, 6),
            'volume': round(volume, 2)
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df

def run_batch_training(symbols: list, timeframe: str = '1h', days: int = 365):
    """Run batch training for multiple symbols"""

    print(f"🚀 Starting batch ML training for {len(symbols)} symbols")
    print("=" * 60)

    train_model = create_ml_training_pipeline()
    results = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Training {symbol}...")
        try:
            trained_strategy = train_model(symbol, timeframe, days)
            results[symbol] = {
                'status': 'success',
                'strategy': trained_strategy,
                'performance': trained_strategy.get_model_performance_summary()
            }
            print(f"✅ {symbol} training completed successfully!")

        except Exception as e:
            results[symbol] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ {symbol} training failed: {e}")

        print("-" * 40)

    # Summary report
    print(f"\n📊 Batch Training Summary:")
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {sum(1 for r in results.values() if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results.values() if r['status'] == 'failed')}")

    # Performance summary for successful trainings
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}

    if successful_results:
        print(f"\n📈 Performance Summary:")
        for symbol, result in successful_results.items():
            perf = result['performance']
            print(f"{symbol:10} | Accuracy: {perf.get('direction_accuracy', 0):.1%} | "
                  f"Features: {perf.get('feature_count', 0):3d}")

    return results

def create_training_config(symbol: str, custom_params: dict = None) -> dict:
    """Create training configuration for a specific symbol"""

    base_config = {
        'symbol': symbol,
        'timeframe': '1h',
        'training_days': 365,
        'validation_split': 0.2,
        'target_periods': 5,  # Predict 5 periods ahead
        'min_confidence': 0.6,
        'ml_weight': 0.6,
        'sentiment_weight': 0.4,
        'model_params': {
            'rf_n_estimators': 100,
            'rf_max_depth': 10,
            'gb_n_estimators': 100,
            'gb_learning_rate': 0.1,
            'lstm_epochs': 50,
            'lstm_batch_size': 32
        }
    }

    if custom_params:
        base_config.update(custom_params)

    return base_config

if __name__ == "__main__":
    # Example usage
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

    # Single symbol training
    print("🎯 Single Symbol Training Example:")
    train_model = create_ml_training_pipeline()
    btc_strategy = train_model('BTCUSDT')

    # Batch training
    print("\n🎯 Batch Training Example:")
    results = run_batch_training(symbols)

    print("\n✅ ML Training Pipeline Demo Complete!")