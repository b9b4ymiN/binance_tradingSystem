import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Advanced feature engineering for ML models"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""

        features_df = df.copy()

        # Price-based features
        features_df['returns'] = df['close'].pct_change()
        features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features_df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features_df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features_df['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1

        # Technical indicators
        features_df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        features_df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        features_df['macd'], features_df['macd_signal'], features_df['macd_hist'] = talib.MACD(df['close'])

        # Moving averages and their relationships
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            features_df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            features_df[f'price_to_sma_{period}'] = df['close'] / features_df[f'sma_{period}'] - 1
            features_df[f'price_to_ema_{period}'] = df['close'] / features_df[f'ema_{period}'] - 1

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
        features_df['bb_upper'] = bb_upper
        features_df['bb_lower'] = bb_lower
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # Volatility features
        features_df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features_df['volatility_20'] = features_df['returns'].rolling(20).std()
        features_df['volatility_ratio'] = features_df['volatility_20'] / features_df['volatility_20'].rolling(50).mean()

        # Volume features
        if 'volume' in df.columns:
            features_df['volume_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
            features_df['volume_ratio'] = df['volume'] / features_df['volume_sma_20']
            features_df['obv'] = talib.OBV(df['close'], df['volume'])
            features_df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])

            # Volume-price relationship
            features_df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            features_df['price_to_vwap'] = df['close'] / features_df['vwap'] - 1

        # Momentum indicators
        features_df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        features_df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        features_df['stoch_k'], features_df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])

        # Time-based features
        features_df['hour'] = pd.to_datetime(df.index).hour if isinstance(df.index, pd.DatetimeIndex) else 0
        features_df['day_of_week'] = pd.to_datetime(df.index).dayofweek if isinstance(df.index, pd.DatetimeIndex) else 0
        features_df['month'] = pd.to_datetime(df.index).month if isinstance(df.index, pd.DatetimeIndex) else 0

        # Market structure features
        features_df['high_low_ratio'] = df['high'] / df['low'] - 1
        features_df['open_close_ratio'] = df['close'] / df['open'] - 1

        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            features_df[f'returns_mean_{window}'] = features_df['returns'].rolling(window).mean()
            features_df[f'returns_std_{window}'] = features_df['returns'].rolling(window).std()
            features_df[f'returns_skew_{window}'] = features_df['returns'].rolling(window).skew()
            features_df[f'returns_kurt_{window}'] = features_df['returns'].rolling(window).kurt()

        # Remove infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)

        self.feature_names = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

        return features_df

    def get_feature_importance_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze feature importance and correlations"""

        features_df = self.create_features(df)
        feature_columns = self.feature_names

        # Calculate correlations with returns
        returns = features_df['returns']
        correlations = {}

        for feature in feature_columns:
            if feature != 'returns':
                corr = features_df[feature].corr(returns)
                if not np.isnan(corr):
                    correlations[feature] = abs(corr)

        # Sort by correlation strength
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'top_features': top_features,
            'total_features': len(feature_columns),
            'correlation_analysis': correlations
        }