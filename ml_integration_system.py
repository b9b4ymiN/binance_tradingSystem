# Machine Learning Integration for Crypto Trading
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import talib
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

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
        
        # Cross-asset features (if multiple symbols available)
        # This would be implemented when multiple asset data is available
        
        # Remove infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        self.feature_names = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return features_df

class MLPricePrediction:
    """Machine Learning models for price prediction"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineering()
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, target_periods: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training ML models"""
        
        # Create features
        features_df = self.feature_engineer.create_features(df)
        
        # Create target variable (future price movement)
        features_df['future_return'] = features_df['close'].shift(-target_periods) / features_df['close'] - 1
        features_df['target_direction'] = (features_df['future_return'] > 0).astype(int)
        features_df['target_magnitude'] = features_df['future_return']
        
        # Remove rows with NaN targets
        features_df = features_df.dropna()
        
        # Prepare feature matrix
        feature_columns = self.feature_engineer.feature_names
        X = features_df[feature_columns].values
        
        # Prepare targets
        y_direction = features_df['target_direction'].values
        y_magnitude = features_df['target_magnitude'].values
        
        return X, y_direction, y_magnitude
    
    def train_models(self, df: pd.DataFrame, target_periods: int = 5):
        """Train ML models for price prediction"""
        
        logger.info("Preparing training data...")
        X, y_direction, y_magnitude = self.prepare_data(df, target_periods)
        
        if len(X) < 100:  # Minimum data requirement
            raise ValueError("Insufficient data for training. Need at least 100 samples.")
        
        # Split data chronologically
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_dir_train, y_dir_test = y_direction[:split_point], y_direction[split_point:]
        y_mag_train, y_mag_test = y_magnitude[:split_point], y_magnitude[split_point:]
        
        # Scale features
        self.feature_engineer.scaler.fit(X_train)
        X_train_scaled = self.feature_engineer.scaler.transform(X_train)
        X_test_scaled = self.feature_engineer.scaler.transform(X_test)
        
        logger.info("Training direction classifier...")
        # Direction classifier (Random Forest)
        self.models['direction_classifier'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.models['direction_classifier'].fit(X_train_scaled, y_dir_train)
        
        # Evaluate direction classifier
        dir_accuracy = self.models['direction_classifier'].score(X_test_scaled, y_dir_test)
        logger.info(f"Direction classifier accuracy: {dir_accuracy:.3f}")
        
        logger.info("Training magnitude regressor...")
        # Magnitude regressor (Gradient Boosting)
        self.models['magnitude_regressor'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.models['magnitude_regressor'].fit(X_train_scaled, y_mag_train)
        
        # Evaluate magnitude regressor
        mag_predictions = self.models['magnitude_regressor'].predict(X_test_scaled)
        mag_mse = mean_squared_error(y_mag_test, mag_predictions)
        logger.info(f"Magnitude regressor MSE: {mag_mse:.6f}")
        
        logger.info("Training LSTM model...")
        # LSTM model for sequential patterns
        self.models['lstm'] = self._build_lstm_model(X_train_scaled, y_mag_train)
        
        self.is_trained = True
        logger.info("All models trained successfully!")
    
    def _build_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray) -> tf.keras.Model:
        """Build and train LSTM model"""
        
        # Reshape data for LSTM (samples, timesteps, features)
        sequence_length = 20
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_train)):
            X_sequences.append(X_train[i-sequence_length:i])
            y_sequences.append(y_train[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        model.fit(
            X_sequences, y_sequences,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return model
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Generate ML-based predictions"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features for the latest data point
        features_df = self.feature_engineer.create_features(df)
        feature_columns = self.feature_engineer.feature_names
        
        # Get the latest features
        latest_features = features_df[feature_columns].iloc[-1:].values
        latest_features_scaled = self.feature_engineer.scaler.transform(latest_features)
        
        # Direction prediction
        direction_prob = self.models['direction_classifier'].predict_proba(latest_features_scaled)[0]
        direction_prediction = direction_prob[1]  # Probability of positive movement
        
        # Magnitude prediction
        magnitude_prediction = self.models['magnitude_regressor'].predict(latest_features_scaled)[0]
        
        # LSTM prediction (requires sequence)
        if len(features_df) >= 20:
            lstm_features = features_df[feature_columns].iloc[-20:].values
            lstm_features_scaled = self.feature_engineer.scaler.transform(lstm_features)
            lstm_features_reshaped = lstm_features_scaled.reshape(1, 20, -1)
            lstm_prediction = self.models['lstm'].predict(lstm_features_reshaped, verbose=0)[0][0]
        else:
            lstm_prediction = 0
        
        # Ensemble prediction
        ensemble_magnitude = (magnitude_prediction + lstm_prediction) / 2
        
        # Feature importance (for Random Forest)
        feature_importance = dict(zip(
            feature_columns,
            self.models['direction_classifier'].feature_importances_
        ))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'direction_probability': direction_prediction,
            'magnitude_prediction': magnitude_prediction,
            'lstm_prediction': lstm_prediction,
            'ensemble_magnitude': ensemble_magnitude,
            'confidence': direction_prob.max(),
            'signal_strength': abs(ensemble_magnitude),
            'top_features': top_features,
            'recommendation': self._generate_recommendation(direction_prediction, ensemble_magnitude)
        }
    
    def _generate_recommendation(self, direction_prob: float, magnitude: float) -> Dict:
        """Generate trading recommendation based on ML predictions"""
        
        # Define thresholds
        high_confidence_threshold = 0.7
        moderate_confidence_threshold = 0.6
        min_magnitude_threshold = 0.01  # 1% minimum expected movement
        
        recommendation = {
            'action': 'hold',
            'confidence': 'low',
            'reason': 'Insufficient signal strength'
        }
        
        if abs(magnitude) >= min_magnitude_threshold:
            if direction_prob >= high_confidence_threshold:
                recommendation = {
                    'action': 'buy',
                    'confidence': 'high',
                    'reason': f'High probability ({direction_prob:.1%}) of positive movement with {magnitude:.1%} expected return'
                }
            elif direction_prob <= (1 - high_confidence_threshold):
                recommendation = {
                    'action': 'sell',
                    'confidence': 'high',
                    'reason': f'High probability ({1-direction_prob:.1%}) of negative movement with {magnitude:.1%} expected return'
                }
            elif direction_prob >= moderate_confidence_threshold:
                recommendation = {
                    'action': 'buy',
                    'confidence': 'moderate',
                    'reason': f'Moderate probability ({direction_prob:.1%}) of positive movement'
                }
            elif direction_prob <= (1 - moderate_confidence_threshold):
                recommendation = {
                    'action': 'sell',
                    'confidence': 'moderate',
                    'reason': f'Moderate probability ({1-direction_prob:.1%}) of negative movement'
                }
        
        return recommendation
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'direction_classifier': self.models['direction_classifier'],
            'magnitude_regressor': self.models['magnitude_regressor'],
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_engineer.feature_names
        }
        
        joblib.dump(model_data, f"{filepath}_ml_models.pkl")
        
        # Save LSTM separately
        self.models['lstm'].save(f"{filepath}_lstm_model.h5")
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(f"{filepath}_ml_models.pkl")
            self.models['direction_classifier'] = model_data['direction_classifier']
            self.models['magnitude_regressor'] = model_data['magnitude_regressor']
            self.feature_engineer = model_data['feature_engineer']
            
            # Load LSTM
            self.models['lstm'] = tf.keras.models.load_model(f"{filepath}_lstm_model.h5")
            
            self.is_trained = True
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

class SentimentAnalysis:
    """Sentiment analysis for crypto markets"""
    
    def __init__(self):
        self.sentiment_sources = []
        
    def analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment for a given symbol"""
        # This would integrate with news APIs like Alpha Vantage, NewsAPI, etc.
        # For now, return mock data
        
        return {
            'overall_sentiment': 0.15,  # -1 to 1 scale
            'sentiment_score': 'positive',
            'news_count': 25,
            'confidence': 0.7,
            'key_topics': ['adoption', 'regulation', 'institutional investment']
        }
    
    def analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze social media sentiment"""
        # This would integrate with Twitter API, Reddit API, etc.
        # For now, return mock data
        
        return {
            'twitter_sentiment': 0.08,
            'reddit_sentiment': 0.12,
            'overall_social_sentiment': 0.10,
            'mention_count': 1250,
            'sentiment_trend': 'improving'
        }

class MLTradingStrategy:
    """ML-enhanced trading strategy"""
    
    def __init__(self, config):
        self.config = config
        self.ml_predictor = MLPricePrediction()
        self.sentiment_analyzer = SentimentAnalysis()
        self.is_trained = False
        
    def train_strategy(self, historical_data: pd.DataFrame):
        """Train the ML components of the strategy"""
        logger.info("Training ML trading strategy...")
        
        try:
            self.ml_predictor.train_models(historical_data)
            self.is_trained = True
            logger.info("ML strategy training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error training ML strategy: {e}")
            raise
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Generate trading signal using ML predictions"""
        
        if not self.is_trained:
            logger.warning("ML models not trained. Using basic signal generation.")
            return None
        
        try:
            # Get ML predictions
            ml_prediction = self.ml_predictor.predict(df)
            
            # Get sentiment analysis
            news_sentiment = self.sentiment_analyzer.analyze_news_sentiment(symbol)
            social_sentiment = self.sentiment_analyzer.analyze_social_sentiment(symbol)
            
            # Combine ML and sentiment signals
            signal = self._combine_signals(ml_prediction, news_sentiment, social_sentiment, df, symbol)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return None
    
    def _combine_signals(self, ml_prediction: Dict, news_sentiment: Dict, 
                        social_sentiment: Dict, df: pd.DataFrame, symbol: str) -> Dict:
        """Combine ML predictions with sentiment analysis"""
        
        # Weight factors
        ml_weight = 0.6
        news_weight = 0.25
        social_weight = 0.15
        
        # Normalize sentiment scores to match ML probability scale
        news_score = (news_sentiment['overall_sentiment'] + 1) / 2  # Convert -1,1 to 0,1
        social_score = (social_sentiment['overall_social_sentiment'] + 1) / 2
        
        # Combined probability
        combined_prob = (
            ml_prediction['direction_probability'] * ml_weight +
            news_score * news_weight +
            social_score * social_weight
        )
        
        # Signal strength
        signal_strength = (
            ml_prediction['signal_strength'] * ml_weight +
            abs(news_sentiment['overall_sentiment']) * news_weight +
            abs(social_sentiment['overall_social_sentiment']) * social_weight
        )
        
        # Calculate position size based on Kelly Criterion and ML confidence
        kelly_fraction = self._calculate_ml_kelly_fraction(
            ml_prediction['direction_probability'],
            ml_prediction['ensemble_magnitude'],
            signal_strength
        )
        
        # Determine action
        current_price = df['close'].iloc[-1]
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        if combined_prob > 0.65 and signal_strength > 0.02:
            action = 'buy'
            confidence = min(0.95, combined_prob + signal_strength)
            stop_loss = current_price - (2.5 * atr)
            take_profit = current_price + (abs(ml_prediction['ensemble_magnitude']) * current_price * 2)
            
        elif combined_prob < 0.35 and signal_strength > 0.02:
            action = 'sell'
            confidence = min(0.95, (1 - combined_prob) + signal_strength)
            stop_loss = current_price + (2.5 * atr)
            take_profit = current_price - (abs(ml_prediction['ensemble_magnitude']) * current_price * 2)
            
        else:
            return None  # No signal
        
        return {
            'action': action,
            'symbol': symbol,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': 'ml_enhanced',
            'confidence': confidence,
            'kelly_fraction': kelly_fraction,
            'ml_prediction': ml_prediction,
            'sentiment_scores': {
                'news': news_sentiment['overall_sentiment'],
                'social': social_sentiment['overall_social_sentiment']
            },
            'signal_components': {
                'ml_probability': ml_prediction['direction_probability'],
                'news_score': news_score,
                'social_score': social_score,
                'combined_probability': combined_prob,
                'signal_strength': signal_strength
            }
        }
    
    def _calculate_ml_kelly_fraction(self, probability: float, expected_return: float, 
                                   confidence: float) -> float:
        """Calculate Kelly fraction based on ML predictions"""
        
        # Adjust probability based on confidence
        adjusted_prob = probability * confidence
        
        # Estimate win/loss ratio from expected return
        if expected_return > 0:
            win_ratio = abs(expected_return) / 0.02  # Assume 2% average loss
        else:
            win_ratio = 0.02 / abs(expected_return)  # Assume 2% average win
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        kelly_fraction = (win_ratio * adjusted_prob - (1 - adjusted_prob)) / win_ratio
        
        # Apply conservative scaling and limits
        kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.1))  # Max 10% position
        
        return kelly_fraction

# Enhanced Pine Script Templates
ENHANCED_PINE_SCRIPT_STRATEGIES = {
    'ml_enhanced_strategy': '''
//@version=6
strategy("ML Enhanced Crypto Strategy", overlay=true, margin_long=100, margin_short=100)

// === INPUT PARAMETERS ===
// Webhook settings
webhook_url = input.string("YOUR_WEBHOOK_URL", title="Webhook URL")
webhook_secret = input.string("YOUR_SECRET", title="Webhook Secret")

// ML Signal parameters
ml_confidence_threshold = input.float(0.65, title="ML Confidence Threshold", minval=0.5, maxval=0.95, step=0.05)
use_sentiment = input.bool(true, title="Use Sentiment Analysis")
sentiment_weight = input.float(0.3, title="Sentiment Weight", minval=0.1, maxval=0.5, step=0.1)

// Risk Management
max_position_size = input.float(0.1, title="Max Position Size", minval=0.01, maxval=0.2, step=0.01)
atr_multiplier = input.float(2.5, title="ATR Stop Loss Multiplier", minval=1.0, maxval=5.0, step=0.5)
risk_reward_ratio = input.float(2.0, title="Risk:Reward Ratio", minval=1.0, maxval=5.0, step=0.5)

// === TECHNICAL INDICATORS ===
atr = ta.atr(14)
rsi = ta.rsi(close, 14)
[macd_line, signal_line, _] = ta.macd(close, 12, 26, 9)

// Bollinger Bands
bb_length = 20
bb_mult = 2.0
bb_basis = ta.sma(close, bb_length)
bb_dev = bb_mult * ta.stdev(close, bb_length)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev

// Moving Averages
ema_20 = ta.ema(close, 20)
ema_50 = ta.ema(close, 50)
sma_200 = ta.sma(close, 200)

// Volume
volume_sma = ta.sma(volume, 20)
volume_ratio = volume / volume_sma

// === ML SIGNAL SIMULATION ===
// In real implementation, this would come from your ML model via webhook
// For simulation, we create a composite signal
ml_direction_prob = 0.5 + (rsi - 50) / 100 + (close - bb_basis) / (bb_upper - bb_lower) * 0.1
ml_magnitude = (close - close[5]) / close[5]
ml_confidence = math.min(0.95, math.abs(ml_magnitude) * 50 + 0.5)

// Sentiment simulation (normally from external API)
news_sentiment = math.sin(bar_index / 10) * 0.3  // Simulated news sentiment
social_sentiment = math.cos(bar_index / 15) * 0.2  // Simulated social sentiment

// Combined signal
combined_signal = ml_direction_prob * (1 - sentiment_weight) + 
                 ((news_sentiment + social_sentiment + 2) / 4) * sentiment_weight

// === ENTRY CONDITIONS ===
// Long conditions
long_ml_signal = combined_signal > ml_confidence_threshold and ml_confidence > 0.6
long_technical = close > ema_20 and ema_20 > ema_50 and close > sma_200
long_volume = volume_ratio > 1.2
long_momentum = rsi > 40 and rsi < 70 and macd_line > signal_line

long_condition = long_ml_signal and long_technical and long_volume and long_momentum

// Short conditions
short_ml_signal = combined_signal < (1 - ml_confidence_threshold) and ml_confidence > 0.6
short_technical = close < ema_20 and ema_20 < ema_50 and close < sma_200
short_volume = volume_ratio > 1.2
short_momentum = rsi < 60 and rsi > 30 and macd_line < signal_line

short_condition = short_ml_signal and short_technical and short_volume and short_momentum

// === POSITION SIZING ===
// Kelly Criterion implementation
kelly_fraction = math.max(0, math.min((ml_direction_prob * 2 - 1) * 0.25, max_position_size))
position_size = kelly_fraction * 100  // Convert to percentage

// === RISK MANAGEMENT ===
var float entry_price = na
var float stop_loss = na
var float take_profit = na

// === STRATEGY EXECUTION ===
if long_condition and strategy.position_size == 0
    entry_price := close
    stop_loss := close - atr * atr_multiplier
    take_profit := close + atr * atr_multiplier * risk_reward_ratio
    
    strategy.entry("Long", strategy.long, qty=position_size)
    strategy.exit("Long Exit", "Long", stop=stop_loss, limit=take_profit)
    
    // Webhook alert
    alert_msg = '{"action": "buy", "symbol": "' + syminfo.ticker + '", "price": ' + str.tostring(close) + 
               ', "stop_loss": ' + str.tostring(stop_loss) + ', "take_profit": ' + str.tostring(take_profit) + 
               ', "strategy": "ml_enhanced", "confidence": ' + str.tostring(ml_confidence) + 
               ', "kelly_fraction": ' + str.tostring(kelly_fraction) + '}'
    alert(alert_msg, alert.freq_once_per_bar_close)

if short_condition and strategy.position_size == 0
    entry_price := close
    stop_loss := close + atr * atr_multiplier
    take_profit := close - atr * atr_multiplier * risk_reward_ratio
    
    strategy.entry("Short", strategy.short, qty=position_size)
    strategy.exit("Short Exit", "Short", stop=stop_loss, limit=take_profit)
    
    // Webhook alert
    alert_msg = '{"action": "sell", "symbol": "' + syminfo.ticker + '", "price": ' + str.tostring(close) + 
               ', "stop_loss": ' + str.tostring(stop_loss) + ', "take_profit": ' + str.tostring(take_profit) + 
               ', "strategy": "ml_enhanced", "confidence": ' + str.tostring(ml_confidence) + 
               ', "kelly_fraction": ' + str.tostring(kelly_fraction) + '}'
    alert(alert_msg, alert.freq_once_per_bar_close)

// === VISUALIZATION ===
plot(ema_20, color=color.blue, title="EMA 20")
plot(ema_50, color=color.orange, title="EMA 50")
plot(sma_200, color=color.red, title="SMA 200")

plotshape(long_condition, style=shape.triangleup, location=location.belowbar, 
          color=color.green, size=size.small, title="ML Long Signal")
plotshape(short_condition, style=shape.triangledown, location=location.abovebar, 
          color=color.red, size=size.small, title="ML Short Signal")

// Display ML metrics
var label ml_label = na
if barstate.islast
    label.delete(ml_label)
    ml_label := label.new(bar_index, high, 
                         "ML Confidence: " + str.tostring(ml_confidence, "#.##") + 
                         "\\nCombined Signal: " + str.tostring(combined_signal, "#.##") + 
                         "\\nKelly Fraction: " + str.tostring(kelly_fraction, "#.##"),
                         color=color.yellow, style=label.style_label_down, size=size.normal)
''',
    
    'adaptive_multi_timeframe': '''
//@version=6
strategy("Adaptive Multi-Timeframe Strategy", overlay=true, margin_long=100, margin_short=100)

// === MULTI-TIMEFRAME ANALYSIS ===
// Higher timeframe trend
htf_timeframe = input.timeframe("4h", title="Higher Timeframe")
htf_trend_length = input.int(50, title="HTF Trend Length")

// Get higher timeframe data
htf_close = request.security(syminfo.tickerid, htf_timeframe, close)
htf_ema = ta.ema(htf_close, htf_trend_length)
htf_trend = htf_close > htf_ema

// Current timeframe indicators
ctf_ema_fast = ta.ema(close, 21)
ctf_ema_slow = ta.ema(close, 50)
ctf_rsi = ta.rsi(close, 14)
ctf_atr = ta.atr(14)

// === MARKET REGIME DETECTION ===
volatility_period = 20
price_change = math.abs(close - close[volatility_period]) / close[volatility_period]
avg_volatility = ta.sma(price_change, volatility_period)

// Trend strength (ADX-like)
plus_dm = math.max(high - high[1], 0)
minus_dm = math.max(low[1] - low, 0)
true_range = math.max(high - low, math.max(math.abs(high - close[1]), math.abs(low - close[1])))

plus_di = 100 * ta.ema(plus_dm, 14) / ta.ema(true_range, 14)
minus_di = 100 * ta.ema(minus_dm, 14) / ta.ema(true_range, 14)
dx = 100 * math.abs(plus_di - minus_di) / (plus_di + minus_di)
adx = ta.ema(dx, 14)

// Market regime classification
is_trending = adx > 25
is_high_volatility = avg_volatility > 0.03  // 3% threshold
is_bull_trend = htf_trend and ctf_ema_fast > ctf_ema_slow
is_bear_trend = not htf_trend and ctf_ema_fast < ctf_ema_slow

// === ADAPTIVE STRATEGY SELECTION ===
// Bull trending market
bull_long_condition = is_bull_trend and is_trending and 
                     close > ctf_ema_fast and ctf_rsi > 45 and ctf_rsi < 70 and
                     volume > ta.sma(volume, 20)

// Bear trending market  
bear_short_condition = is_bear_trend and is_trending and
                      close < ctf_ema_fast and ctf_rsi < 55 and ctf_rsi > 30 and
                      volume > ta.sma(volume, 20)

// High volatility breakout
volatility_long = is_high_volatility and close > ta.highest(high, 20)[1] and volume > ta.sma(volume, 20) * 1.5
volatility_short = is_high_volatility and close < ta.lowest(low, 20)[1] and volume > ta.sma(volume, 20) * 1.5

// Sideways range trading
bb_upper = ta.bb(close, 20, 2)[0]
bb_lower = ta.bb(close, 20, 2)[2]
bb_middle = ta.bb(close, 20, 2)[1]

range_long = not is_trending and close <= bb_lower * 1.01 and ctf_rsi < 35
range_short = not is_trending and close >= bb_upper * 0.99 and ctf_rsi > 65

// === POSITION SIZING ===
base_position = input.float(0.05, title="Base Position Size")
volatility_adj = math.min(2.0, math.max(0.5, 1 / avg_volatility))
adaptive_position = base_position * volatility_adj

// === TRADE EXECUTION ===
var float current_sl = na
var float current_tp = na

// Long entries
if bull_long_condition or volatility_long or range_long
    strategy.entry("Long", strategy.long, qty=adaptive_position * 100)
    current_sl := close - ctf_atr * 2.0
    current_tp := close + ctf_atr * 3.0
    strategy.exit("Long Exit", "Long", stop=current_sl, limit=current_tp)

// Short entries  
if bear_short_condition or volatility_short or range_short
    strategy.entry("Short", strategy.short, qty=adaptive_position * 100)
    current_sl := close + ctf_atr * 2.0
    current_tp := close - ctf_atr * 3.0
    strategy.exit("Short Exit", "Short", stop=current_sl, limit=current_tp)

// === VISUALIZATION ===
plot(ctf_ema_fast, color=color.blue, title="EMA Fast")
plot(ctf_ema_slow, color=color.red, title="EMA Slow")

bgcolor(is_bull_trend ? color.new(color.green, 95) : 
        is_bear_trend ? color.new(color.red, 95) : 
        color.new(color.yellow, 95), title="Market Regime")

plotshape(bull_long_condition, style=shape.triangleup, location=location.belowbar, 
          color=color.green, size=size.small, title="Bull Long")
plotshape(bear_short_condition, style=shape.triangledown, location=location.abovebar, 
          color=color.red, size=size.small, title="Bear Short")
plotshape(volatility_long, style=shape.diamond, location=location.belowbar, 
          color=color.orange, size=size.small, title="Vol Long")
plotshape(volatility_short, style=shape.diamond, location=location.abovebar, 
          color=color.purple, size=size.small, title="Vol Short")
'''
}

def create_ml_training_pipeline():
    """Create a complete ML training pipeline"""
    
    def train_crypto_ml_model(symbol: str, timeframe: str = '1h', days: int = 365):
        """Train ML model for a specific crypto symbol"""
        
        print(f"🤖 Starting ML training pipeline for {symbol}")
        
        # Initialize components
        ml_strategy = MLTradingStrategy(config={})
        
        # In a real implementation, you would fetch historical data here
        # For example:
        # exchange = ccxt.binance()
        # ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=days*24)
        # df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # For now, create sample data
        dates = pd.date_range(start='2023-01-01', periods=days*24, freq='H')
        np.random.seed(42)
        
        # Generate realistic crypto price data
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = [50000]  # Starting price
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices[:-1],
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices[:-1]],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices[:-1]],
            'close': prices[1:],
            'volume': np.random.uniform(100, 1000, len(dates))
        })
        
        df.set_index('timestamp', inplace=True)
        
        print("📊 Training ML models...")
        ml_strategy.train_strategy(df)
        
        print("🔍 Generating sample prediction...")
        latest_signal = ml_strategy.generate_signal(df.tail(100), symbol)
        
        if latest_signal:
            print(f"✅ Sample signal generated:")
            print(f"   Action: {latest_signal['action']}")
            print(f"   Confidence: {latest_signal['confidence']:.2%}")
            print(f"   Kelly Fraction: {latest_signal['kelly_fraction']:.3f}")
            print(f"   ML Probability: {latest_signal['signal_components']['ml_probability']:.2%}")
        
        # Save models
        ml_strategy.ml_predictor.save_models(f"models/{symbol.lower()}")
        print(f"💾 Models saved to models/{symbol.lower()}/")
        
        return ml_strategy
    
    return train_crypto_ml_model

# Example usage
if __name__ == "__main__":
    # Create training pipeline
    train_model = create_ml_training_pipeline()
    
    # Train models for major cryptocurrencies
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    for symbol in symbols:
        try:
            trained_strategy = train_model(symbol)
            print(f"✅ {symbol} ML strategy training completed!\n")
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}\n")
    
    print("🎯 ML Integration System Ready!")
    print("\nNext steps:")
    print("1. Integrate with your live trading system")
    print("2. Set up real-time data feeds") 
    print("3. Configure webhook endpoints")
    print("4. Deploy enhanced Pine Script strategies")
    print("5. Start paper trading to validate performance")
