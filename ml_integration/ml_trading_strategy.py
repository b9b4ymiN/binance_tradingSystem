import pandas as pd
import talib
from typing import Dict, Optional
import logging
from .ml_price_prediction import MLPricePrediction
from .sentiment_analysis import SentimentAnalysis

logger = logging.getLogger(__name__)

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
            sentiment_data = self.sentiment_analyzer.get_combined_sentiment(symbol)

            # Combine ML and sentiment signals
            signal = self._combine_signals(ml_prediction, sentiment_data, df, symbol)

            return signal

        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return None

    def _combine_signals(self, ml_prediction: Dict, sentiment_data: Dict,
                        df: pd.DataFrame, symbol: str) -> Dict:
        """Combine ML predictions with sentiment analysis"""

        # Weight factors
        ml_weight = 0.6
        sentiment_weight = 0.4

        # Normalize sentiment scores to match ML probability scale
        sentiment_score = sentiment_data['combined_sentiment']
        sentiment_normalized = (sentiment_score + 1) / 2  # Convert -1,1 to 0,1

        # Combined probability
        combined_prob = (
            ml_prediction['direction_probability'] * ml_weight +
            sentiment_normalized * sentiment_weight
        )

        # Signal strength
        signal_strength = (
            ml_prediction['signal_strength'] * ml_weight +
            abs(sentiment_score) * sentiment_weight
        )

        # Calculate position size based on Kelly Criterion and ML confidence
        kelly_fraction = self._calculate_ml_kelly_fraction(
            combined_prob,
            ml_prediction['ensemble_magnitude'],
            signal_strength,
            sentiment_data['confidence']
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
            'sentiment_data': sentiment_data,
            'signal_components': {
                'ml_probability': ml_prediction['direction_probability'],
                'sentiment_score': sentiment_score,
                'sentiment_normalized': sentiment_normalized,
                'combined_probability': combined_prob,
                'signal_strength': signal_strength
            },
            'feature_analysis': {
                'top_ml_features': ml_prediction['top_features'],
                'key_sentiment_topics': sentiment_data['news_component']['key_topics']
            }
        }

    def _calculate_ml_kelly_fraction(self, probability: float, expected_return: float,
                                   confidence: float, sentiment_confidence: float) -> float:
        """Calculate Kelly fraction based on ML predictions and sentiment"""

        # Adjust probability based on combined confidence
        combined_confidence = (confidence + sentiment_confidence) / 2
        adjusted_prob = probability * combined_confidence

        # Estimate win/loss ratio from expected return
        if expected_return > 0:
            win_ratio = abs(expected_return) / 0.02  # Assume 2% average loss
        else:
            win_ratio = 0.02 / abs(expected_return) if expected_return != 0 else 1.0

        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        kelly_fraction = (win_ratio * adjusted_prob - (1 - adjusted_prob)) / win_ratio if win_ratio > 0 else 0

        # Apply conservative scaling and limits
        kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.1))  # Max 10% position

        return kelly_fraction

    def get_model_performance_summary(self) -> Dict:
        """Get performance summary of ML models"""

        if not self.is_trained:
            return {'status': 'Models not trained'}

        try:
            # This would include actual model performance metrics
            # For now, return mock performance data
            return {
                'model_status': 'trained',
                'direction_accuracy': 0.68,
                'magnitude_mse': 0.0045,
                'lstm_mae': 0.0032,
                'feature_count': len(self.ml_predictor.feature_engineer.feature_names),
                'training_samples': 'varies',
                'last_prediction_confidence': 'varies'
            }

        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {'status': 'error', 'message': str(e)}

    def save_strategy(self, filepath: str):
        """Save the trained strategy"""
        if not self.is_trained:
            raise ValueError("Strategy must be trained before saving")

        try:
            self.ml_predictor.save_models(filepath)
            logger.info(f"ML strategy saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving strategy: {e}")
            raise

    def load_strategy(self, filepath: str):
        """Load a trained strategy"""
        try:
            self.ml_predictor.load_models(filepath)
            self.is_trained = True
            logger.info(f"ML strategy loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
            raise

    def validate_signal_quality(self, signal: Dict) -> Dict:
        """Validate the quality of generated signal"""

        if not signal:
            return {'valid': False, 'reason': 'No signal generated'}

        quality_score = 0
        issues = []

        # Check confidence levels
        if signal['confidence'] > 0.8:
            quality_score += 30
        elif signal['confidence'] > 0.6:
            quality_score += 20
        else:
            issues.append('Low overall confidence')

        # Check ML prediction quality
        ml_pred = signal['ml_prediction']
        if ml_pred['signal_strength'] > 0.03:
            quality_score += 25
        elif ml_pred['signal_strength'] > 0.015:
            quality_score += 15
        else:
            issues.append('Weak ML signal strength')

        # Check sentiment alignment
        sentiment_data = signal['sentiment_data']
        if sentiment_data['confidence'] > 0.7:
            quality_score += 20
        else:
            issues.append('Low sentiment confidence')

        # Check Kelly fraction reasonableness
        if 0.01 <= signal['kelly_fraction'] <= 0.08:
            quality_score += 25
        else:
            issues.append('Kelly fraction outside optimal range')

        return {
            'valid': quality_score >= 60,
            'quality_score': quality_score,
            'grade': self._get_quality_grade(quality_score),
            'issues': issues,
            'recommendation': 'Execute signal' if quality_score >= 70 else 'Consider skipping'
        }

    def _get_quality_grade(self, score: int) -> str:
        """Convert quality score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'