import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error
from typing import Dict, Tuple
import joblib
import logging
from .feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

class MLPricePrediction:
    """Machine Learning models for price prediction"""

    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineering()
        self.is_trained = False

    def prepare_data(self, df: pd.DataFrame, target_periods: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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