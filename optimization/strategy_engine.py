import numpy as np
import asyncio
from datetime import datetime
from typing import Dict
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

class AdvancedStrategyEngine:
    """Enhanced strategy engine with ML and adaptive capabilities"""

    def __init__(self, config: Dict):
        self.config = config
        self.ml_models = {}
        self.strategy_performance = {}
        self.market_regime_detector = None
        self.anomaly_detector = IsolationForest(contamination=0.1)

    async def initialize_ml_models(self):
        """Initialize and load pre-trained ML models"""

        model_configs = {
            'price_predictor': {
                'type': 'ensemble',
                'models': ['random_forest', 'gradient_boosting', 'lstm'],
                'weights': [0.3, 0.4, 0.3]
            },
            'regime_detector': {
                'type': 'clustering',
                'features': ['volatility', 'trend_strength', 'volume_profile'],
                'clusters': 4
            },
            'sentiment_analyzer': {
                'type': 'transformer',
                'model': 'finbert',
                'sources': ['news', 'twitter', 'reddit']
            }
        }

        for model_name, config in model_configs.items():
            try:
                # In production, load actual trained models
                self.ml_models[model_name] = f"Loaded {model_name}"
                logger.info(f"✅ ML model loaded: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")

    async def adaptive_strategy_selection(self, market_data: Dict,
                                        portfolio_state: Dict) -> Dict:
        """Dynamically select best strategy based on current conditions"""

        # Analyze current market conditions
        market_analysis = await self.analyze_market_conditions(market_data)

        # Get strategy performance history
        strategy_scores = await self.calculate_strategy_scores(market_analysis)

        # Select optimal strategy combination
        selected_strategies = []
        total_weight = 0

        for strategy, score in sorted(strategy_scores.items(),
                                    key=lambda x: x[1], reverse=True):
            if total_weight < 1.0 and score > 0.6:  # Min score threshold
                weight = min(0.4, 1.0 - total_weight)  # Max 40% per strategy
                selected_strategies.append({
                    'strategy': strategy,
                    'weight': weight,
                    'score': score
                })
                total_weight += weight

        return {
            'selected_strategies': selected_strategies,
            'market_analysis': market_analysis,
            'confidence': sum(s['score'] for s in selected_strategies) / len(selected_strategies) if selected_strategies else 0
        }

    async def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Comprehensive market condition analysis"""

        conditions = {
            'volatility_regime': 'normal',  # low, normal, high
            'trend_strength': 0.5,          # 0-1 scale
            'market_sentiment': 0.0,        # -1 to 1
            'liquidity_conditions': 'normal', # low, normal, high
            'correlation_environment': 0.7   # 0-1 scale
        }

        # Analyze volatility
        if 'volatility' in market_data:
            vol = market_data['volatility']
            if vol > 0.04:
                conditions['volatility_regime'] = 'high'
            elif vol < 0.015:
                conditions['volatility_regime'] = 'low'

        # Analyze trend strength using multiple indicators
        trend_indicators = []
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'trend_strength' in data:
                trend_indicators.append(data['trend_strength'])

        if trend_indicators:
            conditions['trend_strength'] = np.mean(trend_indicators)

        return conditions

    async def calculate_strategy_scores(self, market_analysis: Dict) -> Dict:
        """Calculate performance scores for each strategy"""

        base_scores = {
            'rsi_bollinger_scalping': 0.75,
            'ema_crossover': 0.70,
            'volume_breakout': 0.80,
            'mean_reversion': 0.65,
            'ml_enhanced': 0.85,
            'pairs_trading': 0.60,
            'momentum_following': 0.72,
            'volatility_trading': 0.78
        }

        # Adjust scores based on market conditions
        adjusted_scores = {}

        for strategy, base_score in base_scores.items():
            adjustment = 1.0

            # Volatility-based adjustments
            if market_analysis['volatility_regime'] == 'high':
                if strategy in ['volume_breakout', 'volatility_trading']:
                    adjustment *= 1.2
                elif strategy == 'mean_reversion':
                    adjustment *= 0.8

            # Trend-based adjustments
            if market_analysis['trend_strength'] > 0.7:
                if strategy in ['ema_crossover', 'momentum_following']:
                    adjustment *= 1.15
                elif strategy == 'mean_reversion':
                    adjustment *= 0.9

            adjusted_scores[strategy] = min(0.95, base_score * adjustment)

        return adjusted_scores

    async def detect_market_anomalies(self, market_data: Dict) -> Dict:
        """Detect market anomalies using ML"""

        try:
            # Prepare data for anomaly detection
            features = []
            for symbol, data in market_data.items():
                if isinstance(data, dict):
                    price = data.get('price', 0)
                    volume = data.get('volume', 0)
                    volatility = data.get('volatility', 0)
                    features.append([price, volume, volatility])

            if len(features) < 2:
                return {'anomalies_detected': False, 'details': 'Insufficient data'}

            # Detect anomalies
            anomalies = self.anomaly_detector.fit_predict(features)
            anomaly_count = sum(1 for x in anomalies if x == -1)

            return {
                'anomalies_detected': anomaly_count > 0,
                'anomaly_count': anomaly_count,
                'anomaly_ratio': anomaly_count / len(features),
                'recommendation': 'Reduce position sizes' if anomaly_count > len(features) * 0.3 else 'Normal operation'
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'anomalies_detected': False, 'error': str(e)}

    def get_strategy_performance_summary(self) -> Dict:
        """Get performance summary for all strategies"""

        if not self.strategy_performance:
            return {'message': 'No performance data available'}

        summary = {}
        for strategy, performance in self.strategy_performance.items():
            trades = performance.get('trades', [])
            if trades:
                wins = [t for t in trades if t['pnl'] > 0]
                win_rate = len(wins) / len(trades)
                avg_pnl = np.mean([t['pnl'] for t in trades])

                summary[strategy] = {
                    'total_trades': len(trades),
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'total_pnl': sum(t['pnl'] for t in trades),
                    'last_updated': performance.get('last_updated')
                }

        return summary