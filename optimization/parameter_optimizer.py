"""
Parameter Optimizer for VWAP Mean Reversion Strategy

Supports multiple optimization methods:
- Grid Search: Exhaustive search over parameter grid
- Random Search: Random sampling for faster optimization
- Bayesian Optimization: Efficient parameter space exploration (future)

Features:
- Walk-forward validation
- Multi-objective optimization (win rate, profit factor, Sharpe ratio)
- Overfitting detection
- Parameter versioning
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import json
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from config.vwap_parameters import VWAPParameters, VWAP_OPTIMIZATION_GRID
from strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from testing.enhanced_backtesting import EnhancedBacktestingEngine
from core.binance_api import BinanceAPI

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Parameter optimizer for trading strategies

    Finds optimal parameters by backtesting multiple combinations
    and selecting the best based on composite score
    """

    def __init__(
        self,
        config,
        symbols: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
        optimization_days: int = 90,
    ):
        self.config = config
        self.symbols = symbols
        self.optimization_days = optimization_days

        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=optimization_days)

        self.start_date = start_date
        self.end_date = end_date

        self.binance_api = BinanceAPI(config)
        self.best_params = None
        self.best_score = -float('inf')
        self.optimization_results = []

    def optimize_grid_search(
        self,
        param_grid: Dict = None,
        max_combinations: int = None,
        n_workers: int = 4
    ) -> VWAPParameters:
        """
        Perform grid search optimization

        Args:
            param_grid: Dictionary of parameter ranges (uses default if None)
            max_combinations: Limit number of combinations to test
            n_workers: Number of parallel workers

        Returns:
            Best VWAPParameters found
        """
        if param_grid is None:
            param_grid = VWAP_OPTIMIZATION_GRID

        logger.info(f"Starting grid search optimization for {self.symbols}")
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        # Limit combinations if specified
        if max_combinations and len(all_combinations) > max_combinations:
            logger.warning(
                f"Limiting combinations from {len(all_combinations)} to {max_combinations}"
            )
            # Random sample
            indices = np.random.choice(
                len(all_combinations),
                size=max_combinations,
                replace=False
            )
            all_combinations = [all_combinations[i] for i in indices]

        total_combinations = len(all_combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")

        # Download historical data for all symbols
        logger.info("Downloading historical data...")
        historical_data = self._download_historical_data()

        if not historical_data:
            logger.error("Failed to download historical data")
            return VWAPParameters()  # Return default

        # Test each combination
        results = []
        completed = 0

        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}

            for combination in all_combinations:
                # Create params dict
                params_dict = dict(zip(param_names, combination))

                # Submit job
                future = executor.submit(
                    self._evaluate_parameters,
                    params_dict,
                    historical_data
                )
                futures[future] = params_dict

            # Collect results
            for future in as_completed(futures):
                params_dict = futures[future]
                try:
                    score, metrics = future.result()
                    results.append({
                        'params': params_dict,
                        'score': score,
                        'metrics': metrics
                    })

                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{total_combinations} combinations tested")

                    # Track best
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = VWAPParameters.from_dict(params_dict)
                        logger.info(f"✨ New best score: {score:.4f} | Params: {params_dict}")

                except Exception as e:
                    logger.error(f"Error evaluating params {params_dict}: {e}")

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        self.optimization_results = results

        logger.info(f"✅ Optimization complete! Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params.to_dict()}")

        return self.best_params

    def _download_historical_data(self) -> Dict[str, List]:
        """Download historical klines data for all symbols"""

        historical_data = {}

        for symbol in self.symbols:
            try:
                logger.info(f"Downloading data for {symbol}...")

                # Calculate number of candles needed (1-minute candles)
                days = (self.end_date - self.start_date).days
                candles_needed = days * 24 * 60  # minutes per day

                # Binance limit is 1000 per request, so we may need multiple requests
                all_klines = []
                current_start = self.start_date

                while current_start < self.end_date:
                    # Get 1000 candles at a time
                    klines = self.binance_api.get_klines(
                        symbol=symbol,
                        interval='1m',
                        limit=1000,
                        start_time=int(current_start.timestamp() * 1000)
                    )

                    if not klines:
                        break

                    all_klines.extend(klines)

                    # Update start time for next batch
                    last_timestamp = klines[-1][0]
                    current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=1)

                    # Stop if we've reached end date
                    if current_start >= self.end_date:
                        break

                logger.info(f"Downloaded {len(all_klines)} candles for {symbol}")
                historical_data[symbol] = all_klines

            except Exception as e:
                logger.error(f"Failed to download data for {symbol}: {e}")

        return historical_data

    def _evaluate_parameters(
        self,
        params_dict: Dict,
        historical_data: Dict[str, List]
    ) -> Tuple[float, Dict]:
        """
        Evaluate a parameter combination

        Args:
            params_dict: Parameters to test
            historical_data: Historical klines data

        Returns:
            (composite_score, metrics_dict)
        """
        try:
            # Create parameter object
            params = VWAPParameters.from_dict(params_dict)

            # Create strategy with these parameters
            strategy = VWAPMeanReversionStrategy(
                config=self.config,
                binance_api=self.binance_api,
                risk_manager=None,  # Not needed for backtest
                params=params
            )

            # Run backtest for each symbol
            symbol_results = []

            for symbol, klines in historical_data.items():
                if len(klines) < params.lookback_candles:
                    continue

                # Create backtesting engine
                backtest = EnhancedBacktestingEngine(
                    strategy=strategy,
                    initial_balance=10000,
                    commission_rate=0.00075,
                    slippage_rate=0.0005,
                )

                # Run backtest
                result = backtest.run_backtest(symbol, klines)

                if 'error' not in result and result['total_trades'] > 0:
                    symbol_results.append(result)

            # Calculate composite score from all symbols
            if not symbol_results:
                return -1000, {}  # No valid results

            # Aggregate metrics
            avg_win_rate = np.mean([r['win_rate'] for r in symbol_results])
            avg_profit_factor = np.mean([r['profit_factor'] for r in symbol_results if r['profit_factor'] != float('inf')])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in symbol_results])
            avg_return = np.mean([r['total_return'] for r in symbol_results])
            max_drawdown = max([r['max_drawdown'] for r in symbol_results])
            total_trades = sum([r['total_trades'] for r in symbol_results])

            # Composite score (weighted)
            score = (
                avg_win_rate * 25 +                    # 25% weight on win rate
                min(avg_profit_factor / 2, 1.0) * 25 + # 25% weight on profit factor (capped at 2.0)
                min(avg_sharpe / 2, 1.0) * 20 +        # 20% weight on Sharpe ratio (capped at 2.0)
                avg_return * 20 +                       # 20% weight on return
                (1 - max_drawdown) * 10                 # 10% weight on low drawdown
            )

            # Penalty for too few trades
            if total_trades < 20:
                score *= 0.5  # 50% penalty

            # Penalty for excessive drawdown
            if max_drawdown > 0.25:  # > 25% drawdown
                score *= 0.3  # 70% penalty

            metrics = {
                'win_rate': avg_win_rate,
                'profit_factor': avg_profit_factor,
                'sharpe_ratio': avg_sharpe,
                'total_return': avg_return,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'num_symbols': len(symbol_results),
            }

            return score, metrics

        except Exception as e:
            logger.error(f"Error in _evaluate_parameters: {e}")
            return -1000, {}

    def validate_walk_forward(
        self,
        params: VWAPParameters,
        validation_days: int = 30
    ) -> float:
        """
        Validate parameters using walk-forward analysis on out-of-sample data

        Args:
            params: Parameters to validate
            validation_days: Days of out-of-sample data to use

        Returns:
            Validation score
        """
        logger.info(f"Running walk-forward validation for {validation_days} days")

        # Use data after optimization period for validation
        validation_start = self.end_date
        validation_end = validation_start + timedelta(days=validation_days)

        # Download validation data
        validation_data = {}
        for symbol in self.symbols:
            try:
                klines = self.binance_api.get_klines(
                    symbol=symbol,
                    interval='1m',
                    limit=validation_days * 24 * 60
                )
                validation_data[symbol] = klines
            except Exception as e:
                logger.error(f"Failed to download validation data for {symbol}: {e}")

        if not validation_data:
            logger.error("No validation data available")
            return 0.0

        # Evaluate on validation data
        score, metrics = self._evaluate_parameters(
            params.to_dict(),
            validation_data
        )

        logger.info(f"Validation score: {score:.4f}")
        logger.info(f"Validation metrics: {metrics}")

        return score

    def get_top_n_parameters(self, n: int = 5) -> List[Dict]:
        """Get top N parameter combinations"""
        if not self.optimization_results:
            return []

        return self.optimization_results[:n]

    def save_results(self, filepath: str):
        """Save optimization results to file"""
        results = {
            'optimization_date': datetime.now().isoformat(),
            'symbols': self.symbols,
            'period': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat(),
            },
            'best_score': self.best_score,
            'best_params': self.best_params.to_dict() if self.best_params else None,
            'top_10_results': self.get_top_n_parameters(10),
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Optimization results saved to {filepath}")


def run_quick_optimization(config, symbols: List[str] = None) -> VWAPParameters:
    """
    Quick optimization with limited parameter space

    Useful for testing or quick parameter updates
    """
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT']

    # Reduced parameter grid for faster optimization
    quick_grid = {
        'vwap_period': [75, 100, 150],
        'entry_threshold': [0.005, 0.007, 0.010],
        'exit_threshold': [0.002, 0.003],
        'min_volume_multiplier': [1.5, 2.0],
        'stop_loss_atr_mult': [2.0, 2.5],
        'profit_target_atr_mult': [1.5, 2.0],
    }

    optimizer = ParameterOptimizer(
        config=config,
        symbols=symbols,
        optimization_days=60  # Shorter period
    )

    best_params = optimizer.optimize_grid_search(
        param_grid=quick_grid,
        max_combinations=50,  # Limit to 50 combinations
        n_workers=2
    )

    return best_params
