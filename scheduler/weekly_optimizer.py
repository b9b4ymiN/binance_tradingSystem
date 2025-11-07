"""
Weekly Parameter Optimization Scheduler

Automatically optimizes strategy parameters every Sunday at 00:00 UTC
Features:
- Scheduled parameter optimization
- Walk-forward validation
- Automatic parameter updates
- Rollback on poor performance
- Notification system integration
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional

from config.vwap_parameters import VWAPParameters
from optimization.parameter_optimizer import ParameterOptimizer
from core.parameter_manager import ParameterManager

logger = logging.getLogger(__name__)


class WeeklyOptimizer:
    """
    Schedules and manages weekly parameter optimization

    Runs every Sunday at 00:00 UTC to find optimal parameters
    based on recent market data
    """

    def __init__(
        self,
        trading_engine,
        config,
        symbols: List[str] = None,
        optimization_days: int = 90,
        validation_days: int = 14,
        min_improvement_threshold: float = 0.05,  # 5% improvement required
    ):
        self.trading_engine = trading_engine
        self.config = config
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.optimization_days = optimization_days
        self.validation_days = validation_days
        self.min_improvement_threshold = min_improvement_threshold

        self.parameter_manager = ParameterManager(config.db_path)
        self.scheduler = BackgroundScheduler()

        self.is_running = False
        self.last_optimization_time = None
        self.last_optimization_status = None

    def start(self):
        """Start the weekly optimization scheduler"""

        # Schedule for every Sunday at 00:00 UTC
        self.scheduler.add_job(
            func=self.run_weekly_optimization,
            trigger=CronTrigger(day_of_week='sun', hour=0, minute=0),
            id='weekly_parameter_optimization',
            name='Weekly Parameter Optimization',
            replace_existing=True
        )

        # Also add a manual trigger job that can be called anytime
        self.scheduler.add_job(
            func=self.run_weekly_optimization,
            id='manual_optimization',
            name='Manual Parameter Optimization',
            replace_existing=True
        )

        self.scheduler.start()
        self.is_running = True

        logger.info("✅ Weekly parameter optimizer started")
        logger.info(f"📅 Scheduled: Every Sunday at 00:00 UTC")
        logger.info(f"📊 Optimization period: {self.optimization_days} days")
        logger.info(f"✔️  Validation period: {self.validation_days} days")
        logger.info(f"🎯 Symbols: {', '.join(self.symbols)}")

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Weekly optimizer stopped")

    def run_weekly_optimization(self):
        """
        Main optimization routine

        Steps:
        1. Download recent historical data
        2. Run parameter optimization
        3. Validate with walk-forward analysis
        4. Compare with current parameters
        5. Update if better
        6. Send notification
        """

        logger.info("=" * 80)
        logger.info("🚀 STARTING WEEKLY PARAMETER OPTIMIZATION")
        logger.info("=" * 80)

        start_time = datetime.now()
        self.last_optimization_time = start_time

        try:
            # Step 1: Get current active parameters
            current_params = self.parameter_manager.get_active_parameters('vwap_mean_reversion')
            current_version = None

            if current_params:
                logger.info(f"Current parameters version: {current_params['version']}")
                logger.info(f"Current score: {current_params.get('backtest_score', 'N/A')}")
                current_version = current_params['version']
            else:
                logger.info("No current parameters found, using defaults")

            # Step 2: Run optimization
            logger.info(f"\n📊 Running parameter optimization...")
            logger.info(f"Optimization period: {self.optimization_days} days")
            logger.info(f"Symbols: {', '.join(self.symbols)}")

            optimizer = ParameterOptimizer(
                config=self.config,
                symbols=self.symbols,
                optimization_days=self.optimization_days
            )

            # Run grid search (with limited combinations for reasonable time)
            best_params = optimizer.optimize_grid_search(
                max_combinations=100,  # Limit to 100 combinations
                n_workers=4
            )

            optimization_score = optimizer.best_score

            logger.info(f"\n✨ Optimization complete!")
            logger.info(f"Best score: {optimization_score:.4f}")
            logger.info(f"Best parameters: {json.dumps(best_params.to_dict(), indent=2)}")

            # Step 3: Walk-forward validation
            logger.info(f"\n✔️  Running walk-forward validation...")
            validation_score = optimizer.validate_walk_forward(
                best_params,
                validation_days=self.validation_days
            )

            logger.info(f"Validation score: {validation_score:.4f}")

            # Check if validation score is reasonable (not too different from optimization)
            score_difference = abs(validation_score - optimization_score) / optimization_score
            if score_difference > 0.5:  # More than 50% difference
                logger.warning(
                    f"⚠️  Large difference between optimization and validation scores "
                    f"({score_difference*100:.1f}%). Possible overfitting!"
                )

            # Step 4: Compare with current parameters
            should_update = False
            reason = ""

            if current_params is None:
                # No current parameters, use new ones
                should_update = True
                reason = "First parameter set"

            else:
                current_score = current_params.get('backtest_score', 0)

                # Calculate improvement
                improvement = (validation_score - current_score) / current_score if current_score > 0 else 0

                logger.info(f"\n📈 Performance comparison:")
                logger.info(f"Current score: {current_score:.4f}")
                logger.info(f"New score: {validation_score:.4f}")
                logger.info(f"Improvement: {improvement*100:.2f}%")

                if improvement >= self.min_improvement_threshold:
                    should_update = True
                    reason = f"Performance improvement: {improvement*100:.1f}%"
                else:
                    should_update = False
                    reason = f"Insufficient improvement: {improvement*100:.1f}% < {self.min_improvement_threshold*100:.1f}%"

            # Step 5: Update parameters if better
            if should_update:
                logger.info(f"\n✅ UPDATING PARAMETERS")
                logger.info(f"Reason: {reason}")

                # Get top results for reference
                top_params = optimizer.get_top_n_parameters(5)

                # Save new parameters
                new_version = self.parameter_manager.save_parameters(
                    strategy_name='vwap_mean_reversion',
                    parameters=best_params.to_dict(),
                    backtest_score=validation_score,
                    backtest_metrics=optimizer.optimization_results[0]['metrics'] if optimizer.optimization_results else {},
                    optimization_period_start=optimizer.start_date.date(),
                    optimization_period_end=optimizer.end_date.date(),
                    notes=f"Automated weekly optimization. {reason}",
                    top_alternatives=top_params
                )

                # Update strategy in trading engine
                if hasattr(self.trading_engine, 'vwap_strategy'):
                    self.trading_engine.vwap_strategy.update_parameters(best_params)
                    logger.info("Trading engine strategy parameters updated")

                # Save optimization results
                optimizer.save_results(f'optimization_results_{new_version}.json')

                self.last_optimization_status = 'success_updated'

                # Step 6: Send success notification
                self._send_notification(
                    title="✅ Parameter Update Success",
                    message=f"New parameters activated\nVersion: {new_version}\nScore: {validation_score:.4f}\nImprovement: {reason}",
                    priority='normal'
                )

            else:
                logger.info(f"\n⏭️  KEEPING CURRENT PARAMETERS")
                logger.info(f"Reason: {reason}")

                # Still save the results for reference
                self.parameter_manager.save_optimization_run(
                    strategy_name='vwap_mean_reversion',
                    best_params=best_params.to_dict(),
                    best_score=validation_score,
                    total_combinations=len(optimizer.optimization_results),
                    status='completed_no_update',
                    notes=reason
                )

                self.last_optimization_status = 'success_no_update'

                # Send info notification
                self._send_notification(
                    title="ℹ️ Parameters Not Updated",
                    message=f"Current parameters retained\n{reason}\nNew score: {validation_score:.4f}",
                    priority='low'
                )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"\n⏱️  Optimization completed in {duration:.1f} seconds")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Weekly optimization failed: {e}", exc_info=True)
            self.last_optimization_status = 'failed'

            # Save failed run
            try:
                self.parameter_manager.save_optimization_run(
                    strategy_name='vwap_mean_reversion',
                    best_params={},
                    best_score=0,
                    total_combinations=0,
                    status='failed',
                    error_message=str(e),
                    notes='Optimization failed with error'
                )
            except Exception as save_error:
                logger.error(f"Failed to save error log: {save_error}")

            # Send error notification
            self._send_notification(
                title="❌ Optimization Failed",
                message=f"Weekly parameter optimization encountered an error:\n{str(e)}",
                priority='high'
            )

    def run_manual_optimization(self):
        """Trigger optimization manually (useful for testing)"""
        logger.info("📢 Manual optimization triggered")
        self.run_weekly_optimization()

    def get_next_optimization_time(self) -> Optional[datetime]:
        """Get the next scheduled optimization time"""
        job = self.scheduler.get_job('weekly_parameter_optimization')
        if job and job.next_run_time:
            return job.next_run_time
        return None

    def get_last_optimization_status(self) -> Dict:
        """Get status of last optimization run"""
        return {
            'last_run_time': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'status': self.last_optimization_status,
            'next_run_time': self.get_next_optimization_time().isoformat() if self.get_next_optimization_time() else None,
            'is_running': self.is_running
        }

    def _send_notification(self, title: str, message: str, priority: str = 'normal'):
        """
        Send notification about optimization status

        Priority levels: 'low', 'normal', 'high'
        """
        try:
            # Try to import notification system if available
            try:
                from notifications.notification_manager import NotificationManager
                notifier = NotificationManager(self.config)
                notifier.send(title, message, priority)
            except ImportError:
                # Fallback to logging if notification system not available
                logger.info(f"[NOTIFICATION] {title}: {message}")

        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")


def create_weekly_optimizer(trading_engine, config, **kwargs) -> WeeklyOptimizer:
    """
    Factory function to create and configure weekly optimizer

    Usage:
        optimizer = create_weekly_optimizer(
            trading_engine=engine,
            config=config,
            symbols=['BTCUSDT', 'ETHUSDT'],
            optimization_days=90
        )
        optimizer.start()
    """
    return WeeklyOptimizer(trading_engine, config, **kwargs)
