"""
Parameter Management System

Manages strategy parameters with version control, history tracking,
and rollback capabilities.

Features:
- Parameter versioning
- History tracking
- Active parameter management
- Rollback support
- Performance tracking per version
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ParameterManager:
    """
    Manages strategy parameters with version control and history
    """

    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self._initialize_tables()

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_tables(self):
        """Create parameter management tables if they don't exist"""

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Parameter history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameter_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    backtest_score REAL,
                    backtest_metrics TEXT,
                    optimization_period_start DATE,
                    optimization_period_end DATE,
                    activated_at DATETIME,
                    deactivated_at DATETIME,
                    is_active INTEGER DEFAULT 0,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Optimization runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy_name TEXT NOT NULL,
                    best_parameters TEXT,
                    best_score REAL,
                    total_combinations INTEGER,
                    duration_seconds REAL,
                    status TEXT,
                    error_message TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Parameter performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameter_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(version, date)
                )
            ''')

            conn.commit()
            logger.debug("Parameter management tables initialized")

    def save_parameters(
        self,
        strategy_name: str,
        parameters: Dict,
        backtest_score: float,
        backtest_metrics: Dict,
        optimization_period_start: date,
        optimization_period_end: date,
        notes: str = "",
        top_alternatives: List[Dict] = None
    ) -> str:
        """
        Save new parameter version and activate it

        Args:
            strategy_name: Name of the strategy
            parameters: Parameter dictionary
            backtest_score: Backtest performance score
            backtest_metrics: Detailed backtest metrics
            optimization_period_start: Start date of optimization period
            optimization_period_end: End date of optimization period
            notes: Optional notes
            top_alternatives: Top alternative parameter sets

        Returns:
            Version string of saved parameters
        """

        # Generate version string
        version = self._generate_version(strategy_name)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Deactivate current active parameters
            cursor.execute('''
                UPDATE parameter_history
                SET is_active = 0, deactivated_at = ?
                WHERE strategy_name = ? AND is_active = 1
            ''', (datetime.now().isoformat(), strategy_name))

            # Add notes about alternatives
            if top_alternatives:
                notes += f"\n\nTop {len(top_alternatives)} alternatives saved in optimization run."

            # Insert new parameters
            cursor.execute('''
                INSERT INTO parameter_history (
                    version, strategy_name, parameters, backtest_score,
                    backtest_metrics, optimization_period_start, optimization_period_end,
                    activated_at, is_active, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            ''', (
                version,
                strategy_name,
                json.dumps(parameters),
                backtest_score,
                json.dumps(backtest_metrics),
                optimization_period_start.isoformat(),
                optimization_period_end.isoformat(),
                datetime.now().isoformat(),
                notes
            ))

            conn.commit()

        logger.info(f"✅ Parameters saved: {strategy_name} v{version}")
        return version

    def get_active_parameters(self, strategy_name: str) -> Optional[Dict]:
        """Get currently active parameters for a strategy"""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM parameter_history
                WHERE strategy_name = ? AND is_active = 1
                ORDER BY activated_at DESC
                LIMIT 1
            ''', (strategy_name,))

            row = cursor.fetchone()

            if row:
                return {
                    'version': row['version'],
                    'strategy_name': row['strategy_name'],
                    'parameters': json.loads(row['parameters']),
                    'backtest_score': row['backtest_score'],
                    'backtest_metrics': json.loads(row['backtest_metrics']) if row['backtest_metrics'] else {},
                    'activated_at': row['activated_at'],
                    'notes': row['notes']
                }

            return None

    def get_parameter_history(
        self,
        strategy_name: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get parameter history for a strategy"""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM parameter_history
                WHERE strategy_name = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (strategy_name, limit))

            rows = cursor.fetchall()

            history = []
            for row in rows:
                history.append({
                    'version': row['version'],
                    'strategy_name': row['strategy_name'],
                    'parameters': json.loads(row['parameters']),
                    'backtest_score': row['backtest_score'],
                    'is_active': bool(row['is_active']),
                    'activated_at': row['activated_at'],
                    'deactivated_at': row['deactivated_at'],
                    'created_at': row['created_at'],
                    'notes': row['notes']
                })

            return history

    def rollback_to_version(self, strategy_name: str, version: str) -> bool:
        """
        Rollback to a previous parameter version

        Args:
            strategy_name: Strategy name
            version: Version to rollback to

        Returns:
            True if successful, False otherwise
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if version exists
            cursor.execute('''
                SELECT * FROM parameter_history
                WHERE strategy_name = ? AND version = ?
            ''', (strategy_name, version))

            target_version = cursor.fetchone()

            if not target_version:
                logger.error(f"Version {version} not found for {strategy_name}")
                return False

            # Deactivate current
            cursor.execute('''
                UPDATE parameter_history
                SET is_active = 0, deactivated_at = ?
                WHERE strategy_name = ? AND is_active = 1
            ''', (datetime.now().isoformat(), strategy_name))

            # Activate target version
            cursor.execute('''
                UPDATE parameter_history
                SET is_active = 1, activated_at = ?
                WHERE strategy_name = ? AND version = ?
            ''', (datetime.now().isoformat(), strategy_name, version))

            conn.commit()

        logger.info(f"✅ Rolled back {strategy_name} to version {version}")
        return True

    def save_optimization_run(
        self,
        strategy_name: str,
        best_params: Dict,
        best_score: float,
        total_combinations: int,
        duration_seconds: float = None,
        status: str = 'completed',
        error_message: str = None,
        notes: str = ""
    ):
        """Save optimization run information"""

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO optimization_runs (
                    strategy_name, best_parameters, best_score,
                    total_combinations, duration_seconds, status,
                    error_message, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_name,
                json.dumps(best_params),
                best_score,
                total_combinations,
                duration_seconds,
                status,
                error_message,
                notes
            ))

            conn.commit()

        logger.debug(f"Optimization run saved: {strategy_name} - {status}")

    def get_optimization_history(
        self,
        strategy_name: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get optimization run history"""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM optimization_runs
                WHERE strategy_name = ?
                ORDER BY run_date DESC
                LIMIT ?
            ''', (strategy_name, limit))

            rows = cursor.fetchall()

            history = []
            for row in rows:
                history.append({
                    'run_date': row['run_date'],
                    'strategy_name': row['strategy_name'],
                    'best_score': row['best_score'],
                    'total_combinations': row['total_combinations'],
                    'duration_seconds': row['duration_seconds'],
                    'status': row['status'],
                    'error_message': row['error_message'],
                    'notes': row['notes']
                })

            return history

    def track_parameter_performance(
        self,
        version: str,
        strategy_name: str,
        date: date,
        metrics: Dict
    ):
        """Track daily performance for a parameter version"""

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO parameter_performance (
                    version, strategy_name, date, total_trades, winning_trades,
                    total_pnl, win_rate, profit_factor, max_drawdown, sharpe_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version,
                strategy_name,
                date.isoformat(),
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('total_pnl', 0),
                metrics.get('win_rate', 0),
                metrics.get('profit_factor', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('sharpe_ratio', 0)
            ))

            conn.commit()

        logger.debug(f"Performance tracked for {version} on {date}")

    def get_parameter_performance_summary(
        self,
        version: str,
        days: int = 30
    ) -> Optional[Dict]:
        """Get performance summary for a parameter version"""

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    COUNT(*) as trading_days,
                    SUM(total_trades) as total_trades,
                    SUM(winning_trades) as winning_trades,
                    SUM(total_pnl) as total_pnl,
                    AVG(win_rate) as avg_win_rate,
                    AVG(profit_factor) as avg_profit_factor,
                    MAX(max_drawdown) as max_drawdown,
                    AVG(sharpe_ratio) as avg_sharpe_ratio
                FROM parameter_performance
                WHERE version = ?
                AND date >= date('now', '-' || ? || ' days')
            ''', (version, days))

            row = cursor.fetchone()

            if row and row['total_trades']:
                return {
                    'version': version,
                    'trading_days': row['trading_days'],
                    'total_trades': row['total_trades'],
                    'winning_trades': row['winning_trades'],
                    'total_pnl': row['total_pnl'],
                    'avg_win_rate': row['avg_win_rate'],
                    'avg_profit_factor': row['avg_profit_factor'],
                    'max_drawdown': row['max_drawdown'],
                    'avg_sharpe_ratio': row['avg_sharpe_ratio']
                }

            return None

    def _generate_version(self, strategy_name: str) -> str:
        """Generate version string for parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{strategy_name}_{timestamp}"

    def cleanup_old_records(self, days_to_keep: int = 180):
        """Clean up old parameter history (keep last N days)"""

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Keep active parameters and recent ones
            cursor.execute('''
                DELETE FROM parameter_history
                WHERE is_active = 0
                AND created_at < datetime('now', '-' || ? || ' days')
            ''', (days_to_keep,))

            cursor.execute('''
                DELETE FROM optimization_runs
                WHERE created_at < datetime('now', '-' || ? || ' days')
            ''', (days_to_keep,))

            cursor.execute('''
                DELETE FROM parameter_performance
                WHERE date < date('now', '-' || ? || ' days')
            ''', (days_to_keep,))

            deleted = cursor.rowcount
            conn.commit()

        logger.info(f"Cleaned up {deleted} old parameter records")
