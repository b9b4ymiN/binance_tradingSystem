import time
import threading
from datetime import datetime
from prometheus_client import start_http_server
from .metrics_collector import MetricsCollector

class PerformanceMonitor:
    """Advanced performance monitoring with alerts"""

    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self.metrics = MetricsCollector()
        self.alert_thresholds = {
            'max_drawdown': 0.10,  # 10% max drawdown
            'win_rate_min': 0.45,   # Minimum 45% win rate
            'daily_loss_limit': 0.05,  # 5% daily loss limit
            'api_latency_max': 5.0,    # 5 second max API latency
            'error_rate_max': 0.05     # 5% max error rate
        }
        self.monitoring_thread = None
        self.is_monitoring = False

    def start_monitoring(self):
        """Start the monitoring system"""
        self.is_monitoring = True

        # Start Prometheus metrics server
        start_http_server(9999)

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        print("Performance monitoring started on port 9999")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update system metrics
                self.metrics.update_system_metrics()

                # Update trading metrics
                self._update_trading_metrics()

                # Check alert conditions
                self._check_alerts()

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                print(f"Monitoring error: {e}")
                self.metrics.record_error("monitoring")
                time.sleep(60)

    def _update_trading_metrics(self):
        """Update trading-specific metrics"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Calculate daily PnL
                today = datetime.now().date()
                cursor.execute('''
                    SELECT COALESCE(SUM(realized_pnl), 0) FROM trades
                    WHERE DATE(timestamp) = ? AND status = 'FILLED'
                ''', (today,))
                daily_pnl = cursor.fetchone()[0]
                self.metrics.daily_pnl.set(daily_pnl)

                # Calculate win rate (last 100 trades)
                cursor.execute('''
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins
                    FROM (
                        SELECT realized_pnl FROM trades
                        WHERE status = 'FILLED' AND realized_pnl != 0
                        ORDER BY timestamp DESC LIMIT 100
                    )
                ''')
                result = cursor.fetchone()
                if result[0] > 0:
                    win_rate = result[1] / result[0]
                    self.metrics.win_rate.set(win_rate * 100)

                # Calculate current drawdown
                cursor.execute('''
                    SELECT realized_pnl FROM trades
                    WHERE status = 'FILLED'
                    ORDER BY timestamp DESC LIMIT 50
                ''')
                recent_pnls = [row[0] for row in cursor.fetchall()]
                if recent_pnls:
                    cumulative_pnl = []
                    running_total = 0
                    for pnl in reversed(recent_pnls):
                        running_total += pnl
                        cumulative_pnl.append(running_total)

                    peak = cumulative_pnl[0]
                    max_drawdown = 0
                    for value in cumulative_pnl:
                        if value > peak:
                            peak = value
                        drawdown = (peak - value) / peak if peak > 0 else 0
                        max_drawdown = max(max_drawdown, drawdown)

                    self.metrics.drawdown.set(max_drawdown * 100)

                # Count open positions
                cursor.execute('''
                    SELECT COUNT(DISTINCT symbol) FROM trades
                    WHERE status = 'FILLED'
                    GROUP BY symbol
                    HAVING SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) != 0
                ''')
                open_positions = len(cursor.fetchall())
                self.metrics.position_count.set(open_positions)

        except Exception as e:
            print(f"Error updating trading metrics: {e}")
            self.metrics.record_error("metrics_update")

    def _check_alerts(self):
        """Check alert conditions and send notifications"""
        try:
            # Get current metric values
            current_drawdown = self.metrics.drawdown._value._value / 100
            current_win_rate = self.metrics.win_rate._value._value / 100
            daily_pnl = self.metrics.daily_pnl._value._value

            # Check drawdown alert
            if current_drawdown > self.alert_thresholds['max_drawdown']:
                self._send_alert(
                    "HIGH_DRAWDOWN",
                    f"Maximum drawdown exceeded: {current_drawdown:.2%} > {self.alert_thresholds['max_drawdown']:.2%}"
                )

            # Check win rate alert
            if current_win_rate < self.alert_thresholds['win_rate_min']:
                self._send_alert(
                    "LOW_WIN_RATE",
                    f"Win rate below threshold: {current_win_rate:.2%} < {self.alert_thresholds['win_rate_min']:.2%}"
                )

            # Check daily loss limit
            if daily_pnl < -self.alert_thresholds['daily_loss_limit']:
                self._send_alert(
                    "DAILY_LOSS_LIMIT",
                    f"Daily loss limit exceeded: {daily_pnl:.2%}"
                )

        except Exception as e:
            print(f"Error checking alerts: {e}")

    def _send_alert(self, alert_type: str, message: str):
        """Send alert notification"""
        timestamp = datetime.now().isoformat()
        alert_data = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp,
            'severity': 'HIGH'
        }

        # Log alert to database
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        type TEXT,
                        message TEXT,
                        severity TEXT
                    )
                ''')
                cursor.execute('''
                    INSERT INTO alerts (type, message, severity)
                    VALUES (?, ?, ?)
                ''', (alert_type, message, alert_data['severity']))
                conn.commit()
        except Exception as e:
            print(f"Error logging alert: {e}")

        # Print alert (in production, send to Slack/Discord/Email)
        print(f"🚨 ALERT [{alert_type}]: {message}")

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Performance monitoring stopped")
