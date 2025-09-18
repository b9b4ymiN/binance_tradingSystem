import psutil
import asyncio
import time
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class AdvancedMonitoring:
    """Real-time monitoring with predictive alerts"""

    def __init__(self, config: Dict):
        self.config = config
        self.alert_thresholds = {
            'performance': {
                'win_rate_critical': 0.45,
                'drawdown_warning': 0.08,
                'drawdown_critical': 0.12,
                'profit_factor_warning': 1.2
            },
            'system': {
                'cpu_warning': 80,
                'memory_warning': 85,
                'disk_warning': 90,
                'api_latency_warning': 2000  # ms
            },
            'trading': {
                'daily_loss_limit': 0.05,
                'position_count_warning': 8,
                'correlation_warning': 0.8
            }
        }

    async def comprehensive_health_check(self) -> Dict:
        """Comprehensive system health monitoring"""

        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'alerts': [],
            'timestamp': datetime.now().isoformat()
        }

        # System health checks
        system_health = await self.check_system_health()
        health_status['checks']['system'] = system_health

        # Trading system health
        trading_health = await self.check_trading_health()
        health_status['checks']['trading'] = trading_health

        # API connectivity health
        api_health = await self.check_api_health()
        health_status['checks']['api'] = api_health

        # Database health
        db_health = await self.check_database_health()
        health_status['checks']['database'] = db_health

        # Determine overall status
        all_checks = [system_health, trading_health, api_health, db_health]
        if any(check['status'] == 'critical' for check in all_checks):
            health_status['overall_status'] = 'critical'
        elif any(check['status'] == 'warning' for check in all_checks):
            health_status['overall_status'] = 'warning'

        return health_status

    async def check_system_health(self) -> Dict:
        """Check system resource health"""

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        status = 'healthy'
        issues = []

        if cpu_percent > self.alert_thresholds['system']['cpu_warning']:
            status = 'warning' if cpu_percent < 95 else 'critical'
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")

        if memory.percent > self.alert_thresholds['system']['memory_warning']:
            status = 'warning' if memory.percent < 95 else 'critical'
            issues.append(f"High memory usage: {memory.percent:.1f}%")

        if disk.percent > self.alert_thresholds['system']['disk_warning']:
            status = 'warning' if disk.percent < 98 else 'critical'
            issues.append(f"High disk usage: {disk.percent:.1f}%")

        return {
            'status': status,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'issues': issues
        }

    async def check_trading_health(self) -> Dict:
        """Check trading system health"""

        # This would check actual trading metrics
        # For now, return mock data

        return {
            'status': 'healthy',
            'active_positions': 3,
            'daily_trades': 8,
            'current_drawdown': 0.03,
            'win_rate_24h': 0.68,
            'issues': []
        }

    async def check_api_health(self) -> Dict:
        """Check API connectivity and latency"""

        api_checks = {}
        overall_status = 'healthy'

        # Test Binance API
        try:
            start_time = time.time()
            # Mock API call - replace with actual call
            await asyncio.sleep(0.1)  # Simulate API call
            latency = (time.time() - start_time) * 1000

            api_checks['binance'] = {
                'status': 'healthy' if latency < 1000 else 'warning',
                'latency_ms': latency,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            api_checks['binance'] = {
                'status': 'critical',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
            overall_status = 'critical'

        return {
            'status': overall_status,
            'apis': api_checks
        }

    async def check_database_health(self) -> Dict:
        """Check database connectivity and performance"""

        try:
            # Mock database check
            db_size = 50  # MB
            connection_count = 5
            query_performance = 45  # ms

            status = 'healthy'
            issues = []

            if db_size > 1000:  # 1GB
                issues.append(f"Large database size: {db_size}MB")
                status = 'warning'

            if query_performance > 100:
                issues.append(f"Slow queries: {query_performance}ms avg")
                status = 'warning'

            return {
                'status': status,
                'size_mb': db_size,
                'connections': connection_count,
                'avg_query_time_ms': query_performance,
                'issues': issues
            }

        except Exception as e:
            return {
                'status': 'critical',
                'error': str(e)
            }

    async def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""

        report = {
            'system_performance': await self.check_system_health(),
            'trading_performance': await self.check_trading_health(),
            'api_performance': await self.check_api_health(),
            'database_performance': await self.check_database_health(),
            'generated_at': datetime.now().isoformat(),
            'recommendations': []
        }

        # Generate recommendations based on performance
        if report['system_performance']['cpu_percent'] > 80:
            report['recommendations'].append('Consider upgrading CPU or optimizing algorithms')

        if report['system_performance']['memory_percent'] > 85:
            report['recommendations'].append('Increase available memory or optimize memory usage')

        if report['trading_performance']['current_drawdown'] > 0.08:
            report['recommendations'].append('Review and adjust risk management parameters')

        return report

    def create_alert(self, alert_type: str, message: str, level: str = 'INFO') -> Dict:
        """Create structured alert"""

        return {
            'type': alert_type,
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        }