import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class MetricsCollector:
    """Comprehensive metrics collection for trading system"""

    def __init__(self):
        # Define Prometheus metrics
        self.trade_counter = Counter('trades_total', 'Total number of trades', ['strategy', 'side', 'status'])
        self.trade_duration = Histogram('trade_duration_seconds', 'Trade execution duration')
        self.pnl_gauge = Gauge('current_pnl', 'Current profit/loss')
        self.position_count = Gauge('open_positions', 'Number of open positions')
        self.api_latency = Histogram('api_request_duration_seconds', 'API request latency', ['endpoint'])
        self.system_cpu = Gauge('system_cpu_percent', 'System CPU usage')
        self.system_memory = Gauge('system_memory_percent', 'System memory usage')
        self.system_disk = Gauge('system_disk_percent', 'System disk usage')
        self.webhook_requests = Counter('webhook_requests_total', 'Webhook requests', ['status'])
        self.error_counter = Counter('errors_total', 'Total errors', ['type'])

        # Risk metrics
        self.portfolio_risk = Gauge('portfolio_risk_percent', 'Current portfolio risk')
        self.drawdown = Gauge('max_drawdown_percent', 'Maximum drawdown')
        self.win_rate = Gauge('win_rate_percent', 'Current win rate')

        # Performance tracking
        self.daily_pnl = Gauge('daily_pnl', 'Daily PnL')
        self.trade_volume = Gauge('daily_volume', 'Daily trading volume')

    def record_trade(self, strategy: str, side: str, status: str, duration: float):
        """Record trade metrics"""
        self.trade_counter.labels(strategy=strategy, side=side, status=status).inc()
        self.trade_duration.observe(duration)

    def update_system_metrics(self):
        """Update system performance metrics"""
        self.system_cpu.set(psutil.cpu_percent())
        self.system_memory.set(psutil.virtual_memory().percent)
        self.system_disk.set(psutil.disk_usage('/').percent)

    def record_api_latency(self, endpoint: str, duration: float):
        """Record API request latency"""
        self.api_latency.labels(endpoint=endpoint).observe(duration)

    def record_webhook(self, status: str):
        """Record webhook request"""
        self.webhook_requests.labels(status=status).inc()

    def record_error(self, error_type: str):
        """Record error occurrence"""
        self.error_counter.labels(type=error_type).inc()