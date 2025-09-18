import yaml
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProductionDeploymentManager:
    """Manage production deployment and scaling"""

    def __init__(self, config: Dict):
        self.config = config
        self.deployment_config = self.load_deployment_config()

    def load_deployment_config(self) -> Dict:
        """Load production deployment configuration"""

        return {
            'scaling': {
                'min_instances': 1,
                'max_instances': 5,
                'cpu_threshold': 70,
                'memory_threshold': 80
            },
            'resource_limits': {
                'cpu_cores': 2,
                'memory_gb': 4,
                'disk_gb': 50
            },
            'monitoring': {
                'metrics_retention_days': 90,
                'log_retention_days': 30,
                'alert_channels': ['email', 'slack', 'sms']
            },
            'backup': {
                'frequency': 'daily',
                'retention_days': 30,
                'storage_type': 's3'
            }
        }

    async def auto_scaling_check(self, current_metrics: Dict) -> Dict:
        """Check if auto-scaling is needed"""

        scaling_decision = {
            'action': 'none',
            'reason': '',
            'target_instances': 1
        }

        current_cpu = current_metrics.get('cpu_percent', 0)
        current_memory = current_metrics.get('memory_percent', 0)
        current_load = current_metrics.get('trading_load', 0)

        # Scale up conditions
        if (current_cpu > self.deployment_config['scaling']['cpu_threshold'] or
            current_memory > self.deployment_config['scaling']['memory_threshold'] or
            current_load > 0.8):

            scaling_decision['action'] = 'scale_up'
            scaling_decision['reason'] = 'High resource usage detected'
            scaling_decision['target_instances'] = min(
                self.deployment_config['scaling']['max_instances'],
                current_metrics.get('current_instances', 1) + 1
            )

        # Scale down conditions
        elif (current_cpu < 30 and
              current_memory < 40 and
              current_load < 0.3 and
              current_metrics.get('current_instances', 1) > 1):

            scaling_decision['action'] = 'scale_down'
            scaling_decision['reason'] = 'Low resource usage detected'
            scaling_decision['target_instances'] = max(
                self.deployment_config['scaling']['min_instances'],
                current_metrics.get('current_instances', 1) - 1
            )

        return scaling_decision

    def generate_production_config(self) -> str:
        """Generate optimized production configuration"""

        production_config = {
            'version': '3.8',
            'services': {
                'trading-bot': {
                    'image': 'crypto-trading-bot:production',
                    'deploy': {
                        'replicas': 2,
                        'resources': {
                            'limits': {
                                'cpus': '2.0',
                                'memory': '4G'
                            },
                            'reservations': {
                                'cpus': '1.0',
                                'memory': '2G'
                            }
                        },
                        'restart_policy': {
                            'condition': 'on-failure',
                            'delay': '5s',
                            'max_attempts': 3
                        },
                        'update_config': {
                            'parallelism': 1,
                            'delay': '10s',
                            'failure_action': 'rollback'
                        }
                    },
                    'environment': [
                        'USE_TESTNET=false',
                        'LOG_LEVEL=INFO',
                        'ENABLE_METRICS=true'
                    ],
                    'volumes': [
                        './data:/app/data',
                        './logs:/app/logs'
                    ],
                    'ports': ['5000:5000'],
                    'networks': ['trading-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:5000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'deploy': {
                        'resources': {
                            'limits': {'memory': '512M'},
                            'reservations': {'memory': '256M'}
                        }
                    },
                    'volumes': ['redis_data:/data'],
                    'networks': ['trading-network'],
                    'command': 'redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru'
                },
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml',
                        'prometheus_data:/prometheus'
                    ],
                    'networks': ['trading-network']
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'ports': ['3000:3000'],
                    'environment': [
                        'GF_SECURITY_ADMIN_PASSWORD=secure_password_here'
                    ],
                    'volumes': ['grafana_data:/var/lib/grafana'],
                    'networks': ['trading-network']
                },
                'nginx': {
                    'image': 'nginx:alpine',
                    'ports': ['80:80', '443:443'],
                    'volumes': [
                        './nginx/nginx.conf:/etc/nginx/nginx.conf',
                        './nginx/ssl:/etc/nginx/ssl'
                    ],
                    'networks': ['trading-network'],
                    'depends_on': ['trading-bot']
                }
            },
            'volumes': {
                'redis_data': {},
                'prometheus_data': {},
                'grafana_data': {}
            },
            'networks': {
                'trading-network': {
                    'driver': 'overlay',
                    'attachable': True
                }
            }
        }

        return yaml.dump(production_config, default_flow_style=False)

    def create_monitoring_script(self) -> str:
        """Create comprehensive monitoring script"""

        monitoring_script = '''#!/bin/bash
# Production Monitoring Script for Crypto Trading Bot

LOG_FILE="/opt/trading-bot/logs/monitor.log"
ALERT_EMAIL="admin@yourtrading.com"

echo "$(date): Starting monitoring check" >> $LOG_FILE

# Check if trading bot is running
if ! docker ps | grep -q "crypto-trading-bot"; then
    echo "CRITICAL: Trading bot container not running" >> $LOG_FILE
    echo "Trading bot is down" | mail -s "CRITICAL ALERT" $ALERT_EMAIL
    exit 1
fi

# Check CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "WARNING: High CPU usage: $CPU_USAGE%" >> $LOG_FILE
    echo "High CPU usage detected: $CPU_USAGE%" | mail -s "HIGH CPU ALERT" $ALERT_EMAIL
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
if (( $(echo "$MEM_USAGE > 85" | bc -l) )); then
    echo "WARNING: High memory usage: $MEM_USAGE%" >> $LOG_FILE
    echo "High memory usage detected: $MEM_USAGE%" | mail -s "HIGH MEMORY ALERT" $ALERT_EMAIL
fi

# Check disk space
DISK_USAGE=$(df /opt/trading-bot | awk 'NR==2 {print substr($5, 1, length($5)-1)}')
if [ $DISK_USAGE -gt 90 ]; then
    echo "WARNING: High disk usage: $DISK_USAGE%" >> $LOG_FILE
    echo "High disk usage detected: $DISK_USAGE%" | mail -s "HIGH DISK ALERT" $ALERT_EMAIL
fi

# Check trading bot health endpoint
if ! curl -f -s http://localhost:5000/health > /dev/null; then
    echo "CRITICAL: Trading bot health check failed" >> $LOG_FILE
    echo "Trading bot health check failed" | mail -s "HEALTH CHECK FAILED" $ALERT_EMAIL
fi

echo "$(date): Monitoring check completed" >> $LOG_FILE
'''

        return monitoring_script

    def generate_dockerfile(self) -> str:
        """Generate optimized Dockerfile for production"""

        dockerfile_content = '''
# Production Dockerfile for Crypto Trading Bot
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["python", "main.py"]
'''

        return dockerfile_content.strip()

    def get_deployment_status(self) -> Dict:
        """Get current deployment status"""

        return {
            'status': 'active',
            'instances': 2,
            'version': '1.0.0',
            'last_deployed': datetime.now().isoformat(),
            'health_checks': {
                'passing': 2,
                'failing': 0
            },
            'resource_usage': {
                'cpu': '45%',
                'memory': '60%',
                'disk': '25%'
            }
        }