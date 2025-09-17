# Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB, recommended 8GB
- **Storage**: 2GB free space for logs and database
- **Internet**: Stable connection for API calls

### Account Requirements
- **Binance Account**: Regular or Pro account
- **API Access**: API key with trading permissions
- **TradingView Account**: Pro plan for webhook alerts (optional)
- **VPS/Server**: For 24/7 operation (recommended)

## Installation Steps

### 1. Environment Setup

#### Clone Repository
```bash
git clone https://github.com/your-repo/trading_system.git
cd trading_system
```

#### Create Virtual Environment
```bash
# Windows
python -m venv trading_env
trading_env\Scripts\activate

# macOS/Linux
python3 -m venv trading_env
source trading_env/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Binance API Configuration

#### Create API Key
1. Log into Binance account
2. Go to Account → API Management
3. Create new API key with following permissions:
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ❌ Enable Futures (not required)
   - ❌ Enable Withdrawals (security risk)

#### API Key Security
- **Restrict IP Access**: Add your server IP to whitelist
- **Use Testnet First**: Test with paper trading
- **Store Securely**: Never commit keys to version control

### 3. Environment Variables

#### Create .env File
```bash
# Create .env file in project root
touch .env
```

#### Add Configuration
```env
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here

# Webhook Security
WEBHOOK_SECRET=your_secure_webhook_secret

# Trading Configuration
USE_TESTNET=true
MAX_POSITION_RISK=0.02
KELLY_FRACTION=0.25
MAX_DAILY_TRADES=10

# Database Configuration
DATABASE_PATH=trading_data.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=trading_bot.log
```

#### Load Environment Variables
```python
# Add to main.py
from dotenv import load_dotenv
load_dotenv()

config = TradingConfig(
    api_key=os.getenv('BINANCE_API_KEY'),
    api_secret=os.getenv('BINANCE_SECRET_KEY'),
    webhook_secret=os.getenv('WEBHOOK_SECRET'),
    use_testnet=os.getenv('USE_TESTNET', 'true').lower() == 'true'
)
```

### 4. Database Setup

#### Initialize Database
```bash
python -c "
from core.database import DatabaseManager
db = DatabaseManager('trading_data.db')
print('Database initialized successfully')
"
```

#### Verify Tables
```sql
-- Check created tables
.tables
-- Should show: trades, portfolio, performance, risk_metrics
```

### 5. Configuration Validation

#### Test Binance Connection
```python
# test_connection.py
from config.trading_config import TradingConfig
from core.binance_api import BinanceAPI

config = TradingConfig(
    api_key=os.getenv('BINANCE_API_KEY'),
    api_secret=os.getenv('BINANCE_SECRET_KEY'),
    use_testnet=True
)

api = BinanceAPI(config)
try:
    account_info = api.get_account_info()
    print("✅ Binance connection successful")
    print(f"Account type: {account_info.get('accountType')}")
except Exception as e:
    print(f"❌ Binance connection failed: {e}")
```

#### Test Strategy Calculation
```python
# test_strategy.py
from analysis.technical_analysis import TechnicalAnalysis

# Test RSI calculation
prices = [100, 102, 101, 105, 103, 108, 107, 110, 108, 112]
rsi = TechnicalAnalysis.calculate_rsi(prices)
print(f"RSI: {rsi:.2f}")

# Test Bollinger Bands
upper, middle, lower = TechnicalAnalysis.calculate_bollinger_bands(prices)
print(f"BB: Upper={upper:.2f}, Middle={middle:.2f}, Lower={lower:.2f}")
```

## TradingView Integration

### 1. Pine Script Setup

#### Install Strategy
1. Open TradingView chart
2. Go to Pine Editor
3. Copy Pine Script from `utils/pine_scripts.py`
4. Replace `YOUR_WEBHOOK_URL` with your server URL
5. Save and add to chart

#### Configure Alerts
```javascript
// Alert message format
{
  "action": "{{strategy.order.action}}",
  "symbol": "{{ticker}}",
  "price": {{close}},
  "strategy": "rsi_bollinger_scalping",
  "stop_loss": {{strategy.order.action == "buy" ? close * 0.98 : close * 1.02}},
  "take_profit": {{strategy.order.action == "buy" ? close * 1.02 : close * 0.98}}
}
```

### 2. Webhook Configuration

#### Server Setup
```bash
# For production deployment
sudo ufw allow 5000
sudo systemctl enable trading_bot
```

#### Webhook URL Format
```
http://your-server-ip:5000/webhook
```

#### Security Headers
```http
Content-Type: application/json
X-Webhook-Signature: hmac_sha256_signature
```

### 3. Testing Webhook

#### Manual Test
```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "action": "buy",
    "symbol": "BTCUSDT",
    "price": 45000,
    "strategy": "rsi_bollinger_scalping"
  }'
```

#### Expected Response
```json
{
  "status": "success",
  "message": "Signal processed",
  "result": {
    "success": true,
    "order": {
      "orderId": "12345",
      "symbol": "BTCUSDT",
      "status": "FILLED"
    }
  }
}
```

## Production Deployment

### 1. VPS/Server Setup

#### Recommended Specifications
- **CPU**: 2 cores minimum
- **RAM**: 4GB minimum
- **Storage**: 50GB SSD
- **Bandwidth**: Unlimited
- **Uptime**: 99.9% SLA

#### Popular VPS Providers
- DigitalOcean (recommended for Asia)
- AWS EC2 (global presence)
- Vultr (good Thailand connectivity)
- Linode (reliable performance)

### 2. System Configuration

#### Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv nginx supervisor

# CentOS/RHEL
sudo yum install python3 python3-pip nginx supervisor
```

#### Configure Firewall
```bash
# Allow SSH and HTTP/HTTPS
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 5000
sudo ufw enable
```

### 3. Process Management

#### Supervisor Configuration
```ini
# /etc/supervisor/conf.d/trading_bot.conf
[program:trading_bot]
command=/home/trader/trading_system/trading_env/bin/python /home/trader/trading_system/main.py
directory=/home/trader/trading_system
user=trader
autostart=true
autorestart=true
stdout_logfile=/var/log/trading_bot.log
stderr_logfile=/var/log/trading_bot_error.log
environment=PYTHONPATH="/home/trader/trading_system"
```

#### Start Services
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start trading_bot
```

### 4. Monitoring Setup

#### Log Monitoring
```bash
# Real-time log monitoring
tail -f /var/log/trading_bot.log

# Error monitoring
tail -f /var/log/trading_bot_error.log
```

#### Health Check Script
```python
# health_check.py
import requests
import smtplib
from datetime import datetime

def check_health():
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print(f"✅ System healthy at {datetime.now()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    check_health()
```

#### Cron Job for Monitoring
```bash
# Add to crontab (run every 5 minutes)
*/5 * * * * /home/trader/trading_system/trading_env/bin/python /home/trader/trading_system/health_check.py
```

## Security Best Practices

### 1. API Key Security
- **Environment Variables**: Never hardcode keys
- **IP Restrictions**: Whitelist server IPs only
- **Permission Limits**: Only enable required permissions
- **Regular Rotation**: Change keys monthly

### 2. Server Security
- **SSH Key Authentication**: Disable password login
- **Regular Updates**: Keep system packages updated
- **Fail2Ban**: Install intrusion prevention
- **SSL/TLS**: Use HTTPS for webhook endpoints

### 3. Code Security
- **Input Validation**: Validate all webhook data
- **Error Handling**: Don't expose internal errors
- **Logging**: Log security events
- **Backup**: Regular database backups

## Troubleshooting

### Common Issues

#### 1. API Connection Errors
```
Error: 401 Unauthorized
Solution: Check API key and secret
```

#### 2. Database Lock Errors
```
Error: database is locked
Solution: Close all database connections properly
```

#### 3. Webhook Not Receiving Signals
```
Check: Firewall rules, TradingView alert settings, webhook URL
```

#### 4. Strategy Not Executing
```
Check: Risk limits, account balance, market hours
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
from core.binance_api import BinanceAPI
api = BinanceAPI(config)
print(api.get_account_info())
```

### Support Resources
- **Documentation**: Check `/docs/` directory
- **Logs**: Review `trading_bot.log` for errors
- **Community**: Join trading system forums
- **Issues**: Report bugs on GitHub

## Maintenance

### Daily Tasks
- Monitor log files for errors
- Check system resources (CPU, memory)
- Verify webhook connectivity
- Review trading performance

### Weekly Tasks
- Update system packages
- Backup database
- Review strategy performance
- Check for new software updates

### Monthly Tasks
- Rotate API keys
- Performance optimization
- Strategy parameter review
- Security audit

---

**⚠️ Important**: Always test thoroughly with testnet before live trading. Start with small position sizes and gradually increase as you gain confidence in the system.