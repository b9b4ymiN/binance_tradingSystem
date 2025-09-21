import os
import logging
from config.trading_config import TradingConfig
from engine.trading_engine import TradingEngine
from pathlib import Path

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_env_file(path=".env"):
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

def main():
    """Main application entry point"""
    
    # Load configuration from environment variables
    config = TradingConfig(
        api_key=os.getenv('BINANCE_API_KEY', 'your_api_key_here'),
        api_secret=os.getenv('BINANCE_SECRET_KEY', 'your_secret_key_here'),
        webhook_secret=os.getenv('WEBHOOK_SECRET', 'your_webhook_secret'),
        use_testnet=True  # Set to False for live trading
    )
    
    # Initialize trading engine
    trading_engine = TradingEngine(config)
    
    # Start automated trading
    trading_engine.start_auto_trading()
    
    # Start webhook server and dashboard API in separate threads
    from threading import Thread

    def start_webhook_server():
        trading_engine.webhook_handler.run(host='0.0.0.0', port=5000)

    def start_dashboard_api():
        trading_engine.dashboard_api.run(host='0.0.0.0', port=5001)

    try:
        # Start servers in separate threads
        webhook_thread = Thread(target=start_webhook_server, daemon=True)
        dashboard_thread = Thread(target=start_dashboard_api, daemon=True)

        webhook_thread.start()
        dashboard_thread.start()

        logger.info("Trading system started successfully")
        logger.info("Webhook server running on port 5000")
        logger.info("Dashboard API running on port 5001")

        # Keep main thread alive
        webhook_thread.join()

    except KeyboardInterrupt:
        logger.info("Shutting down trading system...")
        trading_engine.stop_trading()


if __name__ == "__main__":
    load_env_file()
    main()