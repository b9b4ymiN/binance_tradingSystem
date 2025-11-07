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

    # Start weekly parameter optimization scheduler
    try:
        from scheduler.weekly_optimizer import WeeklyOptimizer
        weekly_optimizer = WeeklyOptimizer(
            trading_engine=trading_engine,
            config=config,
            symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            optimization_days=90,
            validation_days=14,
            min_improvement_threshold=0.05  # Require 5% improvement to update
        )
        weekly_optimizer.start()
        logger.info("✅ Weekly parameter optimizer started (runs every Sunday 00:00 UTC)")
    except Exception as e:
        logger.warning(f"Failed to start weekly optimizer: {e}")
        weekly_optimizer = None

    # Initialize notification system
    try:
        from notifications.notification_manager import NotificationManager
        notifier = NotificationManager(
            config,
            telegram_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )
        # Test notification on startup
        notifier.send(
            "System Startup",
            "🚀 Trading system started successfully!\n\nVWAP Mean Reversion strategy is active with automated parameter optimization.",
            priority='low'
        )
        logger.info("✅ Notification system initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize notifications: {e}")
        notifier = None

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

        logger.info("=" * 80)
        logger.info("🚀 TRADING SYSTEM STARTED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("📡 Webhook server: http://0.0.0.0:5000")
        logger.info("📊 Dashboard API: http://0.0.0.0:5001")
        logger.info("🎯 Active strategies:")
        logger.info("   - VWAP Mean Reversion (PRIMARY)")
        logger.info("   - RSI-Bollinger Scalping")
        logger.info("   - Breakout Swing Trading")
        logger.info("📅 Parameter optimization: Every Sunday 00:00 UTC")
        logger.info("⚠️  Mode: TESTNET" if config.use_testnet else "🔴 Mode: LIVE TRADING")
        logger.info("=" * 80)

        # Keep main thread alive
        webhook_thread.join()

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("🛑 SHUTTING DOWN TRADING SYSTEM...")
        logger.info("=" * 80)
        trading_engine.stop_trading()
        if weekly_optimizer:
            weekly_optimizer.stop()
        if notifier:
            notifier.send(
                "System Shutdown",
                "Trading system has been stopped.",
                priority='normal'
            )
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    load_env_file()
    main()