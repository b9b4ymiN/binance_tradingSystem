#!/usr/bin/env python3
"""
Production Setup and Configuration Checker
Ensures the trading system is ready for live trading
"""
import os
import logging
from pathlib import Path
from config.trading_config import TradingConfig

logger = logging.getLogger(__name__)

class ProductionChecker:
    """Check if system is ready for production/live trading"""

    def __init__(self):
        self.issues = []
        self.warnings = []

    def check_environment_variables(self) -> bool:
        """Check critical environment variables"""
        required_vars = {
            'BINANCE_API_KEY': 'Binance API Key',
            'BINANCE_SECRET_KEY': 'Binance Secret Key',
            'WEBHOOK_SECRET': 'Webhook Secret Key'
        }

        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value or value in ['your_api_key_here', 'your_secret_key_here', 'your_webhook_secret']:
                missing_vars.append(f"{var} ({description})")

        if missing_vars:
            self.issues.extend([f"Missing or invalid: {var}" for var in missing_vars])
            return False
        return True

    def check_api_configuration(self) -> bool:
        """Check API configuration"""
        try:
            from core.binance_api import BinanceAPI

            config = TradingConfig(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_SECRET_KEY'),
                use_testnet=True  # Always check with testnet first
            )

            api = BinanceAPI(config)

            # Test API connection
            try:
                account_info = api.get_account_info()
                if account_info and 'balances' in account_info:
                    print("✅ API connection successful")
                    print(f"   Account type: {account_info.get('accountType', 'Unknown')}")
                    print(f"   Can trade: {account_info.get('canTrade', False)}")
                    return True
                else:
                    self.issues.append("API connection failed - invalid response")
                    return False
            except Exception as e:
                self.issues.append(f"API connection failed: {str(e)}")
                return False

        except Exception as e:
            self.issues.append(f"API setup error: {str(e)}")
            return False

    def check_risk_settings(self) -> bool:
        """Check risk management settings"""
        config = TradingConfig(
            api_key="dummy",
            api_secret="dummy"
        )

        # Check if risk settings are conservative enough for live trading
        if config.max_position_risk > 0.03:  # More than 3%
            self.warnings.append(f"Position risk is high: {config.max_position_risk*100:.1f}% (recommend ≤ 2%)")

        if config.kelly_fraction > 0.3:  # More than 30%
            self.warnings.append(f"Kelly fraction is high: {config.kelly_fraction*100:.0f}% (recommend ≤ 25%)")

        if config.max_concurrent_positions > 5:
            self.warnings.append(f"Too many concurrent positions: {config.max_concurrent_positions} (recommend ≤ 3)")

        if config.max_daily_trades > 20:
            self.warnings.append(f"Daily trade limit is high: {config.max_daily_trades} (recommend ≤ 10)")

        return len(self.warnings) == 0

    def check_database_integrity(self) -> bool:
        """Check database setup and integrity"""
        try:
            from core.database import DatabaseManager

            db_path = "trading_data.db"
            if not Path(db_path).exists():
                self.issues.append(f"Database not found: {db_path}")
                return False

            db_manager = DatabaseManager(db_path)

            # Test database connection
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                required_tables = ['trades', 'portfolio', 'performance']
                missing_tables = [table for table in required_tables if table not in tables]

                if missing_tables:
                    self.issues.extend([f"Missing table: {table}" for table in missing_tables])
                    return False

            print("✅ Database integrity check passed")
            return True

        except Exception as e:
            self.issues.append(f"Database error: {str(e)}")
            return False

    def check_security_settings(self) -> bool:
        """Check security configurations"""
        # Check if using testnet (should be False for production)
        use_testnet = os.getenv('USE_TESTNET', 'true').lower()
        if use_testnet == 'true':
            self.warnings.append("Still using testnet (set USE_TESTNET=false for live trading)")

        # Check webhook secret strength
        webhook_secret = os.getenv('WEBHOOK_SECRET', '')
        if len(webhook_secret) < 32:
            self.issues.append("Webhook secret is too short (minimum 32 characters)")

        return True

    def generate_production_config(self) -> str:
        """Generate production-ready configuration"""
        config_template = """# Production Trading Configuration
# CRITICAL: Review all settings before live trading

# Binance API (LIVE TRADING)
BINANCE_API_KEY=your_live_api_key_here
BINANCE_SECRET_KEY=your_live_secret_key_here

# Security
WEBHOOK_SECRET=your_strong_webhook_secret_here_min_32_chars

# Trading Configuration (CONSERVATIVE SETTINGS)
USE_TESTNET=false
MAX_POSITION_RISK=0.02    # 2% max risk per trade
KELLY_FRACTION=0.25       # 25% Kelly scaling
MAX_DAILY_TRADES=10       # Maximum 10 trades per day
MAX_CONCURRENT_POSITIONS=3 # Maximum 3 positions

# Database
DATABASE_PATH=trading_data.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=trading_bot.log

# WARNING: Double check all settings before going live!
# 1. Set USE_TESTNET=false only when ready
# 2. Use small position sizes initially
# 3. Monitor all trades closely
# 4. Have stop-loss strategy
"""
        return config_template

    def run_full_check(self) -> bool:
        """Run complete production readiness check"""
        print("=== PRODUCTION READINESS CHECK ===\n")

        checks = [
            ("Environment Variables", self.check_environment_variables),
            ("API Configuration", self.check_api_configuration),
            ("Risk Settings", self.check_risk_settings),
            ("Database Integrity", self.check_database_integrity),
            ("Security Settings", self.check_security_settings)
        ]

        all_passed = True
        for name, check_func in checks:
            print(f"Checking {name}...")
            try:
                result = check_func()
                if result:
                    print(f"✅ {name}: PASSED")
                else:
                    print(f"❌ {name}: FAILED")
                    all_passed = False
            except Exception as e:
                print(f"❌ {name}: ERROR - {e}")
                self.issues.append(f"{name}: {str(e)}")
                all_passed = False
            print()

        # Print summary
        print("=== SUMMARY ===")
        if self.issues:
            print("🚨 CRITICAL ISSUES (must fix before live trading):")
            for issue in self.issues:
                print(f"   - {issue}")
            print()

        if self.warnings:
            print("⚠️  WARNINGS (recommend addressing):")
            for warning in self.warnings:
                print(f"   - {warning}")
            print()

        if all_passed and not self.issues:
            print("✅ System is ready for live trading!")
            print("\n🔴 FINAL CHECKLIST BEFORE GOING LIVE:")
            print("   1. Set USE_TESTNET=false in .env")
            print("   2. Start with small position sizes")
            print("   3. Monitor first few trades manually")
            print("   4. Have emergency stop procedures ready")
            print("   5. Keep sufficient USDT balance")
        else:
            print("❌ System is NOT ready for live trading")
            print("   Fix all critical issues before proceeding")

        return all_passed and not self.issues

def main():
    """Main function to run production check"""
    checker = ProductionChecker()
    is_ready = checker.run_full_check()

    # Generate production config template
    print("\n=== PRODUCTION CONFIG TEMPLATE ===")
    print("Save this as '.env.production':")
    print(checker.generate_production_config())

    return is_ready

if __name__ == '__main__':
    main()