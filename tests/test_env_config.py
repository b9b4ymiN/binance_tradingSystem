import os

import pytest

from config.trading_config import TradingConfig


def _load_env_file(path):
    values = {}
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def test_trading_config_uses_env_values(monkeypatch):
    env_values = _load_env_file(".env")

    required_keys = [
        "BINANCE_API_KEY",
        "BINANCE_SECRET_KEY",
        "WEBHOOK_SECRET",
    ]

    missing = [key for key in required_keys if key not in env_values or not env_values[key]]
    if missing:
        pytest.skip(f"Missing required keys in .env: {missing}")

    if env_values["BINANCE_API_KEY"] == "your_api_key_here" or env_values["BINANCE_SECRET_KEY"] == "your_secret_key_here":
        pytest.skip(".env still contains placeholder Binance credentials")

    for key, value in env_values.items():
        monkeypatch.setenv(key, value)

    config = TradingConfig(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_SECRET_KEY"),
        webhook_secret=os.getenv("WEBHOOK_SECRET"),
        use_testnet=os.getenv("USE_TESTNET", "true").lower() == "true",
        max_position_risk=float(os.getenv("MAX_POSITION_RISK", 0.02)),
        kelly_fraction=float(os.getenv("KELLY_FRACTION", 0.25)),
        max_daily_trades=int(os.getenv("MAX_DAILY_TRADES", 10)),
        db_path=os.getenv("DATABASE_PATH", "trading_data.db"),
    )

    assert config.api_key == env_values["BINANCE_API_KEY"]
    assert config.api_secret == env_values["BINANCE_SECRET_KEY"]
    assert config.webhook_secret == env_values["WEBHOOK_SECRET"]
    assert config.use_testnet == (env_values.get("USE_TESTNET", "true").lower() == "true")
    assert config.db_path == env_values.get("DATABASE_PATH", "trading_data.db")
