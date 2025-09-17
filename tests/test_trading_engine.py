import copy

import pytest

from config.trading_config import TradingConfig
import engine.trading_engine as trading_engine_module


@pytest.fixture
def trading_engine_setup(monkeypatch):
    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return DummyCursor()

        def commit(self):
            return None

    class DummyCursor:
        def execute(self, *args, **kwargs):
            return None

        def fetchone(self):
            return (0,)

    class DummyDBManager:
        def __init__(self, db_path):
            self.db_path = db_path

        def get_connection(self):
            return DummyConnection()

    class DummyAPI:
        instance = None

        def __init__(self, config):
            self.config = config
            self.valid_symbols = {"BTCUSDT"}
            self.orders = []
            DummyAPI.instance = self

        def get_symbol_info(self, symbol):
            if symbol not in self.valid_symbols:
                raise ValueError(f"Symbol {symbol} not found")
            return {"symbol": symbol}

        def place_order(self, symbol, side, order_type, quantity, price=None, stop_price=None, time_in_force="GTC"):
            order = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "price": price if price is not None else 0,
                "stopPrice": stop_price,
                "timeInForce": time_in_force,
                "orderId": len(self.orders) + 1,
                "status": "FILLED" if order_type == "MARKET" else "NEW"
            }
            self.orders.append(order)
            return order

        def get_open_orders(self, symbol=None):
            return []

    class DummyRiskManager:
        instance = None

        def __init__(self, config, db_manager):
            self.config = config
            self.db_manager = db_manager
            self.last_calculate_args = None
            self.last_risk_checked = None
            DummyRiskManager.instance = self

        def calculate_position_size(self, symbol, entry_price, stop_loss, **kwargs):
            self.last_calculate_args = (symbol, entry_price, stop_loss)
            return 0.5

        def check_risk_limits(self, new_position_risk):
            self.last_risk_checked = new_position_risk
            return True

    class DummyRSIStrategy:
        instance = None

        def __init__(self, config, api, risk_manager):
            self.config = config
            self.api = api
            self.risk_manager = risk_manager
            self.generated_signal = {
                "stop_loss": 9500,
                "take_profit": 10500,
                "confidence": 0.8
            }
            DummyRSIStrategy.instance = self

        def generate_signal(self, symbol):
            return copy.deepcopy(self.generated_signal)

    class DummyBreakoutStrategy:
        def __init__(self, config, api, risk_manager):
            self.config = config

        def generate_signal(self, symbol):
            return None

    class DummyWebhookHandler:
        def __init__(self, config, trading_engine):
            self.config = config
            self.trading_engine = trading_engine

        def run(self, *args, **kwargs):
            return None

    monkeypatch.setattr(trading_engine_module, "DatabaseManager", DummyDBManager)
    monkeypatch.setattr(trading_engine_module, "BinanceAPI", DummyAPI)
    monkeypatch.setattr(trading_engine_module, "RiskManager", DummyRiskManager)
    monkeypatch.setattr(trading_engine_module, "RSIBollingerStrategy", DummyRSIStrategy)
    monkeypatch.setattr(trading_engine_module, "BreakoutSwingStrategy", DummyBreakoutStrategy)
    monkeypatch.setattr(trading_engine_module, "WebhookHandler", DummyWebhookHandler)

    config = TradingConfig(api_key="test_key", api_secret="test_secret")
    engine = trading_engine_module.TradingEngine(config)
    engine._log_trade = lambda *args, **kwargs: None

    return engine, DummyAPI.instance, DummyRiskManager.instance, DummyRSIStrategy.instance


def test_process_signal_invalid_symbol(trading_engine_setup):
    engine, dummy_api, _, _ = trading_engine_setup

    result = engine.process_signal({
        "action": "buy",
        "symbol": "INVALID",
        "price": 10000,
        "strategy": "manual"
    })

    assert result == {"error": "Invalid symbol: INVALID"}


def test_process_signal_places_orders_with_generated_levels(trading_engine_setup):
    engine, dummy_api, dummy_risk, dummy_rsi = trading_engine_setup

    dummy_api.valid_symbols = {"BTCUSDT"}
    dummy_rsi.generated_signal = {
        "stop_loss": 9500,
        "take_profit": 10500,
        "confidence": 0.85
    }

    result = engine.process_signal({
        "action": "buy",
        "symbol": "BTCUSDT",
        "price": 10000,
        "strategy": "rsi_bollinger_scalping"
    })

    assert result["success"] is True
    assert len(dummy_api.orders) == 3

    market_order, stop_order, take_profit_order = dummy_api.orders

    assert market_order["type"] == "MARKET"
    assert stop_order["type"] == "STOP_MARKET"
    assert stop_order["stopPrice"] == 9500
    assert take_profit_order["type"] == "LIMIT"
    assert take_profit_order["price"] == 10500

    assert dummy_risk.last_calculate_args == ("BTCUSDT", 10000, 9500)
    assert dummy_risk.last_risk_checked == pytest.approx(250.0)
