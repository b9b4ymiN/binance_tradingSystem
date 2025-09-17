import hmac
import hashlib

import pytest

from config.trading_config import TradingConfig
from core.binance_api import BinanceAPI


def test_generate_signature_matches_expected():
    config = TradingConfig(api_key="key", api_secret="secret")
    api = BinanceAPI(config)
    payload = "symbol=BTCUSDT&timestamp=123456789"

    expected = hmac.new(
        config.api_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    assert api._generate_signature(payload) == expected


def test_make_request_signed_adds_signature(monkeypatch):
    config = TradingConfig(api_key="key", api_secret="secret")
    api = BinanceAPI(config)

    captured = {}

    class DummyResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok"}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = dict(params)
        response = DummyResponse()
        response.url = url
        return response

    monkeypatch.setattr(api.session, "get", fake_get)
    monkeypatch.setattr("core.binance_api.time.time", lambda: 1_600_000_000.123)

    result = api._make_request(
        method="GET",
        endpoint="/api/v3/openOrders",
        params={"symbol": "BTCUSDT"},
        signed=True
    )

    assert result == {"status": "ok"}

    params = captured["params"]
    assert params["symbol"] == "BTCUSDT"
    assert params["recvWindow"] == 60000
    assert params["timestamp"] == 1_600_000_000_123

    expected_signature = hmac.new(
        config.api_secret.encode("utf-8"),
        "symbol=BTCUSDT&timestamp=1600000000123&recvWindow=60000".encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    assert params["signature"] == expected_signature
