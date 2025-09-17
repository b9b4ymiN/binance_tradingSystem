# Webhook Integration Reference

## Environment Prerequisites
- Set `WEBHOOK_SECRET` in `.env` so the webhook handler can validate inbound requests (example: `WEBHOOK_SECRET=qs7fPvmZiQN7VjVnH-ZMyGQk8qz6o_l8vYQbaJ3Xb4`).
- When `python main.py` runs, the server uses that secret for HMAC validation (`webhook/handler.py`).
- Make sure TradingView IPs are present in `config/trading_config.py:29-36` or extend the list to include your alert forwarder.

## Generating the Signature
Sign the exact raw JSON payload with HMAC SHA256 using the shared secret:

```python
import hmac
import hashlib
import json

webhook_secret = "YOUR_WEBHOOK_SECRET"
payload = {
    "action": "buy",
    "symbol": "BTCUSDT",
    "price": 65000,
    "strategy": "rsi_bollinger_scalping",
    "stop_loss": 63700,
    "take_profit": 66800
}

raw_body = json.dumps(payload, separators=(",", ":"))
signature = hmac.new(webhook_secret.encode(), raw_body.encode(), hashlib.sha256).hexdigest()
print(signature)
```

Send the signature in the `X-Webhook-Signature` header.

## Payload Examples and Test Commands

### 1. RSI-Bollinger Buy
```json
{
  "action": "buy",
  "symbol": "BTCUSDT",
  "price": 65000,
  "strategy": "rsi_bollinger_scalping",
  "stop_loss": 63700,
  "take_profit": 66800,
  "confidence": 0.82,
  "notes": "TradingView alert: RSI oversold"
}
```

Example cURL command:
```bash
SIGNATURE=$(python - <<'PY'
import hmac, hashlib, json
secret = "YOUR_WEBHOOK_SECRET"
payload = {
    "action": "buy",
    "symbol": "BTCUSDT",
    "price": 65000,
    "strategy": "rsi_bollinger_scalping",
    "stop_loss": 63700,
    "take_profit": 66800,
    "confidence": 0.82,
    "notes": "TradingView alert: RSI oversold"
}
raw = json.dumps(payload, separators=(",", ":"))
print(hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest())
PY
)

curl -X POST http://127.0.0.1:5000/webhook \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Signature: $SIGNATURE" \
  -d '{"action":"buy","symbol":"BTCUSDT","price":65000,"strategy":"rsi_bollinger_scalping","stop_loss":63700,"take_profit":66800,"confidence":0.82,"notes":"TradingView alert: RSI oversold"}'
```

### 2. RSI-Bollinger Sell
```json
{
  "action": "sell",
  "symbol": "BTCUSDT",
  "price": 65500,
  "strategy": "rsi_bollinger_scalping",
  "stop_loss": 66800,
  "take_profit": 64000,
  "confidence": 0.78,
  "notes": "RSI overbought signal"
}
```

### 3. Breakout Swing Buy
```json
{
  "action": "buy",
  "symbol": "ETHUSDT",
  "price": 3200,
  "strategy": "breakout_swing",
  "stop_loss": 3136,
  "take_profit": 3680,
  "confidence": 0.65,
  "timeframe": "4h"
}
```

### 4. Breakout Swing Sell
```json
{
  "action": "sell",
  "symbol": "ETHUSDT",
  "price": 3150,
  "strategy": "breakout_swing",
  "stop_loss": 3280,
  "take_profit": 2890,
  "confidence": 0.60,
  "timeframe": "4h"
}
```

### 5. Manual Order with Custom TP/SL
```json
{
  "action": "buy",
  "symbol": "ADAUSDT",
  "price": 0.45,
  "strategy": "manual",
  "stop_loss": 0.43,
  "take_profit": 0.49,
  "notes": "Manual override"
}
```

## TradingView Alert Template
Paste the JSON below into the alert message body. If you cannot add custom headers directly from TradingView, use an intermediate webhook service to attach `X-Webhook-Signature`.
```
{
  "action": "buy",
  "symbol": "{{ticker}}",
  "price": {{close}},
  "strategy": "rsi_bollinger_scalping",
  "stop_loss": {{close}} * 0.98,
  "take_profit": {{close}} * 1.03,
  "confidence": 0.80,
  "notes": "RSI oversold"
}
```

## Pre-flight Checklist
- `WEBHOOK_SECRET` in `.env` matches the secret used to sign payloads
- JSON payload includes `action`, `symbol`, `price`, and `strategy`
- Header `X-Webhook-Signature` is present and built from the exact JSON body
- Test locally with cURL/Postman before letting TradingView alerts run unattended
- Tail `trading_bot.log` to monitor execution results and errors
