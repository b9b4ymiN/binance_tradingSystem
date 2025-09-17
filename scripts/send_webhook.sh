#!/usr/bin/env bash
set -euo pipefail

WEBHOOK_URL=${WEBHOOK_URL:-http://127.0.0.1:5000/webhook}
ENV_FILE=${ENV_FILE:-.env}
SCENARIO=${1:-rsi_buy}

if [[ -f "${ENV_FILE}" ]]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +o allexport
fi

if [[ -z "${WEBHOOK_SECRET:-}" ]]; then
  echo "WEBHOOK_SECRET is not set. Export it or provide it in ${ENV_FILE}." >&2
  exit 1
fi

case "${SCENARIO}" in
  rsi_buy)
    PAYLOAD='{"action":"buy","symbol":"BTCUSDT","price":65000,"strategy":"rsi_bollinger_scalping","stop_loss":63700,"take_profit":66800,"confidence":0.82,"notes":"RSI oversold"}'
    ;;
  rsi_sell)
    PAYLOAD='{"action":"sell","symbol":"BTCUSDT","price":65500,"strategy":"rsi_bollinger_scalping","stop_loss":66800,"take_profit":64000,"confidence":0.78,"notes":"RSI overbought"}'
    ;;
  breakout_buy)
    PAYLOAD='{"action":"buy","symbol":"ETHUSDT","price":3200,"strategy":"breakout_swing","stop_loss":3136,"take_profit":3680,"confidence":0.65,"timeframe":"4h"}'
    ;;
  breakout_sell)
    PAYLOAD='{"action":"sell","symbol":"ETHUSDT","price":3150,"strategy":"breakout_swing","stop_loss":3280,"take_profit":2890,"confidence":0.60,"timeframe":"4h"}'
    ;;
  manual_buy)
    PAYLOAD='{"action":"buy","symbol":"ADAUSDT","price":0.45,"strategy":"manual","stop_loss":0.43,"take_profit":0.49,"notes":"Manual override"}'
    ;;
  *)
    echo "Usage: ${0##*/} [rsi_buy|rsi_sell|breakout_buy|breakout_sell|manual_buy]" >&2
    exit 1
    ;;
 esac

SIGNATURE=$(python - "$WEBHOOK_SECRET" "$PAYLOAD" <<'PY'
import sys
import hmac
import hashlib

secret = sys.argv[1]
payload = sys.argv[2]
print(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest())
PY
)

curl -sS -X POST "${WEBHOOK_URL}" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Signature: ${SIGNATURE}" \
  -d "${PAYLOAD}"
