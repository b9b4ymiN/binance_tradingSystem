# Pine Script Strategy Template for TradingView
PINE_SCRIPT_TEMPLATE = '''
//@version=6
strategy("Advanced Crypto Trading Bot", overlay=true, margin_long=100, margin_short=100)

// Input parameters based on research
rsi_period = input(14, "RSI Period")
rsi_oversold = input(30, "RSI Oversold Level")
rsi_overbought = input(70, "RSI Overbought Level")
bb_period = input(20, "Bollinger Bands Period")
bb_std = input(2.0, "Bollinger Bands Std Dev")

// Webhook configuration
webhook_url = input.string("YOUR_WEBHOOK_URL", "Webhook URL")

// Calculate indicators
rsi_value = ta.rsi(close, rsi_period)
bb_basis = ta.sma(close, bb_period)
bb_upper = bb_basis + bb_std * ta.stdev(close, bb_period)
bb_lower = bb_basis - bb_std * ta.stdev(close, bb_period)

// Trading conditions
long_condition = rsi_value < rsi_oversold and close <= bb_lower
short_condition = rsi_value > rsi_overbought and close >= bb_upper

// Entry signals
if long_condition
    strategy.entry("Long", strategy.long)
    alert('{"action": "buy", "symbol": "' + syminfo.ticker + '", "price": ' + str.tostring(close) + ', "strategy": "rsi_bollinger_scalping", "stop_loss": ' + str.tostring(close * 0.98) + ', "take_profit": ' + str.tostring(bb_basis) + '}', alert.freq_once_per_bar_close)

if short_condition
    strategy.entry("Short", strategy.short)
    alert('{"action": "sell", "symbol": "' + syminfo.ticker + '", "price": ' + str.tostring(close) + ', "strategy": "rsi_bollinger_scalping", "stop_loss": ' + str.tostring(close * 1.02) + ', "take_profit": ' + str.tostring(bb_basis) + '}', alert.freq_once_per_bar_close)

// Plot indicators
plot(bb_upper, color=color.red, title="BB Upper")
plot(bb_basis, color=color.blue, title="BB Basis")
plot(bb_lower, color=color.green, title="BB Lower")

// Plot RSI
rsi_plot = plot(rsi_value, color=color.purple, title="RSI")
hline(rsi_overbought, "Overbought", color=color.red)
hline(rsi_oversold, "Oversold", color=color.green)
'''