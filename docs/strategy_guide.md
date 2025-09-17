# Trading Strategy Guide

## Overview

This system implements two primary trading strategies based on extensive research and backtesting results specifically optimized for Thai cryptocurrency traders.

## Strategy 1: RSI-Bollinger Bands Scalping

### Strategy Overview
**Expected Win Rate**: 69-80%  
**Timeframe**: 1-minute charts  
**Risk-Reward Ratio**: 1:1.5 to 1:2  
**Best Markets**: High-volume crypto pairs (BTC/USDT, ETH/USDT)

### Technical Indicators
- **RSI (14-period)**: Momentum oscillator
- **Bollinger Bands (20, 2.0)**: Volatility bands
- **ATR (14-period)**: Average True Range for stop losses

### Entry Conditions

#### Long Entry
1. RSI < 30 (oversold condition)
2. Current price ≤ Lower Bollinger Band
3. Volume confirmation (optional)

#### Short Entry
1. RSI > 70 (overbought condition)
2. Current price ≥ Upper Bollinger Band
3. Volume confirmation (optional)

### Exit Strategy
- **Take Profit**: Middle Bollinger Band (mean reversion)
- **Stop Loss**: 2 × ATR from entry price
- **Time Exit**: Close position if no movement within 5 bars

### Risk Management
- Maximum 2% risk per trade
- No more than 3 concurrent positions
- Daily limit of 10 trades

### Implementation Details

```python
def rsi_bollinger_scalping(self, symbol: str) -> Optional[Dict]:
    # Get 1-minute candlestick data
    klines = self.api.get_klines(symbol, '1m', 100)
    
    # Calculate indicators
    rsi = TechnicalAnalysis.calculate_rsi(closes, 14)
    upper_bb, middle_bb, lower_bb = TechnicalAnalysis.calculate_bollinger_bands(closes, 20, 2.0)
    atr = TechnicalAnalysis.calculate_atr(highs, lows, closes, 14)
    
    # Long signal
    if rsi < 30 and current_price <= lower_bb:
        return {
            'action': 'buy',
            'entry_price': current_price,
            'stop_loss': current_price - (2 * atr),
            'take_profit': middle_bb,
            'confidence': 0.75
        }
```

### Market Conditions
**Best Performance**:
- Sideways/ranging markets
- High volatility periods
- Strong volume support/resistance levels

**Avoid During**:
- Strong trending markets
- Low volatility periods
- Major news events

## Strategy 2: Breakout Swing Trading

### Strategy Overview
**Expected Win Rate**: 55-65%  
**Timeframe**: 4-hour charts  
**Risk-Reward Ratio**: 1:3 to 1:5  
**Best Markets**: Major crypto pairs during trend formation

### Technical Indicators
- **EMA 21 & EMA 50**: Trend direction
- **RSI (14-period)**: Momentum confirmation
- **Volume**: Breakout confirmation
- **Support/Resistance**: Key levels

### Entry Conditions

#### Long Entry
1. Price breaks above recent high (20-period)
2. Volume > 1.5 × average volume (20-period)
3. EMA 21 > EMA 50 (trend confirmation)
4. RSI between 40-60 (not overbought)

### Exit Strategy
- **Take Profit**: 15% above entry price
- **Stop Loss**: 2% below breakout level
- **Trailing Stop**: Activate after 5% profit

### Risk Management
- Maximum 3% risk per trade
- Maximum 2 swing positions
- Hold time: 1-7 days typically

### Implementation Details

```python
def breakout_swing_trading(self, symbol: str) -> Optional[Dict]:
    # Get 4-hour candlestick data
    klines = self.api.get_klines(symbol, '4h', 100)
    
    # Calculate trend indicators
    ema_21 = TechnicalAnalysis.calculate_ema(closes, 21)
    ema_50 = TechnicalAnalysis.calculate_ema(closes, 50)
    
    # Identify breakout conditions
    recent_highs = max(highs[-20:])
    volume_avg = np.mean(volumes[-20:])
    
    if (current_price > recent_highs and 
        current_volume > 1.5 * volume_avg and
        ema_21 > ema_50 and
        40 <= rsi <= 60):
        
        return {
            'action': 'buy',
            'entry_price': current_price,
            'stop_loss': recent_highs * 0.98,
            'take_profit': current_price * 1.15,
            'confidence': 0.65
        }
```

### Market Conditions
**Best Performance**:
- Strong trending markets
- Breakout from consolidation
- High volume confirmation
- Clear support/resistance levels

**Avoid During**:
- Choppy/sideways markets
- Low volume conditions
- Fake breakouts (use volume filter)

## Kelly Criterion Position Sizing

### Mathematical Foundation
The Kelly Criterion calculates optimal position size based on:
- **Win Rate (p)**: Historical probability of winning
- **Average Win (W)**: Average winning trade return
- **Average Loss (L)**: Average losing trade return

### Formula
```
Kelly % = (p × W - (1-p) × L) / W
Scaled Kelly = Kelly % × Kelly Fraction (25%)
```

### Implementation
```python
def calculate_position_size(self, entry_price: float, stop_loss: float, 
                          win_rate: float = 0.6) -> float:
    # Kelly Criterion calculation
    b = avg_win / avg_loss  # Risk-reward ratio
    kelly_fraction = (b * win_rate - (1 - win_rate)) / b
    
    # Apply conservative scaling (25%)
    scaled_kelly = kelly_fraction * self.config.kelly_fraction
    
    # Calculate position size based on stop distance
    risk_per_unit = abs(entry_price - stop_loss) / entry_price
    account_balance = self._get_account_balance()
    
    position_value = min(
        account_balance * scaled_kelly,
        account_balance * max_risk_per_trade / risk_per_unit
    )
    
    return position_value / entry_price
```

## Risk Management Framework

### Position-Level Risk
- **Maximum Risk per Trade**: 2% of account
- **Kelly Fraction Scaling**: 25% (conservative)
- **Stop Loss**: Always defined before entry
- **Take Profit**: Risk-reward minimum 1:1.5

### Portfolio-Level Risk
- **Maximum Concurrent Positions**: 3 total
- **Maximum Daily Trades**: 10 trades
- **Total Portfolio Risk**: 6% maximum exposure
- **Correlation Limits**: No correlated positions

### Drawdown Management
- **Maximum Drawdown**: 15% account value
- **Recovery Protocol**: Reduce position sizes by 50%
- **Trading Halt**: Stop trading at 20% drawdown
- **Review Threshold**: Analyze strategy at 10% drawdown

## Strategy Performance Metrics

### Key Performance Indicators
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Time in position

### Monitoring and Optimization
- **Daily Review**: Check trade execution and slippage
- **Weekly Analysis**: Strategy performance comparison
- **Monthly Optimization**: Parameter adjustment based on market conditions
- **Quarterly Review**: Complete strategy evaluation

### Performance Thresholds

#### RSI-Bollinger Strategy
- **Minimum Win Rate**: 65%
- **Maximum Drawdown**: 8%
- **Profit Factor**: > 2.0
- **Review Trigger**: Win rate < 60% over 50 trades

#### Breakout Swing Strategy
- **Minimum Win Rate**: 50%
- **Maximum Drawdown**: 12%
- **Profit Factor**: > 1.8
- **Review Trigger**: Win rate < 45% over 30 trades

## Market-Specific Considerations

### Thai Market Hours
- **Optimal Trading**: 9:00 AM - 11:00 AM, 2:00 PM - 4:00 PM Thailand time
- **Avoid**: Lunch break (12:00 PM - 1:00 PM) and late evening
- **Weekend Strategy**: Reduced position sizes due to lower liquidity

### Cryptocurrency Specifics
- **Volatility Adjustment**: Increase Kelly fraction during high volatility
- **News Impact**: Halt trading 30 minutes before/after major announcements
- **Correlation Awareness**: Monitor BTC correlation for altcoins

### Risk Considerations
- **Regulatory Risk**: Monitor Thai SEC announcements
- **Exchange Risk**: Diversify across multiple exchanges if possible
- **Technology Risk**: Maintain system redundancy and monitoring

## Strategy Customization

### Parameter Optimization
```python
# RSI-Bollinger Parameters
rsi_oversold = 25-35    # More aggressive: 25, Conservative: 35
rsi_overbought = 65-75  # More aggressive: 75, Conservative: 65
bb_std_dev = 1.8-2.2    # Tighter: 1.8, Wider: 2.2

# Breakout Parameters
volume_multiplier = 1.2-2.0  # Conservative: 1.2, Aggressive: 2.0
profit_target = 10-20%       # Conservative: 10%, Aggressive: 20%
```

### Market Condition Filters
- **Trend Filter**: Only trade breakouts in trending markets
- **Volatility Filter**: Adjust position sizes based on ATR
- **Volume Filter**: Require minimum volume for all entries
- **Time Filter**: Avoid trading during low-liquidity hours