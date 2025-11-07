# VWAP Mean Reversion Trading Strategy with Automated Parameter Optimization

## 🎯 Overview

ระบบเทรดอัตโนมัติที่ใช้กลยุทธ์ **VWAP Mean Reversion** พร้อมระบบหาพารามิเตอร์ที่ดีที่สุดอัตโนมัติทุกวันอาทิตย์

### ✨ Features ใหม่

- ✅ **VWAP Mean Reversion Strategy** - เทรดเมื่อราคาเบี่ยงเบนจาก VWAP แล้วกลับมา
- ✅ **Automated Parameter Optimization** - หาพารามิเตอร์ที่ดีที่สุดทุกอาทิตย์
- ✅ **Walk-Forward Validation** - ป้องกัน overfitting
- ✅ **Enhanced Backtesting** - มี slippage, commission จริง
- ✅ **Trailing Stop Loss** - เลื่อน stop loss ตามกำไร
- ✅ **Partial Profit Taking** - ปิดกำไรบางส่วนที่เป้าหมาย
- ✅ **Parameter Versioning** - เก็บประวัติพารามิเตอร์ทั้งหมด
- ✅ **Telegram/Email Notifications** - แจ้งเตือนอัตโนมัติ

---

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Engine                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ VWAP       │  │ RSI-BB     │  │ Breakout   │            │
│  │ Strategy   │  │ Strategy   │  │ Strategy   │            │
│  └──────┬─────┘  └────────────┘  └────────────┘            │
│         │                                                     │
│         ├─► Auto Signal Generation (every 30s)              │
│         ├─► Position Monitoring (trailing stop, partial TP) │
│         └─► Risk Management (Kelly Criterion)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼────────┐        ┌─────────▼──────────┐
│ Weekly         │        │ Parameter          │
│ Optimizer      │◄──────►│ Manager            │
│                │        │                    │
│ Every Sunday   │        │ • Version Control  │
│ 00:00 UTC      │        │ • History Tracking │
│                │        │ • Performance      │
│ • Download     │        │ • Rollback         │
│   90d data     │        └────────────────────┘
│ • Grid Search  │
│ • Validation   │
│ • Auto Update  │
└────────────────┘
```

---

## 🎓 VWAP Mean Reversion Strategy

### หลักการทำงาน

กลยุทธ์นี้เชื่อว่า **เมื่อราคาเบี่ยงเบนจาก VWAP มากเกินไป จะมีแนวโน้มกลับมาที่ VWAP** (mean reversion)

### Entry Conditions

#### **LONG Signal (ซื้อ)**
```python
✅ Price ≤ VWAP - (entry_threshold * VWAP)
✅ Volume > avg_volume * volume_multiplier
✅ ATR% อยู่ในช่วงที่เหมาะสม (ไม่เงียบหรือผันผวนเกินไป)
✅ Price ใกล้ Lower VWAP Band

เข้า: current_price
Stop Loss: entry_price - (2.0 * ATR)
Take Profit: VWAP + (1.5 * ATR * 0.5)
```

#### **SHORT Signal (ขาย)**
```python
✅ Price ≥ VWAP + (entry_threshold * VWAP)
✅ Volume > avg_volume * volume_multiplier
✅ ATR% อยู่ในช่วงที่เหมาะสม
✅ Price ใกล้ Upper VWAP Band

เข้า: current_price
Stop Loss: entry_price + (2.0 * ATR)
Take Profit: VWAP - (1.5 * ATR * 0.5)
```

### Exit Conditions

1. **Mean Reversion Complete** - ราคากลับมาใกล้ VWAP (ภายใน exit_threshold)
2. **Take Profit Hit** - ราคาถึงเป้าหมาย
3. **Stop Loss Hit** - ราคาถึง stop loss
4. **Max Holding Period** - ถือพอร์ตเกิน 60 นาที (default)
5. **Partial Profit** - ปิด 50% เมื่อกำไร 1%
6. **Trailing Stop** - เลื่อน stop loss เมื่อกำไร 1.5%+

---

## ⚙️ Default Parameters

```python
vwap_period = 100              # VWAP calculation period
entry_threshold = 0.007        # 0.7% deviation to enter
exit_threshold = 0.002         # 0.2% for mean reversion exit
min_volume_multiplier = 1.5    # Volume must be 1.5x average
stop_loss_atr_mult = 2.0       # SL = 2 * ATR
profit_target_atr_mult = 1.5   # TP = 1.5 * ATR
max_holding_periods = 60       # Exit after 60 minutes
min_atr_filter = 0.003         # Don't trade if ATR < 0.3%
max_atr_filter = 0.035         # Don't trade if ATR > 3.5%
```

**พารามิเตอร์เหล่านี้จะถูกหาค่าที่ดีที่สุดใหม่ทุกวันอาทิตย์โดยอัตโนมัติ!**

---

## 🔄 Weekly Parameter Optimization

### ขั้นตอนการทำงาน (ทุกวันอาทิตย์ 00:00 UTC)

1. **Download Historical Data**
   - ดึงข้อมูล 90 วันย้อนหลัง
   - Symbols: BTCUSDT, ETHUSDT, BNBUSDT

2. **Run Grid Search Optimization**
   - ทดสอบ 100+ parameter combinations
   - Backtest กับข้อมูลจริง (มี slippage + commission)
   - คำนวณ composite score:
     ```
     score = (win_rate * 25%) +
             (profit_factor * 25%) +
             (sharpe_ratio * 20%) +
             (total_return * 20%) +
             (low_drawdown * 10%)
     ```

3. **Walk-Forward Validation**
   - ทดสอบพารามิเตอร์กับข้อมูล 14 วันล่าสุด (out-of-sample)
   - ตรวจสอบว่าไม่ overfit

4. **Compare with Current Parameters**
   - ถ้าพารามิเตอร์ใหม่ดีกว่า ≥5% → **อัพเดท**
   - ถ้าไม่ดีกว่า → **เก็บพารามิเตอร์เดิม**

5. **Update & Notify**
   - บันทึกพารามิเตอร์ใหม่ลงฐานข้อมูล
   - อัพเดท strategy ที่กำลังเทรด
   - ส่งการแจ้งเตือนผ่าน Telegram/Email

---

## 📊 Parameter Optimization Grid

พารามิเตอร์ที่จะทดสอบ:

```python
vwap_period: [50, 75, 100, 150, 200]
entry_threshold: [0.003, 0.005, 0.007, 0.010, 0.012, 0.015]
exit_threshold: [0.001, 0.002, 0.003, 0.005]
min_volume_multiplier: [1.2, 1.5, 2.0, 2.5]
stop_loss_atr_mult: [1.5, 2.0, 2.5, 3.0]
profit_target_atr_mult: [1.0, 1.5, 2.0, 2.5]
max_holding_periods: [30, 60, 90, 120]
min_atr_filter: [0.002, 0.003, 0.005]
max_atr_filter: [0.025, 0.030, 0.035, 0.040]
```

**Total Combinations:** 5 × 6 × 4 × 4 × 4 × 4 × 4 × 3 × 4 = **122,880 combinations**
(จำกัดที่ 100 combinations เพื่อความเร็ว)

---

## 🗄️ Database Schema

### New Tables

#### `parameter_history`
```sql
- version (unique ID)
- strategy_name
- parameters (JSON)
- backtest_score
- backtest_metrics (JSON)
- optimization_period_start
- optimization_period_end
- activated_at
- deactivated_at
- is_active (boolean)
- notes
```

#### `optimization_runs`
```sql
- run_date
- strategy_name
- best_parameters (JSON)
- best_score
- total_combinations
- duration_seconds
- status
- error_message
- notes
```

#### `parameter_performance`
```sql
- version
- strategy_name
- date
- total_trades
- winning_trades
- total_pnl
- win_rate
- profit_factor
- max_drawdown
- sharpe_ratio
```

---

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install APScheduler requests
```

### 2. Environment Variables

สร้างไฟล์ `.env`:

```env
# Binance API
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Webhook
WEBHOOK_SECRET=your_webhook_secret

# Telegram Notifications (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_CHAT_ID=your_chat_id

# Email Notifications (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAIL=your_email@gmail.com
```

#### วิธีการสร้าง Telegram Bot:

1. ค้นหา `@BotFather` บน Telegram
2. ส่งคำสั่ง `/newbot`
3. ตั้งชื่อ bot และรับ **token**
4. เริ่มแชทกับ bot ของคุณ
5. ไปที่ `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` เพื่อหา **chat_id**

### 3. Run System

```bash
# Testnet (แนะนำสำหรับการทดสอบ)
python main.py

# Live Trading (ระวัง!)
# แก้ use_testnet=False ใน main.py แล้วรัน
python main.py
```

---

## 📈 Usage Examples

### ดูพารามิเตอร์ปัจจุบัน

```python
from core.parameter_manager import ParameterManager

param_mgr = ParameterManager('trading_data.db')
active_params = param_mgr.get_active_parameters('vwap_mean_reversion')

print(f"Version: {active_params['version']}")
print(f"Score: {active_params['backtest_score']}")
print(f"Parameters: {active_params['parameters']}")
```

### รัน Optimization ด้วยตัวเอง

```python
from optimization.parameter_optimizer import run_quick_optimization
from config.trading_config import TradingConfig

config = TradingConfig(...)
best_params = run_quick_optimization(config, symbols=['BTCUSDT', 'ETHUSDT'])

print(f"Best Parameters: {best_params.to_dict()}")
```

### ทดสอบ Backtest

```python
from testing.enhanced_backtesting import EnhancedBacktestingEngine
from strategies.vwap_mean_reversion import VWAPMeanReversionStrategy

# สร้าง strategy
strategy = VWAPMeanReversionStrategy(config, binance_api, risk_manager)

# สร้าง backtest engine
backtest = EnhancedBacktestingEngine(
    strategy=strategy,
    initial_balance=10000,
    commission_rate=0.00075,
    slippage_rate=0.0005
)

# รัน backtest
result = backtest.run_backtest('BTCUSDT', historical_klines)

print(f"Win Rate: {result['win_rate']*100:.1f}%")
print(f"Profit Factor: {result['profit_factor']:.2f}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result['max_drawdown']*100:.1f}%")
```

### Rollback พารามิเตอร์

```python
from core.parameter_manager import ParameterManager

param_mgr = ParameterManager('trading_data.db')

# ดูประวัติ
history = param_mgr.get_parameter_history('vwap_mean_reversion', limit=10)
for h in history:
    print(f"{h['version']}: Score={h['backtest_score']:.4f}, Active={h['is_active']}")

# Rollback
param_mgr.rollback_to_version('vwap_mean_reversion', 'vwap_mean_reversion_20250520_123045')
```

---

## ⚠️ Important Notes

### 1. Testnet First!

**ห้ามใช้ live trading ทันที!** ทดสอบกับ Binance Testnet อย่างน้อย 2-4 สัปดาห์

```python
config = TradingConfig(
    use_testnet=True  # ✅ ใช้ testnet
)
```

### 2. Parameter Optimization Time

การหาพารามิเตอร์ที่ดีที่สุดใช้เวลา **30-60 นาที** (ขึ้นอยู่กับจำนวน combinations)

### 3. Overfitting Risk

- ระบบมี walk-forward validation เพื่อป้องกัน overfitting
- ถ้า validation score ต่างจาก optimization score > 50% = overfitting
- จำกัด combinations และใช้ out-of-sample data

### 4. Market Conditions

- VWAP mean reversion ทำงานดีใน **ranging market**
- อาจทำงานได้ไม่ดีใน **strong trending market**
- ระบบมี volatility filters เพื่อหลีกเลี่ยงสภาวะตลาดที่ไม่เหมาะสม

### 5. Transaction Costs

- Commission: 0.075% (Binance with BNB discount)
- Slippage: ~0.05% average
- รวม ~0.125% per trade
- VWAP strategy เทรดบ่อย → ต้นทุนสูง

---

## 📁 File Structure

```
binance_tradingSystem/
├── strategies/
│   ├── vwap_mean_reversion.py       ✨ NEW - VWAP strategy
│   ├── rsi_bollinger.py
│   └── breakout_swing.py
│
├── config/
│   └── vwap_parameters.py           ✨ NEW - Dynamic parameters
│
├── optimization/
│   └── parameter_optimizer.py       ✨ NEW - Grid search optimizer
│
├── scheduler/
│   └── weekly_optimizer.py          ✨ NEW - Weekly automation
│
├── core/
│   ├── parameter_manager.py         ✨ NEW - Version control
│   ├── binance_api.py
│   ├── database.py
│   └── risk_manager.py
│
├── testing/
│   ├── enhanced_backtesting.py      ✨ NEW - Realistic backtest
│   └── backtesting_engine.py
│
├── analysis/
│   └── technical_analysis.py        🔧 UPDATED - VWAP utilities
│
├── notifications/
│   └── notification_manager.py      ✨ NEW - Telegram/Email alerts
│
├── engine/
│   └── trading_engine.py            🔧 UPDATED - VWAP integration
│
└── main.py                          🔧 UPDATED - Scheduler startup
```

---

## 🎉 Summary

### ✅ สิ่งที่เพิ่มเข้ามา

1. **VWAP Mean Reversion Strategy** - กลยุทธ์หลักสำหรับเทรดอัตโนมัติ
2. **Parameter Optimization** - หาค่าที่ดีที่สุดอัตโนมัติทุกอาทิตย์
3. **Enhanced Backtesting** - มี slippage, commission, walk-forward validation
4. **Parameter Manager** - เก็บประวัติ, version control, rollback
5. **Weekly Scheduler** - รัน optimization ทุกอาทิตย์อัตโนมัติ
6. **Advanced Position Management** - trailing stop, partial profit taking
7. **Notification System** - Telegram/Email alerts
8. **Better Auto-trading** - เลือกกลยุทธ์ที่เหมาะสมอัตโนมัติ

### 🚀 ขั้นตอนต่อไป

1. ✅ ทดสอบกับ testnet 2-4 สัปดาห์
2. ✅ ตรวจสอบ parameter updates ทุกอาทิตย์
3. ✅ ติดตาม performance metrics
4. ✅ Optimize เพิ่มเติมถ้าจำเป็น
5. ✅ เมื่อพร้อม → เปิด live trading (เริ่มด้วย capital เล็กๆ)

---

## 📞 Support

หากมีปัญหาหรือคำถาม:

1. ตรวจสอบ logs: `trading_bot.log`
2. ตรวจสอบ database: `trading_data.db`
3. ดู parameter history และ optimization runs

---

**Happy Trading! 🚀📈**
