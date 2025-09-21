#!/usr/bin/env python3
"""
Emergency Position Closer
Closes all current trading positions to reduce risk exposure
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Load environment variables
def load_env_file(path='.env'):
    if not Path(path).exists():
        print(f"❌ Environment file {path} not found")
        return
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip())

load_env_file()

from config.trading_config import TradingConfig
from engine.trading_engine import TradingEngine

class PositionCloser:
    def __init__(self):
        self.config = TradingConfig(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY'),
            use_testnet=True
        )
        self.trading_engine = TradingEngine(self.config)
        self.closed_positions = []
        self.failed_positions = []

    def get_current_positions(self):
        """Get current positions from database"""
        positions = []
        try:
            with self.trading_engine.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, quantity, avg_price
                    FROM portfolio
                    WHERE ABS(quantity) > 0.000001
                    ORDER BY ABS(quantity * avg_price) DESC
                ''')

                for symbol, quantity, avg_price in cursor.fetchall():
                    side = 'LONG' if quantity > 0 else 'SHORT'
                    market_value = abs(quantity) * avg_price

                    positions.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'abs_quantity': abs(quantity),
                        'avg_price': avg_price,
                        'side': side,
                        'market_value': market_value
                    })

            return positions
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return []

    def get_current_price(self, symbol):
        """Get current market price for symbol"""
        try:
            return self.trading_engine.binance_api.get_current_price(symbol)
        except Exception as e:
            print(f"⚠️ Warning: Could not get current price for {symbol}: {e}")
            return None

    def close_position(self, position):
        """Close a single position"""
        symbol = position['symbol']
        quantity = position['abs_quantity']
        side = 'SELL' if position['side'] == 'LONG' else 'BUY'  # Opposite side to close

        print(f"\n🔄 Closing {position['side']} position: {symbol}")
        print(f"   Quantity: {quantity:.6f}")
        print(f"   Entry Price: ${position['avg_price']:.2f}")
        print(f"   Market Value: ${position['market_value']:.2f}")

        try:
            # Get current market price
            current_price = self.get_current_price(symbol)
            if current_price:
                print(f"   Current Price: ${current_price:.2f}")

                # Calculate potential P&L
                if position['side'] == 'LONG':
                    pnl = quantity * (current_price - position['avg_price'])
                else:  # SHORT
                    pnl = quantity * (position['avg_price'] - current_price)
                print(f"   Estimated P&L: ${pnl:.2f}")

            # Place market order to close position
            order_result = self.trading_engine.binance_api.place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )

            if order_result.get('status') == 'FILLED':
                print(f"✅ Successfully closed {symbol} position")
                print(f"   Order ID: {order_result.get('orderId')}")
                print(f"   Executed Qty: {order_result.get('executedQty')}")
                print(f"   Avg Price: ${float(order_result.get('cummulativeQuoteQty', 0)) / float(order_result.get('executedQty', 1)):.2f}")

                self.closed_positions.append({
                    'symbol': symbol,
                    'original_side': position['side'],
                    'quantity': quantity,
                    'order_result': order_result,
                    'timestamp': datetime.now().isoformat()
                })

                # Log the trade to update database
                signal_data = {
                    'action': side.lower(),
                    'symbol': symbol,
                    'quantity': quantity,
                    'strategy': 'position_closer',
                    'reason': 'risk_limit_exceeded'
                }

                try:
                    self.trading_engine.lockless_storage.log_trade(order_result, signal_data)
                    print(f"✅ Trade logged successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to log trade: {e}")

                return True

            else:
                print(f"❌ Failed to close {symbol}: Order status {order_result.get('status')}")
                self.failed_positions.append({
                    'symbol': symbol,
                    'reason': f"Order status: {order_result.get('status')}",
                    'order_result': order_result
                })
                return False

        except Exception as e:
            print(f"❌ Error closing {symbol}: {e}")
            self.failed_positions.append({
                'symbol': symbol,
                'reason': str(e),
                'error': True
            })
            return False

    def close_all_positions(self, confirm=True):
        """Close all current positions"""
        print("🚨 EMERGENCY POSITION CLOSER")
        print("=" * 50)

        # Get current positions
        positions = self.get_current_positions()

        if not positions:
            print("✅ No open positions found")
            return True

        print(f"📊 Found {len(positions)} open positions:")
        print()

        total_market_value = 0
        for i, pos in enumerate(positions, 1):
            print(f"{i}. {pos['symbol']} - {pos['side']}")
            print(f"   Quantity: {pos['abs_quantity']:.6f}")
            print(f"   Entry: ${pos['avg_price']:.2f}")
            print(f"   Value: ${pos['market_value']:.2f}")
            total_market_value += pos['market_value']

        print(f"\n💰 Total Market Value: ${total_market_value:.2f}")

        if confirm:
            print("\n⚠️ WARNING: This will close ALL positions!")
            response = input("Do you want to continue? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Operation cancelled by user")
                return False

        print(f"\n🎯 Starting to close {len(positions)} positions...")

        # Close positions one by one
        success_count = 0
        for i, position in enumerate(positions, 1):
            print(f"\n[{i}/{len(positions)}] Processing {position['symbol']}...")

            if self.close_position(position):
                success_count += 1
                # Small delay between orders to avoid rate limits
                time.sleep(1)
            else:
                print(f"⚠️ Failed to close {position['symbol']}, continuing with others...")
                time.sleep(2)  # Longer delay on failure

        # Summary
        print("\n" + "=" * 50)
        print("📋 CLOSING SUMMARY")
        print("=" * 50)

        print(f"✅ Successfully closed: {success_count}/{len(positions)} positions")

        if self.closed_positions:
            print("\n✅ CLOSED POSITIONS:")
            for pos in self.closed_positions:
                print(f"   {pos['symbol']} - {pos['original_side']} - {pos['quantity']:.6f}")

        if self.failed_positions:
            print("\n❌ FAILED POSITIONS:")
            for pos in self.failed_positions:
                print(f"   {pos['symbol']} - {pos['reason']}")

        # Check remaining positions
        remaining_positions = self.get_current_positions()
        if remaining_positions:
            print(f"\n⚠️ WARNING: {len(remaining_positions)} positions still open:")
            for pos in remaining_positions:
                print(f"   {pos['symbol']} - {pos['side']} - {pos['abs_quantity']:.6f}")
        else:
            print("\n🎉 ALL POSITIONS SUCCESSFULLY CLOSED!")

        return success_count == len(positions)

    def save_closing_report(self):
        """Save detailed closing report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'closed_positions': self.closed_positions,
            'failed_positions': self.failed_positions,
            'summary': {
                'total_attempted': len(self.closed_positions) + len(self.failed_positions),
                'successful': len(self.closed_positions),
                'failed': len(self.failed_positions)
            }
        }

        report_file = f"position_closing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Detailed report saved: {report_file}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")

def main():
    """Main function"""
    print("EMERGENCY POSITION CLOSER")
    print("Designed to close all trading positions when risk limits are exceeded")
    print()

    try:
        closer = PositionCloser()

        # Close all positions (with confirmation)
        success = closer.close_all_positions(confirm=True)

        # Save report
        closer.save_closing_report()

        if success:
            print("\n🎉 All positions closed successfully!")
            print("Risk exposure has been reduced to zero.")
            print("You can now resume trading with clean slate.")
        else:
            print("\n⚠️ Some positions could not be closed.")
            print("Please check the failed positions manually.")
            print("Consider using Binance web interface for manual closure.")

        return success

    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("Please close positions manually via Binance interface")
        return False

if __name__ == '__main__':
    main()