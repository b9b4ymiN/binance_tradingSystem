#!/usr/bin/env python3
"""
Order Cancellation Script
Cancels all open orders to clear the way for position closing
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Load environment variables
def load_env_file(path='.env'):
    if not Path(path).exists():
        print(f"Environment file {path} not found")
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

class OrderCanceller:
    def __init__(self):
        self.config = TradingConfig(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY'),
            use_testnet=True
        )
        self.trading_engine = TradingEngine(self.config)
        self.cancelled_orders = []
        self.failed_cancellations = []

    def get_all_open_orders(self):
        """Get all currently open orders"""
        try:
            return self.trading_engine.binance_api.get_open_orders()
        except Exception as e:
            print(f"Error getting open orders: {e}")
            return []

    def cancel_single_order(self, symbol, order_id):
        """Cancel a single order"""
        try:
            print(f"Cancelling order {order_id} for {symbol}...")
            result = self.trading_engine.binance_api.cancel_order(symbol, order_id)

            if result.get('status') == 'CANCELED':
                print(f"Successfully cancelled order {order_id}")
                self.cancelled_orders.append({
                    'symbol': symbol,
                    'order_id': order_id,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            else:
                print(f"Failed to cancel order {order_id}: {result}")
                self.failed_cancellations.append({
                    'symbol': symbol,
                    'order_id': order_id,
                    'reason': f"Status: {result.get('status')}",
                    'result': result
                })
                return False

        except Exception as e:
            print(f"Error cancelling order {order_id}: {e}")
            self.failed_cancellations.append({
                'symbol': symbol,
                'order_id': order_id,
                'reason': str(e),
                'error': True
            })
            return False

    def cancel_all_orders(self, confirm=True):
        """Cancel all open orders"""
        print("ORDER CANCELLATION SCRIPT")
        print("=" * 50)

        # Get all open orders
        open_orders = self.get_all_open_orders()

        if not open_orders:
            print("No open orders found")
            return True

        print(f"Found {len(open_orders)} open orders:")
        print()

        # Group by symbol for better display
        by_symbol = {}
        for order in open_orders:
            symbol = order['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(order)

        for symbol, orders in by_symbol.items():
            print(f"{symbol}: {len(orders)} orders")
            for order in orders:
                print(f"  - {order['side']} {order['type']} (ID: {order['orderId']})")

        if confirm:
            print(f"\nWARNING: This will cancel ALL {len(open_orders)} orders!")
            response = input("Do you want to continue? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Operation cancelled by user")
                return False

        print(f"\nStarting to cancel {len(open_orders)} orders...")

        # Cancel orders one by one
        success_count = 0
        for i, order in enumerate(open_orders, 1):
            print(f"\n[{i}/{len(open_orders)}] Processing order {order['orderId']}...")

            if self.cancel_single_order(order['symbol'], order['orderId']):
                success_count += 1
                # Small delay between cancellations to avoid rate limits
                time.sleep(0.5)
            else:
                print(f"Failed to cancel order {order['orderId']}, continuing...")
                time.sleep(1)  # Longer delay on failure

        # Summary
        print("\n" + "=" * 50)
        print("CANCELLATION SUMMARY")
        print("=" * 50)

        print(f"Successfully cancelled: {success_count}/{len(open_orders)} orders")

        if self.cancelled_orders:
            print("\nCANCELLED ORDERS:")
            for order in self.cancelled_orders:
                print(f"   {order['symbol']} - Order ID: {order['order_id']}")

        if self.failed_cancellations:
            print("\nFAILED CANCELLATIONS:")
            for order in self.failed_cancellations:
                print(f"   {order['symbol']} - Order ID: {order['order_id']} - {order['reason']}")

        # Check remaining orders
        remaining_orders = self.get_all_open_orders()
        if remaining_orders:
            print(f"\nWARNING: {len(remaining_orders)} orders still open")
        else:
            print("\nALL ORDERS SUCCESSFULLY CANCELLED!")

        return success_count == len(open_orders)

def main():
    """Main function"""
    print("ORDER CANCELLATION SCRIPT")
    print("Designed to cancel all open orders before closing positions")
    print()

    try:
        canceller = OrderCanceller()

        # Cancel all orders (with confirmation)
        success = canceller.cancel_all_orders(confirm=True)

        if success:
            print("\nAll orders cancelled successfully!")
            print("You can now proceed to close positions with:")
            print("python close_all_positions.py")
        else:
            print("\nSome orders could not be cancelled.")
            print("Please check the failed orders manually.")
            print("You may need to cancel them via Binance interface.")

        return success

    except Exception as e:
        print(f"Critical error: {e}")
        print("Please cancel orders manually via Binance interface")
        return False

if __name__ == '__main__':
    main()