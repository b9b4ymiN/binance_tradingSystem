#!/usr/bin/env python3
import sqlite3

def analyze_pnl_issue():
    print('=== PnL CALCULATION ANALYSIS ===')

    # Connect to database
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()

    # Look at trades for a specific symbol to see the pattern
    cursor.execute('''
        SELECT id, symbol, side, quantity, price, realized_pnl, timestamp
        FROM trades
        WHERE symbol = 'BTCUSDT' AND status = 'FILLED'
        ORDER BY timestamp
        LIMIT 15
    ''')
    btc_trades = cursor.fetchall()

    print('BTCUSDT trades (chronological order):')
    for trade in btc_trades:
        trade_id, symbol, side, qty, price, pnl, ts = trade
        pnl = pnl or 0
        print('  ID:{} {} {} {:.6f} @ ${:.2f} | P&L: ${:.4f} | {}'.format(
            trade_id, symbol, side, qty, price, pnl, ts))

    # Check portfolio history for BTCUSDT
    print('\nCurrent BTCUSDT portfolio position:')
    cursor.execute('SELECT quantity, avg_price FROM portfolio WHERE symbol = ?', ('BTCUSDT',))
    result = cursor.fetchone()
    if result:
        qty, avg_price = result
        print('  Quantity: {:.6f}, Avg Price: ${:.2f}'.format(qty, avg_price))
    else:
        print('  No current position')

    # Analyze the PnL calculation issue
    print('\n=== PnL CALCULATION ISSUE ANALYSIS ===')

    # Check lockless storage files
    print('Checking lockless storage files...')
    import os
    from pathlib import Path

    lockless_path = Path('trading_data')
    if lockless_path.exists():
        trade_files = list(lockless_path.glob('trades/*.json'))
        print('Trade files in lockless storage:', len(trade_files))

        if trade_files:
            import json
            # Read a few recent trades
            recent_files = sorted(trade_files, reverse=True)[:5]
            print('\nRecent lockless trade files:')
            for trade_file in recent_files:
                try:
                    with open(trade_file, 'r') as f:
                        trade_data = json.load(f)
                    print('  {}: {} {} {:.6f} @ ${:.2f} | PnL: ${:.4f}'.format(
                        trade_data.get('id', 'Unknown'),
                        trade_data.get('symbol', 'Unknown'),
                        trade_data.get('side', 'Unknown'),
                        trade_data.get('quantity', 0),
                        trade_data.get('price', 0),
                        trade_data.get('realized_pnl', 0)
                    ))
                except Exception as e:
                    print(f'  Error reading {trade_file}: {e}')
    else:
        print('No lockless storage directory found')

    print('\n=== ISSUE IDENTIFIED ===')
    print('The issue is in the PnL calculation logic:')
    print('1. Lockless storage sets realized_pnl to 0 for all trades (line 64 in lockless_storage.py)')
    print('2. Database PnL calculation only works for position closes, not new positions')
    print('3. Most trades are opening new positions, not closing existing ones')
    print('4. This results in zero PnL for most trades')

    conn.close()

if __name__ == '__main__':
    analyze_pnl_issue()