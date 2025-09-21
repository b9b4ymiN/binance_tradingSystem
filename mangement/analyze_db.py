#!/usr/bin/env python3
import sqlite3
from datetime import datetime

def analyze_database():
    print('=== DATABASE ANALYSIS ===')

    # Connect to database
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()

    # Check for incomplete records
    cursor.execute('SELECT COUNT(*) FROM trades WHERE status != ?', ('FILLED',))
    incomplete_trades = cursor.fetchone()[0]
    print('Incomplete trades:', incomplete_trades)

    # Check for missing PnL data
    cursor.execute('SELECT COUNT(*) FROM trades WHERE status = ? AND realized_pnl IS NULL', ('FILLED',))
    missing_pnl = cursor.fetchone()[0]
    print('Trades with missing P&L:', missing_pnl)

    # Check for zero PnL
    cursor.execute('SELECT COUNT(*) FROM trades WHERE status = ? AND realized_pnl = 0', ('FILLED',))
    zero_pnl_trades = cursor.fetchone()[0]
    print('Trades with zero P&L:', zero_pnl_trades)

    # Performance summary
    cursor.execute('SELECT SUM(realized_pnl), COUNT(*), AVG(realized_pnl) FROM trades WHERE status = ?', ('FILLED',))
    result = cursor.fetchone()
    total_pnl, total_trades, avg_pnl = result if result else (0, 0, 0)
    total_pnl = total_pnl or 0
    avg_pnl = avg_pnl or 0

    print()
    print('Total P&L: ${:.4f}'.format(total_pnl))
    print('Total completed trades:', total_trades)
    print('Average P&L per trade: ${:.4f}'.format(avg_pnl))

    # Win rate
    cursor.execute('SELECT COUNT(*) FROM trades WHERE status = ? AND realized_pnl > 0', ('FILLED',))
    winning_trades = cursor.fetchone()[0]
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    print('Win rate: {:.1f}% ({}/{})'.format(win_rate, winning_trades, total_trades))

    # Check for missing commission data
    cursor.execute('SELECT COUNT(*) FROM trades WHERE status = ? AND commission IS NULL', ('FILLED',))
    missing_commission = cursor.fetchone()[0]
    print()
    print('Trades with missing commission:', missing_commission)

    print()
    print('Recent trades with P&L details:')
    cursor.execute('''
        SELECT symbol, side, quantity, price, realized_pnl, commission, timestamp
        FROM trades
        WHERE status = ?
        ORDER BY timestamp DESC
        LIMIT 10
    ''', ('FILLED',))
    trades = cursor.fetchall()
    for trade in trades:
        symbol, side, qty, price, pnl, comm, ts = trade
        pnl = pnl or 0
        comm = comm or 0
        print('  {} {} {:.6f} @ ${:.2f} | P&L: ${:.4f} | Commission: ${:.4f} | {}'.format(
            symbol, side, qty, price, pnl, comm, ts))

    # Check portfolio positions
    print()
    print('=== PORTFOLIO ANALYSIS ===')
    cursor.execute('SELECT symbol, quantity, avg_price, unrealized_pnl FROM portfolio')
    positions = cursor.fetchall()
    print('Current positions:', len(positions))
    for pos in positions:
        symbol, qty, avg_price, unrealized_pnl = pos
        unrealized_pnl = unrealized_pnl or 0
        print('  {}: {:.6f} @ ${:.2f} | Unrealized P&L: ${:.4f}'.format(
            symbol, qty, avg_price, unrealized_pnl))

    conn.close()

    print()
    print('=== WEBHOOK STATUS ===')
    print('Webhook server is not currently running.')
    print('To start: python main.py')

if __name__ == '__main__':
    analyze_database()