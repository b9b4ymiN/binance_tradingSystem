import sqlite3
from contextlib import contextmanager
from datetime import datetime
from .db_pool import initialize_pool, get_pool
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Comprehensive database management for trade logging"""

    def __init__(self, db_path: str):
        self.db_path = db_path

        # Initialize connection pool
        self.pool = initialize_pool(db_path, pool_size=5)

        # Initialize database schema
        self.init_database()
    
    def init_database(self):
        """Initialize all required database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trading history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT UNIQUE,
                    strategy TEXT,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'PENDING',
                    commission REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0
                )
            ''')

            # Add missing columns to existing trades table if they don't exist
            try:
                cursor.execute('ALTER TABLE trades ADD COLUMN commission REAL DEFAULT 0')
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute('ALTER TABLE trades ADD COLUMN realized_pnl REAL DEFAULT 0')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Portfolio tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    entry_price REAL,
                    current_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    unrealized_pnl_percent REAL,
                    position_size_usd REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_price_update DATETIME
                )
            ''')

            # Add new columns to existing portfolio table if they don't exist
            new_columns = [
                'entry_price REAL',
                'current_price REAL',
                'market_value REAL',
                'unrealized_pnl REAL',
                'unrealized_pnl_percent REAL',
                'position_size_usd REAL',
                'last_price_update DATETIME'
            ]

            for column in new_columns:
                try:
                    cursor.execute(f'ALTER TABLE portfolio ADD COLUMN {column}')
                except sqlite3.OperationalError:
                    pass  # Column already exists
            
            # Performance metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    date DATE PRIMARY KEY,
                    total_pnl REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    max_drawdown REAL
                )
            ''')
            
            # Risk metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_risk_exposure REAL,
                    position_count INTEGER,
                    kelly_fraction REAL,
                    daily_trades INTEGER
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections using connection pool"""
        with self.pool.get_connection_context() as conn:
            yield conn

    def close_pool(self):
        """Close the database connection pool"""
        if hasattr(self, 'pool'):
            self.pool.close_all()

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics"""
        return self.pool.stats()

    def update_daily_performance(self, date: str = None) -> bool:
        """Update daily performance metrics in the performance table"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Calculate daily metrics from trades
                cursor.execute('''
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(realized_pnl) as total_pnl,
                        AVG(realized_pnl) as avg_pnl,
                        MIN(realized_pnl) as min_pnl,
                        MAX(realized_pnl) as max_pnl
                    FROM trades
                    WHERE DATE(timestamp) = ? AND status = 'FILLED'
                ''', (date,))

                result = cursor.fetchone()
                if not result or result[0] == 0:
                    logger.info(f"No trades found for date {date}")
                    return False

                total_trades, winning_trades, total_pnl, avg_pnl, min_pnl, max_pnl = result
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

                # Calculate max drawdown using realized P&L cumulative curve
                cursor.execute("SELECT COALESCE(realized_pnl, 0) FROM trades WHERE DATE(timestamp) = ? AND status = 'FILLED' ORDER BY timestamp", (date,))
                daily_pnls = [row[0] or 0 for row in cursor.fetchall()]

                cumulative_total = 0.0
                peak = 0.0
                max_drawdown = 0.0

                for pnl_value in daily_pnls:
                    cumulative_total += pnl_value
                    if cumulative_total > peak:
                        peak = cumulative_total
                    drawdown = peak - cumulative_total
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                # Insert or update performance record
                cursor.execute('''
                    INSERT OR REPLACE INTO performance
                    (date, total_pnl, win_rate, total_trades, max_drawdown)
                    VALUES (?, ?, ?, ?, ?)
                ''', (date, total_pnl or 0, win_rate, total_trades, max_drawdown))

                conn.commit()
                logger.info(f"Updated performance for {date}: {total_trades} trades, {total_pnl:.2f} PnL, {win_rate:.1f}% win rate")
                return True

        except Exception as e:
            logger.error(f"Failed to update daily performance for {date}: {e}")
            return False

    def sync_portfolio_from_lockless(self, lockless_storage) -> bool:
        """Sync portfolio table with lockless storage data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Clear existing portfolio data
                cursor.execute('DELETE FROM portfolio')

                # Get positions from lockless storage
                positions = lockless_storage.get_positions()

                # Insert current positions
                for position in positions:
                    cursor.execute('''
                        INSERT INTO portfolio (symbol, quantity, avg_price, last_updated)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        position['symbol'],
                        position['quantity'],
                        position['avg_price'],
                        position.get('last_updated', datetime.now().isoformat())
                    ))

                conn.commit()
                logger.info(f"Synced {len(positions)} portfolio positions from lockless storage")
                return True

        except Exception as e:
            logger.error(f"Failed to sync portfolio from lockless storage: {e}")
            return False

    def update_portfolio_position(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Update portfolio position after a trade"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get current position
                cursor.execute('SELECT quantity, avg_price FROM portfolio WHERE symbol = ?', (symbol,))
                result = cursor.fetchone()

                if result:
                    current_qty, current_avg_price = result
                else:
                    current_qty, current_avg_price = 0, 0

                # Calculate new position
                if side.upper() == 'BUY':
                    new_qty = current_qty + quantity
                    if new_qty != 0:
                        new_avg_price = ((current_qty * current_avg_price) + (quantity * price)) / new_qty
                    else:
                        new_avg_price = price
                else:  # SELL
                    new_qty = current_qty - quantity
                    new_avg_price = current_avg_price if current_qty != 0 else price

                # Update or insert position
                if abs(new_qty) < 0.000001:
                    # Position closed
                    cursor.execute('DELETE FROM portfolio WHERE symbol = ?', (symbol,))
                    logger.info(f"Closed position for {symbol}")
                else:
                    # Update position
                    cursor.execute('''
                        INSERT OR REPLACE INTO portfolio (symbol, quantity, avg_price, last_updated)
                        VALUES (?, ?, ?, ?)
                    ''', (symbol, new_qty, new_avg_price, datetime.now().isoformat()))
                    logger.info(f"Updated position for {symbol}: {new_qty} @ {new_avg_price}")

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to update portfolio position for {symbol}: {e}")
            return False

    def update_position_market_data(self, symbol: str, current_price: float) -> bool:
        """Update position with current market price and calculate unrealized P&L"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get current position
                cursor.execute('''
                    SELECT quantity, avg_price, entry_price FROM portfolio
                    WHERE symbol = ?
                ''', (symbol,))
                result = cursor.fetchone()

                if not result:
                    return False

                quantity, avg_price, entry_price = result
                entry_price = entry_price or avg_price  # Use avg_price if entry_price is None

                # Calculate market metrics
                position_value = abs(quantity) * current_price

                # Calculate unrealized P&L
                if quantity > 0:  # LONG position
                    unrealized_pnl = quantity * (current_price - entry_price)
                else:  # SHORT position
                    unrealized_pnl = abs(quantity) * (entry_price - current_price)

                # Calculate percentage
                if entry_price > 0:
                    unrealized_pnl_percent = (unrealized_pnl / (abs(quantity) * entry_price)) * 100
                else:
                    unrealized_pnl_percent = 0

                # Update position with market data
                cursor.execute('''
                    UPDATE portfolio
                    SET current_price = ?,
                        market_value = ?,
                        unrealized_pnl = ?,
                        unrealized_pnl_percent = ?,
                        position_size_usd = ?,
                        entry_price = COALESCE(entry_price, ?),
                        last_price_update = ?
                    WHERE symbol = ?
                ''', (
                    current_price,
                    position_value,
                    unrealized_pnl,
                    unrealized_pnl_percent,
                    position_value,
                    entry_price,
                    datetime.now().isoformat(),
                    symbol
                ))

                conn.commit()
                logger.debug(f"Updated market data for {symbol}: ${current_price}, P&L: ${unrealized_pnl:.2f} ({unrealized_pnl_percent:.2f}%)")
                return True

        except Exception as e:
            logger.error(f"Failed to update market data for {symbol}: {e}")
            return False

    def update_all_positions_market_data(self, price_fetcher_func) -> int:
        """Update market data for all positions using a price fetcher function"""
        updated_count = 0

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT symbol FROM portfolio WHERE ABS(quantity) > 0.000001')
                symbols = [row[0] for row in cursor.fetchall()]

            for symbol in symbols:
                try:
                    current_price = price_fetcher_func(symbol)
                    if current_price and current_price > 0:
                        if self.update_position_market_data(symbol, current_price):
                            updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to update price for {symbol}: {e}")

            logger.info(f"Updated market data for {updated_count}/{len(symbols)} positions")
            return updated_count

        except Exception as e:
            logger.error(f"Failed to update positions market data: {e}")
            return 0
