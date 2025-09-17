import sqlite3
from contextlib import contextmanager


class DatabaseManager:
    """Comprehensive database management for trade logging"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
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
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            # Portfolio tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
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
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()