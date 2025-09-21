import sqlite3
import threading
import time
import logging
from queue import Queue, Empty
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """Connection pool for SQLite to reduce lock contention"""

    def __init__(self, db_path: str, pool_size: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.active_connections = 0
        self._initialize_pool()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimized settings"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False
        )

        # Optimize for concurrent access
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=10000')
        conn.execute('PRAGMA temp_store=MEMORY')
        conn.execute('PRAGMA busy_timeout=30000')
        conn.execute('PRAGMA foreign_keys=ON')

        return conn

    def _initialize_pool(self):
        """Initialize the connection pool"""
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                self.pool.put(conn)
                logger.debug(f"Added connection to pool (total: {self.pool.qsize()})")
            except Exception as e:
                logger.error(f"Failed to create database connection: {e}")

    def get_connection(self, timeout: Optional[float] = None) -> sqlite3.Connection:
        """Get a connection from the pool"""
        if timeout is None:
            timeout = self.timeout

        try:
            # Try to get existing connection from pool
            conn = self.pool.get(timeout=timeout)

            # Test if connection is still valid
            try:
                conn.execute('SELECT 1')
                with self.lock:
                    self.active_connections += 1
                return conn
            except sqlite3.Error:
                # Connection is dead, create a new one
                logger.warning("Dead connection found, creating new one")
                conn.close()
                conn = self._create_connection()
                with self.lock:
                    self.active_connections += 1
                return conn

        except Empty:
            # Pool is empty, create temporary connection
            logger.warning("Connection pool exhausted, creating temporary connection")
            with self.lock:
                self.active_connections += 1
            return self._create_connection()

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        try:
            with self.lock:
                self.active_connections -= 1

            if self.pool.qsize() < self.pool_size:
                # Test connection before returning to pool
                try:
                    conn.execute('SELECT 1')
                    self.pool.put_nowait(conn)
                    logger.debug(f"Returned connection to pool (total: {self.pool.qsize()})")
                except sqlite3.Error:
                    # Connection is dead, close it
                    logger.warning("Dead connection detected, closing")
                    conn.close()
                except:
                    # Pool is full, close connection
                    conn.close()
            else:
                # Pool is full, close connection
                conn.close()

        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
            try:
                conn.close()
            except:
                pass

    @contextmanager
    def get_connection_context(self, timeout: Optional[float] = None):
        """Context manager for getting and returning connections"""
        conn = None
        try:
            conn = self.get_connection(timeout)
            yield conn
        finally:
            if conn:
                self.return_connection(conn)

    def close_all(self):
        """Close all connections in the pool"""
        logger.info("Closing all database connections")

        # Close connections in pool
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error closing pooled connection: {e}")

        logger.info(f"Database connection pool closed")

    def stats(self) -> dict:
        """Get pool statistics"""
        return {
            'pool_size': self.pool_size,
            'available_connections': self.pool.qsize(),
            'active_connections': self.active_connections,
            'total_capacity': self.pool_size
        }

# Global pool instance (will be initialized by DatabaseManager)
_connection_pool: Optional[SQLiteConnectionPool] = None

def get_pool() -> SQLiteConnectionPool:
    """Get the global connection pool"""
    global _connection_pool
    if _connection_pool is None:
        raise RuntimeError("Database connection pool not initialized")
    return _connection_pool

def initialize_pool(db_path: str, pool_size: int = 5) -> SQLiteConnectionPool:
    """Initialize the global connection pool"""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.close_all()

    _connection_pool = SQLiteConnectionPool(db_path, pool_size)
    logger.info(f"Database connection pool initialized with {pool_size} connections")
    return _connection_pool