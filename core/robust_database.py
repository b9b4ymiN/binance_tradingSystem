import sqlite3
import threading
import time
import logging
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any
from queue import Queue, Empty, Full
import atexit

logger = logging.getLogger(__name__)

class RobustSQLiteManager:
    """Ultra-robust SQLite manager designed for high-frequency trading"""

    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.pool = Queue(maxsize=max_connections)
        self.lock = threading.RLock()
        self.active_connections = 0
        self.total_connections_created = 0
        self.failed_operations = 0
        self.successful_operations = 0

        # Ensure database directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Initialize the pool
        self._initialize_pool()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        # Health monitoring
        self._last_health_check = time.time()
        self._health_check_interval = 30  # seconds

        logger.info(f"RobustSQLiteManager initialized with {max_connections} connections")

    def _create_optimized_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=60.0,  # Increased timeout
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )

            # Apply aggressive optimizations
            optimizations = [
                'PRAGMA journal_mode=WAL',
                'PRAGMA synchronous=NORMAL',
                'PRAGMA cache_size=20000',  # Increased cache
                'PRAGMA temp_store=MEMORY',
                'PRAGMA busy_timeout=60000',  # 60 second timeout
                'PRAGMA wal_autocheckpoint=1000',
                'PRAGMA optimize',
                'PRAGMA foreign_keys=ON'
            ]

            for pragma in optimizations:
                conn.execute(pragma)

            # Test the connection
            conn.execute('SELECT 1').fetchone()

            with self.lock:
                self.total_connections_created += 1

            logger.debug(f"Created new database connection (total: {self.total_connections_created})")
            return conn

        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise

    def _initialize_pool(self):
        """Initialize the connection pool"""
        for i in range(self.max_connections):
            try:
                conn = self._create_optimized_connection()
                self.pool.put(conn, block=False)
            except (Full, Exception) as e:
                logger.warning(f"Failed to add connection {i+1} to pool: {e}")
                break

        logger.info(f"Initialized connection pool with {self.pool.qsize()} connections")

    def _get_connection_with_retry(self, max_retries: int = 3) -> sqlite3.Connection:
        """Get a connection from pool with retry logic"""

        for attempt in range(max_retries):
            try:
                # Try to get from pool first
                try:
                    conn = self.pool.get(timeout=5.0)

                    # Test if connection is still valid
                    conn.execute('SELECT 1')

                    with self.lock:
                        self.active_connections += 1

                    return conn

                except Empty:
                    # Pool is empty, create temporary connection
                    logger.warning(f"Connection pool exhausted (attempt {attempt + 1})")

                except sqlite3.Error:
                    # Connection is dead, discard it
                    logger.warning(f"Dead connection found, creating new one (attempt {attempt + 1})")
                    try:
                        conn.close()
                    except:
                        pass

                # Create new connection as fallback
                conn = self._create_optimized_connection()
                with self.lock:
                    self.active_connections += 1
                return conn

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        raise Exception("Failed to get database connection after all retries")

    def _return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        try:
            with self.lock:
                self.active_connections -= 1

            if self.pool.qsize() < self.max_connections:
                try:
                    # Test connection before returning
                    conn.execute('SELECT 1')
                    self.pool.put_nowait(conn)
                    logger.debug(f"Returned connection to pool (size: {self.pool.qsize()})")
                    return
                except (sqlite3.Error, Full):
                    pass

            # Close if pool is full or connection is bad
            conn.close()
            logger.debug("Closed excess or invalid connection")

        except Exception as e:
            logger.error(f"Error returning connection: {e}")
            try:
                conn.close()
            except:
                pass

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        start_time = time.time()

        try:
            conn = self._get_connection_with_retry()
            yield conn

            # Record successful operation
            with self.lock:
                self.successful_operations += 1

        except Exception as e:
            # Record failed operation
            with self.lock:
                self.failed_operations += 1

            logger.error(f"Database operation failed after {time.time() - start_time:.3f}s: {e}")
            raise

        finally:
            if conn:
                self._return_connection(conn)

            # Periodic health check
            if time.time() - self._last_health_check > self._health_check_interval:
                self._health_check()

    def execute_with_retry(self, query: str, params: tuple = (), max_retries: int = 3) -> Any:
        """Execute query with automatic retry logic"""

        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    result = cursor.execute(query, params)

                    # Auto-commit for non-SELECT queries
                    if not query.strip().upper().startswith('SELECT'):
                        conn.commit()

                    return result.fetchall() if query.strip().upper().startswith('SELECT') else cursor.rowcount

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = 0.1 * (2 ** attempt)
                    logger.warning(f"Database locked, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                time.sleep(0.05 * (2 ** attempt))

        raise Exception(f"Query failed after {max_retries} attempts")

    def _health_check(self):
        """Perform database health check and maintenance"""
        self._last_health_check = time.time()

        try:
            with self.get_connection() as conn:
                # Check WAL mode
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode")
                mode = cursor.fetchone()[0]

                if mode.upper() != 'WAL':
                    logger.warning(f"Database not in WAL mode (current: {mode}), fixing...")
                    cursor.execute("PRAGMA journal_mode=WAL")

                # Check integrity
                cursor.execute("PRAGMA quick_check")
                integrity = cursor.fetchone()[0]

                if integrity != "ok":
                    logger.error(f"Database integrity issue: {integrity}")

                # Optimize if needed
                cursor.execute("PRAGMA optimize")

            logger.debug("Database health check completed successfully")

        except Exception as e:
            logger.error(f"Database health check failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool and operation statistics"""
        with self.lock:
            return {
                'pool_size': self.max_connections,
                'available_connections': self.pool.qsize(),
                'active_connections': self.active_connections,
                'total_created': self.total_connections_created,
                'successful_operations': self.successful_operations,
                'failed_operations': self.failed_operations,
                'success_rate': (self.successful_operations /
                               max(1, self.successful_operations + self.failed_operations)) * 100
            }

    def _cleanup(self):
        """Cleanup all connections on shutdown"""
        logger.info("Cleaning up database connections...")

        # Close all pooled connections
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except (Empty, Exception) as e:
                logger.debug(f"Error during cleanup: {e}")
                break

        logger.info("Database cleanup completed")

    def force_wal_checkpoint(self):
        """Force WAL checkpoint to reduce WAL file size"""
        try:
            with self.get_connection() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.info("WAL checkpoint completed")
        except Exception as e:
            logger.error(f"WAL checkpoint failed: {e}")

# Global instance
_robust_db_manager: Optional[RobustSQLiteManager] = None

def get_robust_db_manager(db_path: str = "trading_data.db") -> RobustSQLiteManager:
    """Get or create the global robust database manager"""
    global _robust_db_manager

    if _robust_db_manager is None:
        _robust_db_manager = RobustSQLiteManager(db_path)

    return _robust_db_manager

def initialize_robust_db(db_path: str = "trading_data.db", max_connections: int = 10) -> RobustSQLiteManager:
    """Initialize the robust database manager"""
    global _robust_db_manager

    if _robust_db_manager is not None:
        _robust_db_manager._cleanup()

    _robust_db_manager = RobustSQLiteManager(db_path, max_connections)
    return _robust_db_manager