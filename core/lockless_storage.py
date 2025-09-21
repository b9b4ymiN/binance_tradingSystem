import json
import os
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class LocklessTradingStorage:
    """
    Lock-free file-based storage for trading data
    Each trade is stored as a separate JSON file to eliminate database locks
    """

    def __init__(self, base_path: str = "trading_data"):
        self.base_path = Path(base_path)
        self.trades_dir = self.base_path / "trades"
        self.portfolio_dir = self.base_path / "portfolio"
        self.performance_dir = self.base_path / "performance"

        # Create directories
        for dir_path in [self.trades_dir, self.portfolio_dir, self.performance_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        logger.info(f"LocklessTradingStorage initialized at {self.base_path}")

    def log_trade(self, order_result: Dict, signal_data: Dict) -> str:
        """Log trade to individual file"""
        try:
            # Calculate average fill price
            fills = order_result.get('fills', [])
            if fills:
                total_qty = sum(float(fill['qty']) for fill in fills)
                total_value = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                avg_price = total_value / total_qty if total_qty > 0 else 0
                commission = sum(float(fill.get('commission', 0)) for fill in fills)
            else:
                executed_qty = float(order_result.get('executedQty', 0))
                cumulative_quote_qty = float(order_result.get('cummulativeQuoteQty', 0))
                if executed_qty > 0 and cumulative_quote_qty > 0:
                    avg_price = cumulative_quote_qty / executed_qty
                else:
                    avg_price = float(order_result.get('price', 0))
                commission = 0

            # Create trade record
            trade_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            trade_data = {
                'id': trade_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': order_result.get('symbol'),
                'side': order_result.get('side'),
                'quantity': float(order_result.get('executedQty', 0)),
                'price': avg_price,
                'order_id': str(order_result.get('orderId')),
                'strategy': signal_data.get('strategy', 'manual'),
                'status': order_result.get('status', 'PENDING').upper(),
                'commission': commission,
                'realized_pnl': 0,
                'fills': fills
            }

            # Write trade to file
            trade_file = self.trades_dir / f"{trade_id}.json"
            with open(trade_file, 'w') as f:
                json.dump(trade_data, f, indent=2)

            # Update portfolio
            self._update_portfolio_atomic(trade_data)

            logger.info(f"Trade logged successfully: {trade_id}")
            return trade_id

        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            raise

    def _update_portfolio_atomic(self, trade_data: Dict):
        """Update portfolio position atomically"""
        symbol = trade_data['symbol']
        side = trade_data['side']
        quantity = trade_data['quantity']
        price = trade_data['price']

        portfolio_file = self.portfolio_dir / f"{symbol}.json"

        with self.lock:
            # Read current position
            current_position = {'quantity': 0, 'avg_price': 0}
            if portfolio_file.exists():
                try:
                    with open(portfolio_file, 'r') as f:
                        current_position = json.load(f)
                except:
                    pass

            current_qty = current_position.get('quantity', 0)
            current_avg_price = current_position.get('avg_price', 0)

            logger.info(f"Updating position: {symbol} {side} {quantity} @ {price}")
            logger.info(f"Current position: {current_qty} @ {current_avg_price}")

            # Calculate new position
            if side == 'BUY':
                new_qty = current_qty + quantity
                if new_qty != 0:
                    new_avg_price = ((current_qty * current_avg_price) + (quantity * price)) / new_qty
                else:
                    new_avg_price = price
            else:  # SELL
                new_qty = current_qty - quantity
                # If this creates a new short position (current_qty was 0), set avg_price to trade price
                if current_qty == 0 and new_qty < 0:
                    new_avg_price = price
                else:
                    new_avg_price = current_avg_price

            logger.info(f"New position: {new_qty} @ {new_avg_price}")

            # Update or remove position
            if abs(new_qty) < 0.000001:
                # Position closed
                if portfolio_file.exists():
                    portfolio_file.unlink()
                logger.info(f"Position closed for {symbol}")
            else:
                # Update position
                new_position = {
                    'symbol': symbol,
                    'quantity': new_qty,
                    'avg_price': new_avg_price,
                    'last_updated': datetime.now().isoformat()
                }

                # Atomic write
                temp_file = portfolio_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(new_position, f, indent=2)
                temp_file.replace(portfolio_file)
                logger.info(f"Position updated for {symbol}: {new_qty} @ {new_avg_price}")

    def get_all_trades(self, limit: int = 1000) -> List[Dict]:
        """Get all trades sorted by timestamp"""
        trades = []

        for trade_file in sorted(self.trades_dir.glob("*.json"), reverse=True):
            if len(trades) >= limit:
                break

            try:
                with open(trade_file, 'r') as f:
                    trade_data = json.load(f)
                    trades.append(trade_data)
            except Exception as e:
                logger.warning(f"Failed to read trade file {trade_file}: {e}")

        return trades

    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        positions = []

        for position_file in self.portfolio_dir.glob("*.json"):
            try:
                with open(position_file, 'r') as f:
                    position_data = json.load(f)
                    if abs(position_data.get('quantity', 0)) > 0.000001:
                        positions.append(position_data)
            except Exception as e:
                logger.warning(f"Failed to read position file {position_file}: {e}")

        return positions

    def get_trade_count(self) -> int:
        """Get total number of trades"""
        return len(list(self.trades_dir.glob("*.json")))

    def get_symbol_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get trades for specific symbol"""
        trades = []

        for trade_data in self.get_all_trades(limit * 2):  # Get more to filter
            if trade_data.get('symbol') == symbol:
                trades.append(trade_data)
                if len(trades) >= limit:
                    break

        return trades

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from trade files"""
        trades = self.get_all_trades()

        if not trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_trade_pnl': 0
            }

        total_pnl = sum(trade.get('realized_pnl', 0) for trade in trades)
        winning_trades = sum(1 for trade in trades if trade.get('realized_pnl', 0) > 0)
        total_trades = len(trades)

        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades
        }

    def cleanup_old_trades(self, days_to_keep: int = 30):
        """Remove old trade files to save space"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        removed_count = 0
        for trade_file in self.trades_dir.glob("*.json"):
            if trade_file.stat().st_mtime < cutoff_time:
                trade_file.unlink()
                removed_count += 1

        logger.info(f"Cleaned up {removed_count} old trade files")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        trade_files = list(self.trades_dir.glob("*.json"))
        position_files = list(self.portfolio_dir.glob("*.json"))

        total_size = sum(f.stat().st_size for f in trade_files + position_files)

        return {
            'total_trades': len(trade_files),
            'total_positions': len(position_files),
            'storage_size_mb': total_size / (1024 * 1024),
            'storage_path': str(self.base_path),
            'avg_file_size_bytes': total_size / max(1, len(trade_files) + len(position_files))
        }

# Global instance
_lockless_storage: Optional[LocklessTradingStorage] = None

def get_lockless_storage(base_path: str = "trading_data") -> LocklessTradingStorage:
    """Get or create the global lockless storage instance"""
    global _lockless_storage

    if _lockless_storage is None:
        _lockless_storage = LocklessTradingStorage(base_path)

    return _lockless_storage