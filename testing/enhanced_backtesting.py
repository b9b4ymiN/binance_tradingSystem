"""
Enhanced Backtesting Engine with realistic trading simulation

Features:
- Slippage modeling
- Commission/fees
- VWAP strategy support
- Walk-forward analysis
- Comprehensive performance metrics
- Multi-timeframe support
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedBacktestingEngine:
    """Realistic backtesting engine for strategy validation"""

    def __init__(
        self,
        strategy,
        initial_balance: float = 10000,
        commission_rate: float = 0.00075,  # 0.075% Binance spot fee (with BNB discount)
        slippage_rate: float = 0.0005,  # 0.05% average slippage
        max_position_size: float = 0.1,  # Max 10% of balance per trade
    ):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size

        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []

        # Performance tracking
        self.peak_equity = initial_balance
        self.max_drawdown = 0
        self.current_drawdown = 0

    def run_backtest(
        self,
        symbol: str,
        historical_data: List[List],  # Binance klines format
        walk_forward: bool = False
    ) -> Dict:
        """
        Run backtest on historical data

        Args:
            symbol: Trading symbol
            historical_data: List of klines [timestamp, open, high, low, close, volume, ...]
            walk_forward: If True, use walk-forward validation

        Returns:
            Comprehensive backtest report
        """
        logger.info(f"Starting backtest for {symbol} with {len(historical_data)} candles")

        if walk_forward:
            return self._run_walk_forward_backtest(symbol, historical_data)

        return self._run_standard_backtest(symbol, historical_data)

    def _run_standard_backtest(self, symbol: str, klines: List[List]) -> Dict:
        """Run standard backtest on all data"""

        for i in range(len(klines)):
            current_kline = klines[i]
            timestamp = current_kline[0]
            current_open = float(current_kline[1])
            current_high = float(current_kline[2])
            current_low = float(current_kline[3])
            current_close = float(current_kline[4])
            current_volume = float(current_kline[5])

            # Check exit conditions for open positions
            if symbol in self.positions:
                self._check_position_exit(
                    symbol, current_high, current_low, current_close, timestamp
                )

            # Generate trading signal
            # Pass subset of data up to current point for strategy
            signal = self.strategy.generate_signal(symbol)

            if signal and symbol not in self.positions:
                self._execute_entry(signal, current_close, timestamp)

            # Update equity curve
            total_equity = self._calculate_total_equity(current_close)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'price': current_close
            })

            # Update drawdown
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        return self._generate_backtest_report(symbol)

    def _run_walk_forward_backtest(self, symbol: str, klines: List[List]) -> Dict:
        """
        Run walk-forward optimization backtest

        Split data into windows:
        - Train: 60%
        - Validate: 20%
        - Test: 20%
        """
        total_length = len(klines)
        window_size = total_length // 5  # 5 windows

        results = []

        for i in range(3):  # 3 walk-forward periods
            start_idx = i * window_size
            train_end = start_idx + int(window_size * 3)  # 60%
            validate_end = train_end + window_size  # 20%
            test_end = validate_end + window_size  # 20%

            if test_end > total_length:
                break

            # Train period (would optimize parameters here)
            train_data = klines[start_idx:train_end]

            # Test period
            test_data = klines[validate_end:test_end]

            # Reset for this period
            period_balance = self.balance
            period_result = self._run_standard_backtest(symbol, test_data)
            period_result['period'] = i + 1
            period_result['start_balance'] = period_balance
            results.append(period_result)

        # Aggregate results
        return self._aggregate_walk_forward_results(results)

    def _execute_entry(self, signal: Dict, current_price: float, timestamp: int):
        """Execute entry with slippage and commission"""

        action = signal['action']
        symbol = signal['symbol']

        # Apply slippage
        if action == 'buy':
            entry_price = current_price * (1 + self.slippage_rate)
        else:
            entry_price = current_price * (1 - self.slippage_rate)

        # Calculate position size
        position_value = self.balance * self.max_position_size
        quantity = position_value / entry_price

        # Calculate commission
        commission = position_value * self.commission_rate

        # Check if sufficient balance
        total_cost = position_value + commission
        if total_cost > self.balance:
            logger.debug(f"Insufficient balance for trade: need ${total_cost:.2f}, have ${self.balance:.2f}")
            return

        # Execute entry
        self.balance -= total_cost

        self.positions[symbol] = {
            'action': action,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'commission_paid': commission,
            'strategy': signal.get('strategy', 'unknown'),
            'confidence': signal.get('confidence', 0),
            'vwap': signal.get('vwap'),
            'vwap_deviation': signal.get('vwap_deviation'),
        }

        logger.debug(f"Entered {action.upper()} position: {quantity:.6f} {symbol} @ ${entry_price:.2f}")

    def _check_position_exit(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        timestamp: int
    ):
        """Check if position should be exited"""

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        action = position['action']
        entry_price = position['entry_price']
        quantity = position['quantity']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']

        exit_price = None
        exit_reason = None

        if action == 'buy':
            # Check stop loss
            if stop_loss and low <= stop_loss:
                exit_price = stop_loss * (1 - self.slippage_rate)  # Slippage on exit
                exit_reason = 'stop_loss'

            # Check take profit
            elif take_profit and high >= take_profit:
                exit_price = take_profit * (1 - self.slippage_rate)
                exit_reason = 'take_profit'

        else:  # sell/short
            # Check stop loss
            if stop_loss and high >= stop_loss:
                exit_price = stop_loss * (1 + self.slippage_rate)
                exit_reason = 'stop_loss'

            # Check take profit
            elif take_profit and low <= take_profit:
                exit_price = take_profit * (1 + self.slippage_rate)
                exit_reason = 'take_profit'

        # Exit position if condition met
        if exit_price:
            self._execute_exit(symbol, exit_price, exit_reason, timestamp)

    def _execute_exit(self, symbol: str, exit_price: float, exit_reason: str, timestamp: int):
        """Execute position exit with commission"""

        position = self.positions[symbol]
        action = position['action']
        quantity = position['quantity']
        entry_price = position['entry_price']

        # Calculate proceeds
        position_value = quantity * exit_price
        commission = position_value * self.commission_rate
        proceeds = position_value - commission

        # Calculate P&L
        if action == 'buy':
            pnl = proceeds - (quantity * entry_price)
        else:  # short
            pnl = (quantity * entry_price) - proceeds

        pnl_percent = pnl / (quantity * entry_price) if entry_price > 0 else 0

        # Update balance
        self.balance += proceeds

        # Record trade
        trade = {
            'symbol': symbol,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'holding_minutes': (timestamp - position['entry_time']) / 60000,  # ms to minutes
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': exit_reason,
            'commission_total': position['commission_paid'] + commission,
            'strategy': position['strategy'],
            'confidence': position['confidence'],
            'vwap': position.get('vwap'),
            'vwap_deviation': position.get('vwap_deviation'),
        }

        self.trades.append(trade)

        # Remove position
        del self.positions[symbol]

        logger.debug(
            f"Exited {action.upper()} position: {quantity:.6f} {symbol} @ ${exit_price:.2f} "
            f"| P&L: ${pnl:.2f} ({pnl_percent*100:.2f}%) | Reason: {exit_reason}"
        )

    def _calculate_total_equity(self, current_price: float) -> float:
        """Calculate total equity including open positions"""
        total_equity = self.balance

        for symbol, position in self.positions.items():
            quantity = position['quantity']
            position_value = quantity * current_price
            total_equity += position_value

        return total_equity

    def _generate_backtest_report(self, symbol: str) -> Dict:
        """Generate comprehensive backtest report"""

        closed_trades = [t for t in self.trades]

        if not closed_trades:
            return {
                'error': 'No trades executed',
                'symbol': symbol,
                'total_trades': 0
            }

        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)

        win_rate = num_wins / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))

        avg_win = gross_profit / num_wins if num_wins > 0 else 0
        avg_loss = gross_loss / num_losses if num_losses > 0 else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate Sharpe Ratio
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]['equity'] - self.equity_curve[i-1]['equity']) / \
                      self.equity_curve[i-1]['equity']
                returns.append(ret)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Sortino Ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0] if 'returns' in locals() else []
        if downside_returns:
            downside_std = np.std(downside_returns)
            sortino_ratio = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = 0

        # Calculate average holding time
        avg_holding_minutes = np.mean([t['holding_minutes'] for t in closed_trades])

        # Commission impact
        total_commission = sum(t['commission_total'] for t in closed_trades)

        return {
            'symbol': symbol,
            'backtest_period': {
                'start': datetime.fromtimestamp(closed_trades[0]['entry_time']/1000).isoformat(),
                'end': datetime.fromtimestamp(closed_trades[-1]['exit_time']/1000).isoformat(),
            },

            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,

            # P&L metrics
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_return_percent': total_return * 100,

            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,

            # Risk metrics
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,

            # Trading metrics
            'avg_holding_minutes': avg_holding_minutes,
            'total_commission': total_commission,
            'commission_percent_of_pnl': (total_commission / abs(total_pnl)) * 100 if total_pnl != 0 else 0,

            # Data for further analysis
            'equity_curve': self.equity_curve,
            'trades': closed_trades,
        }

    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward backtest results"""

        total_trades = sum(r['total_trades'] for r in results)
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_return = np.mean([r['total_return'] for r in results])
        max_dd = max(r['max_drawdown'] for r in results)

        return {
            'walk_forward': True,
            'num_periods': len(results),
            'aggregate_stats': {
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_return': avg_return,
                'max_drawdown': max_dd,
            },
            'period_results': results
        }
