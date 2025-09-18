from typing import Dict, List

class BacktestingEngine:
    """Simple backtesting engine for strategy validation"""

    def __init__(self, strategy_func, initial_balance=10000):
        self.strategy_func = strategy_func
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def run_backtest(self, historical_data: List[Dict], symbol: str = "BTCUSDT"):
        """Run backtest on historical data"""

        for i, data_point in enumerate(historical_data):
            current_price = data_point['close']

            # Generate strategy signal
            signal = self.strategy_func(historical_data[:i+1], symbol)

            if signal:
                trade_result = self._execute_backtest_trade(signal, current_price, data_point['timestamp'])
                if trade_result:
                    self.trades.append(trade_result)

            # Update equity curve
            total_equity = self._calculate_total_equity(current_price)
            self.equity_curve.append({
                'timestamp': data_point['timestamp'],
                'equity': total_equity,
                'price': current_price
            })

        return self._generate_backtest_report()

    def _execute_backtest_trade(self, signal: Dict, current_price: float, timestamp: str):
        """Execute trade in backtest environment"""
        symbol = signal['symbol']
        action = signal['action']

        if action == 'buy' and symbol not in self.positions:
            # Calculate position size (simplified)
            position_value = self.balance * 0.1  # 10% of balance
            quantity = position_value / current_price

            if quantity * current_price <= self.balance:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': timestamp,
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit')
                }
                self.balance -= quantity * current_price

                return {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': current_price,
                    'status': 'open'
                }

        elif action == 'sell' and symbol in self.positions:
            position = self.positions[symbol]
            quantity = position['quantity']
            entry_price = position['entry_price']

            # Calculate PnL
            pnl = (current_price - entry_price) * quantity
            pnl_percent = pnl / (entry_price * quantity)

            self.balance += quantity * current_price
            del self.positions[symbol]

            return {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': current_price,
                'entry_price': entry_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'status': 'closed'
            }

        return None

    def _calculate_total_equity(self, current_price: float):
        """Calculate total equity including open positions"""
        total_equity = self.balance

        for symbol, position in self.positions.items():
            # Assume all positions are against current_price for simplicity
            position_value = position['quantity'] * current_price
            total_equity += position_value

        return total_equity

    def _generate_backtest_report(self):
        """Generate comprehensive backtest report"""
        closed_trades = [t for t in self.trades if t.get('status') == 'closed']

        if not closed_trades:
            return {'error': 'No closed trades found'}

        # Calculate metrics
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in closed_trades if t['pnl'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = (self.balance + sum(p['quantity'] * p['entry_price'] for p in self.positions.values()) - self.initial_balance) / self.initial_balance

        avg_win = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t['pnl'] for t in closed_trades if t['pnl'] < 0) / losing_trades if losing_trades > 0 else 0

        # Calculate maximum drawdown
        peak_equity = self.initial_balance
        max_drawdown = 0

        for point in self.equity_curve:
            if point['equity'] > peak_equity:
                peak_equity = point['equity']

            drawdown = (peak_equity - point['equity']) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'final_balance': self.balance,
            'equity_curve': self.equity_curve,
            'trades': closed_trades
        }