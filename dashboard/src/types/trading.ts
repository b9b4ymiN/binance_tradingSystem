export interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  commission: number;
  timestamp: string;
  strategy: string;
  pnl?: number;
  status: 'FILLED' | 'PENDING' | 'CANCELLED';
}

export interface Position {
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  quantity: number;
  position_size_usd: number;
  market_value: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  realized_pnl: number;
  percentage: number;
  value: number;
}

export interface PerformanceMetrics {
  total_pnl: number;
  daily_pnl: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  max_drawdown: number;
  sharpe_ratio: number;
  portfolio_value: number;
  available_balance: number;
}

export interface StrategyPerformance {
  name: string;
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  avg_trade_duration: number;
  max_drawdown: number;
  active: boolean;
}

export interface SystemHealth {
  status: 'HEALTHY' | 'WARNING' | 'ERROR';
  uptime: number;
  api_latency: number;
  error_rate: number;
  last_trade_time: string;
  memory_usage: number;
  cpu_usage: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  change_24h: number;
  change_percent_24h: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
}

export interface Alert {
  id: string;
  type: 'SUCCESS' | 'WARNING' | 'ERROR' | 'INFO';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface ChartData {
  timestamp: string;
  value: number;
  label?: string;
}

export interface Balance {
  asset: string;
  free: number;
  locked: number;
  total: number;
  usd_value: number;
}

export interface AccountBalance {
  balances: Balance[];
  total_value_usdt: number;
  account_type: string;
  can_trade: boolean;
  can_withdraw: boolean;
  can_deposit: boolean;
  last_updated: string;
}