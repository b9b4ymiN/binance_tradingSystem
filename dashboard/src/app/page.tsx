'use client'

import { MetricCard } from '../components/MetricCard'
import { PositionsTable } from '../components/PositionsTable'
import { TradesTable } from '../components/TradesTable'
import { PnLChart } from '../components/PnLChart'
import { SystemHealth } from '../components/SystemHealth'
import { BalanceTable } from '../components/BalanceTable'
import { useOptimizedTradingData, useOptimizedChartData, useOptimizedSystemHealth } from '../hooks/useOptimizedTradingData'

export default function DashboardPage() {
  // Use optimized data hooks with caching and smart polling
  const { data: tradingData, loading: tradingLoading, error: tradingError, lastUpdate } = useOptimizedTradingData(true, 60000, true)
  const { data: chartData, loading: chartLoading } = useOptimizedChartData(30)
  const { health, loading: healthLoading, isOnline } = useOptimizedSystemHealth()

  // Extract data with fallbacks
  const performance = tradingData?.performance || {
    total_pnl: 0,
    daily_pnl: 0,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    win_rate: 0,
    avg_win: 0,
    avg_loss: 0,
    max_drawdown: 0,
    sharpe_ratio: 0,
    portfolio_value: 0,
    available_balance: 0
  }

  const positions = tradingData?.positions || []
  const trades = tradingData?.recent_trades || []
  const accountBalance = tradingData?.account_balance || null
  const systemHealth = health || {
    status: 'HEALTHY' as const,
    uptime: 0,
    api_latency: 0,
    error_rate: 0,
    last_trade_time: new Date().toISOString(),
    memory_usage: 0,
    cpu_usage: 0
  }

  const loading = tradingLoading && chartLoading && healthLoading

  return (
    <div className="space-y-6">
      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total P&L"
          value={performance.total_pnl}
          change={performance.daily_pnl}
          changeType="currency"
          color={performance.total_pnl >= 0 ? 'success' : 'danger'}
          loading={loading}
          icon={<span className="text-2xl">💰</span>}
        />
        <MetricCard
          title="Win Rate"
          value={`${performance.win_rate}%`}
          change={5.2}
          changeType="percentage"
          color={performance.win_rate >= 50 ? 'success' : 'warning'}
          loading={loading}
          icon={<span className="text-2xl">🎯</span>}
        />
        <MetricCard
          title="Portfolio Value"
          value={performance.portfolio_value}
          change={performance.daily_pnl / performance.portfolio_value * 100}
          changeType="percentage"
          loading={loading}
          icon={<span className="text-2xl">📊</span>}
        />
        <MetricCard
          title="Total Trades"
          value={performance.total_trades ?? '0'}
          change={12}
          changeType="number"
          loading={loading}
          icon={<span className="text-2xl">⚡</span>}
        />
      </div>

      {/* Charts and System Health */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <PnLChart data={chartData} loading={chartLoading} height={350} />
        </div>
        <div>
          <SystemHealth health={systemHealth} loading={healthLoading} />
        </div>
      </div>

      {/* Error State */}
      {tradingError && (
        <div className="bg-danger-50 border border-danger-200 rounded-lg p-4">
          <div className="flex">
            <div className="text-danger-400">⚠️</div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-danger-800">Connection Error</h3>
              <p className="text-sm text-danger-700 mt-1">{tradingError}</p>
            </div>
          </div>
        </div>
      )}

      {/* Account Balance */}
      <BalanceTable balance={accountBalance} loading={tradingLoading} />

      {/* Positions */}
      <PositionsTable positions={positions} loading={tradingLoading} />

      {/* Recent Trades */}
      <TradesTable trades={trades} loading={tradingLoading} />
    </div>
  )
}