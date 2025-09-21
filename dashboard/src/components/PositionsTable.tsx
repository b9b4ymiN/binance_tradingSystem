import { Position } from '../types/trading'
import { Card, CardHeader, CardTitle, CardContent } from './Card'
import { formatCurrency, formatPercent, getPercentageColor, cn } from '../lib/utils'

interface PositionsTableProps {
  positions: Position[]
  loading?: boolean
}

export function PositionsTable({ positions, loading = false }: PositionsTableProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Current Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="animate-pulse flex space-x-4">
                <div className="h-4 bg-gray-200 rounded w-20"></div>
                <div className="h-4 bg-gray-200 rounded w-16"></div>
                <div className="h-4 bg-gray-200 rounded w-24"></div>
                <div className="h-4 bg-gray-200 rounded w-20"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (positions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Current Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">📊</div>
            <p>No open positions</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Current Positions</CardTitle>
      </CardHeader>
      <CardContent  >
        <div className="overflow-x-auto">
          <table className="w-full text-gray-700">
            <thead>
              <tr className="border-b border-gray-100">
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Symbol</th>
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Side</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Size</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Invest</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Entry Price</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Current Price</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Mkt.Value</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">P&L</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">%</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position, index) => (
                <tr
                  key={`${position.symbol}-${index}`}
                  className="border-b border-gray-50 hover:bg-gray-50/50 transition-colors"
                >
                  <td className="py-4 px-6">
                    <div className="font-medium text-gray-900">{position.symbol}</div>
                  </td>
                  <td className="py-4 px-6">
                    <span
                      className={cn(
                        'inline-flex px-2 py-1 rounded-full text-xs font-medium',
                        position.side === 'LONG'
                          ? 'bg-success-100 text-success-700'
                          : 'bg-danger-100 text-danger-700'
                      )}
                    >
                      {position.side}
                    </span>
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm">
                    {Math.abs(position.quantity) ?? '-'}
                  </td>
                   <td className="py-4 px-6 text-right font-mono text-sm">
                    {formatCurrency(position.position_size_usd) ?? '-'}
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm">
                    {formatCurrency(position.entry_price)}
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm">
                    {formatCurrency(position.current_price)}
                  </td>
                   <td className="py-4 px-6 text-right font-mono text-sm">
                    {formatCurrency(position.market_value)}
                  </td>
                  <td className={cn(
                    'py-4 px-6 text-right font-mono text-sm font-medium',
                    getPercentageColor(position.unrealized_pnl)
                  )}>
                    {formatCurrency(position.unrealized_pnl)}
                  </td>
                  <td className={cn(
                    'py-4 px-6 text-right font-mono text-sm font-medium',
                    getPercentageColor(position.unrealized_pnl)
                  )}>
                    {formatPercent(position.unrealized_pnl_percent)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}