import { Trade } from '../types/trading'
import { Card, CardHeader, CardTitle, CardContent } from './Card'
import { formatCurrency, getPercentageColor, cn } from '../lib/utils'

interface TradesTableProps {
  trades: Trade[]
  loading?: boolean
  title?: string
  showPnL?: boolean
}

export function TradesTable({
  trades,
  loading = false,
  title = 'Recent Trades',
  showPnL = true
}: TradesTableProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
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

  if (trades.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">📈</div>
            <p>No trades found</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-100">
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Time</th>
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Symbol</th>
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Side</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Quantity</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Price</th>
                {showPnL && (
                  <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">P&L</th>
                )}
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Strategy</th>
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => (
                <tr
                  key={trade.id}
                  className="border-b border-gray-50 hover:bg-gray-50/50 transition-colors"
                >
                  <td className="py-4 px-6 text-sm text-gray-600">
                    {new Date(trade.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="py-4 px-6">
                    <div className="font-medium text-gray-900">{trade.symbol}</div>
                  </td>
                  <td className="py-4 px-6">
                    <span
                      className={cn(
                        'inline-flex px-2 py-1 rounded-full text-xs font-medium',
                        trade.side === 'BUY'
                          ? 'bg-success-100 text-success-700'
                          : 'bg-danger-100 text-danger-700'
                      )}
                    >
                      {trade.side}
                    </span>
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm text-gray-600">
                    {trade.quantity.toFixed(6)}
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm text-gray-600">
                    {formatCurrency(trade.price)}
                  </td>
                  {showPnL && (
                    <td className={cn(
                      'py-4 px-6 text-right font-mono text-sm font-medium',
                      trade.pnl ? getPercentageColor(trade.pnl) : 'text-gray-600'
                    )}>
                      {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                    </td>
                  )}
                  <td className="py-4 px-6 text-sm text-gray-600">
                    <span className="inline-flex px-2 py-1 rounded bg-gray-100 text-gray-700 text-xs">
                      {trade.strategy}
                    </span>
                  </td>
                  <td className="py-4 px-6">
                    <span
                      className={cn(
                        'inline-flex px-2 py-1 rounded-full text-xs font-medium',
                        trade.status === 'FILLED'
                          ? 'bg-success-100 text-success-700'
                          : trade.status === 'PENDING'
                          ? 'bg-warning-100 text-warning-700'
                          : 'bg-gray-100 text-gray-700'
                      )}
                    >
                      {trade.status}
                    </span>
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