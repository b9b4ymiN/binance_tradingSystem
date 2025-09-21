import { Card, CardHeader, CardTitle, CardContent } from './Card'
import { formatCurrency, cn } from '../lib/utils'

interface Balance {
  asset: string
  free: number
  locked: number
  total: number
  usd_value: number
}

interface AccountBalance {
  balances: Balance[]
  total_value_usdt: number
  account_type: string
  can_trade: boolean
  can_withdraw: boolean
  can_deposit: boolean
  last_updated: string
}

interface BalanceTableProps {
  balance: AccountBalance | null
  loading?: boolean
}

export function BalanceTable({ balance, loading = false }: BalanceTableProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Account Balance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="animate-pulse flex space-x-4">
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

  if (!balance || balance.balances.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Account Balance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">💰</div>
            <p>No balance data available</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const getAssetIcon = (asset: string) => {
    const icons: { [key: string]: string } = {
      'BTC': '₿',
      'ETH': 'Ξ',
      'USDT': '$',
      'ADA': '₳',
      'BNB': '🟡',
      'DOT': '●',
      'SOL': '◉'
    }
    return icons[asset] || '●'
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          Account Balance
          <div className="text-right">
            <div className="text-sm text-gray-600">Total Value</div>
            <div className="text-lg font-bold text-green-600">
              {formatCurrency(balance.total_value_usdt)}
            </div>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent >
        <div className="space-y-2 p-4">
          {/* Account Status */}
          <div className="flex items-center space-x-4 pb-4 border-b border-gray-100">
            <div className="flex items-center space-x-2">
              <span className={cn(
                'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
                balance.can_trade ? 'bg-success-100 text-success-700' : 'bg-danger-100 text-danger-700'
              )}>
                {balance.can_trade ? '✅ Trading' : '❌ No Trading'}
              </span>
              <span className="text-xs text-gray-500">{balance.account_type}</span>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-gray-700">
            <thead>
              <tr className="border-b border-gray-100">
                <th className="text-left py-3 px-6 text-sm font-medium text-gray-600">Asset</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Free</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Locked</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">Total</th>
                <th className="text-right py-3 px-6 text-sm font-medium text-gray-600">USD Value</th>
              </tr>
            </thead>
            <tbody>
              {balance.balances.filter(item=> item.usd_value !== 0).map((bal) => (
                <tr
                  key={bal.asset}
                  className="border-b border-gray-50 hover:bg-gray-50/50 transition-colors"
                >
                  <td className="py-4 px-6">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getAssetIcon(bal.asset)}</span>
                      <div>
                        <div className="font-medium text-gray-900">{bal.asset}</div>
                      </div>
                    </div>
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm">
                    {bal.free.toFixed(8)}
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm">
                    <span className={cn(
                      bal.locked > 0 ? 'text-warning-600' : 'text-gray-400'
                    )}>
                      {bal.locked.toFixed(8)}
                    </span>
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm font-medium">
                    {bal.total.toFixed(8)}
                  </td>
                  <td className="py-4 px-6 text-right font-mono text-sm font-medium text-green-600">
                    {formatCurrency(bal.usd_value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="p-4 bg-gray-50 text-xs text-gray-500">
          Last updated: {new Date(balance.last_updated).toLocaleString()}
        </div>
      </CardContent>
    </Card>
  )
}