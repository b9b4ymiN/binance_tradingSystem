import { SystemHealth as SystemHealthType } from '../types/trading'
import { Card, CardHeader, CardTitle, CardContent } from './Card'
import { cn } from '../lib/utils'

interface SystemHealthProps {
  health: SystemHealthType
  loading?: boolean
}

export function SystemHealth({ health, loading = false }: SystemHealthProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Health</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="animate-pulse flex justify-between">
                <div className="h-4 bg-gray-200 rounded w-24"></div>
                <div className="h-4 bg-gray-200 rounded w-16"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'HEALTHY':
        return 'text-success-600 bg-success-100'
      case 'WARNING':
        return 'text-warning-600 bg-warning-100'
      case 'ERROR':
        return 'text-danger-600 bg-danger-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'HEALTHY':
        return '✅'
      case 'WARNING':
        return '⚠️'
      case 'ERROR':
        return '❌'
      default:
        return '❓'
    }
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between ">
          System Health
          <span className={cn(
            'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
            getStatusColor(health.status)
          )}>
            <span className="mr-1">{getStatusIcon(health.status)}</span>
            {health.status}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Uptime</p>
              <p className="text-lg font-semibold text-gray-800">{formatUptime(health.uptime)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">API Latency</p>
              <p className="text-lg font-semibold text-gray-800">{health.api_latency}ms</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Error Rate</p>
              <p className="text-lg font-semibold text-gray-800">{(health.error_rate * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Memory Usage</p>
              <p className="text-lg font-semibold text-gray-800">{health.memory_usage}%</p>
            </div>
          </div>

          <div>
            <p className="text-sm text-gray-600">CPU Usage</p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className={cn(
                  'h-2 rounded-full transition-all duration-300',
                  health.cpu_usage > 80 ? 'bg-danger-500' :
                  health.cpu_usage > 60 ? 'bg-warning-500' : 'bg-success-500'
                )}
                style={{ width: `${health.cpu_usage}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600 mt-1">{health.cpu_usage}%</p>
          </div>

          <div>
            <p className="text-sm text-gray-600">Last Trade</p>
            <p className="text-sm font-medium text-gray-300">
              {new Date(health.last_trade_time).toLocaleString()}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}