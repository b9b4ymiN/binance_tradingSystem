import { useState, useEffect, useCallback, useRef } from 'react'
import TradingAPI, { DashboardOverview } from '../lib/api'
import { PerformanceMetrics, Position, Trade, SystemHealth, ChartData, AccountBalance } from '../types/trading'

// Cache management
interface CacheEntry<T> {
  data: T
  timestamp: number
  expires: number
}

class DataCache {
  private cache = new Map<string, CacheEntry<any>>()

  set<T>(key: string, data: T, ttlMs: number = 30000): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      expires: Date.now() + ttlMs
    })
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key)
    if (!entry || Date.now() > entry.expires) {
      this.cache.delete(key)
      return null
    }
    return entry.data
  }

  clear(): void {
    this.cache.clear()
  }

  invalidate(pattern: string): void {
    for (const key of this.cache.keys()) {
      if (key.includes(pattern)) {
        this.cache.delete(key)
      }
    }
  }
}

const cache = new DataCache()

interface UseOptimizedTradingDataReturn {
  data: DashboardOverview | null
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
  lastUpdate: number
}

export function useOptimizedTradingData(
  autoRefresh = true,
  interval = 60000,
  enableCache = true
): UseOptimizedTradingDataReturn {
  const [data, setData] = useState<DashboardOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState(0)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const fetchData = useCallback(async (force = false) => {
    // Check cache first
    if (enableCache && !force) {
      const cachedData = cache.get<DashboardOverview>('dashboard-overview')
      if (cachedData) {
        setData(cachedData)
        setLoading(false)
        return
      }
    }

    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()

    try {
      setError(null)

      // Fetch only essential data frequently, others less frequently
      const now = Date.now()
      const shouldFetchAll = !lastUpdate || (now - lastUpdate) > 60000 // Full refresh every minute

      let overview: DashboardOverview

      if (shouldFetchAll) {
        // Full data fetch
        overview = await TradingAPI.getDashboardOverview()
        setLastUpdate(now)

        // Cache individual components with different TTLs
        if (enableCache) {
          cache.set('dashboard-overview', overview, 250000) // 250s cache
          cache.set('positions', overview.positions, 200000) // 200s cache for positions
          cache.set('performance', overview.performance, 300000) // 300s cache for performance
          cache.set('account-balance', overview.account_balance, 450000) // 450s cache for balance
        }
      } else {
        // Incremental update - only fetch frequently changing data
        const [positions, performance] = await Promise.all([
          TradingAPI.getPositions(),
          TradingAPI.getPerformanceMetrics()
        ])

        // Use cached data for less frequently changing items
        const cachedBalance = cache.get<AccountBalance>('account-balance')
        const cachedHealth = cache.get<SystemHealth>('system-health')
        const cachedTrades = cache.get<Trade[]>('recent-trades')

        overview = {
          positions,
          performance,
          account_balance: cachedBalance || data?.account_balance || {
            balances: [],
            total_value_usdt: 0,
            account_type: 'SPOT',
            can_trade: false,
            can_withdraw: false,
            can_deposit: false,
            last_updated: new Date().toISOString()
          },
          system_health: cachedHealth || data?.system_health || {
            status: 'HEALTHY' as const,
            uptime: 0,
            api_latency: 0,
            error_rate: 0,
            last_trade_time: new Date().toISOString(),
            memory_usage: 0,
            cpu_usage: 0
          },
          recent_trades: cachedTrades || data?.recent_trades || []
        }

        if (enableCache) {
          cache.set('positions', positions, 200000)
          cache.set('performance', performance, 300000)
        }
      }

      setData(overview)
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return // Request was cancelled, ignore
      }

      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch trading data'
      setError(errorMessage)
      console.error('Error fetching trading data:', err)
    } finally {
      setLoading(false)
    }
  }, [enableCache, lastUpdate, data])

  useEffect(() => {
    fetchData()

    if (autoRefresh) {
      intervalRef.current = setInterval(() => fetchData(), interval)
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [fetchData, autoRefresh, interval])

  // Clear cache when component unmounts
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  return {
    data,
    loading,
    error,
    refetch: () => fetchData(true),
    lastUpdate
  }
}

// Optimized hook for chart data with longer cache
export function useOptimizedChartData(days = 30): {
  data: ChartData[]
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
} {
  const [data, setData] = useState<ChartData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    const cacheKey = `chart-data-${days}`

    // Check cache first (5 minute cache for chart data)
    const cachedData = cache.get<ChartData[]>(cacheKey)
    if (cachedData) {
      setData(cachedData)
      setLoading(false)
      return
    }

    try {
      setError(null)
      const chartData = await TradingAPI.getPnLChartData(days)
      setData(chartData)

      // Cache for 5 minutes (chart data doesn't change as frequently)
      cache.set(cacheKey, chartData, 300000)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch chart data'
      setError(errorMessage)
      console.error('Error fetching chart data:', err)
    } finally {
      setLoading(false)
    }
  }, [days])

  useEffect(() => {
    fetchData()

    // Refresh chart data every 1 minute
    const interval = setInterval(fetchData, 60000)
    return () => clearInterval(interval)
  }, [fetchData])

  return {
    data,
    loading,
    error,
    refetch: fetchData
  }
}

// Optimized system health with adaptive polling
export function useOptimizedSystemHealth(): {
  health: SystemHealth | null
  loading: boolean
  error: string | null
  isOnline: boolean
} {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isOnline, setIsOnline] = useState(false)
  const [pollInterval, setPollInterval] = useState(10000) // Adaptive polling

  const checkHealth = useCallback(async () => {
    try {
      setError(null)
      const healthData = await TradingAPI.getSystemHealth()
      setHealth(healthData)
      setIsOnline(true)

      // Reduce polling frequency if system is healthy
      if (healthData.status === 'HEALTHY') {
        setPollInterval(150000) // Poll every 150s when healthy
      } else {
        setPollInterval(50000) // Poll every 50s when issues detected
      }

      // Cache health data
      cache.set('system-health', healthData, 120000)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'System health check failed'
      setError(errorMessage)
      setIsOnline(false)
      setPollInterval(5000) // Poll more frequently when offline
      console.error('Health check failed:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    checkHealth()

    const intervalId = setInterval(checkHealth, pollInterval)
    return () => clearInterval(intervalId)
  }, [checkHealth, pollInterval])

  return {
    health,
    loading,
    error,
    isOnline
  }
}