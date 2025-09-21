import { useState, useEffect, useCallback } from 'react'
import TradingAPI, { DashboardOverview } from '../lib/api'
import { PerformanceMetrics, Position, Trade, SystemHealth, ChartData } from '../types/trading'

interface UseTradingDataReturn {
  data: DashboardOverview | null
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
}

export function useTradingData(autoRefresh = true, interval = 30000): UseTradingDataReturn {
  const [data, setData] = useState<DashboardOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      setError(null)
      const overview = await TradingAPI.getDashboardOverview()
      setData(overview)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch trading data'
      setError(errorMessage)
      console.error('Error fetching trading data:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()

    if (autoRefresh) {
      const intervalId = setInterval(fetchData, interval)
      return () => clearInterval(intervalId)
    }
  }, [fetchData, autoRefresh, interval])

  return {
    data,
    loading,
    error,
    refetch: fetchData
  }
}

interface UseChartDataReturn {
  data: ChartData[]
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
}

export function usePnLChartData(days = 30): UseChartDataReturn {
  const [data, setData] = useState<ChartData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      setError(null)
      const chartData = await TradingAPI.getPnLChartData(days)
      setData(chartData)
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
  }, [fetchData])

  return {
    data,
    loading,
    error,
    refetch: fetchData
  }
}

interface UseSystemHealthReturn {
  health: SystemHealth | null
  loading: boolean
  error: string | null
  isOnline: boolean
}

export function useSystemHealth(): UseSystemHealthReturn {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isOnline, setIsOnline] = useState(false)

  const checkHealth = useCallback(async () => {
    try {
      setError(null)
      const healthData = await TradingAPI.getSystemHealth()
      setHealth(healthData)
      setIsOnline(true)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'System health check failed'
      setError(errorMessage)
      setIsOnline(false)
      console.error('Health check failed:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    checkHealth()

    // Check health every 10 seconds
    const intervalId = setInterval(checkHealth, 10000)
    return () => clearInterval(intervalId)
  }, [checkHealth])

  return {
    health,
    loading,
    error,
    isOnline
  }
}