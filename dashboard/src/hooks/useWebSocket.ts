import { useEffect, useRef, useState, useCallback } from 'react'
import { io, Socket } from 'socket.io-client'
import { DashboardOverview } from '../lib/api'

interface UseWebSocketReturn {
  isConnected: boolean
  lastUpdate: number
  sendMessage: (event: string, data?: any) => void
  forceRefresh: () => void
}

interface WebSocketHookOptions {
  autoConnect?: boolean
  reconnectAttempts?: number
  reconnectDelay?: number
}

const WS_URL = process.env.NODE_ENV === 'production'
  ? 'ws://trading-system:5002'
  : 'ws://localhost:5002'

export function useWebSocket(
  onDashboardUpdate?: (data: DashboardOverview) => void,
  onTradeAlert?: (data: any) => void,
  options: WebSocketHookOptions = {}
): UseWebSocketReturn {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectDelay = 1000
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(0)
  const socketRef = useRef<Socket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return
    }

    try {
      const socket = io(WS_URL, {
        transports: ['websocket', 'polling'],
        timeout: 10000,
        forceNew: true
      })

      socketRef.current = socket

      socket.on('connect', () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        reconnectCountRef.current = 0

        // Subscribe to all data types
        socket.emit('subscribe', { type: 'all' })
      })

      socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason)
        setIsConnected(false)

        // Auto-reconnect logic
        if (reason !== 'io client disconnect' && reconnectCountRef.current < reconnectAttempts) {
          const delay = reconnectDelay * Math.pow(2, reconnectCountRef.current)
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectCountRef.current++
            console.log(`Reconnecting... attempt ${reconnectCountRef.current}`)
            connect()
          }, delay)
        }
      })

      socket.on('dashboard_overview', (data: DashboardOverview) => {
        console.log('Received dashboard overview:', data)
        setLastUpdate(Date.now())
        onDashboardUpdate?.(data)
      })

      socket.on('dashboard_update', (data: DashboardOverview) => {
        console.log('Received dashboard update:', data)
        setLastUpdate(Date.now())
        onDashboardUpdate?.(data)
      })

      socket.on('performance_update', (data: { performance: any }) => {
        console.log('Received performance update:', data)
        setLastUpdate(Date.now())
        // Create partial dashboard update
        onDashboardUpdate?.({
          performance: data.performance,
          positions: [],
          system_health: {
            status: 'HEALTHY',
            uptime: 0,
            api_latency: 0,
            error_rate: 0,
            last_trade_time: new Date().toISOString(),
            memory_usage: 0,
            cpu_usage: 0
          },
          recent_trades: [],
          account_balance: {
            balances: [],
            total_value_usdt: 0,
            account_type: 'SPOT',
            can_trade: false,
            can_withdraw: false,
            can_deposit: false,
            last_updated: new Date().toISOString()
          }
        })
      })

      socket.on('positions_update', (data: { positions: any[] }) => {
        console.log('Received positions update:', data)
        setLastUpdate(Date.now())
        // Handle positions update
      })

      socket.on('trade_alert', (data: any) => {
        console.log('Received trade alert:', data)
        onTradeAlert?.(data)

        // Show notification
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('New Trade Executed', {
            body: `${data.trade?.side} ${data.trade?.quantity} ${data.trade?.symbol} @ ${data.trade?.price}`,
            icon: '/favicon.ico'
          })
        }
      })

      socket.on('refresh_complete', (data: { status: string, message?: string }) => {
        console.log('Refresh complete:', data)
        if (data.status === 'error') {
          console.error('Refresh failed:', data.message)
        }
      })

      socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error)
        setIsConnected(false)
      })

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setIsConnected(false)
    }
  }, [WS_URL, onDashboardUpdate, onTradeAlert, reconnectAttempts, reconnectDelay])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
    }

    setIsConnected(false)
  }, [])

  const sendMessage = useCallback((event: string, data?: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data)
    } else {
      console.warn('WebSocket not connected, cannot send message:', event)
    }
  }, [])

  const forceRefresh = useCallback(() => {
    sendMessage('request_refresh')
  }, [sendMessage])

  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }

    return () => {
      disconnect()
    }
  }, [autoConnect, connect, disconnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [])

  return {
    isConnected,
    lastUpdate,
    sendMessage,
    forceRefresh
  }
}