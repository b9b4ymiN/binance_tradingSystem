"""
WebSocket handler for real-time dashboard updates
"""

import json
import logging
import asyncio
from typing import Set, Dict, Any
from flask import Flask
from flask_socketio import SocketIO, emit, disconnect
from threading import Thread
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class DashboardWebSocket:
    """WebSocket handler for real-time dashboard updates"""

    def __init__(self, app: Flask, trading_engine, db_manager):
        self.app = app
        self.trading_engine = trading_engine
        self.db_manager = db_manager
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
        self.active_connections: Set[str] = set()
        self.last_data_hash: Dict[str, str] = {}
        self.setup_websocket_handlers()
        self.start_broadcast_thread()

    def setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""

        @self.socketio.on('connect')
        def handle_connect():
            sid = self.socketio.server.eio.get_session(None).sid
            self.active_connections.add(sid)
            logger.info(f"Client connected: {sid}")

            # Send initial data immediately
            try:
                self.send_initial_data(sid)
            except Exception as e:
                logger.error(f"Error sending initial data: {e}")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            sid = self.socketio.server.eio.get_session(None).sid
            self.active_connections.discard(sid)
            logger.info(f"Client disconnected: {sid}")

        @self.socketio.on('request_refresh')
        def handle_refresh_request():
            """Handle client request for data refresh"""
            try:
                # Force update performance and portfolio
                self.trading_engine._update_daily_performance()
                self.trading_engine._sync_portfolio_to_database()

                # Broadcast updated data
                self.broadcast_dashboard_update()

                emit('refresh_complete', {'status': 'success'})
            except Exception as e:
                logger.error(f"Error handling refresh request: {e}")
                emit('refresh_complete', {'status': 'error', 'message': str(e)})

        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to specific data types"""
            sid = self.socketio.server.eio.get_session(None).sid
            subscription_type = data.get('type', 'all')
            logger.info(f"Client {sid} subscribed to: {subscription_type}")

            # Send relevant data based on subscription
            if subscription_type in ['all', 'performance']:
                self.send_performance_data(sid)
            if subscription_type in ['all', 'positions']:
                self.send_position_data(sid)
            if subscription_type in ['all', 'trades']:
                self.send_trade_data(sid)

    def send_initial_data(self, sid: str):
        """Send initial dashboard data to connected client"""
        try:
            # Get dashboard overview data
            from .dashboard_api import DashboardAPI
            dashboard_api = DashboardAPI(self.trading_engine, self.db_manager)

            overview_data = {
                'performance': dashboard_api._get_performance_metrics(),
                'positions': dashboard_api._get_current_positions(),
                'system_health': dashboard_api._get_system_health(),
                'account_balance': dashboard_api._get_account_balance(),
                'timestamp': datetime.now().isoformat()
            }

            self.socketio.emit('dashboard_overview', overview_data, room=sid)

        except Exception as e:
            logger.error(f"Error sending initial data to {sid}: {e}")

    def send_performance_data(self, sid: str):
        """Send performance metrics to specific client"""
        try:
            from .dashboard_api import DashboardAPI
            dashboard_api = DashboardAPI(self.trading_engine, self.db_manager)

            performance_data = {
                'performance': dashboard_api._get_performance_metrics(),
                'timestamp': datetime.now().isoformat()
            }

            self.socketio.emit('performance_update', performance_data, room=sid)

        except Exception as e:
            logger.error(f"Error sending performance data to {sid}: {e}")

    def send_position_data(self, sid: str):
        """Send position data to specific client"""
        try:
            from .dashboard_api import DashboardAPI
            dashboard_api = DashboardAPI(self.trading_engine, self.db_manager)

            position_data = {
                'positions': dashboard_api._get_current_positions(),
                'timestamp': datetime.now().isoformat()
            }

            self.socketio.emit('positions_update', position_data, room=sid)

        except Exception as e:
            logger.error(f"Error sending position data to {sid}: {e}")

    def send_trade_data(self, sid: str):
        """Send recent trades to specific client"""
        try:
            from .dashboard_api import DashboardAPI
            dashboard_api = DashboardAPI(self.trading_engine, self.db_manager)

            trade_data = {
                'trades': dashboard_api._get_recent_trades(limit=20),
                'timestamp': datetime.now().isoformat()
            }

            self.socketio.emit('trades_update', trade_data, room=sid)

        except Exception as e:
            logger.error(f"Error sending trade data to {sid}: {e}")

    def broadcast_dashboard_update(self):
        """Broadcast dashboard updates to all connected clients"""
        if not self.active_connections:
            return

        try:
            from .dashboard_api import DashboardAPI
            dashboard_api = DashboardAPI(self.trading_engine, self.db_manager)

            # Get latest data
            overview_data = {
                'performance': dashboard_api._get_performance_metrics(),
                'positions': dashboard_api._get_current_positions(),
                'system_health': dashboard_api._get_system_health(),
                'timestamp': datetime.now().isoformat()
            }

            # Check if data has changed
            data_hash = hash(str(overview_data))
            if self.last_data_hash.get('overview') != data_hash:
                self.socketio.emit('dashboard_update', overview_data, broadcast=True)
                self.last_data_hash['overview'] = data_hash
                logger.debug(f"Broadcasted dashboard update to {len(self.active_connections)} clients")

        except Exception as e:
            logger.error(f"Error broadcasting dashboard update: {e}")

    def broadcast_trade_alert(self, trade_data: Dict[str, Any]):
        """Broadcast new trade alert to all connected clients"""
        if not self.active_connections:
            return

        try:
            alert_data = {
                'type': 'new_trade',
                'trade': trade_data,
                'timestamp': datetime.now().isoformat()
            }

            self.socketio.emit('trade_alert', alert_data, broadcast=True)
            logger.info(f"Broadcasted trade alert to {len(self.active_connections)} clients")

        except Exception as e:
            logger.error(f"Error broadcasting trade alert: {e}")

    def start_broadcast_thread(self):
        """Start background thread for periodic data broadcasting"""
        def broadcast_loop():
            while True:
                try:
                    if self.active_connections:
                        self.broadcast_dashboard_update()
                    time.sleep(10)  # Broadcast every 10 seconds
                except Exception as e:
                    logger.error(f"Error in broadcast loop: {e}")
                    time.sleep(30)  # Wait longer on error

        broadcast_thread = Thread(target=broadcast_loop, daemon=True)
        broadcast_thread.start()
        logger.info("Dashboard WebSocket broadcast thread started")

    def notify_trade_executed(self, order_result: Dict, signal_data: Dict):
        """Notify clients when a trade is executed"""
        try:
            trade_data = {
                'symbol': order_result.get('symbol'),
                'side': order_result.get('side'),
                'quantity': float(order_result.get('executedQty', 0)),
                'price': float(order_result.get('price', 0)),
                'strategy': signal_data.get('strategy', 'manual'),
                'timestamp': datetime.now().isoformat()
            }

            self.broadcast_trade_alert(trade_data)

            # Force refresh dashboard data
            self.broadcast_dashboard_update()

        except Exception as e:
            logger.error(f"Error notifying trade execution: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            'active_connections': len(self.active_connections),
            'last_broadcast': self.last_data_hash.get('overview_time', 'Never'),
            'status': 'running' if hasattr(self, 'socketio') else 'stopped'
        }

    def run(self, host='0.0.0.0', port=5002, debug=False):
        """Run the WebSocket server"""
        logger.info(f"Starting WebSocket server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)