import hmac
import hashlib
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from config.trading_config import TradingConfig

logger = logging.getLogger(__name__)


class WebhookHandler:
    """Secure TradingView webhook handler with comprehensive validation"""
    
    def __init__(self, config: TradingConfig, trading_engine):
        self.config = config
        self.trading_engine = trading_engine
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for webhook handling"""
        
        @self.app.route('/webhook', methods=['POST'])
        def handle_webhook():
            try:
                # Security validation
                if not self._validate_request(request):
                    return jsonify({'error': 'Unauthorized'}), 401
                
                # Parse webhook data
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Invalid JSON'}), 400
                
                # Validate required fields
                required_fields = ['action', 'symbol', 'price', 'strategy']
                if not all(field in data for field in required_fields):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Process trading signal
                result = self.trading_engine.process_signal(data)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Signal processed',
                    'result': result
                }), 200
                
            except Exception as e:
                logger.error(f"Webhook processing error: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '1.0.0'
            }), 200
    
    def _validate_request(self, request) -> bool:
        """Validate webhook request security"""
        
        # Check IP whitelist
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if client_ip not in self.config.allowed_ips:
            logger.warning(f"Unauthorized IP: {client_ip}")
            return False
        
        # Validate webhook signature (if implemented)
        signature = request.headers.get('X-Webhook-Signature')
        if signature:
            expected_signature = self._generate_webhook_signature(request.get_data())
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Invalid webhook signature")
                return False
        
        return True
    
    def _generate_webhook_signature(self, payload: bytes) -> str:
        """Generate webhook signature for validation"""
        return hmac.new(
            self.config.webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the webhook server"""
        logger.info(f"Starting webhook server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)