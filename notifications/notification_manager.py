"""
Notification Manager

Sends alerts and notifications through multiple channels:
- Telegram Bot
- Email (SMTP)
- Console logging

Priority levels: low, normal, high, critical
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Multi-channel notification system for trading alerts
    """

    def __init__(self, config, telegram_token: str = None, telegram_chat_id: str = None,
                 email_config: dict = None):
        self.config = config

        # Telegram configuration
        self.telegram_token = telegram_token or getattr(config, 'telegram_token', None)
        self.telegram_chat_id = telegram_chat_id or getattr(config, 'telegram_chat_id', None)
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)

        # Email configuration
        self.email_config = email_config or getattr(config, 'email_config', {})
        self.email_enabled = bool(self.email_config.get('smtp_server'))

        if self.telegram_enabled:
            logger.info("✅ Telegram notifications enabled")
        else:
            logger.info("⚠️ Telegram notifications disabled (no credentials)")

        if self.email_enabled:
            logger.info("✅ Email notifications enabled")
        else:
            logger.info("⚠️ Email notifications disabled (no SMTP config)")

    def send(self, title: str, message: str, priority: str = 'normal'):
        """
        Send notification through all enabled channels

        Args:
            title: Notification title
            message: Notification message
            priority: 'low', 'normal', 'high', 'critical'
        """
        # Always log
        log_level = {
            'low': logging.DEBUG,
            'normal': logging.INFO,
            'high': logging.WARNING,
            'critical': logging.ERROR
        }.get(priority, logging.INFO)

        logger.log(log_level, f"[{priority.upper()}] {title}: {message}")

        # Send via Telegram
        if self.telegram_enabled:
            self._send_telegram(title, message, priority)

        # Send via Email for high priority
        if self.email_enabled and priority in ['high', 'critical']:
            self._send_email(title, message, priority)

    def send_trade_alert(self, symbol: str, action: str, price: float,
                        quantity: float, strategy: str, reason: str = ""):
        """Send trade execution alert"""
        emoji = "📈" if action.lower() == 'buy' else "📉"

        message = f"""
{emoji} *Trade Executed*

*Symbol:* {symbol}
*Action:* {action.upper()}
*Price:* ${price:.2f}
*Quantity:* {quantity:.6f}
*Strategy:* {strategy}
*Reason:* {reason}
*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()

        self.send("Trade Alert", message, priority='normal')

    def send_parameter_update_alert(self, strategy: str, version: str,
                                   score: float, improvement: float):
        """Send parameter update notification"""
        message = f"""
✨ *Parameters Updated*

*Strategy:* {strategy}
*New Version:* {version}
*Score:* {score:.4f}
*Improvement:* {improvement:.2%}
*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

New parameters are now active!
        """.strip()

        self.send("Parameter Update", message, priority='normal')

    def send_error_alert(self, error_type: str, error_message: str, context: str = ""):
        """Send error notification"""
        message = f"""
❌ *Error Alert*

*Type:* {error_type}
*Message:* {error_message}
*Context:* {context}
*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the logs for more details.
        """.strip()

        self.send("System Error", message, priority='high')

    def send_performance_alert(self, metric: str, value: float, threshold: float, status: str):
        """Send performance threshold alert"""
        emoji = "⚠️" if status == 'warning' else "🚨"

        message = f"""
{emoji} *Performance Alert*

*Metric:* {metric}
*Current Value:* {value:.2f}
*Threshold:* {threshold:.2f}
*Status:* {status.upper()}
*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()

        priority = 'high' if status == 'critical' else 'normal'
        self.send("Performance Alert", message, priority=priority)

    def _send_telegram(self, title: str, message: str, priority: str):
        """Send notification via Telegram Bot"""
        try:
            # Add emoji based on priority
            emoji = {
                'low': 'ℹ️',
                'normal': '📢',
                'high': '⚠️',
                'critical': '🚨'
            }.get(priority, '📢')

            # Format message for Telegram
            text = f"{emoji} *{title}*\n\n{message}"

            # Send via Telegram Bot API
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': text,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.debug(f"Telegram notification sent: {title}")
            else:
                logger.warning(f"Telegram notification failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def _send_email(self, title: str, message: str, priority: str):
        """Send notification via Email"""
        try:
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender = self.email_config.get('sender_email')
            password = self.email_config.get('sender_password')
            recipient = self.email_config.get('recipient_email')

            if not all([smtp_server, sender, password, recipient]):
                logger.debug("Email config incomplete, skipping email notification")
                return

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{priority.upper()}] {title}"
            msg['From'] = sender
            msg['To'] = recipient

            # Create HTML version
            html_message = f"""
            <html>
              <head></head>
              <body>
                <h2 style="color: {'#d32f2f' if priority in ['high', 'critical'] else '#1976d2'};">
                  {title}
                </h2>
                <p><strong>Priority:</strong> {priority.upper()}</p>
                <hr>
                <pre style="font-family: monospace; background-color: #f5f5f5; padding: 10px;">
{message}
                </pre>
                <hr>
                <p style="color: #666; font-size: 12px;">
                  Sent by Binance Trading System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
              </body>
            </html>
            """

            part_html = MIMEText(html_message, 'html')
            msg.attach(part_html)

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)

            logger.debug(f"Email notification sent: {title}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def test_notification(self):
        """Send test notification to verify configuration"""
        self.send(
            title="Test Notification",
            message="This is a test notification from your Binance Trading System. If you received this, notifications are working correctly! 🎉",
            priority='low'
        )


# Factory function
def create_notification_manager(config, **kwargs) -> NotificationManager:
    """
    Create NotificationManager with config

    Usage:
        notifier = create_notification_manager(
            config,
            telegram_token="YOUR_TOKEN",
            telegram_chat_id="YOUR_CHAT_ID"
        )
    """
    return NotificationManager(config, **kwargs)


# Example configuration for main.py or config file:
"""
# Add to trading_config.py:

@dataclass
class TradingConfig:
    ...
    # Telegram notifications
    telegram_token: str = ""  # Get from @BotFather
    telegram_chat_id: str = ""  # Your chat ID

    # Email notifications
    email_config: dict = field(default_factory=lambda: {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'your_email@gmail.com',
        'sender_password': 'your_app_password',  # Use app password for Gmail
        'recipient_email': 'your_email@gmail.com'
    })
"""
