"""
Webhook Security Utilities
"""
import hmac
import hashlib
import secrets
import string
import re
import ipaddress
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from functools import wraps

from flask import request, current_app
from config.webhook import webhook_config

logger = logging.getLogger(__name__)

class WebhookSecurityError(Exception):
    """Base exception cho webhook security"""
    pass

class InvalidSignatureError(WebhookSecurityError):
    """Lỗi signature không hợp lệ"""
    pass

class RateLimitExceededError(WebhookSecurityError):
    """Lỗi vượt quá rate limit"""
    pass

class IPNotAllowedError(WebhookSecurityError):
    """Lỗi IP không được phép"""
    pass

class WebhookSecurity:
    """Class quản lý security cho webhooks"""

    def __init__(self):
        self.rate_limit_cache = {}  # In-memory rate limiting cache
        self.cleanup_cache()

    def generate_secret(self, length: int = None) -> str:
        """Tạo secret key cho webhook"""
        if length is None:
            length = webhook_config.SECRET_KEY_LENGTH

        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def generate_signature(self, payload: str, secret: str, algorithm: str = None) -> str:
        """Tạo HMAC signature cho payload"""
        if algorithm is None:
            algorithm = webhook_config.SIGNATURE_ALGORITHM

        if algorithm.lower() == 'sha256':
            hash_func = hashlib.sha256
        elif algorithm.lower() == 'sha512':
            hash_func = hashlib.sha512
        elif algorithm.lower() == 'sha1':
            hash_func = hashlib.sha1
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Ensure payload is bytes
        if isinstance(payload, str):
            payload = payload.encode('utf-8')

        # Ensure secret is bytes
        if isinstance(secret, str):
            secret = secret.encode('utf-8')

        signature = hmac.new(secret, payload, hash_func)
        return f"{algorithm.lower()}={signature.hexdigest()}"

    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Xác minh HMAC signature"""
        try:
            # Parse signature format: "algorithm=signature"
            if '=' not in signature:
                raise InvalidSignatureError("Invalid signature format")

            algorithm, expected_signature = signature.split('=', 1)

            # Generate signature with the same algorithm
            actual_signature = self.generate_signature(payload, secret, algorithm)

            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(actual_signature, signature)

        except Exception as e:
            logger.error(f"Signature verification failed: {str(e)}")
            raise InvalidSignatureError(f"Signature verification failed: {str(e)}")

    def validate_ip_address(self, ip_address: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return False

    def is_ip_allowed(self, ip_address: str, whitelist: List[str] = None, blacklist: List[str] = None) -> bool:
        """Kiểm tra IP có được phép truy cập không"""
        if whitelist is None:
            whitelist = webhook_config.TRUSTED_IPS

        if blacklist is None:
            blacklist = []

        try:
            client_ip = ipaddress.ip_address(ip_address)

            # Check blacklist first
            for blocked_ip in blacklist:
                try:
                    blocked_network = ipaddress.ip_network(blocked_ip, strict=False)
                    if client_ip in blocked_network:
                        return False
                except ValueError:
                    # Invalid IP in blacklist, skip
                    continue

            # If whitelist is enabled, check whitelist
            if webhook_config.ENABLE_IP_WHITELIST and whitelist:
                for allowed_ip in whitelist:
                    try:
                        allowed_network = ipaddress.ip_network(allowed_ip, strict=False)
                        if client_ip in allowed_network:
                            return True
                    except ValueError:
                        # Invalid IP in whitelist, skip
                        continue
                return False  # IP not in whitelist

            return True  # IP allowed by default

        except ValueError:
            return False

    def check_rate_limit(self, webhook_id: str, limit: int = None, window: int = None) -> bool:
        """Kiểm tra rate limit cho webhook"""
        if limit is None:
            limit = webhook_config.DEFAULT_REQUESTS_PER_MINUTE
        if window is None:
            window = webhook_config.RATE_LIMIT_WINDOW

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window)

        # Get or create rate limit entry
        if webhook_id not in self.rate_limit_cache:
            self.rate_limit_cache[webhook_id] = []

        # Clean old entries
        self.rate_limit_cache[webhook_id] = [
            timestamp for timestamp in self.rate_limit_cache[webhook_id]
            if timestamp > window_start
        ]

        # Check if limit exceeded
        if len(self.rate_limit_cache[webhook_id]) >= limit:
            return False

        # Add current request
        self.rate_limit_cache[webhook_id].append(now)
        return True

    def validate_request_size(self, data: bytes) -> bool:
        """Validate request size"""
        max_size = webhook_config.MAX_REQUEST_SIZE
        return len(data) <= max_size

    def validate_url(self, url: str) -> bool:
        """Validate webhook URL"""
        if len(url) > webhook_config.MAX_URL_LENGTH:
            return False

        pattern = webhook_config.URL_PATTERN
        return bool(re.match(pattern, url))

    def validate_headers(self, headers: Dict[str, str]) -> bool:
        """Validate custom headers"""
        total_size = sum(len(k) + len(v) for k, v in headers.items())
        return total_size <= webhook_config.MAX_HEADER_SIZE

    def log_security_event(self, webhook_id: int, event_type: str, details: Dict[str, Any] = None):
        """Log security event"""
        try:
            from models.webhook import WebhookSecurityLog

            log_entry = WebhookSecurityLog(
                webhook_id=webhook_id,
                event_type=event_type,
                ip_address=request.remote_addr if request else None,
                user_agent=request.headers.get('User-Agent') if request else None,
                details=details or {}
            )

            # In a real application, you would save this to database
            # For now, just log to logger
            logger.warning(f"Webhook security event: {event_type} for webhook {webhook_id}")

        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")

    def cleanup_cache(self):
        """Clean up expired rate limit cache entries"""
        # This would typically run as a background task
        pass

# Global webhook security instance
webhook_security = WebhookSecurity()

def require_webhook_auth(f):
    """Decorator để yêu cầu webhook authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get webhook ID from URL or headers
        webhook_id = kwargs.get('webhook_id') or request.headers.get('X-Webhook-ID')

        if not webhook_id:
            raise WebhookSecurityError("Webhook ID required")

        # Get signature from headers
        signature = request.headers.get('X-Webhook-Signature')
        if not signature:
            raise InvalidSignatureError("Signature required")

        # Get raw request data
        payload = request.get_data()

        # Validate request size
        if not webhook_security.validate_request_size(payload):
            raise WebhookSecurityError("Request too large")

        # Get webhook secret (in real app, fetch from database)
        # For now, assume it's passed or stored securely
        secret = getattr(request, 'webhook_secret', None)
        if not secret:
            raise WebhookSecurityError("Webhook secret not found")

        # Verify signature
        if not webhook_security.verify_signature(payload.decode('utf-8'), signature, secret):
            webhook_security.log_security_event(
                webhook_id,
                'signature_verification_failed',
                {'signature': signature[:10] + '...'}
            )
            raise InvalidSignatureError("Invalid signature")

        # Check IP restrictions
        client_ip = request.remote_addr
        if client_ip and not webhook_security.is_ip_allowed(client_ip):
            webhook_security.log_security_event(
                webhook_id,
                'ip_not_allowed',
                {'ip_address': client_ip}
            )
            raise IPNotAllowedError("IP address not allowed")

        return f(*args, **kwargs)

    return decorated_function

def rate_limit_webhook(f):
    """Decorator để rate limit webhook requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        webhook_id = kwargs.get('webhook_id') or request.headers.get('X-Webhook-ID')

        if not webhook_id:
            raise WebhookSecurityError("Webhook ID required")

        # Check rate limit
        if not webhook_security.check_rate_limit(webhook_id):
            webhook_security.log_security_event(
                webhook_id,
                'rate_limit_exceeded',
                {'limit': webhook_config.DEFAULT_REQUESTS_PER_MINUTE}
            )
            raise RateLimitExceededError("Rate limit exceeded")

        return f(*args, **kwargs)

    return decorated_function