"""
WebSocket authentication middleware for Flask-SocketIO
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from functools import wraps
from flask import request, g
from flask_jwt_extended import decode_token

from ..config.websocket import get_websocket_settings
from ..utils.auth import auth_service
from ..utils.redis_manager import redis_manager


class WebSocketAuthMiddleware:
    """WebSocket authentication middleware."""

    def __init__(self):
        self.settings = get_websocket_settings()

    def authenticate(self, auth_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate WebSocket connection.

        Args:
            auth_data: Authentication data from client

        Returns:
            User information if authenticated, None otherwise
        """
        try:
            if not self.settings.WS_AUTH_REQUIRED:
                return {"user_id": None, "username": "anonymous", "is_authenticated": False}

            token = auth_data.get('token')
            if not token:
                return None

            # Verify JWT token
            payload = auth_service.verify_token(token)
            user_id = payload.get('sub')

            if not user_id:
                return None

            # Get user from database
            user = auth_service.get_user_from_token(token)
            if not user:
                return None

            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "is_authenticated": True,
                "account_tier": user.account_tier
            }

        except Exception as e:
            print(f"WebSocket authentication error: {e}")
            return None

    async def check_rate_limit(self, user_id: str, ip_address: str) -> bool:
        """Check if connection is within rate limits.

        Args:
            user_id: User ID (can be None for anonymous)
            ip_address: Client IP address

        Returns:
            True if within limits, False otherwise
        """
        try:
            # Check per-IP limits
            ip_connections = await redis_manager.get_cache(f"ws_connections:{ip_address}")
            ip_connections = ip_connections or 0
            if ip_connections >= self.settings.WS_MAX_CONNECTIONS_PER_IP:
                return False

            # Check per-user limits (if authenticated)
            if user_id:
                user_connections = await redis_manager.get_cache(f"ws_user_connections:{user_id}")
                user_connections = user_connections or 0
                if user_connections >= self.settings.WS_MAX_CONNECTIONS_PER_IP:
                    return False

            return True

        except Exception as e:
            print(f"Rate limit check error: {e}")
            return False

    async def increment_connection_count(self, user_id: Optional[int], ip_address: str):
        """Increment connection counters.

        Args:
            user_id: User ID (can be None for anonymous)
            ip_address: Client IP address
        """
        try:
            # Increment IP connection count
            current_ip_count = await redis_manager.get_cache(f"ws_connections:{ip_address}")
            current_ip_count = current_ip_count or 0
            await redis_manager.set_cache(f"ws_connections:{ip_address}", current_ip_count + 1, 3600)

            # Increment user connection count (if authenticated)
            if user_id:
                current_user_count = await redis_manager.get_cache(f"ws_user_connections:{user_id}")
                current_user_count = current_user_count or 0
                await redis_manager.set_cache(f"ws_user_connections:{user_id}", current_user_count + 1, 3600)

        except Exception as e:
            print(f"Error incrementing connection count: {e}")

    async def decrement_connection_count(self, user_id: Optional[int], ip_address: str):
        """Decrement connection counters.

        Args:
            user_id: User ID (can be None for anonymous)
            ip_address: Client IP address
        """
        try:
            # Decrement IP connection count
            current_ip_count = await redis_manager.get_cache(f"ws_connections:{ip_address}")
            if current_ip_count and current_ip_count > 0:
                await redis_manager.set_cache(f"ws_connections:{ip_address}", current_ip_count - 1, 3600)

            # Decrement user connection count (if authenticated)
            if user_id:
                current_user_count = await redis_manager.get_cache(f"ws_user_connections:{user_id}")
                if current_user_count and current_user_count > 0:
                    await redis_manager.set_cache(f"ws_user_connections:{user_id}", current_user_count - 1, 3600)

        except Exception as e:
            print(f"Error decrementing connection count: {e}")

    def validate_token_expiry(self, token: str) -> bool:
        """Validate if token is not expired and will remain valid for WebSocket session.

        Args:
            token: JWT token to validate

        Returns:
            True if token is valid for WebSocket session, False otherwise
        """
        try:
            payload = auth_service.verify_token(token)
            exp_timestamp = payload.get('exp')

            if not exp_timestamp:
                return False

            # Check if token will expire within threshold
            exp_datetime = datetime.fromtimestamp(exp_timestamp)
            threshold_datetime = datetime.utcnow() + timedelta(seconds=self.settings.WS_TOKEN_REFRESH_THRESHOLD)

            return exp_datetime > threshold_datetime

        except Exception as e:
            print(f"Token validation error: {e}")
            return False

    def get_connection_metadata(self, auth_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get connection metadata for logging and monitoring.

        Args:
            auth_data: Authentication data from client

        Returns:
            Dictionary with connection metadata
        """
        return {
            "ip_address": getattr(request, 'remote_addr', 'unknown'),
            "user_agent": getattr(request, 'user_agent', {}).get('string', 'unknown'),
            "timestamp": datetime.utcnow().isoformat(),
            "auth_method": "token" if auth_data.get('token') else "none",
            "has_credentials": bool(auth_data.get('token'))
        }


# Global WebSocket auth middleware instance
websocket_auth = WebSocketAuthMiddleware()


# Utility functions
def get_websocket_auth() -> WebSocketAuthMiddleware:
    """Get WebSocket authentication middleware instance."""
    return websocket_auth


def require_websocket_auth(func: Callable) -> Callable:
    """Decorator to require WebSocket authentication.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_middleware = get_websocket_auth()

        # Get auth data from function arguments or request
        auth_data = kwargs.get('auth_data') or getattr(func, '_auth_data', None)

        if not auth_data:
            emit('error', {
                'message': 'Authentication required',
                'code': 'AUTH_REQUIRED'
            })
            return False

        # Authenticate
        user_info = auth_middleware.authenticate(auth_data)
        if not user_info:
            emit('error', {
                'message': 'Authentication failed',
                'code': 'AUTH_FAILED'
            })
            return False

        # Check rate limits
        ip_address = getattr(request, 'remote_addr', 'unknown')
        if not auth_middleware.check_rate_limit(user_info.get('user_id'), ip_address):
            emit('error', {
                'message': 'Connection limit exceeded',
                'code': 'RATE_LIMIT_EXCEEDED'
            })
            return False

        # Validate token expiry
        token = auth_data.get('token')
        if token and not auth_middleware.validate_token_expiry(token):
            emit('error', {
                'message': 'Token expires too soon',
                'code': 'TOKEN_EXPIRES_SOON'
            })
            return False

        # Increment connection count
        auth_middleware.increment_connection_count(user_info.get('user_id'), ip_address)

        # Set user info in function context
        g.websocket_user = user_info

        return func(*args, **kwargs)

    return wrapper


def cleanup_websocket_auth(user_id: Optional[int], ip_address: str):
    """Clean up WebSocket authentication counters.

    Args:
        user_id: User ID (can be None for anonymous)
        ip_address: Client IP address
    """
    try:
        auth_middleware = get_websocket_auth()
        auth_middleware.decrement_connection_count(user_id, ip_address)
    except Exception as e:
        print(f"Error cleaning up WebSocket auth: {e}")
</content>
</line_count>