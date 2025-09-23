"""
Authentication and authorization utilities
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
from flask import request, g, current_app
from werkzeug.exceptions import Unauthorized, Forbidden, BadRequest

from ..config import get_settings
from ..models import User
from .redis_manager import redis_manager


class AuthService:
    """Authentication service with JWT and rate limiting."""

    def __init__(self):
        self.settings = get_settings()

    def create_access_token(self, user_id: int, username: str, email: str, **kwargs) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(
            minutes=self.settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

        payload = {
            "sub": str(user_id),
            "username": username,
            "email": email,
            "type": "access",
            "exp": expire,
            "iat": datetime.utcnow(),
            **kwargs
        }

        return jwt.encode(
            payload,
            self.settings.JWT_SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM
        )

    def create_refresh_token(self, user_id: int, username: str, email: str, **kwargs) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(
            days=self.settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )

        payload = {
            "sub": str(user_id),
            "username": username,
            "email": email,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow(),
            **kwargs
        }

        return jwt.encode(
            payload,
            self.settings.JWT_SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM
        )

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.settings.JWT_SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise Unauthorized("Token has expired")
        except jwt.InvalidTokenError:
            raise Unauthorized("Invalid token")

    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from JWT token."""
        try:
            payload = self.verify_token(token)
            user_id = payload.get("sub")

            if not user_id:
                return None

            # Get user from database
            from ..app.extensions import db
            user = db.session.query(User).filter(User.id == int(user_id)).first()

            if not user or not user.is_active:
                return None

            return user

        except Exception:
            return None

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key using bcrypt."""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(api_key.encode('utf-8'), salt).decode('utf-8')

    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash."""
        import bcrypt
        try:
            return bcrypt.checkpw(api_key.encode('utf-8'), hashed_key.encode('utf-8'))
        except Exception:
            return False

    async def check_rate_limit(self, user_id: str, limit_type: str) -> Dict[str, Any]:
        """Check rate limit for user."""
        # Get appropriate limits based on type
        if limit_type == "requests":
            limit_value = self.settings.RATE_LIMIT_REQUESTS_PER_MINUTE
            window_seconds = 60
        elif limit_type == "storage":
            limit_value = self.settings.RATE_LIMIT_STORAGE_PER_HOUR
            window_seconds = 3600
        elif limit_type == "downloads":
            limit_value = self.settings.RATE_LIMIT_DOWNLOADS_PER_HOUR
            window_seconds = 3600
        else:
            limit_value = 100
            window_seconds = 60

        return await redis_manager.check_rate_limit(user_id, limit_type, limit_value, window_seconds)

    async def increment_rate_limit(self, user_id: str, limit_type: str) -> Dict[str, Any]:
        """Increment rate limit counter."""
        # Get appropriate limits based on type
        if limit_type == "requests":
            limit_value = self.settings.RATE_LIMIT_REQUESTS_PER_MINUTE
            window_seconds = 60
        elif limit_type == "storage":
            limit_value = self.settings.RATE_LIMIT_STORAGE_PER_HOUR
            window_seconds = 3600
        elif limit_type == "downloads":
            limit_value = self.settings.RATE_LIMIT_DOWNLOADS_PER_HOUR
            window_seconds = 3600
        else:
            limit_value = 100
            window_seconds = 60

        return await redis_manager.increment_rate_limit(user_id, limit_type, window_seconds)

    def require_auth(self, func):
        """Decorator to require authentication."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check for JWT token in Authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise Unauthorized("Missing or invalid authorization header")

            token = auth_header.split(' ')[1]
            user = self.get_user_from_token(token)

            if not user:
                raise Unauthorized("Invalid or expired token")

            # Check if user is active
            if not user.is_active:
                raise Forbidden("User account is disabled")

            # Set user in flask g object
            g.current_user = user
            g.user_id = user.id

            return func(*args, **kwargs)

        return wrapper

    def require_api_key(self, func):
        """Decorator to require API key authentication."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check for API key in header
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                raise Unauthorized("Missing API key")

            # Get user by API key
            from ..app.extensions import db
            user = db.session.query(User).filter(User.api_key.isnot(None)).first()

            if not user:
                raise Unauthorized("Invalid API key")

            # Verify API key
            if not self.verify_api_key(api_key, user.api_key):
                raise Unauthorized("Invalid API key")

            # Check if API key is expired
            if user.is_api_key_expired():
                raise Unauthorized("API key has expired")

            # Update last used timestamp
            user.api_key_last_used = datetime.utcnow()
            db.session.commit()

            # Set user in flask g object
            g.current_user = user
            g.user_id = user.id

            return await func(*args, **kwargs)

        return wrapper

    def require_premium(self, func):
        """Decorator to require premium account."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not g.current_user.is_premium:
                raise Forbidden("Premium account required")
            return func(*args, **kwargs)
        return wrapper

    def get_current_user(self) -> Optional[User]:
        """Get current user from flask g object."""
        return getattr(g, 'current_user', None)

    def get_user_id(self) -> Optional[int]:
        """Get current user ID from flask g object."""
        return getattr(g, 'user_id', None)

    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission."""
        # Basic permission check based on account tier
        permissions_map = {
            "free": ["basic_tts", "view_own_requests"],
            "basic": ["basic_tts", "view_own_requests", "priority_support"],
            "premium": ["premium_tts", "view_own_requests", "priority_support", "batch_processing"],
            "enterprise": ["premium_tts", "view_own_requests", "priority_support", "batch_processing", "custom_voices"]
        }

        user_permissions = permissions_map.get(user.account_tier, [])
        return permission in user_permissions


# Global auth service instance
auth_service = AuthService()


# Utility functions for easy access
def get_auth_service() -> AuthService:
    """Get auth service instance."""
    return auth_service


def require_auth(f):
    """Decorator to require authentication."""
    return auth_service.require_auth(f)


def require_api_key(f):
    """Decorator to require API key authentication."""
    return auth_service.require_api_key(f)


def require_premium(f):
    """Decorator to require premium account."""
    return auth_service.require_premium(f)


def get_current_user() -> Optional[User]:
    """Get current user."""
    return auth_service.get_current_user()


def get_user_id() -> Optional[int]:
    """Get current user ID."""
    return auth_service.get_user_id()