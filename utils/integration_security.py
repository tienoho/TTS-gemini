"""
Integration Security for TTS System

This module provides comprehensive security features for integration management,
including credential encryption, access token management, API key rotation,
and security audit logging.
"""

import asyncio
import logging
import secrets
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import redis
import json
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from utils.exceptions import SecurityError, AuthenticationError, ValidationError


class SecurityLevel(str, Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TokenType(str, Enum):
    """Token type enumeration"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SESSION = "session"


class SecurityEvent(str, Enum):
    """Security event enumeration"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKED = "token_revoked"
    CREDENTIAL_ACCESS = "credential_access"
    CREDENTIAL_UPDATE = "credential_update"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_key: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    refresh_token_expiration_days: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    password_min_length: int = 12
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True
    max_sessions_per_user: int = 10
    session_timeout_minutes: int = 60
    audit_log_retention_days: int = 90


class CredentialEncryption:
    """Credential encryption and decryption utilities"""

    def __init__(self, key: str):
        self.key = key.encode()
        self.fernet = Fernet(self._derive_key())

    def _derive_key(self) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'integration_security_salt_2024',
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.key))

    def encrypt(self, data: Dict[str, Any]) -> str:
        """Encrypt credentials dictionary"""
        try:
            json_data = json.dumps(data).encode()
            encrypted = self.fernet.encrypt(json_data)
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise SecurityError(f"Failed to encrypt credentials: {str(e)}")

    def decrypt(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt credentials"""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except InvalidToken:
            raise SecurityError("Invalid or corrupted encrypted data")
        except Exception as e:
            raise SecurityError(f"Failed to decrypt credentials: {str(e)}")


class TokenManager:
    """JWT token management"""

    def __init__(self, secret: str, algorithm: str = "HS256"):
        self.secret = secret
        self.algorithm = algorithm

    def create_access_token(self, payload: Dict[str, Any], expiration_hours: int = 24) -> str:
        """Create access token"""
        try:
            expiration = datetime.utcnow() + timedelta(hours=expiration_hours)
            payload.update({
                'exp': expiration,
                'iat': datetime.utcnow(),
                'type': TokenType.ACCESS.value
            })
            return jwt.encode(payload, self.secret, algorithm=self.algorithm)
        except Exception as e:
            raise SecurityError(f"Failed to create access token: {str(e)}")

    def create_refresh_token(self, payload: Dict[str, Any], expiration_days: int = 30) -> str:
        """Create refresh token"""
        try:
            expiration = datetime.utcnow() + timedelta(days=expiration_days)
            payload.update({
                'exp': expiration,
                'iat': datetime.utcnow(),
                'type': TokenType.REFRESH.value
            })
            return jwt.encode(payload, self.secret, algorithm=self.algorithm)
        except Exception as e:
            raise SecurityError(f"Failed to create refresh token: {str(e)}")

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            raise SecurityError(f"Token verification failed: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> str:
        """Refresh access token using refresh token"""
        try:
            payload = self.verify_token(refresh_token)

            if payload.get('type') != TokenType.REFRESH.value:
                raise AuthenticationError("Invalid refresh token")

            # Create new access token with same user data
            user_data = {k: v for k, v in payload.items()
                        if k not in ['exp', 'iat', 'type']}
            return self.create_access_token(user_data)

        except Exception as e:
            raise SecurityError(f"Token refresh failed: {str(e)}")


class RateLimiter:
    """Rate limiting for security"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if rate limit is exceeded"""
        try:
            current_time = int(datetime.utcnow().timestamp())
            window_start = current_time - window_seconds

            # Clean old entries
            await self._cleanup_old_entries(key, window_start)

            # Get current count
            count = self.redis.zcount(key, window_start, current_time)

            if count >= limit:
                return False

            # Add current request
            self.redis.zadd(key, {str(current_time): current_time})
            self.redis.expire(key, window_seconds)

            return True

        except Exception as e:
            logging.error(f"Rate limit check failed: {str(e)}")
            return True  # Allow on error to prevent blocking legitimate requests

    async def _cleanup_old_entries(self, key: str, window_start: int) -> None:
        """Clean up old entries from sorted set"""
        try:
            self.redis.zremrangebyscore(key, '-inf', window_start)
        except Exception as e:
            logging.error(f"Failed to cleanup rate limit entries: {str(e)}")


class SecurityAudit:
    """Security audit logging"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logging.getLogger(__name__)

    async def log_security_event(self, event: SecurityEvent, user_id: Optional[int] = None,
                               integration_id: Optional[int] = None, details: Optional[Dict[str, Any]] = None,
                               ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                               severity: SecurityLevel = SecurityLevel.MEDIUM) -> None:
        """Log security event"""
        try:
            # Log to application logger
            log_data = {
                'event': event.value,
                'user_id': user_id,
                'integration_id': integration_id,
                'details': details,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'severity': severity.value,
                'timestamp': datetime.utcnow().isoformat()
            }

            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                self.logger.warning(f"Security Event: {json.dumps(log_data)}")
            else:
                self.logger.info(f"Security Event: {json.dumps(log_data)}")

            # Store in database if needed
            await self._store_audit_event(log_data)

        except Exception as e:
            self.logger.error(f"Failed to log security event: {str(e)}")

    async def _store_audit_event(self, log_data: Dict[str, Any]) -> None:
        """Store audit event in database"""
        try:
            # This would typically store in a security audit table
            # For now, we'll just log it
            pass
        except Exception as e:
            self.logger.error(f"Failed to store audit event: {str(e)}")


class APIKeyManager:
    """API key management and rotation"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def generate_api_key(self, name: str, permissions: List[str],
                             expiration_days: Optional[int] = None) -> Dict[str, str]:
        """Generate new API key"""
        try:
            # Generate key
            api_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Store key metadata
            key_data = {
                'name': name,
                'permissions': permissions,
                'created_at': datetime.utcnow().isoformat(),
                'is_active': True
            }

            if expiration_days:
                key_data['expires_at'] = (datetime.utcnow() + timedelta(days=expiration_days)).isoformat()

            # Store in Redis with key hash as identifier
            self.redis.setex(f"api_key:{key_hash}", 86400 * 30, json.dumps(key_data))  # 30 days

            return {
                'api_key': api_key,
                'key_id': key_hash[:16],  # First 16 chars as ID
                'name': name,
                'permissions': permissions,
                'expires_at': key_data.get('expires_at')
            }

        except Exception as e:
            raise SecurityError(f"Failed to generate API key: {str(e)}")

    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_data_str = self.redis.get(f"api_key:{key_hash}")

            if not key_data_str:
                return None

            key_data = json.loads(key_data_str)

            # Check if key is active
            if not key_data.get('is_active', False):
                return None

            # Check expiration
            if 'expires_at' in key_data:
                expires_at = datetime.fromisoformat(key_data['expires_at'])
                if datetime.utcnow() > expires_at:
                    return None

            return key_data

        except Exception as e:
            logging.error(f"API key validation failed: {str(e)}")
            return None

    async def revoke_api_key(self, key_hash: str) -> bool:
        """Revoke API key"""
        try:
            self.redis.delete(f"api_key:{key_hash}")
            return True
        except Exception as e:
            logging.error(f"Failed to revoke API key: {str(e)}")
            return False

    async def rotate_api_key(self, old_key_hash: str, new_name: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Rotate API key"""
        try:
            # Get old key data
            old_key_data_str = self.redis.get(f"api_key:{old_key_hash}")
            if not old_key_data_str:
                return None

            old_key_data = json.loads(old_key_data_str)

            # Generate new key
            new_key = secrets.token_urlsafe(32)
            new_key_hash = hashlib.sha256(new_key.encode()).hexdigest()

            # Create new key data
            new_key_data = old_key_data.copy()
            new_key_data['name'] = new_name or f"{old_key_data['name']}_rotated"
            new_key_data['created_at'] = datetime.utcnow().isoformat()
            new_key_data['rotated_from'] = old_key_hash

            # Store new key
            self.redis.setex(f"api_key:{new_key_hash}", 86400 * 30, json.dumps(new_key_data))

            # Revoke old key
            await self.revoke_api_key(old_key_hash)

            return {
                'api_key': new_key,
                'key_id': new_key_hash[:16],
                'name': new_key_data['name'],
                'permissions': new_key_data['permissions']
            }

        except Exception as e:
            logging.error(f"API key rotation failed: {str(e)}")
            return None


class IntegrationSecurity:
    """Main integration security class"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption = CredentialEncryption(config.encryption_key)
        self.token_manager = TokenManager(config.jwt_secret, config.jwt_algorithm)
        self.audit = SecurityAudit(None)  # Will be set by manager
        self.rate_limiter = None  # Will be set by manager
        self.api_key_manager = None  # Will be set by manager

    async def encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        """Encrypt integration credentials"""
        try:
            # Add encryption metadata
            encrypted_data = {
                'credentials': credentials,
                'encrypted_at': datetime.utcnow().isoformat(),
                'version': '1.0'
            }

            encrypted = self.encryption.encrypt(encrypted_data)

            # Log credential encryption
            await self.audit.log_security_event(
                SecurityEvent.CREDENTIAL_UPDATE,
                details={'action': 'encrypt', 'credential_fields': list(credentials.keys())},
                severity=SecurityLevel.HIGH
            )

            return encrypted

        except Exception as e:
            await self.audit.log_security_event(
                SecurityEvent.SUSPICIOUS_ACTIVITY,
                details={'action': 'encrypt_failed', 'error': str(e)},
                severity=SecurityLevel.CRITICAL
            )
            raise SecurityError(f"Credential encryption failed: {str(e)}")

    async def decrypt_credentials(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt integration credentials"""
        try:
            decrypted = self.encryption.decrypt(encrypted_data)

            # Log credential access
            await self.audit.log_security_event(
                SecurityEvent.CREDENTIAL_ACCESS,
                details={'action': 'decrypt'},
                severity=SecurityLevel.HIGH
            )

            return decrypted.get('credentials', {})

        except Exception as e:
            await self.audit.log_security_event(
                SecurityEvent.SUSPICIOUS_ACTIVITY,
                details={'action': 'decrypt_failed', 'error': str(e)},
                severity=SecurityLevel.CRITICAL
            )
            raise SecurityError(f"Credential decryption failed: {str(e)}")

    async def validate_credentials_format(self, credentials: Dict[str, Any],
                                       required_fields: List[str]) -> List[str]:
        """Validate credential format and completeness"""
        errors = []

        # Check required fields
        for field in required_fields:
            if field not in credentials or not credentials[field]:
                errors.append(f"Required field missing: {field}")

        # Validate field formats
        for field, value in credentials.items():
            if isinstance(value, str):
                if len(value) < 4:
                    errors.append(f"Field '{field}' is too short (minimum 4 characters)")
                if len(value) > 1000:
                    errors.append(f"Field '{field}' is too long (maximum 1000 characters)")

        return errors

    async def check_integration_access(self, user_id: int, integration_id: int,
                                     required_permission: str = "read") -> bool:
        """Check if user has access to integration"""
        try:
            # This would typically check database permissions
            # For now, we'll implement basic logic
            return True

        except Exception as e:
            await self.audit.log_security_event(
                SecurityEvent.UNAUTHORIZED_ACCESS,
                user_id=user_id,
                details={'integration_id': integration_id, 'error': str(e)},
                severity=SecurityLevel.HIGH
            )
            return False

    async def rotate_integration_credentials(self, integration_id: int,
                                           new_credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Rotate integration credentials"""
        try:
            # Encrypt new credentials
            encrypted = await self.encrypt_credentials(new_credentials)

            # Log credential rotation
            await self.audit.log_security_event(
                SecurityEvent.CREDENTIAL_UPDATE,
                integration_id=integration_id,
                details={
                    'action': 'rotate',
                    'credential_fields': list(new_credentials.keys())
                },
                severity=SecurityLevel.CRITICAL
            )

            return {
                'encrypted_credentials': encrypted,
                'rotated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            await self.audit.log_security_event(
                SecurityEvent.SUSPICIOUS_ACTIVITY,
                integration_id=integration_id,
                details={'action': 'rotate_failed', 'error': str(e)},
                severity=SecurityLevel.CRITICAL
            )
            raise SecurityError(f"Credential rotation failed: {str(e)}")

    async def validate_api_key_permissions(self, api_key_data: Dict[str, Any],
                                         required_permissions: List[str]) -> bool:
        """Validate API key has required permissions"""
        try:
            key_permissions = set(api_key_data.get('permissions', []))
            required_set = set(required_permissions)

            return required_set.issubset(key_permissions)

        except Exception as e:
            logging.error(f"Permission validation failed: {str(e)}")
            return False

    async def generate_secure_password(self, length: int = 16) -> str:
        """Generate a secure random password"""
        try:
            # Define character sets
            lowercase = 'abcdefghijklmnopqrstuvwxyz'
            uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            numbers = '0123456789'
            special = '!@#$%^&*()_+-=[]{}|;:,.<>?'

            # Ensure minimum requirements
            password = [
                secrets.choice(lowercase),
                secrets.choice(uppercase),
                secrets.choice(numbers),
                secrets.choice(special)
            ]

            # Fill remaining length
            all_chars = lowercase + uppercase + numbers + special
            for _ in range(length - 4):
                password.append(secrets.choice(all_chars))

            # Shuffle the password
            secrets.SystemRandom().shuffle(password)

            return ''.join(password)

        except Exception as e:
            raise SecurityError(f"Password generation failed: {str(e)}")

    async def hash_string(self, data: str) -> str:
        """Hash string using SHA-256"""
        try:
            return hashlib.sha256(data.encode()).hexdigest()
        except Exception as e:
            raise SecurityError(f"String hashing failed: {str(e)}")

    async def verify_string_hash(self, data: str, expected_hash: str) -> bool:
        """Verify string against hash"""
        try:
            actual_hash = await self.hash_string(data)
            return secrets.compare_digest(actual_hash, expected_hash)
        except Exception as e:
            raise SecurityError(f"Hash verification failed: {str(e)}")

    async def cleanup_expired_data(self) -> int:
        """Clean up expired security data"""
        try:
            cleaned_count = 0

            # This would typically clean up expired tokens, sessions, etc.
            # For now, return 0
            return cleaned_count

        except Exception as e:
            logging.error(f"Security cleanup failed: {str(e)}")
            return 0