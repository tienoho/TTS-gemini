"""
Tests for Integration Security

This module contains comprehensive tests for integration security features,
including credential encryption, token management, rate limiting, and audit logging.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from cryptography.fernet import InvalidToken

from utils.integration_security import (
    IntegrationSecurity, SecurityConfig, CredentialEncryption,
    TokenManager, RateLimiter, SecurityAudit, APIKeyManager,
    SecurityLevel, TokenType, SecurityEvent
)
from utils.exceptions import SecurityError, AuthenticationError


@pytest.fixture
def security_config():
    """Security configuration fixture"""
    return SecurityConfig(
        encryption_key="test-encryption-key-32-chars-long",
        jwt_secret="test-jwt-secret-key-32-chars-long",
        jwt_algorithm="HS256",
        jwt_expiration_hours=24,
        max_login_attempts=5,
        lockout_duration_minutes=15
    )


@pytest.fixture
def integration_security(security_config):
    """Integration security instance"""
    return IntegrationSecurity(security_config)


class TestCredentialEncryption:
    """Test credential encryption/decryption"""

    def test_encrypt_decrypt_credentials(self, integration_security):
        """Test successful credential encryption and decryption"""
        # Arrange
        credentials = {
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "token": "test_token_12345"
        }

        # Act
        encrypted = integration_security.encryption.encrypt(credentials)
        decrypted = integration_security.encryption.decrypt(encrypted)

        # Assert
        assert decrypted == credentials
        assert isinstance(encrypted, str)
        assert len(encrypted) > 0

    def test_encrypt_invalid_data(self, integration_security):
        """Test encryption with invalid data"""
        # Act & Assert
        with pytest.raises(SecurityError):
            integration_security.encryption.encrypt(None)

    def test_decrypt_invalid_token(self, integration_security):
        """Test decryption with invalid token"""
        # Act & Assert
        with pytest.raises(SecurityError):
            integration_security.encryption.decrypt("invalid_encrypted_data")

    def test_decrypt_corrupted_data(self, integration_security):
        """Test decryption with corrupted data"""
        # Arrange
        credentials = {"access_key": "test"}
        encrypted = integration_security.encryption.encrypt(credentials)
        corrupted = encrypted[:-5] + "xxxxx"  # Corrupt the data

        # Act & Assert
        with pytest.raises(SecurityError):
            integration_security.encryption.decrypt(corrupted)


class TestTokenManager:
    """Test JWT token management"""

    def test_create_access_token(self, integration_security):
        """Test access token creation"""
        # Arrange
        payload = {"user_id": 1, "username": "test_user"}

        # Act
        token = integration_security.token_manager.create_access_token(payload)

        # Assert
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        decoded = integration_security.token_manager.verify_token(token)
        assert decoded["user_id"] == 1
        assert decoded["username"] == "test_user"
        assert decoded["type"] == TokenType.ACCESS.value

    def test_create_refresh_token(self, integration_security):
        """Test refresh token creation"""
        # Arrange
        payload = {"user_id": 1, "username": "test_user"}

        # Act
        token = integration_security.token_manager.create_refresh_token(payload)

        # Assert
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        decoded = integration_security.token_manager.verify_token(token)
        assert decoded["user_id"] == 1
        assert decoded["username"] == "test_user"
        assert decoded["type"] == TokenType.REFRESH.value

    def test_verify_valid_token(self, integration_security):
        """Test verification of valid token"""
        # Arrange
        payload = {"user_id": 1, "username": "test_user"}
        token = integration_security.token_manager.create_access_token(payload)

        # Act
        decoded = integration_security.token_manager.verify_token(token)

        # Assert
        assert decoded["user_id"] == 1
        assert decoded["username"] == "test_user"
        assert decoded["type"] == TokenType.ACCESS.value

    def test_verify_expired_token(self, integration_security):
        """Test verification of expired token"""
        # Arrange
        payload = {"user_id": 1, "username": "test_user"}
        token = integration_security.token_manager.create_access_token(payload, expiration_hours=-1)

        # Act & Assert
        with pytest.raises(AuthenticationError):
            integration_security.token_manager.verify_token(token)

    def test_verify_invalid_token(self, integration_security):
        """Test verification of invalid token"""
        # Act & Assert
        with pytest.raises(AuthenticationError):
            integration_security.token_manager.verify_token("invalid_token")

    def test_refresh_access_token(self, integration_security):
        """Test access token refresh"""
        # Arrange
        payload = {"user_id": 1, "username": "test_user"}
        refresh_token = integration_security.token_manager.create_refresh_token(payload)

        # Act
        new_access_token = integration_security.token_manager.refresh_access_token(refresh_token)

        # Assert
        assert isinstance(new_access_token, str)
        assert len(new_access_token) > 0

        # Verify new token
        decoded = integration_security.token_manager.verify_token(new_access_token)
        assert decoded["user_id"] == 1
        assert decoded["username"] == "test_user"
        assert decoded["type"] == TokenType.ACCESS.value

    def test_refresh_with_invalid_token(self, integration_security):
        """Test refresh with invalid refresh token"""
        # Act & Assert
        with pytest.raises(SecurityError):
            integration_security.token_manager.refresh_access_token("invalid_token")


class TestRateLimiter:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_check_rate_limit_under_limit(self, integration_security):
        """Test rate limit check when under limit"""
        # Arrange
        mock_redis = Mock()
        mock_redis.zcount.return_value = 5  # Under limit
        mock_redis.zadd.return_value = True
        mock_redis.expire.return_value = True

        rate_limiter = RateLimiter(mock_redis)

        # Act
        result = await rate_limiter.check_rate_limit("test_key", 10)

        # Assert
        assert result is True
        mock_redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_rate_limit_at_limit(self, integration_security):
        """Test rate limit check when at limit"""
        # Arrange
        mock_redis = Mock()
        mock_redis.zcount.return_value = 10  # At limit
        mock_redis.zadd.return_value = True

        rate_limiter = RateLimiter(mock_redis)

        # Act
        result = await rate_limiter.check_rate_limit("test_key", 10)

        # Assert
        assert result is False
        mock_redis.zadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_rate_limit_redis_error(self, integration_security):
        """Test rate limit check with Redis error"""
        # Arrange
        mock_redis = Mock()
        mock_redis.zcount.side_effect = Exception("Redis error")

        rate_limiter = RateLimiter(mock_redis)

        # Act
        result = await rate_limiter.check_rate_limit("test_key", 10)

        # Assert
        assert result is True  # Should allow on error to prevent blocking


class TestSecurityAudit:
    """Test security audit logging"""

    @pytest.mark.asyncio
    async def test_log_security_event(self, integration_security):
        """Test security event logging"""
        # Arrange
        mock_db = Mock()
        audit = SecurityAudit(mock_db)

        # Act
        await audit.log_security_event(
            SecurityEvent.LOGIN_SUCCESS,
            user_id=1,
            integration_id=1,
            details={"ip": "127.0.0.1"},
            ip_address="127.0.0.1",
            user_agent="test-agent",
            severity=SecurityLevel.MEDIUM
        )

        # Assert - In real implementation, this would check database calls
        # For now, just ensure no exceptions are raised

    @pytest.mark.asyncio
    async def test_log_high_severity_event(self, integration_security):
        """Test logging high severity event"""
        # Arrange
        mock_db = Mock()
        audit = SecurityAudit(mock_db)

        # Act
        await audit.log_security_event(
            SecurityEvent.SUSPICIOUS_ACTIVITY,
            details={"suspicious_action": "unauthorized_access"},
            severity=SecurityLevel.CRITICAL
        )

        # Assert - Should trigger warning level logging


class TestAPIKeyManager:
    """Test API key management"""

    @pytest.mark.asyncio
    async def test_generate_api_key(self, integration_security):
        """Test API key generation"""
        # Arrange
        mock_redis = Mock()
        mock_redis.setex.return_value = True

        api_manager = APIKeyManager(mock_redis)

        # Act
        result = await api_manager.generate_api_key(
            "Test Key",
            ["read", "write"],
            expiration_days=30
        )

        # Assert
        assert "api_key" in result
        assert "key_id" in result
        assert "name" in result
        assert "permissions" in result
        assert result["name"] == "Test Key"
        assert result["permissions"] == ["read", "write"]
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, integration_security):
        """Test successful API key validation"""
        # Arrange
        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({
            "name": "Test Key",
            "permissions": ["read", "write"],
            "is_active": True,
            "created_at": datetime.utcnow().isoformat()
        })

        api_manager = APIKeyManager(mock_redis)

        # Act
        result = await api_manager.validate_api_key("test_api_key")

        # Assert
        assert result is not None
        assert result["name"] == "Test Key"
        assert result["permissions"] == ["read", "write"]

    @pytest.mark.asyncio
    async def test_validate_api_key_not_found(self, integration_security):
        """Test API key validation when key not found"""
        # Arrange
        mock_redis = Mock()
        mock_redis.get.return_value = None

        api_manager = APIKeyManager(mock_redis)

        # Act
        result = await api_manager.validate_api_key("nonexistent_key")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_inactive(self, integration_security):
        """Test API key validation when key is inactive"""
        # Arrange
        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({
            "name": "Test Key",
            "permissions": ["read", "write"],
            "is_active": False,
            "created_at": datetime.utcnow().isoformat()
        })

        api_manager = APIKeyManager(mock_redis)

        # Act
        result = await api_manager.validate_api_key("inactive_key")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, integration_security):
        """Test API key revocation"""
        # Arrange
        mock_redis = Mock()
        mock_redis.delete.return_value = True

        api_manager = APIKeyManager(mock_redis)

        # Act
        result = await api_manager.revoke_api_key("test_key_hash")

        # Assert
        assert result is True
        mock_redis.delete.assert_called_once_with("api_key:test_key_hash")

    @pytest.mark.asyncio
    async def test_rotate_api_key(self, integration_security):
        """Test API key rotation"""
        # Arrange
        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({
            "name": "Original Key",
            "permissions": ["read", "write"],
            "is_active": True,
            "created_at": datetime.utcnow().isoformat()
        })
        mock_redis.setex.return_value = True
        mock_redis.delete.return_value = True

        api_manager = APIKeyManager(mock_redis)

        # Act
        result = await api_manager.rotate_api_key("old_key_hash", "Rotated Key")

        # Assert
        assert result is not None
        assert "api_key" in result
        assert result["name"] == "Rotated Key"
        assert result["permissions"] == ["read", "write"]


class TestIntegrationSecurityMain:
    """Test main IntegrationSecurity class"""

    @pytest.mark.asyncio
    async def test_encrypt_credentials_integration(self, integration_security):
        """Test credential encryption through main class"""
        # Arrange
        credentials = {
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        }

        # Act
        encrypted = await integration_security.encrypt_credentials(credentials)

        # Assert
        assert isinstance(encrypted, str)
        assert len(encrypted) > 0

    @pytest.mark.asyncio
    async def test_decrypt_credentials_integration(self, integration_security):
        """Test credential decryption through main class"""
        # Arrange
        credentials = {
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        }
        encrypted = await integration_security.encrypt_credentials(credentials)

        # Act
        decrypted = await integration_security.decrypt_credentials(encrypted)

        # Assert
        assert decrypted == credentials

    @pytest.mark.asyncio
    async def test_validate_credentials_format_valid(self, integration_security):
        """Test credential format validation with valid data"""
        # Arrange
        credentials = {
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        }
        required_fields = ["access_key", "secret_key"]

        # Act
        errors = await integration_security.validate_credentials_format(credentials, required_fields)

        # Assert
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_credentials_format_missing_fields(self, integration_security):
        """Test credential format validation with missing fields"""
        # Arrange
        credentials = {"access_key": "AKIAIOSFODNN7EXAMPLE"}  # Missing secret_key
        required_fields = ["access_key", "secret_key"]

        # Act
        errors = await integration_security.validate_credentials_format(credentials, required_fields)

        # Assert
        assert len(errors) == 1
        assert "secret_key" in errors[0]

    @pytest.mark.asyncio
    async def test_generate_secure_password(self, integration_security):
        """Test secure password generation"""
        # Act
        password = await integration_security.generate_secure_password(16)

        # Assert
        assert isinstance(password, str)
        assert len(password) == 16
        assert any(c.islower() for c in password)
        assert any(c.isupper() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    @pytest.mark.asyncio
    async def test_hash_string(self, integration_security):
        """Test string hashing"""
        # Arrange
        data = "test_string"

        # Act
        hash_result = await integration_security.hash_string(data)

        # Assert
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 produces 64 character hex string

    @pytest.mark.asyncio
    async def test_verify_string_hash(self, integration_security):
        """Test string hash verification"""
        # Arrange
        data = "test_string"
        expected_hash = await integration_security.hash_string(data)

        # Act
        result = await integration_security.verify_string_hash(data, expected_hash)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_string_hash_invalid(self, integration_security):
        """Test string hash verification with invalid hash"""
        # Arrange
        data = "test_string"
        invalid_hash = "invalid_hash"

        # Act
        result = await integration_security.verify_string_hash(data, invalid_hash)

        # Assert
        assert result is False


class TestSecurityErrorHandling:
    """Test security error handling"""

    @pytest.mark.asyncio
    async def test_encrypt_credentials_error_handling(self, integration_security):
        """Test error handling in credential encryption"""
        # Arrange
        credentials = {"invalid": None}  # This might cause JSON serialization error

        # Act & Assert
        with pytest.raises(SecurityError):
            await integration_security.encrypt_credentials(credentials)

    @pytest.mark.asyncio
    async def test_decrypt_credentials_error_handling(self, integration_security):
        """Test error handling in credential decryption"""
        # Act & Assert
        with pytest.raises(SecurityError):
            await integration_security.decrypt_credentials("invalid_data")

    @pytest.mark.asyncio
    async def test_token_manager_error_handling(self, integration_security):
        """Test error handling in token manager"""
        # Act & Assert
        with pytest.raises(SecurityError):
            integration_security.token_manager.create_access_token(None)

    @pytest.mark.asyncio
    async def test_api_key_manager_error_handling(self, integration_security):
        """Test error handling in API key manager"""
        # Arrange
        mock_redis = Mock()
        mock_redis.setex.side_effect = Exception("Redis error")

        api_manager = APIKeyManager(mock_redis)

        # Act & Assert
        with pytest.raises(SecurityError):
            await api_manager.generate_api_key("Test", ["read"])


class TestSecurityEdgeCases:
    """Test security edge cases"""

    @pytest.mark.asyncio
    async def test_encrypt_empty_credentials(self, integration_security):
        """Test encryption of empty credentials"""
        # Arrange
        credentials = {}

        # Act
        encrypted = await integration_security.encrypt_credentials(credentials)
        decrypted = await integration_security.decrypt_credentials(encrypted)

        # Assert
        assert decrypted == {}

    @pytest.mark.asyncio
    async def test_encrypt_large_credentials(self, integration_security):
        """Test encryption of large credentials"""
        # Arrange
        large_credentials = {
            "large_field": "x" * 10000,  # 10KB field
            "another_field": "y" * 5000   # 5KB field
        }

        # Act
        encrypted = await integration_security.encrypt_credentials(large_credentials)
        decrypted = await integration_security.decrypt_credentials(encrypted)

        # Assert
        assert decrypted == large_credentials

    @pytest.mark.asyncio
    async def test_token_with_special_characters(self, integration_security):
        """Test token creation with special characters in payload"""
        # Arrange
        payload = {
            "user_id": 1,
            "special_data": "Special chars: !@#$%^&*()_+-=[]{}|;:,.<>?",
            "unicode_data": "Unicode: 你好世界 🌍"
        }

        # Act
        token = integration_security.token_manager.create_access_token(payload)
        decoded = integration_security.token_manager.verify_token(token)

        # Assert
        assert decoded["user_id"] == 1
        assert decoded["special_data"] == payload["special_data"]
        assert decoded["unicode_data"] == payload["unicode_data"]

    @pytest.mark.asyncio
    async def test_concurrent_access(self, integration_security):
        """Test concurrent access to security functions"""
        # Arrange
        credentials = {"access_key": "test", "secret_key": "secret"}

        # Act - Run multiple encryption/decryption operations concurrently
        tasks = []
        for i in range(10):
            tasks.append(integration_security.encrypt_credentials(credentials))
            tasks.append(integration_security.hash_string(f"test_{i}"))

        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 20
        assert all(isinstance(result, str) for result in results)