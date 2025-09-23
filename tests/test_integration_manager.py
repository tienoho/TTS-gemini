"""
Tests for Integration Manager

This module contains comprehensive tests for integration lifecycle management,
credential handling, validation, and error recovery.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from utils.integration_manager import IntegrationManager
from utils.integration_security import IntegrationSecurity
from models.integration import (
    IntegrationConfig, IntegrationType, IntegrationStatus,
    IntegrationDB, CloudStorageProvider, IntegrationCredential
)
from utils.exceptions import IntegrationError, ValidationError


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    return Mock(spec=Session)


@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.decr.return_value = True
    redis_mock.zcount.return_value = 0
    redis_mock.zadd.return_value = True
    redis_mock.zremrangebyscore.return_value = True
    return redis_mock


@pytest.fixture
def mock_security_manager():
    """Mock security manager"""
    security_mock = Mock(spec=IntegrationSecurity)
    security_mock.encrypt_credentials = AsyncMock(return_value="encrypted_credentials")
    security_mock.decrypt_credentials = AsyncMock(return_value={
        "access_key": "test_key",
        "secret_key": "test_secret"
    })
    return security_mock


@pytest.fixture
def integration_manager(mock_db_session, mock_redis_client, mock_security_manager):
    """Integration manager instance"""
    return IntegrationManager(mock_db_session, mock_redis_client, mock_security_manager)


class TestIntegrationManagerCreation:
    """Test IntegrationManager creation and initialization"""

    def test_manager_creation(self, mock_db_session, mock_redis_client, mock_security_manager):
        """Test creating integration manager"""
        manager = IntegrationManager(mock_db_session, mock_redis_client, mock_security_manager)

        assert manager.db == mock_db_session
        assert manager.redis == mock_redis_client
        assert manager.security == mock_security_manager
        assert manager._active_integrations == {}
        assert manager._rate_limiters == {}

    def test_manager_with_real_services(self):
        """Test manager with real service instances"""
        # This would test with actual database and Redis connections
        # For now, just ensure the structure is correct
        pass


class TestIntegrationCRUD:
    """Test integration CRUD operations"""

    @pytest.mark.asyncio
    async def test_create_integration_success(self, integration_manager, mock_db_session, mock_redis_client):
        """Test successful integration creation"""
        # Arrange
        config = IntegrationConfig(
            name="Test Integration",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential(
                access_key="test_key",
                secret_key="test_secret"
            ),
            rate_limit=1000
        )

        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.name = "Test Integration"

        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None
        mock_db_session.refresh.return_value = None

        # Mock the query to return the integration
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.create_integration(config, user_id=1)

        # Assert
        assert result is not None
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called()
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_integration_validation_error(self, integration_manager):
        """Test integration creation with validation error"""
        # Arrange
        config = IntegrationConfig(
            name="",  # Invalid: empty name
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential()
        )

        # Act & Assert
        with pytest.raises(ValidationError):
            await integration_manager.create_integration(config)

    @pytest.mark.asyncio
    async def test_create_integration_database_error(self, integration_manager, mock_db_session):
        """Test integration creation with database error"""
        # Arrange
        config = IntegrationConfig(
            name="Test Integration",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential(
                access_key="test_key",
                secret_key="test_secret"
            )
        )

        mock_db_session.add.side_effect = Exception("Database error")
        mock_db_session.rollback.return_value = None

        # Act & Assert
        with pytest.raises(IntegrationError):
            await integration_manager.create_integration(config)

        mock_db_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_integration_success(self, integration_manager, mock_db_session):
        """Test successful integration retrieval"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.name = "Test Integration"

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.get_integration(1)

        # Assert
        assert result == mock_integration
        mock_db_session.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_integration_not_found(self, integration_manager, mock_db_session):
        """Test integration retrieval when not found"""
        # Arrange
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.get_integration(999)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_update_integration_success(self, integration_manager, mock_db_session):
        """Test successful integration update"""
        # Arrange
        config = IntegrationConfig(
            name="Updated Integration",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential(
                access_key="updated_key",
                secret_key="updated_secret"
            )
        )

        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.name = "Original Integration"

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.update_integration(1, config, user_id=1)

        # Assert
        assert result == mock_integration
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_integration_success(self, integration_manager, mock_db_session):
        """Test successful integration deletion"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.delete_integration(1, user_id=1)

        # Assert
        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_integration)
        mock_db_session.commit.assert_called_once()


class TestIntegrationTesting:
    """Test integration testing functionality"""

    @pytest.mark.asyncio
    async def test_test_integration_success(self, integration_manager, mock_db_session):
        """Test successful integration test"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.integration_type = IntegrationType.CLOUD_STORAGE
        mock_integration.settings = {"provider": "aws_s3"}

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Mock handler
        mock_handler = Mock()
        mock_handler.test_connection = AsyncMock(return_value={
            "success": True,
            "message": "Test successful",
            "response_time_ms": 150
        })

        with patch.object(integration_manager, '_get_integration_handler', return_value=mock_handler):
            # Act
            result = await integration_manager.test_integration(1)

            # Assert
            assert result["success"] is True
            assert result["message"] == "Test successful"
            mock_handler.test_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_integration_failure(self, integration_manager, mock_db_session):
        """Test integration test failure"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.integration_type = IntegrationType.CLOUD_STORAGE

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Mock handler that raises exception
        mock_handler = Mock()
        mock_handler.test_connection = AsyncMock(side_effect=Exception("Connection failed"))

        with patch.object(integration_manager, '_get_integration_handler', return_value=mock_handler):
            # Act & Assert
            with pytest.raises(IntegrationError):
                await integration_manager.test_integration(1)


class TestIntegrationActivation:
    """Test integration activation/deactivation"""

    @pytest.mark.asyncio
    async def test_activate_integration_success(self, integration_manager, mock_db_session):
        """Test successful integration activation"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.is_active = False
        mock_integration.rate_limit = 1000

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.activate_integration(1)

        # Assert
        assert result is True
        assert mock_integration.is_active is True
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_integration_success(self, integration_manager, mock_db_session):
        """Test successful integration deactivation"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.is_active = True

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.deactivate_integration(1)

        # Assert
        assert result is True
        assert mock_integration.is_active is False
        mock_db_session.commit.assert_called_once()


class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_check_rate_limit_under_limit(self, integration_manager, mock_redis_client):
        """Test rate limit check when under limit"""
        # Arrange
        mock_redis_client.get.return_value = b"5"  # 5 requests in window

        # Act
        result = await integration_manager._check_rate_limit(1)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_at_limit(self, integration_manager, mock_redis_client):
        """Test rate limit check when at limit"""
        # Arrange
        mock_redis_client.get.return_value = b"1000"  # At limit

        # Act
        result = await integration_manager._check_rate_limit(1)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_rate_limiter(self, integration_manager, mock_redis_client):
        """Test rate limiter initialization"""
        # Act
        await integration_manager._initialize_rate_limiter(1, 1000)

        # Assert
        mock_redis_client.setex.assert_called_once_with(
            "rate_limit:1",
            60,
            1000
        )


class TestRetryLogic:
    """Test retry logic functionality"""

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, integration_manager):
        """Test successful execution with retry"""
        # Arrange
        mock_func = AsyncMock(return_value="success")

        # Act
        result = await integration_manager.execute_with_retry(
            1, "test_operation", mock_func, "arg1", "arg2", key="value"
        )

        # Assert
        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", key="value")

    @pytest.mark.asyncio
    async def test_execute_with_retry_with_failures(self, integration_manager):
        """Test execution with retry after failures"""
        # Arrange
        mock_func = AsyncMock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            "success"  # Third attempt succeeds
        ])

        # Act
        result = await integration_manager.execute_with_retry(
            1, "test_operation", mock_func
        )

        # Assert
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self, integration_manager):
        """Test execution when all retries are exhausted"""
        # Arrange
        mock_func = AsyncMock(side_effect=Exception("Always fails"))

        # Act & Assert
        with pytest.raises(IntegrationError):
            await integration_manager.execute_with_retry(
                1, "test_operation", mock_func
            )

        assert mock_func.call_count == 4  # Initial + 3 retries


class TestIntegrationValidation:
    """Test integration validation"""

    @pytest.mark.asyncio
    async def test_validate_cloud_storage_config_success(self, integration_manager):
        """Test successful cloud storage config validation"""
        # Arrange
        config = IntegrationConfig(
            name="Test S3",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential(
                access_key="AKIAIOSFODNN7EXAMPLE",
                secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                endpoint_url="https://s3.amazonaws.com"
            )
        )

        # Act & Assert (should not raise)
        await integration_manager._validate_cloud_storage_config(config)

    @pytest.mark.asyncio
    async def test_validate_cloud_storage_config_invalid_provider(self, integration_manager):
        """Test cloud storage config validation with invalid provider"""
        # Arrange
        config = IntegrationConfig(
            name="Test Invalid",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider="invalid_provider",
            credentials=IntegrationCredential(
                endpoint_url="https://example.com"
            )
        )

        # Act & Assert
        with pytest.raises(ValidationError):
            await integration_manager._validate_cloud_storage_config(config)

    @pytest.mark.asyncio
    async def test_validate_notification_config_success(self, integration_manager):
        """Test successful notification config validation"""
        # Arrange
        config = IntegrationConfig(
            name="Test Slack",
            integration_type=IntegrationType.NOTIFICATION,
            provider="slack",
            credentials=IntegrationCredential(
                token="xoxb-test-token"
            )
        )

        # Act & Assert (should not raise)
        await integration_manager._validate_notification_config(config)


class TestErrorHandling:
    """Test error handling and recovery"""

    @pytest.mark.asyncio
    async def test_database_error_handling(self, integration_manager, mock_db_session):
        """Test handling of database errors"""
        # Arrange
        mock_db_session.query.side_effect = Exception("Database connection error")

        # Act & Assert
        with pytest.raises(IntegrationError):
            await integration_manager.get_integration(1)

    @pytest.mark.asyncio
    async def test_redis_error_handling(self, integration_manager, mock_redis_client):
        """Test handling of Redis errors"""
        # Arrange
        mock_redis_client.get.side_effect = Exception("Redis connection error")

        # Act & Assert
        with pytest.raises(IntegrationError):
            await integration_manager._check_rate_limit(1)

    @pytest.mark.asyncio
    async def test_cleanup_inactive_integrations(self, integration_manager, mock_db_session):
        """Test cleanup of inactive integrations"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_integration]
        mock_db_session.query.return_value = mock_query

        # Act
        result = await integration_manager.cleanup_inactive_integrations()

        # Assert
        assert result == 1
        mock_db_session.delete.assert_called_once_with(mock_integration)


class TestIntegrationContext:
    """Test integration context manager"""

    @pytest.mark.asyncio
    async def test_integration_context_success(self, integration_manager, mock_db_session):
        """Test successful integration context usage"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.is_active = True

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Act
        async with integration_manager.integration_context(1, integration_manager) as integration:
            # Assert
            assert integration == mock_integration

    @pytest.mark.asyncio
    async def test_integration_context_inactive(self, integration_manager, mock_db_session):
        """Test integration context with inactive integration"""
        # Arrange
        mock_integration = Mock(spec=IntegrationDB)
        mock_integration.id = 1
        mock_integration.is_active = False

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_integration
        mock_db_session.query.return_value = mock_query

        # Act & Assert
        with pytest.raises(IntegrationError):
            async with integration_manager.integration_context(1, integration_manager):
                pass