"""
Tests for Integration Models

This module contains comprehensive tests for integration data models,
including validation, serialization, and database operations.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pydantic import ValidationError

from models.integration import (
    IntegrationConfig, IntegrationStatus, IntegrationType, IntegrationStatusInfo,
    IntegrationTemplate, IntegrationCredential, CloudStorageProvider,
    NotificationProvider, DatabaseProvider, APIProtocol
)


class TestIntegrationCredential:
    """Test IntegrationCredential model"""

    def test_valid_credential_creation(self):
        """Test creating valid credentials"""
        credentials = IntegrationCredential(
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            endpoint_url="https://s3.amazonaws.com",
            region="us-east-1"
        )

        assert credentials.access_key == "AKIAIOSFODNN7EXAMPLE"
        assert credentials.secret_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert credentials.endpoint_url == "https://s3.amazonaws.com"
        assert credentials.region == "us-east-1"

    def test_credential_with_minimal_data(self):
        """Test creating credentials with minimal required data"""
        credentials = IntegrationCredential()

        assert credentials.access_key is None
        assert credentials.secret_key is None
        assert credentials.endpoint_url is None
        assert credentials.region is None

    def test_credential_serialization(self):
        """Test credential serialization"""
        credentials = IntegrationCredential(
            access_key="test_key",
            secret_key="test_secret",
            additional_config={"timeout": 30}
        )

        serialized = credentials.dict()
        assert serialized["access_key"] == "test_key"
        assert serialized["secret_key"] == "test_secret"
        assert serialized["additional_config"]["timeout"] == 30


class TestIntegrationConfig:
    """Test IntegrationConfig model"""

    def test_valid_config_creation(self):
        """Test creating valid integration config"""
        config = IntegrationConfig(
            name="Test Integration",
            description="Test integration for unit tests",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential(
                access_key="AKIAIOSFODNN7EXAMPLE",
                secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                region="us-east-1"
            ),
            settings={"bucket_name": "test-bucket"},
            rate_limit=1000,
            timeout=30,
            retry_attempts=3,
            retry_delay=1,
            is_active=True,
            tags=["test", "aws"]
        )

        assert config.name == "Test Integration"
        assert config.integration_type == IntegrationType.CLOUD_STORAGE
        assert config.provider == CloudStorageProvider.AWS_S3.value
        assert config.rate_limit == 1000
        assert config.timeout == 30
        assert config.is_active is True
        assert "test" in config.tags

    def test_config_validation_name_required(self):
        """Test that name is required"""
        with pytest.raises(ValidationError):
            IntegrationConfig(
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                credentials=IntegrationCredential()
            )

    def test_config_validation_name_length(self):
        """Test name length validation"""
        # Test name too long
        with pytest.raises(ValidationError):
            IntegrationConfig(
                name="a" * 256,  # Too long
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                credentials=IntegrationCredential()
            )

    def test_config_validation_rate_limit(self):
        """Test rate limit validation"""
        # Test negative rate limit
        with pytest.raises(ValidationError):
            IntegrationConfig(
                name="Test Integration",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                credentials=IntegrationCredential(),
                rate_limit=-1
            )

    def test_config_validation_timeout(self):
        """Test timeout validation"""
        # Test zero timeout
        with pytest.raises(ValidationError):
            IntegrationConfig(
                name="Test Integration",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                credentials=IntegrationCredential(),
                timeout=0
            )

    def test_config_serialization(self):
        """Test config serialization"""
        config = IntegrationConfig(
            name="Test Integration",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential(
                access_key="test_key",
                secret_key="test_secret"
            ),
            settings={"test": "value"}
        )

        serialized = config.dict()
        assert serialized["name"] == "Test Integration"
        assert serialized["integration_type"] == IntegrationType.CLOUD_STORAGE.value
        assert serialized["settings"]["test"] == "value"


class TestIntegrationStatusInfo:
    """Test IntegrationStatusInfo model"""

    def test_status_info_creation(self):
        """Test creating status info"""
        now = datetime.utcnow()
        status_info = IntegrationStatusInfo(
            status=IntegrationStatus.ACTIVE,
            last_check=now,
            last_success=now,
            last_error=now - timedelta(minutes=5),
            error_message="Test error",
            response_time_ms=150,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            uptime_percentage=95.0,
            metadata={"test": "data"}
        )

        assert status_info.status == IntegrationStatus.ACTIVE
        assert status_info.total_requests == 100
        assert status_info.successful_requests == 95
        assert status_info.failed_requests == 5
        assert status_info.uptime_percentage == 95.0
        assert status_info.metadata["test"] == "data"

    def test_status_info_defaults(self):
        """Test status info with default values"""
        status_info = IntegrationStatusInfo()

        assert status_info.status == IntegrationStatus.ACTIVE
        assert status_info.total_requests == 0
        assert status_info.successful_requests == 0
        assert status_info.failed_requests == 0
        assert status_info.uptime_percentage == 100.0

    def test_status_info_calculations(self):
        """Test uptime percentage calculation"""
        status_info = IntegrationStatusInfo(
            total_requests=100,
            successful_requests=90,
            failed_requests=10
        )

        assert status_info.uptime_percentage == 90.0

        # Test with zero total requests
        status_info_zero = IntegrationStatusInfo(
            total_requests=0,
            successful_requests=0,
            failed_requests=0
        )

        assert status_info_zero.uptime_percentage == 100.0


class TestIntegrationTemplate:
    """Test IntegrationTemplate model"""

    def test_template_creation(self):
        """Test creating integration template"""
        template = IntegrationTemplate(
            template_id="test-template",
            name="Test Template",
            description="Test template for unit tests",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            base_config=IntegrationConfig(
                name="Test Integration",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                credentials=IntegrationCredential()
            ),
            required_fields=["access_key", "secret_key"],
            optional_fields=["region"],
            validation_rules={
                "credentials.access_key": {"type": "string", "min_length": 16}
            },
            default_settings={"timeout": 30},
            is_public=True,
            tags=["test", "template"],
            created_by="test_user"
        )

        assert template.template_id == "test-template"
        assert template.name == "Test Template"
        assert template.is_public is True
        assert "access_key" in template.required_fields
        assert "region" in template.optional_fields
        assert template.created_by == "test_user"

    def test_template_validation_template_id(self):
        """Test template ID validation"""
        with pytest.raises(ValidationError):
            IntegrationTemplate(
                template_id="",  # Empty ID
                name="Test Template",
                description="Test",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                base_config=IntegrationConfig(
                    name="Test",
                    integration_type=IntegrationType.CLOUD_STORAGE,
                    provider=CloudStorageProvider.AWS_S3.value,
                    credentials=IntegrationCredential()
                )
            )

    def test_template_validation_name(self):
        """Test template name validation"""
        with pytest.raises(ValidationError):
            IntegrationTemplate(
                template_id="test-template",
                name="",  # Empty name
                description="Test",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                base_config=IntegrationConfig(
                    name="Test",
                    integration_type=IntegrationType.CLOUD_STORAGE,
                    provider=CloudStorageProvider.AWS_S3.value,
                    credentials=IntegrationCredential()
                )
            )

    def test_template_field_overlap_validation(self):
        """Test that required and optional fields don't overlap"""
        with pytest.raises(ValidationError):
            IntegrationTemplate(
                template_id="test-template",
                name="Test Template",
                description="Test",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                base_config=IntegrationConfig(
                    name="Test",
                    integration_type=IntegrationType.CLOUD_STORAGE,
                    provider=CloudStorageProvider.AWS_S3.value,
                    credentials=IntegrationCredential()
                ),
                required_fields=["field1"],
                optional_fields=["field1"]  # Overlap with required
            )


class TestIntegrationEnums:
    """Test integration enums"""

    def test_integration_type_enum(self):
        """Test IntegrationType enum"""
        assert IntegrationType.CLOUD_STORAGE.value == "cloud_storage"
        assert IntegrationType.NOTIFICATION.value == "notification"
        assert IntegrationType.DATABASE.value == "database"
        assert IntegrationType.API.value == "api"
        assert IntegrationType.FILE_PROCESSING.value == "file_processing"

    def test_integration_status_enum(self):
        """Test IntegrationStatus enum"""
        assert IntegrationStatus.ACTIVE.value == "active"
        assert IntegrationStatus.INACTIVE.value == "inactive"
        assert IntegrationStatus.ERROR.value == "error"
        assert IntegrationStatus.TESTING.value == "testing"
        assert IntegrationStatus.MAINTENANCE.value == "maintenance"

    def test_cloud_storage_provider_enum(self):
        """Test CloudStorageProvider enum"""
        assert CloudStorageProvider.AWS_S3.value == "aws_s3"
        assert CloudStorageProvider.GOOGLE_CLOUD.value == "google_cloud"
        assert CloudStorageProvider.AZURE_BLOB.value == "azure_blob"
        assert CloudStorageProvider.MINIO.value == "minio"

    def test_notification_provider_enum(self):
        """Test NotificationProvider enum"""
        assert NotificationProvider.SLACK.value == "slack"
        assert NotificationProvider.DISCORD.value == "discord"
        assert NotificationProvider.TEAMS.value == "teams"
        assert NotificationProvider.EMAIL.value == "email"
        assert NotificationProvider.WEBHOOK.value == "webhook"

    def test_database_provider_enum(self):
        """Test DatabaseProvider enum"""
        assert DatabaseProvider.POSTGRESQL.value == "postgresql"
        assert DatabaseProvider.MONGODB.value == "mongodb"
        assert DatabaseProvider.REDIS.value == "redis"
        assert DatabaseProvider.MYSQL.value == "mysql"
        assert DatabaseProvider.SQLITE.value == "sqlite"

    def test_api_protocol_enum(self):
        """Test APIProtocol enum"""
        assert APIProtocol.REST.value == "rest"
        assert APIProtocol.GRAPHQL.value == "graphql"
        assert APIProtocol.WEBSOCKET.value == "websocket"
        assert APIProtocol.SOAP.value == "soap"


class TestIntegrationModelEdgeCases:
    """Test edge cases for integration models"""

    def test_config_with_empty_credentials(self):
        """Test config with empty credentials"""
        config = IntegrationConfig(
            name="Test Integration",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential()
        )

        assert config.credentials.access_key is None
        assert config.credentials.secret_key is None

    def test_config_with_max_values(self):
        """Test config with maximum values"""
        config = IntegrationConfig(
            name="a" * 255,  # Max length
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            credentials=IntegrationCredential(),
            rate_limit=1000000,  # Large rate limit
            timeout=300,  # Max timeout
            retry_attempts=10,
            retry_delay=300
        )

        assert len(config.name) == 255
        assert config.rate_limit == 1000000
        assert config.timeout == 300

    def test_status_info_with_extreme_values(self):
        """Test status info with extreme values"""
        status_info = IntegrationStatusInfo(
            total_requests=999999,
            successful_requests=999999,
            failed_requests=0,
            uptime_percentage=100.0,
            response_time_ms=1  # Very fast response
        )

        assert status_info.total_requests == 999999
        assert status_info.uptime_percentage == 100.0
        assert status_info.response_time_ms == 1

    def test_template_with_complex_validation_rules(self):
        """Test template with complex validation rules"""
        complex_rules = {
            "credentials.access_key": {
                "type": "string",
                "min_length": 16,
                "max_length": 128,
                "pattern": "^[A-Z0-9]+$"
            },
            "credentials.secret_key": {
                "type": "string",
                "min_length": 32,
                "max_length": 128
            },
            "settings.timeout": {
                "type": "integer",
                "min": 1,
                "max": 300
            }
        }

        template = IntegrationTemplate(
            template_id="complex-template",
            name="Complex Template",
            description="Template with complex validation",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            base_config=IntegrationConfig(
                name="Test",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                credentials=IntegrationCredential()
            ),
            validation_rules=complex_rules
        )

        assert template.validation_rules == complex_rules
        assert "credentials.access_key" in template.validation_rules
        assert "settings.timeout" in template.validation_rules