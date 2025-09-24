"""
Integration Models for TTS System

This module defines the data models for managing integrations with external services.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class IntegrationStatus(str, Enum):
    """Integration status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"
    MAINTENANCE = "maintenance"


class IntegrationType(str, Enum):
    """Integration type enumeration"""
    CLOUD_STORAGE = "cloud_storage"
    NOTIFICATION = "notification"
    DATABASE = "database"
    API = "api"
    FILE_PROCESSING = "file_processing"
    WEBHOOK = "webhook"


class CloudStorageProvider(str, Enum):
    """Cloud storage provider enumeration"""
    AWS_S3 = "aws_s3"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BLOB = "azure_blob"
    MINIO = "minio"


class NotificationProvider(str, Enum):
    """Notification provider enumeration"""
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"


class DatabaseProvider(str, Enum):
    """Database provider enumeration"""
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class APIProtocol(str, Enum):
    """API protocol enumeration"""
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    SOAP = "soap"


class IntegrationCredential(BaseModel):
    """Integration credential model"""
    access_key: Optional[str] = Field(None, description="Access key or API key")
    secret_key: Optional[str] = Field(None, description="Secret key or API secret")
    token: Optional[str] = Field(None, description="Access token")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    endpoint_url: Optional[str] = Field(None, description="Service endpoint URL")
    region: Optional[str] = Field(None, description="Service region")
    additional_config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IntegrationConfig(BaseModel):
    """Integration configuration model"""
    name: str = Field(..., description="Integration name")
    description: Optional[str] = Field(None, description="Integration description")
    integration_type: IntegrationType = Field(..., description="Type of integration")
    provider: str = Field(..., description="Service provider")
    credentials: IntegrationCredential = Field(..., description="Integration credentials")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Integration settings")
    rate_limit: Optional[int] = Field(None, description="Requests per minute limit")
    timeout: Optional[int] = Field(30, description="Request timeout in seconds")
    retry_attempts: Optional[int] = Field(3, description="Number of retry attempts")
    retry_delay: Optional[int] = Field(1, description="Delay between retries in seconds")
    is_active: bool = Field(True, description="Whether integration is active")
    tags: List[str] = Field(default_factory=list, description="Integration tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('rate_limit')
    def validate_rate_limit(cls, v):
        if v is not None and v < 0:
            raise ValueError('Rate limit must be non-negative')
        return v

    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v


class IntegrationStatusInfo(BaseModel):
    """Integration status information model"""
    status: IntegrationStatus = Field(..., description="Current status")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last status check time")
    last_success: Optional[datetime] = Field(None, description="Last successful operation")
    last_error: Optional[datetime] = Field(None, description="Last error occurrence")
    error_message: Optional[str] = Field(None, description="Last error message")
    response_time_ms: Optional[int] = Field(None, description="Last response time in milliseconds")
    total_requests: int = Field(0, description="Total requests made")
    successful_requests: int = Field(0, description="Successful requests count")
    failed_requests: int = Field(0, description="Failed requests count")
    uptime_percentage: float = Field(100.0, description="Uptime percentage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Status metadata")


class IntegrationTemplate(BaseModel):
    """Integration template model"""
    template_id: str = Field(..., description="Template unique identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    integration_type: IntegrationType = Field(..., description="Integration type")
    provider: str = Field(..., description="Service provider")
    base_config: IntegrationConfig = Field(..., description="Base configuration")
    required_fields: List[str] = Field(default_factory=list, description="Required credential fields")
    optional_fields: List[str] = Field(default_factory=list, description="Optional credential fields")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Validation rules")
    default_settings: Dict[str, Any] = Field(default_factory=dict, description="Default settings")
    is_public: bool = Field(False, description="Whether template is publicly available")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    created_by: Optional[str] = Field(None, description="Template creator")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


# SQLAlchemy Models
class IntegrationDB(Base):
    """Database model for integrations"""
    __tablename__ = "integrations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    integration_type = Column(SQLEnum(IntegrationType), nullable=False)
    provider = Column(String(100), nullable=False)
    credentials = Column(JSON, nullable=False)
    settings = Column(JSON, default=dict)
    rate_limit = Column(Integer, nullable=True)
    timeout = Column(Integer, default=30)
    retry_attempts = Column(Integer, default=3)
    retry_delay = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    tags = Column(JSON, default=list)
    request_metadata = Column(JSON, default=dict)
    status_info = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(Integer, nullable=True)
    organization_id = Column(Integer, nullable=True)


class IntegrationTemplateDB(Base):
    """Database model for integration templates"""
    __tablename__ = "integration_templates"

    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    integration_type = Column(SQLEnum(IntegrationType), nullable=False)
    provider = Column(String(100), nullable=False)
    base_config = Column(JSON, nullable=False)
    required_fields = Column(JSON, default=list)
    optional_fields = Column(JSON, default=list)
    validation_rules = Column(JSON, default=dict)
    default_settings = Column(JSON, default=dict)
    is_public = Column(Boolean, default=False)
    tags = Column(JSON, default=list)
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class IntegrationLogDB(Base):
    """Database model for integration logs"""
    __tablename__ = "integration_logs"

    id = Column(Integer, primary_key=True, index=True)
    integration_id = Column(Integer, nullable=False)
    operation = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # success, error, warning
    message = Column(Text, nullable=True)
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    error_code = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class IntegrationAuditDB(Base):
    """Database model for integration audit trail"""
    __tablename__ = "integration_audit"

    id = Column(Integer, primary_key=True, index=True)
    integration_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=True)
    action = Column(String(100), nullable=False)  # create, update, delete, test, enable, disable
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())