"""
Integration Configuration for TTS System

This module provides configuration settings for the integration ecosystem,
including security policies, rate limiting, timeout settings, and provider configurations.
"""

import os
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseSettings, Field, validator
from pydantic.types import SecretStr


class IntegrationConfig(BaseSettings):
    """Integration system configuration"""

    # General Settings
    INTEGRATION_ENABLED: bool = Field(default=True, env="INTEGRATION_ENABLED")
    INTEGRATION_MAX_PER_USER: int = Field(default=50, env="INTEGRATION_MAX_PER_USER")
    INTEGRATION_MAX_PER_ORGANIZATION: int = Field(default=500, env="INTEGRATION_MAX_PER_ORGANIZATION")

    # Security Settings
    INTEGRATION_ENCRYPTION_ENABLED: bool = Field(default=True, env="INTEGRATION_ENCRYPTION_ENABLED")
    INTEGRATION_CREDENTIAL_ENCRYPTION_KEY: SecretStr = Field(
        default=SecretStr("your-encryption-key-here-change-in-production"),
        env="INTEGRATION_CREDENTIAL_ENCRYPTION_KEY"
    )
    INTEGRATION_JWT_SECRET: SecretStr = Field(
        default=SecretStr("your-jwt-secret-here-change-in-production"),
        env="INTEGRATION_JWT_SECRET"
    )
    INTEGRATION_JWT_ALGORITHM: str = Field(default="HS256", env="INTEGRATION_JWT_ALGORITHM")
    INTEGRATION_JWT_EXPIRATION_HOURS: int = Field(default=24, env="INTEGRATION_JWT_EXPIRATION_HOURS")

    # Rate Limiting
    INTEGRATION_GLOBAL_RATE_LIMIT: int = Field(default=10000, env="INTEGRATION_GLOBAL_RATE_LIMIT")
    INTEGRATION_PER_USER_RATE_LIMIT: int = Field(default=1000, env="INTEGRATION_PER_USER_RATE_LIMIT")
    INTEGRATION_RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, env="INTEGRATION_RATE_LIMIT_WINDOW_SECONDS")
    INTEGRATION_BURST_LIMIT: int = Field(default=100, env="INTEGRATION_BURST_LIMIT")

    # Timeout Settings
    INTEGRATION_DEFAULT_TIMEOUT: int = Field(default=30, env="INTEGRATION_DEFAULT_TIMEOUT")
    INTEGRATION_CONNECTION_TIMEOUT: int = Field(default=10, env="INTEGRATION_CONNECTION_TIMEOUT")
    INTEGRATION_READ_TIMEOUT: int = Field(default=30, env="INTEGRATION_READ_TIMEOUT")
    INTEGRATION_TEST_TIMEOUT: int = Field(default=15, env="INTEGRATION_TEST_TIMEOUT")

    # Retry Settings
    INTEGRATION_DEFAULT_RETRY_ATTEMPTS: int = Field(default=3, env="INTEGRATION_DEFAULT_RETRY_ATTEMPTS")
    INTEGRATION_RETRY_DELAY: float = Field(default=1.0, env="INTEGRATION_RETRY_DELAY")
    INTEGRATION_MAX_RETRY_DELAY: float = Field(default=60.0, env="INTEGRATION_MAX_RETRY_DELAY")
    INTEGRATION_RETRY_BACKOFF_FACTOR: float = Field(default=2.0, env="INTEGRATION_RETRY_BACKOFF_FACTOR")

    # Template Settings
    INTEGRATION_TEMPLATE_ENABLED: bool = Field(default=True, env="INTEGRATION_TEMPLATE_ENABLED")
    INTEGRATION_TEMPLATE_MAX_PER_USER: int = Field(default=20, env="INTEGRATION_TEMPLATE_MAX_PER_USER")
    INTEGRATION_TEMPLATE_CACHE_TTL: int = Field(default=3600, env="INTEGRATION_TEMPLATE_CACHE_TTL")

    # Audit and Logging
    INTEGRATION_AUDIT_ENABLED: bool = Field(default=True, env="INTEGRATION_AUDIT_ENABLED")
    INTEGRATION_AUDIT_LOG_RETENTION_DAYS: int = Field(default=90, env="INTEGRATION_AUDIT_LOG_RETENTION_DAYS")
    INTEGRATION_SECURITY_LOG_LEVEL: str = Field(default="WARNING", env="INTEGRATION_SECURITY_LOG_LEVEL")

    # Provider-specific Settings
    INTEGRATION_AWS_S3_REGION: str = Field(default="us-east-1", env="INTEGRATION_AWS_S3_REGION")
    INTEGRATION_AWS_S3_MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, env="INTEGRATION_AWS_S3_MAX_FILE_SIZE")  # 100MB
    INTEGRATION_AWS_S3_ALLOWED_BUCKETS: List[str] = Field(default_factory=list, env="INTEGRATION_AWS_S3_ALLOWED_BUCKETS")

    INTEGRATION_SLACK_MAX_MESSAGE_LENGTH: int = Field(default=40000, env="INTEGRATION_SLACK_MAX_MESSAGE_LENGTH")
    INTEGRATION_SLACK_RATE_LIMIT: int = Field(default=100, env="INTEGRATION_SLACK_RATE_LIMIT")

    INTEGRATION_DATABASE_CONNECTION_POOL_SIZE: int = Field(default=10, env="INTEGRATION_DATABASE_CONNECTION_POOL_SIZE")
    INTEGRATION_DATABASE_MAX_CONNECTIONS: int = Field(default=20, env="INTEGRATION_DATABASE_MAX_CONNECTIONS")
    INTEGRATION_DATABASE_CONNECTION_TIMEOUT: int = Field(default=30, env="INTEGRATION_DATABASE_CONNECTION_TIMEOUT")

    INTEGRATION_REDIS_CONNECTION_POOL_SIZE: int = Field(default=10, env="INTEGRATION_REDIS_CONNECTION_POOL_SIZE")
    INTEGRATION_REDIS_MAX_CONNECTIONS: int = Field(default=20, env="INTEGRATION_REDIS_MAX_CONNECTIONS")

    # API Settings
    INTEGRATION_API_MAX_REQUEST_SIZE: int = Field(default=10 * 1024 * 1024, env="INTEGRATION_API_MAX_REQUEST_SIZE")  # 10MB
    INTEGRATION_API_REQUEST_TIMEOUT: int = Field(default=60, env="INTEGRATION_API_REQUEST_TIMEOUT")
    INTEGRATION_API_MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="INTEGRATION_API_MAX_CONCURRENT_REQUESTS")

    # Webhook Settings
    INTEGRATION_WEBHOOK_TIMEOUT: int = Field(default=30, env="INTEGRATION_WEBHOOK_TIMEOUT")
    INTEGRATION_WEBHOOK_MAX_RETRIES: int = Field(default=3, env="INTEGRATION_WEBHOOK_MAX_RETRIES")
    INTEGRATION_WEBHOOK_RETRY_DELAY: float = Field(default=5.0, env="INTEGRATION_WEBHOOK_RETRY_DELAY")
    INTEGRATION_WEBHOOK_MAX_SIZE: int = Field(default=1024 * 1024, env="INTEGRATION_WEBHOOK_MAX_SIZE")  # 1MB

    # File Processing Settings
    INTEGRATION_FILE_PROCESSING_MAX_SIZE: int = Field(default=50 * 1024 * 1024, env="INTEGRATION_FILE_PROCESSING_MAX_SIZE")  # 50MB
    INTEGRATION_FILE_PROCESSING_ALLOWED_EXTENSIONS: List[str] = Field(
        default_factory=lambda: ['.txt', '.csv', '.json', '.xml', '.pdf', '.docx'],
        env="INTEGRATION_FILE_PROCESSING_ALLOWED_EXTENSIONS"
    )
    INTEGRATION_FILE_PROCESSING_TIMEOUT: int = Field(default=300, env="INTEGRATION_FILE_PROCESSING_TIMEOUT")  # 5 minutes

    # Notification Settings
    INTEGRATION_NOTIFICATION_MAX_BATCH_SIZE: int = Field(default=100, env="INTEGRATION_NOTIFICATION_MAX_BATCH_SIZE")
    INTEGRATION_NOTIFICATION_RETRY_ATTEMPTS: int = Field(default=3, env="INTEGRATION_NOTIFICATION_RETRY_ATTEMPTS")
    INTEGRATION_NOTIFICATION_QUEUE_SIZE: int = Field(default=1000, env="INTEGRATION_NOTIFICATION_QUEUE_SIZE")

    # Monitoring and Health Check
    INTEGRATION_HEALTH_CHECK_ENABLED: bool = Field(default=True, env="INTEGRATION_HEALTH_CHECK_ENABLED")
    INTEGRATION_HEALTH_CHECK_INTERVAL: int = Field(default=60, env="INTEGRATION_HEALTH_CHECK_INTERVAL")  # seconds
    INTEGRATION_METRICS_ENABLED: bool = Field(default=True, env="INTEGRATION_METRICS_ENABLED")
    INTEGRATION_METRICS_RETENTION_HOURS: int = Field(default=24, env="INTEGRATION_METRICS_RETENTION_HOURS")

    # Error Handling
    INTEGRATION_CIRCUIT_BREAKER_ENABLED: bool = Field(default=True, env="INTEGRATION_CIRCUIT_BREAKER_ENABLED")
    INTEGRATION_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, env="INTEGRATION_CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    INTEGRATION_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=60, env="INTEGRATION_CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    INTEGRATION_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = Field(default=3, env="INTEGRATION_CIRCUIT_BREAKER_SUCCESS_THRESHOLD")

    # Validation Settings
    INTEGRATION_VALIDATION_ENABLED: bool = Field(default=True, env="INTEGRATION_VALIDATION_ENABLED")
    INTEGRATION_CREDENTIAL_VALIDATION_ENABLED: bool = Field(default=True, env="INTEGRATION_CREDENTIAL_VALIDATION_ENABLED")
    INTEGRATION_SCHEMA_VALIDATION_ENABLED: bool = Field(default=True, env="INTEGRATION_SCHEMA_VALIDATION_ENABLED")

    # Caching Settings
    INTEGRATION_CACHE_ENABLED: bool = Field(default=True, env="INTEGRATION_CACHE_ENABLED")
    INTEGRATION_CACHE_TTL: int = Field(default=300, env="INTEGRATION_CACHE_TTL")  # 5 minutes
    INTEGRATION_CACHE_MAX_SIZE: int = Field(default=1000, env="INTEGRATION_CACHE_MAX_SIZE")

    # Security Policies
    INTEGRATION_ALLOWED_ORIGINS: List[str] = Field(
        default_factory=lambda: ["*"],
        env="INTEGRATION_ALLOWED_ORIGINS"
    )
    INTEGRATION_CREDENTIAL_MASKING_ENABLED: bool = Field(default=True, env="INTEGRATION_CREDENTIAL_MASKING_ENABLED")
    INTEGRATION_IP_WHITELIST: List[str] = Field(default_factory=list, env="INTEGRATION_IP_WHITELIST")
    INTEGRATION_IP_BLACKLIST: List[str] = Field(default_factory=list, env="INTEGRATION_IP_BLACKLIST")

    # Performance Settings
    INTEGRATION_WORKER_COUNT: int = Field(default=4, env="INTEGRATION_WORKER_COUNT")
    INTEGRATION_QUEUE_SIZE: int = Field(default=1000, env="INTEGRATION_QUEUE_SIZE")
    INTEGRATION_BATCH_SIZE: int = Field(default=10, env="INTEGRATION_BATCH_SIZE")
    INTEGRATION_MAX_CONCURRENT_INTEGRATIONS: int = Field(default=50, env="INTEGRATION_MAX_CONCURRENT_INTEGRATIONS")

    # Development Settings
    INTEGRATION_DEBUG_MODE: bool = Field(default=False, env="INTEGRATION_DEBUG_MODE")
    INTEGRATION_LOG_LEVEL: str = Field(default="INFO", env="INTEGRATION_LOG_LEVEL")
    INTEGRATION_LOG_REQUESTS: bool = Field(default=False, env="INTEGRATION_LOG_REQUESTS")
    INTEGRATION_LOG_RESPONSES: bool = Field(default=False, env="INTEGRATION_LOG_RESPONSES")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator('INTEGRATION_JWT_EXPIRATION_HOURS')
    def validate_jwt_expiration(cls, v):
        if v < 1 or v > 168:  # Max 7 days
            raise ValueError('JWT expiration must be between 1 and 168 hours')
        return v

    @validator('INTEGRATION_RATE_LIMIT_WINDOW_SECONDS')
    def validate_rate_limit_window(cls, v):
        if v < 1 or v > 3600:  # Max 1 hour
            raise ValueError('Rate limit window must be between 1 and 3600 seconds')
        return v

    @validator('INTEGRATION_DEFAULT_TIMEOUT')
    def validate_timeout(cls, v):
        if v < 1 or v > 300:  # Max 5 minutes
            raise ValueError('Timeout must be between 1 and 300 seconds')
        return v

    @validator('INTEGRATION_DEFAULT_RETRY_ATTEMPTS')
    def validate_retry_attempts(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Retry attempts must be between 0 and 10')
        return v

    @validator('INTEGRATION_RETRY_DELAY')
    def validate_retry_delay(cls, v):
        if v < 0.1 or v > 300:
            raise ValueError('Retry delay must be between 0.1 and 300 seconds')
        return v


class ProviderConfig:
    """Provider-specific configuration"""

    # AWS S3 Configuration
    AWS_S3_CONFIG = {
        'regions': [
            'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1',
            'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1',
            'ap-northeast-2', 'ap-south-1', 'ca-central-1', 'sa-east-1'
        ],
        'default_region': 'us-east-1',
        'max_bucket_name_length': 63,
        'min_bucket_name_length': 3,
        'allowed_algorithms': ['AES256', 'aws:kms'],
        'default_encryption': 'AES256'
    }

    # Google Cloud Configuration
    GOOGLE_CLOUD_CONFIG = {
        'default_project': None,
        'default_region': 'us-central1',
        'allowed_services': [
            'storage', 'bigquery', 'pubsub', 'cloudfunctions',
            'aiplatform', 'speech', 'translate'
        ],
        'max_project_id_length': 30,
        'min_project_id_length': 6
    }

    # Azure Configuration
    AZURE_CONFIG = {
        'default_region': 'East US',
        'allowed_regions': [
            'East US', 'East US 2', 'West US', 'West US 2',
            'North Europe', 'West Europe', 'Southeast Asia'
        ],
        'max_resource_name_length': 63,
        'min_resource_name_length': 1
    }

    # Slack Configuration
    SLACK_CONFIG = {
        'max_message_length': 40000,
        'max_attachments': 100,
        'max_attachment_size': 20 * 1024 * 1024,  # 20MB
        'allowed_channels': None,  # None means all channels allowed
        'rate_limits': {
            'chat.postMessage': 100,
            'chat.postEphemeral': 100,
            'reactions.add': 100
        }
    }

    # Discord Configuration
    DISCORD_CONFIG = {
        'max_message_length': 2000,
        'max_embed_length': 6000,
        'max_embeds': 10,
        'rate_limits': {
            'webhook': 30,  # per second
            'bot': 5  # per second
        }
    }

    # Database Configuration
    DATABASE_CONFIG = {
        'postgresql': {
            'default_port': 5432,
            'max_connections': 100,
            'connection_timeout': 30,
            'allowed_versions': ['12', '13', '14', '15']
        },
        'mongodb': {
            'default_port': 27017,
            'max_connections': 50,
            'connection_timeout': 30,
            'allowed_versions': ['4.4', '5.0', '6.0', '7.0']
        },
        'redis': {
            'default_port': 6379,
            'max_connections': 200,
            'connection_timeout': 10,
            'allowed_versions': ['6.0', '6.2', '7.0', '7.2']
        },
        'mysql': {
            'default_port': 3306,
            'max_connections': 100,
            'connection_timeout': 30,
            'allowed_versions': ['8.0']
        }
    }

    # API Configuration
    API_CONFIG = {
        'rest': {
            'default_timeout': 30,
            'max_request_size': 10 * 1024 * 1024,  # 10MB
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'],
            'default_headers': {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        },
        'graphql': {
            'default_timeout': 60,
            'max_query_depth': 10,
            'max_query_complexity': 1000,
            'allowed_operations': ['query', 'mutation', 'subscription']
        },
        'websocket': {
            'default_timeout': 30,
            'max_message_size': 64 * 1024,  # 64KB
            'heartbeat_interval': 30,
            'max_connections_per_host': 10
        }
    }


class SecurityPolicies:
    """Security policies for integrations"""

    # Credential Policies
    CREDENTIAL_POLICIES = {
        'min_password_length': 12,
        'require_special_characters': True,
        'require_numbers': True,
        'require_uppercase': True,
        'require_lowercase': True,
        'max_credential_age_days': 90,
        'credential_rotation_required': True
    }

    # Access Control Policies
    ACCESS_POLICIES = {
        'require_mfa_for_sensitive_integrations': True,
        'session_timeout_minutes': 60,
        'max_sessions_per_user': 10,
        'ip_whitelist_enabled': False,
        'ip_blacklist_enabled': True,
        'audit_logging_required': True
    }

    # Data Protection Policies
    DATA_PROTECTION_POLICIES = {
        'encrypt_sensitive_data': True,
        'mask_credentials_in_logs': True,
        'data_retention_days': 90,
        'backup_frequency_hours': 24,
        'require_ssl': True
    }

    # Network Security Policies
    NETWORK_POLICIES = {
        'allowed_origins': ['*'],
        'require_https': True,
        'enable_cors': True,
        'cors_max_age': 86400,  # 24 hours
        'rate_limiting_enabled': True
    }


class ValidationRules:
    """Validation rules for different integration types"""

    # AWS S3 Validation Rules
    AWS_S3_RULES = {
        'bucket_name': {
            'type': 'string',
            'min_length': 3,
            'max_length': 63,
            'pattern': '^[a-z0-9][a-z0-9.-]*[a-z0-9]$'
        },
        'access_key': {
            'type': 'string',
            'min_length': 16,
            'max_length': 128,
            'pattern': '^[A-Z0-9]+$'
        },
        'secret_key': {
            'type': 'string',
            'min_length': 32,
            'max_length': 128
        },
        'region': {
            'type': 'string',
            'pattern': '^[a-z]{2}-[a-z]+-\\d+$'
        }
    }

    # Slack Validation Rules
    SLACK_RULES = {
        'webhook_url': {
            'type': 'string',
            'pattern': '^https://hooks\.slack\.com/workflows/.*$'
        },
        'channel': {
            'type': 'string',
            'pattern': '^#[a-z0-9_-]+$'
        },
        'username': {
            'type': 'string',
            'min_length': 1,
            'max_length': 80
        }
    }

    # Database Validation Rules
    DATABASE_RULES = {
        'postgresql': {
            'host': {
                'type': 'string',
                'min_length': 1,
                'max_length': 253
            },
            'port': {
                'type': 'integer',
                'min': 1,
                'max': 65535
            },
            'database': {
                'type': 'string',
                'min_length': 1,
                'max_length': 63
            },
            'username': {
                'type': 'string',
                'min_length': 1,
                'max_length': 63
            },
            'password': {
                'type': 'string',
                'min_length': 8,
                'max_length': 128
            }
        },
        'mongodb': {
            'connection_string': {
                'type': 'string',
                'pattern': '^mongodb(\+srv)?://.*$'
            },
            'database': {
                'type': 'string',
                'min_length': 1,
                'max_length': 63
            }
        },
        'redis': {
            'host': {
                'type': 'string',
                'min_length': 1,
                'max_length': 253
            },
            'port': {
                'type': 'integer',
                'min': 1,
                'max': 65535
            },
            'password': {
                'type': 'string',
                'min_length': 8,
                'max_length': 128
            }
        }
    }

    # API Validation Rules
    API_RULES = {
        'rest': {
            'endpoint_url': {
                'type': 'string',
                'pattern': '^https?://.*$'
            },
            'token': {
                'type': 'string',
                'min_length': 10,
                'max_length': 1000
            },
            'timeout': {
                'type': 'integer',
                'min': 1,
                'max': 300
            }
        },
        'graphql': {
            'endpoint_url': {
                'type': 'string',
                'pattern': '^https?://.*$'
            },
            'query': {
                'type': 'string',
                'min_length': 10,
                'max_length': 10000
            }
        },
        'websocket': {
            'endpoint_url': {
                'type': 'string',
                'pattern': '^(ws|wss)://.*$'
            },
            'message': {
                'type': 'string',
                'max_length': 65536
            }
        }
    }


# Global configuration instance
integration_config = IntegrationConfig()