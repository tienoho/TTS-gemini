"""
Batch Processing Configuration
"""

import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator


class BatchProcessingConfig(BaseModel):
    """
    Configuration cho batch processing system
    """

    # Batch Size Limits
    MAX_BATCH_SIZE: int = Field(default=100, description="Maximum items per batch")
    MIN_BATCH_SIZE: int = Field(default=1, description="Minimum items per batch")
    DEFAULT_CHUNK_SIZE: int = Field(default=25, description="Default chunk size for processing")

    # Processing Concurrency
    MAX_CONCURRENCY: int = Field(default=5, description="Maximum parallel processing workers")
    MIN_CONCURRENCY: int = Field(default=1, description="Minimum parallel processing workers")
    DEFAULT_CONCURRENCY: int = Field(default=3, description="Default parallel processing workers")

    # Timeout Settings
    BATCH_TIMEOUT_SECONDS: int = Field(default=3600, description="Batch processing timeout (1 hour)")
    ITEM_TIMEOUT_SECONDS: int = Field(default=300, description="Individual item timeout (5 minutes)")
    QUEUE_TIMEOUT_SECONDS: int = Field(default=30, description="Queue operation timeout")

    # Retry Policies
    MAX_RETRIES: int = Field(default=3, description="Maximum retry attempts per item")
    RETRY_DELAY_BASE: float = Field(default=1.0, description="Base delay between retries (seconds)")
    RETRY_DELAY_MAX: float = Field(default=60.0, description="Maximum delay between retries")

    # Priority Settings
    HIGH_PRIORITY_THRESHOLD: int = Field(default=10, description="Batch size threshold for high priority")
    LOW_PRIORITY_THRESHOLD: int = Field(default=50, description="Batch size threshold for low priority")

    # Resource Allocation
    MAX_MEMORY_USAGE_MB: int = Field(default=1024, description="Maximum memory usage per batch")
    MAX_CPU_USAGE_PERCENT: int = Field(default=80, description="Maximum CPU usage percentage")

    # Queue Settings
    QUEUE_CLEANUP_INTERVAL: int = Field(default=300, description="Queue cleanup interval (seconds)")
    DEAD_LETTER_RETENTION_DAYS: int = Field(default=7, description="Dead letter queue retention period")

    # Progress Updates
    PROGRESS_UPDATE_INTERVAL: float = Field(default=1.0, description="Progress update interval (seconds)")
    ENABLE_REAL_TIME_UPDATES: bool = Field(default=True, description="Enable real-time progress updates")

    # Storage Settings
    TEMP_FILE_RETENTION_HOURS: int = Field(default=24, description="Temporary file retention period")
    MAX_STORAGE_PER_BATCH_MB: int = Field(default=100, description="Maximum storage per batch")

    # Rate Limiting
    BATCHES_PER_MINUTE: int = Field(default=10, description="Maximum batches per minute")
    ITEMS_PER_MINUTE: int = Field(default=500, description="Maximum items per minute")

    # Quality of Service
    ENABLE_AUDIO_ENHANCEMENT: bool = Field(default=True, description="Enable audio quality enhancement")
    COMPRESSION_QUALITY: str = Field(default="high", description="Audio compression quality")

    # Monitoring and Logging
    ENABLE_DETAILED_LOGGING: bool = Field(default=False, description="Enable detailed batch processing logs")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level for batch operations")

    # WebSocket Settings
    WS_BATCH_UPDATE_INTERVAL: float = Field(default=0.5, description="WebSocket batch update interval")
    WS_MAX_CONNECTIONS_PER_BATCH: int = Field(default=100, description="Max WebSocket connections per batch")

    # Error Handling
    FAIL_FAST_ON_ERROR: bool = Field(default=False, description="Stop batch on first error")
    ERROR_THRESHOLD_PERCENT: float = Field(default=10.0, description="Error threshold percentage to fail batch")

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, description="Circuit breaker failure threshold")
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=60, description="Circuit breaker recovery timeout")

    # Cache Settings
    CACHE_TTL_BATCH_STATUS: int = Field(default=86400, description="Batch status cache TTL (24 hours)")
    CACHE_TTL_PROGRESS: int = Field(default=3600, description="Progress cache TTL (1 hour)")

    # Validation
    @validator('MAX_BATCH_SIZE')
    def validate_max_batch_size(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('MAX_BATCH_SIZE must be between 1 and 1000')
        return v

    @validator('MAX_CONCURRENCY')
    def validate_max_concurrency(cls, v):
        if v < 1 or v > 50:
            raise ValueError('MAX_CONCURRENCY must be between 1 and 50')
        return v

    @validator('BATCH_TIMEOUT_SECONDS')
    def validate_batch_timeout(cls, v):
        if v < 60 or v > 86400:  # 1 minute to 24 hours
            raise ValueError('BATCH_TIMEOUT_SECONDS must be between 60 and 86400')
        return v



class BatchEnvironmentConfig:
    """
    Environment-specific batch configurations
    """

    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Development environment configuration"""
        return {
            "MAX_CONCURRENCY": 2,
            "MAX_BATCH_SIZE": 10,
            "BATCH_TIMEOUT_SECONDS": 1800,  # 30 minutes
            "ENABLE_DETAILED_LOGGING": True,
            "LOG_LEVEL": "DEBUG",
            "FAIL_FAST_ON_ERROR": True,
            "ENABLE_REAL_TIME_UPDATES": True
        }

    @staticmethod
    def get_testing_config() -> Dict[str, Any]:
        """Testing environment configuration"""
        return {
            "MAX_CONCURRENCY": 1,
            "MAX_BATCH_SIZE": 5,
            "BATCH_TIMEOUT_SECONDS": 300,  # 5 minutes
            "ITEM_TIMEOUT_SECONDS": 30,  # 30 seconds
            "ENABLE_DETAILED_LOGGING": True,
            "LOG_LEVEL": "DEBUG",
            "FAIL_FAST_ON_ERROR": True,
            "ENABLE_REAL_TIME_UPDATES": False,
            "ENABLE_AUDIO_ENHANCEMENT": False
        }

    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Production environment configuration"""
        return {
            "MAX_CONCURRENCY": 10,
            "MAX_BATCH_SIZE": 100,
            "BATCH_TIMEOUT_SECONDS": 7200,  # 2 hours
            "ITEM_TIMEOUT_SECONDS": 600,  # 10 minutes
            "ENABLE_DETAILED_LOGGING": False,
            "LOG_LEVEL": "WARNING",
            "FAIL_FAST_ON_ERROR": False,
            "ENABLE_REAL_TIME_UPDATES": True,
            "ENABLE_AUDIO_ENHANCEMENT": True,
            "COMPRESSION_QUALITY": "high",
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD": 10,
            "CIRCUIT_BREAKER_RECOVERY_TIMEOUT": 300
        }

    @staticmethod
    def get_staging_config() -> Dict[str, Any]:
        """Staging environment configuration"""
        return {
            "MAX_CONCURRENCY": 5,
            "MAX_BATCH_SIZE": 50,
            "BATCH_TIMEOUT_SECONDS": 3600,  # 1 hour
            "ENABLE_DETAILED_LOGGING": True,
            "LOG_LEVEL": "INFO",
            "FAIL_FAST_ON_ERROR": False,
            "ENABLE_REAL_TIME_UPDATES": True,
            "ENABLE_AUDIO_ENHANCEMENT": True,
            "COMPRESSION_QUALITY": "medium"
        }


class BatchConfigManager:
    """
    Manager cho batch configuration với environment-specific overrides
    """

    def __init__(self):
        self.base_config = BatchProcessingConfig()
        self.env_config = self._load_environment_config()

    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env = os.getenv('FLASK_ENV', 'development').lower()

        config_map = {
            'development': BatchEnvironmentConfig.get_development_config(),
            'testing': BatchEnvironmentConfig.get_testing_config(),
            'production': BatchEnvironmentConfig.get_production_config(),
            'staging': BatchEnvironmentConfig.get_staging_config()
        }

        return config_map.get(env, {})

    def get_config(self) -> BatchProcessingConfig:
        """Get merged configuration"""
        # Start with base config
        config_dict = self.base_config.model_dump()

        # Apply environment overrides
        for key, value in self.env_config.items():
            if hasattr(self.base_config, key):
                config_dict[key] = value

        return BatchProcessingConfig(**config_dict)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get specific configuration value"""
        # Check environment config first
        if key in self.env_config:
            return self.env_config[key]

        # Check base config
        if hasattr(self.base_config, key):
            return getattr(self.base_config, key)

        return default

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        feature_map = {
            'real_time_updates': self.get_config_value('ENABLE_REAL_TIME_UPDATES', True),
            'detailed_logging': self.get_config_value('ENABLE_DETAILED_LOGGING', False),
            'audio_enhancement': self.get_config_value('ENABLE_AUDIO_ENHANCEMENT', True),
            'fail_fast': self.get_config_value('FAIL_FAST_ON_ERROR', False)
        }

        return feature_map.get(feature, False)

    def get_queue_config(self) -> Dict[str, Any]:
        """Get queue-specific configuration"""
        return {
            'max_concurrency': self.get_config_value('MAX_CONCURRENCY'),
            'chunk_size': self.get_config_value('DEFAULT_CHUNK_SIZE'),
            'timeout': self.get_config_value('QUEUE_TIMEOUT_SECONDS'),
            'cleanup_interval': self.get_config_value('QUEUE_CLEANUP_INTERVAL'),
            'dead_letter_retention_days': self.get_config_value('DEAD_LETTER_RETENTION_DAYS')
        }

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing-specific configuration"""
        return {
            'max_retries': self.get_config_value('MAX_RETRIES'),
            'retry_delay_base': self.get_config_value('RETRY_DELAY_BASE'),
            'retry_delay_max': self.get_config_value('RETRY_DELAY_MAX'),
            'batch_timeout': self.get_config_value('BATCH_TIMEOUT_SECONDS'),
            'item_timeout': self.get_config_value('ITEM_TIMEOUT_SECONDS'),
            'fail_fast': self.get_config_value('FAIL_FAST_ON_ERROR'),
            'error_threshold': self.get_config_value('ERROR_THRESHOLD_PERCENT')
        }

    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource allocation configuration"""
        return {
            'max_memory_mb': self.get_config_value('MAX_MEMORY_USAGE_MB'),
            'max_cpu_percent': self.get_config_value('MAX_CPU_USAGE_PERCENT'),
            'max_storage_mb': self.get_config_value('MAX_STORAGE_PER_BATCH_MB'),
            'temp_file_retention_hours': self.get_config_value('TEMP_FILE_RETENTION_HOURS')
        }


# Global batch configuration instance
batch_config_manager = BatchConfigManager()


def get_batch_config() -> BatchProcessingConfig:
    """Get batch processing configuration"""
    return batch_config_manager.get_config()


def get_batch_config_manager() -> BatchConfigManager:
    """Get batch configuration manager"""
    return batch_config_manager


# Utility functions for easy access
def get_max_batch_size() -> int:
    """Get maximum batch size"""
    return batch_config_manager.get_config_value('MAX_BATCH_SIZE', 100)


def get_max_concurrency() -> int:
    """Get maximum processing concurrency"""
    return batch_config_manager.get_config_value('MAX_CONCURRENCY', 5)


def get_batch_timeout() -> int:
    """Get batch processing timeout"""
    return batch_config_manager.get_config_value('BATCH_TIMEOUT_SECONDS', 3600)


def is_real_time_updates_enabled() -> bool:
    """Check if real-time updates are enabled"""
    return batch_config_manager.is_feature_enabled('real_time_updates')


def should_fail_fast() -> bool:
    """Check if batch should fail fast on errors"""
    return batch_config_manager.is_feature_enabled('fail_fast')