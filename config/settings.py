"""
Application settings with Pydantic
"""

import os
from typing import Optional, List, Annotated
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = Field(default="TTS API", env="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")

    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=5000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")

    # Database
    DATABASE_URL: str = Field(
        default="postgresql://tts_user:tts_password@localhost/tts_db",
        env="DATABASE_URL"
    )
    SQLALCHEMY_DATABASE_URI: str = Field(
        default="postgresql://tts_user:tts_password@localhost/tts_db",
        env="SQLALCHEMY_DATABASE_URI"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    REDIS_POOL_SIZE: int = Field(default=20, env="REDIS_POOL_SIZE")

    # JWT
    JWT_SECRET_KEY: str = Field(default="your-jwt-secret-key", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")

    # Gemini API
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GEMINI_BASE_URL: str = Field(default="https://generativelanguage.googleapis.com", env="GEMINI_BASE_URL")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    RATE_LIMIT_STORAGE_PER_HOUR: int = Field(default=100000000, env="RATE_LIMIT_STORAGE_PER_HOUR")  # 100MB
    RATE_LIMIT_DOWNLOADS_PER_HOUR: int = Field(default=50, env="RATE_LIMIT_DOWNLOADS_PER_HOUR")

    # File Storage
    MAX_FILE_SIZE_MB: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    ALLOWED_AUDIO_FORMATS: List[str] = ["mp3", "wav", "flac", "ogg"]
    UPLOAD_FOLDER: str = Field(default="uploads", env="UPLOAD_FOLDER")
    OUTPUT_FOLDER: str = Field(default="outputs", env="OUTPUT_FOLDER")

    # Cloud Storage
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    AWS_S3_BUCKET: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")
    AWS_S3_REGION: str = Field(default="us-east-1", env="AWS_S3_REGION")

    GCS_BUCKET: Optional[str] = Field(default=None, env="GCS_BUCKET")
    GCS_CREDENTIALS_FILE: Optional[str] = Field(default=None, env="GCS_CREDENTIALS_FILE")

    # Email (for notifications)
    SMTP_SERVER: Optional[str] = Field(default=None, env="SMTP_SERVER")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    EMAIL_FROM: str = Field(default="noreply@tts-api.com", env="EMAIL_FROM")

    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")

    # Security  
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_CREDENTIALS: bool = True

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")

    # Feature Flags
    ENABLE_CLOUD_STORAGE: bool = Field(default=False, env="ENABLE_CLOUD_STORAGE")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    ENABLE_HEALTH_CHECKS: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    ENABLE_API_DOCS: bool = Field(default=True, env="ENABLE_API_DOCS")
    ENABLE_MONITORING: bool = Field(default=False, env="ENABLE_MONITORING")

    # CORS Settings
    cors_methods_internal: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS",
        env="CORS_METHODS",
        alias="CORS_METHODS"
    )
    cors_headers_internal: str = Field(
        default="Content-Type,Authorization,X-Requested-With",
        env="CORS_HEADERS",
        alias="CORS_HEADERS"
    )

    @property
    def CORS_METHODS(self) -> List[str]:
        """Get CORS methods as list."""
        return [i.strip() for i in self.cors_methods_internal.split(",")]

    @property
    def CORS_HEADERS(self) -> List[str]:
        """Get CORS headers as list."""
        return [i.strip() for i in self.cors_headers_internal.split(",")]

    # Environment
    TESTING: bool = Field(default=False, env="TESTING")
    COMPOSE_PROJECT_NAME: str = Field(default="tts-api-dev", env="COMPOSE_PROJECT_NAME")

    # Worker Settings
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        env="CELERY_BROKER_URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        env="CELERY_RESULT_BACKEND"
    )
    CELERY_WORKER_CONCURRENCY: int = Field(default=4, env="CELERY_WORKER_CONCURRENCY")

    # Cache Settings
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=60, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")

    # Retry Settings
    RETRY_ATTEMPTS: int = Field(default=3, env="RETRY_ATTEMPTS")
    RETRY_BACKOFF_FACTOR: float = Field(default=2.0, env="RETRY_BACKOFF_FACTOR")
    RETRY_MAX_DELAY: int = Field(default=300, env="RETRY_MAX_DELAY")  # 5 minutes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False



    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.DEBUG or os.getenv("ENVIRONMENT") == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.is_development


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload application settings."""
    global _settings
    _settings = Settings()
    return _settings


class PluginConfig(BaseSettings):
    """Plugin configuration settings."""

    # Flask Application
    flask_app: str = Field(default="app", env="FLASK_APP")
    flask_env: str = Field(default="development", env="FLASK_ENV")

    # Security
    secret_key: str = Field(default="your-plugin-secret-key", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="your-plugin-jwt-secret", env="JWT_SECRET_KEY")
    jwt_access_token_expires: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRES")
    jwt_refresh_token_expires: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRES")

    # Database
    database_url: str = Field(
        default="postgresql://plugin_user:plugin_password@localhost/plugin_db",
        env="DATABASE_URL"
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )

    # Gemini API
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")

    # File Configuration
    max_audio_file_size: int = Field(default=50, env="MAX_AUDIO_FILE_SIZE")  # MB
    supported_audio_formats: str = Field(default="mp3,wav,flac,ogg", env="SUPPORTED_AUDIO_FORMATS")
    default_voice_name: str = Field(default="default", env="DEFAULT_VOICE_NAME")
    max_text_length: int = Field(default=5000, env="MAX_TEXT_LENGTH")
    upload_folder: str = Field(default="uploads", env="UPLOAD_FOLDER")
    max_content_length: int = Field(default=52428800, env="MAX_CONTENT_LENGTH")  # 50MB

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_premium_per_minute: int = Field(default=120, env="RATE_LIMIT_PREMIUM_PER_MINUTE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="plugin.log", env="LOG_FILE")
    log_max_size: int = Field(default=10485760, env="LOG_MAX_SIZE")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    # CORS
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8080", env="CORS_ORIGINS")
    cors_methods: str = Field(default="GET,POST,PUT,DELETE,OPTIONS", env="CORS_METHODS")
    cors_headers: str = Field(default="Content-Type,Authorization,X-Requested-With", env="CORS_HEADERS")

    # Environment
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    compose_project_name: str = Field(default="tts-plugin-dev", env="COMPOSE_PROJECT_NAME")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False



    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or self.flask_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.is_development