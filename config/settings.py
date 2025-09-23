"""
Application settings with Pydantic
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = Field(default="TTS API", env="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")

    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
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
    ALLOWED_AUDIO_FORMATS: List[str] = Field(
        default=["mp3", "wav", "flac", "ogg"],
        env="ALLOWED_AUDIO_FORMATS"
    )
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
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    CORS_CREDENTIALS: bool = Field(default=True, env="CORS_CREDENTIALS")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")

    # Feature Flags
    ENABLE_CLOUD_STORAGE: bool = Field(default=False, env="ENABLE_CLOUD_STORAGE")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    ENABLE_HEALTH_CHECKS: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    ENABLE_API_DOCS: bool = Field(default=True, env="ENABLE_API_DOCS")

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

    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from environment variable."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return ["http://localhost:3000", "http://localhost:8080"]

    @validator("ALLOWED_AUDIO_FORMATS", pre=True)
    def assemble_audio_formats(cls, v):
        """Parse allowed audio formats from environment variable."""
        if isinstance(v, str):
            return [i.strip().lower() for i in v.split(",")]
        return v

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