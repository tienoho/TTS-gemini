"""
Testing configuration for Flask TTS API
"""

import os
from datetime import timedelta

from pydantic_settings import BaseSettings


class TestingConfig(BaseSettings):
    """Testing configuration settings."""

    # Flask settings
    DEBUG: bool = True
    TESTING: bool = True
    SECRET_KEY: str = 'test-secret-key'

    # Database
    SQLALCHEMY_DATABASE_URI: str = 'sqlite:///test_tts_api.db'
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        'pool_pre_ping': True,
    }

    # JWT settings
    JWT_SECRET_KEY: str = 'test-jwt-secret'
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(minutes=5)
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(hours=1)

    # Redis (use in-memory for testing)
    REDIS_URL: str = 'redis://localhost:6379/1'

    # Gemini API (use mock for testing)
    GEMINI_API_KEY: str = 'test-gemini-api-key'

    # Audio settings
    MAX_AUDIO_FILE_SIZE: int = 1048576  # 1MB for testing
    SUPPORTED_AUDIO_FORMATS: list = ['mp3', 'wav']
    DEFAULT_VOICE_NAME: str = 'Alnilam'
    MAX_TEXT_LENGTH: int = 1000  # Shorter for testing

    # Rate limiting (more permissive for testing)
    RATE_LIMIT_PER_MINUTE: int = 1000
    RATE_LIMIT_PREMIUM_PER_MINUTE: int = 5000

    # File storage
    UPLOAD_FOLDER: str = 'test_uploads/audio'
    MAX_CONTENT_LENGTH: int = 2097152  # 2MB for testing

    # CORS (allow all for testing)
    CORS_ORIGINS: list = ['*']

    # Logging
    LOG_LEVEL: str = 'DEBUG'
    LOG_FILE: str = 'logs/test_tts_api.log'

    # Monitoring (disabled for testing)
    ENABLE_MONITORING: bool = False

    # Disable CSRF for testing
    WTF_CSRF_ENABLED: bool = False

    class Config:
        """Pydantic configuration."""
        env_file = '.env.test'
        case_sensitive = False