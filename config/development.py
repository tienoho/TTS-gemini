"""
Development configuration for Flask TTS API
"""

import os
from datetime import timedelta

from pydantic_settings import BaseSettings


class DevelopmentConfig(BaseSettings):
    """Development configuration settings."""

    # Flask settings
    DEBUG: bool = True
    TESTING: bool = False
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-secret-key-change-this')

    # Database
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        'DATABASE_URL', 'sqlite:///dev_tts_api.db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

    # JWT settings
    JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret')
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(days=30)

    # Redis
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    # Gemini API
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')

    # Audio settings
    MAX_AUDIO_FILE_SIZE: int = int(os.getenv('MAX_AUDIO_FILE_SIZE', '10485760'))  # 10MB
    SUPPORTED_AUDIO_FORMATS: list = ['mp3', 'wav', 'ogg', 'flac']
    DEFAULT_VOICE_NAME: str = os.getenv('DEFAULT_VOICE_NAME', 'Alnilam')
    MAX_TEXT_LENGTH: int = int(os.getenv('MAX_TEXT_LENGTH', '5000'))

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
    RATE_LIMIT_PREMIUM_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PREMIUM_PER_MINUTE', '1000'))

    # File storage
    UPLOAD_FOLDER: str = os.getenv('UPLOAD_FOLDER', 'uploads/audio')
    MAX_CONTENT_LENGTH: int = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB

    # CORS
    CORS_ORIGINS: list = os.getenv(
        'CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080'
    ).split(',')

    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'DEBUG')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/dev_tts_api.log')

    # Monitoring
    ENABLE_MONITORING: bool = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'

    class Config:
        """Pydantic configuration."""
        env_file = '.env'
        case_sensitive = False