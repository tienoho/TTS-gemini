"""
Production configuration for Flask TTS API
"""

import os
from datetime import timedelta


class ProductionConfig:
    """Production configuration settings."""

    # Flask settings
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY')

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20,
    }

    # JWT settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=30)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)

    # Redis
    REDIS_URL = os.getenv('REDIS_URL')

    # Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Audio settings
    MAX_AUDIO_FILE_SIZE = int(os.getenv('MAX_AUDIO_FILE_SIZE', '10485760'))  # 10MB
    SUPPORTED_AUDIO_FORMATS = ['mp3', 'wav', 'ogg', 'flac']
    DEFAULT_VOICE_NAME = os.getenv('DEFAULT_VOICE_NAME', 'Alnilam')
    MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '5000'))

    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
    RATE_LIMIT_PREMIUM_PER_MINUTE = int(os.getenv('RATE_LIMIT_PREMIUM_PER_MINUTE', '1000'))

    # File storage
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/uploads/audio')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB

    # CORS - Restrictive in production
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '').split(',') if os.getenv('CORS_ORIGINS') else []

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', '/app/logs/tts_api.log')

    # Monitoring
    ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'