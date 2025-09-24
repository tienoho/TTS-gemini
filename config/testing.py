"""
Testing configuration for Flask TTS API
"""

from datetime import timedelta


class TestingConfig:
    """Testing configuration settings."""

    # Flask settings
    DEBUG = True
    TESTING = True
    SECRET_KEY = 'test-secret-key'

    # Database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test_tts_api.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
    }

    # JWT settings
    JWT_SECRET_KEY = 'test-jwt-secret'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(hours=1)

    # Redis (use in-memory for testing)
    REDIS_URL = 'redis://localhost:6379/1'

    # Gemini API (use mock for testing)
    GEMINI_API_KEY = 'test-gemini-api-key'

    # Audio settings
    MAX_AUDIO_FILE_SIZE = 1048576  # 1MB for testing
    SUPPORTED_AUDIO_FORMATS = ['mp3', 'wav']
    DEFAULT_VOICE_NAME = 'Alnilam'
    MAX_TEXT_LENGTH = 1000  # Shorter for testing

    # Rate limiting (more permissive for testing)
    RATE_LIMIT_PER_MINUTE = 1000
    RATE_LIMIT_PREMIUM_PER_MINUTE = 5000

    # File storage
    UPLOAD_FOLDER = 'test_uploads/audio'
    MAX_CONTENT_LENGTH = 2097152  # 2MB for testing

    # CORS (allow all for testing)
    CORS_ORIGINS = ['*']

    # Logging
    LOG_LEVEL = 'DEBUG'
    LOG_FILE = 'logs/test_tts_api.log'

    # Monitoring (disabled for testing)
    ENABLE_MONITORING = False

    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False