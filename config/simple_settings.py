"""
Simple settings without Pydantic for temporary use
"""
import os
from typing import List, Optional

class SimpleSettings:
    """Simple settings class without Pydantic validation."""
    
    # Application
    APP_NAME: str = "TTS API"
    APP_VERSION: str = "1.0.0" 
    DEBUG: bool = False
    SECRET_KEY: str = "your-secret-key-here"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    WORKERS: int = 4

    # Database
    DATABASE_URL: str = "postgresql://tts_user:tts_password@localhost/tts_db"
    SQLALCHEMY_DATABASE_URI: str = "postgresql://tts_user:tts_password@localhost/tts_db" 
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 20

    # JWT
    JWT_SECRET_KEY: str = "your-jwt-secret-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_STORAGE_PER_HOUR: int = 1000
    RATE_LIMIT_DOWNLOADS_PER_HOUR: int = 500

    # File Storage
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_AUDIO_FORMATS: List[str] = ["mp3", "wav", "flac", "ogg"]
    UPLOAD_FOLDER: str = "uploads"
    OUTPUT_FOLDER: str = "outputs"

    # Cloud Storage
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_S3_REGION: str = "us-east-1"
    
    GCS_BUCKET: Optional[str] = None
    GCS_CREDENTIALS_FILE: Optional[str] = None

    # Email
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAIL_FROM: str = "noreply@tts-api.com"

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None

    # Security
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_CREDENTIALS: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[str] = None

    # Feature Flags
    ENABLE_CLOUD_STORAGE: bool = False
    ENABLE_METRICS: bool = True
    ENABLE_HEALTH_CHECKS: bool = True
    ENABLE_API_DOCS: bool = True
    ENABLE_MONITORING: bool = False

    # CORS Settings
    CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS: List[str] = ["Content-Type", "Authorization", "X-Requested-With"]

    # Environment
    TESTING: bool = False
    COMPOSE_PROJECT_NAME: str = "tts-api-dev"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_WORKER_CONCURRENCY: int = 4

    # Cache
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_ENABLED: bool = True

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60

    # Retry
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_FACTOR: float = 2.0
    RETRY_MAX_DELAY: int = 60

    # WebSocket Settings
    WS_MAX_CONNECTIONS: int = 1000
    WS_MAX_CONNECTIONS_PER_IP: int = 10
    WS_CONNECTION_TIMEOUT: int = 30
    WS_PING_INTERVAL: int = 20
    WS_PING_TIMEOUT: int = 10
    WS_CLOSE_TIMEOUT: int = 5
    WS_MAX_MESSAGE_SIZE: int = 65536
    WS_MESSAGE_RATE_LIMIT: int = 100
    WS_RATE_LIMIT_WINDOW: int = 60
    WS_MAX_ROOMS_PER_CONNECTION: int = 10
    WS_ROOM_CLEANUP_INTERVAL: int = 300
    WS_INACTIVE_ROOM_TIMEOUT: int = 3600
    WS_AUTH_REQUIRED: bool = True
    WS_JWT_ALGORITHM: str = "HS256"
    WS_TOKEN_REFRESH_THRESHOLD: int = 300
    WS_CORS_CREDENTIALS: bool = True
    WS_HEALTH_CHECK_INTERVAL: int = 30
    WS_CONNECTION_CLEANUP_INTERVAL: int = 60
    WS_METRICS_COLLECTION_INTERVAL: int = 10
    WS_MAX_RECONNECT_ATTEMPTS: int = 5
    WS_RECONNECT_BACKOFF_BASE: float = 1.0
    WS_RECONNECT_BACKOFF_MAX: int = 60
    WS_ENABLE_HEARTBEAT: bool = True
    WS_ENABLE_METRICS: bool = True
    WS_ENABLE_LOGGING: bool = True
    WS_ENABLE_COMPRESSION: bool = False
    WS_SEND_BUFFER_SIZE: int = 4096
    WS_RECEIVE_BUFFER_SIZE: int = 4096

    def __init__(self):
        # Load from environment variables if available
        self._load_from_env()
    
    @property
    def ping_timeout_total(self) -> int:
        """Total timeout for ping operations."""
        return self.WS_PING_INTERVAL + self.WS_PING_TIMEOUT

    @property  
    def connection_timeout_total(self) -> int:
        """Total timeout for connection operations."""
        return self.WS_CONNECTION_TIMEOUT + self.WS_CLOSE_TIMEOUT
    
    def _load_from_env(self):
        """Load settings from environment variables."""
        # Load common environment variables
        self.DATABASE_URL = os.getenv('DATABASE_URL', self.DATABASE_URL)
        self.SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', self.DATABASE_URL)
        self.REDIS_URL = os.getenv('REDIS_URL', self.REDIS_URL)
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
        self.SECRET_KEY = os.getenv('SECRET_KEY', self.SECRET_KEY)
        self.JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', self.JWT_SECRET_KEY)
        
        # Load CORS origins from env
        cors_origins_env = os.getenv('CORS_ORIGINS')
        if cors_origins_env:
            self.CORS_ORIGINS = [origin.strip() for origin in cors_origins_env.split(',')]

# Global settings instance
_simple_settings = None

def get_simple_settings() -> SimpleSettings:
    """Get simple settings instance."""
    global _simple_settings
    if _simple_settings is None:
        _simple_settings = SimpleSettings()
    return _simple_settings