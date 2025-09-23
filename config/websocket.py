"""
WebSocket configuration settings
"""

from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class WebSocketSettings(BaseSettings):
    """WebSocket-specific settings."""

    # Connection Settings
    WS_MAX_CONNECTIONS: int = Field(1000, env="WS_MAX_CONNECTIONS")
    WS_MAX_CONNECTIONS_PER_IP: int = Field(10, env="WS_MAX_CONNECTIONS_PER_IP")
    WS_CONNECTION_TIMEOUT: int = Field(30, env="WS_CONNECTION_TIMEOUT")  # seconds
    WS_PING_INTERVAL: int = Field(20, env="WS_PING_INTERVAL")  # seconds
    WS_PING_TIMEOUT: int = Field(10, env="WS_PING_TIMEOUT")  # seconds
    WS_CLOSE_TIMEOUT: int = Field(5, env="WS_CLOSE_TIMEOUT")  # seconds

    # Message Settings
    WS_MAX_MESSAGE_SIZE: int = Field(65536, env="WS_MAX_MESSAGE_SIZE")  # 64KB
    WS_MESSAGE_RATE_LIMIT: int = Field(100, env="WS_MESSAGE_RATE_LIMIT")  # messages per minute
    WS_RATE_LIMIT_WINDOW: int = Field(60, env="WS_RATE_LIMIT_WINDOW")  # seconds

    # Room Settings
    WS_MAX_ROOMS_PER_CONNECTION: int = Field(10, env="WS_MAX_ROOMS_PER_CONNECTION")
    WS_ROOM_CLEANUP_INTERVAL: int = Field(300, env="WS_ROOM_CLEANUP_INTERVAL")  # 5 minutes
    WS_INACTIVE_ROOM_TIMEOUT: int = Field(3600, env="WS_INACTIVE_ROOM_TIMEOUT")  # 1 hour

    # Authentication
    WS_AUTH_REQUIRED: bool = Field(True, env="WS_AUTH_REQUIRED")
    WS_JWT_ALGORITHM: str = Field("HS256", env="WS_JWT_ALGORITHM")
    WS_TOKEN_REFRESH_THRESHOLD: int = Field(300, env="WS_TOKEN_REFRESH_THRESHOLD")  # 5 minutes

    # CORS Settings
    WS_CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        env="WS_CORS_ORIGINS"
    )
    WS_CORS_CREDENTIALS: bool = Field(True, env="WS_CORS_CREDENTIALS")

    # Health Monitoring
    WS_HEALTH_CHECK_INTERVAL: int = Field(30, env="WS_HEALTH_CHECK_INTERVAL")  # seconds
    WS_CONNECTION_CLEANUP_INTERVAL: int = Field(60, env="WS_CONNECTION_CLEANUP_INTERVAL")  # seconds
    WS_METRICS_COLLECTION_INTERVAL: int = Field(10, env="WS_METRICS_COLLECTION_INTERVAL")  # seconds

    # Error Handling
    WS_MAX_RECONNECT_ATTEMPTS: int = Field(5, env="WS_MAX_RECONNECT_ATTEMPTS")
    WS_RECONNECT_BACKOFF_BASE: float = Field(1.0, env="WS_RECONNECT_BACKOFF_BASE")
    WS_RECONNECT_BACKOFF_MAX: int = Field(60, env="WS_RECONNECT_BACKOFF_MAX")

    # Feature Flags
    WS_ENABLE_HEARTBEAT: bool = Field(True, env="WS_ENABLE_HEARTBEAT")
    WS_ENABLE_METRICS: bool = Field(True, env="WS_ENABLE_METRICS")
    WS_ENABLE_LOGGING: bool = Field(True, env="WS_ENABLE_LOGGING")
    WS_ENABLE_COMPRESSION: bool = Field(False, env="WS_ENABLE_COMPRESSION")

    # Buffer Settings
    WS_SEND_BUFFER_SIZE: int = Field(4096, env="WS_SEND_BUFFER_SIZE")
    WS_RECEIVE_BUFFER_SIZE: int = Field(4096, env="WS_RECEIVE_BUFFER_SIZE")

    class Config:
        env_prefix = "WS_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def ping_timeout_total(self) -> int:
        """Total timeout for ping operations."""
        return self.WS_PING_INTERVAL + self.WS_PING_TIMEOUT

    @property
    def connection_timeout_total(self) -> int:
        """Total timeout for connection operations."""
        return self.WS_CONNECTION_TIMEOUT + self.WS_CLOSE_TIMEOUT


# Global WebSocket settings instance
_ws_settings: Optional[WebSocketSettings] = None


def get_websocket_settings() -> WebSocketSettings:
    """Get WebSocket settings."""
    global _ws_settings
    if _ws_settings is None:
        _ws_settings = WebSocketSettings()
    return _ws_settings


def reload_websocket_settings() -> WebSocketSettings:
    """Reload WebSocket settings."""
    global _ws_settings
    _ws_settings = WebSocketSettings()
    return _ws_settings