"""
API Routes for Flask TTS API
"""

from .auth import auth_bp
from .tts import tts_bp
from .websocket import register_websocket_handlers
from .analytics import analytics_bp
from .plugin import plugin_bp

__all__ = ['auth_bp', 'tts_bp', 'register_websocket_handlers', 'analytics_bp', 'plugin_bp']