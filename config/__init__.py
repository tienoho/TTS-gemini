"""
Configuration package for Flask TTS API
"""

from .development import DevelopmentConfig
from .production import ProductionConfig
from .testing import TestingConfig
from .settings import get_settings, Settings
from .plugin import PluginConfig, plugin_config

__all__ = [
    'DevelopmentConfig',
    'ProductionConfig',
    'TestingConfig',
    'get_settings',
    'Settings',
    'PluginConfig',
    'plugin_config'
]