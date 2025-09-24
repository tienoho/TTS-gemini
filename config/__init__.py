"""
Configuration package for Flask TTS API
"""

from .development import DevelopmentConfig
from .production import ProductionConfig
from .testing import TestingConfig

__all__ = [
    'DevelopmentConfig',
    'ProductionConfig', 
    'TestingConfig'
]