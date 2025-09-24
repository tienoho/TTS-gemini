"""
Gemini API configuration for TTS system
"""

import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings


class GeminiConfig(BaseSettings):
    """Gemini API configuration settings."""

    # API Configuration
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    GEMINI_API_VERSION: str = os.getenv('GEMINI_API_VERSION', 'v1')

    # Rate Limiting
    GEMINI_REQUESTS_PER_MINUTE: int = int(os.getenv('GEMINI_REQUESTS_PER_MINUTE', '60'))
    GEMINI_REQUESTS_PER_HOUR: int = int(os.getenv('GEMINI_REQUESTS_PER_HOUR', '1000'))
    GEMINI_BURST_LIMIT: int = int(os.getenv('GEMINI_BURST_LIMIT', '10'))

    # Audio Quality Settings
    DEFAULT_AUDIO_FORMAT: str = os.getenv('DEFAULT_AUDIO_FORMAT', 'mp3')
    DEFAULT_SAMPLE_RATE: int = int(os.getenv('DEFAULT_SAMPLE_RATE', '22050'))
    DEFAULT_AUDIO_ENCODING: str = os.getenv('DEFAULT_AUDIO_ENCODING', 'LINEAR16')

    # Supported Audio Formats
    SUPPORTED_AUDIO_FORMATS: List[str] = ['mp3', 'wav', 'ogg', 'flac', 'webm']

    # Voice and Language Settings
    DEFAULT_VOICE_NAME: str = os.getenv('DEFAULT_VOICE_NAME', 'Alnilam')
    DEFAULT_LANGUAGE_CODE: str = os.getenv('DEFAULT_LANGUAGE_CODE', 'vi-VN')

    # Language to Voice Mappings
    LANGUAGE_VOICE_MAP: Dict[str, List[str]] = {
        'vi-VN': ['Alnilam', 'Sirius', 'Vega', 'Altair'],
        'en-US': ['Alnilam', 'Sirius', 'Vega', 'Altair'],
        'ja-JP': ['Alnilam', 'Sirius'],
        'ko-KR': ['Alnilam', 'Sirius'],
        'zh-CN': ['Alnilam', 'Sirius'],
        'fr-FR': ['Alnilam', 'Sirius'],
        'de-DE': ['Alnilam', 'Sirius'],
        'es-ES': ['Alnilam', 'Sirius'],
        'pt-BR': ['Alnilam', 'Sirius'],
        'ru-RU': ['Alnilam', 'Sirius'],
        'ar-SA': ['Alnilam', 'Sirius'],
        'hi-IN': ['Alnilam', 'Sirius'],
        'th-TH': ['Alnilam', 'Sirius'],
        'id-ID': ['Alnilam', 'Sirius'],
    }

    # Voice Gender Mappings
    VOICE_GENDER_MAP: Dict[str, str] = {
        'Alnilam': 'neutral',
        'Sirius': 'male',
        'Vega': 'female',
        'Altair': 'neutral',
    }

    # Circuit Breaker Settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5'))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = int(os.getenv('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60'))
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = os.getenv('CIRCUIT_BREAKER_EXPECTED_EXCEPTION', 'Exception')

    # Retry Settings
    RETRY_MAX_ATTEMPTS: int = int(os.getenv('RETRY_MAX_ATTEMPTS', '3'))
    RETRY_BASE_DELAY: float = float(os.getenv('RETRY_BASE_DELAY', '1.0'))
    RETRY_MAX_DELAY: float = float(os.getenv('RETRY_MAX_DELAY', '60.0'))
    RETRY_BACKOFF_MULTIPLIER: float = float(os.getenv('RETRY_BACKOFF_MULTIPLIER', '2.0'))

    # Cache Settings
    CACHE_TTL_VOICE_MAPPINGS: int = int(os.getenv('CACHE_TTL_VOICE_MAPPINGS', '3600'))  # 1 hour
    CACHE_TTL_API_RESPONSES: int = int(os.getenv('CACHE_TTL_API_RESPONSES', '300'))    # 5 minutes

    # Request Settings
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_TEXT_LENGTH: int = int(os.getenv('MAX_TEXT_LENGTH', '5000'))
    MAX_AUDIO_FILE_SIZE: int = int(os.getenv('MAX_AUDIO_FILE_SIZE', '10485760'))  # 10MB

    # File Storage
    TEMP_AUDIO_DIR: str = os.getenv('TEMP_AUDIO_DIR', 'temp/audio')
    AUDIO_STORAGE_DIR: str = os.getenv('AUDIO_STORAGE_DIR', 'storage/audio')
    CLEANUP_INTERVAL: int = int(os.getenv('CLEANUP_INTERVAL', '3600'))  # 1 hour

    # Quality Enhancement
    ENABLE_NOISE_REDUCTION: bool = os.getenv('ENABLE_NOISE_REDUCTION', 'true').lower() == 'true'
    ENABLE_AUDIO_NORMALIZATION: bool = os.getenv('ENABLE_AUDIO_NORMALIZATION', 'true').lower() == 'true'
    ENABLE_VOICE_ENHANCEMENT: bool = os.getenv('ENABLE_VOICE_ENHANCEMENT', 'true').lower() == 'true'

    # Validation Settings
    MIN_TEXT_LENGTH: int = int(os.getenv('MIN_TEXT_LENGTH', '1'))
    MAX_SSML_LENGTH: int = int(os.getenv('MAX_SSML_LENGTH', '10000'))

    # Error Handling
    ENABLE_DETAILED_ERROR_MESSAGES: bool = os.getenv('ENABLE_DETAILED_ERROR_MESSAGES', 'false').lower() == 'true'

    class Config:
        """Pydantic configuration."""
        env_file = '.env'
        case_sensitive = False

    def get_available_voices_for_language(self, language_code: str) -> List[str]:
        """Get available voices for a specific language."""
        return self.LANGUAGE_VOICE_MAP.get(language_code, [self.DEFAULT_VOICE_NAME])

    def get_voice_gender(self, voice_name: str) -> str:
        """Get gender for a specific voice."""
        return self.VOICE_GENDER_MAP.get(voice_name, 'neutral')

    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported."""
        return language_code in self.LANGUAGE_VOICE_MAP

    def is_voice_supported(self, voice_name: str, language_code: str) -> bool:
        """Check if a voice is supported for a specific language."""
        available_voices = self.get_available_voices_for_language(language_code)
        return voice_name in available_voices

    def get_supported_languages(self) -> List[str]:
        """Get list of all supported languages."""
        return list(self.LANGUAGE_VOICE_MAP.keys())


# Global configuration instance - commented out to avoid pydantic_settings error
# gemini_config = GeminiConfig()