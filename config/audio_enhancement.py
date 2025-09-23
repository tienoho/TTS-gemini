"""
Audio Enhancement Configuration for TTS system
Provides comprehensive configuration for audio enhancement features
"""

import os
import json
from typing import Optional, List, Dict, Any


class AudioEnhancementSettings:
    """Audio Enhancement Settings."""

    def __init__(self):
        """Initialize audio enhancement settings with default values."""

        # Audio Enhancement Features
        self.ENABLE_AUDIO_ENHANCEMENT = self._get_bool_env("ENABLE_AUDIO_ENHANCEMENT", True)
        self.ENABLE_REAL_TIME_ENHANCEMENT = self._get_bool_env("ENABLE_REAL_TIME_ENHANCEMENT", True)
        self.ENABLE_QUALITY_ANALYSIS = self._get_bool_env("ENABLE_QUALITY_ANALYSIS", True)
        self.ENABLE_ENHANCEMENT_PRESETS = self._get_bool_env("ENABLE_ENHANCEMENT_PRESETS", True)

        # Processing Limits
        self.MAX_AUDIO_FILE_SIZE_MB = self._get_int_env("MAX_AUDIO_FILE_SIZE_MB", 50)
        self.MAX_BATCH_SIZE = self._get_int_env("MAX_BATCH_SIZE", 10)
        self.MAX_CONCURRENT_ENHANCEMENTS = self._get_int_env("MAX_CONCURRENT_ENHANCEMENTS", 5)
        self.PROCESSING_TIMEOUT_SECONDS = self._get_int_env("PROCESSING_TIMEOUT_SECONDS", 300)

        # Quality Analysis Settings
        self.QUALITY_ANALYSIS_ENABLED = self._get_bool_env("QUALITY_ANALYSIS_ENABLED", True)
        self.SNR_THRESHOLD = self._get_float_env("SNR_THRESHOLD", 20.0)
        self.THD_THRESHOLD = self._get_float_env("THD_THRESHOLD", 5.0)
        self.QUALITY_SCORE_THRESHOLD = self._get_float_env("QUALITY_SCORE_THRESHOLD", 7.0)

        # Enhancement Algorithm Settings
        self.DEFAULT_NOISE_REDUCTION_FACTOR = self._get_float_env("DEFAULT_NOISE_REDUCTION_FACTOR", 0.1)
        self.DEFAULT_NORMALIZATION_LEVEL = self._get_float_env("DEFAULT_NORMALIZATION_LEVEL", -12.0)
        self.DEFAULT_COMPRESSION_THRESHOLD = self._get_float_env("DEFAULT_COMPRESSION_THRESHOLD", -25.0)
        self.DEFAULT_COMPRESSION_RATIO = self._get_float_env("DEFAULT_COMPRESSION_RATIO", 3.0)

        # Real-time Processing Settings
        self.REAL_TIME_CHUNK_SIZE = self._get_int_env("REAL_TIME_CHUNK_SIZE", 1024)
        self.REAL_TIME_BUFFER_SIZE = self._get_int_env("REAL_TIME_BUFFER_SIZE", 8192)
        self.REAL_TIME_TARGET_LATENCY_MS = self._get_int_env("REAL_TIME_TARGET_LATENCY_MS", 50)
        self.REAL_TIME_SAMPLE_RATE = self._get_int_env("REAL_TIME_SAMPLE_RATE", 44100)

        # Adaptive Processing Settings
        self.ADAPTIVE_PROCESSING_ENABLED = self._get_bool_env("ADAPTIVE_PROCESSING_ENABLED", True)
        self.ADAPTIVE_QUALITY_MONITORING = self._get_bool_env("ADAPTIVE_QUALITY_MONITORING", True)
        self.ADAPTIVE_AUTO_ADJUST = self._get_bool_env("ADAPTIVE_AUTO_ADJUST", True)
        self.ADAPTIVE_UPDATE_INTERVAL = self._get_int_env("ADAPTIVE_UPDATE_INTERVAL", 10)

        # Enhancement Presets Settings
        self.MAX_USER_PRESETS_PER_USER = self._get_int_env("MAX_USER_PRESETS_PER_USER", 50)
        self.MAX_SYSTEM_PRESETS = self._get_int_env("MAX_SYSTEM_PRESETS", 20)
        self.PRESET_RATING_ENABLED = self._get_bool_env("PRESET_RATING_ENABLED", True)
        self.PRESET_USAGE_TRACKING = self._get_bool_env("PRESET_USAGE_TRACKING", True)

        # A/B Testing Settings
        self.AB_TEST_ENABLED = self._get_bool_env("AB_TEST_ENABLED", True)
        self.AB_TEST_MIN_SAMPLES = self._get_int_env("AB_TEST_MIN_SAMPLES", 10)
        self.AB_TEST_CONFIDENCE_THRESHOLD = self._get_float_env("AB_TEST_CONFIDENCE_THRESHOLD", 0.95)
        self.AB_TEST_DURATION_DAYS = self._get_int_env("AB_TEST_DURATION_DAYS", 7)

        # Performance Settings
        self.ENHANCEMENT_WORKER_COUNT = self._get_int_env("ENHANCEMENT_WORKER_COUNT", 4)
        self.QUALITY_ANALYSIS_WORKER_COUNT = self._get_int_env("QUALITY_ANALYSIS_WORKER_COUNT", 2)
        self.ENABLE_CACHING = self._get_bool_env("ENABLE_CACHING", True)
        self.CACHE_TTL_SECONDS = self._get_int_env("CACHE_TTL_SECONDS", 3600)

        # Audio Format Support
        self.SUPPORTED_INPUT_FORMATS = self._get_list_env("SUPPORTED_INPUT_FORMATS",
                                                         ["mp3", "wav", "flac", "ogg", "m4a", "aac"])
        self.SUPPORTED_OUTPUT_FORMATS = self._get_list_env("SUPPORTED_OUTPUT_FORMATS",
                                                          ["mp3", "wav", "flac", "ogg"])

        # Quality Metrics Weights
        self.QUALITY_WEIGHTS = self._get_dict_env("QUALITY_WEIGHTS",
                                                 {"clarity": 0.4, "noise": 0.3, "distortion": 0.3})

        # Enhancement Algorithm Parameters
        self.NOISE_REDUCTION_MIN_FACTOR = self._get_float_env("NOISE_REDUCTION_MIN_FACTOR", 0.01)
        self.NOISE_REDUCTION_MAX_FACTOR = self._get_float_env("NOISE_REDUCTION_MAX_FACTOR", 0.5)
        self.NORMALIZATION_MIN_LEVEL = self._get_float_env("NORMALIZATION_MIN_LEVEL", -50.0)
        self.NORMALIZATION_MAX_LEVEL = self._get_float_env("NORMALIZATION_MAX_LEVEL", 0.0)
        self.COMPRESSION_MIN_THRESHOLD = self._get_float_env("COMPRESSION_MIN_THRESHOLD", -50.0)
        self.COMPRESSION_MAX_THRESHOLD = self._get_float_env("COMPRESSION_MAX_THRESHOLD", 0.0)
        self.COMPRESSION_MIN_RATIO = self._get_float_env("COMPRESSION_MIN_RATIO", 1.0)
        self.COMPRESSION_MAX_RATIO = self._get_float_env("COMPRESSION_MAX_RATIO", 20.0)

        # Equalization Settings
        self.EQUALIZATION_BANDS = self._get_list_env("EQUALIZATION_BANDS", ["low", "mid", "high"])
        self.EQUALIZATION_MIN_GAIN = self._get_float_env("EQUALIZATION_MIN_GAIN", -20.0)
        self.EQUALIZATION_MAX_GAIN = self._get_float_env("EQUALIZATION_MAX_GAIN", 20.0)

        # Reverb Settings
        self.REVERB_MIN_ROOM_SIZE = self._get_float_env("REVERB_MIN_ROOM_SIZE", 0.0)
        self.REVERB_MAX_ROOM_SIZE = self._get_float_env("REVERB_MAX_ROOM_SIZE", 1.0)
        self.REVERB_MIN_DAMPING = self._get_float_env("REVERB_MIN_DAMPING", 0.0)
        self.REVERB_MAX_DAMPING = self._get_float_env("REVERB_MAX_DAMPING", 1.0)
        self.REVERB_MIN_WET_LEVEL = self._get_float_env("REVERB_MIN_WET_LEVEL", 0.0)
        self.REVERB_MAX_WET_LEVEL = self._get_float_env("REVERB_MAX_WET_LEVEL", 1.0)

        # Storage Settings
        self.ENHANCEMENT_STORAGE_PATH = self._get_str_env("ENHANCEMENT_STORAGE_PATH", "enhancements")
        self.QUALITY_REPORTS_PATH = self._get_str_env("QUALITY_REPORTS_PATH", "quality_reports")
        self.PRESET_STORAGE_PATH = self._get_str_env("PRESET_STORAGE_PATH", "presets")

        # Monitoring and Analytics
        self.ENABLE_ENHANCEMENT_METRICS = self._get_bool_env("ENABLE_ENHANCEMENT_METRICS", True)
        self.ENABLE_QUALITY_TRACKING = self._get_bool_env("ENABLE_QUALITY_TRACKING", True)
        self.ENABLE_USAGE_ANALYTICS = self._get_bool_env("ENABLE_USAGE_ANALYTICS", True)
        self.METRICS_RETENTION_DAYS = self._get_int_env("METRICS_RETENTION_DAYS", 90)

        # Error Handling
        self.MAX_RETRY_ATTEMPTS = self._get_int_env("MAX_RETRY_ATTEMPTS", 3)
        self.RETRY_BACKOFF_FACTOR = self._get_float_env("RETRY_BACKOFF_FACTOR", 2.0)
        self.ERROR_RATE_THRESHOLD = self._get_float_env("ERROR_RATE_THRESHOLD", 0.05)

        # Resource Management
        self.MEMORY_LIMIT_MB = self._get_int_env("MEMORY_LIMIT_MB", 1024)
        self.CPU_LIMIT_PERCENT = self._get_int_env("CPU_LIMIT_PERCENT", 80)
        self.ENABLE_RESOURCE_MONITORING = self._get_bool_env("ENABLE_RESOURCE_MONITORING", True)

        # API Settings
        self.API_RATE_LIMIT_PER_MINUTE = self._get_int_env("API_RATE_LIMIT_PER_MINUTE", 30)
        self.API_TIMEOUT_SECONDS = self._get_int_env("API_TIMEOUT_SECONDS", 300)
        self.ENABLE_API_AUTHENTICATION = self._get_bool_env("ENABLE_API_AUTHENTICATION", True)

        # Notification Settings
        self.ENABLE_ENHANCEMENT_NOTIFICATIONS = self._get_bool_env("ENABLE_ENHANCEMENT_NOTIFICATIONS", True)
        self.NOTIFICATION_EMAIL_ENABLED = self._get_bool_env("NOTIFICATION_EMAIL_ENABLED", False)
        self.NOTIFICATION_WEBHOOK_ENABLED = self._get_bool_env("NOTIFICATION_WEBHOOK_ENABLED", False)

        # Advanced Features
        self.ENABLE_MACHINE_LEARNING = self._get_bool_env("ENABLE_MACHINE_LEARNING", False)
        self.ENABLE_AUTO_QUALITY_OPTIMIZATION = self._get_bool_env("ENABLE_AUTO_QUALITY_OPTIMIZATION", True)
        self.ENABLE_BATCH_OPTIMIZATION = self._get_bool_env("ENABLE_BATCH_OPTIMIZATION", True)

        # Integration Settings
        self.ENABLE_CLOUD_STORAGE_INTEGRATION = self._get_bool_env("ENABLE_CLOUD_STORAGE_INTEGRATION", False)
        self.ENABLE_DATABASE_PERSISTENCE = self._get_bool_env("ENABLE_DATABASE_PERSISTENCE", True)
        self.ENABLE_REDIS_CACHING = self._get_bool_env("ENABLE_REDIS_CACHING", True)

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_str_env(self, key: str, default: str) -> str:
        """Get string environment variable."""
        return os.getenv(key, default)

    def _get_list_env(self, key: str, default: List[str]) -> List[str]:
        """Get list environment variable."""
        value = os.getenv(key)
        if value:
            return [item.strip().lower() for item in value.split(",")]
        return default

    def _get_dict_env(self, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Get dictionary environment variable."""
        value = os.getenv(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return default

    @property
    def is_enhancement_enabled(self) -> bool:
        """Check if audio enhancement is enabled."""
        return self.ENABLE_AUDIO_ENHANCEMENT

    @property
    def is_real_time_enabled(self) -> bool:
        """Check if real-time enhancement is enabled."""
        return self.ENABLE_REAL_TIME_ENHANCEMENT and self.ENABLE_AUDIO_ENHANCEMENT

    @property
    def is_quality_analysis_enabled(self) -> bool:
        """Check if quality analysis is enabled."""
        return self.ENABLE_QUALITY_ANALYSIS and self.ENABLE_AUDIO_ENHANCEMENT

    @property
    def is_preset_system_enabled(self) -> bool:
        """Check if preset system is enabled."""
        return self.ENABLE_ENHANCEMENT_PRESETS and self.ENABLE_AUDIO_ENHANCEMENT

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024

    @property
    def real_time_target_latency_seconds(self) -> float:
        """Get real-time target latency in seconds."""
        return self.REAL_TIME_TARGET_LATENCY_MS / 1000.0

    def get_enhancement_config(self, enhancement_type: str) -> Dict[str, Any]:
        """Get configuration for specific enhancement type.

        Args:
            enhancement_type: Type of enhancement

        Returns:
            Configuration dictionary
        """
        configs = {
            'noise_reduction': {
                'min_factor': self.NOISE_REDUCTION_MIN_FACTOR,
                'max_factor': self.NOISE_REDUCTION_MAX_FACTOR,
                'default_factor': self.DEFAULT_NOISE_REDUCTION_FACTOR
            },
            'normalization': {
                'min_level': self.NORMALIZATION_MIN_LEVEL,
                'max_level': self.NORMALIZATION_MAX_LEVEL,
                'default_level': self.DEFAULT_NORMALIZATION_LEVEL
            },
            'compression': {
                'min_threshold': self.COMPRESSION_MIN_THRESHOLD,
                'max_threshold': self.COMPRESSION_MAX_THRESHOLD,
                'min_ratio': self.COMPRESSION_MIN_RATIO,
                'max_ratio': self.COMPRESSION_MAX_RATIO,
                'default_threshold': self.DEFAULT_COMPRESSION_THRESHOLD,
                'default_ratio': self.DEFAULT_COMPRESSION_RATIO
            },
            'equalization': {
                'bands': self.EQUALIZATION_BANDS,
                'min_gain': self.EQUALIZATION_MIN_GAIN,
                'max_gain': self.EQUALIZATION_MAX_GAIN
            },
            'reverb': {
                'min_room_size': self.REVERB_MIN_ROOM_SIZE,
                'max_room_size': self.REVERB_MAX_ROOM_SIZE,
                'min_damping': self.REVERB_MIN_DAMPING,
                'max_damping': self.REVERB_MAX_DAMPING,
                'min_wet_level': self.REVERB_MIN_WET_LEVEL,
                'max_wet_level': self.REVERB_MAX_WET_LEVEL
            }
        }

        return configs.get(enhancement_type, {})

    def validate_enhancement_settings(self, enhancement_type: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate enhancement settings against configuration limits.

        Args:
            enhancement_type: Type of enhancement
            settings: Settings to validate

        Returns:
            Validated settings

        Raises:
            ValueError: If settings are invalid
        """
        config = self.get_enhancement_config(enhancement_type)
        validated_settings = {}

        if enhancement_type == 'noise_reduction':
            factor = settings.get('noise_factor', config.get('default_factor', 0.1))
            validated_settings['noise_factor'] = max(
                config.get('min_factor', 0.01),
                min(config.get('max_factor', 0.5), factor)
            )

        elif enhancement_type == 'normalization':
            level = settings.get('target_level', config.get('default_level', -12.0))
            validated_settings['target_level'] = max(
                config.get('min_level', -50.0),
                min(config.get('max_level', 0.0), level)
            )

        elif enhancement_type == 'compression':
            threshold = settings.get('threshold', config.get('default_threshold', -25.0))
            ratio = settings.get('ratio', config.get('default_ratio', 3.0))

            validated_settings['threshold'] = max(
                config.get('min_threshold', -50.0),
                min(config.get('max_threshold', 0.0), threshold)
            )
            validated_settings['ratio'] = max(
                config.get('min_ratio', 1.0),
                min(config.get('max_ratio', 20.0), ratio)
            )

        elif enhancement_type == 'equalization':
            for band in config.get('bands', ['low', 'mid', 'high']):
                gain_key = f'{band}_gain'
                gain = settings.get(gain_key, 0.0)
                validated_settings[gain_key] = max(
                    config.get('min_gain', -20.0),
                    min(config.get('max_gain', 20.0), gain)
                )

        elif enhancement_type == 'reverb':
            room_size = settings.get('room_size', 0.5)
            damping = settings.get('damping', 0.5)
            wet_level = settings.get('wet_level', 0.3)

            validated_settings['room_size'] = max(
                config.get('min_room_size', 0.0),
                min(config.get('max_room_size', 1.0), room_size)
            )
            validated_settings['damping'] = max(
                config.get('min_damping', 0.0),
                min(config.get('max_damping', 1.0), damping)
            )
            validated_settings['wet_level'] = max(
                config.get('min_wet_level', 0.0),
                min(config.get('max_wet_level', 1.0), wet_level)
            )

        return validated_settings


# Global audio enhancement settings instance
_audio_enhancement_settings: Optional[AudioEnhancementSettings] = None


def get_audio_enhancement_settings() -> AudioEnhancementSettings:
    """Get audio enhancement settings."""
    global _audio_enhancement_settings
    if _audio_enhancement_settings is None:
        _audio_enhancement_settings = AudioEnhancementSettings()
    return _audio_enhancement_settings


def reload_audio_enhancement_settings() -> AudioEnhancementSettings:
    """Reload audio enhancement settings."""
    global _audio_enhancement_settings
    _audio_enhancement_settings = AudioEnhancementSettings()
    return _audio_enhancement_settings