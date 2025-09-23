"""
Voice Cloning Configuration for TTS System
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class VoiceCloningConfig:
    """Configuration class for voice cloning system."""

    def __init__(self):
        """Initialize voice cloning configuration."""
        # Base paths
        self.base_dir = Path(__file__).parent.parent
        self.voice_library_path = self.base_dir / "voice_library"
        self.temp_path = self.base_dir / "temp" / "voice_cloning"
        self.logs_path = self.base_dir / "logs" / "voice_cloning"

        # Training configuration
        self.training_config = {
            'min_sample_duration': 30.0,  # seconds
            'max_sample_duration': 300.0,  # seconds
            'min_samples_required': 5,
            'max_samples_allowed': 50,
            'target_sample_rate': 22050,  # Hz
            'batch_size': 8,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 10,
            'validation_split': 0.2,
            'test_split': 0.1
        }

        # Quality thresholds
        self.quality_config = {
            'min_snr_ratio': 15.0,  # dB
            'min_quality_score': 5.0,  # 1-10 scale
            'excellent_threshold': 9.0,
            'good_threshold': 7.0,
            'fair_threshold': 5.0,
            'poor_threshold': 3.0,
            'quality_weights': {
                'clarity': 0.25,
                'naturalness': 0.25,
                'pronunciation': 0.20,
                'consistency': 0.15,
                'expressiveness': 0.15
            }
        }

        # Storage configuration
        self.storage_config = {
            'max_storage_per_org': 10 * 1024 * 1024 * 1024,  # 10GB
            'max_models_per_org': 100,
            'max_versions_per_model': 10,
            'cleanup_days': 30,
            'compression_enabled': True,
            'compression_level': 6,  # 1-9, higher = more compression
            'auto_cleanup_enabled': True
        }

        # Audio preprocessing configuration
        self.preprocessing_config = {
            'target_sample_rate': 22050,
            'target_channels': 1,
            'target_bit_depth': 16,
            'noise_reduction_enabled': True,
            'normalization_enabled': True,
            'silence_removal_enabled': True,
            'target_peak_db': -3.0,
            'target_rms_db': -20.0,
            'silence_thresh_db': -40.0,
            'min_silence_duration_ms': 500,
            'keep_silence_ms': 300
        }

        # Model configuration
        self.model_config = {
            'supported_formats': ['wav', 'mp3', 'flac', 'ogg'],
            'supported_languages': ['vi', 'en', 'ja', 'ko', 'zh'],
            'max_model_size_mb': 500,  # Maximum model file size
            'model_version': '1.0.0',
            'supported_model_types': ['standard', 'premium', 'custom'],
            'feature_types': ['mfcc', 'mel_spectrogram', 'pitch', 'spectral']
        }

        # API configuration
        self.api_config = {
            'max_concurrent_trainings': 5,
            'training_timeout_hours': 24,
            'max_file_size_mb': 100,  # Maximum upload file size
            'supported_upload_formats': ['wav', 'mp3', 'flac'],
            'rate_limit_per_minute': 60,
            'rate_limit_per_hour': 1000,
            'enable_progress_streaming': True,
            'progress_update_interval': 5  # seconds
        }

        # WebSocket configuration
        self.websocket_config = {
            'enabled': True,
            'heartbeat_interval': 30,  # seconds
            'reconnect_interval': 5,  # seconds
            'max_message_size': 1024 * 1024,  # 1MB
            'supported_events': [
                'training_started',
                'training_progress',
                'training_completed',
                'training_failed',
                'training_cancelled',
                'quality_assessment_started',
                'quality_assessment_completed'
            ]
        }

        # Security configuration
        self.security_config = {
            'enable_encryption': True,
            'encryption_algorithm': 'AES256',
            'enable_access_tokens': True,
            'token_expiration_hours': 24,
            'enable_audit_logging': True,
            'max_failed_attempts': 5,
            'lockout_duration_minutes': 30,
            'require_organization_scoping': True
        }

        # Performance configuration
        self.performance_config = {
            'enable_gpu_training': True,
            'gpu_memory_fraction': 0.8,
            'enable_mixed_precision': True,
            'enable_model_optimization': True,
            'optimization_level': 2,  # 0-3, higher = more optimization
            'enable_caching': True,
            'cache_size_mb': 1024,  # 1GB cache
            'enable_parallel_processing': True,
            'max_workers': 4
        }

        # Monitoring configuration
        self.monitoring_config = {
            'enable_metrics_collection': True,
            'metrics_retention_days': 90,
            'enable_health_checks': True,
            'health_check_interval': 60,  # seconds
            'enable_alerting': True,
            'alert_thresholds': {
                'training_failure_rate': 0.1,  # 10% failure rate
                'average_training_time': 3600,  # 1 hour
                'quality_score_drop': 2.0  # 2 point drop
            }
        }

        # Development configuration
        self.development_config = {
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'enable_detailed_logging': os.getenv('DETAILED_LOGGING', 'false').lower() == 'true',
            'mock_training': os.getenv('MOCK_TRAINING', 'false').lower() == 'true',
            'mock_training_time': int(os.getenv('MOCK_TRAINING_TIME', '60')),  # seconds
            'enable_profiling': os.getenv('ENABLE_PROFILING', 'false').lower() == 'true',
            'profile_output_path': str(self.logs_path / 'profiles')
        }

        # Load environment-specific overrides
        self._load_environment_config()

    def _load_environment_config(self):
        """Load environment-specific configuration."""
        env = os.getenv('FLASK_ENV', 'development')

        if env == 'production':
            # Production overrides
            self.api_config.update({
                'max_concurrent_trainings': 10,
                'training_timeout_hours': 48,
                'rate_limit_per_minute': 30,
                'rate_limit_per_hour': 500
            })

            self.security_config.update({
                'enable_encryption': True,
                'enable_access_tokens': True,
                'require_organization_scoping': True
            })

            self.development_config.update({
                'debug_mode': False,
                'enable_detailed_logging': False,
                'mock_training': False
            })

        elif env == 'testing':
            # Testing overrides
            self.training_config.update({
                'min_samples_required': 2,
                'max_samples_allowed': 5,
                'epochs': 5,
                'patience': 2
            })

            self.development_config.update({
                'debug_mode': True,
                'mock_training': True,
                'mock_training_time': 10
            })

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.training_config.copy()

    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality assessment configuration."""
        return self.quality_config.copy()

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self.storage_config.copy()

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get audio preprocessing configuration."""
        return self.preprocessing_config.copy()

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.model_config.copy()

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.api_config.copy()

    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration."""
        return self.websocket_config.copy()

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.security_config.copy()

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.performance_config.copy()

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.monitoring_config.copy()

    def get_development_config(self) -> Dict[str, Any]:
        """Get development configuration."""
        return self.development_config.copy()

    def get_path_config(self) -> Dict[str, str]:
        """Get path configuration."""
        return {
            'base_dir': str(self.base_dir),
            'voice_library_path': str(self.voice_library_path),
            'temp_path': str(self.temp_path),
            'logs_path': str(self.logs_path)
        }

    def update_config(self, section: str, key: str, value: Any):
        """Update configuration value."""
        if hasattr(self, f"{section}_config"):
            config_attr = getattr(self, f"{section}_config")
            if isinstance(config_attr, dict):
                config_attr[key] = value
            else:
                setattr(config_attr, key, value)
        else:
            raise ValueError(f"Configuration section '{section}' not found")

    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if hasattr(self, f"{section}_config"):
            config_attr = getattr(self, f"{section}_config")
            if isinstance(config_attr, dict):
                return config_attr.get(key, default)
            else:
                return getattr(config_attr, key, default)
        return default

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate training config
        if self.training_config['min_samples_required'] <= 0:
            issues.append("min_samples_required must be greater than 0")

        if self.training_config['batch_size'] <= 0:
            issues.append("batch_size must be greater than 0")

        if not 0 < self.training_config['learning_rate'] < 1:
            issues.append("learning_rate must be between 0 and 1")

        # Validate quality config
        if not 1 <= self.quality_config['min_quality_score'] <= 10:
            issues.append("min_quality_score must be between 1 and 10")

        # Validate storage config
        if self.storage_config['max_storage_per_org'] <= 0:
            issues.append("max_storage_per_org must be greater than 0")

        # Validate API config
        if self.api_config['max_file_size_mb'] <= 0:
            issues.append("max_file_size_mb must be greater than 0")

        # Validate paths
        for path_key, path_value in self.get_path_config().items():
            path_obj = Path(path_value)
            if not path_obj.parent.exists():
                issues.append(f"Parent directory for {path_key} does not exist: {path_value}")

        return issues

    def create_directories(self):
        """Create necessary directories."""
        paths = [
            self.voice_library_path,
            self.temp_path,
            self.logs_path,
            self.voice_library_path / "models",
            self.voice_library_path / "samples",
            self.voice_library_path / "temp",
            self.voice_library_path / "archives"
        ]

        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'training_config': self.training_config,
            'quality_config': self.quality_config,
            'storage_config': self.storage_config,
            'preprocessing_config': self.preprocessing_config,
            'model_config': self.model_config,
            'api_config': self.api_config,
            'websocket_config': self.websocket_config,
            'security_config': self.security_config,
            'performance_config': self.performance_config,
            'monitoring_config': self.monitoring_config,
            'development_config': self.development_config,
            'path_config': self.get_path_config()
        }


# Global configuration instance
voice_cloning_config = VoiceCloningConfig()

def get_voice_cloning_config() -> VoiceCloningConfig:
    """Get voice cloning configuration instance."""
    return voice_cloning_config

def reload_voice_cloning_config():
    """Reload voice cloning configuration."""
    global voice_cloning_config
    voice_cloning_config = VoiceCloningConfig()
    return voice_cloning_config