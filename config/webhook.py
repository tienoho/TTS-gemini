"""
Webhook Configuration Settings
"""
from typing import Dict, Any, List
from pydantic import BaseSettings, validator
import os

class WebhookConfig(BaseSettings):
    """Cấu hình cho webhook system"""

    # Delivery settings
    DEFAULT_TIMEOUT: int = 30  # seconds
    MAX_TIMEOUT: int = 300  # seconds
    CONNECTION_TIMEOUT: int = 10  # seconds

    # Retry settings
    DEFAULT_MAX_ATTEMPTS: int = 3
    DEFAULT_BACKOFF_MULTIPLIER: float = 2.0
    DEFAULT_INITIAL_DELAY: int = 1  # seconds
    DEFAULT_MAX_DELAY: int = 300  # seconds

    # Rate limiting
    DEFAULT_REQUESTS_PER_MINUTE: int = 60
    DEFAULT_BURST_LIMIT: int = 10
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Batch delivery
    BATCH_SIZE: int = 10
    BATCH_TIMEOUT: int = 5  # seconds
    MAX_BATCH_SIZE: int = 100

    # Dead letter queue
    DEAD_LETTER_RETENTION_DAYS: int = 30
    MAX_DEAD_LETTER_SIZE: int = 1000

    # Security settings
    SIGNATURE_ALGORITHM: str = "sha256"
    SECRET_KEY_LENGTH: int = 32
    MAX_REQUEST_SIZE: int = 1024 * 1024  # 1MB

    # IP restrictions
    ENABLE_IP_WHITELIST: bool = False
    ENABLE_IP_BLACKLIST: bool = False
    TRUSTED_IPS: List[str] = []

    # Queue settings
    QUEUE_MAX_SIZE: int = 10000
    QUEUE_WORKERS: int = 4

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_RETENTION_HOURS: int = 24

    # Event filtering
    ENABLE_EVENT_FILTERING: bool = True
    MAX_EVENTS_PER_WEBHOOK: int = 50

    # Custom headers
    DEFAULT_USER_AGENT: str = "TTS-Webhook/1.0"
    ENABLE_CUSTOM_HEADERS: bool = True

    # Validation
    URL_PATTERN: str = r"^https?://.*"
    MAX_URL_LENGTH: int = 500
    MAX_HEADER_SIZE: int = 8192  # 8KB

    class Config:
        env_prefix = "WEBHOOK_"
        case_sensitive = False

    @validator("TRUSTED_IPS", pre=True)
    def parse_trusted_ips(cls, v):
        if isinstance(v, str):
            return [ip.strip() for ip in v.split(",") if ip.strip()]
        return v or []

    def get_retry_config(self) -> Dict[str, Any]:
        """Lấy cấu hình retry mặc định"""
        return {
            "max_attempts": self.DEFAULT_MAX_ATTEMPTS,
            "backoff_multiplier": self.DEFAULT_BACKOFF_MULTIPLIER,
            "initial_delay": self.DEFAULT_INITIAL_DELAY,
            "max_delay": self.DEFAULT_MAX_DELAY
        }

    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Lấy cấu hình rate limit mặc định"""
        return {
            "requests_per_minute": self.DEFAULT_REQUESTS_PER_MINUTE,
            "burst_limit": self.DEFAULT_BURST_LIMIT,
            "window_seconds": self.RATE_LIMIT_WINDOW
        }

    def get_batch_config(self) -> Dict[str, Any]:
        """Lấy cấu hình batch delivery"""
        return {
            "batch_size": self.BATCH_SIZE,
            "batch_timeout": self.BATCH_TIMEOUT,
            "max_batch_size": self.MAX_BATCH_SIZE
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Lấy cấu hình security"""
        return {
            "signature_algorithm": self.SIGNATURE_ALGORITHM,
            "secret_key_length": self.SECRET_KEY_LENGTH,
            "max_request_size": self.MAX_REQUEST_SIZE,
            "enable_ip_whitelist": self.ENABLE_IP_WHITELIST,
            "enable_ip_blacklist": self.ENABLE_IP_BLACKLIST,
            "trusted_ips": self.TRUSTED_IPS
        }

    def get_queue_config(self) -> Dict[str, Any]:
        """Lấy cấu hình queue"""
        return {
            "max_size": self.QUEUE_MAX_SIZE,
            "workers": self.QUEUE_WORKERS
        }

# Global webhook configuration instance
webhook_config = WebhookConfig()

# Event type configurations
EVENT_CONFIGS = {
    "tts.completed": {
        "description": "TTS conversion completed successfully",
        "required_fields": ["request_id", "audio_url", "duration", "text_length"],
        "optional_fields": ["quality_score", "enhancement_applied", "voice_settings"]
    },
    "tts.error": {
        "description": "TTS conversion failed",
        "required_fields": ["request_id", "error_code", "error_message"],
        "optional_fields": ["retry_count", "text_length", "voice_settings"]
    },
    "batch.completed": {
        "description": "Batch processing completed",
        "required_fields": ["batch_id", "total_requests", "successful_requests", "failed_requests"],
        "optional_fields": ["processing_time", "average_quality_score", "total_duration"]
    },
    "batch.error": {
        "description": "Batch processing failed",
        "required_fields": ["batch_id", "error_code", "error_message", "failed_requests"],
        "optional_fields": ["total_requests", "successful_requests", "processing_time"]
    },
    "quality_enhancement.completed": {
        "description": "Audio quality enhancement completed",
        "required_fields": ["request_id", "enhancement_type", "original_quality", "enhanced_quality"],
        "optional_fields": ["enhancement_settings", "processing_time", "improvement_score"]
    },
    "quality_enhancement.error": {
        "description": "Audio quality enhancement failed",
        "required_fields": ["request_id", "enhancement_type", "error_code", "error_message"],
        "optional_fields": ["original_quality", "enhancement_settings"]
    },
    "voice_cloning.completed": {
        "description": "Voice cloning completed",
        "required_fields": ["cloning_id", "voice_name", "quality_score", "training_duration"],
        "optional_fields": ["voice_samples_used", "model_size", "accuracy_score"]
    },
    "voice_cloning.error": {
        "description": "Voice cloning failed",
        "required_fields": ["cloning_id", "error_code", "error_message"],
        "optional_fields": ["voice_samples_used", "training_duration", "failure_stage"]
    },
    "audio_enhancement.completed": {
        "description": "Audio enhancement completed",
        "required_fields": ["request_id", "enhancement_type", "improvement_score"],
        "optional_fields": ["original_metrics", "enhanced_metrics", "processing_time"]
    },
    "audio_enhancement.error": {
        "description": "Audio enhancement failed",
        "required_fields": ["request_id", "enhancement_type", "error_code", "error_message"],
        "optional_fields": ["original_metrics", "enhancement_settings"]
    }
}