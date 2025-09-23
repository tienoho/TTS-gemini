"""
Webhook Models cho TTS System
"""
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from . import Base

class WebhookEventType(str, Enum):
    """Các loại event được hỗ trợ"""
    TTS_COMPLETED = "tts.completed"
    TTS_ERROR = "tts.error"
    BATCH_COMPLETED = "batch.completed"
    BATCH_ERROR = "batch.error"
    QUALITY_ENHANCEMENT_COMPLETED = "quality_enhancement.completed"
    QUALITY_ENHANCEMENT_ERROR = "quality_enhancement.error"
    VOICE_CLONING_COMPLETED = "voice_cloning.completed"
    VOICE_CLONING_ERROR = "voice_cloning.error"
    AUDIO_ENHANCEMENT_COMPLETED = "audio_enhancement.completed"
    AUDIO_ENHANCEMENT_ERROR = "audio_enhancement.error"

class WebhookStatus(str, Enum):
    """Trạng thái của webhook"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class DeliveryStatus(str, Enum):
    """Trạng thái delivery của webhook"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"

class Webhook(Base):
    """Model cho webhook configuration"""
    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    url = Column(String(500), nullable=False)
    secret = Column(String(255), nullable=False)  # Secret key cho HMAC signature
    status = Column(SQLEnum(WebhookStatus), default=WebhookStatus.ACTIVE)
    events = Column(JSON, nullable=False)  # List of event types to listen for
    headers = Column(JSON, default=dict)  # Custom headers
    retry_policy = Column(JSON, default=dict)  # Retry configuration
    rate_limit = Column(JSON, default=dict)  # Rate limiting configuration
    timeout = Column(Integer, default=30)  # Request timeout in seconds
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    creator = relationship("User", back_populates="webhooks")
    organization = relationship("Organization", back_populates="webhooks")
    delivery_history = relationship("WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")

    def is_event_enabled(self, event_type: str) -> bool:
        """Kiểm tra xem event type có được enable không"""
        return event_type in self.events

    def get_retry_config(self) -> Dict[str, Any]:
        """Lấy cấu hình retry"""
        return self.retry_policy or {
            "max_attempts": 3,
            "backoff_multiplier": 2,
            "initial_delay": 1,
            "max_delay": 300
        }

    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Lấy cấu hình rate limit"""
        return self.rate_limit or {
            "requests_per_minute": 60,
            "burst_limit": 10
        }

class WebhookDelivery(Base):
    """Model cho webhook delivery history"""
    __tablename__ = "webhook_deliveries"

    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(Integer, ForeignKey("webhooks.id"), nullable=False)
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False)
    signature = Column(String(255), nullable=False)  # HMAC signature
    status = Column(SQLEnum(DeliveryStatus), default=DeliveryStatus.PENDING)
    attempt_count = Column(Integer, default=0)
    last_attempt_at = Column(DateTime, nullable=True)
    next_retry_at = Column(DateTime, nullable=True)
    response_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    delivery_time = Column(Integer, nullable=True)  # Time in milliseconds
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    webhook = relationship("Webhook", back_populates="delivery_history")

    def is_retryable(self) -> bool:
        """Kiểm tra xem có thể retry không"""
        webhook = self.webhook
        max_attempts = webhook.get_retry_config()["max_attempts"]
        return self.attempt_count < max_attempts and self.status != DeliveryStatus.SUCCESS

    def calculate_next_retry_time(self) -> datetime:
        """Tính thời gian retry tiếp theo với exponential backoff"""
        config = self.webhook.get_retry_config()
        delay = min(
            config["initial_delay"] * (config["backoff_multiplier"] ** self.attempt_count),
            config["max_delay"]
        )
        return datetime.utcnow() + timedelta(seconds=delay)

class WebhookRetryAttempt(Base):
    """Model cho retry attempts chi tiết"""
    __tablename__ = "webhook_retry_attempts"

    id = Column(Integer, primary_key=True, index=True)
    delivery_id = Column(Integer, ForeignKey("webhook_deliveries.id"), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    attempted_at = Column(DateTime, default=datetime.utcnow)
    response_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    delivery_time = Column(Integer, nullable=True)

    # Relationships
    delivery = relationship("WebhookDelivery", back_populates="retry_attempts")

class WebhookDeadLetter(Base):
    """Model cho dead letter queue"""
    __tablename__ = "webhook_dead_letters"

    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(Integer, ForeignKey("webhooks.id"), nullable=False)
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False)
    signature = Column(String(255), nullable=False)
    failure_reason = Column(Text, nullable=False)
    last_attempt_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    webhook = relationship("Webhook")

class WebhookSecurityLog(Base):
    """Model cho security logging"""
    __tablename__ = "webhook_security_logs"

    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(Integer, ForeignKey("webhooks.id"), nullable=False)
    event_type = Column(String(100), nullable=False)  # signature_failed, rate_limited, etc.
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    webhook = relationship("Webhook")