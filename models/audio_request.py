"""
AudioRequest model for Flask TTS API with production-ready features
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

Base = declarative_base()


class AudioRequestStatus(str, Enum):
    """Status enum for audio requests."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioRequestPriority(str, Enum):
    """Priority enum for audio requests."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AudioRequest(Base):
    """AudioRequest model for storing TTS requests with production features."""

    __tablename__ = 'requests'

    # Primary key and relationships
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)

    # Request content
    text_content = Column(Text, nullable=False)
    language = Column(String(10), default="vi", index=True)
    voice_settings = Column(JSON, default=dict)  # voice_name, speed, pitch, etc.

    # Status and progress tracking
    status = Column(String(20), default=AudioRequestStatus.PENDING, index=True)
    priority = Column(String(10), default=AudioRequestPriority.NORMAL, index=True)
    progress = Column(Float, default=0.0)  # 0.0 to 100.0
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Output configuration
    output_format = Column(String(10), default="mp3")
    output_path = Column(String(500), nullable=True)  # Cloud storage path
    storage_type = Column(String(20), default="local")  # local, s3, gcs

    # Processing metadata
    processing_time = Column(Float, nullable=True)  # in seconds
    estimated_duration = Column(Float, nullable=True)  # estimated processing time
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)

    # Custom metadata
    metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Cost tracking
    processing_cost = Column(Float, default=0.0)  # API usage cost
    storage_cost = Column(Float, default=0.0)  # Storage cost

    # Flags
    is_public = Column(Boolean, default=False)
    allow_download = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="requests")
    logs = relationship("RequestLog", back_populates="request", cascade="all, delete-orphan")
    audio_files = relationship("AudioFile", back_populates="request", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_requests_user_status', 'user_id', 'status'),
        Index('idx_requests_priority_created', 'priority', 'created_at'),
        Index('idx_requests_processing_cost', 'processing_cost'),
    )

    def __init__(self, user_id: int, text_content: str, **kwargs):
        """Initialize audio request with production features."""
        self.user_id = user_id
        self.text_content = text_content

        # Set defaults for new fields
        if 'language' not in kwargs:
            kwargs['language'] = 'vi'
        if 'voice_settings' not in kwargs:
            kwargs['voice_settings'] = {}
        if 'priority' not in kwargs:
            kwargs['priority'] = AudioRequestPriority.NORMAL
        if 'output_format' not in kwargs:
            kwargs['output_format'] = 'mp3'
        if 'storage_type' not in kwargs:
            kwargs['storage_type'] = 'local'

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of audio request."""
        return f"<AudioRequest(id={self.id}, user_id={self.user_id}, status='{self.status}', priority='{self.priority}')>"

    def mark_as_processing(self) -> None:
        """Mark request as processing."""
        self.status = AudioRequestStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_as_completed(self, processing_time: Optional[float] = None, output_path: Optional[str] = None) -> None:
        """Mark request as completed."""
        self.status = AudioRequestStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if processing_time:
            self.processing_time = processing_time
        if output_path:
            self.output_path = output_path
        self.updated_at = datetime.utcnow()

    def mark_as_failed(self, error_message: str, error_code: Optional[str] = None) -> None:
        """Mark request as failed."""
        self.status = AudioRequestStatus.FAILED
        self.error_message = error_message
        self.error_code = error_code
        self.updated_at = datetime.utcnow()

    def mark_as_cancelled(self) -> None:
        """Mark request as cancelled."""
        self.status = AudioRequestStatus.CANCELLED
        self.updated_at = datetime.utcnow()

    def increment_retry_count(self) -> bool:
        """Increment retry count and return True if can retry."""
        self.retry_count += 1
        return self.retry_count < self.max_retries

    def update_progress(self, progress: float) -> None:
        """Update processing progress."""
        self.progress = max(0.0, min(100.0, progress))
        self.updated_at = datetime.utcnow()

    def can_be_processed(self) -> bool:
        """Check if request can be processed."""
        return (
            self.status in [AudioRequestStatus.PENDING, AudioRequestStatus.FAILED] and
            self.retry_count < self.max_retries
        )

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if request has expired."""
        if self.status in [AudioRequestStatus.COMPLETED, AudioRequestStatus.FAILED, AudioRequestStatus.CANCELLED]:
            return False

        age = datetime.utcnow() - self.created_at
        return age.total_seconds() > (max_age_hours * 3600)

    def get_total_cost(self) -> float:
        """Get total cost for this request."""
        return self.processing_cost + self.storage_cost

    def get_estimated_completion_time(self) -> Optional[datetime]:
        """Get estimated completion time based on progress."""
        if self.status == AudioRequestStatus.COMPLETED:
            return self.completed_at
        if self.status == AudioRequestStatus.PROCESSING and self.estimated_duration:
            return self.started_at + timedelta(seconds=self.estimated_duration)
        return None

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if request has expired."""
        if self.status in [AudioRequestStatus.COMPLETED, AudioRequestStatus.FAILED]:
            return False

        age = datetime.utcnow() - self.created_at
        return age.total_seconds() > (max_age_hours * 3600)

    def get_file_count(self) -> int:
        """Get number of audio files for this request."""
        return len(self.audio_files) if self.audio_files else 0

    def get_total_file_size(self) -> int:
        """Get total size of all audio files."""
        if not self.audio_files:
            return 0
        return sum(file.file_size for file in self.audio_files if file.file_size)

    def get_download_url(self, base_url: str) -> Optional[str]:
        """Get download URL for the request."""
        if self.status == AudioRequestStatus.COMPLETED and self.output_path:
            return f"{base_url}/api/v1/tts/result/{self.id}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert audio request to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'text_content': self.text_content,
            'language': self.language,
            'voice_settings': self.voice_settings,
            'status': self.status,
            'priority': self.priority,
            'progress': self.progress,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'output_format': self.output_format,
            'output_path': self.output_path,
            'storage_type': self.storage_type,
            'processing_time': self.processing_time,
            'estimated_duration': self.estimated_duration,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'processing_cost': self.processing_cost,
            'storage_cost': self.storage_cost,
            'is_public': self.is_public,
            'allow_download': self.allow_download,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'file_count': self.get_file_count(),
            'total_file_size': self.get_total_file_size(),
            'total_cost': self.get_total_cost(),
        }

    def to_dict_with_files(self) -> Dict[str, Any]:
        """Convert audio request to dictionary with file details."""
        data = self.to_dict()
        if self.audio_files:
            data['audio_files'] = [file.to_dict() for file in self.audio_files]
        else:
            data['audio_files'] = []
        return data

    def to_public_dict(self) -> Dict[str, Any]:
        """Convert to public dictionary (without sensitive data)."""
        data = self.to_dict()
        # Remove sensitive fields
        data.pop('user_id', None)
        data.pop('processing_cost', None)
        data.pop('storage_cost', None)
        data.pop('metadata', None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioRequest':
        """Create audio request instance from dictionary."""
        return cls(
            user_id=data['user_id'],
            text_content=data['text_content'],
            language=data.get('language', 'vi'),
            voice_settings=data.get('voice_settings', {}),
            priority=data.get('priority', AudioRequestPriority.NORMAL),
            output_format=data.get('output_format', 'mp3'),
            storage_type=data.get('storage_type', 'local'),
            metadata=data.get('metadata', {}),
            is_public=data.get('is_public', False),
            allow_download=data.get('allow_download', True)
        )

    @staticmethod
    def get_by_user(user_id: int, db_session, limit: int = 50, offset: int = 0, include_completed: bool = True):
        """Get audio requests for a specific user with advanced filtering."""
        if not user_id or user_id <= 0:
            raise ValueError("Invalid user_id provided")

        query = db_session.query(AudioRequest).filter(AudioRequest.user_id == user_id)

        if not include_completed:
            query = query.filter(AudioRequest.status != AudioRequestStatus.COMPLETED)

        return query.order_by(AudioRequest.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def get_by_status(status: str, db_session, limit: int = 50, offset: int = 0):
        """Get audio requests by status."""
        return db_session.query(AudioRequest).filter(
            AudioRequest.status == status
        ).order_by(AudioRequest.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def get_pending_requests(db_session, limit: int = 10, priority: Optional[str] = None):
        """Get pending audio requests for processing with priority support."""
        query = db_session.query(AudioRequest).filter(
            AudioRequest.status == AudioRequestStatus.PENDING
        )

        if priority:
            query = query.filter(AudioRequest.priority == priority)

        return query.order_by(
            AudioRequest.priority.desc(),
            AudioRequest.created_at.asc()
        ).limit(limit).all()

    @staticmethod
    def get_requests_for_retry(db_session, limit: int = 10):
        """Get failed requests that can be retried."""
        return db_session.query(AudioRequest).filter(
            AudioRequest.status == AudioRequestStatus.FAILED,
            AudioRequest.retry_count < AudioRequest.max_retries
        ).order_by(
            AudioRequest.retry_count.asc(),
            AudioRequest.created_at.asc()
        ).limit(limit).all()

    @staticmethod
    def get_expired_requests(db_session, max_age_hours: int = 24):
        """Get expired requests that should be cleaned up."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        return db_session.query(AudioRequest).filter(
            AudioRequest.created_at < cutoff_time,
            AudioRequest.status.in_([
                AudioRequestStatus.PENDING,
                AudioRequestStatus.PROCESSING
            ])
        ).all()

    @staticmethod
    def get_user_stats(user_id: int, db_session) -> Dict[str, Any]:
        """Get comprehensive statistics for user's audio requests."""
        if not user_id or user_id <= 0:
            raise ValueError("Invalid user_id provided")

        from sqlalchemy import func, case

        stats = db_session.query(
            func.count(AudioRequest.id).label('total'),
            func.sum(case((AudioRequest.status == AudioRequestStatus.COMPLETED, 1), else_=0)).label('completed'),
            func.sum(case((AudioRequest.status == AudioRequestStatus.FAILED, 1), else_=0)).label('failed'),
            func.sum(case((AudioRequest.status == AudioRequestStatus.PROCESSING, 1), else_=0)).label('processing'),
            func.sum(case((AudioRequest.status == AudioRequestStatus.PENDING, 1), else_=0)).label('pending'),
            func.sum(case((AudioRequest.status == AudioRequestStatus.CANCELLED, 1), else_=0)).label('cancelled'),
            func.avg(AudioRequest.processing_time).label('avg_processing_time'),
            func.sum(AudioRequest.processing_cost).label('total_processing_cost'),
            func.sum(AudioRequest.storage_cost).label('total_storage_cost'),
            func.avg(AudioRequest.retry_count).label('avg_retry_count')
        ).filter(AudioRequest.user_id == user_id).first()

        return {
            'total_requests': stats.total or 0,
            'completed': stats.completed or 0,
            'failed': stats.failed or 0,
            'processing': stats.processing or 0,
            'pending': stats.pending or 0,
            'cancelled': stats.cancelled or 0,
            'success_rate': (stats.completed / stats.total * 100) if stats.total > 0 else 0,
            'avg_processing_time': float(stats.avg_processing_time) if stats.avg_processing_time else 0.0,
            'total_processing_cost': float(stats.total_processing_cost) if stats.total_processing_cost else 0.0,
            'total_storage_cost': float(stats.total_storage_cost) if stats.total_storage_cost else 0.0,
            'total_cost': float(stats.total_processing_cost or 0) + float(stats.total_storage_cost or 0),
            'avg_retry_count': float(stats.avg_retry_count) if stats.avg_retry_count else 0.0,
        }

    @staticmethod
    def get_user_stats(user_id: int, db_session) -> dict:
        """Get statistics for user's audio requests."""
        if not user_id or user_id <= 0:
            raise ValueError("Invalid user_id provided")

        from sqlalchemy import func

        stats = db_session.query(
            func.count(AudioRequest.id).label('total'),
            func.sum(func.case((AudioRequest.status == AudioRequestStatus.COMPLETED, 1), else_=0)).label('completed'),
            func.sum(func.case((AudioRequest.status == AudioRequestStatus.FAILED, 1), else_=0)).label('failed'),
            func.sum(func.case((AudioRequest.status == AudioRequestStatus.PROCESSING, 1), else_=0)).label('processing'),
            func.sum(func.case((AudioRequest.status == AudioRequestStatus.PENDING, 1), else_=0)).label('pending'),
            func.avg(AudioRequest.processing_time).label('avg_processing_time')
        ).filter(AudioRequest.user_id == user_id).first()

        return {
            'total_requests': stats.total or 0,
            'completed': stats.completed or 0,
            'failed': stats.failed or 0,
            'processing': stats.processing or 0,
            'pending': stats.pending or 0,
            'avg_processing_time': float(stats.avg_processing_time) if stats.avg_processing_time else 0.0,
        }