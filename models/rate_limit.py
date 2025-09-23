"""
RateLimit model for managing API rate limiting
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import Column, DateTime, Integer, String, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class RateLimitType(str, Enum):
    """Rate limit type enum."""

    REQUEST = "request"  # TTS requests
    STORAGE = "storage"  # Storage operations
    API = "api"  # General API calls
    DOWNLOAD = "download"  # File downloads


class RateLimitWindow(str, Enum):
    """Rate limit window enum."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class RateLimit(Base):
    """RateLimit model for tracking user rate limits."""

    __tablename__ = 'rate_limits'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)

    # Rate limit configuration
    limit_type = Column(String(20), nullable=False, index=True)  # request, storage, api, download
    window_type = Column(String(10), nullable=False, index=True)  # minute, hour, day, month
    limit_value = Column(Integer, nullable=False)  # Maximum allowed in window
    current_count = Column(Integer, default=0)

    # Window tracking
    window_start = Column(DateTime, nullable=False, index=True)
    window_end = Column(DateTime, nullable=False)

    # Metadata
    metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")

    # Indexes for performance
    __table_args__ = (
        Index('idx_rate_limits_user_type_window', 'user_id', 'limit_type', 'window_start'),
        Index('idx_rate_limits_window_end', 'window_end'),
    )

    def __init__(self, user_id: int, limit_type: str, window_type: str, limit_value: int, **kwargs):
        """Initialize rate limit."""
        self.user_id = user_id
        self.limit_type = limit_type
        self.window_type = window_type
        self.limit_value = limit_value

        # Calculate window
        now = datetime.utcnow()
        self.window_start, self.window_end = self._calculate_window(now, window_type)

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of rate limit."""
        return f"<RateLimit(id={self.id}, user_id={self.user_id}, type='{self.limit_type}', window='{self.window_type}', count={self.current_count}/{self.limit_value})>"

    def _calculate_window(self, now: datetime, window_type: str) -> tuple[datetime, datetime]:
        """Calculate window start and end times."""
        if window_type == RateLimitWindow.MINUTE:
            start = now.replace(second=0, microsecond=0)
            end = start + timedelta(minutes=1)
        elif window_type == RateLimitWindow.HOUR:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif window_type == RateLimitWindow.DAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif window_type == RateLimitWindow.MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        else:
            raise ValueError(f"Invalid window type: {window_type}")

        return start, end

    def is_expired(self) -> bool:
        """Check if rate limit window has expired."""
        return datetime.utcnow() >= self.window_end

    def is_under_limit(self) -> bool:
        """Check if current count is under the limit."""
        return self.current_count < self.limit_value

    def can_make_request(self) -> bool:
        """Check if user can make another request."""
        if self.is_expired():
            return True
        return self.is_under_limit()

    def increment_count(self) -> bool:
        """Increment count and return True if still under limit."""
        if self.is_expired():
            # Reset for new window
            now = datetime.utcnow()
            self.window_start, self.window_end = self._calculate_window(now, self.window_type)
            self.current_count = 0
            self.updated_at = now

        self.current_count += 1
        self.updated_at = datetime.utcnow()
        return self.is_under_limit()

    def get_remaining_count(self) -> int:
        """Get remaining requests in current window."""
        if self.is_expired():
            return self.limit_value
        return max(0, self.limit_value - self.current_count)

    def get_reset_time(self) -> datetime:
        """Get time when rate limit resets."""
        return self.window_end

    def get_seconds_until_reset(self) -> int:
        """Get seconds until rate limit resets."""
        reset_time = self.get_reset_time()
        now = datetime.utcnow()
        if reset_time <= now:
            return 0
        return int((reset_time - now).total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        """Convert rate limit to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'limit_type': self.limit_type,
            'window_type': self.window_type,
            'limit_value': self.limit_value,
            'current_count': self.current_count,
            'remaining_count': self.get_remaining_count(),
            'window_start': self.window_start.isoformat() if self.window_start else None,
            'window_end': self.window_end.isoformat() if self.window_end else None,
            'reset_seconds': self.get_seconds_until_reset(),
            'is_under_limit': self.is_under_limit(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def get_or_create(
        cls,
        user_id: int,
        limit_type: str,
        window_type: str,
        limit_value: int,
        db_session
    ) -> 'RateLimit':
        """Get existing rate limit or create new one."""
        now = datetime.utcnow()

        # Try to find existing non-expired rate limit
        existing = db_session.query(cls).filter(
            cls.user_id == user_id,
            cls.limit_type == limit_type,
            cls.window_type == window_type,
            cls.window_end > now
        ).first()

        if existing:
            return existing

        # Create new rate limit
        new_limit = cls(
            user_id=user_id,
            limit_type=limit_type,
            window_type=window_type,
            limit_value=limit_value
        )

        db_session.add(new_limit)
        return new_limit

    @staticmethod
    def get_user_limits(user_id: int, db_session):
        """Get all rate limits for a user."""
        return db_session.query(RateLimit).filter(
            RateLimit.user_id == user_id
        ).order_by(RateLimit.limit_type, RateLimit.window_type).all()

    @staticmethod
    def cleanup_expired(db_session):
        """Clean up expired rate limits."""
        now = datetime.utcnow()
        deleted_count = db_session.query(RateLimit).filter(
            RateLimit.window_end <= now
        ).delete()

        return deleted_count

    @staticmethod
    def get_user_stats(user_id: int, db_session, hours: int = 24) -> Dict[str, Any]:
        """Get rate limit statistics for a user."""
        from datetime import timedelta
        from sqlalchemy import func

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        stats = db_session.query(
            func.count(RateLimit.id).label('total_limits'),
            func.sum(RateLimit.current_count).label('total_requests'),
            func.avg(RateLimit.current_count).label('avg_requests_per_limit'),
            func.max(RateLimit.current_count).label('max_requests_in_window')
        ).filter(
            RateLimit.user_id == user_id,
            RateLimit.created_at >= cutoff_time
        ).first()

        # Get current active limits
        active_limits = db_session.query(RateLimit).filter(
            RateLimit.user_id == user_id,
            RateLimit.window_end > datetime.utcnow()
        ).all()

        return {
            'total_limits': stats.total_limits or 0,
            'total_requests': stats.total_requests or 0,
            'avg_requests_per_limit': float(stats.avg_requests_per_limit) if stats.avg_requests_per_limit else 0,
            'max_requests_in_window': stats.max_requests_in_window or 0,
            'active_limits': len(active_limits),
            'active_limits_details': [limit.to_dict() for limit in active_limits],
        }