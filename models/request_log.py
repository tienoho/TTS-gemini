"""
RequestLog model for comprehensive logging of TTS requests
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import Column, DateTime, Integer, String, Text, JSON, ForeignKey, Index, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class LogLevel(str, Enum):
    """Log level enum."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RequestLog(Base):
    """RequestLog model for storing detailed logs of TTS requests."""

    __tablename__ = 'request_logs'

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey('requests.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)

    # Log details
    level = Column(String(20), default=LogLevel.INFO, index=True)
    message = Column(Text, nullable=False)
    error_code = Column(String(50), nullable=True)
    details = Column(JSON, default=dict)  # Additional structured data

    # Context information
    source = Column(String(100), default="api")  # api, worker, background
    component = Column(String(100), default="unknown")  # tts, auth, storage, etc.
    operation = Column(String(100), nullable=True)  # create, process, upload, etc.

    # Performance metrics
    duration = Column(Integer, nullable=True)  # in milliseconds
    memory_usage = Column(Integer, nullable=True)  # in bytes
    cpu_usage = Column(Float, nullable=True)  # percentage

    # External service calls
    external_service = Column(String(100), nullable=True)  # gemini, s3, redis, etc.
    external_duration = Column(Integer, nullable=True)  # in milliseconds
    external_cost = Column(Float, nullable=True)  # API cost

    # Request context
    ip_address = Column(String(45), nullable=True)  # IPv4/IPv6
    user_agent = Column(String(500), nullable=True)
    request_id_header = Column(String(100), nullable=True)  # X-Request-ID
    correlation_id = Column(String(100), nullable=True)  # For tracing

    # Metadata
    metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    request = relationship("AudioRequest", back_populates="logs")
    user = relationship("User")

    # Indexes for performance
    __table_args__ = (
        Index('idx_logs_request_level', 'request_id', 'level'),
        Index('idx_logs_user_created', 'user_id', 'created_at'),
        Index('idx_logs_component_operation', 'component', 'operation'),
        Index('idx_logs_correlation_id', 'correlation_id'),
    )

    def __init__(self, request_id: int, user_id: int, message: str, **kwargs):
        """Initialize request log."""
        self.request_id = request_id
        self.user_id = user_id
        self.message = message

        # Set defaults
        if 'level' not in kwargs:
            kwargs['level'] = LogLevel.INFO
        if 'source' not in kwargs:
            kwargs['source'] = 'api'

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of request log."""
        return f"<RequestLog(id={self.id}, request_id={self.request_id}, level='{self.level}', message='{self.message[:50]}...')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert log to dictionary."""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'level': self.level,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'source': self.source,
            'component': self.component,
            'operation': self.operation,
            'duration': self.duration,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'external_service': self.external_service,
            'external_duration': self.external_duration,
            'external_cost': self.external_cost,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_id_header': self.request_id_header,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def create_error_log(
        cls,
        request_id: int,
        user_id: int,
        message: str,
        error_code: Optional[str] = None,
        **kwargs
    ) -> 'RequestLog':
        """Create an error log entry."""
        return cls(
            request_id=request_id,
            user_id=user_id,
            message=message,
            level=LogLevel.ERROR,
            error_code=error_code,
            **kwargs
        )

    @classmethod
    def create_performance_log(
        cls,
        request_id: int,
        user_id: int,
        operation: str,
        duration: int,
        **kwargs
    ) -> 'RequestLog':
        """Create a performance log entry."""
        return cls(
            request_id=request_id,
            user_id=user_id,
            message=f"Operation '{operation}' completed",
            level=LogLevel.INFO,
            operation=operation,
            duration=duration,
            **kwargs
        )

    @staticmethod
    def get_logs_by_request(request_id: int, db_session, limit: int = 100):
        """Get logs for a specific request."""
        return db_session.query(RequestLog).filter(
            RequestLog.request_id == request_id
        ).order_by(RequestLog.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_logs_by_user(user_id: int, db_session, limit: int = 100, offset: int = 0):
        """Get logs for a specific user."""
        return db_session.query(RequestLog).filter(
            RequestLog.user_id == user_id
        ).order_by(RequestLog.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def get_error_logs(db_session, limit: int = 50, component: Optional[str] = None):
        """Get error logs with optional component filter."""
        query = db_session.query(RequestLog).filter(
            RequestLog.level.in_([LogLevel.ERROR, LogLevel.CRITICAL])
        )

        if component:
            query = query.filter(RequestLog.component == component)

        return query.order_by(RequestLog.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_performance_summary(db_session, hours: int = 24):
        """Get performance summary for the last N hours."""
        from datetime import timedelta
        from sqlalchemy import func

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        summary = db_session.query(
            func.count(RequestLog.id).label('total_logs'),
            func.sum(RequestLog.duration).label('total_duration'),
            func.avg(RequestLog.duration).label('avg_duration'),
            func.sum(RequestLog.memory_usage).label('total_memory'),
            func.avg(RequestLog.memory_usage).label('avg_memory'),
            func.count(func.case((RequestLog.level == LogLevel.ERROR, 1))).label('error_count'),
            func.count(func.case((RequestLog.level == LogLevel.CRITICAL, 1))).label('critical_count')
        ).filter(
            RequestLog.created_at >= cutoff_time,
            RequestLog.duration.isnot(None)
        ).first()

        return {
            'total_logs': summary.total_logs or 0,
            'total_duration_ms': summary.total_duration or 0,
            'avg_duration_ms': float(summary.avg_duration) if summary.avg_duration else 0,
            'total_memory_bytes': summary.total_memory or 0,
            'avg_memory_bytes': float(summary.avg_memory) if summary.avg_memory else 0,
            'error_count': summary.error_count or 0,
            'critical_count': summary.critical_count or 0,
            'error_rate': (summary.error_count / summary.total_logs * 100) if summary.total_logs > 0 else 0,
        }