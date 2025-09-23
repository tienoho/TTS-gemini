"""
Metrics model for tracking system metrics and analytics
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import Column, DateTime, Integer, String, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class MetricType(str, Enum):
    """Metric type enum."""

    COUNTER = "counter"  # Incrementing values
    GAUGE = "gauge"  # Current values
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurements


class MetricCategory(str, Enum):
    """Metric category enum."""

    SYSTEM = "system"  # System metrics
    API = "api"  # API metrics
    TTS = "tts"  # TTS processing metrics
    STORAGE = "storage"  # Storage metrics
    USER = "user"  # User metrics
    BUSINESS = "business"  # Business metrics


class SystemMetric(Base):
    """SystemMetric model for tracking system performance metrics."""

    __tablename__ = 'system_metrics'

    id = Column(Integer, primary_key=True, index=True)

    # Metric identification
    name = Column(String(100), nullable=False, index=True)  # e.g., "cpu_usage", "memory_usage"
    type = Column(String(20), nullable=False, index=True)  # counter, gauge, histogram, timer
    category = Column(String(20), nullable=False, index=True)  # system, api, tts, etc.

    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)  # %, bytes, seconds, count, etc.

    # Metadata
    labels = Column(JSON, default=dict)  # Additional dimensions
    metadata = Column(JSON, default=dict)

    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes for performance
    __table_args__ = (
        Index('idx_metrics_name_timestamp', 'name', 'timestamp'),
        Index('idx_metrics_category_timestamp', 'category', 'timestamp'),
        Index('idx_metrics_type_timestamp', 'type', 'timestamp'),
    )

    def __init__(self, name: str, value: float, metric_type: str, category: str, **kwargs):
        """Initialize system metric."""
        self.name = name
        self.value = value
        self.type = metric_type
        self.category = category

        # Set defaults
        if 'unit' not in kwargs:
            kwargs['unit'] = self._get_default_unit(name)

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of system metric."""
        return f"<SystemMetric(name='{self.name}', value={self.value}, category='{self.category}')>"

    def _get_default_unit(self, name: str) -> Optional[str]:
        """Get default unit for metric name."""
        unit_map = {
            'cpu_usage': '%',
            'memory_usage': 'bytes',
            'disk_usage': 'bytes',
            'network_io': 'bytes',
            'request_count': 'count',
            'response_time': 'seconds',
            'error_rate': '%',
            'processing_time': 'seconds',
            'queue_length': 'count',
            'active_connections': 'count',
        }
        return unit_map.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'category': self.category,
            'value': self.value,
            'unit': self.unit,
            'labels': self.labels,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def create_counter(cls, name: str, value: float = 1.0, category: str = MetricCategory.SYSTEM, **kwargs) -> 'SystemMetric':
        """Create a counter metric."""
        return cls(name=name, value=value, metric_type=MetricType.COUNTER, category=category, **kwargs)

    @classmethod
    def create_gauge(cls, name: str, value: float, category: str = MetricCategory.SYSTEM, **kwargs) -> 'SystemMetric':
        """Create a gauge metric."""
        return cls(name=name, value=value, metric_type=MetricType.GAUGE, category=category, **kwargs)

    @classmethod
    def create_histogram(cls, name: str, value: float, category: str = MetricCategory.SYSTEM, **kwargs) -> 'SystemMetric':
        """Create a histogram metric."""
        return cls(name=name, value=value, metric_type=MetricType.HISTOGRAM, category=category, **kwargs)

    @classmethod
    def create_timer(cls, name: str, value: float, category: str = MetricCategory.SYSTEM, **kwargs) -> 'SystemMetric':
        """Create a timer metric."""
        return cls(name=name, value=value, metric_type=MetricType.TIMER, category=category, **kwargs)

    @staticmethod
    def get_metrics_by_name(name: str, db_session, limit: int = 100, hours: Optional[int] = None):
        """Get metrics by name with optional time filter."""
        query = db_session.query(SystemMetric).filter(SystemMetric.name == name)

        if hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = query.filter(SystemMetric.timestamp >= cutoff_time)

        return query.order_by(SystemMetric.timestamp.desc()).limit(limit).all()

    @staticmethod
    def get_metrics_by_category(category: str, db_session, limit: int = 100, hours: Optional[int] = None):
        """Get metrics by category with optional time filter."""
        query = db_session.query(SystemMetric).filter(SystemMetric.category == category)

        if hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = query.filter(SystemMetric.timestamp >= cutoff_time)

        return query.order_by(SystemMetric.timestamp.desc()).limit(limit).all()

    @staticmethod
    def get_latest_metrics(db_session, names: Optional[list] = None, categories: Optional[list] = None):
        """Get latest values for specified metrics."""
        from sqlalchemy import func

        query = db_session.query(
            SystemMetric.name,
            SystemMetric.category,
            SystemMetric.value,
            SystemMetric.unit,
            SystemMetric.labels,
            func.max(SystemMetric.timestamp).label('latest_timestamp')
        )

        if names:
            query = query.filter(SystemMetric.name.in_(names))

        if categories:
            query = query.filter(SystemMetric.category.in_(categories))

        query = query.group_by(
            SystemMetric.name,
            SystemMetric.category,
            SystemMetric.value,
            SystemMetric.unit,
            SystemMetric.labels
        )

        results = query.all()

        return {
            result.name: {
                'value': result.value,
                'unit': result.unit,
                'labels': result.labels,
                'category': result.category,
                'timestamp': result.latest_timestamp.isoformat() if result.latest_timestamp else None,
            }
            for result in results
        }

    @staticmethod
    def get_aggregated_metrics(db_session, hours: int = 1, group_by: str = 'name'):
        """Get aggregated metrics over time period."""
        from sqlalchemy import func
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        if group_by == 'name':
            group_fields = [SystemMetric.name, SystemMetric.category, SystemMetric.unit]
        elif group_by == 'category':
            group_fields = [SystemMetric.category]
        else:
            raise ValueError(f"Invalid group_by value: {group_by}")

        # Build aggregation query
        query = db_session.query(
            *group_fields,
            func.count(SystemMetric.id).label('count'),
            func.sum(SystemMetric.value).label('sum'),
            func.avg(SystemMetric.value).label('avg'),
            func.min(SystemMetric.value).label('min'),
            func.max(SystemMetric.value).label('max'),
            func.max(SystemMetric.timestamp).label('latest_timestamp')
        ).filter(
            SystemMetric.timestamp >= cutoff_time
        ).group_by(*group_fields)

        results = query.all()

        return [
            {
                'name': result.name if hasattr(result, 'name') else None,
                'category': result.category if hasattr(result, 'category') else None,
                'unit': result.unit if hasattr(result, 'unit') else None,
                'count': result.count,
                'sum': float(result.sum),
                'avg': float(result.avg),
                'min': float(result.min),
                'max': float(result.max),
                'latest_timestamp': result.latest_timestamp.isoformat() if result.latest_timestamp else None,
            }
            for result in results
        ]

    @staticmethod
    def cleanup_old_metrics(db_session, days: int = 30):
        """Clean up metrics older than specified days."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        deleted_count = db_session.query(SystemMetric).filter(
            SystemMetric.timestamp < cutoff_time
        ).delete()

        return deleted_count