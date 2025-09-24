"""
Advanced Analytics Models for TTS System Dashboard
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, DateTime, Integer, String, Float, JSON, Boolean, Text, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MetricPeriod(str, Enum):
    """Time period for analytics aggregation."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReportType(str, Enum):
    """Report type enum."""

    USAGE = "usage"
    PERFORMANCE = "performance"
    USER_BEHAVIOR = "user_behavior"
    BUSINESS = "business"
    CUSTOM = "custom"


class UsageMetric(Base):
    """Usage metrics for tracking system usage patterns."""

    __tablename__ = 'usage_metrics'

    id = Column(Integer, primary_key=True, index=True)

    # Time dimensions
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    period = Column(String(20), nullable=False, index=True)  # minute, hour, day, week, month
    date_key = Column(String(10), index=True)  # YYYY-MM-DD format

    # Usage dimensions
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    endpoint = Column(String(100), index=True)  # API endpoint used
    voice_type = Column(String(50), index=True)  # Voice type used
    language = Column(String(10), index=True)  # Language code

    # Usage metrics
    request_count = Column(Integer, default=0)
    audio_duration_seconds = Column(Float, default=0.0)  # Total audio duration processed
    data_processed_mb = Column(Float, default=0.0)  # Data processed in MB
    error_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)

    # Performance metrics
    avg_response_time = Column(Float, default=0.0)  # Average response time in seconds
    p95_response_time = Column(Float, default=0.0)  # 95th percentile response time
    throughput = Column(Float, default=0.0)  # Requests per second

    # Cost and billing
    estimated_cost = Column(Float, default=0.0)  # Estimated cost for usage
    tokens_used = Column(Integer, default=0)  # API tokens consumed

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes for performance
    __table_args__ = (
        Index('idx_usage_metrics_timestamp_period', 'timestamp', 'period'),
        Index('idx_usage_metrics_user_date', 'user_id', 'date_key'),
        Index('idx_usage_metrics_org_date', 'organization_id', 'date_key'),
        Index('idx_usage_metrics_endpoint_date', 'endpoint', 'date_key'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'period': self.period,
            'date_key': self.date_key,
            'user_id': self.user_id,
            'organization_id': self.organization_id,
            'endpoint': self.endpoint,
            'voice_type': self.voice_type,
            'language': self.language,
            'request_count': self.request_count,
            'audio_duration_seconds': self.audio_duration_seconds,
            'data_processed_mb': self.data_processed_mb,
            'error_count': self.error_count,
            'success_count': self.success_count,
            'avg_response_time': self.avg_response_time,
            'p95_response_time': self.p95_response_time,
            'throughput': self.throughput,
            'estimated_cost': self.estimated_cost,
            'tokens_used': self.tokens_used,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class UserBehavior(Base):
    """User behavior tracking for analytics."""

    __tablename__ = 'user_behavior'

    id = Column(Integer, primary_key=True, index=True)

    # User identification
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    session_id = Column(String(100), index=True)  # Unique session identifier

    # Session information
    session_start = Column(DateTime, default=datetime.utcnow, index=True)
    session_end = Column(DateTime, nullable=True)
    session_duration_seconds = Column(Float, default=0.0)

    # User actions
    actions_count = Column(Integer, default=0)
    pages_visited = Column(JSON, default=list)  # List of pages/URLs visited
    features_used = Column(JSON, default=list)  # List of features used
    errors_encountered = Column(JSON, default=list)  # List of errors encountered

    # Device and browser info
    user_agent = Column(Text, nullable=True)
    device_type = Column(String(50), default='unknown')  # desktop, mobile, tablet
    browser = Column(String(50), default='unknown')
    os = Column(String(50), default='unknown')

    # Geographic info
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    country = Column(String(2), nullable=True)  # ISO country code
    region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)

    # Engagement metrics
    engagement_score = Column(Float, default=0.0)  # Calculated engagement score
    conversion_events = Column(JSON, default=list)  # List of conversion events
    retention_category = Column(String(20), default='new')  # new, returning, loyal

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_user_behavior_user_session', 'user_id', 'session_id'),
        Index('idx_user_behavior_session_start', 'session_start'),
        Index('idx_user_behavior_engagement', 'engagement_score'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'organization_id': self.organization_id,
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat() if self.session_start else None,
            'session_end': self.session_end.isoformat() if self.session_end else None,
            'session_duration_seconds': self.session_duration_seconds,
            'actions_count': self.actions_count,
            'pages_visited': self.pages_visited,
            'features_used': self.features_used,
            'errors_encountered': self.errors_encountered,
            'user_agent': self.user_agent,
            'device_type': self.device_type,
            'browser': self.browser,
            'os': self.os,
            'ip_address': self.ip_address,
            'country': self.country,
            'region': self.region,
            'city': self.city,
            'engagement_score': self.engagement_score,
            'conversion_events': self.conversion_events,
            'retention_category': self.retention_category,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class PerformanceMetric(Base):
    """System performance metrics for monitoring."""

    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True, index=True)

    # Time dimensions
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    period = Column(String(20), nullable=False, index=True)

    # System resources
    cpu_usage_percent = Column(Float, default=0.0)
    memory_usage_mb = Column(Float, default=0.0)
    disk_usage_mb = Column(Float, default=0.0)
    network_io_mb = Column(Float, default=0.0)

    # Application metrics
    active_connections = Column(Integer, default=0)
    queue_length = Column(Integer, default=0)
    thread_count = Column(Integer, default=0)
    db_connections = Column(Integer, default=0)

    # Response times
    avg_response_time = Column(Float, default=0.0)
    p50_response_time = Column(Float, default=0.0)  # 50th percentile
    p95_response_time = Column(Float, default=0.0)  # 95th percentile
    p99_response_time = Column(Float, default=0.0)  # 99th percentile

    # Error rates
    error_rate_percent = Column(Float, default=0.0)
    timeout_rate_percent = Column(Float, default=0.0)
    retry_rate_percent = Column(Float, default=0.0)

    # TTS specific metrics
    tts_processing_time = Column(Float, default=0.0)
    audio_generation_rate = Column(Float, default=0.0)  # Audio files per second
    voice_cloning_success_rate = Column(Float, default=0.0)

    # Cache metrics
    cache_hit_rate = Column(Float, default=0.0)
    cache_size_mb = Column(Float, default=0.0)

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_performance_metrics_timestamp', 'timestamp'),
        Index('idx_performance_metrics_period', 'period'),
        Index('idx_performance_metrics_cpu_memory', 'cpu_usage_percent', 'memory_usage_mb'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'period': self.period,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'disk_usage_mb': self.disk_usage_mb,
            'network_io_mb': self.network_io_mb,
            'active_connections': self.active_connections,
            'queue_length': self.queue_length,
            'thread_count': self.thread_count,
            'db_connections': self.db_connections,
            'avg_response_time': self.avg_response_time,
            'p50_response_time': self.p50_response_time,
            'p95_response_time': self.p95_response_time,
            'p99_response_time': self.p99_response_time,
            'error_rate_percent': self.error_rate_percent,
            'timeout_rate_percent': self.timeout_rate_percent,
            'retry_rate_percent': self.retry_rate_percent,
            'tts_processing_time': self.tts_processing_time,
            'audio_generation_rate': self.audio_generation_rate,
            'voice_cloning_success_rate': self.voice_cloning_success_rate,
            'cache_hit_rate': self.cache_hit_rate,
            'cache_size_mb': self.cache_size_mb,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class BusinessMetric(Base):
    """Business intelligence metrics."""

    __tablename__ = 'business_metrics'

    id = Column(Integer, primary_key=True, index=True)

    # Time dimensions
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    period = Column(String(20), nullable=False, index=True)
    date_key = Column(String(10), index=True)

    # Revenue metrics
    revenue = Column(Float, default=0.0)
    cost = Column(Float, default=0.0)
    profit = Column(Float, default=0.0)
    margin_percent = Column(Float, default=0.0)

    # Customer metrics
    new_customers = Column(Integer, default=0)
    active_customers = Column(Integer, default=0)
    churned_customers = Column(Integer, default=0)
    customer_lifetime_value = Column(Float, default=0.0)

    # Usage metrics
    total_requests = Column(Integer, default=0)
    paid_requests = Column(Integer, default=0)
    free_requests = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)

    # Product metrics
    popular_voices = Column(JSON, default=list)  # List of most used voices
    popular_languages = Column(JSON, default=list)  # List of most used languages
    feature_adoption = Column(JSON, default=dict)  # Feature usage statistics

    # Growth metrics
    growth_rate_percent = Column(Float, default=0.0)
    retention_rate_percent = Column(Float, default=0.0)
    conversion_rate_percent = Column(Float, default=0.0)

    # Organization metrics
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    tier = Column(String(20), index=True)  # Organization tier

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_business_metrics_timestamp_period', 'timestamp', 'period'),
        Index('idx_business_metrics_org_date', 'organization_id', 'date_key'),
        Index('idx_business_metrics_revenue', 'revenue'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'period': self.period,
            'date_key': self.date_key,
            'revenue': self.revenue,
            'cost': self.cost,
            'profit': self.profit,
            'margin_percent': self.margin_percent,
            'new_customers': self.new_customers,
            'active_customers': self.active_customers,
            'churned_customers': self.churned_customers,
            'customer_lifetime_value': self.customer_lifetime_value,
            'total_requests': self.total_requests,
            'paid_requests': self.paid_requests,
            'free_requests': self.free_requests,
            'api_calls': self.api_calls,
            'popular_voices': self.popular_voices,
            'popular_languages': self.popular_languages,
            'feature_adoption': self.feature_adoption,
            'growth_rate_percent': self.growth_rate_percent,
            'retention_rate_percent': self.retention_rate_percent,
            'conversion_rate_percent': self.conversion_rate_percent,
            'organization_id': self.organization_id,
            'tier': self.tier,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class TimeSeriesData(Base):
    """Time series data for trend analysis and forecasting."""

    __tablename__ = 'time_series_data'

    id = Column(Integer, primary_key=True, index=True)

    # Data identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # usage, performance, business
    data_source = Column(String(50), nullable=False, index=True)  # api, system, user, etc.

    # Time dimensions
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    period = Column(String(20), nullable=False, index=True)
    date_key = Column(String(10), index=True)

    # Data values
    value = Column(Float, nullable=False)
    previous_value = Column(Float, nullable=True)
    change_percent = Column(Float, default=0.0)

    # Statistical measures
    mean = Column(Float, nullable=True)
    median = Column(Float, nullable=True)
    std_dev = Column(Float, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)

    # Trend analysis
    trend_direction = Column(String(10), default='stable')  # up, down, stable
    trend_strength = Column(Float, default=0.0)  # Strength of trend (0-1)
    seasonality_score = Column(Float, default=0.0)  # Seasonality score (0-1)

    # Forecasting
    forecast_value = Column(Float, nullable=True)
    forecast_confidence = Column(Float, default=0.0)  # Confidence level (0-1)
    forecast_error = Column(Float, nullable=True)  # Forecast error

    # Anomaly detection
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)
    anomaly_confidence = Column(Float, default=0.0)

    # Metadata
    labels = Column(JSON, default=dict)  # Additional dimensions
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_time_series_metric_timestamp', 'metric_name', 'timestamp'),
        Index('idx_time_series_type_period', 'metric_type', 'period'),
        Index('idx_time_series_anomaly', 'is_anomaly', 'anomaly_score'),
        Index('idx_time_series_trend', 'trend_direction', 'trend_strength'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'metric_name': self.metric_name,
            'metric_type': self.metric_type,
            'data_source': self.data_source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'period': self.period,
            'date_key': self.date_key,
            'value': self.value,
            'previous_value': self.previous_value,
            'change_percent': self.change_percent,
            'mean': self.mean,
            'median': self.median,
            'std_dev': self.std_dev,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'seasonality_score': self.seasonality_score,
            'forecast_value': self.forecast_value,
            'forecast_confidence': self.forecast_confidence,
            'forecast_error': self.forecast_error,
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_score,
            'anomaly_confidence': self.anomaly_confidence,
            'labels': self.labels,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class AnalyticsAlert(Base):
    """Alert system for analytics anomalies and thresholds."""

    __tablename__ = 'analytics_alerts'

    id = Column(Integer, primary_key=True, index=True)

    # Alert identification
    alert_name = Column(String(100), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False, index=True)  # threshold, anomaly, trend
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical

    # Alert conditions
    metric_name = Column(String(100), nullable=False, index=True)
    threshold_value = Column(Float, nullable=True)
    threshold_operator = Column(String(10), nullable=True)  # >, <, >=, <=, ==, !=

    # Alert status
    status = Column(String(20), default='active', index=True)  # active, resolved, acknowledged
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Alert details
    current_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=True)
    deviation_percent = Column(Float, default=0.0)

    # Alert context
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)

    # Alert metadata
    description = Column(Text, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    request_metadata = Column(JSON, default=dict)

    # Notification settings
    notification_channels = Column(JSON, default=list)  # email, slack, webhook
    notification_sent = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_analytics_alerts_status', 'status'),
        Index('idx_analytics_alerts_severity', 'severity'),
        Index('idx_analytics_alerts_metric', 'metric_name', 'triggered_at'),
        Index('idx_analytics_alerts_org', 'organization_id', 'status'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'alert_name': self.alert_name,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'metric_name': self.metric_name,
            'threshold_value': self.threshold_value,
            'threshold_operator': self.threshold_operator,
            'status': self.status,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'current_value': self.current_value,
            'expected_value': self.expected_value,
            'deviation_percent': self.deviation_percent,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'description': self.description,
            'resolution_notes': self.resolution_notes,
            'metadata': self.request_metadata,
            'notification_channels': self.notification_channels,
            'notification_sent': self.notification_sent,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AnalyticsReport(Base):
    """Generated analytics reports."""

    __tablename__ = 'analytics_reports'

    id = Column(Integer, primary_key=True, index=True)

    # Report identification
    report_name = Column(String(200), nullable=False, index=True)
    report_type = Column(String(50), nullable=False, index=True)  # usage, performance, user_behavior, business
    report_format = Column(String(10), nullable=False, default='json')  # json, csv, pdf, excel

    # Report parameters
    date_range_start = Column(DateTime, nullable=False, index=True)
    date_range_end = Column(DateTime, nullable=False, index=True)
    filters = Column(JSON, default=dict)  # Report filters applied
    parameters = Column(JSON, default=dict)  # Report parameters

    # Report content
    data = Column(JSON, nullable=True)  # Report data
    summary = Column(JSON, default=dict)  # Report summary statistics
    charts = Column(JSON, default=list)  # Chart configurations
    insights = Column(JSON, default=list)  # AI-generated insights

    # Report metadata
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    generated_by = Column(Integer, ForeignKey('users.id'), nullable=True)  # User who generated the report

    # Report status
    status = Column(String(20), default='generating', index=True)  # generating, completed, failed
    progress_percent = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    # File information
    file_path = Column(String(500), nullable=True)  # Path to generated file
    file_size_bytes = Column(Integer, nullable=True)
    download_url = Column(String(500), nullable=True)

    # Scheduling (for automated reports)
    is_scheduled = Column(Boolean, default=False)
    schedule_config = Column(JSON, default=dict)  # Cron expression, frequency, etc.
    next_run_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    generated_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_analytics_reports_type_status', 'report_type', 'status'),
        Index('idx_analytics_reports_org_date', 'organization_id', 'date_range_start'),
        Index('idx_analytics_reports_scheduled', 'is_scheduled', 'next_run_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'report_name': self.report_name,
            'report_type': self.report_type,
            'report_format': self.report_format,
            'date_range_start': self.date_range_start.isoformat() if self.date_range_start else None,
            'date_range_end': self.date_range_end.isoformat() if self.date_range_end else None,
            'filters': self.filters,
            'parameters': self.parameters,
            'data': self.data,
            'summary': self.summary,
            'charts': self.charts,
            'insights': self.insights,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'generated_by': self.generated_by,
            'status': self.status,
            'progress_percent': self.progress_percent,
            'error_message': self.error_message,
            'file_path': self.file_path,
            'file_size_bytes': self.file_size_bytes,
            'download_url': self.download_url,
            'is_scheduled': self.is_scheduled,
            'schedule_config': self.schedule_config,
            'next_run_at': self.next_run_at.isoformat() if self.next_run_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
        }