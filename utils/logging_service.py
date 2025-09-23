"""
Comprehensive logging service for TTS API
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from functools import wraps

from flask import request, g
from ..config import get_settings
from ..models import RequestLog, SystemMetric
from .redis_manager import redis_manager


class LoggingService:
    """Comprehensive logging service for TTS API."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

    def log_request(
        self,
        request_id: int,
        user_id: int,
        message: str,
        level: str = "info",
        **kwargs
    ) -> RequestLog:
        """Log a request event."""
        from ..app.extensions import db

        # Create log entry
        log_entry = RequestLog(
            request_id=request_id,
            user_id=user_id,
            message=message,
            level=level.upper(),
            **kwargs
        )

        # Add request context if available
        if request:
            log_entry.ip_address = request.remote_addr
            log_entry.user_agent = request.headers.get('User-Agent', '')
            log_entry.request_id_header = request.headers.get('X-Request-ID', '')

        # Add correlation ID if available
        if hasattr(g, 'correlation_id'):
            log_entry.correlation_id = g.correlation_id

        # Add to database
        db.session.add(log_entry)
        db.session.commit()

        # Also log to Redis for real-time monitoring
        self._log_to_redis(log_entry)

        return log_entry

    def log_error(
        self,
        request_id: int,
        user_id: int,
        message: str,
        error_code: Optional[str] = None,
        **kwargs
    ) -> RequestLog:
        """Log an error event."""
        return self.log_request(
            request_id=request_id,
            user_id=user_id,
            message=message,
            level="error",
            error_code=error_code,
            **kwargs
        )

    def log_performance(
        self,
        request_id: int,
        user_id: int,
        operation: str,
        duration: int,
        **kwargs
    ) -> RequestLog:
        """Log a performance event."""
        return self.log_request(
            request_id=request_id,
            user_id=user_id,
            message=f"Operation '{operation}' completed",
            level="info",
            operation=operation,
            duration=duration,
            **kwargs
        )

    def log_system_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "gauge",
        category: str = "system",
        **kwargs
    ) -> SystemMetric:
        """Log a system metric."""
        from ..app.extensions import db

        # Create metric entry
        metric = SystemMetric(
            name=name,
            value=value,
            type=metric_type,
            category=category,
            **kwargs
        )

        # Add to database
        db.session.add(metric)
        db.session.commit()

        # Also send to Redis for real-time metrics
        self._send_metric_to_redis(metric)

        return metric

    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: int,
        user_id: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log API request for monitoring."""
        # Log to system metrics
        self.log_system_metric(
            name="api_request_duration",
            value=duration,
            metric_type="histogram",
            category="api",
            labels={
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code),
                "user_id": str(user_id) if user_id else "anonymous"
            }
        )

        # Log to system metrics
        self.log_system_metric(
            name="api_request_count",
            value=1,
            metric_type="counter",
            category="api",
            labels={
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code)
            }
        )

    def log_tts_processing(
        self,
        request_id: int,
        user_id: int,
        text_length: int,
        processing_time: int,
        voice_name: str,
        **kwargs
    ) -> None:
        """Log TTS processing metrics."""
        # Log processing time
        self.log_system_metric(
            name="tts_processing_time",
            value=processing_time,
            metric_type="histogram",
            category="tts",
            labels={
                "voice_name": voice_name,
                "text_length": str(text_length)
            }
        )

        # Log request count
        self.log_system_metric(
            name="tts_request_count",
            value=1,
            metric_type="counter",
            category="tts",
            labels={"voice_name": voice_name}
        )

        # Log text length distribution
        self.log_system_metric(
            name="tts_text_length",
            value=text_length,
            metric_type="histogram",
            category="tts"
        )

    def log_queue_operation(
        self,
        operation: str,
        queue_name: str,
        duration: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log queue operation metrics."""
        # Log queue operation count
        self.log_system_metric(
            name="queue_operation_count",
            value=1,
            metric_type="counter",
            category="system",
            labels={
                "operation": operation,
                "queue_name": queue_name
            }
        )

        # Log duration if provided
        if duration:
            self.log_system_metric(
                name="queue_operation_duration",
                value=duration,
                metric_type="histogram",
                category="system",
                labels={
                    "operation": operation,
                    "queue_name": queue_name
                }
            )

    def get_request_logs(self, request_id: int, limit: int = 100) -> List[RequestLog]:
        """Get logs for a specific request."""
        from ..app.extensions import db
        return db.session.query(RequestLog).filter(
            RequestLog.request_id == request_id
        ).order_by(RequestLog.created_at.desc()).limit(limit).all()

    def get_user_logs(self, user_id: int, limit: int = 100, offset: int = 0) -> List[RequestLog]:
        """Get logs for a specific user."""
        from ..app.extensions import db
        return db.session.query(RequestLog).filter(
            RequestLog.user_id == user_id
        ).order_by(RequestLog.created_at.desc()).offset(offset).limit(limit).all()

    def get_error_logs(self, limit: int = 50, component: Optional[str] = None) -> List[RequestLog]:
        """Get error logs."""
        from ..app.extensions import db
        query = db.session.query(RequestLog).filter(
            RequestLog.level.in_(["ERROR", "CRITICAL"])
        )

        if component:
            query = query.filter(RequestLog.component == component)

        return query.order_by(RequestLog.created_at.desc()).limit(limit).all()

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary."""
        from ..app.extensions import db
        from sqlalchemy import func
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        summary = db.session.query(
            func.count(RequestLog.id).label('total_logs'),
            func.sum(RequestLog.duration).label('total_duration'),
            func.avg(RequestLog.duration).label('avg_duration'),
            func.count(func.case((RequestLog.level == "ERROR", 1))).label('error_count'),
            func.count(func.case((RequestLog.level == "CRITICAL", 1))).label('critical_count')
        ).filter(
            RequestLog.created_at >= cutoff_time,
            RequestLog.duration.isnot(None)
        ).first()

        return {
            'total_logs': summary.total_logs or 0,
            'total_duration_ms': summary.total_duration or 0,
            'avg_duration_ms': float(summary.avg_duration) if summary.avg_duration else 0,
            'error_count': summary.error_count or 0,
            'critical_count': summary.critical_count or 0,
            'error_rate': (summary.error_count / summary.total_logs * 100) if summary.total_logs > 0 else 0,
        }

    def _log_to_redis(self, log_entry: RequestLog) -> None:
        """Log to Redis for real-time monitoring."""
        try:
            # Create log data for Redis
            log_data = {
                "id": log_entry.id,
                "request_id": log_entry.request_id,
                "user_id": log_entry.user_id,
                "level": log_entry.level,
                "message": log_entry.message,
                "timestamp": log_entry.created_at.isoformat() if log_entry.created_at else None,
            }

            # Add to Redis stream for real-time processing
            redis_key = "logs:stream"
            # Note: In a real implementation, you would use Redis streams
            # For now, we'll just use a simple list
            # await redis_manager.redis.xadd(redis_key, log_data)

        except Exception as e:
            self.logger.error(f"Failed to log to Redis: {e}")

    def _send_metric_to_redis(self, metric: SystemMetric) -> None:
        """Send metric to Redis for real-time monitoring."""
        try:
            # Create metric data for Redis
            metric_data = {
                "name": metric.name,
                "value": metric.value,
                "type": metric.type,
                "category": metric.category,
                "timestamp": metric.timestamp.isoformat() if metric.timestamp else None,
            }

            # Add to Redis for real-time metrics
            redis_key = "metrics:stream"
            # Note: In a real implementation, you would use Redis streams
            # For now, we'll just use a simple hash
            # await redis_manager.redis.xadd(redis_key, metric_data)

        except Exception as e:
            self.logger.error(f"Failed to send metric to Redis: {e}")

    @contextmanager
    def log_operation(self, operation: str, request_id: int, user_id: int):
        """Context manager to log operation duration."""
        start_time = datetime.utcnow()

        try:
            yield
        finally:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.log_performance(
                request_id=request_id,
                user_id=user_id,
                operation=operation,
                duration=duration
            )

    def log_function_call(self, func):
        """Decorator to log function calls."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()

            try:
                result = func(*args, **kwargs)
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                # Log successful function call
                self.logger.info(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        "function_name": func.__name__,
                        "duration_ms": duration,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )

                return result

            except Exception as e:
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                # Log function call failure
                self.logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        "function_name": func.__name__,
                        "duration_ms": duration,
                        "error": str(e),
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )

                raise

        return wrapper


# Global logging service instance
logging_service = LoggingService()


# Utility functions for easy access
def get_logging_service() -> LoggingService:
    """Get logging service instance."""
    return logging_service


def log_request(request_id: int, user_id: int, message: str, level: str = "info", **kwargs):
    """Log a request event."""
    return logging_service.log_request(request_id, user_id, message, level, **kwargs)


def log_error(request_id: int, user_id: int, message: str, error_code: Optional[str] = None, **kwargs):
    """Log an error event."""
    return logging_service.log_error(request_id, user_id, message, error_code, **kwargs)


def log_performance(request_id: int, user_id: int, operation: str, duration: int, **kwargs):
    """Log a performance event."""
    return logging_service.log_performance(request_id, user_id, operation, duration, **kwargs)


def log_system_metric(name: str, value: float, metric_type: str = "gauge", category: str = "system", **kwargs):
    """Log a system metric."""
    return logging_service.log_system_metric(name, value, metric_type, category, **kwargs)


def log_operation(operation: str, request_id: int, user_id: int):
    """Context manager to log operation duration."""
    return logging_service.log_operation(operation, request_id, user_id)