"""
Celery application for background task processing
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import time
import asyncio

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready, worker_shutdown

from ..config import get_settings
from ..models import AudioRequest, RequestLog, SystemMetric
from ..utils.redis_manager import redis_manager
from ..utils.logging_service import logging_service
from ..utils.auth import auth_service


# Initialize Celery app
settings = get_settings()

celery_app = Celery(
    'tts_worker',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['workers.tasks']
)

# Celery configuration
celery_app.conf.update(
    # Worker settings
    worker_prefetch_multiplier=1,  # Disable prefetch for fair queuing
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    worker_disable_rate_limits=False,
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,

    # Task routing
    task_routes={
        'workers.tasks.process_tts_request': {'queue': 'tts_priority'},
        'workers.tasks.process_tts_batch': {'queue': 'tts_batch'},
        'workers.tasks.cleanup_old_files': {'queue': 'maintenance'},
        'workers.tasks.update_metrics': {'queue': 'monitoring'},
    },

    # Queue definitions
    task_create_missing_queues=True,
    task_default_queue='tts_normal',
    task_default_exchange='tts_exchange',
    task_default_exchange_type='direct',
    task_default_routing_key='tts_normal',

    # Retry settings
    task_always_eager=False,  # Run tasks asynchronously
    task_ignore_result=False,  # Store results
    result_expires=3600,  # Results expire after 1 hour

    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-files-every-hour': {
            'task': 'workers.tasks.cleanup_old_files',
            'schedule': crontab(minute=0),  # Every hour
        },
        'update-metrics-every-minute': {
            'task': 'workers.tasks.update_metrics',
            'schedule': 60.0,  # Every minute
        },
        'retry-failed-requests-every-5-minutes': {
            'task': 'workers.tasks.retry_failed_requests',
            'schedule': 300.0,  # Every 5 minutes
        },
        'cleanup-expired-rate-limits-every-hour': {
            'task': 'workers.tasks.cleanup_expired_rate_limits',
            'schedule': crontab(minute=30),  # Every hour at :30
        },
    },

    # Logging
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    worker_log_color=False,

    # Security
    worker_hijack_root_logger=False,
    worker_log_file=os.path.join(os.getcwd(), 'logs', 'celery_worker.log'),
)

# Worker event handlers
@worker_ready.connect
def worker_ready_handler(**kwargs):
    """Handler called when worker is ready."""
    print("🚀 Celery worker is ready!")
    logging_service.log_system_metric(
        name="celery_worker_status",
        value=1,
        metric_type="gauge",
        category="system",
        labels={"status": "ready"}
    )


@worker_shutdown.connect
def worker_shutdown_handler(**kwargs):
    """Handler called when worker is shutting down."""
    print("👋 Celery worker is shutting down...")
    logging_service.log_system_metric(
        name="celery_worker_status",
        value=0,
        metric_type="gauge",
        category="system",
        labels={"status": "shutdown"}
    )


# Context manager for database sessions
class DatabaseTask:
    """Context manager for database operations in tasks."""

    def __init__(self, task_self):
        self.task_self = task_self
        self.db = None

    def __enter__(self):
        from ..app.extensions import db
        self.db = db
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.session.rollback()
            self.task_self.retry(countdown=60)  # Retry after 1 minute
        else:
            self.db.session.commit()


# Circuit breaker for external services
class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def is_open(self):
        """Check if circuit breaker is open."""
        if self.state == "open":
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Global circuit breaker for Gemini API
gemini_circuit_breaker = CircuitBreaker(
    failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
)


# Utility functions
def get_priority_queue_name(priority: str) -> str:
    """Get queue name for priority."""
    priority_map = {
        "urgent": "tts_queue_urgent",
        "high": "tts_queue_high",
        "normal": "tts_queue_normal",
        "low": "tts_queue_low"
    }
    return priority_map.get(priority, "tts_queue_normal")


def calculate_backoff_delay(attempt: int, base_delay: int = 60) -> int:
    """Calculate exponential backoff delay."""
    return min(base_delay * (2 ** attempt), 3600)  # Max 1 hour


# Initialize app for Celery
if __name__ == '__main__':
    celery_app.start()