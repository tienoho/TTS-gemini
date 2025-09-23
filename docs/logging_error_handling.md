# Comprehensive Logging và Error Handling

## Tổng quan
Hệ thống logging và error handling được thiết kế để đảm bảo observability, debugging, và monitoring của TTS API.

## Logging Architecture

### Log Levels
```python
import logging
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
```

### Structured Logging
```python
import json
import structlog
from datetime import datetime

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info("TTS request started", request_id=12345, user_id=1, text_length=100)
logger.error("Gemini API error", request_id=12345, error_code="QUOTA_EXCEEDED", retry_count=2)
```

### Database Logging
```python
# models/log.py
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class LogEntry(Base):
    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True)
    request_id = Column(Integer, nullable=True)
    user_id = Column(Integer, nullable=True)
    level = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    metadata = Column(JSON, default=dict)
    source = Column(String(100), default='api')
    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'level': self.level,
            'message': self.message,
            'metadata': self.metadata,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }

# Logging utility
class DatabaseLogger:
    def __init__(self, db_session):
        self.db_session = db_session

    def log_request(self, level: str, message: str, request_id: int = None,
                   user_id: int = None, metadata: dict = None):
        """Log entry to database"""
        log_entry = LogEntry(
            request_id=request_id,
            user_id=user_id,
            level=level,
            message=message,
            metadata=metadata or {},
            source='api'
        )

        self.db_session.add(log_entry)
        self.db_session.commit()

        return log_entry.id

    def get_logs(self, request_id: int = None, user_id: int = None,
                level: str = None, limit: int = 100):
        """Retrieve logs from database"""
        query = self.db_session.query(LogEntry)

        if request_id:
            query = query.filter(LogEntry.request_id == request_id)
        if user_id:
            query = query.filter(LogEntry.user_id == user_id)
        if level:
            query = query.filter(LogEntry.level == level)

        return query.order_by(LogEntry.timestamp.desc()).limit(limit).all()
```

## Error Handling Strategy

### Custom Exception Classes
```python
# utils/exceptions.py
from typing import Dict, Any, Optional

class TTSAPIError(Exception):
    """Base exception for TTS API"""
    def __init__(self, message: str, error_code: str = None,
                 status_code: int = 500, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or 'INTERNAL_ERROR'
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class AuthenticationError(TTSAPIError):
    """Authentication related errors"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_ERROR", 401)

class AuthorizationError(TTSAPIError):
    """Authorization related errors"""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "AUTHORIZATION_ERROR", 403)

class ValidationError(TTSAPIError):
    """Input validation errors"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", 400, details)

class RateLimitError(TTSAPIError):
    """Rate limiting errors"""
    def __init__(self, message: str, retry_after: int = None):
        details = {'retry_after': retry_after} if retry_after else {}
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429, details)

class TTSProcessingError(TTSAPIError):
    """TTS processing errors"""
    def __init__(self, message: str, request_id: int = None):
        details = {'request_id': request_id} if request_id else {}
        super().__init__(message, "TTS_PROCESSING_ERROR", 500, details)

class GeminiAPIError(TTSAPIError):
    """Gemini API errors"""
    def __init__(self, message: str, api_error_code: str = None):
        details = {'gemini_error_code': api_error_code} if api_error_code else {}
        super().__init__(message, "GEMINI_API_ERROR", 502, details)

class DatabaseError(TTSAPIError):
    """Database related errors"""
    def __init__(self, message: str, operation: str = None):
        details = {'operation': operation} if operation else {}
        super().__init__(message, "DATABASE_ERROR", 500, details)

class RedisError(TTSAPIError):
    """Redis related errors"""
    def __init__(self, message: str, operation: str = None):
        details = {'operation': operation} if operation else {}
        super().__init__(message, "REDIS_ERROR", 500, details)
```

### Error Handler Middleware
```python
# app/error_handlers.py
from flask import jsonify, request, current_app
from werkzeug.exceptions import HTTPException
import traceback
import time

from utils.exceptions import TTSAPIError, AuthenticationError, ValidationError
from utils.logging import DatabaseLogger

def register_error_handlers(app):
    """Register all error handlers"""

    @app.errorhandler(TTSAPIError)
    def handle_tts_api_error(error):
        """Handle TTS API specific errors"""
        response = {
            'error': error.error_code,
            'message': error.message,
            'timestamp': time.time(),
            'path': request.path,
            'method': request.method
        }

        if error.details:
            response['details'] = error.details

        # Log error
        logger = DatabaseLogger(app.db_session)
        logger.log_request(
            level='ERROR',
            message=error.message,
            metadata={
                'error_code': error.error_code,
                'status_code': error.status_code,
                'path': request.path,
                'method': request.method,
                'user_id': getattr(request, 'user_id', None),
                'details': error.details
            }
        )

        return jsonify(response), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle HTTP exceptions"""
        response = {
            'error': 'HTTP_ERROR',
            'message': error.description,
            'timestamp': time.time(),
            'path': request.path,
            'method': request.method
        }

        # Log error
        logger = DatabaseLogger(app.db_session)
        logger.log_request(
            level='WARNING',
            message=f"HTTP {error.code}: {error.description}",
            metadata={
                'http_code': error.code,
                'path': request.path,
                'method': request.method
            }
        )

        return jsonify(response), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors"""
        # Log full traceback
        error_id = str(time.time())
        current_app.logger.error(
            f"Unexpected error {error_id}: {str(error)}",
            exc_info=True
        )

        response = {
            'error': 'INTERNAL_SERVER_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': time.time(),
            'path': request.path,
            'method': request.method
        }

        # Log to database
        logger = DatabaseLogger(app.db_session)
        logger.log_request(
            level='CRITICAL',
            message=f"Unexpected error: {str(error)}",
            metadata={
                'error_id': error_id,
                'path': request.path,
                'method': request.method,
                'traceback': traceback.format_exc()
            }
        )

        return jsonify(response), 500
```

### Request Logging Middleware
```python
# app/middleware.py
import time
import uuid
from flask import request, g

class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        # Generate request ID
        request_id = str(uuid.uuid4())
        g.request_id = request_id

        # Record start time
        start_time = time.time()

        def new_start_response(status, response_headers, exc_info=None):
            # Calculate duration
            duration = time.time() - start_time

            # Log request
            self.log_request(
                request_id=request_id,
                method=request.method,
                path=request.path,
                status_code=status.split()[0],
                duration=duration,
                user_agent=request.headers.get('User-Agent'),
                ip_address=request.remote_addr
            )

            return start_response(status, response_headers, exc_info)

        return self.app(environ, new_start_response)

    def log_request(self, request_id, method, path, status_code, duration,
                   user_agent, ip_address):
        """Log HTTP request"""
        log_data = {
            'request_id': request_id,
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration': duration,
            'user_agent': user_agent,
            'ip_address': ip_address,
            'timestamp': time.time()
        }

        # Log to structured logger
        logger = structlog.get_logger()
        logger.info("HTTP Request", **log_data)

        # Log to database for important requests
        if int(status_code) >= 400 or duration > 5.0:  # Slow or error requests
            db_logger = DatabaseLogger(current_app.db_session)
            db_logger.log_request(
                level='WARNING' if int(status_code) < 500 else 'ERROR',
                message=f"HTTP {status_code} in {duration:.2f}s",
                metadata=log_data
            )
```

## Request Tracing

### Trace Context
```python
# utils/tracing.py
import uuid
from contextvars import ContextVar
from flask import g

class RequestTracer:
    REQUEST_ID: ContextVar[str] = ContextVar('request_id')

    @classmethod
    def get_request_id(cls) -> str:
        """Get current request ID"""
        try:
            return cls.REQUEST_ID.get()
        except LookupError:
            return str(uuid.uuid4())

    @classmethod
    def set_request_id(cls, request_id: str):
        """Set current request ID"""
        cls.REQUEST_ID.set(request_id)

    def __enter__(self):
        request_id = self.get_request_id()
        g.request_id = request_id
        return request_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Usage in views
@tts_bp.route('/generate', methods=['POST'])
def generate_audio():
    with RequestTracer() as request_id:
        logger.info("Starting TTS generation", request_id=request_id)
        # Process request
        return jsonify({'request_id': request_id})
```

### Distributed Tracing
```python
# utils/distributed_tracing.py
import time
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

class DistributedTracer:
    @classmethod
    def trace_tts_request(cls, request_id: str, user_id: int):
        """Trace TTS request processing"""
        with tracer.start_as_current_span(f"tts_request_{request_id}") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("service.name", "tts-api")

            try:
                # Simulate processing
                time.sleep(0.1)
                span.add_event("Request queued")

                # Call Gemini API
                with tracer.start_as_current_span("gemini_api_call") as api_span:
                    api_span.set_attribute("api.endpoint", "generateContent")
                    time.sleep(0.05)
                    api_span.add_event("API call completed")

                span.set_status(Status(StatusCode.OK))
                return True

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
```

## Performance Monitoring

### Metrics Collection
```python
# utils/metrics.py
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class MetricsCollector:
    _instance = None
    _lock = Lock()

    # Counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    auth_failures: int = 0

    # Histograms
    request_durations: list = field(default_factory=list)
    processing_times: list = field(default_factory=list)

    # Gauges
    active_connections: int = 0
    queue_length: int = 0

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def record_request(self, duration: float, success: bool = True):
        """Record request metrics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.request_durations.append(duration)

    def record_processing_time(self, processing_time: float):
        """Record TTS processing time"""
        self.processing_times.append(processing_time)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        total = self.total_requests
        if total == 0:
            return {'error_rate': 0, 'avg_duration': 0}

        return {
            'total_requests': total,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'error_rate': self.failed_requests / total,
            'avg_request_duration': sum(self.request_durations) / len(self.request_durations) if self.request_durations else 0,
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'queue_length': self.queue_length,
            'active_connections': self.active_connections
        }
```

### Health Checks
```python
# utils/health_checks.py
import time
from typing import Dict, Any
from datetime import datetime

class HealthChecker:
    def __init__(self, db_session, redis_client):
        self.db_session = db_session
        self.redis_client = redis_client
        self.last_check = None
        self.check_results = {}

    def check_database(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            start_time = time.time()
            self.db_session.execute("SELECT 1")
            response_time = time.time() - start_time

            return {
                'status': 'healthy',
                'response_time': response_time,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def check_redis(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            start_time = time.time()
            self.redis_client.ping()
            response_time = time.time() - start_time

            return {
                'status': 'healthy',
                'response_time': response_time,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def check_gemini_api(self) -> Dict[str, Any]:
        """Check Gemini API health"""
        try:
            # Simple API call to check health
            # Implementation depends on Gemini API client
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        checks = {
            'database': self.check_database(),
            'redis': self.check_redis(),
            'gemini_api': self.check_gemini_api()
        }

        overall_status = 'healthy'
        for service, result in checks.items():
            if result['status'] != 'healthy':
                overall_status = 'unhealthy'
                break

        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'services': checks
        }
```

## Alerting System

### Alert Manager
```python
# utils/alerting.py
import requests
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class AlertRule:
    name: str
    condition: str  # e.g., "error_rate > 0.05"
    threshold: float
    severity: str  # info, warning, error, critical
    channels: List[str]  # email, slack, webhook

class AlertManager:
    def __init__(self):
        self.rules = []
        self.alert_history = []

    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules.append(rule)

    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules"""
        alerts = []

        for rule in self.rules:
            if self.evaluate_condition(rule.condition, metrics):
                alert = {
                    'rule': rule.name,
                    'severity': rule.severity,
                    'message': f"Alert: {rule.name}",
                    'metrics': metrics,
                    'timestamp': time.time()
                }
                alerts.append(alert)

                # Send alert
                self.send_alert(alert, rule.channels)

        return alerts

    def evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        # Simple condition evaluation
        # In production, use a more robust expression evaluator
        if 'error_rate' in condition:
            threshold = float(condition.split('>')[1].strip())
            return metrics.get('error_rate', 0) > threshold

        return False

    def send_alert(self, alert: Dict[str, Any], channels: List[str]):
        """Send alert to specified channels"""
        for channel in channels:
            if channel == 'email':
                self.send_email_alert(alert)
            elif channel == 'slack':
                self.send_slack_alert(alert)
            elif channel == 'webhook':
                self.send_webhook_alert(alert)

    def send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert"""
        # Implementation using SMTP or email service
        pass

    def send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if webhook_url:
            requests.post(webhook_url, json={
                'text': f"🚨 {alert['severity'].upper()}: {alert['message']}"
            })

    def send_webhook_alert(self, alert: Dict[str, Any]):
        """Send webhook alert"""
        webhook_url = os.getenv('ALERT_WEBHOOK_URL')
        if webhook_url:
            requests.post(webhook_url, json=alert)
```

### Alert Rules Configuration
```python
# Configuration
alert_manager = AlertManager()

# Add alert rules
alert_manager.add_rule(AlertRule(
    name="High Error Rate",
    condition="error_rate > 0.05",
    threshold=0.05,
    severity="warning",
    channels=["email", "slack"]
))

alert_manager.add_rule(AlertRule(
    name="Critical Error Rate",
    condition="error_rate > 0.10",
    threshold=0.10,
    severity="critical",
    channels=["email", "slack"]
))

alert_manager.add_rule(AlertRule(
    name="Queue Too Long",
    condition="queue_length > 100",
    threshold=100,
    severity="warning",
    channels=["slack"]
))
```

## Log Aggregation

### ELK Stack Integration
```python
# utils/elk_integration.py
import json
from datetime import datetime

class ELKLogger:
    def __init__(self, elasticsearch_url: str, index_prefix: str = "tts-api"):
        self.elasticsearch_url = elasticsearch_url
        self.index_prefix = index_prefix

    def log_to_elasticsearch(self, log_entry: Dict[str, Any]):
        """Send log to Elasticsearch"""
        index_name = f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"

        log_data = {
            '@timestamp': datetime.utcnow().isoformat(),
            'level': log_entry.get('level', 'INFO'),
            'message': log_entry.get('message', ''),
            'request_id': log_entry.get('request_id'),
            'user_id': log_entry.get('user_id'),
            'source': log_entry.get('source', 'api'),
            'metadata': log_entry.get('metadata', {}),
            'host': os.getenv('HOSTNAME', 'unknown')
        }

        # Send to Elasticsearch
        try:
            response = requests.post(
                f"{self.elasticsearch_url}/{index_name}/_doc",
                json=log_data,
                headers={'Content-Type': 'application/json'}
            )
            return response.status_code == 201
        except Exception as e:
            print(f"Failed to send log to Elasticsearch: {e}")
            return False
```

### Log Rotation
```python
# utils/log_rotation.py
import os
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path

class LogRotator:
    def __init__(self, log_dir: str, max_files: int = 30):
        self.log_dir = Path(log_dir)
        self.max_files = max_files

    def rotate_logs(self):
        """Rotate log files"""
        if not self.log_dir.exists():
            return

        # Compress old log files
        for log_file in self.log_dir.glob("*.log"):
            if self._should_compress(log_file):
                self._compress_file(log_file)

        # Delete old compressed files
        self._cleanup_old_files()

    def _should_compress(self, log_file: Path) -> bool:
        """Check if file should be compressed"""
        if log_file.suffix == '.gz':
            return False

        # Compress files older than 1 day
        return datetime.fromtimestamp(log_file.stat().st_mtime) < datetime.now() - timedelta(days=1)

    def _compress_file(self, log_file: Path):
        """Compress log file"""
        compressed_file = log_file.with_suffix(log_file.suffix + '.gz')

        with open(log_file, 'rb') as f_in:
            with gzip.open(compressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove original file
        log_file.unlink()

    def _cleanup_old_files(self):
        """Clean up old compressed files"""
        compressed_files = sorted(
            self.log_dir.glob("*.log.gz"),
            key=lambda x: x.stat().st_mtime
        )

        # Keep only max_files most recent
        for old_file in compressed_files[:-self.max_files]:
            old_file.unlink()
```

## Testing Error Handling

### Unit Tests
```python
# tests/test_error_handling.py
import pytest
from unittest.mock import patch, MagicMock

from utils.exceptions import TTSAPIError, ValidationError, RateLimitError
from app.error_handlers import handle_tts_api_error

def test_tts_api_error_handling():
    """Test TTS API error handling"""
    error = ValidationError("Invalid input", {"field": "text"})

    with patch('app.error_handlers.DatabaseLogger') as mock_logger:
        response, status_code = handle_tts_api_error(error)

        assert status_code == 400
        response_data = json.loads(response.data)
        assert response_data['error'] == 'VALIDATION_ERROR'
        assert response_data['message'] == 'Invalid input'
        assert 'details' in response_data

def test_rate_limit_error_handling():
    """Test rate limit error handling"""
    error = RateLimitError("Too many requests", 60)

    response, status_code = handle_tts_api_error(error)

    assert status_code == 429
    response_data = json.loads(response.data)
    assert response_data['details']['retry_after'] == 60
```

### Integration Tests
```python
# tests/test_logging_integration.py
def test_request_logging_middleware(client):
    """Test request logging middleware"""
    with patch('app.middleware.DatabaseLogger') as mock_logger:
        response = client.post('/api/v1/tts/generate', json={'text': 'test'})

        # Check that logging was called
        mock_logger.return_value.log_request.assert_called_once()
        call_args = mock_logger.return_value.log_request.call_args

        # Verify log data
        assert call_args[1]['level'] == 'WARNING'  # Should be warning due to auth error
        assert 'path' in call_args[1]['metadata']
        assert 'method' in call_args[1]['metadata']
```

## Configuration

### Logging Configuration
```yaml
# logging_config.yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'
    class: pythonjsonlogger.jsonlogger.JsonFormatter

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: json
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: logs/tts_api.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  database:
    class: app.handlers.DatabaseHandler
    level: WARNING

root:
  level: INFO
  handlers: [console, file, database]

loggers:
  app:
    level: DEBUG
    handlers: [console, file, database]
    propagate: false

  celery:
    level: INFO
    handlers: [console, file]
    propagate: false
```

### Environment Variables
```bash
# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=/app/logs/tts_api.log

# Error handling
ERROR_INCLUDE_TRACEBACK=false
ERROR_WEBHOOK_URL=https://hooks.slack.com/services/...

# Monitoring
HEALTH_CHECK_INTERVAL=30
METRICS_COLLECTION_INTERVAL=60
ALERT_CHECK_INTERVAL=30
```

## Best Practices

### 1. Structured Logging
- Use JSON format for all logs
- Include request ID in all log entries
- Add relevant context (user_id, request_id, etc.)
- Use appropriate log levels

### 2. Error Handling
- Never expose internal error details to clients
- Use custom exception classes
- Implement proper retry logic
- Log errors with full context

### 3. Monitoring
- Implement health checks for all services
- Collect relevant metrics
- Set up alerting for critical issues
- Monitor performance metrics

### 4. Security
- Sanitize sensitive data in logs
- Use secure communication for log aggregation
- Implement log access controls
- Regular log rotation and cleanup

### 5. Performance
- Use async logging for high-throughput scenarios
- Implement log sampling for debug logs
- Use connection pooling for database logging
- Monitor logging performance impact