# Retry Logic với Exponential Backoff

## Tổng quan
Hệ thống retry logic được thiết kế để handle các transient failures một cách thông minh, sử dụng exponential backoff để tránh overload hệ thống.

## Retry Strategies

### 1. Exponential Backoff
```python
# utils/retry/strategies.py
import random
import time
from typing import Callable, Any, Tuple, Optional
from functools import wraps

class RetryStrategy:
    """Base retry strategy"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)

        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if retry should be attempted"""
        return attempt < self.max_retries

def exponential_backoff_retry(max_retries: int = 3, base_delay: float = 1.0,
                             max_delay: float = 60.0, backoff_factor: float = 2.0,
                             jitter: bool = True):
    """Decorator for exponential backoff retry"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            strategy = RetryStrategy(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter
            )

            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not strategy.should_retry(attempt, e):
                        raise e

                    delay = strategy.calculate_delay(attempt)
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator
```

### 2. Circuit Breaker Pattern
```python
# utils/retry/circuit_breaker.py
import time
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: tuple = (Exception,)
    success_threshold: int = 3

class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = Lock()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == CircuitState.OPEN:
                    if self._should_attempt_reset():
                        self.state = CircuitState.HALF_OPEN
                        self.success_count = 0
                    else:
                        raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)

                with self.lock:
                    if self.state == CircuitState.HALF_OPEN:
                        self.success_count += 1
                        if self.success_count >= self.config.success_threshold:
                            self._reset()

                return result

            except self.config.expected_exception as e:
                with self.lock:
                    self._record_failure()
                raise e

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def _record_failure(self):
        """Record failure and update state"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

    def _reset(self):
        """Reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
```

### 3. Retry with Dead Letter Queue
```python
# utils/retry/dead_letter_queue.py
import json
from typing import Dict, Any, Callable
from datetime import datetime
from redis import Redis

class DeadLetterQueue:
    """Dead letter queue for permanently failed requests"""

    def __init__(self, redis_client: Redis, queue_name: str = 'dead_letter_queue'):
        self.redis = redis_client
        self.queue_name = queue_name

    def add_to_dlq(self, request_data: Dict[str, Any], error: Exception,
                   retry_count: int, max_retries: int):
        """Add failed request to dead letter queue"""
        dlq_item = {
            'request_id': request_data.get('id'),
            'user_id': request_data.get('user_id'),
            'text': request_data.get('text', '')[:500],  # Truncate text
            'voice_name': request_data.get('voice_name'),
            'error_message': str(error),
            'retry_count': retry_count,
            'max_retries': max_retries,
            'timestamp': datetime.utcnow().isoformat(),
            'headers': {
                'user_agent': request_data.get('user_agent', ''),
                'ip_address': request_data.get('ip_address', '')
            }
        }

        # Add to Redis list
        self.redis.lpush(self.queue_name, json.dumps(dlq_item))

        # Log to database
        self._log_to_database(dlq_item)

    def get_dlq_items(self, limit: int = 100) -> list:
        """Get items from dead letter queue"""
        items = self.redis.lrange(self.queue_name, 0, limit - 1)
        return [json.loads(item) for item in items]

    def requeue_item(self, index: int) -> bool:
        """Requeue item from dead letter queue"""
        item = self.redis.lindex(self.queue_name, index)
        if item:
            # Move to retry queue
            self.redis.lpush('retry_queue', item)
            self.redis.lrem(self.queue_name, 1, item)
            return True
        return False

    def cleanup_old_items(self, days_old: int = 30):
        """Clean up old items from dead letter queue"""
        # Implementation for cleanup
        pass

    def _log_to_database(self, dlq_item: Dict[str, Any]):
        """Log to database for analysis"""
        # Implementation for database logging
        pass
```

## Application Integration

### TTS Processing with Retry
```python
# utils/retry/tts_retry.py
from utils.retry.strategies import exponential_backoff_retry
from utils.retry.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from utils.retry.dead_letter_queue import DeadLetterQueue
from redis import Redis

class TTSRetryManager:
    """Manage retry logic for TTS processing"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.dlq = DeadLetterQueue(redis_client)

        # Circuit breaker for Gemini API
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=(Exception,),
            success_threshold=3
        ))

    @exponential_backoff_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def process_with_retry(self, request_id: int, user_id: int):
        """Process TTS request with retry logic"""
        try:
            # Apply circuit breaker
            result = self.circuit_breaker(self._process_tts_request)(request_id, user_id)
            return result

        except Exception as e:
            # Add to dead letter queue if max retries exceeded
            request_data = self._get_request_data(request_id)
            self.dlq.add_to_dlq(
                request_data=request_data,
                error=e,
                retry_count=3,
                max_retries=3
            )
            raise e

    def _process_tts_request(self, request_id: int, user_id: int):
        """Actual TTS processing logic"""
        # Implementation of TTS processing
        pass

    def _get_request_data(self, request_id: int) -> Dict[str, Any]:
        """Get request data for dead letter queue"""
        # Implementation to get request data
        pass
```

### API Error Handling with Retry
```python
# app/error_handlers.py
from utils.retry.strategies import exponential_backoff_retry
from utils.retry.circuit_breaker import CircuitBreaker

class RetryableError(Exception):
    """Exception that should trigger retry"""
    pass

# Circuit breaker for external API calls
gemini_api_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=(ConnectionError, TimeoutError, RetryableError),
    success_threshold=3
))

@exponential_backoff_retry(max_retries=3, base_delay=0.5, max_delay=10.0)
def call_gemini_api_with_retry(text: str, voice_name: str):
    """Call Gemini API with retry logic"""
    try:
        return gemini_api_breaker(_call_gemini_api)(text, voice_name)
    except Exception as e:
        # Log error and raise
        logger.error(f"Gemini API call failed after retries: {e}")
        raise e

def _call_gemini_api(text: str, voice_name: str):
    """Actual Gemini API call"""
    # Implementation
    pass
```

## Database Retry Logic

### SQLAlchemy Retry
```python
# utils/retry/database_retry.py
from sqlalchemy.exc import DisconnectionError, OperationalError
from utils.retry.strategies import exponential_backoff_retry

class DatabaseRetryManager:
    """Manage database operations with retry"""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    @exponential_backoff_retry(
        max_retries=3,
        base_delay=0.1,
        max_delay=2.0,
        backoff_factor=2.0
    )
    def execute_with_retry(self, operation: Callable):
        """Execute database operation with retry"""
        try:
            return operation()
        except (DisconnectionError, OperationalError) as e:
            # Rollback and retry
            if hasattr(operation, '__self__'):
                operation.__self__.rollback()
            raise e

    def get_session_with_retry(self):
        """Get database session with retry"""
        return self.execute_with_retry(lambda: self.session_factory())

# Usage
db_retry = DatabaseRetryManager(session_factory)

def update_request_status(request_id: int, status: str):
    """Update request status with retry"""
    def operation():
        session = db_retry.get_session_with_retry()
        request = session.query(AudioRequest).get(request_id)
        request.status = status
        session.commit()
        return request

    return db_retry.execute_with_retry(operation)
```

### Redis Retry
```python
# utils/retry/redis_retry.py
from redis.exceptions import ConnectionError, TimeoutError
from utils.retry.strategies import exponential_backoff_retry

class RedisRetryManager:
    """Manage Redis operations with retry"""

    def __init__(self, redis_client):
        self.redis = redis_client

    @exponential_backoff_retry(
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0,
        backoff_factor=2.0
    )
    def execute_with_retry(self, operation: Callable, *args, **kwargs):
        """Execute Redis operation with retry"""
        try:
            return operation(*args, **kwargs)
        except (ConnectionError, TimeoutError) as e:
            raise e

# Usage
redis_retry = RedisRetryManager(redis_client)

def enqueue_request(request_data: dict):
    """Enqueue request with retry"""
    return redis_retry.execute_with_retry(
        lambda: redis_client.lpush('tts_queue', json.dumps(request_data))
    )
```

## Worker Retry Logic

### Celery Task Retry
```python
# app/tasks/retry_tasks.py
from celery import Task
from utils.retry.strategies import exponential_backoff_retry

class RetryableTask(Task):
    """Base task class with retry logic"""

    def __init__(self):
        super().__init__()
        self.max_retries = 3
        self.base_delay = 60  # 1 minute
        self.max_delay = 3600  # 1 hour

    def apply_async_with_retry(self, args=None, kwargs=None, **options):
        """Apply task with retry configuration"""
        options.setdefault('retry', True)
        options.setdefault('retry_policy', {
            'max_retries': self.max_retries,
            'interval_start': self.base_delay,
            'interval_step': self.base_delay * 2,
            'interval_max': self.max_delay,
        })
        return self.apply_async(args, kwargs, **options)

@celery_app.task(
    base=RetryableTask,
    bind=True,
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=3600,
    autoretry_for=(Exception,),
    retry_jitter=True
)
def robust_tts_task(self, request_id: int):
    """Robust TTS task with comprehensive retry logic"""
    try:
        # Task implementation
        return process_tts_request(request_id)
    except Exception as e:
        # Log error with context
        logger.error(
            f"TTS task failed",
            request_id=request_id,
            retry_count=self.request.retries,
            error=str(e),
            exc_info=True
        )

        # Retry with exponential backoff
        raise self.retry(countdown=60 * (2 ** self.request.retries))
```

### AsyncIO Retry
```python
# utils/retry/async_retry.py
import asyncio
import random
from typing import Callable, Any, TypeVar, Tuple
from functools import wraps

T = TypeVar('T')

async def async_exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """Async retry decorator with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                        if jitter:
                            jitter_amount = delay * 0.1
                            delay += random.uniform(-jitter_amount, jitter_amount)

                        await asyncio.sleep(delay)
                    else:
                        raise last_exception

            raise last_exception
        return wrapper
    return decorator

# Usage
@async_exponential_backoff_retry(max_retries=3, base_delay=0.5, max_delay=10.0)
async def async_api_call(text: str, voice_name: str):
    """Async API call with retry"""
    # Implementation
    pass
```

## Configuration

### Retry Configuration
```python
# config/retry_config.py
class RetryConfig:
    """Retry configuration for different services"""

    # TTS Processing
    TTS_MAX_RETRIES = 3
    TTS_BASE_DELAY = 1.0
    TTS_MAX_DELAY = 30.0
    TTS_BACKOFF_FACTOR = 2.0

    # Gemini API
    GEMINI_API_MAX_RETRIES = 5
    GEMINI_API_BASE_DELAY = 0.5
    GEMINI_API_MAX_DELAY = 10.0
    GEMINI_API_BACKOFF_FACTOR = 2.0

    # Database
    DB_MAX_RETRIES = 3
    DB_BASE_DELAY = 0.1
    DB_MAX_DELAY = 2.0
    DB_BACKOFF_FACTOR = 2.0

    # Redis
    REDIS_MAX_RETRIES = 3
    REDIS_BASE_DELAY = 0.1
    REDIS_MAX_DELAY = 1.0
    REDIS_BACKOFF_FACTOR = 2.0

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3

    # Dead Letter Queue
    DLQ_MAX_SIZE = 1000
    DLQ_RETENTION_DAYS = 30
```

### Environment Variables
```bash
# Retry settings
TTS_MAX_RETRIES=3
TTS_BASE_DELAY=1.0
GEMINI_API_MAX_RETRIES=5
DB_MAX_RETRIES=3
CIRCUIT_BREAKER_ENABLED=true
DEAD_LETTER_QUEUE_ENABLED=true
```

## Monitoring và Metrics

### Retry Metrics
```python
# utils/retry/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Retry metrics
RETRY_ATTEMPTS_TOTAL = Counter(
    'retry_attempts_total',
    'Total retry attempts',
    ['service', 'status']
)

RETRY_DELAY_SECONDS = Histogram(
    'retry_delay_seconds',
    'Retry delay in seconds',
    ['service'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state',
    ['service', 'state']
)

DEAD_LETTER_QUEUE_SIZE = Gauge(
    'dead_letter_queue_size',
    'Dead letter queue size',
    ['queue_name']
)

def record_retry_attempt(service: str, success: bool):
    """Record retry attempt"""
    status = 'success' if success else 'failed'
    RETRY_ATTEMPTS_TOTAL.labels(service=service, status=status).inc()

def record_retry_delay(service: str, delay: float):
    """Record retry delay"""
    RETRY_DELAY_SECONDS.labels(service=service).observe(delay)

def update_circuit_breaker_state(service: str, state: str):
    """Update circuit breaker state"""
    state_value = 1 if state == 'closed' else 0
    CIRCUIT_BREAKER_STATE.labels(service=service, state=state).set(state_value)

def update_dlq_size(queue_name: str, size: int):
    """Update dead letter queue size"""
    DEAD_LETTER_QUEUE_SIZE.labels(queue_name=queue_name).set(size)
```

### Retry Dashboard
```json
{
  "dashboard": {
    "title": "Retry Logic Dashboard",
    "panels": [
      {
        "title": "Retry Success Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(retry_attempts_total{status=\"success\"}[5m]) / rate(retry_attempts_total[5m])",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Circuit Breaker Status",
        "type": "stat",
        "targets": [
          {
            "expr": "circuit_breaker_state",
            "legendFormat": "{{service}} - {{state}}"
          }
        ]
      },
      {
        "title": "Dead Letter Queue Size",
        "type": "graph",
        "targets": [
          {
            "expr": "dead_letter_queue_size",
            "legendFormat": "{{queue_name}}"
          }
        ]
      }
    ]
  }
}
```

## Testing

### Unit Tests
```python
# tests/test_retry.py
import pytest
from unittest.mock import patch, MagicMock
from utils.retry.strategies import exponential_backoff_retry

def test_exponential_backoff_retry_success():
    """Test successful retry"""
    call_count = 0

    @exponential_backoff_retry(max_retries=3)
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 3

def test_exponential_backoff_retry_exhaustion():
    """Test retry exhaustion"""
    @exponential_backoff_retry(max_retries=2)
    def always_failing_function():
        raise ConnectionError("Persistent failure")

    with pytest.raises(ConnectionError):
        always_failing_function()

def test_circuit_breaker():
    """Test circuit breaker"""
    from utils.retry.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
    breaker = CircuitBreaker(config)

    @breaker
    def failing_function():
        raise Exception("Always fails")

    # First call - should fail
    with pytest.raises(Exception):
        failing_function()

    # Second call - should fail
    with pytest.raises(Exception):
        failing_function()

    # Third call - should be blocked by circuit breaker
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        failing_function()
```

### Integration Tests
```python
# tests/test_retry_integration.py
def test_tts_retry_integration():
    """Test TTS retry integration"""
    from utils.retry.tts_retry import TTSRetryManager

    retry_manager = TTSRetryManager(redis_client)

    # Mock TTS processing to fail twice then succeed
    with patch('utils.retry.tts_retry.TTSRetryManager._process_tts_request') as mock_process:
        mock_process.side_effect = [Exception("Fail 1"), Exception("Fail 2"), "Success"]

        result = retry_manager.process_with_retry(1, 1)

        assert result == "Success"
        assert mock_process.call_count == 3

def test_database_retry():
    """Test database retry"""
    from utils.retry.database_retry import DatabaseRetryManager

    db_retry = DatabaseRetryManager(session_factory)

    with patch.object(db_retry, 'execute_with_retry') as mock_retry:
        mock_retry.side_effect = [OperationalError("Connection lost", None, None), "Success"]

        result = db_retry.execute_with_retry(lambda: "Success")

        assert result == "Success"
        assert mock_retry.call_count == 2
```

## Best Practices

### 1. Retry Design
- Only retry on transient failures
- Use exponential backoff with jitter
- Implement circuit breaker for external services
- Set appropriate retry limits

### 2. Error Classification
- Distinguish between retryable and non-retryable errors
- Use dead letter queues for permanent failures
- Log retry attempts with context

### 3. Resource Management
- Implement timeouts for all operations
- Use connection pooling
- Monitor retry overhead
- Clean up resources on failure

### 4. Monitoring
- Track retry success rates
- Monitor circuit breaker states
- Alert on high retry rates
- Measure retry delays

### 5. Configuration
- Make retry parameters configurable
- Use different settings for different environments
- Document retry behavior
- Test retry logic thoroughly

## Deployment Considerations

### Production Configuration
```python
# Production retry settings
PRODUCTION_RETRY_CONFIG = {
    'tts_processing': {
        'max_retries': 5,
        'base_delay': 2.0,
        'max_delay': 60.0,
        'circuit_breaker': {
            'failure_threshold': 10,
            'recovery_timeout': 120
        }
    },
    'gemini_api': {
        'max_retries': 3,
        'base_delay': 1.0,
        'max_delay': 30.0
    }
}
```

### Load Testing
```python
# Load testing retry logic
def test_retry_under_load():
    """Test retry logic under high load"""
    import asyncio
    import time

    async def load_test():
        tasks = []
        for i in range(100):
            task = async_api_call_with_retry(f"test_{i}")
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        print(f"Success rate: {success_count}/100")
        print(f"Total time: {end_time - start_time:.2f}s")

    asyncio.run(load_test())
```

## Summary

Retry logic implementation provides:

1. **Resilience**: Handle transient failures gracefully
2. **Performance**: Prevent cascade failures with circuit breakers
3. **Observability**: Comprehensive monitoring and metrics
4. **Maintainability**: Clean separation of retry logic
5. **Scalability**: Handle high load with appropriate backoff

Key components:
- Exponential backoff strategies
- Circuit breaker pattern
- Dead letter queues
- Comprehensive monitoring
- Integration with all system components