# Background Worker System

## Tổng quan
Hệ thống background workers xử lý các task TTS một cách asynchronous, đảm bảo API responsive và có thể scale horizontally.

## Celery Implementation

### Configuration
```python
# config/development.py
CELERY_BROKER_URL = 'redis://localhost:6379/2'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/3'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

# Worker settings
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000
CELERY_WORKER_DISABLE_RATE_LIMITS = False

# Task routing
CELERY_TASK_ROUTES = {
    'app.tasks.tts_tasks.*': {'queue': 'tts_queue'},
    'app.tasks.cleanup_tasks.*': {'queue': 'cleanup_queue'},
}
```

### Celery Application
```python
# app/celery_app.py
from celery import Celery
from config import DevelopmentConfig, ProductionConfig, TestingConfig
import os

def create_celery_app():
    """Create Celery application"""
    config_name = os.getenv('FLASK_ENV', 'development')
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
    }

    config = config_map[config_name]

    celery = Celery(__name__)
    celery.config_from_object(config, namespace='CELERY')

    # Update configuration with Flask app settings
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery_app = create_celery_app()
```

### TTS Processing Tasks
```python
# app/tasks/tts_tasks.py
import asyncio
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

from app.celery_app import celery_app
from app.extensions import db
from models import AudioRequest, AudioFile
from utils.audio_processor import AudioProcessor
from utils.security import SecurityUtils

@celery_app.task(
    name='app.tasks.tts_tasks.process_tts_request',
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def process_tts_request(self, request_id: int):
    """Process TTS request asynchronously"""
    try:
        # Get request from database
        audio_request = db.session.query(AudioRequest).get(request_id)
        if not audio_request:
            self.retry(countdown=60)
            return

        if audio_request.status != 'pending':
            return

        # Mark as processing
        audio_request.mark_as_processing()
        db.session.commit()

        # Initialize audio processor
        audio_processor = AudioProcessor(api_key=os.getenv('GEMINI_API_KEY'))

        # Generate audio
        audio_data, mime_type = asyncio.run(audio_processor.generate_audio(
            text=audio_request.text_content,
            voice_name=audio_request.voice_name,
            output_format=audio_request.output_format
        ))

        # Calculate file hash
        file_hash = SecurityUtils.calculate_audio_hash(audio_data)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{audio_request.output_format}"
        ) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Create audio file record
        filename = f"tts_{request_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{audio_request.output_format}"
        file_size = len(audio_data)

        audio_file = AudioFile(
            request_id=request_id,
            file_path=temp_file_path,
            filename=filename,
            mime_type=mime_type,
            file_size=file_size,
            checksum=file_hash
        )

        # Save to database
        db.session.add(audio_file)

        # Update request status
        processing_time = 0  # Calculate actual time in production
        audio_request.mark_as_completed(processing_time)

        db.session.commit()

        # Clean up temporary file
        os.unlink(temp_file_path)

        return {
            'request_id': request_id,
            'status': 'completed',
            'file_size': file_size
        }

    except Exception as e:
        # Handle errors
        error_message = str(e)
        audio_request.mark_as_failed(error_message)
        db.session.commit()

        # Retry logic
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=self._get_retry_delay(self.request.retries))

        return {
            'request_id': request_id,
            'status': 'failed',
            'error': error_message
        }

    def _get_retry_delay(self, retry_count: int) -> int:
        """Calculate retry delay with exponential backoff"""
        return min(60 * (2 ** retry_count), 3600)  # Max 1 hour

@celery_app.task(name='app.tasks.tts_tasks.batch_process_requests')
def batch_process_requests(request_ids: list):
    """Process multiple TTS requests in batch"""
    results = []

    for request_id in request_ids:
        result = process_tts_request.delay(request_id)
        results.append(result.id)

    return {
        'task_ids': results,
        'count': len(request_ids)
    }
```

### Queue Management Tasks
```python
# app/tasks/queue_tasks.py
from app.celery_app import celery_app
from app.extensions import db
from models import AudioRequest
from redis import Redis

redis_client = Redis.from_url(os.getenv('REDIS_URL'))

@celery_app.task(name='app.tasks.queue_tasks.enqueue_tts_request')
def enqueue_tts_request(request_data: Dict[str, Any]):
    """Enqueue TTS request to Redis"""
    queue_manager = TTSQueueManager(redis_client)

    # Add priority based on user type
    priority = 1 if request_data.get('is_premium', False) else 0
    queue_manager.enqueue_request(request_data, priority)

    return {'request_id': request_data['id'], 'queued': True}

@celery_app.task(name='app.tasks.queue_tasks.process_queue')
def process_queue(max_requests: int = 10):
    """Process requests from queue"""
    queue_manager = TTSQueueManager(redis_client)
    processed_count = 0

    while processed_count < max_requests:
        request_data = queue_manager.dequeue_request()
        if not request_data:
            break

        # Submit to TTS processing
        process_tts_request.delay(request_data['request_id'])
        processed_count += 1

    return {'processed': processed_count}

@celery_app.task(name='app.tasks.queue_tasks.retry_failed_requests')
def retry_failed_requests():
    """Retry failed requests"""
    retry_manager = RetryManager(redis_client)
    retry_items = retry_manager.get_retry_items()

    retried_count = 0
    for item in retry_items:
        if retried_count >= 5:  # Limit batch size
            break

        process_tts_request.delay(item['request_id'])
        retried_count += 1

    return {'retried': retried_count}
```

### Cleanup Tasks
```python
# app/tasks/cleanup_tasks.py
import os
from datetime import datetime, timedelta

from app.celery_app import celery_app
from app.extensions import db
from models import AudioRequest, AudioFile

@celery_app.task(name='app.tasks.cleanup_tasks.cleanup_old_files')
def cleanup_old_files(days_old: int = 30):
    """Clean up old audio files"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)

    # Get old completed requests
    old_requests = db.session.query(AudioRequest).filter(
        AudioRequest.status == 'completed',
        AudioRequest.updated_at < cutoff_date
    ).all()

    deleted_files = 0
    freed_space = 0

    for request in old_requests:
        for audio_file in request.audio_files:
            # Delete physical file
            if os.path.exists(audio_file.file_path):
                file_size = os.path.getsize(audio_file.file_path)
                os.remove(audio_file.file_path)
                freed_space += file_size
                deleted_files += 1

        # Delete from database
        db.session.delete(request)

    db.session.commit()

    return {
        'deleted_files': deleted_files,
        'freed_space': freed_space,
        'deleted_requests': len(old_requests)
    }

@celery_app.task(name='app.tasks.cleanup_tasks.cleanup_failed_requests')
def cleanup_failed_requests(days_old: int = 7):
    """Clean up old failed requests"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)

    old_requests = db.session.query(AudioRequest).filter(
        AudioRequest.status == 'failed',
        AudioRequest.updated_at < cutoff_date
    ).all()

    for request in old_requests:
        # Delete associated files
        for audio_file in request.audio_files:
            if os.path.exists(audio_file.file_path):
                os.remove(audio_file.file_path)

        db.session.delete(request)

    db.session.commit()

    return {'deleted_requests': len(old_requests)}
```

## AsyncIO Implementation (Alternative)

### Async Worker
```python
# app/workers/async_worker.py
import asyncio
import os
from datetime import datetime
from typing import Dict, Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from models import AudioRequest, AudioFile
from utils.audio_processor import AudioProcessor

class AsyncTTSWorker:
    def __init__(self):
        self.redis = Redis.from_url(os.getenv('REDIS_URL'))
        self.engine = create_async_engine(os.getenv('DATABASE_URL'))
        self.audio_processor = AudioProcessor(os.getenv('GEMINI_API_KEY'))

    async def start(self):
        """Start worker"""
        while True:
            try:
                # Get request from queue
                request_data = await self._dequeue_request()
                if not request_data:
                    await asyncio.sleep(1)
                    continue

                # Process request
                await self._process_request(request_data)

            except Exception as e:
                print(f"Worker error: {e}")
                await asyncio.sleep(5)

    async def _dequeue_request(self) -> Dict[str, Any]:
        """Dequeue request from Redis"""
        # Implementation
        pass

    async def _process_request(self, request_data: Dict[str, Any]):
        """Process TTS request"""
        async with AsyncSession(self.engine) as session:
            # Get request
            result = await session.execute(
                "SELECT * FROM audio_requests WHERE id = :request_id",
                {"request_id": request_data['request_id']}
            )
            audio_request = result.first()

            if not audio_request:
                return

            try:
                # Generate audio
                audio_data, mime_type = await self.audio_processor.generate_audio(
                    text=audio_request.text_content,
                    voice_name=audio_request.voice_name,
                    output_format=audio_request.output_format
                )

                # Save audio file
                await self._save_audio_file(session, audio_request.id, audio_data, mime_type)

                # Update request status
                await session.execute(
                    "UPDATE audio_requests SET status = 'completed', updated_at = :now WHERE id = :request_id",
                    {"now": datetime.utcnow(), "request_id": audio_request.id}
                )

                await session.commit()

            except Exception as e:
                # Update request status to failed
                await session.execute(
                    "UPDATE audio_requests SET status = 'failed', error_message = :error, updated_at = :now WHERE id = :request_id",
                    {"error": str(e), "now": datetime.utcnow(), "request_id": audio_request.id}
                )
                await session.commit()

    async def _save_audio_file(self, session: AsyncSession, request_id: int, audio_data: bytes, mime_type: str):
        """Save audio file to storage"""
        # Implementation
        pass
```

## Worker Management

### Starting Workers
```bash
# Celery workers
celery -A app.celery_app worker --loglevel=info --concurrency=4

# Beat scheduler for periodic tasks
celery -A app.celery_app beat --loglevel=info

# Flower monitoring
celery -A app.celery_app flower
```

### Worker Configuration
```python
# celeryconfig.py
from celery.schedules import crontab

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000
worker_disable_rate_limits = False

# Task routing
task_routes = {
    'app.tasks.tts_tasks.*': {'queue': 'tts_queue'},
    'app.tasks.cleanup_tasks.*': {'queue': 'cleanup_queue'},
}

# Periodic tasks
beat_schedule = {
    'process-queue-every-30-seconds': {
        'task': 'app.tasks.queue_tasks.process_queue',
        'schedule': 30.0,
    },
    'retry-failed-requests-every-5-minutes': {
        'task': 'app.tasks.queue_tasks.retry_failed_requests',
        'schedule': 300.0,
    },
    'cleanup-old-files-daily': {
        'task': 'app.tasks.cleanup_tasks.cleanup_old_files',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
}
```

## Monitoring & Health Checks

### Worker Health Check
```python
# app/tasks/health_tasks.py
from app.celery_app import celery_app
import redis

@celery_app.task(name='app.tasks.health_tasks.worker_health_check')
def worker_health_check():
    """Worker health check"""
    redis_client = redis.from_url(os.getenv('REDIS_URL'))

    # Update worker heartbeat
    worker_id = os.getenv('WORKER_ID', 'worker-1')
    redis_client.hset(f'worker_status:{worker_id}', mapping={
        'last_seen': datetime.utcnow().isoformat(),
        'status': 'healthy'
    })

    return {'worker_id': worker_id, 'status': 'healthy'}

@celery_app.task(name='app.tasks.health_tasks.get_queue_status')
def get_queue_status():
    """Get queue status"""
    redis_client = redis.from_url(os.getenv('REDIS_URL'))

    return {
        'tts_queue_length': redis_client.llen('tts_queue'),
        'processing_requests': redis_client.scard('processing_requests'),
        'failed_requests': redis_client.llen('tts_retry_queue'),
        'timestamp': datetime.utcnow().isoformat()
    }
```

### Metrics Collection
```python
# app/tasks/metrics_tasks.py
from app.celery_app import celery_app
from app.extensions import db
from models import AudioRequest
from datetime import datetime, timedelta

@celery_app.task(name='app.tasks.metrics_tasks.collect_metrics')
def collect_metrics():
    """Collect system metrics"""
    # Request metrics
    total_requests = db.session.query(AudioRequest).count()
    completed_requests = db.session.query(AudioRequest).filter(
        AudioRequest.status == 'completed'
    ).count()
    failed_requests = db.session.query(AudioRequest).filter(
        AudioRequest.status == 'failed'
    ).count()

    # Performance metrics
    avg_processing_time = db.session.query(
        db.func.avg(AudioRequest.processing_time)
    ).filter(
        AudioRequest.processing_time.isnot(None)
    ).scalar()

    metrics = {
        'total_requests': total_requests,
        'completed_requests': completed_requests,
        'failed_requests': failed_requests,
        'success_rate': (completed_requests / total_requests * 100) if total_requests > 0 else 0,
        'avg_processing_time': float(avg_processing_time) if avg_processing_time else 0,
        'timestamp': datetime.utcnow().isoformat()
    }

    # Store metrics in database or send to monitoring system
    return metrics
```

## Error Handling & Retry Logic

### Task Retry Configuration
```python
@celery_app.task(
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=700,
    retry_jitter=True
)
def robust_tts_task(self, request_id: int):
    """Robust TTS task with retry logic"""
    try:
        # Task implementation
        return process_tts_request(request_id)
    except Exception as e:
        # Log error
        logger.error(f"TTS task failed: {e}", extra={
            'request_id': request_id,
            'retry_count': self.request.retries
        })

        # Retry with exponential backoff
        raise self.retry(countdown=60 * (2 ** self.request.retries))
```

### Dead Letter Queue
```python
# app/tasks/dead_letter_tasks.py
from app.celery_app import celery_app

@celery_app.task(name='app.tasks.dead_letter_tasks.handle_dead_letter')
def handle_dead_letter(request_data: Dict[str, Any]):
    """Handle permanently failed requests"""
    # Log to dead letter queue
    redis_client = redis.from_url(os.getenv('REDIS_URL'))
    redis_client.lpush('tts_dead_letter_queue', json.dumps(request_data))

    # Send notification
    send_failure_notification(request_data)

    return {'handled': True}

def send_failure_notification(request_data: Dict[str, Any]):
    """Send failure notification"""
    # Implementation for email/webhook notifications
    pass
```

## Scaling & Load Balancing

### Horizontal Scaling
```bash
# Start multiple workers
celery -A app.celery_app worker --concurrency=4 --pool=solo -n worker1@%h
celery -A app.celery_app worker --concurrency=4 --pool=solo -n worker2@%h
celery -A app.celery_app worker --concurrency=4 --pool=solo -n worker3@%h

# Start workers with different queues
celery -A app.celery_app worker -Q tts_queue --concurrency=8 -n tts-worker@%h
celery -A app.celery_app worker -Q cleanup_queue --concurrency=2 -n cleanup-worker@%h
```

### Load Balancing
```python
# Custom task router
class CustomRouter:
    def route_for_task(self, task, args=None, kwargs=None):
        if task == 'app.tasks.tts_tasks.process_tts_request':
            # Route based on user priority
            if args and args[0]:  # request_id
                # Check user priority and route accordingly
                return {'queue': 'tts_priority_queue' if self._is_premium_user(args[0]) else 'tts_queue'}
        return None

    def _is_premium_user(self, request_id: int) -> bool:
        # Implementation
        pass
```

## Testing

### Unit Tests
```python
# tests/test_tts_tasks.py
import pytest
from unittest.mock import patch, MagicMock
from app.tasks.tts_tasks import process_tts_request

@patch('app.tasks.tts_tasks.AudioProcessor')
def test_process_tts_request_success(mock_processor):
    """Test successful TTS processing"""
    mock_processor_instance = MagicMock()
    mock_processor.return_value = mock_processor_instance
    mock_processor_instance.generate_audio.return_value = (b'audio_data', 'audio/wav')

    result = process_tts_request(1)

    assert result['status'] == 'completed'
    mock_processor_instance.generate_audio.assert_called_once()

@patch('app.tasks.tts_tasks.AudioProcessor')
def test_process_tts_request_retry(mock_processor):
    """Test TTS processing with retry"""
    mock_processor_instance = MagicMock()
    mock_processor.return_value = mock_processor_instance
    mock_processor_instance.generate_audio.side_effect = Exception("API Error")

    with pytest.raises(Exception):
        process_tts_request(1)
```

### Integration Tests
```python
# tests/test_celery_integration.py
def test_celery_task_execution(celery_app, celery_worker):
    """Test Celery task execution"""
    from app.tasks.tts_tasks import process_tts_request

    # Submit task
    task = process_tts_request.delay(1)

    # Wait for result
    result = task.get(timeout=30)

    assert result is not None
    assert 'request_id' in result
```

## Deployment

### Docker Configuration
```dockerfile
# Dockerfile.worker
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Start Celery worker
CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  tts-worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tts_db
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3

  tts-beat:
    build:
      context: .
      dockerfile: Dockerfile.worker
    command: celery -A app.celery_app beat --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
```

## Performance Optimization

### Task Optimization
```python
# Use singleton pattern for heavy objects
class OptimizedAudioProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.processor = AudioProcessor(os.getenv('GEMINI_API_KEY'))
        return cls._instance

@celery_app.task
def optimized_tts_task(request_id: int):
    processor = OptimizedAudioProcessor().processor
    # Use processor
```

### Memory Management
```python
# Limit memory usage
@celery_app.task
def memory_efficient_task(request_id: int):
    import gc

    try:
        # Process task
        result = process_request(request_id)
        return result
    finally:
        # Clean up memory
        gc.collect()
```

### Connection Pooling
```python
# Redis connection pooling
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=20,
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=redis_pool)