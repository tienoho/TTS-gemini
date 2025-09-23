# Performance Optimization và Scalability

## Tổng quan

Tài liệu này mô tả các chiến lược tối ưu performance và scalability cho hệ thống TTS API, đảm bảo hệ thống có thể xử lý tải cao và mở rộng theo nhu cầu.

## 1. Application Performance

### 1.1. Database Optimization

#### PostgreSQL Performance Tuning

```sql
-- Tạo indexes cho performance
CREATE INDEX idx_requests_status_created ON requests(status, created_at);
CREATE INDEX idx_requests_user_id ON requests(user_id);
CREATE INDEX idx_logs_request_id ON logs(request_id);
CREATE INDEX idx_logs_timestamp ON logs(timestamp);
CREATE INDEX idx_rate_limit_user_timestamp ON rate_limit(user_id, timestamp);

-- Partitioning cho bảng requests (nếu có nhiều data)
CREATE TABLE requests_y2024m01 PARTITION OF requests
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Connection pooling configuration
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

#### Query Optimization

```python
# Sử dụng eager loading để tránh N+1 queries
requests = Request.query.options(
    joinedload(Request.user),
    joinedload(Request.logs)
).filter(Request.status == 'completed').all()

# Batch operations cho bulk updates
from sqlalchemy import update

stmt = update(Request).where(
    Request.id.in_(request_ids)
).values(status='processing')
session.execute(stmt)
session.commit()
```

### 1.2. Redis Optimization

#### Connection Pooling

```python
# config/redis.py
import redis.asyncio as redis

redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=50,
    decode_responses=True,
    retry_on_timeout=True
)

redis_client = redis.Redis(connection_pool=redis_pool)
```

#### Pipeline Operations

```python
# Batch Redis operations
async def batch_update_status(request_ids, status):
    async with redis_client.pipeline() as pipe:
        for request_id in request_ids:
            pipe.hset(f"tts_status:{request_id}", "status", status)
            pipe.hset(f"tts_status:{request_id}", "updated_at", datetime.utcnow().isoformat())
        await pipe.execute()
```

### 1.3. Application Caching

#### Multi-level Caching Strategy

```python
# utils/cache.py
import asyncio
from functools import lru_cache
from redis.asyncio import Redis

class CacheManager:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.local_cache = {}

    @lru_cache(maxsize=1000)
    def get_user_settings(self, user_id: str):
        """Local cache với LRU eviction"""
        return self._get_from_redis(f"user_settings:{user_id}")

    async def get_request_status(self, request_id: str):
        """Redis cache cho status updates"""
        cache_key = f"tts_status:{request_id}"

        # Check Redis first
        cached = await self.redis.hgetall(cache_key)
        if cached:
            return cached

        # Fallback to database
        return await self._get_from_db(request_id)

    async def invalidate_user_cache(self, user_id: str):
        """Invalidate cache khi user data thay đổi"""
        await self.redis.delete(f"user_settings:{user_id}")
        self.get_user_settings.cache_clear()
```

## 2. Scalability Architecture

### 2.1. Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx.conf
upstream tts_backend {
    least_conn;
    server tts-app-1:8000 weight=3;
    server tts-app-2:8000 weight=3;
    server tts-app-3:8000 weight=2;
    server tts-app-4:8000 weight=2;
}

server {
    listen 80;
    server_name api.tts.example.com;

    location / {
        proxy_pass http://tts_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Health checks
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

#### Docker Compose Scaling

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  tts-app:
    image: tts-api:latest
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    environment:
      - REDIS_URL=redis://redis-cluster:6379
      - DATABASE_URL=postgresql://db:5432/tts_prod
    depends_on:
      - redis-cluster
      - db

  redis-cluster:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    deploy:
      replicas: 3
    volumes:
      - redis-data:/data

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=tts_prod
      - POSTGRES_USER=tts_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    deploy:
      replicas: 1
      placement:
        constraints: [node.role == manager]
```

### 2.2. Database Scaling

#### Read Replicas

```yaml
# docker-compose.replicas.yml
services:
  db-primary:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=tts_prod
      - POSTGRES_USER=tts_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-primary:/var/lib/postgresql/data
    command: postgres -c wal_level=replica

  db-replica-1:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=tts_prod
      - POSTGRES_USER=tts_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-replica-1:/var/lib/postgresql/data
    command: postgres -c hot_standby=on
    depends_on:
      - db-primary
```

#### Connection Pooling với PgBouncer

```ini
# pgbouncer.ini
[databases]
tts_prod = host=db-primary port=5432 dbname=tts_prod

[pgbouncer]
pool_mode = transaction
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

max_client_conn = 1000
default_pool_size = 25
min_pool_size = 5
reserve_pool_size = 5
max_db_connections = 50
```

### 2.3. Redis Clustering

#### Redis Cluster Configuration

```python
# config/redis_cluster.py
from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster

class RedisClusterManager:
    def __init__(self):
        self.cluster = RedisCluster(
            startup_nodes=[
                {"host": "redis-node-1", "port": 6379},
                {"host": "redis-node-2", "port": 6379},
                {"host": "redis-node-3", "port": 6379},
            ],
            decode_responses=True,
            max_connections=32,
            retry_on_timeout=True,
        )

    async def get_queue_length(self, queue_name: str) -> int:
        """Get queue length across all cluster nodes"""
        total_length = 0
        for node in self.cluster.get_nodes():
            length = await self.cluster.llen(queue_name)
            total_length += length
        return total_length
```

## 3. Background Processing Optimization

### 3.1. Celery Worker Optimization

#### Worker Configuration

```python
# celery_app.py
from celery import Celery
from celery.schedules import crontab

app = Celery('tts_processor')
app.config_from_object('config.celery')

# Worker optimization
app.conf.update(
    worker_prefetch_multiplier=1,  # Disable prefetch for fair queuing
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    worker_disable_rate_limits=False,
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

# Task routing cho load balancing
app.conf.task_routes = {
    'tasks.process_tts': {'queue': 'tts_priority'},
    'tasks.cleanup_old_files': {'queue': 'maintenance'},
}
```

#### Priority Queues

```python
# tasks.py
@app.task(queue='tts_priority', priority=5)
def process_tts_priority(request_id: str):
    """High priority TTS processing"""
    pass

@app.task(queue='tts_normal', priority=1)
def process_tts_normal(request_id: str):
    """Normal priority TTS processing"""
    pass

@app.task(queue='maintenance', priority=0)
def cleanup_old_files():
    """Maintenance tasks"""
    pass
```

### 3.2. AsyncIO Workers

#### Async Worker Implementation

```python
# workers/async_worker.py
import asyncio
import aioredis
from concurrent.futures import ThreadPoolExecutor
import logging

class AsyncTTSWorker:
    def __init__(self, redis_url: str, db_url: str):
        self.redis_url = redis_url
        self.db_url = db_url
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start async worker"""
        self.redis = aioredis.from_url(self.redis_url)
        self.logger.info("Async TTS worker started")

        while True:
            try:
                # Process queue with timeout
                request_data = await asyncio.wait_for(
                    self.redis.blpop('tts_queue', timeout=1.0),
                    timeout=1.0
                )

                if request_data:
                    await self.process_request(request_data[1])

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)

    async def process_request(self, request_data: bytes):
        """Process single TTS request"""
        try:
            request_info = json.loads(request_data)

            # Run CPU-intensive TTS processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_process_tts,
                request_info
            )

            await self._update_status(request_info['id'], 'completed', result)

        except Exception as e:
            self.logger.error(f"Failed to process request {request_info['id']}: {e}")
            await self._update_status(request_info['id'], 'failed', str(e))

    def _sync_process_tts(self, request_info: dict):
        """Synchronous TTS processing (runs in thread pool)"""
        # TTS processing logic here
        pass
```

## 4. Monitoring và Performance Metrics

### 4.1. Application Metrics

#### Custom Metrics Collection

```python
# utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
TTS_REQUESTS = Counter(
    'tts_requests_total',
    'Total number of TTS requests',
    ['status', 'user_id', 'voice']
)

TTS_PROCESSING_TIME = Histogram(
    'tts_processing_duration_seconds',
    'Time spent processing TTS requests',
    ['voice', 'text_length']
)

QUEUE_LENGTH = Gauge(
    'tts_queue_length',
    'Current queue length'
)

ACTIVE_WORKERS = Gauge(
    'tts_active_workers',
    'Number of active workers'
)

def track_tts_request(user_id: str, voice: str, text_length: int):
    """Track TTS request metrics"""
    TTS_REQUESTS.labels(status='received', user_id=user_id, voice=voice).inc()

def track_processing_time(voice: str, text_length: int, duration: float):
    """Track processing time"""
    TTS_PROCESSING_TIME.labels(voice=voice, text_length=str(text_length)).observe(duration)

def update_queue_length(length: int):
    """Update queue length metric"""
    QUEUE_LENGTH.set(length)
```

### 4.2. Performance Testing

#### Load Testing với Locust

```python
# tests/load_test.py
from locust import HttpUser, task, between
import json

class TTSLoadTest(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def test_tts_request(self):
        """Test TTS request endpoint"""
        payload = {
            "text": "This is a test text for load testing",
            "voice_settings": {
                "language": "en-US",
                "voice": "en-US-Wavenet-F",
                "speed": 1.0,
                "pitch": 0.0
            },
            "user_id": "load_test_user"
        }

        with self.client.post(
            "/tts/request",
            json=payload,
            headers={"Authorization": f"Bearer {self.token}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def test_status_check(self):
        """Test status check endpoint"""
        self.client.get(
            f"/tts/status/{self.request_id}",
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

#### Performance Benchmarks

```bash
# Load testing commands
# Ramp up test
locust -f tests/load_test.py --host http://localhost:8000 -u 100 -r 10 --run-time 5m

# Stress test
locust -f tests/load_test.py --host http://localhost:8000 -u 1000 -r 100 --run-time 10m

# Spike test
locust -f tests/load_test.py --host http://localhost:8000 -u 500 -r 500 --run-time 2m
```

## 5. Caching Strategies

### 5.1. Response Caching

#### API Response Caching

```python
# middleware/cache_middleware.py
from functools import wraps
import hashlib
import json

def cache_response(timeout: int = 300):
    """Cache decorator cho API responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, args, kwargs)

            # Check cache first
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                return json.loads(cached_response)

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await redis_client.setex(
                cache_key,
                timeout,
                json.dumps(result)
            )

            return result
        return wrapper
    return decorator

def generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key từ function name và parameters"""
    key_data = {
        'function': func_name,
        'args': str(args),
        'kwargs': str(sorted(kwargs.items()))
    }
    return hashlib.md5(json.dumps(key_data).encode()).hexdigest()
```

### 5.2. Static Content Caching

#### Nginx Static Caching

```nginx
# Static file caching
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header X-Cache "STATIC";
}

# API response caching
location /api/v1/status/ {
    expires 30s;
    add_header Cache-Control "public";
    add_header X-Cache "API";
}
```

## 6. Database Connection Optimization

### 6.1. Connection Pooling

#### SQLAlchemy Connection Pool

```python
# config/database.py
from sqlalchemy.pool import QueuePool
from sqlalchemy import create_engine

def create_db_engine(database_url: str):
    """Create database engine với optimized connection pool"""
    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,   # Recycle connections after 1 hour
        echo=False,          # Disable SQL logging in production
        future=True          # Use SQLAlchemy 2.0 style
    )
```

### 6.2. Read/Write Splitting

```python
# utils/db_router.py
class DatabaseRouter:
    def __init__(self, read_engine, write_engine):
        self.read_engine = read_engine
        self.write_engine = write_engine

    def get_engine(self, operation: str = 'read'):
        """Route database operations to appropriate engine"""
        if operation.lower() in ['write', 'update', 'delete', 'insert']:
            return self.write_engine
        return self.read_engine

# Usage
db_router = DatabaseRouter(read_engine, write_engine)

# In repository
def get_request_status(self, request_id: str):
    with db_router.get_engine('read').connect() as conn:
        result = conn.execute(
            text("SELECT status, progress FROM requests WHERE id = :id"),
            {"id": request_id}
        )
        return result.fetchone()
```

## 7. Memory Management

### 7.1. Memory Optimization

#### Worker Memory Management

```python
# workers/memory_manager.py
import gc
import psutil
import logging
from typing import Dict, Any

class MemoryManager:
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
        self.logger = logging.getLogger(__name__)

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        memory_percent = self.process.memory_percent()
        return memory_percent < self.max_memory_percent

    def force_garbage_collection(self):
        """Force garbage collection if needed"""
        if not self.check_memory_usage():
            self.logger.warning("High memory usage detected, forcing GC")
            gc.collect()
            # Clear any caches
            self._clear_caches()

    def _clear_caches(self):
        """Clear various caches to free memory"""
        # Clear local caches
        if hasattr(self, 'local_cache'):
            self.local_cache.clear()

        # Clear Redis caches if needed
        # Implementation depends on cache manager
```

### 7.2. Streaming Large Responses

```python
# utils/streaming.py
from fastapi.responses import StreamingResponse
import io

async def stream_audio_file(file_path: str):
    """Stream large audio files to reduce memory usage"""
    async def generate():
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):  # 8KB chunks
                yield chunk

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=output.mp3"}
    )
```

## 8. Deployment Optimization

### 8.1. Container Optimization

#### Multi-stage Docker Build

```dockerfile
# Dockerfile.optimized
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim as runtime

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1

# Security optimizations
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 8.2. Startup Optimization

#### Application Preloading

```python
# app/preload.py
import asyncio
from app.extensions import db, redis_client
from models import Request, User
from utils.cache import CacheManager

async def preload_application():
    """Preload application data and connections"""
    # Warm up database connections
    await db.engine.connect()

    # Preload frequently accessed data
    cache_manager = CacheManager(redis_client)

    # Load user settings
    users = await User.query.all()
    for user in users:
        await cache_manager.get_user_settings(user.id)

    # Preload voice configurations
    await cache_manager.preload_voice_configs()

    print("Application preloading completed")
```

## 9. Performance Monitoring

### 9.1. Real-time Monitoring

#### Health Check Endpoints

```python
# routes/health.py
from fastapi import APIRouter
import psutil
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check với metrics"""
    # System metrics
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu_percent = psutil.cpu_percent(interval=1)

    # Application metrics
    queue_length = await redis_client.llen('tts_queue')
    active_connections = len(db.engine.pool.checkedout())

    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent
        },
        "application": {
            "queue_length": queue_length,
            "active_db_connections": active_connections,
            "uptime": time.time() - start_time
        }
    }
```

### 9.2. Performance Profiling

#### Production Profiling

```python
# middleware/profiling.py
import cProfile
import pstats
import io
from functools import wraps

class ProductionProfiler:
    def __init__(self, enable_profiling: bool = False):
        self.enable_profiling = enable_profiling

    def profile_function(self, func):
        """Profile function execution"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.enable_profiling:
                return await func(*args, **kwargs)

            pr = cProfile.Profile()
            pr.enable()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                pr.disable()
                self._save_profile_stats(pr, func.__name__)

        return wrapper

    def _save_profile_stats(self, pr: cProfile.Profile, function_name: str):
        """Save profiling statistics"""
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions

        # Log or save to file
        self.logger.info(f"Profile for {function_name}: {s.getvalue()}")
```

## 10. Cost Optimization

### 10.1. Resource Optimization

#### Auto-scaling Configuration

```yaml
# docker-compose.autoscale.yml
services:
  tts-app:
    image: tts-api:latest
    deploy:
      replicas: 2-10  # Min 2, max 10 replicas
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    environment:
      - AUTOSCALING_ENABLED=true
      - SCALE_DOWN_THRESHOLD=30  # Scale down if CPU < 30%
      - SCALE_UP_THRESHOLD=70    # Scale up if CPU > 70%

  autoscaler:
    image: autoscaler:latest
    environment:
      - DOCKER_HOST=tcp://docker-socket-proxy:2376
      - SCALE_CHECK_INTERVAL=30  # Check every 30 seconds
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

### 10.2. Gemini API Cost Optimization

```python
# utils/gemini_optimizer.py
class GeminiCostOptimizer:
    def __init__(self):
        self.batch_size = 10
        self.max_batch_wait = 5  # seconds

    async def optimize_batch_request(self, requests: List[dict]):
        """Batch requests để optimize API calls"""
        if len(requests) < self.batch_size:
            # Process individually
            return await self._process_individual_requests(requests)

        # Batch processing
        return await self._process_batch_requests(requests)

    async def _process_batch_requests(self, requests: List[dict]):
        """Process requests in batches"""
        results = []
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_result = await self._call_gemini_batch(batch)
            results.extend(batch_result)
            await asyncio.sleep(0.1)  # Rate limiting
        return results
```

## Kết luận

Các chiến lược tối ưu performance và scalability này đảm bảo hệ thống TTS API có thể:

1. **Xử lý tải cao** với horizontal scaling và load balancing
2. **Tối ưu resource usage** với caching và connection pooling
3. **Giảm latency** với async processing và streaming
4. **Giảm chi phí** với batching và auto-scaling
5. **Monitor performance** với comprehensive metrics và alerting

Việc implement các optimizations này sẽ tạo ra một hệ thống production-ready có thể scale từ vài requests đến hàng nghìn requests per minute.