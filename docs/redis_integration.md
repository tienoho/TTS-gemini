# Redis Integration cho TTS API

## Tổng quan
Redis được sử dụng để quản lý queue, caching, và rate limiting trong hệ thống TTS API. Hỗ trợ horizontal scaling và high availability.

## Redis Configuration

### Environment Variables
```bash
# Redis connection
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password_here

# Queue settings
TTS_QUEUE_NAME=tts_queue
TTS_PRIORITY_QUEUE=tts_priority_queue
TTS_RETRY_QUEUE=tts_retry_queue

# Cache settings
CACHE_TTL=3600  # 1 hour
STATUS_CACHE_TTL=300  # 5 minutes
```

## Queue Management

### 1. Main TTS Queue
```python
import redis
import json
from typing import Dict, Any

class TTSQueueManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.queue_name = "tts_queue"

    def enqueue_request(self, request_data: Dict[str, Any], priority: int = 0):
        """Enqueue TTS request với priority"""
        queue_item = {
            'request_id': request_data['id'],
            'user_id': request_data['user_id'],
            'text': request_data['text_content'],
            'voice_settings': {
                'voice_name': request_data['voice_name'],
                'speed': request_data['speed'],
                'pitch': request_data['pitch']
            },
            'priority': priority,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Add to main queue
        self.redis.lpush(self.queue_name, json.dumps(queue_item))

        # Add to priority queue if priority > 0
        if priority > 0:
            self.redis.zadd(f"{self.queue_name}_priority",
                          {json.dumps(queue_item): priority})

    def dequeue_request(self) -> Dict[str, Any]:
        """Dequeue next request (priority first, then FIFO)"""
        # Try priority queue first
        priority_item = self.redis.zpopmax(f"{self.queue_name}_priority")
        if priority_item:
            return json.loads(priority_item[0][0])

        # Fall back to regular queue
        item = self.redis.rpop(self.queue_name)
        return json.loads(item) if item else None

    def get_queue_length(self) -> int:
        """Get current queue length"""
        return self.redis.llen(self.queue_name)
```

### 2. Status Tracking
```python
class StatusManager:
    def __init__(self, redis_client):
        self.redis = redis_client

    def set_status(self, request_id: str, status: str, progress: int = 0):
        """Update request status với progress"""
        status_key = f"tts_status:{request_id}"
        status_data = {
            'status': status,
            'progress': progress,
            'updated_at': datetime.utcnow().isoformat()
        }

        self.redis.hset(status_key, mapping=status_data)
        self.redis.expire(status_key, 3600)  # Expire after 1 hour

    def get_status(self, request_id: str) -> Dict[str, Any]:
        """Get current status của request"""
        status_key = f"tts_status:{request_id}"
        return self.redis.hgetall(status_key)

    def update_progress(self, request_id: str, progress: int):
        """Update progress percentage"""
        self.redis.hset(f"tts_status:{request_id}", 'progress', progress)
```

### 3. Retry Queue
```python
class RetryManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.retry_queue = "tts_retry_queue"

    def add_to_retry(self, request_data: Dict[str, Any], retry_count: int = 0):
        """Add failed request to retry queue"""
        retry_item = {
            **request_data,
            'retry_count': retry_count,
            'next_retry_at': self._calculate_next_retry(retry_count)
        }

        self.redis.lpush(self.retry_queue, json.dumps(retry_item))

    def get_retry_items(self) -> List[Dict[str, Any]]:
        """Get items ready for retry"""
        current_time = datetime.utcnow().timestamp()
        items = self.redis.lrange(self.retry_queue, 0, -1)

        ready_items = []
        for item_str in items:
            item = json.loads(item_str)
            if item['next_retry_at'] <= current_time:
                ready_items.append(item)

        return ready_items

    def _calculate_next_retry(self, retry_count: int) -> float:
        """Calculate next retry time với exponential backoff"""
        delay = min(30 * (2 ** retry_count), 3600)  # Max 1 hour
        return datetime.utcnow().timestamp() + delay
```

## Caching Strategy

### 1. Status Cache
```python
class StatusCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    def cache_status(self, request_id: str, status_data: Dict[str, Any]):
        """Cache status data"""
        cache_key = f"tts_status_cache:{request_id}"
        self.redis.setex(
            cache_key,
            300,  # 5 minutes TTL
            json.dumps(status_data)
        )

    def get_cached_status(self, request_id: str) -> Dict[str, Any]:
        """Get cached status"""
        cache_key = f"tts_status_cache:{request_id}"
        cached = self.redis.get(cache_key)
        return json.loads(cached) if cached else None
```

### 2. Rate Limiting
```python
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def check_rate_limit(self, user_id: str, endpoint: str, limit: int, window: int = 60):
        """Check if user exceeded rate limit"""
        window_key = f"rate_limit:{user_id}:{endpoint}"
        current_time = int(time.time())

        # Clean old entries
        self.redis.zremrangebyscore(window_key, 0, current_time - window)

        # Count requests in current window
        request_count = self.redis.zcard(window_key)

        if request_count >= limit:
            return False, request_count

        # Add current request
        self.redis.zadd(window_key, {str(current_time): current_time})
        self.redis.expire(window_key, window)

        return True, request_count + 1

    def get_remaining_time(self, user_id: str, endpoint: str, window: int = 60):
        """Get remaining time until rate limit resets"""
        window_key = f"rate_limit:{user_id}:{endpoint}"
        oldest = self.redis.zrange(window_key, 0, 0, withscores=True)

        if not oldest:
            return 0

        return max(0, window - (int(time.time()) - int(oldest[0][1])))
```

## Redis Keys Structure

### Queue Keys
```
# Main processing queue
LIST: tts_queue

# Priority queue (sorted by priority score)
ZSET: tts_queue_priority

# Retry queue for failed requests
LIST: tts_retry_queue

# Dead letter queue for permanently failed requests
LIST: tts_dead_letter_queue
```

### Status Keys
```
# Request status tracking
HASH: tts_status:{request_id}
# Fields: status, progress, worker_id, started_at, completed_at

# Cached status for fast retrieval
STRING: tts_status_cache:{request_id}

# Worker heartbeat
HASH: worker_status:{worker_id}
# Fields: last_seen, current_job, queue_length
```

### Rate Limiting Keys
```
# User rate limiting per endpoint
ZSET: rate_limit:{user_id}:{endpoint}

# Global rate limiting
ZSET: global_rate_limit:{endpoint}
```

### Cache Keys
```
# Audio file cache (for frequently requested files)
STRING: audio_cache:{checksum}

# User session cache
HASH: user_session:{user_id}

# API response cache
STRING: api_cache:{hash_of_request}
```

## Monitoring và Metrics

### Queue Metrics
```python
def get_queue_metrics(redis_client):
    """Get queue performance metrics"""
    metrics = {
        'queue_length': redis_client.llen('tts_queue'),
        'priority_queue_length': redis_client.zcard('tts_queue_priority'),
        'retry_queue_length': redis_client.llen('tts_retry_queue'),
        'processing_requests': redis_client.scard('processing_requests'),
        'failed_requests': redis_client.llen('tts_dead_letter_queue')
    }

    # Calculate processing rate
    completed_today = redis_client.get('completed_requests_today')
    metrics['completed_today'] = int(completed_today) if completed_today else 0

    return metrics
```

### Worker Status
```python
def get_worker_status(redis_client):
    """Get status of all workers"""
    workers = redis_client.keys('worker_status:*')
    worker_status = []

    for worker_key in workers:
        status = redis_client.hgetall(worker_key)
        worker_id = worker_key.decode().split(':')[-1]
        worker_status.append({
            'worker_id': worker_id,
            'last_seen': status.get(b'last_seen', b'').decode(),
            'current_job': status.get(b'current_job', b'').decode(),
            'status': 'active' if self._is_worker_alive(status) else 'inactive'
        })

    return worker_status
```

## Error Handling

### Redis Connection Issues
```python
class RedisManager:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self):
        try:
            return redis.from_url(self.redis_url)
        except redis.ConnectionError:
            # Fallback to in-memory storage or database
            logger.warning("Redis connection failed, using fallback storage")
            return self._create_fallback_client()

    def _create_fallback_client(self):
        """Create fallback client using database"""
        # Implementation for fallback storage
        pass
```

## Performance Optimization

### Connection Pooling
```python
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=20,
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=redis_pool)
```

### Pipeline Operations
```python
def batch_update_status(self, updates: List[Dict]):
    """Batch update multiple request statuses"""
    with self.redis.pipeline() as pipe:
        for update in updates:
            status_key = f"tts_status:{update['request_id']}"
            pipe.hset(status_key, mapping=update['status_data'])
            pipe.expire(status_key, 3600)
        pipe.execute()
```

## Security Considerations

### Data Encryption
```python
import base64

def encrypt_sensitive_data(data: str, key: str) -> str:
    """Encrypt sensitive data before storing in Redis"""
    cipher = Fernet(key)
    return base64.b64encode(cipher.encrypt(data.encode())).decode()

def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
    """Decrypt sensitive data from Redis"""
    cipher = Fernet(key)
    return cipher.decrypt(base64.b64decode(encrypted_data)).decode()
```

### Key Namespacing
```python
class RedisKeyManager:
    NAMESPACE = "tts_api"

    @classmethod
    def get_queue_key(cls, queue_name: str) -> str:
        return f"{cls.NAMESPACE}:queue:{queue_name}"

    @classmethod
    def get_status_key(cls, request_id: str) -> str:
        return f"{cls.NAMESPACE}:status:{request_id}"

    @classmethod
    def get_cache_key(cls, key: str) -> str:
        return f"{cls.NAMESPACE}:cache:{key}"