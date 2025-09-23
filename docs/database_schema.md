# Database Schema cho TTS API với PostgreSQL

## Tổng quan
Database schema được thiết kế để hỗ trợ hệ thống TTS với queue management, logging chi tiết, và monitoring. Sử dụng PostgreSQL để đảm bảo performance và reliability.

## Tables chính

### 1. users
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_premium BOOLEAN DEFAULT FALSE,
    api_key VARCHAR(100) UNIQUE,
    api_key_expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. audio_requests
```sql
CREATE TABLE audio_requests (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    text_content TEXT NOT NULL,
    voice_name VARCHAR(50) DEFAULT 'Alnilam',
    output_format VARCHAR(10) DEFAULT 'wav',
    speed FLOAT DEFAULT 1.0,
    pitch FLOAT DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    processing_time FLOAT, -- in seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. audio_files
```sql
CREATE TABLE audio_files (
    id SERIAL PRIMARY KEY,
    request_id INTEGER REFERENCES audio_requests(id) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    mime_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL, -- in bytes
    checksum VARCHAR(64) NOT NULL, -- SHA256 hash
    duration FLOAT, -- in seconds
    sample_rate INTEGER,
    channels INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. logs (NEW - Structured logging)
```sql
CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    request_id INTEGER REFERENCES audio_requests(id),
    user_id INTEGER REFERENCES users(id),
    level VARCHAR(10) NOT NULL, -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    source VARCHAR(100), -- component name
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_logs_request_id ON logs(request_id);
CREATE INDEX idx_logs_user_id ON logs(user_id);
CREATE INDEX idx_logs_level ON logs(level);
CREATE INDEX idx_logs_timestamp ON logs(timestamp);
```

### 5. queue_status (NEW - Queue management)
```sql
CREATE TABLE queue_status (
    id SERIAL PRIMARY KEY,
    request_id INTEGER REFERENCES audio_requests(id) UNIQUE NOT NULL,
    queue_name VARCHAR(50) DEFAULT 'tts_queue',
    priority INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'queued', -- queued, processing, completed, failed
    worker_id VARCHAR(100),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    next_retry_at TIMESTAMP,
    enqueued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_queue_status_request_id ON queue_status(request_id);
CREATE INDEX idx_queue_status_status ON queue_status(status);
CREATE INDEX idx_queue_status_priority ON queue_status(priority DESC);
CREATE INDEX idx_queue_status_next_retry ON queue_status(next_retry_at) WHERE next_retry_at IS NOT NULL;
```

### 6. rate_limits (NEW - Rate limiting)
```sql
CREATE TABLE rate_limits (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    UNIQUE(user_id, endpoint, window_start)
);

-- Indexes
CREATE INDEX idx_rate_limits_user_id ON rate_limits(user_id);
CREATE INDEX idx_rate_limits_window ON rate_limits(window_start, window_end);
```

### 7. metrics (NEW - System metrics)
```sql
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    value NUMERIC NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_metrics_name ON metrics(name);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);
```

## Redis Keys Structure

### Queue Management
```bash
# Main TTS queue
LIST: tts_queue

# Queue status tracking
HASH: tts_status:{request_id}
# Fields: status, progress, worker_id, started_at, completed_at

# Priority queue (sorted set)
ZSET: tts_priority_queue

# Failed requests for retry
LIST: tts_retry_queue
```

### Caching
```bash
# Status cache
HASH: tts_status_cache:{request_id}

# User rate limiting
ZSET: rate_limit:{user_id}

# Audio file cache
STRING: audio_cache:{checksum}
```

### Session & Authentication
```bash
# JWT blacklist
SET: jwt_blacklist:{token_id}

# User sessions
HASH: user_session:{user_id}
```

## Relationships

```mermaid
erDiagram
    users ||--o{ audio_requests : creates
    audio_requests ||--o{ audio_files : has
    audio_requests ||--o{ logs : generates
    audio_requests ||--o{ queue_status : tracks
    users ||--o{ rate_limits : has
    users ||--o{ logs : generates

    audio_requests {
        int id PK
        int user_id FK
        text text_content
        string voice_name
        string output_format
        float speed
        float pitch
        string status
        jsonb metadata
        float processing_time
        timestamp created_at
        timestamp updated_at
    }

    queue_status {
        int id PK
        int request_id FK
        string queue_name
        int priority
        string status
        string worker_id
        int retry_count
        timestamp next_retry_at
        timestamp enqueued_at
        timestamp started_at
        timestamp completed_at
    }

    logs {
        int id PK
        int request_id FK
        int user_id FK
        string level
        text message
        jsonb metadata
        string source
        timestamp timestamp
    }