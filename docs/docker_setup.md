# Docker và Docker Compose Setup cho Production

## Tổng quan
Hệ thống containerization sử dụng Docker và Docker Compose để đảm bảo consistent deployment, scalability, và easy management trong production environment.

## Docker Architecture

### Service Architecture
```mermaid
graph TB
    A[Load Balancer] --> B[API Service]
    A --> C[API Service]
    A --> D[API Service]

    B --> E[(PostgreSQL)]
    C --> E
    D --> E

    B --> F[(Redis)]
    C --> F
    D --> F

    G[Worker Service] --> E
    G --> F
    H[Worker Service] --> E
    H --> F

    I[Monitoring Service] --> J[(Prometheus)]
    I --> K[(Grafana)]
    I --> L[(ELK Stack)]
```

## Docker Configuration

### Dockerfile cho API Service
```dockerfile
# Dockerfile.api
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app.main:app"]
```

### Dockerfile cho Worker Service
```dockerfile
# Dockerfile.worker
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV C_FORCE_ROOT=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD celery -A app.celery_app inspect ping || exit 1

# Run worker
CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
```

### Gunicorn Configuration
```python
# gunicorn.conf.py
import os
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "/app/logs/access.log"
errorlog = "/app/logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'tts-api'

# Server mechanics
preload_app = True
pidfile = "/app/logs/gunicorn.pid"
user = "app"
group = "app"
tmp_upload_dir = None

# SSL (if needed)
# keyfile = "/app/ssl/private.key"
# certfile = "/app/ssl/certificate.crt"
```

## Docker Compose Configuration

### Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "5000:5000"
    volumes:
      - .:/app
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=development
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tts_dev
      - REDIS_URL=redis://redis:6379/0
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - tts-network

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    volumes:
      - .:/app
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=development
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tts_dev
      - REDIS_URL=redis://redis:6379/0
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - tts-network
    command: ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--concurrency=2"]

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=tts_dev
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - tts-network

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_dev_data:/data
    ports:
      - "6379:6379"
    networks:
      - tts-network

  # Optional: Development monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    networks:
      - tts-network

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_dev_data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - tts-network

volumes:
  postgres_dev_data:
  redis_dev_data:
  grafana_dev_data:

networks:
  tts-network:
    driver: bridge
```

### Production Environment
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tts_prod
      - REDIS_URL=redis://redis:6379/0
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - tts-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tts_prod
      - REDIS_URL=redis://redis:6379/0
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - tts-network
    command: ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--concurrency=4"]

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=tts_prod
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - tts-network
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d tts_prod"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_prod_data:/data
    networks:
      - tts-network
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api
    networks:
      - tts-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  # Monitoring stack
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - tts-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/var/lib/grafana/dashboards
    networks:
      - tts-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

volumes:
  postgres_prod_data:
  redis_prod_data:
  prometheus_data:
  grafana_data:

networks:
  tts-network:
    driver: bridge
```

### Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 50M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private auth;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json application/xml+rss;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=tts:10m rate=1r/s;

    upstream tts_api {
        least_conn;
        server api:5000 weight=10 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        server_name api.tts-service.com;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

        # Health check
        location /health {
            proxy_pass http://tts_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints
        location /api/v1/ {
            # Rate limiting
            limit_req zone=api burst=20 nodelay;

            # TTS specific rate limiting
            location ~ ^/api/v1/tts/generate {
                limit_req zone=tts burst=5 nodelay;
            }

            proxy_pass http://tts_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;

            # WebSocket support (if needed)
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Static files
        location /static/ {
            proxy_pass http://tts_api;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Documentation
        location /docs/ {
            proxy_pass http://tts_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name api.tts-service.com;

        ssl_certificate /etc/nginx/ssl/certificate.crt;
        ssl_certificate_key /etc/nginx/ssl/private.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Include the same configuration as HTTP
        # ... (same as above)
    }
}
```

## Database Initialization

### PostgreSQL Init Script
```sql
-- init.sql
-- Create databases
CREATE DATABASE tts_prod;
CREATE DATABASE tts_dev;

-- Create users
CREATE USER tts_user WITH ENCRYPTED PASSWORD 'tts_password';
CREATE USER tts_dev_user WITH ENCRYPTED PASSWORD 'tts_dev_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE tts_prod TO tts_user;
GRANT ALL PRIVILEGES ON DATABASE tts_dev TO tts_dev_user;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

### Database Migration
```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os

# Import your models
from models import Base

config = context.config

# Set up logging
fileConfig(config.config_file_name)

# Add your model's MetaData object here
target_metadata = Base.metadata

def run_migrations_online():
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = os.getenv('DATABASE_URL')

    connectable = engine_from_config(
        configuration,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()
```

## Environment Configuration

### Environment Variables
```bash
# Application
FLASK_ENV=production
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret

# Database
DATABASE_URL=postgresql://tts_user:tts_password@postgres:5432/tts_prod
POSTGRES_PASSWORD=tts_password

# Redis
REDIS_URL=redis://redis:6379/0

# External APIs
GEMINI_API_KEY=your-gemini-api-key

# AWS S3 (if using)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=tts-audio-files

# Monitoring
GRAFANA_PASSWORD=admin

# SSL
SSL_CERT_PATH=/etc/nginx/ssl/certificate.crt
SSL_KEY_PATH=/etc/nginx/ssl/private.key
```

### Environment File Template
```bash
# .env.example
# Application
FLASK_ENV=production
SECRET_KEY=change-this-in-production
JWT_SECRET_KEY=change-this-in-production
API_HOST=https://api.tts-service.com

# Database
DATABASE_URL=postgresql://tts_user:tts_password@postgres:5432/tts_prod
POSTGRES_PASSWORD=change-this-in-production

# Redis
REDIS_URL=redis://redis:6379/0

# External APIs
GEMINI_API_KEY=your-gemini-api-key

# Storage (choose one)
# AWS S3
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=tts-audio-files
STORAGE_TYPE=s3

# Google Cloud Storage
GCP_PROJECT_ID=your-project-id
GCP_BUCKET_NAME=tts-audio-files
GCP_CREDENTIALS_PATH=/app/credentials.json
STORAGE_TYPE=gcs

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_PASSWORD=change-this-in-production

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=/app/logs/tts_api.log

# Security
ENABLE_RATE_LIMITING=true
ENABLE_CORS=false
ALLOWED_ORIGINS=https://your-frontend.com
```

## Deployment Scripts

### Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Starting TTS API deployment..."

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Build and deploy
echo "🐳 Building Docker images..."
docker-compose -f docker-compose.prod.yml build

# Run database migrations
echo "🗃️ Running database migrations..."
docker-compose -f docker-compose.prod.yml run --rm api alembic upgrade head

# Deploy services
echo "🚀 Deploying services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Run health checks
echo "🏥 Running health checks..."
curl -f http://localhost/api/v1/health || exit 1

# Run smoke tests
echo "🧪 Running smoke tests..."
python scripts/smoke_test.py

echo "✅ Deployment completed successfully!"
```

### Rollback Script
```bash
#!/bin/bash
# rollback.sh

set -e

echo "🔄 Starting rollback..."

# Get previous image
PREVIOUS_IMAGE=$(docker-compose -f docker-compose.prod.yml ps -q api | xargs docker inspect | jq -r '.[0].Image')

# Deploy previous version
echo "📦 Rolling back to previous version..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for rollback to complete
sleep 30

# Verify rollback
echo "✅ Verifying rollback..."
curl -f http://localhost/api/v1/health || exit 1

echo "🔄 Rollback completed successfully!"
```

## Monitoring và Health Checks

### Health Check Endpoints
```python
# app/health.py
from flask import Blueprint, jsonify
from datetime import datetime

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health_check():
    """Basic health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@health_bp.route('/health/detailed')
def detailed_health_check():
    """Detailed health check"""
    from app.extensions import db
    from redis import Redis

    # Check database
    db_status = 'healthy'
    try:
        db.session.execute('SELECT 1')
    except Exception:
        db_status = 'unhealthy'

    # Check Redis
    redis_status = 'healthy'
    try:
        redis_client = Redis.from_url(current_app.config['REDIS_URL'])
        redis_client.ping()
    except Exception:
        redis_status = 'unhealthy'

    # Check external APIs
    gemini_status = 'healthy'
    try:
        # Add Gemini API health check
        pass
    except Exception:
        gemini_status = 'unhealthy'

    overall_status = 'healthy'
    if db_status != 'healthy' or redis_status != 'unhealthy':
        overall_status = 'unhealthy'

    return jsonify({
        'status': overall_status,
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': db_status,
            'redis': redis_status,
            'gemini_api': gemini_status
        },
        'version': '1.0.0'
    })
```

### Docker Health Checks
```yaml
# docker-compose.prod.yml health checks
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tts_user -d tts_prod"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Security Considerations

### Docker Security
```bash
# Security scanning
docker scan tts-api:latest
docker scan tts-worker:latest

# Run as non-root
USER app

# Remove unnecessary packages
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Use specific package versions
RUN pip install --no-cache-dir -r requirements.txt --only-binary=all
```

### Network Security
```yaml
# docker-compose.prod.yml network security
networks:
  tts-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
    driver_opts:
      com.docker.network.bridge.name: tts-bridge

# Service isolation
services:
  api:
    networks:
      - tts-network
    depends_on:
      - postgres
      - redis
    # Prevent container escape
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
      - /var/cache/nginx
```

## Scaling Configuration

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  api:
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 2
        delay: 10s
        order: start-first

  worker:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

### Auto-scaling Rules
```yaml
# Auto-scaling configuration (example for cloud providers)
auto_scaling:
  min_replicas: 2
  max_replicas: 10
  target_cpu_utilization_percentage: 70
  target_memory_utilization_percentage: 80

scaling_rules:
  - name: "High CPU Usage"
    condition: "cpu_usage > 70%"
    action: "scale_out"
    cooldown: 300

  - name: "Low CPU Usage"
    condition: "cpu_usage < 30%"
    action: "scale_in"
    cooldown: 600
```

## Backup và Recovery

### Database Backup
```bash
#!/bin/bash
# backup.sh

# Backup PostgreSQL
docker exec postgres pg_dump -U tts_user tts_prod > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup Redis
docker exec redis redis-cli SAVE

# Backup volumes
docker run --rm -v tts_postgres_prod_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
docker run --rm -v tts_redis_prod_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup.tar.gz /data

# Upload to cloud storage
aws s3 cp *.tar.gz s3://tts-backups/
```

### Recovery Script
```bash
#!/bin/bash
# recovery.sh

# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore volumes
aws s3 cp s3://tts-backups/postgres_backup.tar.gz .
docker run --rm -v tts_postgres_prod_data:/data -v $(pwd):/backup alpine sh -c "cd / && tar xzf /backup/postgres_backup.tar.gz"

# Start services
docker-compose -f docker-compose.prod.yml up -d
```

## Configuration Management

### ConfigMap (Kubernetes)
```yaml
# k8s-configmap.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tts-api-config
data:
  FLASK_ENV: "production"
  LOG_LEVEL: "INFO"
  WORKER_CONCURRENCY: "4"
  API_HOST: "https://api.tts-service.com"
```

### Secrets Management
```yaml
# k8s-secret.yml
apiVersion: v1
kind: Secret
metadata:
  name: tts-api-secrets
type: Opaque
data:
  jwt-secret-key: <base64-encoded-secret>
  database-password: <base64-encoded-password>
  gemini-api-key: <base64-encoded-api-key>
```

## Best Practices

### 1. Security
- Run containers as non-root user
- Use secrets management
- Implement network segmentation
- Regular security scanning
- Keep images minimal

### 2. Performance
- Use multi-stage builds
- Implement health checks
- Configure resource limits
- Use appropriate base images
- Optimize layer caching

### 3. Reliability
- Implement graceful shutdown
- Use init containers
- Configure restart policies
- Implement circuit breakers
- Use rolling updates

### 4. Monitoring
- Implement comprehensive health checks
- Use structured logging
- Set up alerting
- Monitor resource usage
- Track performance metrics

### 5. Maintenance
- Regular backup procedures
- Update strategies
- Configuration management
- Documentation
- Disaster recovery plans

## Testing

### Docker Tests
```python
# tests/test_docker.py
import subprocess
import time
import requests

def test_docker_compose():
    """Test Docker Compose setup"""
    # Start services
    result = subprocess.run(['docker-compose', '-f', 'docker-compose.test.yml', 'up', '-d'],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Wait for services to be ready
    time.sleep(30)

    # Test API health
    response = requests.get('http://localhost:5000/api/v1/health')
    assert response.status_code == 200

    # Test database connection
    response = requests.get('http://localhost:5000/api/v1/health/detailed')
    data = response.json()
    assert data['services']['database'] == 'healthy'

    # Clean up
    subprocess.run(['docker-compose', '-f', 'docker-compose.test.yml', 'down'])

def test_container_security():
    """Test container security"""
    # Check if containers run as non-root
    result = subprocess.run(['docker', 'exec', 'tts-api', 'whoami'],
                          capture_output=True, text=True)
    assert 'app' in result.stdout

    # Check if unnecessary packages are removed
    result = subprocess.run(['docker', 'exec', 'tts-api', 'dpkg', '-l'],
                          capture_output=True, text=True)
    # Verify minimal package installation
```

## Deployment Pipeline

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy TTS API

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          docker-compose -f docker-compose.test.yml up -d
          sleep 30
          python -m pytest tests/ -v

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -f Dockerfile.api -t tts-api:${{ github.sha }}
          docker build -f Dockerfile.worker -t tts-worker:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          ./scripts/deploy.sh
```

## Summary

Docker setup provides:

1. **Containerization**: Consistent deployment across environments
2. **Scalability**: Easy horizontal scaling with load balancing
3. **Security**: Isolated services with security best practices
4. **Monitoring**: Comprehensive health checks and monitoring
5. **Management**: Easy deployment, scaling, and maintenance

Key components:
- Multi-stage Docker builds
- Docker Compose for orchestration
- Nginx load balancer
- PostgreSQL and Redis persistence
- Health checks and monitoring
- Security hardening
- Backup and recovery procedures