# Hướng dẫn Setup và Deployment

## Mục lục

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt Development](#cài-đặt-development)
3. [Cài đặt Production](#cài-đặt-production)
4. [Cấu hình Database](#cấu-hình-database)
5. [Cấu hình Redis](#cấu-hình-redis)
6. [Environment Variables](#environment-variables)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

## Yêu cầu hệ thống

### Minimum Requirements
- **OS**: Linux, macOS, hoặc Windows
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB free space
- **Network**: Internet connection cho API calls

### Software Requirements
- **Python**: 3.8+
- **PostgreSQL**: 13+ (production) hoặc SQLite (development)
- **Redis**: 6+ (cho caching)
- **Git**: 2.0+

## Cài đặt Development

### 1. Clone Repository

```bash
git clone <repository-url>
cd flask-tts-api
```

### 2. Tạo Virtual Environment

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Cài đặt Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### 4. Cấu hình Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env với thông tin của bạn
nano .env  # hoặc sử dụng editor khác
```

### 5. Database Setup

```bash
# Khởi tạo database migrations
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

# Hoặc với SQLite (development)
export DATABASE_URL=sqlite:///dev_tts_api.db
```

### 6. Chạy Application

```bash
# Development server
flask run

# Hoặc với Python
python app/main.py
```

### 7. Verify Installation

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Register test user
curl -X POST http://localhost:5000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPassword123"
  }'
```

## Cài đặt Production

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    postgresql \
    postgresql-contrib \
    redis-server \
    nginx \
    supervisor \
    curl \
    wget

# CentOS/RHEL
sudo yum install -y \
    python39 \
    python39-devel \
    postgresql-server \
    postgresql-contrib \
    redis \
    nginx \
    supervisor
```

### 2. Database Setup

```bash
# PostgreSQL setup
sudo -u postgres createuser --interactive --pwprompt tts_user
sudo -u postgres createdb -O tts_user tts_api

# Configure PostgreSQL
sudo nano /etc/postgresql/13/main/pg_hba.conf
# Thêm dòng: local   tts_api   tts_user   md5

sudo systemctl restart postgresql
```

### 3. Redis Setup

```bash
# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: supervised systemd
# Set: bind 127.0.0.1 ::1

sudo systemctl restart redis
```

### 4. Application Setup

```bash
# Tạo application user
sudo useradd -m -s /bin/bash tts_app
sudo -u tts_app mkdir -p /home/tts_app/flask-tts-api

# Copy application files
sudo cp -r . /home/tts_app/flask-tts-api/
sudo chown -R tts_app:tts_app /home/tts_app/flask-tts-api

# Setup virtual environment
sudo -u tts_app python3.9 -m venv /home/tts_app/flask-tts-api/venv
sudo -u tts_app /home/tts_app/flask-tts-api/venv/bin/pip install -r requirements.txt
```

### 5. Configuration

```bash
# Production environment file
sudo -u tts_app nano /home/tts_app/flask-tts-api/.env

# Production settings
FLASK_ENV=production
DEBUG=false
DATABASE_URL=postgresql://tts_user:password@localhost:5432/tts_api
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-production-secret-key
JWT_SECRET_KEY=your-production-jwt-secret
GEMINI_API_KEY=your-gemini-api-key
```

### 6. Database Migration

```bash
# Run migrations
sudo -u tts_app /home/tts_app/flask-tts-api/venv/bin/flask db upgrade
```

### 7. Systemd Service

```bash
# Tạo service file
sudo nano /etc/systemd/system/flask-tts-api.service

# Nội dung service file:
[Unit]
Description=Flask TTS API
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=tts_app
Group=tts_app
WorkingDirectory=/home/tts_app/flask-tts-api
Environment=PATH=/home/tts_app/flask-tts-api/venv/bin
ExecStart=/home/tts_app/flask-tts-api/venv/bin/gunicorn \
    --bind 127.0.0.1:8000 \
    --workers 4 \
    --worker-class gevent \
    --log-level info \
    --access-logfile /home/tts_app/flask-tts-api/logs/access.log \
    --error-logfile /home/tts_app/flask-tts-api/logs/error.log \
    app.main:app

[Install]
WantedBy=multi-user.target

# Enable và start service
sudo systemctl daemon-reload
sudo systemctl enable flask-tts-api
sudo systemctl start flask-tts-api
```

### 8. Nginx Configuration

```bash
# Tạo nginx config
sudo nano /etc/nginx/sites-available/flask-tts-api

# Nội dung config:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    }

    # Static files
    location /static {
        alias /home/tts_app/flask-tts-api/app/static;
    }

    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8000/api/v1/health;
        access_log off;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/flask-tts-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 9. SSL Certificate (Optional)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d your-domain.com

# Auto renewal
sudo crontab -e
# Thêm dòng: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Cấu hình Database

### PostgreSQL Production Settings

```sql
-- Tối ưu hóa PostgreSQL cho production
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = '0.9';
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = '100';
ALTER SYSTEM SET random_page_cost = '1.1';
ALTER SYSTEM SET effective_io_concurrency = '200';
ALTER SYSTEM SET work_mem = '6553kB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';

-- Restart PostgreSQL
sudo systemctl restart postgresql
```

### Database Backup

```bash
# Tạo backup script
sudo nano /home/tts_app/backup-db.sh

#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U tts_user -d tts_api > /home/tts_app/backups/db_backup_$DATE.sql
find /home/tts_app/backups -name "db_backup_*.sql" -mtime +7 -delete

# Make executable
sudo chmod +x /home/tts_app/backup-db.sh

# Add to crontab
sudo crontab -e
# Thêm dòng: 0 2 * * * /home/tts_app/backup-db.sh
```

## Cấu hình Redis

### Redis Production Settings

```bash
# Edit redis config
sudo nano /etc/redis/redis.conf

# Important settings:
supervised systemd
bind 127.0.0.1 ::1
port 6379
timeout 0
tcp-keepalive 300
daemonize no
supervised no
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis
slave-serve-stale-data yes
slave-read-only no
repl-diskless-sync no
repl-diskless-sync-delay 5
repl-disable-tcp-nodelay no
slave-priority 100
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
lua-time-limit 5000
slowlog-log-slower-than 10000
slowlog-max-len 128
latency-monitor-threshold 0
notify-keyspace-events ""
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
activerehashing yes
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit slave 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
hz 10
aof-rewrite-incremental-fsync yes

# Restart Redis
sudo systemctl restart redis
```

## Environment Variables

### Production Environment Template

```env
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-super-secret-production-key
DEBUG=false
TESTING=false

# Database
DATABASE_URL=postgresql://tts_user:secure_password@localhost:5432/tts_api

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRES=1800
JWT_REFRESH_TOKEN_EXPIRES=604800

# API Keys
GEMINI_API_KEY=your-gemini-api-key

# Audio Settings
MAX_AUDIO_FILE_SIZE=10485760
SUPPORTED_AUDIO_FORMATS=mp3,wav,ogg,flac
DEFAULT_VOICE_NAME=Alnilam
MAX_TEXT_LENGTH=5000

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PREMIUM_PER_MINUTE=1000

# File Storage
UPLOAD_FOLDER=/home/tts_app/flask-tts-api/uploads/audio
MAX_CONTENT_LENGTH=16777216

# Logging
LOG_LEVEL=INFO
LOG_FILE=/home/tts_app/flask-tts-api/logs/tts_api.log
LOG_MAX_SIZE=10485760
LOG_BACKUP_COUNT=5

# CORS
CORS_ORIGINS=https://yourdomain.com

# Monitoring
ENABLE_MONITORING=true
```

## Deployment

### Docker Deployment

```bash
# Build và deploy với Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale application
docker-compose -f docker-compose.prod.yml up -d --scale app=3

# Update application
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Deployment

```bash
# Pull latest changes
cd /home/tts_app/flask-tts-api
sudo -u tts_app git pull origin main

# Install dependencies
sudo -u tts_app /home/tts_app/flask-tts-api/venv/bin/pip install -r requirements.txt

# Run migrations
sudo -u tts_app /home/tts_app/flask-tts-api/venv/bin/flask db upgrade

# Restart service
sudo systemctl restart flask-tts-api
```

### Blue-Green Deployment

```bash
# Setup blue-green deployment
# 1. Deploy to green environment
# 2. Test green environment
# 3. Switch traffic to green
# 4. Remove blue environment
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# Test connection
psql -h localhost -U tts_user -d tts_api
```

#### 2. Redis Connection Issues

```bash
# Check Redis status
sudo systemctl status redis

# Test connection
redis-cli ping

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log
```

#### 3. Application Issues

```bash
# Check application logs
sudo tail -f /home/tts_app/flask-tts-api/logs/error.log

# Check system logs
sudo journalctl -u flask-tts-api -f

# Test health endpoint
curl http://localhost:8000/api/v1/health
```

#### 4. Memory Issues

```bash
# Check memory usage
free -h

# Check application memory
ps aux | grep gunicorn

# Monitor Redis memory
redis-cli info memory
```

#### 5. Performance Issues

```bash
# Check database performance
psql -h localhost -U tts_user -d tts_api -c "SELECT * FROM pg_stat_activity;"

# Check Redis performance
redis-cli info stats

# Monitor system resources
htop
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug
flask run --debug

# Check for syntax errors
python -m py_compile app/main.py
```

### Backup và Recovery

```bash
# Database backup
pg_dump -h localhost -U tts_user -d tts_api > backup.sql

# Database restore
psql -h localhost -U tts_user -d tts_api < backup.sql

# File system backup
tar -czf backup_$(date +%Y%m%d).tar.gz /home/tts_app/flask-tts-api/
```

### Security Checklist

- [ ] Change default passwords
- [ ] Configure firewall
- [ ] Setup SSL/TLS
- [ ] Configure fail2ban
- [ ] Regular security updates
- [ ] Monitor logs
- [ ] Backup strategy
- [ ] Access control

## Monitoring và Maintenance

### Health Checks

```bash
# Application health
curl http://localhost:8000/api/v1/health

# Database health
psql -h localhost -U tts_user -d tts_api -c "SELECT 1;"

# Redis health
redis-cli ping
```

### Log Rotation

```bash
# Configure logrotate
sudo nano /etc/logrotate.d/flask-tts-api

/home/tts_app/flask-tts-api/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 tts_app tts_app
}
```

### Regular Maintenance

```bash
# Clean old audio files
find /home/tts_app/flask-tts-api/uploads -type f -mtime +30 -delete

# Clean old logs
find /home/tts_app/flask-tts-api/logs -name "*.log.*" -mtime +30 -delete

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Restart services weekly
sudo systemctl restart flask-tts-api redis postgresql
```

---

**Chúc bạn setup thành công! 🎵**