# Monitoring và Metrics với Prometheus/Grafana

## Tổng quan
Hệ thống monitoring sử dụng Prometheus để thu thập metrics và Grafana để visualization, đảm bảo observability và performance monitoring của TTS API.

## Prometheus Setup

### Installation và Configuration
```bash
# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64

# Configuration
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'tts-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
```

### Prometheus Metrics Endpoint
```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from flask import Response
import time
import psutil

# Custom registry
REGISTRY = CollectorRegistry()

# Counters
TTS_REQUESTS_TOTAL = Counter(
    'tts_requests_total',
    'Total number of TTS requests',
    ['status', 'voice_name', 'output_format'],
    registry=REGISTRY
)

TTS_REQUESTS_FAILED = Counter(
    'tts_requests_failed_total',
    'Total number of failed TTS requests',
    ['error_type', 'voice_name'],
    registry=REGISTRY
)

API_REQUESTS_TOTAL = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

# Histograms
REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY
)

TTS_PROCESSING_TIME = Histogram(
    'tts_processing_duration_seconds',
    'TTS processing time in seconds',
    ['voice_name', 'output_format'],
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY
)

# Gauges
ACTIVE_USERS = Gauge(
    'tts_active_users',
    'Number of active users',
    registry=REGISTRY
)

QUEUE_LENGTH = Gauge(
    'tts_queue_length',
    'Current TTS queue length',
    ['queue_name'],
    registry=REGISTRY
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=REGISTRY
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage',
    registry=REGISTRY
)

# Metrics collection functions
def update_system_metrics():
    """Update system metrics"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    SYSTEM_CPU_USAGE.set(cpu_percent)

    # Memory usage
    memory = psutil.virtual_memory()
    SYSTEM_MEMORY_USAGE.set(memory.percent)

def record_api_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record API request metrics"""
    API_REQUESTS_TOTAL.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    ).inc()

    REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)

def record_tts_request(voice_name: str, output_format: str, success: bool, duration: float = None):
    """Record TTS request metrics"""
    status = 'success' if success else 'failed'

    TTS_REQUESTS_TOTAL.labels(
        status=status,
        voice_name=voice_name,
        output_format=output_format
    ).inc()

    if duration:
        TTS_PROCESSING_TIME.labels(
            voice_name=voice_name,
            output_format=output_format
        ).observe(duration)

def record_tts_failure(error_type: str, voice_name: str):
    """Record TTS failure metrics"""
    TTS_REQUESTS_FAILED.labels(
        error_type=error_type,
        voice_name=voice_name
    ).inc()

def update_queue_metrics(queue_lengths: dict):
    """Update queue length metrics"""
    for queue_name, length in queue_lengths.items():
        QUEUE_LENGTH.labels(queue_name=queue_name).set(length)

def update_active_users(count: int):
    """Update active users gauge"""
    ACTIVE_USERS.set(count)

# Flask endpoint for metrics
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    update_system_metrics()
    return Response(generate_latest(REGISTRY), mimetype='text/plain; charset=utf-8')
```

### Metrics Middleware
```python
# app/middleware.py
import time
from flask import request, g
from app.metrics import record_api_request, REQUEST_ID

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        start_time = time.time()

        def new_start_response(status, response_headers, exc_info=None):
            duration = time.time() - start_time
            status_code = status.split()[0]

            # Record metrics
            record_api_request(
                method=request.method,
                endpoint=request.path,
                status_code=status_code,
                duration=duration
            )

            return start_response(status, response_headers, exc_info)

        return self.app(environ, new_start_response)
```

## Grafana Setup

### Installation
```bash
# Install Grafana
wget https://dl.grafana.com/oss/release/grafana-10.0.0.linux-amd64.tar.gz
tar -zxvf grafana-10.0.0.linux-amd64.tar.gz

# Start Grafana
cd grafana-10.0.0
./bin/grafana-server
```

### Data Source Configuration
```json
{
  "name": "Prometheus",
  "type": "prometheus",
  "url": "http://localhost:9090",
  "access": "proxy",
  "isDefault": true
}
```

### Dashboard Configuration

#### 1. System Overview Dashboard
```json
{
  "dashboard": {
    "title": "TTS API - System Overview",
    "tags": ["tts", "api", "system"],
    "timezone": "utc",
    "panels": [
      {
        "title": "System CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "system_cpu_usage_percent",
            "legendFormat": "CPU Usage %"
          }
        ]
      },
      {
        "title": "System Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "system_memory_usage_percent",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "TTS Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(tts_requests_total[5m])",
            "legendFormat": "{{status}} - {{voice_name}}"
          }
        ]
      }
    ]
  }
}
```

#### 2. TTS Performance Dashboard
```json
{
  "dashboard": {
    "title": "TTS API - Performance",
    "tags": ["tts", "performance"],
    "panels": [
      {
        "title": "TTS Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(tts_processing_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile - {{voice_name}}"
          }
        ]
      },
      {
        "title": "Queue Length",
        "type": "graph",
        "targets": [
          {
            "expr": "tts_queue_length",
            "legendFormat": "{{queue_name}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(tts_requests_failed_total[5m]) / rate(tts_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Voice Usage Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "tts_requests_total",
            "legendFormat": "{{voice_name}}"
          }
        ]
      }
    ]
  }
}
```

#### 3. User Analytics Dashboard
```json
{
  "dashboard": {
    "title": "TTS API - User Analytics",
    "tags": ["tts", "users"],
    "panels": [
      {
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "tts_active_users",
            "legendFormat": "Active Users"
          }
        ]
      },
      {
        "title": "Daily Active Users",
        "type": "graph",
        "targets": [
          {
            "expr": "count(count(api_requests_total) by (user_id))",
            "legendFormat": "Daily Active Users"
          }
        ]
      },
      {
        "title": "User Request Patterns",
        "type": "heatmap",
        "targets": [
          {
            "expr": "sum(rate(api_requests_total[1h])) by (hour)",
            "legendFormat": "Requests per Hour"
          }
        ]
      }
    ]
  }
}
```

## Alerting Rules

### Prometheus Alert Rules
```yaml
# alert_rules.yml
groups:
  - name: tts_api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(tts_requests_failed_total[5m]) / rate(tts_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}% for the last 5 minutes"

      - alert: HighQueueLength
        expr: tts_queue_length > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High queue length"
          description: "Queue length is {{ $value }}"

      - alert: SlowProcessingTime
        expr: histogram_quantile(0.95, rate(tts_processing_duration_seconds_bucket[10m])) > 60
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow TTS processing"
          description: "95th percentile processing time is {{ $value }}s"

      - alert: HighCPUUsage
        expr: system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"

      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "Service {{ $labels.job }} is down"
```

### Grafana Alerting
```json
{
  "name": "TTS API Alerts",
  "type": "prometheus",
  "isDefault": true,
  "url": "http://prometheus:9090",
  "alert": {
    "conditions": [
      {
        "type": "query",
        "query": {
          "params": ["A", "5m", "now"]
        },
        "reducer": {
          "type": "avg",
          "params": []
        },
        "evaluator": {
          "type": "gt",
          "params": [0.05]
        },
        "operator": {
          "type": "and"
        }
      }
    ],
    "frequency": "60s",
    "handler": 1,
    "name": "High Error Rate Alert",
    "message": "Error rate is above 5% for the last 5 minutes",
    "noDataState": "no_data",
    "notifications": [
      {
        "id": 1
      }
    ]
  }
}
```

## Application Metrics Collection

### Custom Metrics Collector
```python
# app/monitoring.py
import time
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Thread, Event
from app.metrics import (
    update_queue_metrics, update_active_users,
    record_tts_request, record_tts_failure
)

class MetricsCollector:
    def __init__(self, redis_client, db_session):
        self.redis_client = redis_client
        self.db_session = db_session
        self.stop_event = Event()
        self.collector_thread = None

    def start(self):
        """Start metrics collection"""
        self.collector_thread = Thread(target=self._collect_metrics, daemon=True)
        self.collector_thread.start()

    def stop(self):
        """Stop metrics collection"""
        self.stop_event.set()
        if self.collector_thread:
            self.collector_thread.join()

    def _collect_metrics(self):
        """Collect metrics periodically"""
        while not self.stop_event.is_set():
            try:
                self._collect_queue_metrics()
                self._collect_user_metrics()
                self._collect_performance_metrics()

                # Sleep for 30 seconds
                self.stop_event.wait(30)

            except Exception as e:
                print(f"Metrics collection error: {e}")
                self.stop_event.wait(60)  # Wait longer on error

    def _collect_queue_metrics(self):
        """Collect queue metrics"""
        queue_lengths = {
            'tts_queue': self.redis_client.llen('tts_queue'),
            'retry_queue': self.redis_client.llen('tts_retry_queue'),
            'processing': self.redis_client.scard('processing_requests')
        }

        update_queue_metrics(queue_lengths)

    def _collect_user_metrics(self):
        """Collect user activity metrics"""
        # Count active users in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        active_users = self.db_session.query(AudioRequest.user_id).filter(
            AudioRequest.created_at >= one_hour_ago
        ).distinct().count()

        update_active_users(active_users)

    def _collect_performance_metrics(self):
        """Collect performance metrics"""
        # Calculate success rate
        total_requests = self.db_session.query(AudioRequest).count()
        failed_requests = self.db_session.query(AudioRequest).filter(
            AudioRequest.status == 'failed'
        ).count()

        if total_requests > 0:
            success_rate = (total_requests - failed_requests) / total_requests
            # Store in custom metrics or send to external system
```

### Business Metrics
```python
# app/business_metrics.py
from sqlalchemy import func, and_
from datetime import datetime, timedelta

class BusinessMetricsCollector:
    def __init__(self, db_session):
        self.db_session = db_session

    def get_daily_stats(self, date: datetime = None):
        """Get daily statistics"""
        if date is None:
            date = datetime.utcnow().date()

        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())

        stats = self.db_session.query(
            func.count(AudioRequest.id).label('total_requests'),
            func.sum(func.case((AudioRequest.status == 'completed', 1), else_=0)).label('completed'),
            func.sum(func.case((AudioRequest.status == 'failed', 1), else_=0)).label('failed'),
            func.avg(AudioRequest.processing_time).label('avg_processing_time'),
            func.sum(AudioFile.file_size).label('total_audio_size')
        ).filter(
            and_(
                AudioRequest.created_at >= start_of_day,
                AudioRequest.created_at <= end_of_day
            )
        ).first()

        return {
            'date': date.isoformat(),
            'total_requests': stats.total_requests or 0,
            'completed_requests': stats.completed or 0,
            'failed_requests': stats.failed or 0,
            'success_rate': (stats.completed / stats.total_requests * 100) if stats.total_requests else 0,
            'avg_processing_time': float(stats.avg_processing_time) if stats.avg_processing_time else 0,
            'total_audio_size': stats.total_audio_size or 0
        }

    def get_user_engagement_metrics(self):
        """Get user engagement metrics"""
        # Users active in last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)

        active_users = self.db_session.query(
            func.count(func.distinct(AudioRequest.user_id))
        ).filter(
            AudioRequest.created_at >= seven_days_ago
        ).scalar()

        # New users today
        today = datetime.utcnow().date()
        new_users = self.db_session.query(
            func.count(User.id)
        ).filter(
            func.date(User.created_at) == today
        ).scalar()

        return {
            'active_users_7d': active_users,
            'new_users_today': new_users,
            'engagement_rate': active_users / max(new_users, 1) * 100
        }

    def get_voice_usage_stats(self):
        """Get voice usage statistics"""
        voice_stats = self.db_session.query(
            AudioRequest.voice_name,
            func.count(AudioRequest.id).label('request_count'),
            func.avg(AudioRequest.processing_time).label('avg_time')
        ).group_by(AudioRequest.voice_name).all()

        return {
            voice.voice_name: {
                'request_count': voice.request_count,
                'avg_processing_time': float(voice.avg_time) if voice.avg_time else 0
            }
            for voice in voice_stats
        }
```

## Log Aggregation với ELK Stack

### Elasticsearch Configuration
```json
{
  "index": "tts-api-logs",
  "mappings": {
    "properties": {
      "@timestamp": {"type": "date"},
      "level": {"type": "keyword"},
      "message": {"type": "text"},
      "request_id": {"type": "keyword"},
      "user_id": {"type": "keyword"},
      "source": {"type": "keyword"},
      "metadata": {"type": "object"},
      "host": {"type": "keyword"}
    }
  }
}
```

### Logstash Configuration
```json
input {
  file {
    path => "/app/logs/*.json"
    codec => "json"
  }
}

filter {
  mutate {
    add_field => {
      "index_name" => "tts-api-logs"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{index_name}"
  }
}
```

### Kibana Dashboard
```json
{
  "dashboard": {
    "title": "TTS API - Error Analysis",
    "panels": [
      {
        "title": "Error Rate Over Time",
        "type": "line",
        "query": "level:ERROR OR level:CRITICAL",
        "timeField": "@timestamp"
      },
      {
        "title": "Top Error Messages",
        "type": "table",
        "query": "level:ERROR",
        "aggs": {
          "terms": {"field": "message.keyword", "size": 10}
        }
      },
      {
        "title": "Request Failures by Voice",
        "type": "pie",
        "query": "level:ERROR AND metadata.error_type:voice_error",
        "aggs": {
          "terms": {"field": "metadata.voice_name.keyword"}
        }
      }
    ]
  }
}
```

## Health Checks

### Application Health Check
```python
# app/health.py
from flask import jsonify
from app.monitoring import HealthChecker

def health_check():
    """Application health check endpoint"""
    checker = HealthChecker(db_session, redis_client)
    health_status = checker.overall_health()

    status_code = 200 if health_status['status'] == 'healthy' else 503

    return jsonify(health_status), status_code

def detailed_health_check():
    """Detailed health check for monitoring systems"""
    checker = HealthChecker(db_session, redis_client)

    health_details = {
        'overall': checker.overall_health(),
        'database': checker.check_database(),
        'redis': checker.check_redis(),
        'gemini_api': checker.check_gemini_api(),
        'timestamp': datetime.utcnow().isoformat()
    }

    return jsonify(health_details)
```

### Kubernetes Health Checks
```yaml
# k8s health checks
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/detailed
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Performance Monitoring

### APM Integration
```python
# app/apm.py
from elasticapm.contrib.flask import ElasticAPM
from elasticapm.handlers import LoggingHandler

def init_apm(app):
    """Initialize Application Performance Monitoring"""
    app.config['ELASTIC_APM'] = {
        'SERVICE_NAME': 'tts-api',
        'SERVER_URL': 'http://apm-server:8200',
        'ENVIRONMENT': app.config['ENV'],
        'DEBUG': app.config['DEBUG'],
    }

    apm = ElasticAPM(app)

    # Add logging handler
    handler = LoggingHandler(client=apm.client)
    handler.setLevel(logging.WARNING)
    app.logger.addHandler(handler)

    return apm
```

### Database Performance Monitoring
```python
# app/db_monitoring.py
import time
from sqlalchemy import event
from app.metrics import DB_CONNECTION_TIME, DB_QUERY_TIME

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', time.time())

@event.listens_for(Engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = time.time() - conn.info['query_start_time']

    DB_QUERY_TIME.observe(total_time)

    # Log slow queries
    if total_time > 1.0:  # Log queries slower than 1 second
        logger.warning(
            "Slow database query",
            query=statement[:100],  # First 100 chars
            duration=total_time,
            parameters=str(parameters)[:200]
        )

@event.listens_for(Engine, "connect")
def receive_connect(dbapi_connection, connection_record):
    """Monitor database connection time"""
    start_time = time.time()
    return

@event.listens_for(Engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Monitor connection pool usage"""
    checkout_time = time.time() - connection_record.info.get('checkout_start', time.time())
    DB_CONNECTION_TIME.observe(checkout_time)
```

## Configuration

### Environment Variables
```bash
# Prometheus
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8000

# Grafana
GRAFANA_URL=http://localhost:3000
GRAFANA_API_KEY=your_grafana_api_key

# Alerting
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_EMAIL_FROM=noreply@tts-api.com
ALERT_EMAIL_TO=admin@company.com

# APM
ELASTIC_APM_SERVER_URL=http://apm-server:8200
ELASTIC_APM_SERVICE_NAME=tts-api
ELASTIC_APM_ENVIRONMENT=production
```

### Docker Compose Services
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/var/lib/grafana/dashboards

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
```

## Best Practices

### 1. Metrics Collection
- Use appropriate metric types (counter, histogram, gauge)
- Include relevant labels for filtering
- Set appropriate bucket sizes for histograms
- Collect metrics at appropriate intervals

### 2. Alerting
- Set meaningful thresholds
- Use multiple severity levels
- Include context in alert messages
- Test alerts regularly

### 3. Dashboard Design
- Keep dashboards focused on specific areas
- Use appropriate visualization types
- Include time range selectors
- Add drill-down capabilities

### 4. Performance
- Minimize metrics collection overhead
- Use sampling for high-frequency events
- Implement proper indexing for log queries
- Monitor monitoring system performance

### 5. Security
- Secure Prometheus endpoints
- Use authentication for Grafana
- Encrypt sensitive data in logs
- Implement access controls for monitoring data

## Testing

### Metrics Tests
```python
# tests/test_metrics.py
import pytest
from unittest.mock import patch
from app.metrics import record_api_request, record_tts_request

def test_api_request_metrics():
    """Test API request metrics recording"""
    with patch('app.metrics.API_REQUESTS_TOTAL') as mock_counter, \
         patch('app.metrics.REQUEST_DURATION') as mock_histogram:

        record_api_request('POST', '/api/v1/tts/generate', '200', 0.5)

        mock_counter.labels.assert_called_with(
            method='POST',
            endpoint='/api/v1/tts/generate',
            status_code='200'
        )
        mock_counter.labels().inc.assert_called_once()

        mock_histogram.labels.assert_called_with(
            method='POST',
            endpoint='/api/v1/tts/generate'
        )
        mock_histogram.labels().observe.assert_called_with(0.5)

def test_tts_request_metrics():
    """Test TTS request metrics recording"""
    with patch('app.metrics.TTS_REQUESTS_TOTAL') as mock_counter, \
         patch('app.metrics.TTS_PROCESSING_TIME') as mock_histogram:

        record_tts_request('Alnilam', 'wav', True, 15.5)

        mock_counter.labels.assert_called_with(
            status='success',
            voice_name='Alnilam',
            output_format='wav'
        )
        mock_counter.labels().inc.assert_called_once()

        mock_histogram.labels.assert_called_with(
            voice_name='Alnilam',
            output_format='wav'
        )
        mock_histogram.labels().observe.assert_called_with(15.5)
```

### Health Check Tests
```python
# tests/test_health.py
def test_health_check_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200

    data = response.get_json()
    assert data['status'] == 'healthy'
    assert 'database' in data['services']
    assert 'redis' in data['services']
    assert 'gemini_api' in data['services']

def test_detailed_health_check(client):
    """Test detailed health check"""
    response = client.get('/health/detailed')
    assert response.status_code == 200

    data = response.get_json()
    assert 'overall' in data
    assert 'database' in data
    assert 'redis' in data
    assert 'gemini_api' in data
```

## Troubleshooting

### Common Issues

#### High Cardinality
```python
# Avoid high cardinality labels
# Bad:
API_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, user_id=user_id, ip=ip)

# Good:
API_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status_code)
```

#### Memory Usage
```python
# Monitor Prometheus memory usage
# Set appropriate retention periods
# Use recording rules for complex queries
```

#### Alert Fatigue
```python
# Implement alert silencing
# Use maintenance windows
# Set appropriate alert thresholds
# Group related alerts
```

#### Performance Impact
```python
# Use async metrics collection
# Implement metrics sampling
# Monitor collection overhead
# Optimize query performance
```

## Deployment

### Production Setup
```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Configure reverse proxy
location /metrics {
    proxy_pass http://tts-api:8000;
    auth_basic "Prometheus";
    auth_basic_user_file /etc/nginx/.htpasswd;
}
```

### Scaling Considerations
```yaml
# Horizontal scaling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: tts-api
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Backup và Recovery
```bash
# Backup Prometheus data
tar czf prometheus_backup.tar.gz /prometheus/data/

# Backup Grafana dashboards
curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
     http://grafana:3000/api/dashboards/db > dashboards_backup.json
```

## Summary

Hệ thống monitoring và metrics được thiết kế để:

1. **Observability**: Theo dõi toàn bộ hệ thống với metrics chi tiết
2. **Performance**: Monitoring performance và bottlenecks
3. **Alerting**: Tự động cảnh báo khi có vấn đề
4. **Debugging**: Log aggregation và analysis
5. **Business Intelligence**: Metrics cho business decisions

Các thành phần chính:
- Prometheus: Thu thập và lưu trữ metrics
- Grafana: Visualization và dashboarding
- Alertmanager: Quản lý alerts
- ELK Stack: Log aggregation và search
- APM: Application performance monitoring