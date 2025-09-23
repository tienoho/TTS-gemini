# 🚀 TTS System with Business Intelligence

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Flask Version](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/database-postgresql-blue.svg)](https://postgresql.org)

**Advanced Text-to-Speech API with Enterprise-Grade Business Intelligence**

A production-ready Flask-based TTS (Text-to-Speech) system powered by Google Gemini AI, featuring comprehensive Business Intelligence capabilities, multi-tenancy support, advanced analytics, and enterprise-grade security.

## 🌟 Key Features

### 🎵 Core TTS Features
- **Google Gemini AI Integration** - High-quality text-to-speech conversion
- **Voice Cloning** - Custom voice generation and management
- **Audio Enhancement** - Real-time audio quality improvement
- **Batch Processing** - Bulk TTS request processing
- **Multi-format Support** - MP3, WAV, OGG, FLAC output formats
- **Real-time Processing** - WebSocket support for live audio streaming

### 📊 Advanced Business Intelligence
- **Revenue Analytics** - Comprehensive financial tracking and forecasting
- **Customer Segmentation** - Advanced customer behavior analysis
- **KPI Dashboard** - Real-time key performance indicators
- **Usage Pattern Analysis** - AI-powered usage insights
- **Anomaly Detection** - Automated system anomaly identification
- **Predictive Analytics** - Revenue and demand forecasting
- **Custom Reporting** - Flexible report generation and scheduling
- **Business Insights** - AI-generated actionable recommendations

### 🔐 Enterprise Security
- **JWT Authentication** - Secure token-based authentication
- **Multi-tenancy** - Organization-based data isolation
- **Rate Limiting** - Intelligent request throttling
- **Input Sanitization** - Comprehensive data validation
- **SQL Injection Prevention** - Database security measures
- **CORS Protection** - Cross-origin request handling
- **File Upload Security** - Secure file handling with integrity checks

### 🏗️ Technical Excellence
- **Docker Ready** - Complete containerization support
- **Database Flexibility** - PostgreSQL/SQLite support
- **Redis Caching** - High-performance data caching
- **Async Processing** - Background job processing
- **Plugin System** - Extensible architecture
- **Comprehensive Testing** - Full test coverage
- **API Documentation** - Auto-generated documentation
- **Monitoring & Logging** - Advanced system monitoring

## 📋 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   Load Balancer │    │   API Gateway   │
│   (React/Angular)│    │   (Nginx)       │    │   (Flask)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                    │
┌───────────────────────────────────────────────────┼───────────────────────────────────────────────────┐
│                    Application Layer              │              Business Logic Layer              │
│                                                   │                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Auth      │  │   TTS       │  │   BI        │ │  │   Revenue   │  │   Customer  │  │   Analytics │ │
│  │  Service    │  │  Service    │  │  Service    │ │  │  Manager    │  │  Manager    │  │  Engine     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │  └─────────────┘  └─────────────┘  └─────────────┘ │
└───────────────────────────────────────────────────┼───────────────────────────────────────────────────┘
                                                    │
┌───────────────────────────────────────────────────┼───────────────────────────────────────────────────┐
│                   Infrastructure Layer            │              Data Storage Layer                │
│                                                   │                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Redis     │  │   Celery    │  │   File      │ │  │ PostgreSQL  │  │   Redis     │  │   File      │ │
│  │   Cache     │  │   Workers   │  │   Storage   │ │  │   Database  │  │   Cache     │  │   Storage   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │  └─────────────┘  └─────────────┘  └─────────────┘ │
└───────────────────────────────────────────────────┴───────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### System Requirements

- **Python 3.8+**
- **Redis** (for caching and sessions)
- **PostgreSQL** (production) or **SQLite** (development)
- **Docker & Docker Compose** (recommended)

### 1. Clone Repository

```bash
git clone <repository-url>
cd tts-gemini
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

### 4. Database Setup

```bash
# Initialize database
flask db init
flask db migrate
flask db upgrade

# Seed initial data (optional)
flask seed-db
```

### 5. Start Services

```bash
# Start Redis (if not using Docker)
redis-server

# Start application
flask run

# Or with Python
python app/main.py
```

### 6. Verify Installation

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Should return: {"status": "healthy", "version": "1.0.0"}
```

## 📚 API Documentation

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/register` | User registration |
| `POST` | `/api/v1/auth/login` | User authentication |
| `POST` | `/api/v1/auth/refresh` | Token refresh |
| `GET` | `/api/v1/auth/profile` | Get user profile |
| `PUT` | `/api/v1/auth/profile` | Update user profile |
| `POST` | `/api/v1/auth/api-key` | Regenerate API key |
| `POST` | `/api/v1/auth/logout` | User logout |

### Text-to-Speech Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/tts/generate` | Generate audio from text |
| `GET` | `/api/v1/tts/` | List TTS requests (paginated) |
| `GET` | `/api/v1/tts/{id}` | Get TTS request details |
| `GET` | `/api/v1/tts/{id}/download` | Download audio file |
| `DELETE` | `/api/v1/tts/{id}` | Delete TTS request |
| `GET` | `/api/v1/tts/stats` | User statistics |

### Business Intelligence Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/bi/revenue` | Revenue analytics |
| `GET` | `/api/v1/bi/customers` | Customer analytics |
| `GET` | `/api/v1/bi/usage` | Usage analytics |
| `GET` | `/api/v1/bi/kpis` | KPI dashboard |
| `POST` | `/api/v1/bi/reports` | Generate custom reports |
| `GET` | `/api/v1/bi/insights` | AI business insights |
| `GET` | `/api/v1/bi/forecasting` | Financial forecasting |

### Advanced Features Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/batch/tts` | Batch TTS processing |
| `GET` | `/api/v1/voice-cloning/list` | Voice cloning management |
| `POST` | `/api/v1/audio-enhancement` | Audio quality enhancement |
| `GET` | `/api/v1/integrations` | Third-party integrations |
| `GET` | `/api/v1/webhooks` | Webhook management |

## 🔧 Configuration

### Environment Variables

```env
# Flask Configuration
FLASK_APP=app.main:create_app
FLASK_ENV=development
SECRET_KEY=your-super-secret-key-here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/tts_db
# Alternative for development
DATABASE_URL=sqlite:///tts_db.sqlite

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRES=3600
JWT_REFRESH_TOKEN_EXPIRES=86400

# Google Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# Audio Configuration
MAX_AUDIO_FILE_SIZE=10485760
SUPPORTED_AUDIO_FORMATS=mp3,wav,ogg,flac
DEFAULT_VOICE_NAME=Alnilam
MAX_TEXT_LENGTH=5000

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PREMIUM_PER_MINUTE=1000

# File Storage
UPLOAD_FOLDER=uploads/audio
MAX_CONTENT_LENGTH=16777216

# Business Intelligence
BI_CACHE_TTL=3600
BI_MAX_FORECAST_MONTHS=24
BI_ANOMALY_DETECTION_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/tts_api.log

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

## 📊 Business Intelligence Features

### Revenue Analytics
- **Real-time Revenue Tracking** - Monitor revenue streams in real-time
- **Financial Forecasting** - AI-powered revenue predictions
- **Revenue Attribution** - Track revenue sources and channels
- **Profit Margin Analysis** - Comprehensive profitability insights

### Customer Analytics
- **Customer Segmentation** - Advanced customer grouping and analysis
- **Churn Prediction** - Identify customers at risk of leaving
- **Cohort Analysis** - Track customer behavior over time
- **Lifetime Value Calculation** - Customer value prediction

### Usage Analytics
- **Pattern Recognition** - AI-powered usage pattern detection
- **Anomaly Detection** - Automated system anomaly identification
- **Performance Monitoring** - Real-time system performance tracking
- **Resource Optimization** - Usage-based resource recommendations

### KPI Management
- **Custom KPI Definition** - Define organization-specific KPIs
- **Real-time Dashboard** - Live KPI monitoring dashboard
- **Performance Tracking** - Historical KPI performance analysis
- **Alert System** - Automated KPI threshold alerts

## 🎵 Text-to-Speech Features

### Core TTS Capabilities
- **Multiple Voice Support** - Various voice options and styles
- **Language Support** - Multi-language text processing
- **Audio Quality Control** - Adjustable audio quality settings
- **Real-time Processing** - Low-latency audio generation

### Voice Cloning
- **Custom Voice Creation** - Create personalized voices
- **Voice Library Management** - Organize and manage voice assets
- **Voice Quality Analysis** - Automated voice quality assessment
- **Voice Training** - Continuous voice model improvement

### Audio Enhancement
- **Real-time Enhancement** - Live audio quality improvement
- **Noise Reduction** - Advanced noise filtering
- **Audio Normalization** - Consistent audio levels
- **Format Optimization** - Optimal audio format selection

## 🧪 Testing

### Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test modules
pytest tests/test_auth.py -v
pytest tests/test_tts.py -v
pytest tests/test_bi_service.py -v

# Run Business Intelligence tests
pytest tests/run_bi_tests.py -v

# Performance testing
pytest tests/test_batch_performance.py -v
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=app --cov-report=term-missing --cov-report=html

# Coverage for specific modules
pytest --cov=utils.bi_service --cov-report=term-missing
```

## 🐳 Docker Deployment

### Development Environment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild specific service
docker-compose build tts-api
docker-compose up -d tts-api
```

### Production Deployment

```bash
# Build production image
docker build -t tts-gemini:latest .

# Run with production configuration
docker run -d \
  --name tts-gemini \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your-api-key \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  --restart unless-stopped \
  tts-gemini:latest

# Or use production docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  tts-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/tts_db
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=tts_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password

  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
```

## 📈 Monitoring & Analytics

### System Monitoring
- **Health Checks** - Automated system health verification
- **Performance Metrics** - Real-time performance tracking
- **Error Monitoring** - Comprehensive error tracking
- **Resource Usage** - CPU, memory, and disk monitoring

### Business Intelligence Dashboard
- **Revenue Dashboard** - Financial performance visualization
- **Customer Dashboard** - Customer behavior analytics
- **Usage Dashboard** - System usage patterns
- **KPI Dashboard** - Key performance indicators

### Alert System
- **Revenue Alerts** - Revenue threshold notifications
- **System Alerts** - System performance alerts
- **Customer Alerts** - Customer behavior alerts
- **Custom Alerts** - User-defined alert conditions

## 🔒 Security Features

### Authentication & Authorization
- **JWT Tokens** - Secure token-based authentication
- **Role-based Access Control** - Granular permission system
- **API Key Management** - Secure API key handling
- **Session Management** - Secure session handling

### Data Protection
- **Input Validation** - Comprehensive input sanitization
- **SQL Injection Prevention** - Database security measures
- **XSS Protection** - Cross-site scripting prevention
- **CSRF Protection** - Cross-site request forgery prevention

### Network Security
- **CORS Configuration** - Secure cross-origin policies
- **Rate Limiting** - Request throttling and abuse prevention
- **IP Whitelisting** - Network access control
- **SSL/TLS** - Secure communication protocols

## 🚀 Production Deployment

### Deployment Checklist

- [ ] **Environment Setup** - Configure production environment
- [ ] **Database Migration** - Run database migrations
- [ ] **SSL Configuration** - Enable HTTPS
- [ ] **Load Balancer** - Configure load balancing
- [ ] **Monitoring Setup** - Implement monitoring systems
- [ ] **Backup Strategy** - Configure data backups
- [ ] **Security Hardening** - Apply security measures
- [ ] **Performance Tuning** - Optimize system performance

### Production Configuration

```bash
# Production environment variables
export FLASK_ENV=production
export SECRET_KEY=your-production-secret-key
export DATABASE_URL=postgresql://prod_user:prod_pass@prod_db:5432/tts_prod
export REDIS_URL=redis://prod_redis:6379/0
export GEMINI_API_KEY=your-production-gemini-key

# Start with gunicorn
gunicorn --bind 0.0.0.0:5000 \
  --workers 4 \
  --worker-class gevent \
  --worker-connections 1000 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --log-level info \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  app.main:app
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd tts-gemini

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run tests
pytest

# Code formatting
black app/ tests/
isort app/ tests/

# Type checking
mypy app/

# Security scanning
bandit -r app/
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Create** a Pull Request

### Code Standards

- Follow **PEP 8** style guidelines
- Write comprehensive **docstrings**
- Include **type hints** for all functions
- Add **unit tests** for new features
- Update **documentation** as needed

## 📚 Additional Documentation

- [API Documentation](docs/api_docs.md) - Complete API reference
- [Setup Guide](docs/setup_guide.md) - Detailed setup instructions
- [Database Schema](docs/database_schema.md) - Database structure
- [Security Guide](security_audit_report.md) - Security considerations
- [Performance Guide](docs/performance_optimization.md) - Performance tuning

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

## 🙏 Acknowledgments

- **[Flask](https://flask.palletsprojects.com/)** - Web framework
- **[Google Gemini AI](https://ai.google.dev/)** - TTS engine
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - Database ORM
- **[Redis](https://redis.io/)** - Caching and sessions
- **[PostgreSQL](https://postgresql.org/)** - Primary database
- **[Docker](https://docker.com/)** - Containerization

## 📞 Support

### Getting Help

1. **Check Issues** - Browse existing [GitHub Issues](../../issues)
2. **Create Issue** - Report bugs or request features
3. **Documentation** - Review detailed documentation
4. **Community** - Join our developer community

### Contact Information

- **Email**: support@tts-gemini.com
- **Documentation**: [docs.tts-gemini.com](https://docs.tts-gemini.com)
- **API Status**: [status.tts-gemini.com](https://status.tts-gemini.com)

---

**🎵 Happy coding with TTS-Gemini!**

*Built with ❤️ by the TTS-Gemini team*