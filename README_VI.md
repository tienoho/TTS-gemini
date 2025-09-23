# 🚀 Hệ Thống TTS với Trí Tuệ Kinh Doanh

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Flask Version](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/database-postgresql-blue.svg)](https://postgresql.org)

**API Text-to-Speech Tiên Tiến với Trí Tuệ Kinh Doanh Doanh Nghiệp**

Hệ thống TTS (Text-to-Speech) dựa trên Flask, được xây dựng sẵn sàng cho production, sử dụng Google Gemini AI, với khả năng Trí Tuệ Kinh Doanh toàn diện, hỗ trợ đa tổ chức, phân tích nâng cao và bảo mật cấp doanh nghiệp.

## 🌟 Tính Năng Chính

### 🎵 Tính Năng TTS Cốt Lõi
- **Tích Hợp Google Gemini AI** - Chuyển đổi text-to-speech chất lượng cao
- **Sao Chép Giọng Nói** - Tạo và quản lý giọng nói tùy chỉnh
- **Cải Thiện Âm Thanh** - Cải thiện chất lượng âm thanh thời gian thực
- **Xử Lý Hàng Loạt** - Xử lý yêu cầu TTS số lượng lớn
- **Hỗ Trợ Đa Định Dạng** - Đầu ra MP3, WAV, OGG, FLAC
- **Xử Lý Thời Gian Thực** - Hỗ trợ WebSocket cho streaming âm thanh trực tiếp

### 📊 Trí Tuệ Kinh Doanh Nâng Cao
- **Phân Tích Doanh Thu** - Theo dõi và dự báo tài chính toàn diện
- **Phân Đoạn Khách Hàng** - Phân tích hành vi khách hàng nâng cao
- **Bảng Điều Khiển KPI** - Chỉ số hiệu suất chính thời gian thực
- **Phân Tích Mẫu Sử Dụng** - Thông tin chi tiết sử dụng được hỗ trợ bởi AI
- **Phát Hiện Bất Thường** - Xác định bất thường hệ thống tự động
- **Phân Tích Dự Đoán** - Dự báo doanh thu và nhu cầu
- **Báo Cáo Tùy Chỉnh** - Tạo và lập lịch báo cáo linh hoạt
- **Thông Tin Chi Tiết Kinh Doanh** - Đề xuất hành động được tạo bởi AI

### 🔐 Bảo Mật Doanh Nghiệp
- **Xác Thực JWT** - Xác thực dựa trên token bảo mật
- **Đa Tổ Chức** - Cách ly dữ liệu dựa trên tổ chức
- **Giới Hạn Tỷ Lệ** - Kiểm soát yêu cầu thông minh
- **Làm Sạch Dữ Liệu Đầu Vào** - Xác thực dữ liệu toàn diện
- **Ngăn Chặn SQL Injection** - Biện pháp bảo mật cơ sở dữ liệu
- **Bảo Vệ CORS** - Xử lý yêu cầu cross-origin
- **Bảo Mật Upload File** - Xử lý file bảo mật với kiểm tra tính toàn vẹn

### 🏗️ Xuất Sắc Về Kỹ Thuật
- **Sẵn Sàng Docker** - Hỗ trợ containerization hoàn chỉnh
- **Tính Linh Hoạt Cơ Sở Dữ Liệu** - Hỗ trợ PostgreSQL/SQLite
- **Bộ Nhớ Cache Redis** - Bộ nhớ đệm dữ liệu hiệu suất cao
- **Xử Lý Bất Đồng Bộ** - Xử lý công việc nền
- **Hệ Thống Plugin** - Kiến trúc có thể mở rộng
- **Testing Toàn Diện** - Độ bao phủ test đầy đủ
- **Tài Liệu API** - Tài liệu tự động tạo
- **Giám Sát & Ghi Log** - Giám sát hệ thống nâng cao

## 📋 Kiến Trúc Hệ Thống

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

## 🚀 Bắt Đầu Nhanh

### Yêu Cầu Hệ Thống

- **Python 3.8+**
- **Redis** (để caching và sessions)
- **PostgreSQL** (production) hoặc **SQLite** (development)
- **Docker & Docker Compose** (khuyến nghị)

### 1. Clone Repository

```bash
git clone <repository-url>
cd tts-gemini
```

### 2. Thiết Lập Môi Trường

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Cấu Hình

```bash
# Copy template môi trường
cp .env.example .env

# Chỉnh sửa cấu hình
nano .env  # hoặc editor bạn thích
```

### 4. Thiết Lập Cơ Sở Dữ Liệu

```bash
# Khởi tạo cơ sở dữ liệu
flask db init
flask db migrate
flask db upgrade

# Seed dữ liệu ban đầu (tùy chọn)
flask seed-db
```

### 5. Khởi Động Dịch Vụ

```bash
# Khởi động Redis (nếu không sử dụng Docker)
redis-server

# Khởi động ứng dụng
flask run

# Hoặc với Python
python app/main.py
```

### 6. Xác Minh Cài Đặt

```bash
# Kiểm tra health
curl http://localhost:5000/api/v1/health

# Sẽ trả về: {"status": "healthy", "version": "1.0.0"}
```

## 📚 Tài Liệu API

### Endpoints Xác Thực

| Phương Thức | Endpoint | Mô Tả |
|-------------|----------|--------|
| `POST` | `/api/v1/auth/register` | Đăng ký người dùng |
| `POST` | `/api/v1/auth/login` | Xác thực người dùng |
| `POST` | `/api/v1/auth/refresh` | Làm mới token |
| `GET` | `/api/v1/auth/profile` | Lấy thông tin người dùng |
| `PUT` | `/api/v1/auth/profile` | Cập nhật thông tin người dùng |
| `POST` | `/api/v1/auth/api-key` | Tạo lại API key |
| `POST` | `/api/v1/auth/logout` | Đăng xuất người dùng |

### Endpoints Text-to-Speech

| Phương Thức | Endpoint | Mô Tả |
|-------------|----------|--------|
| `POST` | `/api/v1/tts/generate` | Tạo âm thanh từ text |
| `GET` | `/api/v1/tts/` | Liệt kê yêu cầu TTS (có phân trang) |
| `GET` | `/api/v1/tts/{id}` | Lấy chi tiết yêu cầu TTS |
| `GET` | `/api/v1/tts/{id}/download` | Tải xuống file âm thanh |
| `DELETE` | `/api/v1/tts/{id}` | Xóa yêu cầu TTS |
| `GET` | `/api/v1/tts/stats` | Thống kê người dùng |

### Endpoints Trí Tuệ Kinh Doanh

| Phương Thức | Endpoint | Mô Tả |
|-------------|----------|--------|
| `GET` | `/api/v1/bi/revenue` | Phân tích doanh thu |
| `GET` | `/api/v1/bi/customers` | Phân tích khách hàng |
| `GET` | `/api/v1/bi/usage` | Phân tích sử dụng |
| `GET` | `/api/v1/bi/kpis` | Bảng điều khiển KPI |
| `POST` | `/api/v1/bi/reports` | Tạo báo cáo tùy chỉnh |
| `GET` | `/api/v1/bi/insights` | Thông tin chi tiết kinh doanh AI |
| `GET` | `/api/v1/bi/forecasting` | Dự báo tài chính |

### Endpoints Tính Năng Nâng Cao

| Phương Thức | Endpoint | Mô Tả |
|-------------|----------|--------|
| `POST` | `/api/v1/batch/tts` | Xử lý TTS hàng loạt |
| `GET` | `/api/v1/voice-cloning/list` | Quản lý sao chép giọng nói |
| `POST` | `/api/v1/audio-enhancement` | Cải thiện chất lượng âm thanh |
| `GET` | `/api/v1/integrations` | Tích hợp bên thứ ba |
| `GET` | `/api/v1/webhooks` | Quản lý webhook |

## 🔧 Cấu Hình

### Biến Môi Trường

```env
# Cấu hình Flask
FLASK_APP=app.main:create_app
FLASK_ENV=development
SECRET_KEY=your-super-secret-key-here

# Cấu hình Cơ sở dữ liệu
DATABASE_URL=postgresql://user:password@localhost/tts_db
# Thay thế cho development
DATABASE_URL=sqlite:///tts_db.sqlite

# Cấu hình Redis
REDIS_URL=redis://localhost:6379/0

# Cấu hình JWT
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRES=3600
JWT_REFRESH_TOKEN_EXPIRES=86400

# Google Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# Cấu hình Âm thanh
MAX_AUDIO_FILE_SIZE=10485760
SUPPORTED_AUDIO_FORMATS=mp3,wav,ogg,flac
DEFAULT_VOICE_NAME=Alnilam
MAX_TEXT_LENGTH=5000

# Giới hạn Tỷ lệ
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PREMIUM_PER_MINUTE=1000

# Lưu trữ File
UPLOAD_FOLDER=uploads/audio
MAX_CONTENT_LENGTH=16777216

# Trí tuệ Kinh doanh
BI_CACHE_TTL=3600
BI_MAX_FORECAST_MONTHS=24
BI_ANOMALY_DETECTION_ENABLED=true

# Ghi log
LOG_LEVEL=INFO
LOG_FILE=logs/tts_api.log

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

## 📊 Tính Năng Trí Tuệ Kinh Doanh

### Phân Tích Doanh Thu
- **Theo Dõi Doanh Thu Thời Gian Thực** - Giám sát luồng doanh thu theo thời gian thực
- **Dự Báo Tài Chính** - Dự đoán doanh thu được hỗ trợ bởi AI
- **Phân Bổ Doanh Thu** - Theo dõi nguồn và kênh doanh thu
- **Phân Tích Biên Lợi Nhuận** - Thông tin chi tiết về lợi nhuận toàn diện

### Phân Tích Khách Hàng
- **Phân Đoạn Khách Hàng** - Nhóm và phân tích khách hàng nâng cao
- **Dự Đoán Rời Bỏ** - Xác định khách hàng có nguy cơ rời bỏ
- **Phân Tích Nhóm** - Theo dõi hành vi khách hàng theo thời gian
- **Tính Toán Giá Trị Vòng Đời** - Dự đoán giá trị khách hàng

### Phân Tích Sử Dụng
- **Nhận Dạng Mẫu** - Phát hiện mẫu sử dụng được hỗ trợ bởi AI
- **Phát Hiện Bất Thường** - Xác định bất thường hệ thống tự động
- **Giám Sát Hiệu Suất** - Theo dõi hiệu suất hệ thống thời gian thực
- **Tối Ưu Hóa Tài Nguyên** - Đề xuất tài nguyên dựa trên sử dụng

### Quản Lý KPI
- **Định Nghĩa KPI Tùy Chỉnh** - Xác định KPI cụ thể cho tổ chức
- **Bảng Điều Khiển Thời Gian Thực** - Giám sát KPI trực tiếp
- **Theo Dõi Hiệu Suất** - Phân tích hiệu suất KPI lịch sử
- **Hệ Thống Cảnh Báo** - Cảnh báo ngưỡng KPI tự động

## 🎵 Tính Năng Text-to-Speech

### Khả Năng TTS Cốt Lõi
- **Hỗ Trợ Nhiều Giọng Nói** - Nhiều tùy chọn và kiểu giọng nói
- **Hỗ Trợ Đa Ngôn Ngữ** - Xử lý text đa ngôn ngữ
- **Kiểm Soát Chất Lượng Âm Thanh** - Cài đặt chất lượng âm thanh có thể điều chỉnh
- **Xử Lý Thời Gian Thực** - Tạo âm thanh độ trễ thấp

### Sao Chép Giọng Nói
- **Tạo Giọng Nói Tùy Chỉnh** - Tạo giọng nói cá nhân hóa
- **Quản Lý Thư Viện Giọng Nói** - Tổ chức và quản lý tài sản giọng nói
- **Phân Tích Chất Lượng Giọng Nói** - Đánh giá chất lượng giọng nói tự động
- **Đào Tạo Giọng Nói** - Cải thiện mô hình giọng nói liên tục

### Cải Thiện Âm Thanh
- **Cải Thiện Thời Gian Thực** - Cải thiện chất lượng âm thanh trực tiếp
- **Giảm Nhiễu** - Lọc nhiễu nâng cao
- **Chuẩn Hóa Âm Thanh** - Mức âm thanh nhất quán
- **Tối Ưu Hóa Định Dạng** - Lựa chọn định dạng âm thanh tối ưu

## 🧪 Testing

### Thực Thi Test

```bash
# Chạy tất cả tests
pytest

# Chạy với coverage
pytest --cov=app --cov-report=html

# Chạy module test cụ thể
pytest tests/test_auth.py -v
pytest tests/test_tts.py -v
pytest tests/test_bi_service.py -v

# Chạy tests Trí tuệ Kinh doanh
pytest tests/run_bi_tests.py -v

# Test hiệu suất
pytest tests/test_batch_performance.py -v
```

### Coverage Test

```bash
# Tạo báo cáo coverage
pytest --cov=app --cov-report=term-missing --cov-report=html

# Coverage cho module cụ thể
pytest --cov=utils.bi_service --cov-report=term-missing
```

## 🐳 Triển Khai Docker

### Môi Trường Development

```bash
# Build và khởi động tất cả dịch vụ
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng dịch vụ
docker-compose down

# Rebuild dịch vụ cụ thể
docker-compose build tts-api
docker-compose up -d tts-api
```

### Triển Khai Production

```bash
# Build image production
docker build -t tts-gemini:latest .

# Chạy với cấu hình production
docker run -d \
  --name tts-gemini \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your-api-key \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  --restart unless-stopped \
  tts-gemini:latest

# Hoặc sử dụng production docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Cấu Hình Docker

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

## 📈 Giám Sát & Phân Tích

### Giám Sát Hệ Thống
- **Kiểm Tra Tình Trạng** - Xác minh tình trạng hệ thống tự động
- **Số Liệu Hiệu Suất** - Theo dõi hiệu suất thời gian thực
- **Giám Sát Lỗi** - Theo dõi lỗi toàn diện
- **Sử Dụng Tài Nguyên** - Giám sát CPU, bộ nhớ và đĩa

### Bảng Điều Khiển Trí Tuệ Kinh Doanh
- **Bảng Doanh Thu** - Trực quan hóa hiệu suất tài chính
- **Bảng Khách Hàng** - Phân tích hành vi khách hàng
- **Bảng Sử Dụng** - Mẫu sử dụng hệ thống
- **Bảng KPI** - Chỉ số hiệu suất chính

### Hệ Thống Cảnh Báo
- **Cảnh Báo Doanh Thu** - Thông báo ngưỡng doanh thu
- **Cảnh Báo Hệ Thống** - Cảnh báo hiệu suất hệ thống
- **Cảnh Báo Khách Hàng** - Cảnh báo hành vi khách hàng
- **Cảnh Báo Tùy Chỉnh** - Điều kiện cảnh báo do người dùng định nghĩa

## 🔒 Tính Năng Bảo Mật

### Xác Thực & Ủy Quyền
- **Token JWT** - Xác thực dựa trên token bảo mật
- **Kiểm Soát Truy Cập Dựa Trên Vai Trò** - Hệ thống quyền hạn chi tiết
- **Quản Lý API Key** - Xử lý API key bảo mật
- **Quản Lý Session** - Xử lý session bảo mật

### Bảo Vệ Dữ Liệu
- **Xác Thực Đầu Vào** - Làm sạch đầu vào toàn diện
- **Ngăn Chặn SQL Injection** - Biện pháp bảo mật cơ sở dữ liệu
- **Bảo Vệ XSS** - Ngăn chặn cross-site scripting
- **Bảo Vệ CSRF** - Ngăn chặn cross-site request forgery

### Bảo Mật Mạng
- **Cấu Hình CORS** - Chính sách cross-origin bảo mật
- **Giới Hạn Tỷ Lệ** - Kiểm soát yêu cầu và ngăn chặn lạm dụng
- **Danh Sách Trắng IP** - Kiểm soát truy cập mạng
- **SSL/TLS** - Giao thức liên lạc bảo mật

## 🚀 Triển Khai Production

### Checklist Triển Khai

- [ ] **Thiết Lập Môi Trường** - Cấu hình môi trường production
- [ ] **Migration Cơ Sở Dữ Liệu** - Chạy migration cơ sở dữ liệu
- [ ] **Cấu Hình SSL** - Kích hoạt HTTPS
- [ ] **Load Balancer** - Cấu hình cân bằng tải
- [ ] **Thiết Lập Giám Sát** - Triển khai hệ thống giám sát
- [ ] **Chiến Lược Backup** - Cấu hình backup dữ liệu
- [ ] **Tăng Cường Bảo Mật** - Áp dụng biện pháp bảo mật
- [ ] **Tối Ưu Hiệu Suất** - Tối ưu hiệu suất hệ thống

### Cấu Hình Production

```bash
# Biến môi trường production
export FLASK_ENV=production
export SECRET_KEY=your-production-secret-key
export DATABASE_URL=postgresql://prod_user:prod_pass@prod_db:5432/tts_prod
export REDIS_URL=redis://prod_redis:6379/0
export GEMINI_API_KEY=your-production-gemini-key

# Khởi động với gunicorn
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

## 🤝 Đóng Góp

### Thiết Lập Development

```bash
# Clone repository
git clone <repository-url>
cd tts-gemini

# Thiết lập môi trường development
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Chạy tests
pytest

# Định dạng code
black app/ tests/
isort app/ tests/

# Kiểm tra kiểu
mypy app/

# Quét bảo mật
bandit -r app/
```

### Hướng Dẫn Đóng Góp

1. **Fork** repository
2. **Tạo** feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** thay đổi: `git commit -m 'Add amazing feature'`
4. **Push** lên branch: `git push origin feature/amazing-feature`
5. **Tạo** Pull Request

### Tiêu Chuẩn Code

- Tuân thủ **hướng dẫn kiểu PEP 8**
- Viết **docstrings** toàn diện
- Bao gồm **gợi ý kiểu** cho tất cả hàm
- Thêm **unit tests** cho tính năng mới
- Cập nhật **tài liệu** khi cần

## 📚 Tài Liệu Bổ Sung

- [Tài Liệu API](docs/api_docs.md) - Tài liệu tham khảo API hoàn chỉnh
- [Hướng Dẫn Thiết Lập](docs/setup_guide.md) - Hướng dẫn thiết lập chi tiết
- [Schema Cơ Sở Dữ Liệu](docs/database_schema.md) - Cấu trúc cơ sở dữ liệu
- [Hướng Dẫn Bảo Mật](security_audit_report.md) - Cân nhắc bảo mật
- [Hướng Dẫn Hiệu Suất](docs/performance_optimization.md) - Tối ưu hiệu suất

## 📄 Giấy Phép

Được phân phối theo **Giấy phép MIT**. Xem [LICENSE](LICENSE) để biết thêm thông tin.

## 🙏 Lời Cảm Ơn

- **[Flask](https://flask.palletsprojects.com/)** - Web framework
- **[Google Gemini AI](https://ai.google.dev/)** - TTS engine
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - Database ORM
- **[Redis](https://redis.io/)** - Caching và sessions
- **[PostgreSQL](https://postgresql.org/)** - Cơ sở dữ liệu chính
- **[Docker](https://docker.com/)** - Containerization

## 📞 Hỗ Trợ

### Nhận Sự Giúp Đỡ

1. **Kiểm Tra Issues** - Duyệt [GitHub Issues](../../issues) hiện có
2. **Tạo Issue** - Báo cáo lỗi hoặc yêu cầu tính năng
3. **Tài Liệu** - Xem lại tài liệu chi tiết
4. **Cộng Đồng** - Tham gia cộng đồng developer

### Thông Tin Liên Hệ

- **Email**: support@tts-gemini.com
- **Tài Liệu**: [docs.tts-gemini.com](https://docs.tts-gemini.com)
- **Trạng Thái API**: [status.tts-gemini.com](https://status.tts-gemini.com)

---

**🎵 Lập trình vui vẻ với TTS-Gemini!**

*Được xây dựng với ❤️ bởi đội ngũ TTS-Gemini*