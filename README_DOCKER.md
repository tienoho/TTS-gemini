# TTS API - Docker Development Setup

Hướng dẫn chạy thử TTS API sử dụng Docker.

## 📋 Yêu cầu hệ thống

- Docker Desktop
- Docker Compose
- Ít nhất 4GB RAM trống
- Ít nhất 2GB dung lượng ổ cứng

## 🚀 Cách chạy nhanh

### 1. Clone và cấu hình

```bash
# Tạo file .env từ template
cp .env.example .env

# Chỉnh sửa .env và thêm GEMINI_API_KEY
# Lấy API key từ: https://makersuite.google.com/app/apikey
```

### 2. Chạy ứng dụng

```bash
# Chạy tất cả services
docker-compose up --build

# Hoặc chạy ở background
docker-compose up --build -d
```

### 3. Kiểm tra hoạt động

```bash
# Kiểm tra health check
curl http://localhost:5000/api/v1/health

# Kiểm tra API root
curl http://localhost:5000/
```

## 🌐 Endpoints có sẵn

| Endpoint | Mô tả | URL |
|----------|--------|-----|
| API Root | Thông tin API | http://localhost:5000/ |
| Health Check | Kiểm tra trạng thái | http://localhost:5000/api/v1/health |
| Auth | Xác thực | http://localhost:5000/api/v1/auth |
| TTS | Text-to-Speech | http://localhost:5000/api/v1/tts |
| Redis Commander | Quản lý Redis | http://localhost:8081 |
| Monitoring | Dashboard theo dõi | http://localhost:8080 |

## 🛠️ Lệnh hữu ích

```bash
# Xem logs
docker-compose logs -f

# Dừng services
docker-compose down

# Restart services
docker-compose restart

# Vào container để debug
docker-compose exec app bash

# Chạy tests
docker-compose exec app pytest

# Xem trạng thái services
docker-compose ps
```

## ⚙️ Cấu hình

### Environment Variables quan trọng

```env
# Bắt buộc
GEMINI_API_KEY=your-actual-gemini-api-key

# Database (SQLite cho development)
DATABASE_URL=sqlite:///tts_api.db

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET_KEY=your-jwt-secret-key
```

### Cấu hình Database

- **Development**: SQLite (mặc định)
- **Production**: PostgreSQL (trong docker-compose)

### Cấu hình Redis

- Cache và session storage
- Rate limiting
- WebSocket connections

## 🧪 Test API

### Health Check
```bash
curl http://localhost:5000/api/v1/health
```

### Text-to-Speech (cần API key)
```bash
curl -X POST http://localhost:5000/api/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Xin chào, đây là test TTS API",
    "voice_name": "Alnilam",
    "language": "vi"
  }'
```

## 🔧 Troubleshooting

### Lỗi thường gặp

1. **Port đã được sử dụng**
   ```bash
   # Kiểm tra port
   netstat -ano | findstr :5000

   # Thay đổi port trong docker-compose.yml
   ```

2. **GEMINI_API_KEY không hợp lệ**
   - Kiểm tra API key trong .env
   - Đảm bảo API key có quyền truy cập Gemini API

3. **Database connection error**
   ```bash
   # Reset database
   docker-compose down -v
   docker-compose up --build -d
   ```

4. **Memory không đủ**
   - Tăng RAM allocation cho Docker Desktop
   - Đóng các ứng dụng không cần thiết

### Logs

```bash
# Xem logs của tất cả services
docker-compose logs

# Xem logs của service cụ thể
docker-compose logs app
docker-compose logs db
docker-compose logs redis
```

## 📊 Monitoring

- **Application Metrics**: http://localhost:8080
- **Redis Management**: http://localhost:8081
- **API Documentation**: http://localhost:5000/api/v1/health

## 🛑 Dừng ứng dụng

```bash
# Dừng và xóa containers
docker-compose down

# Dừng và xóa cả volumes (database data)
docker-compose down -v
```

## 📝 Ghi chú

- Database sẽ được tạo tự động khi chạy lần đầu
- Logs được lưu trong thư mục `./logs`
- Uploaded files được lưu trong `./uploads`
- Trong development mode, debug=True để dễ debug

## 🔐 Security Notes

- Thay đổi tất cả secret keys trong production
- Không commit .env file vào git
- Sử dụng HTTPS trong production
- Cấu hình firewall phù hợp