# API Endpoints Specification

## Tổng quan
API được thiết kế theo RESTful principles với authentication, rate limiting, và comprehensive error handling.

## Base URL
```
Production: https://api.tts-service.com/api/v1
Development: http://localhost:5000/api/v1
```

## Authentication
Tất cả endpoints (trừ health check) yêu cầu authentication qua JWT token hoặc API key.

### Headers
```
Authorization: Bearer <jwt_token>
X-API-Key: <api_key>
Content-Type: application/json
```

## Endpoints

### 1. Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-22T10:30:00Z",
  "version": "1.0.0",
  "environment": "production",
  "services": {
    "database": "connected",
    "redis": "connected",
    "gemini_api": "available"
  }
}
```

### 2. Authentication Endpoints

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "message": "User created successfully",
  "user_id": 1,
  "api_key": "sk-abc123..."
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "username": "testuser",
    "email": "test@example.com",
    "is_premium": false
  }
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Authorization: Bearer <refresh_token>
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

#### Generate API Key
```http
POST /api/v1/auth/api-key
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "api_key": "sk-abc123..."
}
```

### 3. TTS Request Endpoints

#### Create TTS Request
```http
POST /api/v1/tts/request
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "text": "Text to convert to speech",
  "voice_settings": {
    "voice_name": "Alnilam",
    "language": "vi",
    "speed": 1.0,
    "pitch": 0.0
  },
  "output_format": "wav",
  "metadata": {
    "custom_field": "value"
  }
}
```

**Voice Settings:**
- `voice_name`: Tên giọng nói (Alnilam, Puck, Charon, etc.)
- `language`: Ngôn ngữ (vi, en, ja, ko, etc.)
- `speed`: Tốc độ (0.5 - 2.0)
- `pitch`: Cao độ (-10.0 - 10.0)

**Response:**
```json
{
  "request_id": "12345",
  "status": "queued",
  "message": "TTS request has been queued for processing",
  "estimated_time": "10-30 seconds",
  "queue_position": 3
}
```

#### Get Request Status
```http
GET /api/v1/tts/status/{request_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "request_id": "12345",
  "status": "processing",
  "progress": 75,
  "created_at": "2025-01-22T10:00:00Z",
  "updated_at": "2025-01-22T10:00:15Z",
  "voice_settings": {
    "voice_name": "Alnilam",
    "language": "vi",
    "speed": 1.0,
    "pitch": 0.0
  },
  "logs": [
    {
      "timestamp": "2025-01-22T10:00:00Z",
      "level": "INFO",
      "message": "Request queued for processing"
    },
    {
      "timestamp": "2025-01-22T10:00:05Z",
      "level": "INFO",
      "message": "Started processing request"
    }
  ],
  "download_url": null
}
```

#### Download Audio File
```http
GET /api/v1/tts/result/{request_id}
Authorization: Bearer <access_token>
```

**Response:** Audio file binary data với headers:
```
Content-Type: audio/wav
Content-Disposition: attachment; filename="tts_12345_20250122_100000.wav"
```

#### Cancel Request
```http
DELETE /api/v1/tts/request/{request_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "message": "Request cancelled successfully"
}
```

#### List User Requests
```http
GET /api/v1/tts/requests?page=1&per_page=10&status=completed&sort_by=created_at&sort_order=desc
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `page`: Số trang (default: 1)
- `per_page`: Số items mỗi trang (default: 10, max: 100)
- `status`: Lọc theo status (pending, processing, completed, failed)
- `sort_by`: Sắp xếp theo (created_at, updated_at, status)
- `sort_order`: Thứ tự sắp xếp (asc, desc)

**Response:**
```json
{
  "requests": [
    {
      "id": 12345,
      "status": "completed",
      "text_content": "Text to convert...",
      "voice_name": "Alnilam",
      "output_format": "wav",
      "created_at": "2025-01-22T10:00:00Z",
      "updated_at": "2025-01-22T10:00:15Z",
      "processing_time": 12.5,
      "file_size": 245760,
      "download_url": "/api/v1/tts/result/12345"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total_pages": 5,
    "total_items": 45,
    "has_next": true,
    "has_prev": false
  }
}
```

### 4. User Management Endpoints

#### Get User Profile
```http
GET /api/v1/users/profile
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "id": 1,
  "username": "testuser",
  "email": "test@example.com",
  "is_premium": false,
  "created_at": "2025-01-22T10:00:00Z",
  "stats": {
    "total_requests": 150,
    "completed_requests": 145,
    "failed_requests": 5,
    "avg_processing_time": 8.5
  }
}
```

#### Update User Profile
```http
PUT /api/v1/users/profile
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "email": "newemail@example.com"
}
```

**Response:**
```json
{
  "message": "Profile updated successfully",
  "user": {
    "id": 1,
    "username": "testuser",
    "email": "newemail@example.com"
  }
}
```

#### Get User Statistics
```http
GET /api/v1/users/stats
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "total_requests": 150,
  "completed_requests": 145,
  "failed_requests": 5,
  "processing_requests": 0,
  "pending_requests": 0,
  "avg_processing_time": 8.5,
  "total_audio_duration": 1250.5,
  "total_storage_used": 24576000,
  "requests_today": 10,
  "requests_this_month": 150
}
```

### 5. Admin Endpoints

#### Get System Statistics
```http
GET /api/v1/admin/stats
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "total_users": 1250,
  "active_users": 890,
  "premium_users": 125,
  "total_requests": 45678,
  "requests_today": 1234,
  "success_rate": 98.5,
  "avg_processing_time": 8.2,
  "queue_status": {
    "pending": 45,
    "processing": 12,
    "failed": 3
  },
  "system_health": {
    "database": "healthy",
    "redis": "healthy",
    "gemini_api": "healthy"
  }
}
```

#### Get Failed Requests
```http
GET /api/v1/admin/failed-requests?page=1&per_page=20
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "failed_requests": [
    {
      "id": 12345,
      "user_id": 1,
      "text_content": "Text...",
      "error_message": "Gemini API quota exceeded",
      "created_at": "2025-01-22T10:00:00Z",
      "retry_count": 2
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_pages": 15,
    "total_items": 300
  }
}
```

## Error Responses

### Standard Error Format
```json
{
  "error": "ErrorType",
  "message": "Human readable error message",
  "code": "ERROR_CODE",
  "details": {},
  "timestamp": "2025-01-22T10:30:00Z"
}
```

### Common Error Codes

#### Authentication Errors
- `INVALID_TOKEN`: Token không hợp lệ
- `TOKEN_EXPIRED`: Token đã hết hạn
- `INSUFFICIENT_PERMISSIONS`: Không có quyền truy cập
- `INVALID_CREDENTIALS`: Thông tin đăng nhập sai

#### Rate Limiting Errors
- `RATE_LIMIT_EXCEEDED`: Vượt quá giới hạn rate limit
- `QUOTA_EXCEEDED`: Vượt quá quota sử dụng

#### Validation Errors
- `INVALID_INPUT`: Dữ liệu đầu vào không hợp lệ
- `MISSING_REQUIRED_FIELD`: Thiếu trường bắt buộc
- `INVALID_FORMAT`: Định dạng không đúng

#### TTS Processing Errors
- `TEXT_TOO_LONG`: Text quá dài
- `UNSUPPORTED_VOICE`: Giọng nói không được hỗ trợ
- `GEMINI_API_ERROR`: Lỗi từ Gemini API
- `QUOTA_EXCEEDED`: Vượt quá quota API

#### System Errors
- `DATABASE_ERROR`: Lỗi cơ sở dữ liệu
- `REDIS_ERROR`: Lỗi Redis
- `INTERNAL_SERVER_ERROR`: Lỗi hệ thống

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `202`: Accepted (async processing)
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `409`: Conflict
- `422`: Unprocessable Entity
- `429`: Too Many Requests
- `500`: Internal Server Error

## Rate Limiting

### Default Limits
- Standard users: 100 requests/minute
- Premium users: 1000 requests/minute
- TTS generation: 10 requests/minute per user

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642852800
X-RateLimit-Retry-After: 60
```

## Pagination

### Standard Pagination Response
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total_pages": 5,
    "total_items": 45,
    "has_next": true,
    "has_prev": false,
    "next_page": 2,
    "prev_page": null
  }
}
```

### Query Parameters
- `page`: Số trang (default: 1)
- `per_page`: Items mỗi trang (default: 10, max: 100)
- `offset`: Offset (alternative to page)

## Filtering & Sorting

### List Requests with Filters
```http
GET /api/v1/tts/requests?status=completed&created_after=2025-01-01&created_before=2025-01-31&sort_by=created_at&sort_order=desc
```

### Supported Filters
- `status`: pending, processing, completed, failed
- `created_after`: ISO datetime
- `created_before`: ISO datetime
- `voice_name`: Filter by voice
- `output_format`: wav, mp3

### Supported Sorting
- `created_at`: Thời gian tạo
- `updated_at`: Thời gian cập nhật
- `processing_time`: Thời gian xử lý
- `file_size`: Kích thước file

## Webhook Notifications

### Webhook Configuration
```http
POST /api/v1/webhooks
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "url": "https://your-app.com/webhook/tts",
  "events": ["request.completed", "request.failed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Events
- `request.queued`: Request được thêm vào queue
- `request.processing`: Bắt đầu xử lý
- `request.completed`: Xử lý thành công
- `request.failed`: Xử lý thất bại
- `request.cancelled`: Request bị hủy

### Webhook Payload
```json
{
  "event": "request.completed",
  "timestamp": "2025-01-22T10:30:00Z",
  "data": {
    "request_id": "12345",
    "user_id": 1,
    "status": "completed",
    "download_url": "/api/v1/tts/result/12345",
    "processing_time": 12.5
  }
}
```

## API Versioning

### Version Header
```http
Accept: application/vnd.tts.v1+json
```

### URL Versioning
```http
GET /api/v1/tts/requests
GET /api/v2/tts/requests
```

## Caching

### Cache Headers
```http
Cache-Control: public, max-age=300
ETag: "abc123"
Last-Modified: Wed, 22 Jan 2025 10:00:00 GMT
```

### Conditional Requests
```http
GET /api/v1/tts/status/12345
If-None-Match: "abc123"
```

## File Upload/Download

### Upload Audio File (Future Feature)
```http
POST /api/v1/audio/upload
Content-Type: multipart/form-data

{
  "file": "audio_file.wav",
  "metadata": {
    "title": "Sample Audio",
    "description": "Test audio file"
  }
}
```

### Download with Range Support
```http
GET /api/v1/tts/result/12345
Range: bytes=0-1023
```

## Testing Endpoints

### Test TTS Request
```http
POST /api/v1/test/tts
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "text": "This is a test",
  "voice_name": "Alnilam"
}
```

### Test Authentication
```http
GET /api/v1/test/auth
Authorization: Bearer <access_token>
```

## Documentation

### OpenAPI/Swagger
```http
GET /api/v1/docs
GET /api/v1/swagger.json
```

### API Documentation
```http
GET /api/v1/docs/api
```

## Monitoring Endpoints

### Metrics
```http
GET /api/v1/metrics
Authorization: Bearer <admin_token>
```

### Health Check Details
```http
GET /api/v1/health/detailed
Authorization: Bearer <admin_token>
```

## Security Features

### CORS Configuration
```http
Access-Control-Allow-Origin: https://your-frontend.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key
```

### Security Headers
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### Input Validation
- SQL injection prevention
- XSS protection
- Path traversal protection
- Rate limiting
- Input sanitization

## Performance Features

### Compression
```http
Accept-Encoding: gzip, deflate
Content-Encoding: gzip
```

### Connection Pooling
- Database connection pooling
- Redis connection pooling
- HTTP connection pooling

### Async Processing
- Non-blocking TTS processing
- Queue-based architecture
- Background workers

## Error Handling Best Practices

### Retry Logic
- Exponential backoff
- Circuit breaker pattern
- Dead letter queues

### Logging
- Structured JSON logging
- Request tracing
- Performance monitoring

### Monitoring
- Health checks
- Metrics collection
- Alerting system