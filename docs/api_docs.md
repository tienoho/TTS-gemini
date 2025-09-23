# API Documentation

## Authentication

### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

### Get Profile
```http
GET /api/v1/auth/profile
Authorization: Bearer <access_token>
```

## Text-to-Speech

### Generate Audio
```http
POST /api/v1/tts/generate
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "text": "string",
  "voice_name": "Alnilam",
  "output_format": "wav",
  "speed": 1.0,
  "pitch": 0.0
}
```

### Get Audio Requests
```http
GET /api/v1/tts/?page=1&per_page=10&status=pending&sort_by=created_at&sort_order=desc
Authorization: Bearer <access_token>
```

### Get Specific Request
```http
GET /api/v1/tts/{request_id}
Authorization: Bearer <access_token>
```

### Download Audio
```http
GET /api/v1/tts/{request_id}/download
Authorization: Bearer <access_token>
```

### Delete Request
```http
DELETE /api/v1/tts/{request_id}
Authorization: Bearer <access_token>
```

### Get User Stats
```http
GET /api/v1/tts/stats
Authorization: Bearer <access_token>
```

## Error Responses

### Standard Error Format
```json
{
  "error": "ErrorType",
  "message": "Human readable message",
  "details": {},
  "timestamp": "2023-01-01T00:00:00Z",
  "path": "/api/v1/endpoint",
  "request_id": "uuid"
}
```

### Common HTTP Status Codes

- `200` - Success
- `201` - Created
- `202` - Accepted (processing)
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `413` - Payload Too Large
- `429` - Too Many Requests
- `500` - Internal Server Error

## Rate Limiting

- **Free users**: 100 requests per minute
- **Premium users**: 1000 requests per minute

Rate limit headers:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Window reset timestamp
- `X-RateLimit-Retry-After`: Seconds until reset

## File Formats

### Supported Audio Formats
- `wav` - Waveform Audio File Format
- `mp3` - MPEG Audio Layer III
- `ogg` - Ogg Vorbis
- `flac` - Free Lossless Audio Codec

### File Size Limits
- Maximum file size: 10MB per request
- Maximum text length: 5000 characters

## Webhooks (Future)

### Request Status Updates
```http
POST /your-webhook-url
Content-Type: application/json

{
  "event": "request.completed",
  "request_id": "uuid",
  "status": "completed",
  "download_url": "https://api.example.com/download/uuid",
  "timestamp": "2023-01-01T00:00:00Z"
}
```

## SDK Examples

### Python
```python
import requests

# Authentication
auth_response = requests.post('http://localhost:5000/api/v1/auth/login', json={
    'username': 'your_username',
    'password': 'your_password'
})

access_token = auth_response.json()['tokens']['access_token']

# Generate audio
headers = {'Authorization': f'Bearer {access_token}'}
tts_response = requests.post('http://localhost:5000/api/v1/tts/generate', json={
    'text': 'Hello, world!',
    'voice_name': 'Alnilam'
}, headers=headers)

print(f"Request ID: {tts_response.json()['request_id']}")
```

### JavaScript
```javascript
// Authentication
const loginResponse = await fetch('http://localhost:5000/api/v1/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    username: 'your_username',
    password: 'your_password'
  })
});

const { tokens } = await loginResponse.json();
const accessToken = tokens.access_token;

// Generate audio
const ttsResponse = await fetch('http://localhost:5000/api/v1/tts/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${accessToken}`
  },
  body: JSON.stringify({
    text: 'Hello, world!',
    voice_name: 'Alnilam'
  })
});

const result = await ttsResponse.json();
console.log('Request ID:', result.request_id);
```

## Best Practices

1. **Authentication**: Always include valid access tokens
2. **Rate Limiting**: Respect rate limits and handle 429 responses
3. **Error Handling**: Implement proper error handling for all endpoints
4. **File Downloads**: Use appropriate timeouts for file downloads
5. **Text Length**: Keep text under 5000 characters per request
6. **Voice Selection**: Use valid voice names only
7. **Format Support**: Check supported formats before requesting

## Monitoring

### Health Check
```http
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2023-01-01T00:00:00Z",
  "version": "1.0.0",
  "environment": "production"
}
```

### Metrics
- Request count and timing
- Error rates by endpoint
- Audio generation success rates
- Storage usage statistics