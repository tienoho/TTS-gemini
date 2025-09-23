# 📖 TTS System Usage Examples

## Tổng Quan

Tài liệu này cung cấp các ví dụ thực tế về cách sử dụng hệ thống TTS với Trí Tuệ Kinh Doanh, từ các thao tác cơ bản đến nâng cao.

## 🔐 1. Authentication & Authorization

### Đăng Ký Tài Khoản Mới

```bash
curl -X POST http://localhost:5000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password123",
    "full_name": "John Doe",
    "organization_name": "Acme Corp"
  }'
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user_id": "uuid-here",
  "access_token": "jwt_access_token_here",
  "refresh_token": "jwt_refresh_token_here",
  "expires_in": 3600
}
```

### Đăng Nhập

```bash
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password123"
  }'
```

### Sử Dụng API với JWT Token

```bash
# Lưu token sau khi đăng nhập
ACCESS_TOKEN="your_jwt_token_here"

# Sử dụng token trong các request tiếp theo
curl -X GET http://localhost:5000/api/v1/auth/profile \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

## 🎵 2. Text-to-Speech Operations

### Tạo Audio Cơ Bản

```bash
curl -X POST http://localhost:5000/api/v1/tts/generate \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Xin chào, đây là hệ thống TTS với trí tuệ kinh doanh!",
    "voice_name": "Alnilam",
    "language": "vi-VN",
    "audio_format": "mp3",
    "speed": 1.0,
    "pitch": 0.0
  }'
```

**Response:**
```json
{
  "request_id": "req_123456789",
  "status": "processing",
  "text": "Xin chào, đây là hệ thống TTS với trí tuệ kinh doanh!",
  "estimated_duration": 3.2,
  "message": "Audio generation started"
}
```

### Kiểm Tra Trạng Thái Request

```bash
curl -X GET http://localhost:5000/api/v1/tts/req_123456789 \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**Response khi hoàn thành:**
```json
{
  "request_id": "req_123456789",
  "status": "completed",
  "text": "Xin chào, đây là hệ thống TTS với trí tuệ kinh doanh!",
  "audio_url": "/api/v1/tts/req_123456789/download",
  "file_size": 245760,
  "duration": 3.2,
  "created_at": "2025-01-23T10:30:00Z",
  "completed_at": "2025-01-23T10:30:03Z"
}
```

### Tải Xuống Audio File

```bash
curl -X GET http://localhost:5000/api/v1/tts/req_123456789/download \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -o output.mp3
```

### Xử Lý Hàng Loạt TTS

```bash
curl -X POST http://localhost:5000/api/v1/batch/tts \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Text đầu tiên cần chuyển đổi thành audio",
      "Text thứ hai với giọng nói khác nhau",
      "Text thứ ba với cài đặt tùy chỉnh"
    ],
    "voice_name": "Alnilam",
    "language": "vi-VN",
    "audio_format": "mp3",
    "webhook_url": "https://your-app.com/webhook/batch-complete"
  }'
```

**Response:**
```json
{
  "batch_id": "batch_987654321",
  "status": "processing",
  "total_items": 3,
  "estimated_completion": "2025-01-23T10:35:00Z",
  "progress_url": "/api/v1/batch/tts/batch_987654321/progress"
}
```

### Theo Dõi Tiến Trình Batch

```bash
curl -X GET http://localhost:5000/api/v1/batch/tts/batch_987654321/progress \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**Response:**
```json
{
  "batch_id": "batch_987654321",
  "status": "processing",
  "total_items": 3,
  "completed_items": 2,
  "failed_items": 0,
  "progress_percent": 66.7,
  "current_item": "Text thứ hai với giọng nói khác nhau",
  "estimated_time_remaining": "45 seconds"
}
```

## 📊 3. Business Intelligence Operations

### Lấy Phân Tích Doanh Thu

```bash
curl -X GET "http://localhost:5000/api/v1/bi/revenue?days=30&forecast=true&forecast_months=6" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**Response:**
```json
{
  "timestamp": "2025-01-23T10:30:00Z",
  "period_days": 30,
  "period_start": "2024-12-24T10:30:00Z",
  "period_end": "2025-01-23T10:30:00Z",
  "organization_id": "org_123456789",
  "revenue_dashboard": {
    "total_revenue": 125000.50,
    "total_cost": 37500.25,
    "total_profit": 87499.75,
    "profit_margin": 69.9,
    "revenue_by_type": {
      "subscription": {"amount": 75000.00, "count": 150, "avg_amount": 500.00},
      "pay_per_use": {"amount": 50000.50, "count": 500, "avg_amount": 100.01}
    },
    "revenue_growth_percent": 15.5,
    "average_revenue_per_stream": 192.31
  },
  "revenue_forecast": {
    "forecast_method": "linear_regression",
    "forecast_periods": 6,
    "forecast_data": [
      {
        "period": "2025-02",
        "forecasted_revenue": 135000.00,
        "confidence_interval_low": 128000.00,
        "confidence_interval_high": 142000.00
      }
    ]
  }
}
```

### Phân Tích Khách Hàng

```bash
curl -X GET "http://localhost:5000/api/v1/bi/customers?days=90&segmentation=true&churn=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**Response:**
```json
{
  "timestamp": "2025-01-23T10:30:00Z",
  "period_days": 90,
  "organization_id": "org_123456789",
  "customer_dashboard": {
    "new_customers": 45,
    "total_customers": 1250,
    "active_customers": 1180,
    "churned_customers": 70,
    "churn_rate_percent": 5.6,
    "segment_distribution": {
      "enterprise": 25,
      "sme": 180,
      "startup": 450,
      "individual": 595
    },
    "average_lifetime_value": 1250.00,
    "customer_acquisition_rate": 0.5
  },
  "customer_segmentation": {
    "segments": {
      "enterprise": {
        "count": 25,
        "avg_revenue": 5000.00,
        "features": ["advanced_analytics", "custom_integration"]
      }
    }
  },
  "churn_prediction": {
    "high_risk_customers": 15,
    "medium_risk_customers": 35,
    "risk_factors": ["usage_frequency", "support_tickets"]
  }
}
```

### Phân Tích Sử Dụng Hệ Thống

```bash
curl -X GET "http://localhost:5000/api/v1/bi/usage?days=30&patterns=true&anomalies=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**Response:**
```json
{
  "timestamp": "2025-01-23T10:30:00Z",
  "period_days": 30,
  "organization_id": "org_123456789",
  "usage_dashboard": {
    "total_requests": 15750,
    "successful_requests": 15400,
    "failed_requests": 350,
    "average_response_time": 0.45,
    "peak_usage_hour": 14,
    "error_rate_percent": 2.2,
    "throughput_per_hour": 525
  },
  "usage_patterns": {
    "patterns": [
      {
        "type": "daily_pattern",
        "description": "Daily usage pattern detected",
        "strength": 0.7,
        "peak_hours": [9, 10, 11, 14, 15, 16]
      }
    ],
    "anomalies": [
      {
        "timestamp": "2025-01-20T10:30:00Z",
        "metric": "requests",
        "value": 1200,
        "expected_range": "800 - 1000",
        "severity": "medium"
      }
    ]
  }
}
```

### Bảng Điều Khiển KPI

```bash
curl -X GET "http://localhost:5000/api/v1/bi/kpis?days=30&detailed=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**Response:**
```json
{
  "timestamp": "2025-01-23T10:30:00Z",
  "period_days": 30,
  "organization_id": "org_123456789",
  "kpi_dashboard": {
    "revenue": {
      "current_value": 125000.50,
      "target_value": 100000.00,
      "change_percent": 25.0,
      "performance_status": "exceeded"
    },
    "profit_margin": {
      "current_value": 69.9,
      "target_value": 65.0,
      "change_percent": 7.5,
      "performance_status": "exceeded"
    },
    "customer_acquisition_cost": {
      "current_value": 45.50,
      "target_value": 50.00,
      "change_percent": -9.0,
      "performance_status": "on_track"
    }
  },
  "detailed_kpis": {
    "system_uptime": {
      "current_value": 99.95,
      "target_value": 99.90,
      "performance_status": "exceeded"
    },
    "error_rate": {
      "current_value": 0.8,
      "target_value": 1.0,
      "performance_status": "on_track"
    }
  }
}
```

## 📋 4. Báo Cáo & Analytics

### Tạo Báo Cáo Tùy Chỉnh

```bash
curl -X POST http://localhost:5000/api/v1/bi/reports \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "monthly_business_review",
    "date_from": "2024-12-01T00:00:00Z",
    "date_to": "2025-01-01T00:00:00Z",
    "format": "pdf",
    "parameters": {
      "include_charts": true,
      "include_recommendations": true,
      "language": "vi"
    }
  }'
```

**Response:**
```json
{
  "message": "Report generation started",
  "report_id": "report_456789123",
  "status": "processing",
  "estimated_completion": "within_5_minutes"
}
```

### Kiểm Tra Trạng Thái Báo Cáo

```bash
curl -X GET http://localhost:5000/api/v1/bi/reports/report_456789123 \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**Response khi hoàn thành:**
```json
{
  "report_id": "report_456789123",
  "report_name": "Monthly Business Review - December 2024",
  "report_type": "monthly_business_review",
  "status": "completed",
  "format": "pdf",
  "generated_at": "2025-01-23T10:35:00Z",
  "download_url": "/api/v1/bi/reports/report_456789123/download",
  "summary": "Comprehensive business review with revenue analysis, customer insights, and strategic recommendations"
}
```

### Tải Xuống Báo Cáo

```bash
curl -X GET http://localhost:5000/api/v1/bi/reports/report_456789123/download \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -o "monthly_business_review_dec_2024.pdf"
```

### Lập Lịch Báo Cáo Định Kỳ

```bash
curl -X POST http://localhost:5000/api/v1/bi/reports/scheduled \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "weekly_performance",
    "schedule_config": {
      "frequency": "weekly",
      "day_of_week": "monday",
      "time": "09:00",
      "timezone": "Asia/Ho_Chi_Minh",
      "recipients": ["manager@company.com", "admin@company.com"],
      "format": "pdf"
    }
  }'
```

**Response:**
```json
{
  "message": "Report scheduled successfully",
  "scheduled_report_id": "sched_789123456",
  "report_type": "weekly_performance",
  "next_run": "2025-01-27T09:00:00Z"
}
```

## 🎯 5. Advanced Features

### Voice Cloning

```bash
# Upload dữ liệu training
curl -X POST http://localhost:5000/api/v1/voice-cloning/train \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "audio_files=@voice_sample1.wav" \
  -F "audio_files=@voice_sample2.wav" \
  -F "voice_name=My Custom Voice" \
  -F "language=vi-VN"
```

**Response:**
```json
{
  "training_id": "train_321654987",
  "status": "training_started",
  "estimated_completion": "2025-01-23T12:30:00Z",
  "voice_name": "My Custom Voice"
}
```

### Audio Enhancement

```bash
curl -X POST http://localhost:5000/api/v1/audio-enhancement \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "/api/v1/tts/req_123456789/download",
    "enhancement_type": "noise_reduction",
    "parameters": {
      "noise_reduction_level": "medium",
      "normalize_audio": true,
      "output_format": "mp3"
    }
  }'
```

### Webhook Integration

```bash
# Tạo webhook endpoint
curl -X POST http://localhost:5000/api/v1/webhooks \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "TTS Completion Webhook",
    "url": "https://your-app.com/webhooks/tts-complete",
    "events": ["tts.completed", "tts.failed", "batch.completed"],
    "secret": "your_webhook_secret"
  }'
```

**Response:**
```json
{
  "webhook_id": "wh_654321789",
  "name": "TTS Completion Webhook",
  "url": "https://your-app.com/webhooks/tts-complete",
  "status": "active"
}
```

### Real-time WebSocket

```javascript
// JavaScript WebSocket client
const ws = new WebSocket('ws://localhost:5000/api/v1/ws/tts');

ws.onopen = function(event) {
  console.log('Connected to TTS WebSocket');

  // Subscribe to real-time updates
  ws.send(JSON.stringify({
    "action": "subscribe",
    "channels": ["tts_progress", "system_alerts"]
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);

  if (data.type === 'tts_progress') {
    updateProgressBar(data.progress_percent);
  }
};
```

## 🔍 6. Error Handling & Troubleshooting

### Xử Lý Lỗi Thường Gặp

```bash
# Lỗi 401 Unauthorized
curl -X POST http://localhost:5000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your_refresh_token"}'

# Lỗi 429 Rate Limited
curl -X GET http://localhost:5000/api/v1/auth/profile \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Retry-After: 60"
```

### Kiểm Tra Health Status

```bash
curl -X GET http://localhost:5000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-23T10:30:00Z",
  "version": "1.0.0",
  "environment": "development",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "gemini_api": "healthy"
  }
}
```

### Debug Mode

```bash
# Kích hoạt debug logging
curl -X POST http://localhost:5000/api/v1/debug/enable \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"level": "DEBUG", "duration_minutes": 30}'
```

## 📈 7. Performance Optimization

### Batch Processing với Progress Tracking

```python
import requests
import time
import json

def process_large_batch(texts, access_token):
    # Tạo batch request
    response = requests.post(
        'http://localhost:5000/api/v1/batch/tts',
        headers={'Authorization': f'Bearer {access_token}'},
        json={
            'texts': texts,
            'voice_name': 'Alnilam',
            'language': 'vi-VN'
        }
    )

    batch_data = response.json()
    batch_id = batch_data['batch_id']

    # Theo dõi tiến trình
    while True:
        progress = requests.get(
            f'http://localhost:5000/api/v1/batch/tts/{batch_id}/progress',
            headers={'Authorization': f'Bearer {access_token}'}
        ).json()

        print(f"Progress: {progress['progress_percent']:.1f}%")

        if progress['status'] == 'completed':
            break

        time.sleep(5)

    return batch_id
```

### Concurrent Request Handling

```python
import asyncio
import aiohttp
import json

async def process_concurrent_requests(texts, access_token):
    async with aiohttp.ClientSession() as session:

        tasks = []
        for text in texts:
            task = session.post(
                'http://localhost:5000/api/v1/tts/generate',
                headers={
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                },
                json={'text': text, 'voice_name': 'Alnilam'}
            )
            tasks.append(task)

        # Thực hiện tất cả requests đồng thời
        responses = await asyncio.gather(*tasks)

        results = []
        for response in responses:
            result = await response.json()
            results.append(result)

        return results
```

## 🎨 8. Integration Examples

### Python SDK Usage

```python
from tts_client import TTSClient

# Khởi tạo client
client = TTSClient(
    base_url="http://localhost:5000",
    api_key="your_api_key"
)

# Tạo audio đơn giản
result = client.generate_audio(
    text="Xin chào thế giới!",
    voice_name="Alnilam",
    language="vi-VN"
)

print(f"Audio URL: {result['audio_url']}")

# Lấy analytics
analytics = client.get_revenue_analytics(days=30)
print(f"Total Revenue: ${analytics['total_revenue']}")
```

### React Frontend Integration

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function TTSComponent() {
  const [text, setText] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const generateAudio = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post('/api/v1/tts/generate', {
        text: text,
        voice_name: 'Alnilam',
        language: 'vi-VN'
      }, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });

      // Poll for completion
      const checkCompletion = setInterval(async () => {
        const statusResponse = await axios.get(
          `/api/v1/tts/${response.data.request_id}`,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
          }
        );

        if (statusResponse.data.status === 'completed') {
          setAudioUrl(statusResponse.data.audio_url);
          setIsLoading(false);
          clearInterval(checkCompletion);
        }
      }, 2000);

    } catch (error) {
      console.error('Error generating audio:', error);
      setIsLoading(false);
    }
  };

  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Nhập text cần chuyển thành audio..."
      />
      <button onClick={generateAudio} disabled={isLoading}>
        {isLoading ? 'Đang xử lý...' : 'Tạo Audio'}
      </button>

      {audioUrl && (
        <audio controls>
          <source src={audioUrl} type="audio/mpeg" />
        </audio>
      )}
    </div>
  );
}
```

### Node.js Backend Integration

```javascript
const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

// TTS endpoint
app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice_name = 'Alnilam' } = req.body;

    // Gọi TTS API
    const response = await axios.post('http://localhost:5000/api/v1/tts/generate', {
      text,
      voice_name,
      language: 'vi-VN'
    }, {
      headers: {
        'Authorization': `Bearer ${process.env.TTS_API_KEY}`
      }
    });

    // Trả về kết quả
    res.json({
      request_id: response.data.request_id,
      status: response.data.status
    });

  } catch (error) {
    console.error('TTS Error:', error.response?.data || error.message);
    res.status(500).json({ error: 'TTS generation failed' });
  }
});

// Business Intelligence endpoint
app.get('/api/analytics/revenue', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:5000/api/v1/bi/revenue', {
      headers: {
        'Authorization': `Bearer ${process.env.TTS_API_KEY}`
      },
      params: {
        days: 30,
        forecast: true
      }
    });

    res.json(response.data);
  } catch (error) {
    console.error('Analytics Error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Analytics retrieval failed' });
  }
});

app.listen(3000, () => {
  console.log('Integration server running on port 3000');
});
```

## 🔐 9. Security Best Practices

### API Key Management

```bash
# Tạo API key mới
curl -X POST http://localhost:5000/api/v1/auth/api-key \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Sử dụng API key trong server-to-server communication
curl -X POST http://localhost:5000/api/v1/tts/generate \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Secure text", "voice_name": "Alnilam"}'
```

### Rate Limiting Handling

```javascript
async function makeRateLimitedRequest(url, data, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });

      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After') || 60;
        console.log(`Rate limited. Retrying after ${retryAfter} seconds...`);
        await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
        continue;
      }

      return await response.json();
    } catch (error) {
      if (attempt === maxRetries) {
        throw error;
      }
      console.log(`Attempt ${attempt} failed, retrying...`);
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }
}
```

## 📊 10. Monitoring & Analytics Integration

### Real-time Dashboard Data

```javascript
async function updateDashboard() {
  try {
    // Lấy revenue analytics
    const revenueResponse = await fetch('/api/v1/bi/revenue?days=30', {
      headers: { 'Authorization': `Bearer ${accessToken}` }
    });
    const revenueData = await revenueResponse.json();

    // Lấy customer analytics
    const customerResponse = await fetch('/api/v1/bi/customers?days=30', {
      headers: { 'Authorization': `Bearer ${accessToken}` }
    });
    const customerData = await customerResponse.json();

    // Lấy usage analytics
    const usageResponse = await fetch('/api/v1/bi/usage?days=7', {
      headers: { 'Authorization': `Bearer ${accessToken}` }
    });
    const usageData = await usageResponse.json();

    // Cập nhật dashboard
    updateRevenueChart(revenueData);
    updateCustomerChart(customerData);
    updateUsageChart(usageData);

  } catch (error) {
    console.error('Dashboard update failed:', error);
  }
}

// Tự động cập nhật mỗi 5 phút
setInterval(updateDashboard, 5 * 60 * 1000);
```

### Alert System Integration

```javascript
// WebSocket connection cho real-time alerts
const ws = new WebSocket('ws://localhost:5000/api/v1/ws/alerts');

ws.onmessage = function(event) {
  const alert = JSON.parse(event.data);

  switch (alert.type) {
    case 'revenue_drop':
      showNotification('Cảnh báo: Doanh thu giảm', alert.message, 'warning');
      break;
    case 'high_churn':
      showNotification('Cảnh báo: Tỷ lệ rời bỏ cao', alert.message, 'error');
      break;
    case 'system_downtime':
      showNotification('Cảnh báo: Hệ thống gián đoạn', alert.message, 'critical');
      break;
    default:
      showNotification('Thông báo hệ thống', alert.message, 'info');
  }
};
```

---

*These usage examples demonstrate how to effectively integrate with and utilize the TTS system with Business Intelligence features. From basic text-to-speech operations to advanced analytics and monitoring, these examples cover the full spectrum of system capabilities.*