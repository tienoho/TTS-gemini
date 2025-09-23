# Input Validation và Sanitization

## Tổng quan
Hệ thống validation và sanitization được thiết kế để đảm bảo data integrity, security, và prevent các loại attacks như injection, XSS, và malformed input.

## Input Validation Framework

### Pydantic Models
```python
# utils/validation/models.py
from pydantic import BaseModel, Field, validator, constr
from typing import Optional, Dict, Any
from enum import Enum
import re

class VoiceName(str, Enum):
    ALNILAM = "Alnilam"
    PUCK = "Puck"
    CHARON = "Charon"
    KORE = "Kore"
    FENRIR = "Fenrir"
    AO = "Ao"

class OutputFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"

class TTSRequestSchema(BaseModel):
    """Schema for TTS request validation"""
    text: constr(min_length=1, max_length=5000, strip_whitespace=True) = Field(
        ..., description="Text to convert to speech"
    )
    voice_name: VoiceName = Field(
        default=VoiceName.ALNILAM, description="Voice to use for synthesis"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV, description="Output audio format"
    )
    speed: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier"
    )
    pitch: float = Field(
        default=0.0, ge=-10.0, le=10.0, description="Pitch adjustment"
    )
    language: Optional[str] = Field(
        default=None, min_length=2, max_length=10, description="Language code"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )

    @validator('text')
    def validate_text(cls, v):
        """Validate text content"""
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')

        # Check for potentially harmful content
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                raise ValueError('Text contains potentially harmful content')

        # Check for extremely long words (potential DoS)
        words = v.split()
        for word in words:
            if len(word) > 100:
                raise ValueError('Text contains extremely long words')

        return v

    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata"""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError('Metadata must be a dictionary')

        # Limit metadata size
        if len(str(v)) > 1000:
            raise ValueError('Metadata too large')

        return v

class UserRegistrationSchema(BaseModel):
    """Schema for user registration"""
    username: constr(min_length=3, max_length=50, strip_whitespace=True, regex=r'^[a-zA-Z0-9_]+$') = Field(
        ..., description="Username (alphanumeric and underscore only)"
    )
    email: str = Field(
        ..., description="Email address"
    )
    password: constr(min_length=8, max_length=128) = Field(
        ..., description="Password"
    )

    @validator('email')
    def validate_email(cls, v):
        """Validate email format"""
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, v):
            raise ValueError('Invalid email format')
        return v.lower().strip()

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')

        # Check for common passwords
        common_passwords = ['password', '123456', '123456789', 'qwerty', 'abc123']
        if v.lower() in common_passwords:
            raise ValueError('Password is too common')

        # Check for password complexity
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')

        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')

        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')

        return v

class PaginationSchema(BaseModel):
    """Schema for pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=10, ge=1, le=100, description="Items per page")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", regex=r'^(asc|desc)$', description="Sort order")

class FilterSchema(BaseModel):
    """Schema for filtering parameters"""
    status: Optional[str] = Field(default=None, regex=r'^(pending|processing|completed|failed)$')
    created_after: Optional[str] = Field(default=None, description="ISO datetime")
    created_before: Optional[str] = Field(default=None, description="ISO datetime")
    voice_name: Optional[VoiceName] = Field(default=None)
    output_format: Optional[OutputFormat] = Field(default=None)
```

## Sanitization Utilities

### Text Sanitization
```python
# utils/sanitization/text_sanitizer.py
import re
import html
from typing import Optional
from bleach import clean, ALLOWED_TAGS, ALLOWED_ATTRIBUTES

class TextSanitizer:
    """Comprehensive text sanitization"""

    @staticmethod
    def sanitize_text(text: str, allow_html: bool = False) -> str:
        """Sanitize text input"""
        if not text:
            return text

        # Step 1: Basic cleaning
        text = text.strip()
        if not text:
            return text

        # Step 2: Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Step 3: Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Step 4: HTML sanitization
        if allow_html:
            # Allow limited HTML tags
            allowed_tags = ALLOWED_TAGS + ['p', 'br', 'strong', 'em', 'u']
            text = clean(text, tags=allowed_tags, attributes=ALLOWED_ATTRIBUTES)
        else:
            # Escape HTML completely
            text = html.escape(text)

        # Step 5: Remove potential script content
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)

        # Step 6: Limit length
        if len(text) > 10000:
            text = text[:10000] + '...'

        return text.strip()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security"""
        if not filename:
            return 'unnamed_file'

        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_').replace('..', '_')

        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        # Remove control characters
        filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]

        return filename.strip('_')

    @staticmethod
    def sanitize_json_input(data: str) -> str:
        """Sanitize JSON input"""
        # Remove potential JSON injection
        data = re.sub(r'[\x00-\x1f\x7f]', '', data)

        # Remove comments (potential security issue)
        data = re.sub(r'//.*', '', data)
        data = re.sub(r'/\*.*?\*/', '', data, flags=re.DOTALL)

        return data.strip()

    @staticmethod
    def sanitize_sql_input(text: str) -> str:
        """Sanitize SQL input (additional layer)"""
        # Remove SQL injection patterns
        sql_patterns = [
            r';\s*(drop|delete|update|insert|create|alter)',
            r'union\s+select',
            r'--\s',
            r'/\*.*?\*/'
        ]

        for pattern in sql_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode characters"""
        import unicodedata

        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)

        # Remove non-printable characters
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')

        return text
```

### Request Sanitization
```python
# utils/sanitization/request_sanitizer.py
from flask import request
from typing import Dict, Any, Optional
import json

class RequestSanitizer:
    """Sanitize HTTP requests"""

    @staticmethod
    def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize HTTP headers"""
        sanitized = {}

        for key, value in headers.items():
            # Skip sensitive headers
            sensitive_headers = [
                'authorization', 'cookie', 'x-api-key', 'x-auth-token',
                'x-forwarded-for', 'x-real-ip', 'x-client-ip'
            ]

            if key.lower() in sensitive_headers:
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = TextSanitizer.sanitize_text(str(value))

        return sanitized

    @staticmethod
    def sanitize_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize query parameters"""
        sanitized = {}

        for key, value in params.items():
            if isinstance(value, str):
                sanitized[key] = TextSanitizer.sanitize_text(value)
            elif isinstance(value, list):
                sanitized[key] = [TextSanitizer.sanitize_text(str(item)) for item in value]
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def sanitize_json_body(body: str) -> str:
        """Sanitize JSON request body"""
        try:
            # Parse and re-serialize to validate JSON
            data = json.loads(body)
            sanitized_data = RequestSanitizer.sanitize_json_data(data)
            return json.dumps(sanitized_data)
        except json.JSONDecodeError:
            # If not valid JSON, sanitize as text
            return TextSanitizer.sanitize_json_input(body)

    @staticmethod
    def sanitize_json_data(data: Any) -> Any:
        """Recursively sanitize JSON data"""
        if isinstance(data, dict):
            return {key: RequestSanitizer.sanitize_json_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [RequestSanitizer.sanitize_json_data(item) for item in data]
        elif isinstance(data, str):
            return TextSanitizer.sanitize_text(data)
        else:
            return data

    @staticmethod
    def sanitize_file_upload(filename: str, content_type: str) -> Tuple[bool, str]:
        """Sanitize file upload"""
        # Check filename
        safe_filename = TextSanitizer.sanitize_filename(filename)

        # Check content type
        allowed_types = [
            'audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/aac',
            'text/plain', 'application/json'
        ]

        if content_type not in allowed_types:
            return False, "File type not allowed"

        # Check for suspicious patterns in filename
        suspicious_patterns = [
            r'\.exe$', r'\.bat$', r'\.cmd$', r'\.scr$', r'\.pif$',
            r'\.com$', r'\.jar$', r'\.zip$', r'\.rar$', r'\.7z$'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return False, "Suspicious file extension"

        return True, safe_filename
```

## Validation Middleware

### Request Validation Middleware
```python
# app/middleware/validation.py
from flask import request, jsonify, g
from functools import wraps
from utils.validation.models import TTSRequestSchema, PaginationSchema
from utils.sanitization.request_sanitizer import RequestSanitizer
import json

class ValidationMiddleware:
    """Middleware for request validation and sanitization"""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        # Sanitize request data
        self._sanitize_request()

        # Validate request
        self._validate_request()

        return self.app(environ, start_response)

    def _sanitize_request(self):
        """Sanitize incoming request"""
        # Sanitize headers
        sanitized_headers = RequestSanitizer.sanitize_headers(dict(request.headers))
        g.sanitized_headers = sanitized_headers

        # Sanitize query parameters
        sanitized_params = RequestSanitizer.sanitize_query_params(request.args.to_dict())
        g.sanitized_params = sanitized_params

        # Sanitize JSON body if present
        if request.is_json:
            sanitized_body = RequestSanitizer.sanitize_json_body(request.get_data(as_text=True))
            g.sanitized_body = sanitized_body
            # Replace request data
            request._cached_json = None
            request.get_json.cache_clear()

    def _validate_request(self):
        """Validate request based on endpoint"""
        if request.endpoint:
            validator = self._get_validator_for_endpoint(request.endpoint)
            if validator:
                validator()

    def _get_validator_for_endpoint(self, endpoint: str):
        """Get validator function for endpoint"""
        validators = {
            'tts.generate_audio': self._validate_tts_request,
            'tts.get_audio_requests': self._validate_pagination,
            'auth.register': self._validate_user_registration,
            'auth.login': self._validate_user_login
        }
        return validators.get(endpoint)

    def _validate_tts_request(self):
        """Validate TTS request"""
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")

        try:
            data = request.get_json()
            schema = TTSRequestSchema(**data)
            g.validated_data = schema.dict()
        except Exception as e:
            raise ValidationError(f"Invalid request data: {str(e)}")

    def _validate_pagination(self):
        """Validate pagination parameters"""
        try:
            params = request.args.to_dict()
            schema = PaginationSchema(**params)
            g.pagination_params = schema.dict()
        except Exception as e:
            raise ValidationError(f"Invalid pagination parameters: {str(e)}")

    def _validate_user_registration(self):
        """Validate user registration"""
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")

        try:
            data = request.get_json()
            schema = UserRegistrationSchema(**data)
            g.validated_data = schema.dict()
        except Exception as e:
            raise ValidationError(f"Invalid registration data: {str(e)}")

    def _validate_user_login(self):
        """Validate user login"""
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")

        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            raise ValidationError("Username and password are required")
```

## Security Validation

### SQL Injection Prevention
```python
# utils/validation/sql_validator.py
import re
from typing import Any, Dict, List

class SQLValidator:
    """SQL injection prevention utilities"""

    @staticmethod
    def validate_sql_identifier(identifier: str) -> bool:
        """Validate SQL identifier"""
        if not identifier or not isinstance(identifier, str):
            return False

        # Check for valid identifier pattern
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, identifier))

    @staticmethod
    def validate_order_by_clause(order_by: str) -> bool:
        """Validate ORDER BY clause"""
        if not order_by:
            return False

        # Split by comma and validate each field
        fields = [field.strip() for field in order_by.split(',')]
        for field in fields:
            if '.' in field:
                table, column = field.split('.', 1)
                if not SQLValidator.validate_sql_identifier(table) or not SQLValidator.validate_sql_identifier(column):
                    return False
            else:
                if not SQLValidator.validate_sql_identifier(field):
                    return False

        return True

    @staticmethod
    def validate_where_clause(where_clause: str) -> bool:
        """Validate WHERE clause (basic)"""
        if not where_clause:
            return True

        # Basic validation - check for dangerous patterns
        dangerous_patterns = [
            r';\s*(drop|delete|update|insert|create|alter)',
            r'union\s+select',
            r'--\s',
            r'/\*.*?\*/',
            r'xp_|sp_',
            r'exec\s*\(',
            r'script\s+'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, where_clause, re.IGNORECASE):
                return False

        return True

    @staticmethod
    def sanitize_table_name(table_name: str) -> str:
        """Sanitize table name"""
        if not SQLValidator.validate_sql_identifier(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        return table_name

    @staticmethod
    def sanitize_column_name(column_name: str) -> str:
        """Sanitize column name"""
        if not SQLValidator.validate_sql_identifier(column_name):
            raise ValueError(f"Invalid column name: {column_name}")
        return column_name
```

### XSS Prevention
```python
# utils/validation/xss_validator.py
from bleach import clean, ALLOWED_TAGS, ALLOWED_ATTRIBUTES
import re
from typing import Optional

class XSSValidator:
    """XSS prevention utilities"""

    @staticmethod
    def sanitize_html(html_content: str, allowed_tags: list = None) -> str:
        """Sanitize HTML content"""
        if allowed_tags is None:
            allowed_tags = ALLOWED_TAGS + ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3']

        return clean(html_content, tags=allowed_tags, attributes=ALLOWED_ATTRIBUTES)

    @staticmethod
    def is_safe_url(url: str) -> bool:
        """Check if URL is safe"""
        if not url:
            return False

        # Allow relative URLs
        if url.startswith('/'):
            return True

        # Allow http/https URLs
        if url.startswith(('http://', 'https://')):
            # Basic domain validation
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                domain = parsed.netloc.lower()

                # Block localhost and private IPs
                if domain.startswith(('localhost', '127.', '10.', '192.168.', '172.')):
                    return False

                return True
            except:
                return False

        return False

    @staticmethod
    def validate_css_value(css_value: str) -> bool:
        """Validate CSS value for safety"""
        if not css_value:
            return True

        # Check for dangerous CSS
        dangerous_patterns = [
            r'javascript:',
            r'expression\s*\(',
            r'vbscript:',
            r'on\w+\s*:'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, css_value, re.IGNORECASE):
                return False

        return True

    @staticmethod
    def sanitize_json_for_html(json_data: Any) -> str:
        """Sanitize JSON data for safe HTML output"""
        import json
        import html

        def sanitize_value(value):
            if isinstance(value, str):
                return html.escape(value)
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            else:
                return value

        sanitized = sanitize_value(json_data)
        return json.dumps(sanitized)
```

## Rate Limiting Integration

### Advanced Rate Limiting
```python
# utils/validation/rate_limit_validator.py
from flask_limiter.util import get_remote_address
from flask_jwt_extended import get_jwt_identity
from typing import Optional

class RateLimitValidator:
    """Advanced rate limiting with validation"""

    @staticmethod
    def get_user_identifier() -> str:
        """Get user identifier for rate limiting"""
        try:
            # Try JWT identity first
            user_id = get_jwt_identity()
            if user_id:
                return f"user:{user_id}"
        except:
            pass

        # Fall back to IP address
        return f"ip:{get_remote_address()}"

    @staticmethod
    def validate_request_size(request_size: int, max_size: int = 1024 * 1024) -> None:
        """Validate request size"""
        if request_size > max_size:
            raise ValidationError(f"Request size exceeds limit of {max_size} bytes")

    @staticmethod
    def validate_request_frequency(user_id: str, endpoint: str, window_seconds: int = 60) -> None:
        """Validate request frequency"""
        # Implementation using Redis for tracking
        pass

    @staticmethod
    def validate_content_length(content_length: Optional[int], max_length: int = 5000) -> None:
        """Validate content length"""
        if content_length is None:
            raise ValidationError("Content-Length header is required")

        if content_length > max_length:
            raise ValidationError(f"Content length exceeds maximum of {max_length} characters")
```

## Error Handling

### Validation Error Handler
```python
# app/error_handlers.py
from marshmallow import ValidationError as MarshmallowValidationError
from pydantic import ValidationError as PydanticValidationError
from utils.sanitization.request_sanitizer import RequestSanitizer

@app.errorhandler(PydanticValidationError)
def handle_pydantic_validation_error(error):
    """Handle Pydantic validation errors"""
    sanitized_errors = RequestSanitizer.sanitize_json_data(error.errors())

    response = {
        'error': 'VALIDATION_ERROR',
        'message': 'Input validation failed',
        'details': sanitized_errors,
        'timestamp': time.time()
    }

    return jsonify(response), 400

@app.errorhandler(MarshmallowValidationError)
def handle_marshmallow_validation_error(error):
    """Handle Marshmallow validation errors"""
    sanitized_errors = RequestSanitizer.sanitize_json_data(error.messages)

    response = {
        'error': 'VALIDATION_ERROR',
        'message': 'Input validation failed',
        'details': sanitized_errors,
        'timestamp': time.time()
    }

    return jsonify(response), 400
```

## Testing

### Validation Tests
```python
# tests/test_validation.py
import pytest
from utils.validation.models import TTSRequestSchema, UserRegistrationSchema

def test_tts_request_validation():
    """Test TTS request validation"""
    # Valid request
    valid_data = {
        'text': 'Hello, world!',
        'voice_name': 'Alnilam',
        'output_format': 'wav',
        'speed': 1.0,
        'pitch': 0.0
    }

    schema = TTSRequestSchema(**valid_data)
    assert schema.text == 'Hello, world!'
    assert schema.voice_name == 'Alnilam'

def test_tts_request_validation_invalid():
    """Test TTS request validation with invalid data"""
    # Text too long
    invalid_data = {
        'text': 'a' * 5001,
        'voice_name': 'Alnilam'
    }

    with pytest.raises(Exception):
        TTSRequestSchema(**invalid_data)

def test_user_registration_validation():
    """Test user registration validation"""
    # Valid registration
    valid_data = {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'Password123'
    }

    schema = UserRegistrationSchema(**valid_data)
    assert schema.username == 'testuser'
    assert schema.email == 'test@example.com'

def test_user_registration_validation_invalid():
    """Test user registration validation with invalid data"""
    # Weak password
    invalid_data = {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': '123456'
    }

    with pytest.raises(Exception):
        UserRegistrationSchema(**invalid_data)
```

### Sanitization Tests
```python
# tests/test_sanitization.py
from utils.sanitization.text_sanitizer import TextSanitizer

def test_text_sanitization():
    """Test text sanitization"""
    # Test XSS prevention
    malicious_text = '<script>alert("XSS")</script>Hello world'
    sanitized = TextSanitizer.sanitize_text(malicious_text)
    assert '<script>' not in sanitized
    assert 'alert("XSS")' not in sanitized

def test_filename_sanitization():
    """Test filename sanitization"""
    # Test path traversal prevention
    malicious_filename = '../../../etc/passwd'
    sanitized = TextSanitizer.sanitize_filename(malicious_filename)
    assert '..' not in sanitized
    assert '/' not in sanitized

def test_sql_sanitization():
    """Test SQL injection prevention"""
    malicious_sql = "'; DROP TABLE users; --"
    sanitized = TextSanitizer.sanitize_sql_input(malicious_sql)
    assert 'DROP TABLE' not in sanitized
    assert ';' not in sanitized
```

## Configuration

### Validation Settings
```python
# config/validation_config.py
class ValidationConfig:
    """Validation configuration"""

    # Text limits
    MAX_TEXT_LENGTH = 5000
    MAX_WORD_LENGTH = 100

    # File limits
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_FILENAME_LENGTH = 255

    # Request limits
    MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
    MAX_JSON_DEPTH = 10

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 100
    MAX_TTS_REQUESTS_PER_MINUTE = 10

    # Content validation
    ALLOWED_HTML_TAGS = ['p', 'br', 'strong', 'em', 'u']
    BLOCKED_DOMAINS = ['localhost', '127.0.0.1', '10.0.0.0/8']

    # Password policy
    MIN_PASSWORD_LENGTH = 8
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL_CHARS = False
    COMMON_PASSWORDS = ['password', '123456', 'qwerty']
```

### Environment Variables
```bash
# Validation settings
MAX_TEXT_LENGTH=5000
MAX_FILE_SIZE=52428800
ENABLE_STRICT_VALIDATION=true
LOG_VALIDATION_ERRORS=true

# Security settings
BLOCK_SUSPICIOUS_REQUESTS=true
SANITIZE_ALL_INPUTS=true
VALIDATE_CONTENT_TYPE=true
```

## Best Practices

### 1. Input Validation
- Validate all inputs on entry points
- Use whitelisting instead of blacklisting
- Validate data types, formats, and ranges
- Implement proper error messages

### 2. Sanitization
- Sanitize all user inputs before processing
- Use appropriate sanitization for different contexts
- Don't rely solely on client-side validation
- Implement defense in depth

### 3. Security
- Prevent injection attacks (SQL, XSS, etc.)
- Validate file uploads thoroughly
- Implement rate limiting
- Use secure defaults

### 4. Performance
- Validate early to fail fast
- Use efficient validation libraries
- Cache validation schemas when possible
- Avoid redundant validation

### 5. Error Handling
- Provide clear error messages without exposing internals
- Log validation failures for monitoring
- Handle validation errors gracefully
- Implement proper error responses

## Deployment

### Production Configuration
```python
# Production validation settings
PRODUCTION_VALIDATION_CONFIG = {
    'strict_mode': True,
    'log_all_validation_errors': True,
    'enable_request_size_limits': True,
    'enable_rate_limiting': True,
    'sanitize_all_inputs': True,
    'validate_content_types': True
}
```

### Monitoring
```python
# Validation monitoring
VALIDATION_FAILURES = Counter(
    'validation_failures_total',
    'Total validation failures',
    ['validation_type', 'field']
)

VALIDATION_SUCCESS = Counter(
    'validation_success_total',
    'Total successful validations',
    ['validation_type']
)
```

## Summary

Input validation và sanitization system provides:

1. **Security**: Protection against injection attacks and malicious input
2. **Data Integrity**: Ensuring data conforms to expected formats
3. **Error Prevention**: Catching invalid data early in the process
4. **Compliance**: Meeting security and data quality requirements
5. **Monitoring**: Comprehensive logging and metrics for validation

Key components:
- Pydantic models for schema validation
- Text sanitization utilities
- Request validation middleware
- Security validation for SQL/XSS prevention
- Rate limiting integration
- Comprehensive error handling