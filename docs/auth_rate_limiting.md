# JWT Authentication và Rate Limiting

## Tổng quan
Hệ thống authentication sử dụng JWT (JSON Web Tokens) với rate limiting để bảo mật API và ngăn chặn abuse.

## JWT Authentication

### Configuration
```python
# config/development.py
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-secret-key-change-in-production')
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
JWT_ALGORITHM = 'HS256'
```

### User Registration & Login
```python
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, create_refresh_token
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.get_json()

    # Validate input
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing required fields'}), 400

    # Check if user exists
    if User.get_by_username(data['username'], db.session):
        return jsonify({'error': 'Username already exists'}), 409

    if User.get_by_email(data['email'], db.session):
        return jsonify({'error': 'Email already exists'}), 409

    # Create user
    user = User(
        username=data['username'],
        email=data['email'],
        password=data['password']
    )

    db.session.add(user)
    db.session.commit()

    # Generate API key for user
    api_key = user.generate_api_key()

    return jsonify({
        'message': 'User created successfully',
        'user_id': user.id,
        'api_key': api_key
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()

    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password required'}), 400

    # Get user
    user = User.get_by_username(data['username'], db.session)
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401

    if not user.is_active:
        return jsonify({'error': 'Account is disabled'}), 401

    # Generate tokens
    access_token = create_access_token(identity=user.id)
    refresh_token = create_refresh_token(identity=user.id)

    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict()
    }), 200
```

### JWT Middleware
```python
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, get_jwt
from functools import wraps

jwt = JWTManager()

def init_jwt(app):
    jwt.init_app(app)

    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        """Check if token is revoked"""
        jti = jwt_payload['jti']
        return jti in revoked_tokens

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({'error': 'Token has expired'}), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({'error': 'Invalid token'}), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({'error': 'Access token required'}), 401

def require_premium():
    """Decorator to require premium account"""
    def decorator(f):
        @wraps(f)
        @jwt_required()
        def decorated_function(*args, **kwargs):
            user_id = get_jwt_identity()
            user = User.query.get(user_id)

            if not user.is_premium:
                return jsonify({'error': 'Premium account required'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

### API Key Authentication
```python
from flask import request
from functools import wraps

def require_api_key():
    """Decorator to require API key authentication"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')

            if not api_key:
                return jsonify({'error': 'API key required'}), 401

            user = User.get_by_api_key(api_key, db.session)
            if not user:
                return jsonify({'error': 'Invalid API key'}), 401

            if user.is_api_key_expired():
                return jsonify({'error': 'API key expired'}), 401

            # Set user context
            request.user_id = user.id
            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

## Rate Limiting

### Configuration
```python
# config/development.py
RATELIMIT_DEFAULT = "100 per minute"
RATELIMIT_STORAGE_URL = "redis://localhost:6379/1"

# Rate limits by user type
RATELIMIT_PREMIUM = "1000 per minute"
RATELIMIT_STANDARD = "100 per minute"
RATELIMIT_TTS_GENERATE = "10 per minute"
```

### Implementation
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

def get_user_id():
    """Get user ID for rate limiting"""
    try:
        from flask_jwt_extended import get_jwt_identity
        return get_jwt_identity()
    except:
        return get_remote_address()

limiter = Limiter(
    key_func=get_user_id,
    storage_uri="redis://localhost:6379/1",
    default_limits=["100 per minute"]
)

def init_rate_limiting(app):
    limiter.init_app(app)

# Apply rate limiting to routes
@tts_bp.route('/generate', methods=['POST'])
@limiter.limit("10 per minute", key_func=get_user_id)
@jwt_required()
def generate_audio():
    """Generate audio with rate limiting"""
    # Implementation
    pass
```

### Advanced Rate Limiting
```python
class AdvancedRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def check_limit(self, user_id: str, endpoint: str, limit: int, window_seconds: int = 60):
        """Check if request is within rate limit"""
        window_key = f"rate_limit:{user_id}:{endpoint}"
        current_time = int(time.time())

        # Clean old entries
        self.redis.zremrangebyscore(window_key, 0, current_time - window_seconds)

        # Count requests in current window
        request_count = self.redis.zcard(window_key)

        if request_count >= limit:
            return False, self._get_remaining_time(window_key, window_seconds)

        # Add current request
        self.redis.zadd(window_key, {str(current_time): current_time})
        self.redis.expire(window_key, window_seconds)

        return True, 0

    def _get_remaining_time(self, window_key: str, window_seconds: int) -> int:
        """Get remaining time until rate limit resets"""
        oldest = self.redis.zrange(window_key, 0, 0, withscores=True)
        if not oldest:
            return 0

        return max(0, window_seconds - (int(time.time()) - int(oldest[0][1])))

    def get_user_limits(self, user_id: str) -> dict:
        """Get current rate limit status for user"""
        limits = {}
        endpoints = ['/api/v1/tts/generate', '/api/v1/tts/status']

        for endpoint in endpoints:
            window_key = f"rate_limit:{user_id}:{endpoint}"
            current_count = self.redis.zcard(window_key)
            limits[endpoint] = {
                'current_count': current_count,
                'limit': self._get_endpoint_limit(endpoint),
                'remaining': max(0, self._get_endpoint_limit(endpoint) - current_count)
            }

        return limits

    def _get_endpoint_limit(self, endpoint: str) -> int:
        """Get rate limit for specific endpoint"""
        limits = {
            '/api/v1/tts/generate': 10,
            '/api/v1/tts/status': 100,
        }
        return limits.get(endpoint, 100)
```

## Security Features

### Password Security
```python
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt

class PasswordManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return check_password_hash(hashed_password, password)

    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        import secrets
        return f"sk-{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
```

### Token Management
```python
class TokenManager:
    def __init__(self, redis_client):
        self.redis = redis_client

    def revoke_token(self, jti: str):
        """Revoke JWT token"""
        self.redis.sadd('jwt_blacklist', jti)

    def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        return self.redis.sismember('jwt_blacklist', jti)

    def cleanup_expired_tokens(self):
        """Clean up expired tokens from blacklist"""
        # Implementation for cleanup
        pass

    def get_token_info(self, token: str) -> dict:
        """Get information about token"""
        try:
            decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return {
                'user_id': decoded.get('sub'),
                'exp': decoded.get('exp'),
                'jti': decoded.get('jti'),
                'is_expired': decoded.get('exp', 0) < int(time.time())
            }
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}
```

### Input Validation & Sanitization
```python
from marshmallow import Schema, fields, validates, ValidationError
import re

class TTSRequestSchema(Schema):
    text = fields.Str(required=True, validate=lambda x: len(x) <= 5000)
    voice_name = fields.Str(missing='Alnilam')
    output_format = fields.Str(missing='wav', validate=lambda x: x in ['wav', 'mp3'])
    speed = fields.Float(missing=1.0, validate=lambda x: 0.5 <= x <= 2.0)
    pitch = fields.Float(missing=0.0, validate=lambda x: -10.0 <= x <= 10.0)

    @validates('text')
    def validate_text(self, value):
        """Validate text content"""
        if not value.strip():
            raise ValidationError('Text cannot be empty')

        # Check for potentially harmful content
        if len(value) > 5000:
            raise ValidationError('Text too long')

        # Basic XSS prevention
        dangerous_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
        for pattern in dangerous_patterns:
            if pattern.lower() in value.lower():
                raise ValidationError('Text contains potentially harmful content')

class SecurityUtils:
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        if not text:
            return text

        # Remove null bytes
        text = text.replace('\x00', '')

        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for security"""
        if not filename:
            return False

        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False

        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in dangerous_chars):
            return False

        return True
```

## API Endpoints

### Authentication Endpoints
```python
@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh_token():
    """Refresh access token"""
    user_id = get_jwt_identity()
    new_access_token = create_access_token(identity=user_id)
    return jsonify({'access_token': new_access_token}), 200

@auth_bp.route('/revoke', methods=['POST'])
@jwt_required()
def revoke_token():
    """Revoke current token"""
    jti = get_jwt()['jti']
    token_manager.revoke_token(jti)
    return jsonify({'message': 'Token revoked'}), 200

@auth_bp.route('/api-key', methods=['POST'])
@jwt_required()
def generate_api_key():
    """Generate new API key"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    api_key = user.generate_api_key()
    db.session.commit()

    return jsonify({'api_key': api_key}), 200
```

### Protected TTS Endpoints
```python
@tts_bp.route('/generate', methods=['POST'])
@limiter.limit("10 per minute")
@jwt_required()
def generate_audio():
    """Generate audio with authentication and rate limiting"""
    user_id = get_jwt_identity()

    # Check user permissions
    user = User.query.get(user_id)
    if not user.is_active:
        return jsonify({'error': 'Account disabled'}), 403

    # Process request
    # Implementation here

    return jsonify({'message': 'Audio generation started'}), 202
```

## Monitoring & Logging

### Authentication Events
```python
class AuthLogger:
    def __init__(self, logger):
        self.logger = logger

    def log_login(self, user_id: str, ip_address: str, user_agent: str, success: bool):
        """Log login attempts"""
        log_data = {
            'event': 'login',
            'user_id': user_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        }

        if success:
            self.logger.info(f"Successful login: {user_id}", extra=log_data)
        else:
            self.logger.warning(f"Failed login attempt: {user_id}", extra=log_data)

    def log_api_key_usage(self, user_id: str, endpoint: str, ip_address: str):
        """Log API key usage"""
        log_data = {
            'event': 'api_key_usage',
            'user_id': user_id,
            'endpoint': endpoint,
            'ip_address': ip_address,
            'timestamp': datetime.utcnow().isoformat()
        }

        self.logger.info(f"API key used: {user_id}", extra=log_data)
```

### Rate Limiting Events
```python
class RateLimitLogger:
    def __init__(self, logger):
        self.logger = logger

    def log_rate_limit_exceeded(self, user_id: str, endpoint: str, limit: int):
        """Log rate limit exceeded"""
        log_data = {
            'event': 'rate_limit_exceeded',
            'user_id': user_id,
            'endpoint': endpoint,
            'limit': limit,
            'timestamp': datetime.utcnow().isoformat()
        }

        self.logger.warning(f"Rate limit exceeded: {user_id}", extra=log_data)
```

## Error Handling

### Authentication Errors
```python
@jwt.invalid_token_loader
def invalid_token_callback(error):
    """Handle invalid tokens"""
    return jsonify({
        'error': 'Authentication failed',
        'message': 'Invalid or malformed token',
        'code': 'INVALID_TOKEN'
    }), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    """Handle expired tokens"""
    return jsonify({
        'error': 'Authentication failed',
        'message': 'Token has expired',
        'code': 'TOKEN_EXPIRED'
    }), 401
```

### Rate Limiting Errors
```python
@limiter.request_filter
def rate_limit_handler(request):
    """Handle rate limit exceeded"""
    if limiter.current_limit and limiter.current_limit.exceeded:
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': f'Too many requests. Limit: {limiter.current_limit.limit}',
            'retry_after': limiter.current_limit.reset_time
        }), 429
```

## Testing

### Authentication Tests
```python
import pytest
from flask_jwt_extended import create_access_token

def test_jwt_authentication(client):
    """Test JWT authentication"""
    # Register user
    response = client.post('/api/v1/auth/register', json={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'password123'
    })
    assert response.status_code == 201

    # Login
    response = client.post('/api/v1/auth/login', json={
        'username': 'testuser',
        'password': 'password123'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'access_token' in data

    # Test protected endpoint
    headers = {'Authorization': f'Bearer {data["access_token"]}'}
    response = client.get('/api/v1/tts/', headers=headers)
    assert response.status_code == 200
```

### Rate Limiting Tests
```python
def test_rate_limiting(client, auth_headers):
    """Test rate limiting"""
    # Make requests up to limit
    for i in range(10):
        response = client.post('/api/v1/tts/generate',
                             headers=auth_headers,
                             json={'text': 'test'})
        assert response.status_code == 202

    # Next request should be rate limited
    response = client.post('/api/v1/tts/generate',
                         headers=auth_headers,
                         json={'text': 'test'})
    assert response.status_code == 429
    data = response.get_json()
    assert 'retry_after' in data