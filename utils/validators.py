"""
Data validation utilities for Flask TTS API
"""

import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pydantic.types import EmailStr


class UserRegistrationSchema(BaseModel):
    """Schema for user registration."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')

        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')

        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')

        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')

        return v


class UserLoginSchema(BaseModel):
    """Schema for user login."""

    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class TTSRequestSchema(BaseModel):
    """Schema for TTS generation request."""

    text: str = Field(..., min_length=1, max_length=5000)
    voice_name: str = Field(default="Alnilam", min_length=1, max_length=50)
    output_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=0.0, ge=-10.0, le=10.0)

    @validator('text')
    def validate_text(cls, v):
        """Validate and sanitize text input."""
        if not v.strip():
            raise ValueError('Text cannot be empty')

        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())

        return v

    @validator('voice_name')
    def validate_voice_name(cls, v):
        """Validate voice name."""
        # List of supported voices (can be expanded)
        supported_voices = [
            "Alnilam", "Spica", "Vega", "Sirius", "Rigel",
            "Betelgeuse", "Altair", "Aldebaran", "Antares", "Pollux"
        ]

        if v not in supported_voices:
            raise ValueError(f'Voice name must be one of: {", ".join(supported_voices)}')

        return v

    @validator('output_format')
    def validate_output_format(cls, v):
        """Validate output format."""
        allowed_formats = ['wav', 'mp3', 'ogg', 'flac']
        if v not in allowed_formats:
            raise ValueError(f'Output format must be one of: {", ".join(allowed_formats)}')
        return v


class TTSPaginationSchema(BaseModel):
    """Schema for TTS requests pagination."""

    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=10, ge=1, le=100)
    status: Optional[str] = Field(None)
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc")

    @validator('status')
    def validate_status(cls, v):
        """Validate status field."""
        if v is not None:
            allowed_statuses = ['pending', 'processing', 'completed', 'failed']
            if v not in allowed_statuses:
                raise ValueError(f'Status must be one of: {", ".join(allowed_statuses)}')
        return v

    @validator('sort_by')
    def validate_sort_by(cls, v):
        """Validate sort_by field."""
        allowed_fields = ['created_at', 'updated_at', 'text']
        if v not in allowed_fields:
            raise ValueError(f'Sort by must be one of: {", ".join(allowed_fields)}')
        return v

    @validator('sort_order')
    def validate_sort_order(cls, v):
        """Validate sort_order field."""
        allowed_orders = ['asc', 'desc']
        if v not in allowed_orders:
            raise ValueError(f'Sort order must be one of: {", ".join(allowed_orders)}')
        return v


class AudioFileUploadSchema(BaseModel):
    """Schema for audio file upload."""

    file: bytes = Field(...)
    filename: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(...)

    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename."""
        if not v or '/' in v or '\\' in v:
            raise ValueError('Invalid filename')

        # Check for dangerous file extensions
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
        if any(v.lower().endswith(ext) for ext in dangerous_extensions):
            raise ValueError('File type not allowed')

        return v

    @validator('mime_type')
    def validate_mime_type(cls, v):
        """Validate MIME type."""
        allowed_mime_types = ['audio/wav', 'audio/mp3', 'audio/ogg', 'audio/flac']
        if v not in allowed_mime_types:
            raise ValueError(f'MIME type must be one of: {", ".join(allowed_mime_types)}')
        return v

    @validator('file')
    def validate_file_size(cls, v):
        """Validate file size."""
        max_size = 10 * 1024 * 1024  # 10MB
        if len(v) > max_size:
            raise ValueError('File size exceeds 10MB limit')

        return v


class UserUpdateSchema(BaseModel):
    """Schema for user profile update."""

    email: Optional[EmailStr] = None
    current_password: Optional[str] = Field(None, min_length=8, max_length=128)
    new_password: Optional[str] = Field(None, min_length=8, max_length=128)

    @validator('new_password')
    def validate_new_password(cls, v, values):
        """Validate new password."""
        if v and 'current_password' not in values:
            raise ValueError('Current password is required to set new password')

        if v:
            if len(v) < 8:
                raise ValueError('Password must be at least 8 characters long')

            if not re.search(r'[A-Z]', v):
                raise ValueError('Password must contain at least one uppercase letter')

            if not re.search(r'[a-z]', v):
                raise ValueError('Password must contain at least one lowercase letter')

            if not re.search(r'\d', v):
                raise ValueError('Password must contain at least one digit')

        return v


class APIKeySchema(BaseModel):
    """Schema for API key operations."""

    name: str = Field(..., min_length=1, max_length=50)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)

    @validator('name')
    def validate_name(cls, v):
        """Validate API key name."""
        if not re.match(r'^[a-zA-Z0-9\s_-]+$', v):
            raise ValueError('Name can only contain letters, numbers, spaces, underscores, and hyphens')

        return v.strip()


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""

    status: str
    timestamp: str
    version: str
    database: str
    redis: str
    uptime: float


class ErrorResponse(BaseModel):
    """Schema for error response."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    path: str
    request_id: str


class PaginationMeta(BaseModel):
    """Schema for pagination metadata."""

    page: int
    per_page: int
    total_pages: int
    total_items: int
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel):
    """Generic schema for paginated response."""

    data: List[Any]
    meta: PaginationMeta


def validate_positive_int(value: Union[int, str], field_name: str) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Returns:
        Validated integer

    Raises:
        ValueError: If validation fails
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        return int_value
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid integer")


def validate_range(value: Union[int, float], min_val: Union[int, float],
                  max_val: Union[int, float], field_name: str) -> Union[int, float]:
    """Validate that a value is within a specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of the field for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If validation fails
    """
    try:
        num_value = float(value)
        if not (min_val <= num_value <= max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        return num_value
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid number")


def validate_enum(value: str, allowed_values: List[str], field_name: str) -> str:
    """Validate that a value is in a list of allowed values.

    Args:
        value: Value to validate
        allowed_values: List of allowed values
        field_name: Name of the field for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If validation fails
    """
    if value not in allowed_values:
        raise ValueError(f"{field_name} must be one of: {', '.join(allowed_values)}")

    return value


def sanitize_html(text: str) -> str:
    """Enhanced HTML sanitization for text content.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return text

    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove other potentially dangerous tags
    dangerous_tags = [
        'iframe', 'object', 'embed', 'form', 'input', 'button',
        'style', 'link', 'meta', 'base', 'applet', 'frame', 'frameset'
    ]
    for tag in dangerous_tags:
        text = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove event handlers (on*)
    text = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)

    # Remove javascript: URLs
    text = re.sub(r'javascript:[^"\'\s]*', '', text, flags=re.IGNORECASE)

    # Remove data: URLs (except safe ones)
    text = re.sub(r'data:[^;]*text/plain[^;]*;[^,]*', '', text, flags=re.IGNORECASE)

    # Remove potentially dangerous attributes
    dangerous_attrs = ['onclick', 'onload', 'onerror', 'onmouseover', 'onmouseout']
    for attr in dangerous_attrs:
        text = re.sub(f'{attr}\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)

    # Escape HTML entities
    text = text.replace('&', '&')
    text = text.replace('<', '<')
    text = text.replace('>', '>')
    text = text.replace('"', '"')
    text = text.replace("'", '&#x27;')

    return text.strip()


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages to prevent XSS.

    Args:
        message: Error message to sanitize

    Returns:
        Sanitized error message
    """
    if not message:
        return message

    # Remove HTML tags
    message = re.sub(r'<[^>]+>', '', message)

    # Remove script-like content
    message = re.sub(r'javascript:', '', message, flags=re.IGNORECASE)
    message = re.sub(r'on\w+\s*=', '', message, flags=re.IGNORECASE)

    # Limit length to prevent DoS
    if len(message) > 500:
        message = message[:500] + '...'

    return message.strip()


def sanitize_user_agent(user_agent: str) -> str:
    """Sanitize user agent string.

    Args:
        user_agent: User agent string to sanitize

    Returns:
        Sanitized user agent string
    """
    if not user_agent:
        return user_agent

    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"]', '', user_agent)

    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200] + '...'

    return sanitized.strip()