"""
Custom exceptions for TTS system
"""

from typing import Optional, Dict, Any


class TTSException(Exception):
    """Base exception for TTS system."""

    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class GeminiAPIException(TTSException):
    """Exception for Gemini API errors."""

    def __init__(self, message: str, status_code: int = 500, gemini_error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code, details)
        self.gemini_error_code = gemini_error_code


class GeminiQuotaExceededException(GeminiAPIException):
    """Exception for Gemini API quota exceeded errors."""

    def __init__(self, message: str = "Gemini API quota exceeded",
                 retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 429, "QUOTA_EXCEEDED", details)
        self.retry_after = retry_after


class GeminiRateLimitException(GeminiAPIException):
    """Exception for Gemini API rate limiting errors."""

    def __init__(self, message: str = "Gemini API rate limit exceeded",
                 retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 429, "RATE_LIMIT_EXCEEDED", details)
        self.retry_after = retry_after


class GeminiInvalidRequestException(GeminiAPIException):
    """Exception for invalid requests to Gemini API."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, "INVALID_REQUEST", details)


class GeminiAuthenticationException(GeminiAPIException):
    """Exception for Gemini API authentication errors."""

    def __init__(self, message: str = "Gemini API authentication failed",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 401, "AUTHENTICATION_FAILED", details)


class GeminiModelNotFoundException(GeminiAPIException):
    """Exception for model not found errors."""

    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"Gemini model '{model_name}' not found"
        super().__init__(message, 404, "MODEL_NOT_FOUND", details)


class GeminiNetworkException(GeminiAPIException):
    """Exception for network-related errors."""

    def __init__(self, message: str = "Network error occurred while calling Gemini API",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 503, "NETWORK_ERROR", details)


class GeminiTimeoutException(GeminiAPIException):
    """Exception for timeout errors."""

    def __init__(self, timeout: int, details: Optional[Dict[str, Any]] = None):
        message = f"Gemini API request timed out after {timeout} seconds"
        super().__init__(message, 504, "TIMEOUT", details)


class AudioProcessingException(TTSException):
    """Exception for audio processing errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 500, details)


class AudioFormatException(AudioProcessingException):
    """Exception for audio format errors."""

    def __init__(self, message: str, supported_formats: Optional[list] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.supported_formats = supported_formats


class FileStorageException(TTSException):
    """Exception for file storage errors."""

    def __init__(self, message: str, file_path: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 500, details)
        self.file_path = file_path


class ValidationException(TTSException):
    """Exception for input validation errors."""

    def __init__(self, message: str, field: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, details)
        self.field = field


class CircuitBreakerException(TTSException):
    """Exception for circuit breaker activation."""

    def __init__(self, message: str = "Circuit breaker is open",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 503, details)