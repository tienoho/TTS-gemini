"""
Security utilities for Flask TTS API
"""

import hashlib
import os
import re
import secrets
from typing import Any, Dict, List, Optional, Set

import bcrypt
from flask import Request


class SecurityUtils:
    """Security utility functions."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash.

        Args:
            password: Plain text password
            hashed: Hashed password

        Returns:
            True if password matches
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a secure random token.

        Args:
            length: Token length in bytes

        Returns:
            Secure random token as hex string
        """
        return secrets.token_hex(length)

    @staticmethod
    def sanitize_input(text: str, max_length: int = 5000) -> str:
        """Sanitize user input to prevent XSS and other attacks.

        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""

        # Remove null bytes
        text = text.replace('\x00', '')

        # Remove control characters except newlines and tabs
        # Use simple character-by-character approach to prevent ReDoS
        result = []
        for char in text:
            # Allow printable characters, newlines, and tabs
            if char == '\n' or char == '\t' or (' ' <= char <= '~'):
                result.append(char)
            # Remove control characters
        text = ''.join(result)

        # Limit length
        if len(text) > max_length:
            text = text[:max_length]

        return text.strip()

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format.

        Args:
            email: Email address to validate

        Returns:
            True if email is valid
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format.

        Args:
            username: Username to validate

        Returns:
            True if username is valid
        """
        if not isinstance(username, str) or len(username) < 3 or len(username) > 50:
            return False

        # Allow alphanumeric characters, underscores, and hyphens
        pattern = r'^[a-zA-Z0-9_-]+$'
        return re.match(pattern, username) is not None

    @staticmethod
    def is_sql_injection_safe(text: str) -> bool:
        """Check if text contains potential SQL injection patterns.

        Args:
            text: Text to check

        Returns:
            True if text appears safe
        """
        if not isinstance(text, str):
            return False

        dangerous_patterns = [
            # Basic SQL keywords
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT|TRUNCATE)\b)',
            # Comments
            r'(--|#|/\*|\*/)',
            # Tautology patterns
            r'(\bor\b\s+\d+\s*=\s*\d+)',
            r'(\band\b\s+\d+\s*=\s*\d+)',
            # Time-based attacks
            r'(\bwaitfor\b\s+delay)',
            r'(\bsleep\b\s*\()',
            r'(\bbenchmark\b\s*\()',
            # System commands
            r'(\bxp_cmdshell\b)',
            r'(\bsp_configure\b)',
            # Semicolon followed by SQL commands
            r'(;\s*(select|insert|update|delete|drop|create|alter|exec|union|truncate))',
            # Stacked queries
            r'(\bgo\b\s*;?\s*(select|insert|update|delete|drop|create|alter|exec|union))',
            # Hex encoding
            r'(0x[0-9a-fA-F]{2,})',
            # Information schema access
            r'(\binformation_schema\b)',
            r'(\bsys\.objects\b)',
            r'(\bsys\.tables\b)',
            # Dangerous functions
            r'(\bload_file\b)',
            r'(\boutfile\b)',
            # Case variations
            r'(\bselect\b.*\bfrom\b)',
            r'(\bunion\b.*\bselect\b)',
        ]

        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                return False

        # Additional checks for suspicious patterns
        suspicious_chars = [';', '--', '/*', '*/', 'xp_', 'sp_', '0x']
        for char in suspicious_chars:
            if char in text_lower:
                # More context-aware checking
                if char == ';':
                    # Allow semicolons in reasonable contexts (not followed by SQL keywords)
                    if re.search(r';\s*(select|insert|update|delete|drop|create|alter|exec|union)\b', text_lower, re.IGNORECASE):
                        return False
                elif char in ['--', '/*', '*/']:
                    # Comments are generally suspicious in input
                    return False
                elif char in ['xp_', 'sp_']:
                    # System stored procedures
                    return False
                elif char == '0x':
                    # Hex encoding might be used to bypass filters
                    if len(re.findall(r'0x[0-9a-fA-F]{4,}', text_lower)) > 0:
                        return False

        return True

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)

        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)

        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext

        return sanitized.strip()

    @staticmethod
    def check_file_signature(file_data: bytes, allowed_types: Set[str]) -> bool:
        """Check file signature against allowed types.

        Args:
            file_data: File data as bytes
            allowed_types: Set of allowed MIME types

        Returns:
            True if file signature is allowed
        """
        # File signatures (magic bytes)
        signatures = {
            'audio/wav': [b'RIFF'],
            'audio/mpeg': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
            'audio/ogg': [b'OggS'],
            'audio/flac': [b'fLaC'],
        }

        if not file_data:
            return False

        # Check against allowed signatures
        for allowed_type in allowed_types:
            if allowed_type in signatures:
                for signature in signatures[allowed_type]:
                    if file_data.startswith(signature):
                        return True

        return False

    @staticmethod
    def rate_limit_key_func(request: Request) -> str:
        """Generate rate limit key based on user identity.

        Args:
            request: Flask request object

        Returns:
            Rate limit key
        """
        # Use user ID if authenticated, otherwise use IP
        user_id = getattr(request, 'current_user', None)
        if user_id:
            return f"user:{user_id}"
        else:
            # Use client IP with salt to prevent enumeration
            client_ip = request.remote_addr or 'unknown'
            return f"ip:{hashlib.md5(client_ip.encode()).hexdigest()}"

    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Mask sensitive fields in data dictionary.

        Args:
            data: Data dictionary
            sensitive_fields: List of field names to mask

        Returns:
            Dictionary with sensitive fields masked
        """
        masked_data = data.copy()

        for field in sensitive_fields:
            if field in masked_data:
                value = masked_data[field]
                if isinstance(value, str):
                    if len(value) > 8:
                        masked_data[field] = value[:4] + '*' * (len(value) - 8) + value[-4:]
                    else:
                        masked_data[field] = '*' * len(value)
                else:
                    masked_data[field] = '***'

        return masked_data

    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key.

        Returns:
            API key as string
        """
        return 'sk-' + secrets.token_urlsafe(32)

    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Validate API key format.

        Args:
            api_key: API key to validate

        Returns:
            True if format is valid
        """
        pattern = r'^sk-[a-zA-Z0-9_-]{43}$'
        return re.match(pattern, api_key) is not None

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """Compare two strings in constant time to prevent timing attacks.

        Args:
            a: First string
            b: Second string

        Returns:
            True if strings are equal
        """
        if len(a) != len(b):
            return False

        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)

        return result == 0