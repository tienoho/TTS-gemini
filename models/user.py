"""
User model for Flask TTS API with production-ready features
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import bcrypt
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Float, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from werkzeug.security import check_password_hash, generate_password_hash

Base = declarative_base()


class User(Base):
    """User model for storing user information with production features."""

    __tablename__ = 'users'

    # Primary key and basic info
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    # Account status
    is_active = Column(Boolean, default=True, index=True)
    is_premium = Column(Boolean, default=False, index=True)
    account_tier = Column(String(20), default="free", index=True)  # free, basic, premium, enterprise

    # API authentication
    api_key = Column(String(255), unique=True, index=True, nullable=True)  # bcrypt hashed
    api_key_expires_at = Column(DateTime, nullable=True)
    api_key_last_used = Column(DateTime, nullable=True)

    # Usage limits and quotas
    monthly_request_limit = Column(Integer, default=1000)
    monthly_storage_limit = Column(Integer, default=100000000)  # bytes
    current_month_requests = Column(Integer, default=0)
    current_month_storage = Column(Integer, default=0)
    month_reset_date = Column(DateTime, nullable=True)

    # Cost tracking
    total_cost = Column(Float, default=0.0)
    monthly_cost = Column(Float, default=0.0)

    # Preferences
    timezone = Column(String(50), default="UTC")
    language = Column(String(10), default="vi")
    preferences = Column(JSON, default=dict)

    # Security
    last_login_at = Column(DateTime, nullable=True)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255), nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    updated_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Relationships
    audio_requests = relationship("AudioRequest", back_populates="user", cascade="all, delete-orphan")
    logs = relationship("RequestLog", back_populates="user", cascade="all, delete-orphan")
    rate_limits = relationship("RateLimit", back_populates="user", cascade="all, delete-orphan")

    # Self-referential relationships
    created_by_user = relationship("User", remote_side=[id], foreign_keys=[created_by])
    updated_by_user = relationship("User", remote_side=[id], foreign_keys=[updated_by])

    # Indexes for performance
    __table_args__ = (
        Index('idx_users_email_active', 'email', 'is_active'),
        Index('idx_users_api_key_expires', 'api_key_expires_at'),
        Index('idx_users_account_tier', 'account_tier'),
        Index('idx_users_month_reset', 'month_reset_date'),
    )

    def __init__(self, username: str, email: str, password: str, **kwargs):
        """Initialize user with hashed password."""
        self.username = username
        self.email = email
        self.password_hash = generate_password_hash(password)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of user."""
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

    def set_password(self, password: str) -> None:
        """Set user password (hashed)."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Check if provided password matches user's password."""
        return check_password_hash(self.password_hash, password)

    def generate_api_key(self, expires_at: Optional[datetime] = None) -> str:
        """Generate and set API key for user."""
        # Generate secure API key
        api_key = f"sk-{secrets.token_urlsafe(32)}"

        # Hash with bcrypt for security
        salt = bcrypt.gensalt()
        self.api_key = bcrypt.hashpw(api_key.encode('utf-8'), salt).decode('utf-8')
        self.api_key_expires_at = expires_at

        return api_key  # Return original key

    def verify_api_key(self, api_key: str) -> bool:
        """Verify if provided API key matches user's API key."""
        if not self.api_key or not api_key:
            return False

        # Verify using bcrypt
        try:
            return bcrypt.checkpw(api_key.encode('utf-8'), self.api_key.encode('utf-8'))
        except Exception:
            return False

    def is_api_key_expired(self) -> bool:
        """Check if user's API key has expired."""
        if not self.api_key_expires_at:
            return False
        return datetime.utcnow() > self.api_key_expires_at

    def rotate_api_key(self) -> str:
        """Generate new API key and invalidate old one."""
        # Invalidate old key by setting expiration to now
        self.api_key_expires_at = datetime.utcnow()

        # Generate new key
        return self.generate_api_key()

    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_active': self.is_active,
            'is_premium': self.is_premium,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_dict_with_api_key(self) -> dict:
        """Convert user to dictionary including API key info."""
        data = self.to_dict()
        data.update({
            'api_key_expires_at': self.api_key_expires_at.isoformat() if self.api_key_expires_at else None,
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Create user instance from dictionary."""
        return cls(
            username=data['username'],
            email=data['email'],
            password=data['password']
        )

    @staticmethod
    def get_by_username(username: str, db_session) -> Optional['User']:
        """Get user by username using parameterized query."""
        return db_session.query(User).filter(User.username == username).first()

    @staticmethod
    def get_by_email(email: str, db_session) -> Optional['User']:
        """Get user by email using parameterized query."""
        return db_session.query(User).filter(User.email == email).first()

    @staticmethod
    def get_by_api_key(api_key: str, db_session) -> Optional['User']:
        """Get user by API key using bcrypt verification."""
        # Get all users with API keys (this is still not ideal but safer)
        users_with_keys = db_session.query(User).filter(User.api_key.isnot(None)).all()

        for user in users_with_keys:
            if user.verify_api_key(api_key) and not user.is_api_key_expired():
                return user

        return None

    @staticmethod
    def get_active_users(db_session, limit: int = 100) -> list:
        """Get active users."""
        return db_session.query(User).filter(User.is_active == True).limit(limit).all()

    @staticmethod
    def get_premium_users(db_session, limit: int = 100) -> list:
        """Get premium users."""
        return db_session.query(User).filter(
            User.is_active == True,
            User.is_premium == True
        ).limit(limit).all()