"""
Organization models for multi-tenant TTS system
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Float, JSON, Index, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class OrganizationStatus(str, Enum):
    """Organization status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"


class OrganizationTier(str, Enum):
    """Organization tier enumeration."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class MemberRole(str, Enum):
    """Organization member role enumeration."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Organization(Base):
    """Organization model for multi-tenant architecture."""

    __tablename__ = 'organizations'

    # Primary key and basic info
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    slug = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)

    # Status and tier
    status = Column(SQLEnum(OrganizationStatus), default=OrganizationStatus.ACTIVE, index=True)
    tier = Column(SQLEnum(OrganizationTier), default=OrganizationTier.FREE, index=True)

    # Contact information
    email = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)
    website = Column(String(255), nullable=True)

    # Address
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)

    # Billing configuration
    billing_email = Column(String(100), nullable=True)
    tax_id = Column(String(50), nullable=True)
    currency = Column(String(3), default="USD")

    # Resource limits
    max_users = Column(Integer, default=5)  # Maximum number of users
    max_monthly_requests = Column(Integer, default=10000)
    max_storage_bytes = Column(Integer, default=1000000000)  # 1GB
    max_concurrent_requests = Column(Integer, default=10)

    # Current usage
    current_users = Column(Integer, default=0)
    current_month_requests = Column(Integer, default=0)
    current_storage_bytes = Column(Integer, default=0)
    current_month_cost = Column(Float, default=0.0)

    # Cost tracking
    total_cost = Column(Float, default=0.0)
    monthly_cost = Column(Float, default=0.0)

    # Settings and preferences
    settings = Column(JSON, default=dict)
    preferences = Column(JSON, default=dict)

    # Security
    api_key = Column(String(255), unique=True, index=True, nullable=True)
    api_key_expires_at = Column(DateTime, nullable=True)
    webhook_secret = Column(String(255), nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    updated_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Relationships
    members = relationship("OrganizationMember", back_populates="organization", cascade="all, delete-orphan")
    resources = relationship("OrganizationResource", back_populates="organization", cascade="all, delete-orphan")
    billing_records = relationship("OrganizationBilling", back_populates="organization", cascade="all, delete-orphan")
    usage_records = relationship("OrganizationUsage", back_populates="organization", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_orgs_status_tier', 'status', 'tier'),
        Index('idx_orgs_created_at', 'created_at'),
        Index('idx_orgs_api_key_expires', 'api_key_expires_at'),
    )

    def __init__(self, name: str, slug: str, email: str, **kwargs):
        """Initialize organization."""
        self.name = name
        self.slug = slug
        self.email = email
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of organization."""
        return f"<Organization(id={self.id}, name='{self.name}', slug='{self.slug}')>"

    def generate_api_key(self, expires_at: Optional[datetime] = None) -> str:
        """Generate and set API key for organization."""
        api_key = f"org-{secrets.token_urlsafe(32)}"
        self.api_key = api_key  # Store plain for now, should be hashed in production
        self.api_key_expires_at = expires_at
        return api_key

    def is_api_key_expired(self) -> bool:
        """Check if organization's API key has expired."""
        if self.api_key_expires_at is None:
            return False
        return datetime.utcnow() > self.api_key_expires_at

    def rotate_api_key(self) -> str:
        """Generate new API key and invalidate old one."""
        self.api_key_expires_at = datetime.utcnow()
        return self.generate_api_key()

    def can_add_user(self) -> bool:
        """Check if organization can add more users."""
        return self.current_users < self.max_users

    def can_make_request(self) -> bool:
        """Check if organization can make more requests."""
        return self.current_month_requests < self.max_monthly_requests

    def has_storage_capacity(self, additional_bytes: int) -> bool:
        """Check if organization has storage capacity."""
        return (self.current_storage_bytes + additional_bytes) <= self.max_storage_bytes

    def to_dict(self) -> dict:
        """Convert organization to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'status': self.status.value if self.status is not None else None,
            'tier': self.tier.value if self.tier is not None else None,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'current_users': self.current_users,
            'max_users': self.max_users,
            'current_month_requests': self.current_month_requests,
            'max_monthly_requests': self.max_monthly_requests,
            'current_storage_bytes': self.current_storage_bytes,
            'max_storage_bytes': self.max_storage_bytes,
            'current_month_cost': self.current_month_cost,
        }

    @staticmethod
    def get_by_slug(slug: str, db_session) -> Optional['Organization']:
        """Get organization by slug."""
        return db_session.query(Organization).filter(Organization.slug == slug).first()

    @staticmethod
    def get_by_api_key(api_key: str, db_session) -> Optional['Organization']:
        """Get organization by API key."""
        org = db_session.query(Organization).filter(Organization.api_key == api_key).first()
        if org and not org.is_api_key_expired():
            return org
        return None

    @staticmethod
    def get_active_organizations(db_session, limit: int = 100) -> list:
        """Get active organizations."""
        return db_session.query(Organization).filter(
            Organization.status == OrganizationStatus.ACTIVE
        ).limit(limit).all()


class OrganizationMember(Base):
    """Organization member model for managing organization membership."""

    __tablename__ = 'organization_members'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    organization_id = Column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), index=True)

    # Role and permissions
    role = Column(SQLEnum(MemberRole), default=MemberRole.MEMBER, index=True)
    permissions = Column(JSON, default=list)  # Custom permissions

    # Status
    is_active = Column(Boolean, default=True, index=True)
    invited_at = Column(DateTime, default=datetime.utcnow)
    joined_at = Column(DateTime, nullable=True)
    last_active_at = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    invited_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="members")
    user = relationship("User")
    invited_by_user = relationship("User", foreign_keys=[invited_by])

    # Unique constraint to prevent duplicate memberships
    __table_args__ = (
        Index('idx_org_member_org_user', 'organization_id', 'user_id', unique=True),
        Index('idx_org_member_role', 'role'),
    )

    def __repr__(self) -> str:
        """String representation of organization member."""
        return f"<OrganizationMember(id={self.id}, org_id={self.organization_id}, user_id={self.user_id}, role='{self.role.value}')>"

    def to_dict(self) -> dict:
        """Convert member to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'role': self.role.value,
            'is_active': self.is_active,
            'joined_at': self.joined_at.isoformat() if self.joined_at is not None else None,
            'last_active_at': self.last_active_at.isoformat() if self.last_active_at is not None else None,
        }

    @staticmethod
    def get_by_org_and_user(organization_id: int, user_id: int, db_session) -> Optional['OrganizationMember']:
        """Get organization member by organization and user ID."""
        return db_session.query(OrganizationMember).filter(
            OrganizationMember.organization_id == organization_id,
            OrganizationMember.user_id == user_id
        ).first()

    @staticmethod
    def get_members_by_organization(organization_id: int, db_session) -> list:
        """Get all members of an organization."""
        return db_session.query(OrganizationMember).filter(
            OrganizationMember.organization_id == organization_id,
            OrganizationMember.is_active == True
        ).all()

    @staticmethod
    def get_organizations_by_user(user_id: int, db_session) -> list:
        """Get all organizations where user is a member."""
        return db_session.query(OrganizationMember).filter(
            OrganizationMember.user_id == user_id,
            OrganizationMember.is_active == True
        ).all()


class OrganizationResource(Base):
    """Organization resource allocation model."""

    __tablename__ = 'organization_resources'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    organization_id = Column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), index=True)

    # Resource type and allocation
    resource_type = Column(String(50), nullable=False, index=True)  # 'cpu', 'memory', 'storage', 'requests'
    resource_name = Column(String(100), nullable=False)
    allocated_amount = Column(Float, nullable=False)
    used_amount = Column(Float, default=0.0)
    unit = Column(String(20), nullable=False)  # 'cores', 'bytes', 'requests', etc.

    # Limits
    soft_limit = Column(Float, nullable=True)  # Warning threshold
    hard_limit = Column(Float, nullable=True)  # Absolute maximum

    # Status
    is_active = Column(Boolean, default=True, index=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="resources")

    # Indexes
    __table_args__ = (
        Index('idx_org_resource_org_type', 'organization_id', 'resource_type'),
        Index('idx_org_resource_active', 'is_active'),
    )

    def __repr__(self) -> str:
        """String representation of organization resource."""
        return f"<OrganizationResource(id={self.id}, org_id={self.organization_id}, type='{self.resource_type}', allocated={self.allocated_amount})>"

    def to_dict(self) -> dict:
        """Convert resource to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'resource_type': self.resource_type,
            'resource_name': self.resource_name,
            'allocated_amount': self.allocated_amount,
            'used_amount': self.used_amount,
            'unit': self.unit,
            'soft_limit': self.soft_limit,
            'hard_limit': self.hard_limit,
            'is_active': self.is_active,
        }

    @staticmethod
    def get_by_org_and_type(organization_id: int, resource_type: str, db_session) -> list:
        """Get resources by organization and type."""
        return db_session.query(OrganizationResource).filter(
            OrganizationResource.organization_id == organization_id,
            OrganizationResource.resource_type == resource_type,
            OrganizationResource.is_active == True
        ).all()


class OrganizationBilling(Base):
    """Organization billing record model."""

    __tablename__ = 'organization_billing'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    organization_id = Column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), index=True)

    # Billing period
    billing_period_start = Column(DateTime, nullable=False, index=True)
    billing_period_end = Column(DateTime, nullable=False, index=True)

    # Billing details
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    status = Column(String(20), default="pending", index=True)  # pending, paid, failed, cancelled

    # Usage summary
    total_requests = Column(Integer, default=0)
    total_audio_seconds = Column(Float, default=0.0)
    total_storage_bytes = Column(Integer, default=0)

    # Cost breakdown
    base_cost = Column(Float, default=0.0)
    request_cost = Column(Float, default=0.0)
    storage_cost = Column(Float, default=0.0)
    additional_cost = Column(Float, default=0.0)

    # Payment information
    payment_method_id = Column(String(100), nullable=True)
    transaction_id = Column(String(100), nullable=True)
    invoice_url = Column(String(500), nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="billing_records")

    # Indexes
    __table_args__ = (
        Index('idx_org_billing_org_period', 'organization_id', 'billing_period_start', 'billing_period_end'),
        Index('idx_org_billing_status', 'status'),
    )

    def __repr__(self) -> str:
        """String representation of billing record."""
        return f"<OrganizationBilling(id={self.id}, org_id={self.organization_id}, amount={self.amount}, period={self.billing_period_start}-{self.billing_period_end})>"

    def to_dict(self) -> dict:
        """Convert billing record to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'billing_period_start': self.billing_period_start.isoformat(),
            'billing_period_end': self.billing_period_end.isoformat(),
            'amount': self.amount,
            'currency': self.currency,
            'status': self.status,
            'total_requests': self.total_requests,
            'total_audio_seconds': self.total_audio_seconds,
            'total_storage_bytes': self.total_storage_bytes,
            'base_cost': self.base_cost,
            'request_cost': self.request_cost,
            'storage_cost': self.storage_cost,
            'additional_cost': self.additional_cost,
            'transaction_id': self.transaction_id,
            'invoice_url': self.invoice_url,
        }


class OrganizationUsage(Base):
    """Organization usage tracking model."""

    __tablename__ = 'organization_usage'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    organization_id = Column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), index=True)

    # Usage details
    usage_date = Column(DateTime, nullable=False, index=True)
    usage_type = Column(String(50), nullable=False, index=True)  # 'requests', 'storage', 'audio_seconds'

    # Metrics
    count = Column(Integer, default=0)
    amount = Column(Float, default=0.0)  # For storage bytes, audio seconds, etc.
    unit = Column(String(20), nullable=False)

    # Cost
    cost = Column(Float, default=0.0)
    cost_per_unit = Column(Float, default=0.0)

    # Metadata
    request_metadata = Column(JSON, default=dict)  # Additional usage details

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="usage_records")

    # Indexes
    __table_args__ = (
        Index('idx_org_usage_org_date_type', 'organization_id', 'usage_date', 'usage_type'),
        Index('idx_org_usage_date', 'usage_date'),
    )

    def __repr__(self) -> str:
        """String representation of usage record."""
        return f"<OrganizationUsage(id={self.id}, org_id={self.organization_id}, type='{self.usage_type}', count={self.count})>"

    def to_dict(self) -> dict:
        """Convert usage record to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'usage_date': self.usage_date.isoformat(),
            'usage_type': self.usage_type,
            'count': self.count,
            'amount': self.amount,
            'unit': self.unit,
            'cost': self.cost,
            'cost_per_unit': self.cost_per_unit,
            'metadata': self.request_metadata,
        }

    @staticmethod
    def get_usage_by_org_and_date_range(organization_id: int, start_date: datetime, end_date: datetime, db_session) -> list:
        """Get usage records for organization within date range."""
        return db_session.query(OrganizationUsage).filter(
            OrganizationUsage.organization_id == organization_id,
            OrganizationUsage.usage_date >= start_date,
            OrganizationUsage.usage_date <= end_date
        ).all()

    @staticmethod
    def get_daily_usage_summary(organization_id: int, date: datetime, db_session) -> dict:
        """Get daily usage summary for organization."""
        records = db_session.query(OrganizationUsage).filter(
            OrganizationUsage.organization_id == organization_id,
            OrganizationUsage.usage_date >= date,
            OrganizationUsage.usage_date < date + timedelta(days=1)
        ).all()

        summary = {
            'date': date.isoformat(),
            'total_requests': 0,
            'total_audio_seconds': 0.0,
            'total_storage_bytes': 0,
            'total_cost': 0.0,
        }

        for record in records:
            if record.usage_type == 'requests':
                summary['total_requests'] = record.count
            elif record.usage_type == 'audio_seconds':
                summary['total_audio_seconds'] = record.amount
            elif record.usage_type == 'storage':
                summary['total_storage_bytes'] = record.amount
            summary['total_cost'] += record.cost

        return summary