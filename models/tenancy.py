"""
Multi-tenant database schema and row-level security for TTS system
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, text, event, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, Query
from sqlalchemy.sql import operators

from .organization import Organization, OrganizationStatus
from utils.tenant_manager import tenant_manager

Base = declarative_base()


class TenantSecurityManager:
    """Manages row-level security for multi-tenant database."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._tenant_context = None
        self._bypass_security = False

    def set_tenant_context(self, organization_id: int, user_id: Optional[int] = None):
        """Set tenant context for current session."""
        self._tenant_context = {
            'organization_id': organization_id,
            'user_id': user_id
        }

    def clear_tenant_context(self):
        """Clear tenant context."""
        self._tenant_context = None

    def get_tenant_context(self) -> Optional[Dict[str, Any]]:
        """Get current tenant context."""
        return self._tenant_context

    def bypass_security(self, bypass: bool = True):
        """Temporarily bypass security checks."""
        self._bypass_security = bypass

    @contextmanager
    def security_bypass(self):
        """Context manager to temporarily bypass security."""
        original_state = self._bypass_security
        self._bypass_security = True
        try:
            yield
        finally:
            self._bypass_security = original_state

    def is_security_enabled(self) -> bool:
        """Check if security is enabled."""
        return not self._bypass_security and self._tenant_context is not None

    def get_current_organization_id(self) -> Optional[int]:
        """Get current organization ID from tenant context."""
        if self._tenant_context:
            return self._tenant_context.get('organization_id')
        return None


# Global tenant security manager
tenant_security = TenantSecurityManager()


def add_tenant_filter(query: Query, model_class) -> Query:
    """Add tenant filter to query if security is enabled."""
    if not tenant_security.is_security_enabled():
        return query

    organization_id = tenant_security.get_current_organization_id()
    if not organization_id:
        return query

    # Check if model has organization_id column
    if hasattr(model_class, 'organization_id'):
        return query.filter(model_class.organization_id == organization_id)
    elif hasattr(model_class, 'user_id'):
        # For user-scoped models, we need to check user's organization membership
        # This is a simplified approach - in production, you'd want to join with organization_members
        return query
    else:
        # For models without explicit tenant columns, we might need different strategies
        return query


def apply_tenant_security(session: Session):
    """Apply tenant security to all queries in the session."""
    if not tenant_security.is_security_enabled():
        return

    @event.listens_for(session, 'before_flush')
    def before_flush(session, flush_context, instances):
        """Validate tenant access before flush."""
        if tenant_security._bypass_security:
            return

        organization_id = tenant_security.get_current_organization_id()
        if not organization_id:
            return

        for obj in session.new:
            if hasattr(obj, 'organization_id') and obj.organization_id != organization_id:
                raise ValueError(f"Access denied: Cannot create {obj.__class__.__name__} for different organization")

        for obj in session.dirty:
            if hasattr(obj, 'organization_id') and obj.organization_id != organization_id:
                raise ValueError(f"Access denied: Cannot modify {obj.__class__.__name__} from different organization")


def setup_tenant_security(session: Session):
    """Setup tenant security for session."""
    apply_tenant_security(session)


# Enhanced model base with tenant support
class TenantAwareBase:
    """Base class for tenant-aware models."""

    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    updated_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    @classmethod
    def tenant_aware_query(cls, session: Session) -> Query:
        """Get tenant-aware query for this model."""
        query = session.query(cls)
        return add_tenant_filter(query, cls)

    @classmethod
    def get_by_id_tenant_aware(cls, session: Session, id: int):
        """Get record by ID with tenant awareness."""
        query = cls.tenant_aware_query(session)
        return query.filter(cls.id == id).first()

    @classmethod
    def get_all_tenant_aware(cls, session: Session, limit: int = 100):
        """Get all records with tenant awareness."""
        query = cls.tenant_aware_query(session)
        return query.limit(limit).all()

    def validate_tenant_access(self, session: Session) -> bool:
        """Validate that current tenant has access to this record."""
        if not tenant_security.is_security_enabled():
            return True

        organization_id = tenant_security.get_current_organization_id()
        return self.organization_id == organization_id


# Enhanced AudioRequest model with tenant support
class TenantAwareAudioRequest(TenantAwareBase):
    """Tenant-aware audio request model."""

    __tablename__ = 'tenant_aware_audio_requests'

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(5000), nullable=False)
    voice_name = Column(String(50), default="Alnilam")
    output_format = Column(String(10), default="wav")
    speed = Column(Float, default=1.0)
    pitch = Column(Float, default=0.0)

    # Status and processing
    status = Column(String(20), default="pending", index=True)
    file_path = Column(String(500), nullable=True)
    file_url = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Cost tracking
    cost = Column(Float, default=0.0)
    cost_per_character = Column(Float, default=0.0)

    # Processing metadata
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    error_message = Column(String(500), nullable=True)

    # User association (for per-user tracking within organization)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'text': self.text,
            'voice_name': self.voice_name,
            'output_format': self.output_format,
            'speed': self.speed,
            'pitch': self.pitch,
            'status': self.status,
            'file_path': self.file_path,
            'file_url': self.file_url,
            'file_size': self.file_size,
            'duration_seconds': self.duration_seconds,
            'cost': self.cost,
            'cost_per_character': self.cost_per_character,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None,
            'processing_started_at': self.processing_started_at.isoformat() if self.processing_started_at is not None else None,
            'processing_completed_at': self.processing_completed_at.isoformat() if self.processing_completed_at is not None else None,
            'error_message': self.error_message,
        }


# Enhanced AudioFile model with tenant support
class TenantAwareAudioFile(TenantAwareBase):
    """Tenant-aware audio file model."""

    __tablename__ = 'tenant_aware_audio_files'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_url = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    mime_type = Column(String(50), nullable=False)

    # Metadata
    metadata = Column(JSON, default=dict)
    tags = Column(String(500), nullable=True)

    # User association
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)

    # Storage information
    storage_provider = Column(String(50), default="local")
    storage_path = Column(String(500), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'file_url': self.file_url,
            'file_size': self.file_size,
            'duration_seconds': self.duration_seconds,
            'mime_type': self.mime_type,
            'metadata': self.metadata,
            'tags': self.tags,
            'storage_provider': self.storage_provider,
            'storage_path': self.storage_path,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None,
        }


# Enhanced RequestLog model with tenant support
class TenantAwareRequestLog(TenantAwareBase):
    """Tenant-aware request log model."""

    __tablename__ = 'tenant_aware_request_logs'

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    endpoint = Column(String(500), nullable=False)
    user_agent = Column(String(200), nullable=True)

    # Request details
    ip_address = Column(String(45), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)

    # Response details
    status_code = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    request_size = Column(Integer, default=0)
    response_size = Column(Integer, default=0)

    # Error tracking
    error_message = Column(String(500), nullable=True)
    error_stack_trace = Column(String(2000), nullable=True)

    # Additional metadata
    metadata = Column(JSON, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'request_id': self.request_id,
            'method': self.method,
            'endpoint': self.endpoint,
            'user_agent': self.user_agent,
            'ip_address': self.ip_address,
            'user_id': self.user_id,
            'status_code': self.status_code,
            'response_time_ms': self.response_time_ms,
            'request_size': self.request_size,
            'response_size': self.response_size,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'metadata': self.metadata,
        }


# Database migration helper functions
def create_tenant_aware_tables(engine):
    """Create tenant-aware tables."""
    TenantAwareAudioRequest.__table__.create(engine, checkfirst=True)
    TenantAwareAudioFile.__table__.create(engine, checkfirst=True)
    TenantAwareRequestLog.__table__.create(engine, checkfirst=True)


def drop_tenant_aware_tables(engine):
    """Drop tenant-aware tables."""
    TenantAwareRequestLog.__table__.drop(engine, checkfirst=True)
    TenantAwareAudioFile.__table__.drop(engine, checkfirst=True)
    TenantAwareAudioRequest.__table__.drop(engine, checkfirst=True)


def migrate_existing_data_to_tenant_aware(session: Session, organization_id: int):
    """Migrate existing data to tenant-aware structure."""
    from .audio_request import AudioRequest
    from .audio_file import AudioFile
    from .request_log import RequestLog

    # Migrate audio requests
    existing_requests = session.query(AudioRequest).all()
    for request in existing_requests:
        tenant_request = TenantAwareAudioRequest(
            organization_id=organization_id,
            text=request.text,
            voice_name=request.voice_name,
            output_format=request.output_format,
            speed=request.speed,
            pitch=request.pitch,
            status=request.status,
            file_path=request.file_path,
            file_url=request.file_url,
            file_size=request.file_size,
            duration_seconds=request.duration_seconds,
            cost=request.cost,
            cost_per_character=request.cost_per_character,
            processing_started_at=request.processing_started_at,
            processing_completed_at=request.processing_completed_at,
            error_message=request.error_message,
            user_id=request.user_id,
            created_at=request.created_at,
            updated_at=request.updated_at,
            created_by=request.created_by,
            updated_by=request.updated_by
        )
        session.add(tenant_request)

    # Migrate audio files
    existing_files = session.query(AudioFile).all()
    for file in existing_files:
        tenant_file = TenantAwareAudioFile(
            organization_id=organization_id,
            filename=file.filename,
            original_filename=file.original_filename,
            file_path=file.file_path,
            file_url=file.file_url,
            file_size=file.file_size,
            duration_seconds=file.duration_seconds,
            mime_type=file.mime_type,
            metadata=file.metadata,
            tags=file.tags,
            user_id=file.user_id,
            storage_provider=file.storage_provider,
            storage_path=file.storage_path,
            created_at=file.created_at,
            updated_at=file.updated_at,
            created_by=file.created_by,
            updated_by=file.updated_by
        )
        session.add(tenant_file)

    # Migrate request logs
    existing_logs = session.query(RequestLog).all()
    for log in existing_logs:
        tenant_log = TenantAwareRequestLog(
            organization_id=organization_id,
            request_id=log.request_id,
            method=log.method,
            endpoint=log.endpoint,
            user_agent=log.user_agent,
            ip_address=log.ip_address,
            user_id=log.user_id,
            status_code=log.status_code,
            response_time_ms=log.response_time_ms,
            request_size=log.request_size,
            response_size=log.response_size,
            error_message=log.error_message,
            error_stack_trace=log.error_stack_trace,
            metadata=log.metadata,
            created_at=log.created_at,
            updated_at=log.updated_at,
            created_by=log.created_by,
            updated_by=log.updated_by
        )
        session.add(tenant_log)

    session.commit()


# Query helpers for cross-tenant operations (admin only)
def get_tenant_statistics(session: Session) -> Dict[str, Any]:
    """Get statistics across all tenants (admin function)."""
    with tenant_security.security_bypass():
        total_orgs = session.query(Organization).filter(
            Organization.status == OrganizationStatus.ACTIVE
        ).count()

        total_requests = session.query(TenantAwareAudioRequest).count()
        total_files = session.query(TenantAwareAudioFile).count()
        total_logs = session.query(TenantAwareRequestLog).count()

        return {
            'total_organizations': total_orgs,
            'total_requests': total_requests,
            'total_files': total_files,
            'total_logs': total_logs,
            'generated_at': datetime.utcnow().isoformat()
        }


def get_organization_usage_summary(session: Session, organization_id: int) -> Dict[str, Any]:
    """Get usage summary for specific organization."""
    with tenant_security.security_bypass():
        org = session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return None

        request_count = session.query(TenantAwareAudioRequest).filter(
            TenantAwareAudioRequest.organization_id == organization_id
        ).count()

        file_count = session.query(TenantAwareAudioFile).filter(
            TenantAwareAudioFile.organization_id == organization_id
        ).count()

        total_cost = session.query(TenantAwareAudioRequest).filter(
            TenantAwareAudioRequest.organization_id == organization_id
        ).with_entities(TenantAwareAudioRequest.cost).all()

        total_cost = sum(cost[0] for cost in total_cost) if total_cost else 0.0

        return {
            'organization_id': organization_id,
            'organization_name': org.name,
            'total_requests': request_count,
            'total_files': file_count,
            'total_cost': total_cost,
            'current_month_requests': org.current_month_requests,
            'current_month_cost': org.current_month_cost,
            'max_monthly_requests': org.max_monthly_requests,
            'max_storage_bytes': org.max_storage_bytes,
            'current_storage_bytes': org.current_storage_bytes,
        }