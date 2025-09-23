"""
Tenant Management Service for multi-tenant TTS system
"""

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from functools import wraps

from flask import g, request, current_app
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from models.organization import (
    Organization,
    OrganizationMember,
    OrganizationResource,
    OrganizationUsage,
    OrganizationStatus,
    MemberRole
)


class TenantContext:
    """Context manager for tenant-specific operations."""

    def __init__(self, organization_id: int, user_id: Optional[int] = None):
        self.organization_id = organization_id
        self.user_id = user_id
        self._previous_context = None

    def __enter__(self):
        self._previous_context = getattr(g, 'tenant_context', None)
        g.tenant_context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        g.tenant_context = self._previous_context


class TenantManager:
    """Main tenant management service."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._tenant_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_lock = threading.Lock()

    def get_current_tenant(self) -> Optional[TenantContext]:
        """Get current tenant context from Flask g object."""
        return getattr(g, 'tenant_context', None)

    def set_tenant_context(self, organization_id: int, user_id: Optional[int] = None):
        """Set tenant context for current request."""
        g.tenant_context = TenantContext(organization_id, user_id)

    def clear_tenant_context(self):
        """Clear tenant context."""
        if hasattr(g, 'tenant_context'):
            delattr(g, 'tenant_context')

    @contextmanager
    def tenant_context(self, organization_id: int, user_id: Optional[int] = None):
        """Context manager for tenant operations."""
        with TenantContext(organization_id, user_id):
            try:
                yield
            finally:
                self.clear_tenant_context()

    def get_organization_from_request(self, db_session: Session) -> Optional[Organization]:
        """Extract organization from request (API key, user context, etc.)."""
        # Try to get from API key first
        api_key = self._extract_api_key_from_request()
        if api_key:
            org = Organization.get_by_api_key(api_key, db_session)
            if org:
                return org

        # Try to get from current user context
        tenant_context = self.get_current_tenant()
        if tenant_context and tenant_context.organization_id:
            return db_session.query(Organization).filter(
                Organization.id == tenant_context.organization_id,
                Organization.status == OrganizationStatus.ACTIVE
            ).first()

        return None

    def _extract_api_key_from_request(self) -> Optional[str]:
        """Extract API key from request headers."""
        # Check Authorization header
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            return auth_header[7:]

        # Check X-API-Key header
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return api_key

        # Check api_key query parameter
        api_key = request.args.get('api_key')
        if api_key:
            return api_key

        return None

    def validate_organization_access(self, organization_id: int, user_id: int, db_session: Session) -> bool:
        """Validate if user has access to organization."""
        member = OrganizationMember.get_by_org_and_user(organization_id, user_id, db_session)
        return member is not None and member.is_active

    def get_user_organizations(self, user_id: int, db_session: Session) -> List[Organization]:
        """Get all organizations accessible by user."""
        members = OrganizationMember.get_organizations_by_user(user_id, db_session)
        org_ids = [member.organization_id for member in members]
        return db_session.query(Organization).filter(
            Organization.id.in_(org_ids),
            Organization.status == OrganizationStatus.ACTIVE
        ).all()

    def check_resource_availability(self, organization_id: int, resource_type: str,
                                  amount: float, db_session: Session) -> Dict[str, Any]:
        """Check if organization has sufficient resources."""
        resources = OrganizationResource.get_by_org_and_type(organization_id, resource_type, db_session)

        if not resources:
            return {
                'available': False,
                'reason': f'No {resource_type} resources allocated',
                'current_usage': 0,
                'allocated': 0
            }

        total_allocated = sum(r.allocated_amount for r in resources if r.is_active)
        total_used = sum(r.used_amount for r in resources if r.is_active)

        available = total_allocated - total_used

        if available >= amount:
            return {
                'available': True,
                'current_usage': total_used,
                'allocated': total_allocated,
                'available': available,
                'requested': amount
            }
        else:
            return {
                'available': False,
                'reason': 'Insufficient resources',
                'current_usage': total_used,
                'allocated': total_allocated,
                'available': available,
                'requested': amount
            }

    def allocate_resource(self, organization_id: int, resource_type: str, amount: float,
                         db_session: Session) -> bool:
        """Allocate resource usage for organization."""
        resources = OrganizationResource.get_by_org_and_type(organization_id, resource_type, db_session)

        if not resources:
            return False

        # Find resource with available capacity
        for resource in resources:
            if not resource.is_active:
                continue

            available = resource.allocated_amount - resource.used_amount
            if available >= amount:
                resource.used_amount += amount
                resource.last_updated = datetime.utcnow()
                db_session.commit()
                return True

        return False

    def release_resource(self, organization_id: int, resource_type: str, amount: float,
                        db_session: Session) -> bool:
        """Release allocated resource."""
        resources = OrganizationResource.get_by_org_and_type(organization_id, resource_type, db_session)

        if not resources:
            return False

        # Release from the first active resource (simplified logic)
        for resource in resources:
            if not resource.is_active:
                continue

            if resource.used_amount >= amount:
                resource.used_amount -= amount
                resource.last_updated = datetime.utcnow()
                db_session.commit()
                return True

        return False

    def track_usage(self, organization_id: int, usage_type: str, count: int = 1,
                   amount: float = 0.0, unit: str = 'count', cost: float = 0.0,
                   db_session: Session, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Track usage for organization."""
        try:
            usage_record = OrganizationUsage(
                organization_id=organization_id,
                usage_date=datetime.utcnow(),
                usage_type=usage_type,
                count=count,
                amount=amount,
                unit=unit,
                cost=cost,
                cost_per_unit=cost / max(count, 1),
                metadata=metadata or {}
            )

            db_session.add(usage_record)
            db_session.commit()

            # Update organization usage counters
            self._update_organization_usage(organization_id, usage_type, count, amount, cost, db_session)

            return True
        except SQLAlchemyError:
            db_session.rollback()
            return False

    def _update_organization_usage(self, organization_id: int, usage_type: str, count: int,
                                  amount: float, cost: float, db_session: Session):
        """Update organization usage counters."""
        org = db_session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return

        # Update counters based on usage type
        if usage_type == 'requests':
            org.current_month_requests = org.current_month_requests + count
        elif usage_type == 'storage':
            org.current_storage_bytes = org.current_storage_bytes + int(amount)
        elif usage_type == 'audio_seconds':
            # Could track audio seconds if needed
            pass

        org.current_month_cost = org.current_month_cost + cost
        org.monthly_cost = org.monthly_cost + cost
        org.total_cost = org.total_cost + cost

        db_session.commit()

    def get_usage_summary(self, organization_id: int, start_date: datetime,
                         end_date: datetime, db_session: Session) -> Dict[str, Any]:
        """Get usage summary for organization."""
        records = OrganizationUsage.get_usage_by_org_and_date_range(
            organization_id, start_date, end_date, db_session
        )

        summary = {
            'organization_id': organization_id,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'total_requests': 0,
            'total_audio_seconds': 0.0,
            'total_storage_bytes': 0,
            'total_cost': 0.0,
            'usage_by_type': {}
        }

        for record in records:
            summary['total_requests'] += record.count if record.usage_type == 'requests' else 0
            summary['total_audio_seconds'] += record.amount if record.usage_type == 'audio_seconds' else 0
            summary['total_storage_bytes'] += record.amount if record.usage_type == 'storage' else 0
            summary['total_cost'] += record.cost

            if record.usage_type not in summary['usage_by_type']:
                summary['usage_by_type'][record.usage_type] = {
                    'count': 0,
                    'amount': 0.0,
                    'cost': 0.0
                }

            summary['usage_by_type'][record.usage_type]['count'] += record.count
            summary['usage_by_type'][record.usage_type]['amount'] += record.amount
            summary['usage_by_type'][record.usage_type]['cost'] += record.cost

        return summary

    def check_organization_limits(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Check if organization is within limits."""
        org = db_session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return {'within_limits': False, 'reason': 'Organization not found'}

        issues = []

        # Check user limit
        if org.current_users >= org.max_users:
            issues.append(f'User limit exceeded: {org.current_users}/{org.max_users}')

        # Check monthly request limit
        if org.current_month_requests >= org.max_monthly_requests:
            issues.append(f'Monthly request limit exceeded: {org.current_month_requests}/{org.max_monthly_requests}')

        # Check storage limit
        if org.current_storage_bytes >= org.max_storage_bytes:
            issues.append(f'Storage limit exceeded: {org.current_storage_bytes}/{org.max_storage_bytes}')

        return {
            'within_limits': len(issues) == 0,
            'issues': issues,
            'limits': {
                'users': {'current': org.current_users, 'max': org.max_users},
                'requests': {'current': org.current_month_requests, 'max': org.max_monthly_requests},
                'storage': {'current': org.current_storage_bytes, 'max': org.max_storage_bytes}
            }
        }

    def get_organization_cache_key(self, organization_id: int) -> str:
        """Generate cache key for organization."""
        return f"org:{organization_id}"

    def get_cached_organization(self, organization_id: int) -> Optional[Dict]:
        """Get organization from cache."""
        with self._cache_lock:
            cache_key = self.get_organization_cache_key(organization_id)
            if cache_key in self._tenant_cache:
                cached_item = self._tenant_cache[cache_key]
                if time.time() - cached_item['timestamp'] < self._cache_ttl:
                    return cached_item['data']
                else:
                    del self._tenant_cache[cache_key]
            return None

    def cache_organization(self, organization_id: int, data: Dict):
        """Cache organization data."""
        with self._cache_lock:
            cache_key = self.get_organization_cache_key(organization_id)
            self._tenant_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }

    def clear_organization_cache(self, organization_id: int):
        """Clear organization cache."""
        with self._cache_lock:
            cache_key = self.get_organization_cache_key(organization_id)
            self._tenant_cache.pop(cache_key, None)

    def cleanup_cache(self):
        """Clean up expired cache entries."""
        with self._cache_lock:
            current_time = time.time()
            expired_keys = [
                key for key, value in self._tenant_cache.items()
                if current_time - value['timestamp'] >= self._cache_ttl
            ]
            for key in expired_keys:
                del self._tenant_cache[key]


# Global tenant manager instance
tenant_manager = TenantManager()


def require_tenant_context(f):
    """Decorator to require tenant context for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        tenant_context = tenant_manager.get_current_tenant()
        if not tenant_context:
            return {
                'error': 'Tenant context required',
                'message': 'This endpoint requires organization context'
            }, 400
        return f(*args, **kwargs)
    return decorated_function


def tenant_aware(f):
    """Decorator to make function tenant-aware."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        tenant_context = tenant_manager.get_current_tenant()
        if tenant_context:
            kwargs['tenant_context'] = tenant_context
        return f(*args, **kwargs)
    return decorated_function


def with_organization_context(organization_id: int, user_id: Optional[int] = None):
    """Decorator to execute function with specific organization context."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            with tenant_manager.tenant_context(organization_id, user_id):
                return f(*args, **kwargs)
        return decorated_function
    return decorator