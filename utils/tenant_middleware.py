"""
Tenant middleware for Flask application
"""

from functools import wraps
from flask import request, g, jsonify, current_app
from sqlalchemy.orm import Session

from .tenant_manager import tenant_manager
from models.organization import Organization, OrganizationStatus


class TenantMiddleware:
    """Middleware for handling tenant context in Flask requests."""

    @staticmethod
    def before_request(db_session: Session):
        """Process tenant context before each request."""
        try:
            # Extract organization from request
            organization = tenant_manager.get_organization_from_request(db_session)

            if organization:
                # Set tenant context
                tenant_manager.set_tenant_context(
                    organization_id=organization.id,
                    user_id=None  # Will be set by auth middleware if user is authenticated
                )

                # Store organization in Flask g for easy access
                g.current_organization = organization

                # Check if organization is active
                if organization.status != OrganizationStatus.ACTIVE:
                    return jsonify({
                        'error': 'Organization suspended',
                        'message': 'This organization account has been suspended'
                    }), 403

                # Check organization limits
                limits_check = tenant_manager.check_organization_limits(organization.id, db_session)
                if not limits_check['within_limits']:
                    return jsonify({
                        'error': 'Organization limits exceeded',
                        'message': 'Organization has exceeded its usage limits',
                        'issues': limits_check['issues']
                    }), 429

            else:
                # No organization context found
                g.current_organization = None

        except Exception as e:
            current_app.logger.error(f"Tenant middleware error: {e}")
            return jsonify({
                'error': 'Tenant context error',
                'message': 'Failed to establish tenant context'
            }), 500

    @staticmethod
    def after_request(response):
        """Clean up tenant context after each request."""
        tenant_manager.clear_tenant_context()
        if hasattr(g, 'current_organization'):
            delattr(g, 'current_organization')
        return response


def require_organization_context(f):
    """Decorator to require organization context for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'current_organization') or g.current_organization is None:
            return jsonify({
                'error': 'Organization context required',
                'message': 'This endpoint requires organization context. Please provide API key or authenticate as organization user.'
            }), 400
        return f(*args, **kwargs)
    return decorated_function


def require_organization_admin(f):
    """Decorator to require organization admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'current_organization') or g.current_organization is None:
            return jsonify({
                'error': 'Organization context required',
                'message': 'This endpoint requires organization context'
            }), 400

        # Check if current user is admin or owner
        if not hasattr(g, 'current_user') or g.current_user is None:
            return jsonify({
                'error': 'Authentication required',
                'message': 'User authentication required for admin operations'
            }), 401

        # Check user's role in organization
        from models.organization import OrganizationMember, MemberRole
        member = OrganizationMember.get_by_org_and_user(
            g.current_organization.id,
            g.current_user.id,
            kwargs.get('db_session')
        )

        if not member or member.role not in [MemberRole.ADMIN, MemberRole.OWNER]:
            return jsonify({
                'error': 'Insufficient permissions',
                'message': 'Admin or owner privileges required'
            }), 403

        return f(*args, **kwargs)
    return decorated_function


def organization_rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Rate limiting decorator for organization-scoped endpoints."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_organization') or g.current_organization is None:
                return jsonify({
                    'error': 'Organization context required',
                    'message': 'Rate limiting requires organization context'
                }), 400

            # Simple in-memory rate limiting (in production, use Redis)
            import time
            from flask import current_app

            org_id = g.current_organization.id
            current_time = time.time()

            # Get or create rate limit tracker
            if not hasattr(current_app, '_rate_limit_cache'):
                current_app._rate_limit_cache = {}

            cache_key = f"rate_limit:{org_id}:{f.__name__}"
            cache = current_app._rate_limit_cache

            if cache_key not in cache:
                cache[cache_key] = {
                    'requests': [],
                    'max_requests': max_requests,
                    'window_seconds': window_seconds
                }

            tracker = cache[cache_key]

            # Clean old requests
            tracker['requests'] = [
                req_time for req_time in tracker['requests']
                if current_time - req_time < window_seconds
            ]

            # Check if limit exceeded
            if len(tracker['requests']) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {max_requests} per {window_seconds} seconds',
                    'retry_after': window_seconds
                }), 429

            # Add current request
            tracker['requests'].append(current_time)

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def log_organization_request(f):
    """Decorator to log organization requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from models.tenancy import TenantAwareRequestLog

        # Log request start
        request_id = request.headers.get('X-Request-ID', 'unknown')
        start_time = time.time()

        try:
            # Execute the function
            result = f(*args, **kwargs)

            # Log successful request
            if hasattr(g, 'current_organization'):
                db_session = kwargs.get('db_session')
                if db_session:
                    log_entry = TenantAwareRequestLog(
                        organization_id=g.current_organization.id,
                        request_id=request_id,
                        method=request.method,
                        endpoint=request.endpoint,
                        user_agent=request.headers.get('User-Agent', ''),
                        ip_address=request.remote_addr,
                        user_id=getattr(g, 'current_user', {}).get('id') if hasattr(g, 'current_user') else None,
                        status_code=result[1] if isinstance(result, tuple) and len(result) > 1 else 200,
                        response_time_ms=int((time.time() - start_time) * 1000),
                        request_size=len(request.get_data()),
                        response_size=len(str(result[0]).encode()) if isinstance(result, tuple) else 0,
                        metadata={
                            'user_id': getattr(g, 'current_user', {}).get('id'),
                            'organization_id': g.current_organization.id,
                            'organization_name': g.current_organization.name
                        }
                    )
                    db_session.add(log_entry)
                    db_session.commit()

            return result

        except Exception as e:
            # Log failed request
            if hasattr(g, 'current_organization'):
                db_session = kwargs.get('db_session')
                if db_session:
                    log_entry = TenantAwareRequestLog(
                        organization_id=g.current_organization.id,
                        request_id=request_id,
                        method=request.method,
                        endpoint=request.endpoint,
                        user_agent=request.headers.get('User-Agent', ''),
                        ip_address=request.remote_addr,
                        user_id=getattr(g, 'current_user', {}).get('id') if hasattr(g, 'current_user') else None,
                        status_code=500,
                        response_time_ms=int((time.time() - start_time) * 1000),
                        request_size=len(request.get_data()),
                        error_message=str(e),
                        metadata={
                            'user_id': getattr(g, 'current_user', {}).get('id'),
                            'organization_id': g.current_organization.id,
                            'organization_name': g.current_organization.name
                        }
                    )
                    db_session.add(log_entry)
                    db_session.commit()

            raise
    return decorated_function


def track_organization_usage(usage_type: str, count: int = 1, amount: float = 0.0, unit: str = 'count'):
    """Decorator to track organization usage."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            result = f(*args, **kwargs)

            # Track usage if request was successful
            if (hasattr(g, 'current_organization') and
                (isinstance(result, tuple) and result[1] < 400 or not isinstance(result, tuple))):

                db_session = kwargs.get('db_session')
                if db_session:
                    cost = 0.0  # Calculate cost based on usage_type and amount
                    if usage_type == 'requests':
                        cost = count * 0.001  # Example: $0.001 per request
                    elif usage_type == 'storage':
                        cost = amount * 0.0001  # Example: $0.0001 per byte

                    tenant_manager.track_usage(
                        organization_id=g.current_organization.id,
                        usage_type=usage_type,
                        count=count,
                        amount=amount,
                        unit=unit,
                        cost=cost,
                        db_session=db_session,
                        metadata={
                            'endpoint': request.endpoint,
                            'method': request.method,
                            'user_id': getattr(g, 'current_user', {}).get('id')
                        }
                    )

            return result
        return decorated_function
    return decorator


def validate_resource_availability(resource_type: str, amount: float):
    """Decorator to validate resource availability before processing."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_organization'):
                return jsonify({
                    'error': 'Organization context required',
                    'message': 'Resource validation requires organization context'
                }), 400

            db_session = kwargs.get('db_session')
            if db_session:
                availability = tenant_manager.check_resource_availability(
                    g.current_organization.id,
                    resource_type,
                    amount,
                    db_session
                )

                if not availability['available']:
                    return jsonify({
                        'error': 'Insufficient resources',
                        'message': availability['reason'],
                        'current_usage': availability['current_usage'],
                        'allocated': availability['allocated'],
                        'requested': availability['requested']
                    }), 429

            return f(*args, **kwargs)
        return decorated_function
    return decorator


# Import time module for logging
import time