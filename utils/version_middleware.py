"""
Version Middleware cho TTS System

Xử lý request version detection, response version handling, compatibility checking và migration assistance.
"""

import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

from flask import request, g, Response, jsonify
from werkzeug.wrappers import Response as WerkzeugResponse

from utils.version_manager import version_manager
from models.api_version import VersionedResponse, VersionedRequest
from config.api_versions import VERSION_CONFIG


logger = logging.getLogger(__name__)


class VersionMiddleware:
    """Middleware xử lý API versioning"""

    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)

    def init_app(self, app):
        """Initialize middleware với Flask app"""
        app.before_request(self._before_request)
        app.after_request(self._after_request)

        # Store middleware instance
        app.version_middleware = self

    def _before_request(self):
        """Xử lý trước khi request được processed"""
        try:
            # Detect API version
            requested_version = version_manager.detect_version()

            # Validate version exists
            if requested_version not in version_manager.registry.versions:
                return self._create_error_response(
                    "Invalid API version",
                    400,
                    {
                        "requested_version": requested_version,
                        "supported_versions": list(version_manager.registry.versions.keys())
                    }
                )

            # Get version info
            version_info = version_manager.get_version_info(requested_version)

            # Store version info in Flask g object
            g.api_version = requested_version
            g.version_info = version_info
            g.version_status = version_info["status"] if version_info else "unknown"

            # Check for deprecation warnings
            deprecation_warnings = version_manager.get_deprecation_warnings(requested_version)
            if deprecation_warnings:
                g.deprecation_warnings = deprecation_warnings
                logger.warning(f"Deprecated API version used: {requested_version}")

            # Log version usage
            logger.info(f"API Request: {request.method} {request.path} - Version: {requested_version}")

        except Exception as e:
            logger.error(f"Version middleware error: {str(e)}")
            return self._create_error_response("Version processing error", 500)

    def _after_request(self, response: Response):
        """Xử lý sau khi request được processed"""
        try:
            # Skip if no API version detected
            if not hasattr(g, 'api_version'):
                return response

            api_version = g.api_version

            # Add version headers to response
            response.headers['X-API-Version'] = api_version
            response.headers['X-Version-Status'] = g.version_status

            # Add deprecation headers if applicable
            if hasattr(g, 'deprecation_warnings') and g.deprecation_warnings:
                warnings = g.deprecation_warnings[0]  # Get first warning
                if warnings.get('type') == 'deprecation':
                    response.headers['X-API-Deprecated'] = 'true'
                    if warnings.get('replacement_version'):
                        response.headers['X-API-Replacement-Version'] = warnings['replacement_version']
                    if warnings.get('sunset_date'):
                        response.headers['X-API-Sunset-Date'] = warnings['sunset_date']

            # Add feature flags header if available
            if hasattr(g, 'version_info') and g.version_info:
                feature_flags = g.version_info.get('feature_flags', {})
                if feature_flags:
                    response.headers['X-Feature-Flags'] = json.dumps(feature_flags)

            # Log response version info
            logger.info(f"API Response: {response.status_code} - Version: {api_version}")

            return response

        except Exception as e:
            logger.error(f"Version middleware after_request error: {str(e)}")
            return response

    def _create_error_response(self, message: str, status_code: int, details: Dict[str, Any] = None):
        """Tạo error response với version information"""
        error_response = {
            "error": message,
            "timestamp": datetime.now().isoformat(),
            "path": request.path,
            "method": request.method,
            "version_info": {
                "requested_version": getattr(g, 'api_version', 'unknown'),
                "supported_versions": list(version_manager.registry.versions.keys())
            }
        }

        if details:
            error_response["details"] = details

        return jsonify(error_response), status_code

    def create_versioned_response(self, data: Any, status_code: int = 200,
                                deprecation_notice: Dict = None,
                                migration_info: Dict = None) -> Response:
        """
        Tạo versioned response

        Args:
            data: Response data
            status_code: HTTP status code
            deprecation_notice: Deprecation notice information
            migration_info: Migration information

        Returns:
            Flask Response object
        """
        if not hasattr(g, 'api_version'):
            return jsonify(data), status_code

        api_version = g.api_version

        # Create versioned response object
        versioned_response = version_manager.create_versioned_response(
            data=data,
            version=api_version,
            deprecation_notice=deprecation_notice,
            migration_info=migration_info
        )

        # Create Flask response
        response_data = versioned_response.dict()

        response = jsonify(response_data)
        response.status_code = status_code

        return response

    def handle_version_migration(self, target_version: str = None) -> Response:
        """
        Handle version migration request

        Args:
            target_version: Target version for migration

        Returns:
            Flask Response with migration information
        """
        current_version = getattr(g, 'api_version', version_manager.registry.default_version)

        if not target_version:
            target_version = version_manager.registry.current_version

        # Validate versions
        if current_version not in version_manager.registry.versions:
            return self._create_error_response(f"Current version {current_version} không tồn tại", 400)

        if target_version not in version_manager.registry.versions:
            return self._create_error_response(f"Target version {target_version} không tồn tại", 400)

        # Get migration path
        migration_path = version_manager.get_migration_path(current_version, target_version)

        if not migration_path:
            return self._create_error_response(
                f"No migration path from {current_version} to {target_version}",
                400
            )

        # Get migration details
        migration = version_manager.registry.get_migration_path(current_version, target_version)

        migration_info = {
            "from_version": current_version,
            "to_version": target_version,
            "migration_type": migration_path.value,
            "breaking_changes": migration.breaking_changes if migration else [],
            "migration_steps": migration.migration_steps if migration else [],
            "estimated_duration": migration.estimated_duration if migration else None,
            "requires_downtime": migration.requires_downtime if migration else False,
            "rollback_possible": migration.rollback_possible if migration else True
        }

        return self.create_versioned_response(
            data={"migration_info": migration_info},
            migration_info=migration_info
        )

    def get_version_status(self) -> Dict[str, Any]:
        """Lấy status của current version"""
        if not hasattr(g, 'api_version'):
            return {"error": "No version detected"}

        api_version = g.api_version
        version_info = getattr(g, 'version_info', {})

        return {
            "current_version": api_version,
            "status": getattr(g, 'version_status', 'unknown'),
            "version_info": version_info,
            "deprecation_warnings": getattr(g, 'deprecation_warnings', []),
            "supported_versions": list(version_manager.registry.versions.keys()),
            "default_version": version_manager.registry.default_version,
            "latest_version": version_manager.registry.get_latest_version().version
        }

    def validate_request_version(self, required_version: str) -> Tuple[bool, Optional[Response]]:
        """
        Validate request version against required version

        Args:
            required_version: Required API version

        Returns:
            Tuple[bool, Optional[Response]]: (is_valid, error_response)
        """
        if not hasattr(g, 'api_version'):
            return False, self._create_error_response("No API version detected", 400)

        current_version = g.api_version

        is_compatible, message = version_manager.validate_version_compatibility(
            current_version, required_version
        )

        if not is_compatible:
            return False, self._create_error_response(
                f"Version compatibility error: {message}",
                400,
                {
                    "current_version": current_version,
                    "required_version": required_version,
                    "compatibility_message": message
                }
            )

        return True, None

    def add_version_headers(self, response: Response) -> Response:
        """Add version headers to response"""
        if hasattr(g, 'api_version'):
            response.headers['X-API-Version'] = g.api_version
            response.headers['X-Version-Status'] = g.version_status

        return response


class VersionMiddlewareError(Exception):
    """Custom exception for version middleware errors"""
    pass


# Utility functions for use in routes
def get_current_version() -> str:
    """Get current API version from request context"""
    return getattr(g, 'api_version', version_manager.registry.default_version)


def get_version_info() -> Dict[str, Any]:
    """Get version information from request context"""
    return getattr(g, 'version_info', {})


def get_deprecation_warnings() -> list:
    """Get deprecation warnings from request context"""
    return getattr(g, 'deprecation_warnings', [])


def create_versioned_response(data: Any, status_code: int = 200,
                            deprecation_notice: Dict = None,
                            migration_info: Dict = None) -> Response:
    """Create versioned response (convenience function)"""
    from flask import current_app
    middleware = getattr(current_app, 'version_middleware', None)
    if middleware:
        return middleware.create_versioned_response(data, status_code, deprecation_notice, migration_info)
    else:
        # Fallback if middleware not initialized
        return jsonify(data), status_code


def require_api_version(required_version: str):
    """
    Decorator để require specific API version

    Args:
        required_version: Required API version
    """
    def decorator(f):
        def decorated_function(*args, **kwargs):
            from flask import current_app
            middleware = getattr(current_app, 'version_middleware', None)

            if middleware:
                is_valid, error_response = middleware.validate_request_version(required_version)
                if not is_valid:
                    return error_response

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def add_version_headers(f):
    """
    Decorator để add version headers to response
    """
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)

        from flask import current_app
        middleware = getattr(current_app, 'version_middleware', None)

        if middleware and isinstance(response, Response):
            response = middleware.add_version_headers(response)

        return response
    return decorated_function