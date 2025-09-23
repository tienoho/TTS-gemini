"""
Version Management Service cho TTS System

Xử lý version detection, routing, backward compatibility, migration utilities và version validation.
"""

import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from functools import wraps

from flask import request, g, jsonify, current_app
from werkzeug.exceptions import HTTPException

from models.api_version import (
    APIVersionRegistry,
    APIVersion,
    VersionStatus,
    MigrationType,
    VersionedRequest,
    VersionedResponse,
    get_version_status,
    is_version_compatible,
    parse_version_string
)
from config.api_versions import API_VERSION_REGISTRY, VERSION_CONFIG, FEATURE_FLAGS


logger = logging.getLogger(__name__)


class VersionManager:
    """Service quản lý API versioning"""

    def __init__(self, registry: APIVersionRegistry = None):
        self.registry = registry or API_VERSION_REGISTRY
        self._version_cache: Dict[str, Any] = {}
        self._migration_cache: Dict[str, Any] = {}

    def detect_version(self, request_data: Optional[Dict] = None) -> str:
        """
        Detect API version từ request

        Priority order:
        1. URL path parameter (/v1/endpoint)
        2. X-API-Version header
        3. Accept-Version header
        4. Query parameter (?version=v1)
        5. Default version
        """
        if request_data is None:
            request_data = {}

        # 1. Check URL path
        url_version = self._extract_version_from_url()
        if url_version and self._is_valid_version(url_version):
            return url_version

        # 2. Check X-API-Version header
        header_version = request.headers.get('X-API-Version')
        if header_version and self._is_valid_version(header_version):
            return header_version

        # 3. Check Accept-Version header
        accept_version = request.headers.get('Accept-Version')
        if accept_version and self._is_valid_version(accept_version):
            return accept_version

        # 4. Check query parameter
        query_version = request.args.get('version')
        if query_version and self._is_valid_version(query_version):
            return query_version

        # 5. Return default version
        return self.registry.default_version

    def _extract_version_from_url(self) -> Optional[str]:
        """Extract version từ URL path"""
        if not request:
            return None

        path = request.path
        # Pattern: /v{major}.{minor}.{patch}/ hoặc /v{major}.{minor}/ hoặc /v{major}/
        version_pattern = r'/v(\d+)\.(\d+)\.(\d+)/|/v(\d+)\.(\d+)/|/v(\d+)/'
        match = re.search(version_pattern, path)

        if match:
            if match.group(3):  # Full semantic version
                return f"v{match.group(1)}.{match.group(2)}.{match.group(3)}"
            elif match.group(5):  # Major.minor
                return f"v{match.group(4)}.{match.group(5)}.0"
            else:  # Major only
                return f"v{match.group(6)}.0.0"

        return None

    def _is_valid_version(self, version_str: str) -> bool:
        """Validate version string"""
        return version_str in self.registry.versions

    def validate_version_compatibility(self, requested_version: str, required_version: str) -> Tuple[bool, Optional[str]]:
        """
        Validate compatibility giữa requested version và required version

        Returns:
            Tuple[bool, Optional[str]]: (is_compatible, migration_message)
        """
        cache_key = f"{requested_version}_{required_version}"
        if cache_key in self._version_cache:
            return self._version_cache[cache_key]

        if requested_version not in self.registry.versions:
            result = (False, f"Version {requested_version} không tồn tại")
        elif required_version not in self.registry.versions:
            result = (False, f"Required version {required_version} không tồn tại")
        else:
            requested_ver = self.registry.versions[requested_version]
            required_ver = self.registry.versions[required_version]

            if requested_ver.status == VersionStatus.SUNSET:
                result = (False, f"Version {requested_version} đã bị sunset")
            elif requested_ver.status == VersionStatus.DEPRECATED:
                migration_path = self.get_migration_path(requested_version, required_version)
                if migration_path:
                    result = (True, f"Version {requested_version} deprecated. Migration available to {required_version}")
                else:
                    result = (False, f"Version {requested_version} deprecated và không tương thích với {required_version}")
            else:
                is_compatible = requested_ver.is_compatible_with(required_version)
                if is_compatible:
                    result = (True, None)
                else:
                    migration_path = self.get_migration_path(requested_version, required_version)
                    if migration_path:
                        result = (True, f"Migration required from {requested_version} to {required_version}")
                    else:
                        result = (False, f"Version {requested_version} không tương thích với {required_version}")

        self._version_cache[cache_key] = result
        return result

    def get_migration_path(self, from_version: str, to_version: str) -> Optional[MigrationType]:
        """Lấy migration path giữa hai version"""
        cache_key = f"{from_version}_to_{to_version}"
        if cache_key in self._migration_cache:
            return self._migration_cache[cache_key]

        migration = self.registry.get_migration_path(from_version, to_version)
        if migration:
            result = migration.migration_type
        else:
            # Check if direct compatibility exists
            if from_version in self.registry.versions and to_version in self.registry.versions:
                from_ver = self.registry.versions[from_version]
                result = from_ver.migration_paths.get(to_version)
            else:
                result = None

        self._migration_cache[cache_key] = result
        return result

    def check_feature_compatibility(self, version: str, feature: str) -> Tuple[bool, Optional[str]]:
        """
        Check if feature is compatible với version

        Returns:
            Tuple[bool, Optional[str]]: (is_compatible, message)
        """
        if version not in self.registry.versions:
            return False, f"Version {version} không tồn tại"

        version_obj = self.registry.versions[version]

        if feature not in FEATURE_FLAGS:
            return False, f"Feature {feature} không được định nghĩa"

        feature_config = FEATURE_FLAGS[feature]
        required_version = feature_config["required_version"]

        if version_obj.version == required_version:
            return True, None
        elif parse_version_string(version_obj.version) >= parse_version_string(required_version):
            return True, None
        else:
            return False, f"Feature {feature} requires version {required_version} or higher"

    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết về version"""
        if version not in self.registry.versions:
            return None

        version_obj = self.registry.versions[version]
        return {
            "version": version_obj.version,
            "status": version_obj.status.value,
            "release_date": version_obj.release_date.isoformat(),
            "sunset_date": version_obj.sunset_date.isoformat() if version_obj.sunset_date else None,
            "description": version_obj.description,
            "new_features": version_obj.new_features,
            "breaking_changes": version_obj.breaking_changes,
            "deprecated_features": version_obj.deprecated_features,
            "feature_flags": version_obj.feature_flags,
            "is_deprecated": version_obj.is_deprecated(),
            "is_sunset": version_obj.is_sunset(),
            "days_until_sunset": version_obj.days_until_sunset()
        }

    def get_compatible_versions(self, version: str) -> List[str]:
        """Lấy danh sách version tương thích"""
        if version not in self.registry.versions:
            return []

        version_obj = self.registry.versions[version]
        compatible = []

        for other_version in self.registry.versions.keys():
            if version_obj.is_compatible_with(other_version):
                compatible.append(other_version)

        return compatible

    def create_versioned_response(self, data: Any, version: str,
                                deprecation_notice: Optional[Dict] = None,
                                migration_info: Optional[Dict] = None) -> VersionedResponse:
        """Tạo versioned response"""
        if version not in self.registry.versions:
            raise ValueError(f"Invalid version: {version}")

        version_obj = self.registry.versions[version]

        return VersionedResponse(
            data=data,
            version=version_obj.version,
            status=version_obj.status.value,
            deprecation_notice=deprecation_notice,
            migration_info=migration_info,
            feature_flags=version_obj.feature_flags
        )

    def handle_version_migration(self, from_version: str, to_version: str,
                               request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle migration logic giữa các version

        Args:
            from_version: Version hiện tại
            to_version: Version đích
            request_data: Request data

        Returns:
            Dict[str, Any]: Migrated request data
        """
        if from_version == to_version:
            return request_data

        migration_path = self.get_migration_path(from_version, to_version)
        if not migration_path:
            raise ValueError(f"No migration path from {from_version} to {to_version}")

        # Apply migration transformations based on migration type
        migrated_data = request_data.copy()

        if migration_path == MigrationType.AUTOMATIC:
            migrated_data = self._apply_automatic_migration(from_version, to_version, migrated_data)
        elif migration_path == MigrationType.BREAKING:
            migrated_data = self._apply_breaking_changes_migration(from_version, to_version, migrated_data)

        return migrated_data

    def _apply_automatic_migration(self, from_version: str, to_version: str,
                                 request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automatic migration transformations"""
        # Add version-specific parameters
        if from_version.startswith("v1.") and to_version.startswith("v2."):
            # v1 to v2 migration
            if "options" not in request_data:
                request_data["options"] = {}

            # Add default streaming settings for v2
            if to_version >= "v2.0.0":
                request_data["options"]["enable_streaming"] = True

        return request_data

    def _apply_breaking_changes_migration(self, from_version: str, to_version: str,
                                        request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply breaking changes migration transformations"""
        # Handle major version breaking changes
        if from_version == "v1.2.0" and to_version == "v2.0.0":
            # Authentication changes
            if "auth" in request_data:
                request_data["authentication"] = request_data.pop("auth")

            # Response format changes
            if "format" in request_data:
                format_mapping = {
                    "mp3": "audio/mpeg",
                    "wav": "audio/wav",
                    "ogg": "audio/ogg"
                }
                old_format = request_data["format"]
                if old_format in format_mapping:
                    request_data["format"] = format_mapping[old_format]

        return request_data

    def validate_feature_flags(self, version: str, requested_flags: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate feature flags cho version

        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        if version not in self.registry.versions:
            return False, [f"Version {version} không tồn tại"]

        version_obj = self.registry.versions[version]
        errors = []

        for flag, value in requested_flags.items():
            if flag not in FEATURE_FLAGS:
                errors.append(f"Feature flag {flag} không được định nghĩa")
                continue

            feature_config = FEATURE_FLAGS[flag]
            required_version = feature_config["required_version"]

            if parse_version_string(version_obj.version) < parse_version_string(required_version):
                errors.append(f"Feature flag {flag} requires version {required_version} or higher")

        return len(errors) == 0, errors

    def get_deprecation_warnings(self, version: str) -> List[Dict[str, Any]]:
        """Lấy danh sách deprecation warnings cho version"""
        if version not in self.registry.versions:
            return []

        version_obj = self.registry.versions[version]
        warnings = []

        if version_obj.is_deprecated():
            # Find deprecation notice
            for notice in self.registry.deprecation_notices:
                if notice.version == version:
                    warnings.append({
                        "type": "deprecation",
                        "version": notice.version,
                        "replacement_version": notice.replacement_version,
                        "sunset_date": notice.sunset_date.isoformat() if notice.sunset_date else None,
                        "message": f"API version {version} is deprecated. Please migrate to {notice.replacement_version or 'newer version'}",
                        "migration_guide": notice.migration_guide
                    })
                    break

        # Check for deprecated features
        for feature in version_obj.deprecated_features:
            warnings.append({
                "type": "feature_deprecation",
                "feature": feature,
                "message": f"Feature {feature} is deprecated in version {version}"
            })

        return warnings

    def get_supported_versions(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả supported versions"""
        versions = []
        for version_str, version_obj in self.registry.versions.items():
            versions.append({
                "version": version_str,
                "status": version_obj.status.value,
                "release_date": version_obj.release_date.isoformat(),
                "description": version_obj.description,
                "is_current": version_str == self.registry.current_version,
                "is_deprecated": version_obj.is_deprecated(),
                "is_sunset": version_obj.is_sunset()
            })

        return sorted(versions, key=lambda x: parse_version_string(x["version"]), reverse=True)


# Global version manager instance
version_manager = VersionManager()


def require_version(min_version: str = None, max_version: str = None):
    """
    Decorator để enforce version requirements

    Args:
        min_version: Minimum required version
        max_version: Maximum supported version
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get requested version
            requested_version = version_manager.detect_version()

            # Validate version exists
            if requested_version not in version_manager.registry.versions:
                return jsonify({
                    "error": "Invalid API version",
                    "requested_version": requested_version,
                    "supported_versions": list(version_manager.registry.versions.keys())
                }), 400

            # Check minimum version
            if min_version:
                if parse_version_string(requested_version) < parse_version_string(min_version):
                    return jsonify({
                        "error": "API version too old",
                        "requested_version": requested_version,
                        "minimum_version": min_version,
                        "message": f"Version {min_version} or higher is required"
                    }), 400

            # Check maximum version
            if max_version:
                if parse_version_string(requested_version) > parse_version_string(max_version):
                    return jsonify({
                        "error": "API version too new",
                        "requested_version": requested_version,
                        "maximum_version": max_version,
                        "message": f"Version {max_version} or lower is supported"
                    }), 400

            # Add version info to request context
            g.requested_version = requested_version
            g.version_info = version_manager.get_version_info(requested_version)

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def version_compatibility_check(required_version: str):
    """
    Decorator để check version compatibility

    Args:
        required_version: Version required for the endpoint
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            requested_version = version_manager.detect_version()

            is_compatible, message = version_manager.validate_version_compatibility(
                requested_version, required_version
            )

            if not is_compatible:
                return jsonify({
                    "error": "Version compatibility error",
                    "requested_version": requested_version,
                    "required_version": required_version,
                    "message": message
                }), 400

            # Add compatibility info to request context
            g.requested_version = requested_version
            g.required_version = required_version
            g.compatibility_message = message

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def feature_flag_check(*required_flags: str):
    """
    Decorator để check feature flags

    Args:
        *required_flags: Required feature flags
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            requested_version = version_manager.detect_version()

            for flag in required_flags:
                is_compatible, message = version_manager.check_feature_compatibility(
                    requested_version, flag
                )

                if not is_compatible:
                    return jsonify({
                        "error": "Feature not supported",
                        "requested_version": requested_version,
                        "feature": flag,
                        "message": message
                    }), 400

            return f(*args, **kwargs)
        return decorated_function
    return decorator