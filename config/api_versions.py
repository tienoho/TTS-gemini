"""
API Version Configuration cho TTS System

Định nghĩa các API versions, compatibility settings, migration policies và deprecation timelines.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from models.api_version import (
    APIVersion,
    APIVersionRegistry,
    VersionCompatibility,
    VersionMigration,
    MigrationType,
    VersionStatus,
    DeprecationNotice
)


# API Version Definitions
API_VERSIONS = {
    "v1.0.0": APIVersion(
        version="v1.0.0",
        status=VersionStatus.ACTIVE,
        release_date=datetime(2024, 1, 1),
        description="Initial TTS API release với basic functionality",
        changelog=[
            "Initial release",
            "Basic TTS functionality",
            "Simple voice synthesis",
            "Basic audio format support"
        ],
        new_features=[
            "Text-to-speech synthesis",
            "Multiple voice support",
            "Basic audio format conversion",
            "Simple API endpoints"
        ],
        feature_flags={
            "enable_streaming": False,
            "enable_batch_processing": False,
            "enable_voice_cloning": False,
            "enable_real_time_processing": False
        },
        sunset_date=datetime(2025, 12, 1)
    ),

    "v1.1.0": APIVersion(
        version="v1.1.0",
        status=VersionStatus.ACTIVE,
        release_date=datetime(2024, 3, 15),
        description="Enhanced TTS với streaming và batch processing",
        changelog=[
            "Added streaming support",
            "Batch processing capabilities",
            "Improved voice quality",
            "Enhanced error handling"
        ],
        new_features=[
            "Real-time audio streaming",
            "Batch TTS processing",
            "Voice quality improvements",
            "Enhanced error responses"
        ],
        feature_flags={
            "enable_streaming": True,
            "enable_batch_processing": True,
            "enable_voice_cloning": False,
            "enable_real_time_processing": False
        },
        sunset_date=datetime(2025, 12, 1)
    ),

    "v1.2.0": APIVersion(
        version="v1.2.0",
        status=VersionStatus.ACTIVE,
        release_date=datetime(2024, 6, 1),
        description="Voice cloning và advanced audio processing",
        changelog=[
            "Voice cloning functionality",
            "Advanced audio processing",
            "Enhanced API security",
            "Performance optimizations"
        ],
        new_features=[
            "Voice cloning capabilities",
            "Advanced audio enhancement",
            "Enhanced security features",
            "Performance improvements"
        ],
        feature_flags={
            "enable_streaming": True,
            "enable_batch_processing": True,
            "enable_voice_cloning": True,
            "enable_real_time_processing": False
        },
        sunset_date=datetime(2025, 12, 1)
    ),

    "v2.0.0": APIVersion(
        version="v2.0.0",
        status=VersionStatus.ACTIVE,
        release_date=datetime(2024, 9, 1),
        description="Major release với real-time processing và enhanced features",
        changelog=[
            "Real-time processing capabilities",
            "Enhanced streaming protocols",
            "Improved batch processing",
            "Advanced voice synthesis",
            "Breaking changes in API structure"
        ],
        breaking_changes=[
            "Changed authentication mechanism",
            "Modified request/response format",
            "Updated error codes",
            "New rate limiting structure"
        ],
        new_features=[
            "Real-time TTS processing",
            "Enhanced WebSocket support",
            "Advanced voice synthesis",
            "Improved streaming performance"
        ],
        feature_flags={
            "enable_streaming": True,
            "enable_batch_processing": True,
            "enable_voice_cloning": True,
            "enable_real_time_processing": True
        },
        sunset_date=datetime(2026, 12, 1)
    ),

    "v2.1.0": APIVersion(
        version="v2.1.0",
        status=VersionStatus.ACTIVE,
        release_date=datetime(2024, 11, 1),
        description="Enhanced real-time processing và multi-tenant support",
        changelog=[
            "Multi-tenant architecture",
            "Enhanced real-time processing",
            "Improved scalability",
            "Advanced analytics"
        ],
        new_features=[
            "Multi-tenant support",
            "Enhanced real-time capabilities",
            "Advanced analytics integration",
            "Improved scalability"
        ],
        feature_flags={
            "enable_streaming": True,
            "enable_batch_processing": True,
            "enable_voice_cloning": True,
            "enable_real_time_processing": True,
            "enable_multi_tenancy": True
        },
        sunset_date=datetime(2026, 12, 1)
    ),

    "v2.2.0": APIVersion(
        version="v2.2.0",
        status=VersionStatus.DEVELOPMENT,
        release_date=datetime(2025, 1, 15),
        description="Next generation TTS với AI enhancements",
        changelog=[
            "AI-powered voice synthesis",
            "Enhanced natural language processing",
            "Advanced emotion detection",
            "Improved audio quality"
        ],
        new_features=[
            "AI-enhanced voice synthesis",
            "Natural language processing",
            "Emotion-aware synthesis",
            "Ultra-high quality audio"
        ],
        feature_flags={
            "enable_streaming": True,
            "enable_batch_processing": True,
            "enable_voice_cloning": True,
            "enable_real_time_processing": True,
            "enable_multi_tenancy": True,
            "enable_ai_enhancement": True,
            "enable_emotion_detection": True
        },
        sunset_date=datetime(2027, 12, 1)
    )
}


# Version Compatibility Matrix
VERSION_COMPATIBILITY = {
    "v1.0.0": VersionCompatibility(
        current_version="v1.0.0",
        compatible_versions=["v1.0.0", "v1.1.0"],
        incompatible_versions=["v2.0.0", "v2.1.0", "v2.2.0"],
        migration_required=True,
        migration_type=MigrationType.BREAKING
    ),
    "v1.1.0": VersionCompatibility(
        current_version="v1.1.0",
        compatible_versions=["v1.0.0", "v1.1.0", "v1.2.0"],
        incompatible_versions=["v2.0.0", "v2.1.0", "v2.2.0"],
        migration_required=True,
        migration_type=MigrationType.BREAKING
    ),
    "v1.2.0": VersionCompatibility(
        current_version="v1.2.0",
        compatible_versions=["v1.0.0", "v1.1.0", "v1.2.0"],
        incompatible_versions=["v2.0.0", "v2.1.0", "v2.2.0"],
        migration_required=True,
        migration_type=MigrationType.BREAKING
    ),
    "v2.0.0": VersionCompatibility(
        current_version="v2.0.0",
        compatible_versions=["v2.0.0", "v2.1.0"],
        incompatible_versions=["v1.0.0", "v1.1.0", "v1.2.0", "v2.2.0"],
        migration_required=False,
        migration_type=MigrationType.AUTOMATIC
    ),
    "v2.1.0": VersionCompatibility(
        current_version="v2.1.0",
        compatible_versions=["v2.0.0", "v2.1.0", "v2.2.0"],
        incompatible_versions=["v1.0.0", "v1.1.0", "v1.2.0"],
        migration_required=False,
        migration_type=MigrationType.AUTOMATIC
    ),
    "v2.2.0": VersionCompatibility(
        current_version="v2.2.0",
        compatible_versions=["v2.1.0", "v2.2.0"],
        incompatible_versions=["v1.0.0", "v1.1.0", "v1.2.0", "v2.0.0"],
        migration_required=False,
        migration_type=MigrationType.AUTOMATIC
    )
}


# Migration Paths
MIGRATION_PATHS = {
    "v1.0.0_to_v1.1.0": VersionMigration(
        from_version="v1.0.0",
        to_version="v1.1.0",
        migration_type=MigrationType.AUTOMATIC,
        breaking_changes=[],
        migration_steps=[
            "Update API endpoints to include streaming support",
            "Add batch processing parameters",
            "Update error handling for new response codes"
        ],
        estimated_duration=5,
        rollback_possible=True,
        requires_downtime=False,
        pre_migration_checks=[
            "Verify API key validity",
            "Check current usage patterns"
        ],
        post_migration_checks=[
            "Test streaming functionality",
            "Verify batch processing works"
        ]
    ),

    "v1.1.0_to_v1.2.0": VersionMigration(
        from_version="v1.1.0",
        to_version="v1.2.0",
        migration_type=MigrationType.AUTOMATIC,
        breaking_changes=[],
        migration_steps=[
            "Enable voice cloning endpoints",
            "Update audio processing parameters",
            "Add security headers"
        ],
        estimated_duration=3,
        rollback_possible=True,
        requires_downtime=False,
        pre_migration_checks=[
            "Check voice cloning permissions"
        ],
        post_migration_checks=[
            "Test voice cloning functionality"
        ]
    ),

    "v1.2.0_to_v2.0.0": VersionMigration(
        from_version="v1.2.0",
        to_version="v2.0.0",
        migration_type=MigrationType.BREAKING,
        breaking_changes=[
            "Authentication mechanism changed",
            "Request/response format updated",
            "Error codes modified"
        ],
        migration_steps=[
            "Update authentication to new system",
            "Modify request format to match v2.0.0",
            "Update error handling for new codes",
            "Test real-time processing features"
        ],
        estimated_duration=30,
        rollback_possible=True,
        requires_downtime=True,
        pre_migration_checks=[
            "Backup current configuration",
            "Test new authentication",
            "Prepare migration scripts"
        ],
        post_migration_checks=[
            "Verify all endpoints work",
            "Test real-time processing",
            "Validate authentication"
        ]
    ),

    "v2.0.0_to_v2.1.0": VersionMigration(
        from_version="v2.0.0",
        to_version="v2.1.0",
        migration_type=MigrationType.AUTOMATIC,
        breaking_changes=[],
        migration_steps=[
            "Enable multi-tenant features",
            "Update tenant-specific endpoints",
            "Configure tenant isolation"
        ],
        estimated_duration=10,
        rollback_possible=True,
        requires_downtime=False,
        pre_migration_checks=[
            "Verify tenant configuration"
        ],
        post_migration_checks=[
            "Test multi-tenant functionality",
            "Verify tenant isolation"
        ]
    ),

    "v2.1.0_to_v2.2.0": VersionMigration(
        from_version="v2.1.0",
        to_version="v2.2.0",
        migration_type=MigrationType.AUTOMATIC,
        breaking_changes=[],
        migration_steps=[
            "Enable AI enhancement features",
            "Update emotion detection parameters",
            "Configure advanced audio processing"
        ],
        estimated_duration=15,
        rollback_possible=True,
        requires_downtime=False,
        pre_migration_checks=[
            "Check AI service availability"
        ],
        post_migration_checks=[
            "Test AI-enhanced synthesis",
            "Verify emotion detection"
        ]
    )
}


# Deprecation Policies
DEPRECATION_POLICIES = {
    "v1.0.0": {
        "deprecated_date": datetime(2024, 6, 1),
        "sunset_date": datetime(2024, 12, 1),
        "replacement_version": "v1.1.0",
        "migration_required": True
    },
    "v1.1.0": {
        "deprecated_date": datetime(2024, 9, 1),
        "sunset_date": datetime(2025, 3, 1),
        "replacement_version": "v1.2.0",
        "migration_required": True
    },
    "v1.2.0": {
        "deprecated_date": datetime(2024, 12, 1),
        "sunset_date": datetime(2025, 6, 1),
        "replacement_version": "v2.0.0",
        "migration_required": True
    }
}


# Version Configuration Settings
VERSION_CONFIG = {
    "default_version": "v2.1.0",
    "current_version": "v2.1.0",
    "minimum_supported_version": "v1.0.0",
    "auto_migration_enabled": True,
    "deprecation_warnings_enabled": True,
    "breaking_change_notifications": True,
    "version_header_detection": True,
    "url_version_detection": True,
    "feature_flag_enforcement": True
}


# Feature Flags Configuration
FEATURE_FLAGS = {
    "enable_streaming": {
        "description": "Enable real-time audio streaming",
        "available_from": "v1.1.0",
        "required_version": "v1.1.0"
    },
    "enable_batch_processing": {
        "description": "Enable batch TTS processing",
        "available_from": "v1.1.0",
        "required_version": "v1.1.0"
    },
    "enable_voice_cloning": {
        "description": "Enable voice cloning functionality",
        "available_from": "v1.2.0",
        "required_version": "v1.2.0"
    },
    "enable_real_time_processing": {
        "description": "Enable real-time TTS processing",
        "available_from": "v2.0.0",
        "required_version": "v2.0.0"
    },
    "enable_multi_tenancy": {
        "description": "Enable multi-tenant architecture",
        "available_from": "v2.1.0",
        "required_version": "v2.1.0"
    },
    "enable_ai_enhancement": {
        "description": "Enable AI-powered enhancements",
        "available_from": "v2.2.0",
        "required_version": "v2.2.0"
    },
    "enable_emotion_detection": {
        "description": "Enable emotion-aware synthesis",
        "available_from": "v2.2.0",
        "required_version": "v2.2.0"
    }
}


# Create API Version Registry
def create_api_version_registry() -> APIVersionRegistry:
    """Tạo API version registry với tất cả cấu hình"""
    registry = APIVersionRegistry(
        versions=API_VERSIONS.copy(),
        current_version=VERSION_CONFIG["current_version"],
        default_version=VERSION_CONFIG["default_version"],
        deprecation_notices=[],
        migration_registry=MIGRATION_PATHS.copy()
    )

    # Add compatibility matrix to each version
    for version_str, compatibility in VERSION_COMPATIBILITY.items():
        if version_str in registry.versions:
            registry.versions[version_str].compatibility_matrix[version_str] = compatibility

    # Add migration paths to versions
    for migration in MIGRATION_PATHS.values():
        from_version = registry.versions[migration.from_version]
        to_version = migration.to_version
        from_version.migration_paths[to_version] = migration.migration_type.value

    # Add deprecation notices
    for version_str, policy in DEPRECATION_POLICIES.items():
        if version_str in registry.versions:
            notice = DeprecationNotice(
                version=version_str,
                replacement_version=policy["replacement_version"],
                deprecation_date=policy["deprecated_date"],
                sunset_date=policy["sunset_date"],
                migration_guide=f"Migration guide from {version_str} to {policy['replacement_version']}",
                breaking_changes=registry.versions[version_str].breaking_changes
            )
            registry.deprecation_notices.append(notice)

            # Update version status if deprecated
            if datetime.now() >= policy["deprecated_date"]:
                registry.versions[version_str].status = VersionStatus.DEPRECATED
            if datetime.now() >= policy["sunset_date"]:
                registry.versions[version_str].status = VersionStatus.SUNSET

    return registry


# Global registry instance
API_VERSION_REGISTRY = create_api_version_registry()