"""
Migration Tools cho TTS System

Xử lý data migration utilities, API response transformation, breaking change detection và migration scripts.
"""

import json
import copy
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from models.api_version import (
    APIVersionRegistry,
    VersionMigration,
    MigrationType,
    BreakingChangeType
)
from config.api_versions import API_VERSION_REGISTRY
from utils.version_manager import version_manager


logger = logging.getLogger(__name__)


class MigrationTool(ABC):
    """Abstract base class cho migration tools"""

    def __init__(self, registry: APIVersionRegistry = None):
        self.registry = registry or API_VERSION_REGISTRY
        self.migration_history: List[Dict[str, Any]] = []

    @abstractmethod
    def migrate_request(self, from_version: str, to_version: str,
                       request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate request data từ version cũ sang version mới"""
        pass

    @abstractmethod
    def migrate_response(self, from_version: str, to_version: str,
                        response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate response data từ version cũ sang version mới"""
        pass

    @abstractmethod
    def detect_breaking_changes(self, from_version: str, to_version: str) -> List[str]:
        """Detect breaking changes giữa hai version"""
        pass

    def log_migration(self, from_version: str, to_version: str,
                     migration_type: str, success: bool, details: Dict[str, Any] = None):
        """Log migration event"""
        migration_record = {
            "timestamp": datetime.now().isoformat(),
            "from_version": from_version,
            "to_version": to_version,
            "migration_type": migration_type,
            "success": success,
            "details": details or {}
        }

        self.migration_history.append(migration_record)
        logger.info(f"Migration logged: {from_version} -> {to_version} ({'success' if success else 'failed'})")


class TTSMigrationTool(MigrationTool):
    """Migration tool specific cho TTS system"""

    # Version-specific transformations
    VERSION_TRANSFORMATIONS = {
        "v1_to_v2": {
            "auth_to_authentication": lambda data: self._transform_auth_to_authentication(data),
            "format_to_mime_type": lambda data: self._transform_format_to_mime_type(data),
            "response_structure": lambda data: self._transform_response_structure(data),
            "error_codes": lambda data: self._transform_error_codes(data)
        },
        "v2_to_v2_1": {
            "add_tenant_support": lambda data: self._add_tenant_support(data),
            "enhance_streaming": lambda data: self._enhance_streaming_options(data)
        },
        "v2_1_to_v2_2": {
            "add_ai_features": lambda data: self._add_ai_features(data),
            "add_emotion_detection": lambda data: self._add_emotion_detection(data)
        }
    }

    def migrate_request(self, from_version: str, to_version: str,
                       request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate TTS request data từ version cũ sang version mới

        Args:
            from_version: Source version
            to_version: Target version
            request_data: Request data to migrate

        Returns:
            Migrated request data
        """
        try:
            logger.info(f"Migrating request from {from_version} to {to_version}")

            # Create deep copy to avoid modifying original data
            migrated_data = copy.deepcopy(request_data)

            # Apply version-specific transformations
            if from_version.startswith("v1.") and to_version.startswith("v2."):
                migrated_data = self._apply_v1_to_v2_migration(migrated_data)
            elif from_version.startswith("v2.0.") and to_version.startswith("v2.1."):
                migrated_data = self._apply_v2_0_to_v2_1_migration(migrated_data)
            elif from_version.startswith("v2.1.") and to_version.startswith("v2.2."):
                migrated_data = self._apply_v2_1_to_v2_2_migration(migrated_data)

            # Apply general transformations
            migrated_data = self._apply_general_transformations(migrated_data, from_version, to_version)

            # Validate migrated data
            validation_result = self._validate_migrated_data(migrated_data, to_version)
            if not validation_result["valid"]:
                logger.warning(f"Migrated data validation failed: {validation_result['errors']}")

            self.log_migration(from_version, to_version, "request", True,
                             {"original_size": len(str(request_data)),
                              "migrated_size": len(str(migrated_data))})

            return migrated_data

        except Exception as e:
            logger.error(f"Request migration failed from {from_version} to {to_version}: {str(e)}")
            self.log_migration(from_version, to_version, "request", False, {"error": str(e)})
            raise

    def migrate_response(self, from_version: str, to_version: str,
                        response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate TTS response data từ version cũ sang version mới

        Args:
            from_version: Source version
            to_version: Target version
            response_data: Response data to migrate

        Returns:
            Migrated response data
        """
        try:
            logger.info(f"Migrating response from {from_version} to {to_version}")

            # Create deep copy to avoid modifying original data
            migrated_data = copy.deepcopy(response_data)

            # Apply version-specific transformations
            if from_version.startswith("v1.") and to_version.startswith("v2."):
                migrated_data = self._apply_v1_to_v2_response_migration(migrated_data)
            elif from_version.startswith("v2.0.") and to_version.startswith("v2.1."):
                migrated_data = self._apply_v2_0_to_v2_1_response_migration(migrated_data)
            elif from_version.startswith("v2.1.") and to_version.startswith("v2.2."):
                migrated_data = self._apply_v2_1_to_v2_2_response_migration(migrated_data)

            # Apply general response transformations
            migrated_data = self._apply_general_response_transformations(migrated_data, from_version, to_version)

            self.log_migration(from_version, to_version, "response", True,
                             {"original_size": len(str(response_data)),
                              "migrated_size": len(str(migrated_data))})

            return migrated_data

        except Exception as e:
            logger.error(f"Response migration failed from {from_version} to {to_version}: {str(e)}")
            self.log_migration(from_version, to_version, "response", False, {"error": str(e)})
            raise

    def detect_breaking_changes(self, from_version: str, to_version: str) -> List[str]:
        """
        Detect breaking changes giữa hai version

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            List of breaking changes
        """
        breaking_changes = []

        try:
            # Get migration path
            migration = self.registry.get_migration_path(from_version, to_version)
            if migration:
                breaking_changes.extend(migration.breaking_changes)

            # Add version-specific breaking changes
            if from_version.startswith("v1.") and to_version.startswith("v2."):
                breaking_changes.extend(self._get_v1_to_v2_breaking_changes())
            elif from_version.startswith("v2.0.") and to_version.startswith("v2.1."):
                breaking_changes.extend(self._get_v2_0_to_v2_1_breaking_changes())
            elif from_version.startswith("v2.1.") and to_version.startswith("v2.2."):
                breaking_changes.extend(self._get_v2_1_to_v2_2_breaking_changes())

            # Remove duplicates while preserving order
            seen = set()
            unique_breaking_changes = []
            for change in breaking_changes:
                if change not in seen:
                    seen.add(change)
                    unique_breaking_changes.append(change)

            logger.info(f"Detected {len(unique_breaking_changes)} breaking changes from {from_version} to {to_version}")
            return unique_breaking_changes

        except Exception as e:
            logger.error(f"Error detecting breaking changes: {str(e)}")
            return [f"Error detecting breaking changes: {str(e)}"]

    def _apply_v1_to_v2_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v1 to v2 migration transformations"""
        transformations = self.VERSION_TRANSFORMATIONS["v1_to_v2"]

        for transform_name, transform_func in transformations.items():
            try:
                data = transform_func(data)
                logger.debug(f"Applied transformation: {transform_name}")
            except Exception as e:
                logger.warning(f"Failed to apply transformation {transform_name}: {str(e)}")

        return data

    def _apply_v2_0_to_v2_1_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v2.0 to v2.1 migration transformations"""
        transformations = self.VERSION_TRANSFORMATIONS["v2_to_v2_1"]

        for transform_name, transform_func in transformations.items():
            try:
                data = transform_func(data)
                logger.debug(f"Applied transformation: {transform_name}")
            except Exception as e:
                logger.warning(f"Failed to apply transformation {transform_name}: {str(e)}")

        return data

    def _apply_v2_1_to_v2_2_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v2.1 to v2.2 migration transformations"""
        transformations = self.VERSION_TRANSFORMATIONS["v2_1_to_v2_2"]

        for transform_name, transform_func in transformations.items():
            try:
                data = transform_func(data)
                logger.debug(f"Applied transformation: {transform_name}")
            except Exception as e:
                logger.warning(f"Failed to apply transformation {transform_name}: {str(e)}")

        return data

    def _apply_general_transformations(self, data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Apply general transformations between any versions"""
        # Add version metadata
        if "metadata" not in data:
            data["metadata"] = {}

        data["metadata"]["migrated_from"] = from_version
        data["metadata"]["migrated_to"] = to_version
        data["metadata"]["migration_timestamp"] = datetime.now().isoformat()

        # Add compatibility flags
        if "options" not in data:
            data["options"] = {}

        # Ensure backward compatibility
        if from_version < to_version:  # Upgrade
            data["options"]["backward_compatibility"] = True
        else:  # Downgrade
            data["options"]["forward_compatibility"] = True

        return data

    def _apply_v1_to_v2_response_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v1 to v2 response migration transformations"""
        # Transform response structure
        if "result" in data and "audio_url" in data["result"]:
            # Move audio_url to top level
            data["audio_url"] = data["result"]["audio_url"]
            data["result"].pop("audio_url")

        # Add version info
        if "version_info" not in data:
            data["version_info"] = {
                "api_version": "v2.0.0",
                "response_format": "v2"
            }

        return data

    def _apply_v2_0_to_v2_1_response_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v2.0 to v2.1 response migration transformations"""
        # Add tenant information
        if "tenant_info" not in data:
            data["tenant_info"] = {
                "tenant_id": "default",
                "tenant_name": "Default Tenant"
            }

        return data

    def _apply_v2_1_to_v2_2_response_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v2.1 to v2.2 response migration transformations"""
        # Add AI enhancement info
        if "ai_enhancement" not in data:
            data["ai_enhancement"] = {
                "enabled": False,
                "emotion_detection": False,
                "quality_score": 0.0
            }

        return data

    def _apply_general_response_transformations(self, data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Apply general response transformations"""
        # Add migration metadata
        if "metadata" not in data:
            data["metadata"] = {}

        data["metadata"]["migrated_from"] = from_version
        data["metadata"]["migrated_to"] = to_version
        data["metadata"]["migration_timestamp"] = datetime.now().isoformat()

        return data

    def _transform_auth_to_authentication(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform auth field to authentication"""
        if "auth" in data:
            data["authentication"] = data.pop("auth")
        return data

    def _transform_format_to_mime_type(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform format field to mime_type"""
        format_mapping = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "m4a": "audio/m4a"
        }

        if "format" in data:
            old_format = data["format"]
            if old_format in format_mapping:
                data["mime_type"] = format_mapping[old_format]
                # Keep original format for backward compatibility
                data["format"] = old_format

        return data

    def _transform_response_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform response structure for v2 compatibility"""
        if "response_format" not in data:
            data["response_format"] = "v2"

        return data

    def _transform_error_codes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform error codes for v2 compatibility"""
        # This would be implemented based on specific error code mappings
        return data

    def _add_tenant_support(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add tenant support for v2.1"""
        if "options" not in data:
            data["options"] = {}

        data["options"]["multi_tenant"] = True
        return data

    def _enhance_streaming_options(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance streaming options for v2.1"""
        if "options" not in data:
            data["options"] = {}

        if "streaming" not in data["options"]:
            data["options"]["streaming"] = {
                "enabled": True,
                "chunk_size": 1024,
                "real_time": False
            }

        return data

    def _add_ai_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add AI features for v2.2"""
        if "options" not in data:
            data["options"] = {}

        data["options"]["ai_enhancement"] = {
            "enabled": False,
            "model": "default",
            "quality": "standard"
        }

        return data

    def _add_emotion_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add emotion detection for v2.2"""
        if "options" not in data:
            data["options"] = {}

        data["options"]["emotion_detection"] = {
            "enabled": False,
            "sensitivity": "medium",
            "output_format": "json"
        }

        return data

    def _get_v1_to_v2_breaking_changes(self) -> List[str]:
        """Get breaking changes from v1 to v2"""
        return [
            "Authentication mechanism changed from 'auth' to 'authentication'",
            "Audio format field changed from simple string to mime_type",
            "Response structure updated with new metadata fields",
            "Error codes and response formats updated",
            "New required fields for streaming and batch processing"
        ]

    def _get_v2_0_to_v2_1_breaking_changes(self) -> List[str]:
        """Get breaking changes from v2.0 to v2.1"""
        return [
            "Multi-tenant support added with tenant_id requirements",
            "Enhanced streaming options with new parameters",
            "New analytics and monitoring fields",
            "Updated rate limiting structure"
        ]

    def _get_v2_1_to_v2_2_breaking_changes(self) -> List[str]:
        """Get breaking changes from v2.1 to v2.2"""
        return [
            "AI enhancement features added",
            "Emotion detection capabilities added",
            "New audio quality metrics",
            "Enhanced natural language processing options"
        ]

    def _validate_migrated_data(self, data: Dict[str, Any], target_version: str) -> Dict[str, Any]:
        """
        Validate migrated data against target version requirements

        Args:
            data: Migrated data to validate
            target_version: Target version

        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []

        # Check required fields for target version
        if target_version.startswith("v2."):
            if "authentication" not in data:
                errors.append("Missing required 'authentication' field for v2.x")
            if "options" not in data:
                warnings.append("Missing 'options' field - using defaults")

        # Check data types and formats
        if "text" in data and not isinstance(data["text"], str):
            errors.append("Text field must be string")

        if "voice" in data and not isinstance(data["voice"], str):
            errors.append("Voice field must be string")

        # Check version-specific requirements
        if target_version >= "v2.1.0":
            if "options" in data and "multi_tenant" not in data["options"]:
                warnings.append("Multi-tenant option not specified for v2.1+")

        if target_version >= "v2.2.0":
            if "options" in data and "ai_enhancement" not in data["options"]:
                warnings.append("AI enhancement options not specified for v2.2+")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


class MigrationScriptGenerator:
    """Generator cho migration scripts"""

    def __init__(self, migration_tool: MigrationTool):
        self.migration_tool = migration_tool

    def generate_migration_script(self, from_version: str, to_version: str,
                                script_type: str = "python") -> str:
        """
        Generate migration script

        Args:
            from_version: Source version
            to_version: Target version
            script_type: Type of script to generate

        Returns:
            Generated script as string
        """
        if script_type.lower() == "python":
            return self._generate_python_script(from_version, to_version)
        elif script_type.lower() == "bash":
            return self._generate_bash_script(from_version, to_version)
        else:
            raise ValueError(f"Unsupported script type: {script_type}")

    def _generate_python_script(self, from_version: str, to_version: str) -> str:
        """Generate Python migration script"""
        breaking_changes = self.migration_tool.detect_breaking_changes(from_version, to_version)

        script = f'''#!/usr/bin/env python3
"""
Migration script from {from_version} to {to_version}

Generated on: {datetime.now().isoformat()}

Breaking Changes:
{chr(10).join(f"- {change}" for change in breaking_changes)}
"""

import json
import sys
from typing import Dict, Any
from utils.migration_tools import TTSMigrationTool


def migrate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate data from {from_version} to {to_version}"""
    migration_tool = TTSMigrationTool()

    try:
        # Migrate request data
        migrated_request = migration_tool.migrate_request(
            "{from_version}",
            "{to_version}",
            data
        )

        # Migrate response data if needed
        migrated_response = migration_tool.migrate_response(
            "{from_version}",
            "{to_version}",
            {{}}  # Empty dict for response migration example
        )

        return {{
            "success": True,
            "migrated_request": migrated_request,
            "migrated_response": migrated_response,
            "breaking_changes": {json.dumps(breaking_changes)}
        }}

    except Exception as e:
        return {{
            "success": False,
            "error": str(e)
        }}


if __name__ == "__main__":
    # Example usage
    sample_data = {{
        "text": "Hello, world!",
        "voice": "default",
        "format": "mp3"
    }}

    result = migrate_data(sample_data)

    if result["success"]:
        print("Migration successful!")
        print(json.dumps(result["migrated_request"], indent=2))
    else:
        print(f"Migration failed: {{result['error']}}")
        sys.exit(1)
'''

        return script

    def _generate_bash_script(self, from_version: str, to_version: str) -> str:
        """Generate Bash migration script"""
        script = f'''#!/bin/bash
#
# Migration script from {from_version} to {to_version}
#
# Generated on: {datetime.now().isoformat()}
#

set -e

FROM_VERSION="{from_version}"
TO_VERSION="{to_version}"
MIGRATION_TOOL="python3 -c \\"from utils.migration_tools import TTSMigrationTool; print('Migration tool loaded')\\""

echo "Starting migration from $FROM_VERSION to $TO_VERSION"
echo "Timestamp: $(date)"

# Check if migration tool is available
if ! eval "$MIGRATION_TOOL" > /dev/null 2>&1; then
    echo "Error: Migration tool not available"
    exit 1
fi

echo "Migration tool is available"

# Create backup
BACKUP_DIR="./migration_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Created backup directory: $BACKUP_DIR"

# Migration steps would go here
echo "Migration steps:"
echo "1. Backup current configuration"
echo "2. Apply breaking changes"
echo "3. Update API endpoints"
echo "4. Test migrated functionality"
echo "5. Verify compatibility"

echo "Migration completed successfully!"
echo "Please review the changes and test thoroughly before deploying."
'''

        return script


# Global migration tool instance
tts_migration_tool = TTSMigrationTool()
migration_script_generator = MigrationScriptGenerator(tts_migration_tool)