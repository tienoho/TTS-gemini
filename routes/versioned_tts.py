"""
Version-aware API Routes cho TTS System

Cung cấp version-specific endpoints, automatic version detection, backward compatibility và migration endpoints.
"""

from flask import Blueprint, request, jsonify, g, Response
from typing import Dict, Any, Optional
import logging

from utils.version_manager import (
    version_manager,
    require_version,
    version_compatibility_check,
    feature_flag_check
)
from utils.version_middleware import (
    create_versioned_response,
    get_current_version,
    get_version_info,
    get_deprecation_warnings
)
from utils.migration_tools import tts_migration_tool, migration_script_generator
from models.api_version import VersionedResponse
from config.api_versions import VERSION_CONFIG


logger = logging.getLogger(__name__)

# Create blueprint
versioned_tts_bp = Blueprint('versioned_tts', __name__, url_prefix='/api')


@versioned_tts_bp.route('/tts', methods=['POST'])
@require_version(min_version="v1.0.0")
def text_to_speech():
    """
    Text-to-Speech endpoint với version awareness

    Supports multiple API versions với automatic migration và backward compatibility.
    """
    try:
        # Get current version
        current_version = get_current_version()
        version_info = get_version_info()

        # Get request data
        request_data = request.get_json() or {}

        # Log version-specific request
        logger.info(f"TTS Request - Version: {current_version}, Text length: {len(request_data.get('text', ''))}")

        # Handle version-specific logic
        if current_version.startswith("v1."):
            return _handle_v1_tts_request(request_data, current_version)
        elif current_version.startswith("v2."):
            return _handle_v2_tts_request(request_data, current_version)
        else:
            return create_versioned_response(
                {"error": f"Unsupported version: {current_version}"},
                status_code=400
            )

    except Exception as e:
        logger.error(f"TTS request failed: {str(e)}")
        return create_versioned_response(
            {"error": "Internal server error", "details": str(e)},
            status_code=500
        )


@versioned_tts_bp.route('/tts/stream', methods=['POST'])
@require_version(min_version="v1.1.0")
@feature_flag_check("enable_streaming")
def text_to_speech_stream():
    """
    Streaming Text-to-Speech endpoint

    Requires v1.1.0+ và streaming feature flag.
    """
    try:
        current_version = get_current_version()

        # Get request data
        request_data = request.get_json() or {}

        logger.info(f"Streaming TTS Request - Version: {current_version}")

        # Handle streaming logic based on version
        if current_version >= "v2.0.0":
            return _handle_v2_streaming_tts(request_data)
        else:
            return _handle_v1_streaming_tts(request_data)

    except Exception as e:
        logger.error(f"Streaming TTS request failed: {str(e)}")
        return create_versioned_response(
            {"error": "Streaming request failed", "details": str(e)},
            status_code=500
        )


@versioned_tts_bp.route('/tts/batch', methods=['POST'])
@require_version(min_version="v1.1.0")
@feature_flag_check("enable_batch_processing")
def batch_text_to_speech():
    """
    Batch Text-to-Speech endpoint

    Requires v1.1.0+ và batch processing feature flag.
    """
    try:
        current_version = get_current_version()
        request_data = request.get_json() or {}

        logger.info(f"Batch TTS Request - Version: {current_version}, Items: {len(request_data.get('texts', []))}")

        # Handle batch logic based on version
        if current_version >= "v2.0.0":
            return _handle_v2_batch_tts(request_data)
        else:
            return _handle_v1_batch_tts(request_data)

    except Exception as e:
        logger.error(f"Batch TTS request failed: {str(e)}")
        return create_versioned_response(
            {"error": "Batch request failed", "details": str(e)},
            status_code=500
        )


@versioned_tts_bp.route('/voice-cloning', methods=['POST'])
@require_version(min_version="v1.2.0")
@feature_flag_check("enable_voice_cloning")
def voice_cloning():
    """
    Voice Cloning endpoint

    Requires v1.2.0+ và voice cloning feature flag.
    """
    try:
        current_version = get_current_version()
        request_data = request.get_json() or {}

        logger.info(f"Voice Cloning Request - Version: {current_version}")

        # Handle voice cloning based on version
        if current_version >= "v2.0.0":
            return _handle_v2_voice_cloning(request_data)
        else:
            return _handle_v1_voice_cloning(request_data)

    except Exception as e:
        logger.error(f"Voice cloning request failed: {str(e)}")
        return create_versioned_response(
            {"error": "Voice cloning failed", "details": str(e)},
            status_code=500
        )


@versioned_tts_bp.route('/version', methods=['GET'])
def get_version_info():
    """
    Get current API version information

    Returns version details, compatibility info, and deprecation warnings.
    """
    try:
        current_version = get_current_version()
        version_info = get_version_info()
        deprecation_warnings = get_deprecation_warnings()

        response_data = {
            "current_version": current_version,
            "version_info": version_info,
            "supported_versions": list(version_manager.registry.versions.keys()),
            "default_version": version_manager.registry.default_version,
            "latest_version": version_manager.registry.get_latest_version().version,
            "deprecation_warnings": deprecation_warnings,
            "feature_flags": version_info.get("feature_flags", {}) if version_info else {},
            "compatibility_info": {
                "compatible_versions": version_manager.get_compatible_versions(current_version),
                "is_deprecated": version_info.get("is_deprecated", False) if version_info else False,
                "is_sunset": version_info.get("is_sunset", False) if version_info else False
            }
        }

        return create_versioned_response(response_data)

    except Exception as e:
        logger.error(f"Get version info failed: {str(e)}")
        return create_versioned_response(
            {"error": "Failed to get version info", "details": str(e)},
            status_code=500
        )


@versioned_tts_bp.route('/version/migrate', methods=['POST'])
@version_compatibility_check("v1.0.0")
def migrate_version():
    """
    Migrate API usage từ current version to target version

    Provides migration assistance và breaking change information.
    """
    try:
        current_version = get_current_version()
        request_data = request.get_json() or {}

        # Get target version (default to latest)
        target_version = request_data.get('target_version', version_manager.registry.current_version)

        # Validate target version
        if target_version not in version_manager.registry.versions:
            return create_versioned_response(
                {"error": f"Invalid target version: {target_version}"},
                status_code=400
            )

        # Get migration information
        migration_path = version_manager.get_migration_path(current_version, target_version)

        if not migration_path:
            return create_versioned_response(
                {"error": f"No migration path from {current_version} to {target_version}"},
                status_code=400
            )

        # Get breaking changes
        breaking_changes = tts_migration_tool.detect_breaking_changes(current_version, target_version)

        # Get migration steps
        migration = version_manager.registry.get_migration_path(current_version, target_version)

        migration_info = {
            "from_version": current_version,
            "to_version": target_version,
            "migration_type": migration_path.value,
            "breaking_changes": breaking_changes,
            "migration_steps": migration.migration_steps if migration else [],
            "estimated_duration": migration.estimated_duration if migration else None,
            "requires_downtime": migration.requires_downtime if migration else False,
            "rollback_possible": migration.rollback_possible if migration else True,
            "pre_migration_checks": migration.pre_migration_checks if migration else [],
            "post_migration_checks": migration.post_migration_checks if migration else []
        }

        # Generate migration script if requested
        script_type = request_data.get('script_type', 'python')
        if request_data.get('generate_script'):
            try:
                migration_script = migration_script_generator.generate_migration_script(
                    current_version, target_version, script_type
                )
                migration_info["migration_script"] = migration_script
            except Exception as e:
                migration_info["script_generation_error"] = str(e)

        return create_versioned_response(
            {"migration_info": migration_info},
            migration_info=migration_info
        )

    except Exception as e:
        logger.error(f"Migration request failed: {str(e)}")
        return create_versioned_response(
            {"error": "Migration request failed", "details": str(e)},
            status_code=500
        )


@versioned_tts_bp.route('/version/compatibility', methods=['POST'])
def check_version_compatibility():
    """
    Check compatibility between versions

    Validates if migration between versions is possible.
    """
    try:
        request_data = request.get_json() or {}

        from_version = request_data.get('from_version', get_current_version())
        to_version = request_data.get('to_version')

        if not to_version:
            return create_versioned_response(
                {"error": "Target version is required"},
                status_code=400
            )

        # Validate versions exist
        if from_version not in version_manager.registry.versions:
            return create_versioned_response(
                {"error": f"Source version {from_version} không tồn tại"},
                status_code=400
            )

        if to_version not in version_manager.registry.versions:
            return create_versioned_response(
                {"error": f"Target version {to_version} không tồn tại"},
                status_code=400
            )

        # Check compatibility
        is_compatible, message = version_manager.validate_version_compatibility(from_version, to_version)

        # Get detailed compatibility info
        compatibility_info = {
            "from_version": from_version,
            "to_version": to_version,
            "is_compatible": is_compatible,
            "compatibility_message": message,
            "migration_path": version_manager.get_migration_path(from_version, to_version),
            "breaking_changes": tts_migration_tool.detect_breaking_changes(from_version, to_version) if not is_compatible else []
        }

        return create_versioned_response(
            {"compatibility_info": compatibility_info}
        )

    except Exception as e:
        logger.error(f"Compatibility check failed: {str(e)}")
        return create_versioned_response(
            {"error": "Compatibility check failed", "details": str(e)},
            status_code=500
        )


@versioned_tts_bp.route('/version/features', methods=['GET'])
def get_available_features():
    """
    Get available features cho current version

    Returns feature flags và capabilities based on API version.
    """
    try:
        current_version = get_current_version()
        version_info = get_version_info()

        if not version_info:
            return create_versioned_response(
                {"error": "Version info not available"},
                status_code=500
            )

        # Get feature compatibility
        features_info = {}
        for feature, config in version_manager.registry.versions[current_version].feature_flags.items():
            is_compatible, message = version_manager.check_feature_compatibility(current_version, feature)
            features_info[feature] = {
                "enabled": config,
                "available": is_compatible,
                "message": message
            }

        response_data = {
            "version": current_version,
            "features": features_info,
            "version_specific_features": version_info.get("new_features", []),
            "deprecated_features": version_info.get("deprecated_features", [])
        }

        return create_versioned_response(response_data)

    except Exception as e:
        logger.error(f"Get features failed: {str(e)}")
        return create_versioned_response(
            {"error": "Failed to get features", "details": str(e)},
            status_code=500
        )


# Version-specific handler functions
def _handle_v1_tts_request(request_data: Dict[str, Any], version: str) -> Response:
    """Handle v1.x TTS requests"""
    # Basic v1 functionality
    required_fields = ['text', 'voice']
    missing_fields = [field for field in required_fields if field not in request_data]

    if missing_fields:
        return create_versioned_response(
            {"error": f"Missing required fields: {', '.join(missing_fields)}"},
            status_code=400
        )

    # Simulate TTS processing
    response_data = {
        "audio_url": f"/audio/{request_data['voice']}/output.mp3",
        "format": request_data.get('format', 'mp3'),
        "duration": len(request_data['text']) * 0.1,  # Simple estimation
        "version": version
    }

    return create_versioned_response(response_data)


def _handle_v2_tts_request(request_data: Dict[str, Any], version: str) -> Response:
    """Handle v2.x TTS requests với enhanced features"""
    required_fields = ['text', 'authentication']
    missing_fields = [field for field in required_fields if field not in request_data]

    if missing_fields:
        return create_versioned_response(
            {"error": f"Missing required fields: {', '.join(missing_fields)}"},
            status_code=400
        )

    # Enhanced v2 functionality
    response_data = {
        "audio_url": f"/v2/audio/{request_data['voice']}/output.mp3",
        "format": request_data.get('format', 'audio/mpeg'),
        "mime_type": request_data.get('format', 'audio/mpeg'),
        "duration": len(request_data['text']) * 0.1,
        "quality_score": 0.95,
        "processing_time": 0.5,
        "version": version,
        "options": request_data.get('options', {}),
        "metadata": {
            "model_version": "v2.1.0",
            "processing_engine": "enhanced"
        }
    }

    return create_versioned_response(response_data)


def _handle_v1_streaming_tts(request_data: Dict[str, Any]) -> Response:
    """Handle v1.x streaming TTS"""
    response_data = {
        "stream_url": f"/stream/{request_data.get('voice', 'default')}",
        "format": request_data.get('format', 'mp3'),
        "chunk_size": 1024,
        "version": "v1.x"
    }

    return create_versioned_response(response_data)


def _handle_v2_streaming_tts(request_data: Dict[str, Any]) -> Response:
    """Handle v2.x streaming TTS với enhanced features"""
    response_data = {
        "stream_url": f"/v2/stream/{request_data.get('voice', 'default')}",
        "format": request_data.get('format', 'audio/mpeg'),
        "chunk_size": request_data.get('chunk_size', 2048),
        "real_time": request_data.get('real_time', False),
        "quality": request_data.get('quality', 'high'),
        "version": "v2.x",
        "websocket_url": "/v2/ws/tts-stream"
    }

    return create_versioned_response(response_data)


def _handle_v1_batch_tts(request_data: Dict[str, Any]) -> Response:
    """Handle v1.x batch TTS"""
    texts = request_data.get('texts', [])
    if not texts:
        return create_versioned_response(
            {"error": "No texts provided for batch processing"},
            status_code=400
        )

    response_data = {
        "batch_id": f"batch_{hash(str(texts))}",
        "total_items": len(texts),
        "estimated_time": len(texts) * 2,
        "version": "v1.x"
    }

    return create_versioned_response(response_data)


def _handle_v2_batch_tts(request_data: Dict[str, Any]) -> Response:
    """Handle v2.x batch TTS với enhanced features"""
    texts = request_data.get('texts', [])
    if not texts:
        return create_versioned_response(
            {"error": "No texts provided for batch processing"},
            status_code=400
        )

    response_data = {
        "batch_id": f"v2_batch_{hash(str(texts))}",
        "total_items": len(texts),
        "estimated_time": len(texts) * 1.5,
        "priority": request_data.get('priority', 'normal'),
        "webhook_url": request_data.get('webhook_url'),
        "version": "v2.x",
        "options": request_data.get('options', {}),
        "progress_url": "/v2/batch/progress"
    }

    return create_versioned_response(response_data)


def _handle_v1_voice_cloning(request_data: Dict[str, Any]) -> Response:
    """Handle v1.x voice cloning"""
    required_fields = ['name', 'audio_file']
    missing_fields = [field for field in required_fields if field not in request_data]

    if missing_fields:
        return create_versioned_response(
            {"error": f"Missing required fields: {', '.join(missing_fields)}"},
            status_code=400
        )

    response_data = {
        "voice_id": f"voice_{request_data['name']}",
        "status": "training",
        "estimated_training_time": 30,
        "version": "v1.x"
    }

    return create_versioned_response(response_data)


def _handle_v2_voice_cloning(request_data: Dict[str, Any]) -> Response:
    """Handle v2.x voice cloning với enhanced features"""
    required_fields = ['name', 'audio_file', 'authentication']
    missing_fields = [field for field in required_fields if field not in request_data]

    if missing_fields:
        return create_versioned_response(
            {"error": f"Missing required fields: {', '.join(missing_fields)}"},
            status_code=400
        )

    response_data = {
        "voice_id": f"v2_voice_{request_data['name']}",
        "status": "training",
        "estimated_training_time": 25,
        "quality": request_data.get('quality', 'high'),
        "language": request_data.get('language', 'vi'),
        "version": "v2.x",
        "training_progress_url": "/v2/voice-cloning/progress",
        "options": request_data.get('options', {})
    }

    return create_versioned_response(response_data)