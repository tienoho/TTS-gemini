"""
TTS (Text-to-Speech) routes for Flask TTS API with production-ready features
"""

import asyncio
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request, send_file, g, current_app
from flask_jwt_extended import get_jwt_identity, jwt_required
from sqlalchemy import desc
from werkzeug.utils import secure_filename

from app.extensions import db
from models import AudioFile, AudioRequest, AudioRequestPriority
from utils.audio_processor import AudioProcessor
from utils.security import SecurityUtils
from utils.validators import TTSPaginationSchema, TTSRequestSchema, sanitize_error_message
from utils.auth import get_auth_service, require_auth, require_api_key
from utils.redis_manager import redis_manager
from utils.logging_service import logging_service
from config import get_settings

tts_bp = Blueprint('tts', __name__, url_prefix='/api/v1/tts')

# Initialize audio processor (will be set by application factory)
audio_processor = None

settings = get_settings()
auth_service = get_auth_service()


def init_audio_processor(api_key: str):
    """Initialize audio processor with API key.

    Args:
        api_key: Google Gemini API key
    """
    global audio_processor
    audio_processor = AudioProcessor(api_key)


@tts_bp.route('/request', methods=['POST'])
@require_auth
def create_tts_request():
    """Create a new TTS request with production features."""
    start_time = datetime.utcnow()
    user_id = auth_service.get_user_id()

    try:
        # Check rate limit
        rate_limit_result = asyncio.run(
            auth_service.check_rate_limit(str(user_id), "requests")
        )

        if not rate_limit_result["is_under_limit"]:
            logging_service.log_error(
                request_id=0,
                user_id=user_id,
                message="Rate limit exceeded",
                error_code="RATE_LIMIT_EXCEEDED"
            )

            return jsonify({
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {rate_limit_result['limit_value']}, "
                          f"Remaining: {rate_limit_result['remaining']}, "
                          f"Reset in: {rate_limit_result['reset_seconds']} seconds"
            }), 429

        # Get request data
        data = request.get_json() or {}

        # Basic validation
        if not data.get('text'):
            return jsonify({
                "error": "Missing text",
                "message": "Text content is required"
            }), 400

        # Sanitize text input
        text = SecurityUtils.sanitize_input(data['text'])

        # Check text length limits
        if len(text) > 5000:
            return jsonify({
                "error": "Text too long",
                "message": "Text must be less than 5000 characters"
            }), 400

        if len(text) < 10:
            return jsonify({
                "error": "Text too short",
                "message": "Text must be at least 10 characters"
            }), 400

        # Extract voice settings
        voice_settings = data.get('voice_settings', {})
        if not voice_settings:
            voice_settings = {
                "language": data.get('language', 'vi'),
                "voice_name": data.get('voice_name', 'default'),
                "speed": data.get('speed', 1.0),
                "pitch": data.get('pitch', 0.0)
            }

        # Determine priority
        priority = data.get('priority', 'normal')
        if priority not in ['low', 'normal', 'high', 'urgent']:
            priority = 'normal'

        # Create audio request with production features
        audio_request = AudioRequest(
            user_id=user_id,
            text_content=text,
            language=voice_settings.get('language', 'vi'),
            voice_settings=voice_settings,
            priority=priority,
            output_format=data.get('output_format', 'mp3'),
            storage_type=data.get('storage_type', 'local'),
            metadata={
                'ip_address': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', ''),
                'request_id_header': request.headers.get('X-Request-ID', ''),
                'timestamp': datetime.utcnow().isoformat(),
                'estimated_cost': _calculate_estimated_cost(text, priority)
            }
        )

        # Save to database
        db.session.add(audio_request)
        db.session.commit()

        # Log request creation
        logging_service.log_request(
            request_id=audio_request.id,
            user_id=user_id,
            message="TTS request created successfully",
            level="info",
            component="api",
            operation="create_request",
            details={
                "text_length": len(text),
                "priority": priority,
                "output_format": audio_request.output_format
            }
        )

        # Enqueue for processing
        asyncio.run(
            redis_manager.enqueue_request(
                {
                    "id": audio_request.id,
                    "user_id": user_id,
                    "text": text,
                    "voice_settings": voice_settings,
                    "output_format": audio_request.output_format,
                    "priority": priority
                },
                priority=priority
            )
        )

        # Increment rate limit
        asyncio.run(auth_service.increment_rate_limit(str(user_id), "requests"))

        # Log API performance
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        logging_service.log_api_request(
            endpoint="/api/v1/tts/request",
            method="POST",
            status_code=202,
            duration=processing_time,
            user_id=user_id
        )

        return jsonify({
            "message": "TTS request created successfully",
            "request_id": audio_request.id,
            "status": audio_request.status,
            "priority": audio_request.priority,
            "estimated_time": _get_estimated_processing_time(priority),
            "queue_position": asyncio.run(redis_manager.get_queue_length(priority))
        }), 202

    except Exception as e:
        db.session.rollback()
        error_msg = sanitize_error_message(str(e))

        logging_service.log_error(
            request_id=0,
            user_id=user_id,
            message=f"TTS request creation failed: {error_msg}",
            error_code="REQUEST_CREATION_ERROR"
        )

        return jsonify({
            "error": "Request creation failed",
            "message": error_msg
        }), 400


@tts_bp.route('/status/<int:request_id>', methods=['GET'])
@require_auth
def get_tts_status(request_id: int):
    """Get TTS request status with detailed information."""
    user_id = auth_service.get_user_id()

    try:
        # Get request from database
        audio_request = db.session.query(AudioRequest).filter(
            AudioRequest.id == request_id,
            AudioRequest.user_id == user_id
        ).first()

        if not audio_request:
            return jsonify({
                "error": "Request not found",
                "message": "Audio request not found or access denied"
            }), 404

        # Get status from cache first
        cached_status = asyncio.run(redis_manager.get_request_status(str(request_id)))

        # Get logs for this request
        logs = logging_service.get_request_logs(request_id, limit=10)

        # Calculate progress
        progress = audio_request.progress
        if audio_request.status == "processing":
            # Estimate progress based on time elapsed
            elapsed = datetime.utcnow() - audio_request.started_at
            estimated_total = audio_request.estimated_duration or 30  # seconds
            progress = min(90, (elapsed.total_seconds() / estimated_total) * 100)

        response_data = {
            "request_id": audio_request.id,
            "status": audio_request.status,
            "priority": audio_request.priority,
            "progress": progress,
            "created_at": audio_request.created_at.isoformat() if audio_request.created_at else None,
            "started_at": audio_request.started_at.isoformat() if audio_request.started_at else None,
            "completed_at": audio_request.completed_at.isoformat() if audio_request.completed_at else None,
            "estimated_completion": _get_estimated_completion_time(audio_request),
            "processing_time": audio_request.processing_time,
            "error_message": audio_request.error_message,
            "error_code": audio_request.error_code,
            "retry_count": audio_request.retry_count,
            "max_retries": audio_request.max_retries,
            "output_path": audio_request.output_path,
            "storage_type": audio_request.storage_type,
            "processing_cost": audio_request.processing_cost,
            "storage_cost": audio_request.storage_cost,
            "total_cost": audio_request.get_total_cost(),
            "logs": [
                {
                    "level": log.level,
                    "message": log.message,
                    "timestamp": log.created_at.isoformat() if log.created_at else None,
                    "duration": log.duration
                }
                for log in logs
            ]
        }

        # Add download URL if completed
        if audio_request.status == "completed" and audio_request.output_path:
            response_data["download_url"] = f"{request.host_url}api/v1/tts/result/{request_id}"

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to get status",
            "message": sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/result/<int:request_id>', methods=['GET'])
@require_auth
def download_tts_result(request_id: int):
    """Download TTS result file."""
    user_id = auth_service.get_user_id()

    try:
        # Check rate limit for downloads
        rate_limit_result = asyncio.run(
            auth_service.check_rate_limit(str(user_id), "downloads")
        )

        if not rate_limit_result["is_under_limit"]:
            return jsonify({
                "error": "Download rate limit exceeded",
                "message": f"Too many downloads. Limit: {rate_limit_result['limit_value']}, "
                          f"Remaining: {rate_limit_result['remaining']}"
            }), 429

        # Get request
        audio_request = db.session.query(AudioRequest).filter(
            AudioRequest.id == request_id,
            AudioRequest.user_id == user_id
        ).first()

        if not audio_request:
            return jsonify({
                "error": "Request not found",
                "message": "Audio request not found or access denied"
            }), 404

        # Check if request is completed
        if audio_request.status != "completed":
            return jsonify({
                "error": "Request not ready",
                "message": f"Audio is not ready for download. Status: {audio_request.status}"
            }), 400

        # Check if download is allowed
        if not audio_request.allow_download:
            return jsonify({
                "error": "Download not allowed",
                "message": "Download is disabled for this request"
            }), 403

        # Get audio files
        audio_files = db.session.query(AudioFile).filter(
            AudioFile.request_id == request_id
        ).all()

        if not audio_files:
            return jsonify({
                "error": "No audio files",
                "message": "No audio files found for this request"
            }), 404

        audio_file = audio_files[0]

        # Security checks
        if not audio_file.file_path or '..' in audio_file.file_path:
            return jsonify({
                "error": "Invalid file path",
                "message": "File path contains invalid characters"
            }), 400

        if not os.path.exists(audio_file.file_path):
            return jsonify({
                "error": "File not found",
                "message": "Audio file not found on disk"
            }), 404

        # Additional security check - ensure file is within expected directory
        upload_dir = os.path.abspath(settings.OUTPUT_FOLDER)
        file_path = os.path.abspath(audio_file.file_path)

        if not file_path.startswith(upload_dir):
            return jsonify({
                "error": "Access denied",
                "message": "File access not allowed"
            }), 403

        # Increment download rate limit
        asyncio.run(auth_service.increment_rate_limit(str(user_id), "downloads"))

        # Log download
        logging_service.log_request(
            request_id=request_id,
            user_id=user_id,
            message="Audio file downloaded",
            level="info",
            component="api",
            operation="download_file"
        )

        return send_file(
            audio_file.file_path,
            as_attachment=True,
            download_name=audio_file.filename,
            mimetype=audio_file.mime_type
        )

    except Exception as e:
        return jsonify({
            "error": "Download failed",
            "message": sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/request/<int:request_id>', methods=['DELETE'])
@require_auth
def cancel_tts_request(request_id: int):
    """Cancel a TTS request if it's still pending or processing."""
    user_id = auth_service.get_user_id()

    try:
        # Get request
        audio_request = db.session.query(AudioRequest).filter(
            AudioRequest.id == request_id,
            AudioRequest.user_id == user_id
        ).first()

        if not audio_request:
            return jsonify({
                "error": "Request not found",
                "message": "Audio request not found or access denied"
            }), 404

        # Check if request can be cancelled
        if audio_request.status not in ["pending", "processing"]:
            return jsonify({
                "error": "Cannot cancel request",
                "message": f"Request status '{audio_request.status}' cannot be cancelled"
            }), 400

        # Mark as cancelled
        audio_request.mark_as_cancelled()
        db.session.commit()

        # Log cancellation
        logging_service.log_request(
            request_id=request_id,
            user_id=user_id,
            message="TTS request cancelled by user",
            level="info",
            component="api",
            operation="cancel_request"
        )

        return jsonify({
            "message": "Request cancelled successfully",
            "request_id": request_id,
            "status": audio_request.status
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "error": "Cancellation failed",
            "message": sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/requests', methods=['GET'])
@require_auth
def get_tts_requests():
    """Get paginated list of TTS requests with advanced filtering."""
    user_id = auth_service.get_user_id()

    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        status = request.args.get('status')
        priority = request.args.get('priority')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')

        # Validate pagination
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 20

        # Build query
        query = db.session.query(AudioRequest).filter(
            AudioRequest.user_id == user_id
        )

        # Apply filters
        if status:
            query = query.filter(AudioRequest.status == status)

        if priority:
            query = query.filter(AudioRequest.priority == priority)

        # Apply sorting
        sort_column = getattr(AudioRequest, sort_by, AudioRequest.created_at)
        if sort_order == 'desc':
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)

        # Get total count
        total = query.count()

        # Apply pagination
        requests = query.offset((page - 1) * per_page).limit(per_page).all()

        # Calculate pagination metadata
        total_pages = (total + per_page - 1) // per_page

        return jsonify({
            "requests": [req.to_dict() for req in requests],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "total_items": total,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "filters": {
                "status": status,
                "priority": priority,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve requests",
            "message": sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/stats', methods=['GET'])
@require_auth
def get_user_tts_stats():
    """Get comprehensive user TTS statistics."""
    user_id = auth_service.get_user_id()

    try:
        # Get stats from database
        stats = AudioRequest.get_user_stats(user_id, db.session)

        # Get queue positions
        queue_lengths = asyncio.run(redis_manager.get_all_queue_lengths())

        # Get recent activity
        recent_requests = db.session.query(AudioRequest).filter(
            AudioRequest.user_id == user_id
        ).order_by(AudioRequest.created_at.desc()).limit(5).all()

        return jsonify({
            "stats": stats,
            "queue_status": queue_lengths,
            "recent_requests": [req.to_dict() for req in recent_requests],
            "limits": {
                "requests_per_minute": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                "storage_per_hour": settings.RATE_LIMIT_STORAGE_PER_HOUR,
                "downloads_per_hour": settings.RATE_LIMIT_DOWNLOADS_PER_HOUR
            }
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve statistics",
            "message": sanitize_error_message(str(e))
        }), 400


# Utility functions
def _calculate_estimated_cost(text: str, priority: str) -> float:
    """Calculate estimated cost for TTS request."""
    # Base cost per character
    base_rate = 0.0001  # $0.0001 per character

    # Priority multiplier
    priority_multipliers = {
        "low": 0.8,
        "normal": 1.0,
        "high": 1.5,
        "urgent": 2.0
    }

    multiplier = priority_multipliers.get(priority, 1.0)
    estimated_cost = len(text) * base_rate * multiplier

    return round(estimated_cost, 6)


def _get_estimated_processing_time(priority: str) -> str:
    """Get estimated processing time based on priority."""
    time_estimates = {
        "low": "30-60 seconds",
        "normal": "10-30 seconds",
        "high": "5-15 seconds",
        "urgent": "1-5 seconds"
    }

    return time_estimates.get(priority, "10-30 seconds")


def _get_estimated_completion_time(request: AudioRequest) -> Optional[str]:
    """Get estimated completion time for request."""
    if request.status == "completed":
        return request.completed_at.isoformat() if request.completed_at else None

    if request.status == "processing" and request.estimated_duration:
        estimated_completion = request.started_at + timedelta(seconds=request.estimated_duration)
        return estimated_completion.isoformat()

    return None