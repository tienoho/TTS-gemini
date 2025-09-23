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
from utils.progress_streamer import (
    create_progress_tracker,
    update_progress,
    mark_request_completed,
    mark_request_failed,
    ProgressStatus
)
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


@tts_bp.route('/generate', methods=['POST'])
@jwt_required()
def generate_audio() -> Tuple[Dict, int]:
    """Generate audio from text.

    Returns:
        JSON response with request details
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Validate request data
        schema = TTSRequestSchema()
        validated_data = schema.load(data)

        # Sanitize text input
        text = SecurityUtils.sanitize_input(validated_data['text'])

        # Check text length
        if len(text) > 5000:
            return jsonify({
                'error': 'Text too long',
                'message': 'Text must be less than 5000 characters'
            }), 400

        # Create audio request
        audio_request = AudioRequest(
            user_id=current_user_id,
            text_content=text,
            voice_name=validated_data.get('voice_name', 'Alnilam'),
            output_format=validated_data.get('output_format', 'wav'),
            speed=validated_data.get('speed', 1.0),
            pitch=validated_data.get('pitch', 0.0),
            metadata={
                'ip_address': request.remote_addr,
                'user_agent': request.user_agent.string,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

        # Mark as processing
        audio_request.mark_as_processing()

        # Save to database
        db.session.add(audio_request)
        db.session.commit()

        # Process audio asynchronously (in a real app, use a task queue)
        process_audio_task(audio_request.id, current_user_id)

        return jsonify({
            'message': 'Audio generation started',
            'request_id': audio_request.id,
            'status': audio_request.status,
            'estimated_time': '10-30 seconds'
        }), 202

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Audio generation failed',
            'message': sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/', methods=['GET'])
@jwt_required()
def get_audio_requests() -> Tuple[Dict, int]:
    """Get paginated list of audio requests.

    Returns:
        JSON response with paginated audio requests
    """
    try:
        current_user_id = get_jwt_identity()
        query_params = request.args

        # Validate pagination parameters
        schema = TTSPaginationSchema()
        pagination_data = schema.load(query_params)

        # Build query
        query = db.session.query(AudioRequest).filter(
            AudioRequest.user_id == current_user_id
        )

        # Apply filters
        if pagination_data.get('status'):
            query = query.filter(AudioRequest.status == pagination_data['status'])

        # Apply sorting
        sort_column = getattr(AudioRequest, pagination_data.get('sort_by', 'created_at'))
        if pagination_data.get('sort_order', 'desc') == 'desc':
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)

        # Get total count
        total = query.count()

        # Apply pagination
        requests = query.offset(
            (pagination_data['page'] - 1) * pagination_data['per_page']
        ).limit(pagination_data['per_page']).all()

        # Calculate pagination metadata
        total_pages = (total + pagination_data['per_page'] - 1) // pagination_data['per_page']

        return jsonify({
            'requests': [req.to_dict() for req in requests],
            'pagination': {
                'page': pagination_data['page'],
                'per_page': pagination_data['per_page'],
                'total_pages': total_pages,
                'total_items': total,
                'has_next': pagination_data['page'] < total_pages,
                'has_prev': pagination_data['page'] > 1
            }
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve requests',
            'message': sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/<int:request_id>', methods=['GET'])
@jwt_required()
def get_audio_request(request_id: int) -> Tuple[Dict, int]:
    """Get specific audio request details.

    Args:
        request_id: Audio request ID

    Returns:
        JSON response with request details
    """
    try:
        current_user_id = get_jwt_identity()

        # Get request
        audio_request = db.session.query(AudioRequest).filter(
            AudioRequest.id == request_id,
            AudioRequest.user_id == current_user_id
        ).first()

        if not audio_request:
            return jsonify({
                'error': 'Request not found',
                'message': 'Audio request not found or access denied'
            }), 404

        return jsonify({
            'request': audio_request.to_dict_with_files()
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve request',
            'message': sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/<int:request_id>/download', methods=['GET'])
@jwt_required()
def download_audio(request_id: int) -> Tuple[Dict, int]:
    """Download audio file.

    Args:
        request_id: Audio request ID

    Returns:
        Audio file as attachment
    """
    try:
        current_user_id = get_jwt_identity()

        # Get request
        audio_request = db.session.query(AudioRequest).filter(
            AudioRequest.id == request_id,
            AudioRequest.user_id == current_user_id
        ).first()

        if not audio_request:
            return jsonify({
                'error': 'Request not found',
                'message': 'Audio request not found or access denied'
            }), 404

        # Check if request is completed
        if audio_request.status != 'completed':
            return jsonify({
                'error': 'Request not ready',
                'message': 'Audio is not ready for download yet'
            }), 400

        # Get audio files
        audio_files = AudioFile.get_by_request(request_id, db.session)
        if not audio_files:
            return jsonify({
                'error': 'No audio files',
                'message': 'No audio files found for this request'
            }), 404
# Return first audio file with path traversal protection
audio_file = audio_files[0]

# Validate file path to prevent directory traversal
if not audio_file.file_path or '..' in audio_file.file_path:
    return jsonify({
        'error': 'Invalid file path',
        'message': 'File path contains invalid characters'
    }), 400

# Ensure file exists and is within allowed directory
if not os.path.exists(audio_file.file_path):
    return jsonify({
        'error': 'File not found',
        'message': 'Audio file not found on disk'
    }), 404

# Additional security check - ensure file is within expected directory
upload_dir = os.path.abspath(current_app.config.get('UPLOAD_FOLDER', 'uploads/audio'))
file_path = os.path.abspath(audio_file.file_path)

if not file_path.startswith(upload_dir):
    return jsonify({
        'error': 'Access denied',
        'message': 'File access not allowed'
    }), 403

return send_file(
    audio_file.file_path,
    as_attachment=True,
    download_name=audio_file.filename,
    mimetype=audio_file.mime_type
)

    except Exception as e:
        return jsonify({
            'error': 'Download failed',
            'message': sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/<int:request_id>', methods=['DELETE'])
@jwt_required()
def delete_audio_request(request_id: int) -> Tuple[Dict, int]:
    """Delete audio request and associated files.

    Args:
        request_id: Audio request ID

    Returns:
        JSON response confirming deletion
    """
    try:
        current_user_id = get_jwt_identity()

        # Get request
        audio_request = db.session.query(AudioRequest).filter(
            AudioRequest.id == request_id,
            AudioRequest.user_id == current_user_id
        ).first()

        if not audio_request:
            return jsonify({
                'error': 'Request not found',
                'message': 'Audio request not found or access denied'
            }), 404

        # Delete associated files from disk
        for audio_file in audio_request.audio_files:
            if os.path.exists(audio_file.file_path):
                os.remove(audio_file.file_path)

        # Delete from database
        db.session.delete(audio_request)
        db.session.commit()

        return jsonify({
            'message': 'Audio request deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Deletion failed',
            'message': sanitize_error_message(str(e))
        }), 400


@tts_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_user_stats() -> Tuple[Dict, int]:
    """Get user's audio generation statistics.

    Returns:
        JSON response with user statistics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get stats
        stats = AudioRequest.get_user_stats(current_user_id, db.session)

        return jsonify({
            'stats': stats
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'message': sanitize_error_message(str(e))
        }), 400


def process_audio_task(request_id: int, user_id: int = None) -> None:
    """Process audio generation task with WebSocket progress streaming.

    Args:
        request_id: Audio request ID to process
        user_id: User ID for authorization check (optional for backward compatibility)
    """
    try:
        # Get request with authorization check if user_id provided
        if user_id:
            audio_request = db.session.query(AudioRequest).filter(
                AudioRequest.id == request_id,
                AudioRequest.user_id == user_id
            ).first()
        else:
            audio_request = db.session.get(AudioRequest, request_id)

        if not audio_request:
            return

        # Create progress tracker
        tracker = create_progress_tracker(str(request_id), user_id)

        # Update progress: Starting
        update_progress(
            str(request_id),
            ProgressStatus.PROCESSING,
            "Starting audio generation...",
            10,
            {"stage": "initialization"}
        )

        # Generate audio with progress updates
        if audio_processor:
            # Update progress: Processing text
            update_progress(
                str(request_id),
                ProgressStatus.PROCESSING,
                "Processing text content...",
                30,
                {"stage": "text_processing"}
            )

            # Generate audio
            audio_data, mime_type = asyncio.run(audio_processor.generate_audio(
                text=audio_request.text_content,
                voice_name=audio_request.voice_name,
                output_format=audio_request.output_format
            ))

            # Update progress: Generating audio
            update_progress(
                str(request_id),
                ProgressStatus.PROCESSING,
                "Generating audio file...",
                70,
                {"stage": "audio_generation"}
            )

            # Calculate file hash
            file_hash = SecurityUtils().calculate_audio_hash(audio_data)

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{audio_request.output_format}"
            ) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Create audio file record
            filename = f"tts_{request_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{audio_request.output_format}"
            file_size = len(audio_data)

            audio_file = AudioFile(
                request_id=request_id,
                file_path=temp_file_path,
                filename=filename,
                mime_type=mime_type,
                file_size=file_size,
                checksum=file_hash
            )

            # Save to database
            db.session.add(audio_file)

            # Update progress: Finalizing
            update_progress(
                str(request_id),
                ProgressStatus.PROCESSING,
                "Finalizing audio file...",
                90,
                {"stage": "finalization"}
            )

            # Update request status
            processing_time = 0  # Would calculate actual time in real implementation
            audio_request.mark_as_completed(processing_time)

            # Mark as completed
            mark_request_completed(
                str(request_id),
                "Audio generation completed successfully",
                {
                    "file_size": file_size,
                    "mime_type": mime_type,
                    "filename": filename,
                    "processing_time": processing_time
                }
            )

            db.session.commit()

        else:
            # Mark as failed if no audio processor
            mark_request_failed(
                str(request_id),
                "Audio processor not initialized"
            )
            audio_request.mark_as_failed("Audio processor not initialized")
            db.session.commit()

    except Exception as e:
        # Mark as failed
        mark_request_failed(
            str(request_id),
            f"Audio generation failed: {str(e)}"
        )
        audio_request.mark_as_failed(str(e))
        db.session.commit()