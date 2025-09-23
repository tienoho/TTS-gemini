"""
Voice Cloning API Routes for TTS System
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from sqlalchemy.orm import Session

from app.extensions import db
from models.voice_cloning import (
    VoiceModel, VoiceSample, VoiceQualityMetrics, VoiceVersion, VoiceTestResult,
    VoiceModelStatus, VoiceSampleStatus, VoiceModelType
)
from utils.voice_trainer import get_voice_training_service
from utils.voice_library import get_voice_library_manager
from utils.audio_preprocessor import get_audio_preprocessor
from utils.voice_quality import get_voice_quality_assessor
from utils.exceptions import ValidationException, AudioProcessingException
from utils.validators import VoiceModelSchema, VoiceSampleSchema, TrainingConfigSchema
from config.voice_cloning import get_voice_cloning_config

voice_cloning_bp = Blueprint('voice_cloning', __name__)
logger = logging.getLogger(__name__)


@voice_cloning_bp.route('/voice-models', methods=['POST'])
@jwt_required()
def create_voice_model():
    """Create a new voice model.

    Request Body:
    {
        "name": "My Voice Model",
        "description": "Optional description",
        "language": "vi",
        "gender": "male",
        "age_group": "adult",
        "accent": "northern",
        "model_type": "standard"
    }

    Returns:
        JSON response with created voice model
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Validate request data
        schema = VoiceModelSchema()
        validated_data = schema.load(data)

        # Get user's organization
        from models.user import User
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        organization_id = user.organization_id
        if not organization_id:
            return jsonify({'error': 'User must be part of an organization'}), 400

        # Create voice model
        voice_model = VoiceModel(
            name=validated_data['name'],
            description=validated_data.get('description'),
            organization_id=organization_id,
            created_by=current_user_id,
            language=validated_data.get('language', 'vi'),
            gender=validated_data.get('gender'),
            age_group=validated_data.get('age_group'),
            accent=validated_data.get('accent'),
            model_type=VoiceModelType(validated_data.get('model_type', 'standard'))
        )

        db.session.add(voice_model)
        db.session.commit()

        logger.info(f"Voice model created: {voice_model.id} by user {current_user_id}")
        return jsonify({
            'message': 'Voice model created successfully',
            'voice_model': voice_model.to_dict()
        }), 201

    except ValidationException as e:
        return jsonify({'error': 'Validation error', 'message': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating voice model: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to create voice model'}), 500


@voice_cloning_bp.route('/voice-models', methods=['GET'])
@jwt_required()
def get_voice_models():
    """Get voice models for the current user/organization.

    Query Parameters:
    - language: Filter by language
    - status: Filter by status
    - quality_min: Minimum quality score
    - limit: Number of results (default: 50)
    - offset: Pagination offset (default: 0)

    Returns:
        JSON response with voice models list
    """
    try:
        current_user_id = get_jwt_identity()
        from models.user import User
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        organization_id = user.organization_id
        if not organization_id:
            return jsonify({'error': 'User must be part of an organization'}), 400

        # Get query parameters
        language = request.args.get('language')
        status = request.args.get('status')
        quality_min = request.args.get('quality_min', type=float)
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        # Build query
        query = db.session.query(VoiceModel).filter(
            VoiceModel.organization_id == organization_id,
            VoiceModel.deleted_at.is_(None)
        )

        if language:
            query = query.filter(VoiceModel.language == language)

        if status:
            query = query.filter(VoiceModel.status == status)

        if quality_min is not None:
            query = query.filter(VoiceModel.quality_score >= quality_min)

        # Order by creation date (newest first)
        query = query.order_by(VoiceModel.created_at.desc())

        # Apply pagination
        total_count = query.count()
        voice_models = query.offset(offset).limit(limit).all()

        # Convert to dictionaries
        result = []
        for model in voice_models:
            model_dict = model.to_dict()

            # Add sample count
            sample_count = db.session.query(db.func.count(VoiceSample.id)).filter(
                VoiceSample.voice_model_id == model.id,
                VoiceSample.status.in_([VoiceSampleStatus.PROCESSED, VoiceSampleStatus.VALIDATED])
            ).scalar()
            model_dict['sample_count'] = sample_count

            result.append(model_dict)

        return jsonify({
            'voice_models': result,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total_count
        }), 200

    except Exception as e:
        logger.error(f"Error getting voice models: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to get voice models'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>', methods=['GET'])
@jwt_required()
def get_voice_model(model_id: int):
    """Get a specific voice model.

    Returns:
        JSON response with voice model details
    """
    try:
        current_user_id = get_jwt_identity()
        from models.user import User
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        organization_id = user.organization_id
        if not organization_id:
            return jsonify({'error': 'User must be part of an organization'}), 400

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.organization_id == organization_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found'}), 404

        model_dict = voice_model.to_dict()

        # Add additional information
        model_dict['sample_count'] = db.session.query(db.func.count(VoiceSample.id)).filter(
            VoiceSample.voice_model_id == model_id,
            VoiceSample.status.in_([VoiceSampleStatus.PROCESSED, VoiceSampleStatus.VALIDATED])
        ).scalar()

        # Add latest quality metrics
        latest_metrics = VoiceQualityMetrics.get_latest_by_model(model_id, db.session)
        if latest_metrics:
            model_dict['quality_metrics'] = latest_metrics.to_dict()

        # Add version information
        active_version = VoiceVersion.get_active_version(model_id, db.session)
        if active_version:
            model_dict['active_version'] = active_version.to_dict()

        return jsonify({'voice_model': model_dict}), 200

    except Exception as e:
        logger.error(f"Error getting voice model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to get voice model'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>', methods=['PUT'])
@jwt_required()
def update_voice_model(model_id: int):
    """Update a voice model.

    Request Body:
    {
        "name": "Updated Name",
        "description": "Updated description",
        "is_public": true,
        "tags": ["tag1", "tag2"]
    }

    Returns:
        JSON response with updated voice model
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        # Update fields
        if 'name' in data:
            voice_model.name = data['name']

        if 'description' in data:
            voice_model.description = data['description']

        if 'is_public' in data:
            voice_model.is_public = data['is_public']

        if 'tags' in data:
            voice_model.tags = data['tags']

        voice_model.updated_at = datetime.utcnow()
        db.session.commit()

        logger.info(f"Voice model updated: {model_id} by user {current_user_id}")
        return jsonify({
            'message': 'Voice model updated successfully',
            'voice_model': voice_model.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating voice model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to update voice model'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>', methods=['DELETE'])
@jwt_required()
def delete_voice_model(model_id: int):
    """Delete a voice model.

    Returns:
        JSON response confirming deletion
    """
    try:
        current_user_id = get_jwt_identity()

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        # Soft delete
        voice_model.deleted_at = datetime.utcnow()
        voice_model.is_active = False
        db.session.commit()

        logger.info(f"Voice model deleted: {model_id} by user {current_user_id}")
        return jsonify({'message': 'Voice model deleted successfully'}), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting voice model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to delete voice model'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>/samples', methods=['POST'])
@jwt_required()
def upload_voice_sample(model_id: int):
    """Upload a voice sample for a model.

    Request should include:
    - audio file in form data
    - filename parameter

    Returns:
        JSON response with uploaded sample information
    """
    try:
        current_user_id = get_jwt_identity()

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        # Check if file is provided
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Validate file
        config = get_voice_cloning_config()
        max_file_size = config.get_config_value('api', 'max_file_size_mb') * 1024 * 1024
        if len(audio_file.read()) > max_file_size:
            return jsonify({'error': 'File too large'}), 400
        audio_file.seek(0)  # Reset file pointer

        # Validate file format
        allowed_formats = config.get_config_value('api', 'supported_upload_formats')
        if not any(fmt in audio_file.content_type.lower() for fmt in allowed_formats):
            return jsonify({'error': f'Unsupported file format. Allowed: {allowed_formats}'}), 400

        # Save file temporarily
        filename = secure_filename(audio_file.filename)
        temp_path = config.get_path_config()['temp_path']
        import os
        os.makedirs(temp_path, exist_ok=True)

        temp_file_path = os.path.join(temp_path, f"{model_id}_{current_user_id}_{filename}")
        audio_file.save(temp_file_path)

        try:
            # Preprocess audio
            preprocessor = get_audio_preprocessor()
            validation_result = await preprocessor.validate_audio_for_cloning(temp_file_path)

            if not validation_result['is_valid']:
                return jsonify({
                    'error': 'Audio validation failed',
                    'issues': validation_result['issues'],
                    'recommendations': validation_result['recommendations']
                }), 400

            # Create voice sample record
            import hashlib
            with open(temp_file_path, 'rb') as f:
                file_data = f.read()
                checksum = hashlib.sha256(file_data).hexdigest()

            # Check if sample already exists
            existing_sample = VoiceSample.get_by_checksum(checksum, db.session)
            if existing_sample:
                os.remove(temp_file_path)
                return jsonify({'error': 'This audio sample has already been uploaded'}), 409

            # Move to permanent storage
            samples_dir = config.voice_library_path / "samples" / str(voice_model.organization_id) / str(model_id)
            samples_dir.mkdir(parents=True, exist_ok=True)

            permanent_path = samples_dir / filename
            import shutil
            shutil.move(temp_file_path, permanent_path)

            # Create sample record
            sample = VoiceSample(
                voice_model_id=model_id,
                uploaded_by=current_user_id,
                filename=filename,
                original_filename=audio_file.filename,
                file_path=str(permanent_path),
                file_size=os.path.getsize(permanent_path),
                mime_type=audio_file.content_type,
                checksum=checksum,
                duration=validation_result['duration'],
                sample_rate=validation_result['sample_rate'],
                quality_score=validation_result['quality_score'],
                snr_ratio=validation_result['snr_db']
            )

            db.session.add(sample)
            db.session.commit()

            logger.info(f"Voice sample uploaded: {sample.id} for model {model_id}")
            return jsonify({
                'message': 'Voice sample uploaded successfully',
                'sample': sample.to_dict(),
                'validation_result': validation_result
            }), 201

        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise

    except ValidationException as e:
        return jsonify({'error': 'Validation error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error uploading voice sample for model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to upload voice sample'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>/samples', methods=['GET'])
@jwt_required()
def get_voice_samples(model_id: int):
    """Get voice samples for a model.

    Returns:
        JSON response with voice samples list
    """
    try:
        current_user_id = get_jwt_identity()

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        # Get samples
        samples = VoiceSample.get_by_voice_model(model_id, db.session)

        return jsonify({
            'samples': [sample.to_dict() for sample in samples],
            'total_count': len(samples)
        }), 200

    except Exception as e:
        logger.error(f"Error getting voice samples for model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to get voice samples'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>/train', methods=['POST'])
@jwt_required()
def train_voice_model(model_id: int):
    """Start training a voice model.

    Request Body:
    {
        "training_config": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "epochs": 100
        }
    }

    Returns:
        JSON response with training initiation status
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        # Validate training config
        training_config = data.get('training_config', {})
        schema = TrainingConfigSchema()
        validated_config = schema.load(training_config)

        # Start training asynchronously
        training_service = get_voice_training_service(db.session)

        # Run training in background task
        asyncio.create_task(
            training_service.train_voice_model(model_id, current_user_id, validated_config)
        )

        logger.info(f"Voice model training started: {model_id} by user {current_user_id}")
        return jsonify({
            'message': 'Voice model training started',
            'voice_model_id': model_id,
            'estimated_duration': '30-60 minutes'
        }), 202

    except ValidationException as e:
        return jsonify({'error': 'Validation error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error starting voice model training {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to start training'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>/training-status', methods=['GET'])
@jwt_required()
def get_training_status(model_id: int):
    """Get training status for a voice model.

    Returns:
        JSON response with training status
    """
    try:
        current_user_id = get_jwt_identity()

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        # Get training service
        training_service = get_voice_training_service(db.session)
        status = await training_service.get_training_progress(model_id)

        return jsonify({'training_status': status}), 200

    except ValidationException as e:
        return jsonify({'error': 'Validation error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error getting training status for model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to get training status'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>/cancel-training', methods=['POST'])
@jwt_required()
def cancel_training(model_id: int):
    """Cancel training for a voice model.

    Returns:
        JSON response confirming cancellation
    """
    try:
        current_user_id = get_jwt_identity()

        # Get training service
        training_service = get_voice_training_service(db.session)
        success = await training_service.cancel_training(model_id, current_user_id)

        if not success:
            return jsonify({'error': 'No active training to cancel'}), 400

        logger.info(f"Voice model training cancelled: {model_id} by user {current_user_id}")
        return jsonify({'message': 'Training cancelled successfully'}), 200

    except ValidationException as e:
        return jsonify({'error': 'Validation error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error cancelling training for model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to cancel training'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>/quality', methods=['POST'])
@jwt_required()
def assess_voice_quality(model_id: int):
    """Assess voice model quality.

    Request Body:
    {
        "test_texts": ["Test text 1", "Test text 2"],
        "assessment_method": "automated"
    }

    Returns:
        JSON response with quality assessment
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        if voice_model.status != VoiceModelStatus.TRAINED:
            return jsonify({'error': 'Voice model must be trained before quality assessment'}), 400

        # Get quality assessor
        quality_assessor = get_voice_quality_assessor(db.session)
        test_texts = data.get('test_texts')
        assessment_method = data.get('assessment_method', 'automated')

        # Run quality assessment
        quality_result = await quality_assessor.assess_voice_model_quality(
            model_id, test_texts, assessment_method
        )

        logger.info(f"Voice quality assessed: {model_id} by user {current_user_id}")
        return jsonify({
            'message': 'Voice quality assessed successfully',
            'quality_assessment': quality_result
        }), 200

    except ValidationException as e:
        return jsonify({'error': 'Validation error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error assessing voice quality for model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to assess voice quality'}), 500


@voice_cloning_bp.route('/voice-models/<int:model_id>/test', methods=['POST'])
@jwt_required()
def test_voice_model(model_id: int):
    """Test a voice model with sample text.

    Request Body:
    {
        "text": "Sample text to test",
        "output_format": "wav"
    }

    Returns:
        JSON response with test result
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Get voice model
        voice_model = db.session.query(VoiceModel).filter(
            VoiceModel.id == model_id,
            VoiceModel.created_by == current_user_id,
            VoiceModel.deleted_at.is_(None)
        ).first()

        if not voice_model:
            return jsonify({'error': 'Voice model not found or access denied'}), 404

        if not voice_model.can_be_used():
            return jsonify({'error': 'Voice model is not ready for use'}), 400

        # Validate input
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text is required for testing'}), 400

        if len(text) > 1000:
            return jsonify({'error': 'Text too long (max 1000 characters)'}), 400

        # Create test result record
        test_result = VoiceTestResult(
            voice_model_id=model_id,
            test_type='inference',
            test_name='manual_test',
            test_input=text,
            success=True,  # Will be updated based on actual result
            tested_by=current_user_id,
            tested_at=datetime.utcnow()
        )

        db.session.add(test_result)
        db.session.commit()

        # In a real implementation, this would:
        # 1. Generate audio using the voice model
        # 2. Save the audio file
        # 3. Return the audio URL

        # For now, simulate the test
        test_result.success = True
        test_result.test_result = {
            'text_length': len(text),
            'estimated_duration': len(text) * 0.1,  # Rough estimate
            'status': 'completed'
        }
        db.session.commit()

        logger.info(f"Voice model test completed: {model_id} by user {current_user_id}")
        return jsonify({
            'message': 'Voice model test completed',
            'test_result': test_result.to_dict()
        }), 200

    except ValidationException as e:
        return jsonify({'error': 'Validation error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error testing voice model {model_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to test voice model'}), 500


@voice_cloning_bp.route('/public-voices', methods=['GET'])
def get_public_voices():
    """Get public voice models.

    Query Parameters:
    - language: Filter by language
    - quality_min: Minimum quality score
    - limit: Number of results (default: 20)
    - offset: Pagination offset (default: 0)

    Returns:
        JSON response with public voice models
    """
    try:
        # Get query parameters
        language = request.args.get('language')
        quality_min = request.args.get('quality_min', type=float)
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)

        # Build query for public models
        query = db.session.query(VoiceModel).filter(
            VoiceModel.is_public == True,
            VoiceModel.is_active == True,
            VoiceModel.status == VoiceModelStatus.TRAINED,
            VoiceModel.deleted_at.is_(None)
        )

        if language:
            query = query.filter(VoiceModel.language == language)

        if quality_min is not None:
            query = query.filter(VoiceModel.quality_score >= quality_min)

        # Order by quality score
        query = query.order_by(VoiceModel.quality_score.desc())

        # Apply pagination
        total_count = query.count()
        voice_models = query.offset(offset).limit(limit).all()

        # Convert to dictionaries
        result = []
        for model in voice_models:
            model_dict = model.to_dict()

            # Add sample count
            model_dict['sample_count'] = db.session.query(db.func.count(VoiceSample.id)).filter(
                VoiceSample.voice_model_id == model.id,
                VoiceSample.status.in_([VoiceSampleStatus.PROCESSED, VoiceSampleStatus.VALIDATED])
            ).scalar()

            # Add latest quality metrics
            latest_metrics = VoiceQualityMetrics.get_latest_by_model(model.id, db.session)
            if latest_metrics:
                model_dict['quality_metrics'] = latest_metrics.to_dict()

            result.append(model_dict)

        return jsonify({
            'public_voices': result,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total_count
        }), 200

    except Exception as e:
        logger.error(f"Error getting public voices: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to get public voices'}), 500


@voice_cloning_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_voice_statistics():
    """Get voice cloning statistics for the current user/organization.

    Returns:
        JSON response with statistics
    """
    try:
        current_user_id = get_jwt_identity()
        from models.user import User
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        organization_id = user.organization_id
        if not organization_id:
            return jsonify({'error': 'User must be part of an organization'}), 400

        # Get library manager
        library_manager = get_voice_library_manager(db.session)
        statistics = await library_manager.get_voice_statistics(organization_id)

        return jsonify({'statistics': statistics}), 200

    except Exception as e:
        logger.error(f"Error getting voice statistics: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': 'Failed to get statistics'}), 500