"""
Audio Enhancement routes for Flask TTS API with production-ready features
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
from models import AudioFile
from models.audio_enhancement import AudioEnhancement, EnhancementPreset, AudioQualityMetric
from utils.audio_enhancer import audio_enhancer
from utils.audio_quality_analyzer import audio_quality_analyzer
from utils.security import SecurityUtils
from utils.validators import AudioEnhancementSchema, QualityAnalysisSchema, BatchEnhancementSchema
from utils.auth import get_auth_service, require_auth, require_api_key
from utils.redis_manager import redis_manager
from utils.logging_service import logging_service
from config import get_settings

enhancement_bp = Blueprint('audio_enhancement', __name__, url_prefix='/api/v1/audio')

settings = get_settings()
auth_service = get_auth_service()


@enhancement_bp.route('/enhance', methods=['POST'])
@jwt_required()
def enhance_audio() -> Tuple[Dict, int]:
    """Apply audio enhancement to an existing audio file.

    Returns:
        JSON response with enhancement results
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Validate request data
        schema = AudioEnhancementSchema()
        validated_data = schema.load(data)

        # Get audio file
        audio_file_id = validated_data['audio_file_id']
        audio_file = db.session.query(AudioFile).filter(
            AudioFile.id == audio_file_id
        ).first()

        if not audio_file:
            return jsonify({
                'error': 'Audio file not found',
                'message': 'Audio file not found'
            }), 404

        # Check if user owns the audio file
        if audio_file.request.user_id != current_user_id:
            return jsonify({
                'error': 'Access denied',
                'message': 'You do not have permission to enhance this audio file'
            }), 403

        # Validate enhancement settings
        enhancement_type = validated_data['enhancement_type']
        settings = validated_data.get('settings', {})

        validated_settings = audio_enhancer.validate_enhancement_settings(enhancement_type, settings)

        # Read audio file data
        if not os.path.exists(audio_file.file_path):
            return jsonify({
                'error': 'File not found',
                'message': 'Audio file not found on disk'
            }), 404

        with open(audio_file.file_path, 'rb') as f:
            audio_data = f.read()

        # Apply enhancement
        enhanced_data, processing_info = audio_enhancer.enhance_audio(
            audio_data=audio_data,
            enhancement_type=enhancement_type,
            settings=validated_settings
        )

        # Create enhancement record
        enhancement = AudioEnhancement(
            audio_file_id=audio_file_id,
            user_id=current_user_id,
            enhancement_type=enhancement_type,
            settings=validated_settings,
            original_snr=processing_info.get('original_snr'),
            enhanced_snr=processing_info.get('enhanced_snr'),
            processing_time=processing_info['processing_time'],
            file_size_before=processing_info['original_size'],
            file_size_after=processing_info['enhanced_size'],
            success=True
        )

        # Save enhanced audio file
        enhanced_filename = f"enhanced_{audio_file.filename}"
        enhanced_file_path = audio_file.file_path.replace(audio_file.filename, enhanced_filename)

        with open(enhanced_file_path, 'wb') as f:
            f.write(enhanced_data)

        # Create enhanced audio file record
        enhanced_audio_file = AudioFile(
            request_id=audio_file.request_id,
            file_path=enhanced_file_path,
            filename=enhanced_filename,
            mime_type=audio_file.mime_type,
            file_size=len(enhanced_data),
            checksum=SecurityUtils().calculate_audio_hash(enhanced_data)
        )

        # Save to database
        db.session.add(enhancement)
        db.session.add(enhanced_audio_file)
        db.session.commit()

        return jsonify({
            'message': 'Audio enhancement completed successfully',
            'enhancement_id': enhancement.id,
            'enhanced_file_id': enhanced_audio_file.id,
            'processing_info': processing_info,
            'improvement_metrics': enhancement.get_improvement_metrics()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Audio enhancement failed',
            'message': str(e)
        }), 400


@enhancement_bp.route('/quality', methods=['POST'])
@jwt_required()
def analyze_audio_quality() -> Tuple[Dict, int]:
    """Analyze audio quality of an existing audio file.

    Returns:
        JSON response with quality analysis results
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Validate request data
        schema = QualityAnalysisSchema()
        validated_data = schema.load(data)

        # Get audio file
        audio_file_id = validated_data['audio_file_id']
        audio_file = db.session.query(AudioFile).filter(
            AudioFile.id == audio_file_id
        ).first()

        if not audio_file:
            return jsonify({
                'error': 'Audio file not found',
                'message': 'Audio file not found'
            }), 404

        # Check if user owns the audio file
        if audio_file.request.user_id != current_user_id:
            return jsonify({
                'error': 'Access denied',
                'message': 'You do not have permission to analyze this audio file'
            }), 403

        # Read audio file data
        if not os.path.exists(audio_file.file_path):
            return jsonify({
                'error': 'File not found',
                'message': 'Audio file not found on disk'
            }), 404

        with open(audio_file.file_path, 'rb') as f:
            audio_data = f.read()

        # Analyze audio quality
        analysis_results = audio_quality_analyzer.analyze_audio_quality(
            audio_data=audio_data,
            analysis_method=validated_data.get('analysis_method', 'algorithmic')
        )

        # Create quality metric record
        quality_metric = audio_quality_analyzer.create_quality_metric(
            audio_file_id=audio_file_id,
            analysis_results=analysis_results
        )

        # Save to database
        db.session.add(quality_metric)
        db.session.commit()

        # Get recommendations
        recommendations = audio_quality_analyzer.get_quality_recommendations(
            analysis_results['overall_quality_score'],
            analysis_results
        )

        return jsonify({
            'message': 'Audio quality analysis completed',
            'quality_metric_id': quality_metric.id,
            'analysis_results': analysis_results,
            'recommendations': recommendations,
            'quality_grade': quality_metric.get_quality_grade()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Audio quality analysis failed',
            'message': str(e)
        }), 400


@enhancement_bp.route('/enhance/batch', methods=['POST'])
@jwt_required()
def batch_enhance_audio() -> Tuple[Dict, int]:
    """Apply enhancement to multiple audio files.

    Returns:
        JSON response with batch enhancement results
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Validate request data
        schema = BatchEnhancementSchema()
        validated_data = schema.load(data)

        audio_file_ids = validated_data['audio_file_ids']
        enhancement_type = validated_data['enhancement_type']
        settings = validated_data.get('settings', {})

        # Validate enhancement settings
        validated_settings = audio_enhancer.validate_enhancement_settings(enhancement_type, settings)

        results = []
        failed_files = []

        # Process each audio file
        for audio_file_id in audio_file_ids:
            try:
                # Get audio file
                audio_file = db.session.query(AudioFile).filter(
                    AudioFile.id == audio_file_id
                ).first()

                if not audio_file:
                    failed_files.append({
                        'audio_file_id': audio_file_id,
                        'error': 'Audio file not found'
                    })
                    continue

                # Check ownership
                if audio_file.request.user_id != current_user_id:
                    failed_files.append({
                        'audio_file_id': audio_file_id,
                        'error': 'Access denied'
                    })
                    continue

                # Read audio file data
                if not os.path.exists(audio_file.file_path):
                    failed_files.append({
                        'audio_file_id': audio_file_id,
                        'error': 'File not found on disk'
                    })
                    continue

                with open(audio_file.file_path, 'rb') as f:
                    audio_data = f.read()

                # Apply enhancement
                enhanced_data, processing_info = audio_enhancer.enhance_audio(
                    audio_data=audio_data,
                    enhancement_type=enhancement_type,
                    settings=validated_settings
                )

                # Create enhancement record
                enhancement = AudioEnhancement(
                    audio_file_id=audio_file_id,
                    user_id=current_user_id,
                    enhancement_type=enhancement_type,
                    settings=validated_settings,
                    processing_time=processing_info['processing_time'],
                    file_size_before=processing_info['original_size'],
                    file_size_after=processing_info['enhanced_size'],
                    success=True
                )

                # Save enhanced audio file
                enhanced_filename = f"enhanced_{audio_file.filename}"
                enhanced_file_path = audio_file.file_path.replace(audio_file.filename, enhanced_filename)

                with open(enhanced_file_path, 'wb') as f:
                    f.write(enhanced_data)

                # Create enhanced audio file record
                enhanced_audio_file = AudioFile(
                    request_id=audio_file.request_id,
                    file_path=enhanced_file_path,
                    filename=enhanced_filename,
                    mime_type=audio_file.mime_type,
                    file_size=len(enhanced_data),
                    checksum=SecurityUtils().calculate_audio_hash(enhanced_data)
                )

                # Save to database
                db.session.add(enhancement)
                db.session.add(enhanced_audio_file)

                results.append({
                    'audio_file_id': audio_file_id,
                    'enhancement_id': enhancement.id,
                    'enhanced_file_id': enhanced_audio_file.id,
                    'processing_info': processing_info,
                    'improvement_metrics': enhancement.get_improvement_metrics()
                })

            except Exception as e:
                failed_files.append({
                    'audio_file_id': audio_file_id,
                    'error': str(e)
                })

        # Commit all successful enhancements
        db.session.commit()

        return jsonify({
            'message': f'Batch enhancement completed. {len(results)} successful, {len(failed_files)} failed',
            'results': results,
            'failed_files': failed_files,
            'total_processed': len(audio_file_ids),
            'success_count': len(results),
            'failure_count': len(failed_files)
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Batch enhancement failed',
            'message': str(e)
        }), 400


@enhancement_bp.route('/enhancement/presets', methods=['GET'])
@jwt_required()
def get_enhancement_presets() -> Tuple[Dict, int]:
    """Get enhancement presets.

    Returns:
        JSON response with enhancement presets
    """
    try:
        current_user_id = get_jwt_identity()
        preset_type = request.args.get('type', None)
        include_system = request.args.get('include_system', 'true').lower() == 'true'

        presets = []

        # Get system presets if requested
        if include_system:
            system_presets = EnhancementPreset.get_system_presets(db.session)
            presets.extend([preset.to_dict() for preset in system_presets])

        # Get user presets
        user_presets = EnhancementPreset.get_user_presets(current_user_id, db.session)
        presets.extend([preset.to_dict() for preset in user_presets])

        # Filter by type if specified
        if preset_type:
            presets = [p for p in presets if p['enhancement_type'] == preset_type]

        return jsonify({
            'presets': presets,
            'total_count': len(presets)
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve presets',
            'message': str(e)
        }), 400


@enhancement_bp.route('/enhancement/presets', methods=['POST'])
@jwt_required()
def create_enhancement_preset() -> Tuple[Dict, int]:
    """Create a custom enhancement preset.

    Returns:
        JSON response with created preset
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Validate required fields
        if not data.get('name'):
            return jsonify({
                'error': 'Validation error',
                'message': 'Preset name is required'
            }), 400

        if not data.get('enhancement_type'):
            return jsonify({
                'error': 'Validation error',
                'message': 'Enhancement type is required'
            }), 400

        if not data.get('settings'):
            return jsonify({
                'error': 'Validation error',
                'message': 'Enhancement settings are required'
            }), 400

        # Validate enhancement settings
        enhancement_type = data['enhancement_type']
        settings = data['settings']

        validated_settings = audio_enhancer.validate_enhancement_settings(enhancement_type, settings)

        # Create preset
        preset = EnhancementPreset(
            name=data['name'],
            description=data.get('description'),
            user_id=current_user_id,
            is_system_preset=False,
            enhancement_type=enhancement_type,
            settings=validated_settings
        )

        # Save to database
        db.session.add(preset)
        db.session.commit()

        return jsonify({
            'message': 'Enhancement preset created successfully',
            'preset': preset.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to create preset',
            'message': str(e)
        }), 400


@enhancement_bp.route('/enhancement/presets/<int:preset_id>', methods=['PUT'])
@jwt_required()
def update_enhancement_preset(preset_id: int) -> Tuple[Dict, int]:
    """Update an enhancement preset.

    Args:
        preset_id: Preset ID to update

    Returns:
        JSON response with updated preset
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Get preset
        preset = db.session.query(EnhancementPreset).filter(
            EnhancementPreset.id == preset_id,
            EnhancementPreset.user_id == current_user_id
        ).first()

        if not preset:
            return jsonify({
                'error': 'Preset not found',
                'message': 'Enhancement preset not found or access denied'
            }), 404

        # Update fields
        if 'name' in data:
            preset.name = data['name']

        if 'description' in data:
            preset.description = data['description']

        if 'settings' in data:
            # Validate new settings
            validated_settings = audio_enhancer.validate_enhancement_settings(
                preset.enhancement_type, data['settings']
            )
            preset.settings = validated_settings

        # Save to database
        db.session.commit()

        return jsonify({
            'message': 'Enhancement preset updated successfully',
            'preset': preset.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to update preset',
            'message': str(e)
        }), 400


@enhancement_bp.route('/enhancement/presets/<int:preset_id>', methods=['DELETE'])
@jwt_required()
def delete_enhancement_preset(preset_id: int) -> Tuple[Dict, int]:
    """Delete an enhancement preset.

    Args:
        preset_id: Preset ID to delete

    Returns:
        JSON response confirming deletion
    """
    try:
        current_user_id = get_jwt_identity()

        # Get preset
        preset = db.session.query(EnhancementPreset).filter(
            EnhancementPreset.id == preset_id,
            EnhancementPreset.user_id == current_user_id
        ).first()

        if not preset:
            return jsonify({
                'error': 'Preset not found',
                'message': 'Enhancement preset not found or access denied'
            }), 404

        # Check if it's a system preset
        if preset.is_system_preset:
            return jsonify({
                'error': 'Cannot delete system preset',
                'message': 'System presets cannot be deleted'
            }), 400

        # Delete from database
        db.session.delete(preset)
        db.session.commit()

        return jsonify({
            'message': 'Enhancement preset deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to delete preset',
            'message': str(e)
        }), 400


@enhancement_bp.route('/enhancement/history', methods=['GET'])
@jwt_required()
def get_enhancement_history() -> Tuple[Dict, int]:
    """Get user's enhancement history.

    Returns:
        JSON response with enhancement history
    """
    try:
        current_user_id = get_jwt_identity()
        limit = request.args.get('limit', 50, type=int)

        # Get enhancement history
        enhancements = AudioEnhancement.get_by_user(current_user_id, db.session, limit=limit)

        return jsonify({
            'enhancements': [enhancement.to_dict() for enhancement in enhancements],
            'total_count': len(enhancements)
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve enhancement history',
            'message': str(e)
        }), 400


@enhancement_bp.route('/enhancement/stats', methods=['GET'])
@jwt_required()
def get_enhancement_stats() -> Tuple[Dict, int]:
    """Get user's enhancement statistics.

    Returns:
        JSON response with enhancement statistics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get stats
        stats = AudioEnhancement.get_enhancement_stats(db.session, user_id=current_user_id)

        return jsonify({
            'stats': stats
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve enhancement statistics',
            'message': str(e)
        }), 400