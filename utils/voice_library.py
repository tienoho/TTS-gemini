"""
Voice Library Management for TTS Voice Cloning System
"""

import asyncio
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from models.voice_cloning import (
    VoiceModel, VoiceSample, VoiceQualityMetrics, VoiceVersion,
    VoiceModelStatus, VoiceModelType, VoiceQualityScore
)
from utils.exceptions import ValidationException, AudioProcessingException


class VoiceLibraryManager:
    """Manages voice library operations including storage, search, and organization."""

    def __init__(self, db_session: Session, storage_path: str = "voice_library"):
        """Initialize voice library manager."""
        self.db_session = db_session
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(__name__)

        # Storage configuration
        self.max_storage_per_org = 10 * 1024 * 1024 * 1024  # 10GB per organization
        self.max_models_per_org = 100
        self.max_versions_per_model = 10
        self.cleanup_days = 30  # Days to keep failed models

        # Search configuration
        self.search_batch_size = 50
        self.max_search_results = 200

        # Ensure storage directories exist
        self._ensure_storage_directories()

    def _ensure_storage_directories(self):
        """Ensure all necessary storage directories exist."""
        directories = [
            self.storage_path,
            self.storage_path / "models",
            self.storage_path / "samples",
            self.storage_path / "temp",
            self.storage_path / "archives"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    async def store_voice_model(
        self,
        voice_model: VoiceModel,
        model_data: bytes,
        organization_id: int
    ) -> str:
        """Store a trained voice model in the library."""
        try:
            # Check storage limits
            await self._check_storage_limits(organization_id)

            # Generate storage path
            model_dir = self.storage_path / "models" / str(organization_id) / str(voice_model.id)
            model_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with version
            version = VoiceVersion.get_active_version(voice_model.id, self.db_session)
            if not version:
                version_number = "1.0.0"
            else:
                # Increment version number
                major, minor, patch = map(int, version.version_number.split('.'))
                version_number = f"{major}.{minor}.{patch + 1}"

            filename = f"{voice_model.name}_v{version_number}.h5"
            model_path = model_dir / filename

            # Save model file
            with open(model_path, 'wb') as f:
                f.write(model_data)

            # Create version record
            version = VoiceVersion(
                voice_model_id=voice_model.id,
                version_number=version_number,
                model_path=str(model_path),
                model_size_bytes=len(model_data),
                checksum=self._calculate_checksum(model_data),
                created_by=voice_model.created_by
            )

            # Deactivate previous versions
            previous_versions = VoiceVersion.get_version_history(voice_model.id, self.db_session)
            for prev_version in previous_versions:
                if prev_version.is_active:
                    prev_version.deactivate()

            # Activate new version
            version.activate()
            self.db_session.add(version)

            # Update model path and size
            voice_model.model_path = str(model_path)
            voice_model.model_size_bytes = len(model_data)

            self.db_session.commit()

            return str(model_path)

        except Exception as e:
            self.logger.error(f"Error storing voice model {voice_model.id}: {str(e)}")
            raise AudioProcessingException(f"Failed to store voice model: {str(e)}")

    async def retrieve_voice_model(
        self,
        voice_model_id: int,
        organization_id: int,
        version_number: str = None
    ) -> Tuple[bytes, str]:
        """Retrieve a voice model from the library."""
        try:
            # Get voice model
            voice_model = self.db_session.query(VoiceModel).filter(
                VoiceModel.id == voice_model_id,
                VoiceModel.organization_id == organization_id
            ).first()

            if not voice_model:
                raise ValidationException("Voice model not found or access denied")

            if not voice_model.can_be_used():
                raise ValidationException("Voice model is not available for use")

            # Get specific version or active version
            if version_number:
                version = VoiceVersion.get_by_model_and_version(
                    voice_model_id, version_number, self.db_session
                )
            else:
                version = VoiceVersion.get_active_version(voice_model_id, self.db_session)

            if not version:
                raise ValidationException("Voice model version not found")

            # Read model file
            if not os.path.exists(version.model_path):
                raise ValidationException("Voice model file not found on disk")

            with open(version.model_path, 'rb') as f:
                model_data = f.read()

            # Verify integrity
            if not version.verify_integrity(model_data):
                raise ValidationException("Voice model file integrity check failed")

            # Update usage statistics
            voice_model.total_inferences += 1
            voice_model.last_used_at = datetime.utcnow()
            self.db_session.commit()

            return model_data, version.version_number

        except Exception as e:
            self.logger.error(f"Error retrieving voice model {voice_model_id}: {str(e)}")
            raise AudioProcessingException(f"Failed to retrieve voice model: {str(e)}")

    async def search_voices(
        self,
        organization_id: int = None,
        language: str = None,
        gender: str = None,
        quality_min: float = None,
        search_text: str = None,
        tags: List[str] = None,
        model_type: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search for voices in the library."""
        try:
            # Build query
            query = self.db_session.query(VoiceModel).filter(
                VoiceModel.is_active == True,
                VoiceModel.deleted_at.is_(None)
            )

            # Apply organization filter
            if organization_id:
                query = query.filter(VoiceModel.organization_id == organization_id)
            else:
                # Only show public models if no organization specified
                query = query.filter(VoiceModel.is_public == True)

            # Apply filters
            if language:
                query = query.filter(VoiceModel.language == language)

            if gender:
                query = query.filter(VoiceModel.gender == gender)

            if quality_min is not None:
                query = query.filter(VoiceModel.quality_score >= quality_min)

            if model_type:
                query = query.filter(VoiceModel.model_type == model_type)

            # Text search
            if search_text:
                search_filter = or_(
                    VoiceModel.name.ilike(f'%{search_text}%'),
                    VoiceModel.description.ilike(f'%{search_text}%'),
                    VoiceModel.accent.ilike(f'%{search_text}%')
                )
                query = query.filter(search_filter)

            # Tags filter
            if tags:
                for tag in tags:
                    query = query.filter(VoiceModel.tags.contains([tag]))

            # Order by quality and creation date
            query = query.order_by(
                desc(VoiceModel.quality_score),
                desc(VoiceModel.created_at)
            )

            # Apply pagination
            total_count = query.count()
            voices = query.offset(offset).limit(limit).all()

            # Get quality metrics for each voice
            result_voices = []
            for voice in voices:
                voice_dict = voice.to_dict()

                # Add latest quality metrics
                latest_metrics = VoiceQualityMetrics.get_latest_by_model(voice.id, self.db_session)
                if latest_metrics:
                    voice_dict['quality_metrics'] = latest_metrics.to_dict()

                # Add sample count
                sample_count = self.db_session.query(func.count(VoiceSample.id)).filter(
                    VoiceSample.voice_model_id == voice.id,
                    VoiceSample.status.in_([VoiceSampleStatus.PROCESSED, VoiceSampleStatus.VALIDATED])
                ).scalar()
                voice_dict['sample_count'] = sample_count

                result_voices.append(voice_dict)

            return {
                'voices': result_voices,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total_count
            }

        except Exception as e:
            self.logger.error(f"Error searching voices: {str(e)}")
            raise AudioProcessingException(f"Voice search failed: {str(e)}")

    async def get_voice_statistics(
        self,
        organization_id: int = None
    ) -> Dict[str, Any]:
        """Get voice library statistics."""
        try:
            # Build base query
            query = self.db_session.query(VoiceModel)
            if organization_id:
                query = query.filter(VoiceModel.organization_id == organization_id)

            # Get basic counts
            total_models = query.count()
            active_models = query.filter(VoiceModel.is_active == True).count()
            public_models = query.filter(VoiceModel.is_public == True).count()
            trained_models = query.filter(VoiceModel.status == VoiceModelStatus.TRAINED).count()

            # Get quality distribution
            quality_stats = self.db_session.query(
                func.count(VoiceModel.id).label('count'),
                func.avg(VoiceModel.quality_score).label('avg_score'),
                func.min(VoiceModel.quality_score).label('min_score'),
                func.max(VoiceModel.quality_score).label('max_score')
            ).filter(
                VoiceModel.organization_id == organization_id if organization_id else True
            ).first()

            # Get language distribution
            language_stats = self.db_session.query(
                VoiceModel.language,
                func.count(VoiceModel.id).label('count')
            ).filter(
                VoiceModel.organization_id == organization_id if organization_id else True
            ).group_by(VoiceModel.language).all()

            # Get storage statistics
            storage_stats = self.db_session.query(
                func.sum(VoiceModel.model_size_bytes).label('total_size'),
                func.avg(VoiceModel.model_size_bytes).label('avg_size'),
                func.count(VoiceSample.id).label('total_samples'),
                func.sum(VoiceSample.file_size).label('samples_size')
            ).filter(
                VoiceModel.organization_id == organization_id if organization_id else True
            ).first()

            # Get recent activity
            recent_models = query.filter(
                VoiceModel.created_at >= datetime.utcnow() - timedelta(days=30)
            ).count()

            return {
                'total_models': total_models,
                'active_models': active_models,
                'public_models': public_models,
                'trained_models': trained_models,
                'recent_models': recent_models,
                'quality_stats': {
                    'count': quality_stats.count or 0,
                    'average_score': float(quality_stats.avg_score) if quality_stats.avg_score else 0,
                    'min_score': float(quality_stats.min_score) if quality_stats.min_score else 0,
                    'max_score': float(quality_stats.max_score) if quality_stats.max_score else 0
                },
                'language_distribution': {
                    lang: count for lang, count in language_stats
                },
                'storage_stats': {
                    'models_size': storage_stats.total_size or 0,
                    'models_avg_size': float(storage_stats.avg_size) if storage_stats.avg_size else 0,
                    'total_samples': storage_stats.total_samples or 0,
                    'samples_size': storage_stats.samples_size or 0
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting voice statistics: {str(e)}")
            raise AudioProcessingException(f"Failed to get statistics: {str(e)}")

    async def create_voice_version(
        self,
        voice_model_id: int,
        user_id: int,
        model_data: bytes,
        version_tag: str = None,
        change_log: str = None
    ) -> VoiceVersion:
        """Create a new version of a voice model."""
        try:
            # Get voice model
            voice_model = self.db_session.query(VoiceModel).filter(
                VoiceModel.id == voice_model_id,
                VoiceModel.created_by == user_id
            ).first()

            if not voice_model:
                raise ValidationException("Voice model not found or access denied")

            # Check version limits
            version_count = self.db_session.query(func.count(VoiceVersion.id)).filter(
                VoiceVersion.voice_model_id == voice_model_id,
                VoiceVersion.is_active == True
            ).scalar()

            if version_count >= self.max_versions_per_model:
                raise ValidationException(
                    f"Maximum versions per model ({self.max_versions_per_model}) reached"
                )

            # Generate new version number
            latest_version = VoiceVersion.get_active_version(voice_model_id, self.db_session)
            if latest_version:
                major, minor, patch = map(int, latest_version.version_number.split('.'))
                new_version_number = f"{major}.{minor}.{patch + 1}"
            else:
                new_version_number = "1.0.0"

            # Create version record
            version = VoiceVersion(
                voice_model_id=voice_model_id,
                version_number=new_version_number,
                version_tag=version_tag,
                change_log=change_log,
                model_path="",  # Will be set after saving
                model_size_bytes=len(model_data),
                checksum=self._calculate_checksum(model_data),
                created_by=user_id
            )

            self.db_session.add(version)
            self.db_session.commit()

            # Store model file
            model_path = await self.store_voice_model(voice_model, model_data, voice_model.organization_id)
            version.model_path = model_path

            self.db_session.commit()

            return version

        except Exception as e:
            self.logger.error(f"Error creating voice version: {str(e)}")
            raise AudioProcessingException(f"Failed to create version: {str(e)}")

    async def rollback_voice_version(
        self,
        voice_model_id: int,
        user_id: int,
        version_number: str
    ) -> VoiceVersion:
        """Rollback to a previous version of a voice model."""
        try:
            # Get voice model
            voice_model = self.db_session.query(VoiceModel).filter(
                VoiceModel.id == voice_model_id,
                VoiceModel.created_by == user_id
            ).first()

            if not voice_model:
                raise ValidationException("Voice model not found or access denied")

            # Get target version
            target_version = VoiceVersion.get_by_model_and_version(
                voice_model_id, version_number, self.db_session
            )

            if not target_version:
                raise ValidationException(f"Version {version_number} not found")

            # Deactivate current active version
            current_version = VoiceVersion.get_active_version(voice_model_id, self.db_session)
            if current_version:
                current_version.deactivate()

            # Activate target version
            target_version.activate()
            self.db_session.commit()

            return target_version

        except Exception as e:
            self.logger.error(f"Error rolling back voice version: {str(e)}")
            raise AudioProcessingException(f"Failed to rollback version: {str(e)}")

    async def _check_storage_limits(self, organization_id: int):
        """Check if organization has reached storage limits."""
        # Get current storage usage
        storage_query = self.db_session.query(
            func.sum(VoiceModel.model_size_bytes).label('models_size'),
            func.sum(VoiceSample.file_size).label('samples_size')
        ).filter(
            VoiceModel.organization_id == organization_id
        ).outerjoin(VoiceSample, VoiceModel.id == VoiceSample.voice_model_id)

        storage_result = storage_query.first()
        current_usage = (storage_result.models_size or 0) + (storage_result.samples_size or 0)

        if current_usage >= self.max_storage_per_org:
            raise ValidationException(
                f"Storage limit exceeded. Current usage: {current_usage / (1024*1024*1024)".2f"}GB, "
                f"Limit: {self.max_storage_per_org / (1024*1024*1024)".2f"}GB"
            )

        # Check model count limits
        model_count = self.db_session.query(func.count(VoiceModel.id)).filter(
            VoiceModel.organization_id == organization_id,
            VoiceModel.is_active == True
        ).scalar()

        if model_count >= self.max_models_per_org:
            raise ValidationException(
                f"Model limit exceeded. Current count: {model_count}, "
                f"Limit: {self.max_models_per_org}"
            )

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of data."""
        import hashlib
        return hashlib.sha256(data).hexdigest()

    async def cleanup_old_models(
        self,
        organization_id: int = None,
        days_old: int = None
    ) -> Dict[str, int]:
        """Clean up old or failed models."""
        try:
            if days_old is None:
                days_old = self.cleanup_days

            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # Find models to clean up
            query = self.db_session.query(VoiceModel).filter(
                VoiceModel.created_at < cutoff_date,
                or_(
                    VoiceModel.status == VoiceModelStatus.FAILED,
                    and_(
                        VoiceModel.status == VoiceModelStatus.TRAINED,
                        VoiceModel.is_active == False
                    )
                )
            )

            if organization_id:
                query = query.filter(VoiceModel.organization_id == organization_id)

            models_to_cleanup = query.all()
            cleanup_count = 0
            freed_space = 0

            for model in models_to_cleanup:
                # Delete model files
                if model.model_path and os.path.exists(model.model_path):
                    try:
                        file_size = os.path.getsize(model.model_path)
                        os.remove(model.model_path)
                        freed_space += file_size
                    except OSError as e:
                        self.logger.warning(f"Failed to delete model file {model.model_path}: {e}")

                # Delete sample files
                samples = VoiceSample.get_by_voice_model(model.id, self.db_session)
                for sample in samples:
                    if sample.file_path and os.path.exists(sample.file_path):
                        try:
                            file_size = os.path.getsize(sample.file_path)
                            os.remove(sample.file_path)
                            freed_space += file_size
                        except OSError as e:
                            self.logger.warning(f"Failed to delete sample file {sample.file_path}: {e}")

                # Mark model as deleted
                model.deleted_at = datetime.utcnow()
                cleanup_count += 1

            self.db_session.commit()

            return {
                'models_cleaned': cleanup_count,
                'space_freed_bytes': freed_space
            }

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise AudioProcessingException(f"Cleanup failed: {str(e)}")

    async def export_voice_model(
        self,
        voice_model_id: int,
        user_id: int,
        include_samples: bool = False
    ) -> str:
        """Export a voice model for backup or transfer."""
        try:
            # Get voice model
            voice_model = self.db_session.query(VoiceModel).filter(
                VoiceModel.id == voice_model_id,
                VoiceModel.created_by == user_id
            ).first()

            if not voice_model:
                raise ValidationException("Voice model not found or access denied")

            # Create export directory
            export_dir = self.storage_path / "temp" / f"export_{voice_model_id}_{int(datetime.utcnow().timestamp())}"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Export model file
            if voice_model.model_path and os.path.exists(voice_model.model_path):
                shutil.copy2(voice_model.model_path, export_dir / "model.h5")

            # Export metadata
            metadata = {
                'voice_model': voice_model.to_dict(),
                'exported_at': datetime.utcnow().isoformat(),
                'exported_by': user_id,
                'version': '1.0'
            }

            import json
            with open(export_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Export samples if requested
            if include_samples:
                samples = VoiceSample.get_by_voice_model(voice_model_id, self.db_session)
                samples_dir = export_dir / "samples"
                samples_dir.mkdir(exist_ok=True)

                for sample in samples:
                    if sample.file_path and os.path.exists(sample.file_path):
                        shutil.copy2(sample.file_path, samples_dir / sample.filename)

            # Create archive
            archive_path = self.storage_path / "archives" / f"voice_model_{voice_model_id}_export.zip"
            shutil.make_archive(
                str(archive_path).replace('.zip', ''),
                'zip',
                export_dir
            )

            # Clean up temp directory
            shutil.rmtree(export_dir)

            return str(archive_path)

        except Exception as e:
            self.logger.error(f"Error exporting voice model: {str(e)}")
            raise AudioProcessingException(f"Export failed: {str(e)}")

    async def import_voice_model(
        self,
        archive_path: str,
        organization_id: int,
        user_id: int,
        model_name: str = None
    ) -> VoiceModel:
        """Import a voice model from an export archive."""
        try:
            # Check storage limits
            await self._check_storage_limits(organization_id)

            # Extract archive
            import tempfile
            import zipfile
            import json

            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Read metadata
                metadata_path = Path(temp_dir) / "metadata.json"
                if not metadata_path.exists():
                    raise ValidationException("Invalid export archive: missing metadata")

                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Create new voice model
                original_model = metadata['voice_model']
                voice_model = VoiceModel(
                    name=model_name or f"{original_model['name']}_imported",
                    organization_id=organization_id,
                    created_by=user_id,
                    language=original_model['language'],
                    gender=original_model.get('gender'),
                    age_group=original_model.get('age_group'),
                    accent=original_model.get('accent'),
                    model_type=original_model.get('model_type'),
                    model_version=original_model.get('model_version', '1.0.0'),
                    description=original_model.get('description', '') + ' (Imported)',
                    status=VoiceModelStatus.TRAINED,
                    quality_score=original_model.get('quality_score', 5.0),
                    is_public=False,
                    is_active=True
                )

                self.db_session.add(voice_model)
                self.db_session.commit()

                # Import model file
                model_file = Path(temp_dir) / "model.h5"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        model_data = f.read()

                    await self.store_voice_model(voice_model, model_data, organization_id)

                # Import samples if present
                samples_dir = Path(temp_dir) / "samples"
                if samples_dir.exists():
                    for sample_file in samples_dir.glob("*"):
                        if sample_file.is_file():
                            await self._import_sample_file(
                                str(sample_file), voice_model.id, user_id
                            )

                self.db_session.commit()
                return voice_model

        except Exception as e:
            self.logger.error(f"Error importing voice model: {str(e)}")
            raise AudioProcessingException(f"Import failed: {str(e)}")

    async def _import_sample_file(
        self,
        file_path: str,
        voice_model_id: int,
        user_id: int
    ):
        """Import a sample file into the voice model."""
        try:
            file_path = Path(file_path)

            # Calculate checksum
            with open(file_path, 'rb') as f:
                file_data = f.read()
                checksum = self._calculate_checksum(file_data)

            # Check if sample already exists
            existing_sample = VoiceSample.get_by_checksum(checksum, self.db_session)
            if existing_sample:
                return existing_sample

            # Create new sample
            sample = VoiceSample(
                voice_model_id=voice_model_id,
                uploaded_by=user_id,
                filename=file_path.name,
                original_filename=file_path.name,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                mime_type="audio/wav",  # Assume WAV for imported files
                checksum=checksum,
                status=VoiceSampleStatus.PROCESSED  # Mark as processed for imported files
            )

            self.db_session.add(sample)
            return sample

        except Exception as e:
            self.logger.error(f"Error importing sample file: {str(e)}")
            raise


# Global instance
voice_library_manager = None

def get_voice_library_manager(db_session: Session) -> VoiceLibraryManager:
    """Get or create voice library manager instance."""
    global voice_library_manager
    if voice_library_manager is None:
        voice_library_manager = VoiceLibraryManager(db_session)
    return voice_library_manager