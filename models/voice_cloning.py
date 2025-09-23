"""
Voice Cloning Models for TTS System
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import Column, DateTime, Integer, String, Float, JSON, Index, ForeignKey, Text, Boolean, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class VoiceModelStatus(str, Enum):
    """Voice model status enumeration."""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    DELETED = "deleted"


class VoiceModelType(str, Enum):
    """Voice model type enumeration."""
    STANDARD = "standard"
    PREMIUM = "premium"
    CUSTOM = "custom"


class VoiceSampleStatus(str, Enum):
    """Voice sample status enumeration."""
    UPLOADED = "uploaded"
    PROCESSED = "processed"
    VALIDATED = "validated"
    REJECTED = "rejected"


class VoiceQualityScore(str, Enum):
    """Voice quality score enumeration."""
    EXCELLENT = "excellent"  # 9-10
    GOOD = "good"           # 7-8
    FAIR = "fair"           # 5-6
    POOR = "poor"           # 3-4
    UNUSABLE = "unusable"   # 1-2


class VoiceModel(Base):
    """Voice model for voice cloning system."""

    __tablename__ = 'voice_models'

    # Primary key and basic info
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Foreign keys
    organization_id = Column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), index=True)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Voice characteristics
    language = Column(String(10), nullable=False, default="vi", index=True)
    gender = Column(String(10), nullable=True)  # male, female, other
    age_group = Column(String(20), nullable=True)  # child, teen, adult, senior
    accent = Column(String(50), nullable=True)

    # Model configuration
    model_type = Column(SQLEnum(VoiceModelType), default=VoiceModelType.STANDARD, index=True)
    model_version = Column(String(20), default="1.0.0")
    model_path = Column(String(500), nullable=True)
    model_size_bytes = Column(Integer, default=0)

    # Training configuration
    training_config = Column(JSON, default=dict)
    training_parameters = Column(JSON, default=dict)

    # Status and progress
    status = Column(SQLEnum(VoiceModelStatus), default=VoiceModelStatus.CREATED, index=True)
    training_progress = Column(Float, default=0.0)  # 0.0 to 100.0
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    training_error = Column(Text, nullable=True)

    # Quality metrics
    quality_score = Column(Float, default=0.0)  # 1.0 to 10.0
    quality_feedback = Column(JSON, default=dict)
    quality_assessed_at = Column(DateTime, nullable=True)

    # Usage statistics
    total_training_time_seconds = Column(Float, default=0.0)
    total_samples_used = Column(Integer, default=0)
    total_inferences = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)

    # Access control
    is_public = Column(Boolean, default=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    access_token = Column(String(255), unique=True, index=True, nullable=True)

    # Metadata
    metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Relationships
    samples = relationship("VoiceSample", back_populates="voice_model", cascade="all, delete-orphan")
    quality_metrics = relationship("VoiceQualityMetrics", back_populates="voice_model", cascade="all, delete-orphan")
    versions = relationship("VoiceVersion", back_populates="voice_model", cascade="all, delete-orphan")
    test_results = relationship("VoiceTestResult", back_populates="voice_model", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_voice_models_org_status', 'organization_id', 'status'),
        Index('idx_voice_models_public_active', 'is_public', 'is_active'),
        Index('idx_voice_models_language_gender', 'language', 'gender'),
        Index('idx_voice_models_created_at', 'created_at'),
    )

    def __init__(self, name: str, organization_id: int, created_by: int, **kwargs):
        """Initialize voice model."""
        self.name = name
        self.organization_id = organization_id
        self.created_by = created_by
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of voice model."""
        return f"<VoiceModel(id={self.id}, name='{self.name}', status='{self.status.value}')>"

    def generate_access_token(self) -> str:
        """Generate access token for voice model."""
        token = f"voice-{secrets.token_urlsafe(32)}"
        self.access_token = token
        return token

    def start_training(self):
        """Start training process."""
        self.status = VoiceModelStatus.TRAINING
        self.training_started_at = datetime.utcnow()
        self.training_progress = 0.0

    def complete_training(self):
        """Mark training as completed."""
        self.status = VoiceModelStatus.TRAINED
        self.training_completed_at = datetime.utcnow()
        self.training_progress = 100.0

    def fail_training(self, error_message: str):
        """Mark training as failed."""
        self.status = VoiceModelStatus.FAILED
        self.training_error = error_message
        self.training_progress = 0.0

    def update_progress(self, progress: float):
        """Update training progress."""
        self.training_progress = max(0.0, min(100.0, progress))

    def is_training_complete(self) -> bool:
        """Check if training is complete."""
        return self.status == VoiceModelStatus.TRAINED

    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        return self.status == VoiceModelStatus.TRAINING

    def can_be_used(self) -> bool:
        """Check if voice model can be used for inference."""
        return (self.status == VoiceModelStatus.TRAINED and
                self.is_active and
                self.model_path is not None)

    def get_quality_grade(self) -> VoiceQualityScore:
        """Get quality grade based on score."""
        if self.quality_score >= 9.0:
            return VoiceQualityScore.EXCELLENT
        elif self.quality_score >= 7.0:
            return VoiceQualityScore.GOOD
        elif self.quality_score >= 5.0:
            return VoiceQualityScore.FAIR
        elif self.quality_score >= 3.0:
            return VoiceQualityScore.POOR
        else:
            return VoiceQualityScore.UNUSABLE

    def to_dict(self) -> dict:
        """Convert voice model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'organization_id': self.organization_id,
            'created_by': self.created_by,
            'language': self.language,
            'gender': self.gender,
            'age_group': self.age_group,
            'accent': self.accent,
            'model_type': self.model_type.value if self.model_type else None,
            'model_version': self.model_version,
            'model_size_bytes': self.model_size_bytes,
            'status': self.status.value if self.status else None,
            'training_progress': self.training_progress,
            'training_started_at': self.training_started_at.isoformat() if self.training_started_at else None,
            'training_completed_at': self.training_completed_at.isoformat() if self.training_completed_at else None,
            'quality_score': self.quality_score,
            'quality_grade': self.get_quality_grade().value,
            'total_training_time_seconds': self.total_training_time_seconds,
            'total_samples_used': self.total_samples_used,
            'total_inferences': self.total_inferences,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'is_public': self.is_public,
            'is_active': self.is_active,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def get_by_organization(organization_id: int, db_session, include_deleted: bool = False):
        """Get voice models by organization."""
        query = db_session.query(VoiceModel).filter(
            VoiceModel.organization_id == organization_id
        )
        if not include_deleted:
            query = query.filter(VoiceModel.deleted_at.is_(None))
        return query.order_by(VoiceModel.created_at.desc()).all()

    @staticmethod
    def get_public_models(db_session, language: str = None, limit: int = 50):
        """Get public voice models."""
        query = db_session.query(VoiceModel).filter(
            VoiceModel.is_public == True,
            VoiceModel.is_active == True,
            VoiceModel.status == VoiceModelStatus.TRAINED,
            VoiceModel.deleted_at.is_(None)
        )
        if language:
            query = query.filter(VoiceModel.language == language)
        return query.order_by(VoiceModel.quality_score.desc()).limit(limit).all()

    @staticmethod
    def get_training_models(db_session):
        """Get models currently in training."""
        return db_session.query(VoiceModel).filter(
            VoiceModel.status == VoiceModelStatus.TRAINING
        ).all()


class VoiceSample(Base):
    """Voice sample for training voice models."""

    __tablename__ = 'voice_samples'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    voice_model_id = Column(Integer, ForeignKey('voice_models.id', ondelete='CASCADE'), index=True)
    uploaded_by = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Sample information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # in bytes
    mime_type = Column(String(50), nullable=False)
    checksum = Column(String(64), nullable=False, unique=True)  # SHA256 hash

    # Audio properties
    duration = Column(Float, nullable=True)  # in seconds
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, default=1)
    bit_depth = Column(Integer, nullable=True)

    # Processing status
    status = Column(SQLEnum(VoiceSampleStatus), default=VoiceSampleStatus.UPLOADED, index=True)
    processed_at = Column(DateTime, nullable=True)
    validated_at = Column(DateTime, nullable=True)

    # Quality metrics
    quality_score = Column(Float, default=0.0)
    snr_ratio = Column(Float, nullable=True)  # Signal-to-noise ratio
    transcription = Column(Text, nullable=True)
    phoneme_coverage = Column(JSON, default=dict)

    # Processing results
    features_path = Column(String(500), nullable=True)
    spectrogram_path = Column(String(500), nullable=True)
    mfcc_features = Column(JSON, default=dict)

    # Metadata
    metadata = Column(JSON, default=dict)
    processing_notes = Column(Text, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    voice_model = relationship("VoiceModel", back_populates="samples")

    # Indexes
    __table_args__ = (
        Index('idx_voice_samples_model_status', 'voice_model_id', 'status'),
        Index('idx_voice_samples_checksum', 'checksum'),
        Index('idx_voice_samples_created_at', 'created_at'),
    )

    def __init__(self, voice_model_id: int, uploaded_by: int, filename: str,
                 original_filename: str, file_path: str, file_size: int,
                 mime_type: str, checksum: str, **kwargs):
        """Initialize voice sample."""
        self.voice_model_id = voice_model_id
        self.uploaded_by = uploaded_by
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_size = file_size
        self.mime_type = mime_type
        self.checksum = checksum
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of voice sample."""
        return f"<VoiceSample(id={self.id}, filename='{self.filename}', status='{self.status.value}')>"

    def calculate_checksum(self, file_data: bytes) -> str:
        """Calculate SHA256 checksum of file data."""
        return hashlib.sha256(file_data).hexdigest()

    def verify_integrity(self, file_data: bytes) -> bool:
        """Verify file integrity by comparing checksums."""
        return self.checksum == self.calculate_checksum(file_data)

    def mark_processed(self):
        """Mark sample as processed."""
        self.status = VoiceSampleStatus.PROCESSED
        self.processed_at = datetime.utcnow()

    def mark_validated(self):
        """Mark sample as validated."""
        self.status = VoiceSampleStatus.VALIDATED
        self.validated_at = datetime.utcnow()

    def mark_rejected(self, reason: str):
        """Mark sample as rejected."""
        self.status = VoiceSampleStatus.REJECTED
        self.processing_notes = reason

    def is_valid(self) -> bool:
        """Check if sample is valid for training."""
        return self.status in [VoiceSampleStatus.PROCESSED, VoiceSampleStatus.VALIDATED]

    def get_file_extension(self) -> str:
        """Get file extension from filename."""
        return self.filename.split('.')[-1].lower() if '.' in self.filename else ''

    def is_audio_format(self) -> bool:
        """Check if file is an audio format."""
        audio_extensions = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a']
        return self.get_file_extension() in audio_extensions

    def format_file_size(self) -> str:
        """Format file size for display."""
        size = float(self.file_size)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size".1f"} {unit}"
            size /= 1024.0
        return f"{size".1f"} TB"

    def to_dict(self) -> dict:
        """Convert voice sample to dictionary."""
        return {
            'id': self.id,
            'voice_model_id': self.voice_model_id,
            'uploaded_by': self.uploaded_by,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'formatted_size': self.format_file_size(),
            'mime_type': self.mime_type,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'status': self.status.value if self.status else None,
            'quality_score': self.quality_score,
            'snr_ratio': self.snr_ratio,
            'transcription': self.transcription,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
        }

    @staticmethod
    def get_by_voice_model(voice_model_id: int, db_session):
        """Get samples by voice model."""
        return db_session.query(VoiceSample).filter(
            VoiceSample.voice_model_id == voice_model_id
        ).order_by(VoiceSample.created_at.desc()).all()

    @staticmethod
    def get_valid_samples(voice_model_id: int, db_session):
        """Get valid samples for training."""
        return db_session.query(VoiceSample).filter(
            VoiceSample.voice_model_id == voice_model_id,
            VoiceSample.status.in_([VoiceSampleStatus.PROCESSED, VoiceSampleStatus.VALIDATED])
        ).all()

    @staticmethod
    def get_by_checksum(checksum: str, db_session):
        """Get sample by checksum."""
        return db_session.query(VoiceSample).filter(VoiceSample.checksum == checksum).first()


class VoiceQualityMetrics(Base):
    """Voice quality metrics for trained models."""

    __tablename__ = 'voice_quality_metrics'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    voice_model_id = Column(Integer, ForeignKey('voice_models.id', ondelete='CASCADE'), index=True)

    # Quality dimensions
    clarity_score = Column(Float, default=0.0)  # 1.0 to 10.0
    naturalness_score = Column(Float, default=0.0)  # 1.0 to 10.0
    pronunciation_score = Column(Float, default=0.0)  # 1.0 to 10.0
    consistency_score = Column(Float, default=0.0)  # 1.0 to 10.0
    expressiveness_score = Column(Float, default=0.0)  # 1.0 to 10.0

    # Overall scores
    overall_score = Column(Float, default=0.0)  # 1.0 to 10.0
    weighted_score = Column(Float, default=0.0)  # Weighted average

    # Detailed metrics
    word_error_rate = Column(Float, nullable=True)  # WER percentage
    character_error_rate = Column(Float, nullable=True)  # CER percentage
    mel_cepstral_distortion = Column(Float, nullable=True)  # MCD score
    f0_frame_error = Column(Float, nullable=True)  # F0 error percentage

    # Audio characteristics
    speaking_rate = Column(Float, nullable=True)  # words per minute
    pitch_mean = Column(Float, nullable=True)  # average pitch in Hz
    pitch_std = Column(Float, nullable=True)  # pitch standard deviation
    energy_mean = Column(Float, nullable=True)  # average energy
    energy_std = Column(Float, nullable=True)  # energy standard deviation

    # Test conditions
    test_dataset = Column(String(100), nullable=True)
    test_samples_count = Column(Integer, default=0)
    test_duration_seconds = Column(Float, default=0.0)

    # Assessment metadata
    assessment_method = Column(String(50), default="automated")  # automated, manual, hybrid
    assessor_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    assessment_notes = Column(Text, nullable=True)

    # Audit fields
    assessed_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    voice_model = relationship("VoiceModel", back_populates="quality_metrics")

    # Indexes
    __table_args__ = (
        Index('idx_voice_quality_model_date', 'voice_model_id', 'assessed_at'),
        Index('idx_voice_quality_overall_score', 'overall_score'),
    )

    def __repr__(self) -> str:
        """String representation of quality metrics."""
        return f"<VoiceQualityMetrics(id={self.id}, model_id={self.voice_model_id}, overall={self.overall_score})>"

    def calculate_weighted_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall score."""
        if weights is None:
            weights = {
                'clarity': 0.25,
                'naturalness': 0.25,
                'pronunciation': 0.20,
                'consistency': 0.15,
                'expressiveness': 0.15
            }

        weighted_score = (
            self.clarity_score * weights['clarity'] +
            self.naturalness_score * weights['naturalness'] +
            self.pronunciation_score * weights['pronunciation'] +
            self.consistency_score * weights['consistency'] +
            self.expressiveness_score * weights['expressiveness']
        )

        self.weighted_score = weighted_score
        return weighted_score

    def get_quality_grade(self) -> VoiceQualityScore:
        """Get quality grade based on overall score."""
        if self.overall_score >= 9.0:
            return VoiceQualityScore.EXCELLENT
        elif self.overall_score >= 7.0:
            return VoiceQualityScore.GOOD
        elif self.overall_score >= 5.0:
            return VoiceQualityScore.FAIR
        elif self.overall_score >= 3.0:
            return VoiceQualityScore.POOR
        else:
            return VoiceQualityScore.UNUSABLE

    def to_dict(self) -> dict:
        """Convert quality metrics to dictionary."""
        return {
            'id': self.id,
            'voice_model_id': self.voice_model_id,
            'clarity_score': self.clarity_score,
            'naturalness_score': self.naturalness_score,
            'pronunciation_score': self.pronunciation_score,
            'consistency_score': self.consistency_score,
            'expressiveness_score': self.expressiveness_score,
            'overall_score': self.overall_score,
            'weighted_score': self.weighted_score,
            'word_error_rate': self.word_error_rate,
            'character_error_rate': self.character_error_rate,
            'mel_cepstral_distortion': self.mel_cepstral_distortion,
            'f0_frame_error': self.f0_frame_error,
            'speaking_rate': self.speaking_rate,
            'pitch_mean': self.pitch_mean,
            'pitch_std': self.pitch_std,
            'energy_mean': self.energy_mean,
            'energy_std': self.energy_std,
            'assessment_method': self.assessment_method,
            'assessed_at': self.assessed_at.isoformat() if self.assessed_at else None,
        }

    @staticmethod
    def get_latest_by_model(voice_model_id: int, db_session):
        """Get latest quality metrics for a voice model."""
        return db_session.query(VoiceQualityMetrics).filter(
            VoiceQualityMetrics.voice_model_id == voice_model_id
        ).order_by(VoiceQualityMetrics.assessed_at.desc()).first()

    @staticmethod
    def get_quality_trend(voice_model_id: int, db_session, limit: int = 10):
        """Get quality trend for a voice model."""
        return db_session.query(VoiceQualityMetrics).filter(
            VoiceQualityMetrics.voice_model_id == voice_model_id
        ).order_by(VoiceQualityMetrics.assessed_at.desc()).limit(limit).all()


class VoiceVersion(Base):
    """Voice model versioning system."""

    __tablename__ = 'voice_versions'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    voice_model_id = Column(Integer, ForeignKey('voice_models.id', ondelete='CASCADE'), index=True)

    # Version information
    version_number = Column(String(20), nullable=False, index=True)  # e.g., "1.0.0", "1.1.0"
    version_tag = Column(String(50), nullable=True)  # e.g., "stable", "beta", "improved"
    change_log = Column(Text, nullable=True)

    # Version metadata
    model_path = Column(String(500), nullable=False)
    model_size_bytes = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA256 hash of model file

    # Version status
    is_active = Column(Boolean, default=True, index=True)
    is_deprecated = Column(Boolean, default=False, index=True)
    deprecation_reason = Column(Text, nullable=True)

    # Performance metrics
    quality_score = Column(Float, default=0.0)
    inference_speed_ms = Column(Float, nullable=True)  # Average inference time
    memory_usage_mb = Column(Float, nullable=True)

    # Migration information
    parent_version_id = Column(Integer, ForeignKey('voice_versions.id'), nullable=True)
    migration_notes = Column(Text, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    activated_at = Column(DateTime, nullable=True)
    deprecated_at = Column(DateTime, nullable=True)

    # Relationships
    voice_model = relationship("VoiceModel", back_populates="versions")
    parent_version = relationship("VoiceVersion", remote_side=[id])

    # Indexes
    __table_args__ = (
        Index('idx_voice_versions_model_active', 'voice_model_id', 'is_active'),
        Index('idx_voice_versions_version_number', 'version_number'),
    )

    def __init__(self, voice_model_id: int, version_number: str, model_path: str,
                 model_size_bytes: int, checksum: str, created_by: int, **kwargs):
        """Initialize voice version."""
        self.voice_model_id = voice_model_id
        self.version_number = version_number
        self.model_path = model_path
        self.model_size_bytes = model_size_bytes
        self.checksum = checksum
        self.created_by = created_by
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of voice version."""
        return f"<VoiceVersion(id={self.id}, model_id={self.voice_model_id}, version='{self.version_number}')>"

    def activate(self):
        """Activate this version."""
        self.is_active = True
        self.activated_at = datetime.utcnow()

    def deactivate(self):
        """Deactivate this version."""
        self.is_active = False

    def deprecate(self, reason: str = None):
        """Mark version as deprecated."""
        self.is_deprecated = True
        self.deprecated_at = datetime.utcnow()
        if reason:
            self.deprecation_reason = reason

    def verify_integrity(self, file_data: bytes) -> bool:
        """Verify model file integrity."""
        return self.checksum == hashlib.sha256(file_data).hexdigest()

    def to_dict(self) -> dict:
        """Convert voice version to dictionary."""
        return {
            'id': self.id,
            'voice_model_id': self.voice_model_id,
            'version_number': self.version_number,
            'version_tag': self.version_tag,
            'change_log': self.change_log,
            'model_size_bytes': self.model_size_bytes,
            'quality_score': self.quality_score,
            'inference_speed_ms': self.inference_speed_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'is_active': self.is_active,
            'is_deprecated': self.is_deprecated,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
        }

    @staticmethod
    def get_by_model_and_version(voice_model_id: int, version_number: str, db_session):
        """Get specific version of a voice model."""
        return db_session.query(VoiceVersion).filter(
            VoiceVersion.voice_model_id == voice_model_id,
            VoiceVersion.version_number == version_number
        ).first()

    @staticmethod
    def get_active_version(voice_model_id: int, db_session):
        """Get active version of a voice model."""
        return db_session.query(VoiceVersion).filter(
            VoiceVersion.voice_model_id == voice_model_id,
            VoiceVersion.is_active == True
        ).order_by(VoiceVersion.created_at.desc()).first()

    @staticmethod
    def get_version_history(voice_model_id: int, db_session):
        """Get version history for a voice model."""
        return db_session.query(VoiceVersion).filter(
            VoiceVersion.voice_model_id == voice_model_id
        ).order_by(VoiceVersion.created_at.desc()).all()


class VoiceTestResult(Base):
    """Test results for voice models."""

    __tablename__ = 'voice_test_results'

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    voice_model_id = Column(Integer, ForeignKey('voice_models.id', ondelete='CASCADE'), index=True)

    # Test information
    test_type = Column(String(50), nullable=False, index=True)  # 'quality', 'performance', 'compatibility'
    test_name = Column(String(100), nullable=False)
    test_description = Column(Text, nullable=True)

    # Test parameters
    test_parameters = Column(JSON, default=dict)
    test_input = Column(Text, nullable=True)  # Input text for TTS test
    expected_output = Column(Text, nullable=True)

    # Test results
    test_result = Column(JSON, default=dict)  # Detailed test results
    success = Column(Boolean, default=False, index=True)
    error_message = Column(Text, nullable=True)

    # Performance metrics
    execution_time_ms = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)

    # Quality metrics (for quality tests)
    quality_score = Column(Float, nullable=True)
    clarity_rating = Column(Float, nullable=True)
    naturalness_rating = Column(Float, nullable=True)

    # Test environment
    test_environment = Column(JSON, default=dict)  # OS, Python version, etc.
    tested_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Audit fields
    tested_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    voice_model = relationship("VoiceModel", back_populates="test_results")

    # Indexes
    __table_args__ = (
        Index('idx_voice_test_model_type', 'voice_model_id', 'test_type'),
        Index('idx_voice_test_success', 'success'),
        Index('idx_voice_test_date', 'tested_at'),
    )

    def __repr__(self) -> str:
        """String representation of test result."""
        return f"<VoiceTestResult(id={self.id}, model_id={self.voice_model_id}, type='{self.test_type}', success={self.success})>"

    def to_dict(self) -> dict:
        """Convert test result to dictionary."""
        return {
            'id': self.id,
            'voice_model_id': self.voice_model_id,
            'test_type': self.test_type,
            'test_name': self.test_name,
            'test_description': self.test_description,
            'success': self.success,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'quality_score': self.quality_score,
            'clarity_rating': self.clarity_rating,
            'naturalness_rating': self.naturalness_rating,
            'tested_at': self.tested_at.isoformat() if self.tested_at else None,
        }

    @staticmethod
    def get_by_model_and_type(voice_model_id: int, test_type: str, db_session):
        """Get test results by model and type."""
        return db_session.query(VoiceTestResult).filter(
            VoiceTestResult.voice_model_id == voice_model_id,
            VoiceTestResult.test_type == test_type
        ).order_by(VoiceTestResult.tested_at.desc()).all()

    @staticmethod
    def get_latest_test(voice_model_id: int, test_type: str, db_session):
        """Get latest test result for model and type."""
        return db_session.query(VoiceTestResult).filter(
            VoiceTestResult.voice_model_id == voice_model_id,
            VoiceTestResult.test_type == test_type
        ).order_by(VoiceTestResult.tested_at.desc()).first()