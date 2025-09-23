"""
Audio Enhancement models for TTS system
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AudioEnhancement(Base):
    """Audio Enhancement model for storing enhancement configurations and results."""

    __tablename__ = 'audio_enhancements'

    id = Column(Integer, primary_key=True, index=True)
    audio_file_id = Column(Integer, ForeignKey('audio_files.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Enhancement settings
    enhancement_type = Column(String(50), nullable=False)  # 'noise_reduction', 'normalization', etc.
    settings = Column(JSON, nullable=False)  # JSON object with enhancement parameters
    preset_id = Column(Integer, ForeignKey('enhancement_presets.id'), nullable=True)

    # Quality metrics before enhancement
    original_snr = Column(Float, nullable=True)
    original_thd = Column(Float, nullable=True)
    original_quality_score = Column(Float, nullable=True)

    # Quality metrics after enhancement
    enhanced_snr = Column(Float, nullable=True)
    enhanced_thd = Column(Float, nullable=True)
    enhanced_quality_score = Column(Float, nullable=True)

    # Processing info
    processing_time = Column(Float, nullable=True)  # in seconds
    file_size_before = Column(Integer, nullable=True)  # in bytes
    file_size_after = Column(Integer, nullable=True)  # in bytes
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    audio_file = relationship("AudioFile", back_populates="enhancements")
    user = relationship("User", back_populates="audio_enhancements")
    preset = relationship("EnhancementPreset", back_populates="enhancements")

    def __init__(self, audio_file_id: int, user_id: int, enhancement_type: str,
                 settings: dict, **kwargs):
        """Initialize audio enhancement."""
        self.audio_file_id = audio_file_id
        self.user_id = user_id
        self.enhancement_type = enhancement_type
        self.settings = settings
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of audio enhancement."""
        return f"<AudioEnhancement(id={self.id}, type='{self.enhancement_type}', quality_score={self.enhanced_quality_score})>"

    def to_dict(self) -> dict:
        """Convert enhancement to dictionary."""
        return {
            'id': self.id,
            'audio_file_id': self.audio_file_id,
            'user_id': self.user_id,
            'enhancement_type': self.enhancement_type,
            'settings': self.settings,
            'preset_id': self.preset_id,
            'original_snr': self.original_snr,
            'original_thd': self.original_thd,
            'original_quality_score': self.original_quality_score,
            'enhanced_snr': self.enhanced_snr,
            'enhanced_thd': self.enhanced_thd,
            'enhanced_quality_score': self.enhanced_quality_score,
            'processing_time': self.processing_time,
            'file_size_before': self.file_size_before,
            'file_size_after': self.file_size_after,
            'success': self.success,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AudioEnhancement':
        """Create enhancement instance from dictionary."""
        return cls(
            audio_file_id=data['audio_file_id'],
            user_id=data['user_id'],
            enhancement_type=data['enhancement_type'],
            settings=data['settings'],
            preset_id=data.get('preset_id'),
            original_snr=data.get('original_snr'),
            original_thd=data.get('original_thd'),
            original_quality_score=data.get('original_quality_score'),
            enhanced_snr=data.get('enhanced_snr'),
            enhanced_thd=data.get('enhanced_thd'),
            enhanced_quality_score=data.get('enhanced_quality_score'),
            processing_time=data.get('processing_time'),
            file_size_before=data.get('file_size_before'),
            file_size_after=data.get('file_size_after'),
            success=data.get('success', True),
            error_message=data.get('error_message')
        )

    def get_improvement_metrics(self) -> dict:
        """Get improvement metrics from enhancement."""
        improvements = {}

        if self.original_quality_score is not None and self.enhanced_quality_score is not None:
            improvements['quality_score_improvement'] = self.enhanced_quality_score - self.original_quality_score

        if self.original_snr is not None and self.enhanced_snr is not None:
            improvements['snr_improvement'] = self.enhanced_snr - self.original_snr

        if self.file_size_before is not None and self.file_size_after is not None:
            improvements['size_change_percent'] = ((self.file_size_after - self.file_size_before) / self.file_size_before) * 100

        return improvements

    @staticmethod
    def get_by_audio_file(audio_file_id: int, db_session) -> List['AudioEnhancement']:
        """Get all enhancements for a specific audio file."""
        return db_session.query(AudioEnhancement).filter(
            AudioEnhancement.audio_file_id == audio_file_id
        ).order_by(AudioEnhancement.created_at.desc()).all()

    @staticmethod
    def get_by_user(user_id: int, db_session, limit: int = 50) -> List['AudioEnhancement']:
        """Get enhancements by user."""
        return db_session.query(AudioEnhancement).filter(
            AudioEnhancement.user_id == user_id
        ).order_by(AudioEnhancement.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_enhancement_stats(db_session, user_id: int = None) -> dict:
        """Get enhancement statistics."""
        from sqlalchemy import func

        query = db_session.query(
            func.count(AudioEnhancement.id).label('total_enhancements'),
            func.avg(AudioEnhancement.enhanced_quality_score).label('avg_quality_score'),
            func.avg(AudioEnhancement.processing_time).label('avg_processing_time'),
            func.sum(AudioEnhancement.file_size_after - AudioEnhancement.file_size_before).label('total_size_change')
        )

        if user_id:
            query = query.filter(AudioEnhancement.user_id == user_id)

        stats = query.first()

        return {
            'total_enhancements': stats.total_enhancements or 0,
            'avg_quality_score': float(stats.avg_quality_score) if stats.avg_quality_score else 0,
            'avg_processing_time': float(stats.avg_processing_time) if stats.avg_processing_time else 0,
            'total_size_change': stats.total_size_change or 0,
        }


class EnhancementPreset(Base):
    """Enhancement Preset model for storing reusable enhancement configurations."""

    __tablename__ = 'enhancement_presets'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # None for system presets
    is_system_preset = Column(Boolean, default=False)

    # Preset configuration
    enhancement_type = Column(String(50), nullable=False)
    settings = Column(JSON, nullable=False)

    # Metadata
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)  # Average user rating
    rating_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="enhancement_presets")
    enhancements = relationship("AudioEnhancement", back_populates="preset")

    def __init__(self, name: str, enhancement_type: str, settings: dict,
                 description: str = None, user_id: int = None, is_system_preset: bool = False):
        """Initialize enhancement preset."""
        self.name = name
        self.description = description
        self.user_id = user_id
        self.is_system_preset = is_system_preset
        self.enhancement_type = enhancement_type
        self.settings = settings

    def __repr__(self) -> str:
        """String representation of enhancement preset."""
        return f"<EnhancementPreset(id={self.id}, name='{self.name}', type='{self.enhancement_type}')>"

    def to_dict(self) -> dict:
        """Convert preset to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'user_id': self.user_id,
            'is_system_preset': self.is_system_preset,
            'enhancement_type': self.enhancement_type,
            'settings': self.settings,
            'usage_count': self.usage_count,
            'rating': self.rating,
            'rating_count': self.rating_count,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EnhancementPreset':
        """Create preset instance from dictionary."""
        return cls(
            name=data['name'],
            description=data.get('description'),
            user_id=data.get('user_id'),
            is_system_preset=data.get('is_system_preset', False),
            enhancement_type=data['enhancement_type'],
            settings=data['settings']
        )

    def increment_usage(self):
        """Increment usage count."""
        self.usage_count += 1

    def update_rating(self, new_rating: float):
        """Update preset rating."""
        if self.rating_count is not None and self.rating_count == 0:
            self.rating = new_rating
        elif self.rating_count is not None and self.rating is not None:
            self.rating = (self.rating * self.rating_count + new_rating) / (self.rating_count + 1)
        if self.rating_count is not None:
            self.rating_count += 1

    @staticmethod
    def get_system_presets(db_session) -> List['EnhancementPreset']:
        """Get all system presets."""
        return db_session.query(EnhancementPreset).filter(
            EnhancementPreset.is_system_preset == True
        ).order_by(EnhancementPreset.name).all()

    @staticmethod
    def get_user_presets(user_id: int, db_session) -> List['EnhancementPreset']:
        """Get presets for a specific user."""
        return db_session.query(EnhancementPreset).filter(
            EnhancementPreset.user_id == user_id
        ).order_by(EnhancementPreset.created_at.desc()).all()

    @staticmethod
    def get_popular_presets(db_session, limit: int = 10) -> List['EnhancementPreset']:
        """Get most popular presets."""
        return db_session.query(EnhancementPreset).filter(
            EnhancementPreset.usage_count > 0
        ).order_by(EnhancementPreset.usage_count.desc()).limit(limit).all()

    @staticmethod
    def get_preset_by_type(enhancement_type: str, db_session) -> List['EnhancementPreset']:
        """Get presets by enhancement type."""
        return db_session.query(EnhancementPreset).filter(
            EnhancementPreset.enhancement_type == enhancement_type
        ).order_by(EnhancementPreset.name).all()


class AudioQualityMetric(Base):
    """Audio Quality Metric model for storing detailed quality analysis."""

    __tablename__ = 'audio_quality_metrics'

    id = Column(Integer, primary_key=True, index=True)
    audio_file_id = Column(Integer, ForeignKey('audio_files.id'), nullable=False)
    enhancement_id = Column(Integer, ForeignKey('audio_enhancements.id'), nullable=True)

    # Quality metrics
    snr = Column(Float, nullable=True)  # Signal-to-Noise Ratio
    thd = Column(Float, nullable=True)  # Total Harmonic Distortion
    rms = Column(Float, nullable=True)  # Root Mean Square
    peak = Column(Float, nullable=True)  # Peak amplitude
    dynamic_range = Column(Float, nullable=True)

    # Frequency analysis
    spectral_centroid = Column(Float, nullable=True)
    spectral_rolloff = Column(Float, nullable=True)
    zero_crossing_rate = Column(Float, nullable=True)

    # Quality scores
    overall_quality_score = Column(Float, nullable=True)  # 1-10 scale
    clarity_score = Column(Float, nullable=True)
    noise_score = Column(Float, nullable=True)
    distortion_score = Column(Float, nullable=True)

    # Analysis metadata
    analysis_method = Column(String(50), nullable=False)  # 'algorithmic', 'ai', 'manual'
    confidence = Column(Float, default=1.0)  # Confidence in analysis
    processing_time = Column(Float, nullable=True)  # Analysis time in seconds

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    audio_file = relationship("AudioFile", back_populates="quality_metrics")
    enhancement = relationship("AudioEnhancement", back_populates="quality_metrics")

    def __init__(self, audio_file_id: int, analysis_method: str = 'algorithmic', **kwargs):
        """Initialize audio quality metric."""
        self.audio_file_id = audio_file_id
        self.analysis_method = analysis_method
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of audio quality metric."""
        return f"<AudioQualityMetric(id={self.id}, quality_score={self.overall_quality_score})>"

    def to_dict(self) -> dict:
        """Convert quality metric to dictionary."""
        return {
            'id': self.id,
            'audio_file_id': self.audio_file_id,
            'enhancement_id': self.enhancement_id,
            'snr': self.snr,
            'thd': self.thd,
            'rms': self.rms,
            'peak': self.peak,
            'dynamic_range': self.dynamic_range,
            'spectral_centroid': self.spectral_centroid,
            'spectral_rolloff': self.spectral_rolloff,
            'zero_crossing_rate': self.zero_crossing_rate,
            'overall_quality_score': self.overall_quality_score,
            'clarity_score': self.clarity_score,
            'noise_score': self.noise_score,
            'distortion_score': self.distortion_score,
            'analysis_method': self.analysis_method,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AudioQualityMetric':
        """Create quality metric instance from dictionary."""
        return cls(
            audio_file_id=data['audio_file_id'],
            enhancement_id=data.get('enhancement_id'),
            analysis_method=data.get('analysis_method', 'algorithmic'),
            snr=data.get('snr'),
            thd=data.get('thd'),
            rms=data.get('rms'),
            peak=data.get('peak'),
            dynamic_range=data.get('dynamic_range'),
            spectral_centroid=data.get('spectral_centroid'),
            spectral_rolloff=data.get('spectral_rolloff'),
            zero_crossing_rate=data.get('zero_crossing_rate'),
            overall_quality_score=data.get('overall_quality_score'),
            clarity_score=data.get('clarity_score'),
            noise_score=data.get('noise_score'),
            distortion_score=data.get('distortion_score'),
            confidence=data.get('confidence', 1.0),
            processing_time=data.get('processing_time')
        )

    def get_quality_grade(self) -> str:
        """Get quality grade based on overall score."""
        if self.overall_quality_score is None:
            return 'Unknown'

        score = self.overall_quality_score
        if score >= 9.0:
            return 'Excellent'
        elif score >= 8.0:
            return 'Very Good'
        elif score >= 7.0:
            return 'Good'
        elif score >= 6.0:
            return 'Fair'
        elif score >= 5.0:
            return 'Poor'
        else:
            return 'Very Poor'

    @staticmethod
    def get_by_audio_file(audio_file_id: int, db_session) -> List['AudioQualityMetric']:
        """Get quality metrics for a specific audio file."""
        return db_session.query(AudioQualityMetric).filter(
            AudioQualityMetric.audio_file_id == audio_file_id
        ).order_by(AudioQualityMetric.created_at.desc()).all()

    @staticmethod
    def get_latest_by_audio_file(audio_file_id: int, db_session) -> Optional['AudioQualityMetric']:
        """Get latest quality metric for a specific audio file."""
        return db_session.query(AudioQualityMetric).filter(
            AudioQualityMetric.audio_file_id == audio_file_id
        ).order_by(AudioQualityMetric.created_at.desc()).first()

    @staticmethod
    def get_quality_stats(db_session) -> dict:
        """Get overall quality statistics."""
        from sqlalchemy import func

        stats = db_session.query(
            func.count(AudioQualityMetric.id).label('total_analyses'),
            func.avg(AudioQualityMetric.overall_quality_score).label('avg_quality_score'),
            func.min(AudioQualityMetric.overall_quality_score).label('min_quality_score'),
            func.max(AudioQualityMetric.overall_quality_score).label('max_quality_score'),
            func.avg(AudioQualityMetric.snr).label('avg_snr'),
            func.avg(AudioQualityMetric.thd).label('avg_thd')
        ).first()

        return {
            'total_analyses': stats.total_analyses or 0,
            'avg_quality_score': float(stats.avg_quality_score) if stats.avg_quality_score else 0,
            'min_quality_score': float(stats.min_quality_score) if stats.min_quality_score else 0,
            'max_quality_score': float(stats.max_quality_score) if stats.max_quality_score else 0,
            'avg_snr': float(stats.avg_snr) if stats.avg_snr else 0,
            'avg_thd': float(stats.avg_thd) if stats.avg_thd else 0,
        }