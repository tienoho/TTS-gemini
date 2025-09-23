"""
AudioFile model for Flask TTS API
"""

import hashlib
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AudioFile(Base):
    """AudioFile model for storing generated audio files."""

    __tablename__ = 'audio_files'

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey('audio_requests.id'), nullable=False)
    file_path = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    mime_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)  # in bytes
    checksum = Column(String(64), nullable=False)  # SHA256 hash
    duration = Column(Float, nullable=True)  # in seconds
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    request = relationship("AudioRequest", back_populates="audio_files")

    def __init__(self, request_id: int, file_path: str, filename: str,
                 mime_type: str, file_size: int, checksum: str, **kwargs):
        """Initialize audio file."""
        self.request_id = request_id
        self.file_path = file_path
        self.filename = filename
        self.mime_type = mime_type
        self.file_size = file_size
        self.checksum = checksum
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of audio file."""
        return f"<AudioFile(id={self.id}, filename='{self.filename}', size={self.file_size})>"

    def calculate_checksum(self, file_data: bytes) -> str:
        """Calculate SHA256 checksum of file data."""
        return hashlib.sha256(file_data).hexdigest()

    def verify_integrity(self, file_data: bytes) -> bool:
        """Verify file integrity by comparing checksums."""
        return str(self.checksum) == self.calculate_checksum(file_data)

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
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def to_dict(self) -> dict:
        """Convert audio file to dictionary."""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'file_path': self.file_path,
            'filename': self.filename,
            'mime_type': self.mime_type,
            'file_size': self.file_size,
            'formatted_size': self.format_file_size(),
            'checksum': self.checksum,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AudioFile':
        """Create audio file instance from dictionary."""
        return cls(
            request_id=data['request_id'],
            file_path=data['file_path'],
            filename=data['filename'],
            mime_type=data['mime_type'],
            file_size=data['file_size'],
            checksum=data['checksum'],
            duration=data.get('duration'),
            sample_rate=data.get('sample_rate'),
            channels=data.get('channels', 1)
        )

    @staticmethod
    def get_by_request(request_id: int, db_session, user_id: int = None):
        """Get audio files for a specific request."""
        if not request_id or request_id <= 0:
            raise ValueError("Invalid request_id provided")

        from .audio_request import AudioRequest

        query = db_session.query(AudioFile).filter(
            AudioFile.request_id == request_id
        )

        # If user_id is provided, add authorization check
        if user_id:
            query = query.join(AudioRequest).filter(
                AudioRequest.user_id == user_id
            )

        return query.order_by(AudioFile.created_at.desc()).all()

    @staticmethod
    def get_by_checksum(checksum: str, db_session) -> Optional['AudioFile']:
        """Get audio file by checksum."""
        return db_session.query(AudioFile).filter(AudioFile.checksum == checksum).first()

    @staticmethod
    def get_total_size(db_session) -> int:
        """Get total size of all audio files."""
        from sqlalchemy import func
        result = db_session.query(func.sum(AudioFile.file_size)).first()
        return result[0] or 0

    @staticmethod
    def get_file_count(db_session) -> int:
        """Get total count of audio files."""
        from sqlalchemy import func
        result = db_session.query(func.count(AudioFile.id)).first()
        return result[0] or 0

    @staticmethod
    def cleanup_orphaned_files(db_session, max_age_days: int = 30):
        """Clean up orphaned audio files (files without valid requests)."""
        from datetime import timedelta
        from .audio_request import AudioRequest

        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

        # Get files for requests that don't exist or are very old
        orphaned_files = db_session.query(AudioFile).filter(
            ~AudioFile.request_id.in_(
                db_session.query(AudioRequest.id)
            ) |
            (AudioFile.created_at < cutoff_date)
        ).all()

        # Delete orphaned files
        for file in orphaned_files:
            # In a real application, you'd also delete the physical file
            db_session.delete(file)

        return len(orphaned_files)

    @staticmethod
    def get_storage_stats(db_session) -> dict:
        """Get storage statistics."""
        from sqlalchemy import func

        stats = db_session.query(
            func.count(AudioFile.id).label('total_files'),
            func.sum(AudioFile.file_size).label('total_size'),
            func.avg(AudioFile.file_size).label('avg_file_size'),
            func.min(AudioFile.file_size).label('min_file_size'),
            func.max(AudioFile.file_size).label('max_file_size')
        ).first()

        return {
            'total_files': stats.total_files or 0,
            'total_size': stats.total_size or 0,
            'avg_file_size': float(stats.avg_file_size) if stats.avg_file_size else 0,
            'min_file_size': stats.min_file_size or 0,
            'max_file_size': stats.max_file_size or 0,
        }

    @staticmethod
    def get_format_distribution(db_session) -> dict:
        """Get distribution of audio formats."""
        from sqlalchemy import func

        format_stats = db_session.query(
            AudioFile.mime_type,
            func.count(AudioFile.id).label('count'),
            func.sum(AudioFile.file_size).label('total_size')
        ).group_by(AudioFile.mime_type).all()

        return {
            row.mime_type: {
                'count': row.count,
                'total_size': row.total_size or 0
            }
            for row in format_stats
        }