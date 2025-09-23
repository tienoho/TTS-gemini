"""
File management utilities for TTS system
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import aiofiles
from PIL import Image

from config.gemini import gemini_config
from utils.exceptions import FileStorageException


class FileManager:
    """Manages file storage and cleanup operations for TTS system."""

    def __init__(self):
        """Initialize the file manager."""
        self.config = gemini_config
        self.logger = logging.getLogger(__name__)

        # Create necessary directories
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        directories = [
            self.config.TEMP_AUDIO_DIR,
            self.config.AUDIO_STORAGE_DIR,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {directory}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent security issues.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove path separators and dangerous characters
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)

        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)

        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext

        return sanitized.strip()

    def _validate_file_path(self, file_path: str, base_directory: str) -> bool:
        """Validate that file path is within allowed directory.

        Args:
            file_path: File path to validate
            base_directory: Base directory that should contain the file

        Returns:
            True if path is valid
        """
        try:
            # Get absolute paths
            abs_file_path = os.path.abspath(file_path)
            abs_base_directory = os.path.abspath(base_directory)

            # Check if file path is within base directory
            return abs_file_path.startswith(abs_base_directory)
        except Exception:
            return False

    async def save_audio_file(
        self,
        audio_data: bytes,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> str:
        """Save audio data to file with security checks.

        Args:
            audio_data: Audio data as bytes
            filename: Desired filename
            subdirectory: Optional subdirectory within storage

        Returns:
            Full path to saved file

        Raises:
            FileStorageException: If file cannot be saved
        """
        try:
            # Sanitize filename
            safe_filename = self._sanitize_filename(filename)

            # Validate filename
            if not safe_filename:
                raise FileStorageException("Invalid filename")

            # Determine storage directory
            if subdirectory:
                storage_dir = os.path.join(self.config.AUDIO_STORAGE_DIR, subdirectory)
            else:
                storage_dir = self.config.AUDIO_STORAGE_DIR

            # Ensure directory exists
            Path(storage_dir).mkdir(parents=True, exist_ok=True)

            # Generate full path
            file_path = os.path.join(storage_dir, safe_filename)

            # Security validation
            if not self._validate_file_path(file_path, storage_dir):
                raise FileStorageException("Invalid file path - security violation")

            # Save file asynchronously
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_data)

            self.logger.info(f"Audio file saved: {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to save audio file: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg, file_path)

    async def save_temp_audio_file(
        self,
        audio_data: bytes,
        filename: str
    ) -> str:
        """Save audio data to temporary file.

        Args:
            audio_data: Audio data as bytes
            filename: Desired filename

        Returns:
            Full path to temporary file

        Raises:
            FileStorageException: If file cannot be saved
        """
        try:
            # Sanitize filename
            safe_filename = self._sanitize_filename(filename)

            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(
                prefix="tts_",
                suffix=f"_{safe_filename}",
                dir=self.config.TEMP_AUDIO_DIR
            )

            # Write data to temporary file
            async with aiofiles.open(temp_fd, 'wb') as f:
                await f.write(audio_data)

            self.logger.info(f"Temporary audio file created: {temp_path}")
            return temp_path

        except Exception as e:
            error_msg = f"Failed to save temporary audio file: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg)

    async def read_audio_file(self, file_path: str) -> bytes:
        """Read audio data from file.

        Args:
            file_path: Path to audio file

        Returns:
            Audio data as bytes

        Raises:
            FileStorageException: If file cannot be read
        """
        try:
            # Security validation
            if not self._validate_file_path(file_path, self.config.AUDIO_STORAGE_DIR):
                raise FileStorageException("Invalid file path - security violation")

            # Check if file exists
            if not os.path.exists(file_path):
                raise FileStorageException(f"File not found: {file_path}")

            # Read file asynchronously
            async with aiofiles.open(file_path, 'rb') as f:
                audio_data = await f.read()

            self.logger.info(f"Audio file read: {file_path}")
            return audio_data

        except Exception as e:
            error_msg = f"Failed to read audio file: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg, file_path)

    async def delete_file(self, file_path: str) -> bool:
        """Delete a file.

        Args:
            file_path: Path to file to delete

        Returns:
            True if file was deleted successfully

        Raises:
            FileStorageException: If file cannot be deleted
        """
        try:
            # Security validation
            if not self._validate_file_path(file_path, self.config.AUDIO_STORAGE_DIR):
                raise FileStorageException("Invalid file path - security violation")

            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found for deletion: {file_path}")
                return False

            # Delete file
            os.remove(file_path)

            self.logger.info(f"File deleted: {file_path}")
            return True

        except Exception as e:
            error_msg = f"Failed to delete file: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg, file_path)

    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified age.

        Args:
            max_age_hours: Maximum age of files to keep (hours)

        Returns:
            Number of files cleaned up
        """
        try:
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            # Clean up temporary directory
            temp_dir = Path(self.config.TEMP_AUDIO_DIR)
            if temp_dir.exists():
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        # Check file age
                        file_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_modified_time < cutoff_time:
                            file_path.unlink()
                            cleaned_count += 1
                            self.logger.info(f"Cleaned up temp file: {file_path}")

            self.logger.info(f"Cleaned up {cleaned_count} temporary files")
            return cleaned_count

        except Exception as e:
            error_msg = f"Failed to cleanup temporary files: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg)

    async def cleanup_old_files(self, max_age_days: int = 7) -> int:
        """Clean up old files from storage directory.

        Args:
            max_age_days: Maximum age of files to keep (days)

        Returns:
            Number of files cleaned up
        """
        try:
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(days=max_age_days)

            # Clean up storage directory
            storage_dir = Path(self.config.AUDIO_STORAGE_DIR)
            if storage_dir.exists():
                for file_path in storage_dir.rglob("*"):
                    if file_path.is_file():
                        # Check file age
                        file_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_modified_time < cutoff_time:
                            file_path.unlink()
                            cleaned_count += 1
                            self.logger.info(f"Cleaned up old file: {file_path}")

            self.logger.info(f"Cleaned up {cleaned_count} old files")
            return cleaned_count

        except Exception as e:
            error_msg = f"Failed to cleanup old files: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg)

    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information

        Raises:
            FileStorageException: If file information cannot be retrieved
        """
        try:
            # Security validation
            if not self._validate_file_path(file_path, self.config.AUDIO_STORAGE_DIR):
                raise FileStorageException("Invalid file path - security violation")

            # Check if file exists
            if not os.path.exists(file_path):
                raise FileStorageException(f"File not found: {file_path}")

            # Get file stats
            stat = os.stat(file_path)

            # Get file extension
            _, extension = os.path.splitext(file_path)

            file_info = {
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": stat.st_size,
                "extension": extension,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
            }

            return file_info

        except Exception as e:
            error_msg = f"Failed to get file info: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg, file_path)

    async def list_files(
        self,
        subdirectory: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List files in storage directory.

        Args:
            subdirectory: Optional subdirectory to list
            pattern: Optional pattern to filter files

        Returns:
            List of file information dictionaries
        """
        try:
            # Determine directory to list
            if subdirectory:
                search_dir = os.path.join(self.config.AUDIO_STORAGE_DIR, subdirectory)
            else:
                search_dir = self.config.AUDIO_STORAGE_DIR

            # Security validation
            if not self._validate_file_path(search_dir, self.config.AUDIO_STORAGE_DIR):
                raise FileStorageException("Invalid directory path - security violation")

            files_info = []

            # List files
            search_path = Path(search_dir)
            if search_path.exists():
                for file_path in search_path.rglob("*"):
                    if file_path.is_file():
                        # Apply pattern filter if provided
                        if pattern and pattern not in file_path.name:
                            continue

                        try:
                            file_info = await self.get_file_info(str(file_path))
                            files_info.append(file_info)
                        except Exception as e:
                            self.logger.warning(f"Failed to get info for {file_path}: {str(e)}")

            return files_info

        except Exception as e:
            error_msg = f"Failed to list files: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg)

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            total_files = 0
            total_size = 0
            storage_dir = Path(self.config.AUDIO_STORAGE_DIR)

            if storage_dir.exists():
                for file_path in storage_dir.rglob("*"):
                    if file_path.is_file():
                        total_files += 1
                        total_size += file_path.stat().st_size

            # Get temp directory stats
            temp_files = 0
            temp_size = 0
            temp_dir = Path(self.config.TEMP_AUDIO_DIR)

            if temp_dir.exists():
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        temp_files += 1
                        temp_size += file_path.stat().st_size

            return {
                "storage_directory": self.config.AUDIO_STORAGE_DIR,
                "temp_directory": self.config.TEMP_AUDIO_DIR,
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "temp_files": temp_files,
                "temp_size_bytes": temp_size,
                "temp_size_mb": temp_size / (1024 * 1024),
            }

        except Exception as e:
            error_msg = f"Failed to get storage stats: {str(e)}"
            self.logger.error(error_msg)
            raise FileStorageException(error_msg)


# Global file manager instance
file_manager = FileManager()