"""
Audio processing utilities for TTS API
"""

import asyncio
import base64
import hashlib
import mimetypes
import os
import struct
import tempfile
from typing import Dict, Optional, Tuple

import aiofiles

from .gemini_tts import gemini_tts_service
from .exceptions import AudioProcessingException, ValidationException


class AudioProcessor:
    """Handles audio processing using Gemini TTS Service."""

    def __init__(self):
        """Initialize the audio processor."""
        self.tts_service = gemini_tts_service

    def parse_audio_mime_type(self, mime_type: str) -> Dict[str, Optional[int]]:
        """Parses bits per sample and rate from an audio MIME type string.

        Args:
            mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

        Returns:
            A dictionary with "bits_per_sample" and "rate" keys.
        """
        bits_per_sample = 16
        rate = 24000

        # Extract rate from parameters
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass  # Keep default rate
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass  # Keep default bits_per_sample

        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Converts audio data to WAV format.

        Args:
            audio_data: Raw audio data as bytes
            mime_type: MIME type of the audio data

        Returns:
            WAV formatted audio data as bytes
        """
        parameters = self.parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"] or 16
        sample_rate = parameters["rate"] or 24000
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size

        # WAV header format
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",          # ChunkID
            chunk_size,       # ChunkSize (total file size - 8 bytes)
            b"WAVE",          # Format
            b"fmt ",          # Subchunk1ID
            16,               # Subchunk1Size (16 for PCM)
            1,                # AudioFormat (1 for PCM)
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",          # Subchunk2ID
            data_size         # Subchunk2Size (size of audio data)
        )
        return header + audio_data

    async def generate_audio(
        self,
        text: str,
        voice_name: str = "Alnilam",
        language: str = "vi-VN",
        output_format: str = "mp3",
        enhance_quality: bool = True
    ) -> Tuple[bytes, str]:
        """Generate audio from text using Gemini TTS Service.

        Args:
            text: Text to convert to speech
            voice_name: Voice to use for synthesis
            language: Language code for synthesis
            output_format: Desired output format (mp3, wav, etc.)
            enhance_quality: Whether to apply quality enhancements

        Returns:
            Tuple of (audio_data, filename)

        Raises:
            AudioProcessingException: If audio generation fails
        """
        try:
            # Use Gemini TTS service to generate audio
            audio_data, filename = await self.tts_service.text_to_speech(
                text=text,
                voice=voice_name,
                language=language,
                audio_format=output_format,
                enhance_quality=enhance_quality
            )

            return audio_data, filename

        except ValidationException as e:
            raise AudioProcessingException(f"Validation error: {str(e)}")
        except Exception as e:
            raise AudioProcessingException(f"Audio generation failed: {str(e)}")

    def get_available_voices(self, language: str = None):
        """Get available voices for language."""
        return self.tts_service.get_available_voices(language)

    def get_supported_languages(self):
        """Get supported languages."""
        return self.tts_service.get_supported_languages()

    def get_voice_info(self, voice_name: str):
        """Get voice information."""
        return self.tts_service.get_voice_info(voice_name)

    def calculate_audio_hash(self, audio_data: bytes) -> str:
        """Calculate SHA256 hash of audio data.

        Args:
            audio_data: Audio data as bytes

        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(audio_data).hexdigest()

    def get_audio_duration(self, audio_data: bytes, mime_type: str) -> float:
        """Estimate audio duration from data.

        Args:
            audio_data: Audio data as bytes
            mime_type: MIME type of audio

        Returns:
            Estimated duration in seconds
        """
        parameters = self.parse_audio_mime_type(mime_type)
        sample_rate = parameters["rate"] or 24000
        bits_per_sample = parameters["bits_per_sample"] or 16
        bytes_per_sample = bits_per_sample // 8
        num_channels = 1

        # Calculate duration: (data_size / bytes_per_sample / channels) / sample_rate
        data_size = len(audio_data)
        duration = (data_size / bytes_per_sample / num_channels) / sample_rate

        return duration

    async def save_audio_file(
        self,
        audio_data: bytes,
        filename: str,
        upload_folder: str
    ) -> str:
        """Save audio data to file with path traversal protection.

        Args:
            audio_data: Audio data as bytes
            filename: Desired filename
            upload_folder: Directory to save file

        Returns:
            Full path to saved file

        Raises:
            ValueError: If filename contains invalid characters
        """
        # Validate filename to prevent path traversal
        if not filename or '..' in filename or '/' in filename or '\\' in filename:
            raise ValueError("Invalid filename")

        # Sanitize filename
        safe_filename = self.sanitize_filename(filename)

        # Ensure upload folder exists
        os.makedirs(upload_folder, exist_ok=True)

        # Generate full path with absolute path checking
        file_path = os.path.join(upload_folder, safe_filename)
        normalized_path = os.path.normpath(file_path)

        # Security check: ensure path is within upload folder
        upload_abs = os.path.abspath(upload_folder)
        normalized_abs = os.path.abspath(normalized_path)

        if not normalized_abs.startswith(upload_abs):
            raise ValueError("Invalid file path - path traversal detected")

        # Save file asynchronously
        async with aiofiles.open(normalized_path, 'wb') as f:
            await f.write(audio_data)

        return normalized_path

    def sanitize_filename(self, filename: str) -> str:
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

    def validate_audio_format(self, mime_type: str, supported_formats: list) -> bool:
        """Validate if audio format is supported.

        Args:
            mime_type: MIME type to validate
            supported_formats: List of supported formats

        Returns:
            True if format is supported
        """
        for fmt in supported_formats:
            if fmt in mime_type.lower():
                return True
        return False

    def get_file_extension(self, mime_type: str) -> str:
        """Get file extension from MIME type.

        Args:
            mime_type: MIME type

        Returns:
            File extension (with dot)
        """
        extension = mimetypes.guess_extension(mime_type)
        return extension if extension else ".wav"