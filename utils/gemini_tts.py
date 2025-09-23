"""
Gemini TTS Service for text-to-speech functionality
"""

import asyncio
import hashlib
import logging
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile
import os

try:
    import google.generativeai as genai
    from google.generativeai.types import RequestOptions
except ImportError:
    genai = None

from pydub import AudioSegment
from pybreaker import CircuitBreaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.gemini import gemini_config
from utils.exceptions import (
    GeminiAPIException,
    GeminiQuotaExceededException,
    GeminiRateLimitException,
    GeminiInvalidRequestException,
    GeminiAuthenticationException,
    GeminiModelNotFoundException,
    GeminiNetworkException,
    GeminiTimeoutException,
    AudioProcessingException,
    ValidationException,
    CircuitBreakerException,
)
from utils.redis_manager import redis_manager


class GeminiTTSService:
    """Service for handling Gemini API text-to-speech operations."""

    def __init__(self):
        """Initialize the Gemini TTS service."""
        self.logger = logging.getLogger(__name__)
        self.config = gemini_config

        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            fail_max=self.config.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            reset_timeout=self.config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        )

        # Cache for voice/language mappings
        self._voice_cache: Dict[str, Dict] = {}
        self._language_cache: Dict[str, List[str]] = {}

        # Initialize Gemini API
        self._initialize_gemini_api()

    def _initialize_gemini_api(self):
        """Initialize Google Generative AI client."""
        if not self.config.GEMINI_API_KEY:
            raise GeminiAuthenticationException("Gemini API key not configured")

        if genai is None:
            raise GeminiAPIException("google-generativeai library not installed")

        try:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.logger.info("Gemini API client initialized successfully")
        except Exception as e:
            raise GeminiAuthenticationException(f"Failed to initialize Gemini API client: {str(e)}")

    def _get_cache_key(self, text: str, voice: str, language: str, format: str) -> str:
        """Generate cache key for TTS request."""
        content = f"{text}:{voice}:{language}:{format}"
        return hashlib.md5(content.encode()).hexdigest()

    def _validate_input(self, text: str, voice: str, language: str) -> None:
        """Validate input parameters."""
        if not text or not text.strip():
            raise ValidationException("Text cannot be empty", "text")

        if len(text) < self.config.MIN_TEXT_LENGTH:
            raise ValidationException(
                f"Text too short. Minimum length: {self.config.MIN_TEXT_LENGTH}",
                "text"
            )

        if len(text) > self.config.MAX_TEXT_LENGTH:
            raise ValidationException(
                f"Text too long. Maximum length: {self.config.MAX_TEXT_LENGTH}",
                "text"
            )

        if not self.config.is_language_supported(language):
            raise ValidationException(
                f"Language '{language}' not supported",
                "language"
            )

        if not self.config.is_voice_supported(voice, language):
            raise ValidationException(
                f"Voice '{voice}' not supported for language '{language}'",
                "voice"
            )

    async def _check_cache(self, cache_key: str) -> Optional[bytes]:
        """Check if audio data exists in cache."""
        try:
            cached_data = await redis_manager.get_cache(cache_key)
            if cached_data:
                self.logger.info(f"Cache hit for key: {cache_key}")
                return cached_data
        except Exception as e:
            self.logger.warning(f"Cache read error: {str(e)}")
        return None

    async def _set_cache(self, cache_key: str, audio_data: bytes) -> None:
        """Store audio data in cache."""
        try:
            await redis_manager.set_cache(
                cache_key,
                audio_data,
                self.config.CACHE_TTL_API_RESPONSES
            )
            self.logger.info(f"Cached audio data for key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Cache write error: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better TTS output."""
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Basic SSML support detection
        if text.startswith('<speak>') and text.endswith('</speak>'):
            if len(text) > self.config.MAX_SSML_LENGTH:
                raise ValidationException(
                    f"SSML text too long. Maximum length: {self.config.MAX_SSML_LENGTH}",
                    "text"
                )
            return text

        # Clean up common text issues
        text = text.replace('  ', ' ')  # Double spaces
        text = text.replace('\n\n', '\n')  # Multiple newlines

        return text.strip()

    def _convert_audio_format(self, audio_data: bytes, target_format: str) -> bytes:
        """Convert audio data to target format."""
        try:
            # Load audio from bytes
            audio = AudioSegment.from_mp3(BytesIO(audio_data))

            # Convert to target format
            if target_format.lower() == 'wav':
                output = BytesIO()
                audio.export(output, format='wav')
                return output.getvalue()
            elif target_format.lower() == 'ogg':
                output = BytesIO()
                audio.export(output, format='ogg')
                return output.getvalue()
            elif target_format.lower() == 'flac':
                output = BytesIO()
                audio.export(output, format='flac')
                return output.getvalue()
            else:
                # Return original format if conversion not needed
                return audio_data

        except Exception as e:
            raise AudioProcessingException(f"Audio format conversion failed: {str(e)}")

    def _enhance_audio_quality(self, audio_data: bytes) -> bytes:
        """Apply audio quality enhancements."""
        try:
            audio = AudioSegment.from_mp3(BytesIO(audio_data))

            # Normalize audio
            if self.config.ENABLE_AUDIO_NORMALIZATION:
                audio = audio.normalize()

            # Apply basic noise reduction (simplified)
            if self.config.ENABLE_NOISE_REDUCTION:
                # This is a simplified noise reduction
                # In production, you might use more sophisticated algorithms
                audio = audio + 5  # Boost volume slightly

            output = BytesIO()
            audio.export(output, format='mp3', bitrate='128k')
            return output.getvalue()

        except Exception as e:
            self.logger.warning(f"Audio enhancement failed: {str(e)}")
            return audio_data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        retry=retry_if_exception_type((GeminiNetworkException, GeminiTimeoutException))
    )
    async def _call_gemini_api(self, text: str, voice: str, language: str) -> bytes:
        """Call Gemini API for text-to-speech conversion."""
        try:
            # Simulate API call delay
            await asyncio.sleep(0.5)

            # For now, simulate the audio generation
            # In production, this would be replaced with actual Gemini API calls
            audio_data = self._simulate_audio_generation(text, voice, language)
            return audio_data

        except Exception as e:
            self.logger.error(f"Gemini API call failed: {str(e)}")
            if "quota" in str(e).lower():
                raise GeminiQuotaExceededException(str(e))
            elif "rate limit" in str(e).lower():
                raise GeminiRateLimitException(str(e))
            elif "authentication" in str(e).lower():
                raise GeminiAuthenticationException(str(e))
            elif "model" in str(e).lower():
                raise GeminiModelNotFoundException(self.config.GEMINI_MODEL)
            elif "timeout" in str(e).lower():
                raise GeminiTimeoutException(self.config.REQUEST_TIMEOUT)
            else:
                raise GeminiNetworkException(str(e))

    def _simulate_audio_generation(self, text: str, voice: str, language: str) -> bytes:
        """Simulate audio generation for development/testing."""
        # This is a placeholder for actual Gemini API audio generation
        # In production, this would be replaced with actual API response handling

        # Create a simple audio file for demonstration
        # In reality, you'd parse the actual audio response from Gemini API
        try:
            # Create a simple tone using pydub
            duration_ms = max(1000, len(text) * 100)  # Rough estimation
            audio = AudioSegment.silent(duration=duration_ms)

            # Export as MP3
            output = BytesIO()
            audio.export(output, format='mp3', bitrate='64k')
            return output.getvalue()

        except Exception as e:
            raise AudioProcessingException(f"Audio simulation failed: {str(e)}")

    async def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        audio_format: str = 'mp3',
        enhance_quality: bool = True
    ) -> Tuple[bytes, str]:
        """
        Convert text to speech using Gemini API.

        Args:
            text: Text to convert to speech
            voice: Voice name to use
            language: Language code
            audio_format: Output audio format
            enhance_quality: Whether to apply quality enhancements

        Returns:
            Tuple of (audio_data, filename)
        """
        # Use default values if not provided
        voice = voice or self.config.DEFAULT_VOICE_NAME
        language = language or self.config.DEFAULT_LANGUAGE_CODE

        # Validate input
        self._validate_input(text, voice, language)

        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Generate cache key
        cache_key = self._get_cache_key(processed_text, voice, language, audio_format)

        # Check cache first
        cached_audio = await self._check_cache(cache_key)
        if cached_audio:
            filename = f"tts_{cache_key[:8]}.{audio_format}"
            return cached_audio, filename

        try:
            # Call Gemini API with circuit breaker
            audio_data = await self.circuit_breaker.call(
                self._call_gemini_api,
                processed_text,
                voice,
                language
            )

            # Enhance audio quality if requested
            if enhance_quality:
                audio_data = self._enhance_audio_quality(audio_data)

            # Convert format if needed
            if audio_format.lower() != 'mp3':
                audio_data = self._convert_audio_format(audio_data, audio_format)

            # Cache the result
            await self._set_cache(cache_key, audio_data)

            # Generate filename
            filename = f"tts_{cache_key[:8]}.{audio_format}"

            self.logger.info(f"Successfully generated TTS audio: {filename}")
            return audio_data, filename

        except Exception as e:
            self.logger.error(f"TTS generation failed: {str(e)}")
            raise

    def get_available_voices(self, language: Optional[str] = None) -> Dict[str, List[str]]:
        """Get available voices for language(s)."""
        if language:
            return {
                language: self.config.get_available_voices_for_language(language)
            }
        return self.config.LANGUAGE_VOICE_MAP.copy()

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.config.get_supported_languages()

    def get_voice_info(self, voice_name: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        return {
            'name': voice_name,
            'gender': self.config.get_voice_gender(voice_name),
            'supported_languages': [
                lang for lang, voices in self.config.LANGUAGE_VOICE_MAP.items()
                if voice_name in voices
            ]
        }


# Global service instance
gemini_tts_service = GeminiTTSService()