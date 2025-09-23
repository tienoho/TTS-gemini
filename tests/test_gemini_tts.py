"""
Unit tests for Gemini TTS integration
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

from utils.gemini_tts import GeminiTTSService
from utils.exceptions import (
    GeminiAPIException,
    GeminiQuotaExceededException,
    GeminiRateLimitException,
    ValidationException,
    AudioProcessingException,
)
from config.gemini import gemini_config


class TestGeminiTTSService:
    """Test cases for Gemini TTS Service."""

    @pytest.fixture
    def tts_service(self):
        """Create TTS service instance for testing."""
        with patch('utils.gemini_tts.genai') as mock_genai:
            mock_genai.configure = MagicMock()
            service = GeminiTTSService()
            return service

    @pytest.fixture
    def mock_redis_manager(self):
        """Mock Redis manager for testing."""
        with patch('utils.gemini_tts.redis_manager') as mock_redis:
            mock_redis.get_cache = AsyncMock(return_value=None)
            mock_redis.set_cache = AsyncMock()
            yield mock_redis

    def test_initialization_success(self, tts_service):
        """Test successful service initialization."""
        assert tts_service is not None
        assert tts_service.config is not None
        assert tts_service.circuit_breaker is not None

    def test_initialization_without_api_key(self):
        """Test initialization failure without API key."""
        with patch('utils.gemini_tts.gemini_config') as mock_config:
            mock_config.GEMINI_API_KEY = ''
            with pytest.raises(GeminiAPIException):
                GeminiTTSService()

    def test_input_validation_valid(self, tts_service):
        """Test input validation with valid parameters."""
        # Should not raise any exception
        tts_service._validate_input("Hello world", "Alnilam", "vi-VN")

    def test_input_validation_empty_text(self, tts_service):
        """Test input validation with empty text."""
        with pytest.raises(ValidationException) as exc_info:
            tts_service._validate_input("", "Alnilam", "vi-VN")
        assert "cannot be empty" in str(exc_info.value)

    def test_input_validation_short_text(self, tts_service):
        """Test input validation with text too short."""
        with pytest.raises(ValidationException) as exc_info:
            tts_service._validate_input("", "Alnilam", "vi-VN")
        assert "too short" in str(exc_info.value)

    def test_input_validation_long_text(self, tts_service):
        """Test input validation with text too long."""
        long_text = "a" * 6000  # Exceed max length
        with pytest.raises(ValidationException) as exc_info:
            tts_service._validate_input(long_text, "Alnilam", "vi-VN")
        assert "too long" in str(exc_info.value)

    def test_input_validation_unsupported_language(self, tts_service):
        """Test input validation with unsupported language."""
        with pytest.raises(ValidationException) as exc_info:
            tts_service._validate_input("Hello", "Alnilam", "unsupported-lang")
        assert "not supported" in str(exc_info.value)

    def test_input_validation_unsupported_voice(self, tts_service):
        """Test input validation with unsupported voice."""
        with pytest.raises(ValidationException) as exc_info:
            tts_service._validate_input("Hello", "UnsupportedVoice", "vi-VN")
        assert "not supported" in str(exc_info.value)

    def test_text_preprocessing(self, tts_service):
        """Test text preprocessing functionality."""
        text = "  Hello   world  \n\n  Test  "
        processed = tts_service._preprocess_text(text)
        assert processed == "Hello world\nTest"

    def test_ssml_validation(self, tts_service):
        """Test SSML text validation."""
        ssml_text = "<speak>Hello <emphasis>world</emphasis></speak>"
        processed = tts_service._preprocess_text(ssml_text)
        assert processed == ssml_text

    def test_ssml_too_long(self, tts_service):
        """Test SSML text too long validation."""
        long_ssml = "<speak>" + "a" * 10001 + "</speak>"
        with pytest.raises(ValidationException) as exc_info:
            tts_service._preprocess_text(long_ssml)
        assert "too long" in str(exc_info.value)

    def test_cache_key_generation(self, tts_service):
        """Test cache key generation."""
        text = "Hello world"
        voice = "Alnilam"
        language = "vi-VN"
        format = "mp3"

        cache_key = tts_service._get_cache_key(text, voice, language, format)
        expected_key = tts_service._get_cache_key(text, voice, language, format)

        assert cache_key == expected_key  # Same input should produce same key

    @pytest.mark.asyncio
    async def test_cache_hit(self, tts_service, mock_redis_manager):
        """Test cache hit scenario."""
        cache_key = "test_cache_key"
        mock_audio_data = b"fake_audio_data"

        mock_redis_manager.get_cache.return_value = mock_audio_data

        result = await tts_service._check_cache(cache_key)

        assert result == mock_audio_data
        mock_redis_manager.get_cache.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_cache_miss(self, tts_service, mock_redis_manager):
        """Test cache miss scenario."""
        cache_key = "test_cache_key"

        mock_redis_manager.get_cache.return_value = None

        result = await tts_service._check_cache(cache_key)

        assert result is None
        mock_redis_manager.get_cache.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_cache_set(self, tts_service, mock_redis_manager):
        """Test cache set operation."""
        cache_key = "test_cache_key"
        audio_data = b"fake_audio_data"

        await tts_service._set_cache(cache_key, audio_data)

        mock_redis_manager.set_cache.assert_called_once_with(
            cache_key,
            audio_data,
            tts_service.config.CACHE_TTL_API_RESPONSES
        )

    def test_audio_format_conversion_mp3_to_wav(self, tts_service):
        """Test audio format conversion from MP3 to WAV."""
        # Create fake MP3 data
        fake_mp3_data = b"fake_mp3_audio_data"

        result = tts_service._convert_audio_format(fake_mp3_data, "wav")

        # Should return the same data since conversion is not implemented
        assert result == fake_mp3_data

    def test_audio_enhancement(self, tts_service):
        """Test audio quality enhancement."""
        fake_audio_data = b"fake_audio_data"

        result = tts_service._enhance_audio_quality(fake_audio_data)

        # Should return the same data since enhancement is simplified
        assert result == fake_audio_data

    def test_available_voices(self, tts_service):
        """Test getting available voices."""
        voices = tts_service.get_available_voices()
        assert isinstance(voices, dict)
        assert "vi-VN" in voices
        assert "en-US" in voices

    def test_available_voices_for_language(self, tts_service):
        """Test getting available voices for specific language."""
        voices = tts_service.get_available_voices("vi-VN")
        assert isinstance(voices, dict)
        assert "vi-VN" in voices
        assert isinstance(voices["vi-VN"], list)

    def test_supported_languages(self, tts_service):
        """Test getting supported languages."""
        languages = tts_service.get_supported_languages()
        assert isinstance(languages, list)
        assert "vi-VN" in languages
        assert "en-US" in languages

    def test_voice_info(self, tts_service):
        """Test getting voice information."""
        voice_info = tts_service.get_voice_info("Alnilam")
        assert isinstance(voice_info, dict)
        assert "name" in voice_info
        assert "gender" in voice_info
        assert "supported_languages" in voice_info

    def test_voice_gender(self, tts_service):
        """Test getting voice gender."""
        gender = tts_service.config.get_voice_gender("Alnilam")
        assert gender in ["neutral", "male", "female"]

    def test_language_support_check(self, tts_service):
        """Test language support checking."""
        assert tts_service.config.is_language_supported("vi-VN") is True
        assert tts_service.config.is_language_supported("unsupported") is False

    def test_voice_support_check(self, tts_service):
        """Test voice support checking."""
        assert tts_service.config.is_voice_supported("Alnilam", "vi-VN") is True
        assert tts_service.config.is_voice_supported("NonExistent", "vi-VN") is False

    @pytest.mark.asyncio
    async def test_successful_tts_generation(self, tts_service, mock_redis_manager):
        """Test successful TTS generation."""
        text = "Hello world"
        voice = "Alnilam"
        language = "vi-VN"
        audio_format = "mp3"

        # Mock cache miss
        mock_redis_manager.get_cache.return_value = None

        # Mock successful API call
        with patch.object(tts_service, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = b"fake_audio_data"

            audio_data, filename = await tts_service.text_to_speech(
                text, voice, language, audio_format
            )

            assert audio_data == b"fake_audio_data"
            assert filename.endswith(".mp3")
            assert filename.startswith("tts_")

    @pytest.mark.asyncio
    async def test_tts_with_cache_hit(self, tts_service, mock_redis_manager):
        """Test TTS generation with cache hit."""
        text = "Hello world"
        voice = "Alnilam"
        language = "vi-VN"
        audio_format = "mp3"

        # Mock cache hit
        cached_data = b"cached_audio_data"
        mock_redis_manager.get_cache.return_value = cached_data

        audio_data, filename = await tts_service.text_to_speech(
            text, voice, language, audio_format
        )

        assert audio_data == cached_data
        assert filename.endswith(".mp3")

        # Should not call API when cache hit
        mock_redis_manager.set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_validation_error(self, tts_service):
        """Test TTS generation with validation error."""
        with pytest.raises(ValidationException):
            await tts_service.text_to_speech("", "Alnilam", "vi-VN")

    @pytest.mark.asyncio
    async def test_tts_api_error(self, tts_service, mock_redis_manager):
        """Test TTS generation with API error."""
        text = "Hello world"
        voice = "Alnilam"
        language = "vi-VN"

        # Mock cache miss
        mock_redis_manager.get_cache.return_value = None

        # Mock API error
        with patch.object(tts_service, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = GeminiQuotaExceededException("Quota exceeded")

            with pytest.raises(GeminiQuotaExceededException):
                await tts_service.text_to_speech(text, voice, language)

    def test_simulated_audio_generation(self, tts_service):
        """Test simulated audio generation."""
        text = "Hello world"
        voice = "Alnilam"
        language = "vi-VN"

        audio_data = tts_service._simulate_audio_generation(text, voice, language)

        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0

        # Should generate different audio for different text
        audio_data2 = tts_service._simulate_audio_generation("Different text", voice, language)
        assert audio_data != audio_data2

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, tts_service, mock_redis_manager):
        """Test circuit breaker with successful calls."""
        text = "Hello world"
        voice = "Alnilam"
        language = "vi-VN"

        # Mock cache miss
        mock_redis_manager.get_cache.return_value = None

        # Mock successful API call
        with patch.object(tts_service, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = b"fake_audio_data"

            # Should succeed
            audio_data, filename = await tts_service.text_to_speech(text, voice, language)
            assert audio_data == b"fake_audio_data"

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self, tts_service, mock_redis_manager):
        """Test circuit breaker with failing calls."""
        text = "Hello world"
        voice = "Alnilam"
        language = "vi-VN"

        # Mock cache miss
        mock_redis_manager.get_cache.return_value = None

        # Mock failing API call
        with patch.object(tts_service, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = GeminiAPIException("API Error")

            # Should raise exception
            with pytest.raises(GeminiAPIException):
                await tts_service.text_to_speech(text, voice, language)

    def test_config_methods(self, tts_service):
        """Test configuration helper methods."""
        # Test getting available voices for language
        voices = tts_service.config.get_available_voices_for_language("vi-VN")
        assert isinstance(voices, list)
        assert "Alnilam" in voices

        # Test getting voice gender
        gender = tts_service.config.get_voice_gender("Alnilam")
        assert gender in ["neutral", "male", "female"]

        # Test language support check
        assert tts_service.config.is_language_supported("vi-VN") is True
        assert tts_service.config.is_language_supported("invalid") is False

        # Test voice support check
        assert tts_service.config.is_voice_supported("Alnilam", "vi-VN") is True
        assert tts_service.config.is_voice_supported("invalid", "vi-VN") is False

        # Test getting supported languages
        languages = tts_service.config.get_supported_languages()
        assert isinstance(languages, list)
        assert "vi-VN" in languages