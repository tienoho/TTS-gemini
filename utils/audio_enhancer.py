"""
Audio Enhancement Service for TTS system
Provides comprehensive audio quality enhancement capabilities
"""

import asyncio
import io
import logging
import numpy as np
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import wave
import struct
import math

try:
    import librosa
    import librosa.effects
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

from .exceptions import AudioProcessingException, ValidationException
from models.audio_enhancement import AudioEnhancement, EnhancementPreset, AudioQualityMetric


class AudioEnhancer:
    """Comprehensive audio enhancement service with multiple algorithms."""

    def __init__(self):
        """Initialize the audio enhancer."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit

        if not HAS_AUDIO_LIBS:
            self.logger.warning("Audio enhancement libraries not available. Install librosa and soundfile for full functionality.")

    def validate_audio_file(self, audio_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate audio file before processing.

        Args:
            audio_data: Audio file data as bytes
            filename: Original filename

        Returns:
            Dictionary with validation results

        Raises:
            ValidationException: If file is invalid
        """
        if len(audio_data) > self.max_file_size:
            raise ValidationException(f"File size exceeds maximum limit of {self.max_file_size} bytes")

        if len(audio_data) == 0:
            raise ValidationException("Audio file is empty")

        # Check file extension
        file_ext = Path(filename).suffix.lower().lstrip('.')
        if file_ext not in self.supported_formats:
            raise ValidationException(f"Unsupported format: {file_ext}")

        return {
            'valid': True,
            'size': len(audio_data),
            'format': file_ext,
            'filename': filename
        }

    def load_audio_from_bytes(self, audio_data: bytes, sample_rate: int = None) -> Tuple[np.ndarray, int]:
        """Load audio data from bytes into numpy array.

        Args:
            audio_data: Audio data as bytes
            sample_rate: Target sample rate (optional)

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            AudioProcessingException: If loading fails
        """
        if not HAS_AUDIO_LIBS:
            raise AudioProcessingException("Audio processing libraries not available")

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            # Load audio using librosa
            audio, sr = librosa.load(temp_path, sr=sample_rate, mono=True)

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

            return audio, sr

        except Exception as e:
            raise AudioProcessingException(f"Failed to load audio: {str(e)}")

    def save_audio_to_bytes(self, audio: np.ndarray, sample_rate: int, output_format: str = 'wav') -> bytes:
        """Save audio array to bytes in specified format.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate
            output_format: Output format (wav, mp3, flac, etc.)

        Returns:
            Audio data as bytes

        Raises:
            AudioProcessingException: If saving fails
        """
        if not HAS_AUDIO_LIBS:
            raise AudioProcessingException("Audio processing libraries not available")

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as temp_file:
                temp_path = temp_file.name

            # Save audio using soundfile
            sf.write(temp_path, audio, sample_rate, format=output_format)

            # Read back as bytes
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

            return audio_bytes

        except Exception as e:
            raise AudioProcessingException(f"Failed to save audio: {str(e)}")

    def apply_noise_reduction(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Apply noise reduction using spectral gating.

        Args:
            audio: Input audio array
            noise_factor: Noise reduction factor (0.0 to 1.0)

        Returns:
            Noise-reduced audio array
        """
        if not HAS_AUDIO_LIBS:
            return audio

        try:
            # Use librosa for noise reduction
            # Simple noise reduction using spectral subtraction
            stft = librosa.stft(audio)
            magnitude, phase = librosa.magphase(stft)

            # Estimate noise floor
            noise_floor = np.mean(magnitude[:, :10], axis=1, keepdims=True)

            # Apply spectral subtraction
            magnitude_clean = magnitude - noise_factor * noise_floor
            magnitude_clean = np.maximum(magnitude_clean, 0)

            # Reconstruct audio
            stft_clean = magnitude_clean * phase
            audio_clean = librosa.istft(stft_clean)

            return audio_clean

        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {str(e)}")
            return audio

    def apply_normalization(self, audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
        """Apply volume normalization to target level.

        Args:
            audio: Input audio array
            target_level: Target RMS level in dB

        Returns:
            Normalized audio array
        """
        try:
            # Calculate current RMS
            current_rms = np.sqrt(np.mean(audio**2))

            if current_rms == 0:
                return audio

            # Convert target level to linear scale
            target_linear = 10**(target_level / 20.0)

            # Calculate normalization factor
            normalization_factor = target_linear / current_rms

            # Apply normalization with clipping protection
            normalized_audio = audio * normalization_factor
            normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

            return normalized_audio

        except Exception as e:
            self.logger.warning(f"Normalization failed: {str(e)}")
            return audio

    def apply_compression(self, audio: np.ndarray, threshold: float = -20.0,
                         ratio: float = 4.0, attack_time: float = 0.001,
                         release_time: float = 0.1) -> np.ndarray:
        """Apply dynamic range compression.

        Args:
            audio: Input audio array
            threshold: Compression threshold in dB
            ratio: Compression ratio
            attack_time: Attack time in seconds
            release_time: Release time in seconds

        Returns:
            Compressed audio array
        """
        try:
            # Convert threshold to linear scale
            threshold_linear = 10**(threshold / 20.0)

            # Simple compressor implementation
            compressed = np.copy(audio)

            # Apply compression
            over_threshold = np.abs(audio) > threshold_linear
            compressed[over_threshold] = np.sign(audio[over_threshold]) * (
                threshold_linear + (np.abs(audio[over_threshold]) - threshold_linear) / ratio
            )

            return compressed

        except Exception as e:
            self.logger.warning(f"Compression failed: {str(e)}")
            return audio

    def apply_equalization(self, audio: np.ndarray, sample_rate: int,
                          low_gain: float = 0.0, mid_gain: float = 0.0,
                          high_gain: float = 0.0) -> np.ndarray:
        """Apply simple 3-band equalization.

        Args:
            audio: Input audio array
            sample_rate: Sample rate
            low_gain: Low frequency gain in dB
            mid_gain: Mid frequency gain in dB
            high_gain: High frequency gain in dB

        Returns:
            Equalized audio array
        """
        if not HAS_AUDIO_LIBS:
            return audio

        try:
            # Define frequency bands
            nyquist = sample_rate / 2
            low_cutoff = 250 / nyquist
            high_cutoff = 8000 / nyquist

            # Create simple EQ filters
            stft = librosa.stft(audio)
            magnitude, phase = librosa.magphase(stft)

            # Apply frequency-dependent gain
            freqs = librosa.fft_frequencies(sr=sample_rate)
            low_mask = freqs < 250
            mid_mask = (freqs >= 250) & (freqs < 8000)
            high_mask = freqs >= 8000

            # Apply gains
            magnitude[low_mask] *= 10**(low_gain / 20.0)
            magnitude[mid_mask] *= 10**(mid_gain / 20.0)
            magnitude[high_mask] *= 10**(high_gain / 20.0)

            # Reconstruct audio
            stft_eq = magnitude * phase
            audio_eq = librosa.istft(stft_eq)

            return audio_eq

        except Exception as e:
            self.logger.warning(f"Equalization failed: {str(e)}")
            return audio

    def apply_reverb(self, audio: np.ndarray, room_size: float = 0.5,
                    damping: float = 0.5, wet_level: float = 0.3) -> np.ndarray:
        """Apply reverb effect.

        Args:
            audio: Input audio array
            room_size: Room size (0.0 to 1.0)
            damping: High frequency damping (0.0 to 1.0)
            wet_level: Wet/dry mix (0.0 to 1.0)

        Returns:
            Audio with reverb effect
        """
        if not HAS_AUDIO_LIBS:
            return audio

        try:
            # Simple reverb using librosa
            # Create impulse response for reverb
            delay_samples = int(0.1 * 44100)  # 100ms delay
            decay = 0.5

            # Create reverb tail
            reverb_length = len(audio) + delay_samples
            reverb = np.zeros(reverb_length)

            for i in range(1, 5):  # Multiple reflections
                delay = delay_samples * i
                if delay < len(reverb):
                    reverb[delay:delay+len(audio)] += audio * (decay ** i)

            # Mix dry and wet signals
            output = audio * (1 - wet_level) + reverb[:len(audio)] * wet_level

            return output

        except Exception as e:
            self.logger.warning(f"Reverb failed: {str(e)}")
            return audio

    def apply_enhancement_preset(self, audio: np.ndarray, sample_rate: int,
                                preset: EnhancementPreset) -> np.ndarray:
        """Apply enhancement using a preset configuration.

        Args:
            audio: Input audio array
            sample_rate: Sample rate
            preset: Enhancement preset to apply

        Returns:
            Enhanced audio array
        """
        settings = preset.settings

        # Apply enhancements in sequence
        enhanced = audio

        # Noise reduction
        if settings.get('noise_reduction', False):
            noise_factor = settings.get('noise_factor', 0.1)
            enhanced = self.apply_noise_reduction(enhanced, noise_factor)

        # Normalization
        if settings.get('normalization', False):
            target_level = settings.get('target_level', -3.0)
            enhanced = self.apply_normalization(enhanced, target_level)

        # Compression
        if settings.get('compression', False):
            threshold = settings.get('threshold', -20.0)
            ratio = settings.get('ratio', 4.0)
            enhanced = self.apply_compression(enhanced, threshold, ratio)

        # Equalization
        if settings.get('equalization', False):
            low_gain = settings.get('low_gain', 0.0)
            mid_gain = settings.get('mid_gain', 0.0)
            high_gain = settings.get('high_gain', 0.0)
            enhanced = self.apply_equalization(enhanced, sample_rate, low_gain, mid_gain, high_gain)

        # Reverb
        if settings.get('reverb', False):
            room_size = settings.get('room_size', 0.5)
            damping = settings.get('damping', 0.5)
            wet_level = settings.get('wet_level', 0.3)
            enhanced = self.apply_reverb(enhanced, room_size, damping, wet_level)

        return enhanced

    def enhance_audio(self, audio_data: bytes, enhancement_type: str,
                     settings: Dict[str, Any], sample_rate: int = None) -> Tuple[bytes, Dict[str, Any]]:
        """Apply audio enhancement with comprehensive processing.

        Args:
            audio_data: Input audio data as bytes
            enhancement_type: Type of enhancement to apply
            settings: Enhancement settings
            sample_rate: Target sample rate (optional)

        Returns:
            Tuple of (enhanced_audio_bytes, processing_info)

        Raises:
            AudioProcessingException: If enhancement fails
        """
        start_time = time.time()

        try:
            # Load audio
            audio, sr = self.load_audio_from_bytes(audio_data, sample_rate)
            original_size = len(audio_data)

            # Store original for comparison
            audio_original = audio.copy()

            # Apply enhancement based on type
            if enhancement_type == 'noise_reduction':
                noise_factor = settings.get('noise_factor', 0.1)
                audio_enhanced = self.apply_noise_reduction(audio, noise_factor)

            elif enhancement_type == 'normalization':
                target_level = settings.get('target_level', -3.0)
                audio_enhanced = self.apply_normalization(audio, target_level)

            elif enhancement_type == 'compression':
                threshold = settings.get('threshold', -20.0)
                ratio = settings.get('ratio', 4.0)
                audio_enhanced = self.apply_compression(audio, threshold, ratio)

            elif enhancement_type == 'equalization':
                low_gain = settings.get('low_gain', 0.0)
                mid_gain = settings.get('mid_gain', 0.0)
                high_gain = settings.get('high_gain', 0.0)
                audio_enhanced = self.apply_equalization(audio, sr, low_gain, mid_gain, high_gain)

            elif enhancement_type == 'reverb':
                room_size = settings.get('room_size', 0.5)
                damping = settings.get('damping', 0.5)
                wet_level = settings.get('wet_level', 0.3)
                audio_enhanced = self.apply_reverb(audio, room_size, damping, wet_level)

            elif enhancement_type == 'full_enhancement':
                # Apply all enhancements
                audio_enhanced = self.apply_noise_reduction(audio, settings.get('noise_factor', 0.1))
                audio_enhanced = self.apply_normalization(audio_enhanced, settings.get('target_level', -3.0))
                audio_enhanced = self.apply_compression(audio_enhanced, settings.get('threshold', -20.0),
                                                      settings.get('ratio', 4.0))
                audio_enhanced = self.apply_equalization(audio_enhanced, sr, settings.get('low_gain', 0.0),
                                                       settings.get('mid_gain', 0.0), settings.get('high_gain', 0.0))

            else:
                raise ValidationException(f"Unknown enhancement type: {enhancement_type}")

            # Save enhanced audio
            enhanced_bytes = self.save_audio_to_bytes(audio_enhanced, sr)

            # Calculate processing metrics
            processing_time = time.time() - start_time
            compression_ratio = original_size / len(enhanced_bytes) if len(enhanced_bytes) > 0 else 1.0

            processing_info = {
                'original_size': original_size,
                'enhanced_size': len(enhanced_bytes),
                'compression_ratio': compression_ratio,
                'processing_time': processing_time,
                'sample_rate': sr,
                'enhancement_type': enhancement_type,
                'settings': settings
            }

            return enhanced_bytes, processing_info

        except Exception as e:
            raise AudioProcessingException(f"Audio enhancement failed: {str(e)}")

    async def enhance_audio_async(self, audio_data: bytes, enhancement_type: str,
                                 settings: Dict[str, Any], sample_rate: int = None) -> Tuple[bytes, Dict[str, Any]]:
        """Async version of enhance_audio.

        Args:
            audio_data: Input audio data as bytes
            enhancement_type: Type of enhancement to apply
            settings: Enhancement settings
            sample_rate: Target sample rate (optional)

        Returns:
            Tuple of (enhanced_audio_bytes, processing_info)
        """
        # Run enhancement in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.enhance_audio, audio_data, enhancement_type, settings, sample_rate
        )

    def create_enhancement_preset(self, name: str, enhancement_type: str,
                                 settings: Dict[str, Any], description: str = None,
                                 user_id: int = None) -> EnhancementPreset:
        """Create a new enhancement preset.

        Args:
            name: Preset name
            enhancement_type: Type of enhancement
            settings: Enhancement settings
            description: Optional description
            user_id: User ID (None for system presets)

        Returns:
            EnhancementPreset instance
        """
        return EnhancementPreset(
            name=name,
            enhancement_type=enhancement_type,
            settings=settings,
            description=description,
            user_id=user_id,
            is_system_preset=(user_id is None)
        )

    def get_default_presets(self) -> List[Dict[str, Any]]:
        """Get default enhancement presets.

        Returns:
            List of default preset configurations
        """
        return [
            {
                'name': 'Clean Speech',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.15,
                    'normalization': True,
                    'target_level': -6.0,
                    'compression': True,
                    'threshold': -25.0,
                    'ratio': 3.0,
                    'equalization': True,
                    'low_gain': 1.0,
                    'mid_gain': 0.5,
                    'high_gain': -1.0
                },
                'description': 'Clean and clear speech enhancement'
            },
            {
                'name': 'Studio Quality',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.1,
                    'normalization': True,
                    'target_level': -3.0,
                    'compression': True,
                    'threshold': -20.0,
                    'ratio': 4.0,
                    'equalization': True,
                    'low_gain': 2.0,
                    'mid_gain': 1.0,
                    'high_gain': 0.5,
                    'reverb': True,
                    'room_size': 0.3,
                    'damping': 0.4,
                    'wet_level': 0.2
                },
                'description': 'Professional studio quality enhancement'
            },
            {
                'name': 'Voice Over',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.2,
                    'normalization': True,
                    'target_level': -8.0,
                    'compression': True,
                    'threshold': -30.0,
                    'ratio': 2.5,
                    'equalization': True,
                    'low_gain': 3.0,
                    'mid_gain': 1.5,
                    'high_gain': -2.0
                },
                'description': 'Optimized for voice over and narration'
            }
        ]

    def validate_enhancement_settings(self, enhancement_type: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate enhancement settings.

        Args:
            enhancement_type: Type of enhancement
            settings: Settings to validate

        Returns:
            Validated settings with defaults

        Raises:
            ValidationException: If settings are invalid
        """
        validated_settings = {}

        if enhancement_type == 'noise_reduction':
            validated_settings['noise_factor'] = max(0.0, min(1.0, settings.get('noise_factor', 0.1)))

        elif enhancement_type == 'normalization':
            validated_settings['target_level'] = max(-50.0, min(0.0, settings.get('target_level', -3.0)))

        elif enhancement_type == 'compression':
            validated_settings['threshold'] = max(-50.0, min(0.0, settings.get('threshold', -20.0)))
            validated_settings['ratio'] = max(1.0, min(20.0, settings.get('ratio', 4.0)))

        elif enhancement_type == 'equalization':
            validated_settings['low_gain'] = max(-20.0, min(20.0, settings.get('low_gain', 0.0)))
            validated_settings['mid_gain'] = max(-20.0, min(20.0, settings.get('mid_gain', 0.0)))
            validated_settings['high_gain'] = max(-20.0, min(20.0, settings.get('high_gain', 0.0)))

        elif enhancement_type == 'reverb':
            validated_settings['room_size'] = max(0.0, min(1.0, settings.get('room_size', 0.5)))
            validated_settings['damping'] = max(0.0, min(1.0, settings.get('damping', 0.5)))
            validated_settings['wet_level'] = max(0.0, min(1.0, settings.get('wet_level', 0.3)))

        elif enhancement_type == 'full_enhancement':
            # Validate all settings
            validated_settings['noise_factor'] = max(0.0, min(1.0, settings.get('noise_factor', 0.1)))
            validated_settings['target_level'] = max(-50.0, min(0.0, settings.get('target_level', -3.0)))
            validated_settings['threshold'] = max(-50.0, min(0.0, settings.get('threshold', -20.0)))
            validated_settings['ratio'] = max(1.0, min(20.0, settings.get('ratio', 4.0)))
            validated_settings['low_gain'] = max(-20.0, min(20.0, settings.get('low_gain', 0.0)))
            validated_settings['mid_gain'] = max(-20.0, min(20.0, settings.get('mid_gain', 0.0)))
            validated_settings['high_gain'] = max(-20.0, min(20.0, settings.get('high_gain', 0.0)))
            validated_settings['room_size'] = max(0.0, min(1.0, settings.get('room_size', 0.5)))
            validated_settings['damping'] = max(0.0, min(1.0, settings.get('damping', 0.5)))
            validated_settings['wet_level'] = max(0.0, min(1.0, settings.get('wet_level', 0.3)))

        return validated_settings


# Global audio enhancer instance
audio_enhancer = AudioEnhancer()