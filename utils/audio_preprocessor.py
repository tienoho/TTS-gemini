"""
Audio Preprocessing for Voice Cloning System
"""

import asyncio
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence

from .exceptions import AudioProcessingException, ValidationException


class AudioPreprocessor:
    """Handles audio preprocessing for voice cloning including cleaning, normalization, and feature extraction."""

    def __init__(self):
        """Initialize audio preprocessor."""
        self.logger = logging.getLogger(__name__)

        # Preprocessing configuration
        self.target_sample_rate = 22050
        self.target_channels = 1
        self.target_bit_depth = 16

        # Quality thresholds
        self.min_duration = 1.0  # Minimum 1 second
        self.max_duration = 300.0  # Maximum 5 minutes
        self.min_snr = 10.0  # Minimum signal-to-noise ratio
        self.max_noise_level = -30.0  # Maximum noise level in dB

        # Normalization settings
        self.target_peak = -3.0  # Target peak level in dB
        self.target_rms = -20.0  # Target RMS level in dB

        # Silence detection
        self.silence_thresh = -40  # Silence threshold in dB
        self.min_silence_duration = 500  # Minimum silence duration in ms
        self.keep_silence = 300  # Keep silence duration in ms

    async def preprocess_audio_file(
        self,
        file_path: str,
        output_path: str = None,
        apply_noise_reduction: bool = True,
        apply_normalization: bool = True,
        apply_silence_removal: bool = True
    ) -> Dict[str, Any]:
        """Preprocess an audio file for voice cloning."""
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)

            original_duration = len(audio_data) / sample_rate

            # Validate basic requirements
            await self._validate_audio_basic(audio_data, sample_rate, original_duration)

            # Apply preprocessing steps
            processed_audio = audio_data.copy()

            # 1. Noise reduction
            if apply_noise_reduction:
                processed_audio = await self._reduce_noise(processed_audio, sample_rate)

            # 2. Normalization
            if apply_normalization:
                processed_audio = await self._normalize_audio(processed_audio, sample_rate)

            # 3. Silence removal
            if apply_silence_removal:
                processed_audio = await self._remove_silence(processed_audio, sample_rate)

            # Validate processed audio
            final_duration = len(processed_audio) / sample_rate
            await self._validate_processed_audio(processed_audio, sample_rate, final_duration)

            # Save processed audio
            if output_path is None:
                output_path = file_path.replace('.wav', '_processed.wav')

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save with target format
            sf.write(
                output_path,
                processed_audio,
                self.target_sample_rate,
                subtype='PCM_16'
            )

            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                processed_audio, sample_rate, original_duration, final_duration
            )

            return {
                'original_path': file_path,
                'processed_path': output_path,
                'original_duration': original_duration,
                'processed_duration': final_duration,
                'sample_rate': sample_rate,
                'target_sample_rate': self.target_sample_rate,
                'quality_metrics': quality_metrics,
                'preprocessing_applied': {
                    'noise_reduction': apply_noise_reduction,
                    'normalization': apply_normalization,
                    'silence_removal': apply_silence_removal
                }
            }

        except Exception as e:
            self.logger.error(f"Audio preprocessing error: {str(e)}")
            raise AudioProcessingException(f"Preprocessing failed: {str(e)}")

    async def _validate_audio_basic(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        duration: float
    ):
        """Validate basic audio requirements."""
        if duration < self.min_duration:
            raise ValidationException(
                f"Audio too short: {duration".2f"}s. Minimum required: {self.min_duration}s"
            )

        if duration > self.max_duration:
            raise ValidationException(
                f"Audio too long: {duration".2f"}s. Maximum allowed: {self.max_duration}s"
            )

        if sample_rate < 8000:
            raise ValidationException(
                f"Sample rate too low: {sample_rate}Hz. Minimum required: 8000Hz"
            )

        if sample_rate > 48000:
            raise ValidationException(
                f"Sample rate too high: {sample_rate}Hz. Maximum allowed: 48000Hz"
            )

    async def _validate_processed_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        duration: float
    ):
        """Validate processed audio quality."""
        if duration < self.min_duration:
            raise ValidationException(
                f"Processed audio too short: {duration".2f"}s. Minimum required: {self.min_duration}s"
            )

        # Check for audio quality issues
        if np.max(np.abs(audio_data)) < 0.01:
            raise ValidationException("Processed audio is too quiet")

        if np.max(np.abs(audio_data)) > 0.99:
            raise ValidationException("Processed audio is too loud (clipping detected)")

        # Check for excessive silence
        silence_ratio = self._calculate_silence_ratio(audio_data)
        if silence_ratio > 0.8:  # More than 80% silence
            raise ValidationException(
                f"Too much silence in processed audio: {silence_ratio".1%"}"
            )

    async def _reduce_noise(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply noise reduction to audio."""
        try:
            # Simple noise reduction using spectral gating
            # Calculate noise profile from quiet segments
            noise_profile = self._estimate_noise_profile(audio_data)

            if noise_profile is not None:
                # Apply spectral subtraction
                processed_audio = self._spectral_subtraction(audio_data, noise_profile)
            else:
                processed_audio = audio_data

            return processed_audio

        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {str(e)}")
            return audio_data

    async def _normalize_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Normalize audio levels."""
        try:
            # Calculate current peak and RMS
            current_peak = np.max(np.abs(audio_data))
            current_rms = np.sqrt(np.mean(audio_data ** 2))

            # Calculate target levels
            target_peak_linear = 10 ** (self.target_peak / 20)
            target_rms_linear = 10 ** (self.target_rms / 20)

            # Normalize to target RMS first
            if current_rms > 0:
                rms_normalized = audio_data * (target_rms_linear / current_rms)

                # Then apply peak normalization
                peak_after_rms = np.max(np.abs(rms_normalized))
                if peak_after_rms > target_peak_linear:
                    peak_normalized = rms_normalized * (target_peak_linear / peak_after_rms)
                else:
                    peak_normalized = rms_normalized
            else:
                peak_normalized = audio_data

            return peak_normalized

        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {str(e)}")
            return audio_data

    async def _remove_silence(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Remove silence from audio."""
        try:
            # Convert to pydub AudioSegment for silence detection
            # Normalize to 16-bit for pydub
            normalized_data = np.int16(audio_data * 32767)
            audio_segment = AudioSegment(
                normalized_data.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )

            # Split on silence
            chunks = split_on_silence(
                audio_segment,
                min_silence_len=self.min_silence_duration,
                silence_thresh=self.silence_thresh,
                keep_silence=self.keep_silence
            )

            # If no chunks found, return original
            if not chunks:
                return audio_data

            # Concatenate non-silent chunks
            processed_segment = chunks[0]
            for chunk in chunks[1:]:
                processed_segment += chunk

            # Convert back to numpy array
            samples = np.array(processed_segment.get_array_of_samples())
            processed_audio = samples.astype(np.float32) / 32767.0

            return processed_audio

        except Exception as e:
            self.logger.warning(f"Silence removal failed: {str(e)}")
            return audio_data

    def _estimate_noise_profile(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Estimate noise profile from audio."""
        try:
            # Use the first and last 10% of audio as potential noise segments
            length = len(audio_data)
            noise_segment_length = int(length * 0.1)

            noise_segments = []
            if noise_segment_length > 0:
                noise_segments.append(audio_data[:noise_segment_length])
                noise_segments.append(audio_data[-noise_segment_length:])

            if not noise_segments:
                return None

            # Average the noise segments
            noise_profile = np.mean(np.array(noise_segments), axis=0)

            return noise_profile

        except Exception:
            return None

    def _spectral_subtraction(
        self,
        audio_data: np.ndarray,
        noise_profile: np.ndarray
    ) -> np.ndarray:
        """Apply spectral subtraction for noise reduction."""
        try:
            # Parameters
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor

            # Compute STFT
            stft = librosa.stft(audio_data)
            noise_stft = librosa.stft(noise_profile)

            # Magnitude and phase
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            noise_magnitude = np.abs(noise_stft)

            # Spectral subtraction
            enhanced_magnitude = magnitude - alpha * noise_magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)

            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)

            return enhanced_audio

        except Exception as e:
            self.logger.warning(f"Spectral subtraction failed: {str(e)}")
            return audio_data

    async def _calculate_quality_metrics(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        original_duration: float,
        processed_duration: float
    ) -> Dict[str, Any]:
        """Calculate quality metrics for processed audio."""
        try:
            # Basic metrics
            peak_level = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            dynamic_range = 20 * np.log10(peak_level / rms_level) if rms_level > 0 else 0

            # Signal-to-noise ratio
            snr = self._calculate_snr(audio_data)

            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))

            # Zero-crossing rate
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))

            # Silence ratio
            silence_ratio = self._calculate_silence_ratio(audio_data)

            # Audio quality score (1-10 scale)
            quality_score = self._calculate_overall_quality_score(
                snr, dynamic_range, silence_ratio, spectral_centroid, sample_rate
            )

            return {
                'peak_level_db': 20 * np.log10(peak_level) if peak_level > 0 else -100,
                'rms_level_db': 20 * np.log10(rms_level) if rms_level > 0 else -100,
                'dynamic_range_db': dynamic_range,
                'snr_db': snr,
                'spectral_centroid_hz': spectral_centroid,
                'spectral_rolloff_hz': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'silence_ratio': silence_ratio,
                'quality_score': quality_score,
                'duration_reduction_percent': ((original_duration - processed_duration) / original_duration) * 100 if original_duration > 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {str(e)}")
            return {
                'peak_level_db': 0,
                'rms_level_db': 0,
                'dynamic_range_db': 0,
                'snr_db': 0,
                'spectral_centroid_hz': 0,
                'spectral_rolloff_hz': 0,
                'zero_crossing_rate': 0,
                'silence_ratio': 0,
                'quality_score': 5.0,
                'duration_reduction_percent': 0
            }

    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            # Simple SNR calculation
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data) * 0.1  # Rough noise estimate

            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 20.0  # High SNR if no noise detected

            return max(-20.0, min(50.0, snr))  # Clamp to reasonable range

        except Exception:
            return 0.0

    def _calculate_silence_ratio(self, audio_data: np.ndarray) -> float:
        """Calculate ratio of silence in audio."""
        try:
            # Detect silence (very low amplitude)
            silence_threshold = 0.01
            silence_frames = np.sum(np.abs(audio_data) < silence_threshold)
            total_frames = len(audio_data)

            return silence_frames / total_frames if total_frames > 0 else 0

        except Exception:
            return 0.0

    def _calculate_overall_quality_score(
        self,
        snr: float,
        dynamic_range: float,
        silence_ratio: float,
        spectral_centroid: float,
        sample_rate: int
    ) -> float:
        """Calculate overall quality score (1-10 scale)."""
        try:
            # SNR score (40% weight)
            snr_score = min(10.0, max(1.0, (snr + 20) / 7))  # Normalize -20dB to 50dB -> 1 to 10

            # Dynamic range score (20% weight)
            dr_score = min(10.0, max(1.0, dynamic_range / 6))  # Normalize 0dB to 60dB -> 1 to 10

            # Silence ratio score (20% weight)
            silence_score = max(1.0, 10.0 - (silence_ratio * 10))  # 0% silence -> 10, 100% silence -> 0

            # Spectral content score (20% weight)
            # Good spectral content should be in mid-frequency range
            spectral_score = 1.0
            if 100 < spectral_centroid < sample_rate / 4:
                spectral_score = 10.0
            elif spectral_centroid < 50 or spectral_centroid > sample_rate / 2:
                spectral_score = 5.0

            # Calculate weighted score
            overall_score = (
                snr_score * 0.4 +
                dr_score * 0.2 +
                silence_score * 0.2 +
                spectral_score * 0.2
            )

            return max(1.0, min(10.0, overall_score))

        except Exception:
            return 5.0

    async def extract_features(
        self,
        audio_path: str,
        feature_types: List[str] = None
    ) -> Dict[str, Any]:
        """Extract audio features for voice cloning."""
        try:
            if feature_types is None:
                feature_types = ['mfcc', 'mel_spectrogram', 'pitch', 'spectral']

            # Load audio
            audio_data, sample_rate = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)

            features = {}

            # Extract MFCC
            if 'mfcc' in feature_types:
                mfcc = librosa.feature.mfcc(
                    y=audio_data,
                    sr=sample_rate,
                    n_mfcc=13,
                    n_fft=2048,
                    hop_length=512
                )
                features['mfcc'] = mfcc
                features['mfcc_mean'] = np.mean(mfcc, axis=1)
                features['mfcc_std'] = np.std(mfcc, axis=1)

            # Extract mel spectrogram
            if 'mel_spectrogram' in feature_types:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_data,
                    sr=sample_rate,
                    n_mels=128,
                    n_fft=2048,
                    hop_length=512
                )
                features['mel_spectrogram'] = mel_spec
                features['mel_spectrogram_db'] = librosa.power_to_db(mel_spec)

            # Extract pitch
            if 'pitch' in feature_types:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio_data,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sample_rate,
                    frame_length=2048,
                    hop_length=512
                )
                features['pitch'] = f0
                features['voiced_flag'] = voiced_flag
                features['pitch_mean'] = np.mean(f0[voiced_flag]) if np.any(voiced_flag) else 0
                features['pitch_std'] = np.std(f0[voiced_flag]) if np.any(voiced_flag) else 0

            # Extract spectral features
            if 'spectral' in feature_types:
                spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)

                features['spectral_centroid'] = spectral_centroid
                features['spectral_rolloff'] = spectral_rolloff
                features['spectral_bandwidth'] = spectral_bandwidth

                features['spectral_centroid_mean'] = np.mean(spectral_centroid)
                features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
                features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

            # Add metadata
            features['duration'] = len(audio_data) / sample_rate
            features['sample_rate'] = sample_rate
            features['feature_types'] = feature_types

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            raise AudioProcessingException(f"Feature extraction failed: {str(e)}")

    async def convert_audio_format(
        self,
        input_path: str,
        output_path: str,
        target_format: str = 'wav',
        target_sample_rate: int = None,
        target_channels: int = None
    ) -> Dict[str, Any]:
        """Convert audio file to different format."""
        try:
            if target_sample_rate is None:
                target_sample_rate = self.target_sample_rate

            if target_channels is None:
                target_channels = self.target_channels

            # Load audio
            audio_data, original_sample_rate = librosa.load(input_path, sr=None, mono=(target_channels == 1))

            # Resample if needed
            if original_sample_rate != target_sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sample_rate, target_sr=target_sample_rate)

            # Convert to mono if needed
            if target_channels == 1 and len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)

            # Save in target format
            if target_format.lower() == 'wav':
                sf.write(output_path, audio_data, target_sample_rate, subtype='PCM_16')
            elif target_format.lower() == 'flac':
                sf.write(output_path, audio_data, target_sample_rate, format='FLAC')
            elif target_format.lower() == 'mp3':
                # For MP3, we'd need additional libraries like pydub with ffmpeg
                # For now, save as WAV and note the conversion
                temp_wav = output_path.replace('.mp3', '_temp.wav')
                sf.write(temp_wav, audio_data, target_sample_rate, subtype='PCM_16')

                # Convert to MP3 using pydub
                audio_segment = AudioSegment.from_wav(temp_wav)
                audio_segment.export(output_path, format='mp3')
                os.remove(temp_wav)
            else:
                raise ValidationException(f"Unsupported format: {target_format}")

            # Get file info
            file_size = os.path.getsize(output_path)
            duration = len(audio_data) / target_sample_rate

            return {
                'input_path': input_path,
                'output_path': output_path,
                'original_sample_rate': original_sample_rate,
                'target_sample_rate': target_sample_rate,
                'channels': target_channels,
                'duration': duration,
                'file_size': file_size,
                'format': target_format
            }

        except Exception as e:
            self.logger.error(f"Audio format conversion error: {str(e)}")
            raise AudioProcessingException(f"Format conversion failed: {str(e)}")

    async def validate_audio_for_cloning(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """Validate audio file for voice cloning training."""
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(file_path, sr=self.target_sample_rate, mono=True)
            duration = len(audio_data) / sample_rate

            # Basic validation
            issues = []

            if duration < self.min_duration:
                issues.append(f"Duration too short: {duration".2f"}s (min: {self.min_duration}s)")

            if duration > self.max_duration:
                issues.append(f"Duration too long: {duration".2f"}s (max: {self.max_duration}s)")

            if sample_rate < 8000:
                issues.append(f"Sample rate too low: {sample_rate}Hz (min: 8000Hz)")

            # Quality checks
            snr = self._calculate_snr(audio_data)
            if snr < self.min_snr:
                issues.append(f"Signal-to-noise ratio too low: {snr".1f"}dB (min: {self.min_snr}dB)")

            # Check for excessive silence
            silence_ratio = self._calculate_silence_ratio(audio_data)
            if silence_ratio > 0.7:
                issues.append(f"Too much silence: {silence_ratio".1%"}")

            # Check for clipping
            peak_level = np.max(np.abs(audio_data))
            if peak_level > 0.98:
                issues.append("Audio clipping detected")

            # Calculate quality score
            quality_score = self._calculate_overall_quality_score(
                snr, 20 * np.log10(peak_level / np.sqrt(np.mean(audio_data ** 2))), silence_ratio,
                np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)), sample_rate
            )

            is_valid = len(issues) == 0

            return {
                'is_valid': is_valid,
                'duration': duration,
                'sample_rate': sample_rate,
                'snr_db': snr,
                'silence_ratio': silence_ratio,
                'peak_level': peak_level,
                'quality_score': quality_score,
                'issues': issues,
                'recommendations': self._get_validation_recommendations(issues)
            }

        except Exception as e:
            self.logger.error(f"Audio validation error: {str(e)}")
            raise AudioProcessingException(f"Validation failed: {str(e)}")

    def _get_validation_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations for fixing validation issues."""
        recommendations = []

        for issue in issues:
            if "Duration too short" in issue:
                recommendations.append("Record longer audio samples (minimum 30 seconds)")
            elif "Duration too long" in issue:
                recommendations.append("Split long recordings into shorter segments")
            elif "Sample rate too low" in issue:
                recommendations.append("Use higher quality recording settings (minimum 8kHz)")
            elif "Signal-to-noise ratio too low" in issue:
                recommendations.append("Record in a quiet environment, reduce background noise")
            elif "Too much silence" in issue:
                recommendations.append("Remove long pauses and silence from recordings")
            elif "Audio clipping" in issue:
                recommendations.append("Reduce recording volume to prevent distortion")

        if not recommendations:
            recommendations.append("Audio is suitable for voice cloning")

        return recommendations


# Global instance
audio_preprocessor = AudioPreprocessor()

def get_audio_preprocessor() -> AudioPreprocessor:
    """Get audio preprocessor instance."""
    return audio_preprocessor