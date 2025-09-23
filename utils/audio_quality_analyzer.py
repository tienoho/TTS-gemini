"""
Audio Quality Analyzer for TTS system
Provides comprehensive audio quality analysis and metrics
"""

import asyncio
import logging
import numpy as np
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import math

try:
    import librosa
    import librosa.feature
    import soundfile as sf
    from scipy import signal
    from scipy.fft import fft, fftfreq
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

from .exceptions import AudioProcessingException, ValidationException
from models.audio_enhancement import AudioQualityMetric


class AudioQualityAnalyzer:
    """Comprehensive audio quality analysis service."""

    def __init__(self):
        """Initialize the audio quality analyzer."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit

        if not HAS_AUDIO_LIBS:
            self.logger.warning("Audio analysis libraries not available. Install librosa, scipy, and soundfile for full functionality.")

    def validate_audio_file(self, audio_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate audio file before analysis.

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

    def calculate_snr(self, audio: np.ndarray, noise_duration: float = 1.0) -> float:
        """Calculate Signal-to-Noise Ratio.

        Args:
            audio: Input audio array
            noise_duration: Duration to use for noise estimation (seconds)

        Returns:
            SNR in dB
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Estimate noise from the beginning of the audio
            noise_samples = int(noise_duration * 44100)  # Assume 44.1kHz
            if noise_samples > len(audio):
                noise_samples = len(audio) // 4

            if noise_samples > 0:
                noise_segment = audio[:noise_samples]
                noise_power = np.mean(noise_segment**2)

                # Signal power (excluding noise segment)
                signal_segment = audio[noise_samples:]
                if len(signal_segment) > 0:
                    signal_power = np.mean(signal_segment**2)
                else:
                    signal_power = np.mean(audio**2)
            else:
                noise_power = np.mean(audio**2)
                signal_power = noise_power

            if noise_power == 0:
                return float('inf')

            snr = 10 * np.log10(signal_power / noise_power)
            return max(snr, 0.0)  # Ensure non-negative SNR

        except Exception as e:
            self.logger.warning(f"SNR calculation failed: {str(e)}")
            return 0.0

    def calculate_thd(self, audio: np.ndarray, sample_rate: int, fundamental_freq: float = 440.0) -> float:
        """Calculate Total Harmonic Distortion.

        Args:
            audio: Input audio array
            sample_rate: Sample rate
            fundamental_freq: Fundamental frequency to analyze

        Returns:
            THD as percentage
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Perform FFT
            n = len(audio)
            yf = fft(audio)
            xf = fftfreq(n, 1/sample_rate)

            # Find fundamental frequency bin
            fundamental_idx = np.argmin(np.abs(xf - fundamental_freq))

            # Calculate harmonic frequencies
            harmonic_freqs = [fundamental_freq * (i+2) for i in range(5)]  # 2nd to 6th harmonics

            # Calculate power of fundamental
            fundamental_power = np.abs(yf[fundamental_idx])**2

            # Calculate power of harmonics
            harmonic_power = 0
            for freq in harmonic_freqs:
                idx = np.argmin(np.abs(xf - freq))
                if idx < len(yf):
                    harmonic_power += np.abs(yf[idx])**2

            # Calculate THD
            if fundamental_power == 0:
                return 0.0

            thd = np.sqrt(harmonic_power / fundamental_power) * 100  # Convert to percentage
            return min(thd, 100.0)  # Cap at 100%

        except Exception as e:
            self.logger.warning(f"THD calculation failed: {str(e)}")
            return 0.0

    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate Root Mean Square of audio signal.

        Args:
            audio: Input audio array

        Returns:
            RMS value
        """
        try:
            if len(audio) == 0:
                return 0.0

            return np.sqrt(np.mean(audio**2))

        except Exception as e:
            self.logger.warning(f"RMS calculation failed: {str(e)}")
            return 0.0

    def calculate_peak(self, audio: np.ndarray) -> float:
        """Calculate peak amplitude of audio signal.

        Args:
            audio: Input audio array

        Returns:
            Peak amplitude
        """
        try:
            if len(audio) == 0:
                return 0.0

            return np.max(np.abs(audio))

        except Exception as e:
            self.logger.warning(f"Peak calculation failed: {str(e)}")
            return 0.0

    def calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range of audio signal.

        Args:
            audio: Input audio array

        Returns:
            Dynamic range in dB
        """
        try:
            if len(audio) == 0:
                return 0.0

            peak = self.calculate_peak(audio)
            rms = self.calculate_rms(audio)

            if peak == 0 or rms == 0:
                return 0.0

            return 20 * np.log10(peak / rms)

        except Exception as e:
            self.logger.warning(f"Dynamic range calculation failed: {str(e)}")
            return 0.0

    def calculate_spectral_centroid(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate spectral centroid.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Spectral centroid in Hz
        """
        if not HAS_AUDIO_LIBS:
            return 0.0

        try:
            if len(audio) == 0:
                return 0.0

            # Calculate spectral centroid using librosa
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            return float(np.mean(centroid))

        except Exception as e:
            self.logger.warning(f"Spectral centroid calculation failed: {str(e)}")
            return 0.0

    def calculate_spectral_rolloff(self, audio: np.ndarray, sample_rate: int, rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff.

        Args:
            audio: Input audio array
            sample_rate: Sample rate
            rolloff_percent: Roll-off percentage

        Returns:
            Spectral rolloff frequency in Hz
        """
        if not HAS_AUDIO_LIBS:
            return 0.0

        try:
            if len(audio) == 0:
                return 0.0

            # Calculate spectral rolloff using librosa
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=rolloff_percent)
            return float(np.mean(rolloff))

        except Exception as e:
            self.logger.warning(f"Spectral rolloff calculation failed: {str(e)}")
            return 0.0

    def calculate_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Calculate zero crossing rate.

        Args:
            audio: Input audio array

        Returns:
            Zero crossing rate
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Calculate zero crossings
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
            return zero_crossings / len(audio)

        except Exception as e:
            self.logger.warning(f"Zero crossing rate calculation failed: {str(e)}")
            return 0.0

    def calculate_clarity_score(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate clarity score based on frequency content and noise.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Clarity score (0-10)
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Calculate SNR
            snr = self.calculate_snr(audio)

            # Calculate spectral centroid
            centroid = self.calculate_spectral_centroid(audio, sample_rate)

            # Calculate spectral rolloff
            rolloff = self.calculate_spectral_rolloff(audio, sample_rate)

            # Normalize metrics
            snr_score = min(snr / 30.0, 1.0)  # Normalize SNR to 0-1
            centroid_score = min(centroid / 5000.0, 1.0)  # Normalize centroid to 0-1
            rolloff_score = min(rolloff / 10000.0, 1.0)  # Normalize rolloff to 0-1

            # Combine scores
            clarity = (snr_score * 0.4 + centroid_score * 0.3 + rolloff_score * 0.3)

            # Convert to 0-10 scale
            return clarity * 10.0

        except Exception as e:
            self.logger.warning(f"Clarity score calculation failed: {str(e)}")
            return 0.0

    def calculate_noise_score(self, audio: np.ndarray) -> float:
        """Calculate noise score based on noise characteristics.

        Args:
            audio: Input audio array

        Returns:
            Noise score (0-10, where 10 is no noise)
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Calculate SNR
            snr = self.calculate_snr(audio)

            # Calculate dynamic range
            dynamic_range = self.calculate_dynamic_range(audio)

            # Normalize metrics
            snr_score = min(snr / 40.0, 1.0)  # Normalize SNR to 0-1
            dynamic_range_score = min(dynamic_range / 60.0, 1.0)  # Normalize dynamic range to 0-1

            # Combine scores (higher SNR and dynamic range = less noise)
            noise_score = (snr_score * 0.7 + dynamic_range_score * 0.3)

            # Convert to 0-10 scale (invert for noise score)
            return (1.0 - noise_score) * 10.0

        except Exception as e:
            self.logger.warning(f"Noise score calculation failed: {str(e)}")
            return 0.0

    def calculate_distortion_score(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate distortion score based on harmonic distortion.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Distortion score (0-10, where 10 is no distortion)
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Calculate THD
            thd = self.calculate_thd(audio, sample_rate)

            # Calculate peak level (avoid clipping)
            peak = self.calculate_peak(audio)

            # Normalize metrics
            thd_score = max(0.0, 1.0 - (thd / 10.0))  # Normalize THD to 0-1
            peak_score = 1.0 if peak < 0.95 else max(0.0, 1.0 - (peak - 0.95) * 10.0)  # Penalize clipping

            # Combine scores
            distortion_score = (thd_score * 0.7 + peak_score * 0.3)

            # Convert to 0-10 scale
            return distortion_score * 10.0

        except Exception as e:
            self.logger.warning(f"Distortion score calculation failed: {str(e)}")
            return 0.0

    def calculate_overall_quality_score(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate overall quality score.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Overall quality score (0-10)
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Calculate individual scores
            clarity_score = self.calculate_clarity_score(audio, sample_rate)
            noise_score = self.calculate_noise_score(audio)
            distortion_score = self.calculate_distortion_score(audio, sample_rate)

            # Weight the scores
            overall_score = (
                clarity_score * 0.4 +      # Clarity is most important
                noise_score * 0.3 +        # Low noise is important
                distortion_score * 0.3     # Low distortion is important
            )

            # Ensure score is within bounds
            return max(0.0, min(10.0, overall_score))

        except Exception as e:
            self.logger.warning(f"Overall quality score calculation failed: {str(e)}")
            return 0.0

    def analyze_audio_quality(self, audio_data: bytes, analysis_method: str = 'algorithmic') -> Dict[str, Any]:
        """Perform comprehensive audio quality analysis.

        Args:
            audio_data: Audio data as bytes
            analysis_method: Analysis method ('algorithmic', 'ai', 'manual')

        Returns:
            Dictionary with quality metrics

        Raises:
            AudioProcessingException: If analysis fails
        """
        start_time = time.time()

        try:
            # Load audio
            audio, sample_rate = self.load_audio_from_bytes(audio_data)

            # Calculate basic metrics
            snr = self.calculate_snr(audio)
            thd = self.calculate_thd(audio, sample_rate)
            rms = self.calculate_rms(audio)
            peak = self.calculate_peak(audio)
            dynamic_range = self.calculate_dynamic_range(audio)

            # Calculate frequency domain metrics
            spectral_centroid = self.calculate_spectral_centroid(audio, sample_rate)
            spectral_rolloff = self.calculate_spectral_rolloff(audio, sample_rate)
            zero_crossing_rate = self.calculate_zero_crossing_rate(audio)

            # Calculate quality scores
            overall_quality_score = self.calculate_overall_quality_score(audio, sample_rate)
            clarity_score = self.calculate_clarity_score(audio, sample_rate)
            noise_score = self.calculate_noise_score(audio)
            distortion_score = self.calculate_distortion_score(audio, sample_rate)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Determine quality grade
            quality_grade = self.get_quality_grade(overall_quality_score)

            return {
                'snr': snr,
                'thd': thd,
                'rms': rms,
                'peak': peak,
                'dynamic_range': dynamic_range,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'overall_quality_score': overall_quality_score,
                'clarity_score': clarity_score,
                'noise_score': noise_score,
                'distortion_score': distortion_score,
                'quality_grade': quality_grade,
                'analysis_method': analysis_method,
                'processing_time': processing_time,
                'sample_rate': sample_rate,
                'duration': len(audio) / sample_rate,
                'samples': len(audio)
            }

        except Exception as e:
            raise AudioProcessingException(f"Audio quality analysis failed: {str(e)}")

    async def analyze_audio_quality_async(self, audio_data: bytes, analysis_method: str = 'algorithmic') -> Dict[str, Any]:
        """Async version of analyze_audio_quality.

        Args:
            audio_data: Audio data as bytes
            analysis_method: Analysis method

        Returns:
            Dictionary with quality metrics
        """
        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.analyze_audio_quality, audio_data, analysis_method
        )

    def get_quality_grade(self, quality_score: float) -> str:
        """Get quality grade based on quality score.

        Args:
            quality_score: Quality score (0-10)

        Returns:
            Quality grade string
        """
        if quality_score >= 9.0:
            return 'Excellent'
        elif quality_score >= 8.0:
            return 'Very Good'
        elif quality_score >= 7.0:
            return 'Good'
        elif quality_score >= 6.0:
            return 'Fair'
        elif quality_score >= 5.0:
            return 'Poor'
        else:
            return 'Very Poor'

    def create_quality_metric(self, audio_file_id: int, analysis_results: Dict[str, Any],
                            enhancement_id: int = None) -> AudioQualityMetric:
        """Create AudioQualityMetric from analysis results.

        Args:
            audio_file_id: Audio file ID
            analysis_results: Analysis results dictionary
            enhancement_id: Enhancement ID (optional)

        Returns:
            AudioQualityMetric instance
        """
        return AudioQualityMetric(
            audio_file_id=audio_file_id,
            enhancement_id=enhancement_id,
            analysis_method=analysis_results.get('analysis_method', 'algorithmic'),
            snr=analysis_results.get('snr'),
            thd=analysis_results.get('thd'),
            rms=analysis_results.get('rms'),
            peak=analysis_results.get('peak'),
            dynamic_range=analysis_results.get('dynamic_range'),
            spectral_centroid=analysis_results.get('spectral_centroid'),
            spectral_rolloff=analysis_results.get('spectral_rolloff'),
            zero_crossing_rate=analysis_results.get('zero_crossing_rate'),
            overall_quality_score=analysis_results.get('overall_quality_score'),
            clarity_score=analysis_results.get('clarity_score'),
            noise_score=analysis_results.get('noise_score'),
            distortion_score=analysis_results.get('distortion_score'),
            confidence=0.95,  # Default confidence
            processing_time=analysis_results.get('processing_time')
        )

    def compare_audio_quality(self, audio1_data: bytes, audio2_data: bytes) -> Dict[str, Any]:
        """Compare quality between two audio files.

        Args:
            audio1_data: First audio data
            audio2_data: Second audio data

        Returns:
            Dictionary with comparison results
        """
        try:
            # Analyze both audio files
            quality1 = self.analyze_audio_quality(audio1_data, 'algorithmic')
            quality2 = self.analyze_audio_quality(audio2_data, 'algorithmic')

            # Calculate differences
            snr_diff = quality2['snr'] - quality1['snr']
            quality_diff = quality2['overall_quality_score'] - quality1['overall_quality_score']

            # Determine which is better
            if quality2['overall_quality_score'] > quality1['overall_quality_score']:
                better_audio = 'audio2'
                improvement = quality_diff
            elif quality1['overall_quality_score'] > quality2['overall_quality_score']:
                better_audio = 'audio1'
                improvement = -quality_diff
            else:
                better_audio = 'equal'
                improvement = 0.0

            return {
                'audio1_quality': quality1,
                'audio2_quality': quality2,
                'snr_difference': snr_diff,
                'quality_difference': quality_diff,
                'better_audio': better_audio,
                'improvement': improvement,
                'comparison_method': 'algorithmic'
            }

        except Exception as e:
            raise AudioProcessingException(f"Audio comparison failed: {str(e)}")

    def get_quality_recommendations(self, quality_score: float, quality_metrics: Dict[str, Any]) -> List[str]:
        """Get quality improvement recommendations.

        Args:
            quality_score: Overall quality score
            quality_metrics: Quality metrics dictionary

        Returns:
            List of recommendations
        """
        recommendations = []

        if quality_score < 6.0:
            recommendations.append("Audio quality is poor. Consider re-recording or applying noise reduction.")

        if quality_metrics.get('snr', 0) < 20.0:
            recommendations.append("Signal-to-noise ratio is low. Apply noise reduction enhancement.")

        if quality_metrics.get('thd', 0) > 5.0:
            recommendations.append("High distortion detected. Consider using compression or re-recording.")

        if quality_metrics.get('peak', 0) > 0.95:
            recommendations.append("Audio is clipping. Reduce recording level or apply normalization.")

        if quality_metrics.get('dynamic_range', 0) < 30.0:
            recommendations.append("Dynamic range is limited. Consider using compression to improve consistency.")

        if quality_metrics.get('spectral_centroid', 0) < 1000.0:
            recommendations.append("Audio lacks high-frequency content. Consider using equalization to brighten the sound.")

        if not recommendations:
            recommendations.append("Audio quality is good. No major improvements needed.")

        return recommendations


# Global audio quality analyzer instance
audio_quality_analyzer = AudioQualityAnalyzer()