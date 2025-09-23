"""
Real-time Audio Enhancement for TTS system
Provides low-latency streaming audio enhancement with adaptive processing
"""

import asyncio
import logging
import numpy as np
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import deque
from pathlib import Path
import statistics

try:
    import librosa
    import librosa.effects
    import soundfile as sf
    from scipy import signal
    has_audio_libs = True
except ImportError:
    has_audio_libs = False

from .audio_enhancer import audio_enhancer
from .audio_quality_analyzer import audio_quality_analyzer
from .exceptions import AudioProcessingException, ValidationException


class RealTimeEnhancer:
    """Real-time audio enhancement with low latency and adaptive processing."""

    def __init__(self):
        """Initialize the real-time enhancer."""
        self.logger = logging.getLogger(__name__)

        # Processing configuration
        self.sample_rate = 44100
        self.chunk_size = 1024  # Audio frames per chunk
        self.buffer_size = 8192  # Total buffer size
        self.target_latency = 0.05  # Target latency in seconds (50ms)

        # Adaptive processing settings
        self.adaptive_enabled = True
        self.quality_monitoring_enabled = True
        self.auto_adjust_enabled = True

        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.audio_queue = None
        self.result_queue = None

        # Quality tracking
        self.quality_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=50)
        self.current_quality_score = 0.0

        # Enhancement settings
        self.current_enhancement_type = 'adaptive'
        self.current_settings = self._get_default_adaptive_settings()

        if not has_audio_libs:
            self.logger.warning("Audio processing libraries not available. Real-time enhancement will be limited.")

    def _get_default_adaptive_settings(self) -> Dict[str, Any]:
        """Get default adaptive enhancement settings.

        Returns:
            Dictionary with default settings
        """
        return {
            'noise_reduction': True,
            'noise_factor': 0.1,
            'normalization': True,
            'target_level': -12.0,
            'compression': True,
            'threshold': -25.0,
            'ratio': 3.0,
            'equalization': True,
            'low_gain': 1.0,
            'mid_gain': 0.5,
            'high_gain': 0.0,
            'adaptive_noise_reduction': True,
            'adaptive_compression': True,
            'quality_threshold': 7.0
        }

    def start_processing(self, enhancement_type: str = 'adaptive',
                        settings: Dict[str, Any] = None) -> bool:
        """Start real-time audio enhancement processing.

        Args:
            enhancement_type: Type of enhancement to apply
            settings: Enhancement settings

        Returns:
            True if started successfully
        """
        try:
            if self.is_processing:
                self.logger.warning("Real-time enhancement is already running")
                return False

            # Update settings
            self.current_enhancement_type = enhancement_type
            if settings:
                self.current_settings.update(settings)

            # Initialize queues
            self.audio_queue = queue.Queue(maxsize=100)  # Input audio queue
            self.result_queue = queue.Queue(maxsize=100)  # Processed audio queue

            # Start processing thread
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

            self.logger.info(f"Real-time enhancement started with type: {enhancement_type}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start real-time enhancement: {str(e)}")
            self.is_processing = False
            return False

    def stop_processing(self) -> bool:
        """Stop real-time audio enhancement processing.

        Returns:
            True if stopped successfully
        """
        try:
            if not self.is_processing:
                return True

            self.is_processing = False

            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)

            # Clear queues
            if self.audio_queue:
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break

            if self.result_queue:
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break

            self.logger.info("Real-time enhancement stopped")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping real-time enhancement: {str(e)}")
            return False

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process a single audio chunk.

        Args:
            audio_chunk: Input audio chunk as numpy array

        Returns:
            Processed audio chunk or None if processing fails
        """
        try:
            start_time = time.time()

            # Validate input
            if len(audio_chunk) == 0:
                return audio_chunk

            # Apply adaptive settings if enabled
            if self.adaptive_enabled:
                self._update_adaptive_settings(audio_chunk)

            # Apply enhancement based on type
            if self.current_enhancement_type == 'adaptive':
                processed_chunk = self._apply_adaptive_enhancement(audio_chunk)
            elif self.current_enhancement_type == 'noise_reduction':
                processed_chunk = self._apply_noise_reduction(audio_chunk)
            elif self.current_enhancement_type == 'normalization':
                processed_chunk = self._apply_normalization(audio_chunk)
            elif self.current_enhancement_type == 'compression':
                processed_chunk = self._apply_compression(audio_chunk)
            elif self.current_enhancement_type == 'equalization':
                processed_chunk = self._apply_equalization(audio_chunk)
            else:
                processed_chunk = audio_chunk

            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Monitor quality if enabled
            if self.quality_monitoring_enabled:
                self._monitor_quality(processed_chunk, audio_chunk)

            return processed_chunk

        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {str(e)}")
            return audio_chunk  # Return original on error

    def _processing_loop(self):
        """Main processing loop for real-time enhancement."""
        try:
            while self.is_processing:
                try:
                    # Get audio chunk from queue (non-blocking with timeout)
                    audio_chunk = self.audio_queue.get(timeout=0.1)

                    # Process chunk
                    processed_chunk = self.process_audio_chunk(audio_chunk)

                    # Put result in output queue
                    if processed_chunk is not None:
                        self.result_queue.put(processed_chunk, timeout=0.1)

                except queue.Empty:
                    # No audio to process, continue loop
                    continue
                except queue.Full:
                    self.logger.warning("Result queue is full, dropping processed chunk")
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {str(e)}")

        except Exception as e:
            self.logger.error(f"Fatal error in processing loop: {str(e)}")
        finally:
            self.is_processing = False

    def _apply_adaptive_enhancement(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply adaptive enhancement based on audio characteristics.

        Args:
            audio_chunk: Input audio chunk

        Returns:
            Adaptively enhanced audio chunk
        """
        try:
            # Analyze current audio characteristics
            rms = np.sqrt(np.mean(audio_chunk**2))
            peak = np.max(np.abs(audio_chunk))

            # Adjust settings based on audio characteristics
            adaptive_settings = self.current_settings.copy()

            # Adaptive noise reduction
            if adaptive_settings.get('adaptive_noise_reduction', True):
                if rms < 0.01:  # Very quiet audio
                    adaptive_settings['noise_factor'] = 0.05
                elif rms > 0.1:  # Loud audio
                    adaptive_settings['noise_factor'] = 0.15
                else:
                    adaptive_settings['noise_factor'] = 0.1

            # Adaptive compression
            if adaptive_settings.get('adaptive_compression', True):
                if peak > 0.9:  # Near clipping
                    adaptive_settings['threshold'] = -15.0
                    adaptive_settings['ratio'] = 6.0
                elif peak < 0.3:  # Low dynamic range
                    adaptive_settings['threshold'] = -30.0
                    adaptive_settings['ratio'] = 2.0
                else:
                    adaptive_settings['threshold'] = -25.0
                    adaptive_settings['ratio'] = 3.0

            # Apply enhancements with adaptive settings
            enhanced_chunk = audio_chunk.copy()

            # Apply noise reduction
            if adaptive_settings.get('noise_reduction', True):
                enhanced_chunk = self._apply_noise_reduction(enhanced_chunk, adaptive_settings['noise_factor'])

            # Apply normalization
            if adaptive_settings.get('normalization', True):
                enhanced_chunk = self._apply_normalization(enhanced_chunk, adaptive_settings['target_level'])

            # Apply compression
            if adaptive_settings.get('compression', True):
                enhanced_chunk = self._apply_compression(
                    enhanced_chunk,
                    adaptive_settings['threshold'],
                    adaptive_settings['ratio']
                )

            # Apply equalization
            if adaptive_settings.get('equalization', True):
                enhanced_chunk = self._apply_equalization(
                    enhanced_chunk,
                    adaptive_settings['low_gain'],
                    adaptive_settings['mid_gain'],
                    adaptive_settings['high_gain']
                )

            return enhanced_chunk

        except Exception as e:
            self.logger.warning(f"Adaptive enhancement failed: {str(e)}")
            return audio_chunk

    def _apply_noise_reduction(self, audio_chunk: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Apply noise reduction to audio chunk.

        Args:
            audio_chunk: Input audio chunk
            noise_factor: Noise reduction factor

        Returns:
            Noise-reduced audio chunk
        """
        try:
            if not has_audio_libs or len(audio_chunk) == 0:
                return audio_chunk

            # Simple noise reduction using spectral subtraction
            # For real-time processing, use a simpler approach
            if len(audio_chunk) < 64:
                return audio_chunk

            # Estimate noise floor from quieter sections
            noise_floor = np.percentile(np.abs(audio_chunk), 10)  # 10th percentile as noise floor

            # Apply spectral subtraction
            magnitude = np.abs(audio_chunk)
            phase = np.angle(audio_chunk) if np.iscomplexobj(audio_chunk) else 0

            # Subtract noise
            clean_magnitude = magnitude - noise_factor * noise_floor
            clean_magnitude = np.maximum(clean_magnitude, 0)

            # Reconstruct signal
            if np.iscomplexobj(audio_chunk):
                clean_audio = clean_magnitude * np.exp(1j * phase)
            else:
                clean_audio = clean_magnitude * np.sign(audio_chunk)

            return clean_audio

        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {str(e)}")
            return audio_chunk

    def _apply_normalization(self, audio_chunk: np.ndarray, target_level: float = -12.0) -> np.ndarray:
        """Apply normalization to audio chunk.

        Args:
            audio_chunk: Input audio chunk
            target_level: Target RMS level in dB

        Returns:
            Normalized audio chunk
        """
        try:
            if len(audio_chunk) == 0:
                return audio_chunk

            # Calculate current RMS
            current_rms = np.sqrt(np.mean(audio_chunk**2))

            if current_rms == 0:
                return audio_chunk

            # Convert target level to linear scale
            target_linear = 10**(target_level / 20.0)

            # Calculate normalization factor
            normalization_factor = target_linear / current_rms

            # Apply normalization with clipping protection
            normalized_audio = audio_chunk * normalization_factor
            normalized_audio = np.clip(normalized_audio, -0.95, 0.95)  # Leave some headroom

            return normalized_audio

        except Exception as e:
            self.logger.warning(f"Normalization failed: {str(e)}")
            return audio_chunk

    def _apply_compression(self, audio_chunk: np.ndarray, threshold: float = -25.0,
                          ratio: float = 3.0) -> np.ndarray:
        """Apply compression to audio chunk.

        Args:
            audio_chunk: Input audio chunk
            threshold: Compression threshold in dB
            ratio: Compression ratio

        Returns:
            Compressed audio chunk
        """
        try:
            if len(audio_chunk) == 0:
                return audio_chunk

            # Convert threshold to linear scale
            threshold_linear = 10**(threshold / 20.0)

            # Apply compression
            compressed = np.copy(audio_chunk)

            # Find samples above threshold
            over_threshold = np.abs(audio_chunk) > threshold_linear

            if np.any(over_threshold):
                # Apply compression
                compressed[over_threshold] = np.sign(audio_chunk[over_threshold]) * (
                    threshold_linear + (np.abs(audio_chunk[over_threshold]) - threshold_linear) / ratio
                )

            return compressed

        except Exception as e:
            self.logger.warning(f"Compression failed: {str(e)}")
            return audio_chunk

    def _apply_equalization(self, audio_chunk: np.ndarray, low_gain: float = 1.0,
                           mid_gain: float = 0.5, high_gain: float = 0.0) -> np.ndarray:
        """Apply equalization to audio chunk.

        Args:
            audio_chunk: Input audio chunk
            low_gain: Low frequency gain in dB
            mid_gain: Mid frequency gain in dB
            high_gain: High frequency gain in dB

        Returns:
            Equalized audio chunk
        """
        try:
            if not has_audio_libs or len(audio_chunk) == 0:
                return audio_chunk

            # Simple frequency-based equalization
            # For real-time processing, use a simple FIR filter approach
            if len(audio_chunk) < 32:
                return audio_chunk

            # Apply gains based on frequency content
            # This is a simplified approach - in practice, you'd use proper EQ filters
            equalized = audio_chunk.copy()

            # Simple low-frequency boost/cut (approximate)
            if low_gain != 0.0:
                # Apply low-pass filter effect
                if low_gain > 0:
                    # Boost low frequencies
                    equalized = equalized * (1.0 + low_gain * 0.3)
                else:
                    # Cut low frequencies
                    equalized = equalized * (1.0 + low_gain * 0.1)

            # Simple high-frequency boost/cut
            if high_gain != 0.0:
                # Apply high-pass filter effect
                if high_gain > 0:
                    # Boost high frequencies
                    equalized = equalized * (1.0 + high_gain * 0.2)
                else:
                    # Cut high frequencies
                    equalized = equalized * (1.0 + high_gain * 0.1)

            return equalized

        except Exception as e:
            self.logger.warning(f"Equalization failed: {str(e)}")
            return audio_chunk

    def _update_adaptive_settings(self, audio_chunk: np.ndarray):
        """Update adaptive settings based on audio characteristics.

        Args:
            audio_chunk: Recent audio chunk for analysis
        """
        try:
            if not self.adaptive_enabled or len(audio_chunk) == 0:
                return

            # Analyze audio characteristics
            rms = np.sqrt(np.mean(audio_chunk**2))
            peak = np.max(np.abs(audio_chunk))
            dynamic_range = 20 * np.log10(peak / rms) if rms > 0 else 0

            # Update quality score
            if self.quality_monitoring_enabled:
                # Simple quality estimation
                quality_score = 7.0  # Base score

                if rms < 0.01:
                    quality_score -= 2.0  # Very quiet
                elif rms > 0.3:
                    quality_score -= 1.0  # Very loud

                if peak > 0.95:
                    quality_score -= 2.0  # Clipping

                if dynamic_range < 20:
                    quality_score -= 1.0  # Low dynamic range

                self.current_quality_score = max(0.0, min(10.0, quality_score))

            # Adjust settings based on quality
            if self.auto_adjust_enabled and self.current_quality_score < self.current_settings.get('quality_threshold', 7.0):
                # Quality is below threshold, adjust settings
                if self.current_quality_score < 5.0:
                    # Poor quality - increase noise reduction
                    self.current_settings['noise_factor'] = min(0.2, self.current_settings['noise_factor'] + 0.02)
                elif self.current_quality_score < 7.0:
                    # Moderate quality - slight adjustments
                    self.current_settings['noise_factor'] = min(0.15, self.current_settings['noise_factor'] + 0.01)

        except Exception as e:
            self.logger.warning(f"Adaptive settings update failed: {str(e)}")

    def _monitor_quality(self, processed_chunk: np.ndarray, original_chunk: np.ndarray):
        """Monitor audio quality in real-time.

        Args:
            processed_chunk: Processed audio chunk
            original_chunk: Original audio chunk
        """
        try:
            if not self.quality_monitoring_enabled or len(processed_chunk) == 0:
                return

            # Calculate quality metrics
            processed_rms = np.sqrt(np.mean(processed_chunk**2))
            original_rms = np.sqrt(np.mean(original_chunk**2))
            processed_peak = np.max(np.abs(processed_chunk))
            original_peak = np.max(np.abs(original_chunk))

            # Simple quality indicators
            rms_stability = 1.0 - min(abs(processed_rms - original_rms) / max(original_rms, 0.001), 1.0)
            peak_control = 1.0 if processed_peak <= 0.95 else max(0.0, 1.0 - (processed_peak - 0.95) * 5.0)

            # Combined quality score
            quality_score = (rms_stability * 0.6 + peak_control * 0.4) * 10.0
            self.quality_history.append(quality_score)

        except Exception as e:
            self.logger.warning(f"Quality monitoring failed: {str(e)}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        try:
            avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0.0
            avg_quality = statistics.mean(self.quality_history) if self.quality_history else 0.0

            return {
                'is_processing': self.is_processing,
                'enhancement_type': self.current_enhancement_type,
                'avg_processing_time': avg_processing_time,
                'current_latency': avg_processing_time,
                'avg_quality_score': avg_quality,
                'quality_history_size': len(self.quality_history),
                'processing_times_size': len(self.processing_times),
                'buffer_size': self.buffer_size,
                'chunk_size': self.chunk_size,
                'adaptive_enabled': self.adaptive_enabled,
                'quality_monitoring_enabled': self.quality_monitoring_enabled,
                'auto_adjust_enabled': self.auto_adjust_enabled
            }

        except Exception as e:
            self.logger.error(f"Error getting processing stats: {str(e)}")
            return {
                'is_processing': self.is_processing,
                'error': str(e)
            }

    def update_enhancement_settings(self, enhancement_type: str, settings: Dict[str, Any]) -> bool:
        """Update enhancement settings in real-time.

        Args:
            enhancement_type: New enhancement type
            settings: New settings

        Returns:
            True if updated successfully
        """
        try:
            self.current_enhancement_type = enhancement_type
            self.current_settings.update(settings)

            # Validate settings
            if enhancement_type in ['noise_reduction', 'normalization', 'compression', 'equalization', 'adaptive']:
                validated_settings = audio_enhancer.validate_enhancement_settings(enhancement_type, settings)
                self.current_settings.update(validated_settings)

            self.logger.info(f"Enhancement settings updated to: {enhancement_type}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating enhancement settings: {str(e)}")
            return False

    def reset_adaptive_settings(self) -> bool:
        """Reset adaptive settings to defaults.

        Returns:
            True if reset successfully
        """
        try:
            self.current_settings = self._get_default_adaptive_settings()
            self.quality_history.clear()
            self.processing_times.clear()
            self.current_quality_score = 0.0

            self.logger.info("Adaptive settings reset to defaults")
            return True

        except Exception as e:
            self.logger.error(f"Error resetting adaptive settings: {str(e)}")
            return False

    async def process_stream_async(self, audio_stream: Any, output_callback: Callable = None) -> bool:
        """Process audio stream asynchronously.

        Args:
            audio_stream: Async audio stream generator
            output_callback: Optional callback for processed audio

        Returns:
            True if processing completed successfully
        """
        try:
            async for audio_chunk in audio_stream:
                if not self.is_processing:
                    break

                # Process chunk
                processed_chunk = self.process_audio_chunk(audio_chunk)

                # Call output callback if provided
                if output_callback and processed_chunk is not None:
                    await output_callback(processed_chunk)

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)

            return True

        except Exception as e:
            self.logger.error(f"Error in async stream processing: {str(e)}")
            return False


# Global real-time enhancer instance
realtime_enhancer = RealTimeEnhancer()