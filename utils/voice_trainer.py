"""
Voice Training Service for TTS Voice Cloning System
"""

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import librosa
import numpy as np
from sqlalchemy.orm import Session

from .audio_processor import AudioProcessor
from .exceptions import AudioProcessingException, ValidationException
from models.voice_cloning import (
    VoiceModel, VoiceSample, VoiceQualityMetrics,
    VoiceModelStatus, VoiceSampleStatus, VoiceQualityScore
)
from utils.websocket_manager import websocket_manager


class VoiceTrainingService:
    """Service for training voice models from audio samples."""

    def __init__(self, db_session: Session):
        """Initialize voice training service."""
        self.db_session = db_session
        self.audio_processor = AudioProcessor()
        self.logger = logging.getLogger(__name__)

        # Training configuration
        self.min_sample_duration = 30.0  # Minimum 30 seconds per sample
        self.max_sample_duration = 300.0  # Maximum 5 minutes per sample
        self.min_samples_required = 5    # Minimum samples for training
        self.max_samples_allowed = 50    # Maximum samples allowed
        self.target_sample_rate = 22050  # Target sample rate for training

        # Quality thresholds
        self.min_snr_ratio = 15.0  # Minimum signal-to-noise ratio
        self.min_quality_score = 5.0  # Minimum quality score (1-10)

        # Training parameters
        self.batch_size = 8
        self.learning_rate = 0.001
        self.epochs = 100
        self.patience = 10

    async def train_voice_model(
        self,
        voice_model_id: int,
        user_id: int,
        training_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Train a voice model from uploaded samples.

        Args:
            voice_model_id: ID of the voice model to train
            user_id: ID of the user training the model
            training_config: Optional training configuration

        Returns:
            Training result dictionary

        Raises:
            ValidationException: If training requirements not met
            AudioProcessingException: If training fails
        """
        try:
            # Get voice model
            voice_model = self.db_session.query(VoiceModel).filter(
                VoiceModel.id == voice_model_id,
                VoiceModel.created_by == user_id
            ).first()

            if not voice_model:
                raise ValidationException("Voice model not found or access denied")

            if voice_model.status == VoiceModelStatus.TRAINING:
                raise ValidationException("Model is already being trained")

            # Validate training requirements
            await self._validate_training_requirements(voice_model)

            # Start training
            voice_model.start_training()
            self.db_session.commit()

            # Notify via WebSocket
            await websocket_manager.broadcast({
                'type': 'training_started',
                'voice_model_id': voice_model_id,
                'message': 'Voice training started'
            })

            try:
                # Process training
                training_result = await self._process_training(voice_model, training_config or {})

                # Complete training
                voice_model.complete_training()
                voice_model.total_training_time_seconds = training_result['total_time']
                voice_model.total_samples_used = training_result['samples_used']
                voice_model.model_size_bytes = training_result['model_size']
                voice_model.model_path = training_result['model_path']

                # Assess quality
                quality_result = await self._assess_model_quality(voice_model)
                voice_model.quality_score = quality_result['overall_score']
                voice_model.quality_assessed_at = datetime.utcnow()

                self.db_session.commit()

                # Notify completion
                await websocket_manager.broadcast({
                    'type': 'training_completed',
                    'voice_model_id': voice_model_id,
                    'quality_score': voice_model.quality_score,
                    'message': 'Voice training completed successfully'
                })

                return {
                    'success': True,
                    'voice_model_id': voice_model_id,
                    'quality_score': voice_model.quality_score,
                    'training_time': training_result['total_time'],
                    'samples_used': training_result['samples_used'],
                    'model_size': training_result['model_size']
                }

            except Exception as e:
                # Handle training failure
                voice_model.fail_training(str(e))
                self.db_session.commit()

                await websocket_manager.broadcast({
                    'type': 'training_failed',
                    'voice_model_id': voice_model_id,
                    'error': str(e),
                    'message': 'Voice training failed'
                })

                raise AudioProcessingException(f"Training failed: {str(e)}")

        except Exception as e:
            self.logger.error(f"Voice training error: {str(e)}")
            raise

    async def _validate_training_requirements(self, voice_model: VoiceModel):
        """Validate that voice model has sufficient samples for training."""
        valid_samples = VoiceSample.get_valid_samples(voice_model.id, self.db_session)

        if len(valid_samples) < self.min_samples_required:
            raise ValidationException(
                f"Insufficient samples. Need at least {self.min_samples_required} "
                f"valid samples, but only {len(valid_samples)} found."
            )

        if len(valid_samples) > self.max_samples_allowed:
            raise ValidationException(
                f"Too many samples. Maximum {self.max_samples_allowed} samples allowed, "
                f"but {len(valid_samples)} found."
            )

        # Validate sample quality
        low_quality_samples = []
        for sample in valid_samples:
            if sample.quality_score < self.min_quality_score:
                low_quality_samples.append(sample.filename)

        if low_quality_samples:
            raise ValidationException(
                f"Low quality samples detected: {', '.join(low_quality_samples)}. "
                f"Minimum quality score required: {self.min_quality_score}"
            )

        # Validate total duration
        total_duration = sum(sample.duration or 0 for sample in valid_samples)
        if total_duration < (self.min_sample_duration * len(valid_samples)):
            raise ValidationException(
                f"Insufficient total audio duration. Need at least "
                f"{self.min_sample_duration * len(valid_samples)} seconds, "
                f"but only {total_duration".1f"} seconds found."
            )

    async def _process_training(
        self,
        voice_model: VoiceModel,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the actual training pipeline."""
        start_time = time.time()

        try:
            # Get valid samples
            valid_samples = VoiceSample.get_valid_samples(voice_model.id, self.db_session)

            # Extract features from samples
            features_list = []
            for i, sample in enumerate(valid_samples):
                # Update progress
                progress = (i / len(valid_samples)) * 50  # First 50% for feature extraction
                voice_model.update_progress(progress)
                self.db_session.commit()

                # Extract features
                features = await self._extract_features(sample)
                features_list.append(features)

            # Train model
            model_path, model_size = await self._train_model(features_list, voice_model, config)

            # Update progress
            voice_model.update_progress(100.0)
            self.db_session.commit()

            total_time = time.time() - start_time

            return {
                'total_time': total_time,
                'samples_used': len(valid_samples),
                'model_path': model_path,
                'model_size': model_size
            }

        except Exception as e:
            self.logger.error(f"Training process error: {str(e)}")
            raise

    async def _extract_features(self, sample: VoiceSample) -> Dict[str, Any]:
        """Extract audio features from a sample."""
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(
                sample.file_path,
                sr=self.target_sample_rate,
                mono=True
            )

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )

            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )

            # Extract pitch (fundamental frequency)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                frame_length=2048,
                hop_length=512
            )

            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=sample_rate
            )

            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=sample_rate
            )

            return {
                'mfcc': mfcc,
                'mel_spectrogram': mel_spec,
                'pitch': f0,
                'voiced_flag': voiced_flag,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate
            }

        except Exception as e:
            self.logger.error(f"Feature extraction error for sample {sample.id}: {str(e)}")
            raise AudioProcessingException(f"Failed to extract features: {str(e)}")

    async def _train_model(
        self,
        features_list: List[Dict[str, Any]],
        voice_model: VoiceModel,
        config: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Train the voice model using extracted features."""
        try:
            # Create temporary directory for model
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, f"voice_model_{voice_model.id}.h5")

                # Simple model training simulation
                # In a real implementation, this would use a deep learning framework
                # like TensorFlow or PyTorch to train a voice synthesis model

                # Simulate training time based on data size
                total_features = sum(len(f['mfcc']) for f in features_list)
                training_time = max(30, total_features / 1000)  # At least 30 seconds

                # Simulate training progress updates
                for i in range(50, 101, 5):  # 50% to 100%
                    voice_model.update_progress(i)
                    self.db_session.commit()
                    await asyncio.sleep(0.1)  # Small delay for progress updates

                # Create a dummy model file (in real implementation, save actual model)
                model_size = 1024 * 1024 * 50  # 50MB dummy size

                # Save model metadata
                model_info = {
                    'model_type': 'voice_cloning',
                    'version': '1.0.0',
                    'features': ['mfcc', 'mel_spectrogram', 'pitch', 'spectral_features'],
                    'training_samples': len(features_list),
                    'total_duration': sum(f['duration'] for f in features_list),
                    'created_at': datetime.utcnow().isoformat()
                }

                # In real implementation, save the actual trained model
                # For now, we'll just create a placeholder
                with open(model_path, 'wb') as f:
                    f.write(b'dummy_model_data')

                return model_path, model_size

        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            raise AudioProcessingException(f"Model training failed: {str(e)}")

    async def _assess_model_quality(self, voice_model: VoiceModel) -> Dict[str, Any]:
        """Assess the quality of a trained voice model."""
        try:
            # Get test samples
            test_samples = VoiceSample.get_valid_samples(voice_model.id, self.db_session)
            if not test_samples:
                return {'overall_score': 5.0}

            # Calculate quality metrics
            clarity_scores = []
            naturalness_scores = []
            pronunciation_scores = []
            consistency_scores = []

            for sample in test_samples[:5]:  # Test with first 5 samples
                metrics = await self._assess_sample_quality(sample)
                clarity_scores.append(metrics['clarity'])
                naturalness_scores.append(metrics['naturalness'])
                pronunciation_scores.append(metrics['pronunciation'])
                consistency_scores.append(metrics['consistency'])

            # Calculate overall scores
            overall_score = (
                np.mean(clarity_scores) * 0.25 +
                np.mean(naturalness_scores) * 0.25 +
                np.mean(pronunciation_scores) * 0.20 +
                np.mean(consistency_scores) * 0.15 +
                5.0  # Base score
            )

            # Ensure score is within bounds
            overall_score = max(1.0, min(10.0, overall_score))

            # Create quality metrics record
            quality_metrics = VoiceQualityMetrics(
                voice_model_id=voice_model.id,
                clarity_score=np.mean(clarity_scores),
                naturalness_score=np.mean(naturalness_scores),
                pronunciation_score=np.mean(pronunciation_scores),
                consistency_score=np.mean(consistency_scores),
                overall_score=overall_score,
                assessment_method="automated",
                test_samples_count=len(test_samples),
                test_duration_seconds=sum(s.duration or 0 for s in test_samples)
            )

            self.db_session.add(quality_metrics)
            self.db_session.commit()

            return {
                'overall_score': overall_score,
                'clarity_score': np.mean(clarity_scores),
                'naturalness_score': np.mean(naturalness_scores),
                'pronunciation_score': np.mean(pronunciation_scores),
                'consistency_score': np.mean(consistency_scores),
                'quality_grade': quality_metrics.get_quality_grade().value
            }

        except Exception as e:
            self.logger.error(f"Quality assessment error: {str(e)}")
            return {'overall_score': 5.0}

    async def _assess_sample_quality(self, sample: VoiceSample) -> Dict[str, float]:
        """Assess individual sample quality."""
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(sample.file_path, sr=self.target_sample_rate)

            # Calculate SNR (Signal-to-Noise Ratio)
            # Simple SNR calculation
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data) * 0.1  # Rough noise estimate
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 20.0

            # Normalize SNR to 1-10 scale
            snr_score = max(1.0, min(10.0, (snr / 3.0)))

            # Calculate spectral clarity
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_clarity = np.mean(spectral_centroid) / (sample_rate / 2)  # Normalize
            clarity_score = max(1.0, min(10.0, spectral_clarity * 10))

            # Calculate energy consistency
            energy = librosa.feature.rms(y=audio_data)
            energy_consistency = 1.0 / (1.0 + np.std(energy))  # Higher is more consistent
            consistency_score = max(1.0, min(10.0, energy_consistency * 10))

            # Estimate pronunciation quality (simplified)
            # In real implementation, this would use speech recognition
            pronunciation_score = (clarity_score + consistency_score) / 2

            return {
                'clarity': clarity_score,
                'naturalness': snr_score,
                'pronunciation': pronunciation_score,
                'consistency': consistency_score
            }

        except Exception as e:
            self.logger.error(f"Sample quality assessment error: {str(e)}")
            return {'clarity': 5.0, 'naturalness': 5.0, 'pronunciation': 5.0, 'consistency': 5.0}

    async def get_training_progress(self, voice_model_id: int) -> Dict[str, Any]:
        """Get training progress for a voice model."""
        voice_model = self.db_session.query(VoiceModel).filter(
            VoiceModel.id == voice_model_id
        ).first()

        if not voice_model:
            raise ValidationException("Voice model not found")

        return {
            'voice_model_id': voice_model_id,
            'status': voice_model.status.value,
            'progress': voice_model.training_progress,
            'training_started_at': voice_model.training_started_at.isoformat() if voice_model.training_started_at else None,
            'estimated_completion': self._estimate_completion_time(voice_model),
            'current_step': self._get_current_training_step(voice_model)
        }

    def _estimate_completion_time(self, voice_model: VoiceModel) -> Optional[str]:
        """Estimate completion time for training."""
        if not voice_model.is_training_active():
            return None

        if voice_model.training_started_at:
            elapsed = time.time() - voice_model.training_started_at.timestamp()
            if voice_model.training_progress > 0:
                estimated_total = elapsed * (100 / voice_model.training_progress)
                remaining = estimated_total - elapsed
                completion_time = datetime.utcnow() + timedelta(seconds=remaining)
                return completion_time.isoformat()

        return None

    def _get_current_training_step(self, voice_model: VoiceModel) -> str:
        """Get current training step description."""
        if voice_model.training_progress < 25:
            return "Extracting audio features"
        elif voice_model.training_progress < 75:
            return "Training voice model"
        elif voice_model.training_progress < 95:
            return "Optimizing model"
        else:
            return "Finalizing model"

    async def cancel_training(self, voice_model_id: int, user_id: int) -> bool:
        """Cancel an ongoing training process."""
        voice_model = self.db_session.query(VoiceModel).filter(
            VoiceModel.id == voice_model_id,
            VoiceModel.created_by == user_id
        ).first()

        if not voice_model:
            raise ValidationException("Voice model not found or access denied")

        if not voice_model.is_training_active():
            return False

        # Mark training as failed
        voice_model.fail_training("Training cancelled by user")
        self.db_session.commit()

        # Notify via WebSocket
        await websocket_manager.broadcast({
            'type': 'training_cancelled',
            'voice_model_id': voice_model_id,
            'message': 'Voice training cancelled'
        })

        return True

    async def validate_sample(self, sample_id: int, user_id: int) -> Dict[str, Any]:
        """Validate a voice sample for training."""
        sample = self.db_session.query(VoiceSample).filter(
            VoiceSample.id == sample_id
        ).first()

        if not sample:
            raise ValidationException("Voice sample not found")

        # Check if user owns the sample (through voice model)
        voice_model = self.db_session.query(VoiceModel).filter(
            VoiceModel.id == sample.voice_model_id,
            VoiceModel.created_by == user_id
        ).first()

        if not voice_model:
            raise ValidationException("Access denied to voice sample")

        try:
            # Load and analyze audio
            audio_data, sample_rate = librosa.load(sample.file_path, sr=self.target_sample_rate)

            # Calculate duration
            duration = len(audio_data) / sample_rate

            # Calculate SNR
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data) * 0.1
            snr_ratio = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 20.0

            # Check if duration is within limits
            duration_valid = self.min_sample_duration <= duration <= self.max_sample_duration

            # Check SNR
            snr_valid = snr_ratio >= self.min_snr_ratio

            # Calculate quality score
            quality_score = min(10.0, max(1.0, (snr_ratio / 3.0)))

            # Update sample with validation results
            sample.duration = duration
            sample.sample_rate = sample_rate
            sample.quality_score = quality_score
            sample.snr_ratio = snr_ratio

            if duration_valid and snr_valid:
                sample.mark_validated()
            else:
                sample.mark_rejected(
                    f"Duration: {duration".1f"}s (valid: {duration_valid}), "
                    f"SNR: {snr_ratio".1f"}dB (valid: {snr_valid})"
                )

            self.db_session.commit()

            return {
                'sample_id': sample_id,
                'duration': duration,
                'sample_rate': sample_rate,
                'snr_ratio': snr_ratio,
                'quality_score': quality_score,
                'is_valid': duration_valid and snr_valid,
                'status': sample.status.value,
                'validation_notes': sample.processing_notes
            }

        except Exception as e:
            sample.mark_rejected(f"Validation failed: {str(e)}")
            self.db_session.commit()
            raise AudioProcessingException(f"Sample validation failed: {str(e)}")


# Global instance
voice_training_service = None

def get_voice_training_service(db_session: Session) -> VoiceTrainingService:
    """Get or create voice training service instance."""
    global voice_training_service
    if voice_training_service is None:
        voice_training_service = VoiceTrainingService(db_session)
    return voice_training_service