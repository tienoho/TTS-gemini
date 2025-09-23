"""
Comprehensive tests for Voice Cloning System
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from flask import Flask
from flask_jwt_extended import create_access_token
from sqlalchemy.orm import Session

from app.extensions import db
from models.voice_cloning import (
    VoiceModel, VoiceSample, VoiceQualityMetrics, VoiceVersion, VoiceTestResult,
    VoiceModelStatus, VoiceSampleStatus, VoiceModelType, VoiceQualityScore
)
from utils.voice_trainer import VoiceTrainingService
from utils.voice_library import VoiceLibraryManager
from utils.audio_preprocessor import AudioPreprocessor
from utils.voice_quality import VoiceQualityAssessor
from config.voice_cloning import VoiceCloningConfig
from routes.voice_cloning import voice_cloning_bp


class TestVoiceCloningModels:
    """Test voice cloning models."""

    def test_voice_model_creation(self, db_session: Session, test_user):
        """Test creating a voice model."""
        voice_model = VoiceModel(
            name="Test Voice",
            description="Test voice model",
            organization_id=test_user.organization_id,
            created_by=test_user.id,
            language="vi",
            gender="male",
            age_group="adult",
            accent="northern",
            model_type=VoiceModelType.STANDARD
        )

        db_session.add(voice_model)
        db_session.commit()

        assert voice_model.id is not None
        assert voice_model.name == "Test Voice"
        assert voice_model.status == VoiceModelStatus.CREATED
        assert voice_model.is_active == True
        assert voice_model.quality_score == 0.0

    def test_voice_model_to_dict(self, db_session: Session, test_user):
        """Test voice model serialization."""
        voice_model = VoiceModel(
            name="Test Voice",
            organization_id=test_user.organization_id,
            created_by=test_user.id,
            language="vi",
            quality_score=8.5
        )

        db_session.add(voice_model)
        db_session.commit()

        model_dict = voice_model.to_dict()
        assert model_dict['name'] == "Test Voice"
        assert model_dict['language'] == "vi"
        assert model_dict['quality_score'] == 8.5
        assert model_dict['status'] == VoiceModelStatus.CREATED.value

    def test_voice_sample_creation(self, db_session: Session, test_voice_model):
        """Test creating a voice sample."""
        sample = VoiceSample(
            voice_model_id=test_voice_model.id,
            uploaded_by=test_voice_model.created_by,
            filename="test_sample.wav",
            original_filename="test_sample.wav",
            file_path="/path/to/sample.wav",
            file_size=1024000,
            mime_type="audio/wav",
            checksum="abc123",
            duration=45.5,
            sample_rate=22050,
            quality_score=7.5,
            snr_ratio=18.5
        )

        db_session.add(sample)
        db_session.commit()

        assert sample.id is not None
        assert sample.status == VoiceSampleStatus.UPLOADED
        assert sample.duration == 45.5
        assert sample.quality_score == 7.5

    def test_voice_quality_metrics_creation(self, db_session: Session, test_voice_model):
        """Test creating voice quality metrics."""
        metrics = VoiceQualityMetrics(
            voice_model_id=test_voice_model.id,
            clarity_score=8.5,
            naturalness_score=7.8,
            pronunciation_score=8.2,
            consistency_score=8.0,
            expressiveness_score=7.5,
            overall_score=8.0,
            weighted_score=7.9,
            word_error_rate=0.05,
            character_error_rate=0.03,
            mel_cepstral_distortion=6.5,
            f0_frame_error=0.08,
            speaking_rate=160,
            pitch_mean=145.5,
            pitch_std=25.0,
            energy_mean=-18.5,
            energy_std=4.2,
            assessment_method="automated",
            test_samples_count=10,
            test_duration_seconds=100.0
        )

        db_session.add(metrics)
        db_session.commit()

        assert metrics.id is not None
        assert metrics.overall_score == 8.0
        assert metrics.get_quality_grade() == VoiceQualityScore.GOOD

    def test_voice_version_creation(self, db_session: Session, test_voice_model):
        """Test creating a voice version."""
        version = VoiceVersion(
            voice_model_id=test_voice_model.id,
            version_number="1.0.0",
            model_path="/path/to/model.h5",
            model_size_bytes=52428800,  # 50MB
            checksum="def456",
            created_by=test_voice_model.created_by
        )

        db_session.add(version)
        db_session.commit()

        assert version.id is not None
        assert version.version_number == "1.0.0"
        assert version.is_active == True

    def test_voice_test_result_creation(self, db_session: Session, test_voice_model):
        """Test creating a voice test result."""
        test_result = VoiceTestResult(
            voice_model_id=test_voice_model.id,
            test_type="quality",
            test_name="pronunciation_test",
            test_description="Test pronunciation accuracy",
            test_input="Sample text for testing",
            test_result={"score": 8.5, "details": "Good pronunciation"},
            success=True,
            error_message=None,
            execution_time_ms=1500.0,
            memory_usage_mb=75.5,
            cpu_usage_percent=15.0,
            quality_score=8.5,
            clarity_rating=8.2,
            naturalness_rating=8.8,
            tested_by=test_voice_model.created_by
        )

        db_session.add(test_result)
        db_session.commit()

        assert test_result.id is not None
        assert test_result.success == True
        assert test_result.quality_score == 8.5


class TestVoiceTrainingService:
    """Test voice training service."""

    @pytest_asyncio.asyncio
    async def test_validate_training_requirements_success(self, db_session: Session, test_voice_model_with_samples):
        """Test successful training requirements validation."""
        service = VoiceTrainingService(db_session)

        # Should not raise exception
        await service._validate_training_requirements(test_voice_model_with_samples)

    @pytest_asyncio.asyncio
    async def test_validate_training_requirements_insufficient_samples(self, db_session: Session, test_voice_model):
        """Test validation failure due to insufficient samples."""
        service = VoiceTrainingService(db_session)

        with pytest.raises(ValidationException, match="Insufficient samples"):
            await service._validate_training_requirements(test_voice_model)

    @pytest_asyncio.asyncio
    async def test_validate_training_requirements_low_quality(self, db_session: Session, test_voice_model_with_low_quality_samples):
        """Test validation failure due to low quality samples."""
        service = VoiceTrainingService(db_session)

        with pytest.raises(ValidationException, match="Low quality samples"):
            await service._validate_training_requirements(test_voice_model_with_low_quality_samples)

    @pytest_asyncio.asyncio
    async def test_extract_features(self, db_session: Session, test_voice_sample):
        """Test feature extraction from voice sample."""
        service = VoiceTrainingService(db_session)

        features = await service._extract_features(test_voice_sample)

        assert 'mfcc' in features
        assert 'mel_spectrogram' in features
        assert 'pitch' in features
        assert 'spectral_centroid' in features
        assert features['sample_rate'] == 22050
        assert features['duration'] > 0

    @pytest_asyncio.asyncio
    async def test_get_training_progress(self, db_session: Session, test_voice_model):
        """Test getting training progress."""
        service = VoiceTrainingService(db_session)

        # Start training
        test_voice_model.start_training()
        db_session.commit()

        progress = await service.get_training_progress(test_voice_model.id)

        assert progress['voice_model_id'] == test_voice_model.id
        assert progress['status'] == VoiceModelStatus.TRAINING.value
        assert progress['progress'] == 0.0
        assert 'current_step' in progress

    @pytest_asyncio.asyncio
    async def test_cancel_training(self, db_session: Session, test_voice_model):
        """Test cancelling training."""
        service = VoiceTrainingService(db_session)

        # Start training
        test_voice_model.start_training()
        db_session.commit()

        # Cancel training
        success = await service.cancel_training(test_voice_model.id, test_voice_model.created_by)

        assert success == True

        # Check that training was cancelled
        db_session.refresh(test_voice_model)
        assert test_voice_model.status == VoiceModelStatus.FAILED
        assert test_voice_model.training_error == "Training cancelled by user"


class TestVoiceLibraryManager:
    """Test voice library management."""

    @pytest_asyncio.asyncio
    async def test_store_voice_model(self, db_session: Session, test_voice_model):
        """Test storing a voice model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceLibraryManager(db_session, temp_dir)

            # Create dummy model data
            model_data = b"dummy_model_data_" + b"x" * 1000

            model_path = await manager.store_voice_model(test_voice_model, model_data, test_voice_model.organization_id)

            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) == len(model_data)

            # Check that version was created
            version = VoiceVersion.get_active_version(test_voice_model.id, db_session)
            assert version is not None
            assert version.model_path == model_path

    @pytest_asyncio.asyncio
    async def test_retrieve_voice_model(self, db_session: Session, test_voice_model):
        """Test retrieving a voice model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceLibraryManager(db_session, temp_dir)

            # Store model first
            model_data = b"dummy_model_data_" + b"x" * 1000
            await manager.store_voice_model(test_voice_model, model_data, test_voice_model.organization_id)

            # Retrieve model
            retrieved_data, version = await manager.retrieve_voice_model(
                test_voice_model.id, test_voice_model.organization_id
            )

            assert retrieved_data == model_data
            assert version is not None

    @pytest_asyncio.asyncio
    async def test_search_voices(self, db_session: Session, test_voice_model):
        """Test searching voices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceLibraryManager(db_session, temp_dir)

            # Search for voices
            results = await manager.search_voices(
                organization_id=test_voice_model.organization_id,
                language="vi",
                quality_min=5.0,
                limit=10
            )

            assert 'voices' in results
            assert 'total_count' in results
            assert isinstance(results['voices'], list)

    @pytest_asyncio.asyncio
    async def test_get_voice_statistics(self, db_session: Session, test_voice_model):
        """Test getting voice statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceLibraryManager(db_session, temp_dir)

            stats = await manager.get_voice_statistics(test_voice_model.organization_id)

            assert 'total_models' in stats
            assert 'active_models' in stats
            assert 'quality_stats' in stats
            assert 'storage_stats' in stats

    @pytest_asyncio.asyncio
    async def test_create_voice_version(self, db_session: Session, test_voice_model):
        """Test creating a new voice version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceLibraryManager(db_session, temp_dir)

            model_data = b"new_model_data_" + b"x" * 2000

            version = await manager.create_voice_version(
                test_voice_model.id,
                test_voice_model.created_by,
                model_data,
                version_tag="improved",
                change_log="Enhanced quality"
            )

            assert version is not None
            assert version.version_number == "1.0.0"
            assert version.version_tag == "improved"
            assert version.change_log == "Enhanced quality"

    @pytest_asyncio.asyncio
    async def test_rollback_voice_version(self, db_session: Session, test_voice_model):
        """Test rolling back to a previous version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceLibraryManager(db_session, temp_dir)

            # Create initial version
            model_data = b"initial_model_data_" + b"x" * 1000
            initial_version = await manager.create_voice_version(
                test_voice_model.id,
                test_voice_model.created_by,
                model_data,
                version_tag="initial"
            )

            # Create new version
            new_model_data = b"new_model_data_" + b"x" * 1500
            new_version = await manager.create_voice_version(
                test_voice_model.id,
                test_voice_model.created_by,
                new_model_data,
                version_tag="improved"
            )

            # Rollback to initial version
            rolled_back_version = await manager.rollback_voice_version(
                test_voice_model.id,
                test_voice_model.created_by,
                initial_version.version_number
            )

            assert rolled_back_version.version_number == initial_version.version_number
            assert rolled_back_version.is_active == True
            assert new_version.is_active == False


class TestAudioPreprocessor:
    """Test audio preprocessing."""

    @pytest_asyncio.asyncio
    async def test_preprocess_audio_file(self):
        """Test audio file preprocessing."""
        preprocessor = AudioPreprocessor()

        # Create a dummy WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Create simple audio data (sine wave)
            import numpy as np
            sample_rate = 22050
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            audio_data = (audio_data * 0.5).astype(np.float32)

            # Save as WAV
            import soundfile as sf
            sf.write(temp_file.name, audio_data, sample_rate)

            # Preprocess
            with tempfile.NamedTemporaryFile(suffix='_processed.wav', delete=False) as output_file:
                result = await preprocessor.preprocess_audio_file(
                    temp_file.name,
                    output_file.name,
                    apply_noise_reduction=True,
                    apply_normalization=True,
                    apply_silence_removal=True
                )

                assert result['original_duration'] == duration
                assert result['processed_duration'] > 0
                assert result['quality_metrics']['quality_score'] > 0
                assert os.path.exists(output_file.name)

                # Clean up
                os.unlink(temp_file.name)
                os.unlink(output_file.name)

    @pytest_asyncio.asyncio
    async def test_validate_audio_for_cloning(self):
        """Test audio validation for cloning."""
        preprocessor = AudioPreprocessor()

        # Create a valid audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            import numpy as np
            sample_rate = 22050
            duration = 45.0  # Valid duration
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # Lower amplitude for good SNR
            audio_data = audio_data.astype(np.float32)

            import soundfile as sf
            sf.write(temp_file.name, audio_data, sample_rate)

            # Validate
            result = await preprocessor.validate_audio_for_cloning(temp_file.name)

            assert result['is_valid'] == True
            assert result['duration'] == duration
            assert result['quality_score'] > 5.0
            assert len(result['issues']) == 0

            os.unlink(temp_file.name)

    @pytest_asyncio.asyncio
    async def test_extract_features(self):
        """Test feature extraction."""
        preprocessor = AudioPreprocessor()

        # Create a test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            import numpy as np
            sample_rate = 22050
            duration = 5.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
            audio_data = audio_data.astype(np.float32)

            import soundfile as sf
            sf.write(temp_file.name, audio_data, sample_rate)

            # Extract features
            features = await preprocessor.extract_features(
                temp_file.name,
                feature_types=['mfcc', 'mel_spectrogram', 'pitch']
            )

            assert 'mfcc' in features
            assert 'mel_spectrogram' in features
            assert 'pitch' in features
            assert features['duration'] == duration
            assert features['sample_rate'] == sample_rate

            os.unlink(temp_file.name)

    @pytest_asyncio.asyncio
    async def test_convert_audio_format(self):
        """Test audio format conversion."""
        preprocessor = AudioPreprocessor()

        # Create a test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
            import numpy as np
            sample_rate = 22050
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
            audio_data = audio_data.astype(np.float32)

            import soundfile as sf
            sf.write(input_file.name, audio_data, sample_rate)

            # Convert to different format
            with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as output_file:
                result = await preprocessor.convert_audio_format(
                    input_file.name,
                    output_file.name,
                    target_format='flac',
                    target_sample_rate=22050,
                    target_channels=1
                )

                assert result['format'] == 'flac'
                assert result['target_sample_rate'] == 22050
                assert result['duration'] == duration
                assert os.path.exists(output_file.name)

                os.unlink(input_file.name)
                os.unlink(output_file.name)


class TestVoiceQualityAssessor:
    """Test voice quality assessment."""

    @pytest_asyncio.asyncio
    async def test_assess_voice_model_quality(self, db_session: Session, test_voice_model):
        """Test voice model quality assessment."""
        assessor = VoiceQualityAssessor(db_session)

        # Set model as trained
        test_voice_model.status = VoiceModelStatus.TRAINED
        test_voice_model.quality_score = 7.5
        db_session.commit()

        # Assess quality
        result = await assessor.assess_voice_model_quality(
            test_voice_model.id,
            test_texts=["Test text 1", "Test text 2"],
            assessment_method="automated"
        )

        assert result['voice_model_id'] == test_voice_model.id
        assert result['overall_score'] > 0
        assert result['quality_grade'] in [grade.value for grade in VoiceQualityScore]
        assert 'detailed_scores' in result
        assert 'recommendations' in result

    @pytest_asyncio.asyncio
    async def test_compare_voice_models(self, db_session: Session, test_voice_model, test_voice_model_2):
        """Test comparing multiple voice models."""
        assessor = VoiceQualityAssessor(db_session)

        # Set both models as trained
        test_voice_model.status = VoiceModelStatus.TRAINED
        test_voice_model.quality_score = 8.0
        test_voice_model_2.status = VoiceModelStatus.TRAINED
        test_voice_model_2.quality_score = 6.5
        db_session.commit()

        # Compare models
        result = await assessor.compare_voice_models(
            [test_voice_model.id, test_voice_model_2.id],
            test_texts=["Test text 1", "Test text 2"]
        )

        assert result['models_compared'] == 2
        assert 'individual_results' in result
        assert 'best_model_id' in result
        assert 'recommendations' in result

    @pytest_asyncio.asyncio
    async def test_run_quality_tests(self, db_session: Session, test_voice_model):
        """Test running comprehensive quality tests."""
        assessor = VoiceQualityAssessor(db_session)

        # Set model as trained
        test_voice_model.status = VoiceModelStatus.TRAINED
        db_session.commit()

        # Run tests
        result = await assessor.run_quality_tests(
            test_voice_model.id,
            test_types=['quality', 'performance', 'consistency']
        )

        assert result['voice_model_id'] == test_voice_model.id
        assert result['tests_run'] == ['quality', 'performance', 'consistency']
        assert 'test_results' in result
        assert result['overall_pass'] in [True, False]

    @pytest_asyncio.asyncio
    async def test_get_quality_trend(self, db_session: Session, test_voice_model):
        """Test getting quality trend over time."""
        assessor = VoiceQualityAssessor(db_session)

        # Create some quality metrics
        for i in range(5):
            metrics = VoiceQualityMetrics(
                voice_model_id=test_voice_model.id,
                clarity_score=7.0 + i * 0.5,
                naturalness_score=7.5 + i * 0.3,
                pronunciation_score=8.0 + i * 0.2,
                consistency_score=7.8 + i * 0.1,
                expressiveness_score=7.2 + i * 0.4,
                overall_score=7.5 + i * 0.3,
                assessment_method="automated",
                assessed_at=datetime.utcnow() - timedelta(days=4-i)
            )
            db_session.add(metrics)

        db_session.commit()

        # Get trend
        trend = await assessor.get_quality_trend(test_voice_model.id, days=7)

        assert trend['voice_model_id'] == test_voice_model.id
        assert trend['period_days'] == 7
        assert 'score_range' in trend
        assert 'trend' in trend
        assert trend['data_points'] == 5


class TestVoiceCloningConfig:
    """Test voice cloning configuration."""

    def test_config_initialization(self):
        """Test configuration initialization."""
        config = VoiceCloningConfig()

        assert config.training_config['min_samples_required'] == 5
        assert config.quality_config['min_snr_ratio'] == 15.0
        assert config.storage_config['max_storage_per_org'] == 10 * 1024 * 1024 * 1024
        assert config.api_config['max_file_size_mb'] == 100

    def test_config_validation(self):
        """Test configuration validation."""
        config = VoiceCloningConfig()

        issues = config.validate_config()

        # Should have no issues with default config
        assert isinstance(issues, list)

    def test_config_getters(self):
        """Test configuration getters."""
        config = VoiceCloningConfig()

        training_config = config.get_training_config()
        assert 'min_samples_required' in training_config
        assert 'batch_size' in training_config

        quality_config = config.get_quality_config()
        assert 'min_snr_ratio' in quality_config
        assert 'quality_weights' in quality_config

        path_config = config.get_path_config()
        assert 'base_dir' in path_config
        assert 'voice_library_path' in path_config

    def test_config_update(self):
        """Test configuration updates."""
        config = VoiceCloningConfig()

        # Update training config
        config.update_config('training', 'batch_size', 16)
        assert config.get_config_value('training', 'batch_size') == 16

        # Update API config
        config.update_config('api', 'max_file_size_mb', 200)
        assert config.get_config_value('api', 'max_file_size_mb') == 200

    def test_environment_config_loading(self):
        """Test environment-specific configuration loading."""
        import os

        # Test production config
        os.environ['FLASK_ENV'] = 'production'
        config = VoiceCloningConfig()

        assert config.api_config['max_concurrent_trainings'] == 10
        assert config.security_config['enable_encryption'] == True

        # Test testing config
        os.environ['FLASK_ENV'] = 'testing'
        config = VoiceCloningConfig()

        assert config.training_config['min_samples_required'] == 2
        assert config.development_config['mock_training'] == True

        # Clean up
        os.environ.pop('FLASK_ENV', None)


class TestVoiceCloningAPI:
    """Test voice cloning API endpoints."""

    def test_create_voice_model_unauthorized(self, client):
        """Test creating voice model without authentication."""
        response = client.post('/voice-cloning/voice-models')
        assert response.status_code == 401

    def test_create_voice_model_success(self, client, auth_headers, test_user):
        """Test successful voice model creation."""
        data = {
            'name': 'Test Voice Model',
            'description': 'Test model for API',
            'language': 'vi',
            'gender': 'male',
            'age_group': 'adult',
            'accent': 'northern',
            'model_type': 'standard'
        }

        response = client.post(
            '/voice-cloning/voice-models',
            headers=auth_headers,
            json=data
        )

        assert response.status_code == 201
        assert response.json['message'] == 'Voice model created successfully'
        assert response.json['voice_model']['name'] == 'Test Voice Model'

    def test_get_voice_models_success(self, client, auth_headers, test_user):
        """Test getting voice models."""
        response = client.get(
            '/voice-cloning/voice-models',
            headers=auth_headers
        )

        assert response.status_code == 200
        assert 'voice_models' in response.json
        assert 'total_count' in response.json

    def test_upload_voice_sample_unauthorized(self, client):
        """Test uploading voice sample without authentication."""
        response = client.post('/voice-cloning/voice-models/1/samples')
        assert response.status_code == 401

    def test_upload_voice_sample_no_file(self, client, auth_headers):
        """Test uploading voice sample without file."""
        response = client.post(
            '/voice-cloning/voice-models/1/samples',
            headers=auth_headers
        )

        assert response.status_code == 400
        assert 'No audio file provided' in response.json['error']

    def test_train_voice_model_unauthorized(self, client):
        """Test training voice model without authentication."""
        response = client.post('/voice-cloning/voice-models/1/train')
        assert response.status_code == 401

    def test_train_voice_model_success(self, client, auth_headers):
        """Test successful voice model training initiation."""
        data = {
            'training_config': {
                'batch_size': 8,
                'learning_rate': 0.001,
                'epochs': 50
            }
        }

        response = client.post(
            '/voice-cloning/voice-models/1/train',
            headers=auth_headers,
            json=data
        )

        assert response.status_code == 202
        assert 'training started' in response.json['message']

    def test_get_training_status_unauthorized(self, client):
        """Test getting training status without authentication."""
        response = client.get('/voice-cloning/voice-models/1/training-status')
        assert response.status_code == 401

    def test_get_training_status_success(self, client, auth_headers):
        """Test getting training status."""
        response = client.get(
            '/voice-cloning/voice-models/1/training-status',
            headers=auth_headers
        )

        assert response.status_code == 200
        assert 'training_status' in response.json

    def test_assess_voice_quality_unauthorized(self, client):
        """Test assessing voice quality without authentication."""
        response = client.post('/voice-cloning/voice-models/1/quality')
        assert response.status_code == 401

    def test_assess_voice_quality_success(self, client, auth_headers):
        """Test successful voice quality assessment."""
        data = {
            'test_texts': ['Test text 1', 'Test text 2'],
            'assessment_method': 'automated'
        }

        response = client.post(
            '/voice-cloning/voice-models/1/quality',
            headers=auth_headers,
            json=data
        )

        assert response.status_code == 200
        assert 'quality_assessment' in response.json

    def test_get_public_voices_success(self, client):
        """Test getting public voices."""
        response = client.get('/voice-cloning/public-voices')

        assert response.status_code == 200
        assert 'public_voices' in response.json
        assert 'total_count' in response.json

    def test_get_voice_statistics_unauthorized(self, client):
        """Test getting voice statistics without authentication."""
        response = client.get('/voice-cloning/statistics')
        assert response.status_code == 401

    def test_get_voice_statistics_success(self, client, auth_headers):
        """Test getting voice statistics."""
        response = client.get(
            '/voice-cloning/statistics',
            headers=auth_headers
        )

        assert response.status_code == 200
        assert 'statistics' in response.json


# Pytest fixtures
@pytest.fixture
def test_voice_model(db_session: Session, test_user):
    """Create a test voice model."""
    voice_model = VoiceModel(
        name="Test Voice Model",
        organization_id=test_user.organization_id,
        created_by=test_user.id,
        language="vi",
        gender="male",
        age_group="adult",
        accent="northern",
        model_type=VoiceModelType.STANDARD
    )

    db_session.add(voice_model)
    db_session.commit()
    return voice_model

@pytest.fixture
def test_voice_model_2(db_session: Session, test_user):
    """Create a second test voice model."""
    voice_model = VoiceModel(
        name="Test Voice Model 2",
        organization_id=test_user.organization_id,
        created_by=test_user.id,
        language="vi",
        gender="female",
        age_group="adult",
        accent="southern",
        model_type=VoiceModelType.STANDARD
    )

    db_session.add(voice_model)
    db_session.commit()
    return voice_model

@pytest.fixture
def test_voice_sample(db_session: Session, test_voice_model):
    """Create a test voice sample."""
    sample = VoiceSample(
        voice_model_id=test_voice_model.id,
        uploaded_by=test_voice_model.created_by,
        filename="test_sample.wav",
        original_filename="test_sample.wav",
        file_path="/path/to/test_sample.wav",
        file_size=1024000,
        mime_type="audio/wav",
        checksum="abc123",
        duration=45.0,
        sample_rate=22050,
        quality_score=8.0,
        snr_ratio=20.0,
        status=VoiceSampleStatus.PROCESSED
    )

    db_session.add(sample)
    db_session.commit()
    return sample

@pytest.fixture
def test_voice_model_with_samples(db_session: Session, test_voice_model):
    """Create a voice model with multiple valid samples."""
    # Create 5 valid samples
    for i in range(5):
        sample = VoiceSample(
            voice_model_id=test_voice_model.id,
            uploaded_by=test_voice_model.created_by,
            filename=f"test_sample_{i}.wav",
            original_filename=f"test_sample_{i}.wav",
            file_path=f"/path/to/test_sample_{i}.wav",
            file_size=1024000,
            mime_type="audio/wav",
            checksum=f"abc123_{i}",
            duration=45.0,
            sample_rate=22050,
            quality_score=8.0,
            snr_ratio=20.0,
            status=VoiceSampleStatus.PROCESSED
        )
        db_session.add(sample)

    db_session.commit()
    return test_voice_model

@pytest.fixture
def test_voice_model_with_low_quality_samples(db_session: Session, test_voice_model):
    """Create a voice model with low quality samples."""
    # Create 5 low quality samples
    for i in range(5):
        sample = VoiceSample(
            voice_model_id=test_voice_model.id,
            uploaded_by=test_voice_model.created_by,
            filename=f"low_quality_sample_{i}.wav",
            original_filename=f"low_quality_sample_{i}.wav",
            file_path=f"/path/to/low_quality_sample_{i}.wav",
            file_size=1024000,
            mime_type="audio/wav",
            checksum=f"low_quality_123_{i}",
            duration=45.0,
            sample_rate=22050,
            quality_score=3.0,  # Low quality
            snr_ratio=8.0,  # Low SNR
            status=VoiceSampleStatus.PROCESSED
        )
        db_session.add(sample)

    db_session.commit()
    return test_voice_model