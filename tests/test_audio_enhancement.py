"""
Tests for Audio Enhancement System
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from models.audio_enhancement import AudioEnhancement, EnhancementPreset, AudioQualityMetric
from utils.audio_enhancer import AudioEnhancer
from utils.audio_quality_analyzer import AudioQualityAnalyzer
from utils.enhancement_presets import EnhancementPresetManager
from utils.realtime_enhancer import RealTimeEnhancer
from config.audio_enhancement import AudioEnhancementSettings


class TestAudioEnhancementModels:
    """Test Audio Enhancement Models."""

    def test_audio_enhancement_creation(self):
        """Test AudioEnhancement model creation."""
        enhancement = AudioEnhancement(
            audio_file_id=1,
            user_id=1,
            enhancement_type='noise_reduction',
            settings={'noise_factor': 0.1}
        )

        assert enhancement.audio_file_id == 1
        assert enhancement.user_id == 1
        assert enhancement.enhancement_type == 'noise_reduction'
        assert enhancement.settings == {'noise_factor': 0.1}
        assert enhancement.success == True

    def test_enhancement_to_dict(self):
        """Test AudioEnhancement to_dict method."""
        enhancement = AudioEnhancement(
            audio_file_id=1,
            user_id=1,
            enhancement_type='noise_reduction',
            settings={'noise_factor': 0.1}
        )

        result = enhancement.to_dict()

        assert result['audio_file_id'] == 1
        assert result['user_id'] == 1
        assert result['enhancement_type'] == 'noise_reduction'
        assert result['settings'] == {'noise_factor': 0.1}

    def test_enhancement_from_dict(self):
        """Test AudioEnhancement from_dict method."""
        data = {
            'audio_file_id': 1,
            'user_id': 1,
            'enhancement_type': 'noise_reduction',
            'settings': {'noise_factor': 0.1}
        }

        enhancement = AudioEnhancement.from_dict(data)

        assert enhancement.audio_file_id == 1
        assert enhancement.user_id == 1
        assert enhancement.enhancement_type == 'noise_reduction'
        assert enhancement.settings == {'noise_factor': 0.1}

    def test_enhancement_improvement_metrics(self):
        """Test improvement metrics calculation."""
        enhancement = AudioEnhancement(
            audio_file_id=1,
            user_id=1,
            enhancement_type='noise_reduction',
            settings={'noise_factor': 0.1},
            original_quality_score=5.0,
            enhanced_quality_score=8.0,
            original_snr=15.0,
            enhanced_snr=25.0,
            file_size_before=1000,
            file_size_after=1200
        )

        metrics = enhancement.get_improvement_metrics()

        assert metrics['quality_score_improvement'] == 3.0
        assert metrics['snr_improvement'] == 10.0
        assert metrics['size_change_percent'] == 20.0

    def test_enhancement_preset_creation(self):
        """Test EnhancementPreset model creation."""
        preset = EnhancementPreset(
            name='Test Preset',
            enhancement_type='noise_reduction',
            settings={'noise_factor': 0.1}
        )

        assert preset.name == 'Test Preset'
        assert preset.enhancement_type == 'noise_reduction'
        assert preset.settings == {'noise_factor': 0.1}
        assert preset.is_system_preset == False

    def test_preset_to_dict(self):
        """Test EnhancementPreset to_dict method."""
        preset = EnhancementPreset(
            name='Test Preset',
            enhancement_type='noise_reduction',
            settings={'noise_factor': 0.1}
        )

        result = preset.to_dict()

        assert result['name'] == 'Test Preset'
        assert result['enhancement_type'] == 'noise_reduction'
        assert result['settings'] == {'noise_factor': 0.1}

    def test_preset_rating_update(self):
        """Test preset rating update."""
        preset = EnhancementPreset(
            name='Test Preset',
            enhancement_type='noise_reduction',
            settings={'noise_factor': 0.1}
        )

        preset.update_rating(4.5)

        assert preset.rating == 4.5
        assert preset.rating_count == 1

        preset.update_rating(3.5)

        assert preset.rating == 4.0  # Average of 4.5 and 3.5
        assert preset.rating_count == 2

    def test_audio_quality_metric_creation(self):
        """Test AudioQualityMetric model creation."""
        metric = AudioQualityMetric(
            audio_file_id=1,
            analysis_method='algorithmic',
            snr=25.0,
            thd=2.0,
            overall_quality_score=8.5
        )

        assert metric.audio_file_id == 1
        assert metric.analysis_method == 'algorithmic'
        assert metric.snr == 25.0
        assert metric.thd == 2.0
        assert metric.overall_quality_score == 8.5

    def test_quality_metric_to_dict(self):
        """Test AudioQualityMetric to_dict method."""
        metric = AudioQualityMetric(
            audio_file_id=1,
            analysis_method='algorithmic',
            snr=25.0,
            thd=2.0,
            overall_quality_score=8.5
        )

        result = metric.to_dict()

        assert result['audio_file_id'] == 1
        assert result['analysis_method'] == 'algorithmic'
        assert result['snr'] == 25.0
        assert result['thd'] == 2.0
        assert result['overall_quality_score'] == 8.5

    def test_quality_grade_calculation(self):
        """Test quality grade calculation."""
        metric = AudioQualityMetric(
            audio_file_id=1,
            analysis_method='algorithmic',
            overall_quality_score=9.2
        )

        assert metric.get_quality_grade() == 'Excellent'

        metric.overall_quality_score = 7.5
        assert metric.get_quality_grade() == 'Good'

        metric.overall_quality_score = 4.5
        assert metric.get_quality_grade() == 'Poor'


class TestAudioEnhancer:
    """Test Audio Enhancer functionality."""

    def test_audio_enhancer_initialization(self):
        """Test AudioEnhancer initialization."""
        enhancer = AudioEnhancer()

        assert enhancer.supported_formats == ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac']
        assert enhancer.max_file_size == 50 * 1024 * 1024

    def test_validate_audio_file(self):
        """Test audio file validation."""
        enhancer = AudioEnhancer()

        # Test valid file
        audio_data = b'fake audio data'
        result = enhancer.validate_audio_file(audio_data, 'test.wav')

        assert result['valid'] == True
        assert result['size'] == len(audio_data)
        assert result['format'] == 'wav'

        # Test empty file
        with pytest.raises(Exception):  # Should raise ValidationException
            enhancer.validate_audio_file(b'', 'test.wav')

        # Test unsupported format
        with pytest.raises(Exception):  # Should raise ValidationException
            enhancer.validate_audio_file(b'data', 'test.xyz')

    def test_noise_reduction(self):
        """Test noise reduction functionality."""
        enhancer = AudioEnhancer()

        # Create test audio with noise
        np.random.seed(42)
        audio = np.random.randn(1000) * 0.1  # Low amplitude noise

        result = enhancer.apply_noise_reduction(audio, 0.1)

        assert len(result) == len(audio)
        assert isinstance(result, np.ndarray)

    def test_normalization(self):
        """Test audio normalization."""
        enhancer = AudioEnhancer()

        # Create test audio
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = enhancer.apply_normalization(audio, -3.0)

        assert len(result) == len(audio)
        assert isinstance(result, np.ndarray)

    def test_compression(self):
        """Test dynamic range compression."""
        enhancer = AudioEnhancer()

        # Create test audio
        audio = np.array([0.1, 0.5, 0.8, 0.3, 0.9])

        result = enhancer.apply_compression(audio, -20.0, 4.0)

        assert len(result) == len(audio)
        assert isinstance(result, np.ndarray)

    def test_enhancement_settings_validation(self):
        """Test enhancement settings validation."""
        enhancer = AudioEnhancer()

        # Test noise reduction settings
        settings = enhancer.validate_enhancement_settings('noise_reduction', {'noise_factor': 0.5})
        assert settings['noise_factor'] == 0.5

        # Test out of bounds
        settings = enhancer.validate_enhancement_settings('noise_reduction', {'noise_factor': 2.0})
        assert settings['noise_factor'] == 1.0  # Should be clamped to max

    def test_create_enhancement_preset(self):
        """Test preset creation."""
        enhancer = AudioEnhancer()

        preset = enhancer.create_enhancement_preset(
            name='Test Preset',
            enhancement_type='noise_reduction',
            settings={'noise_factor': 0.1}
        )

        assert preset.name == 'Test Preset'
        assert preset.enhancement_type == 'noise_reduction'
        assert preset.settings == {'noise_factor': 0.1}


class TestAudioQualityAnalyzer:
    """Test Audio Quality Analyzer functionality."""

    def test_analyzer_initialization(self):
        """Test AudioQualityAnalyzer initialization."""
        analyzer = AudioQualityAnalyzer()

        assert analyzer.supported_formats == ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac']
        assert analyzer.max_file_size == 50 * 1024 * 1024

    def test_snr_calculation(self):
        """Test SNR calculation."""
        analyzer = AudioQualityAnalyzer()

        # Create test audio with signal and noise
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100)  # 440Hz tone
        noise = np.random.randn(1000) * 0.1
        audio = signal + noise

        snr = analyzer.calculate_snr(audio)

        assert isinstance(snr, float)
        assert snr > 0  # Should be positive

    def test_thd_calculation(self):
        """Test THD calculation."""
        analyzer = AudioQualityAnalyzer()

        # Create test audio with harmonics
        t = np.arange(1000) / 44100
        fundamental = np.sin(2 * np.pi * 440 * t)
        harmonic = 0.1 * np.sin(2 * np.pi * 880 * t)  # 2nd harmonic
        audio = fundamental + harmonic

        thd = analyzer.calculate_thd(audio, 44100, 440)

        assert isinstance(thd, float)
        assert thd >= 0

    def test_rms_calculation(self):
        """Test RMS calculation."""
        analyzer = AudioQualityAnalyzer()

        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        rms = analyzer.calculate_rms(audio)

        expected_rms = np.sqrt(np.mean(np.array([0.1, 0.2, 0.3, 0.4, 0.5])**2))
        assert abs(rms - expected_rms) < 1e-10

    def test_peak_calculation(self):
        """Test peak calculation."""
        analyzer = AudioQualityAnalyzer()

        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        peak = analyzer.calculate_peak(audio)

        assert peak == 0.5

    def test_quality_score_calculation(self):
        """Test overall quality score calculation."""
        analyzer = AudioQualityAnalyzer()

        # Create high quality test audio
        np.random.seed(42)
        audio = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100) * 0.5

        quality_score = analyzer.calculate_overall_quality_score(audio, 44100)

        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 10

    def test_audio_quality_analysis(self):
        """Test comprehensive audio quality analysis."""
        analyzer = AudioQualityAnalyzer()

        # Create test audio
        np.random.seed(42)
        audio = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100) * 0.5

        # Convert to bytes (simulate audio file)
        audio_bytes = audio.tobytes()

        # Mock the load_audio_from_bytes method
        with patch.object(analyzer, 'load_audio_from_bytes', return_value=(audio, 44100)):
            results = analyzer.analyze_audio_quality(audio_bytes)

            assert 'snr' in results
            assert 'thd' in results
            assert 'overall_quality_score' in results
            assert 'quality_grade' in results
            assert results['analysis_method'] == 'algorithmic'


class TestEnhancementPresets:
    """Test Enhancement Preset Management."""

    def test_preset_manager_initialization(self):
        """Test EnhancementPresetManager initialization."""
        mock_db = Mock()
        manager = EnhancementPresetManager(mock_db)

        assert manager.ab_test_min_samples == 10
        assert manager.ab_test_confidence_threshold == 0.95
        assert manager.ab_test_duration_days == 7

    def test_get_default_presets(self):
        """Test getting default presets."""
        mock_db = Mock()
        manager = EnhancementPresetManager(mock_db)

        presets = manager.get_default_presets()

        assert len(presets) > 0
        assert all('name' in preset for preset in presets)
        assert all('enhancement_type' in preset for preset in presets)
        assert all('settings' in preset for preset in presets)

    def test_preset_categories(self):
        """Test preset categories."""
        mock_db = Mock()
        manager = EnhancementPresetManager(mock_db)

        categories = manager.categories

        assert 'speech' in categories
        assert 'music' in categories
        assert 'podcast' in categories
        assert categories['speech'] == 'Speech Enhancement'


class TestRealTimeEnhancer:
    """Test Real-time Audio Enhancement."""

    def test_realtime_enhancer_initialization(self):
        """Test RealTimeEnhancer initialization."""
        enhancer = RealTimeEnhancer()

        assert enhancer.sample_rate == 44100
        assert enhancer.chunk_size == 1024
        assert enhancer.buffer_size == 8192
        assert enhancer.target_latency == 0.05
        assert not enhancer.is_processing

    def test_adaptive_settings(self):
        """Test adaptive settings functionality."""
        enhancer = RealTimeEnhancer()

        settings = enhancer._get_default_adaptive_settings()

        assert 'noise_reduction' in settings
        assert 'normalization' in settings
        assert 'compression' in settings
        assert settings['adaptive_noise_reduction'] == True

    def test_processing_stats(self):
        """Test processing statistics."""
        enhancer = RealTimeEnhancer()

        stats = enhancer.get_processing_stats()

        assert 'is_processing' in stats
        assert 'enhancement_type' in stats
        assert 'avg_processing_time' in stats
        assert not stats['is_processing']

    def test_enhancement_settings_update(self):
        """Test enhancement settings update."""
        enhancer = RealTimeEnhancer()

        success = enhancer.update_enhancement_settings(
            'noise_reduction',
            {'noise_factor': 0.2}
        )

        assert success
        assert enhancer.current_enhancement_type == 'noise_reduction'
        assert enhancer.current_settings['noise_factor'] == 0.2


class TestAudioEnhancementSettings:
    """Test Audio Enhancement Configuration."""

    def test_settings_initialization(self):
        """Test AudioEnhancementSettings initialization."""
        settings = AudioEnhancementSettings()

        assert settings.ENABLE_AUDIO_ENHANCEMENT == True
        assert settings.MAX_AUDIO_FILE_SIZE_MB == 50
        assert settings.REAL_TIME_CHUNK_SIZE == 1024
        assert settings.SUPPORTED_INPUT_FORMATS == ["mp3", "wav", "flac", "ogg", "m4a", "aac"]

    def test_settings_properties(self):
        """Test settings properties."""
        settings = AudioEnhancementSettings()

        assert settings.is_enhancement_enabled == True
        assert settings.is_real_time_enabled == True
        assert settings.is_quality_analysis_enabled == True
        assert settings.max_file_size_bytes == 50 * 1024 * 1024
        assert settings.real_time_target_latency_seconds == 0.05

    def test_enhancement_config(self):
        """Test enhancement configuration retrieval."""
        settings = AudioEnhancementSettings()

        config = settings.get_enhancement_config('noise_reduction')

        assert 'min_factor' in config
        assert 'max_factor' in config
        assert 'default_factor' in config

    def test_settings_validation(self):
        """Test enhancement settings validation."""
        settings = AudioEnhancementSettings()

        # Test noise reduction validation
        validated = settings.validate_enhancement_settings(
            'noise_reduction',
            {'noise_factor': 0.5}
        )

        assert validated['noise_factor'] == 0.5

        # Test out of bounds
        validated = settings.validate_enhancement_settings(
            'noise_reduction',
            {'noise_factor': 2.0}
        )

        assert validated['noise_factor'] == 1.0  # Should be clamped


class TestIntegration:
    """Integration tests for Audio Enhancement System."""

    def test_enhancer_quality_analyzer_integration(self):
        """Test integration between enhancer and quality analyzer."""
        enhancer = AudioEnhancer()
        analyzer = AudioQualityAnalyzer()

        # Create test audio
        np.random.seed(42)
        audio = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100) * 0.5

        # Apply enhancement
        enhanced = enhancer.apply_noise_reduction(audio, 0.1)

        # Analyze quality
        quality_score = analyzer.calculate_overall_quality_score(enhanced, 44100)

        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 10

    def test_preset_manager_with_enhancer(self):
        """Test integration between preset manager and enhancer."""
        mock_db = Mock()
        preset_manager = EnhancementPresetManager(mock_db)
        enhancer = AudioEnhancer()

        # Get default presets
        presets = preset_manager.get_default_presets()

        assert len(presets) > 0

        # Test preset validation
        for preset_data in presets[:1]:  # Test first preset
            validated_settings = enhancer.validate_enhancement_settings(
                preset_data['enhancement_type'],
                preset_data['settings']
            )

            assert isinstance(validated_settings, dict)
            assert len(validated_settings) > 0


if __name__ == '__main__':
    # Run basic tests
    test_classes = [
        TestAudioEnhancementModels,
        TestAudioEnhancer,
        TestAudioQualityAnalyzer,
        TestEnhancementPresets,
        TestRealTimeEnhancer,
        TestAudioEnhancementSettings,
        TestIntegration
    ]

    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")
        test_instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(test_instance)
                       if method.startswith('test_') and callable(getattr(test_instance, method))]

        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")

    print("\nAudio Enhancement System Tests Completed!")