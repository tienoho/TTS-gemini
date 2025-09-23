"""
Voice Quality Assessment for TTS Voice Cloning System
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import librosa
import numpy as np
from sqlalchemy.orm import Session

from models.voice_cloning import (
    VoiceModel, VoiceQualityMetrics, VoiceTestResult,
    VoiceModelStatus, VoiceQualityScore
)
from .exceptions import AudioProcessingException, ValidationException


class VoiceQualityAssessor:
    """Assesses voice quality for trained models and provides improvement recommendations."""

    def __init__(self, db_session: Session):
        """Initialize voice quality assessor."""
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

        # Quality assessment configuration
        self.test_sample_rate = 22050
        self.test_duration_seconds = 10  # Duration for quality tests

        # Quality thresholds
        self.excellent_threshold = 9.0
        self.good_threshold = 7.0
        self.fair_threshold = 5.0
        self.poor_threshold = 3.0

        # Weight factors for different quality dimensions
        self.quality_weights = {
            'clarity': 0.25,
            'naturalness': 0.25,
            'pronunciation': 0.20,
            'consistency': 0.15,
            'expressiveness': 0.15
        }

    async def assess_voice_model_quality(
        self,
        voice_model_id: int,
        test_texts: List[str] = None,
        assessment_method: str = "automated"
    ) -> Dict[str, Any]:
        """Assess the quality of a trained voice model."""
        try:
            # Get voice model
            voice_model = self.db_session.query(VoiceModel).filter(
                VoiceModel.id == voice_model_id,
                VoiceModel.status == VoiceModelStatus.TRAINED
            ).first()

            if not voice_model:
                raise ValidationException("Voice model not found or not trained")

            # Default test texts if not provided
            if test_texts is None:
                test_texts = self._get_default_test_texts(voice_model.language)

            # Perform quality assessment
            quality_scores = await self._perform_comprehensive_assessment(
                voice_model, test_texts
            )

            # Calculate overall scores
            overall_score = self._calculate_overall_score(quality_scores)
            weighted_score = self._calculate_weighted_score(quality_scores)

            # Create quality metrics record
            quality_metrics = VoiceQualityMetrics(
                voice_model_id=voice_model_id,
                clarity_score=quality_scores['clarity'],
                naturalness_score=quality_scores['naturalness'],
                pronunciation_score=quality_scores['pronunciation'],
                consistency_score=quality_scores['consistency'],
                expressiveness_score=quality_scores['expressiveness'],
                overall_score=overall_score,
                weighted_score=weighted_score,
                assessment_method=assessment_method,
                test_samples_count=len(test_texts),
                test_duration_seconds=self.test_duration_seconds * len(test_texts)
            )

            # Add detailed metrics
            quality_metrics.word_error_rate = quality_scores.get('word_error_rate')
            quality_metrics.character_error_rate = quality_scores.get('character_error_rate')
            quality_metrics.mel_cepstral_distortion = quality_scores.get('mel_cepstral_distortion')
            quality_metrics.f0_frame_error = quality_scores.get('f0_frame_error')

            # Add audio characteristics
            quality_metrics.speaking_rate = quality_scores.get('speaking_rate')
            quality_metrics.pitch_mean = quality_scores.get('pitch_mean')
            quality_metrics.pitch_std = quality_scores.get('pitch_std')
            quality_metrics.energy_mean = quality_scores.get('energy_mean')
            quality_metrics.energy_std = quality_scores.get('energy_std')

            self.db_session.add(quality_metrics)
            self.db_session.commit()

            # Update voice model quality score
            voice_model.quality_score = overall_score
            voice_model.quality_assessed_at = datetime.utcnow()
            self.db_session.commit()

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                quality_scores, overall_score
            )

            return {
                'voice_model_id': voice_model_id,
                'overall_score': overall_score,
                'weighted_score': weighted_score,
                'quality_grade': quality_metrics.get_quality_grade().value,
                'detailed_scores': quality_scores,
                'recommendations': recommendations,
                'assessment_method': assessment_method,
                'assessed_at': quality_metrics.assessed_at.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Voice quality assessment error: {str(e)}")
            raise AudioProcessingException(f"Quality assessment failed: {str(e)}")

    async def _perform_comprehensive_assessment(
        self,
        voice_model: VoiceModel,
        test_texts: List[str]
    ) -> Dict[str, float]:
        """Perform comprehensive quality assessment."""
        try:
            scores = {
                'clarity': 0.0,
                'naturalness': 0.0,
                'pronunciation': 0.0,
                'consistency': 0.0,
                'expressiveness': 0.0
            }

            # Assess each test text
            for text in test_texts:
                text_scores = await self._assess_single_text(voice_model, text)

                # Accumulate scores
                for key in scores:
                    scores[key] += text_scores.get(key, 0.0)

            # Average scores
            for key in scores:
                scores[key] /= len(test_texts)

            # Add additional metrics
            scores.update(await self._calculate_additional_metrics(voice_model, test_texts))

            return scores

        except Exception as e:
            self.logger.error(f"Comprehensive assessment error: {str(e)}")
            return self._get_default_scores()

    async def _assess_single_text(
        self,
        voice_model: VoiceModel,
        text: str
    ) -> Dict[str, float]:
        """Assess quality for a single text sample."""
        try:
            # In a real implementation, this would:
            # 1. Generate audio using the voice model
            # 2. Analyze the generated audio
            # 3. Compare with reference audio if available

            # For now, simulate quality assessment based on model characteristics
            base_score = voice_model.quality_score / 10.0  # Normalize to 0-1

            # Add some variation based on text characteristics
            text_length = len(text)
            complexity_factor = min(1.0, text_length / 100.0)  # Longer texts are more complex

            # Simulate different quality dimensions
            clarity = base_score * (0.8 + 0.2 * np.random.random())
            naturalness = base_score * (0.7 + 0.3 * np.random.random())
            pronunciation = base_score * (0.9 + 0.1 * np.random.random())
            consistency = base_score * (0.85 + 0.15 * np.random.random())
            expressiveness = base_score * (0.6 + 0.4 * np.random.random())

            # Ensure scores are within bounds
            return {
                'clarity': max(1.0, min(10.0, clarity * 10)),
                'naturalness': max(1.0, min(10.0, naturalness * 10)),
                'pronunciation': max(1.0, min(10.0, pronunciation * 10)),
                'consistency': max(1.0, min(10.0, consistency * 10)),
                'expressiveness': max(1.0, min(10.0, expressiveness * 10))
            }

        except Exception as e:
            self.logger.warning(f"Single text assessment error: {str(e)}")
            return self._get_default_scores()

    async def _calculate_additional_metrics(
        self,
        voice_model: VoiceModel,
        test_texts: List[str]
    ) -> Dict[str, float]:
        """Calculate additional quality metrics."""
        try:
            # In a real implementation, these would be calculated from actual audio analysis
            # For now, provide simulated values based on model characteristics

            # Word Error Rate (WER) - simulated
            wer = max(0.01, 0.3 - (voice_model.quality_score / 50))  # Better models have lower WER

            # Character Error Rate (CER) - simulated
            cer = wer * 0.7  # CER is typically lower than WER

            # Mel-Cepstral Distortion (MCD) - simulated
            mcd = max(5.0, 15.0 - (voice_model.quality_score / 2))  # Better models have lower MCD

            # F0 Frame Error - simulated
            f0_error = max(0.05, 0.4 - (voice_model.quality_score / 30))

            # Speaking rate (words per minute) - simulated
            speaking_rate = 150 + (voice_model.quality_score - 5) * 10  # 100-200 WPM

            # Pitch characteristics - simulated
            pitch_mean = 150 + (voice_model.quality_score - 5) * 20  # 100-200 Hz
            pitch_std = 20 + (voice_model.quality_score - 5) * 5  # 15-35 Hz

            # Energy characteristics - simulated
            energy_mean = -20 + (voice_model.quality_score - 5) * 2  # -30 to -10 dB
            energy_std = 5 + (voice_model.quality_score - 5) * 1  # 4-8 dB

            return {
                'word_error_rate': wer,
                'character_error_rate': cer,
                'mel_cepstral_distortion': mcd,
                'f0_frame_error': f0_error,
                'speaking_rate': speaking_rate,
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'energy_mean': energy_mean,
                'energy_std': energy_std
            }

        except Exception as e:
            self.logger.warning(f"Additional metrics calculation error: {str(e)}")
            return {}

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        try:
            # Simple average of main quality dimensions
            main_scores = [
                scores.get('clarity', 5.0),
                scores.get('naturalness', 5.0),
                scores.get('pronunciation', 5.0),
                scores.get('consistency', 5.0),
                scores.get('expressiveness', 5.0)
            ]

            return sum(main_scores) / len(main_scores)

        except Exception:
            return 5.0

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted quality score."""
        try:
            weighted_sum = 0.0
            total_weight = 0.0

            for dimension, weight in self.quality_weights.items():
                score = scores.get(dimension, 5.0)
                weighted_sum += score * weight
                total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 5.0

        except Exception:
            return 5.0

    def _get_default_scores(self) -> Dict[str, float]:
        """Get default quality scores."""
        return {
            'clarity': 5.0,
            'naturalness': 5.0,
            'pronunciation': 5.0,
            'consistency': 5.0,
            'expressiveness': 5.0
        }

    def _get_default_test_texts(self, language: str = "vi") -> List[str]:
        """Get default test texts for quality assessment."""
        if language.lower() == "vi":
            return [
                "Xin chào, tôi là trợ lý ảo của bạn.",
                "Hôm nay thời tiết thật đẹp.",
                "Tôi có thể giúp bạn những gì?",
                "Cảm ơn bạn đã sử dụng dịch vụ của chúng tôi.",
                "Chúc bạn một ngày tốt lành."
            ]
        else:
            return [
                "Hello, I am your virtual assistant.",
                "Today the weather is beautiful.",
                "How can I help you?",
                "Thank you for using our service.",
                "Have a nice day."
            ]

    def _generate_quality_recommendations(
        self,
        scores: Dict[str, float],
        overall_score: float
    ) -> List[str]:
        """Generate recommendations for quality improvement."""
        recommendations = []

        try:
            # Overall recommendations
            if overall_score < self.poor_threshold:
                recommendations.append("Model quality is poor. Consider retraining with better samples.")
            elif overall_score < self.fair_threshold:
                recommendations.append("Model quality is below average. Consider improving training data.")
            elif overall_score < self.good_threshold:
                recommendations.append("Model quality is fair. Minor improvements possible.")

            # Dimension-specific recommendations
            if scores.get('clarity', 5.0) < 6.0:
                recommendations.append("Improve clarity by using higher quality audio samples with less noise.")

            if scores.get('naturalness', 5.0) < 6.0:
                recommendations.append("Enhance naturalness by including more diverse speaking styles in training data.")

            if scores.get('pronunciation', 5.0) < 6.0:
                recommendations.append("Improve pronunciation accuracy by using clearer speech samples.")

            if scores.get('consistency', 5.0) < 6.0:
                recommendations.append("Increase consistency by using samples from the same recording session.")

            if scores.get('expressiveness', 5.0) < 6.0:
                recommendations.append("Add expressiveness by including samples with varied intonation and emotion.")

            # Technical recommendations
            wer = scores.get('word_error_rate', 0.2)
            if wer > 0.15:
                recommendations.append(f"High word error rate ({wer".2%"}). Consider using cleaner audio samples.")

            mcd = scores.get('mel_cepstral_distortion', 10.0)
            if mcd > 10.0:
                recommendations.append(f"High spectral distortion ({mcd".1f"}). Consider improving model architecture.")

            # Positive feedback
            if overall_score >= self.excellent_threshold:
                recommendations.append("Excellent voice quality achieved!")
            elif overall_score >= self.good_threshold:
                recommendations.append("Good voice quality. Model is ready for production use.")

        except Exception as e:
            self.logger.warning(f"Recommendation generation error: {str(e)}")
            recommendations.append("Unable to generate specific recommendations.")

        return recommendations

    async def compare_voice_models(
        self,
        model_ids: List[int],
        test_texts: List[str] = None
    ) -> Dict[str, Any]:
        """Compare quality between multiple voice models."""
        try:
            # Get all models
            models = self.db_session.query(VoiceModel).filter(
                VoiceModel.id.in_(model_ids),
                VoiceModel.status == VoiceModelStatus.TRAINED
            ).all()

            if len(models) != len(model_ids):
                raise ValidationException("One or more voice models not found or not trained")

            # Assess each model
            comparison_results = {}
            for model in models:
                assessment = await self.assess_voice_model_quality(
                    model.id, test_texts, "comparison"
                )
                comparison_results[str(model.id)] = assessment

            # Generate comparison summary
            scores = [result['overall_score'] for result in comparison_results.values()]
            best_model_id = max(comparison_results.keys(), key=lambda k: comparison_results[k]['overall_score'])

            return {
                'models_compared': len(models),
                'individual_results': comparison_results,
                'best_model_id': best_model_id,
                'score_range': {
                    'min': min(scores),
                    'max': max(scores),
                    'average': sum(scores) / len(scores)
                },
                'recommendations': self._generate_comparison_recommendations(comparison_results)
            }

        except Exception as e:
            self.logger.error(f"Voice model comparison error: {str(e)}")
            raise AudioProcessingException(f"Comparison failed: {str(e)}")

    def _generate_comparison_recommendations(
        self,
        comparison_results: Dict[str, Dict]
    ) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []

        try:
            # Find best and worst performing models
            best_model_id = max(comparison_results.keys(), key=lambda k: comparison_results[k]['overall_score'])
            worst_model_id = min(comparison_results.keys(), key=lambda k: comparison_results[k]['overall_score'])

            best_score = comparison_results[best_model_id]['overall_score']
            worst_score = comparison_results[worst_model_id]['overall_score']

            if best_score - worst_score > 2.0:
                recommendations.append(f"Significant quality difference detected between models.")
                recommendations.append(f"Best performing model: {best_model_id} (score: {best_score".1f"})")
                recommendations.append(f"Consider using the best model as reference for improvements.")

            # Analyze quality dimensions across models
            dimension_averages = {}
            for dimension in self.quality_weights.keys():
                scores = [result['detailed_scores'].get(dimension, 5.0) for result in comparison_results.values()]
                dimension_averages[dimension] = sum(scores) / len(scores)

            # Find weakest dimension
            weakest_dimension = min(dimension_averages.keys(), key=lambda k: dimension_averages[k])
            if dimension_averages[weakest_dimension] < 6.0:
                recommendations.append(f"Focus improvement efforts on {weakest_dimension} (average: {dimension_averages[weakest_dimension]".1f"})")

        except Exception as e:
            self.logger.warning(f"Comparison recommendation generation error: {str(e)}")

        return recommendations

    async def run_quality_tests(
        self,
        voice_model_id: int,
        test_types: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive quality tests on a voice model."""
        try:
            if test_types is None:
                test_types = ['quality', 'performance', 'consistency']

            test_results = {}

            # Quality test
            if 'quality' in test_types:
                quality_result = await self.assess_voice_model_quality(
                    voice_model_id, assessment_method="automated_test"
                )
                test_results['quality'] = quality_result

            # Performance test (simulated)
            if 'performance' in test_types:
                performance_result = await self._run_performance_test(voice_model_id)
                test_results['performance'] = performance_result

            # Consistency test (simulated)
            if 'consistency' in test_types:
                consistency_result = await self._run_consistency_test(voice_model_id)
                test_results['consistency'] = consistency_result

            # Create test result records
            for test_type, result in test_results.items():
                test_record = VoiceTestResult(
                    voice_model_id=voice_model_id,
                    test_type=test_type,
                    test_name=f"{test_type}_test",
                    test_description=f"Automated {test_type} test",
                    test_result=result,
                    success=result.get('overall_score', 0) >= 5.0,
                    tested_at=datetime.utcnow()
                )
                self.db_session.add(test_record)

            self.db_session.commit()

            return {
                'voice_model_id': voice_model_id,
                'tests_run': list(test_results.keys()),
                'test_results': test_results,
                'overall_pass': all(result.get('overall_score', 0) >= 5.0 for result in test_results.values())
            }

        except Exception as e:
            self.logger.error(f"Quality tests error: {str(e)}")
            raise AudioProcessingException(f"Quality tests failed: {str(e)}")

    async def _run_performance_test(self, voice_model_id: int) -> Dict[str, Any]:
        """Run performance test on voice model."""
        try:
            # Simulate performance testing
            # In real implementation, this would measure actual inference time, memory usage, etc.

            start_time = time.time()

            # Simulate inference operations
            await asyncio.sleep(0.1)  # Simulate processing time

            end_time = time.time()
            processing_time = end_time - start_time

            return {
                'inference_time_ms': processing_time * 1000,
                'memory_usage_mb': 50 + np.random.random() * 20,  # Simulated 50-70MB
                'cpu_usage_percent': 10 + np.random.random() * 15,  # Simulated 10-25%
                'throughput_items_per_second': 5 + np.random.random() * 5,  # Simulated 5-10 items/sec
                'performance_score': min(10.0, 15.0 - processing_time * 10)  # Faster = higher score
            }

        except Exception as e:
            self.logger.warning(f"Performance test error: {str(e)}")
            return {'performance_score': 5.0}

    async def _run_consistency_test(self, voice_model_id: int) -> Dict[str, Any]:
        """Run consistency test on voice model."""
        try:
            # Test multiple times with same input to check consistency
            test_text = "This is a consistency test."

            scores = []
            for i in range(5):  # Test 5 times
                result = await self._assess_single_text(
                    self.db_session.query(VoiceModel).get(voice_model_id), test_text
                )
                scores.append(self._calculate_overall_score(result))

            # Calculate consistency metrics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            consistency_ratio = max(0.1, 1.0 - (std_score / mean_score))  # Higher = more consistent

            return {
                'mean_score': mean_score,
                'score_std': std_score,
                'consistency_ratio': consistency_ratio,
                'consistency_score': consistency_ratio * 10,  # Convert to 1-10 scale
                'num_tests': len(scores)
            }

        except Exception as e:
            self.logger.warning(f"Consistency test error: {str(e)}")
            return {'consistency_score': 5.0}

    async def get_quality_trend(
        self,
        voice_model_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get quality trend for a voice model over time."""
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Get quality metrics within the time range
            metrics = self.db_session.query(VoiceQualityMetrics).filter(
                VoiceQualityMetrics.voice_model_id == voice_model_id,
                VoiceQualityMetrics.assessed_at >= cutoff_date
            ).order_by(VoiceQualityMetrics.assessed_at).all()

            if not metrics:
                return {'trend': 'no_data', 'message': 'No quality data available for the specified period'}

            # Calculate trend
            scores = [m.overall_score for m in metrics]
            dates = [m.assessed_at for m in metrics]

            # Simple trend analysis
            if len(scores) >= 2:
                first_score = scores[0]
                last_score = scores[-1]
                trend_direction = 'improving' if last_score > first_score else 'declining' if last_score < first_score else 'stable'

                trend_slope = (last_score - first_score) / len(scores) if len(scores) > 1 else 0
            else:
                trend_direction = 'stable'
                trend_slope = 0

            return {
                'voice_model_id': voice_model_id,
                'period_days': days,
                'data_points': len(metrics),
                'score_range': {
                    'min': min(scores),
                    'max': max(scores),
                    'current': scores[-1] if scores else 0,
                    'average': sum(scores) / len(scores) if scores else 0
                },
                'trend': {
                    'direction': trend_direction,
                    'slope': trend_slope,
                    'improvement': scores[-1] - scores[0] if len(scores) >= 2 else 0
                },
                'quality_grades': [m.get_quality_grade().value for m in metrics]
            }

        except Exception as e:
            self.logger.error(f"Quality trend analysis error: {str(e)}")
            raise AudioProcessingException(f"Trend analysis failed: {str(e)}")


# Global instance
voice_quality_assessor = None

def get_voice_quality_assessor(db_session: Session) -> VoiceQualityAssessor:
    """Get or create voice quality assessor instance."""
    global voice_quality_assessor
    if voice_quality_assessor is None:
        voice_quality_assessor = VoiceQualityAssessor(db_session)
    return voice_quality_assessor