"""
Enhancement Presets Management for TTS Audio Enhancement System
Provides comprehensive preset management with A/B testing capabilities
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import random
import statistics

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from models.audio_enhancement import EnhancementPreset, AudioEnhancement
from utils.audio_enhancer import audio_enhancer
from utils.exceptions import ValidationException, AudioProcessingException


class EnhancementPresetManager:
    """Manages enhancement presets with A/B testing and analytics."""

    def __init__(self, db_session: Session):
        """Initialize enhancement preset manager."""
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

        # Preset categories
        self.categories = {
            'speech': 'Speech Enhancement',
            'music': 'Music Enhancement',
            'podcast': 'Podcast Enhancement',
            'audiobook': 'Audiobook Enhancement',
            'voiceover': 'Voice Over Enhancement',
            'custom': 'Custom Presets'
        }

        # A/B testing configuration
        self.ab_test_min_samples = 10
        self.ab_test_confidence_threshold = 0.95
        self.ab_test_duration_days = 7

    def get_default_presets(self) -> List[Dict[str, Any]]:
        """Get comprehensive default enhancement presets.

        Returns:
            List of default preset configurations
        """
        return [
            # Speech Enhancement Presets
            {
                'name': 'Clean Speech',
                'category': 'speech',
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
                'description': 'Clean and clear speech enhancement for presentations and talks',
                'tags': ['speech', 'clean', 'professional'],
                'is_system_preset': True
            },
            {
                'name': 'Studio Quality Speech',
                'category': 'speech',
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
                'description': 'Professional studio quality enhancement for broadcast speech',
                'tags': ['speech', 'studio', 'broadcast', 'professional'],
                'is_system_preset': True
            },
            {
                'name': 'Voice Over Enhancement',
                'category': 'voiceover',
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
                'description': 'Optimized for voice over and narration work',
                'tags': ['voiceover', 'narration', 'optimized'],
                'is_system_preset': True
            },

            # Music Enhancement Presets
            {
                'name': 'Music Clarity',
                'category': 'music',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.05,
                    'normalization': True,
                    'target_level': -12.0,
                    'compression': True,
                    'threshold': -15.0,
                    'ratio': 6.0,
                    'equalization': True,
                    'low_gain': 0.0,
                    'mid_gain': 2.0,
                    'high_gain': 1.0
                },
                'description': 'Enhanced clarity for music recordings',
                'tags': ['music', 'clarity', 'recording'],
                'is_system_preset': True
            },
            {
                'name': 'Concert Recording',
                'category': 'music',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.08,
                    'normalization': True,
                    'target_level': -8.0,
                    'compression': True,
                    'threshold': -18.0,
                    'ratio': 5.0,
                    'equalization': True,
                    'low_gain': 1.5,
                    'mid_gain': 0.5,
                    'high_gain': 2.0,
                    'reverb': True,
                    'room_size': 0.6,
                    'damping': 0.3,
                    'wet_level': 0.4
                },
                'description': 'Live concert recording enhancement',
                'tags': ['music', 'concert', 'live', 'recording'],
                'is_system_preset': True
            },

            # Podcast Enhancement Presets
            {
                'name': 'Podcast Standard',
                'category': 'podcast',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.12,
                    'normalization': True,
                    'target_level': -16.0,
                    'compression': True,
                    'threshold': -22.0,
                    'ratio': 4.0,
                    'equalization': True,
                    'low_gain': 2.0,
                    'mid_gain': 0.0,
                    'high_gain': 1.0
                },
                'description': 'Standard podcast audio enhancement',
                'tags': ['podcast', 'standard', 'balanced'],
                'is_system_preset': True
            },
            {
                'name': 'Interview Enhancement',
                'category': 'podcast',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.18,
                    'normalization': True,
                    'target_level': -14.0,
                    'compression': True,
                    'threshold': -28.0,
                    'ratio': 3.5,
                    'equalization': True,
                    'low_gain': 1.5,
                    'mid_gain': 1.0,
                    'high_gain': 0.5
                },
                'description': 'Enhanced clarity for interview recordings',
                'tags': ['podcast', 'interview', 'clarity'],
                'is_system_preset': True
            },

            # Audiobook Enhancement Presets
            {
                'name': 'Audiobook Narration',
                'category': 'audiobook',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.15,
                    'normalization': True,
                    'target_level': -18.0,
                    'compression': True,
                    'threshold': -25.0,
                    'ratio': 3.0,
                    'equalization': True,
                    'low_gain': 2.5,
                    'mid_gain': 1.0,
                    'high_gain': -1.0
                },
                'description': 'Optimized for audiobook narration',
                'tags': ['audiobook', 'narration', 'comfortable'],
                'is_system_preset': True
            },
            {
                'name': 'Audiobook Dramatic',
                'category': 'audiobook',
                'enhancement_type': 'full_enhancement',
                'settings': {
                    'noise_reduction': True,
                    'noise_factor': 0.1,
                    'normalization': True,
                    'target_level': -12.0,
                    'compression': True,
                    'threshold': -20.0,
                    'ratio': 4.5,
                    'equalization': True,
                    'low_gain': 1.0,
                    'mid_gain': 2.0,
                    'high_gain': 1.5,
                    'reverb': True,
                    'room_size': 0.4,
                    'damping': 0.5,
                    'wet_level': 0.3
                },
                'description': 'Dramatic enhancement for storytelling',
                'tags': ['audiobook', 'dramatic', 'storytelling'],
                'is_system_preset': True
            }
        ]

    def create_system_presets(self):
        """Create system default presets in database."""
        try:
            default_presets = self.get_default_presets()

            for preset_data in default_presets:
                # Check if preset already exists
                existing = self.db_session.query(EnhancementPreset).filter(
                    EnhancementPreset.name == preset_data['name'],
                    EnhancementPreset.is_system_preset == True
                ).first()

                if not existing:
                    preset = EnhancementPreset(
                        name=preset_data['name'],
                        description=preset_data['description'],
                        user_id=None,
                        is_system_preset=True,
                        enhancement_type=preset_data['enhancement_type'],
                        settings=preset_data['settings']
                    )

                    # Add tags as JSON metadata
                    if 'tags' in preset_data:
                        preset.settings['tags'] = preset_data['tags']

                    self.db_session.add(preset)

            self.db_session.commit()

        except Exception as e:
            self.logger.error(f"Error creating system presets: {str(e)}")
            self.db_session.rollback()
            raise AudioProcessingException(f"Failed to create system presets: {str(e)}")

    def get_presets_by_category(self, category: str, user_id: int = None) -> List[EnhancementPreset]:
        """Get presets by category.

        Args:
            category: Preset category
            user_id: User ID (optional, for user-specific presets)

        Returns:
            List of EnhancementPreset objects
        """
        try:
            query = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.is_system_preset == True
            )

            # Filter by category if specified
            if category and category != 'all':
                # For now, we'll use enhancement_type as a proxy for category
                # In a real implementation, you might have a separate category field
                category_mapping = {
                    'speech': ['full_enhancement'],
                    'music': ['full_enhancement'],
                    'podcast': ['full_enhancement'],
                    'audiobook': ['full_enhancement'],
                    'voiceover': ['full_enhancement']
                }

                if category in category_mapping:
                    query = query.filter(
                        EnhancementPreset.enhancement_type.in_(category_mapping[category])
                    )

            # Add user presets if user_id provided
            if user_id:
                user_presets = self.db_session.query(EnhancementPreset).filter(
                    EnhancementPreset.user_id == user_id
                ).all()
                system_presets = query.all()
                return system_presets + user_presets

            return query.order_by(EnhancementPreset.name).all()

        except Exception as e:
            self.logger.error(f"Error getting presets by category: {str(e)}")
            raise AudioProcessingException(f"Failed to get presets: {str(e)}")

    def create_user_preset(self, user_id: int, name: str, enhancement_type: str,
                          settings: Dict[str, Any], description: str = None) -> EnhancementPreset:
        """Create a custom preset for a user.

        Args:
            user_id: User ID
            name: Preset name
            enhancement_type: Type of enhancement
            settings: Enhancement settings
            description: Optional description

        Returns:
            Created EnhancementPreset object

        Raises:
            ValidationException: If preset name already exists for user
        """
        try:
            # Validate settings
            validated_settings = audio_enhancer.validate_enhancement_settings(enhancement_type, settings)

            # Check if preset name already exists for this user
            existing = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.name == name,
                EnhancementPreset.user_id == user_id
            ).first()

            if existing:
                raise ValidationException(f"Preset with name '{name}' already exists for this user")

            # Create preset
            preset = EnhancementPreset(
                name=name,
                description=description,
                user_id=user_id,
                is_system_preset=False,
                enhancement_type=enhancement_type,
                settings=validated_settings
            )

            self.db_session.add(preset)
            self.db_session.commit()

            return preset

        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Error creating user preset: {str(e)}")
            if isinstance(e, ValidationException):
                raise
            raise AudioProcessingException(f"Failed to create preset: {str(e)}")

    def update_user_preset(self, preset_id: int, user_id: int, updates: Dict[str, Any]) -> EnhancementPreset:
        """Update a user's custom preset.

        Args:
            preset_id: Preset ID
            user_id: User ID
            updates: Fields to update

        Returns:
            Updated EnhancementPreset object

        Raises:
            ValidationException: If preset not found or access denied
        """
        try:
            # Get preset
            preset = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.id == preset_id,
                EnhancementPreset.user_id == user_id
            ).first()

            if not preset:
                raise ValidationException("Preset not found or access denied")

            if preset.is_system_preset:
                raise ValidationException("Cannot modify system presets")

            # Update fields
            if 'name' in updates:
                # Check if new name conflicts
                existing = self.db_session.query(EnhancementPreset).filter(
                    EnhancementPreset.name == updates['name'],
                    EnhancementPreset.user_id == user_id,
                    EnhancementPreset.id != preset_id
                ).first()

                if existing:
                    raise ValidationException(f"Preset with name '{updates['name']}' already exists")

                preset.name = updates['name']

            if 'description' in updates:
                preset.description = updates['description']

            if 'settings' in updates:
                # Validate new settings
                validated_settings = audio_enhancer.validate_enhancement_settings(
                    preset.enhancement_type, updates['settings']
                )
                preset.settings = validated_settings

            self.db_session.commit()
            return preset

        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Error updating user preset: {str(e)}")
            if isinstance(e, ValidationException):
                raise
            raise AudioProcessingException(f"Failed to update preset: {str(e)}")

    def delete_user_preset(self, preset_id: int, user_id: int) -> bool:
        """Delete a user's custom preset.

        Args:
            preset_id: Preset ID
            user_id: User ID

        Returns:
            True if deleted successfully

        Raises:
            ValidationException: If preset not found or access denied
        """
        try:
            # Get preset
            preset = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.id == preset_id,
                EnhancementPreset.user_id == user_id
            ).first()

            if not preset:
                raise ValidationException("Preset not found or access denied")

            if preset.is_system_preset:
                raise ValidationException("Cannot delete system presets")

            # Delete preset
            self.db_session.delete(preset)
            self.db_session.commit()

            return True

        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Error deleting user preset: {str(e)}")
            if isinstance(e, ValidationException):
                raise
            raise AudioProcessingException(f"Failed to delete preset: {str(e)}")

    def start_ab_test(self, user_id: int, preset_a_id: int, preset_b_id: int,
                     test_name: str = None) -> Dict[str, Any]:
        """Start an A/B test between two presets.

        Args:
            user_id: User ID
            preset_a_id: First preset ID
            preset_b_id: Second preset ID
            test_name: Optional test name

        Returns:
            Dictionary with test configuration

        Raises:
            ValidationException: If presets not found or invalid
        """
        try:
            # Get presets
            preset_a = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.id == preset_a_id
            ).first()

            preset_b = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.id == preset_b_id
            ).first()

            if not preset_a or not preset_b:
                raise ValidationException("One or both presets not found")

            # Check user access
            if not preset_a.is_system_preset and preset_a.user_id != user_id:
                raise ValidationException("Access denied to preset A")

            if not preset_b.is_system_preset and preset_b.user_id != user_id:
                raise ValidationException("Access denied to preset B")

            # Create test configuration
            test_config = {
                'test_id': f"ab_test_{user_id}_{int(datetime.utcnow().timestamp())}",
                'user_id': user_id,
                'preset_a_id': preset_a_id,
                'preset_b_id': preset_b_id,
                'preset_a_name': preset_a.name,
                'preset_b_name': preset_b.name,
                'test_name': test_name or f"A/B Test: {preset_a.name} vs {preset_b.name}",
                'start_date': datetime.utcnow(),
                'end_date': datetime.utcnow() + timedelta(days=self.ab_test_duration_days),
                'status': 'active',
                'results': {
                    'preset_a': {'uses': 0, 'avg_quality_score': 0.0, 'ratings': []},
                    'preset_b': {'uses': 0, 'avg_quality_score': 0.0, 'ratings': []}
                }
            }

            # Store test configuration (in a real app, you'd have a dedicated A/B test table)
            # For now, we'll store it in a JSON field or cache
            self._store_ab_test_config(test_config)

            return test_config

        except Exception as e:
            self.logger.error(f"Error starting A/B test: {str(e)}")
            if isinstance(e, ValidationException):
                raise
            raise AudioProcessingException(f"Failed to start A/B test: {str(e)}")

    def record_ab_test_result(self, test_id: str, preset_id: int,
                             quality_score: float, user_rating: int = None) -> bool:
        """Record a result from an A/B test.

        Args:
            test_id: Test ID
            preset_id: Preset ID used
            quality_score: Quality score achieved
            user_rating: Optional user rating (1-5)

        Returns:
            True if recorded successfully
        """
        try:
            # Get test configuration
            test_config = self._get_ab_test_config(test_id)
            if not test_config:
                raise ValidationException("A/B test not found")

            if test_config['status'] != 'active':
                raise ValidationException("A/B test is not active")

            # Record result
            results = test_config['results']
            if preset_id == test_config['preset_a_id']:
                preset_key = 'preset_a'
            elif preset_id == test_config['preset_b_id']:
                preset_key = 'preset_b'
            else:
                raise ValidationException("Preset not part of this test")

            results[preset_key]['uses'] += 1
            results[preset_key]['ratings'].append(user_rating or 0)

            # Update average quality score
            if results[preset_key]['uses'] > 0:
                results[preset_key]['avg_quality_score'] = (
                    sum(r for r in results[preset_key]['ratings'] if r > 0) /
                    len([r for r in results[preset_key]['ratings'] if r > 0])
                )

            # Check if test should end
            total_uses = results['preset_a']['uses'] + results['preset_b']['uses']
            if total_uses >= self.ab_test_min_samples:
                # Perform statistical analysis
                winner = self._analyze_ab_test_results(results)
                if winner:
                    test_config['status'] = 'completed'
                    test_config['winner'] = winner

            # Save updated configuration
            self._store_ab_test_config(test_config)

            return True

        except Exception as e:
            self.logger.error(f"Error recording A/B test result: {str(e)}")
            if isinstance(e, ValidationException):
                raise
            raise AudioProcessingException(f"Failed to record test result: {str(e)}")

    def _analyze_ab_test_results(self, results: Dict[str, Any]) -> Optional[str]:
        """Analyze A/B test results and determine winner.

        Args:
            results: Test results dictionary

        Returns:
            Winner preset key ('preset_a' or 'preset_b') or None if inconclusive
        """
        try:
            a_scores = [r for r in results['preset_a']['ratings'] if r > 0]
            b_scores = [r for r in results['preset_b']['ratings'] if r > 0]

            if len(a_scores) < 3 or len(b_scores) < 3:
                return None

            # Simple statistical comparison
            a_mean = statistics.mean(a_scores)
            b_mean = statistics.mean(b_scores)

            # Calculate standard deviations
            a_std = statistics.stdev(a_scores) if len(a_scores) > 1 else 0
            b_std = statistics.stdev(b_scores) if len(b_scores) > 1 else 0

            # Determine winner based on mean scores
            if b_mean > a_mean + 0.5:  # Significant improvement
                return 'preset_b'
            elif a_mean > b_mean + 0.5:  # Significant improvement
                return 'preset_a'
            else:
                return None

        except Exception as e:
            self.logger.warning(f"Error analyzing A/B test results: {str(e)}")
            return None

    def get_preset_statistics(self, preset_id: int) -> Dict[str, Any]:
        """Get statistics for a preset.

        Args:
            preset_id: Preset ID

        Returns:
            Dictionary with preset statistics
        """
        try:
            # Get preset
            preset = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.id == preset_id
            ).first()

            if not preset:
                raise ValidationException("Preset not found")

            # Get usage statistics
            enhancements = self.db_session.query(AudioEnhancement).filter(
                AudioEnhancement.preset_id == preset_id
            ).all()

            total_uses = len(enhancements)
            successful_uses = len([e for e in enhancements if e.success])
            avg_quality_improvement = 0.0
            avg_processing_time = 0.0

            if enhancements:
                quality_improvements = [e.get_improvement_metrics().get('quality_score_improvement', 0)
                                      for e in enhancements if e.enhanced_quality_score is not None]
                if quality_improvements:
                    avg_quality_improvement = statistics.mean(quality_improvements)

                processing_times = [e.processing_time for e in enhancements if e.processing_time]
                if processing_times:
                    avg_processing_time = statistics.mean(processing_times)

            # Get rating statistics
            avg_rating = preset.rating
            rating_count = preset.rating_count

            return {
                'preset_id': preset_id,
                'preset_name': preset.name,
                'total_uses': total_uses,
                'successful_uses': successful_uses,
                'success_rate': successful_uses / total_uses if total_uses > 0 else 0,
                'avg_quality_improvement': avg_quality_improvement,
                'avg_processing_time': avg_processing_time,
                'avg_rating': avg_rating,
                'rating_count': rating_count,
                'usage_trend': self._calculate_usage_trend(enhancements)
            }

        except Exception as e:
            self.logger.error(f"Error getting preset statistics: {str(e)}")
            raise AudioProcessingException(f"Failed to get statistics: {str(e)}")

    def _calculate_usage_trend(self, enhancements: List[AudioEnhancement]) -> str:
        """Calculate usage trend for enhancements.

        Args:
            enhancements: List of enhancement records

        Returns:
            Trend description
        """
        if len(enhancements) < 2:
            return 'insufficient_data'

        # Sort by date
        sorted_enhancements = sorted(enhancements, key=lambda x: x.created_at)

        # Simple trend analysis
        recent_count = len([e for e in sorted_enhancements[-7:]])  # Last 7 days
        older_count = len(sorted_enhancements) - recent_count

        if recent_count > older_count:
            return 'increasing'
        elif recent_count < older_count:
            return 'decreasing'
        else:
            return 'stable'

    def _store_ab_test_config(self, test_config: Dict[str, Any]):
        """Store A/B test configuration (placeholder implementation)."""
        # In a real implementation, this would store to database or cache
        # For now, we'll use a simple in-memory storage
        if not hasattr(self, '_ab_tests'):
            self._ab_tests = {}

        self._ab_tests[test_config['test_id']] = test_config

    def _get_ab_test_config(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test configuration (placeholder implementation)."""
        if not hasattr(self, '_ab_tests'):
            self._ab_tests = {}

        return self._ab_tests.get(test_id)

    def get_popular_presets(self, limit: int = 10, category: str = None) -> List[EnhancementPreset]:
        """Get most popular presets.

        Args:
            limit: Maximum number of presets to return
            category: Optional category filter

        Returns:
            List of popular presets ordered by usage
        """
        try:
            query = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.usage_count > 0
            )

            # Filter by category if specified
            if category and category != 'all':
                category_mapping = {
                    'speech': ['full_enhancement'],
                    'music': ['full_enhancement'],
                    'podcast': ['full_enhancement'],
                    'audiobook': ['full_enhancement'],
                    'voiceover': ['full_enhancement']
                }

                if category in category_mapping:
                    query = query.filter(
                        EnhancementPreset.enhancement_type.in_(category_mapping[category])
                    )

            return query.order_by(
                desc(EnhancementPreset.usage_count),
                desc(EnhancementPreset.rating)
            ).limit(limit).all()

        except Exception as e:
            self.logger.error(f"Error getting popular presets: {str(e)}")
            raise AudioProcessingException(f"Failed to get popular presets: {str(e)}")

    def clone_preset(self, preset_id: int, user_id: int, new_name: str) -> EnhancementPreset:
        """Clone an existing preset for a user.

        Args:
            preset_id: Source preset ID
            user_id: User ID to create clone for
            new_name: Name for the cloned preset

        Returns:
            Cloned EnhancementPreset object
        """
        try:
            # Get source preset
            source_preset = self.db_session.query(EnhancementPreset).filter(
                EnhancementPreset.id == preset_id
            ).first()

            if not source_preset:
                raise ValidationException("Source preset not found")

            # Check if user can access source preset
            if not source_preset.is_system_preset and source_preset.user_id != user_id:
                raise ValidationException("Access denied to source preset")

            # Create clone
            clone = EnhancementPreset(
                name=new_name,
                description=f"Clone of {source_preset.name}: {source_preset.description}",
                user_id=user_id,
                is_system_preset=False,
                enhancement_type=source_preset.enhancement_type,
                settings=source_preset.settings.copy()
            )

            self.db_session.add(clone)
            self.db_session.commit()

            return clone

        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Error cloning preset: {str(e)}")
            if isinstance(e, ValidationException):
                raise
            raise AudioProcessingException(f"Failed to clone preset: {str(e)}")


# Global instance
enhancement_preset_manager = None

def get_enhancement_preset_manager(db_session: Session) -> EnhancementPresetManager:
    """Get or create enhancement preset manager instance."""
    global enhancement_preset_manager
    if enhancement_preset_manager is None:
        enhancement_preset_manager = EnhancementPresetManager(db_session)
    return enhancement_preset_manager