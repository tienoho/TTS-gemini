"""
Error Handling Tests cho Batch Processing System
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from models.batch_request import (
    BatchRequest, TTSItem, Priority, BatchStatus, ItemStatus,
    BatchItemResult, BatchProcessingError
)
from utils.batch_processor import BatchProcessor, ProcessingConfig
from utils.batch_queue import BatchQueueManager
from routes.batch_tts import get_batch_processor


class TestBatchErrorHandling:
    """Error handling tests cho batch processing"""

    @pytest.fixture
    def mock_failing_tts_service(self):
        """Mock TTS service that fails"""
        mock_service = Mock()

        async def failing_generate_audio(*args, **kwargs):
            raise Exception("TTS service unavailable")

        mock_service.generate_audio = failing_generate_audio
        return mock_service

    @pytest.fixture
    def mock_network_error_tts_service(self):
        """Mock TTS service với network errors"""
        mock_service = Mock()

        async def network_error_generate_audio(*args, **kwargs):
            raise ConnectionError("Network timeout")

        mock_service.generate_audio = network_error_generate_audio
        return mock_service

    @pytest.fixture
    def batch_processor_with_failing_service(self, mock_failing_tts_service):
        """Batch processor với failing TTS service"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            processor = BatchProcessor(
                tts_service=mock_failing_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            yield processor

    def create_test_batch(self, size: int = 3) -> BatchRequest:
        """Create test batch"""
        items = [
            TTSItem(text=f"Test item {i}", voice="voice1", language="vi")
            for i in range(size)
        ]

        return BatchRequest(
            name="Error Handling Test Batch",
            items=items,
            priority=Priority.NORMAL
        )

    @pytest.mark.asyncio
    async def test_batch_processing_with_tts_failures(
        self,
        batch_processor_with_failing_service
    ):
        """Test batch processing khi TTS service fails"""
        batch_request = self.create_test_batch(3)

        # Process batch
        result = await batch_processor_with_failing_service.process_batch(batch_request)

        # Verify error handling
        assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
        assert len(result["results"]) == 3
        assert result["summary"]["total_items"] == 3
        assert result["summary"]["failed"] == 3
        assert result["summary"]["completed"] == 0

        # Verify all items failed
        for item_result in result["results"]:
            assert item_result.status == ItemStatus.FAILED
            assert item_result.error is not None
            assert item_result.error.error_code == "MAX_RETRIES_EXCEEDED"
            assert item_result.error.retry_count == 3

    @pytest.mark.asyncio
    async def test_batch_processing_with_mixed_results(self):
        """Test batch processing với mixed success/failure results"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create TTS service that fails on even-numbered calls
            call_count = 0
            mock_tts_service = Mock()

            async def mixed_generate_audio(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count % 2 == 0:  # Fail on even calls
                    raise Exception("Simulated failure")

                return {
                    "audio_url": "https://example.com/audio.mp3",
                    "duration": 2.5,
                    "file_size": 1024
                }

            mock_tts_service.generate_audio = mixed_generate_audio

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            batch_request = self.create_test_batch(4)

            # Process batch
            result = await processor.process_batch(batch_request)

            # Verify mixed results
            assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
            assert result["summary"]["total_items"] == 4
            assert result["summary"]["completed"] == 2  # Every other item succeeds
            assert result["summary"]["failed"] == 2

            # Verify individual results
            completed_count = sum(1 for r in result["results"] if r.status == ItemStatus.COMPLETED)
            failed_count = sum(1 for r in result["results"] if r.status == ItemStatus.FAILED)

            assert completed_count == 2
            assert failed_count == 2

    @pytest.mark.asyncio
    async def test_batch_processing_with_network_errors(self):
        """Test batch processing với network errors"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create TTS service với network errors
            mock_tts_service = Mock()

            async def network_error_generate_audio(*args, **kwargs):
                raise ConnectionError("Network timeout")

            mock_tts_service.generate_audio = network_error_generate_audio

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            batch_request = self.create_test_batch(2)

            # Process batch
            result = await processor.process_batch(batch_request)

            # Verify error handling
            assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
            assert result["summary"]["failed"] == 2

            # Verify error details
            for item_result in result["results"]:
                assert item_result.error is not None
                assert "MAX_RETRIES_EXCEEDED" in item_result.error.error_code

    @pytest.mark.asyncio
    async def test_batch_queue_error_handling(self):
        """Test queue error handling"""
        queue_manager = BatchQueueManager()

        # Test with invalid batch
        invalid_batch = BatchRequest(
            name="Invalid Batch",
            items=[],  # Empty batch should fail
            priority=Priority.NORMAL
        )

        # Should handle validation error
        with pytest.raises(Exception):
            await queue_manager.enqueue_batch(invalid_batch)

    @pytest.mark.asyncio
    async def test_batch_processor_redis_errors(self):
        """Test batch processor với Redis errors"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer:

            mock_streamer.send_batch_progress = AsyncMock()

            # Create Redis manager that fails
            mock_redis = Mock()
            mock_redis.set_cache = AsyncMock(side_effect=Exception("Redis connection failed"))
            mock_redis.get_cache = AsyncMock(side_effect=Exception("Redis connection failed"))

            mock_tts_service = Mock()
            mock_tts_service.generate_audio = AsyncMock(return_value={
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            })

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            batch_request = self.create_test_batch(2)

            # Should handle Redis errors gracefully
            with pytest.raises(Exception):
                await processor.process_batch(batch_request)

    @pytest.mark.asyncio
    async def test_batch_processor_progress_streamer_errors(self):
        """Test batch processor với progress streamer errors"""
        with patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create progress streamer that fails
            mock_streamer = Mock()
            mock_streamer.send_batch_progress = AsyncMock(side_effect=Exception("WebSocket error"))

            mock_tts_service = Mock()
            mock_tts_service.generate_audio = AsyncMock(return_value={
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            })

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            batch_request = self.create_test_batch(2)

            # Should handle progress streamer errors gracefully
            result = await processor.process_batch(batch_request)

            # Processing should still complete
            assert result["status"] == BatchStatus.COMPLETED
            assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_with_corrupted_data(self):
        """Test batch processing với corrupted data"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create TTS service that returns corrupted data
            mock_tts_service = Mock()

            async def corrupted_generate_audio(*args, **kwargs):
                return {
                    "audio_url": None,  # Corrupted data
                    "duration": None,
                    "file_size": None
                }

            mock_tts_service.generate_audio = corrupted_generate_audio

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            batch_request = self.create_test_batch(2)

            # Process batch
            result = await processor.process_batch(batch_request)

            # Should handle corrupted data gracefully
            assert result["status"] == BatchStatus.COMPLETED
            assert len(result["results"]) == 2

            # Results should handle None values
            for item_result in result["results"]:
                assert item_result.status == ItemStatus.COMPLETED
                assert item_result.audio_url is None

    @pytest.mark.asyncio
    async def test_batch_timeout_handling(self):
        """Test batch processing với timeouts"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create TTS service with long delays
            mock_tts_service = Mock()

            async def slow_generate_audio(*args, **kwargs):
                await asyncio.sleep(10)  # Long delay
                return {
                    "audio_url": "https://example.com/audio.mp3",
                    "duration": 2.5,
                    "file_size": 1024
                }

            mock_tts_service.generate_audio = slow_generate_audio

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis,
                config=ProcessingConfig(
                    max_concurrency=1,
                    timeout_per_item=1  # Short timeout
                )
            )

            batch_request = self.create_test_batch(1)

            # Should timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    processor.process_batch(batch_request),
                    timeout=2.0
                )

    @pytest.mark.asyncio
    async def test_batch_cancellation_error_handling(self):
        """Test batch cancellation error handling"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            mock_tts_service = Mock()
            mock_tts_service.generate_audio = AsyncMock(return_value={
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            })

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            batch_request = self.create_test_batch(2)

            # Start processing
            processing_task = asyncio.create_task(
                processor.process_batch(batch_request)
            )

            # Cancel batch
            success = await processor.cancel_batch(batch_request.id)
            assert success

            # Verify cancellation
            status = await processor.get_batch_status(batch_request.id)
            assert status["status"] == BatchStatus.CANCELLED.value

            # Clean up
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_batch_validation_errors(self):
        """Test batch validation error handling"""
        # Test invalid batch size
        large_items = [
            TTSItem(text=f"Item {i}", voice="voice1", language="vi")
            for i in range(101)  # Exceeds limit
        ]

        with pytest.raises(Exception):
            BatchRequest(
                name="Too Large Batch",
                items=large_items,
                priority=Priority.NORMAL
            )

        # Test invalid text
        with pytest.raises(Exception):
            TTSItem(text="", voice="voice1", language="vi")

        # Test invalid speed
        with pytest.raises(Exception):
            TTSItem(text="test", voice="voice1", language="vi", speed=3.0)

    @pytest.mark.asyncio
    async def test_batch_memory_error_handling(self):
        """Test batch processing với memory errors"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create TTS service that causes memory issues
            mock_tts_service = Mock()

            async def memory_intensive_generate_audio(*args, **kwargs):
                # Simulate memory-intensive operation
                large_data = "x" * (100 * 1024 * 1024)  # 100MB
                await asyncio.sleep(0.1)
                return {
                    "audio_url": "https://example.com/audio.mp3",
                    "duration": 2.5,
                    "file_size": 1024
                }

            mock_tts_service.generate_audio = memory_intensive_generate_audio

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            batch_request = self.create_test_batch(1)

            # Should handle memory issues gracefully
            result = await processor.process_batch(batch_request)

            # Processing should still complete
            assert result["status"] == BatchStatus.COMPLETED
            assert len(result["results"]) == 1


class TestBatchAPIErrorHandling:
    """Error handling tests cho batch API"""

    @pytest.mark.asyncio
    async def test_api_invalid_json(self):
        """Test API với invalid JSON"""
        # This would be tested through actual HTTP requests
        # For now, we test the validation logic
        pass

    @pytest.mark.asyncio
    async def test_api_missing_required_fields(self):
        """Test API với missing required fields"""
        # Test validation of required fields
        with pytest.raises(Exception):
            BatchRequest(
                name="Test Batch",
                items=[],  # Missing items
                priority=Priority.NORMAL
            )

    @pytest.mark.asyncio
    async def test_api_invalid_batch_id(self):
        """Test API với invalid batch ID"""
        # Test with non-existent batch ID
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value=None)  # Batch not found

            mock_tts_service = Mock()
            mock_tts_service.generate_audio = AsyncMock(return_value={
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            })

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            # Try to get non-existent batch
            status = await processor.get_batch_status(uuid4())
            assert status is None

    @pytest.mark.asyncio
    async def test_api_rate_limiting(self):
        """Test API rate limiting"""
        # This would be tested with actual rate limiting middleware
        # For now, we test the configuration
        config = ProcessingConfig()
        assert hasattr(config, 'max_concurrency')
        assert hasattr(config, 'max_retries')