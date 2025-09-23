"""
Simple Performance tests cho Batch Processing
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from models.batch_request import (
    BatchRequest, TTSItem, Priority, BatchStatus
)
from utils.batch_processor import BatchProcessor, ProcessingConfig


class TestBatchPerformance:
    """Simple performance tests cho batch processing"""

    @pytest.fixture
    def mock_tts_service(self):
        """Mock TTS service"""
        mock_service = Mock()

        async def generate_audio(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing
            return {
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            }

        mock_service.generate_audio = generate_audio
        return mock_service

    @pytest.fixture
    def batch_processor(self, mock_tts_service):
        """Batch processor với mocks"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            yield processor

    def create_batch(self, size: int) -> BatchRequest:
        """Create test batch"""
        items = [
            TTSItem(text=f"Item {i}", voice="voice1", language="vi")
            for i in range(size)
        ]

        return BatchRequest(
            name="Performance Test",
            items=items,
            priority=Priority.NORMAL
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("batch_size", [10, 25, 50])
    async def test_batch_processing_performance(self, batch_processor, batch_size):
        """Test performance với different batch sizes"""
        batch_request = self.create_batch(batch_size)

        start_time = time.time()
        result = await batch_processor.process_batch(batch_request)
        end_time = time.time()

        processing_time = end_time - start_time

        # Basic assertions
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == batch_size
        assert result["summary"]["completed"] == batch_size

        # Performance checks
        if batch_size <= 25:
            assert processing_time < 5.0, "Batch took too long"
        elif batch_size <= 50:
            assert processing_time < 10.0, "Batch took too long"

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_tts_service):
        """Test concurrent batch processing"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create multiple processors
            processors = []
            for i in range(2):
                processor = BatchProcessor(
                    tts_service=mock_tts_service,
                    progress_streamer=mock_streamer,
                    redis_manager=mock_redis
                )
                processors.append(processor)

            # Create batches
            batches = [self.create_batch(10) for i in range(2)]

            # Process concurrently
            tasks = [
                processor.process_batch(batch)
                for processor, batch in zip(processors, batches)
            ]

            results = await asyncio.gather(*tasks)

            # Verify results
            assert len(results) == 2
            assert all(r["status"] == BatchStatus.COMPLETED for r in results)
            assert all(len(r["results"]) == 10 for r in results)