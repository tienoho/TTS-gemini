"""
Integration tests cho Batch Processing với WebSocket
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from models.batch_request import (
    BatchRequest, TTSItem, Priority, BatchStatus, ItemStatus,
    BatchItemResult, BatchProcessingError
)
from utils.batch_processor import BatchProcessor, ProcessingConfig
from utils.batch_queue import BatchQueueManager
from utils.progress_streamer import progress_streamer
from routes.batch_tts import get_batch_processor


class TestBatchWebSocketIntegration:
    """Integration tests cho batch processing với WebSocket"""

    @pytest.fixture
    async def mock_websocket_client(self):
        """Mock WebSocket client"""
        client = Mock()
        client.sid = "test_session_123"
        client.send = AsyncMock()
        return client

    @pytest.fixture
    async def batch_processor_with_mocks(self):
        """Batch processor với mocked dependencies"""
        with patch('utils.batch_processor.GeminiTTS') as mock_tts, \
             patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            # Setup mocks
            mock_tts.generate_audio = Mock(return_value={
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            })

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            processor = BatchProcessor(
                tts_service=mock_tts,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            yield processor

    @pytest.fixture
    def sample_batch_request(self):
        """Sample batch request"""
        items = [
            TTSItem(text="Hello world 1", voice="voice1", language="vi"),
            TTSItem(text="Hello world 2", voice="voice2", language="vi"),
            TTSItem(text="Hello world 3", voice="voice1", language="vi")
        ]

        return BatchRequest(
            name="Integration Test Batch",
            items=items,
            priority=Priority.NORMAL
        )

    async def test_batch_progress_streaming(
        self,
        batch_processor_with_mocks,
        sample_batch_request,
        mock_websocket_client
    ):
        """Test batch progress streaming qua WebSocket"""
        # Subscribe client to batch updates
        success = progress_streamer.subscribe_to_batch(
            str(sample_batch_request.id),
            mock_websocket_client.sid
        )
        assert success

        # Process batch
        result = await batch_processor_with_mocks.process_batch(sample_batch_request)

        # Verify batch completed successfully
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == 3

        # Verify progress updates were sent
        assert batch_processor_with_mocks.progress_streamer.send_batch_progress.call_count >= 2

        # Verify WebSocket messages were sent
        # Note: In real implementation, this would be verified through WebSocket manager

    async def test_batch_item_progress_updates(
        self,
        batch_processor_with_mocks,
        sample_batch_request
    ):
        """Test individual item progress updates"""
        # Setup mock to track item updates
        item_update_calls = []

        async def mock_send_batch_progress(batch_id, status, progress_data, item_updates=None):
            if item_updates:
                item_update_calls.extend(item_updates)

        batch_processor_with_mocks.progress_streamer.send_batch_progress = mock_send_batch_progress

        # Process batch
        result = await batch_processor_with_mocks.process_batch(sample_batch_request)

        # Verify all items completed
        assert all(r.status == ItemStatus.COMPLETED for r in result["results"])

    async def test_batch_queue_integration(
        self,
        batch_processor_with_mocks,
        sample_batch_request
    ):
        """Test batch queue integration"""
        # Create queue manager
        queue_manager = BatchQueueManager()

        # Enqueue batch
        success = await queue_manager.enqueue_batch(sample_batch_request)
        assert success

        # Dequeue batch
        queue_item = await queue_manager.dequeue_highest_priority()
        assert queue_item is not None
        assert queue_item.batch_request.id == sample_batch_request.id

        # Process dequeued batch
        result = await batch_processor_with_mocks.process_batch(queue_item.batch_request)

        # Verify processing completed
        assert result["status"] == BatchStatus.COMPLETED

    async def test_batch_status_api_integration(
        self,
        batch_processor_with_mocks,
        sample_batch_request
    ):
        """Test batch status API integration"""
        # Process batch
        await batch_processor_with_mocks.process_batch(sample_batch_request)

        # Get batch status
        status = await batch_processor_with_mocks.get_batch_status(sample_batch_request.id)

        assert status is not None
        assert status["id"] == str(sample_batch_request.id)
        assert status["status"] == BatchStatus.COMPLETED.value
        assert status["total_items"] == 3
        assert status["completed_items"] == 3

    async def test_batch_cancellation_integration(
        self,
        batch_processor_with_mocks,
        sample_batch_request
    ):
        """Test batch cancellation integration"""
        # Start processing
        processing_task = asyncio.create_task(
            batch_processor_with_mocks.process_batch(sample_batch_request)
        )

        # Cancel batch
        success = await batch_processor_with_mocks.cancel_batch(sample_batch_request.id)
        assert success

        # Verify cancellation
        status = await batch_processor_with_mocks.get_batch_status(sample_batch_request.id)
        assert status["status"] == BatchStatus.CANCELLED.value

        # Clean up
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass

    async def test_batch_with_mixed_results(
        self,
        batch_processor_with_mocks
    ):
        """Test batch processing với mixed success/failure results"""
        # Create batch with items that will have different outcomes
        items = [
            TTSItem(text="Success item 1", voice="voice1", language="vi"),
            TTSItem(text="Success item 2", voice="voice2", language="vi"),
            TTSItem(text="Failure item", voice="voice1", language="vi")  # This will fail
        ]

        batch_request = BatchRequest(
            name="Mixed Results Batch",
            items=items,
            priority=Priority.NORMAL
        )

        # Setup TTS service to fail on third item
        call_count = 0
        def mock_generate_audio(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Fail on third call
                raise Exception("Simulated failure")
            return {
                "audio_url": f"https://example.com/audio{call_count}.mp3",
                "duration": 2.5,
                "file_size": 1024
            }

        batch_processor_with_mocks.tts_service.generate_audio = mock_generate_audio

        # Process batch
        result = await batch_processor_with_mocks.process_batch(batch_request)

        # Verify mixed results
        assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
        assert result["summary"]["total_items"] == 3
        assert result["summary"]["completed"] == 2
        assert result["summary"]["failed"] == 1
        assert result["summary"]["success_rate"] == pytest.approx(66.67)

    async def test_batch_progress_with_real_time_updates(
        self,
        batch_processor_with_mocks,
        sample_batch_request,
        mock_websocket_client
    ):
        """Test real-time progress updates"""
        # Subscribe to batch updates
        success = progress_streamer.subscribe_to_batch(
            str(sample_batch_request.id),
            mock_websocket_client.sid
        )
        assert success

        # Setup progress tracking
        progress_updates = []

        async def capture_progress(batch_id, status, progress_data, item_updates=None):
            progress_updates.append({
                "batch_id": batch_id,
                "status": status,
                "progress": progress_data,
                "item_updates": item_updates
            })

        batch_processor_with_mocks.progress_streamer.send_batch_progress = capture_progress

        # Process batch
        result = await batch_processor_with_mocks.process_batch(sample_batch_request)

        # Verify progress updates were captured
        assert len(progress_updates) >= 2  # At least initialization and completion

        # Verify first update is initialization
        first_update = progress_updates[0]
        assert first_update["status"] == BatchStatus.PROCESSING
        assert first_update["progress"]["total_items"] == 3

        # Verify final update is completion
        final_update = progress_updates[-1]
        assert final_update["status"] == BatchStatus.COMPLETED
        assert final_update["progress"]["completed"] == 3

    async def test_batch_queue_priority_integration(
        self,
        batch_processor_with_mocks
    ):
        """Test priority queue integration"""
        queue_manager = BatchQueueManager()

        # Create batches with different priorities
        high_priority_batch = BatchRequest(
            name="High Priority Batch",
            items=[TTSItem(text="High priority", voice="voice1", language="vi")],
            priority=Priority.HIGH
        )

        normal_priority_batch = BatchRequest(
            name="Normal Priority Batch",
            items=[TTSItem(text="Normal priority", voice="voice1", language="vi")],
            priority=Priority.NORMAL
        )

        low_priority_batch = BatchRequest(
            name="Low Priority Batch",
            items=[TTSItem(text="Low priority", voice="voice1", language="vi")],
            priority=Priority.LOW
        )

        # Enqueue batches
        await queue_manager.enqueue_batch(high_priority_batch)
        await queue_manager.enqueue_batch(normal_priority_batch)
        await queue_manager.enqueue_batch(low_priority_batch)

        # Verify dequeue order (high priority first)
        dequeued_high = await queue_manager.dequeue_highest_priority()
        assert dequeued_high.batch_request.priority == Priority.HIGH

        dequeued_normal = await queue_manager.dequeue_highest_priority()
        assert dequeued_normal.batch_request.priority == Priority.NORMAL

        dequeued_low = await queue_manager.dequeue_highest_priority()
        assert dequeued_low.batch_request.priority == Priority.LOW

    async def test_batch_error_handling_integration(
        self,
        batch_processor_with_mocks,
        sample_batch_request
    ):
        """Test error handling integration"""
        # Setup TTS service to fail
        batch_processor_with_mocks.tts_service.generate_audio = Mock(
            side_effect=Exception("Network error")
        )

        # Process batch
        result = await batch_processor_with_mocks.process_batch(sample_batch_request)

        # Verify error handling
        assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
        assert len(result["results"]) == 3
        assert all(r.status == ItemStatus.FAILED for r in result["results"])
        assert all(r.error is not None for r in result["results"])

        # Verify error details
        for result_item in result["results"]:
            assert result_item.error.error_code == "MAX_RETRIES_EXCEEDED"
            assert result_item.error.retry_count == 3

    async def test_batch_config_integration(
        self,
        batch_processor_with_mocks,
        sample_batch_request
    ):
        """Test configuration integration"""
        # Test with custom config
        custom_config = ProcessingConfig(
            max_concurrency=2,
            chunk_size=2,
            max_retries=1
        )

        batch_processor_with_mocks.update_config(custom_config)

        assert batch_processor_with_mocks.config.max_concurrency == 2
        assert batch_processor_with_mocks.config.chunk_size == 2
        assert batch_processor_with_mocks.config.max_retries == 1

        # Process batch with new config
        result = await batch_processor_with_mocks.process_batch(sample_batch_request)

        # Verify processing completed
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == 3


class TestBatchWebSocketEdgeCases:
    """Test edge cases cho batch WebSocket integration"""

    async def test_batch_with_empty_items(self):
        """Test batch với empty items list"""
        # This should be handled by Pydantic validation at creation time
        with pytest.raises(Exception):
            BatchRequest(
                name="Empty Batch",
                items=[],
                priority=Priority.NORMAL
            )

    async def test_batch_with_max_items(self):
        """Test batch với maximum items"""
        # Create 100 items (max allowed)
        items = [
            TTSItem(text=f"Item {i}", voice="voice1", language="vi")
            for i in range(100)
        ]

        batch_request = BatchRequest(
            name="Max Items Batch",
            items=items,
            priority=Priority.NORMAL
        )

        # Should be valid
        assert len(batch_request.items) == 100

    async def test_concurrent_batch_processing(
        self,
        batch_processor_with_mocks
    ):
        """Test concurrent batch processing"""
        # Create multiple batches
        batches = []
        for i in range(3):
            items = [
                TTSItem(text=f"Batch {i} Item {j}", voice="voice1", language="vi")
                for j in range(2)
            ]
            batch = BatchRequest(
                name=f"Concurrent Batch {i}",
                items=items,
                priority=Priority.NORMAL
            )
            batches.append(batch)

        # Process batches concurrently
        tasks = [
            batch_processor_with_mocks.process_batch(batch)
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)

        # Verify all batches completed
        assert len(results) == 3
        assert all(r["status"] == BatchStatus.COMPLETED for r in results)
        assert all(len(r["results"]) == 2 for r in results)

    async def test_batch_progress_with_network_issues(
        self,
        batch_processor_with_mocks,
        sample_batch_request
    ):
        """Test batch progress với network issues"""
        # Setup intermittent failures
        call_count = 0
        def mock_generate_audio(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Fail every other call
                raise Exception("Network timeout")
            return {
                "audio_url": f"https://example.com/audio{call_count}.mp3",
                "duration": 2.5,
                "file_size": 1024
            }

        batch_processor_with_mocks.tts_service.generate_audio = mock_generate_audio

        # Process batch
        result = await batch_processor_with_mocks.process_batch(sample_batch_request)

        # Verify partial success
        assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
        assert result["summary"]["total_items"] == 3
        assert result["summary"]["completed"] == 2  # Every other item succeeds
        assert result["summary"]["failed"] == 1

    async def test_batch_memory_management(
        self,
        batch_processor_with_mocks
    ):
        """Test memory management cho large batches"""
        # Create large batch
        items = [
            TTSItem(text=f"Large batch item {i}", voice="voice1", language="vi")
            for i in range(50)
        ]

        large_batch = BatchRequest(
            name="Large Batch Test",
            items=items,
            priority=Priority.NORMAL
        )

        # Process large batch
        result = await batch_processor_with_mocks.process_batch(large_batch)

        # Verify processing completed without memory issues
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == 50
        assert result["summary"]["total_items"] == 50
        assert result["summary"]["completed"] == 50

        # Verify cleanup
        assert large_batch.id not in batch_processor_with_mocks.active_batches