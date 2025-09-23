"""
Unit tests cho Batch Processing Service
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from utils.batch_processor import BatchProcessor, ProcessingConfig
from models.batch_request import (
    BatchRequest, TTSItem, BatchStatus, ItemStatus,
    BatchItemResult, BatchProcessingError, Priority
)


class TestProcessingConfig:
    """Test cases cho ProcessingConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ProcessingConfig()

        assert config.max_concurrency == 5
        assert config.chunk_size == 10
        assert config.timeout_per_item == 300
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.progress_update_interval == 1.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = ProcessingConfig(
            max_concurrency=10,
            chunk_size=20,
            timeout_per_item=600,
            max_retries=5,
            retry_delay=2.0,
            progress_update_interval=0.5
        )

        assert config.max_concurrency == 10
        assert config.chunk_size == 20
        assert config.timeout_per_item == 600
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.progress_update_interval == 0.5


class TestBatchProcessor:
    """Test cases cho BatchProcessor"""

    @pytest.fixture
    def mock_tts_service(self):
        """Mock TTS service"""
        mock_service = Mock()
        mock_service.generate_audio = Mock(return_value={
            "audio_url": "https://example.com/audio.mp3",
            "duration": 2.5,
            "file_size": 1024
        })
        return mock_service

    @pytest.fixture
    def mock_progress_streamer(self):
        """Mock progress streamer"""
        mock_streamer = Mock()
        mock_streamer.send_batch_progress = AsyncMock()
        return mock_streamer

    @pytest.fixture
    def mock_redis_manager(self):
        """Mock Redis manager"""
        mock_redis = Mock()
        mock_redis.set_cache = AsyncMock()
        mock_redis.get_cache = AsyncMock(return_value={})
        mock_redis.update_json_field = AsyncMock()
        return mock_redis

    @pytest.fixture
    def batch_processor(self, mock_tts_service, mock_progress_streamer, mock_redis_manager):
        """Batch processor instance với mocks"""
        return BatchProcessor(
            tts_service=mock_tts_service,
            progress_streamer=mock_progress_streamer,
            redis_manager=mock_redis_manager
        )

    @pytest.fixture
    def sample_batch_request(self):
        """Sample batch request cho testing"""
        items = [
            TTSItem(text="Hello world", voice="voice1", language="vi"),
            TTSItem(text="Goodbye world", voice="voice2", language="vi"),
            TTSItem(text="Test message", voice="voice1", language="vi")
        ]

        return BatchRequest(
            name="Test Batch",
            items=items,
            priority=Priority.NORMAL
        )

    def test_batch_processor_initialization(self, batch_processor):
        """Test batch processor initialization"""
        assert batch_processor.config.max_concurrency == 5
        assert batch_processor.config.chunk_size == 10
        assert isinstance(batch_processor.executor, type(batch_processor.executor))
        assert len(batch_processor.active_batches) == 0

    def test_create_chunks(self, batch_processor):
        """Test chunk creation"""
        items = [TTSItem(text=f"Text {i}", voice="voice1") for i in range(25)]

        chunks = batch_processor._create_chunks(items, 10)

        assert len(chunks) == 3  # 25 items / 10 chunk_size = 3 chunks
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 5

    def test_generate_summary(self, batch_processor):
        """Test summary generation"""
        results = [
            BatchItemResult(
                item_id=uuid4(),
                status=ItemStatus.COMPLETED,
                processing_time=1.0
            ),
            BatchItemResult(
                item_id=uuid4(),
                status=ItemStatus.COMPLETED,
                processing_time=2.0
            ),
            BatchItemResult(
                item_id=uuid4(),
                status=ItemStatus.FAILED,
                error=BatchProcessingError(
                    item_id=uuid4(),
                    error_code="ERROR",
                    error_message="Failed"
                ),
                processing_time=0.5
            )
        ]

        summary = batch_processor._generate_summary(results)

        assert summary["total_items"] == 3
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert summary["skipped"] == 0
        assert summary["success_rate"] == pytest.approx(66.67)
        assert summary["average_processing_time"] == pytest.approx(1.17)
        assert summary["total_processing_time"] == 3.5

    @patch('asyncio.create_task')
    async def test_process_batch_success(self, mock_create_task, batch_processor, sample_batch_request):
        """Test successful batch processing"""
        # Setup mocks
        batch_processor.redis_manager.set_cache = AsyncMock()
        batch_processor.redis_manager.get_cache = AsyncMock(return_value={})
        batch_processor.progress_streamer.send_batch_progress = AsyncMock()

        # Process batch
        result = await batch_processor.process_batch(sample_batch_request)

        # Verify results
        assert result["batch_id"] == sample_batch_request.id
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == 3
        assert result["summary"]["total_items"] == 3
        assert result["summary"]["completed"] == 3

        # Verify Redis operations
        assert batch_processor.redis_manager.set_cache.call_count >= 3

        # Verify progress updates
        assert batch_processor.progress_streamer.send_batch_progress.call_count >= 2

    async def test_process_batch_with_failures(self, batch_processor, sample_batch_request):
        """Test batch processing với failures"""
        # Setup mock to raise exception
        batch_processor.tts_service.generate_audio = Mock(
            side_effect=Exception("Processing failed")
        )

        batch_processor.redis_manager.set_cache = AsyncMock()
        batch_processor.redis_manager.get_cache = AsyncMock(return_value={})
        batch_processor.progress_streamer.send_batch_progress = AsyncMock()

        # Process batch
        result = await batch_processor.process_batch(sample_batch_request)

        # Verify results
        assert result["batch_id"] == sample_batch_request.id
        assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
        assert len(result["results"]) == 3
        assert result["summary"]["total_items"] == 3
        assert result["summary"]["failed"] == 3
        assert result["summary"]["success_rate"] == 0.0

    async def test_process_single_item_success(self, batch_processor, sample_batch_request):
        """Test single item processing success"""
        item = sample_batch_request.items[0]

        result = await asyncio.get_event_loop().run_in_executor(
            batch_processor.executor,
            batch_processor._process_single_item,
            sample_batch_request.id,
            item
        )

        assert result.item_id == item.id
        assert result.status == ItemStatus.COMPLETED
        assert result.audio_url == "https://example.com/audio.mp3"
        assert result.duration == 2.5
        assert result.file_size == 1024
        assert result.processing_time > 0

    async def test_process_single_item_with_retries(self, batch_processor, sample_batch_request):
        """Test single item processing với retries"""
        item = sample_batch_request.items[0]

        # Setup mock to fail first 2 times, succeed on 3rd
        call_count = 0
        def mock_generate_audio(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            }

        batch_processor.tts_service.generate_audio = Mock(side_effect=mock_generate_audio)

        result = await asyncio.get_event_loop().run_in_executor(
            batch_processor.executor,
            batch_processor._process_single_item,
            sample_batch_request.id,
            item
        )

        assert result.item_id == item.id
        assert result.status == ItemStatus.COMPLETED
        assert call_count == 3  # 2 failures + 1 success

    async def test_process_single_item_max_retries_exceeded(self, batch_processor, sample_batch_request):
        """Test single item processing when max retries exceeded"""
        item = sample_batch_request.items[0]

        # Setup mock to always fail
        batch_processor.tts_service.generate_audio = Mock(
            side_effect=Exception("Persistent failure")
        )

        result = await asyncio.get_event_loop().run_in_executor(
            batch_processor.executor,
            batch_processor._process_single_item,
            sample_batch_request.id,
            item
        )

        assert result.item_id == item.id
        assert result.status == ItemStatus.FAILED
        assert result.error is not None
        assert result.error.error_code == "MAX_RETRIES_EXCEEDED"
        assert result.error.retry_count == 3

    async def test_get_batch_status(self, batch_processor):
        """Test get batch status"""
        batch_id = uuid4()
        expected_data = {
            "id": str(batch_id),
            "status": "processing",
            "total_items": 5,
            "completed_items": 2
        }

        batch_processor.redis_manager.get_cache = AsyncMock(return_value=expected_data)

        result = await batch_processor.get_batch_status(batch_id)

        assert result == expected_data
        batch_processor.redis_manager.get_cache.assert_called_once_with(f"batch:{batch_id}")

    async def test_cancel_batch(self, batch_processor):
        """Test cancel batch"""
        batch_id = uuid4()
        batch_data = {
            "id": str(batch_id),
            "status": "processing",
            "total_items": 5
        }

        batch_processor.redis_manager.get_cache = AsyncMock(return_value=batch_data)
        batch_processor.redis_manager.update_json_field = AsyncMock()

        result = await batch_processor.cancel_batch(batch_id)

        assert result == True
        batch_processor.redis_manager.update_json_field.assert_called_once()

    async def test_cancel_nonexistent_batch(self, batch_processor):
        """Test cancel nonexistent batch"""
        batch_id = uuid4()

        batch_processor.redis_manager.get_cache = AsyncMock(return_value=None)

        result = await batch_processor.cancel_batch(batch_id)

        assert result == False

    async def test_cancel_completed_batch(self, batch_processor):
        """Test cancel completed batch"""
        batch_id = uuid4()
        batch_data = {
            "id": str(batch_id),
            "status": "completed",
            "total_items": 5
        }

        batch_processor.redis_manager.get_cache = AsyncMock(return_value=batch_data)

        result = await batch_processor.cancel_batch(batch_id)

        assert result == False

    def test_update_config(self, batch_processor):
        """Test update configuration"""
        new_config = ProcessingConfig(
            max_concurrency=10,
            chunk_size=20
        )

        # Mock executor shutdown
        batch_processor.executor.shutdown = Mock()

        batch_processor.update_config(new_config)

        assert batch_processor.config.max_concurrency == 10
        assert batch_processor.config.chunk_size == 20
        batch_processor.executor.shutdown.assert_called_once()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_process_batch_items_with_chunks(self, mock_sleep, batch_processor, sample_batch_request):
        """Test processing batch items với chunking"""
        batch_processor.redis_manager.set_cache = AsyncMock()
        batch_processor.redis_manager.get_cache = AsyncMock(return_value={})
        batch_processor.progress_streamer.send_batch_progress = AsyncMock()

        # Process batch
        results = await batch_processor._process_batch_items(sample_batch_request)

        assert len(results) == 3
        mock_sleep.assert_called_once()

    async def test_initialize_batch(self, batch_processor, sample_batch_request):
        """Test batch initialization"""
        batch_processor.redis_manager.set_cache = AsyncMock()

        await batch_processor._initialize_batch(sample_batch_request)

        # Verify Redis storage
        batch_processor.redis_manager.set_cache.assert_called_once()
        call_args = batch_processor.redis_manager.set_cache.call_args[0]

        assert call_args[0] == f"batch:{sample_batch_request.id}"
        stored_data = call_args[1]
        assert stored_data["id"] == str(sample_batch_request.id)
        assert stored_data["total_items"] == 3
        assert stored_data["status"] == "pending"

        # Verify active batch tracking
        assert sample_batch_request.id in batch_processor.active_batches

    async def test_finalize_batch(self, batch_processor, sample_batch_request):
        """Test batch finalization"""
        batch_id = sample_batch_request.id

        # Setup initial batch data
        initial_data = {
            "id": str(batch_id),
            "status": "processing",
            "total_items": 3,
            "results": []
        }
        batch_processor.redis_manager.get_cache = AsyncMock(return_value=initial_data)
        batch_processor.redis_manager.set_cache = AsyncMock()

        # Create sample results
        results = [
            BatchItemResult(
                item_id=uuid4(),
                status=ItemStatus.COMPLETED,
                processing_time=1.0
            ),
            BatchItemResult(
                item_id=uuid4(),
                status=ItemStatus.FAILED,
                error=BatchProcessingError(
                    item_id=uuid4(),
                    error_code="ERROR",
                    error_message="Failed"
                ),
                processing_time=0.5
            )
        ]

        await batch_processor._finalize_batch(batch_id, results)

        # Verify Redis updates
        assert batch_processor.redis_manager.set_cache.call_count >= 2

        # Verify cleanup
        assert batch_id not in batch_processor.active_batches