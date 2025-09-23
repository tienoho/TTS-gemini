"""
Unit tests cho Batch Request Models
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from pydantic import ValidationError

from models.batch_request import (
    TTSItem, BatchRequest, BatchStatus, Priority, ItemStatus,
    BatchProcessingError, BatchItemResult, BatchResponse,
    BatchStatusResponse, BatchResultsResponse
)


class TestTTSItem:
    """Test cases cho TTSItem model"""

    def test_valid_tts_item_creation(self):
        """Test tạo TTSItem hợp lệ"""
        item = TTSItem(
            text="Hello world",
            voice="voice1",
            language="vi",
            speed=1.0,
            pitch=0.0,
            volume=1.0,
            output_format="mp3"
        )

        assert item.text == "Hello world"
        assert item.voice == "voice1"
        assert item.language == "vi"
        assert item.speed == 1.0
        assert item.pitch == 0.0
        assert item.volume == 1.0
        assert item.output_format == "mp3"
        assert isinstance(item.id, uuid4().__class__)

    def test_tts_item_validation_text_required(self):
        """Test validation cho text field"""
        with pytest.raises(ValidationError):
            TTSItem(text="", voice="voice1", language="vi")

    def test_tts_item_validation_text_length(self):
        """Test validation cho text length"""
        # Text quá ngắn
        with pytest.raises(ValidationError):
            TTSItem(text="", voice="voice1", language="vi")

        # Text quá dài
        long_text = "a" * 5001
        with pytest.raises(ValidationError):
            TTSItem(text=long_text, voice="voice1", language="vi")

    def test_tts_item_validation_speed_range(self):
        """Test validation cho speed range"""
        # Speed quá thấp
        with pytest.raises(ValidationError):
            TTSItem(text="test", voice="voice1", language="vi", speed=0.4)

        # Speed quá cao
        with pytest.raises(ValidationError):
            TTSItem(text="test", voice="voice1", language="vi", speed=2.1)

    def test_tts_item_validation_pitch_range(self):
        """Test validation cho pitch range"""
        # Pitch quá thấp
        with pytest.raises(ValidationError):
            TTSItem(text="test", voice="voice1", language="vi", pitch=-10.1)

        # Pitch quá cao
        with pytest.raises(ValidationError):
            TTSItem(text="test", voice="voice1", language="vi", pitch=10.1)

    def test_tts_item_validation_volume_range(self):
        """Test validation cho volume range"""
        # Volume quá thấp
        with pytest.raises(ValidationError):
            TTSItem(text="test", voice="voice1", language="vi", volume=0.0)

        # Volume quá cao
        with pytest.raises(ValidationError):
            TTSItem(text="test", voice="voice1", language="vi", volume=2.1)

    def test_tts_item_default_values(self):
        """Test default values"""
        item = TTSItem(text="test", voice="voice1")

        assert item.language == "vi"
        assert item.speed == 1.0
        assert item.pitch == 0.0
        assert item.volume == 1.0
        assert item.output_format == "mp3"


class TestBatchRequest:
    """Test cases cho BatchRequest model"""

    def test_valid_batch_request_creation(self):
        """Test tạo BatchRequest hợp lệ"""
        items = [
            TTSItem(text="Hello", voice="voice1"),
            TTSItem(text="World", voice="voice2")
        ]

        batch = BatchRequest(
            name="Test Batch",
            items=items,
            priority=Priority.NORMAL,
            webhook_url="https://example.com/webhook"
        )

        assert batch.name == "Test Batch"
        assert len(batch.items) == 2
        assert batch.priority == Priority.NORMAL
        assert batch.webhook_url == "https://example.com/webhook"
        assert isinstance(batch.id, uuid4().__class__)
        assert isinstance(batch.created_at, datetime)

    def test_batch_request_validation_min_items(self):
        """Test validation cho minimum items"""
        with pytest.raises(ValidationError):
            BatchRequest(items=[])

    def test_batch_request_validation_max_items(self):
        """Test validation cho maximum items"""
        # Tạo 101 items (exceed limit)
        items = [TTSItem(text=f"Text {i}", voice="voice1") for i in range(101)]

        with pytest.raises(ValidationError):
            BatchRequest(items=items)

    def test_batch_request_priority_enum(self):
        """Test priority enum values"""
        for priority in [Priority.LOW, Priority.NORMAL, Priority.HIGH]:
            batch = BatchRequest(
                items=[TTSItem(text="test", voice="voice1")],
                priority=priority
            )
            assert batch.priority == priority

    def test_batch_request_expiry_validation(self):
        """Test expiry time validation"""
        future_time = datetime.utcnow() + timedelta(hours=1)
        past_time = datetime.utcnow() - timedelta(hours=1)

        # Valid expiry
        batch = BatchRequest(
            items=[TTSItem(text="test", voice="voice1")],
            expires_at=future_time
        )
        assert batch.expires_at == future_time

        # Invalid expiry (past time)
        with pytest.raises(ValidationError):
            BatchRequest(
                items=[TTSItem(text="test", voice="voice1")],
                expires_at=past_time
            )


class TestBatchProcessingError:
    """Test cases cho BatchProcessingError model"""

    def test_batch_processing_error_creation(self):
        """Test tạo BatchProcessingError"""
        item_id = uuid4()
        error = BatchProcessingError(
            item_id=item_id,
            error_code="TEST_ERROR",
            error_message="Test error message",
            retry_count=2
        )

        assert error.item_id == item_id
        assert error.error_code == "TEST_ERROR"
        assert error.error_message == "Test error message"
        assert error.retry_count == 2
        assert isinstance(error.timestamp, datetime)


class TestBatchItemResult:
    """Test cases cho BatchItemResult model"""

    def test_batch_item_result_success(self):
        """Test BatchItemResult cho successful processing"""
        item_id = uuid4()
        result = BatchItemResult(
            item_id=item_id,
            status=ItemStatus.COMPLETED,
            audio_url="https://example.com/audio.mp3",
            duration=2.5,
            file_size=1024,
            processing_time=1.2
        )

        assert result.item_id == item_id
        assert result.status == ItemStatus.COMPLETED
        assert result.audio_url == "https://example.com/audio.mp3"
        assert result.duration == 2.5
        assert result.file_size == 1024
        assert result.processing_time == 1.2
        assert result.error is None

    def test_batch_item_result_failure(self):
        """Test BatchItemResult cho failed processing"""
        item_id = uuid4()
        error = BatchProcessingError(
            item_id=item_id,
            error_code="PROCESSING_FAILED",
            error_message="Processing failed"
        )

        result = BatchItemResult(
            item_id=item_id,
            status=ItemStatus.FAILED,
            error=error,
            processing_time=0.5
        )

        assert result.item_id == item_id
        assert result.status == ItemStatus.FAILED
        assert result.error == error
        assert result.processing_time == 0.5
        assert result.audio_url is None


class TestBatchResponse:
    """Test cases cho BatchResponse model"""

    def test_batch_response_creation(self):
        """Test tạo BatchResponse"""
        batch_id = uuid4()
        created_at = datetime.utcnow()

        response = BatchResponse(
            batch_id=batch_id,
            status=BatchStatus.PENDING,
            message="Batch submitted successfully",
            created_at=created_at,
            estimated_completion=datetime.utcnow() + timedelta(minutes=30)
        )

        assert response.batch_id == batch_id
        assert response.status == BatchStatus.PENDING
        assert response.message == "Batch submitted successfully"
        assert response.created_at == created_at


class TestBatchStatusResponse:
    """Test cases cho BatchStatusResponse model"""

    def test_batch_status_response_creation(self):
        """Test tạo BatchStatusResponse"""
        batch_id = uuid4()
        created_at = datetime.utcnow()

        response = BatchStatusResponse(
            batch_id=batch_id,
            status=BatchStatus.PROCESSING,
            name="Test Batch",
            priority=Priority.HIGH,
            created_at=created_at,
            total_items=10,
            completed_items=3,
            failed_items=0,
            pending_items=7
        )

        assert response.batch_id == batch_id
        assert response.status == BatchStatus.PROCESSING
        assert response.name == "Test Batch"
        assert response.priority == Priority.HIGH
        assert response.total_items == 10
        assert response.completed_items == 3
        assert response.failed_items == 0
        assert response.pending_items == 7


class TestBatchResultsResponse:
    """Test cases cho BatchResultsResponse model"""

    def test_batch_results_response_creation(self):
        """Test tạo BatchResultsResponse"""
        batch_id = uuid4()
        results = [
            BatchItemResult(
                item_id=uuid4(),
                status=ItemStatus.COMPLETED,
                audio_url="https://example.com/audio1.mp3"
            ),
            BatchItemResult(
                item_id=uuid4(),
                status=ItemStatus.FAILED,
                error=BatchProcessingError(
                    item_id=uuid4(),
                    error_code="ERROR",
                    error_message="Failed"
                )
            )
        ]

        summary = {
            "total_items": 2,
            "completed": 1,
            "failed": 1,
            "success_rate": 50.0
        }

        response = BatchResultsResponse(
            batch_id=batch_id,
            status=BatchStatus.PARTIALLY_COMPLETED,
            total_items=2,
            results=results,
            summary=summary,
            download_url="https://example.com/download"
        )

        assert response.batch_id == batch_id
        assert response.status == BatchStatus.PARTIALLY_COMPLETED
        assert response.total_items == 2
        assert len(response.results) == 2
        assert response.summary == summary
        assert response.download_url == "https://example.com/download"


class TestEnumValues:
    """Test cases cho enum values"""

    def test_priority_enum(self):
        """Test Priority enum"""
        assert Priority.LOW == "low"
        assert Priority.NORMAL == "normal"
        assert Priority.HIGH == "high"

    def test_batch_status_enum(self):
        """Test BatchStatus enum"""
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.CANCELLED == "cancelled"
        assert BatchStatus.PARTIALLY_COMPLETED == "partially_completed"

    def test_item_status_enum(self):
        """Test ItemStatus enum"""
        assert ItemStatus.PENDING == "pending"
        assert ItemStatus.PROCESSING == "processing"
        assert ItemStatus.COMPLETED == "completed"
        assert ItemStatus.FAILED == "failed"
        assert ItemStatus.SKIPPED == "skipped"