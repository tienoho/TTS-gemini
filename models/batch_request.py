from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class ItemStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TTSItem(BaseModel):
    """Individual TTS item trong batch"""
    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(..., description="Voice identifier")
    language: str = Field(default="vi", description="Language code")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    pitch: float = Field(default=0.0, ge=-10.0, le=10.0, description="Pitch adjustment")
    volume: float = Field(default=1.0, ge=0.1, le=2.0, description="Volume level")
    output_format: str = Field(default="mp3", description="Audio format")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


class BatchProcessingError(BaseModel):
    """Error details cho individual items"""
    item_id: UUID
    error_code: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0


class BatchItemResult(BaseModel):
    """Result cho individual TTS item"""
    item_id: UUID
    status: ItemStatus
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    error: Optional[BatchProcessingError] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchRequest(BaseModel):
    """Batch TTS request model"""
    id: UUID = Field(default_factory=uuid4)
    name: Optional[str] = Field(default=None, max_length=200, description="Batch name")
    items: List[TTSItem] = Field(..., description="List of TTS items")
    priority: Priority = Field(default=Priority.NORMAL, description="Processing priority")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for completion notification")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Batch metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    @validator('items')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 items per batch allowed')
        return v

    @validator('expires_at')
    def validate_expiry(cls, v, values):
        if v and v <= values.get('created_at', datetime.utcnow()):
            raise ValueError('Expiry time must be after creation time')
        return v


class BatchResponse(BaseModel):
    """Response model cho batch operations"""
    batch_id: UUID
    status: BatchStatus
    message: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None
    progress: Optional[Dict[str, Any]] = None


class BatchStatusResponse(BaseModel):
    """Detailed batch status response"""
    batch_id: UUID
    status: BatchStatus
    name: Optional[str]
    priority: Priority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    total_items: int
    completed_items: int
    failed_items: int
    pending_items: int
    estimated_completion: Optional[datetime] = None
    results: List[BatchItemResult] = Field(default_factory=list)
    errors: List[BatchProcessingError] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class BatchResultsResponse(BaseModel):
    """Batch results response"""
    batch_id: UUID
    status: BatchStatus
    total_items: int
    results: List[BatchItemResult]
    summary: Dict[str, Any]
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class BatchListResponse(BaseModel):
    """List of batches response"""
    batches: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    has_more: bool


class BatchProgressUpdate(BaseModel):
    """WebSocket progress update message"""
    batch_id: UUID
    type: str = "batch_progress"
    status: BatchStatus
    progress: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    item_updates: Optional[List[BatchItemResult]] = None