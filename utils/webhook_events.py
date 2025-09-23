"""
Webhook Events Management
"""
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging

from config.webhook import EVENT_CONFIGS, webhook_config
from models.webhook import WebhookEventType

logger = logging.getLogger(__name__)

class EventPriority(str, Enum):
    """Priority levels cho events"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class WebhookEvent:
    """Class đại diện cho một webhook event"""

    def __init__(self, event_type: str, data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL):
        self.event_type = event_type
        self.data = data
        self.priority = priority
        self.timestamp = datetime.utcnow()
        self.id = f"{event_type}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "type": self.event_type,
            "data": self.data,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat()
        }

    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict(), default=str)

class EventTemplate:
    """Template cho các loại event khác nhau"""

    @staticmethod
    def tts_completed(request_id: str, audio_url: str, duration: float, text_length: int, **kwargs) -> Dict[str, Any]:
        """Template cho TTS completed event"""
        data = {
            "request_id": request_id,
            "audio_url": audio_url,
            "duration": duration,
            "text_length": text_length,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def tts_error(request_id: str, error_code: str, error_message: str, text_length: int = None, **kwargs) -> Dict[str, Any]:
        """Template cho TTS error event"""
        data = {
            "request_id": request_id,
            "error_code": error_code,
            "error_message": error_message,
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }
        if text_length:
            data["text_length"] = text_length
        data.update(kwargs)
        return data

    @staticmethod
    def batch_completed(batch_id: str, total_requests: int, successful_requests: int, failed_requests: int, **kwargs) -> Dict[str, Any]:
        """Template cho batch completed event"""
        data = {
            "batch_id": batch_id,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def batch_error(batch_id: str, error_code: str, error_message: str, failed_requests: int, **kwargs) -> Dict[str, Any]:
        """Template cho batch error event"""
        data = {
            "batch_id": batch_id,
            "error_code": error_code,
            "error_message": error_message,
            "failed_requests": failed_requests,
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def quality_enhancement_completed(request_id: str, enhancement_type: str, original_quality: float, enhanced_quality: float, **kwargs) -> Dict[str, Any]:
        """Template cho quality enhancement completed event"""
        data = {
            "request_id": request_id,
            "enhancement_type": enhancement_type,
            "original_quality": original_quality,
            "enhanced_quality": enhanced_quality,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def quality_enhancement_error(request_id: str, enhancement_type: str, error_code: str, error_message: str, **kwargs) -> Dict[str, Any]:
        """Template cho quality enhancement error event"""
        data = {
            "request_id": request_id,
            "enhancement_type": enhancement_type,
            "error_code": error_code,
            "error_message": error_message,
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def voice_cloning_completed(cloning_id: str, voice_name: str, quality_score: float, training_duration: float, **kwargs) -> Dict[str, Any]:
        """Template cho voice cloning completed event"""
        data = {
            "cloning_id": cloning_id,
            "voice_name": voice_name,
            "quality_score": quality_score,
            "training_duration": training_duration,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def voice_cloning_error(cloning_id: str, error_code: str, error_message: str, **kwargs) -> Dict[str, Any]:
        """Template cho voice cloning error event"""
        data = {
            "cloning_id": cloning_id,
            "error_code": error_code,
            "error_message": error_message,
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def audio_enhancement_completed(request_id: str, enhancement_type: str, improvement_score: float, **kwargs) -> Dict[str, Any]:
        """Template cho audio enhancement completed event"""
        data = {
            "request_id": request_id,
            "enhancement_type": enhancement_type,
            "improvement_score": improvement_score,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

    @staticmethod
    def audio_enhancement_error(request_id: str, enhancement_type: str, error_code: str, error_message: str, **kwargs) -> Dict[str, Any]:
        """Template cho audio enhancement error event"""
        data = {
            "request_id": request_id,
            "enhancement_type": enhancement_type,
            "error_code": error_code,
            "error_message": error_message,
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }
        data.update(kwargs)
        return data

class EventFilter:
    """Filter và routing cho events"""

    def __init__(self):
        self.filters: Dict[str, List[Callable]] = {}

    def add_filter(self, event_type: str, filter_func: Callable[[Dict[str, Any]], bool]):
        """Thêm filter cho event type"""
        if event_type not in self.filters:
            self.filters[event_type] = []
        self.filters[event_type].append(filter_func)

    def should_send_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Kiểm tra xem event có nên được gửi không"""
        if event_type not in self.filters:
            return True  # No filters, send by default

        for filter_func in self.filters[event_type]:
            try:
                if not filter_func(data):
                    return False
            except Exception as e:
                logger.error(f"Event filter error for {event_type}: {str(e)}")
                return False

        return True

    def get_filtered_data(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Lấy data đã được filter"""
        # Apply field filtering based on event config
        if event_type in EVENT_CONFIGS:
            config = EVENT_CONFIGS[event_type]
            required_fields = config.get("required_fields", [])
            optional_fields = config.get("optional_fields", [])

            filtered_data = {}
            for field in required_fields + optional_fields:
                if field in data:
                    filtered_data[field] = data[field]

            return filtered_data
        return data

# Global event filter instance
event_filter = EventFilter()

class EventManager:
    """Manager cho webhook events"""

    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue = []

    def create_event(self, event_type: str, data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL) -> WebhookEvent:
        """Tạo webhook event"""
        event = WebhookEvent(event_type, data, priority)

        # Validate event data
        if not self.validate_event_data(event_type, data):
            raise ValueError(f"Invalid data for event type: {event_type}")

        return event

    def validate_event_data(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Validate event data theo config"""
        if event_type not in EVENT_CONFIGS:
            return True  # Unknown event type, allow it

        config = EVENT_CONFIGS[event_type]
        required_fields = config.get("required_fields", [])

        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field '{field}' for event type '{event_type}'")
                return False

        return True

    def format_event_payload(self, event: WebhookEvent, webhook_id: int = None) -> Dict[str, Any]:
        """Format event payload cho webhook delivery"""
        # Filter data based on event config
        filtered_data = event_filter.get_filtered_data(event.event_type, event.data)

        payload = {
            "event_id": event.id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "data": filtered_data
        }

        if webhook_id:
            payload["webhook_id"] = webhook_id

        return payload

    def add_custom_event_type(self, event_type: str, config: Dict[str, Any]):
        """Thêm custom event type"""
        EVENT_CONFIGS[event_type] = config
        logger.info(f"Added custom event type: {event_type}")

    def get_event_config(self, event_type: str) -> Dict[str, Any]:
        """Lấy config cho event type"""
        return EVENT_CONFIGS.get(event_type, {})

    def list_event_types(self) -> List[str]:
        """Liệt kê tất cả event types"""
        return list(EVENT_CONFIGS.keys())

# Global event manager instance
event_manager = EventManager()

# Convenience functions for creating common events
def create_tts_completed_event(request_id: str, audio_url: str, duration: float, text_length: int, **kwargs) -> WebhookEvent:
    """Tạo TTS completed event"""
    data = EventTemplate.tts_completed(request_id, audio_url, duration, text_length, **kwargs)
    return event_manager.create_event(WebhookEventType.TTS_COMPLETED, data)

def create_tts_error_event(request_id: str, error_code: str, error_message: str, text_length: int = None, **kwargs) -> WebhookEvent:
    """Tạo TTS error event"""
    data = EventTemplate.tts_error(request_id, error_code, error_message, text_length, **kwargs)
    return event_manager.create_event(WebhookEventType.TTS_ERROR, data, EventPriority.HIGH)

def create_batch_completed_event(batch_id: str, total_requests: int, successful_requests: int, failed_requests: int, **kwargs) -> WebhookEvent:
    """Tạo batch completed event"""
    data = EventTemplate.batch_completed(batch_id, total_requests, successful_requests, failed_requests, **kwargs)
    return event_manager.create_event(WebhookEventType.BATCH_COMPLETED, data)

def create_batch_error_event(batch_id: str, error_code: str, error_message: str, failed_requests: int, **kwargs) -> WebhookEvent:
    """Tạo batch error event"""
    data = EventTemplate.batch_error(batch_id, error_code, error_message, failed_requests, **kwargs)
    return event_manager.create_event(WebhookEventType.BATCH_ERROR, data, EventPriority.CRITICAL)

def create_quality_enhancement_completed_event(request_id: str, enhancement_type: str, original_quality: float, enhanced_quality: float, **kwargs) -> WebhookEvent:
    """Tạo quality enhancement completed event"""
    data = EventTemplate.quality_enhancement_completed(request_id, enhancement_type, original_quality, enhanced_quality, **kwargs)
    return event_manager.create_event(WebhookEventType.QUALITY_ENHANCEMENT_COMPLETED, data)

def create_voice_cloning_completed_event(cloning_id: str, voice_name: str, quality_score: float, training_duration: float, **kwargs) -> WebhookEvent:
    """Tạo voice cloning completed event"""
    data = EventTemplate.voice_cloning_completed(cloning_id, voice_name, quality_score, training_duration, **kwargs)
    return event_manager.create_event(WebhookEventType.VOICE_CLONING_COMPLETED, data)

def create_audio_enhancement_completed_event(request_id: str, enhancement_type: str, improvement_score: float, **kwargs) -> WebhookEvent:
    """Tạo audio enhancement completed event"""
    data = EventTemplate.audio_enhancement_completed(request_id, enhancement_type, improvement_score, **kwargs)
    return event_manager.create_event(WebhookEventType.AUDIO_ENHANCEMENT_COMPLETED, data)