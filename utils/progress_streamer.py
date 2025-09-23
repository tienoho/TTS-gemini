"""
Progress streaming service for real-time TTS updates
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
from threading import Lock

from flask import current_app

from .websocket_manager import get_websocket_manager, broadcast_to_room, broadcast_to_user
from .redis_manager import redis_manager
from models.batch_request import BatchStatus, BatchProgressUpdate


class ProgressStatus(str, Enum):
    """Progress status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressEvent:
    """Represents a progress event."""

    def __init__(self, request_id: str, status: ProgressStatus, message: str = "",
                 progress: int = 0, metadata: Optional[Dict] = None):
        self.request_id = request_id
        self.status = status
        self.message = message
        self.progress = progress
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "message": self.message,
            "progress": self.progress,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ProgressTracker:
    """Tracks progress for a specific request."""

    def __init__(self, request_id: str, user_id: Optional[int] = None):
        self.request_id = request_id
        self.user_id = user_id
        self.status = ProgressStatus.PENDING
        self.progress = 0
        self.message = ""
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.events: List[ProgressEvent] = []
        self.is_active = True

    def update_status(self, status: ProgressStatus, message: str = "", progress: int = None,
                     metadata: Optional[Dict] = None):
        """Update progress status."""
        self.status = status
        self.message = message
        if progress is not None:
            self.progress = progress
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = datetime.utcnow()

        # Create event
        event = ProgressEvent(self.request_id, status, message, self.progress, metadata)
        self.events.append(event)

        # Keep only last 50 events to prevent memory issues
        if len(self.events) > 50:
            self.events = self.events[-50:]

    def get_latest_event(self) -> Optional[ProgressEvent]:
        """Get the latest progress event."""
        return self.events[-1] if self.events else None

    def is_completed(self) -> bool:
        """Check if request is completed."""
        return self.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active
        }


class ProgressStreamingService:
    """Service for managing and streaming progress updates."""

    def __init__(self):
        self.trackers: Dict[str, ProgressTracker] = {}
        self._lock = Lock()
        self._cleanup_task = None
        self._event_listeners: Dict[str, List[Callable]] = {}

        # Start background cleanup task
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_trackers())

    async def _cleanup_inactive_trackers(self):
        """Background task to cleanup inactive progress trackers."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes

                with self._lock:
                    inactive_requests = []
                    cutoff_time = datetime.utcnow() - timedelta(hours=2)

                    for request_id, tracker in self.trackers.items():
                        if tracker.updated_at < cutoff_time and tracker.is_completed():
                            inactive_requests.append(request_id)

                    for request_id in inactive_requests:
                        del self.trackers[request_id]

                    if inactive_requests:
                        logging.info(f"Cleaned up {len(inactive_requests)} inactive progress trackers")

            except Exception as e:
                logging.error(f"Error in cleanup task: {e}")

    def create_tracker(self, request_id: str, user_id: Optional[int] = None) -> ProgressTracker:
        """Create a new progress tracker."""
        with self._lock:
            if request_id in self.trackers:
                return self.trackers[request_id]

            tracker = ProgressTracker(request_id, user_id)
            self.trackers[request_id] = tracker
            return tracker

    def get_tracker(self, request_id: str) -> Optional[ProgressTracker]:
        """Get progress tracker by request ID."""
        with self._lock:
            return self.trackers.get(request_id)

    def update_progress(self, request_id: str, status: ProgressStatus, message: str = "",
                       progress: int = None, metadata: Optional[Dict] = None):
        """Update progress for a request."""
        with self._lock:
            tracker = self.trackers.get(request_id)
            if not tracker:
                return False

            tracker.update_status(status, message, progress, metadata)

            # Broadcast update
            asyncio.create_task(self._broadcast_progress_update(tracker))

            return True

    async def _broadcast_progress_update(self, tracker: ProgressTracker):
        """Broadcast progress update to connected clients."""
        try:
            event_data = tracker.get_latest_event().to_dict()

            # Broadcast to room (request-specific)
            room_id = f"tts_{tracker.request_id}"
            await broadcast_to_room(room_id, "progress_update", event_data)

            # Broadcast to user if user_id is available
            if tracker.user_id:
                await broadcast_to_user(tracker.user_id, "progress_update", event_data)

            # Trigger event listeners
            await self._trigger_event_listeners("progress_update", event_data)

        except Exception as e:
            logging.error(f"Error broadcasting progress update for {tracker.request_id}: {e}")

    def subscribe_to_request(self, request_id: str, sid: str) -> bool:
        """Subscribe a WebSocket connection to request progress updates."""
        try:
            room_id = f"tts_{request_id}"
            asyncio.create_task(get_websocket_manager().join_room(sid, room_id))
            return True
        except Exception as e:
            logging.error(f"Error subscribing to request {request_id}: {e}")
            return False

    def unsubscribe_from_request(self, request_id: str, sid: str) -> bool:
        """Unsubscribe a WebSocket connection from request progress updates."""
        try:
            room_id = f"tts_{request_id}"
            asyncio.create_task(get_websocket_manager().leave_room(sid, room_id))
            return True
        except Exception as e:
            logging.error(f"Error unsubscribing from request {request_id}: {e}")
            return False

    def register_event_listener(self, event_type: str, callback: Callable):
        """Register an event listener."""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    async def _trigger_event_listeners(self, event_type: str, data: Any):
        """Trigger event listeners."""
        if event_type not in self._event_listeners:
            return

        for callback in self._event_listeners[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logging.error(f"Error in event listener for {event_type}: {e}")

    def get_all_trackers(self) -> Dict[str, ProgressTracker]:
        """Get all progress trackers."""
        with self._lock:
            return self.trackers.copy()

    def get_user_trackers(self, user_id: int) -> Dict[str, ProgressTracker]:
        """Get progress trackers for a specific user."""
        with self._lock:
            return {
                request_id: tracker
                for request_id, tracker in self.trackers.items()
                if tracker.user_id == user_id
            }

    def get_active_trackers(self) -> Dict[str, ProgressTracker]:
        """Get active (non-completed) progress trackers."""
        with self._lock:
            return {
                request_id: tracker
                for request_id, tracker in self.trackers.items()
                if not tracker.is_completed()
            }

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request and update its status."""
        return self.update_progress(
            request_id,
            ProgressStatus.CANCELLED,
            "Request was cancelled"
        )

    def mark_request_completed(self, request_id: str, message: str = "Request completed successfully",
                              metadata: Optional[Dict] = None) -> bool:
        """Mark a request as completed."""
        return self.update_progress(
            request_id,
            ProgressStatus.COMPLETED,
            message,
            100,
            metadata
        )

    def mark_request_failed(self, request_id: str, error_message: str,
                           metadata: Optional[Dict] = None) -> bool:
        """Mark a request as failed."""
        return self.update_progress(
            request_id,
            ProgressStatus.FAILED,
            error_message,
            metadata=metadata
        )

    def get_request_history(self, request_id: str, limit: int = 50) -> List[Dict]:
        """Get progress history for a request."""
        with self._lock:
            tracker = self.trackers.get(request_id)
            if not tracker:
                return []

            events = tracker.events[-limit:] if limit > 0 else tracker.events
            return [event.to_dict() for event in events]

    async def broadcast_status_update(self, request_id: str, status: ProgressStatus,
                                    message: str = "", progress: int = None):
        """Broadcast a status update for a request."""
        if self.update_progress(request_id, status, message, progress):
            # Additional broadcast logic can be added here if needed
            pass

    def cleanup_user_trackers(self, user_id: int):
        """Clean up all trackers for a specific user."""
        with self._lock:
            user_requests = [
                request_id for request_id, tracker in self.trackers.items()
                if tracker.user_id == user_id
            ]

            for request_id in user_requests:
                del self.trackers[request_id]

            logging.info(f"Cleaned up {len(user_requests)} trackers for user {user_id}")

    # Batch Progress Integration Methods
    async def send_batch_progress(
        self,
        batch_id: str,
        status: BatchStatus,
        progress_data: Dict[str, Any],
        item_updates: Optional[List[Dict]] = None
    ):
        """
        Send batch progress update via WebSocket
        """
        try:
            # Create batch progress update message
            update_message = BatchProgressUpdate(
                batch_id=batch_id,
                status=status,
                progress=progress_data,
                item_updates=item_updates
            )

            # Broadcast to batch-specific room
            room_id = f"batch_{batch_id}"
            await broadcast_to_room(room_id, "batch_progress", update_message.dict())

            # Also broadcast to general progress room
            await broadcast_to_room("batch_updates", "batch_progress", update_message.dict())

            logging.info(f"Batch progress update sent for {batch_id}: {status.value}")

        except Exception as e:
            logging.error(f"Error sending batch progress for {batch_id}: {str(e)}")

    def subscribe_to_batch(self, batch_id: str, sid: str) -> bool:
        """
        Subscribe to batch progress updates
        """
        try:
            # Subscribe to batch-specific room
            batch_room = f"batch_{batch_id}"
            asyncio.create_task(get_websocket_manager().join_room(sid, batch_room))

            # Also subscribe to general batch updates room
            asyncio.create_task(get_websocket_manager().join_room(sid, "batch_updates"))

            return True
        except Exception as e:
            logging.error(f"Error subscribing to batch {batch_id}: {e}")
            return False

    def unsubscribe_from_batch(self, batch_id: str, sid: str) -> bool:
        """
        Unsubscribe from batch progress updates
        """
        try:
            # Unsubscribe from batch-specific room
            batch_room = f"batch_{batch_id}"
            asyncio.create_task(get_websocket_manager().leave_room(sid, batch_room))

            # Also unsubscribe from general batch updates room
            asyncio.create_task(get_websocket_manager().leave_room(sid, "batch_updates"))

            return True
        except Exception as e:
            logging.error(f"Error unsubscribing from batch {batch_id}: {e}")
            return False

    async def send_batch_item_update(
        self,
        batch_id: str,
        item_result: Dict[str, Any]
    ):
        """
        Send individual item progress update
        """
        try:
            # Send item-specific update
            item_room = f"batch_{batch_id}_item_{item_result.get('item_id')}"
            await broadcast_to_room(item_room, "item_progress", item_result)

            # Also include in batch progress update
            await self.send_batch_progress(
                batch_id,
                BatchStatus.PROCESSING,
                {"item_updated": True},
                item_updates=[item_result]
            )

        except Exception as e:
            logging.error(f"Error sending batch item update for {batch_id}: {str(e)}")

    async def send_batch_completed(
        self,
        batch_id: str,
        summary: Dict[str, Any],
        results: List[Dict[str, Any]]
    ):
        """
        Send batch completion notification
        """
        try:
            completion_data = {
                "batch_id": batch_id,
                "status": BatchStatus.COMPLETED,
                "summary": summary,
                "results_count": len(results),
                "completed_at": datetime.utcnow().isoformat()
            }

            # Send to batch room
            batch_room = f"batch_{batch_id}"
            await broadcast_to_room(batch_room, "batch_completed", completion_data)

            # Send to general completion room
            await broadcast_to_room("batch_completions", "batch_completed", completion_data)

            logging.info(f"Batch completion notification sent for {batch_id}")

        except Exception as e:
            logging.error(f"Error sending batch completion for {batch_id}: {str(e)}")

    async def send_batch_failed(
        self,
        batch_id: str,
        error_message: str,
        failed_items: List[Dict[str, Any]] = None
    ):
        """
        Send batch failure notification
        """
        try:
            failure_data = {
                "batch_id": batch_id,
                "status": BatchStatus.FAILED,
                "error_message": error_message,
                "failed_at": datetime.utcnow().isoformat(),
                "failed_items_count": len(failed_items) if failed_items else 0
            }

            # Send to batch room
            batch_room = f"batch_{batch_id}"
            await broadcast_to_room(batch_room, "batch_failed", failure_data)

            # Send to general failure room
            await broadcast_to_room("batch_failures", "batch_failed", failure_data)

            logging.warning(f"Batch failure notification sent for {batch_id}: {error_message}")

        except Exception as e:
            logging.error(f"Error sending batch failure for {batch_id}: {str(e)}")


# Global progress streaming service instance
progress_streamer = ProgressStreamingService()


# Utility functions for easy access
def get_progress_streamer() -> ProgressStreamingService:
    """Get progress streaming service instance."""
    return progress_streamer


def create_progress_tracker(request_id: str, user_id: Optional[int] = None) -> ProgressTracker:
    """Create a new progress tracker."""
    return progress_streamer.create_tracker(request_id, user_id)


def update_progress(request_id: str, status: ProgressStatus, message: str = "",
                   progress: int = None, metadata: Optional[Dict] = None) -> bool:
    """Update progress for a request."""
    return progress_streamer.update_progress(request_id, status, message, progress, metadata)


def subscribe_to_request(request_id: str, sid: str) -> bool:
    """Subscribe to request progress updates."""
    return progress_streamer.subscribe_to_request(request_id, sid)


def unsubscribe_from_request(request_id: str, sid: str) -> bool:
    """Unsubscribe from request progress updates."""
    return progress_streamer.unsubscribe_from_request(request_id, sid)


def mark_request_completed(request_id: str, message: str = "Request completed successfully",
                          metadata: Optional[Dict] = None) -> bool:
    """Mark a request as completed."""
    return progress_streamer.mark_request_completed(request_id, message, metadata)


def mark_request_failed(request_id: str, error_message: str,
                       metadata: Optional[Dict] = None) -> bool:
    """Mark a request as failed."""
    return progress_streamer.mark_request_failed(request_id, error_message, metadata)


def cancel_request(request_id: str) -> bool:
    """Cancel a request."""
    return progress_streamer.cancel_request(request_id)