"""
WebSocket connection manager for real-time TTS progress updates
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
from collections import defaultdict
import logging
from threading import Lock

from flask import current_app
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_jwt_extended import decode_token

from ..config.websocket import get_websocket_settings
from .auth import auth_service
from .redis_manager import redis_manager


class WebSocketConnection:
    """Represents a WebSocket connection."""

    def __init__(self, sid: str, user_id: Optional[int] = None, ip_address: str = ""):
        self.sid = sid
        self.user_id = user_id
        self.ip_address = ip_address
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.rooms: Set[str] = set()
        self.is_authenticated = user_id is not None
        self.message_count = 0
        self.last_message_time = datetime.utcnow()

    def update_ping(self):
        """Update last ping timestamp."""
        self.last_ping = datetime.utcnow()

    def add_room(self, room_id: str):
        """Add connection to a room."""
        self.rooms.add(room_id)

    def remove_room(self, room_id: str):
        """Remove connection from a room."""
        self.rooms.discard(room_id)

    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """Check if connection is still alive."""
        return (datetime.utcnow() - self.last_ping).total_seconds() < timeout_seconds

    def can_send_message(self, rate_limit: int = 100, window_seconds: int = 60) -> bool:
        """Check if connection can send more messages based on rate limit."""
        now = datetime.utcnow()
        time_diff = (now - self.last_message_time).total_seconds()

        if time_diff > window_seconds:
            self.message_count = 0
            self.last_message_time = now
            return True

        return self.message_count < rate_limit


class WebSocketManager:
    """Manages WebSocket connections, rooms, and message broadcasting."""

    def __init__(self):
        self.settings = get_websocket_settings()
        self.connections: Dict[str, WebSocketConnection] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)  # room_id -> set of sids
        self.user_connections: Dict[int, Set[str]] = defaultdict(set)  # user_id -> set of sids
        self.ip_connections: Dict[str, Set[str]] = defaultdict(set)  # ip -> set of sids
        self._lock = Lock()
        self._cleanup_task = None
        self._metrics_task = None

        # Message handlers
        self._message_handlers: Dict[str, Callable] = {}

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background cleanup and metrics tasks."""
        if self.settings.WS_ENABLE_HEARTBEAT:
            self._cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())
            self._metrics_task = asyncio.create_task(self._collect_metrics())

    async def _cleanup_inactive_connections(self):
        """Background task to cleanup inactive connections."""
        while True:
            try:
                await asyncio.sleep(self.settings.WS_CONNECTION_CLEANUP_INTERVAL)

                with self._lock:
                    inactive_sids = []
                    for sid, conn in self.connections.items():
                        if not conn.is_alive(self.settings.ping_timeout_total):
                            inactive_sids.append(sid)

                    for sid in inactive_sids:
                        await self._disconnect_connection(sid, "Connection timeout")

            except Exception as e:
                logging.error(f"Error in cleanup task: {e}")

    async def _collect_metrics(self):
        """Background task to collect connection metrics."""
        while True:
            try:
                await asyncio.sleep(self.settings.WS_METRICS_COLLECTION_INTERVAL)

                with self._lock:
                    total_connections = len(self.connections)
                    authenticated_connections = sum(1 for conn in self.connections.values() if conn.is_authenticated)
                    total_rooms = len(self.rooms)

                    # Log metrics
                    if self.settings.WS_ENABLE_METRICS:
                        logging.info(
                            f"WebSocket Metrics - Connections: {total_connections}, "
                            f"Authenticated: {authenticated_connections}, "
                            f"Rooms: {total_rooms}"
                        )

            except Exception as e:
                logging.error(f"Error in metrics collection: {e}")

    def register_message_handler(self, event_type: str, handler: Callable):
        """Register a message handler for specific event type."""
        self._message_handlers[event_type] = handler

    async def handle_connect(self, sid: str, auth_data: Optional[Dict] = None, ip_address: str = ""):
        """Handle new WebSocket connection."""
        try:
            # Check connection limits
            if not await self._check_connection_limits(ip_address):
                return False

            # Authenticate connection
            user_id = None
            if self.settings.WS_AUTH_REQUIRED and auth_data:
                user_id = await self._authenticate_connection(auth_data)

            # Create connection object
            connection = WebSocketConnection(sid, user_id, ip_address)

            with self._lock:
                self.connections[sid] = connection
                if user_id:
                    self.user_connections[user_id].add(sid)
                self.ip_connections[ip_address].add(sid)

            logging.info(f"WebSocket connection established: {sid} (User: {user_id}, IP: {ip_address})")
            return True

        except Exception as e:
            logging.error(f"Error establishing connection {sid}: {e}")
            return False

    async def handle_disconnect(self, sid: str):
        """Handle WebSocket disconnection."""
        try:
            await self._disconnect_connection(sid, "Client disconnected")
        except Exception as e:
            logging.error(f"Error handling disconnect {sid}: {e}")

    async def _disconnect_connection(self, sid: str, reason: str):
        """Internal method to disconnect a connection."""
        with self._lock:
            if sid not in self.connections:
                return

            connection = self.connections[sid]

            # Leave all rooms
            for room_id in connection.rooms:
                await self.leave_room(sid, room_id)

            # Remove from tracking
            if connection.user_id:
                self.user_connections[connection.user_id].discard(sid)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]

            self.ip_connections[connection.ip_address].discard(sid)
            if not self.ip_connections[connection.ip_address]:
                del self.ip_connections[connection.ip_address]

            del self.connections[sid]

        logging.info(f"WebSocket connection disconnected: {sid} - {reason}")

    async def _check_connection_limits(self, ip_address: str) -> bool:
        """Check if connection limits are exceeded."""
        with self._lock:
            # Check total connections
            if len(self.connections) >= self.settings.WS_MAX_CONNECTIONS:
                logging.warning(f"Max connections limit reached: {len(self.connections)}")
                return False

            # Check per-IP connections
            ip_connection_count = len(self.ip_connections[ip_address])
            if ip_connection_count >= self.settings.WS_MAX_CONNECTIONS_PER_IP:
                logging.warning(f"Max connections per IP limit reached for {ip_address}: {ip_connection_count}")
                return False

        return True

    async def _authenticate_connection(self, auth_data: Dict) -> Optional[int]:
        """Authenticate WebSocket connection."""
        try:
            token = auth_data.get('token')
            if not token:
                return None

            # Verify JWT token
            payload = auth_service.verify_token(token)
            user_id = payload.get('sub')

            if user_id:
                return int(user_id)

        except Exception as e:
            logging.warning(f"WebSocket authentication failed: {e}")

        return None

    async def join_room(self, sid: str, room_id: str):
        """Join a connection to a room."""
        try:
            with self._lock:
                if sid not in self.connections:
                    return False

                connection = self.connections[sid]
                if len(connection.rooms) >= self.settings.WS_MAX_ROOMS_PER_CONNECTION:
                    logging.warning(f"Max rooms per connection limit reached for {sid}")
                    return False

                # Join SocketIO room
                join_room(room_id, sid=connection.sid)

                # Update connection and room tracking
                connection.add_room(room_id)
                self.rooms[room_id].add(sid)

                logging.info(f"Connection {sid} joined room {room_id}")
                return True

        except Exception as e:
            logging.error(f"Error joining room {room_id} for {sid}: {e}")
            return False

    async def leave_room(self, sid: str, room_id: str):
        """Leave a connection from a room."""
        try:
            with self._lock:
                if sid not in self.connections:
                    return

                connection = self.connections[sid]
                connection.remove_room(room_id)
                self.rooms[room_id].discard(sid)

                # Leave SocketIO room
                leave_room(room_id, sid=connection.sid)

                # Cleanup empty rooms
                if not self.rooms[room_id]:
                    del self.rooms[room_id]

                logging.info(f"Connection {sid} left room {room_id}")

        except Exception as e:
            logging.error(f"Error leaving room {room_id} for {sid}: {e}")

    async def broadcast_to_room(self, room_id: str, event: str, data: Any, exclude_sid: Optional[str] = None):
        """Broadcast message to all connections in a room."""
        try:
            # Check if room exists
            with self._lock:
                room_sids = self.rooms.get(room_id, set())

            if not room_sids:
                return

            # Send to SocketIO room
            emit(event, data, to=room_id, skip_sid=exclude_sid)

            logging.debug(f"Broadcasted {event} to room {room_id} ({len(room_sids)} connections)")

        except Exception as e:
            logging.error(f"Error broadcasting to room {room_id}: {e}")

    async def broadcast_to_user(self, user_id: int, event: str, data: Any, exclude_sid: Optional[str] = None):
        """Broadcast message to all connections of a specific user."""
        try:
            with self._lock:
                user_sids = self.user_connections.get(user_id, set())

            if not user_sids:
                return

            # Send to each connection
            for sid in user_sids:
                if sid != exclude_sid:
                    try:
                        emit(event, data, to=sid)
                    except Exception as e:
                        logging.error(f"Error sending to user connection {sid}: {e}")

            logging.debug(f"Broadcasted {event} to user {user_id} ({len(user_sids)} connections)")

        except Exception as e:
            logging.error(f"Error broadcasting to user {user_id}: {e}")

    async def send_to_connection(self, sid: str, event: str, data: Any):
        """Send message to a specific connection."""
        try:
            with self._lock:
                if sid not in self.connections:
                    return False

                connection = self.connections[sid]

                # Check rate limit
                if not connection.can_send_message(self.settings.WS_MESSAGE_RATE_LIMIT, self.settings.WS_RATE_LIMIT_WINDOW):
                    logging.warning(f"Rate limit exceeded for connection {sid}")
                    return False

                # Update message count
                connection.message_count += 1

            # Send via SocketIO
            emit(event, data, to=sid)
            return True

        except Exception as e:
            logging.error(f"Error sending to connection {sid}: {e}")
            return False

    async def handle_ping(self, sid: str):
        """Handle ping from client."""
        try:
            with self._lock:
                if sid in self.connections:
                    self.connections[sid].update_ping()
        except Exception as e:
            logging.error(f"Error handling ping for {sid}: {e}")

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        with self._lock:
            return len(self.connections)

    def get_room_count(self) -> int:
        """Get total number of active rooms."""
        with self._lock:
            return len(self.rooms)

    def get_user_connections(self, user_id: int) -> int:
        """Get number of connections for a specific user."""
        with self._lock:
            return len(self.user_connections.get(user_id, set()))

    def get_connection_info(self, sid: str) -> Optional[Dict]:
        """Get information about a specific connection."""
        with self._lock:
            if sid not in self.connections:
                return None

            conn = self.connections[sid]
            return {
                'sid': conn.sid,
                'user_id': conn.user_id,
                'ip_address': conn.ip_address,
                'connected_at': conn.connected_at.isoformat(),
                'last_ping': conn.last_ping.isoformat(),
                'rooms': list(conn.rooms),
                'is_authenticated': conn.is_authenticated,
                'message_count': conn.message_count
            }

    def get_room_info(self, room_id: str) -> Optional[Dict]:
        """Get information about a specific room."""
        with self._lock:
            if room_id not in self.rooms:
                return None

            return {
                'room_id': room_id,
                'connection_count': len(self.rooms[room_id]),
                'connections': list(self.rooms[room_id])
            }

    async def cleanup_user_connections(self, user_id: int):
        """Cleanup all connections for a specific user."""
        try:
            with self._lock:
                user_sids = self.user_connections.get(user_id, set()).copy()

            for sid in user_sids:
                await self._disconnect_connection(sid, "User logout")

        except Exception as e:
            logging.error(f"Error cleaning up user connections for {user_id}: {e}")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Utility functions for easy access
def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance."""
    return websocket_manager


async def broadcast_to_room(room_id: str, event: str, data: Any, exclude_sid: Optional[str] = None):
    """Broadcast message to a room."""
    await websocket_manager.broadcast_to_room(room_id, event, data, exclude_sid)


async def broadcast_to_user(user_id: int, event: str, data: Any, exclude_sid: Optional[str] = None):
    """Broadcast message to a user."""
    await websocket_manager.broadcast_to_user(user_id, event, data, exclude_sid)


async def send_to_connection(sid: str, event: str, data: Any):
    """Send message to a specific connection."""
    await websocket_manager.send_to_connection(sid, event, data)