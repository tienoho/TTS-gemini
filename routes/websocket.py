"""
WebSocket routes for real-time TTS progress updates
"""

import json
from typing import Dict, Any, Optional
from flask import request
from flask_socketio import emit, join_room, leave_room

from ..config.websocket import get_websocket_settings
from ..utils.websocket_manager import get_websocket_manager
from ..utils.progress_streamer import (
    get_progress_streamer,
    subscribe_to_request,
    unsubscribe_from_request,
    ProgressStatus
)
from ..utils.auth import auth_service


# WebSocket event handlers
def register_websocket_handlers(socketio):
    """Register WebSocket event handlers with SocketIO."""

    @socketio.on('connect')
    def handle_connect(auth):
        """Handle WebSocket connection."""
        try:
            settings = get_websocket_settings()
            manager = get_websocket_manager()

            # Get client IP address
            ip_address = getattr(request, 'remote_addr', 'unknown')

            # Handle connection
            success = manager.handle_connect(request.sid, auth, ip_address)

            if success:
                emit('connected', {
                    'message': 'Connected successfully',
                    'session_id': request.sid
                })
            else:
                emit('error', {
                    'message': 'Connection failed - limits exceeded',
                    'code': 'CONNECTION_LIMIT_EXCEEDED'
                })
                return False  # Reject connection

        except Exception as e:
            emit('error', {
                'message': f'Connection error: {str(e)}',
                'code': 'CONNECTION_ERROR'
            })
            return False

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        try:
            manager = get_websocket_manager()
            manager.handle_disconnect(request.sid)
        except Exception as e:
            print(f"Error handling disconnect: {e}")

    @socketio.on('authenticate')
    def handle_authenticate(auth_data):
        """Handle WebSocket authentication."""
        try:
            settings = get_websocket_settings()
            manager = get_websocket_manager()

            if not settings.WS_AUTH_REQUIRED:
                emit('authenticated', {'message': 'Authentication not required'})
                return

            token = auth_data.get('token')
            if not token:
                emit('error', {
                    'message': 'Authentication token required',
                    'code': 'AUTH_TOKEN_MISSING'
                })
                return

            # Verify token
            try:
                payload = auth_service.verify_token(token)
                user_id = payload.get('sub')

                if user_id:
                    emit('authenticated', {
                        'message': 'Authenticated successfully',
                        'user_id': user_id
                    })
                else:
                    emit('error', {
                        'message': 'Invalid token payload',
                        'code': 'INVALID_TOKEN'
                    })

            except Exception as e:
                emit('error', {
                    'message': f'Authentication failed: {str(e)}',
                    'code': 'AUTH_FAILED'
                })

        except Exception as e:
            emit('error', {
                'message': f'Authentication error: {str(e)}',
                'code': 'AUTH_ERROR'
            })

    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Handle subscription to request progress updates."""
        try:
            request_id = data.get('request_id')
            if not request_id:
                emit('error', {
                    'message': 'Request ID is required',
                    'code': 'MISSING_REQUEST_ID'
                })
                return

            # Subscribe to request updates
            success = subscribe_to_request(request_id, request.sid)

            if success:
                emit('subscribed', {
                    'message': f'Subscribed to request {request_id}',
                    'request_id': request_id
                })
            else:
                emit('error', {
                    'message': 'Failed to subscribe to request',
                    'code': 'SUBSCRIPTION_FAILED'
                })

        except Exception as e:
            emit('error', {
                'message': f'Subscription error: {str(e)}',
                'code': 'SUBSCRIPTION_ERROR'
            })

    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Handle unsubscription from request progress updates."""
        try:
            request_id = data.get('request_id')
            if not request_id:
                emit('error', {
                    'message': 'Request ID is required',
                    'code': 'MISSING_REQUEST_ID'
                })
                return

            # Unsubscribe from request updates
            success = unsubscribe_from_request(request_id, request.sid)

            if success:
                emit('unsubscribed', {
                    'message': f'Unsubscribed from request {request_id}',
                    'request_id': request_id
                })
            else:
                emit('error', {
                    'message': 'Failed to unsubscribe from request',
                    'code': 'UNSUBSCRIPTION_FAILED'
                })

        except Exception as e:
            emit('error', {
                'message': f'Unsubscription error: {str(e)}',
                'code': 'UNSUBSCRIPTION_ERROR'
            })

    @socketio.on('ping')
    def handle_ping():
        """Handle ping from client."""
        try:
            manager = get_websocket_manager()
            manager.handle_ping(request.sid)

            emit('pong', {
                'timestamp': json.dumps('pong')
            })
        except Exception as e:
            emit('error', {
                'message': f'Ping error: {str(e)}',
                'code': 'PING_ERROR'
            })

    @socketio.on('get_progress')
    def handle_get_progress(data):
        """Handle request for current progress of a request."""
        try:
            request_id = data.get('request_id')
            if not request_id:
                emit('error', {
                    'message': 'Request ID is required',
                    'code': 'MISSING_REQUEST_ID'
                })
                return

            # Get progress tracker
            streamer = get_progress_streamer()
            tracker = streamer.get_tracker(request_id)

            if not tracker:
                emit('error', {
                    'message': 'Request not found',
                    'code': 'REQUEST_NOT_FOUND'
                })
                return

            # Get latest event
            latest_event = tracker.get_latest_event()
            if latest_event:
                emit('progress_update', latest_event.to_dict())
            else:
                emit('progress_update', {
                    'request_id': request_id,
                    'status': tracker.status.value,
                    'message': tracker.message,
                    'progress': tracker.progress,
                    'timestamp': tracker.updated_at.isoformat()
                })

        except Exception as e:
            emit('error', {
                'message': f'Get progress error: {str(e)}',
                'code': 'GET_PROGRESS_ERROR'
            })

    @socketio.on('get_active_requests')
    def handle_get_active_requests():
        """Handle request for list of active requests."""
        try:
            # Get active trackers
            streamer = get_progress_streamer()
            active_trackers = streamer.get_active_trackers()

            # Convert to list of dicts
            active_requests = []
            for request_id, tracker in active_trackers.items():
                latest_event = tracker.get_latest_event()
                if latest_event:
                    active_requests.append(latest_event.to_dict())
                else:
                    active_requests.append({
                        'request_id': request_id,
                        'status': tracker.status.value,
                        'message': tracker.message,
                        'progress': tracker.progress,
                        'timestamp': tracker.updated_at.isoformat()
                    })

            emit('active_requests', {
                'requests': active_requests,
                'count': len(active_requests)
            })

        except Exception as e:
            emit('error', {
                'message': f'Get active requests error: {str(e)}',
                'code': 'GET_ACTIVE_REQUESTS_ERROR'
            })

    @socketio.on('get_connection_info')
    def handle_get_connection_info():
        """Handle request for connection information."""
        try:
            manager = get_websocket_manager()
            connection_info = manager.get_connection_info(request.sid)

            if connection_info:
                emit('connection_info', connection_info)
            else:
                emit('error', {
                    'message': 'Connection not found',
                    'code': 'CONNECTION_NOT_FOUND'
                })

        except Exception as e:
            emit('error', {
                'message': f'Get connection info error: {str(e)}',
                'code': 'GET_CONNECTION_INFO_ERROR'
            })

    @socketio.on('join_room')
    def handle_join_room(data):
        """Handle joining a custom room."""
        try:
            room_id = data.get('room_id')
            if not room_id:
                emit('error', {
                    'message': 'Room ID is required',
                    'code': 'MISSING_ROOM_ID'
                })
                return

            # Join room
            join_room(room_id)

            emit('room_joined', {
                'message': f'Joined room {room_id}',
                'room_id': room_id
            })

        except Exception as e:
            emit('error', {
                'message': f'Join room error: {str(e)}',
                'code': 'JOIN_ROOM_ERROR'
            })

    @socketio.on('leave_room')
    def handle_leave_room(data):
        """Handle leaving a custom room."""
        try:
            room_id = data.get('room_id')
            if not room_id:
                emit('error', {
                    'message': 'Room ID is required',
                    'code': 'MISSING_ROOM_ID'
                })
                return

            # Leave room
            leave_room(room_id)

            emit('room_left', {
                'message': f'Left room {room_id}',
                'room_id': room_id
            })

        except Exception as e:
            emit('error', {
                'message': f'Leave room error: {str(e)}',
                'code': 'LEAVE_ROOM_ERROR'
            })

    @socketio.on('broadcast_room')
    def handle_broadcast_room(data):
        """Handle broadcasting to a room."""
        try:
            room_id = data.get('room_id')
            event = data.get('event', 'message')
            message = data.get('message', '')

            if not room_id:
                emit('error', {
                    'message': 'Room ID is required',
                    'code': 'MISSING_ROOM_ID'
                })
                return

            # Broadcast to room
            emit(event, {
                'message': message,
                'from': request.sid,
                'timestamp': json.dumps('broadcast')
            }, to=room_id)

            emit('broadcast_sent', {
                'message': 'Message broadcasted successfully',
                'room_id': room_id,
                'event': event
            })

        except Exception as e:
            emit('error', {
                'message': f'Broadcast error: {str(e)}',
                'code': 'BROADCAST_ERROR'
            })

    @socketio.on('error')
    def handle_error(error_data):
        """Handle client error reports."""
        try:
            print(f"Client error: {error_data}")
            # Log error for monitoring
        except Exception as e:
            print(f"Error handling client error: {e}")


# Utility functions
def emit_progress_update(request_id: str, status: ProgressStatus, message: str = "",
                        progress: int = None, metadata: Optional[Dict] = None):
    """Emit progress update to all subscribers."""
    try:
        from ..utils.progress_streamer import update_progress
        update_progress(request_id, status, message, progress, metadata)
    except Exception as e:
        print(f"Error emitting progress update: {e}")


def emit_to_room(room_id: str, event: str, data: Any):
    """Emit message to a specific room."""
    try:
        emit(event, data, to=room_id)
    except Exception as e:
        print(f"Error emitting to room {room_id}: {e}")


def emit_to_user(user_id: int, event: str, data: Any):
    """Emit message to all connections of a user."""
    try:
        from ..utils.websocket_manager import broadcast_to_user
        from ..utils.progress_streamer import get_progress_streamer
        import asyncio

        # Get progress streamer and create task
        streamer = get_progress_streamer()
        asyncio.create_task(broadcast_to_user(user_id, event, data))
    except Exception as e:
        print(f"Error emitting to user {user_id}: {e}")
</content>
</line_count>