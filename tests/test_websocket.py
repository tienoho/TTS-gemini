"""
Tests for WebSocket functionality
"""

import pytest
import pytest_asyncio
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from utils.websocket_manager import WebSocketManager, WebSocketConnection
from utils.progress_streamer import (
    ProgressStreamingService,
    ProgressTracker,
    ProgressStatus,
    ProgressEvent
)
from utils.websocket_auth import WebSocketAuthMiddleware
from utils.websocket_health import (
    WebSocketHealthMonitor,
    ConnectionHealth,
    HealthStatus
)
from config.websocket import WebSocketSettings


class TestWebSocketConnection:
    """Test WebSocket connection class."""

    def test_connection_creation(self):
        """Test creating a WebSocket connection."""
        conn = WebSocketConnection("test_sid", 1, "127.0.0.1")

        assert conn.sid == "test_sid"
        assert conn.user_id == 1
        assert conn.ip_address == "127.0.0.1"
        assert conn.is_authenticated is True
        assert len(conn.rooms) == 0
        assert conn.message_count == 0

    def test_connection_room_management(self):
        """Test adding and removing rooms."""
        conn = WebSocketConnection("test_sid")

        # Add rooms
        conn.add_room("room1")
        conn.add_room("room2")
        assert "room1" in conn.rooms
        assert "room2" in conn.rooms
        assert len(conn.rooms) == 2

        # Remove room
        conn.remove_room("room1")
        assert "room1" not in conn.rooms
        assert len(conn.rooms) == 1

    def test_connection_ping_update(self):
        """Test ping timestamp update."""
        conn = WebSocketConnection("test_sid")
        old_ping = conn.last_ping
        conn.update_ping()
        assert conn.last_ping > old_ping

    def test_connection_rate_limit(self):
        """Test rate limiting functionality."""
        conn = WebSocketConnection("test_sid")

        # Should allow messages initially
        assert conn.can_send_message() is True

        # Fill up message count
        conn.message_count = 100
        conn.last_message_time = datetime.utcnow()

        # Should not allow more messages
        assert conn.can_send_message(100, 60) is False

    def test_connection_alive_status(self):
        """Test connection alive status."""
        conn = WebSocketConnection("test_sid")
        conn.update_ping()

        # Should be alive
        assert conn.is_alive(60) is True

        # Simulate old ping
        conn.last_ping = datetime.utcnow() - timedelta(minutes=5)
        assert conn.is_alive(60) is False


class TestWebSocketManager:
    """Test WebSocket manager functionality."""

    @pytest.fixture
    def manager(self):
        """Create WebSocket manager instance."""
        return WebSocketManager()

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.connections) == 0
        assert len(manager.rooms) == 0
        assert len(manager.user_connections) == 0
        assert len(manager.ip_connections) == 0

    @pytest.mark.asyncio
    async def test_connection_handling(self, manager):
        """Test connection handling."""
        # Test successful connection
        success = await manager.handle_connect("sid1", {"token": "test"}, "127.0.0.1")
        assert success is True
        assert "sid1" in manager.connections

        # Test connection info
        conn_info = manager.get_connection_info("sid1")
        assert conn_info is not None
        assert conn_info["sid"] == "sid1"
        assert conn_info["ip_address"] == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_room_management(self, manager):
        """Test room management."""
        # Setup connection
        await manager.handle_connect("sid1", None, "127.0.0.1")

        # Join room
        success = await manager.join_room("sid1", "test_room")
        assert success is True

        # Check room membership
        room_info = manager.get_room_info("test_room")
        assert room_info is not None
        assert room_info["connection_count"] == 1
        assert "sid1" in room_info["connections"]

        # Leave room
        await manager.leave_room("sid1", "test_room")
        room_info = manager.get_room_info("test_room")
        assert room_info is None

    @pytest.mark.asyncio
    async def test_broadcasting(self, manager):
        """Test message broadcasting."""
        # Setup connections
        await manager.handle_connect("sid1", None, "127.0.0.1")
        await manager.handle_connect("sid2", None, "127.0.0.2")
        await manager.join_room("sid1", "test_room")
        await manager.join_room("sid2", "test_room")

        # Mock emit function
        with patch('utils.websocket_manager.emit') as mock_emit:
            await manager.broadcast_to_room("test_room", "test_event", {"data": "test"})

            # Should have been called
            mock_emit.assert_called()

    @pytest.mark.asyncio
    async def test_user_broadcasting(self, manager):
        """Test user-specific broadcasting."""
        # Setup connections with user IDs
        await manager.handle_connect("sid1", None, "127.0.0.1")
        await manager.handle_connect("sid2", None, "127.0.0.2")

        # Mock the connection objects to have user IDs
        manager.connections["sid1"].user_id = 1
        manager.connections["sid2"].user_id = 2
        manager.user_connections[1].add("sid1")
        manager.user_connections[2].add("sid2")

        # Mock emit function
        with patch('utils.websocket_manager.emit') as mock_emit:
            await manager.broadcast_to_user(1, "test_event", {"data": "test"})

            # Should have been called once for user 1
            assert mock_emit.call_count == 1

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, manager):
        """Test connection cleanup."""
        # Setup connection
        await manager.handle_connect("sid1", None, "127.0.0.1")
        await manager.join_room("sid1", "test_room")

        # Disconnect
        await manager.handle_disconnect("sid1")

        # Should be cleaned up
        assert "sid1" not in manager.connections
        assert len(manager.rooms) == 0


class TestProgressStreamingService:
    """Test progress streaming service."""

    @pytest.fixture
    def streamer(self):
        """Create progress streaming service instance."""
        return ProgressStreamingService()

    def test_tracker_creation(self, streamer):
        """Test progress tracker creation."""
        tracker = streamer.create_tracker("test_request", 1)

        assert tracker.request_id == "test_request"
        assert tracker.user_id == 1
        assert tracker.status == ProgressStatus.PENDING
        assert tracker.progress == 0

    def test_progress_updates(self, streamer):
        """Test progress updates."""
        tracker = streamer.create_tracker("test_request", 1)

        # Update progress
        success = streamer.update_progress(
            "test_request",
            ProgressStatus.PROCESSING,
            "Processing...",
            50,
            {"stage": "processing"}
        )

        assert success is True
        assert tracker.status == ProgressStatus.PROCESSING
        assert tracker.message == "Processing..."
        assert tracker.progress == 50
        assert "stage" in tracker.metadata

    def test_request_completion(self, streamer):
        """Test request completion."""
        tracker = streamer.create_tracker("test_request", 1)

        # Mark as completed
        success = streamer.mark_request_completed(
            "test_request",
            "Completed successfully",
            {"result": "success"}
        )

        assert success is True
        assert tracker.status == ProgressStatus.COMPLETED
        assert tracker.progress == 100
        assert tracker.is_completed() is True

    def test_request_failure(self, streamer):
        """Test request failure."""
        tracker = streamer.create_tracker("test_request", 1)

        # Mark as failed
        success = streamer.mark_request_failed(
            "test_request",
            "Processing failed"
        )

        assert success is True
        assert tracker.status == ProgressStatus.FAILED
        assert tracker.is_completed() is True

    def test_request_cancellation(self, streamer):
        """Test request cancellation."""
        tracker = streamer.create_tracker("test_request", 1)

        # Cancel request
        success = streamer.cancel_request("test_request")

        assert success is True
        assert tracker.status == ProgressStatus.CANCELLED
        assert tracker.is_completed() is True

    def test_subscription_management(self, streamer):
        """Test subscription management."""
        # Subscribe
        success = streamer.subscribe_to_request("test_request", "sid1")
        assert success is True

        # Unsubscribe
        success = streamer.unsubscribe_from_request("test_request", "sid1")
        assert success is True

    def test_tracker_retrieval(self, streamer):
        """Test tracker retrieval."""
        # Create tracker
        tracker = streamer.create_tracker("test_request", 1)

        # Retrieve tracker
        retrieved = streamer.get_tracker("test_request")
        assert retrieved is tracker

        # Test non-existent tracker
        non_existent = streamer.get_tracker("non_existent")
        assert non_existent is None

    def test_user_trackers(self, streamer):
        """Test user tracker filtering."""
        # Create trackers for different users
        tracker1 = streamer.create_tracker("request1", 1)
        tracker2 = streamer.create_tracker("request2", 1)
        tracker3 = streamer.create_tracker("request3", 2)

        # Get user trackers
        user1_trackers = streamer.get_user_trackers(1)
        assert len(user1_trackers) == 2
        assert "request1" in user1_trackers
        assert "request2" in user1_trackers

        user2_trackers = streamer.get_user_trackers(2)
        assert len(user2_trackers) == 1
        assert "request3" in user2_trackers


class TestWebSocketAuthMiddleware:
    """Test WebSocket authentication middleware."""

    @pytest.fixture
    def auth_middleware(self):
        """Create auth middleware instance."""
        return WebSocketAuthMiddleware()

    def test_authentication_required(self, auth_middleware):
        """Test authentication requirement."""
        # Test with no auth required
        with patch.object(auth_middleware.settings, 'WS_AUTH_REQUIRED', False):
            result = auth_middleware.authenticate({})
            assert result is not None
            assert result["is_authenticated"] is False

    def test_token_authentication(self, auth_middleware):
        """Test token-based authentication."""
        # Mock auth service
        with patch('utils.websocket_auth.auth_service') as mock_auth:
            mock_auth.verify_token.return_value = {"sub": "1", "username": "test"}
            mock_auth.get_user_from_token.return_value = Mock(
                id=1,
                username="test",
                email="test@example.com",
                account_tier="free"
            )

            auth_data = {"token": "valid_token"}
            result = auth_middleware.authenticate(auth_data)

            assert result is not None
            assert result["user_id"] == 1
            assert result["is_authenticated"] is True

    def test_invalid_token(self, auth_middleware):
        """Test invalid token handling."""
        with patch('utils.websocket_auth.auth_service') as mock_auth:
            mock_auth.verify_token.side_effect = Exception("Invalid token")

            auth_data = {"token": "invalid_token"}
            result = auth_middleware.authenticate(auth_data)

            assert result is None

    @pytest.mark.asyncio
    async def test_rate_limit_check(self, auth_middleware):
        """Test rate limit checking."""
        with patch('utils.websocket_auth.redis_manager') as mock_redis:
            mock_redis.get_cache.return_value = 5

            # Test within limits
            is_allowed = await auth_middleware.check_rate_limit("1", "127.0.0.1")
            assert is_allowed is True

            # Test over limits
            with patch.object(auth_middleware.settings, 'WS_MAX_CONNECTIONS_PER_IP', 5):
                is_allowed = await auth_middleware.check_rate_limit("1", "127.0.0.1")
                assert is_allowed is False

    @pytest.mark.asyncio
    async def test_connection_count_management(self, auth_middleware):
        """Test connection count management."""
        with patch('utils.websocket_auth.redis_manager') as mock_redis:
            mock_redis.get_cache.return_value = 0

            # Test increment
            await auth_middleware.increment_connection_count(1, "127.0.0.1")

            # Test decrement
            await auth_middleware.decrement_connection_count(1, "127.0.0.1")


class TestWebSocketHealthMonitor:
    """Test WebSocket health monitoring."""

    @pytest.fixture
    def health_monitor(self):
        """Create health monitor instance."""
        return WebSocketHealthMonitor()

    def test_connection_registration(self, health_monitor):
        """Test connection registration."""
        health_monitor.register_connection("sid1", 1, "127.0.0.1")

        health = health_monitor.get_connection_health("sid1")
        assert health is not None
        assert health.sid == "sid1"
        assert health.user_id == 1

    def test_health_recording(self, health_monitor):
        """Test health metric recording."""
        health_monitor.register_connection("sid1")

        # Record ping/pong
        health_monitor.record_ping("sid1")
        health_monitor.record_pong("sid1")

        # Record message
        health_monitor.record_message("sid1")

        health = health_monitor.get_connection_health("sid1")
        assert health.ping_count == 1
        assert health.pong_count == 1
        assert health.message_count == 1

    def test_health_score_calculation(self, health_monitor):
        """Test health score calculation."""
        health_monitor.register_connection("sid1")

        # Perfect health
        health_monitor.record_ping("sid1")
        health_monitor.record_pong("sid1")
        health_monitor.record_message("sid1")

        health = health_monitor.get_connection_health("sid1")
        health.update_status()

        assert health.get_health_score() == 100.0
        assert health.status == HealthStatus.HEALTHY

    def test_unhealthy_detection(self, health_monitor):
        """Test unhealthy connection detection."""
        health_monitor.register_connection("sid1")

        # Record many errors
        for _ in range(10):
            health_monitor.record_error("sid1")

        health = health_monitor.get_connection_health("sid1")
        health.update_status()

        assert health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]

    def test_health_summary(self, health_monitor):
        """Test health summary generation."""
        # Register multiple connections
        health_monitor.register_connection("sid1", 1, "127.0.0.1")
        health_monitor.register_connection("sid2", 2, "127.0.0.2")

        # Record some activity
        health_monitor.record_ping("sid1")
        health_monitor.record_pong("sid1")
        health_monitor.record_message("sid1")

        summary = health_monitor.get_health_summary()

        assert summary["total_connections"] == 2
        assert "healthy" in summary
        assert "avg_health_score" in summary

    def test_unhealthy_connections_filtering(self, health_monitor):
        """Test unhealthy connections filtering."""
        # Register connections
        health_monitor.register_connection("sid1")
        health_monitor.register_connection("sid2")

        # Make one connection unhealthy
        for _ in range(10):
            health_monitor.record_error("sid1")

        health_monitor.get_connection_health("sid1").update_status()

        unhealthy = health_monitor.get_unhealthy_connections()

        assert len(unhealthy) >= 1
        assert any(h.sid == "sid1" for h in unhealthy)


class TestWebSocketSettings:
    """Test WebSocket settings."""

    def test_settings_initialization(self):
        """Test settings initialization."""
        settings = WebSocketSettings()

        assert settings.WS_MAX_CONNECTIONS == 1000
        assert settings.WS_PING_INTERVAL == 20
        assert settings.WS_AUTH_REQUIRED is True

    def test_settings_properties(self):
        """Test computed properties."""
        settings = WebSocketSettings()

        assert settings.ping_timeout_total == 30  # PING_INTERVAL + PING_TIMEOUT
        assert settings.connection_timeout_total == 35  # CONNECTION_TIMEOUT + CLOSE_TIMEOUT


# Integration Tests
class TestWebSocketIntegration:
    """Integration tests for WebSocket components."""

    @pytest.mark.asyncio
    async def test_full_websocket_flow(self):
        """Test full WebSocket flow from connection to progress streaming."""
        # Initialize components
        manager = WebSocketManager()
        streamer = ProgressStreamingService()
        health_monitor = WebSocketHealthMonitor()

        # Simulate connection
        success = await manager.handle_connect("test_sid", None, "127.0.0.1")
        assert success is True

        # Register for health monitoring
        health_monitor.register_connection("test_sid", None, "127.0.0.1")

        # Create progress tracker
        tracker = streamer.create_tracker("test_request", None)

        # Subscribe to progress updates
        success = streamer.subscribe_to_request("test_request", "test_sid")
        assert success is True

        # Update progress
        success = streamer.update_progress(
            "test_request",
            ProgressStatus.PROCESSING,
            "Processing...",
            50
        )
        assert success is True

        # Verify tracker state
        assert tracker.status == ProgressStatus.PROCESSING
        assert tracker.progress == 50

        # Record health metrics
        health_monitor.record_ping("test_sid")
        health_monitor.record_pong("test_sid")
        health_monitor.record_message("test_sid")

        # Verify health
        health = health_monitor.get_connection_health("test_sid")
        assert health is not None
        assert health.ping_count == 1
        assert health.pong_count == 1

        # Cleanup
        await manager.handle_disconnect("test_sid")
        health_monitor.unregister_connection("test_sid")

        # Verify cleanup
        assert "test_sid" not in manager.connections
        assert health_monitor.get_connection_health("test_sid") is None

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in WebSocket components."""
        manager = WebSocketManager()
        streamer = ProgressStreamingService()

        # Test invalid connection
        success = await manager.handle_connect("", None, "")
        assert success is True  # Should handle gracefully

        # Test invalid progress update
        success = streamer.update_progress("non_existent", ProgressStatus.PROCESSING, "test")
        assert success is False

        # Test invalid subscription
        success = streamer.subscribe_to_request("non_existent", "invalid_sid")
        assert success is True  # Should handle gracefully


# Performance Tests
class TestWebSocketPerformance:
    """Performance tests for WebSocket components."""

    @pytest.mark.asyncio
    async def test_multiple_connections(self):
        """Test handling multiple connections."""
        manager = WebSocketManager()

        # Create many connections
        for i in range(100):
            success = await manager.handle_connect(f"sid_{i}", None, f"127.0.0.{i % 255}")
            assert success is True

        # Verify all connections
        assert manager.get_connection_count() == 100

        # Test room operations
        for i in range(100):
            await manager.join_room(f"sid_{i}", "test_room")

        room_info = manager.get_room_info("test_room")
        assert room_info["connection_count"] == 100

        # Cleanup
        for i in range(100):
            await manager.handle_disconnect(f"sid_{i}")

        assert manager.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_progress_broadcasting(self):
        """Test progress broadcasting to multiple subscribers."""
        manager = WebSocketManager()
        streamer = ProgressStreamingService()

        # Create multiple connections
        sids = []
        for i in range(10):
            sid = f"sid_{i}"
            sids.append(sid)
            await manager.handle_connect(sid, None, f"127.0.0.{i}")
            streamer.subscribe_to_request("test_request", sid)

        # Update progress
        success = streamer.update_progress(
            "test_request",
            ProgressStatus.COMPLETED,
            "Done",
            100
        )
        assert success is True

        # Verify tracker
        tracker = streamer.get_tracker("test_request")
        assert tracker.status == ProgressStatus.COMPLETED

        # Cleanup
        for sid in sids:
            await manager.handle_disconnect(sid)
</content>
</line_count>