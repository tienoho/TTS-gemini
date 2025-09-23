"""
WebSocket connection health monitoring service
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
from threading import Lock

from flask import current_app

from .websocket_manager import get_websocket_manager
from .redis_manager import redis_manager
from ..config.websocket import get_websocket_settings


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ConnectionHealth:
    """Represents the health of a WebSocket connection."""

    def __init__(self, sid: str, user_id: Optional[int] = None, ip_address: str = ""):
        self.sid = sid
        self.user_id = user_id
        self.ip_address = ip_address
        self.last_ping = datetime.utcnow()
        self.last_pong = datetime.utcnow()
        self.ping_count = 0
        self.pong_count = 0
        self.message_count = 0
        self.error_count = 0
        self.latency_samples: List[float] = []
        self.status = HealthStatus.HEALTHY
        self.last_status_change = datetime.utcnow()

    def record_ping(self):
        """Record a ping sent."""
        self.ping_count += 1
        self.last_ping = datetime.utcnow()

    def record_pong(self):
        """Record a pong received."""
        self.pong_count += 1
        self.last_pong = datetime.utcnow()

        # Calculate latency
        if self.last_ping:
            latency = (self.last_pong - self.last_ping).total_seconds() * 1000  # ms
            self.latency_samples.append(latency)

            # Keep only last 10 samples
            if len(self.latency_samples) > 10:
                self.latency_samples = self.latency_samples[-10:]

    def record_message(self):
        """Record a message sent/received."""
        self.message_count += 1

    def record_error(self):
        """Record an error."""
        self.error_count += 1

    def get_latency(self) -> float:
        """Get average latency."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    def get_health_score(self) -> float:
        """Calculate health score (0-100)."""
        score = 100.0

        # Reduce score based on ping/pong ratio
        if self.ping_count > 0:
            pong_ratio = self.pong_count / self.ping_count
            score -= (1 - pong_ratio) * 20

        # Reduce score based on error rate
        if self.message_count > 0:
            error_rate = self.error_count / self.message_count
            score -= error_rate * 30

        # Reduce score based on latency
        latency = self.get_latency()
        if latency > 1000:  # High latency
            score -= 20
        elif latency > 500:  # Medium latency
            score -= 10

        return max(0, min(100, score))

    def update_status(self):
        """Update health status based on current metrics."""
        old_status = self.status
        score = self.get_health_score()

        if score >= 90:
            self.status = HealthStatus.HEALTHY
        elif score >= 70:
            self.status = HealthStatus.DEGRADED
        elif score >= 50:
            self.status = HealthStatus.UNHEALTHY
        else:
            self.status = HealthStatus.CRITICAL

        if old_status != self.status:
            self.last_status_change = datetime.utcnow()

    def is_stale(self, timeout_seconds: int = 60) -> bool:
        """Check if connection is stale."""
        return (datetime.utcnow() - self.last_pong).total_seconds() > timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sid": self.sid,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "status": self.status.value,
            "health_score": self.get_health_score(),
            "latency_ms": self.get_latency(),
            "ping_count": self.ping_count,
            "pong_count": self.pong_count,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_ping": self.last_ping.isoformat(),
            "last_pong": self.last_pong.isoformat(),
            "last_status_change": self.last_status_change.isoformat()
        }


class WebSocketHealthMonitor:
    """Monitors WebSocket connection health."""

    def __init__(self):
        self.settings = get_websocket_settings()
        self.connection_health: Dict[str, ConnectionHealth] = {}
        self._lock = Lock()
        self._monitoring_task = None
        self._cleanup_task = None
        self._alert_callbacks: List[Callable] = []

        # Start monitoring tasks
        self._start_monitoring_tasks()

    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        if self.settings.WS_ENABLE_HEARTBEAT:
            self._monitoring_task = asyncio.create_task(self._monitor_connections())
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_connections())

    async def _monitor_connections(self):
        """Background task to monitor connection health."""
        while True:
            try:
                await asyncio.sleep(self.settings.WS_HEALTH_CHECK_INTERVAL)

                with self._lock:
                    # Update health for all connections
                    for sid, health in self.connection_health.items():
                        health.update_status()

                        # Check for stale connections
                        if health.is_stale(self.settings.ping_timeout_total):
                            health.status = HealthStatus.CRITICAL

                    # Trigger alerts for unhealthy connections
                    await self._check_and_alert()

            except Exception as e:
                logging.error(f"Error in health monitoring: {e}")

    async def _cleanup_stale_connections(self):
        """Background task to cleanup stale connection health data."""
        while True:
            try:
                await asyncio.sleep(self.settings.WS_CONNECTION_CLEANUP_INTERVAL * 2)

                with self._lock:
                    stale_sids = []
                    cutoff_time = datetime.utcnow() - timedelta(hours=1)

                    for sid, health in self.connection_health.items():
                        if health.last_pong < cutoff_time:
                            stale_sids.append(sid)

                    for sid in stale_sids:
                        del self.connection_health[sid]

                    if stale_sids:
                        logging.info(f"Cleaned up {len(stale_sids)} stale connection health records")

            except Exception as e:
                logging.error(f"Error in cleanup task: {e}")

    async def _check_and_alert(self):
        """Check connections and trigger alerts."""
        try:
            unhealthy_connections = []
            critical_connections = []

            for health in self.connection_health.values():
                if health.status == HealthStatus.UNHEALTHY:
                    unhealthy_connections.append(health)
                elif health.status == HealthStatus.CRITICAL:
                    critical_connections.append(health)

            # Trigger alerts
            if unhealthy_connections:
                await self._trigger_alerts("unhealthy_connections", unhealthy_connections)

            if critical_connections:
                await self._trigger_alerts("critical_connections", critical_connections)

        except Exception as e:
            logging.error(f"Error checking and alerting: {e}")

    async def _trigger_alerts(self, alert_type: str, connections: List[ConnectionHealth]):
        """Trigger alerts for connection issues."""
        try:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_type, connections)
                    else:
                        callback(alert_type, connections)
                except Exception as e:
                    logging.error(f"Error in alert callback: {e}")

        except Exception as e:
            logging.error(f"Error triggering alerts: {e}")

    def register_connection(self, sid: str, user_id: Optional[int] = None, ip_address: str = ""):
        """Register a new connection for health monitoring."""
        with self._lock:
            if sid not in self.connection_health:
                self.connection_health[sid] = ConnectionHealth(sid, user_id, ip_address)
                logging.debug(f"Registered connection {sid} for health monitoring")

    def unregister_connection(self, sid: str):
        """Unregister a connection from health monitoring."""
        with self._lock:
            if sid in self.connection_health:
                del self.connection_health[sid]
                logging.debug(f"Unregistered connection {sid} from health monitoring")

    def record_ping(self, sid: str):
        """Record ping for a connection."""
        with self._lock:
            if sid in self.connection_health:
                self.connection_health[sid].record_ping()

    def record_pong(self, sid: str):
        """Record pong for a connection."""
        with self._lock:
            if sid in self.connection_health:
                self.connection_health[sid].record_pong()

    def record_message(self, sid: str):
        """Record message for a connection."""
        with self._lock:
            if sid in self.connection_health:
                self.connection_health[sid].record_message()

    def record_error(self, sid: str):
        """Record error for a connection."""
        with self._lock:
            if sid in self.connection_health:
                self.connection_health[sid].record_error()

    def get_connection_health(self, sid: str) -> Optional[ConnectionHealth]:
        """Get health information for a specific connection."""
        with self._lock:
            return self.connection_health.get(sid)

    def get_all_connection_health(self) -> Dict[str, ConnectionHealth]:
        """Get health information for all connections."""
        with self._lock:
            return self.connection_health.copy()

    def get_unhealthy_connections(self) -> List[ConnectionHealth]:
        """Get list of unhealthy connections."""
        with self._lock:
            return [
                health for health in self.connection_health.values()
                if health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
            ]

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all connections."""
        with self._lock:
            total_connections = len(self.connection_health)
            healthy = 0
            degraded = 0
            unhealthy = 0
            critical = 0

            total_latency = 0
            total_health_score = 0

            for health in self.connection_health.values():
                total_latency += health.get_latency()
                total_health_score += health.get_health_score()

                if health.status == HealthStatus.HEALTHY:
                    healthy += 1
                elif health.status == HealthStatus.DEGRADED:
                    degraded += 1
                elif health.status == HealthStatus.UNHEALTHY:
                    unhealthy += 1
                elif health.status == HealthStatus.CRITICAL:
                    critical += 1

            avg_latency = total_latency / total_connections if total_connections > 0 else 0
            avg_health_score = total_health_score / total_connections if total_connections > 0 else 0

            return {
                "total_connections": total_connections,
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "critical": critical,
                "avg_latency_ms": avg_latency,
                "avg_health_score": avg_health_score,
                "unhealthy_connections": len(self.get_unhealthy_connections())
            }

    def register_alert_callback(self, callback: Callable):
        """Register an alert callback."""
        self._alert_callbacks.append(callback)

    def unregister_alert_callback(self, callback: Callable):
        """Unregister an alert callback."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        try:
            # Get WebSocket manager stats
            manager = get_websocket_manager()
            connection_count = manager.get_connection_count()
            room_count = manager.get_room_count()

            # Get health summary
            health_summary = self.get_health_summary()

            # Get Redis stats
            redis_stats = await redis_manager.health_check()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "websocket_connections": connection_count,
                "websocket_rooms": room_count,
                "health_summary": health_summary,
                "redis_status": redis_stats.get("status", "unknown"),
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
            }

        except Exception as e:
            logging.error(f"Error getting monitoring stats: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }


# Global health monitor instance
websocket_health_monitor = WebSocketHealthMonitor()


# Utility functions
def get_websocket_health_monitor() -> WebSocketHealthMonitor:
    """Get WebSocket health monitor instance."""
    return websocket_health_monitor


def register_connection_health(sid: str, user_id: Optional[int] = None, ip_address: str = ""):
    """Register connection for health monitoring."""
    websocket_health_monitor.register_connection(sid, user_id, ip_address)


def unregister_connection_health(sid: str):
    """Unregister connection from health monitoring."""
    websocket_health_monitor.unregister_connection(sid)


def record_ping_health(sid: str):
    """Record ping for health monitoring."""
    websocket_health_monitor.record_ping(sid)


def record_pong_health(sid: str):
    """Record pong for health monitoring."""
    websocket_health_monitor.record_pong(sid)


def record_message_health(sid: str):
    """Record message for health monitoring."""
    websocket_health_monitor.record_message(sid)


def record_error_health(sid: str):
    """Record error for health monitoring."""
    websocket_health_monitor.record_error(sid)


def get_connection_health_info(sid: str) -> Optional[Dict]:
    """Get health information for a connection."""
    health = websocket_health_monitor.get_connection_health(sid)
    return health.to_dict() if health else None


def get_health_summary() -> Dict[str, Any]:
    """Get health summary."""
    return websocket_health_monitor.get_health_summary()


def get_unhealthy_connections() -> List[Dict]:
    """Get list of unhealthy connections."""
    unhealthy = websocket_health_monitor.get_unhealthy_connections()
    return [health.to_dict() for health in unhealthy]
</content>
</line_count>