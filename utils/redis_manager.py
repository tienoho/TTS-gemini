"""
Redis Manager for queue management, caching, and rate limiting
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import aioredis
from aioredis import Redis
from aioredis.exceptions import RedisError, ConnectionError

from ..config import get_settings


class RedisManager:
    """Redis manager for queue operations, caching, and rate limiting."""

    def __init__(self):
        self.settings = get_settings()
        self.redis: Optional[Redis] = None
        self.is_connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis = aioredis.from_url(
                self.settings.REDIS_URL,
                decode_responses=True,
                retry_on_timeout=True,
                max_connections=20,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            await self.redis.ping()
            self.is_connected = True
            print("✅ Connected to Redis")
        except (ConnectionError, RedisError) as e:
            print(f"❌ Failed to connect to Redis: {e}")
            self.is_connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.is_connected = False

    @asynccontextmanager
    async def get_connection(self):
        """Context manager for Redis connections."""
        if not self.is_connected:
            await self.connect()

        try:
            yield self.redis
        except Exception as e:
            print(f"Redis operation error: {e}")
            raise

    # Queue Operations
    async def enqueue_request(self, request_data: Dict[str, Any], priority: str = "normal") -> str:
        """Enqueue TTS request to Redis."""
        async with self.get_connection() as redis:
            # Generate unique request ID
            request_id = f"tts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(request_data)) % 10000}"

            # Add metadata
            queue_data = {
                "id": request_id,
                "data": request_data,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat(),
                "attempts": 0,
                "max_attempts": 3
            }

            # Choose queue based on priority
            queue_name = f"tts_queue_{priority}"

            # Add to queue
            await redis.lpush(queue_name, json.dumps(queue_data))

            # Update queue statistics
            await self._update_queue_stats(priority, "enqueued")

            return request_id

    async def dequeue_request(self, priority: str = "normal", timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Dequeue TTS request from Redis."""
        async with self.get_connection() as redis:
            queue_name = f"tts_queue_{priority}"

            try:
                # Try to get request from queue
                result = await redis.brpop(queue_name, timeout=timeout)
                if not result:
                    return None

                request_data = json.loads(result[1])

                # Update queue statistics
                await self._update_queue_stats(priority, "dequeued")

                return request_data

            except Exception as e:
                print(f"Error dequeuing request: {e}")
                return None

    async def get_queue_length(self, priority: str = "normal") -> int:
        """Get queue length for specific priority."""
        async with self.get_connection() as redis:
            queue_name = f"tts_queue_{priority}"
            return await redis.llen(queue_name)

    async def get_all_queue_lengths(self) -> Dict[str, int]:
        """Get lengths of all queues."""
        async with self.get_connection() as redis:
            queues = ["urgent", "high", "normal", "low"]
            result = {}

            for priority in queues:
                queue_name = f"tts_queue_{priority}"
                length = await redis.llen(queue_name)
                result[priority] = length

            return result

    # Status Caching
    async def set_request_status(self, request_id: str, status: str, **kwargs) -> None:
        """Set request status in cache."""
        async with self.get_connection() as redis:
            status_key = f"tts_status:{request_id}"
            status_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat(),
                **kwargs
            }

            await redis.hset(status_key, mapping=status_data)
            await redis.expire(status_key, 86400)  # 24 hours TTL

    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request status from cache."""
        async with self.get_connection() as redis:
            status_key = f"tts_status:{request_id}"
            status_data = await redis.hgetall(status_key)

            if not status_data:
                return None

            return status_data

    async def update_request_progress(self, request_id: str, progress: float, **kwargs) -> None:
        """Update request progress."""
        async with self.get_connection() as redis:
            status_key = f"tts_status:{request_id}"
            progress_data = {
                "progress": progress,
                "updated_at": datetime.utcnow().isoformat(),
                **kwargs
            }

            await redis.hset(status_key, mapping=progress_data)

    # Rate Limiting
    async def check_rate_limit(self, user_id: str, limit_type: str, limit_value: int, window_seconds: int) -> Dict[str, Any]:
        """Check rate limit for user."""
        async with self.get_connection() as redis:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)

            # Create rate limit key
            rate_limit_key = f"rate_limit:{user_id}:{limit_type}:{window_start.timestamp()}"

            # Get current count
            current_count = await redis.get(rate_limit_key)
            current_count = int(current_count) if current_count else 0

            # Check if under limit
            is_under_limit = current_count < limit_value

            return {
                "is_under_limit": is_under_limit,
                "current_count": current_count,
                "limit_value": limit_value,
                "remaining": max(0, limit_value - current_count),
                "reset_time": window_start + timedelta(seconds=window_seconds),
                "window_seconds": window_seconds
            }

    async def increment_rate_limit(self, user_id: str, limit_type: str, window_seconds: int) -> Dict[str, Any]:
        """Increment rate limit counter."""
        async with self.get_connection() as redis:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)

            # Create rate limit key
            rate_limit_key = f"rate_limit:{user_id}:{limit_type}:{window_start.timestamp()}"

            # Increment counter
            new_count = await redis.incr(rate_limit_key)

            # Set expiration if first request in window
            if new_count == 1:
                await redis.expire(rate_limit_key, window_seconds)

            # Get limit value (this would typically come from user settings)
            limit_value = self._get_default_limit(limit_type)

            return {
                "current_count": new_count,
                "limit_value": limit_value,
                "remaining": max(0, limit_value - new_count),
                "reset_time": window_start + timedelta(seconds=window_seconds)
            }

    def _get_default_limit(self, limit_type: str) -> int:
        """Get default limit for rate limit type."""
        limits = {
            "requests": 100,  # requests per minute
            "storage": 1000000,  # bytes per hour
            "downloads": 50,  # downloads per hour
        }
        return limits.get(limit_type, 100)

    # Caching
    async def set_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cache value."""
        async with self.get_connection() as redis:
            await redis.setex(key, ttl, json.dumps(value))

    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value."""
        async with self.get_connection() as redis:
            cached_value = await redis.get(key)
            if cached_value:
                return json.loads(cached_value)
            return None

    async def delete_cache(self, key: str) -> None:
        """Delete cache value."""
        async with self.get_connection() as redis:
            await redis.delete(key)

    # Statistics and Monitoring
    async def _update_queue_stats(self, priority: str, operation: str) -> None:
        """Update queue statistics."""
        async with self.get_connection() as redis:
            stats_key = f"queue_stats:{priority}:{operation}"
            await redis.incr(stats_key)

            # Set expiration for stats (24 hours)
            await redis.expire(stats_key, 86400)

    async def get_queue_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get queue statistics."""
        async with self.get_connection() as redis:
            stats = {}
            priorities = ["urgent", "high", "normal", "low"]
            operations = ["enqueued", "dequeued", "processed", "failed"]

            for priority in priorities:
                priority_stats = {}
                for operation in operations:
                    stats_key = f"queue_stats:{priority}:{operation}"
                    count = await redis.get(stats_key)
                    priority_stats[operation] = int(count) if count else 0
                stats[priority] = priority_stats

            return stats

    # Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            async with self.get_connection() as redis:
                # Test basic operations
                await redis.ping()

                # Get connection info
                info = await redis.info("server")

                return {
                    "status": "healthy",
                    "connection": "connected",
                    "version": info.get("redis_version", "unknown"),
                    "uptime": info.get("uptime_in_seconds", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "0B"),
                    "total_connections_received": info.get("total_connections_received", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connection": "disconnected",
                "error": str(e)
            }

    # Cleanup Operations
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired keys."""
        async with self.get_connection() as redis:
            # Clean up old rate limit keys
            pattern = "rate_limit:*"
            keys = await redis.keys(pattern)

            if keys:
                await redis.delete(*keys)

            return len(keys)

    async def cleanup_old_status_keys(self, days: int = 7) -> int:
        """Clean up old status keys."""
        async with self.get_connection() as redis:
            # Clean up status keys older than specified days
            pattern = "tts_status:*"
            keys = await redis.keys(pattern)

            deleted_count = 0
            cutoff_time = datetime.utcnow() - timedelta(days=days)

            for key in keys:
                # Check if key has expired
                ttl = await redis.ttl(key)
                if ttl == -1:  # Key has no expiration
                    # Check last update time from key content
                    status_data = await redis.hgetall(key)
                    updated_at_str = status_data.get("updated_at")

                    if updated_at_str:
                        try:
                            updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                            if updated_at < cutoff_time:
                                await redis.delete(key)
                                deleted_count += 1
                        except:
                            pass

            return deleted_count


# Global Redis Manager instance
redis_manager = RedisManager()


# Utility functions for easy access
async def get_redis_manager() -> RedisManager:
    """Get Redis manager instance."""
    return redis_manager


async def enqueue_tts_request(request_data: Dict[str, Any], priority: str = "normal") -> str:
    """Enqueue TTS request."""
    return await redis_manager.enqueue_request(request_data, priority)


async def get_tts_status(request_id: str) -> Optional[Dict[str, Any]]:
    """Get TTS request status."""
    return await redis_manager.get_request_status(request_id)


async def update_tts_progress(request_id: str, progress: float, **kwargs) -> None:
    """Update TTS request progress."""
    await redis_manager.update_request_progress(request_id, progress, **kwargs)


async def check_user_rate_limit(user_id: str, limit_type: str, limit_value: int, window_seconds: int) -> Dict[str, Any]:
    """Check user rate limit."""
    return await redis_manager.check_rate_limit(user_id, limit_type, limit_value, window_seconds)