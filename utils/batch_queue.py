"""
Batch Queue Management System
Redis-based queue system cho batch processing với priority queuing và batch splitting
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import heapq

from models.batch_request import BatchRequest, Priority, BatchStatus
from utils.redis_manager import redis_manager


class BatchQueueItem:
    """Queue item cho batch processing"""

    def __init__(self, batch_request: BatchRequest, priority_score: int = 0):
        self.batch_request = batch_request
        self.priority_score = priority_score
        self.queued_at = datetime.utcnow()
        self.retry_count = 0
        self.max_retries = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary cho Redis storage"""
        return {
            "batch_request": self.batch_request.dict(),
            "priority_score": self.priority_score,
            "queued_at": self.queued_at.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchQueueItem':
        """Create from dictionary"""
        batch_request = BatchRequest(**data["batch_request"])
        item = cls(batch_request, data["priority_score"])
        item.queued_at = datetime.fromisoformat(data["queued_at"])
        item.retry_count = data["retry_count"]
        item.max_retries = data["max_retries"]
        return item

    def __lt__(self, other: 'BatchQueueItem') -> bool:
        """For priority queue comparison (higher priority first)"""
        return self.priority_score > other.priority_score


class BatchQueueManager:
    """Manager cho batch queue system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.priority_queues = {
            Priority.HIGH: "batch_queue_high",
            Priority.NORMAL: "batch_queue_normal",
            Priority.LOW: "batch_queue_low"
        }
        self.dead_letter_queue = "batch_queue_dead_letter"
        self.processing_queue = "batch_queue_processing"
        self.batch_splits = "batch_splits"

    def _calculate_priority_score(self, batch_request: BatchRequest) -> int:
        """Calculate priority score cho batch request"""
        base_score = {
            Priority.HIGH: 1000,
            Priority.NORMAL: 500,
            Priority.LOW: 100
        }.get(batch_request.priority, 500)

        # Bonus points cho small batches (processed faster)
        size_bonus = max(0, 50 - len(batch_request.items))

        # Time bonus (older requests get higher priority)
        age_hours = (datetime.utcnow() - batch_request.created_at).total_seconds() / 3600
        age_bonus = min(100, age_hours * 10)

        return int(base_score + size_bonus + age_bonus)

    async def enqueue_batch(self, batch_request: BatchRequest) -> bool:
        """
        Enqueue batch request vào priority queue
        Returns True if successful
        """
        try:
            # Check if batch needs splitting
            if len(batch_request.items) > 50:  # Split large batches
                await self._split_and_enqueue_batch(batch_request)
                return True

            # Calculate priority score
            priority_score = self._calculate_priority_score(batch_request)

            # Create queue item
            queue_item = BatchQueueItem(batch_request, priority_score)

            # Get appropriate queue name
            queue_name = self.priority_queues[batch_request.priority]

            # Add to Redis sorted set (using priority score as score)
            await redis_manager.redis.zadd(
                queue_name,
                {json.dumps(queue_item.to_dict()): priority_score}
            )

            self.logger.info(f"Batch {batch_request.id} enqueued with priority {batch_request.priority}")
            return True

        except Exception as e:
            self.logger.error(f"Error enqueuing batch {batch_request.id}: {str(e)}")
            return False

    async def _split_and_enqueue_batch(self, batch_request: BatchRequest):
        """Split large batch và enqueue từng phần"""
        items = batch_request.items
        chunk_size = 25  # Split into chunks of 25 items

        # Create sub-batches
        for i in range(0, len(items), chunk_size):
            chunk_items = items[i:i + chunk_size]

            # Create new batch request cho chunk này
            chunk_batch = BatchRequest(
                name=f"{batch_request.name}_part_{i//chunk_size + 1}",
                items=chunk_items,
                priority=batch_request.priority,
                webhook_url=batch_request.webhook_url,
                metadata={
                    **(batch_request.metadata or {}),
                    "parent_batch_id": str(batch_request.id),
                    "chunk_index": i//chunk_size,
                    "total_chunks": (len(items) + chunk_size - 1) // chunk_size
                }
            )

            # Enqueue chunk
            await self.enqueue_batch(chunk_batch)

        # Store split information
        split_info = {
            "parent_batch_id": str(batch_request.id),
            "total_chunks": (len(items) + chunk_size - 1) // chunk_size,
            "chunk_size": chunk_size,
            "created_at": datetime.utcnow().isoformat()
        }

        await redis_manager.set_cache(
            f"{self.batch_splits}:{batch_request.id}",
            split_info,
            ttl=86400  # 24 hours
        )

        self.logger.info(f"Batch {batch_request.id} split into {(len(items) + chunk_size - 1) // chunk_size} chunks")

    async def dequeue_batch(self, priority: Priority = Priority.NORMAL) -> Optional[BatchQueueItem]:
        """
        Dequeue batch với highest priority từ specified priority level
        Returns BatchQueueItem if available, None otherwise
        """
        try:
            queue_name = self.priority_queues[priority]

            # Get highest priority item (lowest score first for min-heap behavior)
            result = await redis_manager.redis.zpopmax(queue_name)

            if not result:
                return None

            score, item_data = result[0]
            item_dict = json.loads(item_data)

            queue_item = BatchQueueItem.from_dict(item_dict)
            self.logger.info(f"Batch {queue_item.batch_request.id} dequeued from {priority.value} queue")

            return queue_item

        except Exception as e:
            self.logger.error(f"Error dequeuing batch from {priority.value} queue: {str(e)}")
            return None

    async def dequeue_highest_priority(self) -> Optional[BatchQueueItem]:
        """
        Dequeue batch với highest priority từ all queues
        Returns BatchQueueItem if available, None otherwise
        """
        try:
            # Check queues theo priority order: HIGH -> NORMAL -> LOW
            for priority in [Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                batch_item = await self.dequeue_batch(priority)
                if batch_item:
                    return batch_item

            return None

        except Exception as e:
            self.logger.error(f"Error dequeuing highest priority batch: {str(e)}")
            return None

    async def mark_processing(self, batch_id: UUID) -> bool:
        """Mark batch as being processed"""
        try:
            # Move from queue to processing set
            processing_key = f"{self.processing_queue}:{batch_id}"

            await redis_manager.set_cache(
                processing_key,
                {
                    "batch_id": str(batch_id),
                    "started_at": datetime.utcnow().isoformat(),
                    "status": "processing"
                },
                ttl=3600  # 1 hour timeout
            )

            return True

        except Exception as e:
            self.logger.error(f"Error marking batch {batch_id} as processing: {str(e)}")
            return False

    async def mark_completed(self, batch_id: UUID, success: bool = True) -> bool:
        """Mark batch as completed hoặc failed"""
        try:
            processing_key = f"{self.processing_queue}:{batch_id}"
            await redis_manager.delete_cache(processing_key)

            if not success:
                # Move to dead letter queue if failed
                await self._move_to_dead_letter_queue(batch_id)

            return True

        except Exception as e:
            self.logger.error(f"Error marking batch {batch_id} as completed: {str(e)}")
            return False

    async def _move_to_dead_letter_queue(self, batch_id: UUID):
        """Move failed batch to dead letter queue"""
        try:
            # Get batch data from processing queue
            processing_key = f"{self.processing_queue}:{batch_id}"
            processing_data = await redis_manager.get_cache(processing_key)

            if processing_data:
                # Add to dead letter queue
                dead_letter_item = {
                    **processing_data,
                    "failed_at": datetime.utcnow().isoformat(),
                    "reason": "max_retries_exceeded"
                }

                await redis_manager.redis.lpush(
                    self.dead_letter_queue,
                    json.dumps(dead_letter_item)
                )

                self.logger.warning(f"Batch {batch_id} moved to dead letter queue")

        except Exception as e:
            self.logger.error(f"Error moving batch {batch_id} to dead letter queue: {str(e)}")

    async def retry_batch(self, batch_id: UUID) -> bool:
        """Retry failed batch"""
        try:
            # Get from dead letter queue
            dead_letter_data = await redis_manager.redis.rpop(self.dead_letter_queue)

            if not dead_letter_data:
                return False

            dead_letter_item = json.loads(dead_letter_data)

            # Check if still within retry limit
            if dead_letter_item.get("retry_count", 0) >= 3:
                self.logger.error(f"Batch {batch_id} exceeded max retries")
                return False

            # Recreate batch request
            batch_request = BatchRequest(**dead_letter_item["batch_request"])

            # Increment retry count
            dead_letter_item["retry_count"] = dead_letter_item.get("retry_count", 0) + 1

            # Re-enqueue with higher priority
            batch_request.priority = Priority.HIGH  # Escalate priority for retries

            await self.enqueue_batch(batch_request)

            self.logger.info(f"Batch {batch_id} re-queued for retry (attempt {dead_letter_item['retry_count']})")
            return True

        except Exception as e:
            self.logger.error(f"Error retrying batch {batch_id}: {str(e)}")
            return False

    async def get_queue_lengths(self) -> Dict[str, int]:
        """Get lengths of all queues"""
        try:
            lengths = {}

            for priority, queue_name in self.priority_queues.items():
                length = await redis_manager.redis.zcard(queue_name)
                lengths[priority.value] = length

            # Dead letter queue
            dead_letter_length = await redis_manager.redis.llen(self.dead_letter_queue)
            lengths["dead_letter"] = dead_letter_length

            return lengths

        except Exception as e:
            self.logger.error(f"Error getting queue lengths: {str(e)}")
            return {}

    async def get_queue_items(self, priority: Priority, limit: int = 10) -> List[Dict[str, Any]]:
        """Get items from specific priority queue"""
        try:
            queue_name = self.priority_queues[priority]

            # Get items from sorted set
            items = await redis_manager.redis.zrevrange(
                queue_name,
                0,
                limit - 1,
                withscores=True
            )

            result = []
            for item_data, score in items:
                item_dict = json.loads(item_data)
                item_dict["priority_score"] = score
                result.append(item_dict)

            return result

        except Exception as e:
            self.logger.error(f"Error getting queue items for {priority.value}: {str(e)}")
            return []

    async def cleanup_dead_letter_queue(self, days: int = 7) -> int:
        """Clean up old items from dead letter queue"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            cleaned_count = 0

            # Process dead letter queue
            while True:
                item_data = await redis_manager.redis.rpop(self.dead_letter_queue)
                if not item_data:
                    break

                item = json.loads(item_data)
                failed_at = datetime.fromisoformat(item.get("failed_at", ""))

                if failed_at < cutoff_time:
                    cleaned_count += 1
                else:
                    # Put back items that are not old enough
                    await redis_manager.redis.lpush(self.dead_letter_queue, item_data)
                    break

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} items from dead letter queue")

            return cleaned_count

        except Exception as e:
            self.logger.error(f"Error cleaning up dead letter queue: {str(e)}")
            return 0

    async def get_batch_splits(self, parent_batch_id: UUID) -> Optional[Dict[str, Any]]:
        """Get split information cho parent batch"""
        try:
            split_info = await redis_manager.get_cache(f"{self.batch_splits}:{parent_batch_id}")
            return split_info
        except Exception as e:
            self.logger.error(f"Error getting batch splits for {parent_batch_id}: {str(e)}")
            return None

    async def is_batch_being_processed(self, batch_id: UUID) -> bool:
        """Check if batch is currently being processed"""
        try:
            processing_key = f"{self.processing_queue}:{batch_id}"
            return await redis_manager.get_cache(processing_key) is not None
        except Exception as e:
            self.logger.error(f"Error checking if batch {batch_id} is being processed: {str(e)}")
            return False

    async def get_processing_batches(self) -> List[Dict[str, Any]]:
        """Get list of batches currently being processed"""
        try:
            # Get all processing keys
            processing_keys = await redis_manager.redis.keys(f"{self.processing_queue}:*")

            processing_batches = []
            for key in processing_keys:
                batch_data = await redis_manager.get_cache(key)
                if batch_data:
                    processing_batches.append(batch_data)

            return processing_batches

        except Exception as e:
            self.logger.error(f"Error getting processing batches: {str(e)}")
            return []


# Global queue manager instance
batch_queue_manager = BatchQueueManager()


# Utility functions
async def get_batch_queue_manager() -> BatchQueueManager:
    """Get batch queue manager instance"""
    return batch_queue_manager


async def enqueue_batch_request(batch_request: BatchRequest) -> bool:
    """Enqueue batch request"""
    return await batch_queue_manager.enqueue_batch(batch_request)


async def dequeue_next_batch() -> Optional[BatchQueueItem]:
    """Dequeue next batch for processing"""
    return await batch_queue_manager.dequeue_highest_priority()


async def get_queue_status() -> Dict[str, Any]:
    """Get overall queue status"""
    lengths = await batch_queue_manager.get_queue_lengths()
    processing_batches = await batch_queue_manager.get_processing_batches()

    return {
        "queue_lengths": lengths,
        "processing_count": len(processing_batches),
        "processing_batches": processing_batches,
        "total_queued": sum(lengths.values()),
        "timestamp": datetime.utcnow().isoformat()
    }