import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import UUID

from models.batch_request import (
    BatchRequest, BatchStatus, BatchItemResult, BatchProcessingError,
    ItemStatus, Priority, TTSItem
)
from utils.gemini_tts import gemini_tts_service as GeminiTTS
from utils.progress_streamer import progress_streamer as ProgressStreamer
from utils.redis_manager import RedisManager


@dataclass
class ProcessingConfig:
    """Configuration cho batch processing"""
    max_concurrency: int = 5
    chunk_size: int = 10
    timeout_per_item: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: float = 1.0
    progress_update_interval: float = 1.0


class BatchProcessor:
    """Service để xử lý batch TTS requests"""

    def __init__(
        self,
        tts_service: Any,
        progress_streamer: Any,
        redis_manager: Any,
        config: Optional[ProcessingConfig] = None
    ):
        self.tts_service = tts_service
        self.progress_streamer = progress_streamer
        self.redis_manager = redis_manager
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)

        # Thread pool cho parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrency)

        # Active batch tracking
        self.active_batches: Dict[UUID, Dict[str, Any]] = {}

    async def process_batch(self, batch_request: BatchRequest) -> Dict[str, Any]:
        """Process một batch request"""
        batch_id = batch_request.id

        try:
            # Initialize batch tracking
            await self._initialize_batch(batch_request)

            # Update status to processing
            await self._update_batch_status(batch_id, BatchStatus.PROCESSING)

            # Process items
            results = await self._process_batch_items(batch_request)

            # Finalize batch
            await self._finalize_batch(batch_id, results)

            return {
                "batch_id": batch_id,
                "status": BatchStatus.COMPLETED,
                "results": results,
                "summary": self._generate_summary(results)
            }

        except Exception as e:
            self.logger.error(f"Batch processing failed for {batch_id}: {str(e)}")
            await self._update_batch_status(batch_id, BatchStatus.FAILED)
            raise

    async def _initialize_batch(self, batch_request: BatchRequest):
        """Initialize batch tracking trong Redis"""
        batch_id = batch_request.id

        batch_data = {
            "id": str(batch_id),
            "status": BatchStatus.PENDING.value,
            "name": batch_request.name,
            "priority": batch_request.priority.value,
            "total_items": len(batch_request.items),
            "completed_items": 0,
            "failed_items": 0,
            "pending_items": len(batch_request.items),
            "created_at": batch_request.created_at.isoformat(),
            "started_at": datetime.utcnow().isoformat(),
            "results": [],
            "errors": [],
            "metadata": batch_request.metadata or {}
        }

        await self.redis_manager.set_cache(f"batch:{batch_id}", batch_data)
        self.active_batches[batch_id] = batch_data

    async def _process_batch_items(self, batch_request: BatchRequest) -> List[BatchItemResult]:
        """Process tất cả items trong batch"""
        batch_id = batch_request.id
        items = batch_request.items

        # Split items thành chunks để parallel processing
        chunks = self._create_chunks(items, self.config.chunk_size)

        all_results = []
        completed_count = 0

        for chunk in chunks:
            # Process chunk parallel
            chunk_results = await self._process_chunk(batch_id, chunk)

            # Update progress
            completed_count += len(chunk_results)
            await self._update_batch_progress(batch_id, completed_count, len(items))

            all_results.extend(chunk_results)

            # Small delay between chunks để tránh overload
            await asyncio.sleep(0.1)

        return all_results

    async def _process_chunk(
        self,
        batch_id: UUID,
        items: List[TTSItem]
    ) -> List[BatchItemResult]:
        """Process một chunk của items parallel"""
        loop = asyncio.get_event_loop()

        # Submit tasks to thread pool
        tasks = []
        for item in items:
            task = loop.run_in_executor(
                self.executor,
                self._process_single_item,
                batch_id,
                item
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task failed: {str(e)}")
                # Create failed result
                failed_item = items[len(results)]
                failed_result = BatchItemResult(
                    item_id=failed_item.id,
                    status=ItemStatus.FAILED,
                    error=BatchProcessingError(
                        item_id=failed_item.id,
                        error_code="PROCESSING_ERROR",
                        error_message=str(e)
                    )
                )
                results.append(failed_result)

        return results

    def _process_single_item(self, batch_id: UUID, item: TTSItem) -> BatchItemResult:
        """Process một single TTS item với retry logic"""
        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                # Generate TTS audio
                audio_data = self.tts_service.generate_audio(
                    text=item.text,
                    voice=item.voice,
                    language=item.language,
                    speed=item.speed,
                    pitch=item.pitch,
                    volume=item.volume,
                    output_format=item.output_format
                )

                processing_time = time.time() - start_time

                # Create success result
                result = BatchItemResult(
                    item_id=item.id,
                    status=ItemStatus.COMPLETED,
                    audio_url=audio_data.get("audio_url"),
                    duration=audio_data.get("duration"),
                    file_size=audio_data.get("file_size"),
                    processing_time=processing_time,
                    metadata=item.metadata
                )

                return result

            except Exception as e:
                if attempt == self.config.max_retries:
                    # Final attempt failed - always returns
                    processing_time = time.time() - start_time
                    return BatchItemResult(
                        item_id=item.id,
                        status=ItemStatus.FAILED,
                        error=BatchProcessingError(
                            item_id=item.id,
                            error_code="MAX_RETRIES_EXCEEDED",
                            error_message=f"Failed after {self.config.max_retries} attempts: {str(e)}",
                            retry_count=attempt
                        ),
                        processing_time=processing_time
                    )

                # Wait before retry
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

        # This line should never be reached, but ensures function always returns
        return BatchItemResult(
            item_id=item.id,
            status=ItemStatus.FAILED,
            error=BatchProcessingError(
                item_id=item.id,
                error_code="UNKNOWN_ERROR",
                error_message="Unknown processing error"
            )
        )

    async def _update_batch_progress(self, batch_id: UUID, completed: int, total: int):
        """Update batch progress"""
        progress_data = {
            "completed": completed,
            "total": total,
            "percentage": (completed / total) * 100 if total > 0 else 0
        }

        # Update Redis
        batch_data = await self.redis_manager.get_cache(f"batch:{batch_id}")
        if batch_data:
            batch_data["completed_items"] = completed
            batch_data["pending_items"] = total - completed
            await self.redis_manager.set_cache(f"batch:{batch_id}", batch_data)

        # Send progress update via WebSocket
        await self.progress_streamer.send_batch_progress(
            batch_id,
            BatchStatus.PROCESSING,
            progress_data
        )

    async def _update_batch_status(self, batch_id: UUID, status: BatchStatus):
        """Update batch status"""
        batch_data = await self.redis_manager.get_cache(f"batch:{batch_id}")
        if batch_data:
            batch_data["status"] = status.value
            if status == BatchStatus.COMPLETED:
                batch_data["completed_at"] = datetime.utcnow().isoformat()
            await self.redis_manager.set_cache(f"batch:{batch_id}", batch_data)

        # Send status update
        await self.progress_streamer.send_batch_progress(
            batch_id,
            status,
            {"status": status.value}
        )

    async def _finalize_batch(self, batch_id: UUID, results: List[BatchItemResult]):
        """Finalize batch processing"""
        # Update Redis with final results
        batch_data = await self.redis_manager.get_cache(f"batch:{batch_id}")
        if batch_data:
            batch_data["results"] = [result.dict() for result in results]

            # Calculate summary
            summary = self._generate_summary(results)
            batch_data["summary"] = summary

            await self.redis_manager.set_cache(f"batch:{batch_id}", batch_data)

        # Update final status
        failed_count = summary.get("failed", 0)
        if failed_count > 0:
            final_status = BatchStatus.PARTIALLY_COMPLETED
        else:
            final_status = BatchStatus.COMPLETED

        await self._update_batch_status(batch_id, final_status)

        # Clean up active batch tracking
        self.active_batches.pop(batch_id, None)

    def _generate_summary(self, results: List[BatchItemResult]) -> Dict[str, Any]:
        """Generate processing summary"""
        total_items = len(results)
        completed = sum(1 for r in results if r.status == ItemStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == ItemStatus.FAILED)
        skipped = sum(1 for r in results if r.status == ItemStatus.SKIPPED)

        # Calculate processing times
        processing_times = [r.processing_time for r in results if r.processing_time]
        avg_processing_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )

        return {
            "total_items": total_items,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (completed / total_items) * 100 if total_items > 0 else 0,
            "average_processing_time": avg_processing_time,
            "total_processing_time": sum(processing_times)
        }

    def _create_chunks(self, items: List[TTSItem], chunk_size: int) -> List[List[TTSItem]]:
        """Split items thành chunks"""
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i:i + chunk_size])
        return chunks

    async def get_batch_status(self, batch_id: UUID) -> Optional[Dict[str, Any]]:
        """Get batch status từ Redis"""
        return await self.redis_manager.get_cache(f"batch:{batch_id}")

    async def cancel_batch(self, batch_id: UUID) -> bool:
        """Cancel một batch đang processing"""
        batch_data = await self.get_batch_status(batch_id)
        if not batch_data:
            return False

        current_status = batch_data.get("status")
        if current_status in [BatchStatus.COMPLETED.value, BatchStatus.FAILED.value, BatchStatus.CANCELLED.value]:
            return False

        # Update status to cancelled
        await self._update_batch_status(batch_id, BatchStatus.CANCELLED)

        # Clean up
        self.active_batches.pop(batch_id, None)

        return True

    async def cleanup_expired_batches(self):
        """Cleanup expired batches"""
        # Implementation for cleanup logic
        pass

    def update_config(self, config: ProcessingConfig):
        """Update processing configuration"""
        self.config = config

        # Recreate executor with new concurrency
        self.executor.shutdown(wait=True)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrency)