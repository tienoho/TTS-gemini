"""
Celery tasks for TTS processing and maintenance
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import tempfile
import shutil

from celery import current_task
from celery.exceptions import MaxRetriesExceededError

from ..config import get_settings
from ..models import AudioRequest, RequestLog, SystemMetric, RateLimit
from ..utils.redis_manager import redis_manager
from ..utils.logging_service import logging_service
from ..utils.auth import auth_service
from .celery_app import celery_app, gemini_circuit_breaker, DatabaseTask, calculate_backoff_delay


settings = get_settings()


@celery_app.task(
    bind=True,
    name='workers.tasks.process_tts_request',
    queue='tts_priority',
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3},
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def process_tts_request(self, request_id: int, user_id: int):
    """Process a single TTS request."""
    start_time = datetime.utcnow()

    with DatabaseTask(self) as db:
        try:
            # Get request from database
            request = db.session.query(AudioRequest).filter(
                AudioRequest.id == request_id,
                AudioRequest.user_id == user_id
            ).first()

            if not request:
                raise ValueError(f"Request {request_id} not found")

            # Check if request can be processed
            if not request.can_be_processed():
                logging_service.log_error(
                    request_id=request_id,
                    user_id=user_id,
                    message=f"Request cannot be processed: status={request.status}, retry_count={request.retry_count}"
                )
                return {"status": "skipped", "reason": "cannot_process"}

            # Mark as processing
            request.mark_as_processing()
            db.session.commit()

            # Log processing start
            logging_service.log_performance(
                request_id=request_id,
                user_id=user_id,
                operation="tts_processing_start",
                duration=0
            )

            # Process TTS request
            result = _process_tts_with_gemini(request, db)

            # Mark as completed
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            request.mark_as_completed(processing_time=processing_time)
            db.session.commit()

            # Log success
            logging_service.log_performance(
                request_id=request_id,
                user_id=user_id,
                operation="tts_processing_complete",
                duration=processing_time
            )

            # Update metrics
            logging_service.log_tts_processing(
                request_id=request_id,
                user_id=user_id,
                text_length=len(request.text_content),
                processing_time=processing_time,
                voice_name=request.voice_settings.get('voice_name', 'unknown')
            )

            return {
                "status": "completed",
                "request_id": request_id,
                "processing_time": processing_time,
                "output_path": result.get("output_path")
            }

        except Exception as e:
            # Handle failure
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Check if we should retry
            if request.increment_retry_count():
                # Retry with exponential backoff
                delay = calculate_backoff_delay(request.retry_count)
                logging_service.log_error(
                    request_id=request_id,
                    user_id=user_id,
                    message=f"TTS processing failed, retrying in {delay}s: {str(e)}",
                    error_code="PROCESSING_ERROR"
                )

                # Update status to failed but will retry
                request.mark_as_failed(str(e))
                db.session.commit()

                raise self.retry(countdown=delay)
            else:
                # Max retries exceeded
                request.mark_as_failed(str(e))
                db.session.commit()

                logging_service.log_error(
                    request_id=request_id,
                    user_id=user_id,
                    message=f"TTS processing failed permanently: {str(e)}",
                    error_code="MAX_RETRIES_EXCEEDED"
                )

                return {
                    "status": "failed",
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time": processing_time
                }


def _process_tts_with_gemini(request: AudioRequest, db) -> Dict[str, Any]:
    """Process TTS request using Gemini API."""
    # Check circuit breaker
    if gemini_circuit_breaker.is_open():
        raise Exception("Gemini API circuit breaker is open")

    try:
        # Import here to avoid circular imports
        from ..utils.audio_processor import AudioProcessor

        # Initialize audio processor
        audio_processor = AudioProcessor()

        # Process the request
        result = audio_processor.process_tts_request(
            text=request.text_content,
            voice_settings=request.voice_settings,
            output_format=request.output_format,
            request_id=request.id
        )

        # Record success
        gemini_circuit_breaker.record_success()

        return result

    except Exception as e:
        # Record failure
        gemini_circuit_breaker.record_failure()
        raise e


@celery_app.task(
    bind=True,
    name='workers.tasks.process_tts_batch',
    queue='tts_batch'
)
def process_tts_batch(self, request_ids: List[int], user_id: int):
    """Process multiple TTS requests in batch."""
    start_time = datetime.utcnow()
    results = []

    with DatabaseTask(self) as db:
        try:
            # Get all requests
            requests = db.session.query(AudioRequest).filter(
                AudioRequest.id.in_(request_ids),
                AudioRequest.user_id == user_id
            ).all()

            if not requests:
                return {"status": "no_requests", "count": 0}

            # Process each request
            for request in requests:
                try:
                    # Check if request can be processed
                    if not request.can_be_processed():
                        results.append({
                            "request_id": request.id,
                            "status": "skipped",
                            "reason": "cannot_process"
                        })
                        continue

                    # Mark as processing
                    request.mark_as_processing()

                    # Process TTS request
                    result = _process_tts_with_gemini(request, db)

                    # Mark as completed
                    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    request.mark_as_completed(processing_time=processing_time)

                    results.append({
                        "request_id": request.id,
                        "status": "completed",
                        "processing_time": processing_time,
                        "output_path": result.get("output_path")
                    })

                except Exception as e:
                    # Handle individual request failure
                    request.mark_as_failed(str(e))
                    results.append({
                        "request_id": request.id,
                        "status": "failed",
                        "error": str(e)
                    })

            db.session.commit()

            # Log batch completion
            total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logging_service.log_performance(
                request_id=0,  # Batch operation
                user_id=user_id,
                operation="batch_processing_complete",
                duration=total_time
            )

            return {
                "status": "completed",
                "total_requests": len(requests),
                "successful": len([r for r in results if r["status"] == "completed"]),
                "failed": len([r for r in results if r["status"] == "failed"]),
                "skipped": len([r for r in results if r["status"] == "skipped"]),
                "total_time": total_time,
                "results": results
            }

        except Exception as e:
            db.session.rollback()
            logging_service.log_error(
                request_id=0,
                user_id=user_id,
                message=f"Batch processing failed: {str(e)}",
                error_code="BATCH_PROCESSING_ERROR"
            )
            raise


@celery_app.task(
    name='workers.tasks.cleanup_old_files',
    queue='maintenance'
)
def cleanup_old_files():
    """Clean up old temporary files and completed request files."""
    try:
        cleanup_count = 0
        space_freed = 0

        # Clean up temporary files older than 1 hour
        temp_dir = settings.UPLOAD_FOLDER
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                filepath = os.path.join(temp_dir, filename)
                if os.path.isfile(filepath):
                    # Check file age
                    file_age = datetime.utcnow() - datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_age > timedelta(hours=1):
                        file_size = os.path.getsize(filepath)
                        os.remove(filepath)
                        cleanup_count += 1
                        space_freed += file_size

        # Clean up old output files (older than 7 days)
        output_dir = settings.OUTPUT_FOLDER
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                filepath = os.path.join(output_dir, filename)
                if os.path.isfile(filepath):
                    # Check file age
                    file_age = datetime.utcnow() - datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_age > timedelta(days=7):
                        file_size = os.path.getsize(filepath)
                        os.remove(filepath)
                        cleanup_count += 1
                        space_freed += file_size

        # Log cleanup results
        logging_service.log_system_metric(
            name="file_cleanup_count",
            value=cleanup_count,
            metric_type="counter",
            category="maintenance"
        )

        logging_service.log_system_metric(
            name="space_freed_bytes",
            value=space_freed,
            metric_type="counter",
            category="maintenance"
        )

        return {
            "status": "completed",
            "files_cleaned": cleanup_count,
            "space_freed_bytes": space_freed
        }

    except Exception as e:
        logging_service.log_error(
            request_id=0,
            user_id=0,
            message=f"File cleanup failed: {str(e)}",
            error_code="CLEANUP_ERROR"
        )
        raise


@celery_app.task(
    name='workers.tasks.update_metrics',
    queue='monitoring'
)
def update_metrics():
    """Update system metrics."""
    try:
        # Get queue lengths
        queue_lengths = asyncio.run(redis_manager.get_all_queue_lengths())

        for priority, length in queue_lengths.items():
            logging_service.log_system_metric(
                name="queue_length",
                value=length,
                metric_type="gauge",
                category="system",
                labels={"priority": priority}
            )

        # Get database connection count
        from ..app.extensions import db
        try:
            connection_count = len(db.engine.pool.checkedout())
            logging_service.log_system_metric(
                name="database_connections",
                value=connection_count,
                metric_type="gauge",
                category="system"
            )
        except:
            pass

        # Get system resource usage (simplified)
        try:
            import psutil
            process = psutil.Process()

            logging_service.log_system_metric(
                name="memory_usage_mb",
                value=process.memory_info().rss / 1024 / 1024,
                metric_type="gauge",
                category="system"
            )

            logging_service.log_system_metric(
                name="cpu_usage_percent",
                value=process.cpu_percent(),
                metric_type="gauge",
                category="system"
            )
        except ImportError:
            # psutil not available
            pass

        return {"status": "completed", "metrics_updated": len(queue_lengths)}

    except Exception as e:
        logging_service.log_error(
            request_id=0,
            user_id=0,
            message=f"Metrics update failed: {str(e)}",
            error_code="METRICS_ERROR"
        )
        raise


@celery_app.task(
    name='workers.tasks.retry_failed_requests',
    queue='maintenance'
)
def retry_failed_requests():
    """Retry failed requests that can be retried."""
    with DatabaseTask(None) as db:
        try:
            # Get failed requests that can be retried
            failed_requests = db.session.query(AudioRequest).filter(
                AudioRequest.status == "failed",
                AudioRequest.retry_count < AudioRequest.max_retries
            ).limit(10).all()

            retried_count = 0

            for request in failed_requests:
                try:
                    # Reset status to pending for retry
                    request.status = "pending"
                    request.updated_at = datetime.utcnow()
                    retried_count += 1

                    # Log retry attempt
                    logging_service.log_request(
                        request_id=request.id,
                        user_id=request.user_id,
                        message=f"Retrying failed request (attempt {request.retry_count + 1})",
                        level="info"
                    )

                except Exception as e:
                    logging_service.log_error(
                        request_id=request.id,
                        user_id=request.user_id,
                        message=f"Failed to prepare request for retry: {str(e)}",
                        error_code="RETRY_PREP_ERROR"
                    )

            db.session.commit()

            return {
                "status": "completed",
                "requests_retried": retried_count,
                "total_failed": len(failed_requests)
            }

        except Exception as e:
            db.session.rollback()
            logging_service.log_error(
                request_id=0,
                user_id=0,
                message=f"Retry failed requests task failed: {str(e)}",
                error_code="RETRY_TASK_ERROR"
            )
            raise


@celery_app.task(
    name='workers.tasks.cleanup_expired_rate_limits',
    queue='maintenance'
)
def cleanup_expired_rate_limits():
    """Clean up expired rate limits."""
    try:
        deleted_count = asyncio.run(redis_manager.cleanup_expired_keys())

        logging_service.log_system_metric(
            name="expired_rate_limits_cleaned",
            value=deleted_count,
            metric_type="counter",
            category="maintenance"
        )

        return {
            "status": "completed",
            "keys_cleaned": deleted_count
        }

    except Exception as e:
        logging_service.log_error(
            request_id=0,
            user_id=0,
            message=f"Rate limit cleanup failed: {str(e)}",
            error_code="RATE_LIMIT_CLEANUP_ERROR"
        )
        raise


@celery_app.task(
    name='workers.tasks.health_check',
    queue='monitoring'
)
def health_check():
    """Perform system health check."""
    try:
        health_status = {
            "celery": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": current_task.request.id if current_task else None
        }

        # Check Redis
        try:
            redis_health = asyncio.run(redis_manager.health_check())
            health_status["redis"] = redis_health.get("status", "unknown")
        except:
            health_status["redis"] = "unhealthy"

        # Check database
        try:
            from ..app.extensions import db
            db.session.execute("SELECT 1")
            health_status["database"] = "healthy"
        except:
            health_status["database"] = "unhealthy"

        # Check Gemini API circuit breaker
        health_status["gemini_circuit_breaker"] = "closed" if not gemini_circuit_breaker.is_open() else "open"

        return health_status

    except Exception as e:
        return {
            "celery": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }