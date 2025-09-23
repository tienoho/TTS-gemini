"""
Webhook Delivery Service
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import aiohttp
from asyncio import Queue, Semaphore
import threading

from models.webhook import Webhook, WebhookDelivery, WebhookRetryAttempt, WebhookDeadLetter, DeliveryStatus
from utils.webhook_security import webhook_security
from utils.webhook_events import event_manager, WebhookEvent
from config.webhook import webhook_config

logger = logging.getLogger(__name__)

class WebhookDeliveryError(Exception):
    """Base exception cho webhook delivery"""
    pass

class WebhookService:
    """Service quản lý webhook delivery"""

    def __init__(self):
        self.delivery_queue = Queue(maxsize=webhook_config.QUEUE_MAX_SIZE)
        self.batch_queue = defaultdict(list)  # webhook_id -> events
        self.batch_timers = {}  # webhook_id -> timer
        self.workers = []
        self.batch_workers = []
        self.is_running = False
        self.semaphore = Semaphore(webhook_config.QUEUE_WORKERS)
        self.batch_semaphore = Semaphore(2)  # Fewer workers for batch processing

        # Dead letter queue
        self.dead_letter_queue = Queue(maxsize=webhook_config.MAX_DEAD_LETTER_SIZE)

    async def start(self):
        """Khởi động webhook service"""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting webhook service...")

        # Start delivery workers
        self.workers = []
        for i in range(webhook_config.QUEUE_WORKERS):
            worker = asyncio.create_task(self._delivery_worker())
            self.workers.append(worker)

        # Start batch workers
        self.batch_workers = []
        for i in range(2):
            worker = asyncio.create_task(self._batch_worker())
            self.batch_workers.append(worker)

        # Start dead letter processor
        self.dead_letter_worker = asyncio.create_task(self._dead_letter_processor())

        logger.info(f"Webhook service started with {webhook_config.QUEUE_WORKERS} delivery workers and 2 batch workers")

    async def stop(self):
        """Dừng webhook service"""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping webhook service...")

        # Cancel all workers
        for worker in self.workers + self.batch_workers:
            worker.cancel()

        if hasattr(self, 'dead_letter_worker'):
            self.dead_letter_worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers + self.batch_workers, return_exceptions=True)
        if hasattr(self, 'dead_letter_worker'):
            await asyncio.gather(self.dead_letter_worker, return_exceptions=True)

        logger.info("Webhook service stopped")

    async def send_webhook(self, webhook: Webhook, event: WebhookEvent) -> bool:
        """Gửi webhook event"""
        try:
            # Format payload
            payload = event_manager.format_event_payload(event, webhook.id)

            # Generate signature
            payload_str = json.dumps(payload, default=str)
            signature = webhook_security.generate_signature(payload_str, webhook.secret)

            # Create delivery record
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event_type=event.event_type,
                payload=payload,
                signature=signature,
                status=DeliveryStatus.PENDING
            )

            # Add to queue
            await self.delivery_queue.put((webhook, delivery, payload_str))
            logger.info(f"Queued webhook delivery for webhook {webhook.id}, event {event.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to queue webhook delivery: {str(e)}")
            return False

    async def send_webhook_immediate(self, webhook: Webhook, event: WebhookEvent) -> Tuple[bool, Dict[str, Any]]:
        """Gửi webhook ngay lập tức (không queue)"""
        try:
            # Format payload
            payload = event_manager.format_event_payload(event, webhook.id)
            payload_str = json.dumps(payload, default=str)
            signature = webhook_security.generate_signature(payload_str, webhook.secret)

            # Send HTTP request
            success, response_data = await self._send_http_request(
                webhook.url, payload_str, signature, webhook.headers, webhook.timeout
            )

            # Create delivery record for history
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event_type=event.event_type,
                payload=payload,
                signature=signature,
                status=DeliveryStatus.SUCCESS if success else DeliveryStatus.FAILED,
                response_status=response_data.get('status'),
                response_body=response_data.get('body'),
                error_message=response_data.get('error'),
                delivery_time=response_data.get('delivery_time'),
                completed_at=datetime.utcnow()
            )

            # In real app, save to database
            logger.info(f"Immediate webhook delivery {'successful' if success else 'failed'} for webhook {webhook.id}")

            return success, response_data

        except Exception as e:
            logger.error(f"Immediate webhook delivery failed: {str(e)}")
            return False, {"error": str(e)}

    async def _delivery_worker(self):
        """Worker xử lý webhook delivery"""
        while self.is_running:
            try:
                # Get item from queue with timeout
                try:
                    webhook, delivery, payload_str = await asyncio.wait_for(
                        self.delivery_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                async with self.semaphore:
                    await self._process_delivery(webhook, delivery, payload_str)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Delivery worker error: {str(e)}")

    async def _process_delivery(self, webhook: Webhook, delivery: WebhookDelivery, payload_str: str):
        """Xử lý một delivery"""
        try:
            # Send HTTP request
            success, response_data = await self._send_http_request(
                webhook.url, payload_str, delivery.signature, webhook.headers, webhook.timeout
            )

            # Update delivery record
            delivery.response_status = response_data.get('status')
            delivery.response_body = response_data.get('body')
            delivery.delivery_time = response_data.get('delivery_time')
            delivery.completed_at = datetime.utcnow()

            if success:
                delivery.status = DeliveryStatus.SUCCESS
                logger.info(f"Webhook delivery successful for webhook {webhook.id}")
            else:
                delivery.status = DeliveryStatus.FAILED
                delivery.error_message = response_data.get('error')
                delivery.attempt_count += 1

                # Check if should retry
                if delivery.is_retryable():
                    delivery.status = DeliveryStatus.RETRYING
                    delivery.next_retry_at = delivery.calculate_next_retry_time()
                    # Re-queue for retry
                    await self.delivery_queue.put((webhook, delivery, payload_str))
                    logger.info(f"Webhook delivery failed, retrying for webhook {webhook.id} (attempt {delivery.attempt_count})")
                else:
                    # Move to dead letter queue
                    await self._move_to_dead_letter(webhook, delivery)
                    logger.error(f"Webhook delivery failed permanently for webhook {webhook.id}")

        except Exception as e:
            logger.error(f"Error processing delivery: {str(e)}")
            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)
            delivery.completed_at = datetime.utcnow()

    async def _send_http_request(self, url: str, payload: str, signature: str, headers: Dict[str, str], timeout: int) -> Tuple[bool, Dict[str, Any]]:
        """Gửi HTTP request"""
        default_headers = {
            'Content-Type': 'application/json',
            'User-Agent': webhook_config.DEFAULT_USER_AGENT,
            'X-Webhook-Signature': signature
        }

        # Merge custom headers
        request_headers = {**default_headers, **headers}

        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(url, data=payload, headers=request_headers) as response:
                    delivery_time = int((time.time() - start_time) * 1000)
                    response_body = await response.text()

                    if response.status in (200, 201, 202, 204):
                        return True, {
                            'status': response.status,
                            'body': response_body,
                            'delivery_time': delivery_time
                        }
                    else:
                        return False, {
                            'status': response.status,
                            'body': response_body,
                            'error': f"HTTP {response.status}: {response_body}",
                            'delivery_time': delivery_time
                        }

        except asyncio.TimeoutError:
            delivery_time = int((time.time() - start_time) * 1000)
            return False, {
                'error': f"Request timeout after {timeout}s",
                'delivery_time': delivery_time
            }
        except Exception as e:
            delivery_time = int((time.time() - start_time) * 1000)
            return False, {
                'error': str(e),
                'delivery_time': delivery_time
            }

    async def send_batch_webhook(self, webhook: Webhook, events: List[WebhookEvent]) -> bool:
        """Gửi batch webhook"""
        if not events:
            return True

        try:
            # Format batch payload
            batch_payload = {
                "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}",
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(events),
                "events": []
            }

            for event in events:
                event_payload = event_manager.format_event_payload(event, webhook.id)
                batch_payload["events"].append(event_payload)

            # Generate signature for batch
            payload_str = json.dumps(batch_payload, default=str)
            signature = webhook_security.generate_signature(payload_str, webhook.secret)

            # Send batch request
            success, response_data = await self._send_http_request(
                webhook.url, payload_str, signature, webhook.headers, webhook.timeout
            )

            # Create delivery records for each event
            for event in events:
                delivery = WebhookDelivery(
                    webhook_id=webhook.id,
                    event_type=event.event_type,
                    payload=event_manager.format_event_payload(event, webhook.id),
                    signature=signature,
                    status=DeliveryStatus.SUCCESS if success else DeliveryStatus.FAILED,
                    response_status=response_data.get('status'),
                    response_body=response_data.get('body'),
                    error_message=response_data.get('error'),
                    delivery_time=response_data.get('delivery_time'),
                    completed_at=datetime.utcnow()
                )

                # In real app, save to database
                logger.info(f"Batch delivery {'successful' if success else 'failed'} for event {event.id}")

            return success

        except Exception as e:
            logger.error(f"Batch webhook delivery failed: {str(e)}")
            return False

    async def _batch_worker(self):
        """Worker xử lý batch delivery"""
        while self.is_running:
            try:
                # Process all pending batches
                for webhook_id in list(self.batch_queue.keys()):
                    if not self.batch_queue[webhook_id]:
                        continue

                    async with self.batch_semaphore:
                        events = self.batch_queue[webhook_id]
                        del self.batch_queue[webhook_id]

                        # Cancel timer if exists
                        if webhook_id in self.batch_timers:
                            self.batch_timers[webhook_id].cancel()
                            del self.batch_timers[webhook_id]

                        # Get webhook (in real app, fetch from database)
                        webhook = await self._get_webhook_by_id(webhook_id)
                        if webhook:
                            await self.send_batch_webhook(webhook, events)

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch worker error: {str(e)}")

    def queue_for_batch(self, webhook: Webhook, event: WebhookEvent):
        """Thêm event vào batch queue"""
        webhook_id = webhook.id

        # Add to batch queue
        self.batch_queue[webhook_id].append(event)

        # Set timer if not exists
        if webhook_id not in self.batch_timers:
            self.batch_timers[webhook_id] = asyncio.create_task(
                self._batch_timer(webhook_id)
            )

        # Check if batch is full
        batch_config = webhook_config.get_batch_config()
        if len(self.batch_queue[webhook_id]) >= batch_config["batch_size"]:
            # Trigger immediate batch processing
            if webhook_id in self.batch_timers:
                self.batch_timers[webhook_id].cancel()
                del self.batch_timers[webhook_id]

            # Process batch immediately
            asyncio.create_task(self._process_batch_immediately(webhook_id))

    async def _batch_timer(self, webhook_id: int):
        """Timer cho batch processing"""
        try:
            batch_config = webhook_config.get_batch_config()
            await asyncio.sleep(batch_config["batch_timeout"])

            # Process batch when timer expires
            if webhook_id in self.batch_queue and self.batch_queue[webhook_id]:
                async with self.batch_semaphore:
                    events = self.batch_queue[webhook_id]
                    del self.batch_queue[webhook_id]

                    webhook = await self._get_webhook_by_id(webhook_id)
                    if webhook:
                        await self.send_batch_webhook(webhook, events)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Batch timer error: {str(e)}")

    async def _process_batch_immediately(self, webhook_id: int):
        """Xử lý batch ngay lập tức"""
        try:
            if webhook_id in self.batch_queue and self.batch_queue[webhook_id]:
                async with self.batch_semaphore:
                    events = self.batch_queue[webhook_id]
                    del self.batch_queue[webhook_id]

                    webhook = await self._get_webhook_by_id(webhook_id)
                    if webhook:
                        await self.send_batch_webhook(webhook, events)

        except Exception as e:
            logger.error(f"Immediate batch processing error: {str(e)}")

    async def _move_to_dead_letter(self, webhook: Webhook, delivery: WebhookDelivery):
        """Chuyển delivery vào dead letter queue"""
        try:
            dead_letter = WebhookDeadLetter(
                webhook_id=webhook.id,
                event_type=delivery.event_type,
                payload=delivery.payload,
                signature=delivery.signature,
                failure_reason=delivery.error_message or "Max retries exceeded",
                last_attempt_at=delivery.last_attempt_at
            )

            # Add to dead letter queue
            await self.dead_letter_queue.put(dead_letter)
            logger.warning(f"Moved delivery to dead letter queue for webhook {webhook.id}")

        except Exception as e:
            logger.error(f"Failed to move to dead letter queue: {str(e)}")

    async def _dead_letter_processor(self):
        """Xử lý dead letter queue"""
        while self.is_running:
            try:
                # Process dead letters with low frequency
                try:
                    dead_letter = await asyncio.wait_for(
                        self.dead_letter_queue.get(), timeout=10.0
                    )

                    # In real app, save to database and potentially alert administrators
                    logger.error(f"Dead letter: {dead_letter.event_type} for webhook {dead_letter.webhook_id}")

                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dead letter processor error: {str(e)}")

    async def _get_webhook_by_id(self, webhook_id: int) -> Optional[Webhook]:
        """Lấy webhook từ database (mock implementation)"""
        # In real application, this would fetch from database
        # For now, return None to indicate webhook not found
        return None

    async def get_delivery_stats(self) -> Dict[str, Any]:
        """Lấy thống kê delivery"""
        return {
            "queue_size": self.delivery_queue.qsize(),
            "batch_queue_size": sum(len(events) for events in self.batch_queue.values()),
            "dead_letter_queue_size": self.dead_letter_queue.qsize(),
            "active_workers": len([w for w in self.workers if not w.done()]),
            "active_batch_workers": len([w for w in self.batch_workers if not w.done()])
        }

# Global webhook service instance
webhook_service = WebhookService()

# Convenience functions
async def send_webhook_notification(webhook: Webhook, event: WebhookEvent) -> bool:
    """Gửi webhook notification"""
    return await webhook_service.send_webhook(webhook, event)

async def send_webhook_immediate(webhook: Webhook, event: WebhookEvent) -> Tuple[bool, Dict[str, Any]]:
    """Gửi webhook ngay lập tức"""
    return await webhook_service.send_webhook_immediate(webhook, event)

def queue_webhook_for_batch(webhook: Webhook, event: WebhookEvent):
    """Thêm webhook vào batch queue"""
    webhook_service.queue_for_batch(webhook, event)