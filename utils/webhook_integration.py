"""
Webhook Integration với TTS System
"""
import asyncio
from typing import List, Optional
from datetime import datetime
import logging

from models.webhook import Webhook, WebhookEventType
from utils.webhook_service import webhook_service, send_webhook_notification, queue_webhook_for_batch
from utils.webhook_events import (
    create_tts_completed_event,
    create_tts_error_event,
    create_batch_completed_event,
    create_batch_error_event,
    create_quality_enhancement_completed_event,
    create_quality_enhancement_error_event,
    create_voice_cloning_completed_event,
    create_voice_cloning_error_event,
    create_audio_enhancement_completed_event,
    create_audio_enhancement_error_event
)

logger = logging.getLogger(__name__)

class TTSWebhookIntegration:
    """Integration class cho TTS webhook notifications"""

    def __init__(self):
        self.enabled_events = set()

    async def initialize(self):
        """Khởi tạo webhook integration"""
        # Start webhook service
        await webhook_service.start()

        # Enable all TTS-related events by default
        self.enabled_events = {
            WebhookEventType.TTS_COMPLETED,
            WebhookEventType.TTS_ERROR,
            WebhookEventType.BATCH_COMPLETED,
            WebhookEventType.BATCH_ERROR,
            WebhookEventType.QUALITY_ENHANCEMENT_COMPLETED,
            WebhookEventType.QUALITY_ENHANCEMENT_ERROR,
            WebhookEventType.VOICE_CLONING_COMPLETED,
            WebhookEventType.VOICE_CLONING_ERROR,
            WebhookEventType.AUDIO_ENHANCEMENT_COMPLETED,
            WebhookEventType.AUDIO_ENHANCEMENT_ERROR
        }

        logger.info("TTS Webhook Integration initialized")

    async def shutdown(self):
        """Tắt webhook integration"""
        await webhook_service.stop()
        logger.info("TTS Webhook Integration shutdown")

    async def notify_tts_completed(self, request_id: str, audio_url: str, duration: float,
                                 text_length: int, organization_id: int, use_batch: bool = True, **kwargs):
        """Gửi notification khi TTS hoàn thành"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.TTS_COMPLETED
            )

            if not webhooks:
                return

            # Create event
            event = create_tts_completed_event(
                request_id=request_id,
                audio_url=audio_url,
                duration=duration,
                text_length=text_length,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications
            if use_batch:
                for webhook in webhooks:
                    queue_webhook_for_batch(webhook, event)
            else:
                await asyncio.gather(*[
                    send_webhook_notification(webhook, event) for webhook in webhooks
                ])

            logger.info(f"Sent TTS completed notifications for request {request_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send TTS completed notification: {str(e)}")

    async def notify_tts_error(self, request_id: str, error_code: str, error_message: str,
                             text_length: int = None, organization_id: int = None, **kwargs):
        """Gửi notification khi TTS gặp lỗi"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.TTS_ERROR
            )

            if not webhooks:
                return

            # Create event
            event = create_tts_error_event(
                request_id=request_id,
                error_code=error_code,
                error_message=error_message,
                text_length=text_length,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications immediately (no batching for errors)
            await asyncio.gather(*[
                send_webhook_notification(webhook, event) for webhook in webhooks
            ])

            logger.warning(f"Sent TTS error notifications for request {request_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send TTS error notification: {str(e)}")

    async def notify_batch_completed(self, batch_id: str, total_requests: int,
                                   successful_requests: int, failed_requests: int,
                                   organization_id: int, use_batch: bool = True, **kwargs):
        """Gửi notification khi batch processing hoàn thành"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.BATCH_COMPLETED
            )

            if not webhooks:
                return

            # Create event
            event = create_batch_completed_event(
                batch_id=batch_id,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications
            if use_batch:
                for webhook in webhooks:
                    queue_webhook_for_batch(webhook, event)
            else:
                await asyncio.gather(*[
                    send_webhook_notification(webhook, event) for webhook in webhooks
                ])

            logger.info(f"Sent batch completed notifications for batch {batch_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send batch completed notification: {str(e)}")

    async def notify_batch_error(self, batch_id: str, error_code: str, error_message: str,
                               failed_requests: int, organization_id: int, **kwargs):
        """Gửi notification khi batch processing gặp lỗi"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.BATCH_ERROR
            )

            if not webhooks:
                return

            # Create event
            event = create_batch_error_event(
                batch_id=batch_id,
                error_code=error_code,
                error_message=error_message,
                failed_requests=failed_requests,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications immediately (no batching for errors)
            await asyncio.gather(*[
                send_webhook_notification(webhook, event) for webhook in webhooks
            ])

            logger.warning(f"Sent batch error notifications for batch {batch_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send batch error notification: {str(e)}")

    async def notify_quality_enhancement_completed(self, request_id: str, enhancement_type: str,
                                                 original_quality: float, enhanced_quality: float,
                                                 organization_id: int, **kwargs):
        """Gửi notification khi quality enhancement hoàn thành"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.QUALITY_ENHANCEMENT_COMPLETED
            )

            if not webhooks:
                return

            # Create event
            event = create_quality_enhancement_completed_event(
                request_id=request_id,
                enhancement_type=enhancement_type,
                original_quality=original_quality,
                enhanced_quality=enhanced_quality,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications
            for webhook in webhooks:
                queue_webhook_for_batch(webhook, event)

            logger.info(f"Sent quality enhancement completed notifications for request {request_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send quality enhancement completed notification: {str(e)}")

    async def notify_quality_enhancement_error(self, request_id: str, enhancement_type: str,
                                             error_code: str, error_message: str,
                                             organization_id: int, **kwargs):
        """Gửi notification khi quality enhancement gặp lỗi"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.QUALITY_ENHANCEMENT_ERROR
            )

            if not webhooks:
                return

            # Create event
            event = create_quality_enhancement_error_event(
                request_id=request_id,
                enhancement_type=enhancement_type,
                error_code=error_code,
                error_message=error_message,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications immediately (no batching for errors)
            await asyncio.gather(*[
                send_webhook_notification(webhook, event) for webhook in webhooks
            ])

            logger.warning(f"Sent quality enhancement error notifications for request {request_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send quality enhancement error notification: {str(e)}")

    async def notify_voice_cloning_completed(self, cloning_id: str, voice_name: str,
                                           quality_score: float, training_duration: float,
                                           organization_id: int, **kwargs):
        """Gửi notification khi voice cloning hoàn thành"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.VOICE_CLONING_COMPLETED
            )

            if not webhooks:
                return

            # Create event
            event = create_voice_cloning_completed_event(
                cloning_id=cloning_id,
                voice_name=voice_name,
                quality_score=quality_score,
                training_duration=training_duration,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications
            for webhook in webhooks:
                queue_webhook_for_batch(webhook, event)

            logger.info(f"Sent voice cloning completed notifications for {cloning_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send voice cloning completed notification: {str(e)}")

    async def notify_voice_cloning_error(self, cloning_id: str, error_code: str,
                                       error_message: str, organization_id: int, **kwargs):
        """Gửi notification khi voice cloning gặp lỗi"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.VOICE_CLONING_ERROR
            )

            if not webhooks:
                return

            # Create event
            event = create_voice_cloning_error_event(
                cloning_id=cloning_id,
                error_code=error_code,
                error_message=error_message,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications immediately (no batching for errors)
            await asyncio.gather(*[
                send_webhook_notification(webhook, event) for webhook in webhooks
            ])

            logger.warning(f"Sent voice cloning error notifications for {cloning_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send voice cloning error notification: {str(e)}")

    async def notify_audio_enhancement_completed(self, request_id: str, enhancement_type: str,
                                               improvement_score: float, organization_id: int, **kwargs):
        """Gửi notification khi audio enhancement hoàn thành"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.AUDIO_ENHANCEMENT_COMPLETED
            )

            if not webhooks:
                return

            # Create event
            event = create_audio_enhancement_completed_event(
                request_id=request_id,
                enhancement_type=enhancement_type,
                improvement_score=improvement_score,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications
            for webhook in webhooks:
                queue_webhook_for_batch(webhook, event)

            logger.info(f"Sent audio enhancement completed notifications for request {request_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send audio enhancement completed notification: {str(e)}")

    async def notify_audio_enhancement_error(self, request_id: str, enhancement_type: str,
                                           error_code: str, error_message: str,
                                           organization_id: int, **kwargs):
        """Gửi notification khi audio enhancement gặp lỗi"""
        try:
            # Get webhooks for organization
            webhooks = await self._get_webhooks_for_event(
                organization_id, WebhookEventType.AUDIO_ENHANCEMENT_ERROR
            )

            if not webhooks:
                return

            # Create event
            event = create_audio_enhancement_error_event(
                request_id=request_id,
                enhancement_type=enhancement_type,
                error_code=error_code,
                error_message=error_message,
                organization_id=organization_id,
                **kwargs
            )

            # Send notifications immediately (no batching for errors)
            await asyncio.gather(*[
                send_webhook_notification(webhook, event) for webhook in webhooks
            ])

            logger.warning(f"Sent audio enhancement error notifications for request {request_id} to {len(webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Failed to send audio enhancement error notification: {str(e)}")

    async def _get_webhooks_for_event(self, organization_id: int, event_type: str) -> List[Webhook]:
        """Lấy danh sách webhooks cho event type và organization"""
        try:
            # Mock implementation - in real app, query database
            # This would be something like:
            # webhooks = await Webhook.query.filter_by(
            #     organization_id=organization_id,
            #     status=WebhookStatus.ACTIVE
            # ).all()
            #
            # Then filter by event type:
            # webhooks = [w for w in webhooks if w.is_event_enabled(event_type)]

            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Error getting webhooks for event {event_type}: {str(e)}")
            return []

# Global TTS webhook integration instance
tts_webhook_integration = TTSWebhookIntegration()

# Convenience functions for easy integration
async def notify_tts_success(request_id: str, audio_url: str, duration: float,
                           text_length: int, organization_id: int, **kwargs):
    """Convenience function để notify TTS success"""
    await tts_webhook_integration.notify_tts_completed(
        request_id, audio_url, duration, text_length, organization_id, **kwargs
    )

async def notify_tts_failure(request_id: str, error_code: str, error_message: str,
                           organization_id: int, **kwargs):
    """Convenience function để notify TTS failure"""
    await tts_webhook_integration.notify_tts_error(
        request_id, error_code, error_message, organization_id=organization_id, **kwargs
    )

async def notify_batch_success(batch_id: str, total_requests: int,
                             successful_requests: int, failed_requests: int,
                             organization_id: int, **kwargs):
    """Convenience function để notify batch success"""
    await tts_webhook_integration.notify_batch_completed(
        batch_id, total_requests, successful_requests, failed_requests, organization_id, **kwargs
    )

async def notify_batch_failure(batch_id: str, error_code: str, error_message: str,
                             failed_requests: int, organization_id: int, **kwargs):
    """Convenience function để notify batch failure"""
    await tts_webhook_integration.notify_batch_error(
        batch_id, error_code, error_message, failed_requests, organization_id, **kwargs
    )