"""
Sample Webhook Plugin for TTS System

This is a sample plugin demonstrating how to create a webhook plugin
that integrates with the TTS system using the plugin interface.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.plugin_interface import WebhookPlugin, HookType, EventType, HookContext, EventContext
from models.plugin import PluginType


class SampleWebhookPlugin(WebhookPlugin):
    """Sample webhook plugin implementation."""

    def __init__(self):
        """Initialize sample webhook plugin."""
        super().__init__(
            name="sample_webhook",
            version="1.0.0",
            description="Sample webhook plugin demonstrating plugin interface"
        )

        # Plugin-specific attributes
        self.dependencies = ["webhook_core"]
        self.supported_events = [
            "tts.completed",
            "tts.failed",
            "audio.enhanced",
            "plugin.loaded",
            "plugin.error"
        ]
        self.webhook_url = "https://example.com/webhook"
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "TTS-Webhook-Plugin/1.0"
        }
        self.timeout = 10
        self.retry_attempts = 3
        self.retry_delay = 1.0

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        info = super().get_plugin_info()
        info.update({
            'supported_events': self.supported_events,
            'webhook_url': self.webhook_url,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'features': [
                'event_notifications',
                'retry_mechanism',
                'custom_headers',
                'event_filtering'
            ]
        })
        return info

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize plugin with configuration."""
        try:
            self.log_info("Initializing Sample Webhook Plugin")

            # Load configuration
            if config:
                self.webhook_url = config.get('webhook_url', self.webhook_url)
                self.headers.update(config.get('headers', {}))
                self.timeout = config.get('timeout', self.timeout)
                self.retry_attempts = config.get('retry_attempts', self.retry_attempts)
                self.supported_events = config.get('supported_events', self.supported_events)

            # Validate webhook URL
            if not self.webhook_url.startswith(('http://', 'https://')):
                self.log_warning("Invalid webhook URL format")
                return False

            # Register hooks
            self.register_hook(HookType.PRE_WEBHOOK, self.pre_webhook_hook)
            self.register_hook(HookType.POST_WEBHOOK, self.post_webhook_hook)

            # Register event handlers
            for event in self.supported_events:
                self.register_event_handler(EventType.CUSTOM, self.handle_custom_event)

            self.log_info("Sample Webhook Plugin initialized successfully")
            return True

        except Exception as e:
            self.log_error(f"Failed to initialize plugin: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        try:
            self.log_info("Cleaning up Sample Webhook Plugin")

            # Unregister hooks and handlers
            self.unregister_hook(HookType.PRE_WEBHOOK, self.pre_webhook_hook)
            self.unregister_hook(HookType.POST_WEBHOOK, self.post_webhook_hook)

            for event in self.supported_events:
                self.unregister_event_handler(EventType.CUSTOM, self.handle_custom_event)

            # Clear any cached data
            self.reset_performance_metrics()

            self.log_info("Sample Webhook Plugin cleaned up successfully")

        except Exception as e:
            self.log_error(f"Error during cleanup: {e}")

    async def send_webhook(self, event: str, data: Dict[str, Any],
                          headers: Dict[str, str] = None) -> bool:
        """Send webhook notification."""
        start_time = time.time()

        try:
            # Check if event is supported
            if event not in self.supported_events:
                self.log_warning(f"Unsupported event: {event}")
                return False

            # Pre-processing hook
            pre_result = await self.pre_webhook_hook(event, data)
            if pre_result.get('skip_webhook'):
                self.log_info(f"Webhook skipped for event: {event}")
                return True

            # Prepare webhook payload
            payload = self._prepare_webhook_payload(event, data, pre_result)

            # Merge headers
            request_headers = self.headers.copy()
            if headers:
                request_headers.update(headers)

            # Send webhook with retry logic
            success = await self._send_webhook_with_retry(
                self.webhook_url,
                payload,
                request_headers
            )

            # Post-processing hook
            post_result = await self.post_webhook_hook(event, payload, success)

            # Update performance metrics
            processing_time = time.time() - start_time
            self._performance_metrics['execution_count'] += 1
            self._performance_metrics['total_execution_time'] += processing_time
            self._performance_metrics['last_execution_time'] = processing_time

            # Emit event
            await self.emit_event(EventType.WEBHOOK_SENT, EventContext(
                event_type=EventType.WEBHOOK_SENT,
                plugin_name=self.name,
                data={
                    'event': event,
                    'success': success,
                    'processing_time': processing_time,
                    'attempts': self.retry_attempts
                }
            ))

            return success

        except Exception as e:
            error_time = time.time() - start_time
            self.log_error(f"Webhook failed: {e}")
            self._performance_metrics['error_count'] += 1
            return False

    async def get_supported_events(self) -> List[str]:
        """Get supported webhook events."""
        return self.supported_events.copy()

    async def pre_webhook_hook(self, event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-webhook processing hook."""
        # Event filtering
        if self._should_filter_event(event, data):
            return {'skip_webhook': True}

        # Data enrichment
        enriched_data = data.copy()
        enriched_data['timestamp'] = datetime.utcnow().isoformat()
        enriched_data['plugin_name'] = self.name
        enriched_data['plugin_version'] = self.version

        return {
            'enriched_data': enriched_data
        }

    async def post_webhook_hook(self, event: str, response: Dict[str, Any], success: bool) -> Dict[str, Any]:
        """Post-webhook processing hook."""
        # Log webhook result
        if success:
            self.log_info(f"Webhook sent successfully: {event}")
        else:
            self.log_error(f"Webhook failed: {event}")

        return {
            'logged': True
        }

    async def handle_custom_event(self, context: EventContext) -> None:
        """Handle custom events."""
        self.log_info(f"Custom event received: {context.event_type.value}")

        # Auto-send webhook for supported events
        if context.data.get('event_type') in self.supported_events:
            await self.send_webhook(
                context.data['event_type'],
                context.data
            )

    async def _send_webhook_with_retry(self, url: str, payload: Dict[str, Any],
                                      headers: Dict[str, str]) -> bool:
        """Send webhook with retry logic."""
        import aiohttp

        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:

                        if response.status in [200, 201, 202, 204]:
                            self.log_info(f"Webhook sent successfully on attempt {attempt + 1}")
                            return True
                        else:
                            response_text = await response.text()
                            self.log_warning(
                                f"Webhook failed with status {response.status}: {response_text}"
                            )

            except asyncio.TimeoutError:
                self.log_warning(f"Webhook timeout on attempt {attempt + 1}")
            except Exception as e:
                self.log_warning(f"Webhook error on attempt {attempt + 1}: {e}")

            # Wait before retry
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        return False

    def _prepare_webhook_payload(self, event: str, data: Dict[str, Any],
                                pre_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare webhook payload."""
        payload = {
            'event': event,
            'timestamp': datetime.utcnow().isoformat(),
            'plugin': {
                'name': self.name,
                'version': self.version,
                'description': self.description
            },
            'data': data
        }

        # Add enriched data if available
        if pre_result.get('enriched_data'):
            payload['data'].update(pre_result['enriched_data'])

        return payload

    def _should_filter_event(self, event: str, data: Dict[str, Any]) -> bool:
        """Check if event should be filtered."""
        # Filter based on event type
        if event not in self.supported_events:
            return True

        # Filter based on data content
        if event == "tts.failed" and data.get('error_type') == 'validation':
            # Skip validation errors
            return True

        return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        # Validate webhook URL
        if 'webhook_url' in config:
            url = config['webhook_url']
            if not url.startswith(('http://', 'https://')):
                self.log_warning("Invalid webhook URL format")
                return False

        # Validate timeout
        if 'timeout' in config:
            timeout = config['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                self.log_warning("Invalid timeout value")
                return False

        return True

    def get_required_permissions(self) -> List:
        """Get required permissions."""
        from models.plugin import PluginPermission
        return [PluginPermission.EXECUTE, PluginPermission.WRITE]


# Plugin registration
def register_plugin():
    """Register the plugin."""
    from utils.plugin_interface import plugin_registry

    plugin_class = SampleWebhookPlugin
    plugin_registry.register_plugin_class('sample_webhook', plugin_class)

    print("Sample Webhook Plugin registered successfully!")


# Auto-register when module is imported
register_plugin()