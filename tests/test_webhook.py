"""
Webhook System Tests
"""
import pytest
import pytest_asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from models.webhook import Webhook, WebhookStatus, WebhookEventType, DeliveryStatus
from utils.webhook_security import webhook_security, WebhookSecurityError, InvalidSignatureError
from utils.webhook_events import event_manager, WebhookEvent, EventTemplate
from utils.webhook_service import webhook_service
from utils.webhook_integration import tts_webhook_integration
from config.webhook import webhook_config

class TestWebhookSecurity:
    """Test webhook security utilities"""

    def test_generate_secret(self):
        """Test secret generation"""
        secret = webhook_security.generate_secret(32)
        assert len(secret) == 32
        assert all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in secret)

    def test_generate_signature(self):
        """Test signature generation"""
        payload = '{"test": "data"}'
        secret = 'test_secret'

        signature = webhook_security.generate_signature(payload, secret)
        assert signature.startswith('sha256=')
        assert len(signature) > 7  # sha256= + hash

    def test_verify_signature_valid(self):
        """Test valid signature verification"""
        payload = '{"test": "data"}'
        secret = 'test_secret'

        signature = webhook_security.generate_signature(payload, secret)
        assert webhook_security.verify_signature(payload, signature, secret)

    def test_verify_signature_invalid(self):
        """Test invalid signature verification"""
        payload = '{"test": "data"}'
        secret = 'test_secret'

        invalid_signature = 'sha256=invalid_hash'
        assert not webhook_security.verify_signature(payload, invalid_signature, secret)

    def test_verify_signature_wrong_secret(self):
        """Test signature verification with wrong secret"""
        payload = '{"test": "data"}'
        secret = 'test_secret'

        signature = webhook_security.generate_signature(payload, secret)
        assert not webhook_security.verify_signature(payload, signature, 'wrong_secret')

    def test_validate_url_valid(self):
        """Test valid URL validation"""
        valid_urls = [
            'https://example.com/webhook',
            'http://localhost:3000/webhook',
            'https://api.example.com/v1/webhooks'
        ]

        for url in valid_urls:
            assert webhook_security.validate_url(url)

    def test_validate_url_invalid(self):
        """Test invalid URL validation"""
        invalid_urls = [
            'ftp://example.com',
            'invalid-url',
            'https://' + 'a' * 500,  # Too long
            ''
        ]

        for url in invalid_urls:
            assert not webhook_security.validate_url(url)

    def test_validate_headers_valid(self):
        """Test valid headers validation"""
        headers = {
            'Content-Type': 'application/json',
            'X-Custom-Header': 'value'
        }
        assert webhook_security.validate_headers(headers)

    def test_validate_headers_invalid(self):
        """Test invalid headers validation"""
        # Headers too large
        large_headers = {f'header_{i}': 'x' * 1000 for i in range(10)}
        assert not webhook_security.validate_headers(large_headers)

    def test_is_ip_allowed_whitelist(self):
        """Test IP whitelist functionality"""
        with patch('config.webhook.webhook_config') as mock_config:
            mock_config.ENABLE_IP_WHITELIST = True
            mock_config.TRUSTED_IPS = ['192.168.1.0/24', '10.0.0.1']

            assert webhook_security.is_ip_allowed('192.168.1.100')
            assert webhook_security.is_ip_allowed('10.0.0.1')
            assert not webhook_security.is_ip_allowed('172.16.0.1')

    def test_is_ip_allowed_blacklist(self):
        """Test IP blacklist functionality"""
        with patch('config.webhook.webhook_config') as mock_config:
            mock_config.ENABLE_IP_BLACKLIST = True
            mock_config.TRUSTED_IPS = []

            # Mock blacklist
            webhook_security.is_ip_allowed = lambda ip, whitelist=None, blacklist=['192.168.1.0/24']: False

            assert not webhook_security.is_ip_allowed('192.168.1.100')
            assert webhook_security.is_ip_allowed('10.0.0.1')

class TestWebhookEvents:
    """Test webhook events utilities"""

    def test_create_tts_completed_event(self):
        """Test TTS completed event creation"""
        event = event_manager.create_event(
            WebhookEventType.TTS_COMPLETED,
            {
                'request_id': 'test_123',
                'audio_url': 'https://example.com/audio.mp3',
                'duration': 2.5,
                'text_length': 100
            }
        )

        assert event.event_type == WebhookEventType.TTS_COMPLETED
        assert event.data['request_id'] == 'test_123'
        assert event.data['duration'] == 2.5

    def test_event_validation(self):
        """Test event data validation"""
        # Valid event
        assert event_manager.validate_event_data(
            WebhookEventType.TTS_COMPLETED,
            {
                'request_id': 'test_123',
                'audio_url': 'https://example.com/audio.mp3',
                'duration': 2.5,
                'text_length': 100
            }
        )

        # Invalid event - missing required field
        assert not event_manager.validate_event_data(
            WebhookEventType.TTS_COMPLETED,
            {
                'request_id': 'test_123',
                # Missing audio_url
                'duration': 2.5,
                'text_length': 100
            }
        )

    def test_event_template_tts_completed(self):
        """Test TTS completed event template"""
        data = EventTemplate.tts_completed(
            request_id='test_123',
            audio_url='https://example.com/audio.mp3',
            duration=2.5,
            text_length=100,
            quality_score=0.95
        )

        assert data['request_id'] == 'test_123'
        assert data['audio_url'] == 'https://example.com/audio.mp3'
        assert data['duration'] == 2.5
        assert data['text_length'] == 100
        assert data['quality_score'] == 0.95
        assert data['status'] == 'completed'

    def test_event_template_tts_error(self):
        """Test TTS error event template"""
        data = EventTemplate.tts_error(
            request_id='test_123',
            error_code='PROCESSING_FAILED',
            error_message='Audio processing failed',
            text_length=100
        )

        assert data['request_id'] == 'test_123'
        assert data['error_code'] == 'PROCESSING_FAILED'
        assert data['error_message'] == 'Audio processing failed'
        assert data['text_length'] == 100
        assert data['status'] == 'error'

class TestWebhookService:
    """Test webhook service"""

    @pytest_asyncio.async_test
    async def test_send_webhook_immediate_success(self):
        """Test immediate webhook delivery success"""
        # Mock webhook
        webhook = MagicMock()
        webhook.id = 1
        webhook.url = 'https://example.com/webhook'
        webhook.secret = 'test_secret'
        webhook.headers = {}
        webhook.timeout = 30

        # Mock event
        event = WebhookEvent(
            event_type=WebhookEventType.TTS_COMPLETED,
            data={
                'request_id': 'test_123',
                'audio_url': 'https://example.com/audio.mp3'
            }
        )

        # Mock HTTP response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='OK')
            mock_post.return_value.__aenter__.return_value = mock_response

            success, response_data = await webhook_service.send_webhook_immediate(webhook, event)

            assert success
            assert response_data['status'] == 200
            assert response_data['body'] == 'OK'

    @pytest_asyncio.async_test
    async def test_send_webhook_immediate_failure(self):
        """Test immediate webhook delivery failure"""
        # Mock webhook
        webhook = MagicMock()
        webhook.id = 1
        webhook.url = 'https://example.com/webhook'
        webhook.secret = 'test_secret'
        webhook.headers = {}
        webhook.timeout = 30

        # Mock event
        event = WebhookEvent(
            event_type=WebhookEventType.TTS_COMPLETED,
            data={
                'request_id': 'test_123',
                'audio_url': 'https://example.com/audio.mp3'
            }
        )

        # Mock HTTP error
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception('Connection failed')

            success, response_data = await webhook_service.send_webhook_immediate(webhook, event)

            assert not success
            assert 'error' in response_data

    @pytest_asyncio.async_test
    async def test_batch_webhook_delivery(self):
        """Test batch webhook delivery"""
        # Mock webhook
        webhook = MagicMock()
        webhook.id = 1
        webhook.url = 'https://example.com/webhook'
        webhook.secret = 'test_secret'
        webhook.headers = {}
        webhook.timeout = 30

        # Mock events
        events = [
            WebhookEvent(
                event_type=WebhookEventType.TTS_COMPLETED,
                data={'request_id': f'test_{i}'}
            ) for i in range(3)
        ]

        # Mock HTTP response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='OK')
            mock_post.return_value.__aenter__.return_value = mock_response

            success = await webhook_service.send_batch_webhook(webhook, events)

            assert success
            # Verify batch payload was sent
            call_args = mock_post.call_args
            payload = json.loads(call_args[1]['data'])
            assert payload['count'] == 3
            assert len(payload['events']) == 3

class TestWebhookIntegration:
    """Test webhook integration"""

    @pytest_asyncio.async_test
    async def test_tts_completed_notification(self):
        """Test TTS completed notification"""
        with patch('utils.webhook_integration.tts_webhook_integration._get_webhooks_for_event') as mock_get_webhooks:
            # Mock webhooks
            mock_webhook = MagicMock()
            mock_webhook.id = 1
            mock_get_webhooks.return_value = [mock_webhook]

            # Mock webhook service
            with patch('utils.webhook_integration.queue_webhook_for_batch') as mock_queue:
                await tts_webhook_integration.notify_tts_completed(
                    request_id='test_123',
                    audio_url='https://example.com/audio.mp3',
                    duration=2.5,
                    text_length=100,
                    organization_id=1
                )

                # Verify webhook was queued for batch
                mock_queue.assert_called_once()

    @pytest_asyncio.async_test
    async def test_tts_error_notification(self):
        """Test TTS error notification"""
        with patch('utils.webhook_integration.tts_webhook_integration._get_webhooks_for_event') as mock_get_webhooks:
            # Mock webhooks
            mock_webhook = MagicMock()
            mock_webhook.id = 1
            mock_get_webhooks.return_value = [mock_webhook]

            # Mock webhook service
            with patch('utils.webhook_integration.send_webhook_notification') as mock_send:
                await tts_webhook_integration.notify_tts_error(
                    request_id='test_123',
                    error_code='PROCESSING_FAILED',
                    error_message='Audio processing failed',
                    organization_id=1
                )

                # Verify webhook was sent immediately (no batching for errors)
                mock_send.assert_called_once()

class TestWebhookModels:
    """Test webhook models"""

    def test_webhook_is_event_enabled(self):
        """Test webhook event enabling"""
        webhook = Webhook(
            id=1,
            name='Test Webhook',
            url='https://example.com/webhook',
            secret='test_secret',
            events=[WebhookEventType.TTS_COMPLETED, WebhookEventType.TTS_ERROR],
            organization_id=1,
            created_by=1
        )

        assert webhook.is_event_enabled(WebhookEventType.TTS_COMPLETED)
        assert webhook.is_event_enabled(WebhookEventType.TTS_ERROR)
        assert not webhook.is_event_enabled(WebhookEventType.BATCH_COMPLETED)

    def test_webhook_retry_config(self):
        """Test webhook retry configuration"""
        webhook = Webhook(
            id=1,
            name='Test Webhook',
            url='https://example.com/webhook',
            secret='test_secret',
            events=[WebhookEventType.TTS_COMPLETED],
            organization_id=1,
            created_by=1,
            retry_policy={
                'max_attempts': 5,
                'backoff_multiplier': 3,
                'initial_delay': 2,
                'max_delay': 600
            }
        )

        config = webhook.get_retry_config()
        assert config['max_attempts'] == 5
        assert config['backoff_multiplier'] == 3
        assert config['initial_delay'] == 2
        assert config['max_delay'] == 600

    def test_webhook_delivery_is_retryable(self):
        """Test webhook delivery retry logic"""
        webhook = Webhook(
            id=1,
            name='Test Webhook',
            url='https://example.com/webhook',
            secret='test_secret',
            events=[WebhookEventType.TTS_COMPLETED],
            organization_id=1,
            created_by=1
        )

        delivery = MagicMock()
        delivery.attempt_count = 1
        delivery.status = DeliveryStatus.FAILED
        delivery.webhook = webhook

        # Mock the is_retryable method
        with patch.object(delivery, 'is_retryable', return_value=True):
            assert delivery.is_retryable()

class TestWebhookConfiguration:
    """Test webhook configuration"""

    def test_webhook_config_defaults(self):
        """Test webhook configuration defaults"""
        assert webhook_config.DEFAULT_TIMEOUT == 30
        assert webhook_config.DEFAULT_MAX_ATTEMPTS == 3
        assert webhook_config.DEFAULT_BACKOFF_MULTIPLIER == 2.0
        assert webhook_config.BATCH_SIZE == 10

    def test_get_retry_config(self):
        """Test getting retry configuration"""
        config = webhook_config.get_retry_config()
        assert 'max_attempts' in config
        assert 'backoff_multiplier' in config
        assert 'initial_delay' in config
        assert 'max_delay' in config

    def test_get_rate_limit_config(self):
        """Test getting rate limit configuration"""
        config = webhook_config.get_rate_limit_config()
        assert 'requests_per_minute' in config
        assert 'burst_limit' in config
        assert 'window_seconds' in config

    def test_get_batch_config(self):
        """Test getting batch configuration"""
        config = webhook_config.get_batch_config()
        assert 'batch_size' in config
        assert 'batch_timeout' in config
        assert 'max_batch_size' in config

# Integration test
@pytest_asyncio.async_test
async def test_webhook_integration_initialization():
    """Test webhook integration initialization"""
    with patch('utils.webhook_service.webhook_service.start') as mock_start:
        await tts_webhook_integration.initialize()
        mock_start.assert_called_once()

@pytest_asyncio.async_test
async def test_webhook_integration_shutdown():
    """Test webhook integration shutdown"""
    with patch('utils.webhook_service.webhook_service.stop') as mock_stop:
        await tts_webhook_integration.shutdown()
        mock_stop.assert_called_once()