"""
Webhook API Routes
"""
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

from models.webhook import Webhook, WebhookStatus, WebhookEventType
from utils.webhook_service import webhook_service, send_webhook_immediate
from utils.webhook_events import event_manager, create_tts_completed_event, create_tts_error_event
from utils.webhook_security import webhook_security
from config.webhook import webhook_config

logger = logging.getLogger(__name__)

webhook_bp = Blueprint('webhook', __name__, url_prefix='/api/v1')

# Mock database functions (in real app, these would interact with actual database)
async def get_webhook_by_id(webhook_id: int) -> Webhook:
    """Mock function to get webhook by ID"""
    # In real app, this would query the database
    return None

async def get_webhooks_by_organization(organization_id: int) -> List[Webhook]:
    """Mock function to get webhooks by organization"""
    # In real app, this would query the database
    return []

async def create_webhook(data: Dict[str, Any]) -> Webhook:
    """Mock function to create webhook"""
    # In real app, this would save to database
    return None

async def update_webhook(webhook_id: int, data: Dict[str, Any]) -> Webhook:
    """Mock function to update webhook"""
    # In real app, this would update database
    return None

async def delete_webhook(webhook_id: int) -> bool:
    """Mock function to delete webhook"""
    # In real app, this would delete from database
    return True

async def get_webhook_deliveries(webhook_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Mock function to get webhook delivery history"""
    # In real app, this would query the database
    return []

@webhook_bp.route('/webhooks', methods=['POST'])
async def create_webhook_endpoint():
    """Tạo webhook mới"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['name', 'url', 'events']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'code': 'MISSING_REQUIRED_FIELD'
                }), 400

        # Validate URL
        if not webhook_security.validate_url(data['url']):
            return jsonify({
                'error': 'Invalid URL format',
                'code': 'INVALID_URL'
            }), 400

        # Validate events
        supported_events = [e.value for e in WebhookEventType]
        for event in data['events']:
            if event not in supported_events:
                return jsonify({
                    'error': f'Unsupported event type: {event}',
                    'code': 'UNSUPPORTED_EVENT_TYPE'
                }), 400

        # Validate headers if provided
        if 'headers' in data and not webhook_security.validate_headers(data['headers']):
            return jsonify({
                'error': 'Invalid headers',
                'code': 'INVALID_HEADERS'
            }), 400

        # Generate secret
        secret = webhook_security.generate_secret()

        # Create webhook data
        webhook_data = {
            'name': data['name'],
            'description': data.get('description', ''),
            'url': data['url'],
            'secret': secret,
            'status': WebhookStatus.ACTIVE,
            'events': data['events'],
            'headers': data.get('headers', {}),
            'retry_policy': data.get('retry_policy', {}),
            'rate_limit': data.get('rate_limit', {}),
            'timeout': data.get('timeout', webhook_config.DEFAULT_TIMEOUT),
            'created_by': 1,  # In real app, get from auth context
            'organization_id': 1,  # In real app, get from auth context
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }

        # Create webhook
        webhook = await create_webhook(webhook_data)

        return jsonify({
            'id': webhook.id,
            'name': webhook.name,
            'url': webhook.url,
            'secret': secret,  # Only return secret on creation
            'status': webhook.status.value,
            'events': webhook.events,
            'created_at': webhook.created_at.isoformat()
        }), 201

    except Exception as e:
        logger.error(f"Error creating webhook: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks', methods=['GET'])
async def list_webhooks():
    """Liệt kê webhooks"""
    try:
        # Get query parameters
        status = request.args.get('status')
        event_type = request.args.get('event_type')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        # Get webhooks (in real app, filter by organization)
        webhooks = await get_webhooks_by_organization(1)  # Mock organization ID

        # Apply filters
        if status:
            webhooks = [w for w in webhooks if w.status.value == status]

        if event_type:
            webhooks = [w for w in webhooks if event_type in w.events]

        # Pagination
        total = len(webhooks)
        webhooks = webhooks[offset:offset + limit]

        # Format response
        result = []
        for webhook in webhooks:
            result.append({
                'id': webhook.id,
                'name': webhook.name,
                'description': webhook.description,
                'url': webhook.url,
                'status': webhook.status.value,
                'events': webhook.events,
                'created_at': webhook.created_at.isoformat(),
                'updated_at': webhook.updated_at.isoformat()
            })

        return jsonify({
            'webhooks': result,
            'total': total,
            'limit': limit,
            'offset': offset
        })

    except Exception as e:
        logger.error(f"Error listing webhooks: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks/<int:webhook_id>', methods=['GET'])
async def get_webhook(webhook_id: int):
    """Lấy thông tin webhook"""
    try:
        webhook = await get_webhook_by_id(webhook_id)

        if not webhook:
            return jsonify({
                'error': 'Webhook not found',
                'code': 'WEBHOOK_NOT_FOUND'
            }), 404

        # Get delivery stats
        deliveries = await get_webhook_deliveries(webhook_id, limit=10)
        success_count = sum(1 for d in deliveries if d['status'] == 'success')
        failed_count = sum(1 for d in deliveries if d['status'] == 'failed')

        return jsonify({
            'id': webhook.id,
            'name': webhook.name,
            'description': webhook.description,
            'url': webhook.url,
            'status': webhook.status.value,
            'events': webhook.events,
            'headers': webhook.headers,
            'timeout': webhook.timeout,
            'created_at': webhook.created_at.isoformat(),
            'updated_at': webhook.updated_at.isoformat(),
            'stats': {
                'total_deliveries': len(deliveries),
                'successful_deliveries': success_count,
                'failed_deliveries': failed_count
            }
        })

    except Exception as e:
        logger.error(f"Error getting webhook: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks/<int:webhook_id>', methods=['PUT'])
async def update_webhook(webhook_id: int):
    """Cập nhật webhook"""
    try:
        webhook = await get_webhook_by_id(webhook_id)

        if not webhook:
            return jsonify({
                'error': 'Webhook not found',
                'code': 'WEBHOOK_NOT_FOUND'
            }), 404

        data = request.get_json()

        # Validate URL if provided
        if 'url' in data and not webhook_security.validate_url(data['url']):
            return jsonify({
                'error': 'Invalid URL format',
                'code': 'INVALID_URL'
            }), 400

        # Validate events if provided
        if 'events' in data:
            supported_events = [e.value for e in WebhookEventType]
            for event in data['events']:
                if event not in supported_events:
                    return jsonify({
                        'error': f'Unsupported event type: {event}',
                        'code': 'UNSUPPORTED_EVENT_TYPE'
                    }), 400

        # Validate headers if provided
        if 'headers' in data and not webhook_security.validate_headers(data['headers']):
            return jsonify({
                'error': 'Invalid headers',
                'code': 'INVALID_HEADERS'
            }), 400

        # Update webhook data
        update_data = {
            'updated_at': datetime.utcnow()
        }

        if 'name' in data:
            update_data['name'] = data['name']
        if 'description' in data:
            update_data['description'] = data['description']
        if 'url' in data:
            update_data['url'] = data['url']
        if 'events' in data:
            update_data['events'] = data['events']
        if 'headers' in data:
            update_data['headers'] = data['headers']
        if 'timeout' in data:
            update_data['timeout'] = data['timeout']
        if 'status' in data:
            update_data['status'] = WebhookStatus(data['status'])

        # Update webhook
        updated_webhook = await update_webhook(webhook_id, update_data)

        return jsonify({
            'id': updated_webhook.id,
            'name': updated_webhook.name,
            'description': updated_webhook.description,
            'url': updated_webhook.url,
            'status': updated_webhook.status.value,
            'events': updated_webhook.events,
            'updated_at': updated_webhook.updated_at.isoformat()
        })

    except Exception as e:
        logger.error(f"Error updating webhook: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks/<int:webhook_id>', methods=['DELETE'])
async def delete_webhook_endpoint(webhook_id: int):
    """Xóa webhook"""
    try:
        webhook = await get_webhook_by_id(webhook_id)

        if not webhook:
            return jsonify({
                'error': 'Webhook not found',
                'code': 'WEBHOOK_NOT_FOUND'
            }), 404

        # Delete webhook
        success = await delete_webhook(webhook_id)

        if success:
            return jsonify({'message': 'Webhook deleted successfully'}), 200
        else:
            return jsonify({
                'error': 'Failed to delete webhook',
                'code': 'DELETE_FAILED'
            }), 500

    except Exception as e:
        logger.error(f"Error deleting webhook: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks/<int:webhook_id>/test', methods=['POST'])
async def test_webhook(webhook_id: int):
    """Test webhook"""
    try:
        webhook = await get_webhook_by_id(webhook_id)

        if not webhook:
            return jsonify({
                'error': 'Webhook not found',
                'code': 'WEBHOOK_NOT_FOUND'
            }), 404

        # Create test event
        test_event = create_tts_completed_event(
            request_id="test_request_123",
            audio_url="https://example.com/test.mp3",
            duration=2.5,
            text_length=100,
            test=True
        )

        # Send test webhook
        success, response_data = await send_webhook_immediate(webhook, test_event)

        return jsonify({
            'success': success,
            'response_status': response_data.get('status'),
            'response_body': response_data.get('body'),
            'delivery_time': response_data.get('delivery_time'),
            'error': response_data.get('error')
        })

    except Exception as e:
        logger.error(f"Error testing webhook: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks/<int:webhook_id>/history', methods=['GET'])
async def get_webhook_history(webhook_id: int):
    """Lấy lịch sử delivery của webhook"""
    try:
        webhook = await get_webhook_by_id(webhook_id)

        if not webhook:
            return jsonify({
                'error': 'Webhook not found',
                'code': 'WEBHOOK_NOT_FOUND'
            }), 404

        # Get query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        status = request.args.get('status')

        # Get deliveries
        deliveries = await get_webhook_deliveries(webhook_id, limit=limit + offset)

        # Apply status filter
        if status:
            deliveries = [d for d in deliveries if d['status'] == status]

        # Pagination
        total = len(deliveries)
        deliveries = deliveries[offset:offset + limit]

        return jsonify({
            'deliveries': deliveries,
            'total': total,
            'limit': limit,
            'offset': offset
        })

    except Exception as e:
        logger.error(f"Error getting webhook history: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks/events', methods=['GET'])
async def list_event_types():
    """Liệt kê các event types được hỗ trợ"""
    try:
        event_types = event_manager.list_event_types()
        event_configs = {}

        for event_type in event_types:
            config = event_manager.get_event_config(event_type)
            event_configs[event_type] = config

        return jsonify({
            'event_types': event_configs
        })

    except Exception as e:
        logger.error(f"Error listing event types: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@webhook_bp.route('/webhooks/stats', methods=['GET'])
async def get_webhook_stats():
    """Lấy thống kê webhook system"""
    try:
        # Get service stats
        service_stats = await webhook_service.get_delivery_stats()

        # Get webhook counts (mock data)
        total_webhooks = len(await get_webhooks_by_organization(1))
        active_webhooks = total_webhooks  # Mock: assume all are active

        return jsonify({
            'total_webhooks': total_webhooks,
            'active_webhooks': active_webhooks,
            'service_stats': service_stats,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting webhook stats: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500