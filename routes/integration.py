"""
Integration API Routes for TTS System

This module provides REST API endpoints for managing integrations with external services.
"""

from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy.orm import Session
from pydantic import ValidationError
import logging

from models.integration import (
    IntegrationConfig, IntegrationType, IntegrationStatus,
    IntegrationDB, IntegrationTemplateDB
)
from utils.integration_manager import IntegrationManager
from utils.integration_security import IntegrationSecurity
from utils.exceptions import IntegrationError, ValidationError, AuthenticationError
from utils.database import get_db_session
from config.integration import IntegrationConfig as AppIntegrationConfig

# Create blueprint
integration_bp = Blueprint('integration', __name__, url_prefix='/api/v1/integrations')

# Initialize services
integration_manager = None
integration_security = None

def init_integration_services():
    """Initialize integration services"""
    global integration_manager, integration_security

    if integration_manager is None:
        db_session = get_db_session()
        redis_client = current_app.redis_client
        integration_security = IntegrationSecurity(AppIntegrationConfig())
        integration_manager = IntegrationManager(db_session, redis_client, integration_security)


@integration_bp.before_request
def before_request():
    """Initialize services before each request"""
    init_integration_services()


@integration_bp.route('', methods=['POST'])
@jwt_required()
def create_integration():
    """
    Create a new integration

    ---
    tags:
      - integrations
    summary: Create integration
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/IntegrationConfig'
    responses:
      201:
        description: Integration created successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                data:
                  $ref: '#/components/schemas/Integration'
                message:
                  type: string
      400:
        description: Invalid request data
      401:
        description: Unauthorized
      500:
        description: Internal server error
    """
    try:
        # Get user identity
        user_id = get_jwt_identity()
        if isinstance(user_id, dict):
            user_id = user_id.get('id')

        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'Request body is required'
            }), 400

        # Validate integration config
        try:
            config = IntegrationConfig(**data)
        except ValidationError as e:
            return jsonify({
                'success': False,
                'message': 'Invalid integration configuration',
                'errors': e.errors()
            }), 400

        # Create integration
        integration = integration_manager.create_integration(
            config=config,
            user_id=user_id
        )

        return jsonify({
            'success': True,
            'data': _format_integration_response(integration),
            'message': 'Integration created successfully'
        }), 201

    except ValidationError as e:
        return jsonify({
            'success': False,
            'message': 'Validation error',
            'errors': e.errors()
        }), 400
    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error creating integration: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('', methods=['GET'])
@jwt_required()
def list_integrations():
    """
    List integrations with optional filters

    ---
    tags:
      - integrations
    summary: List integrations
    security:
      - BearerAuth: []
    parameters:
      - name: type
        in: query
        schema:
          type: string
          enum: [cloud_storage, notification, database, api, file_processing, webhook]
        description: Filter by integration type
      - name: provider
        in: query
        schema:
          type: string
        description: Filter by provider
      - name: active
        in: query
        schema:
          type: boolean
        description: Filter by active status
      - name: page
        in: query
        schema:
          type: integer
          minimum: 1
          default: 1
        description: Page number
      - name: per_page
        in: query
        schema:
          type: integer
          minimum: 1
          maximum: 100
          default: 20
        description: Items per page
    responses:
      200:
        description: Integrations retrieved successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                data:
                  type: array
                  items:
                    $ref: '#/components/schemas/Integration'
                pagination:
                  type: object
                  properties:
                    page:
                      type: integer
                    per_page:
                      type: integer
                    total:
                      type: integer
                    pages:
                      type: integer
                message:
                  type: string
      401:
        description: Unauthorized
      500:
        description: Internal server error
    """
    try:
        # Get query parameters
        integration_type = request.args.get('type')
        provider = request.args.get('provider')
        active = request.args.get('active')
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)

        # Convert string parameters
        if integration_type:
            try:
                integration_type = IntegrationType(integration_type)
            except ValueError:
                return jsonify({
                    'success': False,
                    'message': 'Invalid integration type'
                }), 400

        if active is not None:
            active = active.lower() == 'true'

        # Get integrations
        integrations = integration_manager.list_integrations(
            integration_type=integration_type,
            provider=provider,
            is_active=active
        )

        # Apply pagination
        total = len(integrations)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_integrations = integrations[start:end]

        # Format response
        data = [_format_integration_response(integration) for integration in paginated_integrations]

        pagination = {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }

        return jsonify({
            'success': True,
            'data': data,
            'pagination': pagination,
            'message': 'Integrations retrieved successfully'
        }), 200

    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error listing integrations: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('/<int:integration_id>', methods=['PUT'])
@jwt_required()
def update_integration(integration_id: int):
    """
    Update an existing integration

    ---
    tags:
      - integrations
    summary: Update integration
    security:
      - BearerAuth: []
    parameters:
      - name: integration_id
        in: path
        required: true
        schema:
          type: integer
        description: Integration ID
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/IntegrationConfig'
    responses:
      200:
        description: Integration updated successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                data:
                  $ref: '#/components/schemas/Integration'
                message:
                  type: string
      400:
        description: Invalid request data
      401:
        description: Unauthorized
      404:
        description: Integration not found
      500:
        description: Internal server error
    """
    try:
        # Get user identity
        user_id = get_jwt_identity()
        if isinstance(user_id, dict):
            user_id = user_id.get('id')

        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'Request body is required'
            }), 400

        # Validate integration config
        try:
            config = IntegrationConfig(**data)
        except ValidationError as e:
            return jsonify({
                'success': False,
                'message': 'Invalid integration configuration',
                'errors': e.errors()
            }), 400

        # Update integration
        integration = integration_manager.update_integration(
            integration_id=integration_id,
            config=config,
            user_id=user_id
        )

        return jsonify({
            'success': True,
            'data': _format_integration_response(integration),
            'message': 'Integration updated successfully'
        }), 200

    except ValidationError as e:
        return jsonify({
            'success': False,
            'message': 'Validation error',
            'errors': e.errors()
        }), 400
    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error updating integration {integration_id}: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('/<int:integration_id>', methods=['DELETE'])
@jwt_required()
def delete_integration(integration_id: int):
    """
    Delete an integration

    ---
    tags:
      - integrations
    summary: Delete integration
    security:
      - BearerAuth: []
    parameters:
      - name: integration_id
        in: path
        required: true
        schema:
          type: integer
        description: Integration ID
    responses:
      200:
        description: Integration deleted successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                message:
                  type: string
      401:
        description: Unauthorized
      404:
        description: Integration not found
      500:
        description: Internal server error
    """
    try:
        # Get user identity
        user_id = get_jwt_identity()
        if isinstance(user_id, dict):
            user_id = user_id.get('id')

        # Delete integration
        success = integration_manager.delete_integration(
            integration_id=integration_id,
            user_id=user_id
        )

        if success:
            return jsonify({
                'success': True,
                'message': 'Integration deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to delete integration'
            }), 400

    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error deleting integration {integration_id}: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('/<int:integration_id>/test', methods=['POST'])
@jwt_required()
def test_integration(integration_id: int):
    """
    Test integration connectivity and functionality

    ---
    tags:
      - integrations
    summary: Test integration
    security:
      - BearerAuth: []
    parameters:
      - name: integration_id
        in: path
        required: true
        schema:
          type: integer
        description: Integration ID
    responses:
      200:
        description: Integration test completed
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                data:
                  type: object
                  properties:
                    success:
                      type: boolean
                    message:
                      type: string
                    response_time_ms:
                      type: integer
                    details:
                      type: object
                message:
                  type: string
      401:
        description: Unauthorized
      404:
        description: Integration not found
      500:
        description: Internal server error
    """
    try:
        # Test integration
        test_result = integration_manager.test_integration(integration_id)

        return jsonify({
            'success': True,
            'data': test_result,
            'message': 'Integration test completed'
        }), 200

    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error testing integration {integration_id}: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('/<int:integration_id>/status', methods=['GET'])
@jwt_required()
def get_integration_status(integration_id: int):
    """
    Get detailed integration status

    ---
    tags:
      - integrations
    summary: Get integration status
    security:
      - BearerAuth: []
    parameters:
      - name: integration_id
        in: path
        required: true
        schema:
          type: integer
        description: Integration ID
    responses:
      200:
        description: Integration status retrieved successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                data:
                  $ref: '#/components/schemas/IntegrationStatus'
                message:
                  type: string
      401:
        description: Unauthorized
      404:
        description: Integration not found
      500:
        description: Internal server error
    """
    try:
        # Get integration status
        status_info = integration_manager.get_integration_status(integration_id)

        return jsonify({
            'success': True,
            'data': {
                'status': status_info.status.value,
                'last_check': status_info.last_check.isoformat() if status_info.last_check else None,
                'last_success': status_info.last_success.isoformat() if status_info.last_success else None,
                'last_error': status_info.last_error.isoformat() if status_info.last_error else None,
                'error_message': status_info.error_message,
                'response_time_ms': status_info.response_time_ms,
                'total_requests': status_info.total_requests,
                'successful_requests': status_info.successful_requests,
                'failed_requests': status_info.failed_requests,
                'uptime_percentage': status_info.uptime_percentage,
                'metadata': status_info.metadata
            },
            'message': 'Integration status retrieved successfully'
        }), 200

    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error getting integration status {integration_id}: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('/<int:integration_id>/activate', methods=['POST'])
@jwt_required()
def activate_integration(integration_id: int):
    """
    Activate an integration

    ---
    tags:
      - integrations
    summary: Activate integration
    security:
      - BearerAuth: []
    parameters:
      - name: integration_id
        in: path
        required: true
        schema:
          type: integer
        description: Integration ID
    responses:
      200:
        description: Integration activated successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                message:
                  type: string
      401:
        description: Unauthorized
      404:
        description: Integration not found
      500:
        description: Internal server error
    """
    try:
        # Activate integration
        success = integration_manager.activate_integration(integration_id)

        if success:
            return jsonify({
                'success': True,
                'message': 'Integration activated successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to activate integration'
            }), 400

    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error activating integration {integration_id}: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('/<int:integration_id>/deactivate', methods=['POST'])
@jwt_required()
def deactivate_integration(integration_id: int):
    """
    Deactivate an integration

    ---
    tags:
      - integrations
    summary: Deactivate integration
    security:
      - BearerAuth: []
    parameters:
      - name: integration_id
        in: path
        required: true
        schema:
          type: integer
        description: Integration ID
    responses:
      200:
        description: Integration deactivated successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                message:
                  type: string
      401:
        description: Unauthorized
      404:
        description: Integration not found
      500:
        description: Internal server error
    """
    try:
        # Deactivate integration
        success = integration_manager.deactivate_integration(integration_id)

        if success:
            return jsonify({
                'success': True,
                'message': 'Integration deactivated successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to deactivate integration'
            }), 400

    except IntegrationError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error deactivating integration {integration_id}: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@integration_bp.route('/types', methods=['GET'])
def get_integration_types():
    """
    Get available integration types and providers

    ---
    tags:
      - integrations
    summary: Get integration types
    responses:
      200:
        description: Integration types retrieved successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                data:
                  type: object
                  properties:
                    cloud_storage:
                      type: array
                      items:
                        type: string
                    notification:
                      type: array
                      items:
                        type: string
                    database:
                      type: array
                      items:
                        type: string
                    api:
                      type: array
                      items:
                        type: string
                    file_processing:
                      type: array
                      items:
                        type: string
                    webhook:
                      type: array
                      items:
                        type: string
                message:
                  type: string
    """
    try:
        from models.integration import (
            CloudStorageProvider, NotificationProvider,
            DatabaseProvider, APIProtocol
        )

        data = {
            'cloud_storage': [p.value for p in CloudStorageProvider],
            'notification': [p.value for p in NotificationProvider],
            'database': [p.value for p in DatabaseProvider],
            'api': [p.value for p in APIProtocol],
            'file_processing': ['csv', 'json', 'xml', 'pdf', 'docx'],
            'webhook': ['generic', 'github', 'gitlab', 'slack']
        }

        return jsonify({
            'success': True,
            'data': data,
            'message': 'Integration types retrieved successfully'
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting integration types: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


def _format_integration_response(integration: IntegrationDB) -> Dict[str, Any]:
    """Format integration database model for API response"""
    return {
        'id': integration.id,
        'name': integration.name,
        'description': integration.description,
        'integration_type': integration.integration_type.value,
        'provider': integration.provider,
        'settings': integration.settings,
        'rate_limit': integration.rate_limit,
        'timeout': integration.timeout,
        'retry_attempts': integration.retry_attempts,
        'retry_delay': integration.retry_delay,
        'is_active': integration.is_active,
        'tags': integration.tags,
        'metadata': integration.metadata,
        'status_info': integration.status_info,
        'created_at': integration.created_at.isoformat() if integration.created_at else None,
        'updated_at': integration.updated_at.isoformat() if integration.updated_at else None,
        'created_by': integration.created_by,
        'organization_id': integration.organization_id
    }