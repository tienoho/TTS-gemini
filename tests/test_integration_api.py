"""
Tests for Integration API Routes

This module contains comprehensive tests for integration REST API endpoints,
including CRUD operations, authentication, validation, and error handling.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from flask import Flask
from flask_jwt_extended import JWTManager

from routes.integration import integration_bp
from models.integration import IntegrationConfig, IntegrationType, CloudStorageProvider
from utils.exceptions import IntegrationError, ValidationError


@pytest.fixture
def app():
    """Flask app fixture"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-secret-key'
    app.config['JWT_SECRET_KEY'] = 'test-jwt-secret'

    jwt = JWTManager(app)
    app.register_blueprint(integration_bp)

    return app


@pytest.fixture
def client(app):
    """Flask test client"""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Authentication headers"""
    return {'Authorization': 'Bearer test_token_123'}


@pytest.fixture
def mock_integration_manager():
    """Mock integration manager"""
    manager = Mock()
    manager.create_integration = AsyncMock()
    manager.list_integrations = AsyncMock()
    manager.update_integration = AsyncMock()
    manager.delete_integration = AsyncMock()
    manager.test_integration = AsyncMock()
    manager.get_integration_status = AsyncMock()
    manager.activate_integration = AsyncMock()
    manager.deactivate_integration = AsyncMock()
    return manager


@pytest.fixture
def mock_security_manager():
    """Mock security manager"""
    return Mock()


class TestIntegrationAPICreation:
    """Test integration creation API"""

    def test_create_integration_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration creation"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                # Mock successful creation
                mock_integration = Mock()
                mock_integration.id = 1
                mock_integration.name = "Test Integration"
                mock_integration.integration_type = IntegrationType.CLOUD_STORAGE
                mock_integration.provider = CloudStorageProvider.AWS_S3.value
                mock_integration.settings = {}
                mock_integration.rate_limit = 1000
                mock_integration.timeout = 30
                mock_integration.retry_attempts = 3
                mock_integration.retry_delay = 1
                mock_integration.is_active = True
                mock_integration.tags = ["test"]
                mock_integration.metadata = {}
                mock_integration.status_info = {"status": "active"}
                mock_integration.created_at = None
                mock_integration.updated_at = None
                mock_integration.created_by = 1
                mock_integration.organization_id = None

                mock_integration_manager.create_integration.return_value = mock_integration

                request_data = {
                    "name": "Test Integration",
                    "description": "Test integration for API testing",
                    "integration_type": "cloud_storage",
                    "provider": "aws_s3",
                    "credentials": {
                        "access_key": "AKIAIOSFODNN7EXAMPLE",
                        "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                        "region": "us-east-1"
                    },
                    "settings": {"bucket_name": "test-bucket"},
                    "rate_limit": 1000,
                    "timeout": 30,
                    "retry_attempts": 3,
                    "retry_delay": 1,
                    "is_active": True,
                    "tags": ["test", "aws"]
                }

                # Act
                response = client.post('/api/v1/integrations',
                                     data=json.dumps(request_data),
                                     content_type='application/json',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 201
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['data']['name'] == "Test Integration"
                assert response_data['data']['integration_type'] == "cloud_storage"
                assert response_data['data']['provider'] == "aws_s3"
                assert response_data['message'] == "Integration created successfully"

    def test_create_integration_validation_error(self, client, auth_headers, mock_integration_manager):
        """Test integration creation with validation error"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):

                request_data = {
                    "name": "",  # Invalid: empty name
                    "integration_type": "cloud_storage",
                    "provider": "aws_s3",
                    "credentials": {}
                }

                # Act
                response = client.post('/api/v1/integrations',
                                     data=json.dumps(request_data),
                                     content_type='application/json',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 400
                response_data = json.loads(response.data)
                assert response_data['success'] is False
                assert 'Validation error' in response_data['message']

    def test_create_integration_missing_body(self, client, auth_headers, mock_integration_manager):
        """Test integration creation with missing request body"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):

                # Act
                response = client.post('/api/v1/integrations',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 400
                response_data = json.loads(response.data)
                assert response_data['success'] is False
                assert 'Request body is required' in response_data['message']

    def test_create_integration_unauthorized(self, client, mock_integration_manager):
        """Test integration creation without authentication"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):

            request_data = {
                "name": "Test Integration",
                "integration_type": "cloud_storage",
                "provider": "aws_s3",
                "credentials": {}
            }

            # Act
            response = client.post('/api/v1/integrations',
                                 data=json.dumps(request_data),
                                 content_type='application/json')

            # Assert
            assert response.status_code == 401

    def test_create_integration_internal_error(self, client, auth_headers, mock_integration_manager):
        """Test integration creation with internal error"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.create_integration.side_effect = Exception("Database error")

                request_data = {
                    "name": "Test Integration",
                    "integration_type": "cloud_storage",
                    "provider": "aws_s3",
                    "credentials": {}
                }

                # Act
                response = client.post('/api/v1/integrations',
                                     data=json.dumps(request_data),
                                     content_type='application/json',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 500
                response_data = json.loads(response.data)
                assert response_data['success'] is False
                assert 'Internal server error' in response_data['message']


class TestIntegrationAPIListing:
    """Test integration listing API"""

    def test_list_integrations_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration listing"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                # Mock integrations
                mock_integration1 = Mock()
                mock_integration1.id = 1
                mock_integration1.name = "Integration 1"
                mock_integration1.integration_type = IntegrationType.CLOUD_STORAGE
                mock_integration1.provider = CloudStorageProvider.AWS_S3.value
                mock_integration1.settings = {}
                mock_integration1.rate_limit = 1000
                mock_integration1.timeout = 30
                mock_integration1.retry_attempts = 3
                mock_integration1.retry_delay = 1
                mock_integration1.is_active = True
                mock_integration1.tags = ["test"]
                mock_integration1.metadata = {}
                mock_integration1.status_info = {"status": "active"}
                mock_integration1.created_at = None
                mock_integration1.updated_at = None
                mock_integration1.created_by = 1
                mock_integration1.organization_id = None

                mock_integration2 = Mock()
                mock_integration2.id = 2
                mock_integration2.name = "Integration 2"
                mock_integration2.integration_type = IntegrationType.NOTIFICATION
                mock_integration2.provider = "slack"
                mock_integration2.settings = {}
                mock_integration2.rate_limit = 500
                mock_integration2.timeout = 15
                mock_integration2.retry_attempts = 2
                mock_integration2.retry_delay = 1
                mock_integration2.is_active = True
                mock_integration2.tags = ["notification"]
                mock_integration2.metadata = {}
                mock_integration2.status_info = {"status": "active"}
                mock_integration2.created_at = None
                mock_integration2.updated_at = None
                mock_integration2.created_by = 1
                mock_integration2.organization_id = None

                mock_integration_manager.list_integrations.return_value = [mock_integration1, mock_integration2]

                # Act
                response = client.get('/api/v1/integrations',
                                    headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert len(response_data['data']) == 2
                assert response_data['data'][0]['name'] == "Integration 1"
                assert response_data['data'][1]['name'] == "Integration 2"
                assert response_data['message'] == "Integrations retrieved successfully"

    def test_list_integrations_with_filters(self, client, auth_headers, mock_integration_manager):
        """Test integration listing with filters"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration = Mock()
                mock_integration.id = 1
                mock_integration.name = "AWS S3 Integration"
                mock_integration.integration_type = IntegrationType.CLOUD_STORAGE
                mock_integration.provider = CloudStorageProvider.AWS_S3.value
                mock_integration.settings = {}
                mock_integration.rate_limit = 1000
                mock_integration.timeout = 30
                mock_integration.retry_attempts = 3
                mock_integration.retry_delay = 1
                mock_integration.is_active = True
                mock_integration.tags = ["aws", "s3"]
                mock_integration.metadata = {}
                mock_integration.status_info = {"status": "active"}
                mock_integration.created_at = None
                mock_integration.updated_at = None
                mock_integration.created_by = 1
                mock_integration.organization_id = None

                mock_integration_manager.list_integrations.return_value = [mock_integration]

                # Act
                response = client.get('/api/v1/integrations?type=cloud_storage&provider=aws_s3',
                                    headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert len(response_data['data']) == 1
                assert response_data['data'][0]['integration_type'] == "cloud_storage"
                assert response_data['data'][0]['provider'] == "aws_s3"

    def test_list_integrations_with_pagination(self, client, auth_headers, mock_integration_manager):
        """Test integration listing with pagination"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                # Create multiple mock integrations
                mock_integrations = []
                for i in range(5):
                    mock_integration = Mock()
                    mock_integration.id = i + 1
                    mock_integration.name = f"Integration {i + 1}"
                    mock_integration.integration_type = IntegrationType.CLOUD_STORAGE
                    mock_integration.provider = CloudStorageProvider.AWS_S3.value
                    mock_integration.settings = {}
                    mock_integration.rate_limit = 1000
                    mock_integration.timeout = 30
                    mock_integration.retry_attempts = 3
                    mock_integration.retry_delay = 1
                    mock_integration.is_active = True
                    mock_integration.tags = ["test"]
                    mock_integration.metadata = {}
                    mock_integration.status_info = {"status": "active"}
                    mock_integration.created_at = None
                    mock_integration.updated_at = None
                    mock_integration.created_by = 1
                    mock_integration.organization_id = None
                    mock_integrations.append(mock_integration)

                mock_integration_manager.list_integrations.return_value = mock_integrations

                # Act
                response = client.get('/api/v1/integrations?page=1&per_page=2',
                                    headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert len(response_data['data']) == 2
                assert response_data['pagination']['page'] == 1
                assert response_data['pagination']['per_page'] == 2
                assert response_data['pagination']['total'] == 5
                assert response_data['pagination']['pages'] == 3


class TestIntegrationAPIUpdate:
    """Test integration update API"""

    def test_update_integration_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration update"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration = Mock()
                mock_integration.id = 1
                mock_integration.name = "Updated Integration"
                mock_integration.integration_type = IntegrationType.CLOUD_STORAGE
                mock_integration.provider = CloudStorageProvider.AWS_S3.value
                mock_integration.settings = {"bucket_name": "updated-bucket"}
                mock_integration.rate_limit = 2000
                mock_integration.timeout = 60
                mock_integration.retry_attempts = 5
                mock_integration.retry_delay = 2
                mock_integration.is_active = True
                mock_integration.tags = ["updated", "aws"]
                mock_integration.metadata = {}
                mock_integration.status_info = {"status": "active"}
                mock_integration.created_at = None
                mock_integration.updated_at = None
                mock_integration.created_by = 1
                mock_integration.organization_id = None

                mock_integration_manager.update_integration.return_value = mock_integration

                request_data = {
                    "name": "Updated Integration",
                    "description": "Updated integration",
                    "integration_type": "cloud_storage",
                    "provider": "aws_s3",
                    "credentials": {
                        "access_key": "AKIAIOSFODNN7EXAMPLE",
                        "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                        "region": "us-west-2"
                    },
                    "settings": {"bucket_name": "updated-bucket"},
                    "rate_limit": 2000,
                    "timeout": 60,
                    "retry_attempts": 5,
                    "retry_delay": 2,
                    "is_active": True,
                    "tags": ["updated", "aws"]
                }

                # Act
                response = client.put('/api/v1/integrations/1',
                                    data=json.dumps(request_data),
                                    content_type='application/json',
                                    headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['data']['name'] == "Updated Integration"
                assert response_data['data']['rate_limit'] == 2000
                assert response_data['message'] == "Integration updated successfully"

    def test_update_integration_not_found(self, client, auth_headers, mock_integration_manager):
        """Test integration update when integration not found"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.update_integration.side_effect = IntegrationError("Integration not found")

                request_data = {
                    "name": "Updated Integration",
                    "integration_type": "cloud_storage",
                    "provider": "aws_s3",
                    "credentials": {}
                }

                # Act
                response = client.put('/api/v1/integrations/999',
                                    data=json.dumps(request_data),
                                    content_type='application/json',
                                    headers=auth_headers)

                # Assert
                assert response.status_code == 400
                response_data = json.loads(response.data)
                assert response_data['success'] is False
                assert 'Integration not found' in response_data['message']


class TestIntegrationAPIDeletion:
    """Test integration deletion API"""

    def test_delete_integration_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration deletion"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.delete_integration.return_value = True

                # Act
                response = client.delete('/api/v1/integrations/1',
                                       headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['message'] == "Integration deleted successfully"

    def test_delete_integration_not_found(self, client, auth_headers, mock_integration_manager):
        """Test integration deletion when integration not found"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.delete_integration.side_effect = IntegrationError("Integration not found")

                # Act
                response = client.delete('/api/v1/integrations/999',
                                       headers=auth_headers)

                # Assert
                assert response.status_code == 400
                response_data = json.loads(response.data)
                assert response_data['success'] is False
                assert 'Integration not found' in response_data['message']


class TestIntegrationAPITesting:
    """Test integration testing API"""

    def test_test_integration_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration testing"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.test_integration.return_value = {
                    "success": True,
                    "message": "Test successful",
                    "response_time_ms": 150,
                    "details": {"service": "AWS S3"}
                }

                # Act
                response = client.post('/api/v1/integrations/1/test',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['data']['success'] is True
                assert response_data['data']['message'] == "Test successful"
                assert response_data['message'] == "Integration test completed"

    def test_test_integration_failure(self, client, auth_headers, mock_integration_manager):
        """Test integration testing failure"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.test_integration.side_effect = IntegrationError("Connection failed")

                # Act
                response = client.post('/api/v1/integrations/1/test',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 400
                response_data = json.loads(response.data)
                assert response_data['success'] is False
                assert 'Connection failed' in response_data['message']


class TestIntegrationAPIStatus:
    """Test integration status API"""

    def test_get_integration_status_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration status retrieval"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_status_info = Mock()
                mock_status_info.status = IntegrationType.CLOUD_STORAGE
                mock_status_info.last_check = None
                mock_status_info.last_success = None
                mock_status_info.last_error = None
                mock_status_info.error_message = None
                mock_status_info.response_time_ms = 150
                mock_status_info.total_requests = 100
                mock_status_info.successful_requests = 95
                mock_status_info.failed_requests = 5
                mock_status_info.uptime_percentage = 95.0
                mock_status_info.metadata = {}

                mock_integration_manager.get_integration_status.return_value = mock_status_info

                # Act
                response = client.get('/api/v1/integrations/1/status',
                                    headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['data']['total_requests'] == 100
                assert response_data['data']['successful_requests'] == 95
                assert response_data['data']['uptime_percentage'] == 95.0
                assert response_data['message'] == "Integration status retrieved successfully"


class TestIntegrationAPIActivation:
    """Test integration activation/deactivation API"""

    def test_activate_integration_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration activation"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.activate_integration.return_value = True

                # Act
                response = client.post('/api/v1/integrations/1/activate',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['message'] == "Integration activated successfully"

    def test_deactivate_integration_success(self, client, auth_headers, mock_integration_manager):
        """Test successful integration deactivation"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration_manager.deactivate_integration.return_value = True

                # Act
                response = client.post('/api/v1/integrations/1/deactivate',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 200
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['message'] == "Integration deactivated successfully"


class TestIntegrationAPIErrorHandling:
    """Test API error handling"""

    def test_invalid_json(self, client, auth_headers, mock_integration_manager):
        """Test API with invalid JSON"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):

                # Act
                response = client.post('/api/v1/integrations',
                                     data="invalid json",
                                     content_type='application/json',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 400

    def test_method_not_allowed(self, client, auth_headers):
        """Test unsupported HTTP method"""
        # Act
        response = client.patch('/api/v1/integrations',
                              headers=auth_headers)

        # Assert
        assert response.status_code == 405

    def test_invalid_integration_id(self, client, auth_headers, mock_integration_manager):
        """Test API with invalid integration ID"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):

                # Act
                response = client.get('/api/v1/integrations/invalid',
                                    headers=auth_headers)

                # Assert
                assert response.status_code == 404


class TestIntegrationAPIEdgeCases:
    """Test API edge cases"""

    def test_create_integration_with_minimal_data(self, client, auth_headers, mock_integration_manager):
        """Test integration creation with minimal required data"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration = Mock()
                mock_integration.id = 1
                mock_integration.name = "Minimal Integration"
                mock_integration.integration_type = IntegrationType.CLOUD_STORAGE
                mock_integration.provider = CloudStorageProvider.AWS_S3.value
                mock_integration.settings = {}
                mock_integration.rate_limit = None
                mock_integration.timeout = 30
                mock_integration.retry_attempts = 3
                mock_integration.retry_delay = 1
                mock_integration.is_active = True
                mock_integration.tags = []
                mock_integration.metadata = {}
                mock_integration.status_info = {"status": "active"}
                mock_integration.created_at = None
                mock_integration.updated_at = None
                mock_integration.created_by = 1
                mock_integration.organization_id = None

                mock_integration_manager.create_integration.return_value = mock_integration

                request_data = {
                    "name": "Minimal Integration",
                    "integration_type": "cloud_storage",
                    "provider": "aws_s3",
                    "credentials": {}
                }

                # Act
                response = client.post('/api/v1/integrations',
                                     data=json.dumps(request_data),
                                     content_type='application/json',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 201
                response_data = json.loads(response.data)
                assert response_data['success'] is True
                assert response_data['data']['name'] == "Minimal Integration"

    def test_large_request_data(self, client, auth_headers, mock_integration_manager):
        """Test API with large request data"""
        # Arrange
        with patch('routes.integration.integration_manager', mock_integration_manager):
            with patch('routes.integration.get_jwt_identity', return_value=1):
                mock_integration = Mock()
                mock_integration.id = 1
                mock_integration.name = "Large Data Integration"
                mock_integration.integration_type = IntegrationType.CLOUD_STORAGE
                mock_integration.provider = CloudStorageProvider.AWS_S3.value
                mock_integration.settings = {"large_field": "x" * 10000}
                mock_integration.rate_limit = 1000
                mock_integration.timeout = 30
                mock_integration.retry_attempts = 3
                mock_integration.retry_delay = 1
                mock_integration.is_active = True
                mock_integration.tags = ["large", "test"]
                mock_integration.metadata = {"large_metadata": "y" * 5000}
                mock_integration.status_info = {"status": "active"}
                mock_integration.created_at = None
                mock_integration.updated_at = None
                mock_integration.created_by = 1
                mock_integration.organization_id = None

                mock_integration_manager.create_integration.return_value = mock_integration

                # Create large request data
                large_settings = {f"field_{i}": f"value_{i}" * 100 for i in range(100)}
                request_data = {
                    "name": "Large Data Integration",
                    "integration_type": "cloud_storage",
                    "provider": "aws_s3",
                    "credentials": {
                        "access_key": "AKIAIOSFODNN7EXAMPLE",
                        "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
                    },
                    "settings": large_settings
                }

                # Act
                response = client.post('/api/v1/integrations',
                                     data=json.dumps(request_data),
                                     content_type='application/json',
                                     headers=auth_headers)

                # Assert
                assert response.status_code == 201
                response_data = json.loads(response.data)
                assert response_data['success'] is True