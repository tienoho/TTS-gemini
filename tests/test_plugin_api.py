"""
Plugin API tests for TTS system
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock

from flask import Flask
from flask.testing import FlaskClient

from routes.plugin import plugin_bp
from utils.plugin_manager import PluginManager
from utils.plugin_security import PluginSecurityManager
from models import Plugin, PluginStatus, PluginType


@pytest.fixture
def app():
    """Create Flask app for testing."""
    app = Flask(__name__)
    app.register_blueprint(plugin_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def mock_plugin_manager():
    """Mock plugin manager."""
    with patch('routes.plugin.plugin_manager') as mock_pm:
        yield mock_pm


@pytest.fixture
def mock_security_manager():
    """Mock security manager."""
    with patch('routes.plugin.security_manager') as mock_sm:
        yield mock_sm


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    with patch('routes.plugin.db.session') as mock_session:
        yield mock_session


class TestPluginAPI:
    """Test plugin API endpoints."""

    def test_list_plugins_empty(self, client, mock_plugin_manager, mock_db_session):
        """Test listing plugins when no plugins exist."""
        # Mock database query
        mock_db_session.query.return_value.filter.return_value.count.return_value = 0
        mock_db_session.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = []

        response = client.get('/api/v1/plugins')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['plugins'] == []
        assert data['total'] == 0

    def test_list_plugins_with_data(self, client, mock_plugin_manager, mock_db_session):
        """Test listing plugins with data."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'display_name': 'Test Plugin',
            'status': 'active',
            'version': '1.0.0'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.count.return_value = 1
        mock_db_session.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = [mock_plugin]

        response = client.get('/api/v1/plugins')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['plugins']) == 1
        assert data['total'] == 1
        assert data['plugins'][0]['name'] == 'test_plugin'

    def test_list_plugins_with_filters(self, client, mock_plugin_manager, mock_db_session):
        """Test listing plugins with status and type filters."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'tts_plugin',
            'display_name': 'TTS Plugin',
            'status': 'active',
            'plugin_type': 'tts'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.count.return_value = 1
        mock_db_session.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = [mock_plugin]

        # Test with filters
        response = client.get('/api/v1/plugins?status=active&type=tts')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['plugins']) == 1

    def test_list_plugins_invalid_status(self, client, mock_plugin_manager, mock_db_session):
        """Test listing plugins with invalid status filter."""
        response = client.get('/api/v1/plugins?status=invalid_status')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid status' in data['message']

    def test_list_plugins_invalid_type(self, client, mock_plugin_manager, mock_db_session):
        """Test listing plugins with invalid type filter."""
        response = client.get('/api/v1/plugins?type=invalid_type')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid type' in data['message']

    def test_list_active_plugins(self, client, mock_plugin_manager, mock_db_session):
        """Test listing active plugins."""
        # Mock active plugins
        mock_plugin = Mock()
        mock_plugin.get_plugin_info.return_value = {
            'name': 'active_plugin',
            'version': '1.0.0',
            'status': 'active'
        }

        mock_plugin_manager.get_active_plugins.return_value = {
            'active_plugin': mock_plugin
        }

        response = client.get('/api/v1/plugins/active')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['plugins']) == 1
        assert data['plugins'][0]['name'] == 'active_plugin'

    def test_get_plugin_found(self, client, mock_plugin_manager, mock_db_session):
        """Test getting a specific plugin that exists."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'display_name': 'Test Plugin',
            'status': 'active'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin info
        mock_plugin_manager.get_plugin_info.return_value = {
            'runtime_info': 'additional_info'
        }

        response = client.get('/api/v1/plugins/test_plugin')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['name'] == 'test_plugin'
        assert 'runtime_info' in data

    def test_get_plugin_not_found(self, client, mock_plugin_manager, mock_db_session):
        """Test getting a plugin that doesn't exist."""
        # Mock database query returning None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        response = client.get('/api/v1/plugins/nonexistent_plugin')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['message']

    def test_install_plugin_success(self, client, mock_plugin_manager, mock_db_session):
        """Test installing a new plugin successfully."""
        # Mock database operations
        mock_plugin = Mock()
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None

        # Mock plugin creation
        with patch('routes.plugin.Plugin') as mock_plugin_class:
            mock_plugin_class.return_value = mock_plugin
            mock_plugin.to_dict.return_value = {
                'id': 1,
                'name': 'new_plugin',
                'display_name': 'New Plugin'
            }

            response = client.post('/api/v1/plugins', json={
                'name': 'new_plugin',
                'display_name': 'New Plugin',
                'description': 'A new plugin',
                'version': '1.0.0'
            })

            assert response.status_code == 201
            data = json.loads(response.data)
            assert data['message'] == 'Plugin installed successfully'
            assert data['plugin']['name'] == 'new_plugin'

    def test_install_plugin_missing_name(self, client, mock_plugin_manager, mock_db_session):
        """Test installing plugin without name."""
        response = client.post('/api/v1/plugins', json={
            'display_name': 'Plugin Without Name'
        })

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'name is required' in data['message']

    def test_install_plugin_missing_display_name(self, client, mock_plugin_manager, mock_db_session):
        """Test installing plugin without display name."""
        response = client.post('/api/v1/plugins', json={
            'name': 'plugin_without_display_name'
        })

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'display name is required' in data['message']

    def test_install_plugin_already_exists(self, client, mock_plugin_manager, mock_db_session):
        """Test installing plugin that already exists."""
        # Mock existing plugin
        mock_db_session.query.return_value.filter.return_value.first.return_value = Mock()

        response = client.post('/api/v1/plugins', json={
            'name': 'existing_plugin',
            'display_name': 'Existing Plugin'
        })

        assert response.status_code == 409
        data = json.loads(response.data)
        assert 'error' in data
        assert 'already exists' in data['message']

    def test_upload_plugin_file(self, client, mock_plugin_manager, mock_db_session):
        """Test uploading plugin file."""
        # Mock file upload
        with patch('routes.plugin.Path') as mock_path, \
             patch('routes.plugin.secure_filename') as mock_secure_filename:

            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.mkdir.return_value = None
            mock_path_instance.stat.return_value = Mock(st_size=1024)

            mock_secure_filename.return_value = 'test_plugin.py'

            # Create mock file
            mock_file = Mock()
            mock_file.filename = 'test_plugin.py'

            response = client.post('/api/v1/plugins/test_plugin/upload',
                                 data={'file': mock_file},
                                 content_type='multipart/form-data')

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['message'] == 'Plugin file uploaded successfully'
            assert 'file_path' in data
            assert data['file_size'] == 1024

    def test_upload_plugin_no_file(self, client, mock_plugin_manager, mock_db_session):
        """Test uploading plugin without file."""
        response = client.post('/api/v1/plugins/test_plugin/upload')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file provided' in data['message']

    def test_upload_plugin_invalid_extension(self, client, mock_plugin_manager, mock_db_session):
        """Test uploading plugin with invalid file extension."""
        mock_file = Mock()
        mock_file.filename = 'test_plugin.txt'

        response = client.post('/api/v1/plugins/test_plugin/upload',
                             data={'file': mock_file},
                             content_type='multipart/form-data')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Only Python files' in data['message']

    def test_load_plugin_success(self, client, mock_plugin_manager, mock_db_session):
        """Test loading plugin successfully."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'status': 'active'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_result = Mock()
        mock_result.success = True
        mock_result.load_time = 0.5
        mock_result.dependencies_resolved = ['dep1']
        mock_result.dependencies_failed = []
        mock_result.security_violations = []

        mock_plugin_manager.load_plugin.return_value = mock_result
        mock_plugin_manager.is_plugin_active.return_value = True

        response = client.post('/api/v1/plugins/test_plugin/load')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin loaded successfully'
        assert data['load_info']['load_time'] == 0.5

    def test_load_plugin_not_found(self, client, mock_plugin_manager, mock_db_session):
        """Test loading plugin that doesn't exist."""
        # Mock database query returning None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        response = client.post('/api/v1/plugins/nonexistent_plugin/load')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['message']

    def test_load_plugin_file_not_found(self, client, mock_plugin_manager, mock_db_session):
        """Test loading plugin when file doesn't exist."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager to return failure
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = 'Plugin file not found'

        mock_plugin_manager.load_plugin.return_value = mock_result

        response = client.post('/api/v1/plugins/test_plugin/load')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Plugin file not found' in data['message']

    def test_enable_plugin_success(self, client, mock_plugin_manager, mock_db_session):
        """Test enabling plugin successfully."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'status': 'active'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_plugin_manager.is_plugin_active.return_value = False
        mock_plugin_manager.enable_plugin.return_value = True

        response = client.post('/api/v1/plugins/test_plugin/enable')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin enabled successfully'

    def test_enable_plugin_already_active(self, client, mock_plugin_manager, mock_db_session):
        """Test enabling plugin that is already active."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'status': 'active'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_plugin_manager.is_plugin_active.return_value = True

        response = client.post('/api/v1/plugins/test_plugin/enable')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin is already enabled'

    def test_disable_plugin_success(self, client, mock_plugin_manager, mock_db_session):
        """Test disabling plugin successfully."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'status': 'disabled'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_plugin_manager.is_plugin_active.return_value = True
        mock_plugin_manager.disable_plugin.return_value = True

        response = client.post('/api/v1/plugins/test_plugin/disable')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin disabled successfully'

    def test_disable_plugin_already_disabled(self, client, mock_plugin_manager, mock_db_session):
        """Test disabling plugin that is already disabled."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'status': 'disabled'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_plugin_manager.is_plugin_active.return_value = False

        response = client.post('/api/v1/plugins/test_plugin/disable')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin is already disabled'

    def test_reload_plugin_success(self, client, mock_plugin_manager, mock_db_session):
        """Test reloading plugin successfully."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'status': 'active'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_plugin_manager.reload_plugin.return_value = True

        response = client.post('/api/v1/plugins/test_plugin/reload')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin reloaded successfully'

    def test_update_plugin_success(self, client, mock_plugin_manager, mock_db_session):
        """Test updating plugin successfully."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.to_dict.return_value = {
            'id': 1,
            'name': 'test_plugin',
            'display_name': 'Updated Plugin',
            'version': '1.1.0'
        }

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin
        mock_db_session.commit.return_value = None

        response = client.put('/api/v1/plugins/test_plugin', json={
            'display_name': 'Updated Plugin',
            'version': '1.1.0',
            'description': 'Updated description'
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin updated successfully'
        assert data['plugin']['display_name'] == 'Updated Plugin'

    def test_uninstall_plugin_success(self, client, mock_plugin_manager, mock_db_session):
        """Test uninstalling plugin successfully."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_plugin_manager.is_plugin_active.return_value = False
        mock_plugin_manager.disable_plugin.return_value = True

        # Mock file operations
        with patch('routes.plugin.Path') as mock_path, \
             patch('routes.plugin.shutil.rmtree') as mock_rmtree:

            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True

            response = client.delete('/api/v1/plugins/test_plugin')

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['message'] == 'Plugin uninstalled successfully'

    def test_get_plugin_config(self, client, mock_plugin_manager, mock_db_session):
        """Test getting plugin configuration."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.config_schema = {'type': 'object'}
        mock_plugin.default_config = {'key': 'value'}
        mock_plugin.current_config = {'key': 'current_value'}

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        response = client.get('/api/v1/plugins/test_plugin/config')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['plugin_name'] == 'test_plugin'
        assert data['current_config']['key'] == 'current_value'

    def test_update_plugin_config(self, client, mock_plugin_manager, mock_db_session):
        """Test updating plugin configuration."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.current_config = {'key': 'new_value'}

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin
        mock_db_session.commit.return_value = None

        response = client.put('/api/v1/plugins/test_plugin/config', json={
            'config': {'key': 'new_value'}
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Plugin configuration updated successfully'
        assert data['current_config']['key'] == 'new_value'

    def test_get_plugin_statistics(self, client, mock_plugin_manager, mock_db_session):
        """Test getting plugin statistics."""
        # Mock plugin data
        mock_plugin = Mock()
        mock_plugin.execution_count = 10
        mock_plugin.error_count = 2
        mock_plugin.last_executed_at = None
        mock_plugin.last_error_at = None
        mock_plugin.installed_at = None

        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_plugin

        # Mock plugin manager
        mock_plugin_manager.get_statistics.return_value = {
            'total_plugins': 1,
            'active_plugins': 1,
            'is_initialized': True
        }

        # Mock security manager
        mock_security_manager.get_security_violations.return_value = []

        response = client.get('/api/v1/plugins/test_plugin/statistics')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['plugin_name'] == 'test_plugin'
        assert data['database_stats']['execution_count'] == 10
        assert data['database_stats']['error_count'] == 2

    def test_get_all_plugin_statistics(self, client, mock_plugin_manager, mock_db_session):
        """Test getting statistics for all plugins."""
        # Mock database query
        mock_plugin1 = Mock()
        mock_plugin1.status = PluginStatus.ACTIVE

        mock_plugin2 = Mock()
        mock_plugin2.status = PluginStatus.DISABLED

        mock_db_session.query.return_value.all.return_value = [mock_plugin1, mock_plugin2]

        # Mock plugin manager
        mock_plugin_manager.get_statistics.return_value = {
            'total_plugins': 2,
            'active_plugins': 1,
            'is_initialized': True
        }

        # Mock security manager
        mock_security_manager.get_security_report.return_value = {
            'total_violations': 0,
            'violations_by_severity': {},
            'violations_by_type': {}
        }

        response = client.get('/api/v1/plugins/statistics')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total_plugins'] == 2
        assert data['active_plugins'] == 1
        assert data['disabled_plugins'] == 1
        assert data['error_plugins'] == 0


class TestErrorHandling:
    """Test error handling in plugin API."""

    def test_database_error_handling(self, client, mock_plugin_manager, mock_db_session):
        """Test handling database errors."""
        # Mock database error
        mock_db_session.query.side_effect = Exception("Database connection failed")

        response = client.get('/api/v1/plugins')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to list plugins' in data['message']

    def test_plugin_manager_error_handling(self, client, mock_plugin_manager, mock_db_session):
        """Test handling plugin manager errors."""
        # Mock plugin manager error
        mock_plugin_manager.load_plugin.side_effect = Exception("Plugin manager error")

        response = client.post('/api/v1/plugins/test_plugin/load')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to load plugin' in data['message']

    def test_invalid_json_handling(self, client, mock_plugin_manager, mock_db_session):
        """Test handling invalid JSON."""
        response = client.post('/api/v1/plugins',
                             data="invalid json",
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_missing_required_fields(self, client, mock_plugin_manager, mock_db_session):
        """Test handling missing required fields."""
        response = client.post('/api/v1/plugins', json={})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'name is required' in data['message']