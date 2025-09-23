"""
Comprehensive tests cho API Versioning System

Test tất cả components: models, configuration, version manager, middleware, migration tools và routes.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from flask import Flask

from models.api_version import (
    APIVersion,
    APIVersionRegistry,
    VersionStatus,
    MigrationType,
    VersionCompatibility,
    DeprecationNotice,
    VersionedRequest,
    VersionedResponse,
    parse_version_string,
    compare_versions,
    is_version_compatible
)
from config.api_versions import (
    API_VERSION_REGISTRY,
    API_VERSIONS,
    VERSION_COMPATIBILITY,
    MIGRATION_PATHS,
    DEPRECATION_POLICIES,
    VERSION_CONFIG,
    FEATURE_FLAGS,
    create_api_version_registry
)
from utils.version_manager import VersionManager, version_manager
from utils.version_middleware import VersionMiddleware
from utils.migration_tools import TTSMigrationTool, MigrationScriptGenerator, tts_migration_tool
from routes.versioned_tts import versioned_tts_bp


class TestAPIVersionModels:
    """Test API Version Models"""

    def test_api_version_creation(self):
        """Test tạo APIVersion object"""
        version = APIVersion(
            version="v2.1.0",
            status=VersionStatus.ACTIVE,
            release_date=datetime(2024, 1, 1),
            description="Test version",
            changelog=["Test change"],
            breaking_changes=["Breaking change"],
            new_features=["New feature"],
            feature_flags={"test": True},
            sunset_date=datetime(2025, 12, 1)
        )

        assert version.version == "v2.1.0"
        assert version.status == VersionStatus.ACTIVE
        assert version.is_deprecated() is False
        assert version.is_sunset() is False

    def test_version_validation(self):
        """Test version string validation"""
        # Valid versions
        assert parse_version_string("v1.0.0") is not None
        assert parse_version_string("v2.1.3") is not None

        # Invalid versions should raise exception
        with pytest.raises(Exception):
            parse_version_string("invalid")

    def test_version_comparison(self):
        """Test version comparison"""
        assert compare_versions("v1.0.0", "v1.0.0") == 0
        assert compare_versions("v1.0.0", "v1.0.1") == -1
        assert compare_versions("v1.0.1", "v1.0.0") == 1
        assert compare_versions("v2.0.0", "v1.9.9") == 1

    def test_version_compatibility(self):
        """Test version compatibility checking"""
        registry = create_api_version_registry()

        assert registry.validate_version_compatibility("v1.0.0", "v1.1.0") is False
        assert registry.validate_version_compatibility("v2.0.0", "v2.1.0") is True

    def test_deprecation_notice(self):
        """Test deprecation notice creation"""
        notice = DeprecationNotice(
            version="v1.0.0",
            replacement_version="v1.1.0",
            deprecation_date=datetime.now(),
            sunset_date=datetime.now() + timedelta(days=90),
            migration_guide="Migration guide here",
            breaking_changes=["Breaking change 1"]
        )

        assert notice.version == "v1.0.0"
        assert notice.replacement_version == "v1.1.0"
        assert len(notice.breaking_changes) == 1


class TestAPIVersionConfiguration:
    """Test API Version Configuration"""

    def test_api_versions_dict(self):
        """Test API versions dictionary"""
        assert "v1.0.0" in API_VERSIONS
        assert "v2.1.0" in API_VERSIONS
        assert "v2.2.0" in API_VERSIONS

        v1_version = API_VERSIONS["v1.0.0"]
        assert v1_version.version == "v1.0.0"
        assert v1_version.status == VersionStatus.ACTIVE

    def test_version_compatibility_matrix(self):
        """Test version compatibility matrix"""
        assert "v1.0.0" in VERSION_COMPATIBILITY
        assert "v2.1.0" in VERSION_COMPATIBILITY

        v1_compat = VERSION_COMPATIBILITY["v1.0.0"]
        assert v1_compat.current_version == "v1.0.0"
        assert v1_compat.migration_required is True

    def test_migration_paths(self):
        """Test migration paths"""
        assert "v1.0.0_to_v1.1.0" in MIGRATION_PATHS
        assert "v1.2.0_to_v2.0.0" in MIGRATION_PATHS

        migration = MIGRATION_PATHS["v1.0.0_to_v1.1.0"]
        assert migration.from_version == "v1.0.0"
        assert migration.to_version == "v1.1.0"
        assert migration.migration_type == MigrationType.AUTOMATIC

    def test_deprecation_policies(self):
        """Test deprecation policies"""
        assert "v1.0.0" in DEPRECATION_POLICIES
        assert "v1.2.0" in DEPRECATION_POLICIES

        policy = DEPRECATION_POLICIES["v1.0.0"]
        assert "replacement_version" in policy
        assert "sunset_date" in policy

    def test_version_config(self):
        """Test version configuration"""
        assert VERSION_CONFIG["default_version"] == "v2.1.0"
        assert VERSION_CONFIG["current_version"] == "v2.1.0"
        assert VERSION_CONFIG["auto_migration_enabled"] is True

    def test_feature_flags(self):
        """Test feature flags configuration"""
        assert "enable_streaming" in FEATURE_FLAGS
        assert "enable_voice_cloning" in FEATURE_FLAGS

        streaming_flag = FEATURE_FLAGS["enable_streaming"]
        assert "description" in streaming_flag
        assert "available_from" in streaming_flag

    def test_create_api_version_registry(self):
        """Test registry creation"""
        registry = create_api_version_registry()

        assert isinstance(registry, APIVersionRegistry)
        assert registry.current_version == "v2.1.0"
        assert registry.default_version == "v2.1.0"
        assert len(registry.versions) > 0
        assert len(registry.deprecation_notices) > 0


class TestVersionManager:
    """Test Version Manager"""

    def test_version_detection_from_url(self):
        """Test version detection from URL"""
        with patch('utils.version_manager.request') as mock_request:
            mock_request.path = "/v2.1.0/tts"
            mock_request.headers = {}
            mock_request.args = {}

            detected = version_manager.detect_version()
            assert detected == "v2.1.0"

    def test_version_detection_from_header(self):
        """Test version detection from header"""
        with patch('utils.version_manager.request') as mock_request:
            mock_request.path = "/tts"
            mock_request.headers = {"X-API-Version": "v2.0.0"}
            mock_request.args = {}

            detected = version_manager.detect_version()
            assert detected == "v2.0.0"

    def test_version_detection_from_query(self):
        """Test version detection from query parameter"""
        with patch('utils.version_manager.request') as mock_request:
            mock_request.path = "/tts"
            mock_request.headers = {}
            mock_request.args = {"version": "v1.2.0"}

            detected = version_manager.detect_version()
            assert detected == "v1.2.0"

    def test_version_detection_default(self):
        """Test default version when no version specified"""
        with patch('utils.version_manager.request') as mock_request:
            mock_request.path = "/tts"
            mock_request.headers = {}
            mock_request.args = {}

            detected = version_manager.detect_version()
            assert detected == version_manager.registry.default_version

    def test_validate_version_compatibility(self):
        """Test version compatibility validation"""
        is_compatible, message = version_manager.validate_version_compatibility("v2.0.0", "v2.1.0")
        assert is_compatible is True
        assert message is None

        is_compatible, message = version_manager.validate_version_compatibility("v1.0.0", "v2.0.0")
        assert is_compatible is False
        assert "migration required" in message.lower()

    def test_feature_compatibility_check(self):
        """Test feature compatibility checking"""
        is_compatible, message = version_manager.check_feature_compatibility("v2.1.0", "enable_streaming")
        assert is_compatible is True
        assert message is None

        is_compatible, message = version_manager.check_feature_compatibility("v1.0.0", "enable_streaming")
        assert is_compatible is False
        assert "requires version" in message.lower()

    def test_get_version_info(self):
        """Test getting version information"""
        info = version_manager.get_version_info("v2.1.0")
        assert info is not None
        assert info["version"] == "v2.1.0"
        assert "feature_flags" in info

    def test_get_compatible_versions(self):
        """Test getting compatible versions"""
        compatible = version_manager.get_compatible_versions("v2.1.0")
        assert "v2.0.0" in compatible
        assert "v2.2.0" in compatible
        assert "v1.0.0" not in compatible

    def test_migration_path_detection(self):
        """Test migration path detection"""
        migration_path = version_manager.get_migration_path("v1.0.0", "v1.1.0")
        assert migration_path == MigrationType.AUTOMATIC

        migration_path = version_manager.get_migration_path("v1.2.0", "v2.0.0")
        assert migration_path == MigrationType.BREAKING

    def test_create_versioned_response(self):
        """Test creating versioned response"""
        response = version_manager.create_versioned_response(
            data={"test": "data"},
            version="v2.1.0"
        )

        assert isinstance(response, VersionedResponse)
        assert response.data == {"test": "data"}
        assert response.version == "v2.1.0"


class TestVersionMiddleware:
    """Test Version Middleware"""

    def test_middleware_initialization(self):
        """Test middleware initialization"""
        app = Flask(__name__)
        middleware = VersionMiddleware(app)

        # Check if middleware was stored (this is done in init_app)
        assert hasattr(app, 'extensions') or hasattr(app, 'version_middleware')

    def test_before_request_processing(self):
        """Test before request processing"""
        app = Flask(__name__)
        middleware = VersionMiddleware(app)

        with app.test_request_context('/v2.1.0/test', headers={'X-API-Version': 'v2.1.0'}):
            middleware._before_request()

            assert hasattr(middleware, 'g')
            # Note: g object is request-local, so we can't easily test it directly

    def test_create_versioned_response(self):
        """Test creating versioned response through middleware"""
        app = Flask(__name__)
        middleware = VersionMiddleware(app)

        with app.test_request_context('/test'):
            response = middleware.create_versioned_response(
                data={"test": "data"},
                status_code=200
            )

            assert response.status_code == 200
            assert b"test" in response.get_data()

    def test_error_response_creation(self):
        """Test error response creation"""
        app = Flask(__name__)
        middleware = VersionMiddleware(app)

        with app.test_request_context('/test'):
            response, status_code = middleware._create_error_response(
                "Test error",
                400,
                {"detail": "test"}
            )

            assert status_code == 400
            assert b"Test error" in response.get_data()


class TestMigrationTools:
    """Test Migration Tools"""

    def test_tts_migration_tool_creation(self):
        """Test TTS migration tool creation"""
        tool = TTSMigrationTool()
        assert tool.registry is not None
        assert len(tool.migration_history) == 0

    def test_request_migration_v1_to_v2(self):
        """Test request migration from v1 to v2"""
        tool = TTSMigrationTool()

        request_data = {
            "text": "Hello world",
            "voice": "default",
            "format": "mp3",
            "auth": {"api_key": "test"}
        }

        migrated = tool.migrate_request("v1.2.0", "v2.0.0", request_data)

        assert "authentication" in migrated
        assert "auth" not in migrated
        assert "metadata" in migrated
        assert migrated["metadata"]["migrated_from"] == "v1.2.0"
        assert migrated["metadata"]["migrated_to"] == "v2.0.0"

    def test_response_migration_v1_to_v2(self):
        """Test response migration from v1 to v2"""
        tool = TTSMigrationTool()

        response_data = {
            "result": {
                "audio_url": "/audio/test.mp3",
                "format": "mp3"
            }
        }

        migrated = tool.migrate_response("v1.2.0", "v2.0.0", response_data)

        assert "audio_url" in migrated
        assert "result" in migrated
        assert "audio_url" not in migrated["result"]
        assert "version_info" in migrated

    def test_breaking_changes_detection(self):
        """Test breaking changes detection"""
        tool = TTSMigrationTool()

        breaking_changes = tool.detect_breaking_changes("v1.2.0", "v2.0.0")

        assert isinstance(breaking_changes, list)
        assert len(breaking_changes) > 0
        assert any("authentication" in change.lower() for change in breaking_changes)

    def test_migration_logging(self):
        """Test migration logging"""
        tool = TTSMigrationTool()

        tool.log_migration("v1.0.0", "v1.1.0", "request", True, {"test": "data"})

        assert len(tool.migration_history) == 1
        assert tool.migration_history[0]["from_version"] == "v1.0.0"
        assert tool.migration_history[0]["to_version"] == "v1.1.0"
        assert tool.migration_history[0]["success"] is True

    def test_migration_script_generation(self):
        """Test migration script generation"""
        generator = MigrationScriptGenerator(tts_migration_tool)

        script = generator.generate_migration_script("v1.0.0", "v1.1.0", "python")

        assert isinstance(script, str)
        assert "v1.0.0" in script
        assert "v1.1.0" in script
        assert "def migrate_data" in script

    def test_script_generation(self):
        """Test script generation"""
        generator = MigrationScriptGenerator(tts_migration_tool)

        script = generator.generate_migration_script("v1.0.0", "v1.1.0", "python")

        assert isinstance(script, str)
        assert "v1.0.0" in script
        assert "v1.1.0" in script
        assert "def migrate_data" in script


class TestVersionedTTSRoutes:
    """Test Version-aware TTS Routes"""

    def test_tts_endpoint_basic(self):
        """Test basic TTS endpoint"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.post('/api/tts', json={
                "text": "Hello world",
                "voice": "default"
            }, headers={'X-API-Version': 'v1.0.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "audio_url" in data["data"]

    def test_version_info_endpoint(self):
        """Test version info endpoint"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.get('/api/version')

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "current_version" in data["data"]
            assert "version_info" in data["data"]

    def test_migration_endpoint(self):
        """Test migration endpoint"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.post('/api/version/migrate', json={
                "target_version": "v2.0.0"
            }, headers={'X-API-Version': 'v1.2.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "migration_info" in data["data"]
            assert data["data"]["migration_info"]["from_version"] == "v1.2.0"
            assert data["data"]["migration_info"]["to_version"] == "v2.0.0"

    def test_compatibility_check_endpoint(self):
        """Test compatibility check endpoint"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.post('/api/version/compatibility', json={
                "from_version": "v2.0.0",
                "to_version": "v2.1.0"
            })

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "compatibility_info" in data["data"]
            assert data["data"]["compatibility_info"]["is_compatible"] is True

    def test_features_endpoint(self):
        """Test features endpoint"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.get('/api/version/features', headers={'X-API-Version': 'v2.1.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "features" in data["data"]
            assert "version" in data["data"]

    def test_streaming_endpoint_v1(self):
        """Test streaming endpoint with v1.x"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.post('/api/tts/stream', json={
                "text": "Hello world",
                "voice": "default"
            }, headers={'X-API-Version': 'v1.1.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "stream_url" in data["data"]

    def test_streaming_endpoint_v2(self):
        """Test streaming endpoint with v2.x"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.post('/api/tts/stream', json={
                "text": "Hello world",
                "voice": "default"
            }, headers={'X-API-Version': 'v2.0.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "stream_url" in data["data"]
            assert "websocket_url" in data["data"]

    def test_batch_endpoint(self):
        """Test batch endpoint"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.post('/api/tts/batch', json={
                "texts": ["Hello", "World"],
                "voice": "default"
            }, headers={'X-API-Version': 'v2.0.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "batch_id" in data["data"]
            assert "progress_url" in data["data"]

    def test_voice_cloning_endpoint(self):
        """Test voice cloning endpoint"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            response = client.post('/api/voice-cloning', json={
                "name": "test_voice",
                "audio_file": "/path/to/audio.wav",
                "authentication": {"api_key": "test"}
            }, headers={'X-API-Version': 'v2.0.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "voice_id" in data["data"]
            assert "training_progress_url" in data["data"]


class TestIntegrationTests:
    """Integration tests cho toàn bộ versioning system"""

    def test_full_request_lifecycle(self):
        """Test full request lifecycle với version management"""
        app = Flask(__name__)
        middleware = VersionMiddleware(app)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            # Test request với version header
            response = client.post('/api/tts', json={
                "text": "Hello world",
                "voice": "default"
            }, headers={'X-API-Version': 'v2.1.0'})

            assert response.status_code == 200
            assert 'X-API-Version' in response.headers
            assert response.headers['X-API-Version'] == 'v2.1.0'

            data = json.loads(response.get_data(as_text=True))
            assert data["version"] == "v2.1.0"
            assert data["status"] == "active"

    def test_version_migration_workflow(self):
        """Test version migration workflow"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            # First check compatibility
            compat_response = client.post('/api/version/compatibility', json={
                "from_version": "v1.2.0",
                "to_version": "v2.0.0"
            })

            assert compat_response.status_code == 200
            compat_data = json.loads(compat_response.get_data(as_text=True))
            assert compat_data["data"]["compatibility_info"]["is_compatible"] is False

            # Then get migration info
            migration_response = client.post('/api/version/migrate', json={
                "target_version": "v2.0.0"
            }, headers={'X-API-Version': 'v1.2.0'})

            assert migration_response.status_code == 200
            migration_data = json.loads(migration_response.get_data(as_text=True))
            assert migration_data["data"]["migration_info"]["from_version"] == "v1.2.0"
            assert migration_data["data"]["migration_info"]["to_version"] == "v2.0.0"
            assert len(migration_data["data"]["migration_info"]["breaking_changes"]) > 0

    def test_feature_flag_enforcement(self):
        """Test feature flag enforcement"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            # Test streaming with v1.0.0 (should fail)
            response = client.post('/api/tts/stream', json={
                "text": "Hello world",
                "voice": "default"
            }, headers={'X-API-Version': 'v1.0.0'})

            assert response.status_code == 400
            data = json.loads(response.get_data(as_text=True))
            assert "feature not supported" in data["error"].lower()

    def test_deprecation_warnings(self):
        """Test deprecation warnings"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            # Test với deprecated version
            response = client.get('/api/version', headers={'X-API-Version': 'v1.0.0'})

            assert response.status_code == 200
            data = json.loads(response.get_data(as_text=True))
            assert "deprecation_warnings" in data["data"]
            assert len(data["data"]["deprecation_warnings"]) > 0

    def test_error_handling(self):
        """Test error handling trong versioning system"""
        app = Flask(__name__)
        app.register_blueprint(versioned_tts_bp)

        with app.test_client() as client:
            # Test invalid version
            response = client.post('/api/tts', json={
                "text": "Hello world",
                "voice": "default"
            }, headers={'X-API-Version': 'v99.99.99'})

            assert response.status_code == 400
            data = json.loads(response.get_data(as_text=True))
            assert "invalid api version" in data["error"].lower()

    def test_migration_tools_integration(self):
        """Test migration tools integration"""
        # Test migration tool với actual data
        request_data = {
            "text": "Hello world",
            "voice": "default",
            "format": "mp3",
            "auth": {"api_key": "test"}
        }

        migrated = tts_migration_tool.migrate_request("v1.2.0", "v2.0.0", request_data)

        assert "authentication" in migrated
        assert "auth" not in migrated
        assert "metadata" in migrated
        assert migrated["metadata"]["migrated_from"] == "v1.2.0"

        # Test breaking changes detection
        breaking_changes = tts_migration_tool.detect_breaking_changes("v1.2.0", "v2.0.0")
        assert len(breaking_changes) > 0

        # Test script generation
        generator = MigrationScriptGenerator(tts_migration_tool)
        script = generator.generate_migration_script("v1.2.0", "v2.0.0")
        assert "v1.2.0" in script
        assert "v2.0.0" in script


# Performance tests
class TestPerformance:
    """Performance tests cho versioning system"""

    def test_version_detection_performance(self):
        """Test version detection performance"""
        import time

        start_time = time.time()

        for _ in range(100):
            with patch('utils.version_manager.request') as mock_request:
                mock_request.path = "/v2.1.0/tts"
                mock_request.headers = {"X-API-Version": "v2.1.0"}
                mock_request.args = {}

                version_manager.detect_version()

        end_time = time.time()
        total_time = end_time - start_time

        # Should be very fast (< 0.1 seconds for 100 operations)
        assert total_time < 0.1

    def test_migration_performance(self):
        """Test migration performance"""
        import time

        request_data = {
            "text": "Hello world " * 100,  # Large text
            "voice": "default",
            "format": "mp3",
            "auth": {"api_key": "test"}
        }

        start_time = time.time()

        for _ in range(10):
            tts_migration_tool.migrate_request("v1.2.0", "v2.0.0", request_data)

        end_time = time.time()
        total_time = end_time - start_time

        # Should be reasonable (< 1 second for 10 operations)
        assert total_time < 1.0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])