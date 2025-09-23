"""
Plugin loading tests for TTS system
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from utils.plugin_manager import PluginManager, PluginLoadResult
from utils.plugin_interface import PluginBase, TTSPlugin, AudioEnhancementPlugin, WebhookPlugin
from utils.plugin_security import SecurityViolation, SecurityThreat
from models.plugin import Plugin, PluginStatus, PluginType


class MockTTSPlugin(TTSPlugin):
    """Mock TTS plugin for testing."""

    def __init__(self):
        super().__init__("test_tts", "1.0.0", "Test TTS Plugin")

    def get_plugin_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'type': PluginType.TTS,
            'supported_languages': ['vi', 'en'],
            'supported_voices': ['male', 'female']
        }

    async def initialize(self, config=None):
        return True

    async def cleanup(self):
        pass

    async def synthesize(self, text, voice="default", language="vi", options=None):
        return {"success": True, "audio_data": b"mock_audio_data"}

    async def get_available_voices(self):
        return [{"id": "male", "name": "Male Voice"}]

    async def get_available_languages(self):
        return ["vi", "en"]


class MockAudioEnhancementPlugin(AudioEnhancementPlugin):
    """Mock audio enhancement plugin for testing."""

    def __init__(self):
        super().__init__("test_audio_enhancement", "1.0.0", "Test Audio Enhancement Plugin")

    def get_plugin_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'type': PluginType.AUDIO_ENHANCEMENT,
            'supported_formats': ['mp3', 'wav'],
            'enhancement_types': ['noise_reduction']
        }

    async def initialize(self, config=None):
        return True

    async def cleanup(self):
        pass

    async def enhance(self, audio_data, enhancement_type, parameters=None):
        return b"enhanced_" + audio_data

    async def get_enhancement_types(self):
        return ["noise_reduction"]


class MockWebhookPlugin(WebhookPlugin):
    """Mock webhook plugin for testing."""

    def __init__(self):
        super().__init__("test_webhook", "1.0.0", "Test Webhook Plugin")

    def get_plugin_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'type': PluginType.WEBHOOK,
            'supported_events': ['test.event']
        }

    async def initialize(self, config=None):
        return True

    async def cleanup(self):
        pass

    async def send_webhook(self, event, data, headers=None):
        return True

    async def get_supported_events(self):
        return ["test.event"]


@pytest.fixture
async def plugin_manager():
    """Create plugin manager for testing."""
    manager = PluginManager()
    await manager.initialize()
    yield manager
    await manager.cleanup()


@pytest.fixture
def sample_plugin_code():
    """Sample plugin code for testing."""
    return '''
from utils.plugin_interface import TTSPlugin

class TestPlugin(TTSPlugin):
    def __init__(self):
        super().__init__("test_plugin", "1.0.0", "Test Plugin")

    def get_plugin_info(self):
        return {"name": self.name, "version": self.version}

    async def initialize(self, config=None):
        return True

    async def cleanup(self):
        pass

    async def synthesize(self, text, voice="default", language="vi", options=None):
        return {"success": True, "audio_data": b"test_audio"}

    async def get_available_voices(self):
        return []

    async def get_available_languages(self):
        return ["vi"]
'''


class TestPluginLoading:
    """Test plugin loading functionality."""

    async def test_load_plugin_from_instance(self, plugin_manager):
        """Test loading plugin from instance."""
        plugin = MockTTSPlugin()

        result = await plugin_manager.load_plugin(plugin)

        assert result.success == True
        assert result.plugin == plugin
        assert plugin_manager.is_plugin_active("test_tts")

    async def test_load_plugin_from_file(self, plugin_manager, sample_plugin_code):
        """Test loading plugin from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_plugin_code)
            f.flush()

            try:
                result = await plugin_manager.load_plugin(f.name)

                assert result.success == True
                assert result.plugin is not None
                assert "test_plugin" in plugin_manager.get_active_plugins()

            finally:
                os.unlink(f.name)

    async def test_load_plugin_with_dependencies(self, plugin_manager):
        """Test loading plugin with dependencies."""
        # First load a dependency plugin
        dep_plugin = MockTTSPlugin()
        dep_plugin.name = "dependency_plugin"

        result1 = await plugin_manager.load_plugin(dep_plugin)
        assert result1.success == True

        # Then load plugin that depends on it
        plugin = MockTTSPlugin()
        plugin.name = "dependent_plugin"
        plugin.dependencies = ["dependency_plugin"]

        result2 = await plugin_manager.load_plugin(plugin)
        assert result2.success == True
        assert "dependency_plugin" in result2.dependencies_resolved

    async def test_load_plugin_with_security_violations(self, plugin_manager):
        """Test loading plugin with security violations."""
        plugin = MockTTSPlugin()

        # Mock security violations
        with patch('utils.plugin_security.security_manager.validate_and_sandbox_plugin') as mock_validate:
            mock_validate.return_value = [
                SecurityViolation(
                    threat_type=SecurityThreat.MALICIOUS_CODE,
                    severity="high",
                    message="Dangerous code detected",
                    plugin_name="test_plugin"
                )
            ]

            result = await plugin_manager.load_plugin(plugin)

            assert result.success == False
            assert len(result.security_violations) == 1
            assert result.security_violations[0].severity == "high"

    async def test_load_plugin_initialization_failure(self, plugin_manager):
        """Test loading plugin that fails to initialize."""
        plugin = MockTTSPlugin()

        # Mock initialization failure
        with patch.object(plugin, 'initialize', return_value=False):
            result = await plugin_manager.load_plugin(plugin)

            assert result.success == False
            assert "initialization failed" in result.error_message.lower()

    async def test_load_plugin_with_invalid_file(self, plugin_manager):
        """Test loading plugin from invalid file."""
        result = await plugin_manager.load_plugin("/nonexistent/path/plugin.py")

        assert result.success == False
        assert "failed to load" in result.error_message.lower()

    async def test_load_plugin_directory(self, plugin_manager, sample_plugin_code):
        """Test loading plugins from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create plugin file
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_file.write_text(sample_plugin_code)

            results = await plugin_manager.load_all_plugins(temp_dir)

            assert "test_plugin" in results
            assert results["test_plugin"].success == True


class TestPluginLifecycle:
    """Test plugin lifecycle management."""

    async def test_plugin_enable_disable(self, plugin_manager):
        """Test enabling and disabling plugins."""
        plugin = MockTTSPlugin()

        # Load plugin
        result = await plugin_manager.load_plugin(plugin)
        assert result.success == True

        # Disable plugin
        success = await plugin_manager.disable_plugin("test_tts")
        assert success == True
        assert not plugin_manager.is_plugin_active("test_tts")

        # Enable plugin
        success = await plugin_manager.enable_plugin("test_tts")
        assert success == True
        assert plugin_manager.is_plugin_active("test_tts")

    async def test_plugin_reload(self, plugin_manager):
        """Test reloading plugin."""
        plugin = MockTTSPlugin()

        # Load plugin
        result = await plugin_manager.load_plugin(plugin)
        assert result.success == True

        # Reload plugin
        success = await plugin_manager.reload_plugin("test_tts")
        assert success == True
        assert plugin_manager.is_plugin_active("test_tts")

    async def test_plugin_unload(self, plugin_manager):
        """Test unloading plugin."""
        plugin = MockTTSPlugin()

        # Load plugin
        result = await plugin_manager.load_plugin(plugin)
        assert result.success == True

        # Unload plugin
        success = await plugin_manager.unload_plugin("test_tts")
        assert success == True
        assert not plugin_manager.is_plugin_active("test_tts")

    async def test_multiple_plugin_lifecycle(self, plugin_manager):
        """Test lifecycle of multiple plugins."""
        plugins = [
            MockTTSPlugin(),
            MockAudioEnhancementPlugin(),
            MockWebhookPlugin()
        ]

        # Load all plugins
        for plugin in plugins:
            result = await plugin_manager.load_plugin(plugin)
            assert result.success == True

        # Verify all are active
        active_plugins = plugin_manager.get_active_plugins()
        assert len(active_plugins) == 3

        # Disable one plugin
        await plugin_manager.disable_plugin("test_tts")
        assert not plugin_manager.is_plugin_active("test_tts")
        assert plugin_manager.is_plugin_active("test_audio_enhancement")
        assert plugin_manager.is_plugin_active("test_webhook")

        # Cleanup all plugins
        await plugin_manager.cleanup()
        assert len(plugin_manager.get_active_plugins()) == 0


class TestPluginInformation:
    """Test plugin information retrieval."""

    async def test_get_plugin_info(self, plugin_manager):
        """Test getting plugin information."""
        plugin = MockTTSPlugin()
        await plugin_manager.load_plugin(plugin)

        info = plugin_manager.get_plugin_info("test_tts")
        assert info is not None
        assert info['name'] == "test_tts"
        assert info['type'] == PluginType.TTS

    async def test_get_all_plugin_info(self, plugin_manager):
        """Test getting information for all plugins."""
        plugins = [MockTTSPlugin(), MockAudioEnhancementPlugin()]
        for plugin in plugins:
            await plugin_manager.load_plugin(plugin)

        all_info = plugin_manager.get_all_plugin_info()
        assert len(all_info) == 2
        assert "test_tts" in all_info
        assert "test_audio_enhancement" in all_info

    async def test_plugin_state_tracking(self, plugin_manager):
        """Test plugin state tracking."""
        plugin = MockTTSPlugin()
        await plugin_manager.load_plugin(plugin)

        state = plugin_manager.get_plugin_state("test_tts")
        assert state == PluginStatus.ACTIVE

        await plugin_manager.disable_plugin("test_tts")
        state = plugin_manager.get_plugin_state("test_tts")
        assert state == PluginStatus.DISABLED


class TestPluginStatistics:
    """Test plugin statistics."""

    async def test_plugin_manager_statistics(self, plugin_manager):
        """Test plugin manager statistics."""
        plugins = [MockTTSPlugin(), MockAudioEnhancementPlugin()]
        for plugin in plugins:
            await plugin_manager.load_plugin(plugin)

        stats = plugin_manager.get_statistics()

        assert stats['total_plugins'] == 2
        assert stats['active_plugins'] == 2
        assert stats['is_initialized'] == True

    async def test_plugin_performance_metrics(self, plugin_manager):
        """Test plugin performance metrics."""
        plugin = MockTTSPlugin()
        await plugin_manager.load_plugin(plugin)

        # Get initial metrics
        initial_metrics = plugin.get_performance_metrics()
        assert initial_metrics['execution_count'] == 0

        # Simulate some executions
        plugin._performance_metrics['execution_count'] = 5
        plugin._performance_metrics['total_execution_time'] = 2.5

        # Check updated metrics
        updated_metrics = plugin.get_performance_metrics()
        assert updated_metrics['execution_count'] == 5
        assert updated_metrics['total_execution_time'] == 2.5


class TestErrorHandling:
    """Test error handling in plugin loading."""

    async def test_load_corrupted_plugin(self, plugin_manager):
        """Test loading corrupted plugin file."""
        corrupted_code = "import os; eval('malicious code')"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(corrupted_code)
            f.flush()

            try:
                result = await plugin_manager.load_plugin(f.name)
                assert result.success == False
                assert "failed" in result.error_message.lower()

            finally:
                os.unlink(f.name)

    async def test_load_plugin_with_syntax_error(self, plugin_manager):
        """Test loading plugin with syntax error."""
        syntax_error_code = "class TestPlugin(TTSPlugin\n    def __init__(self):"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(syntax_error_code)
            f.flush()

            try:
                result = await plugin_manager.load_plugin(f.name)
                assert result.success == False
                assert "syntax" in result.error_message.lower()

            finally:
                os.unlink(f.name)

    async def test_load_plugin_with_import_error(self, plugin_manager):
        """Test loading plugin with import error."""
        import_error_code = '''
from nonexistent_module import something
class TestPlugin(TTSPlugin):
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(import_error_code)
            f.flush()

            try:
                result = await plugin_manager.load_plugin(f.name)
                assert result.success == False
                assert "import" in result.error_message.lower()

            finally:
                os.unlink(f.name)


class TestPluginValidation:
    """Test plugin validation."""

    async def test_plugin_with_invalid_dependencies(self, plugin_manager):
        """Test plugin with invalid dependencies."""
        plugin = MockTTSPlugin()
        plugin.dependencies = ["nonexistent_dependency"]

        result = await plugin_manager.load_plugin(plugin)

        assert result.success == True  # Plugin loads but dependency fails
        assert "nonexistent_dependency" in result.dependencies_failed

    async def test_plugin_with_circular_dependencies(self, plugin_manager):
        """Test plugin with circular dependencies."""
        # This would require more complex setup to test properly
        # For now, just test that dependency resolution doesn't crash
        plugin = MockTTSPlugin()
        plugin.dependencies = ["self_dependency"]

        result = await plugin_manager.load_plugin(plugin)
        assert result.success == True  # Should handle gracefully


class TestPluginHooksAndEvents:
    """Test plugin hooks and events."""

    async def test_plugin_hook_registration(self, plugin_manager):
        """Test plugin hook registration."""
        plugin = MockTTSPlugin()
        await plugin_manager.load_plugin(plugin)

        # Register a hook
        async def test_hook(context):
            return {"test": "hook_executed"}

        plugin.register_hook("pre_tts", test_hook)

        # Execute hook
        from utils.plugin_interface import HookContext, HookType
        context = HookContext(HookType.PRE_TTS, "test_tts", {"test": "data"})
        result_context = await plugin.execute_hook(HookType.PRE_TTS, context)

        assert result_context.data["test"] == "hook_executed"

    async def test_plugin_event_handling(self, plugin_manager):
        """Test plugin event handling."""
        plugin = MockTTSPlugin()
        await plugin_manager.load_plugin(plugin)

        # Register event handler
        async def test_handler(context):
            context.data["handled"] = True

        plugin.register_event_handler("custom", test_handler)

        # Emit event
        from utils.plugin_interface import EventContext, EventType
        context = EventContext(EventType.CUSTOM, "test_tts", {"test": "data"})
        await plugin.emit_event(EventType.CUSTOM, context)

        assert context.data["handled"] == True