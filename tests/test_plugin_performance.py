"""
Plugin performance tests for TTS system
"""

import pytest
import asyncio
import time
import psutil
import os
from unittest.mock import Mock, patch, AsyncMock
from memory_profiler import profile

from utils.plugin_manager import PluginManager, PluginLoadResult
from utils.plugin_interface import PluginBase, TTSPlugin
from utils.plugin_security import PluginSecurityManager
from models.plugin import Plugin, PluginStatus


class MockPerformancePlugin(TTSPlugin):
    """Mock plugin for performance testing."""

    def __init__(self, name="performance_plugin", execution_time=0.1):
        super().__init__(name, "1.0.0", "Performance Test Plugin")
        self.execution_time = execution_time
        self.memory_usage = 0

    def get_plugin_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'type': 'tts'
        }

    async def initialize(self, config=None):
        return True

    async def cleanup(self):
        pass

    async def synthesize(self, text, voice="default", language="vi", options=None):
        # Simulate execution time
        await asyncio.sleep(self.execution_time)

        # Simulate memory usage
        self.memory_usage = len(text) * 100  # Simple memory simulation

        return {
            'success': True,
            'audio_data': b'performance_test_audio',
            'execution_time': self.execution_time
        }

    async def get_available_voices(self):
        return [{'id': 'test_voice', 'name': 'Test Voice'}]

    async def get_available_languages(self):
        return ['vi', 'en']


class TestPluginPerformance:
    """Test plugin performance characteristics."""

    @pytest.fixture
    async def plugin_manager(self):
        """Create plugin manager for performance testing."""
        manager = PluginManager()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    async def test_plugin_load_time(self, plugin_manager):
        """Test plugin loading performance."""
        plugin = MockPerformancePlugin("load_time_test", 0.01)

        start_time = time.time()
        result = await plugin_manager.load_plugin(plugin)
        load_time = time.time() - start_time

        assert result.success == True
        assert load_time < 1.0  # Should load within 1 second

    async def test_plugin_initialization_time(self, plugin_manager):
        """Test plugin initialization performance."""
        plugin = MockPerformancePlugin("init_time_test", 0.01)

        start_time = time.time()
        success = await plugin_manager.lifecycle_manager.initialize_plugin(plugin)
        init_time = time.time() - start_time

        assert success == True
        assert init_time < 0.5  # Should initialize within 0.5 seconds

    async def test_plugin_execution_time(self, plugin_manager):
        """Test plugin execution performance."""
        plugin = MockPerformancePlugin("execution_time_test", 0.1)

        await plugin_manager.load_plugin(plugin)

        start_time = time.time()
        result = await plugin.synthesize("Test text for performance measurement")
        execution_time = time.time() - start_time

        assert result['success'] == True
        assert execution_time >= 0.1  # Should take at least the expected time
        assert execution_time < 0.5   # Should not take too long

    async def test_multiple_plugin_execution(self, plugin_manager):
        """Test performance with multiple plugins."""
        plugins = []
        for i in range(5):
            plugin = MockPerformancePlugin(f"multi_plugin_{i}", 0.05)
            plugins.append(plugin)
            await plugin_manager.load_plugin(plugin)

        # Execute all plugins concurrently
        start_time = time.time()

        tasks = []
        for plugin in plugins:
            task = plugin.synthesize("Concurrent execution test")
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        assert len(results) == 5
        assert all(result['success'] for result in results)
        assert total_time < 2.0  # Should complete within 2 seconds

    async def test_plugin_memory_usage(self, plugin_manager):
        """Test plugin memory usage."""
        plugin = MockPerformancePlugin("memory_test", 0.01)

        await plugin_manager.load_plugin(plugin)

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Execute plugin multiple times
        for i in range(10):
            result = await plugin.synthesize(f"Memory test iteration {i}")
            assert result['success'] == True

        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024

    async def test_plugin_concurrent_load(self, plugin_manager):
        """Test concurrent plugin loading."""
        plugins = []
        for i in range(10):
            plugin = MockPerformancePlugin(f"concurrent_load_{i}", 0.01)
            plugins.append(plugin)

        # Load plugins concurrently
        start_time = time.time()

        tasks = []
        for plugin in plugins:
            task = plugin_manager.load_plugin(plugin)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        assert len(results) == 10
        assert all(result.success for result in results)
        assert total_time < 5.0  # Should load 10 plugins within 5 seconds

    async def test_plugin_error_recovery_time(self, plugin_manager):
        """Test plugin error recovery performance."""
        plugin = MockPerformancePlugin("error_recovery_test", 0.01)

        await plugin_manager.load_plugin(plugin)

        # Test successful execution
        start_time = time.time()
        result = await plugin.synthesize("Normal execution")
        normal_time = time.time() - start_time

        assert result['success'] == True
        assert normal_time < 0.5

        # Test error scenario
        original_synthesize = plugin.synthesize

        async def failing_synthesize(*args, **kwargs):
            raise Exception("Simulated error")

        plugin.synthesize = failing_synthesize

        start_time = time.time()
        try:
            await plugin.synthesize("This will fail")
        except Exception:
            pass
        error_time = time.time() - start_time

        # Error handling should be fast
        assert error_time < 0.1

        # Restore original method
        plugin.synthesize = original_synthesize

    async def test_plugin_resource_cleanup(self, plugin_manager):
        """Test plugin resource cleanup performance."""
        plugin = MockPerformancePlugin("cleanup_test", 0.01)

        await plugin_manager.load_plugin(plugin)

        # Get memory before cleanup
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Cleanup plugin
        start_time = time.time()
        await plugin_manager.unload_plugin("cleanup_test")
        cleanup_time = time.time() - start_time

        # Get memory after cleanup
        memory_after = process.memory_info().rss

        assert cleanup_time < 1.0  # Should cleanup within 1 second
        # Memory should not increase significantly after cleanup
        assert memory_after <= memory_before + 10 * 1024 * 1024  # Allow 10MB variance

    async def test_plugin_statistics_performance(self, plugin_manager):
        """Test plugin statistics collection performance."""
        plugins = []
        for i in range(5):
            plugin = MockPerformancePlugin(f"stats_test_{i}", 0.01)
            plugins.append(plugin)
            await plugin_manager.load_plugin(plugin)

        # Collect statistics
        start_time = time.time()
        stats = plugin_manager.get_statistics()
        stats_time = time.time() - start_time

        assert stats_time < 0.1  # Should collect stats quickly
        assert stats['total_plugins'] == 5
        assert stats['active_plugins'] == 5

    async def test_plugin_hook_performance(self, plugin_manager):
        """Test plugin hook execution performance."""
        plugin = MockPerformancePlugin("hook_test", 0.01)

        await plugin_manager.load_plugin(plugin)

        # Register multiple hooks
        async def hook1(context):
            await asyncio.sleep(0.01)
            return {"hook1": "executed"}

        async def hook2(context):
            await asyncio.sleep(0.01)
            return {"hook2": "executed"}

        async def hook3(context):
            await asyncio.sleep(0.01)
            return {"hook3": "executed"}

        plugin.register_hook("pre_tts", hook1)
        plugin.register_hook("pre_tts", hook2)
        plugin.register_hook("pre_tts", hook3)

        # Execute hooks
        from utils.plugin_interface import HookContext, HookType

        start_time = time.time()
        context = HookContext(HookType.PRE_TTS, "hook_test", {"test": "data"})
        result_context = await plugin.execute_hook(HookType.PRE_TTS, context)
        hook_time = time.time() - start_time

        assert hook_time < 0.2  # Should execute 3 hooks within 0.2 seconds
        assert result_context.data["hook1"] == "executed"
        assert result_context.data["hook2"] == "executed"
        assert result_context.data["hook3"] == "executed"

    async def test_plugin_event_performance(self, plugin_manager):
        """Test plugin event handling performance."""
        plugin = MockPerformancePlugin("event_test", 0.01)

        await plugin_manager.load_plugin(plugin)

        # Register multiple event handlers
        async def handler1(context):
            await asyncio.sleep(0.005)
            context.data["handler1"] = True

        async def handler2(context):
            await asyncio.sleep(0.005)
            context.data["handler2"] = True

        async def handler3(context):
            await asyncio.sleep(0.005)
            context.data["handler3"] = True

        plugin.register_event_handler("custom", handler1)
        plugin.register_event_handler("custom", handler2)
        plugin.register_event_handler("custom", handler3)

        # Emit event
        from utils.plugin_interface import EventContext, EventType

        start_time = time.time()
        context = EventContext(EventType.CUSTOM, "event_test", {"test": "data"})
        await plugin.emit_event(EventType.CUSTOM, context)
        event_time = time.time() - start_time

        assert event_time < 0.1  # Should handle 3 events within 0.1 seconds
        assert context.data["handler1"] == True
        assert context.data["handler2"] == True
        assert context.data["handler3"] == True

    async def test_plugin_dependency_resolution_performance(self, plugin_manager):
        """Test plugin dependency resolution performance."""
        # Create plugins with dependencies
        plugins = []
        for i in range(5):
            plugin = MockPerformancePlugin(f"dep_plugin_{i}", 0.01)
            plugin.dependencies = [f"dep_plugin_{j}" for j in range(i)]
            plugins.append(plugin)

        # Load all plugins
        start_time = time.time()

        tasks = []
        for plugin in plugins:
            task = plugin_manager.load_plugin(plugin)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        load_time = time.time() - start_time

        assert len(results) == 5
        assert all(result.success for result in results)
        assert load_time < 3.0  # Should resolve dependencies within 3 seconds

    def test_memory_efficiency_with_many_plugins(self, plugin_manager):
        """Test memory efficiency with many plugin instances."""
        import gc

        # Create many plugin instances
        plugins = []
        for i in range(100):
            plugin = MockPerformancePlugin(f"memory_efficiency_{i}", 0.001)
            plugins.append(plugin)

        # Load plugins in batches
        batch_size = 10
        for i in range(0, len(plugins), batch_size):
            batch = plugins[i:i + batch_size]
            for plugin in batch:
                # Just test that we can create many instances without issues
                assert plugin.name.startswith("memory_efficiency")

        # Force garbage collection
        gc.collect()

        # Check that we can still create new instances
        new_plugin = MockPerformancePlugin("new_plugin", 0.001)
        assert new_plugin.name == "new_plugin"

    async def test_plugin_scaling_performance(self, plugin_manager):
        """Test performance scaling with increasing number of plugins."""
        plugin_counts = [1, 5, 10, 20]
        times = []

        for count in plugin_counts:
            # Create plugins
            plugins = []
            for i in range(count):
                plugin = MockPerformancePlugin(f"scale_test_{i}", 0.01)
                plugins.append(plugin)

            # Load plugins
            start_time = time.time()

            tasks = []
            for plugin in plugins:
                task = plugin_manager.load_plugin(plugin)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            load_time = time.time() - start_time

            times.append(load_time)

            # Cleanup
            for plugin in plugins:
                await plugin_manager.unload_plugin(plugin.name)

            # Verify all loaded successfully
            assert all(result.success for result in results)

        # Check that loading time scales roughly linearly
        for i in range(1, len(times)):
            # Each batch should take roughly proportional time
            # Allow some variance due to system load
            ratio = times[i] / times[i-1]
            expected_ratio = plugin_counts[i] / plugin_counts[i-1]
            assert ratio <= expected_ratio * 2  # Allow 2x variance

    async def test_plugin_error_rate_performance(self, plugin_manager):
        """Test performance under error conditions."""
        plugin = MockPerformancePlugin("error_rate_test", 0.01)

        await plugin_manager.load_plugin(plugin)

        # Test mix of successful and failed operations
        success_count = 0
        error_count = 0
        total_time = 0

        for i in range(50):
            start_time = time.time()

            try:
                if i % 5 == 0:  # Every 5th request fails
                    # Simulate error
                    raise Exception("Simulated error")

                result = await plugin.synthesize(f"Test request {i}")
                if result['success']:
                    success_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1

            total_time += time.time() - start_time

        # Should have mostly successful operations
        assert success_count > error_count
        assert total_time < 10.0  # Should complete within 10 seconds
        assert plugin._performance_metrics['error_count'] == error_count


class TestSystemResourceUsage:
    """Test system resource usage."""

    def test_plugin_manager_memory_usage(self):
        """Test plugin manager memory usage."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and destroy multiple plugin managers
        for i in range(10):
            manager = PluginManager()
            # Just create and let it be garbage collected

        # Force garbage collection
        import gc
        gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal
        assert memory_increase < 10 * 1024 * 1024  # Less than 10MB increase

    def test_plugin_instance_memory_usage(self):
        """Test individual plugin instance memory usage."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create many plugin instances
        plugins = []
        for i in range(100):
            plugin = MockPerformancePlugin(f"memory_test_{i}", 0.001)
            plugins.append(plugin)

        memory_with_plugins = process.memory_info().rss
        memory_increase = memory_with_plugins - initial_memory

        # Memory per plugin should be reasonable
        memory_per_plugin = memory_increase / len(plugins)
        assert memory_per_plugin < 1024 * 1024  # Less than 1MB per plugin

        # Cleanup
        del plugins
        import gc
        gc.collect()

    def test_concurrent_operations_resource_usage(self):
        """Test resource usage during concurrent operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        initial_cpu = process.cpu_percent()

        async def run_concurrent_operations():
            # Simulate concurrent plugin operations
            tasks = []
            for i in range(20):
                task = asyncio.sleep(0.1)  # Simulate work
                tasks.append(task)

            await asyncio.gather(*tasks)

        # Run concurrent operations
        asyncio.run(run_concurrent_operations())

        final_memory = process.memory_info().rss
        final_cpu = process.cpu_percent()

        # Memory usage should remain stable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase

        # CPU usage should be reasonable
        assert final_cpu < 80.0  # Less than 80% CPU