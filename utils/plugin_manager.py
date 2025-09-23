"""
Plugin manager for TTS system with production-ready features
"""

import asyncio
import importlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

from ..config.plugin import plugin_config
from ..models.plugin import Plugin, PluginStatus, PluginType, PluginDependency, PluginLog
from .plugin_interface import (
    PluginBase, PluginRegistry, PluginFactory, plugin_registry,
    initialize_plugin, cleanup_plugin, get_plugin_by_name
)
from .plugin_security import security_manager, SecurityViolation, SecurityThreat


@dataclass
class PluginLoadResult:
    """Result of plugin loading operation."""
    success: bool
    plugin: Optional[PluginBase] = None
    error_message: str = ""
    security_violations: List[SecurityViolation] = field(default_factory=list)
    load_time: float = 0.0
    dependencies_resolved: List[str] = field(default_factory=list)
    dependencies_failed: List[str] = field(default_factory=list)


@dataclass
class PluginDependencyInfo:
    """Plugin dependency information."""
    name: str
    version: str
    required: bool = True
    resolved: bool = False
    plugin_instance: Optional[PluginBase] = None
    error_message: str = ""


class DependencyResolver:
    """Resolve plugin dependencies."""

    def __init__(self):
        """Initialize dependency resolver."""
        self.logger = logging.getLogger('plugin.dependency_resolver')
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._resolved_plugins: Set[str] = set()

    def build_dependency_graph(self, plugins: List[Plugin]) -> Dict[str, Set[str]]:
        """Build dependency graph from plugin list."""
        graph = {}

        for plugin in plugins:
            graph[plugin.name] = set()

            # Add dependencies
            for dep in plugin.dependencies:
                if isinstance(dep, dict):
                    dep_name = dep.get('name', '')
                    if dep_name:
                        graph[plugin.name].add(dep_name)
                elif isinstance(dep, str):
                    graph[plugin.name].add(dep)

        self._dependency_graph = graph
        return graph

    def resolve_dependencies(self, plugin_name: str) -> List[PluginDependencyInfo]:
        """Resolve dependencies for a plugin."""
        if plugin_name not in self._dependency_graph:
            return []

        dependencies = []
        to_resolve = list(self._dependency_graph[plugin_name])

        while to_resolve:
            dep_name = to_resolve.pop()
            if dep_name in self._resolved_plugins:
                continue

            # Find plugin instance
            plugin_instance = get_plugin_by_name(dep_name)
            if plugin_instance:
                dependencies.append(PluginDependencyInfo(
                    name=dep_name,
                    version=plugin_instance.version,
                    required=True,
                    resolved=True,
                    plugin_instance=plugin_instance
                ))
                self._resolved_plugins.add(dep_name)
            else:
                dependencies.append(PluginDependencyInfo(
                    name=dep_name,
                    version="unknown",
                    required=True,
                    resolved=False,
                    error_message=f"Plugin {dep_name} not found"
                ))

        return dependencies

    def get_load_order(self) -> List[str]:
        """Get plugin load order based on dependencies."""
        # Simple topological sort
        load_order = []
        visited = set()
        temp_visited = set()

        def visit(plugin_name: str):
            if plugin_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {plugin_name}")
            if plugin_name in visited:
                return

            temp_visited.add(plugin_name)

            # Visit dependencies first
            for dep in self._dependency_graph.get(plugin_name, set()):
                visit(dep)

            temp_visited.remove(plugin_name)
            visited.add(plugin_name)
            load_order.append(plugin_name)

        # Visit all plugins
        for plugin_name in self._dependency_graph:
            if plugin_name not in visited:
                visit(plugin_name)

        return load_order

    def clear(self) -> None:
        """Clear dependency graph."""
        self._dependency_graph.clear()
        self._resolved_plugins.clear()


class PluginLifecycleManager:
    """Manage plugin lifecycle operations."""

    def __init__(self):
        """Initialize lifecycle manager."""
        self.logger = logging.getLogger('plugin.lifecycle_manager')
        self._active_plugins: Dict[str, PluginBase] = {}
        self._plugin_states: Dict[str, PluginStatus] = {}
        self._initialization_tasks: Dict[str, asyncio.Task] = {}

    async def initialize_plugin(self, plugin: PluginBase, config: Dict[str, Any] = None) -> bool:
        """Initialize a plugin."""
        plugin_name = plugin.name

        try:
            self.logger.info(f"Initializing plugin: {plugin_name}")

            # Check if already initializing
            if plugin_name in self._initialization_tasks:
                task = self._initialization_tasks[plugin_name]
                if not task.done():
                    self.logger.warning(f"Plugin {plugin_name} is already being initialized")
                    return await task
                else:
                    # Task completed, remove it
                    del self._initialization_tasks[plugin_name]

            # Create initialization task
            task = asyncio.create_task(initialize_plugin(plugin, config))
            self._initialization_tasks[plugin_name] = task

            # Wait for initialization
            success = await task

            if success:
                self._active_plugins[plugin_name] = plugin
                self._plugin_states[plugin_name] = PluginStatus.ACTIVE
                self.logger.info(f"Plugin {plugin_name} initialized successfully")
            else:
                self._plugin_states[plugin_name] = PluginStatus.ERROR
                self.logger.error(f"Plugin {plugin_name} initialization failed")

            return success

        except Exception as e:
            self._plugin_states[plugin_name] = PluginStatus.ERROR
            self.logger.error(f"Plugin {plugin_name} initialization error: {e}")
            return False
        finally:
            # Clean up task
            if plugin_name in self._initialization_tasks:
                del self._initialization_tasks[plugin_name]

    async def cleanup_plugin(self, plugin_name: str) -> bool:
        """Cleanup a plugin."""
        try:
            plugin = self._active_plugins.get(plugin_name)
            if not plugin:
                self.logger.warning(f"Plugin {plugin_name} not found for cleanup")
                return True

            self.logger.info(f"Cleaning up plugin: {plugin_name}")

            await cleanup_plugin(plugin)

            # Remove from active plugins
            if plugin_name in self._active_plugins:
                del self._active_plugins[plugin_name]

            if plugin_name in self._plugin_states:
                self._plugin_states[plugin_name] = PluginStatus.DISABLED

            self.logger.info(f"Plugin {plugin_name} cleaned up successfully")
            return True

        except Exception as e:
            self.logger.error(f"Plugin {plugin_name} cleanup error: {e}")
            return False

    async def reload_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """Reload a plugin."""
        try:
            # Cleanup first
            await self.cleanup_plugin(plugin_name)

            # Get plugin class and create new instance
            plugin_class = plugin_registry.get_plugin_class(plugin_name)
            if not plugin_class:
                self.logger.error(f"Plugin class {plugin_name} not found")
                return False

            plugin = PluginFactory.create_plugin(plugin_class, plugin_name)
            if not plugin:
                self.logger.error(f"Failed to create plugin instance: {plugin_name}")
                return False

            # Initialize new instance
            return await self.initialize_plugin(plugin, config)

        except Exception as e:
            self.logger.error(f"Plugin {plugin_name} reload error: {e}")
            return False

    def get_plugin_state(self, plugin_name: str) -> Optional[PluginStatus]:
        """Get plugin state."""
        return self._plugin_states.get(plugin_name)

    def get_active_plugins(self) -> Dict[str, PluginBase]:
        """Get active plugins."""
        return self._active_plugins.copy()

    def is_plugin_active(self, plugin_name: str) -> bool:
        """Check if plugin is active."""
        return (plugin_name in self._active_plugins and
                self._plugin_states.get(plugin_name) == PluginStatus.ACTIVE)

    async def cleanup_all_plugins(self) -> None:
        """Cleanup all active plugins."""
        self.logger.info("Cleaning up all plugins")

        cleanup_tasks = []
        for plugin_name in list(self._active_plugins.keys()):
            cleanup_tasks.append(self.cleanup_plugin(plugin_name))

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._active_plugins.clear()
        self._plugin_states.clear()
        self.logger.info("All plugins cleaned up")


class PluginLoader:
    """Load plugins from various sources."""

    def __init__(self):
        """Initialize plugin loader."""
        self.logger = logging.getLogger('plugin.loader')
        self._loaded_modules: Dict[str, Any] = {}

    def load_plugin_from_file(self, file_path: str, plugin_name: str) -> Optional[PluginBase]:
        """Load plugin from Python file."""
        try:
            self.logger.info(f"Loading plugin from file: {file_path}")

            # Load plugin using factory
            plugin = PluginFactory.load_plugin_from_file(file_path, plugin_name)

            if plugin:
                self.logger.info(f"Plugin {plugin_name} loaded successfully from {file_path}")
                return plugin
            else:
                self.logger.error(f"Failed to load plugin {plugin_name} from {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name} from {file_path}: {e}")
            return None

    def load_plugin_from_directory(self, directory_path: str) -> List[PluginBase]:
        """Load all plugins from directory."""
        plugins = []

        try:
            directory = Path(directory_path)
            if not directory.exists():
                self.logger.warning(f"Plugin directory does not exist: {directory_path}")
                return plugins

            # Find Python files
            python_files = list(directory.glob("*.py"))
            python_files.extend(list(directory.glob("*/__init__.py")))

            for file_path in python_files:
                try:
                    # Generate plugin name from file path
                    plugin_name = file_path.stem
                    if plugin_name == "__init__":
                        # For __init__.py files, use parent directory name
                        plugin_name = file_path.parent.name

                    plugin = self.load_plugin_from_file(str(file_path), plugin_name)
                    if plugin:
                        plugins.append(plugin)

                except Exception as e:
                    self.logger.error(f"Error loading plugin from {file_path}: {e}")

        except Exception as e:
            self.logger.error(f"Error loading plugins from directory {directory_path}: {e}")

        return plugins

    def load_plugin_from_package(self, package_name: str) -> Optional[PluginBase]:
        """Load plugin from Python package."""
        try:
            self.logger.info(f"Loading plugin from package: {package_name}")

            # Import the package
            package = importlib.import_module(package_name)

            # Find plugin class
            plugin_class = None
            for name in dir(package):
                obj = getattr(package, name)
                if (isinstance(obj, type) and
                    issubclass(obj, PluginBase) and
                    obj != PluginBase):
                    plugin_class = obj
                    break

            if plugin_class:
                plugin = PluginFactory.create_plugin(plugin_class, package_name)
                if plugin:
                    self.logger.info(f"Plugin {package_name} loaded successfully from package")
                    return plugin

            self.logger.error(f"No plugin class found in package: {package_name}")
            return None

        except Exception as e:
            self.logger.error(f"Error loading plugin from package {package_name}: {e}")
            return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        try:
            # Remove from registry
            plugin_registry.unregister_plugin_instance(plugin_name)

            # Remove loaded module if exists
            if plugin_name in self._loaded_modules:
                del self._loaded_modules[plugin_name]

            # Remove from sys.modules if loaded
            module_name = f"plugins.{plugin_name}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            self.logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False


class PluginManager:
    """Main plugin manager with comprehensive functionality."""

    def __init__(self):
        """Initialize plugin manager."""
        self.logger = logging.getLogger('plugin.manager')
        self.dependency_resolver = DependencyResolver()
        self.lifecycle_manager = PluginLifecycleManager()
        self.loader = PluginLoader()
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_instances: Dict[str, PluginBase] = {}
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize plugin manager."""
        try:
            self.logger.info("Initializing plugin manager")

            # Create plugin directories
            self._create_plugin_directories()

            # Load built-in plugins
            await self._load_builtin_plugins()

            self._is_initialized = True
            self.logger.info("Plugin manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Plugin manager initialization failed: {e}")
            return False

    def _create_plugin_directories(self) -> None:
        """Create necessary plugin directories."""
        directories = [
            plugin_config.PLUGIN_DIR,
            plugin_config.PLUGIN_DATA_DIR,
            plugin_config.PLUGIN_CACHE_DIR,
            plugin_config.PLUGIN_TEMP_DIR,
            plugin_config.PLUGIN_LOG_DIR
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def _load_builtin_plugins(self) -> None:
        """Load built-in plugins."""
        # This would load plugins that come with the system
        # For now, just log that we're ready for plugin loading
        self.logger.info("Ready to load plugins")

    async def load_plugin(self, plugin_source: Union[str, Path, PluginBase],
                         config: Dict[str, Any] = None) -> PluginLoadResult:
        """Load a plugin from various sources."""
        start_time = time.time()

        try:
            # Handle different plugin sources
            if isinstance(plugin_source, PluginBase):
                plugin = plugin_source
                plugin_name = plugin.name
            elif isinstance(plugin_source, (str, Path)):
                plugin_path = str(plugin_source)
                plugin_name = Path(plugin_path).stem

                # Determine load method based on path
                if plugin_path.endswith('.py'):
                    plugin = self.loader.load_plugin_from_file(plugin_path, plugin_name)
                elif Path(plugin_path).is_dir():
                    plugins = self.loader.load_plugin_from_directory(plugin_path)
                    if plugins:
                        plugin = plugins[0]  # Take first plugin for now
                    else:
                        plugin = None
                else:
                    plugin = self.loader.load_plugin_from_package(plugin_path)

                if not plugin:
                    return PluginLoadResult(
                        success=False,
                        error_message=f"Failed to load plugin from {plugin_path}"
                    )
            else:
                return PluginLoadResult(
                    success=False,
                    error_message="Invalid plugin source"
                )

            # Validate plugin security
            security_violations = security_manager.validate_and_sandbox_plugin(
                Plugin(name=plugin_name, display_name=plugin_name),  # Create basic plugin record
                getattr(plugin, '__code__', None)
            )

            if security_violations:
                critical_violations = [v for v in security_violations if v.severity == "critical"]
                if critical_violations:
                    return PluginLoadResult(
                        success=False,
                        error_message="Critical security violations found",
                        security_violations=security_violations
                    )

            # Resolve dependencies
            dependencies = self.dependency_resolver.resolve_dependencies(plugin_name)
            resolved_deps = [dep.name for dep in dependencies if dep.resolved]
            failed_deps = [dep.name for dep in dependencies if not dep.resolved]

            # Initialize plugin
            success = await self.lifecycle_manager.initialize_plugin(plugin, config)

            load_time = time.time() - start_time

            if success:
                # Register plugin
                plugin_registry.register_plugin_instance(plugin_name, plugin)
                self._plugin_instances[plugin_name] = plugin

                return PluginLoadResult(
                    success=True,
                    plugin=plugin,
                    load_time=load_time,
                    security_violations=security_violations,
                    dependencies_resolved=resolved_deps,
                    dependencies_failed=failed_deps
                )
            else:
                return PluginLoadResult(
                    success=False,
                    error_message="Plugin initialization failed",
                    load_time=load_time,
                    security_violations=security_violations,
                    dependencies_failed=failed_deps
                )

        except Exception as e:
            load_time = time.time() - start_time
            return PluginLoadResult(
                success=False,
                error_message=str(e),
                load_time=load_time
            )

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        try:
            self.logger.info(f"Unloading plugin: {plugin_name}")

            # Check if plugin is active
            if not self.lifecycle_manager.is_plugin_active(plugin_name):
                self.logger.warning(f"Plugin {plugin_name} is not active")
                return True

            # Cleanup plugin
            success = await self.lifecycle_manager.cleanup_plugin(plugin_name)

            if success:
                # Unload from loader
                self.loader.unload_plugin(plugin_name)

                # Remove from instances
                if plugin_name in self._plugin_instances:
                    del self._plugin_instances[plugin_name]

                self.logger.info(f"Plugin {plugin_name} unloaded successfully")
                return True
            else:
                self.logger.error(f"Failed to unload plugin {plugin_name}")
                return False

        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    async def reload_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """Reload a plugin."""
        try:
            self.logger.info(f"Reloading plugin: {plugin_name}")

            # Get current plugin instance
            current_plugin = self._plugin_instances.get(plugin_name)
            if not current_plugin:
                self.logger.error(f"Plugin {plugin_name} not found")
                return False

            # Reload plugin
            success = await self.lifecycle_manager.reload_plugin(plugin_name, config)

            if success:
                self.logger.info(f"Plugin {plugin_name} reloaded successfully")
                return True
            else:
                self.logger.error(f"Failed to reload plugin {plugin_name}")
                return False

        except Exception as e:
            self.logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False

    async def load_all_plugins(self, directory_path: str = None) -> Dict[str, PluginLoadResult]:
        """Load all plugins from directory."""
        directory = directory_path or plugin_config.PLUGIN_DIR
        results = {}

        try:
            self.logger.info(f"Loading all plugins from: {directory}")

            # Load plugins
            plugins = self.loader.load_plugin_from_directory(directory)

            # Build dependency graph
            plugin_records = [Plugin(name=p.name, display_name=p.name, dependencies=p.dependencies)
                            for p in plugins]
            self.dependency_resolver.build_dependency_graph(plugin_records)

            # Get load order
            load_order = self.dependency_resolver.get_load_order()

            # Load plugins in dependency order
            for plugin_name in load_order:
                plugin = next((p for p in plugins if p.name == plugin_name), None)
                if plugin:
                    result = await self.load_plugin(plugin)
                    results[plugin_name] = result

        except Exception as e:
            self.logger.error(f"Error loading plugins from {directory}: {e}")

        return results

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information."""
        plugin = self._plugin_instances.get(plugin_name)
        if plugin:
            return plugin.get_plugin_info()
        return None

    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all plugins."""
        return {
            name: plugin.get_plugin_info()
            for name, plugin in self._plugin_instances.items()
        }

    def get_plugin_state(self, plugin_name: str) -> Optional[PluginStatus]:
        """Get plugin state."""
        return self.lifecycle_manager.get_plugin_state(plugin_name)

    def get_active_plugins(self) -> Dict[str, PluginBase]:
        """Get active plugins."""
        return self.lifecycle_manager.get_active_plugins()

    def is_plugin_active(self, plugin_name: str) -> bool:
        """Check if plugin is active."""
        return self.lifecycle_manager.is_plugin_active(plugin_name)

    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        try:
            plugin = self._plugin_instances.get(plugin_name)
            if not plugin:
                self.logger.error(f"Plugin {plugin_name} not found")
                return False

            # Initialize if not already active
            if not self.is_plugin_active(plugin_name):
                success = await self.lifecycle_manager.initialize_plugin(plugin)
                if success:
                    self.logger.info(f"Plugin {plugin_name} enabled successfully")
                    return True
                else:
                    self.logger.error(f"Failed to enable plugin {plugin_name}")
                    return False
            else:
                self.logger.info(f"Plugin {plugin_name} is already active")
                return True

        except Exception as e:
            self.logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return False

    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        try:
            if not self.is_plugin_active(plugin_name):
                self.logger.info(f"Plugin {plugin_name} is already disabled")
                return True

            success = await self.lifecycle_manager.cleanup_plugin(plugin_name)

            if success:
                self.logger.info(f"Plugin {plugin_name} disabled successfully")
                return True
            else:
                self.logger.error(f"Failed to disable plugin {plugin_name}")
                return False

        except Exception as e:
            self.logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup plugin manager."""
        self.logger.info("Cleaning up plugin manager")

        # Cleanup all plugins
        await self.lifecycle_manager.cleanup_all_plugins()

        # Clear registries
        plugin_registry.clear()
        self.dependency_resolver.clear()

        self._plugins.clear()
        self._plugin_instances.clear()

        self.logger.info("Plugin manager cleaned up")

    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        active_plugins = self.get_active_plugins()
        all_states = [self.get_plugin_state(name) for name in self._plugin_instances.keys()]

        return {
            'total_plugins': len(self._plugin_instances),
            'active_plugins': len(active_plugins),
            'inactive_plugins': len(self._plugin_instances) - len(active_plugins),
            'plugin_states': {
                PluginStatus.ACTIVE.value: all_states.count(PluginStatus.ACTIVE),
                PluginStatus.DISABLED.value: all_states.count(PluginStatus.DISABLED),
                PluginStatus.ERROR.value: all_states.count(PluginStatus.ERROR),
                PluginStatus.PENDING.value: all_states.count(PluginStatus.PENDING),
            },
            'is_initialized': self._is_initialized
        }


# Global plugin manager instance
plugin_manager = PluginManager()