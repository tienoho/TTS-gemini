"""
Plugin interface for TTS system with production-ready features
"""

import asyncio
import importlib
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union, Type, TypeVar
from enum import Enum
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

from models.plugin import Plugin, PluginStatus, PluginType, PluginPermission
from config.plugin import plugin_config


class HookType(str, Enum):
    """Hook types for plugin system."""
    PRE_TTS = "pre_tts"
    POST_TTS = "post_tts"
    PRE_AUDIO_ENHANCEMENT = "pre_audio_enhancement"
    POST_AUDIO_ENHANCEMENT = "post_audio_enhancement"
    PRE_WEBHOOK = "pre_webhook"
    POST_WEBHOOK = "post_webhook"
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    ERROR_HANDLING = "error_handling"
    CUSTOM = "custom"


class EventType(str, Enum):
    """Event types for plugin system."""
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"
    PLUGIN_ENABLED = "plugin_enabled"
    PLUGIN_DISABLED = "plugin_disabled"
    PLUGIN_ERROR = "plugin_error"
    TTS_REQUEST = "tts_request"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    WEBHOOK_SENT = "webhook_sent"
    RESOURCE_USAGE = "resource_usage"
    SECURITY_ALERT = "security_alert"
    CUSTOM = "custom"


@dataclass
class HookContext:
    """Context passed to hook functions."""
    hook_type: HookType
    plugin_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EventContext:
    """Context passed to event handlers."""
    event_type: EventType
    plugin_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None


class PluginBase(ABC):
    """Base class for all plugins."""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        """Initialize plugin base."""
        self.name = name
        self.version = version
        self.description = description
        self.status = PluginStatus.PENDING
        self.logger = logging.getLogger(f"plugin.{name}")
        self.config = {}
        self.dependencies = []
        self.permissions = []
        self._hooks = {}
        self._event_handlers = {}
        self._initialized = False
        self._performance_metrics = {
            'execution_count': 0,
            'total_execution_time': 0.0,
            'last_execution_time': None,
            'memory_usage': 0,
            'error_count': 0
        }

    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'type': PluginType.CUSTOM,
            'author': '',
            'homepage': '',
            'license': '',
            'dependencies': self.dependencies,
            'permissions': self.permissions,
            'config_schema': {},
            'default_config': {}
        }

    @abstractmethod
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize plugin with configuration."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics."""
        return self._performance_metrics.copy()

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self._performance_metrics = {
            'execution_count': 0,
            'total_execution_time': 0.0,
            'last_execution_time': None,
            'memory_usage': 0,
            'error_count': 0
        }

    def register_hook(self, hook_type: HookType, hook_function: Callable) -> None:
        """Register a hook function."""
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._hooks[hook_type].append(hook_function)
        self.logger.debug(f"Registered hook {hook_type.value}")

    def unregister_hook(self, hook_type: HookType, hook_function: Callable) -> None:
        """Unregister a hook function."""
        if hook_type in self._hooks and hook_function in self._hooks[hook_type]:
            self._hooks[hook_type].remove(hook_function)
            self.logger.debug(f"Unregistered hook {hook_type.value}")

    def get_hooks(self, hook_type: HookType) -> List[Callable]:
        """Get registered hooks for a type."""
        return self._hooks.get(hook_type, [])

    def register_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered event handler {event_type.value}")

    def unregister_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """Unregister an event handler."""
        if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
            self.logger.debug(f"Unregistered event handler {event_type.value}")

    def get_event_handlers(self, event_type: EventType) -> List[Callable]:
        """Get registered event handlers for a type."""
        return self._event_handlers.get(event_type, [])

    async def execute_hook(self, hook_type: HookType, context: HookContext) -> HookContext:
        """Execute registered hooks for a type."""
        start_time = time.time()

        try:
            for hook in self.get_hooks(hook_type):
                try:
                    if inspect.iscoroutinefunction(hook):
                        result = await hook(context)
                    else:
                        result = hook(context)

                    if result and isinstance(result, dict):
                        context.data.update(result)

                except Exception as e:
                    self.logger.error(f"Hook {hook_type.value} failed: {e}")
                    context.data['hook_errors'] = context.data.get('hook_errors', [])
                    context.data['hook_errors'].append(str(e))

            execution_time = time.time() - start_time
            self._performance_metrics['total_execution_time'] += execution_time
            self._performance_metrics['last_execution_time'] = execution_time

            return context

        except Exception as e:
            self.logger.error(f"Hook execution failed: {e}")
            raise

    async def emit_event(self, event_type: EventType, context: EventContext) -> None:
        """Emit an event to registered handlers."""
        start_time = time.time()

        try:
            for handler in self.get_event_handlers(event_type):
                try:
                    if inspect.iscoroutinefunction(handler):
                        await handler(context)
                    else:
                        handler(context)

                except Exception as e:
                    self.logger.error(f"Event handler {event_type.value} failed: {e}")

            execution_time = time.time() - start_time
            self._performance_metrics['total_execution_time'] += execution_time

        except Exception as e:
            self.logger.error(f"Event emission failed: {e}")
            raise

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        try:
            # Basic validation - can be overridden by subclasses
            return True
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False

    def get_required_permissions(self) -> List[PluginPermission]:
        """Get required permissions for this plugin."""
        return []

    def check_permission(self, permission: PluginPermission, resource: str = "*") -> bool:
        """Check if plugin has a specific permission."""
        required_permissions = self.get_required_permissions()
        return permission in required_permissions

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)
        self._performance_metrics['error_count'] += 1

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)


class TTSPlugin(PluginBase):
    """Base class for TTS plugins."""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        """Initialize TTS plugin."""
        super().__init__(name, version, description)
        self.dependencies = ['tts_core']
        self.permissions = [PluginPermission.EXECUTE]

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get TTS plugin information."""
        info = super().get_plugin_info()
        info.update({
            'type': PluginType.TTS,
            'supported_languages': [],
            'supported_voices': [],
            'supported_formats': ['mp3', 'wav'],
            'max_text_length': 5000,
            'features': []
        })
        return info

    @abstractmethod
    async def synthesize(self, text: str, voice: str = "default", language: str = "vi",
                        options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synthesize text to speech."""
        pass

    @abstractmethod
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices."""
        pass

    @abstractmethod
    async def get_available_languages(self) -> List[str]:
        """Get available languages."""
        pass

    async def pre_tts_hook(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-TTS processing hook."""
        context = HookContext(
            hook_type=HookType.PRE_TTS,
            plugin_name=self.name,
            data={'text': text, 'options': options}
        )
        result_context = await self.execute_hook(HookType.PRE_TTS, context)
        return result_context.data

    async def post_tts_hook(self, audio_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Post-TTS processing hook."""
        context = HookContext(
            hook_type=HookType.POST_TTS,
            plugin_name=self.name,
            data={'audio_data': audio_data, 'metadata': metadata}
        )
        result_context = await self.execute_hook(HookType.POST_TTS, context)
        return result_context.data


class AudioEnhancementPlugin(PluginBase):
    """Base class for audio enhancement plugins."""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        """Initialize audio enhancement plugin."""
        super().__init__(name, version, description)
        self.dependencies = ['audio_core']
        self.permissions = [PluginPermission.EXECUTE]

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get audio enhancement plugin information."""
        info = super().get_plugin_info()
        info.update({
            'type': PluginType.AUDIO_ENHANCEMENT,
            'supported_formats': ['mp3', 'wav', 'ogg'],
            'enhancement_types': [],
            'parameters': {}
        })
        return info

    @abstractmethod
    async def enhance(self, audio_data: bytes, enhancement_type: str,
                     parameters: Dict[str, Any] = None) -> bytes:
        """Enhance audio data."""
        pass

    @abstractmethod
    async def get_enhancement_types(self) -> List[str]:
        """Get available enhancement types."""
        pass

    async def pre_enhancement_hook(self, audio_data: bytes, enhancement_type: str,
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-enhancement processing hook."""
        context = HookContext(
            hook_type=HookType.PRE_AUDIO_ENHANCEMENT,
            plugin_name=self.name,
            data={'audio_data': audio_data, 'enhancement_type': enhancement_type, 'parameters': parameters}
        )
        result_context = await self.execute_hook(HookType.PRE_AUDIO_ENHANCEMENT, context)
        return result_context.data

    async def post_enhancement_hook(self, enhanced_audio: bytes, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Post-enhancement processing hook."""
        context = HookContext(
            hook_type=HookType.POST_AUDIO_ENHANCEMENT,
            plugin_name=self.name,
            data={'enhanced_audio': enhanced_audio, 'original_metadata': original_metadata}
        )
        result_context = await self.execute_hook(HookType.POST_AUDIO_ENHANCEMENT, context)
        return result_context.data


class WebhookPlugin(PluginBase):
    """Base class for webhook plugins."""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        """Initialize webhook plugin."""
        super().__init__(name, version, description)
        self.dependencies = ['webhook_core']
        self.permissions = [PluginPermission.EXECUTE, PluginPermission.WRITE]

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get webhook plugin information."""
        info = super().get_plugin_info()
        info.update({
            'type': PluginType.WEBHOOK,
            'supported_events': [],
            'webhook_url': '',
            'headers': {},
            'timeout': 30
        })
        return info

    @abstractmethod
    async def send_webhook(self, event: str, data: Dict[str, Any],
                          headers: Dict[str, str] = None) -> bool:
        """Send webhook notification."""
        pass

    @abstractmethod
    async def get_supported_events(self) -> List[str]:
        """Get supported webhook events."""
        pass

    async def pre_webhook_hook(self, event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-webhook processing hook."""
        context = HookContext(
            hook_type=HookType.PRE_WEBHOOK,
            plugin_name=self.name,
            data={'event': event, 'data': data}
        )
        result_context = await self.execute_hook(HookType.PRE_WEBHOOK, context)
        return result_context.data

    async def post_webhook_hook(self, event: str, response: Dict[str, Any], success: bool) -> Dict[str, Any]:
        """Post-webhook processing hook."""
        context = HookContext(
            hook_type=HookType.POST_WEBHOOK,
            plugin_name=self.name,
            data={'event': event, 'response': response, 'success': success}
        )
        result_context = await self.execute_hook(HookType.POST_WEBHOOK, context)
        return result_context.data


class PluginRegistry:
    """Registry for managing plugin instances."""

    _instance = None
    _plugins: Dict[str, PluginBase] = {}
    _plugin_classes: Dict[str, Type[PluginBase]] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_plugin_class(self, name: str, plugin_class: Type[PluginBase]) -> None:
        """Register a plugin class."""
        self._plugin_classes[name] = plugin_class
        logging.getLogger('plugin.registry').info(f"Registered plugin class: {name}")

    def unregister_plugin_class(self, name: str) -> None:
        """Unregister a plugin class."""
        if name in self._plugin_classes:
            del self._plugin_classes[name]
            logging.getLogger('plugin.registry').info(f"Unregistered plugin class: {name}")

    def get_plugin_class(self, name: str) -> Optional[Type[PluginBase]]:
        """Get a plugin class by name."""
        return self._plugin_classes.get(name)

    def register_plugin_instance(self, name: str, plugin: PluginBase) -> None:
        """Register a plugin instance."""
        self._plugins[name] = plugin
        logging.getLogger('plugin.registry').info(f"Registered plugin instance: {name}")

    def unregister_plugin_instance(self, name: str) -> None:
        """Unregister a plugin instance."""
        if name in self._plugins:
            del self._plugins[name]
            logging.getLogger('plugin.registry').info(f"Unregistered plugin instance: {name}")

    def get_plugin_instance(self, name: str) -> Optional[PluginBase]:
        """Get a plugin instance by name."""
        return self._plugins.get(name)

    def get_all_plugin_instances(self) -> Dict[str, PluginBase]:
        """Get all plugin instances."""
        return self._plugins.copy()

    def get_plugin_classes(self) -> Dict[str, Type[PluginBase]]:
        """Get all plugin classes."""
        return self._plugin_classes.copy()

    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, PluginBase]:
        """Get plugin instances by type."""
        return {
            name: plugin for name, plugin in self._plugins.items()
            if hasattr(plugin, 'get_plugin_info') and
            plugin.get_plugin_info().get('type') == plugin_type
        }

    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        self._plugin_classes.clear()
        logging.getLogger('plugin.registry').info("Cleared plugin registry")


# Global plugin registry instance
plugin_registry = PluginRegistry()


class PluginFactory:
    """Factory for creating plugin instances."""

    @staticmethod
    def create_plugin(plugin_class: Type[PluginBase], name: str,
                     version: str = "1.0.0", description: str = "",
                     config: Dict[str, Any] = None) -> Optional[PluginBase]:
        """Create a plugin instance."""
        try:
            plugin = plugin_class(name, version, description)

            if config and plugin.validate_config(config):
                plugin.config = config

            return plugin

        except Exception as e:
            logging.getLogger('plugin.factory').error(f"Failed to create plugin {name}: {e}")
            return None

    @staticmethod
    def create_plugin_from_class_name(class_name: str, name: str,
                                    version: str = "1.0.0", description: str = "",
                                    config: Dict[str, Any] = None) -> Optional[PluginBase]:
        """Create a plugin instance from class name."""
        try:
            # Import the module and get the class
            module_name, cls_name = class_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, cls_name)

            return PluginFactory.create_plugin(plugin_class, name, version, description, config)

        except Exception as e:
            logging.getLogger('plugin.factory').error(f"Failed to create plugin from {class_name}: {e}")
            return None

    @staticmethod
    def load_plugin_from_file(file_path: str, plugin_name: str,
                             config: Dict[str, Any] = None) -> Optional[PluginBase]:
        """Load plugin from Python file."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            if spec is None:
                logging.getLogger('plugin.factory').error(f"Could not create module spec for {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)

            # Execute the module
            spec.loader.exec_module(module)

            # Find plugin class in module
            plugin_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginBase) and
                    obj != PluginBase):
                    plugin_class = obj
                    break

            if plugin_class:
                return PluginFactory.create_plugin(plugin_class, plugin_name, config=config)

            logging.getLogger('plugin.factory').error(f"No plugin class found in {file_path}")
            return None

        except Exception as e:
            logging.getLogger('plugin.factory').error(f"Failed to load plugin from {file_path}: {e}")
            return None


# Utility functions for plugin management
async def initialize_plugin(plugin: PluginBase, config: Dict[str, Any] = None) -> bool:
    """Initialize a plugin."""
    try:
        if await plugin.initialize(config):
            plugin._initialized = True
            plugin.status = PluginStatus.ACTIVE
            plugin.log_info("Plugin initialized successfully")
            return True
        return False
    except Exception as e:
        plugin.log_error(f"Plugin initialization failed: {e}")
        plugin.status = PluginStatus.ERROR
        return False


async def cleanup_plugin(plugin: PluginBase) -> None:
    """Cleanup a plugin."""
    try:
        await plugin.cleanup()
        plugin._initialized = False
        plugin.status = PluginStatus.DISABLED
        plugin.log_info("Plugin cleaned up successfully")
    except Exception as e:
        plugin.log_error(f"Plugin cleanup failed: {e}")


def get_plugin_by_name(name: str) -> Optional[PluginBase]:
    """Get plugin instance by name."""
    return plugin_registry.get_plugin_instance(name)


def get_all_plugins() -> Dict[str, PluginBase]:
    """Get all plugin instances."""
    return plugin_registry.get_all_plugin_instances()


def register_plugin_hook(plugin_name: str, hook_type: HookType, hook_function: Callable) -> None:
    """Register a hook for a plugin."""
    plugin = get_plugin_by_name(plugin_name)
    if plugin:
        plugin.register_hook(hook_type, hook_function)


def register_plugin_event_handler(plugin_name: str, event_type: EventType, handler: Callable) -> None:
    """Register an event handler for a plugin."""
    plugin = get_plugin_by_name(plugin_name)
    if plugin:
        plugin.register_event_handler(event_type, handler)