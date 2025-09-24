"""
Plugin configuration for TTS system with production-ready features
"""

import os
from typing import List, Dict, Any, Optional
from datetime import timedelta

from pydantic_settings import BaseSettings


class PluginConfig(BaseSettings):
    """Plugin system configuration settings."""

    # Plugin directories
    PLUGIN_DIR: str = os.getenv('PLUGIN_DIR', 'plugins')
    PLUGIN_DATA_DIR: str = os.getenv('PLUGIN_DATA_DIR', 'plugins/data')
    PLUGIN_CACHE_DIR: str = os.getenv('PLUGIN_CACHE_DIR', 'plugins/cache')
    PLUGIN_TEMP_DIR: str = os.getenv('PLUGIN_TEMP_DIR', 'plugins/temp')
    PLUGIN_LOG_DIR: str = os.getenv('PLUGIN_LOG_DIR', 'plugins/logs')

    # Plugin marketplace settings
    PLUGIN_MARKETPLACE_URL: str = os.getenv('PLUGIN_MARKETPLACE_URL', 'https://marketplace.tts-api.com')
    ENABLE_MARKETPLACE: bool = os.getenv('ENABLE_MARKETPLACE', 'true').lower() == 'true'
    MARKETPLACE_API_KEY: str = os.getenv('MARKETPLACE_API_KEY', '')

    # Security settings
    PLUGIN_SECURITY_ENABLED: bool = os.getenv('PLUGIN_SECURITY_ENABLED', 'true').lower() == 'true'
    PLUGIN_SANDBOX_ENABLED: bool = os.getenv('PLUGIN_SANDBOX_ENABLED', 'true').lower() == 'true'
    PLUGIN_CODE_ANALYSIS_ENABLED: bool = os.getenv('PLUGIN_CODE_ANALYSIS_ENABLED', 'true').lower() == 'true'

    # Security limits
    MAX_PLUGIN_SIZE: int = int(os.getenv('MAX_PLUGIN_SIZE', '10485760'))  # 10MB
    MAX_DEPENDENCIES_PER_PLUGIN: int = int(os.getenv('MAX_DEPENDENCIES_PER_PLUGIN', '10'))
    MAX_EXECUTION_TIME: int = int(os.getenv('MAX_EXECUTION_TIME', '30'))  # seconds
    MAX_MEMORY_USAGE: int = int(os.getenv('MAX_MEMORY_USAGE', '100'))  # MB
    MAX_FILE_ACCESS: int = int(os.getenv('MAX_FILE_ACCESS', '1000'))  # files per execution

    # Network security
    ALLOW_NETWORK_ACCESS: bool = os.getenv('ALLOW_NETWORK_ACCESS', 'false').lower() == 'true'
    ALLOWED_HOSTS: List[str] = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    BLOCKED_HOSTS: List[str] = os.getenv('BLOCKED_HOSTS', '').split(',') if os.getenv('BLOCKED_HOSTS') else []

    # File system security
    ALLOWED_PATHS: List[str] = os.getenv('ALLOWED_PATHS', 'plugins/,uploads/').split(',')
    BLOCKED_PATHS: List[str] = os.getenv('BLOCKED_PATHS', '/etc,/usr/bin,/bin,/sbin,/boot,/sys,/proc').split(',')

    # Plugin isolation settings
    ISOLATION_LEVEL: str = os.getenv('ISOLATION_LEVEL', 'process')  # process, thread, none
    SANDBOX_TYPE: str = os.getenv('SANDBOX_TYPE', 'subprocess')  # subprocess, docker, none

    # Docker sandbox settings (if using Docker isolation)
    DOCKER_IMAGE: str = os.getenv('DOCKER_IMAGE', 'python:3.9-slim')
    DOCKER_NETWORK: str = os.getenv('DOCKER_NETWORK', 'plugin_network')
    DOCKER_VOLUME_MOUNTS: Dict[str, str] = {
        '/app/plugins': '/app/plugins',
        '/app/uploads': '/app/uploads'
    }

    # Performance limits
    PLUGIN_LOAD_TIMEOUT: int = int(os.getenv('PLUGIN_LOAD_TIMEOUT', '30'))  # seconds
    PLUGIN_INIT_TIMEOUT: int = int(os.getenv('PLUGIN_INIT_TIMEOUT', '10'))  # seconds
    PLUGIN_EXECUTION_TIMEOUT: int = int(os.getenv('PLUGIN_EXECUTION_TIMEOUT', '30'))  # seconds

    # Resource limits
    MAX_CONCURRENT_PLUGINS: int = int(os.getenv('MAX_CONCURRENT_PLUGINS', '10'))
    MAX_PLUGIN_INSTANCES: int = int(os.getenv('MAX_PLUGIN_INSTANCES', '50'))
    PLUGIN_CLEANUP_INTERVAL: int = int(os.getenv('PLUGIN_CLEANUP_INTERVAL', '300'))  # seconds

    # Plugin lifecycle settings
    AUTO_RELOAD_PLUGINS: bool = os.getenv('AUTO_RELOAD_PLUGINS', 'true').lower() == 'true'
    PLUGIN_RELOAD_INTERVAL: int = int(os.getenv('PLUGIN_RELOAD_INTERVAL', '60'))  # seconds
    ENABLE_PLUGIN_WATCHDOG: bool = os.getenv('ENABLE_PLUGIN_WATCHDOG', 'true').lower() == 'true'
    WATCHDOG_TIMEOUT: int = int(os.getenv('WATCHDOG_TIMEOUT', '300'))  # seconds

    # Plugin permissions
    DEFAULT_PERMISSIONS: List[str] = os.getenv('DEFAULT_PERMISSIONS', 'read,execute').split(',')
    REQUIRE_ADMIN_APPROVAL: bool = os.getenv('REQUIRE_ADMIN_APPROVAL', 'true').lower() == 'true'
    ALLOW_UNVERIFIED_PLUGINS: bool = os.getenv('ALLOW_UNVERIFIED_PLUGINS', 'false').lower() == 'true'

    # Plugin validation settings
    VALIDATE_DEPENDENCIES: bool = os.getenv('VALIDATE_DEPENDENCIES', 'true').lower() == 'true'
    VALIDATE_SIGNATURES: bool = os.getenv('VALIDATE_SIGNATURES', 'true').lower() == 'true'
    VALIDATE_CODE_QUALITY: bool = os.getenv('VALIDATE_CODE_QUALITY', 'true').lower() == 'true'

    # Code analysis settings
    SECURITY_SCAN_ENABLED: bool = os.getenv('SECURITY_SCAN_ENABLED', 'true').lower() == 'true'
    BLOCKED_IMPORTS: List[str] = os.getenv('BLOCKED_IMPORTS', 'os,subprocess,socket,urllib,requests').split(',')
    BLOCKED_FUNCTIONS: List[str] = os.getenv('BLOCKED_FUNCTIONS', 'eval,exec,open,__import__').split(',')
    MAX_FUNCTION_COMPLEXITY: int = int(os.getenv('MAX_FUNCTION_COMPLEXITY', '10'))
    MAX_NESTING_DEPTH: int = int(os.getenv('MAX_NESTING_DEPTH', '5'))

    # Logging settings
    PLUGIN_LOG_LEVEL: str = os.getenv('PLUGIN_LOG_LEVEL', 'INFO')
    LOG_PLUGIN_EXECUTION: bool = os.getenv('LOG_PLUGIN_EXECUTION', 'true').lower() == 'true'
    LOG_PERFORMANCE_METRICS: bool = os.getenv('LOG_PERFORMANCE_METRICS', 'true').lower() == 'true'
    LOG_SECURITY_EVENTS: bool = os.getenv('LOG_SECURITY_EVENTS', 'true').lower() == 'true'

    # Monitoring and metrics
    ENABLE_PLUGIN_METRICS: bool = os.getenv('ENABLE_PLUGIN_METRICS', 'true').lower() == 'true'
    METRICS_COLLECTION_INTERVAL: int = int(os.getenv('METRICS_COLLECTION_INTERVAL', '60'))  # seconds
    PERFORMANCE_THRESHOLD_WARNING: float = float(os.getenv('PERFORMANCE_THRESHOLD_WARNING', '0.8'))  # 80%
    PERFORMANCE_THRESHOLD_CRITICAL: float = float(os.getenv('PERFORMANCE_THRESHOLD_CRITICAL', '0.9'))  # 90%

    # Error handling
    MAX_ERROR_COUNT: int = int(os.getenv('MAX_ERROR_COUNT', '5'))
    ERROR_RESET_INTERVAL: int = int(os.getenv('ERROR_RESET_INTERVAL', '3600'))  # seconds
    AUTO_DISABLE_ON_ERROR: bool = os.getenv('AUTO_DISABLE_ON_ERROR', 'true').lower() == 'true'

    # Plugin development settings
    DEVELOPMENT_MODE: bool = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'
    HOT_RELOAD_ENABLED: bool = os.getenv('HOT_RELOAD_ENABLED', 'false').lower() == 'true'
    DEBUG_PLUGIN_LOADING: bool = os.getenv('DEBUG_PLUGIN_LOADING', 'false').lower() == 'true'

    # API settings
    PLUGIN_API_ENABLED: bool = os.getenv('PLUGIN_API_ENABLED', 'true').lower() == 'true'
    PLUGIN_API_RATE_LIMIT: int = int(os.getenv('PLUGIN_API_RATE_LIMIT', '100'))  # requests per minute
    PLUGIN_API_TIMEOUT: int = int(os.getenv('PLUGIN_API_TIMEOUT', '30'))  # seconds

    # Webhook settings for plugin events
    PLUGIN_WEBHOOK_ENABLED: bool = os.getenv('PLUGIN_WEBHOOK_ENABLED', 'true').lower() == 'true'
    PLUGIN_WEBHOOK_URL: str = os.getenv('PLUGIN_WEBHOOK_URL', '')
    WEBHOOK_TIMEOUT: int = int(os.getenv('WEBHOOK_TIMEOUT', '10'))  # seconds
    WEBHOOK_RETRIES: int = int(os.getenv('WEBHOOK_RETRIES', '3'))

    # Backup and recovery
    BACKUP_ENABLED: bool = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
    BACKUP_INTERVAL: int = int(os.getenv('BACKUP_INTERVAL', '86400'))  # seconds (24 hours)
    BACKUP_RETENTION_DAYS: int = int(os.getenv('BACKUP_RETENTION_DAYS', '30'))
    AUTO_RECOVERY_ENABLED: bool = os.getenv('AUTO_RECOVERY_ENABLED', 'true').lower() == 'true'

    # Plugin marketplace integration
    MARKETPLACE_CACHE_TTL: int = int(os.getenv('MARKETPLACE_CACHE_TTL', '3600'))  # seconds
    MARKETPLACE_REQUEST_TIMEOUT: int = int(os.getenv('MARKETPLACE_REQUEST_TIMEOUT', '30'))  # seconds
    TRUSTED_PUBLISHERS: List[str] = os.getenv('TRUSTED_PUBLISHERS', '').split(',') if os.getenv('TRUSTED_PUBLISHERS') else []

    # Advanced security features
    ENABLE_MALWARE_SCAN: bool = os.getenv('ENABLE_MALWARE_SCAN', 'true').lower() == 'true'
    ENABLE_BEHAVIOR_ANALYSIS: bool = os.getenv('ENABLE_BEHAVIOR_ANALYSIS', 'true').lower() == 'true'
    BEHAVIOR_ANALYSIS_THRESHOLD: float = float(os.getenv('BEHAVIOR_ANALYSIS_THRESHOLD', '0.7'))

    # Resource management
    CLEANUP_ON_SHUTDOWN: bool = os.getenv('CLEANUP_ON_SHUTDOWN', 'true').lower() == 'true'
    RESOURCE_MONITORING_ENABLED: bool = os.getenv('RESOURCE_MONITORING_ENABLED', 'true').lower() == 'true'
    RESOURCE_MONITORING_INTERVAL: int = int(os.getenv('RESOURCE_MONITORING_INTERVAL', '10'))  # seconds

    class Config:
        """Pydantic configuration."""
        env_file = '.env'
        case_sensitive = False
        extra = 'ignore'  # Ignore extra fields not defined in the model

    def get_plugin_path(self, plugin_name: str) -> str:
        """Get full path for a plugin."""
        return os.path.join(self.PLUGIN_DIR, plugin_name)

    def get_plugin_data_path(self, plugin_name: str) -> str:
        """Get data directory path for a plugin."""
        return os.path.join(self.PLUGIN_DATA_DIR, plugin_name)

    def get_plugin_cache_path(self, plugin_name: str) -> str:
        """Get cache directory path for a plugin."""
        return os.path.join(self.PLUGIN_CACHE_DIR, plugin_name)

    def is_path_allowed(self, path: str) -> bool:
        """Check if a file system path is allowed."""
        normalized_path = os.path.normpath(path)

        # Check blocked paths
        for blocked_path in self.BLOCKED_PATHS:
            if normalized_path.startswith(os.path.normpath(blocked_path)):
                return False

        # Check allowed paths
        if self.ALLOWED_PATHS:
            for allowed_path in self.ALLOWED_PATHS:
                if normalized_path.startswith(os.path.normpath(allowed_path)):
                    return True
            return False

        return True

    def is_host_allowed(self, host: str) -> bool:
        """Check if a network host is allowed."""
        if not self.ALLOW_NETWORK_ACCESS:
            return False

        if host in self.BLOCKED_HOSTS:
            return False

        if '*' in self.ALLOWED_HOSTS or host in self.ALLOWED_HOSTS:
            return True

        return False

    def is_import_allowed(self, import_name: str) -> bool:
        """Check if a Python import is allowed."""
        for blocked_import in self.BLOCKED_IMPORTS:
            if import_name.startswith(blocked_import):
                return False
        return True

    def is_function_allowed(self, function_name: str) -> bool:
        """Check if a Python function is allowed."""
        return function_name not in self.BLOCKED_FUNCTIONS

    def get_security_settings(self) -> Dict[str, Any]:
        """Get all security-related settings."""
        return {
            'security_enabled': self.PLUGIN_SECURITY_ENABLED,
            'sandbox_enabled': self.PLUGIN_SANDBOX_ENABLED,
            'code_analysis_enabled': self.PLUGIN_CODE_ANALYSIS_ENABLED,
            'network_access_allowed': self.ALLOW_NETWORK_ACCESS,
            'allowed_hosts': self.ALLOWED_HOSTS,
            'blocked_hosts': self.BLOCKED_HOSTS,
            'allowed_paths': self.ALLOWED_PATHS,
            'blocked_paths': self.BLOCKED_PATHS,
            'blocked_imports': self.BLOCKED_IMPORTS,
            'blocked_functions': self.BLOCKED_FUNCTIONS,
            'max_plugin_size': self.MAX_PLUGIN_SIZE,
            'max_execution_time': self.MAX_EXECUTION_TIME,
            'max_memory_usage': self.MAX_MEMORY_USAGE,
        }

    def get_performance_settings(self) -> Dict[str, Any]:
        """Get all performance-related settings."""
        return {
            'max_concurrent_plugins': self.MAX_CONCURRENT_PLUGINS,
            'max_plugin_instances': self.MAX_PLUGIN_INSTANCES,
            'plugin_load_timeout': self.PLUGIN_LOAD_TIMEOUT,
            'plugin_init_timeout': self.PLUGIN_INIT_TIMEOUT,
            'plugin_execution_timeout': self.PLUGIN_EXECUTION_TIMEOUT,
            'cleanup_interval': self.PLUGIN_CLEANUP_INTERVAL,
            'resource_monitoring_interval': self.RESOURCE_MONITORING_INTERVAL,
        }


# Global plugin configuration instance
plugin_config = PluginConfig()