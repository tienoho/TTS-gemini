"""
Plugin security system for TTS with production-ready features
"""

import ast
import hashlib
import importlib
import inspect
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field

from config.plugin import plugin_config
from models.plugin import Plugin, PluginPermission, PluginStatus


class SecurityLevel(str, Enum):
    """Security levels for plugin execution."""
    LOW = "low"           # Basic checks only
    MEDIUM = "medium"     # Standard security checks
    HIGH = "high"         # Enhanced security checks
    CRITICAL = "critical" # Maximum security checks


class SecurityThreat(str, Enum):
    """Security threat types."""
    MALICIOUS_CODE = "malicious_code"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CODE_INJECTION = "code_injection"
    FILE_SYSTEM_VIOLATION = "file_system_violation"
    NETWORK_VIOLATION = "network_violation"
    MEMORY_CORRUPTION = "memory_corruption"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityViolation:
    """Security violation record."""
    threat_type: SecurityThreat
    severity: str  # low, medium, high, critical
    message: str
    plugin_name: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    memory_limit: int = 100  # MB
    timeout: int = 30  # seconds
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    allowed_imports: List[str] = field(default_factory=list)
    blocked_imports: List[str] = field(default_factory=list)
    allowed_functions: List[str] = field(default_factory=list)
    blocked_functions: List[str] = field(default_factory=list)
    network_allowed: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=list)


class PluginSandbox:
    """Plugin execution sandbox with security isolation."""

    def __init__(self, config: SandboxConfig = None):
        """Initialize sandbox."""
        self.config = config or SandboxConfig()
        self.logger = logging.getLogger('plugin.sandbox')
        self._active_executions = {}
        self._resource_monitor = ResourceMonitor()

    @contextmanager
    def execute_in_sandbox(self, plugin_name: str, code: str, globals_dict: Dict = None):
        """Execute code in sandbox environment."""
        execution_id = f"{plugin_name}_{int(time.time())}"
        self._active_executions[execution_id] = {
            'start_time': time.time(),
            'memory_start': self._resource_monitor.get_memory_usage(),
            'plugin_name': plugin_name
        }

        try:
            # Set up restricted environment
            restricted_globals = self._create_restricted_globals(globals_dict or {})

            # Execute with timeout and resource monitoring
            with self._resource_monitor.monitor_execution():
                result = self._execute_with_timeout(code, restricted_globals, self.config.timeout)

            # Check resource usage
            self._check_resource_usage(execution_id)

            yield result

        except TimeoutError:
            self._handle_security_violation(
                SecurityThreat.RESOURCE_EXHAUSTION,
                "critical",
                f"Plugin {plugin_name} execution timed out",
                plugin_name
            )
            raise
        except MemoryError:
            self._handle_security_violation(
                SecurityThreat.MEMORY_CORRUPTION,
                "high",
                f"Plugin {plugin_name} exceeded memory limit",
                plugin_name
            )
            raise
        except Exception as e:
            self._handle_security_violation(
                SecurityThreat.MALICIOUS_CODE,
                "medium",
                f"Plugin {plugin_name} execution failed: {str(e)}",
                plugin_name
            )
            raise
        finally:
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]

    def _create_restricted_globals(self, base_globals: Dict) -> Dict:
        """Create restricted globals dictionary."""
        restricted_globals = dict(base_globals)

        # Remove dangerous built-ins
        dangerous_builtins = [
            'eval', 'exec', 'compile', 'open', '__import__',
            'input', 'raw_input', 'reload', 'globals', 'locals'
        ]

        for builtin in dangerous_builtins:
            if builtin in restricted_globals:
                del restricted_globals[builtin]

        # Add safe wrappers
        restricted_globals['__builtins__'] = self._create_safe_builtins()
        restricted_globals['open'] = self._safe_open
        restricted_globals['__import__'] = self._safe_import

        return restricted_globals

    def _create_safe_builtins(self) -> Dict:
        """Create safe builtins dictionary."""
        safe_builtins = {}

        # Allow safe built-ins
        safe_functions = [
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
            'set', 'range', 'enumerate', 'zip', 'sorted', 'reversed',
            'min', 'max', 'sum', 'abs', 'round', 'type', 'isinstance',
            'hasattr', 'getattr', 'setattr', 'delattr', 'print'
        ]

        for func_name in safe_functions:
            if hasattr(__builtins__, func_name):
                safe_builtins[func_name] = getattr(__builtins__, func_name)

        return safe_builtins

    def _safe_open(self, file_path: str, mode: str = 'r', **kwargs):
        """Safe file open function."""
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access denied to path: {file_path}")

        return open(file_path, mode, **kwargs)

    def _safe_import(self, name: str, globals_dict: Dict = None, locals_dict: Dict = None,
                    fromlist: Tuple = (), level: int = 0):
        """Safe import function."""
        if not self._is_import_allowed(name):
            raise ImportError(f"Import not allowed: {name}")

        return __import__(name, globals_dict, locals_dict, fromlist, level)

    def _is_path_allowed(self, path: str) -> bool:
        """Check if file path is allowed."""
        normalized_path = os.path.normpath(path)

        # Check blocked paths
        for blocked_path in self.config.blocked_paths:
            if normalized_path.startswith(os.path.normpath(blocked_path)):
                return False

        # Check allowed paths
        if self.config.allowed_paths:
            for allowed_path in self.config.allowed_paths:
                if normalized_path.startswith(os.path.normpath(allowed_path)):
                    return True
            return False

        return True

    def _is_import_allowed(self, import_name: str) -> bool:
        """Check if import is allowed."""
        # Check blocked imports
        for blocked_import in self.config.blocked_imports:
            if import_name.startswith(blocked_import):
                return False

        # Check allowed imports
        if self.config.allowed_imports:
            for allowed_import in self.config.allowed_imports:
                if import_name.startswith(allowed_import):
                    return True
            return False

        return True

    def _execute_with_timeout(self, code: str, globals_dict: Dict, timeout: int):
        """Execute code with timeout."""
        def execute():
            exec(code, globals_dict)
            return globals_dict

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise TimeoutError(f"Code execution timed out after {timeout} seconds")

    def _check_resource_usage(self, execution_id: str) -> None:
        """Check resource usage for execution."""
        if execution_id not in self._active_executions:
            return

        execution_info = self._active_executions[execution_id]
        current_memory = self._resource_monitor.get_memory_usage()
        memory_used = current_memory - execution_info['memory_start']

        if memory_used > self.config.memory_limit * 1024 * 1024:  # Convert MB to bytes
            raise MemoryError(f"Memory limit exceeded: {memory_used} bytes")

    def _handle_security_violation(self, threat_type: SecurityThreat, severity: str,
                                  message: str, plugin_name: str) -> None:
        """Handle security violation."""
        violation = SecurityViolation(
            threat_type=threat_type,
            severity=severity,
            message=message,
            plugin_name=plugin_name
        )

        self.logger.warning(f"Security violation: {violation}")
        # Could emit event or send notification here


class ResourceMonitor:
    """Monitor system resources."""

    def __init__(self):
        """Initialize resource monitor."""
        self.logger = logging.getLogger('plugin.resource_monitor')

    @contextmanager
    def monitor_execution(self):
        """Monitor resource usage during execution."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        try:
            yield
        finally:
            final_memory = process.memory_info().rss
            memory_used = final_memory - initial_memory

            if memory_used > plugin_config.MAX_MEMORY_USAGE * 1024 * 1024:
                self.logger.warning(f"High memory usage detected: {memory_used / 1024 / 1024:.2f} MB")

    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return 0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.cpu_percent()
        except ImportError:
            return 0.0


class CodeAnalyzer:
    """Analyze plugin code for security vulnerabilities."""

    def __init__(self):
        """Initialize code analyzer."""
        self.logger = logging.getLogger('plugin.code_analyzer')
        self.dangerous_patterns = self._load_dangerous_patterns()

    def _load_dangerous_patterns(self) -> Dict[str, List[str]]:
        """Load dangerous code patterns."""
        return {
            'eval_usage': [
                r'\beval\s*\(',
                r'\beval\s*\(',
                r'\bexec\s*\(',
            ],
            'import_dangerous': [
                r'import\s+(os|subprocess|sys|shutil)',
                r'from\s+(os|subprocess|sys|shutil)\s+import',
            ],
            'file_operations': [
                r'\bopen\s*\(',
                r'\bfile\s*\(',
            ],
            'network_operations': [
                r'\bsocket\s*\.',
                r'\burllib\s*\.',
                r'\brequests\s*\.',
            ],
            'system_calls': [
                r'\bos\.system\s*\(',
                r'\bsubprocess\.',
                r'\bshutil\.',
            ]
        }

    def analyze_code(self, code: str, file_path: str = None) -> List[SecurityViolation]:
        """Analyze code for security issues."""
        violations = []

        try:
            # Parse AST
            tree = ast.parse(code, file_path or '<string>')

            # Check for dangerous patterns
            violations.extend(self._check_ast_patterns(tree, file_path))
            violations.extend(self._check_regex_patterns(code, file_path))

            # Check code complexity
            complexity_violations = self._check_complexity(tree, file_path)
            violations.extend(complexity_violations)

        except SyntaxError as e:
            violations.append(SecurityViolation(
                threat_type=SecurityThreat.MALICIOUS_CODE,
                severity="high",
                message=f"Syntax error in plugin code: {str(e)}",
                plugin_name=file_path or "unknown",
                file_path=file_path,
                line_number=e.lineno if hasattr(e, 'lineno') else None
            ))

        return violations

    def _check_ast_patterns(self, tree: ast.AST, file_path: str) -> List[SecurityViolation]:
        """Check AST for dangerous patterns."""
        violations = []
        dangerous_nodes = []

        for node in ast.walk(tree):
            # Check for eval/exec usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        dangerous_nodes.append(node)
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'call']:
                        dangerous_nodes.append(node)

            # Check for dangerous imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'sys', 'shutil']:
                        dangerous_nodes.append(node)

            elif isinstance(node, ast.ImportFrom):
                if node.module in ['os', 'subprocess', 'sys', 'shutil']:
                    dangerous_nodes.append(node)

        for node in dangerous_nodes:
            violations.append(SecurityViolation(
                threat_type=SecurityThreat.MALICIOUS_CODE,
                severity="high",
                message=f"Dangerous code pattern detected: {type(node).__name__}",
                plugin_name=file_path or "unknown",
                file_path=file_path,
                line_number=getattr(node, 'lineno', None),
                code_snippet=ast.get_source_segment(open(file_path).read(), node) if file_path else None
            ))

        return violations

    def _check_regex_patterns(self, code: str, file_path: str) -> List[SecurityViolation]:
        """Check code using regex patterns."""
        violations = []

        for pattern_name, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    violations.append(SecurityViolation(
                        threat_type=SecurityThreat.MALICIOUS_CODE,
                        severity="medium",
                        message=f"Dangerous pattern '{pattern_name}' detected",
                        plugin_name=file_path or "unknown",
                        file_path=file_path,
                        line_number=code[:match.start()].count('\n') + 1,
                        code_snippet=match.group()
                    ))

        return violations

    def _check_complexity(self, tree: ast.AST, file_path: str) -> List[SecurityViolation]:
        """Check code complexity."""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                if complexity > plugin_config.MAX_FUNCTION_COMPLEXITY:
                    violations.append(SecurityViolation(
                        threat_type=SecurityThreat.MALICIOUS_CODE,
                        severity="low",
                        message=f"High function complexity: {complexity}",
                        plugin_name=file_path or "unknown",
                        file_path=file_path,
                        line_number=node.lineno
                    ))

        return violations

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With,
                                ast.Try, ast.ExceptHandler, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity


class PermissionManager:
    """Manage plugin permissions."""

    def __init__(self):
        """Initialize permission manager."""
        self.logger = logging.getLogger('plugin.permissions')
        self._plugin_permissions: Dict[str, Set[PluginPermission]] = {}
        self._resource_permissions: Dict[str, Dict[str, Set[PluginPermission]]] = {}

    def grant_permission(self, plugin_name: str, permission: PluginPermission,
                        resource: str = "*") -> None:
        """Grant permission to plugin."""
        if plugin_name not in self._plugin_permissions:
            self._plugin_permissions[plugin_name] = set()

        self._plugin_permissions[plugin_name].add(permission)

        if resource != "*":
            if plugin_name not in self._resource_permissions:
                self._resource_permissions[plugin_name] = {}
            if resource not in self._resource_permissions[plugin_name]:
                self._resource_permissions[plugin_name][resource] = set()
            self._resource_permissions[plugin_name][resource].add(permission)

        self.logger.info(f"Granted {permission.value} permission to {plugin_name} for resource {resource}")

    def revoke_permission(self, plugin_name: str, permission: PluginPermission,
                         resource: str = "*") -> None:
        """Revoke permission from plugin."""
        if plugin_name in self._plugin_permissions:
            self._plugin_permissions[plugin_name].discard(permission)

        if resource != "*" and plugin_name in self._resource_permissions:
            if resource in self._resource_permissions[plugin_name]:
                self._resource_permissions[plugin_name][resource].discard(permission)

        self.logger.info(f"Revoked {permission.value} permission from {plugin_name} for resource {resource}")

    def check_permission(self, plugin_name: str, permission: PluginPermission,
                        resource: str = "*") -> bool:
        """Check if plugin has permission."""
        # Check global permissions
        if plugin_name in self._plugin_permissions:
            if permission in self._plugin_permissions[plugin_name]:
                return True

        # Check resource-specific permissions
        if (plugin_name in self._resource_permissions and
            resource in self._resource_permissions[plugin_name]):
            if permission in self._resource_permissions[plugin_name][resource]:
                return True

        return False

    def get_plugin_permissions(self, plugin_name: str) -> Set[PluginPermission]:
        """Get all permissions for a plugin."""
        return self._plugin_permissions.get(plugin_name, set())

    def get_resource_permissions(self, plugin_name: str, resource: str) -> Set[PluginPermission]:
        """Get permissions for a specific resource."""
        if plugin_name in self._resource_permissions:
            return self._resource_permissions[plugin_name].get(resource, set())
        return set()

    def clear_plugin_permissions(self, plugin_name: str) -> None:
        """Clear all permissions for a plugin."""
        if plugin_name in self._plugin_permissions:
            del self._plugin_permissions[plugin_name]

        if plugin_name in self._resource_permissions:
            del self._resource_permissions[plugin_name]

        self.logger.info(f"Cleared all permissions for {plugin_name}")


class SecurityValidator:
    """Validate plugin security."""

    def __init__(self):
        """Initialize security validator."""
        self.logger = logging.getLogger('plugin.security_validator')
        self.code_analyzer = CodeAnalyzer()
        self.sandbox = PluginSandbox()
        self.permission_manager = PermissionManager()

    def validate_plugin(self, plugin: Plugin, code: str = None) -> List[SecurityViolation]:
        """Validate plugin security."""
        violations = []

        # Validate plugin metadata
        violations.extend(self._validate_plugin_metadata(plugin))

        # Validate plugin code if provided
        if code:
            violations.extend(self._validate_plugin_code(plugin.name, code))

        # Check permissions
        violations.extend(self._validate_plugin_permissions(plugin))

        return violations

    def _validate_plugin_metadata(self, plugin: Plugin) -> List[SecurityViolation]:
        """Validate plugin metadata."""
        violations = []

        # Check for suspicious plugin names
        if re.search(r'[^\w\-_.]', plugin.name):
            violations.append(SecurityViolation(
                threat_type=SecurityThreat.MALICIOUS_CODE,
                severity="medium",
                message=f"Suspicious plugin name: {plugin.name}",
                plugin_name=plugin.name
            ))

        # Check for overly long names
        if len(plugin.name) > 100:
            violations.append(SecurityViolation(
                threat_type=SecurityThreat.MALICIOUS_CODE,
                severity="low",
                message=f"Plugin name too long: {len(plugin.name)} characters",
                plugin_name=plugin.name
            ))

        # Check dependencies
        if len(plugin.dependencies) > plugin_config.MAX_DEPENDENCIES_PER_PLUGIN:
            violations.append(SecurityViolation(
                threat_type=SecurityThreat.RESOURCE_EXHAUSTION,
                severity="medium",
                message=f"Too many dependencies: {len(plugin.dependencies)}",
                plugin_name=plugin.name
            ))

        return violations

    def _validate_plugin_code(self, plugin_name: str, code: str) -> List[SecurityViolation]:
        """Validate plugin code."""
        violations = []

        # Analyze code for security issues
        violations.extend(self.code_analyzer.analyze_code(code, plugin_name))

        # Check code size
        if len(code) > plugin_config.MAX_PLUGIN_SIZE:
            violations.append(SecurityViolation(
                threat_type=SecurityThreat.RESOURCE_EXHAUSTION,
                severity="medium",
                message=f"Plugin code too large: {len(code)} bytes",
                plugin_name=plugin_name
            ))

        return violations

    def _validate_plugin_permissions(self, plugin: Plugin) -> List[SecurityViolation]:
        """Validate plugin permissions."""
        violations = []

        # Check for excessive permissions
        if len(plugin.permissions) > 10:
            violations.append(SecurityViolation(
                threat_type=SecurityThreat.PRIVILEGE_ESCALATION,
                severity="medium",
                message=f"Too many permissions requested: {len(plugin.permissions)}",
                plugin_name=plugin.name
            ))

        # Check for dangerous permission combinations
        permissions = set(perm.permission for perm in plugin.permissions)
        dangerous_combinations = [
            {PluginPermission.ADMIN, PluginPermission.EXECUTE},
            {PluginPermission.WRITE, PluginPermission.EXECUTE}
        ]

        for combination in dangerous_combinations:
            if combination.issubset(permissions):
                violations.append(SecurityViolation(
                    threat_type=SecurityThreat.PRIVILEGE_ESCALATION,
                    severity="high",
                    message=f"Dangerous permission combination: {combination}",
                    plugin_name=plugin.name
                ))

        return violations

    def validate_plugin_execution(self, plugin_name: str, operation: str,
                                resource: str = "*") -> bool:
        """Validate if plugin can perform an operation."""
        # Check if plugin has required permissions
        required_permission = self._get_required_permission(operation)
        if required_permission:
            return self.permission_manager.check_permission(plugin_name, required_permission, resource)

        return True

    def _get_required_permission(self, operation: str) -> Optional[PluginPermission]:
        """Get required permission for operation."""
        operation_permissions = {
            'read': PluginPermission.READ,
            'write': PluginPermission.WRITE,
            'execute': PluginPermission.EXECUTE,
            'admin': PluginPermission.ADMIN
        }

        return operation_permissions.get(operation.lower())


class PluginSecurityManager:
    """Main security manager for plugin system."""

    def __init__(self):
        """Initialize security manager."""
        self.logger = logging.getLogger('plugin.security_manager')
        self.validator = SecurityValidator()
        self.sandbox = PluginSandbox()
        self.permission_manager = PermissionManager()
        self._security_violations: List[SecurityViolation] = []

    def validate_and_sandbox_plugin(self, plugin: Plugin, code: str = None) -> bool:
        """Validate plugin and prepare for sandboxing."""
        # Validate plugin
        violations = self.validator.validate_plugin(plugin, code)

        if violations:
            self._security_violations.extend(violations)

            # Check if violations are critical
            critical_violations = [v for v in violations if v.severity == "critical"]
            if critical_violations:
                plugin.status = PluginStatus.ERROR
                self.logger.error(f"Critical security violations found for {plugin.name}: {critical_violations}")
                return False

        # Set up permissions
        self._setup_plugin_permissions(plugin)

        return True

    def _setup_plugin_permissions(self, plugin: Plugin) -> None:
        """Set up plugin permissions."""
        # Grant default permissions
        for permission in plugin_config.DEFAULT_PERMISSIONS:
            try:
                perm_enum = PluginPermission(permission)
                self.permission_manager.grant_permission(plugin.name, perm_enum)
            except ValueError:
                self.logger.warning(f"Invalid default permission: {permission}")

        # Grant plugin-specific permissions
        for perm in plugin.permissions:
            self.permission_manager.grant_permission(plugin.name, perm.permission, perm.resource)

    def check_plugin_permission(self, plugin_name: str, permission: PluginPermission,
                              resource: str = "*") -> bool:
        """Check if plugin has permission."""
        return self.permission_manager.check_permission(plugin_name, permission, resource)

    def execute_plugin_code(self, plugin_name: str, code: str,
                          globals_dict: Dict = None) -> Any:
        """Execute plugin code in sandbox."""
        with self.sandbox.execute_in_sandbox(plugin_name, code, globals_dict) as result:
            return result

    def get_security_violations(self, plugin_name: str = None) -> List[SecurityViolation]:
        """Get security violations."""
        if plugin_name:
            return [v for v in self._security_violations if v.plugin_name == plugin_name]
        return self._security_violations.copy()

    def clear_security_violations(self, plugin_name: str = None) -> None:
        """Clear security violations."""
        if plugin_name:
            self._security_violations = [v for v in self._security_violations if v.plugin_name != plugin_name]
        else:
            self._security_violations.clear()

    def get_security_report(self, plugin_name: str = None) -> Dict[str, Any]:
        """Get security report."""
        violations = self.get_security_violations(plugin_name)

        report = {
            'total_violations': len(violations),
            'violations_by_severity': {},
            'violations_by_type': {},
            'plugins_affected': len(set(v.plugin_name for v in violations))
        }

        for violation in violations:
            # Count by severity
            severity = violation.severity
            report['violations_by_severity'][severity] = report['violations_by_severity'].get(severity, 0) + 1

            # Count by type
            threat_type = violation.threat_type.value
            report['violations_by_type'][threat_type] = report['violations_by_type'].get(threat_type, 0) + 1

        return report


# Global security manager instance
security_manager = PluginSecurityManager()