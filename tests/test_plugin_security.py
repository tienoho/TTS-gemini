"""
Plugin security tests for TTS system
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from utils.plugin_security import (
    PluginSecurityManager, SecurityValidator, PluginSandbox,
    CodeAnalyzer, PermissionManager, ResourceMonitor,
    security_manager, SecurityLevel, SecurityThreat, SecurityViolation, SandboxConfig
)
from utils.plugin_interface import PluginBase
from models.plugin import Plugin, PluginPermission


class MockPlugin(PluginBase):
    """Mock plugin for security testing."""

    def __init__(self, name="test_plugin"):
        super().__init__(name, "1.0.0", "Test Plugin")

    def get_plugin_info(self):
        return {"name": self.name, "version": self.version}

    async def initialize(self, config=None):
        return True

    async def cleanup(self):
        pass


class TestCodeAnalyzer:
    """Test code analysis functionality."""

    def test_analyze_safe_code(self):
        """Test analyzing safe code."""
        analyzer = CodeAnalyzer()

        safe_code = '''
def hello_world():
    print("Hello, World!")
    return "success"

class TestClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
'''

        violations = analyzer.analyze_code(safe_code, "test_plugin.py")
        assert len(violations) == 0

    def test_analyze_dangerous_code(self):
        """Test analyzing dangerous code."""
        analyzer = CodeAnalyzer()

        dangerous_code = '''
import os
import subprocess

def dangerous_function():
    eval("malicious_code")
    exec("print('hacked')")
    os.system("rm -rf /")
    subprocess.call(["rm", "-rf", "/"])
'''

        violations = analyzer.analyze_code(dangerous_code, "test_plugin.py")

        # Should detect dangerous patterns
        assert len(violations) > 0

        # Check for specific violations
        eval_violations = [v for v in violations if "eval" in v.message.lower()]
        assert len(eval_violations) > 0

        os_import_violations = [v for v in violations if "os" in v.message.lower()]
        assert len(os_import_violations) > 0

    def test_analyze_code_with_syntax_error(self):
        """Test analyzing code with syntax error."""
        analyzer = CodeAnalyzer()

        syntax_error_code = '''
def broken_function(:
    print("This is broken"
    return
'''

        violations = analyzer.analyze_code(syntax_error_code, "test_plugin.py")

        # Should detect syntax error
        syntax_violations = [v for v in violations if "syntax" in v.message.lower()]
        assert len(syntax_violations) > 0

    def test_analyze_complex_code(self):
        """Test analyzing complex code."""
        analyzer = CodeAnalyzer()

        complex_code = '''
def complex_function():
    if True:
        for i in range(10):
            if i > 5:
                while i < 15:
                    if i % 2 == 0:
                        try:
                            result = i * 2
                        except Exception as e:
                            print(e)
    return result
'''

        violations = analyzer.analyze_code(complex_code, "test_plugin.py")

        # Should detect high complexity
        complexity_violations = [v for v in violations if "complexity" in v.message.lower()]
        assert len(complexity_violations) > 0

    def test_analyze_regex_patterns(self):
        """Test regex pattern detection."""
        analyzer = CodeAnalyzer()

        regex_test_code = '''
import os
import sys
import socket
import urllib.request

def test_function():
    open("/etc/passwd", "r")
    eval("1+1")
    exec("print('test')")
    socket.socket()
    urllib.request.urlopen("http://example.com")
'''

        violations = analyzer.analyze_code(regex_test_code, "test_plugin.py")

        # Should detect various dangerous patterns
        assert len(violations) > 0

        # Check for file operations
        file_violations = [v for v in violations if "file" in v.message.lower()]
        assert len(file_violations) > 0

        # Check for network operations
        network_violations = [v for v in violations if "network" in v.message.lower()]
        assert len(network_violations) > 0


class TestPluginSandbox:
    """Test plugin sandbox functionality."""

    def test_sandbox_safe_execution(self):
        """Test safe code execution in sandbox."""
        sandbox = PluginSandbox()

        safe_code = '''
result = []
for i in range(5):
    result.append(i * 2)
final_result = sum(result)
'''

        with sandbox.execute_in_sandbox("test_plugin", safe_code) as result:
            assert result['final_result'] == 30

    def test_sandbox_dangerous_execution(self):
        """Test dangerous code execution in sandbox."""
        sandbox = PluginSandbox()

        dangerous_code = '''
import os
os.system("echo 'hacked'")
'''

        with pytest.raises(Exception):
            with sandbox.execute_in_sandbox("test_plugin", dangerous_code):
                pass

    def test_sandbox_timeout(self):
        """Test sandbox timeout."""
        sandbox = PluginSandbox(SandboxConfig(timeout=1))

        timeout_code = '''
import time
time.sleep(5)  # This should timeout
'''

        with pytest.raises(TimeoutError):
            with sandbox.execute_in_sandbox("test_plugin", timeout_code):
                pass

    def test_sandbox_memory_limit(self):
        """Test sandbox memory limit."""
        sandbox = PluginSandbox(SandboxConfig(memory_limit=1))  # 1MB limit

        memory_code = '''
large_list = [0] * (1024 * 1024 * 10)  # 10MB list
result = len(large_list)
'''

        with pytest.raises(MemoryError):
            with sandbox.execute_in_sandbox("test_plugin", memory_code):
                pass

    def test_sandbox_path_restrictions(self):
        """Test sandbox path restrictions."""
        sandbox = PluginSandbox(SandboxConfig(
            allowed_paths=['/tmp'],
            blocked_paths=['/etc', '/bin']
        ))

        blocked_path_code = '''
with open("/etc/passwd", "r") as f:
    content = f.read()
'''

        with pytest.raises(PermissionError):
            with sandbox.execute_in_sandbox("test_plugin", blocked_path_code):
                pass

    def test_sandbox_import_restrictions(self):
        """Test sandbox import restrictions."""
        sandbox = PluginSandbox(SandboxConfig(
            blocked_imports=['os', 'subprocess', 'sys']
        ))

        import_code = '''
import os
import subprocess
result = "imports_worked"
'''

        with pytest.raises(ImportError):
            with sandbox.execute_in_sandbox("test_plugin", import_code):
                pass


class TestPermissionManager:
    """Test permission management."""

    def test_grant_revoke_permission(self):
        """Test granting and revoking permissions."""
        pm = PermissionManager()

        # Grant permission
        pm.grant_permission("test_plugin", PluginPermission.EXECUTE, "audio")

        assert pm.check_permission("test_plugin", PluginPermission.EXECUTE, "audio")

        # Revoke permission
        pm.revoke_permission("test_plugin", PluginPermission.EXECUTE, "audio")

        assert not pm.check_permission("test_plugin", PluginPermission.EXECUTE, "audio")

    def test_global_permissions(self):
        """Test global permissions."""
        pm = PermissionManager()

        # Grant global permission
        pm.grant_permission("test_plugin", PluginPermission.READ)

        assert pm.check_permission("test_plugin", PluginPermission.READ, "any_resource")
        assert pm.check_permission("test_plugin", PluginPermission.READ, "specific_resource")

    def test_resource_specific_permissions(self):
        """Test resource-specific permissions."""
        pm = PermissionManager()

        # Grant resource-specific permission
        pm.grant_permission("test_plugin", PluginPermission.WRITE, "audio")

        assert pm.check_permission("test_plugin", PluginPermission.WRITE, "audio")
        assert not pm.check_permission("test_plugin", PluginPermission.WRITE, "video")

    def test_get_plugin_permissions(self):
        """Test getting plugin permissions."""
        pm = PermissionManager()

        # Grant multiple permissions
        pm.grant_permission("test_plugin", PluginPermission.READ)
        pm.grant_permission("test_plugin", PluginPermission.EXECUTE, "audio")
        pm.grant_permission("test_plugin", PluginPermission.WRITE, "video")

        permissions = pm.get_plugin_permissions("test_plugin")
        assert PluginPermission.READ in permissions
        assert len(permissions) == 1  # Global permissions only

        resource_permissions = pm.get_resource_permissions("test_plugin", "audio")
        assert PluginPermission.EXECUTE in resource_permissions

    def test_clear_plugin_permissions(self):
        """Test clearing plugin permissions."""
        pm = PermissionManager()

        # Grant permissions
        pm.grant_permission("test_plugin", PluginPermission.READ)
        pm.grant_permission("test_plugin", PluginPermission.EXECUTE, "audio")

        assert pm.check_permission("test_plugin", PluginPermission.READ)

        # Clear permissions
        pm.clear_plugin_permissions("test_plugin")

        assert not pm.check_permission("test_plugin", PluginPermission.READ)
        assert not pm.check_permission("test_plugin", PluginPermission.EXECUTE, "audio")


class TestSecurityValidator:
    """Test security validation."""

    def test_validate_safe_plugin(self):
        """Test validating safe plugin."""
        validator = SecurityValidator()
        plugin = MockPlugin("safe_plugin")

        violations = validator.validate_plugin(plugin)

        # Should have no violations for basic plugin
        assert len(violations) == 0

    def test_validate_plugin_with_long_name(self):
        """Test validating plugin with suspicious name."""
        validator = SecurityValidator()
        plugin = MockPlugin("a" * 150)  # Very long name

        violations = validator.validate_plugin(plugin)

        # Should detect long name
        long_name_violations = [v for v in violations if "long" in v.message.lower()]
        assert len(long_name_violations) > 0

    def test_validate_plugin_with_suspicious_name(self):
        """Test validating plugin with suspicious name."""
        validator = SecurityValidator()
        plugin = MockPlugin("suspicious..plugin")

        violations = validator.validate_plugin(plugin)

        # Should detect suspicious characters
        suspicious_violations = [v for v in violations if "suspicious" in v.message.lower()]
        assert len(suspicious_violations) > 0

    def test_validate_plugin_with_too_many_dependencies(self):
        """Test validating plugin with too many dependencies."""
        validator = SecurityValidator()
        plugin = MockPlugin("dependency_heavy_plugin")
        plugin.dependencies = ["dep" + str(i) for i in range(15)]  # Too many dependencies

        violations = validator.validate_plugin(plugin)

        # Should detect too many dependencies
        dep_violations = [v for v in violations if "dependencies" in v.message.lower()]
        assert len(dep_violations) > 0

    def test_validate_plugin_with_dangerous_permissions(self):
        """Test validating plugin with dangerous permissions."""
        validator = SecurityValidator()
        plugin = MockPlugin("dangerous_plugin")

        # Add dangerous permission combination
        from models.plugin import PluginPermission
        dangerous_perm = PluginPermission(PluginPermission.ADMIN)
        plugin.permissions = [dangerous_perm]

        violations = validator.validate_plugin(plugin)

        # Should detect dangerous permissions
        perm_violations = [v for v in violations if "permission" in v.message.lower()]
        assert len(perm_violations) > 0

    def test_validate_plugin_execution(self):
        """Test validating plugin execution permissions."""
        validator = SecurityValidator()

        # Grant permission
        validator.permission_manager.grant_permission("test_plugin", PluginPermission.EXECUTE, "audio")

        # Test valid execution
        assert validator.validate_plugin_execution("test_plugin", "execute", "audio")

        # Test invalid execution
        assert not validator.validate_plugin_execution("test_plugin", "execute", "video")

        # Test execution without permission
        assert not validator.validate_plugin_execution("no_permission_plugin", "execute", "audio")


class TestResourceMonitor:
    """Test resource monitoring."""

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        monitor = ResourceMonitor()

        # Get initial memory
        initial_memory = monitor.get_memory_usage()
        assert isinstance(initial_memory, int)

        # Memory should be non-negative
        assert initial_memory >= 0

    def test_cpu_usage_tracking(self):
        """Test CPU usage tracking."""
        monitor = ResourceMonitor()

        # Get CPU usage
        cpu_usage = monitor.get_cpu_usage()
        assert isinstance(cpu_usage, float)
        assert 0.0 <= cpu_usage <= 100.0

    def test_execution_monitoring(self):
        """Test execution monitoring context manager."""
        monitor = ResourceMonitor()

        # Test successful execution
        with monitor.monitor_execution():
            # Simulate some work
            data = [i * 2 for i in range(1000)]

        # Should not raise any exceptions

        # Test with memory-intensive operation
        with monitor.monitor_execution():
            large_list = [0] * 100000  # Create some memory pressure


class TestPluginSecurityManager:
    """Test main security manager."""

    def test_validate_and_sandbox_plugin(self):
        """Test plugin validation and sandboxing."""
        plugin = MockPlugin("test_plugin")

        # Test successful validation
        success = security_manager.validate_and_sandbox_plugin(plugin)
        assert success == True

        # Check permissions were set up
        assert security_manager.check_plugin_permission("test_plugin", PluginPermission.EXECUTE)

    def test_validate_and_sandbox_plugin_with_violations(self):
        """Test plugin validation with security violations."""
        plugin = MockPlugin("malicious_plugin")
        plugin.name = "a" * 200  # Very long name

        # Mock security violations
        with patch.object(security_manager.validator, 'validate_plugin') as mock_validate:
            mock_validate.return_value = [
                SecurityViolation(
                    threat_type=SecurityThreat.MALICIOUS_CODE,
                    severity="critical",
                    message="Critical security violation",
                    plugin_name="malicious_plugin"
                )
            ]

            success = security_manager.validate_and_sandbox_plugin(plugin)
            assert success == False

    def test_execute_plugin_code(self):
        """Test executing plugin code in sandbox."""
        safe_code = '''
result = []
for i in range(3):
    result.append(i * 2)
final_result = sum(result)
'''

        result = security_manager.execute_plugin_code("test_plugin", safe_code)
        assert result['final_result'] == 6

    def test_execute_plugin_code_with_error(self):
        """Test executing plugin code with error."""
        error_code = '''
import os
os.system("malicious_command")
'''

        with pytest.raises(Exception):
            security_manager.execute_plugin_code("test_plugin", error_code)

    def test_get_security_violations(self):
        """Test getting security violations."""
        # Clear any existing violations
        security_manager.clear_security_violations()

        # Create test violation
        violation = SecurityViolation(
            threat_type=SecurityThreat.MALICIOUS_CODE,
            severity="medium",
            message="Test violation",
            plugin_name="test_plugin"
        )

        security_manager._security_violations.append(violation)

        # Get violations
        violations = security_manager.get_security_violations("test_plugin")
        assert len(violations) == 1
        assert violations[0].message == "Test violation"

        # Get all violations
        all_violations = security_manager.get_security_violations()
        assert len(all_violations) == 1

    def test_clear_security_violations(self):
        """Test clearing security violations."""
        # Create test violation
        violation = SecurityViolation(
            threat_type=SecurityThreat.MALICIOUS_CODE,
            severity="medium",
            message="Test violation",
            plugin_name="test_plugin"
        )

        security_manager._security_violations.append(violation)
        assert len(security_manager._security_violations) == 1

        # Clear violations for specific plugin
        security_manager.clear_security_violations("test_plugin")
        assert len(security_manager._security_violations) == 0

    def test_get_security_report(self):
        """Test getting security report."""
        # Clear existing violations
        security_manager.clear_security_violations()

        # Create test violations
        violations = [
            SecurityViolation(
                threat_type=SecurityThreat.MALICIOUS_CODE,
                severity="high",
                message="High severity violation",
                plugin_name="test_plugin"
            ),
            SecurityViolation(
                threat_type=SecurityThreat.MALICIOUS_CODE,
                severity="medium",
                message="Medium severity violation",
                plugin_name="test_plugin"
            ),
            SecurityViolation(
                threat_type=SecurityThreat.RESOURCE_EXHAUSTION,
                severity="low",
                message="Low severity violation",
                plugin_name="other_plugin"
            )
        ]

        security_manager._security_violations.extend(violations)

        # Get report
        report = security_manager.get_security_report("test_plugin")

        assert report['total_violations'] == 2
        assert report['violations_by_severity']['high'] == 1
        assert report['violations_by_severity']['medium'] == 1
        assert report['violations_by_type']['malicious_code'] == 2
        assert report['plugins_affected'] == 2


class TestSecurityIntegration:
    """Test security integration with plugin system."""

    async def test_plugin_security_integration(self):
        """Test full security integration."""
        plugin = MockPlugin("integration_test_plugin")

        # Validate plugin
        violations = security_manager.validator.validate_plugin(plugin)
        assert len(violations) == 0

        # Setup sandbox
        success = security_manager.validate_and_sandbox_plugin(plugin)
        assert success == True

        # Check permissions
        assert security_manager.check_plugin_permission("integration_test_plugin", PluginPermission.EXECUTE)

        # Test safe code execution
        safe_code = '''
safe_result = 42
safe_list = [1, 2, 3, 4, 5]
safe_sum = sum(safe_list)
'''

        result = security_manager.execute_plugin_code("integration_test_plugin", safe_code)
        assert result['safe_result'] == 42
        assert result['safe_sum'] == 15

    async def test_plugin_security_violation_handling(self):
        """Test security violation handling."""
        plugin = MockPlugin("violation_test_plugin")

        # Test with dangerous code
        dangerous_code = '''
import os
os.system("echo 'this should be blocked'")
'''

        with pytest.raises(Exception):
            security_manager.execute_plugin_code("violation_test_plugin", dangerous_code)

        # Check that violation was recorded
        violations = security_manager.get_security_violations("violation_test_plugin")
        assert len(violations) > 0

    def test_security_levels(self):
        """Test security levels."""
        # Test that security levels are properly defined
        assert SecurityLevel.LOW == "low"
        assert SecurityLevel.MEDIUM == "medium"
        assert SecurityLevel.HIGH == "high"
        assert SecurityLevel.CRITICAL == "critical"

    def test_security_threats(self):
        """Test security threat types."""
        # Test that threat types are properly defined
        assert SecurityThreat.MALICIOUS_CODE == "malicious_code"
        assert SecurityThreat.UNAUTHORIZED_ACCESS == "unauthorized_access"
        assert SecurityThreat.RESOURCE_EXHAUSTION == "resource_exhaustion"
        assert SecurityThreat.CODE_INJECTION == "code_injection"
        assert SecurityThreat.FILE_SYSTEM_VIOLATION == "file_system_violation"
        assert SecurityThreat.NETWORK_VIOLATION == "network_violation"
        assert SecurityThreat.MEMORY_CORRUPTION == "memory_corruption"
        assert SecurityThreat.PRIVILEGE_ESCALATION == "privilege_escalation"