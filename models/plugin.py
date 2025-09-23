"""
Plugin models for TTS system with production-ready features
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Float, JSON, Index, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class PluginStatus(str, Enum):
    """Plugin lifecycle states."""
    PENDING = "pending"
    LOADING = "loading"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    UNINSTALLING = "uninstalling"


class PluginPermission(str, Enum):
    """Plugin permission levels."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class PluginType(str, Enum):
    """Plugin types."""
    TTS = "tts"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    WEBHOOK = "webhook"
    INTEGRATION = "integration"
    CUSTOM = "custom"


class Plugin(Base):
    """Plugin model for storing plugin information with production features."""

    __tablename__ = 'plugins'

    # Primary key and basic info
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False, default="1.0.0")

    # Plugin metadata
    plugin_type = Column(SQLEnum(PluginType), default=PluginType.CUSTOM, index=True)
    status = Column(SQLEnum(PluginStatus), default=PluginStatus.PENDING, index=True)
    author = Column(String(100), nullable=True)
    homepage = Column(String(255), nullable=True)
    license = Column(String(100), nullable=True)

    # Security and trust
    is_official = Column(Boolean, default=False, index=True)
    is_verified = Column(Boolean, default=False, index=True)
    security_hash = Column(String(255), nullable=True)  # Hash of plugin code for integrity
    signature = Column(Text, nullable=True)  # Digital signature for verification

    # Installation info
    installed_at = Column(DateTime, default=datetime.utcnow, index=True)
    installed_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    installation_path = Column(String(500), nullable=True)

    # Configuration
    config_schema = Column(JSON, default=dict)  # JSON schema for plugin configuration
    default_config = Column(JSON, default=dict)
    current_config = Column(JSON, default=dict)

    # Performance and limits
    memory_limit = Column(Integer, default=100)  # MB
    timeout_limit = Column(Integer, default=30)  # seconds
    execution_count = Column(Integer, default=0)
    last_executed_at = Column(DateTime, nullable=True)

    # Dependencies
    dependencies = Column(JSON, default=list)  # List of required plugins/packages
    python_requirements = Column(Text, nullable=True)  # pip requirements

    # Error tracking
    last_error = Column(Text, nullable=True)
    error_count = Column(Integer, default=0)
    last_error_at = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    updated_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Relationships
    versions = relationship("PluginVersion", back_populates="plugin", cascade="all, delete-orphan")
    permissions = relationship("PluginPermission", back_populates="plugin", cascade="all, delete-orphan")
    logs = relationship("PluginLog", back_populates="plugin", cascade="all, delete-orphan")
    dependencies_rel = relationship("PluginDependency", back_populates="plugin", cascade="all, delete-orphan")

    # Self-referential relationships
    created_by_user = relationship("User", remote_side=[id], foreign_keys=[created_by])
    updated_by_user = relationship("User", remote_side=[id], foreign_keys=[updated_by])
    installed_by_user = relationship("User", remote_side=[id], foreign_keys=[installed_by])

    # Indexes for performance
    __table_args__ = (
        Index('idx_plugins_status_type', 'status', 'plugin_type'),
        Index('idx_plugins_author', 'author'),
        Index('idx_plugins_installed_at', 'installed_at'),
        Index('idx_plugins_is_official', 'is_official'),
        Index('idx_plugins_is_verified', 'is_verified'),
    )

    def __init__(self, name: str, display_name: str, description: str = None, **kwargs):
        """Initialize plugin."""
        self.name = name
        self.display_name = display_name
        self.description = description
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of plugin."""
        return f"<Plugin(id={self.id}, name='{self.name}', status='{self.status}')>"

    def calculate_security_hash(self, code_content: str) -> str:
        """Calculate security hash of plugin code."""
        self.security_hash = hashlib.sha256(code_content.encode('utf-8')).hexdigest()
        return self.security_hash

    def verify_integrity(self, code_content: str) -> bool:
        """Verify plugin code integrity."""
        if not self.security_hash:
            return False
        return self.security_hash == hashlib.sha256(code_content.encode('utf-8')).hexdigest()

    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self.status == PluginStatus.ACTIVE

    def can_execute(self) -> bool:
        """Check if plugin can be executed."""
        return self.status in [PluginStatus.ACTIVE, PluginStatus.LOADING]

    def has_error(self) -> bool:
        """Check if plugin has errors."""
        return self.status == PluginStatus.ERROR

    def increment_execution_count(self) -> None:
        """Increment execution count."""
        self.execution_count += 1
        self.last_executed_at = datetime.utcnow()

    def set_error(self, error_message: str) -> None:
        """Set plugin error state."""
        self.status = PluginStatus.ERROR
        self.last_error = error_message
        self.error_count += 1
        self.last_error_at = datetime.utcnow()

    def clear_error(self) -> None:
        """Clear plugin error state."""
        self.last_error = None
        self.error_count = 0
        self.last_error_at = None

    def enable(self) -> None:
        """Enable plugin."""
        if self.status == PluginStatus.DISABLED:
            self.status = PluginStatus.ACTIVE

    def disable(self) -> None:
        """Disable plugin."""
        if self.status == PluginStatus.ACTIVE:
            self.status = PluginStatus.DISABLED

    def to_dict(self) -> dict:
        """Convert plugin to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'version': self.version,
            'plugin_type': self.plugin_type.value if self.plugin_type else None,
            'status': self.status.value if self.status else None,
            'author': self.author,
            'homepage': self.homepage,
            'license': self.license,
            'is_official': self.is_official,
            'is_verified': self.is_verified,
            'installed_at': self.installed_at.isoformat() if self.installed_at else None,
            'installed_by': self.installed_by,
            'installation_path': self.installation_path,
            'config_schema': self.config_schema,
            'default_config': self.default_config,
            'current_config': self.current_config,
            'memory_limit': self.memory_limit,
            'timeout_limit': self.timeout_limit,
            'execution_count': self.execution_count,
            'last_executed_at': self.last_executed_at.isoformat() if self.last_executed_at else None,
            'dependencies': self.dependencies,
            'python_requirements': self.python_requirements,
            'last_error': self.last_error,
            'error_count': self.error_count,
            'last_error_at': self.last_error_at.isoformat() if self.last_error_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Plugin':
        """Create plugin instance from dictionary."""
        return cls(
            name=data['name'],
            display_name=data['display_name'],
            description=data.get('description'),
            version=data.get('version', '1.0.0'),
            plugin_type=PluginType(data['plugin_type']) if data.get('plugin_type') else None,
            author=data.get('author'),
            homepage=data.get('homepage'),
            license=data.get('license'),
            is_official=data.get('is_official', False),
            is_verified=data.get('is_verified', False),
            config_schema=data.get('config_schema', {}),
            default_config=data.get('default_config', {}),
            current_config=data.get('current_config', {}),
            memory_limit=data.get('memory_limit', 100),
            timeout_limit=data.get('timeout_limit', 30),
            dependencies=data.get('dependencies', []),
            python_requirements=data.get('python_requirements')
        )

    @staticmethod
    def get_by_name(name: str, db_session) -> Optional['Plugin']:
        """Get plugin by name."""
        return db_session.query(Plugin).filter(Plugin.name == name).first()

    @staticmethod
    def get_active_plugins(db_session, plugin_type: Optional[PluginType] = None) -> list:
        """Get active plugins."""
        query = db_session.query(Plugin).filter(Plugin.status == PluginStatus.ACTIVE)
        if plugin_type:
            query = query.filter(Plugin.plugin_type == plugin_type)
        return query.all()

    @staticmethod
    def get_plugins_by_status(db_session, status: PluginStatus, limit: int = 100) -> list:
        """Get plugins by status."""
        return db_session.query(Plugin).filter(Plugin.status == status).limit(limit).all()

    @staticmethod
    def get_official_plugins(db_session, limit: int = 100) -> list:
        """Get official plugins."""
        return db_session.query(Plugin).filter(
            Plugin.is_official == True,
            Plugin.status == PluginStatus.ACTIVE
        ).limit(limit).all()


class PluginVersion(Base):
    """Plugin version model for version management."""

    __tablename__ = 'plugin_versions'

    id = Column(Integer, primary_key=True, index=True)
    plugin_id = Column(Integer, ForeignKey('plugins.id'), nullable=False, index=True)
    version = Column(String(50), nullable=False, index=True)
    changelog = Column(Text, nullable=True)
    release_notes = Column(Text, nullable=True)
    is_compatible = Column(Boolean, default=True, index=True)
    min_system_version = Column(String(50), nullable=True)
    max_system_version = Column(String(50), nullable=True)

    # Version metadata
    release_date = Column(DateTime, default=datetime.utcnow, index=True)
    download_url = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)  # bytes
    checksum = Column(String(255), nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)

    # Relationships
    plugin = relationship("Plugin", back_populates="versions")
    created_by_user = relationship("User", remote_side=[id], foreign_keys=[created_by])

    __table_args__ = (
        Index('idx_plugin_versions_plugin_version', 'plugin_id', 'version'),
        Index('idx_plugin_versions_compatible', 'is_compatible'),
    )

    def __repr__(self) -> str:
        """String representation of plugin version."""
        return f"<PluginVersion(id={self.id}, plugin_id={self.plugin_id}, version='{self.version}')>"


class PluginDependency(Base):
    """Plugin dependency model."""

    __tablename__ = 'plugin_dependencies'

    id = Column(Integer, primary_key=True, index=True)
    plugin_id = Column(Integer, ForeignKey('plugins.id'), nullable=False, index=True)
    dependency_name = Column(String(100), nullable=False, index=True)
    dependency_version = Column(String(50), nullable=False)
    dependency_type = Column(String(20), default="plugin", index=True)  # plugin, package, system

    # Dependency metadata
    is_required = Column(Boolean, default=True, index=True)
    is_resolved = Column(Boolean, default=False, index=True)
    resolved_version = Column(String(50), nullable=True)
    install_command = Column(String(255), nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    plugin = relationship("Plugin", back_populates="dependencies_rel")

    __table_args__ = (
        Index('idx_plugin_deps_plugin_name', 'plugin_id', 'dependency_name'),
        Index('idx_plugin_deps_resolved', 'is_resolved'),
    )

    def __repr__(self) -> str:
        """String representation of plugin dependency."""
        return f"<PluginDependency(id={self.id}, plugin_id={self.plugin_id}, name='{self.dependency_name}')>"


class PluginPermission(Base):
    """Plugin permission model."""

    __tablename__ = 'plugin_permissions'

    id = Column(Integer, primary_key=True, index=True)
    plugin_id = Column(Integer, ForeignKey('plugins.id'), nullable=False, index=True)
    permission = Column(SQLEnum(PluginPermission), nullable=False, index=True)
    resource = Column(String(255), nullable=False, index=True)  # e.g., "audio", "webhook", "file"
    conditions = Column(JSON, default=dict)  # Additional conditions for permission

    # Permission metadata
    is_granted = Column(Boolean, default=False, index=True)
    granted_at = Column(DateTime, nullable=True)
    granted_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    plugin = relationship("Plugin", back_populates="permissions")
    granted_by_user = relationship("User", remote_side=[id], foreign_keys=[granted_by])

    __table_args__ = (
        Index('idx_plugin_perms_plugin_resource', 'plugin_id', 'resource'),
        Index('idx_plugin_perms_granted', 'is_granted'),
    )

    def __repr__(self) -> str:
        """String representation of plugin permission."""
        return f"<PluginPermission(id={self.id}, plugin_id={self.plugin_id}, permission='{self.permission}')>"


class PluginLog(Base):
    """Plugin log model for tracking plugin activities."""

    __tablename__ = 'plugin_logs'

    id = Column(Integer, primary_key=True, index=True)
    plugin_id = Column(Integer, ForeignKey('plugins.id'), nullable=False, index=True)
    level = Column(String(20), default="INFO", index=True)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    details = Column(JSON, default=dict)
    source = Column(String(100), nullable=True)  # Function or module name

    # Context information
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    request_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)

    # Performance data
    execution_time = Column(Float, nullable=True)  # seconds
    memory_usage = Column(Integer, nullable=True)  # MB
    cpu_usage = Column(Float, nullable=True)  # percentage

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    plugin = relationship("Plugin", back_populates="logs")
    user = relationship("User", foreign_keys=[user_id])

    __table_args__ = (
        Index('idx_plugin_logs_plugin_level', 'plugin_id', 'level'),
        Index('idx_plugin_logs_created_at', 'created_at'),
        Index('idx_plugin_logs_user', 'user_id'),
    )

    def __repr__(self) -> str:
        """String representation of plugin log."""
        return f"<PluginLog(id={self.id}, plugin_id={self.plugin_id}, level='{self.level}')>"