"""
API Version Models cho TTS System

Xử lý version tracking, compatibility matrix, migration paths và deprecation notices.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from packaging import version
import json


class VersionStatus(str, Enum):
    """Trạng thái của API version"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    DEVELOPMENT = "development"


class BreakingChangeType(str, Enum):
    """Loại breaking changes"""
    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes, backward compatible


class MigrationType(str, Enum):
    """Loại migration"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    BREAKING = "breaking"


class DeprecationNotice(BaseModel):
    """Thông báo deprecated"""
    version: str = Field(..., description="Version bị deprecated")
    replacement_version: Optional[str] = Field(None, description="Version thay thế")
    deprecation_date: datetime = Field(..., description="Ngày deprecated")
    sunset_date: Optional[datetime] = Field(None, description="Ngày sunset")
    migration_guide: Optional[str] = Field(None, description="Hướng dẫn migration")
    breaking_changes: List[str] = Field(default_factory=list, description="Breaking changes")


class VersionCompatibility(BaseModel):
    """Ma trận tương thích giữa các version"""
    current_version: str = Field(..., description="Version hiện tại")
    compatible_versions: List[str] = Field(default_factory=list, description="Các version tương thích")
    incompatible_versions: List[str] = Field(default_factory=list, description="Các version không tương thích")
    migration_required: bool = Field(default=False, description="Cần migration")
    migration_type: MigrationType = Field(default=MigrationType.AUTOMATIC, description="Loại migration")


class APIVersion(BaseModel):
    """Model đại diện cho một API version"""
    version: str = Field(..., description="Version string (semantic versioning)")
    status: VersionStatus = Field(default=VersionStatus.ACTIVE, description="Trạng thái version")
    release_date: datetime = Field(..., description="Ngày release")
    description: str = Field(..., description="Mô tả version")
    changelog: List[str] = Field(default_factory=list, description="Danh sách thay đổi")
    breaking_changes: List[str] = Field(default_factory=list, description="Breaking changes")
    new_features: List[str] = Field(default_factory=list, description="Tính năng mới")
    deprecated_features: List[str] = Field(default_factory=list, description="Tính năng deprecated")
    compatibility_matrix: Dict[str, VersionCompatibility] = Field(default_factory=dict, description="Ma trận tương thích")
    migration_paths: Dict[str, str] = Field(default_factory=dict, description="Đường dẫn migration")
    feature_flags: Dict[str, Any] = Field(default_factory=dict, description="Feature flags")
    sunset_date: Optional[datetime] = Field(None, description="Ngày sunset")

    @validator('version')
    def validate_version(cls, v):
        """Validate semantic versioning format"""
        try:
            version.parse(v)
            return v
        except Exception:
            raise ValueError(f"Invalid semantic version format: {v}")

    def is_compatible_with(self, other_version: str) -> bool:
        """Kiểm tra tương thích với version khác"""
        if other_version in self.compatibility_matrix:
            return other_version in self.compatibility_matrix[other_version].compatible_versions
        return False

    def get_migration_path(self, target_version: str) -> Optional[str]:
        """Lấy đường dẫn migration đến version đích"""
        return self.migration_paths.get(target_version)

    def is_deprecated(self) -> bool:
        """Kiểm tra version có bị deprecated không"""
        return self.status == VersionStatus.DEPRECATED

    def is_sunset(self) -> bool:
        """Kiểm tra version có bị sunset không"""
        return self.status == VersionStatus.SUNSET

    def days_until_sunset(self) -> Optional[int]:
        """Số ngày còn lại đến khi sunset"""
        if not self.sunset_date:
            return None
        remaining = self.sunset_date - datetime.now(self.sunset_date.tzinfo)
        return max(0, remaining.days)


class VersionMigration(BaseModel):
    """Model cho migration giữa các version"""
    from_version: str = Field(..., description="Version nguồn")
    to_version: str = Field(..., description="Version đích")
    migration_type: MigrationType = Field(..., description="Loại migration")
    breaking_changes: List[str] = Field(default_factory=list, description="Breaking changes")
    migration_steps: List[str] = Field(default_factory=list, description="Các bước migration")
    estimated_duration: Optional[int] = Field(None, description="Thời gian ước tính (phút)")
    rollback_possible: bool = Field(default=True, description="Có thể rollback")
    requires_downtime: bool = Field(default=False, description="Cần downtime")
    pre_migration_checks: List[str] = Field(default_factory=list, description="Kiểm tra trước migration")
    post_migration_checks: List[str] = Field(default_factory=list, description="Kiểm tra sau migration")
    created_at: datetime = Field(default_factory=datetime.now, description="Thời gian tạo")


class APIVersionRegistry(BaseModel):
    """Registry quản lý tất cả API versions"""
    versions: Dict[str, APIVersion] = Field(default_factory=dict, description="Danh sách versions")
    current_version: str = Field(..., description="Version hiện tại")
    default_version: str = Field(..., description="Version mặc định")
    deprecation_notices: List[DeprecationNotice] = Field(default_factory=list, description="Thông báo deprecated")
    migration_registry: Dict[str, VersionMigration] = Field(default_factory=dict, description="Registry migration")

    def get_version(self, version_str: str) -> Optional[APIVersion]:
        """Lấy version theo string"""
        return self.versions.get(version_str)

    def get_current_version(self) -> APIVersion:
        """Lấy version hiện tại"""
        return self.versions[self.current_version]

    def get_latest_version(self) -> APIVersion:
        """Lấy version mới nhất"""
        latest = max(self.versions.keys(), key=lambda v: version.parse(v))
        return self.versions[latest]

    def get_active_versions(self) -> List[APIVersion]:
        """Lấy danh sách version đang active"""
        return [v for v in self.versions.values() if v.status == VersionStatus.ACTIVE]

    def get_deprecated_versions(self) -> List[APIVersion]:
        """Lấy danh sách version deprecated"""
        return [v for v in self.versions.values() if v.status == VersionStatus.DEPRECATED]

    def add_version(self, api_version: APIVersion):
        """Thêm version mới"""
        self.versions[api_version.version] = api_version

    def deprecate_version(self, version_str: str, replacement_version: Optional[str] = None,
                         sunset_days: int = 90) -> DeprecationNotice:
        """Deprecated một version"""
        if version_str not in self.versions:
            raise ValueError(f"Version {version_str} không tồn tại")

        version_obj = self.versions[version_str]
        version_obj.status = VersionStatus.DEPRECATED

        notice = DeprecationNotice(
            version=version_str,
            replacement_version=replacement_version,
            deprecation_date=datetime.now(),
            sunset_date=datetime.now() + timedelta(days=sunset_days),
            breaking_changes=version_obj.breaking_changes
        )

        self.deprecation_notices.append(notice)
        return notice

    def sunset_version(self, version_str: str):
        """Sunset một version"""
        if version_str not in self.versions:
            raise ValueError(f"Version {version_str} không tồn tại")

        version_obj = self.versions[version_str]
        version_obj.status = VersionStatus.SUNSET

    def add_migration_path(self, migration: VersionMigration):
        """Thêm migration path"""
        key = f"{migration.from_version}_to_{migration.to_version}"
        self.migration_registry[key] = migration

    def get_migration_path(self, from_version: str, to_version: str) -> Optional[VersionMigration]:
        """Lấy migration path giữa hai version"""
        key = f"{from_version}_to_{to_version}"
        return self.migration_registry.get(key)

    def validate_version_compatibility(self, from_version: str, to_version: str) -> bool:
        """Validate tương thích giữa hai version"""
        if from_version not in self.versions or to_version not in self.versions:
            return False

        from_ver = self.versions[from_version]
        return from_ver.is_compatible_with(to_version)


class VersionedResponse(BaseModel):
    """Response wrapper với version information"""
    data: Any = Field(..., description="Response data")
    version: str = Field(..., description="API version")
    status: str = Field(..., description="Version status")
    deprecation_notice: Optional[DeprecationNotice] = Field(None, description="Thông báo deprecated")
    migration_info: Optional[Dict[str, Any]] = Field(None, description="Thông tin migration")
    feature_flags: Dict[str, Any] = Field(default_factory=dict, description="Feature flags")

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VersionedRequest(BaseModel):
    """Request wrapper với version information"""
    version: Optional[str] = Field(None, description="API version được request")
    data: Dict[str, Any] = Field(default_factory=dict, description="Request data")
    feature_flags: Dict[str, Any] = Field(default_factory=dict, description="Feature flags")
    migration_options: Optional[Dict[str, Any]] = Field(None, description="Migration options")


# Utility functions
def parse_version_string(version_str: str) -> version.Version:
    """Parse version string thành Version object"""
    return version.parse(version_str)


def compare_versions(v1: str, v2: str) -> int:
    """So sánh hai version (-1: v1 < v2, 0: v1 == v2, 1: v1 > v2)"""
    return (parse_version_string(v1) > parse_version_string(v2)) - (parse_version_string(v1) < parse_version_string(v2))


def is_version_compatible(current: str, target: str, registry: APIVersionRegistry) -> bool:
    """Kiểm tra version có tương thích không"""
    return registry.validate_version_compatibility(current, target)


def get_version_status(version_str: str, registry: APIVersionRegistry) -> Optional[VersionStatus]:
    """Lấy trạng thái của version"""
    version_obj = registry.get_version(version_str)
    return version_obj.status if version_obj else None