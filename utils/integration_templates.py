"""
Integration Templates for TTS System

This module provides template-based integration configuration management,
including pre-configured templates, validation, customization, and sharing capabilities.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml

from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator

from models.integration import (
    IntegrationConfig, IntegrationType, IntegrationTemplate,
    IntegrationTemplateDB, CloudStorageProvider, NotificationProvider,
    DatabaseProvider, APIProtocol
)
from utils.exceptions import ValidationError, IntegrationError


class TemplateMetadata(BaseModel):
    """Template metadata model"""
    version: str = Field("1.0.0", description="Template version")
    author: str = Field("", description="Template author")
    description: str = Field("", description="Template description")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    compatibility: List[str] = Field(default_factory=list, description="Compatible versions")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    support_url: Optional[str] = Field(None, description="Support URL")
    changelog: List[Dict[str, Any]] = Field(default_factory=list, description="Change log")


class IntegrationTemplateManager:
    """Manager for integration templates"""

    def __init__(self, db_session: Session, template_dir: Optional[str] = None):
        self.db = db_session
        self.template_dir = Path(template_dir) if template_dir else Path("templates/integrations")
        self.logger = logging.getLogger(__name__)

        # Ensure template directory exists
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Load built-in templates
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Load built-in integration templates"""
        # AWS S3 Template
        aws_s3_template = IntegrationTemplate(
            template_id="aws-s3-basic",
            name="AWS S3 Basic Storage",
            description="Basic AWS S3 cloud storage integration template",
            integration_type=IntegrationType.CLOUD_STORAGE,
            provider=CloudStorageProvider.AWS_S3.value,
            base_config=IntegrationConfig(
                name="AWS S3 Integration",
                description="AWS S3 cloud storage integration",
                integration_type=IntegrationType.CLOUD_STORAGE,
                provider=CloudStorageProvider.AWS_S3.value,
                credentials={
                    "access_key": "",
                    "secret_key": "",
                    "region": "us-east-1",
                    "endpoint_url": ""
                },
                settings={
                    "bucket_name": "",
                    "default_acl": "private",
                    "encryption": "AES256"
                },
                rate_limit=1000,
                timeout=30
            ),
            required_fields=["access_key", "secret_key", "region"],
            optional_fields=["endpoint_url"],
            validation_rules={
                "credentials.access_key": {"type": "string", "min_length": 16, "max_length": 128},
                "credentials.secret_key": {"type": "string", "min_length": 32, "max_length": 128},
                "credentials.region": {"type": "string", "pattern": "^[a-z]{2}-[a-z]+-\\d+$"}
            },
            default_settings={
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1
            },
            is_public=True,
            tags=["aws", "s3", "cloud-storage", "basic"],
            metadata={
                "provider": "Amazon Web Services",
                "service": "Simple Storage Service",
                "difficulty": "beginner"
            }
        )

        # Slack Template
        slack_template = IntegrationTemplate(
            template_id="slack-webhook",
            name="Slack Webhook Notifications",
            description="Slack webhook integration for notifications",
            integration_type=IntegrationType.NOTIFICATION,
            provider=NotificationProvider.SLACK.value,
            base_config=IntegrationConfig(
                name="Slack Integration",
                description="Slack webhook for notifications",
                integration_type=IntegrationType.NOTIFICATION,
                provider=NotificationProvider.SLACK.value,
                credentials={
                    "webhook_url": "",
                    "channel": "#general",
                    "username": "TTS Bot"
                },
                settings={
                    "message_format": "markdown",
                    "include_timestamp": True,
                    "include_metadata": False
                },
                rate_limit=100,
                timeout=10
            ),
            required_fields=["webhook_url"],
            optional_fields=["channel", "username"],
            validation_rules={
                "credentials.webhook_url": {
                    "type": "string",
                    "pattern": "^https://hooks\.slack\.com/workflows/.*$"
                }
            },
            default_settings={
                "timeout": 10,
                "retry_attempts": 2,
                "retry_delay": 1
            },
            is_public=True,
            tags=["slack", "notification", "webhook", "messaging"],
            metadata={
                "provider": "Slack",
                "service": "Webhook",
                "difficulty": "beginner"
            }
        )

        # PostgreSQL Template
        postgresql_template = IntegrationTemplate(
            template_id="postgresql-basic",
            name="PostgreSQL Database",
            description="Basic PostgreSQL database integration template",
            integration_type=IntegrationType.DATABASE,
            provider=DatabaseProvider.POSTGRESQL.value,
            base_config=IntegrationConfig(
                name="PostgreSQL Integration",
                description="PostgreSQL database connection",
                integration_type=IntegrationType.DATABASE,
                provider=DatabaseProvider.POSTGRESQL.value,
                credentials={
                    "endpoint_url": "localhost",
                    "port": 5432,
                    "database": "tts_db",
                    "username": "",
                    "password": ""
                },
                settings={
                    "connection_pool_size": 10,
                    "max_overflow": 20,
                    "pool_timeout": 30,
                    "pool_recycle": 3600
                },
                rate_limit=500,
                timeout=30
            ),
            required_fields=["username", "password", "database"],
            optional_fields=["endpoint_url", "port"],
            validation_rules={
                "credentials.username": {"type": "string", "min_length": 1, "max_length": 63},
                "credentials.password": {"type": "string", "min_length": 8},
                "credentials.database": {"type": "string", "min_length": 1, "max_length": 63}
            },
            default_settings={
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1
            },
            is_public=True,
            tags=["postgresql", "database", "sql", "basic"],
            metadata={
                "provider": "PostgreSQL",
                "service": "Relational Database",
                "difficulty": "intermediate"
            }
        )

        # REST API Template
        rest_api_template = IntegrationTemplate(
            template_id="rest-api-basic",
            name="REST API Integration",
            description="Basic REST API integration template",
            integration_type=IntegrationType.API,
            provider=APIProtocol.REST.value,
            base_config=IntegrationConfig(
                name="REST API Integration",
                description="REST API connection",
                integration_type=IntegrationType.API,
                provider=APIProtocol.REST.value,
                credentials={
                    "endpoint_url": "",
                    "base_url": "",
                    "auth_type": "bearer",
                    "token": ""
                },
                settings={
                    "headers": {
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    "pagination": {
                        "enabled": False,
                        "page_param": "page",
                        "limit_param": "limit",
                        "max_pages": 10
                    },
                    "rate_limiting": {
                        "requests_per_second": 10,
                        "burst_limit": 100
                    }
                },
                rate_limit=1000,
                timeout=30
            ),
            required_fields=["endpoint_url"],
            optional_fields=["token", "base_url"],
            validation_rules={
                "credentials.endpoint_url": {
                    "type": "string",
                    "pattern": "^https?://.*"
                },
                "credentials.token": {
                    "type": "string",
                    "min_length": 10,
                    "max_length": 1000
                }
            },
            default_settings={
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1
            },
            is_public=True,
            tags=["rest", "api", "http", "basic"],
            metadata={
                "protocol": "REST",
                "service": "HTTP API",
                "difficulty": "intermediate"
            }
        )

        # Store built-in templates
        self._builtin_templates = {
            "aws-s3-basic": aws_s3_template,
            "slack-webhook": slack_template,
            "postgresql-basic": postgresql_template,
            "rest-api-basic": rest_api_template
        }

    async def create_template(self, template: IntegrationTemplate, user_id: Optional[int] = None) -> IntegrationTemplateDB:
        """Create a new integration template"""
        try:
            # Validate template
            await self._validate_template(template)

            # Check if template ID already exists
            existing = self.db.query(IntegrationTemplateDB).filter_by(template_id=template.template_id).first()
            if existing:
                raise ValidationError(f"Template ID already exists: {template.template_id}")

            # Create database record
            template_db = IntegrationTemplateDB(
                template_id=template.template_id,
                name=template.name,
                description=template.description,
                integration_type=template.integration_type,
                provider=template.provider,
                base_config=json.dumps(template.base_config.dict()),
                required_fields=json.dumps(template.required_fields),
                optional_fields=json.dumps(template.optional_fields),
                validation_rules=json.dumps(template.validation_rules),
                default_settings=json.dumps(template.default_settings),
                is_public=template.is_public,
                tags=json.dumps(template.tags),
                created_by=user_id
            )

            self.db.add(template_db)
            self.db.commit()
            self.db.refresh(template_db)

            # Save template file
            await self._save_template_file(template)

            self.logger.info(f"Template created: {template.template_id}")
            return template_db

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to create template: {str(e)}")
            raise IntegrationError(f"Failed to create template: {str(e)}")

    async def get_template(self, template_id: str) -> Optional[IntegrationTemplate]:
        """Get template by ID"""
        try:
            # First check built-in templates
            if template_id in self._builtin_templates:
                return self._builtin_templates[template_id]

            # Check database
            template_db = self.db.query(IntegrationTemplateDB).filter_by(template_id=template_id).first()
            if template_db:
                return await self._db_to_template(template_db)

            # Check template files
            template_file = self.template_dir / f"{template_id}.yaml"
            if template_file.exists():
                return await self._load_template_from_file(template_file)

            return None

        except Exception as e:
            self.logger.error(f"Failed to get template {template_id}: {str(e)}")
            raise IntegrationError(f"Failed to get template: {str(e)}")

    async def list_templates(self, integration_type: Optional[IntegrationType] = None,
                           provider: Optional[str] = None, is_public: Optional[bool] = None,
                           tags: Optional[List[str]] = None) -> List[IntegrationTemplate]:
        """List templates with optional filters"""
        try:
            templates = []

            # Add built-in templates
            for template in self._builtin_templates.values():
                if self._matches_filters(template, integration_type, provider, is_public, tags):
                    templates.append(template)

            # Add database templates
            query = self.db.query(IntegrationTemplateDB)
            if integration_type:
                query = query.filter_by(integration_type=integration_type)
            if provider:
                query = query.filter_by(provider=provider)
            if is_public is not None:
                query = query.filter_by(is_public=is_public)

            db_templates = query.all()
            for template_db in db_templates:
                template = await self._db_to_template(template_db)
                if self._matches_filters(template, integration_type, provider, is_public, tags):
                    templates.append(template)

            # Add file-based templates
            for template_file in self.template_dir.glob("*.yaml"):
                if not template_file.name.startswith('_'):
                    template = await self._load_template_from_file(template_file)
                    if template and self._matches_filters(template, integration_type, provider, is_public, tags):
                        templates.append(template)

            return templates

        except Exception as e:
            self.logger.error(f"Failed to list templates: {str(e)}")
            raise IntegrationError(f"Failed to list templates: {str(e)}")

    async def update_template(self, template_id: str, template: IntegrationTemplate,
                            user_id: Optional[int] = None) -> IntegrationTemplateDB:
        """Update an existing template"""
        try:
            # Check if template exists
            template_db = self.db.query(IntegrationTemplateDB).filter_by(template_id=template_id).first()
            if not template_db:
                raise IntegrationError(f"Template not found: {template_id}")

            # Validate template
            await self._validate_template(template)

            # Update database record
            template_db.name = template.name
            template_db.description = template.description
            template_db.provider = template.provider
            template_db.base_config = json.dumps(template.base_config.dict())
            template_db.required_fields = json.dumps(template.required_fields)
            template_db.optional_fields = json.dumps(template.optional_fields)
            template_db.validation_rules = json.dumps(template.validation_rules)
            template_db.default_settings = json.dumps(template.default_settings)
            template_db.is_public = template.is_public
            template_db.tags = json.dumps(template.tags)
            template_db.updated_at = datetime.utcnow()

            self.db.commit()

            # Update template file
            await self._save_template_file(template)

            self.logger.info(f"Template updated: {template_id}")
            return template_db

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to update template {template_id}: {str(e)}")
            raise IntegrationError(f"Failed to update template: {str(e)}")

    async def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        try:
            # Delete from database
            template_db = self.db.query(IntegrationTemplateDB).filter_by(template_id=template_id).first()
            if template_db:
                self.db.delete(template_db)
                self.db.commit()

            # Delete template file
            template_file = self.template_dir / f"{template_id}.yaml"
            if template_file.exists():
                template_file.unlink()

            self.logger.info(f"Template deleted: {template_id}")
            return True

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to delete template {template_id}: {str(e)}")
            raise IntegrationError(f"Failed to delete template: {str(e)}")

    async def create_integration_from_template(self, template_id: str,
                                             custom_config: Optional[Dict[str, Any]] = None,
                                             user_id: Optional[int] = None) -> IntegrationConfig:
        """Create integration configuration from template"""
        try:
            template = await self.get_template(template_id)
            if not template:
                raise IntegrationError(f"Template not found: {template_id}")

            # Start with base configuration
            config = template.base_config.copy()

            # Apply customizations
            if custom_config:
                config = await self._customize_config(config, custom_config)

            # Validate final configuration
            await self._validate_config_from_template(config, template)

            return config

        except Exception as e:
            self.logger.error(f"Failed to create integration from template {template_id}: {str(e)}")
            raise IntegrationError(f"Failed to create integration from template: {str(e)}")

    async def validate_template_config(self, template_id: str, config: IntegrationConfig) -> Dict[str, Any]:
        """Validate configuration against template"""
        try:
            template = await self.get_template(template_id)
            if not template:
                raise IntegrationError(f"Template not found: {template_id}")

            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'suggestions': []
            }

            # Check required fields
            for field in template.required_fields:
                if not self._get_nested_value(config.credentials.dict(), field):
                    validation_result['errors'].append(f"Required field missing: {field}")
                    validation_result['valid'] = False

            # Validate field formats
            for field_path, rules in template.validation_rules.items():
                value = self._get_nested_value(config.credentials.dict(), field_path)
                if value is not None:
                    field_errors = await self._validate_field_value(field_path, value, rules)
                    validation_result['errors'].extend(field_errors)

            # Check for unknown fields
            all_fields = set(template.required_fields + template.optional_fields)
            for field_path in self._get_all_field_paths(config.credentials.dict()):
                if field_path not in all_fields:
                    validation_result['warnings'].append(f"Unknown field: {field_path}")

            # Generate suggestions
            validation_result['suggestions'] = await self._generate_config_suggestions(config, template)

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate template config: {str(e)}")
            raise IntegrationError(f"Failed to validate template config: {str(e)}")

    async def _validate_template(self, template: IntegrationTemplate) -> None:
        """Validate template structure and content"""
        if not template.template_id or len(template.template_id) < 3:
            raise ValidationError("Template ID must be at least 3 characters long")

        if not template.name or len(template.name.strip()) == 0:
            raise ValidationError("Template name is required")

        if len(template.name) > 255:
            raise ValidationError("Template name too long (max 255 characters)")

        # Validate base configuration
        if not template.base_config:
            raise ValidationError("Base configuration is required")

        # Validate field lists don't overlap
        if set(template.required_fields) & set(template.optional_fields):
            raise ValidationError("Required and optional fields must not overlap")

        # Validate validation rules
        for field_path, rules in template.validation_rules.items():
            if not isinstance(rules, dict):
                raise ValidationError(f"Invalid validation rules for field: {field_path}")

    async def _validate_config_from_template(self, config: IntegrationConfig, template: IntegrationTemplate) -> None:
        """Validate configuration against template requirements"""
        # Check required fields
        for field in template.required_fields:
            value = self._get_nested_value(config.credentials.dict(), field)
            if not value or (isinstance(value, str) and not value.strip()):
                raise ValidationError(f"Required field '{field}' is missing or empty")

        # Validate field formats
        for field_path, rules in template.validation_rules.items():
            value = self._get_nested_value(config.credentials.dict(), field_path)
            if value is not None:
                errors = await self._validate_field_value(field_path, value, rules)
                if errors:
                    raise ValidationError(f"Field '{field_path}' validation failed: {', '.join(errors)}")

    async def _validate_field_value(self, field_path: str, value: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate individual field value against rules"""
        errors = []

        # Type validation
        expected_type = rules.get('type')
        if expected_type:
            if expected_type == 'string' and not isinstance(value, str):
                errors.append(f"Expected string, got {type(value).__name__}")
            elif expected_type == 'integer' and not isinstance(value, int):
                errors.append(f"Expected integer, got {type(value).__name__}")
            elif expected_type == 'boolean' and not isinstance(value, bool):
                errors.append(f"Expected boolean, got {type(value).__name__}")

        # String validations
        if isinstance(value, str):
            min_length = rules.get('min_length')
            if min_length is not None and len(value) < min_length:
                errors.append(f"Minimum length is {min_length}, got {len(value)}")

            max_length = rules.get('max_length')
            if max_length is not None and len(value) > max_length:
                errors.append(f"Maximum length is {max_length}, got {len(value)}")

            pattern = rules.get('pattern')
            if pattern and not __import__('re').match(pattern, value):
                errors.append(f"Value does not match required pattern: {pattern}")

        return errors

    async def _customize_config(self, base_config: IntegrationConfig, customizations: Dict[str, Any]) -> IntegrationConfig:
        """Apply customizations to base configuration"""
        # Deep merge customizations with base config
        config_dict = base_config.dict()

        def merge_dicts(base: Dict, custom: Dict) -> Dict:
            result = base.copy()
            for key, value in custom.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        customized_dict = merge_dicts(config_dict, customizations)
        return IntegrationConfig(**customized_dict)

    async def _generate_config_suggestions(self, config: IntegrationConfig, template: IntegrationTemplate) -> List[str]:
        """Generate configuration suggestions"""
        suggestions = []

        # Suggest default values for missing optional fields
        for field in template.optional_fields:
            value = self._get_nested_value(config.credentials.dict(), field)
            if not value:
                suggestions.append(f"Consider setting optional field: {field}")

        # Provider-specific suggestions
        if template.provider == CloudStorageProvider.AWS_S3.value:
            if not config.settings.get('encryption'):
                suggestions.append("Consider enabling encryption for AWS S3")
        elif template.provider == DatabaseProvider.POSTGRESQL.value:
            if config.credentials.get('endpoint_url') == 'localhost':
                suggestions.append("Consider using a remote database for production")

        return suggestions

    def _matches_filters(self, template: IntegrationTemplate, integration_type: Optional[IntegrationType],
                        provider: Optional[str], is_public: Optional[bool], tags: Optional[List[str]]) -> bool:
        """Check if template matches filter criteria"""
        if integration_type and template.integration_type != integration_type:
            return False
        if provider and template.provider != provider:
            return False
        if is_public is not None and template.is_public != is_public:
            return False
        if tags:
            template_tags = set(template.tags)
            filter_tags = set(tags)
            if not template_tags.intersection(filter_tags):
                return False
        return True

    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = field_path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _get_all_field_paths(self, data: Dict, prefix: str = "") -> List[str]:
        """Get all field paths from nested dictionary"""
        paths = []
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths.extend(self._get_all_field_paths(value, current_path))
            else:
                paths.append(current_path)
        return paths

    async def _db_to_template(self, template_db: IntegrationTemplateDB) -> IntegrationTemplate:
        """Convert database model to template object"""
        return IntegrationTemplate(
            template_id=template_db.template_id,
            name=template_db.name,
            description=template_db.description,
            integration_type=template_db.integration_type,
            provider=template_db.provider,
            base_config=IntegrationConfig(**json.loads(template_db.base_config)),
            required_fields=json.loads(template_db.required_fields),
            optional_fields=json.loads(template_db.optional_fields),
            validation_rules=json.loads(template_db.validation_rules),
            default_settings=json.loads(template_db.default_settings),
            is_public=template_db.is_public,
            tags=json.loads(template_db.tags),
            created_by=template_db.created_by,
            created_at=template_db.created_at
        )

    async def _save_template_file(self, template: IntegrationTemplate) -> None:
        """Save template to YAML file"""
        template_file = self.template_dir / f"{template.template_id}.yaml"

        template_dict = {
            'template_id': template.template_id,
            'name': template.name,
            'description': template.description,
            'integration_type': template.integration_type.value,
            'provider': template.provider,
            'base_config': template.base_config.dict(),
            'required_fields': template.required_fields,
            'optional_fields': template.optional_fields,
            'validation_rules': template.validation_rules,
            'default_settings': template.default_settings,
            'is_public': template.is_public,
            'tags': template.tags,
            'metadata': template.metadata
        }

        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(template_dict, f, default_flow_style=False, allow_unicode=True)

    async def _load_template_from_file(self, template_file: Path) -> Optional[IntegrationTemplate]:
        """Load template from YAML file"""
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            return IntegrationTemplate(
                template_id=data['template_id'],
                name=data['name'],
                description=data['description'],
                integration_type=IntegrationType(data['integration_type']),
                provider=data['provider'],
                base_config=IntegrationConfig(**data['base_config']),
                required_fields=data.get('required_fields', []),
                optional_fields=data.get('optional_fields', []),
                validation_rules=data.get('validation_rules', {}),
                default_settings=data.get('default_settings', {}),
                is_public=data.get('is_public', False),
                tags=data.get('tags', []),
                metadata=data.get('metadata', {})
            )

        except Exception as e:
            self.logger.error(f"Failed to load template from file {template_file}: {str(e)}")
            return None

    async def export_template(self, template_id: str, export_path: Optional[str] = None) -> str:
        """Export template to file"""
        try:
            template = await self.get_template(template_id)
            if not template:
                raise IntegrationError(f"Template not found: {template_id}")

            if not export_path:
                export_path = str(self.template_dir / f"{template_id}_export.yaml")

            await self._save_template_file(template)

            self.logger.info(f"Template exported: {template_id} -> {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"Failed to export template {template_id}: {str(e)}")
            raise IntegrationError(f"Failed to export template: {str(e)}")

    async def import_template(self, template_file: str, user_id: Optional[int] = None) -> IntegrationTemplateDB:
        """Import template from file"""
        try:
            template_path = Path(template_file)
            if not template_path.exists():
                raise IntegrationError(f"Template file not found: {template_file}")

            template = await self._load_template_from_file(template_path)
            if not template:
                raise IntegrationError(f"Failed to load template from file: {template_file}")

            return await self.create_template(template, user_id)

        except Exception as e:
            self.logger.error(f"Failed to import template from {template_file}: {str(e)}")
            raise IntegrationError(f"Failed to import template: {str(e)}")