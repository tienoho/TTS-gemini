"""
Integration Manager for TTS System

This module provides comprehensive integration lifecycle management,
credential handling, validation, and error recovery for external service integrations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Type, Callable
from contextlib import asynccontextmanager
import json
import hashlib
import secrets
import time

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import aiohttp
import redis

from models.integration import (
    IntegrationConfig, IntegrationStatus, IntegrationStatusInfo,
    IntegrationType, IntegrationDB, IntegrationLogDB, IntegrationAuditDB,
    CloudStorageProvider, NotificationProvider, DatabaseProvider, APIProtocol
)
from utils.integration_security import IntegrationSecurity
from utils.exceptions import IntegrationError, ValidationError, AuthenticationError


class IntegrationManager:
    """Main integration manager class"""

    def __init__(self, db_session: Session, redis_client: redis.Redis, security_manager: IntegrationSecurity):
        self.db = db_session
        self.redis = redis_client
        self.security = security_manager
        self.logger = logging.getLogger(__name__)
        self._active_integrations: Dict[int, Any] = {}
        self._rate_limiters: Dict[int, Any] = {}

    async def create_integration(self, config: IntegrationConfig, user_id: Optional[int] = None,
                               organization_id: Optional[int] = None) -> IntegrationDB:
        """Create a new integration"""
        try:
            # Validate configuration
            await self._validate_integration_config(config)

            # Encrypt credentials
            encrypted_credentials = await self.security.encrypt_credentials(config.credentials.dict())

            # Create database record
            integration_db = IntegrationDB(
                name=config.name,
                description=config.description,
                integration_type=config.integration_type,
                provider=config.provider,
                credentials=encrypted_credentials,
                settings=config.settings,
                rate_limit=config.rate_limit,
                timeout=config.timeout,
                retry_attempts=config.retry_attempts,
                retry_delay=config.retry_delay,
                is_active=config.is_active,
                tags=config.tags,
                metadata=config.metadata,
                created_by=user_id,
                organization_id=organization_id,
                status_info=IntegrationStatusInfo(status=IntegrationStatus.ACTIVE).dict()
            )

            self.db.add(integration_db)
            self.db.commit()
            self.db.refresh(integration_db)

            # Log audit trail
            await self._log_audit_event(
                integration_id=integration_db.id,
                user_id=user_id,
                action="create",
                new_values=config.dict()
            )

            # Initialize rate limiter if needed
            if config.rate_limit:
                await self._initialize_rate_limiter(integration_db.id, config.rate_limit)

            self.logger.info(f"Integration created: {integration_db.id} - {config.name}")
            return integration_db

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to create integration: {str(e)}")
            raise IntegrationError(f"Failed to create integration: {str(e)}")

    async def update_integration(self, integration_id: int, config: IntegrationConfig,
                               user_id: Optional[int] = None) -> IntegrationDB:
        """Update an existing integration"""
        try:
            integration = self.db.query(IntegrationDB).filter_by(id=integration_id).first()
            if not integration:
                raise IntegrationError("Integration not found")

            # Store old values for audit
            old_values = self._get_integration_dict(integration)

            # Validate new configuration
            await self._validate_integration_config(config)

            # Encrypt new credentials
            encrypted_credentials = await self.security.encrypt_credentials(config.credentials.dict())

            # Update integration
            integration.name = config.name
            integration.description = config.description
            integration.provider = config.provider
            integration.credentials = encrypted_credentials
            integration.settings = config.settings
            integration.rate_limit = config.rate_limit
            integration.timeout = config.timeout
            integration.retry_attempts = config.retry_attempts
            integration.retry_delay = config.retry_delay
            integration.is_active = config.is_active
            integration.tags = config.tags
            integration.metadata = config.metadata
            integration.updated_at = datetime.utcnow()

            self.db.commit()

            # Update rate limiter if needed
            if config.rate_limit:
                await self._initialize_rate_limiter(integration_id, config.rate_limit)
            else:
                await self._remove_rate_limiter(integration_id)

            # Log audit trail
            await self._log_audit_event(
                integration_id=integration_id,
                user_id=user_id,
                action="update",
                old_values=old_values,
                new_values=config.dict()
            )

            self.logger.info(f"Integration updated: {integration_id}")
            return integration

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to update integration {integration_id}: {str(e)}")
            raise IntegrationError(f"Failed to update integration: {str(e)}")

    async def delete_integration(self, integration_id: int, user_id: Optional[int] = None) -> bool:
        """Delete an integration"""
        try:
            integration = self.db.query(IntegrationDB).filter_by(id=integration_id).first()
            if not integration:
                raise IntegrationError("Integration not found")

            # Store values for audit before deletion
            old_values = self._get_integration_dict(integration)

            # Remove from active integrations
            await self._deactivate_integration(integration_id)

            # Delete from database
            self.db.delete(integration)
            self.db.commit()

            # Remove rate limiter
            await self._remove_rate_limiter(integration_id)

            # Log audit trail
            await self._log_audit_event(
                integration_id=integration_id,
                user_id=user_id,
                action="delete",
                old_values=old_values
            )

            self.logger.info(f"Integration deleted: {integration_id}")
            return True

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to delete integration {integration_id}: {str(e)}")
            raise IntegrationError(f"Failed to delete integration: {str(e)}")

    async def get_integration(self, integration_id: int) -> Optional[IntegrationDB]:
        """Get integration by ID"""
        try:
            integration = self.db.query(IntegrationDB).filter_by(id=integration_id).first()
            return integration
        except Exception as e:
            self.logger.error(f"Failed to get integration {integration_id}: {str(e)}")
            raise IntegrationError(f"Failed to get integration: {str(e)}")

    async def list_integrations(self, integration_type: Optional[IntegrationType] = None,
                              provider: Optional[str] = None, is_active: Optional[bool] = None,
                              organization_id: Optional[int] = None) -> List[IntegrationDB]:
        """List integrations with optional filters"""
        try:
            query = self.db.query(IntegrationDB)

            if integration_type:
                query = query.filter_by(integration_type=integration_type)
            if provider:
                query = query.filter_by(provider=provider)
            if is_active is not None:
                query = query.filter_by(is_active=is_active)
            if organization_id:
                query = query.filter_by(organization_id=organization_id)

            return query.all()
        except Exception as e:
            self.logger.error(f"Failed to list integrations: {str(e)}")
            raise IntegrationError(f"Failed to list integrations: {str(e)}")

    async def test_integration(self, integration_id: int) -> Dict[str, Any]:
        """Test integration connectivity and functionality"""
        try:
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationError("Integration not found")

            # Decrypt credentials
            credentials = await self.security.decrypt_credentials(integration.credentials)

            # Get integration type handler
            handler = await self._get_integration_handler(integration.integration_type)

            # Test connection
            test_result = await handler.test_connection(credentials, integration.settings)

            # Update status
            await self._update_integration_status(integration_id, test_result)

            # Log test result
            await self._log_integration_event(
                integration_id=integration_id,
                operation="test",
                status="success" if test_result["success"] else "error",
                message=test_result.get("message", ""),
                response_time_ms=test_result.get("response_time_ms")
            )

            return test_result

        except Exception as e:
            error_result = {
                "success": False,
                "message": str(e),
                "response_time_ms": None
            }
            await self._update_integration_status(integration_id, error_result)
            await self._log_integration_event(
                integration_id=integration_id,
                operation="test",
                status="error",
                message=str(e)
            )
            raise IntegrationError(f"Integration test failed: {str(e)}")

    async def activate_integration(self, integration_id: int) -> bool:
        """Activate an integration"""
        try:
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationError("Integration not found")

            integration.is_active = True
            integration.status_info = json.dumps(
                IntegrationStatusInfo(status=IntegrationStatus.ACTIVE).dict()
            )
            self.db.commit()

            # Initialize rate limiter if needed
            if integration.rate_limit:
                await self._initialize_rate_limiter(integration_id, integration.rate_limit)

            # Log audit trail
            await self._log_audit_event(
                integration_id=integration_id,
                action="enable"
            )

            self.logger.info(f"Integration activated: {integration_id}")
            return True

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to activate integration {integration_id}: {str(e)}")
            raise IntegrationError(f"Failed to activate integration: {str(e)}")

    async def deactivate_integration(self, integration_id: int) -> bool:
        """Deactivate an integration"""
        try:
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationError("Integration not found")

            integration.is_active = False
            integration.status_info = json.dumps(
                IntegrationStatusInfo(status=IntegrationStatus.INACTIVE).dict()
            )
            self.db.commit()

            # Remove from active integrations
            await self._deactivate_integration(integration_id)

            # Remove rate limiter
            await self._remove_rate_limiter(integration_id)

            # Log audit trail
            await self._log_audit_event(
                integration_id=integration_id,
                action="disable"
            )

            self.logger.info(f"Integration deactivated: {integration_id}")
            return True

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to deactivate integration {integration_id}: {str(e)}")
            raise IntegrationError(f"Failed to deactivate integration: {str(e)}")

    async def get_integration_status(self, integration_id: int) -> IntegrationStatusInfo:
        """Get detailed integration status"""
        try:
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationError("Integration not found")

            status_info = IntegrationStatusInfo(**integration.status_info)
            return status_info

        except Exception as e:
            self.logger.error(f"Failed to get integration status {integration_id}: {str(e)}")
            raise IntegrationError(f"Failed to get integration status: {str(e)}")

    async def execute_with_retry(self, integration_id: int, operation: str,
                               func: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic and rate limiting"""
        integration = await self.get_integration(integration_id)
        if not integration:
            raise IntegrationError("Integration not found")

        # Check rate limit
        if not await self._check_rate_limit(integration_id):
            raise IntegrationError("Rate limit exceeded")

        max_retries = integration.retry_attempts
        retry_delay = integration.retry_delay

        for attempt in range(max_retries + 1):
            try:
                # Execute operation
                start_time = time.time()
                result = await func(*args, **kwargs)
                response_time = int((time.time() - start_time) * 1000)

                # Update success status
                await self._update_request_success(integration_id, response_time)

                # Log successful operation
                await self._log_integration_event(
                    integration_id=integration_id,
                    operation=operation,
                    status="success",
                    response_time_ms=response_time
                )

                return result

            except Exception as e:
                # Update error status
                await self._update_request_error(integration_id, str(e))

                # Log error
                await self._log_integration_event(
                    integration_id=integration_id,
                    operation=operation,
                    status="error",
                    message=str(e)
                )

                if attempt < max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed for integration {integration_id}, retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise IntegrationError(f"Operation failed after {max_retries + 1} attempts: {str(e)}")

    async def _validate_integration_config(self, config: IntegrationConfig) -> None:
        """Validate integration configuration"""
        # Basic validation
        if not config.name or len(config.name.strip()) == 0:
            raise ValidationError("Integration name is required")

        if len(config.name) > 255:
            raise ValidationError("Integration name too long (max 255 characters)")

        # Provider-specific validation
        if config.integration_type == IntegrationType.CLOUD_STORAGE:
            await self._validate_cloud_storage_config(config)
        elif config.integration_type == IntegrationType.NOTIFICATION:
            await self._validate_notification_config(config)
        elif config.integration_type == IntegrationType.DATABASE:
            await self._validate_database_config(config)
        elif config.integration_type == IntegrationType.API:
            await self._validate_api_config(config)

    async def _validate_cloud_storage_config(self, config: IntegrationConfig) -> None:
        """Validate cloud storage configuration"""
        if config.provider not in [p.value for p in CloudStorageProvider]:
            raise ValidationError(f"Unsupported cloud storage provider: {config.provider}")

        if not config.credentials.endpoint_url:
            raise ValidationError("Endpoint URL is required for cloud storage")

    async def _validate_notification_config(self, config: IntegrationConfig) -> None:
        """Validate notification configuration"""
        if config.provider not in [p.value for p in NotificationProvider]:
            raise ValidationError(f"Unsupported notification provider: {config.provider}")

        if config.provider in ["slack", "discord", "teams"] and not config.credentials.token:
            raise ValidationError("Access token is required for notification provider")

    async def _validate_database_config(self, config: IntegrationConfig) -> None:
        """Validate database configuration"""
        if config.provider not in [p.value for p in DatabaseProvider]:
            raise ValidationError(f"Unsupported database provider: {config.provider}")

        if not config.credentials.endpoint_url:
            raise ValidationError("Database endpoint is required")

    async def _validate_api_config(self, config: IntegrationConfig) -> None:
        """Validate API configuration"""
        if config.provider not in [p.value for p in APIProtocol]:
            raise ValidationError(f"Unsupported API protocol: {config.provider}")

        if not config.credentials.endpoint_url:
            raise ValidationError("API endpoint is required")

    async def _get_integration_handler(self, integration_type: IntegrationType):
        """Get appropriate handler for integration type"""
        from utils.integration_types import (
            CloudStorageHandler, NotificationHandler,
            DatabaseHandler, APIHandler
        )

        handlers = {
            IntegrationType.CLOUD_STORAGE: CloudStorageHandler,
            IntegrationType.NOTIFICATION: NotificationHandler,
            IntegrationType.DATABASE: DatabaseHandler,
            IntegrationType.API: APIHandler,
        }

        handler_class = handlers.get(integration_type)
        if not handler_class:
            raise IntegrationError(f"No handler available for integration type: {integration_type}")

        return handler_class(self)

    async def _initialize_rate_limiter(self, integration_id: int, rate_limit: int) -> None:
        """Initialize rate limiter for integration"""
        # Simple Redis-based rate limiter
        self.redis.setex(
            f"rate_limit:{integration_id}",
            60,  # 1 minute window
            rate_limit
        )

    async def _check_rate_limit(self, integration_id: int) -> bool:
        """Check if integration is within rate limits"""
        key = f"rate_limit:{integration_id}"
        current = self.redis.get(key)

        if current is None:
            return True

        return int(current) > 0

    async def _decrement_rate_limit(self, integration_id: int) -> None:
        """Decrement rate limit counter"""
        key = f"rate_limit:{integration_id}"
        self.redis.decr(key)

    async def _remove_rate_limiter(self, integration_id: int) -> None:
        """Remove rate limiter for integration"""
        self.redis.delete(f"rate_limit:{integration_id}")

    async def _update_integration_status(self, integration_id: int, test_result: Dict[str, Any]) -> None:
        """Update integration status based on test result"""
        integration = await self.get_integration(integration_id)
        if not integration:
            return

        status_info = IntegrationStatusInfo(**integration.status_info)

        if test_result["success"]:
            status_info.status = IntegrationStatus.ACTIVE
            status_info.last_success = datetime.utcnow()
            status_info.last_check = datetime.utcnow()
            status_info.response_time_ms = test_result.get("response_time_ms")
        else:
            status_info.status = IntegrationStatus.ERROR
            status_info.last_error = datetime.utcnow()
            status_info.last_check = datetime.utcnow()
            status_info.error_message = test_result.get("message", "")

        integration.status_info = json.dumps(status_info.dict())
        self.db.commit()

    async def _update_request_success(self, integration_id: int, response_time_ms: int) -> None:
        """Update integration status after successful request"""
        integration = await self.get_integration(integration_id)
        if not integration:
            return

        status_info = IntegrationStatusInfo(**integration.status_info)
        status_info.total_requests += 1
        status_info.successful_requests += 1
        status_info.last_success = datetime.utcnow()
        status_info.last_check = datetime.utcnow()
        status_info.response_time_ms = response_time_ms

        # Calculate uptime percentage
        if status_info.total_requests > 0:
            status_info.uptime_percentage = (status_info.successful_requests / status_info.total_requests) * 100

        integration.status_info = json.dumps(status_info.dict())
        self.db.commit()

    async def _update_request_error(self, integration_id: int, error_message: str) -> None:
        """Update integration status after failed request"""
        integration = await self.get_integration(integration_id)
        if not integration:
            return

        status_info = IntegrationStatusInfo(**integration.status_info)
        status_info.total_requests += 1
        status_info.failed_requests += 1
        status_info.last_error = datetime.utcnow()
        status_info.last_check = datetime.utcnow()
        status_info.error_message = error_message

        # Calculate uptime percentage
        if status_info.total_requests > 0:
            status_info.uptime_percentage = (status_info.successful_requests / status_info.total_requests) * 100

        integration.status_info = json.dumps(status_info.dict())
        self.db.commit()

    async def _log_integration_event(self, integration_id: int, operation: str, status: str,
                                   message: Optional[str] = None, request_data: Optional[Dict] = None,
                                   response_data: Optional[Dict] = None, response_time_ms: Optional[int] = None,
                                   error_code: Optional[str] = None) -> None:
        """Log integration event"""
        try:
            log_entry = IntegrationLogDB(
                integration_id=integration_id,
                operation=operation,
                status=status,
                message=message,
                request_data=request_data,
                response_data=response_data,
                response_time_ms=response_time_ms,
                error_code=error_code
            )

            self.db.add(log_entry)
            self.db.commit()

        except Exception as e:
            self.logger.error(f"Failed to log integration event: {str(e)}")

    async def _log_audit_event(self, integration_id: int, user_id: Optional[int] = None,
                             action: str, old_values: Optional[Dict] = None,
                             new_values: Optional[Dict] = None, ip_address: Optional[str] = None,
                             user_agent: Optional[str] = None) -> None:
        """Log audit event"""
        try:
            audit_entry = IntegrationAuditDB(
                integration_id=integration_id,
                user_id=user_id,
                action=action,
                old_values=old_values,
                new_values=new_values,
                ip_address=ip_address,
                user_agent=user_agent
            )

            self.db.add(audit_entry)
            self.db.commit()

        except Exception as e:
            self.logger.error(f"Failed to log audit event: {str(e)}")

    def _get_integration_dict(self, integration: IntegrationDB) -> Dict[str, Any]:
        """Convert integration DB model to dictionary"""
        return {
            "id": integration.id,
            "name": integration.name,
            "description": integration.description,
            "integration_type": integration.integration_type.value,
            "provider": integration.provider,
            "settings": integration.settings,
            "rate_limit": integration.rate_limit,
            "timeout": integration.timeout,
            "retry_attempts": integration.retry_attempts,
            "retry_delay": integration.retry_delay,
            "is_active": integration.is_active,
            "tags": integration.tags,
            "metadata": integration.metadata,
            "created_at": integration.created_at.isoformat() if integration.created_at else None,
            "updated_at": integration.updated_at.isoformat() if integration.updated_at else None,
            "created_by": integration.created_by,
            "organization_id": integration.organization_id
        }

    async def cleanup_inactive_integrations(self) -> int:
        """Clean up inactive integrations and their resources"""
        try:
            # Find inactive integrations older than 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            inactive_integrations = self.db.query(IntegrationDB).filter(
                IntegrationDB.is_active == False,
                IntegrationDB.updated_at < cutoff_date
            ).all()

            cleaned_count = 0
            for integration in inactive_integrations:
                await self.delete_integration(integration.id)
                cleaned_count += 1

            self.logger.info(f"Cleaned up {cleaned_count} inactive integrations")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup inactive integrations: {str(e)}")
            raise IntegrationError(f"Failed to cleanup inactive integrations: {str(e)}")


@asynccontextmanager
async def integration_context(integration_id: int, manager: IntegrationManager):
    """Context manager for integration operations"""
    try:
        integration = await manager.get_integration(integration_id)
        if not integration or not integration.is_active:
            raise IntegrationError("Integration not available")

        yield integration
    except Exception as e:
        await manager._log_integration_event(
            integration_id=integration_id,
            operation="context_error",
            status="error",
            message=str(e)
        )
        raise