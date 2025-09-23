"""
Database migration utilities for multi-tenant TTS system
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Boolean, ForeignKey, Float, JSON, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from models.organization import Organization, OrganizationStatus
from models.tenancy import (
    TenantAwareAudioRequest,
    TenantAwareAudioFile,
    TenantAwareRequestLog,
    tenant_security
)

Base = declarative_base()


def create_tenant_aware_tables(engine):
    """Create tenant-aware tables in the database."""
    # Create tables
    TenantAwareAudioRequest.__table__.create(engine, checkfirst=True)
    TenantAwareAudioFile.__table__.create(engine, checkfirst=True)
    TenantAwareRequestLog.__table__.create(engine, checkfirst=True)

    print("Tenant-aware tables created successfully")


def drop_tenant_aware_tables(engine):
    """Drop tenant-aware tables from the database."""
    # Drop tables in reverse order to handle foreign key constraints
    TenantAwareRequestLog.__table__.drop(engine, checkfirst=True)
    TenantAwareAudioFile.__table__.drop(engine, checkfirst=True)
    TenantAwareAudioRequest.__table__.drop(engine, checkfirst=True)

    print("Tenant-aware tables dropped successfully")


def migrate_existing_data_to_tenant_aware(db_session: Session, organization_id: int):
    """Migrate existing data to tenant-aware structure."""
    from models.audio_request import AudioRequest
    from models.audio_file import AudioFile
    from models.request_log import RequestLog

    print(f"Starting migration for organization {organization_id}")

    # Migrate audio requests
    existing_requests = db_session.query(AudioRequest).all()
    print(f"Migrating {len(existing_requests)} audio requests")

    for request in existing_requests:
        tenant_request = TenantAwareAudioRequest(
            organization_id=organization_id,
            text=request.text,
            voice_name=request.voice_name,
            output_format=request.output_format,
            speed=request.speed,
            pitch=request.pitch,
            status=request.status,
            file_path=request.file_path,
            file_url=request.file_url,
            file_size=request.file_size,
            duration_seconds=request.duration_seconds,
            cost=request.cost,
            cost_per_character=request.cost_per_character,
            processing_started_at=request.processing_started_at,
            processing_completed_at=request.processing_completed_at,
            error_message=request.error_message,
            user_id=request.user_id,
            created_at=request.created_at,
            updated_at=request.updated_at,
            created_by=request.created_by,
            updated_by=request.updated_by
        )
        db_session.add(tenant_request)

    # Migrate audio files
    existing_files = db_session.query(AudioFile).all()
    print(f"Migrating {len(existing_files)} audio files")

    for file in existing_files:
        tenant_file = TenantAwareAudioFile(
            organization_id=organization_id,
            filename=file.filename,
            original_filename=file.original_filename,
            file_path=file.file_path,
            file_url=file.file_url,
            file_size=file.file_size,
            duration_seconds=file.duration_seconds,
            mime_type=file.mime_type,
            metadata=file.metadata or {},
            tags=file.tags,
            user_id=file.user_id,
            storage_provider=file.storage_provider,
            storage_path=file.storage_path,
            created_at=file.created_at,
            updated_at=file.updated_at,
            created_by=file.created_by,
            updated_by=file.updated_by
        )
        db_session.add(tenant_file)

    # Migrate request logs
    existing_logs = db_session.query(RequestLog).all()
    print(f"Migrating {len(existing_logs)} request logs")

    for log in existing_logs:
        tenant_log = TenantAwareRequestLog(
            organization_id=organization_id,
            request_id=log.request_id,
            method=log.method,
            endpoint=log.endpoint,
            user_agent=log.user_agent,
            ip_address=log.ip_address,
            user_id=log.user_id,
            status_code=log.status_code,
            response_time_ms=log.response_time_ms,
            request_size=log.request_size,
            response_size=log.response_size,
            error_message=log.error_message,
            error_stack_trace=log.error_stack_trace,
            metadata=log.metadata or {},
            created_at=log.created_at,
            updated_at=log.updated_at,
            created_by=log.created_by,
            updated_by=log.updated_by
        )
        db_session.add(tenant_log)

    db_session.commit()
    print("Migration completed successfully")


def setup_tenant_security_for_session(session: Session):
    """Setup tenant security for database session."""
    # This function can be used to configure session-level security
    # For now, it's a placeholder for future security enhancements
    pass


def get_tenant_statistics(db_session: Session) -> Dict[str, Any]:
    """Get statistics across all tenants (admin function)."""
    with tenant_security.security_bypass():
        total_orgs = db_session.query(Organization).filter(
            Organization.status == OrganizationStatus.ACTIVE
        ).count()

        total_requests = db_session.query(TenantAwareAudioRequest).count()
        total_files = db_session.query(TenantAwareAudioFile).count()
        total_logs = db_session.query(TenantAwareRequestLog).count()

        return {
            'total_organizations': total_orgs,
            'total_requests': total_requests,
            'total_files': total_files,
            'total_logs': total_logs,
            'generated_at': datetime.utcnow().isoformat()
        }


def get_organization_usage_summary(db_session: Session, organization_id: int) -> Optional[Dict[str, Any]]:
    """Get usage summary for specific organization."""
    with tenant_security.security_bypass():
        org = db_session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return None

        request_count = db_session.query(TenantAwareAudioRequest).filter(
            TenantAwareAudioRequest.organization_id == organization_id
        ).count()

        file_count = db_session.query(TenantAwareAudioFile).filter(
            TenantAwareAudioFile.organization_id == organization_id
        ).count()

        # Calculate total cost
        cost_results = db_session.query(TenantAwareAudioRequest.cost).filter(
            TenantAwareAudioRequest.organization_id == organization_id
        ).all()
        total_cost = sum(cost[0] for cost in cost_results) if cost_results else 0.0

        return {
            'organization_id': organization_id,
            'organization_name': org.name,
            'total_requests': request_count,
            'total_files': file_count,
            'total_cost': total_cost,
            'current_month_requests': org.current_month_requests,
            'current_month_cost': org.current_month_cost,
            'max_monthly_requests': org.max_monthly_requests,
            'max_storage_bytes': org.max_storage_bytes,
            'current_storage_bytes': org.current_storage_bytes,
        }


def validate_tenant_data_integrity(db_session: Session) -> Dict[str, Any]:
    """Validate tenant data integrity across all organizations."""
    with tenant_security.security_bypass():
        issues = []

        # Check for orphaned records
        orphaned_requests = db_session.query(TenantAwareAudioRequest).filter(
            ~TenantAwareAudioRequest.organization_id.in_(
                db_session.query(Organization.id).filter(Organization.status == OrganizationStatus.ACTIVE)
            )
        ).count()

        if orphaned_requests > 0:
            issues.append(f"Found {orphaned_requests} orphaned audio requests")

        orphaned_files = db_session.query(TenantAwareAudioFile).filter(
            ~TenantAwareAudioFile.organization_id.in_(
                db_session.query(Organization.id).filter(Organization.status == OrganizationStatus.ACTIVE)
            )
        ).count()

        if orphaned_files > 0:
            issues.append(f"Found {orphaned_files} orphaned audio files")

        orphaned_logs = db_session.query(TenantAwareRequestLog).filter(
            ~TenantAwareRequestLog.organization_id.in_(
                db_session.query(Organization.id).filter(Organization.status == OrganizationStatus.ACTIVE)
            )
        ).count()

        if orphaned_logs > 0:
            issues.append(f"Found {orphaned_logs} orphaned request logs")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }


def cleanup_tenant_data(db_session: Session, organization_id: int) -> bool:
    """Clean up all tenant data for a specific organization."""
    with tenant_security.security_bypass():
        try:
            # Delete in order to handle foreign key constraints
            db_session.query(TenantAwareRequestLog).filter(
                TenantAwareRequestLog.organization_id == organization_id
            ).delete()

            db_session.query(TenantAwareAudioFile).filter(
                TenantAwareAudioFile.organization_id == organization_id
            ).delete()

            db_session.query(TenantAwareAudioRequest).filter(
                TenantAwareAudioRequest.organization_id == organization_id
            ).delete()

            db_session.commit()
            return True
        except Exception as e:
            db_session.rollback()
            print(f"Error cleaning up tenant data: {e}")
            return False


def create_database_indexes(engine):
    """Create additional indexes for tenant-aware tables."""
    with engine.connect() as conn:
        # Index for faster organization-based queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tenant_requests_org_status
            ON tenant_aware_audio_requests(organization_id, status)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tenant_files_org_mime
            ON tenant_aware_audio_files(organization_id, mime_type)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tenant_logs_org_endpoint
            ON tenant_aware_request_logs(organization_id, endpoint)
        """))

        conn.commit()
        print("Database indexes created successfully")


def run_migration(database_url: str, organization_id: int):
    """Run complete migration process."""
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        print("Starting multi-tenant migration...")

        # Create tables
        create_tenant_aware_tables(engine)

        # Create indexes
        create_database_indexes(engine)

        # Migrate existing data
        migrate_existing_data_to_tenant_aware(session, organization_id)

        # Validate migration
        validation_result = validate_tenant_data_integrity(session)
        if validation_result['is_valid']:
            print("Migration completed successfully!")
        else:
            print(f"Migration completed with issues: {validation_result['issues']}")

        return True

    except Exception as e:
        print(f"Migration failed: {e}")
        session.rollback()
        return False
    finally:
        session.close()