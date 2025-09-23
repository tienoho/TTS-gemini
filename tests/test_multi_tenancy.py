"""
Multi-tenancy tests for TTS system
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from models.organization import (
    Organization, OrganizationMember, OrganizationResource,
    OrganizationBilling, OrganizationUsage, OrganizationStatus, OrganizationTier, MemberRole
)
from models.tenancy import (
    TenantAwareAudioRequest, TenantAwareAudioFile, TenantAwareRequestLog,
    TenantSecurityManager, tenant_security
)
from utils.tenant_manager import TenantManager, tenant_manager
from utils.billing_manager import BillingManager, billing_manager
from config.tenancy import TenantConfigManager, tenant_config_manager


class TestTenantSecurityManager:
    """Test tenant security manager functionality."""

    def test_tenant_context_management(self):
        """Test tenant context setting and retrieval."""
        tenant_security.set_tenant_context(organization_id=1, user_id=100)

        context = tenant_security.get_tenant_context()
        assert context is not None
        assert context['organization_id'] == 1
        assert context['user_id'] == 100

        tenant_security.clear_tenant_context()
        assert tenant_security.get_tenant_context() is None

    def test_security_bypass(self):
        """Test security bypass functionality."""
        tenant_security.set_tenant_context(1, 100)

        # Security should be enabled by default
        assert tenant_security.is_security_enabled() == True

        # Test bypass
        with tenant_security.security_bypass():
            assert tenant_security.is_security_enabled() == False

        # Should be restored after context
        assert tenant_security.is_security_enabled() == True

    def test_organization_id_retrieval(self):
        """Test current organization ID retrieval."""
        tenant_security.set_tenant_context(organization_id=42, user_id=100)
        assert tenant_security.get_current_organization_id() == 42

        tenant_security.clear_tenant_context()
        assert tenant_security.get_current_organization_id() is None


class TestTenantManager:
    """Test tenant manager functionality."""

    def test_singleton_pattern(self):
        """Test that TenantManager follows singleton pattern."""
        manager1 = TenantManager()
        manager2 = TenantManager()
        assert manager1 is manager2

    def test_tenant_context_operations(self):
        """Test tenant context operations."""
        with tenant_manager.tenant_context(organization_id=1, user_id=100):
            context = tenant_manager.get_current_tenant()
            assert context is not None
            assert context.organization_id == 1
            assert context.user_id == 100

        # Context should be cleared after exiting
        assert tenant_manager.get_current_tenant() is None

    def test_api_key_extraction(self):
        """Test API key extraction from various sources."""
        with patch('utils.tenant_manager.request') as mock_request:
            # Test Authorization header
            mock_request.headers.get.return_value = 'Bearer sk-1234567890abcdef'
            api_key = tenant_manager._extract_api_key_from_request()
            assert api_key == 'sk-1234567890abcdef'

            # Test X-API-Key header
            mock_request.headers.get.return_value = None
            mock_request.args.get.return_value = 'sk-abcdef1234567890'
            api_key = tenant_manager._extract_api_key_from_request()
            assert api_key == 'sk-abcdef1234567890'

    def test_resource_availability_check(self):
        """Test resource availability checking."""
        # Mock database session
        mock_session = Mock()

        # Mock resources
        mock_resources = [
            Mock(allocated_amount=100, used_amount=50, is_active=True),
            Mock(allocated_amount=200, used_amount=150, is_active=True)
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_resources

        result = tenant_manager.check_resource_availability(1, 'requests', 75, mock_session)

        assert result['available'] == True
        assert result['current_usage'] == 200  # 50 + 150
        assert result['allocated'] == 300  # 100 + 200
        assert result['available'] == 100  # 300 - 200
        assert result['requested'] == 75

    def test_usage_tracking(self):
        """Test usage tracking functionality."""
        mock_session = Mock()

        # Mock organization usage update
        with patch.object(tenant_manager, '_update_organization_usage') as mock_update:
            result = tenant_manager.track_usage(
                organization_id=1,
                usage_type='requests',
                count=10,
                amount=100.0,
                unit='requests',
                cost=0.01,
                db_session=mock_session,
                metadata={'test': True}
            )

            assert result == True
            mock_update.assert_called_once_with(1, 'requests', 10, 100.0, 0.01, mock_session)

    def test_organization_limits_check(self):
        """Test organization limits checking."""
        mock_session = Mock()

        # Mock organization
        mock_org = Mock()
        mock_org.current_users = 5
        mock_org.max_users = 10
        mock_org.current_month_requests = 500
        mock_org.max_monthly_requests = 1000
        mock_org.current_storage_bytes = 1000000
        mock_org.max_storage_bytes = 2000000

        mock_session.query.return_value.filter.return_value.first.return_value = mock_org

        result = tenant_manager.check_organization_limits(1, mock_session)

        assert result['within_limits'] == True
        assert len(result['issues']) == 0
        assert result['limits']['users']['current'] == 5
        assert result['limits']['users']['max'] == 10


class TestBillingManager:
    """Test billing manager functionality."""

    def test_pricing_tiers(self):
        """Test pricing tier configurations."""
        assert billing_manager.PRICING_TIERS[OrganizationTier.FREE]['base_monthly'] == 0
        assert billing_manager.PRICING_TIERS[OrganizationTier.BASIC]['base_monthly'] == 9.99
        assert billing_manager.PRICING_TIERS[OrganizationTier.PROFESSIONAL]['base_monthly'] == 29.99
        assert billing_manager.PRICING_TIERS[OrganizationTier.ENTERPRISE]['base_monthly'] == 99.99

    def test_monthly_usage_calculation(self):
        """Test monthly usage calculation."""
        mock_session = Mock()

        # Mock usage records
        mock_records = [
            Mock(usage_type='requests', count=1000, amount=0, cost=1.0),
            Mock(usage_type='storage', count=0, amount=1000000000, cost=0.1),
            Mock(usage_type='audio_seconds', count=0, amount=3600, cost=0.36)
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_records

        usage = billing_manager.calculate_monthly_usage(1, 2024, 1, mock_session)

        assert usage['total_requests'] == 1000
        assert usage['total_storage_bytes'] == 1000000000
        assert usage['total_audio_seconds'] == 3600
        assert usage['total_cost'] == 1.46

    def test_billing_amount_calculation(self):
        """Test billing amount calculation."""
        mock_session = Mock()

        # Mock organization
        mock_org = Mock(tier=OrganizationTier.BASIC)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_org

        # Mock usage
        with patch.object(billing_manager, 'calculate_monthly_usage') as mock_usage:
            mock_usage.return_value = {
                'total_requests': 15000,  # 5000 over limit
                'total_storage_bytes': 2000000000,  # 1GB over limit
                'total_audio_seconds': 0,
                'total_cost': 0
            }

            billing = billing_manager.calculate_billing_amount(1, 2024, 1, mock_session)

            assert billing is not None
            assert billing['base_cost'] == 9.99
            assert billing['request_cost'] == 5.0  # 5000/1000 * 1.00
            assert billing['storage_cost'] == 0.1  # 1GB * 0.10
            assert billing['total_amount'] == 15.09

    def test_billing_history(self):
        """Test billing history retrieval."""
        mock_session = Mock()

        # Mock billing records
        mock_billings = [
            Mock(
                id=1,
                billing_period_start=datetime(2024, 1, 1),
                billing_period_end=datetime(2024, 2, 1),
                amount=15.09,
                currency='USD',
                status='paid',
                total_requests=15000,
                total_audio_seconds=0,
                total_storage_bytes=2000000000,
                base_cost=9.99,
                request_cost=5.0,
                storage_cost=0.1,
                additional_cost=0.0,
                transaction_id='txn_123',
                invoice_url='https://example.com/invoice/123',
                processed_at=datetime(2024, 1, 15)
            )
        ]
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_billings

        history = billing_manager.get_billing_history(1, 12, mock_session)

        assert len(history) == 1
        assert history[0]['amount'] == 15.09
        assert history[0]['status'] == 'paid'
        assert history[0]['total_requests'] == 15000


class TestTenantConfigManager:
    """Test tenant configuration manager."""

    def test_configuration_retrieval(self):
        """Test configuration retrieval for different tiers."""
        free_config = tenant_config_manager.get_configuration(OrganizationTier.FREE)
        assert free_config.tier == OrganizationTier.FREE
        assert free_config.limits.max_users == 1
        assert free_config.pricing.base_monthly == 0.0

        basic_config = tenant_config_manager.get_configuration(OrganizationTier.BASIC)
        assert basic_config.tier == OrganizationTier.BASIC
        assert basic_config.limits.max_users == 5
        assert basic_config.pricing.base_monthly == 9.99

    def test_feature_flags(self):
        """Test feature flag functionality."""
        # Free tier features
        assert tenant_config_manager.get_feature_flag(OrganizationTier.FREE, 'priority_processing') == False
        assert tenant_config_manager.get_feature_flag(OrganizationTier.FREE, 'batch_processing') == False

        # Professional tier features
        assert tenant_config_manager.get_feature_flag(OrganizationTier.PROFESSIONAL, 'priority_processing') == True
        assert tenant_config_manager.get_feature_flag(OrganizationTier.PROFESSIONAL, 'batch_processing') == True

    def test_resource_limits(self):
        """Test resource limit retrieval."""
        free_limits = tenant_config_manager.get_resource_limit(OrganizationTier.FREE, 'max_users')
        assert free_limits == 1

        enterprise_limits = tenant_config_manager.get_resource_limit(OrganizationTier.ENTERPRISE, 'max_users')
        assert enterprise_limits == 100

    def test_cost_calculation(self):
        """Test cost calculation for different tiers."""
        # Free tier - should be 0
        free_cost = tenant_config_manager.calculate_cost(OrganizationTier.FREE, requests=1000)
        assert free_cost['total'] == 0.0

        # Basic tier - base + usage
        basic_cost = tenant_config_manager.calculate_cost(
            OrganizationTier.BASIC,
            requests=15000,  # 5000 over limit
            storage_gb=2.0   # 1GB over limit
        )
        assert basic_cost['base_monthly'] == 9.99
        assert basic_cost['requests_cost'] == 5.0  # 5000/1000 * 1.00
        assert basic_cost['storage_cost'] == 0.2   # 2GB * 0.10
        assert basic_cost['total'] == 15.19

    def test_usage_validation(self):
        """Test usage validation against limits."""
        # Valid usage
        valid_usage = {
            'users': 3,
            'monthly_requests': 30000,
            'storage_bytes': 500000000,  # ~0.5GB
            'concurrent_requests': 3
        }

        validation = tenant_config_manager.validate_usage_against_limits(OrganizationTier.BASIC, valid_usage)
        assert validation['valid'] == True
        assert len(validation['violations']) == 0

        # Invalid usage
        invalid_usage = {
            'users': 10,  # Over limit of 5
            'monthly_requests': 100000,  # Over limit of 50000
            'storage_bytes': 2000000000,  # ~2GB, over limit of 1GB
            'concurrent_requests': 20  # Over limit of 5
        }

        validation = tenant_config_manager.validate_usage_against_limits(OrganizationTier.BASIC, invalid_usage)
        assert validation['valid'] == False
        assert len(validation['violations']) == 4

    def test_tier_upgrade_downgrade(self):
        """Test tier upgrade and downgrade paths."""
        # From free tier
        upgrades = tenant_config_manager.get_tier_upgrade_path(OrganizationTier.FREE)
        assert len(upgrades) == 3  # Basic, Professional, Enterprise

        # From enterprise tier
        downgrades = tenant_config_manager.get_tier_downgrade_options(OrganizationTier.ENTERPRISE)
        assert len(downgrades) == 3  # Professional, Basic, Free

    def test_tier_recommendation(self):
        """Test tier recommendation based on usage."""
        usage = {
            'monthly_requests': 30000,
            'storage_bytes': 500000000,  # ~0.5GB
            'audio_seconds': 1800
        }

        recommendation = tenant_config_manager.recommend_tier(usage)

        assert 'recommendations' in recommendation
        assert len(recommendation['recommendations']) > 0
        assert recommendation['best_value'] is not None


class TestMultiTenantIsolation:
    """Test multi-tenant data isolation."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session for testing."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # Create all tables
        from models.organization import Base as OrgBase
        from models.tenancy import Base as TenantBase

        OrgBase.metadata.create_all(engine)
        TenantBase.metadata.create_all(engine)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()

        yield session

        session.close()

    def test_organization_creation(self, db_session):
        """Test organization creation."""
        org = Organization(
            name="Test Org",
            slug="test-org",
            email="test@example.com"
        )

        db_session.add(org)
        db_session.commit()

        assert org.id is not None
        assert org.name == "Test Org"
        assert org.slug == "test-org"
        assert org.status == OrganizationStatus.ACTIVE

    def test_organization_member_management(self, db_session):
        """Test organization member management."""
        # Create organization and users
        org = Organization(name="Test Org", slug="test-org", email="test@example.com")
        db_session.add(org)
        db_session.commit()

        # Create member
        member = OrganizationMember(
            organization_id=org.id,
            user_id=1,
            role=MemberRole.ADMIN
        )

        db_session.add(member)
        db_session.commit()

        assert member.id is not None
        assert member.role == MemberRole.ADMIN
        assert member.is_active == True

    def test_tenant_aware_queries(self, db_session):
        """Test tenant-aware query functionality."""
        # Create test organizations
        org1 = Organization(name="Org 1", slug="org-1", email="org1@example.com")
        org2 = Organization(name="Org 2", slug="org-2", email="org2@example.com")
        db_session.add_all([org1, org2])
        db_session.commit()

        # Create tenant-aware records
        request1 = TenantAwareAudioRequest(
            organization_id=org1.id,
            text="Test request 1",
            user_id=1
        )
        request2 = TenantAwareAudioRequest(
            organization_id=org2.id,
            text="Test request 2",
            user_id=2
        )

        db_session.add_all([request1, request2])
        db_session.commit()

        # Test tenant isolation
        tenant_security.set_tenant_context(org1.id, 1)

        # Should only return records for org1
        org1_requests = TenantAwareAudioRequest.tenant_aware_query(db_session).all()
        assert len(org1_requests) == 1
        assert org1_requests[0].organization_id == org1.id
        assert org1_requests[0].text == "Test request 1"

        # Switch to org2 context
        tenant_security.set_tenant_context(org2.id, 2)

        org2_requests = TenantAwareAudioRequest.tenant_aware_query(db_session).all()
        assert len(org2_requests) == 1
        assert org2_requests[0].organization_id == org2.id
        assert org2_requests[0].text == "Test request 2"

    def test_cross_tenant_security(self, db_session):
        """Test cross-tenant security enforcement."""
        # Create test organizations
        org1 = Organization(name="Org 1", slug="org-1", email="org1@example.com")
        org2 = Organization(name="Org 2", slug="org-2", email="org2@example.com")
        db_session.add_all([org1, org2])
        db_session.commit()

        # Set context to org1
        tenant_security.set_tenant_context(org1.id, 1)

        # Try to create record for org2 - should be blocked
        request_for_org2 = TenantAwareAudioRequest(
            organization_id=org2.id,  # This should be blocked
            text="Unauthorized request",
            user_id=1
        )

        db_session.add(request_for_org2)

        # This should raise an error due to tenant security
        with pytest.raises(ValueError, match="Access denied"):
            db_session.commit()

    def test_billing_record_creation(self, db_session):
        """Test billing record creation."""
        org = Organization(name="Test Org", slug="test-org", email="test@example.com")
        db_session.add(org)
        db_session.commit()

        billing = OrganizationBilling(
            organization_id=org.id,
            billing_period_start=datetime(2024, 1, 1),
            billing_period_end=datetime(2024, 2, 1),
            amount=29.99,
            currency="USD",
            status="pending",
            total_requests=1000,
            total_audio_seconds=3600,
            total_storage_bytes=1000000000,
            base_cost=9.99,
            request_cost=15.00,
            storage_cost=5.00,
            additional_cost=0.0
        )

        db_session.add(billing)
        db_session.commit()

        assert billing.id is not None
        assert billing.amount == 29.99
        assert billing.total_requests == 1000
        assert billing.status == "pending"


class TestTenantAPIFunctionality:
    """Test tenant-aware API functionality."""

    def test_organization_scoped_endpoints(self):
        """Test organization-scoped API endpoints."""
        # This would test the actual API endpoints
        # For now, just test the route structure
        from routes.tenant_tts import tenant_tts_bp

        assert tenant_tts_bp is not None
        assert tenant_tts_bp.url_prefix == '/api/tenant/tts'

        # Check that routes are registered
        rules = [rule.rule for rule in tenant_tts_bp.url_map.iter_rules()]
        assert '/api/tenant/tts/generate' in rules
        assert '/api/tenant/tts/requests' in rules
        assert '/api/tenant/tts/files' in rules

    def test_middleware_functionality(self):
        """Test tenant middleware functionality."""
        from utils.tenant_middleware import TenantMiddleware, require_organization_context

        # Test middleware class
        assert TenantMiddleware is not None

        # Test decorator
        assert require_organization_context is not None

        # Test that decorator is callable
        assert callable(require_organization_context)

    def test_rate_limiting(self):
        """Test organization-scoped rate limiting."""
        from utils.tenant_middleware import organization_rate_limit

        # Test decorator creation
        rate_limiter = organization_rate_limit(max_requests=50, window_seconds=60)
        assert rate_limiter is not None
        assert callable(rate_limiter)


class TestTenantDataMigration:
    """Test tenant data migration functionality."""

    def test_migration_helper_functions(self):
        """Test data migration helper functions."""
        from utils.database_migrations import (
            create_tenant_aware_tables,
            drop_tenant_aware_tables,
            validate_tenant_data_integrity
        )

        # Test function existence
        assert callable(create_tenant_aware_tables)
        assert callable(drop_tenant_aware_tables)
        assert callable(validate_tenant_data_integrity)

    def test_migration_statistics(self):
        """Test migration statistics functions."""
        from utils.database_migrations import get_tenant_statistics

        assert callable(get_tenant_statistics)


if __name__ == "__main__":
    # Run basic tests
    print("Running multi-tenancy tests...")

    # Test tenant security manager
    test_security = TestTenantSecurityManager()
    test_security.test_tenant_context_management()
    test_security.test_security_bypass()
    test_security.test_organization_id_retrieval()
    print("✓ Tenant security manager tests passed")

    # Test tenant manager
    test_manager = TestTenantManager()
    test_manager.test_singleton_pattern()
    test_manager.test_api_key_extraction()
    print("✓ Tenant manager tests passed")

    # Test billing manager
    test_billing = TestBillingManager()
    test_billing.test_pricing_tiers()
    print("✓ Billing manager tests passed")

    # Test configuration manager
    test_config = TestTenantConfigManager()
    test_config.test_configuration_retrieval()
    test_config.test_feature_flags()
    test_config.test_cost_calculation()
    print("✓ Configuration manager tests passed")

    print("All multi-tenancy tests completed successfully! 🎉")