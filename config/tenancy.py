"""
Multi-tenant configuration for TTS system
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import timedelta

from models.organization import OrganizationTier, MemberRole


@dataclass
class TenantResourceLimits:
    """Resource limits for a tenant tier."""
    max_users: int = 5
    max_monthly_requests: int = 10000
    max_storage_bytes: int = 1000000000  # 1GB
    max_concurrent_requests: int = 10
    max_request_size_mb: int = 10
    max_audio_duration_seconds: int = 300  # 5 minutes
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 100


@dataclass
class TenantPricing:
    """Pricing configuration for a tenant tier."""
    base_monthly: float = 0.0
    request_cost_per_1000: float = 0.0
    storage_cost_per_gb: float = 0.0
    audio_second_cost: float = 0.0
    support_level: str = "basic"


@dataclass
class TenantFeatures:
    """Feature flags for a tenant tier."""
    priority_processing: bool = False
    custom_voices: bool = False
    batch_processing: bool = False
    api_webhooks: bool = False
    advanced_analytics: bool = False
    custom_models: bool = False
    sso_integration: bool = False
    audit_logs: bool = False
    api_rate_limits: bool = True
    usage_alerts: bool = True


@dataclass
class TenantConfiguration:
    """Complete configuration for a tenant tier."""
    tier: OrganizationTier
    name: str
    description: str
    limits: TenantResourceLimits = field(default_factory=TenantResourceLimits)
    pricing: TenantPricing = field(default_factory=TenantPricing)
    features: TenantFeatures = field(default_factory=TenantFeatures)
    is_active: bool = True


class TenantConfigManager:
    """Manages tenant configurations and settings."""

    def __init__(self):
        self._configurations = self._initialize_configurations()
        self._custom_configs = {}
        self._feature_overrides = {}

    def _initialize_configurations(self) -> Dict[OrganizationTier, TenantConfiguration]:
        """Initialize default configurations for all tiers."""
        return {
            OrganizationTier.FREE: TenantConfiguration(
                tier=OrganizationTier.FREE,
                name="Free",
                description="Free tier with basic features",
                limits=TenantResourceLimits(
                    max_users=1,
                    max_monthly_requests=1000,
                    max_storage_bytes=100000000,  # 100MB
                    max_concurrent_requests=2,
                    max_request_size_mb=5,
                    max_audio_duration_seconds=60,
                    rate_limit_requests_per_minute=10,
                    rate_limit_burst=20
                ),
                pricing=TenantPricing(
                    base_monthly=0.0,
                    request_cost_per_1000=0.0,
                    storage_cost_per_gb=0.0,
                    audio_second_cost=0.0,
                    support_level="community"
                ),
                features=TenantFeatures(
                    priority_processing=False,
                    custom_voices=False,
                    batch_processing=False,
                    api_webhooks=False,
                    advanced_analytics=False,
                    custom_models=False,
                    sso_integration=False,
                    audit_logs=False,
                    api_rate_limits=True,
                    usage_alerts=False
                )
            ),

            OrganizationTier.BASIC: TenantConfiguration(
                tier=OrganizationTier.BASIC,
                name="Basic",
                description="Basic tier with standard features",
                limits=TenantResourceLimits(
                    max_users=5,
                    max_monthly_requests=50000,
                    max_storage_bytes=1000000000,  # 1GB
                    max_concurrent_requests=5,
                    max_request_size_mb=10,
                    max_audio_duration_seconds=180,
                    rate_limit_requests_per_minute=60,
                    rate_limit_burst=100
                ),
                pricing=TenantPricing(
                    base_monthly=9.99,
                    request_cost_per_1000=1.00,
                    storage_cost_per_gb=0.10,
                    audio_second_cost=0.001,
                    support_level="email"
                ),
                features=TenantFeatures(
                    priority_processing=False,
                    custom_voices=False,
                    batch_processing=True,
                    api_webhooks=False,
                    advanced_analytics=False,
                    custom_models=False,
                    sso_integration=False,
                    audit_logs=True,
                    api_rate_limits=True,
                    usage_alerts=True
                )
            ),

            OrganizationTier.PROFESSIONAL: TenantConfiguration(
                tier=OrganizationTier.PROFESSIONAL,
                name="Professional",
                description="Professional tier with advanced features",
                limits=TenantResourceLimits(
                    max_users=25,
                    max_monthly_requests=200000,
                    max_storage_bytes=10000000000,  # 10GB
                    max_concurrent_requests=15,
                    max_request_size_mb=25,
                    max_audio_duration_seconds=600,
                    rate_limit_requests_per_minute=200,
                    rate_limit_burst=300
                ),
                pricing=TenantPricing(
                    base_monthly=29.99,
                    request_cost_per_1000=0.50,
                    storage_cost_per_gb=0.05,
                    audio_second_cost=0.0005,
                    support_level="priority_email"
                ),
                features=TenantFeatures(
                    priority_processing=True,
                    custom_voices=True,
                    batch_processing=True,
                    api_webhooks=True,
                    advanced_analytics=True,
                    custom_models=False,
                    sso_integration=False,
                    audit_logs=True,
                    api_rate_limits=True,
                    usage_alerts=True
                )
            ),

            OrganizationTier.ENTERPRISE: TenantConfiguration(
                tier=OrganizationTier.ENTERPRISE,
                name="Enterprise",
                description="Enterprise tier with full features",
                limits=TenantResourceLimits(
                    max_users=100,
                    max_monthly_requests=1000000,
                    max_storage_bytes=100000000000,  # 100GB
                    max_concurrent_requests=50,
                    max_request_size_mb=50,
                    max_audio_duration_seconds=1800,
                    rate_limit_requests_per_minute=500,
                    rate_limit_burst=1000
                ),
                pricing=TenantPricing(
                    base_monthly=99.99,
                    request_cost_per_1000=0.25,
                    storage_cost_per_gb=0.02,
                    audio_second_cost=0.00025,
                    support_level="dedicated"
                ),
                features=TenantFeatures(
                    priority_processing=True,
                    custom_voices=True,
                    batch_processing=True,
                    api_webhooks=True,
                    advanced_analytics=True,
                    custom_models=True,
                    sso_integration=True,
                    audit_logs=True,
                    api_rate_limits=True,
                    usage_alerts=True
                )
            )
        }

    def get_configuration(self, tier: OrganizationTier) -> TenantConfiguration:
        """Get configuration for a specific tier."""
        # Check for custom configuration first
        if tier in self._custom_configs:
            return self._custom_configs[tier]

        return self._configurations.get(tier, self._configurations[OrganizationTier.FREE])

    def set_custom_configuration(self, tier: OrganizationTier, config: TenantConfiguration):
        """Set custom configuration for a tier."""
        self._custom_configs[tier] = config

    def get_feature_flag(self, tier: OrganizationTier, feature: str) -> bool:
        """Get feature flag value for a tier."""
        config = self.get_configuration(tier)

        # Check for feature overrides
        if tier in self._feature_overrides and feature in self._feature_overrides[tier]:
            return self._feature_overrides[tier][feature]

        return getattr(config.features, feature, False)

    def set_feature_override(self, tier: OrganizationTier, feature: str, enabled: bool):
        """Override a feature flag for a specific tier."""
        if tier not in self._feature_overrides:
            self._feature_overrides[tier] = {}
        self._feature_overrides[tier][feature] = enabled

    def get_resource_limit(self, tier: OrganizationTier, limit: str) -> Any:
        """Get resource limit for a tier."""
        config = self.get_configuration(tier)
        return getattr(config.limits, limit, None)

    def get_pricing(self, tier: OrganizationTier) -> TenantPricing:
        """Get pricing configuration for a tier."""
        config = self.get_configuration(tier)
        return config.pricing

    def calculate_cost(self, tier: OrganizationTier, requests: int = 0,
                      storage_gb: float = 0, audio_seconds: float = 0) -> Dict[str, float]:
        """Calculate cost for given usage."""
        config = self.get_configuration(tier)
        pricing = config.pricing

        # Base monthly cost
        total_cost = pricing.base_monthly

        # Request cost (per 1000 requests)
        if requests > 0:
            request_cost = (requests / 1000) * pricing.request_cost_per_1000
            total_cost += request_cost

        # Storage cost (per GB)
        if storage_gb > 0:
            storage_cost = storage_gb * pricing.storage_cost_per_gb
            total_cost += storage_cost

        # Audio processing cost
        if audio_seconds > 0:
            audio_cost = audio_seconds * pricing.audio_second_cost
            total_cost += audio_cost

        return {
            'base_monthly': pricing.base_monthly,
            'requests_cost': (requests / 1000) * pricing.request_cost_per_1000 if requests > 0 else 0,
            'storage_cost': storage_gb * pricing.storage_cost_per_gb if storage_gb > 0 else 0,
            'audio_cost': audio_seconds * pricing.audio_second_cost if audio_seconds > 0 else 0,
            'total': total_cost
        }

    def validate_usage_against_limits(self, tier: OrganizationTier, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Validate usage against tier limits."""
        config = self.get_configuration(tier)
        limits = config.limits

        violations = []
        warnings = []

        # Check user limit
        if usage.get('users', 0) > limits.max_users:
            violations.append(f"User limit exceeded: {usage.get('users', 0)} > {limits.max_users}")

        # Check request limit
        if usage.get('monthly_requests', 0) > limits.max_monthly_requests:
            violations.append(f"Monthly request limit exceeded: {usage.get('monthly_requests', 0)} > {limits.max_monthly_requests}")

        # Check storage limit
        storage_gb = usage.get('storage_bytes', 0) / (1024 * 1024 * 1024)
        if storage_gb > limits.max_storage_bytes / (1024 * 1024 * 1024):
            violations.append(f"Storage limit exceeded: {storage_gb:.2f}GB > {limits.max_storage_bytes / (1024 * 1024 * 1024):.2f}GB")

        # Check concurrent requests
        if usage.get('concurrent_requests', 0) > limits.max_concurrent_requests:
            violations.append(f"Concurrent request limit exceeded: {usage.get('concurrent_requests', 0)} > {limits.max_concurrent_requests}")

        # Warnings for approaching limits
        if usage.get('users', 0) > limits.max_users * 0.8:
            warnings.append(f"Approaching user limit: {usage.get('users', 0)}/{limits.max_users}")

        if usage.get('monthly_requests', 0) > limits.max_monthly_requests * 0.8:
            warnings.append(f"Approaching request limit: {usage.get('monthly_requests', 0)}/{limits.max_monthly_requests}")

        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'limits': {
                'max_users': limits.max_users,
                'max_monthly_requests': limits.max_monthly_requests,
                'max_storage_bytes': limits.max_storage_bytes,
                'max_concurrent_requests': limits.max_concurrent_requests
            },
            'current_usage': usage
        }

    def get_available_tiers(self) -> List[Dict[str, Any]]:
        """Get all available tiers with their configurations."""
        tiers = []
        for tier, config in self._configurations.items():
            if config.is_active:
                tiers.append({
                    'tier': tier.value,
                    'name': config.name,
                    'description': config.description,
                    'limits': {
                        'max_users': config.limits.max_users,
                        'max_monthly_requests': config.limits.max_monthly_requests,
                        'max_storage_bytes': config.limits.max_storage_bytes,
                        'max_concurrent_requests': config.limits.max_concurrent_requests
                    },
                    'pricing': {
                        'base_monthly': config.pricing.base_monthly,
                        'request_cost_per_1000': config.pricing.request_cost_per_1000,
                        'storage_cost_per_gb': config.pricing.storage_cost_per_gb,
                        'support_level': config.pricing.support_level
                    },
                    'features': {
                        'priority_processing': config.features.priority_processing,
                        'custom_voices': config.features.custom_voices,
                        'batch_processing': config.features.batch_processing,
                        'api_webhooks': config.features.api_webhooks,
                        'advanced_analytics': config.features.advanced_analytics,
                        'custom_models': config.features.custom_models,
                        'sso_integration': config.features.sso_integration,
                        'audit_logs': config.features.audit_logs
                    }
                })
        return tiers

    def get_tier_upgrade_path(self, current_tier: OrganizationTier) -> List[Dict[str, Any]]:
        """Get available upgrade paths from current tier."""
        tier_order = [OrganizationTier.FREE, OrganizationTier.BASIC, OrganizationTier.PROFESSIONAL, OrganizationTier.ENTERPRISE]
        upgrades = []

        try:
            current_index = tier_order.index(current_tier)
            for tier in tier_order[current_index + 1:]:
                config = self.get_configuration(tier)
                if config.is_active:
                    upgrades.append({
                        'tier': tier.value,
                        'name': config.name,
                        'description': config.description,
                        'price_increase': config.pricing.base_monthly - self.get_configuration(current_tier).pricing.base_monthly
                    })
        except ValueError:
            pass

        return upgrades

    def get_tier_downgrade_options(self, current_tier: OrganizationTier) -> List[Dict[str, Any]]:
        """Get available downgrade options from current tier."""
        tier_order = [OrganizationTier.FREE, OrganizationTier.BASIC, OrganizationTier.PROFESSIONAL, OrganizationTier.ENTERPRISE]
        downgrades = []

        try:
            current_index = tier_order.index(current_tier)
            for tier in tier_order[:current_index]:
                config = self.get_configuration(tier)
                if config.is_active:
                    downgrades.append({
                        'tier': tier.value,
                        'name': config.name,
                        'description': config.description,
                        'price_decrease': self.get_configuration(current_tier).pricing.base_monthly - config.pricing.base_monthly
                    })
        except ValueError:
            pass

        return downgrades

    def estimate_tier_cost(self, tier: OrganizationTier, usage: Dict[str, Any]) -> Dict[str, float]:
        """Estimate cost for a tier based on usage patterns."""
        return self.calculate_cost(
            tier=tier,
            requests=usage.get('monthly_requests', 0),
            storage_gb=usage.get('storage_bytes', 0) / (1024 * 1024 * 1024),
            audio_seconds=usage.get('audio_seconds', 0)
        )

    def recommend_tier(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal tier based on usage patterns."""
        recommendations = []
        current_cost = None

        for tier in self._configurations.keys():
            config = self.get_configuration(tier)
            if not config.is_active:
                continue

            # Estimate cost for this tier
            estimated_cost = self.estimate_tier_cost(tier, usage)

            # Check if usage fits within limits
            validation = self.validate_usage_against_limits(tier, usage)

            recommendations.append({
                'tier': tier.value,
                'name': config.name,
                'estimated_monthly_cost': estimated_cost['total'],
                'fits_limits': validation['valid'],
                'limit_violations': len(validation['violations']),
                'warnings': len(validation['warnings'])
            })

        # Sort by cost efficiency (cost per feature)
        recommendations.sort(key=lambda x: x['estimated_monthly_cost'])

        return {
            'current_usage': usage,
            'recommendations': recommendations,
            'best_value': recommendations[0] if recommendations else None
        }

    def get_organization_config(self, organization_id: int, db_session) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific organization."""
        from models.organization import Organization

        org = db_session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return None

        config = self.get_configuration(org.tier)

        return {
            'organization_id': organization_id,
            'tier': org.tier.value,
            'name': config.name,
            'description': config.description,
            'limits': {
                'max_users': config.limits.max_users,
                'max_monthly_requests': config.limits.max_monthly_requests,
                'max_storage_bytes': config.limits.max_storage_bytes,
                'max_concurrent_requests': config.limits.max_concurrent_requests,
                'max_request_size_mb': config.limits.max_request_size_mb,
                'max_audio_duration_seconds': config.limits.max_audio_duration_seconds,
                'rate_limit_requests_per_minute': config.limits.rate_limit_requests_per_minute,
                'rate_limit_burst': config.limits.rate_limit_burst
            },
            'features': {
                'priority_processing': config.features.priority_processing,
                'custom_voices': config.features.custom_voices,
                'batch_processing': config.features.batch_processing,
                'api_webhooks': config.features.api_webhooks,
                'advanced_analytics': config.features.advanced_analytics,
                'custom_models': config.features.custom_models,
                'sso_integration': config.features.sso_integration,
                'audit_logs': config.features.audit_logs,
                'api_rate_limits': config.features.api_rate_limits,
                'usage_alerts': config.features.usage_alerts
            },
            'pricing': {
                'base_monthly': config.pricing.base_monthly,
                'request_cost_per_1000': config.pricing.request_cost_per_1000,
                'storage_cost_per_gb': config.pricing.storage_cost_per_gb,
                'audio_second_cost': config.pricing.audio_second_cost,
                'support_level': config.pricing.support_level
            }
        }


# Global tenant configuration manager
tenant_config_manager = TenantConfigManager()