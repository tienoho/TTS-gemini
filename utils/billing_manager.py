"""
Billing and usage tracking manager for multi-tenant TTS system
"""

import calendar
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

from sqlalchemy.orm import Session

from models.organization import Organization, OrganizationBilling, OrganizationUsage, OrganizationStatus, OrganizationTier
from models.tenancy import TenantAwareAudioRequest, TenantAwareAudioFile
from utils.tenant_manager import tenant_manager


class BillingManager:
    """Manages billing calculations and usage tracking for organizations."""

    # Pricing tiers
    PRICING_TIERS = {
        OrganizationTier.FREE: {
            'base_monthly': 0,
            'request_cost_per_1000': 0,
            'storage_cost_per_gb': 0,
            'max_requests': 10000,
            'max_storage_gb': 1
        },
        OrganizationTier.BASIC: {
            'base_monthly': 9.99,
            'request_cost_per_1000': 1.00,
            'storage_cost_per_gb': 0.10,
            'max_requests': 100000,
            'max_storage_gb': 10
        },
        OrganizationTier.PROFESSIONAL: {
            'base_monthly': 29.99,
            'request_cost_per_1000': 0.50,
            'storage_cost_per_gb': 0.05,
            'max_requests': 500000,
            'max_storage_gb': 100
        },
        OrganizationTier.ENTERPRISE: {
            'base_monthly': 99.99,
            'request_cost_per_1000': 0.25,
            'storage_cost_per_gb': 0.02,
            'max_requests': 2000000,
            'max_storage_gb': 1000
        }
    }

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def calculate_monthly_usage(self, organization_id: int, year: int, month: int, db_session: Session) -> Dict[str, Any]:
        """Calculate usage for a specific month."""
        # Get start and end of month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        end_date = end_date - timedelta(microseconds=1)

        # Get usage records for the month
        usage_records = OrganizationUsage.get_usage_by_org_and_date_range(
            organization_id, start_date, end_date, db_session
        )

        # Calculate totals
        total_requests = 0
        total_audio_seconds = 0.0
        total_storage_bytes = 0
        total_cost = 0.0

        usage_by_type = {}

        for record in usage_records:
            if record.usage_type == 'requests':
                total_requests += record.count
            elif record.usage_type == 'audio_seconds':
                total_audio_seconds += record.amount
            elif record.usage_type == 'storage':
                total_storage_bytes += record.amount

            total_cost += record.cost

            if record.usage_type not in usage_by_type:
                usage_by_type[record.usage_type] = {
                    'count': 0,
                    'amount': 0.0,
                    'cost': 0.0
                }

            usage_by_type[record.usage_type]['count'] += record.count
            usage_by_type[record.usage_type]['amount'] += record.amount
            usage_by_type[record.usage_type]['cost'] += record.cost

        return {
            'organization_id': organization_id,
            'year': year,
            'month': month,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'total_requests': total_requests,
            'total_audio_seconds': total_audio_seconds,
            'total_storage_bytes': total_storage_bytes,
            'total_cost': total_cost,
            'usage_by_type': usage_by_type
        }

    def calculate_billing_amount(self, organization_id: int, year: int, month: int, db_session: Session) -> Dict[str, Any]:
        """Calculate billing amount for a specific month."""
        org = db_session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return None

        # Get usage for the month
        usage = self.calculate_monthly_usage(organization_id, year, month, db_session)

        # Get pricing tier
        tier_config = self.PRICING_TIERS.get(org.tier, self.PRICING_TIERS[OrganizationTier.FREE])

        # Calculate costs
        base_cost = tier_config['base_monthly']

        # Request cost (per 1000 requests)
        request_cost = 0.0
        if usage['total_requests'] > tier_config['max_requests']:
            excess_requests = usage['total_requests'] - tier_config['max_requests']
            request_cost = (excess_requests / 1000) * tier_config['request_cost_per_1000']

        # Storage cost (per GB)
        storage_cost = 0.0
        storage_gb = usage['total_storage_bytes'] / (1024 * 1024 * 1024)
        if storage_gb > tier_config['max_storage_gb']:
            excess_storage = storage_gb - tier_config['max_storage_gb']
            storage_cost = excess_storage * tier_config['storage_cost_per_gb']

        total_amount = base_cost + request_cost + storage_cost

        return {
            'organization_id': organization_id,
            'organization_name': org.name,
            'tier': org.tier.value,
            'year': year,
            'month': month,
            'base_cost': base_cost,
            'request_cost': request_cost,
            'storage_cost': storage_cost,
            'total_amount': total_amount,
            'currency': org.currency,
            'usage': usage,
            'tier_limits': tier_config
        }

    def generate_monthly_billing(self, organization_id: int, year: int, month: int, db_session: Session) -> Optional[OrganizationBilling]:
        """Generate monthly billing record."""
        # Check if billing already exists
        existing_billing = db_session.query(OrganizationBilling).filter(
            OrganizationBilling.organization_id == organization_id,
            OrganizationBilling.billing_period_start == datetime(year, month, 1)
        ).first()

        if existing_billing:
            return existing_billing

        # Calculate billing
        billing_data = self.calculate_billing_amount(organization_id, year, month, db_session)
        if not billing_data:
            return None

        # Get start and end of billing period
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        end_date = end_date - timedelta(microseconds=1)

        # Create billing record
        billing = OrganizationBilling(
            organization_id=organization_id,
            billing_period_start=start_date,
            billing_period_end=end_date,
            amount=float(billing_data['total_amount']),
            currency=billing_data['currency'],
            status='pending',
            total_requests=billing_data['usage']['total_requests'],
            total_audio_seconds=billing_data['usage']['total_audio_seconds'],
            total_storage_bytes=billing_data['usage']['total_storage_bytes'],
            base_cost=billing_data['base_cost'],
            request_cost=billing_data['request_cost'],
            storage_cost=billing_data['storage_cost'],
            additional_cost=0.0  # For any additional charges
        )

        db_session.add(billing)
        db_session.commit()

        return billing

    def generate_all_monthly_billings(self, year: int, month: int, db_session: Session) -> List[OrganizationBilling]:
        """Generate monthly billings for all active organizations."""
        organizations = db_session.query(Organization).filter(
            Organization.status == OrganizationStatus.ACTIVE
        ).all()

        billings = []
        for org in organizations:
            billing = self.generate_monthly_billing(org.id, year, month, db_session)
            if billing:
                billings.append(billing)

        return billings

    def get_billing_history(self, organization_id: int, db_session: Session, limit: int = 12) -> List[Dict[str, Any]]:
        """Get billing history for organization."""
        billings = db_session.query(OrganizationBilling).filter(
            OrganizationBilling.organization_id == organization_id
        ).order_by(OrganizationBilling.billing_period_start.desc()).limit(limit).all()

        history = []
        for billing in billings:
            history.append({
                'id': billing.id,
                'period_start': billing.billing_period_start.isoformat(),
                'period_end': billing.billing_period_end.isoformat(),
                'amount': billing.amount,
                'currency': billing.currency,
                'status': billing.status,
                'total_requests': billing.total_requests,
                'total_audio_seconds': billing.total_audio_seconds,
                'total_storage_bytes': billing.total_storage_bytes,
                'base_cost': billing.base_cost,
                'request_cost': billing.request_cost,
                'storage_cost': billing.storage_cost,
                'additional_cost': billing.additional_cost,
                'transaction_id': billing.transaction_id,
                'invoice_url': billing.invoice_url,
                'processed_at': billing.processed_at.isoformat() if billing.processed_at else None
            })

        return history

    def update_billing_status(self, billing_id: int, status: str, db_session: Session, transaction_id: str = None,
                             invoice_url: str = None) -> bool:
        """Update billing status."""
        billing = db_session.query(OrganizationBilling).filter(
            OrganizationBilling.id == billing_id
        ).first()

        if not billing:
            return False

        billing.status = status
        billing.transaction_id = transaction_id
        billing.invoice_url = invoice_url
        billing.processed_at = datetime.utcnow()

        db_session.commit()
        return True

    def get_organization_cost_summary(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Get cost summary for organization."""
        org = db_session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return None

        # Get current month usage
        current_year = datetime.utcnow().year
        current_month = datetime.utcnow().month
        current_usage = self.calculate_monthly_usage(organization_id, current_year, current_month, db_session)

        # Get billing history
        billing_history = self.get_billing_history(organization_id, db_session, 6)

        # Calculate average monthly cost
        total_billing_cost = sum(billing['amount'] for billing in billing_history if billing['status'] == 'paid')
        average_monthly_cost = total_billing_cost / max(len([b for b in billing_history if b['status'] == 'paid']), 1)

        return {
            'organization_id': organization_id,
            'organization_name': org.name,
            'tier': org.tier.value,
            'current_month': {
                'year': current_year,
                'month': current_month,
                'requests': current_usage['total_requests'],
                'audio_seconds': current_usage['total_audio_seconds'],
                'storage_bytes': current_usage['total_storage_bytes'],
                'cost': current_usage['total_cost']
            },
            'totals': {
                'total_cost': org.total_cost,
                'monthly_cost': org.monthly_cost,
                'current_month_requests': org.current_month_requests,
                'current_month_cost': org.current_month_cost,
                'current_storage_bytes': org.current_storage_bytes
            },
            'billing_history': billing_history,
            'average_monthly_cost': average_monthly_cost,
            'currency': org.currency
        }

    def get_usage_forecast(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Generate usage forecast for organization."""
        org = db_session.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            return None

        # Get last 3 months usage
        current_date = datetime.utcnow()
        usage_data = []

        for i in range(3):
            date = current_date - timedelta(days=30 * i)
            year, month = date.year, date.month
            usage = self.calculate_monthly_usage(organization_id, year, month, db_session)
            usage_data.append(usage)

        # Simple linear forecast
        if len(usage_data) >= 2:
            # Calculate trend
            requests_trend = usage_data[0]['total_requests'] - usage_data[1]['total_requests']
            cost_trend = usage_data[0]['total_cost'] - usage_data[1]['total_cost']

            forecast_requests = usage_data[0]['total_requests'] + requests_trend
            forecast_cost = usage_data[0]['total_cost'] + cost_trend
        else:
            forecast_requests = usage_data[0]['total_requests'] if usage_data else 0
            forecast_cost = usage_data[0]['total_cost'] if usage_data else 0

        # Get tier limits
        tier_config = self.PRICING_TIERS.get(org.tier, self.PRICING_TIERS[OrganizationTier.FREE])

        return {
            'organization_id': organization_id,
            'organization_name': org.name,
            'tier': org.tier.value,
            'current_usage': usage_data[0] if usage_data else {},
            'forecast': {
                'next_month_requests': max(0, forecast_requests),
                'next_month_cost': max(0, forecast_cost),
                'trend_direction': 'increasing' if requests_trend > 0 else 'decreasing' if requests_trend < 0 else 'stable'
            },
            'tier_limits': tier_config,
            'recommendations': self._generate_tier_recommendations(org, usage_data[0] if usage_data else {})
        }

    def _generate_tier_recommendations(self, org: Organization, current_usage: Dict[str, Any]) -> List[str]:
        """Generate tier upgrade/downgrade recommendations."""
        recommendations = []
        tier_config = self.PRICING_TIERS.get(org.tier, self.PRICING_TIERS[OrganizationTier.FREE])

        # Check if current usage exceeds tier limits
        if current_usage.get('total_requests', 0) > tier_config['max_requests'] * 0.8:
            recommendations.append("Consider upgrading tier - approaching request limit")

        storage_gb = current_usage.get('total_storage_bytes', 0) / (1024 * 1024 * 1024)
        if storage_gb > tier_config['max_storage_gb'] * 0.8:
            recommendations.append("Consider upgrading tier - approaching storage limit")

        # Check if current tier is overkill
        if org.tier != OrganizationTier.FREE:
            lower_tier = self._get_lower_tier(org.tier)
            if lower_tier:
                lower_config = self.PRICING_TIERS[lower_tier]
                if (current_usage.get('total_requests', 0) < lower_config['max_requests'] * 0.5 and
                    storage_gb < lower_config['max_storage_gb'] * 0.5):
                    recommendations.append(f"Consider downgrading to {lower_tier.value} tier - current usage is low")

        return recommendations

    def _get_lower_tier(self, tier: OrganizationTier) -> Optional[OrganizationTier]:
        """Get the next lower tier."""
        tier_order = [OrganizationTier.FREE, OrganizationTier.BASIC, OrganizationTier.PROFESSIONAL, OrganizationTier.ENTERPRISE]
        try:
            current_index = tier_order.index(tier)
            if current_index > 0:
                return tier_order[current_index - 1]
        except ValueError:
            pass
        return None

    def process_payment_webhook(self, webhook_data: Dict[str, Any], db_session: Session) -> bool:
        """Process payment webhook and update billing status."""
        try:
            billing_id = webhook_data.get('billing_id')
            status = webhook_data.get('status', 'unknown')
            transaction_id = webhook_data.get('transaction_id')
            invoice_url = webhook_data.get('invoice_url')

            if not billing_id:
                return False

            success = self.update_billing_status(
                billing_id, status, db_session, transaction_id, invoice_url
            )

            if success and status == 'paid':
                # Update organization totals
                billing = db_session.query(OrganizationBilling).filter(
                    OrganizationBilling.id == billing_id
                ).first()

                if billing:
                    org = billing.organization
                    org.total_cost += billing.amount
                    org.monthly_cost += billing.amount
                    db_session.commit()

            return success

        except Exception as e:
            print(f"Error processing payment webhook: {e}")
            return False


# Global billing manager instance
billing_manager = BillingManager()