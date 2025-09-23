"""
Business Intelligence Configuration for TTS System
Defines KPIs, revenue rules, customer segmentation, and forecasting parameters
"""

import os
from datetime import timedelta
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class BusinessIntelligenceConfig(BaseSettings):
    """Business Intelligence configuration settings."""

    # KPI Definitions
    KPI_DEFINITIONS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Revenue Calculation Rules
    REVENUE_CALCULATION_RULES: Dict[str, Any] = Field(default_factory=dict)

    # Customer Segmentation Rules
    CUSTOMER_SEGMENTATION: Dict[str, Any] = Field(default_factory=dict)

    # Forecasting Parameters
    FORECASTING_PARAMETERS: Dict[str, Any] = Field(default_factory=dict)

    # Report Templates
    REPORT_TEMPLATES: Dict[str, Any] = Field(default_factory=dict)

    # Analytics Settings
    ANALYTICS_RETENTION_DAYS: int = Field(default=365)
    REAL_TIME_ANALYTICS_ENABLED: bool = Field(default=True)
    BATCH_PROCESSING_ENABLED: bool = Field(default=True)

    # Alert Thresholds
    ALERT_THRESHOLDS: Dict[str, Any] = Field(default_factory=dict)

    # Data Processing
    BATCH_SIZE: int = Field(default=1000)
    PROCESSING_TIMEOUT_SECONDS: int = Field(default=300)

    # Cache Settings
    CACHE_TTL_SECONDS: int = Field(default=3600)  # 1 hour
    CACHE_ENABLED: bool = Field(default=True)

    # Feature Flags
    ENABLE_ADVANCED_ANALYTICS: bool = Field(default=True)
    ENABLE_PREDICTIVE_MODELING: bool = Field(default=True)
    ENABLE_ANOMALY_DETECTION: bool = Field(default=True)

    class Config:
        """Pydantic configuration."""
        env_file = '.env'
        case_sensitive = False

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize default configuration values."""

        # KPI Definitions
        self.KPI_DEFINITIONS = {
            'revenue': {
                'name': 'Total Revenue',
                'category': 'financial',
                'type': 'currency',
                'formula': 'sum(revenue_streams.amount)',
                'target': 100000,
                'target_period': 'monthly',
                'description': 'Total revenue from all sources'
            },
            'profit_margin': {
                'name': 'Profit Margin',
                'category': 'financial',
                'type': 'percentage',
                'formula': '(profit / revenue) * 100',
                'target': 30.0,
                'target_period': 'monthly',
                'description': 'Percentage of revenue that is profit'
            },
            'customer_acquisition_cost': {
                'name': 'Customer Acquisition Cost',
                'category': 'customer',
                'type': 'currency',
                'formula': 'marketing_cost / new_customers',
                'target': 50.0,
                'target_period': 'monthly',
                'description': 'Cost to acquire a new customer'
            },
            'customer_lifetime_value': {
                'name': 'Customer Lifetime Value',
                'category': 'customer',
                'type': 'currency',
                'formula': 'average_revenue_per_customer * customer_lifespan',
                'target': 500.0,
                'target_period': 'monthly',
                'description': 'Total value a customer brings over their lifetime'
            },
            'churn_rate': {
                'name': 'Churn Rate',
                'category': 'customer',
                'type': 'percentage',
                'formula': '(churned_customers / total_customers) * 100',
                'target': 5.0,
                'target_period': 'monthly',
                'description': 'Percentage of customers who stop using the service'
            },
            'monthly_recurring_revenue': {
                'name': 'Monthly Recurring Revenue',
                'category': 'financial',
                'type': 'currency',
                'formula': 'sum(subscription_revenue)',
                'target': 50000,
                'target_period': 'monthly',
                'description': 'Recurring revenue from subscriptions'
            },
            'average_revenue_per_user': {
                'name': 'Average Revenue Per User',
                'category': 'financial',
                'type': 'currency',
                'formula': 'total_revenue / active_users',
                'target': 25.0,
                'target_period': 'monthly',
                'description': 'Average revenue generated per active user'
            },
            'system_uptime': {
                'name': 'System Uptime',
                'category': 'operational',
                'type': 'percentage',
                'formula': '(uptime_hours / total_hours) * 100',
                'target': 99.9,
                'target_period': 'monthly',
                'description': 'Percentage of time the system is operational'
            },
            'api_response_time': {
                'name': 'API Response Time',
                'category': 'operational',
                'type': 'metric',
                'formula': 'average_response_time',
                'target': 0.5,
                'target_period': 'daily',
                'description': 'Average time for API responses in seconds'
            },
            'error_rate': {
                'name': 'Error Rate',
                'category': 'operational',
                'type': 'percentage',
                'formula': '(error_count / total_requests) * 100',
                'target': 1.0,
                'target_period': 'daily',
                'description': 'Percentage of requests that result in errors'
            }
        }

        # Revenue Calculation Rules
        self.REVENUE_CALCULATION_RULES = {
            'subscription': {
                'recognition_method': 'ratable',
                'period': 'monthly',
                'proration_enabled': True,
                'discount_handling': 'apply_to_period'
            },
            'pay_per_use': {
                'recognition_method': 'immediate',
                'minimum_charge': 0.01,
                'rounding_precision': 2
            },
            'enterprise': {
                'recognition_method': 'milestone',
                'payment_terms': 'net_30',
                'revenue_share': 0.7
            },
            'currency_conversion': {
                'default_currency': 'USD',
                'exchange_rate_api': 'open_exchange_rates',
                'cache_duration_minutes': 60
            }
        }

        # Customer Segmentation Rules
        self.CUSTOMER_SEGMENTATION = {
            'enterprise': {
                'criteria': {
                    'monthly_revenue': {'min': 10000},
                    'user_count': {'min': 100},
                    'contract_value': {'min': 50000}
                },
                'features': ['advanced_analytics', 'custom_integration', 'dedicated_support'],
                'support_level': 'premium'
            },
            'sme': {
                'criteria': {
                    'monthly_revenue': {'min': 1000, 'max': 9999},
                    'user_count': {'min': 10, 'max': 99},
                    'contract_value': {'min': 5000, 'max': 49999}
                },
                'features': ['basic_analytics', 'standard_integration', 'priority_support'],
                'support_level': 'standard'
            },
            'startup': {
                'criteria': {
                    'monthly_revenue': {'min': 100, 'max': 999},
                    'user_count': {'min': 1, 'max': 9},
                    'company_age_months': {'max': 24}
                },
                'features': ['basic_features', 'community_support'],
                'support_level': 'basic',
                'discount_rate': 0.5
            },
            'individual': {
                'criteria': {
                    'monthly_revenue': {'max': 99},
                    'user_count': {'max': 1}
                },
                'features': ['basic_features', 'community_support'],
                'support_level': 'basic'
            },
            'freemium': {
                'criteria': {
                    'monthly_revenue': {'max': 0},
                    'usage_limits': {'requests_per_month': 1000}
                },
                'features': ['limited_features', 'community_support'],
                'support_level': 'basic',
                'conversion_target': 'startup'
            }
        }

        # Forecasting Parameters
        self.FORECASTING_PARAMETERS = {
            'revenue': {
                'method': 'linear_regression',
                'historical_periods': 12,
                'confidence_interval': 0.95,
                'seasonal_adjustment': True,
                'trend_analysis': True,
                'outlier_detection': True
            },
            'usage': {
                'method': 'exponential_smoothing',
                'historical_periods': 30,
                'confidence_interval': 0.90,
                'seasonal_periods': 7,
                'smoothing_level': 0.3
            },
            'customer_growth': {
                'method': 'logistic_regression',
                'historical_periods': 24,
                'confidence_interval': 0.85,
                'market_saturation': 0.8,
                'growth_rate_estimation': True
            },
            'churn': {
                'method': 'survival_analysis',
                'historical_periods': 18,
                'confidence_interval': 0.90,
                'cohort_analysis': True,
                'risk_factors': ['usage_frequency', 'support_tickets', 'payment_issues']
            }
        }

        # Report Templates
        self.REPORT_TEMPLATES = {
            'monthly_business_review': {
                'name': 'Monthly Business Review',
                'type': 'comprehensive',
                'frequency': 'monthly',
                'sections': [
                    'executive_summary',
                    'financial_performance',
                    'customer_metrics',
                    'operational_metrics',
                    'growth_indicators',
                    'risks_and_opportunities',
                    'recommendations'
                ],
                'charts': [
                    'revenue_trend',
                    'customer_acquisition_funnel',
                    'usage_heatmap',
                    'kpi_dashboard'
                ],
                'format': ['pdf', 'json', 'excel']
            },
            'weekly_performance': {
                'name': 'Weekly Performance Report',
                'type': 'operational',
                'frequency': 'weekly',
                'sections': [
                    'key_metrics',
                    'system_performance',
                    'user_activity',
                    'alerts_summary'
                ],
                'charts': [
                    'performance_gauge',
                    'activity_timeline',
                    'error_trends'
                ],
                'format': ['json', 'html']
            },
            'customer_analytics': {
                'name': 'Customer Analytics Report',
                'type': 'customer_focused',
                'frequency': 'monthly',
                'sections': [
                    'customer_segments',
                    'journey_analysis',
                    'engagement_metrics',
                    'churn_prediction',
                    'lifetime_value_analysis'
                ],
                'charts': [
                    'segment_distribution',
                    'journey_funnel',
                    'engagement_heatmap',
                    'clv_histogram'
                ],
                'format': ['pdf', 'json']
            },
            'financial_forecast': {
                'name': 'Financial Forecast Report',
                'type': 'predictive',
                'frequency': 'quarterly',
                'sections': [
                    'forecast_summary',
                    'revenue_projections',
                    'cost_projections',
                    'profit_projections',
                    'scenario_analysis',
                    'risk_assessment'
                ],
                'charts': [
                    'forecast_chart',
                    'confidence_intervals',
                    'scenario_comparison',
                    'sensitivity_analysis'
                ],
                'format': ['pdf', 'excel']
            }
        }

        # Alert Thresholds
        self.ALERT_THRESHOLDS = {
            'revenue_drop': {
                'metric': 'revenue',
                'threshold': -10.0,  # 10% drop
                'period': 'daily',
                'severity': 'high',
                'notification_channels': ['email', 'slack']
            },
            'high_churn': {
                'metric': 'churn_rate',
                'threshold': 5.0,  # 5% churn rate
                'period': 'weekly',
                'severity': 'critical',
                'notification_channels': ['email', 'slack', 'sms']
            },
            'system_downtime': {
                'metric': 'system_uptime',
                'threshold': 99.0,  # Below 99% uptime
                'period': 'hourly',
                'severity': 'critical',
                'notification_channels': ['email', 'slack', 'pagerduty']
            },
            'slow_response': {
                'metric': 'api_response_time',
                'threshold': 2.0,  # Above 2 seconds
                'period': 'hourly',
                'severity': 'medium',
                'notification_channels': ['email', 'slack']
            },
            'high_error_rate': {
                'metric': 'error_rate',
                'threshold': 3.0,  # Above 3% error rate
                'period': 'hourly',
                'severity': 'high',
                'notification_channels': ['email', 'slack']
            }
        }


# Global BI configuration instance
bi_config = BusinessIntelligenceConfig()