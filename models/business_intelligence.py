"""
Business Intelligence Models for TTS System
Extends analytics models with advanced BI capabilities
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, DateTime, Integer, String, Float, JSON, Boolean, Text, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from models.analytics import Base


class RevenueModel(str, Enum):
    """Revenue calculation models."""

    SUBSCRIPTION = "subscription"
    PAY_PER_USE = "pay_per_use"
    HYBRID = "hybrid"
    ENTERPRISE = "enterprise"


class CustomerSegment(str, Enum):
    """Customer segmentation categories."""

    ENTERPRISE = "enterprise"
    SME = "sme"
    STARTUP = "startup"
    INDIVIDUAL = "individual"
    FREEMIUM = "freemium"


class KPICategory(str, Enum):
    """KPI categories for business intelligence."""

    FINANCIAL = "financial"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"
    PRODUCT = "product"
    GROWTH = "growth"


class RevenueStream(Base):
    """Detailed revenue stream tracking."""

    __tablename__ = 'revenue_streams'

    id = Column(Integer, primary_key=True, index=True)

    # Organization and user context
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)

    # Revenue details
    stream_type = Column(String(50), nullable=False, index=True)  # subscription, usage, premium_features
    stream_name = Column(String(100), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default='USD')

    # Billing period
    billing_period_start = Column(DateTime, nullable=False, index=True)
    billing_period_end = Column(DateTime, nullable=False, index=True)
    billing_cycle = Column(String(20), default='monthly')  # monthly, yearly, one_time

    # Revenue recognition
    recognized_at = Column(DateTime, default=datetime.utcnow, index=True)
    recognition_method = Column(String(20), default='immediate')  # immediate, ratable, milestone

    # Cost attribution
    cost_of_revenue = Column(Float, default=0.0)
    gross_margin = Column(Float, default=0.0)

    # Customer context
    customer_segment = Column(String(20), index=True)
    customer_tier = Column(String(20), index=True)
    customer_lifetime_value = Column(Float, default=0.0)

    # Product context
    product_category = Column(String(50), index=True)
    feature_usage = Column(JSON, default=dict)  # Features used that generated revenue

    # Metadata
    transaction_id = Column(String(100), index=True)
    payment_method = Column(String(50))
    payment_status = Column(String(20), default='completed', index=True)
    request_metadata = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes for performance
    __table_args__ = (
        Index('idx_revenue_streams_org_period', 'organization_id', 'billing_period_start'),
        Index('idx_revenue_streams_type_amount', 'stream_type', 'amount'),
        Index('idx_revenue_streams_recognized', 'recognized_at'),
        Index('idx_revenue_streams_customer', 'customer_segment', 'customer_tier'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'stream_type': self.stream_type,
            'stream_name': self.stream_name,
            'amount': self.amount,
            'currency': self.currency,
            'billing_period_start': self.billing_period_start.isoformat() if self.billing_period_start is not None else None,
            'billing_period_end': self.billing_period_end.isoformat() if self.billing_period_end is not None else None,
            'billing_cycle': self.billing_cycle,
            'recognized_at': self.recognized_at.isoformat() if self.recognized_at is not None else None,
            'recognition_method': self.recognition_method,
            'cost_of_revenue': self.cost_of_revenue,
            'gross_margin': self.gross_margin,
            'customer_segment': self.customer_segment,
            'customer_tier': self.customer_tier,
            'customer_lifetime_value': self.customer_lifetime_value,
            'product_category': self.product_category,
            'feature_usage': self.feature_usage,
            'transaction_id': self.transaction_id,
            'payment_method': self.payment_method,
            'payment_status': self.payment_status,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None,
        }


class CustomerJourney(Base):
    """Customer journey tracking for BI analysis."""

    __tablename__ = 'customer_journeys'

    id = Column(Integer, primary_key=True, index=True)

    # Customer identification
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    customer_id = Column(String(100), index=True)  # External customer ID

    # Journey stages
    journey_stage = Column(String(50), index=True)  # awareness, consideration, purchase, retention, advocacy
    journey_step = Column(String(100), index=True)  # Specific step in journey
    previous_stage = Column(String(50))

    # Journey context
    touchpoint = Column(String(100), index=True)  # API, dashboard, email, etc.
    channel = Column(String(50), index=True)  # direct, referral, organic, paid
    campaign_id = Column(String(100))  # Marketing campaign identifier

    # Customer behavior
    session_id = Column(String(100), index=True)
    actions_taken = Column(JSON, default=list)  # List of actions in this step
    time_spent_seconds = Column(Float, default=0.0)
    conversion_value = Column(Float, default=0.0)  # Value of conversion event

    # Attribution
    attribution_model = Column(String(50), default='last_touch')  # last_touch, first_touch, linear, etc.
    attribution_weight = Column(Float, default=1.0)

    # Customer segmentation
    segment = Column(String(50), index=True)
    lifecycle_stage = Column(String(50), index=True)  # new, active, at_risk, churned
    engagement_score = Column(Float, default=0.0)

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Indexes
    __table_args__ = (
        Index('idx_customer_journeys_customer_stage', 'customer_id', 'journey_stage'),
        Index('idx_customer_journeys_org_stage', 'organization_id', 'journey_stage'),
        Index('idx_customer_journeys_touchpoint', 'touchpoint', 'created_at'),
        Index('idx_customer_journeys_lifecycle', 'lifecycle_stage', 'engagement_score'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'customer_id': self.customer_id,
            'journey_stage': self.journey_stage,
            'journey_step': self.journey_step,
            'previous_stage': self.previous_stage,
            'touchpoint': self.touchpoint,
            'channel': self.channel,
            'campaign_id': self.campaign_id,
            'session_id': self.session_id,
            'actions_taken': self.actions_taken,
            'time_spent_seconds': self.time_spent_seconds,
            'conversion_value': self.conversion_value,
            'attribution_model': self.attribution_model,
            'attribution_weight': self.attribution_weight,
            'segment': self.segment,
            'lifecycle_stage': self.lifecycle_stage,
            'engagement_score': self.engagement_score,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
        }


class BusinessKPI(Base):
    """Key Performance Indicators for business intelligence."""

    __tablename__ = 'business_kpis'

    id = Column(Integer, primary_key=True, index=True)

    # KPI identification
    kpi_name = Column(String(100), nullable=False, index=True)
    kpi_category = Column(String(20), nullable=False, index=True)  # financial, customer, operational, product, growth
    kpi_type = Column(String(20), nullable=False, index=True)  # metric, ratio, percentage, currency

    # KPI definition
    kpi_formula = Column(Text, nullable=True)  # Formula for calculation
    kpi_description = Column(Text, nullable=True)
    target_value = Column(Float, nullable=True)
    target_period = Column(String(20), default='monthly')  # daily, weekly, monthly, quarterly, yearly

    # Current value and tracking
    current_value = Column(Float, nullable=False)
    previous_value = Column(Float, nullable=True)
    change_percent = Column(Float, default=0.0)
    change_direction = Column(String(10), default='stable')  # up, down, stable

    # Performance assessment
    performance_status = Column(String(20), default='on_track', index=True)  # on_track, at_risk, off_track, exceeded
    confidence_level = Column(Float, default=1.0)  # 0.0 to 1.0

    # Time dimensions
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), default='monthly', index=True)

    # Organization context
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    department = Column(String(50))  # Sales, Marketing, Operations, etc.
    owner_id = Column(Integer, ForeignKey('users.id'))  # KPI owner

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_business_kpis_org_category', 'organization_id', 'kpi_category'),
        Index('idx_business_kpis_period_status', 'period_start', 'performance_status'),
        Index('idx_business_kpis_name_period', 'kpi_name', 'period_start'),
        Index('idx_business_kpis_owner', 'owner_id', 'period_start'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'kpi_name': self.kpi_name,
            'kpi_category': self.kpi_category,
            'kpi_type': self.kpi_type,
            'kpi_formula': self.kpi_formula,
            'kpi_description': self.kpi_description,
            'target_value': self.target_value,
            'target_period': self.target_period,
            'current_value': self.current_value,
            'previous_value': self.previous_value,
            'change_percent': self.change_percent,
            'change_direction': self.change_direction,
            'performance_status': self.performance_status,
            'confidence_level': self.confidence_level,
            'period_start': self.period_start.isoformat() if self.period_start is not None else None,
            'period_end': self.period_end.isoformat() if self.period_end is not None else None,
            'period_type': self.period_type,
            'organization_id': self.organization_id,
            'department': self.department,
            'owner_id': self.owner_id,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None,
        }


class UsagePattern(Base):
    """Advanced usage pattern analysis for BI."""

    __tablename__ = 'usage_patterns'

    id = Column(Integer, primary_key=True, index=True)

    # Pattern identification
    pattern_name = Column(String(100), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)  # seasonal, trend, cyclical, irregular
    pattern_category = Column(String(50), index=True)  # user_behavior, system_load, revenue, etc.

    # Pattern characteristics
    pattern_strength = Column(Float, default=0.0)  # 0.0 to 1.0
    pattern_confidence = Column(Float, default=0.0)  # 0.0 to 1.0
    pattern_duration_hours = Column(Float, default=0.0)

    # Pattern metrics
    average_value = Column(Float, default=0.0)
    peak_value = Column(Float, default=0.0)
    trough_value = Column(Float, default=0.0)
    volatility = Column(Float, default=0.0)

    # Time characteristics
    peak_time = Column(DateTime, nullable=True)
    trough_time = Column(DateTime, nullable=True)
    cycle_length_hours = Column(Float, nullable=True)  # For cyclical patterns

    # Pattern detection
    detection_method = Column(String(50), default='statistical')  # statistical, ml, rule_based
    detection_confidence = Column(Float, default=0.0)
    detection_metadata = Column(JSON, default=dict)

    # Impact analysis
    business_impact = Column(String(20), default='medium', index=True)  # low, medium, high, critical
    revenue_impact = Column(Float, default=0.0)
    cost_impact = Column(Float, default=0.0)

    # Organization context
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    affected_endpoints = Column(JSON, default=list)  # API endpoints affected
    affected_features = Column(JSON, default=list)  # Features affected

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_usage_patterns_org_type', 'organization_id', 'pattern_type'),
        Index('idx_usage_patterns_category_strength', 'pattern_category', 'pattern_strength'),
        Index('idx_usage_patterns_impact', 'business_impact', 'revenue_impact'),
        Index('idx_usage_patterns_detection', 'detection_method', 'detection_confidence'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'pattern_name': self.pattern_name,
            'pattern_type': self.pattern_type,
            'pattern_category': self.pattern_category,
            'pattern_strength': self.pattern_strength,
            'pattern_confidence': self.pattern_confidence,
            'pattern_duration_hours': self.pattern_duration_hours,
            'average_value': self.average_value,
            'peak_value': self.peak_value,
            'trough_value': self.trough_value,
            'volatility': self.volatility,
            'peak_time': self.peak_time.isoformat() if self.peak_time is not None else None,
            'trough_time': self.trough_time.isoformat() if self.trough_time is not None else None,
            'cycle_length_hours': self.cycle_length_hours,
            'detection_method': self.detection_method,
            'detection_confidence': self.detection_confidence,
            'detection_metadata': self.detection_metadata,
            'business_impact': self.business_impact,
            'revenue_impact': self.revenue_impact,
            'cost_impact': self.cost_impact,
            'organization_id': self.organization_id,
            'affected_endpoints': self.affected_endpoints,
            'affected_features': self.affected_features,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None,
        }


class FinancialProjection(Base):
    """Financial forecasting and projections."""

    __tablename__ = 'financial_projections'

    id = Column(Integer, primary_key=True, index=True)

    # Projection identification
    projection_name = Column(String(100), nullable=False, index=True)
    projection_type = Column(String(50), nullable=False, index=True)  # revenue, cost, profit, cash_flow
    projection_method = Column(String(50), default='linear_regression')  # linear_regression, exponential, etc.

    # Projection parameters
    historical_periods = Column(Integer, default=12)  # Number of historical periods used
    forecast_periods = Column(Integer, default=12)  # Number of periods to forecast
    confidence_interval = Column(Float, default=0.95)  # 0.0 to 1.0

    # Projection results
    projected_values = Column(JSON, nullable=False)  # List of projected values
    confidence_intervals = Column(JSON, default=list)  # Confidence intervals for each projection
    accuracy_metrics = Column(JSON, default=dict)  # MAE, RMSE, MAPE, etc.

    # Model performance
    model_accuracy = Column(Float, default=0.0)
    model_error = Column(Float, default=0.0)
    model_drift = Column(Float, default=0.0)  # Model drift over time

    # Scenario analysis
    scenario_type = Column(String(20), default='baseline', index=True)  # baseline, optimistic, pessimistic
    scenario_assumptions = Column(JSON, default=dict)  # Assumptions for this scenario

    # Organization context
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    department = Column(String(50))

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_financial_projections_org_type', 'organization_id', 'projection_type'),
        Index('idx_financial_projections_scenario', 'scenario_type', 'created_at'),
        Index('idx_financial_projections_method', 'projection_method', 'model_accuracy'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'projection_name': self.projection_name,
            'projection_type': self.projection_type,
            'projection_method': self.projection_method,
            'historical_periods': self.historical_periods,
            'forecast_periods': self.forecast_periods,
            'confidence_interval': self.confidence_interval,
            'projected_values': self.projected_values,
            'confidence_intervals': self.confidence_intervals,
            'accuracy_metrics': self.accuracy_metrics,
            'model_accuracy': self.model_accuracy,
            'model_error': self.model_error,
            'model_drift': self.model_drift,
            'scenario_type': self.scenario_type,
            'scenario_assumptions': self.scenario_assumptions,
            'organization_id': self.organization_id,
            'department': self.department,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class BusinessInsight(Base):
    """AI-generated business insights and recommendations."""

    __tablename__ = 'business_insights'

    id = Column(Integer, primary_key=True, index=True)

    # Insight identification
    insight_title = Column(String(200), nullable=False, index=True)
    insight_type = Column(String(50), nullable=False, index=True)  # opportunity, risk, trend, anomaly
    insight_category = Column(String(50), index=True)  # revenue, customer, operations, product

    # Insight content
    insight_description = Column(Text, nullable=False)
    insight_summary = Column(Text, nullable=True)
    confidence_score = Column(Float, default=0.0)  # 0.0 to 1.0
    impact_score = Column(Float, default=0.0)  # 0.0 to 1.0

    # Insight details
    root_causes = Column(JSON, default=list)  # Root causes identified
    contributing_factors = Column(JSON, default=list)  # Contributing factors
    affected_metrics = Column(JSON, default=list)  # Metrics affected

    # Recommendations
    recommendations = Column(JSON, default=list)  # Recommended actions
    expected_benefits = Column(JSON, default=list)  # Expected benefits of actions
    implementation_effort = Column(String(20), default='medium')  # low, medium, high

    # Time context
    time_period_start = Column(DateTime, nullable=False, index=True)
    time_period_end = Column(DateTime, nullable=False, index=True)
    insight_horizon = Column(String(20), default='short_term')  # short_term, medium_term, long_term

    # Organization context
    organization_id = Column(Integer, ForeignKey('organizations.id'), index=True)
    department = Column(String(50))
    priority = Column(String(20), default='medium', index=True)  # low, medium, high, critical

    # Status and tracking
    status = Column(String(20), default='new', index=True)  # new, reviewed, implemented, dismissed
    reviewed_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    implemented_at = Column(DateTime, nullable=True)

    # Metadata
    request_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_business_insights_org_type', 'organization_id', 'insight_type'),
        Index('idx_business_insights_category_priority', 'insight_category', 'priority'),
        Index('idx_business_insights_status', 'status', 'created_at'),
        Index('idx_business_insights_confidence', 'confidence_score', 'impact_score'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'insight_title': self.insight_title,
            'insight_type': self.insight_type,
            'insight_category': self.insight_category,
            'insight_description': self.insight_description,
            'insight_summary': self.insight_summary,
            'confidence_score': self.confidence_score,
            'impact_score': self.impact_score,
            'root_causes': self.root_causes,
            'contributing_factors': self.contributing_factors,
            'affected_metrics': self.affected_metrics,
            'recommendations': self.recommendations,
            'expected_benefits': self.expected_benefits,
            'implementation_effort': self.implementation_effort,
            'time_period_start': self.time_period_start.isoformat() if self.time_period_start is not None else None,
            'time_period_end': self.time_period_end.isoformat() if self.time_period_end is not None else None,
            'insight_horizon': self.insight_horizon,
            'organization_id': self.organization_id,
            'department': self.department,
            'priority': self.priority,
            'status': self.status,
            'reviewed_by': self.reviewed_by,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at is not None else None,
            'implemented_at': self.implemented_at.isoformat() if self.implemented_at is not None else None,
            'metadata': self.request_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }