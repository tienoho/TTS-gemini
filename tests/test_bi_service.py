"""
Tests for Business Intelligence Service
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np

from utils.bi_service import bi_service, BusinessIntelligenceService
from models.business_intelligence import RevenueStream, CustomerJourney, BusinessKPI
from models.analytics import UsageMetric, BusinessMetric
from models.organization import Organization
from models.user import User


class TestBusinessIntelligenceService:
    """Test cases for BI Service functionality."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return Mock()

    @pytest.fixture
    def sample_organization(self):
        """Create sample organization for testing."""
        org = Mock(spec=Organization)
        org.id = 1
        org.name = "Test Organization"
        org.tier = Mock()
        org.tier.value = "basic"
        return org

    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing."""
        user = Mock(spec=User)
        user.id = 1
        user.organization_id = 1
        return user

    def test_calculate_revenue_metrics(self, mock_db_session, sample_organization):
        """Test revenue metrics calculation."""
        # Mock revenue streams
        revenue_streams = [
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream)
        ]
        revenue_streams[0].amount = 1000.0
        revenue_streams[0].cost_of_revenue = 200.0
        revenue_streams[1].amount = 500.0
        revenue_streams[1].cost_of_revenue = 100.0

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        # Mock previous period revenue
        with patch.object(bi_service, '_get_revenue_for_period', return_value=1200.0):
            result = bi_service.calculate_revenue_metrics(
                sample_organization.id,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow(),
                mock_db_session
            )

        assert result['total_revenue'] == 1500.0
        assert result['total_cost'] == 300.0
        assert result['total_profit'] == 1200.0
        assert result['profit_margin'] == 80.0  # 1200/1500 * 100
        assert result['revenue_growth_percent'] == 25.0  # (1500-1200)/1200 * 100

    def test_calculate_customer_metrics(self, mock_db_session, sample_organization):
        """Test customer metrics calculation."""
        # Mock customer journeys
        customer_journeys = [
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney)
        ]
        customer_journeys[0].customer_id = "customer_1"
        customer_journeys[0].journey_stage = "purchase"
        customer_journeys[0].conversion_value = 100.0
        customer_journeys[1].customer_id = "customer_2"
        customer_journeys[1].journey_stage = "consideration"
        customer_journeys[1].conversion_value = 0.0
        customer_journeys[2].customer_id = "customer_1"
        customer_journeys[2].journey_stage = "retention"
        customer_journeys[2].conversion_value = 50.0

        mock_db_session.query.return_value.filter.return_value.all.return_value = customer_journeys

        # Mock total customers count
        mock_db_session.query.return_value.filter.return_value.scalar.return_value = 5

        # Mock active customers count
        with patch.object(mock_db_session.query.return_value.filter.return_value, 'scalar', return_value=3):
            result = bi_service.calculate_customer_metrics(
                sample_organization.id,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow(),
                mock_db_session
            )

        assert result['new_customers'] == 2  # customer_1 and customer_2
        assert result['total_customers'] == 5
        assert result['active_customers'] == 3
        assert result['churned_customers'] == 2
        assert result['churn_rate_percent'] == 40.0  # 2/5 * 100

    def test_calculate_kpis(self, mock_db_session, sample_organization):
        """Test KPI calculation."""
        # Mock existing KPIs
        existing_kpis = [
            Mock(spec=BusinessKPI),
            Mock(spec=BusinessKPI)
        ]
        existing_kpis[0].kpi_name = "revenue"
        existing_kpis[0].current_value = 10000.0
        existing_kpis[0].target_value = 12000.0
        existing_kpis[0].change_percent = 15.0
        existing_kpis[0].performance_status = "on_track"

        existing_kpis[1].kpi_name = "customer_acquisition"
        existing_kpis[1].current_value = 150
        existing_kpis[1].target_value = 200
        existing_kpis[1].change_percent = -10.0
        existing_kpis[1].performance_status = "at_risk"

        mock_db_session.query.return_value.filter.return_value.all.return_value = existing_kpis

        result = bi_service.calculate_kpis(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'revenue' in result
        assert 'customer_acquisition' in result
        assert result['revenue']['current_value'] == 10000.0
        assert result['customer_acquisition']['performance_status'] == 'at_risk'

    def test_analyze_usage_patterns(self, mock_db_session, sample_organization):
        """Test usage pattern analysis."""
        # Mock usage metrics
        usage_metrics = []
        base_time = datetime.utcnow()

        for i in range(24):  # 24 hours of data
            metric = Mock(spec=UsageMetric)
            metric.timestamp = base_time - timedelta(hours=i)
            metric.request_count = 100 + i * 10  # Increasing pattern
            metric.error_count = 2
            metric.avg_response_time = 0.5
            usage_metrics.append(metric)

        mock_db_session.query.return_value.filter.return_value.all.return_value = usage_metrics

        result = bi_service.analyze_usage_patterns(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'patterns' in result
        assert 'anomalies' in result
        assert 'statistics' in result
        assert 'recommendations' in result
        assert len(result['patterns']) > 0

    def test_generate_financial_forecast(self, mock_db_session, sample_organization):
        """Test financial forecast generation."""
        # Mock historical revenue data
        historical_data = [
            {'period': '2024-01', 'revenue': 10000},
            {'period': '2024-02', 'revenue': 12000},
            {'period': '2024-03', 'revenue': 11000},
            {'period': '2024-04', 'revenue': 13000},
            {'period': '2024-05', 'revenue': 12500},
            {'period': '2024-06', 'revenue': 14000}
        ]

        with patch.object(bi_service, '_get_historical_revenue_data', return_value=historical_data):
            result = bi_service.generate_financial_forecast(
                sample_organization.id, 6, mock_db_session
            )

        assert 'forecast_data' in result
        assert 'accuracy_metrics' in result
        assert 'model_confidence' in result
        assert len(result['forecast_data']) == 6  # 6 months forecast

    def test_generate_business_insights(self, mock_db_session, sample_organization):
        """Test business insights generation."""
        insights = bi_service.generate_business_insights(
            sample_organization.id, mock_db_session
        )

        assert isinstance(insights, list)
        # Check if insights contain required fields
        if insights:
            insight = insights[0]
            assert 'title' in insight
            assert 'type' in insight
            assert 'category' in insight
            assert 'description' in insight
            assert 'confidence_score' in insight
            assert 'impact_score' in insight

    def test_helper_methods(self, mock_db_session, sample_organization):
        """Test helper methods."""
        # Test _get_revenue_for_period
        mock_db_session.query.return_value.filter.return_value.scalar.return_value = 5000.0

        revenue = bi_service._get_revenue_for_period(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert revenue == 5000.0

        # Test _calculate_average_clv
        with patch.object(bi_service, '_get_revenue_for_period', return_value=10000.0):
            clv = bi_service._calculate_average_clv(sample_organization.id, mock_db_session)
            assert clv == 240000.0  # 10000 * 24 months

    def test_kpi_status_determination(self):
        """Test KPI status determination logic."""
        # Test exceeded
        assert bi_service._determine_kpi_status(120, 100) == 'exceeded'

        # Test on_track
        assert bi_service._determine_kpi_status(95, 100) == 'on_track'

        # Test at_risk
        assert bi_service._determine_kpi_status(80, 100) == 'at_risk'

        # Test off_track
        assert bi_service._determine_kpi_status(60, 100) == 'off_track'

    def test_error_handling(self, mock_db_session, sample_organization):
        """Test error handling in BI service."""
        # Test with empty data
        mock_db_session.query.return_value.filter.return_value.all.return_value = []

        result = bi_service.calculate_revenue_metrics(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['total_revenue'] == 0
        assert result['total_cost'] == 0
        assert result['total_profit'] == 0
        assert result['profit_margin'] == 0

    def test_service_initialization(self):
        """Test service initialization."""
        assert isinstance(bi_service, BusinessIntelligenceService)
        assert hasattr(bi_service, 'config')
        assert hasattr(bi_service, 'cache')