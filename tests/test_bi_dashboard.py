"""
Tests for Business Intelligence Dashboard Components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

from utils.bi_dashboard import bi_dashboard, BusinessIntelligenceDashboard
from models.business_intelligence import RevenueStream, CustomerJourney, BusinessKPI, UsagePattern
from models.analytics import UsageMetric
from models.organization import Organization


class TestBusinessIntelligenceDashboard:
    """Test cases for BI Dashboard Components functionality."""

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
        return org

    def test_revenue_dashboard(self, mock_db_session, sample_organization):
        """Test revenue dashboard generation."""
        # Mock revenue data
        revenue_streams = []
        for i in range(30):  # 30 days of revenue data
            revenue = Mock(spec=RevenueStream)
            revenue.amount = 1000 + i * 50
            revenue.created_at = datetime.utcnow() - timedelta(days=i)
            revenue.metadata = {'source': 'subscription' if i % 2 == 0 else 'one_time'}
            revenue_streams.append(revenue)

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        result = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'dashboard_type' in result
        assert 'title' in result
        assert 'charts' in result
        assert 'kpi_cards' in result
        assert 'summary' in result
        assert 'last_updated' in result

        assert result['dashboard_type'] == 'revenue'
        assert result['title'] == 'Revenue Dashboard'

        # Check charts
        charts = result['charts']
        assert len(charts) > 0
        assert all('id' in chart and 'title' in chart and 'data' in chart for chart in charts)

        # Check KPI cards
        kpi_cards = result['kpi_cards']
        assert len(kpi_cards) > 0
        assert all('title' in card and 'value' in card and 'change' in card for card in kpi_cards)

    def test_customer_dashboard(self, mock_db_session, sample_organization):
        """Test customer dashboard generation."""
        # Mock customer journey data
        customer_journeys = []
        for i in range(100):  # 100 customers
            journey = Mock(spec=CustomerJourney)
            journey.customer_id = f"customer_{i}"
            journey.total_sessions = 5 + i % 10
            journey.total_actions = 20 + i % 50
            journey.total_time_spent = 300 + i % 1800
            journey.total_conversions = i % 5
            journey.avg_engagement_score = 3.0 + (i % 7)
            journey.lifecycle_stage = ['new', 'active', 'churned'][i % 3]
            journey.segment = ['premium', 'standard', 'basic'][i % 3]
            customer_journeys.append(journey)

        mock_db_session.query.return_value.filter.return_value.all.return_value = customer_journeys

        result = bi_dashboard.generate_customer_dashboard(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['dashboard_type'] == 'customer'
        assert result['title'] == 'Customer Analytics Dashboard'

        # Check charts
        charts = result['charts']
        assert len(charts) > 0

        # Check KPI cards
        kpi_cards = result['kpi_cards']
        assert len(kpi_cards) > 0

        # Check customer segments
        assert 'customer_segments' in result
        assert 'segment_distribution' in result

    def test_usage_dashboard(self, mock_db_session, sample_organization):
        """Test usage dashboard generation."""
        # Mock usage pattern data
        usage_patterns = []
        for i in range(50):  # 50 usage patterns
            pattern = Mock(spec=UsagePattern)
            pattern.pattern_type = ['daily', 'weekly', 'monthly'][i % 3]
            pattern.frequency = 10 + i % 20
            pattern.avg_duration = 30 + i % 60
            pattern.peak_hours = [9, 10, 11, 14, 15, 16][i % 6]
            pattern.error_rate = 0.02 + (i % 5) * 0.01
            pattern.success_rate = 0.95 - (i % 5) * 0.01
            usage_patterns.append(pattern)

        # Mock usage metrics
        usage_metrics = []
        for i in range(24):  # 24 hours of usage data
            metric = Mock(spec=UsageMetric)
            metric.timestamp = datetime.utcnow() - timedelta(hours=i)
            metric.request_count = 100 + i * 10
            metric.error_count = 2 + i % 5
            metric.avg_response_time = 0.5 + (i % 10) * 0.1
            usage_metrics.append(metric)

        mock_db_session.query.return_value.filter.return_value.all.side_effect = [
            usage_patterns, usage_metrics
        ]

        result = bi_dashboard.generate_usage_dashboard(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=7),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['dashboard_type'] == 'usage'
        assert result['title'] == 'Usage Analytics Dashboard'

        # Check charts
        charts = result['charts']
        assert len(charts) > 0

        # Check KPI cards
        kpi_cards = result['kpi_cards']
        assert len(kpi_cards) > 0

        # Check usage patterns
        assert 'usage_patterns' in result
        assert 'peak_usage_hours' in result

    def test_kpi_dashboard(self, mock_db_session, sample_organization):
        """Test KPI dashboard generation."""
        # Mock KPI data
        kpis = []
        for i in range(10):  # 10 KPIs
            kpi = Mock(spec=BusinessKPI)
            kpi.name = ['Revenue Growth', 'Customer Acquisition', 'Usage Rate',
                       'Conversion Rate', 'Churn Rate', 'ARPU', 'LTV', 'CAC',
                       'ROI', 'NPS'][i]
            kpi.value = 100 + i * 10
            kpi.target = 120 + i * 10
            kpi.unit = ['%', '$', 'users', '%', '%', '$', '$', '$', '%', 'score'][i]
            kpi.category = ['financial', 'customer', 'usage', 'conversion',
                           'retention', 'financial', 'financial', 'financial',
                           'financial', 'customer'][i]
            kpi.trend = ['up', 'up', 'down', 'up', 'down', 'up', 'up', 'down', 'up', 'up'][i]
            kpis.append(kpi)

        mock_db_session.query.return_value.filter.return_value.all.return_value = kpis

        result = bi_dashboard.generate_kpi_dashboard(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['dashboard_type'] == 'kpi'
        assert result['title'] == 'Key Performance Indicators Dashboard'

        # Check KPI cards
        kpi_cards = result['kpi_cards']
        assert len(kpi_cards) > 0
        assert all('name' in card and 'value' in card and 'target' in card for card in kpi_cards)

        # Check KPI categories
        assert 'kpi_categories' in result
        assert 'overall_performance' in result

    def test_insights_dashboard(self, mock_db_session, sample_organization):
        """Test insights dashboard generation."""
        # Mock business insights data
        insights = []
        for i in range(5):  # 5 insights
            insight = Mock()
            insight.id = i + 1
            insight.title = f"Insight {i + 1}"
            insight.description = f"Description for insight {i + 1}"
            insight.category = ['revenue', 'customer', 'usage', 'performance', 'growth'][i]
            insight.severity = ['low', 'medium', 'high', 'critical', 'info'][i]
            insight.actionable = True
            insight.created_at = datetime.utcnow() - timedelta(days=i)
            insights.append(insight)

        mock_db_session.query.return_value.filter.return_value.all.return_value = insights

        result = bi_dashboard.generate_insights_dashboard(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['dashboard_type'] == 'insights'
        assert result['title'] == 'Business Insights Dashboard'

        # Check insights
        dashboard_insights = result['insights']
        assert len(dashboard_insights) > 0
        assert all('title' in insight and 'description' in insight for insight in dashboard_insights)

        # Check insights by category
        assert 'insights_by_category' in result
        assert 'insights_by_severity' in result

    def test_chart_generation_methods(self):
        """Test individual chart generation methods."""
        # Test revenue chart generation
        revenue_data = [
            {'date': '2024-01-01', 'amount': 1000, 'source': 'subscription'},
            {'date': '2024-01-02', 'amount': 1200, 'source': 'one_time'},
            {'date': '2024-01-03', 'amount': 1100, 'source': 'subscription'}
        ]

        chart = bi_dashboard._generate_revenue_chart(revenue_data)

        assert 'data' in chart
        assert 'layout' in chart
        assert 'config' in chart
        assert chart['data'][0]['type'] == 'line'

        # Test customer chart generation
        customer_data = [
            {'segment': 'premium', 'count': 50, 'percentage': 25.0},
            {'segment': 'standard', 'count': 100, 'percentage': 50.0},
            {'segment': 'basic', 'count': 50, 'percentage': 25.0}
        ]

        chart = bi_dashboard._generate_customer_chart(customer_data)

        assert chart['data'][0]['type'] == 'pie'
        assert len(chart['data'][0]['labels']) == 3

        # Test usage chart generation
        usage_data = [
            {'hour': 9, 'requests': 100},
            {'hour': 10, 'requests': 150},
            {'hour': 11, 'requests': 200}
        ]

        chart = bi_dashboard._generate_usage_chart(usage_data)

        assert chart['data'][0]['type'] == 'bar'
        assert len(chart['data'][0]['x']) == 3

    def test_kpi_card_generation(self):
        """Test KPI card generation."""
        kpi_data = {
            'name': 'Revenue Growth',
            'value': 125000,
            'target': 120000,
            'unit': '$',
            'trend': 'up',
            'change': 4.2
        }

        card = bi_dashboard._generate_kpi_card(kpi_data)

        assert 'title' in card
        assert 'value' in card
        assert 'target' in card
        assert 'unit' in card
        assert 'trend' in card
        assert 'change' in card
        assert 'status' in card

        assert card['title'] == 'Revenue Growth'
        assert card['value'] == 125000
        assert card['status'] == 'good'  # Since value > target

    def test_data_aggregation_methods(self):
        """Test data aggregation helper methods."""
        # Test _aggregate_revenue_data
        revenue_streams = [
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream)
        ]
        revenue_streams[0].amount = 1000
        revenue_streams[0].created_at = datetime(2024, 1, 1)
        revenue_streams[0].metadata = {'source': 'subscription'}

        revenue_streams[1].amount = 2000
        revenue_streams[1].created_at = datetime(2024, 1, 2)
        revenue_streams[1].metadata = {'source': 'one_time'}

        revenue_streams[2].amount = 1500
        revenue_streams[2].created_at = datetime(2024, 1, 1)
        revenue_streams[2].metadata = {'source': 'subscription'}

        aggregated = bi_dashboard._aggregate_revenue_data(revenue_streams)

        assert len(aggregated) == 2  # Two dates
        assert aggregated[0]['date'] == '2024-01-01'
        assert aggregated[0]['amount'] == 2500  # 1000 + 1500
        assert aggregated[1]['date'] == '2024-01-02'
        assert aggregated[1]['amount'] == 2000

        # Test _aggregate_customer_data
        customer_journeys = [
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney)
        ]
        customer_journeys[0].segment = 'premium'
        customer_journeys[0].lifecycle_stage = 'active'
        customer_journeys[1].segment = 'standard'
        customer_journeys[1].lifecycle_stage = 'active'
        customer_journeys[2].segment = 'premium'
        customer_journeys[2].lifecycle_stage = 'churned'

        aggregated = bi_dashboard._aggregate_customer_data(customer_journeys)

        assert len(aggregated) == 2  # Two segments
        assert aggregated[0]['segment'] == 'premium'
        assert aggregated[0]['count'] == 2
        assert aggregated[1]['segment'] == 'standard'
        assert aggregated[1]['count'] == 1

    def test_dashboard_filtering(self, mock_db_session, sample_organization):
        """Test dashboard filtering functionality."""
        # Mock revenue data with different time periods
        revenue_streams = []
        for i in range(60):  # 60 days of data
            revenue = Mock(spec=RevenueStream)
            revenue.amount = 1000 + i * 10
            revenue.created_at = datetime.utcnow() - timedelta(days=i)
            revenue.metadata = {'source': 'subscription'}
            revenue_streams.append(revenue)

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        # Test with 30-day filter
        result = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        # Should only include 30 days of data
        charts = result['charts']
        assert len(charts) > 0

        # Test with custom date range
        start_date = datetime.utcnow() - timedelta(days=15)
        end_date = datetime.utcnow() - timedelta(days=5)

        result = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            start_date,
            end_date,
            mock_db_session
        )

        assert result['dashboard_type'] == 'revenue'

    def test_dashboard_caching(self, mock_db_session, sample_organization):
        """Test dashboard caching functionality."""
        # Mock revenue data
        revenue_streams = [
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream)
        ]
        revenue_streams[0].amount = 1000
        revenue_streams[0].created_at = datetime(2024, 1, 1)
        revenue_streams[1].amount = 2000
        revenue_streams[1].created_at = datetime(2024, 1, 2)

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        # Generate dashboard first time
        result1 = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            mock_db_session
        )

        # Generate dashboard second time (should use cache if implemented)
        result2 = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            mock_db_session
        )

        assert result1['dashboard_type'] == result2['dashboard_type']
        assert result1['title'] == result2['title']

    def test_error_handling(self, mock_db_session, sample_organization):
        """Test error handling in dashboard generation."""
        # Test with empty data
        mock_db_session.query.return_value.filter.return_value.all.return_value = []

        result = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'error' in result
        assert 'No revenue data found' in result['error']

        # Test with invalid date range
        result = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            datetime.utcnow(),  # End date before start date
            datetime.utcnow() - timedelta(days=30),
            mock_db_session
        )

        assert 'error' in result
        assert 'Invalid date range' in result['error']

    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        assert isinstance(bi_dashboard, BusinessIntelligenceDashboard)
        assert hasattr(bi_dashboard, 'chart_configs')
        assert hasattr(bi_dashboard, 'kpi_configs')
        assert hasattr(bi_dashboard, 'dashboard_cache')

    def test_chart_configurations(self):
        """Test chart configuration methods."""
        # Test getting chart config
        config = bi_dashboard.get_chart_config('revenue_trend')

        assert 'type' in config
        assert 'title' in config
        assert 'x_axis' in config
        assert 'y_axis' in config

        # Test getting KPI config
        config = bi_dashboard.get_kpi_config('revenue_growth')

        assert 'title' in config
        assert 'unit' in config
        assert 'format' in config
        assert 'thresholds' in config

    def test_dashboard_export(self, mock_db_session, sample_organization):
        """Test dashboard export functionality."""
        # Mock revenue data
        revenue_streams = [
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream)
        ]
        revenue_streams[0].amount = 1000
        revenue_streams[0].created_at = datetime(2024, 1, 1)
        revenue_streams[1].amount = 2000
        revenue_streams[1].created_at = datetime(2024, 1, 2)

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        # Generate dashboard
        dashboard = bi_dashboard.generate_revenue_dashboard(
            sample_organization.id,
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            mock_db_session
        )

        # Test JSON export
        json_export = bi_dashboard.export_dashboard(dashboard, 'json')
        assert isinstance(json_export, str)

        # Test HTML export
        html_export = bi_dashboard.export_dashboard(dashboard, 'html')
        assert isinstance(html_export, str)
        assert '<html>' in html_export

        # Test invalid format
        with pytest.raises(ValueError):
            bi_dashboard.export_dashboard(dashboard, 'invalid_format')