"""
Tests for Business Intelligence Analytics Engine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from utils.bi_analytics import bi_analytics, BusinessIntelligenceAnalytics
from models.business_intelligence import RevenueStream, CustomerJourney
from models.analytics import UsageMetric
from models.organization import Organization


class TestBusinessIntelligenceAnalytics:
    """Test cases for BI Analytics Engine functionality."""

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

    def test_customer_segmentation(self, mock_db_session, sample_organization):
        """Test customer segmentation functionality."""
        # Mock customer journey data
        customer_journeys = []
        for i in range(50):  # Create 50 customers
            journey = Mock(spec=CustomerJourney)
            journey.customer_id = f"customer_{i}"
            journey.total_sessions = np.random.randint(1, 20)
            journey.total_actions = np.random.randint(5, 100)
            journey.total_time_spent = np.random.randint(30, 3600)
            journey.total_conversions = np.random.randint(0, 5)
            journey.avg_engagement_score = np.random.uniform(1, 10)
            journey.lifecycle_stage = "active"
            journey.segment = "premium" if i < 10 else "standard"
            customer_journeys.append(journey)

        mock_db_session.query.return_value.filter.return_value.all.return_value = customer_journeys

        result = bi_analytics.perform_customer_segmentation(
            sample_organization.id, mock_db_session
        )

        assert 'total_customers' in result
        assert 'number_of_segments' in result
        assert 'segments' in result
        assert 'silhouette_score' in result
        assert 'recommendations' in result
        assert result['total_customers'] == 50
        assert result['number_of_segments'] >= 2  # Should find at least 2 clusters

    def test_revenue_attribution(self, mock_db_session, sample_organization):
        """Test revenue attribution analysis."""
        # Mock revenue streams
        revenue_streams = []
        for i in range(10):
            revenue = Mock(spec=RevenueStream)
            revenue.amount = 1000 + i * 100
            revenue.metadata = {'customer_id': f'customer_{i}'}
            revenue_streams.append(revenue)

        # Mock customer journeys
        customer_journeys = []
        for i in range(10):
            for j in range(3):  # Each customer has 3 touchpoints
                journey = Mock(spec=CustomerJourney)
                journey.customer_id = f'customer_{i}"
                journey.touchpoint = ['organic', 'paid', 'direct'][j % 3]
                journey.created_at = datetime.utcnow() - timedelta(days=i)
                customer_journeys.append(journey)

        mock_db_session.query.return_value.filter.return_value.all.side_effect = [
            revenue_streams, customer_journeys
        ]

        result = bi_analytics.perform_revenue_attribution(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'total_revenue' in result
        assert 'attribution_models' in result
        assert 'insights' in result
        assert 'recommendations' in result
        assert result['total_revenue'] == sum(r.amount for r in revenue_streams)

        # Check attribution models
        attribution_models = result['attribution_models']
        assert 'last_touch' in attribution_models
        assert 'first_touch' in attribution_models
        assert 'linear' in attribution_models
        assert 'data_driven' in attribution_models

    def test_anomaly_detection(self, mock_db_session, sample_organization):
        """Test anomaly detection functionality."""
        # Create time series data with some anomalies
        usage_metrics = []
        base_time = datetime.utcnow()

        for i in range(100):  # 100 data points
            metric = Mock(spec=UsageMetric)
            metric.timestamp = base_time - timedelta(hours=i)
            metric.request_count = 100 + np.random.normal(0, 10)

            # Add some anomalies
            if i in [10, 20, 30]:
                metric.request_count = 500  # Anomalous high values
            elif i in [40, 50]:
                metric.request_count = 5    # Anomalous low values

            metric.error_count = 2 + np.random.poisson(1)
            usage_metrics.append(metric)

        mock_db_session.query.return_value.filter.return_value.all.return_value = usage_metrics

        result = bi_analytics.detect_anomalies_advanced(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=4),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'total_data_points' in result
        assert 'anomalies_detected' in result
        assert 'anomaly_rate' in result
        assert 'anomalies' in result
        assert 'insights' in result
        assert 'recommendations' in result

        # Should detect some anomalies
        assert result['anomalies_detected'] > 0
        assert result['anomaly_rate'] > 0

    def test_churn_prediction(self, mock_db_session, sample_organization):
        """Test customer churn prediction."""
        # Mock customer journey data for churn analysis
        customer_journeys = []
        for i in range(100):  # 100 customers
            journey = Mock(spec=CustomerJourney)
            journey.customer_id = f"customer_{i}"
            journey.total_sessions = np.random.randint(1, 50)
            journey.total_actions = np.random.randint(5, 200)
            journey.total_time_spent = np.random.randint(30, 7200)
            journey.total_conversions = np.random.randint(0, 10)
            journey.avg_engagement_score = np.random.uniform(0.5, 9.5)
            customer_journeys.append(journey)

        mock_db_session.query.return_value.filter.return_value.all.return_value = customer_journeys

        result = bi_analytics.predict_customer_churn(
            sample_organization.id, mock_db_session
        )

        assert 'model_performance' in result
        assert 'total_customers_analyzed' in result
        assert 'high_risk_customers' in result
        assert 'average_churn_probability' in result
        assert 'recommendations' in result

        # Check model performance metrics
        performance = result['model_performance']
        assert 'mse' in performance
        assert 'r2_score' in performance
        assert 'accuracy' in performance

    def test_demand_forecasting(self, mock_db_session, sample_organization):
        """Test demand forecasting functionality."""
        # Mock historical usage data
        historical_data = []
        base_date = datetime.utcnow().date()

        for i in range(90):  # 90 days of historical data
            date = base_date - timedelta(days=i)
            requests = 1000 + np.random.normal(0, 100) + i * 5  # Slight upward trend
            historical_data.append({
                'date': date,
                'requests': max(0, int(requests)),
                'avg_response_time': 0.5 + np.random.normal(0, 0.1)
            })

        with patch.object(bi_analytics, '_get_historical_usage_data', return_value=historical_data):
            result = bi_analytics.forecast_demand(
                sample_organization.id, 30, mock_db_session
            )

        assert 'forecast_method' in result
        assert 'historical_periods' in result
        assert 'forecast_days' in result
        assert 'forecast_data' in result
        assert 'accuracy_metrics' in result
        assert 'seasonal_patterns' in result
        assert 'recommendations' in result

        # Check forecast data structure
        forecast_data = result['forecast_data']
        assert len(forecast_data) == 30
        assert all('date' in f and 'forecasted_requests' in f for f in forecast_data)

    def test_cohort_analysis(self, mock_db_session, sample_organization):
        """Test cohort performance analysis."""
        # Mock customer journey data for cohort analysis
        customer_journeys = []
        base_date = datetime.utcnow()

        # Create customers from different months
        for month_offset in range(6):  # 6 months
            for customer_id in range(20):  # 20 customers per month
                journey = Mock(spec=CustomerJourney)
                journey.customer_id = f"customer_{month_offset}_{customer_id}"
                journey.created_at = base_date - timedelta(days=30 * month_offset)
                customer_journeys.append(journey)

        mock_db_session.query.return_value.filter.return_value.all.return_value = customer_journeys

        result = bi_analytics.analyze_cohort_performance(
            sample_organization.id, mock_db_session
        )

        assert 'total_cohorts' in result
        assert 'cohort_retention' in result
        assert 'cohort_revenue' in result
        assert 'cohort_ltv' in result
        assert 'best_performing_cohorts' in result
        assert 'struggling_cohorts' in result
        assert 'insights' in result
        assert 'recommendations' in result

        assert result['total_cohorts'] == 6  # 6 months

    def test_helper_methods(self, mock_db_session, sample_organization):
        """Test helper methods in BI analytics."""
        # Test _get_customer_data_for_segmentation
        customer_journeys = [
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney)
        ]
        customer_journeys[0].customer_id = "customer_1"
        customer_journeys[0].total_sessions = 10
        customer_journeys[0].total_actions = 50
        customer_journeys[0].total_time_spent = 300
        customer_journeys[0].total_conversions = 2
        customer_journeys[0].avg_engagement_score = 5.0
        customer_journeys[0].lifecycle_stage = "active"
        customer_journeys[0].segment = "premium"

        customer_journeys[1].customer_id = "customer_2"
        customer_journeys[1].total_sessions = 5
        customer_journeys[1].total_actions = 20
        customer_journeys[1].total_time_spent = 150
        customer_journeys[1].total_conversions = 1
        customer_journeys[1].avg_engagement_score = 4.0
        customer_journeys[1].lifecycle_stage = "active"
        customer_journeys[1].segment = "standard"

        mock_db_session.query.return_value.filter.return_value.all.return_value = customer_journeys

        customer_data = bi_analytics._get_customer_data_for_segmentation(
            sample_organization.id, mock_db_session
        )

        assert len(customer_data) == 2
        assert customer_data[0]['customer_id'] == "customer_1"
        assert customer_data[0]['avg_engagement_score'] == 5.0  # 50/10

    def test_clustering_methods(self):
        """Test clustering-related methods."""
        # Test _find_optimal_clusters
        X_scaled = np.random.rand(50, 5)  # 50 samples, 5 features

        optimal_clusters = bi_analytics._find_optimal_clusters(X_scaled)

        assert isinstance(optimal_clusters, int)
        assert 2 <= optimal_clusters <= 6  # Should be within reasonable range

        # Test _prepare_customer_features
        customer_data = [
            {
                'total_sessions': 10,
                'total_actions': 50,
                'total_time_spent': 300,
                'total_conversions': 2,
                'avg_engagement_score': 5.0
            },
            {
                'total_sessions': 5,
                'total_actions': 20,
                'total_time_spent': 150,
                'total_conversions': 1,
                'avg_engagement_score': 4.0
            }
        ]

        features = bi_analytics._prepare_customer_features(customer_data)

        assert features.shape == (2, 5)
        assert features[0, 0] == 10  # total_sessions
        assert features[0, 1] == 50  # total_actions

    def test_attribution_methods(self):
        """Test attribution calculation methods."""
        # Create mock data
        revenue_streams = [
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream)
        ]
        revenue_streams[0].amount = 1000.0
        revenue_streams[0].metadata = {'customer_id': 'customer_1'}
        revenue_streams[1].amount = 500.0
        revenue_streams[1].metadata = {'customer_id': 'customer_2'}

        customer_journeys = [
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney)
        ]
        customer_journeys[0].customer_id = 'customer_1'
        customer_journeys[0].touchpoint = 'organic'
        customer_journeys[0].created_at = datetime.utcnow() - timedelta(days=2)
        customer_journeys[1].customer_id = 'customer_1'
        customer_journeys[1].touchpoint = 'paid'
        customer_journeys[1].created_at = datetime.utcnow() - timedelta(days=1)
        customer_journeys[2].customer_id = 'customer_2'
        customer_journeys[2].touchpoint = 'direct'
        customer_journeys[2].created_at = datetime.utcnow()

        # Test last touch attribution
        last_touch = bi_analytics._calculate_last_touch_attribution(
            revenue_streams, customer_journeys
        )

        assert 'model' in last_touch
        assert 'attribution' in last_touch
        assert 'total_attributed_revenue' in last_touch
        assert last_touch['model'] == 'last_touch'
        assert last_touch['total_attributed_revenue'] == 1500.0

        # Test first touch attribution
        first_touch = bi_analytics._calculate_first_touch_attribution(
            revenue_streams, customer_journeys
        )

        assert first_touch['model'] == 'first_touch'
        assert first_touch['total_attributed_revenue'] == 1500.0

    def test_anomaly_detection_methods(self):
        """Test anomaly detection helper methods."""
        # Create time series data
        time_series_data = [
            {
                'timestamp': datetime.utcnow() - timedelta(hours=i),
                'metric': 'requests',
                'value': 100 + i * 5
            }
            for i in range(50)
        ]

        # Add some anomalies
        time_series_data[10]['value'] = 500  # High anomaly
        time_series_data[20]['value'] = 10   # Low anomaly

        # Test _prepare_anomaly_features
        features = bi_analytics._prepare_anomaly_features(time_series_data)

        assert features.shape[0] == len(time_series_data)
        assert features.shape[1] == 5  # 5 features per data point

        # Test _describe_anomaly
        anomaly_description = bi_analytics._describe_anomaly(
            time_series_data[10], 0.8
        )

        assert 'anomaly' in anomaly_description.lower()
        assert 'requests' in anomaly_description.lower()

    def test_forecasting_methods(self):
        """Test forecasting helper methods."""
        # Create sample DataFrame
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'requests': [100 + i * 2 + np.random.normal(0, 5) for i in range(30)]
        }, index=dates)

        # Test _identify_seasonal_patterns
        patterns = bi_analytics._identify_seasonal_patterns(df)

        assert 'daily_pattern' in patterns
        assert 'hourly_pattern' in patterns
        assert 'peak_day' in patterns
        assert 'peak_hour' in patterns

        # Test _calculate_forecast_accuracy_metrics
        accuracy_metrics = bi_analytics._calculate_forecast_accuracy_metrics(
            Mock(), df
        )

        assert 'mae' in accuracy_metrics
        assert 'rmse' in accuracy_metrics
        assert 'mape' in accuracy_metrics

    def test_cohort_methods(self):
        """Test cohort analysis helper methods."""
        # Create mock journey data
        customer_journeys = []
        base_date = datetime.utcnow()

        for month in range(3):
            for customer in range(10):
                journey = Mock(spec=CustomerJourney)
                journey.customer_id = f"customer_{month}_{customer}"
                journey.created_at = base_date - timedelta(days=30 * month)
                customer_journeys.append(journey)

        # Test _create_customer_cohorts
        cohorts = bi_analytics._create_customer_cohorts(customer_journeys)

        assert len(cohorts) == 3  # 3 monthly cohorts
        assert len(cohorts[list(cohorts.keys())[0]]) == 10  # 10 customers per cohort

        # Test _calculate_cohort_retention
        retention = bi_analytics._calculate_cohort_retention(cohorts)

        assert len(retention) == 3
        assert all('cohort_size' in v for v in retention.values())
        assert all('retention_rate' in v for v in retention.values())

    def test_error_handling(self, mock_db_session, sample_organization):
        """Test error handling in BI analytics."""
        # Test with insufficient data
        mock_db_session.query.return_value.filter.return_value.all.return_value = []

        result = bi_analytics.perform_customer_segmentation(
            sample_organization.id, mock_db_session
        )

        assert 'error' in result
        assert 'Insufficient customer data' in result['error']

        # Test anomaly detection with insufficient data
        result = bi_analytics.detect_anomalies_advanced(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'error' in result
        assert 'Insufficient data' in result['error']

    def test_analytics_initialization(self):
        """Test analytics engine initialization."""
        assert isinstance(bi_analytics, BusinessIntelligenceAnalytics)
        assert hasattr(bi_analytics, 'scaler')
        assert hasattr(bi_analytics, 'customer_segments')
        assert hasattr(bi_analytics, 'revenue_attribution_models')