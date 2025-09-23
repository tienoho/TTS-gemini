"""
Tests for Business Intelligence API Routes
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
from flask import Flask
from flask_jwt_extended import create_access_token

from routes.business_intelligence import bi_bp
from utils.bi_service import bi_service
from utils.bi_analytics import bi_analytics
from utils.bi_dashboard import bi_dashboard
from utils.bi_reporting import bi_reporting
from models.organization import Organization


class TestBusinessIntelligenceAPI:
    """Test cases for BI API Routes functionality."""

    @pytest.fixture
    def app(self):
        """Create Flask test app."""
        app = Flask(__name__)
        app.register_blueprint(bi_bp)
        app.config['JWT_SECRET_KEY'] = 'test-secret-key'
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

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

    @pytest.fixture
    def auth_headers(self, sample_organization):
        """Create authentication headers."""
        with patch('utils.tenant_manager.get_organization_by_id') as mock_get_org:
            mock_get_org.return_value = sample_organization
            token = create_access_token(identity=str(sample_organization.id))
            return {'Authorization': f'Bearer {token}'}

    def test_revenue_analytics_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/revenue endpoint."""
        # Mock BI service response
        mock_response = {
            'total_revenue': 50000,
            'monthly_revenue': 25000,
            'revenue_by_source': {'subscription': 40000, 'one_time': 10000},
            'growth_rate': 15.5,
            'period': '30 days'
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_revenue_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = mock_response

            response = client.get('/bi/revenue', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total_revenue'] == 50000
            assert data['growth_rate'] == 15.5
            mock_get_metrics.assert_called_once()

    def test_customer_analytics_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/customers endpoint."""
        # Mock BI service response
        mock_response = {
            'total_customers': 1000,
            'active_customers': 850,
            'new_customers': 150,
            'churned_customers': 50,
            'customer_segments': {
                'premium': 200,
                'standard': 600,
                'basic': 200
            },
            'average_lifetime_value': 250.0
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_customer_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = mock_response

            response = client.get('/bi/customers', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total_customers'] == 1000
            assert data['active_customers'] == 850
            mock_get_metrics.assert_called_once()

    def test_usage_analytics_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/usage endpoint."""
        # Mock BI service response
        mock_response = {
            'total_requests': 50000,
            'average_daily_requests': 1667,
            'peak_usage_hours': [14, 15, 16],
            'error_rate': 0.02,
            'average_response_time': 0.45,
            'usage_patterns': {
                'daily': 60,
                'weekly': 30,
                'monthly': 10
            }
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_usage_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = mock_response

            response = client.get('/bi/usage', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total_requests'] == 50000
            assert data['error_rate'] == 0.02
            mock_get_metrics.assert_called_once()

    def test_kpis_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/kpis endpoint."""
        # Mock BI service response
        mock_response = {
            'kpis': [
                {
                    'name': 'Revenue Growth',
                    'value': 125000,
                    'target': 120000,
                    'unit': '$',
                    'status': 'good'
                },
                {
                    'name': 'Customer Acquisition',
                    'value': 150,
                    'target': 200,
                    'unit': 'users',
                    'status': 'warning'
                }
            ],
            'overall_performance': 'good',
            'kpis_on_target': 8,
            'total_kpis': 10
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_kpi_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = mock_response

            response = client.get('/bi/kpis', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['kpis']) == 2
            assert data['overall_performance'] == 'good'
            mock_get_metrics.assert_called_once()

    def test_reports_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/reports endpoint."""
        # Mock BI reporting response
        mock_response = {
            'reports': [
                {
                    'id': 'report_1',
                    'title': 'Monthly Revenue Report',
                    'type': 'revenue',
                    'status': 'completed',
                    'created_at': datetime.utcnow().isoformat()
                },
                {
                    'id': 'report_2',
                    'title': 'Customer Analytics Report',
                    'type': 'customer',
                    'status': 'completed',
                    'created_at': datetime.utcnow().isoformat()
                }
            ],
            'total_reports': 2
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_reporting, 'get_available_reports') as mock_get_reports:

            mock_get_db.return_value = mock_db_session
            mock_get_reports.return_value = mock_response

            response = client.get('/bi/reports', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['reports']) == 2
            assert data['total_reports'] == 2
            mock_get_reports.assert_called_once()

    def test_forecasting_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/forecasting endpoint."""
        # Mock BI analytics response
        mock_response = {
            'forecast_method': 'exponential_smoothing',
            'forecast_days': 30,
            'forecast_data': [
                {'date': '2024-02-01', 'forecasted_requests': 1800},
                {'date': '2024-02-02', 'forecasted_requests': 1850}
            ],
            'accuracy_metrics': {
                'mae': 50.5,
                'rmse': 65.2,
                'mape': 0.03
            },
            'seasonal_patterns': {
                'peak_day': 'Wednesday',
                'peak_hour': 14
            }
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_analytics, 'forecast_demand') as mock_forecast:

            mock_get_db.return_value = mock_db_session
            mock_forecast.return_value = mock_response

            response = client.get('/bi/forecasting?days=30', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['forecast_days'] == 30
            assert len(data['forecast_data']) == 2
            mock_forecast.assert_called_once()

    def test_insights_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/insights endpoint."""
        # Mock BI analytics response
        mock_response = {
            'insights': [
                {
                    'id': 'insight_1',
                    'title': 'Revenue Growth Opportunity',
                    'description': 'Revenue has increased by 15% this month',
                    'category': 'revenue',
                    'severity': 'info',
                    'actionable': True
                },
                {
                    'id': 'insight_2',
                    'title': 'Customer Churn Alert',
                    'description': 'Customer churn rate has increased',
                    'category': 'customer',
                    'severity': 'warning',
                    'actionable': True
                }
            ],
            'total_insights': 2,
            'insights_by_category': {
                'revenue': 1,
                'customer': 1
            }
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_analytics, 'generate_business_insights') as mock_insights:

            mock_get_db.return_value = mock_db_session
            mock_insights.return_value = mock_response

            response = client.get('/bi/insights', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['insights']) == 2
            assert data['total_insights'] == 2
            mock_insights.assert_called_once()

    def test_anomalies_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/anomalies endpoint."""
        # Mock BI analytics response
        mock_response = {
            'total_data_points': 1000,
            'anomalies_detected': 5,
            'anomaly_rate': 0.005,
            'anomalies': [
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'metric': 'requests',
                    'value': 500,
                    'expected_range': [100, 200],
                    'severity': 'high'
                }
            ],
            'insights': ['Unusual spike in requests detected'],
            'recommendations': ['Investigate the cause of the spike']
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_analytics, 'detect_anomalies_advanced') as mock_detect:

            mock_get_db.return_value = mock_db_session
            mock_detect.return_value = mock_response

            response = client.get('/bi/anomalies', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['anomalies_detected'] == 5
            assert len(data['anomalies']) == 1
            mock_detect.assert_called_once()

    def test_segmentation_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/segmentation endpoint."""
        # Mock BI analytics response
        mock_response = {
            'total_customers': 1000,
            'number_of_segments': 4,
            'segments': [
                {
                    'name': 'High Value',
                    'size': 200,
                    'percentage': 20.0,
                    'characteristics': ['High engagement', 'High spending']
                },
                {
                    'name': 'Regular',
                    'size': 600,
                    'percentage': 60.0,
                    'characteristics': ['Regular usage', 'Medium spending']
                }
            ],
            'silhouette_score': 0.75,
            'recommendations': ['Target high-value customers with premium offers']
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_analytics, 'perform_customer_segmentation') as mock_segment:

            mock_get_db.return_value = mock_db_session
            mock_segment.return_value = mock_response

            response = client.get('/bi/segmentation', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['number_of_segments'] == 4
            assert len(data['segments']) == 2
            mock_segment.assert_called_once()

    def test_attribution_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/attribution endpoint."""
        # Mock BI analytics response
        mock_response = {
            'total_revenue': 50000,
            'attribution_models': {
                'last_touch': {
                    'model': 'last_touch',
                    'attribution': {'organic': 20000, 'paid': 30000},
                    'total_attributed_revenue': 50000
                },
                'first_touch': {
                    'model': 'first_touch',
                    'attribution': {'organic': 25000, 'paid': 25000},
                    'total_attributed_revenue': 50000
                }
            },
            'insights': ['Paid marketing is driving more conversions'],
            'recommendations': ['Increase investment in paid marketing']
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_analytics, 'perform_revenue_attribution') as mock_attribution:

            mock_get_db.return_value = mock_db_session
            mock_attribution.return_value = mock_response

            response = client.get('/bi/attribution', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total_revenue'] == 50000
            assert 'last_touch' in data['attribution_models']
            mock_attribution.assert_called_once()

    def test_churn_prediction_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/churn endpoint."""
        # Mock BI analytics response
        mock_response = {
            'model_performance': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.78,
                'f1_score': 0.80
            },
            'total_customers_analyzed': 1000,
            'high_risk_customers': 50,
            'average_churn_probability': 0.05,
            'churn_prediction': [
                {
                    'customer_id': 'customer_1',
                    'churn_probability': 0.85,
                    'risk_level': 'high',
                    'factors': ['Decreased usage', 'Support tickets']
                }
            ],
            'recommendations': ['Implement retention campaigns for high-risk customers']
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_analytics, 'predict_customer_churn') as mock_churn:

            mock_get_db.return_value = mock_db_session
            mock_churn.return_value = mock_response

            response = client.get('/bi/churn', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['high_risk_customers'] == 50
            assert data['average_churn_probability'] == 0.05
            mock_churn.assert_called_once()

    def test_cohorts_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/cohorts endpoint."""
        # Mock BI analytics response
        mock_response = {
            'total_cohorts': 6,
            'cohort_retention': {
                '2024-01': {
                    'cohort_size': 100,
                    'retention_rate': 0.85,
                    'revenue_per_user': 250.0
                },
                '2024-02': {
                    'cohort_size': 120,
                    'retention_rate': 0.78,
                    'revenue_per_user': 220.0
                }
            },
            'cohort_revenue': {
                '2024-01': 25000,
                '2024-02': 26400
            },
            'best_performing_cohorts': ['2024-01'],
            'struggling_cohorts': ['2024-02'],
            'insights': ['January cohort shows strong retention'],
            'recommendations': ['Analyze factors contributing to January success']
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_analytics, 'analyze_cohort_performance') as mock_cohorts:

            mock_get_db.return_value = mock_db_session
            mock_cohorts.return_value = mock_response

            response = client.get('/bi/cohorts', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total_cohorts'] == 6
            assert '2024-01' in data['cohort_retention']
            mock_cohorts.assert_called_once()

    def test_dashboard_endpoint(self, client, mock_db_session, auth_headers):
        """Test GET /bi/dashboard endpoint."""
        # Mock BI dashboard response
        mock_response = {
            'dashboard_type': 'comprehensive',
            'title': 'Business Intelligence Dashboard',
            'charts': [
                {
                    'id': 'revenue_chart',
                    'title': 'Revenue Trend',
                    'type': 'line',
                    'data': {'x': ['Jan', 'Feb'], 'y': [1000, 1200]}
                }
            ],
            'kpi_cards': [
                {
                    'title': 'Total Revenue',
                    'value': 50000,
                    'unit': '$',
                    'status': 'good'
                }
            ],
            'last_updated': datetime.utcnow().isoformat()
        }

        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_dashboard, 'generate_comprehensive_dashboard') as mock_dashboard:

            mock_get_db.return_value = mock_db_session
            mock_dashboard.return_value = mock_response

            response = client.get('/bi/dashboard', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['dashboard_type'] == 'comprehensive'
            assert len(data['charts']) == 1
            mock_dashboard.assert_called_once()

    def test_unauthorized_access(self, client, mock_db_session):
        """Test unauthorized access to BI endpoints."""
        response = client.get('/bi/revenue')

        assert response.status_code == 401

    def test_invalid_date_range(self, client, auth_headers):
        """Test API with invalid date range."""
        response = client.get('/bi/revenue?start_date=2024-12-31&end_date=2024-01-01', headers=auth_headers)

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_query_parameters(self, client, mock_db_session, auth_headers):
        """Test API endpoints with query parameters."""
        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_revenue_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = {'total_revenue': 50000}

            # Test with date range parameters
            response = client.get(
                '/bi/revenue?start_date=2024-01-01&end_date=2024-01-31',
                headers=auth_headers
            )

            assert response.status_code == 200
            mock_get_metrics.assert_called_once()

            # Check that date parameters were passed correctly
            call_args = mock_get_metrics.call_args
            assert len(call_args[0]) >= 2  # At least org_id and date parameters

    def test_error_handling(self, client, mock_db_session, auth_headers):
        """Test error handling in BI API endpoints."""
        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_revenue_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.side_effect = Exception("Database connection error")

            response = client.get('/bi/revenue', headers=auth_headers)

            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert 'Database connection error' in data['error']

    def test_api_response_format(self, client, mock_db_session, auth_headers):
        """Test API response format consistency."""
        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_revenue_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = {
                'total_revenue': 50000,
                'growth_rate': 15.5
            }

            response = client.get('/bi/revenue', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)

            # Check response structure
            assert isinstance(data, dict)
            assert 'total_revenue' in data
            assert 'growth_rate' in data

            # Check data types
            assert isinstance(data['total_revenue'], (int, float))
            assert isinstance(data['growth_rate'], (int, float))

    def test_api_pagination(self, client, mock_db_session, auth_headers):
        """Test API pagination support."""
        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_reporting, 'get_available_reports') as mock_get_reports:

            mock_get_db.return_value = mock_db_session
            mock_get_reports.return_value = {
                'reports': [
                    {'id': f'report_{i}', 'title': f'Report {i}'}
                    for i in range(25)  # More than default page size
                ],
                'total_reports': 25,
                'page': 1,
                'per_page': 10,
                'total_pages': 3
            }

            response = client.get('/bi/reports?page=1&per_page=10', headers=auth_headers)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['reports']) == 10
            assert data['page'] == 1
            assert data['total_pages'] == 3

    def test_api_filtering(self, client, mock_db_session, auth_headers):
        """Test API filtering capabilities."""
        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_customer_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = {
                'total_customers': 1000,
                'customer_segments': {'premium': 200, 'standard': 800}
            }

            # Test with segment filter
            response = client.get('/bi/customers?segment=premium', headers=auth_headers)

            assert response.status_code == 200
            mock_get_metrics.assert_called_once()

    def test_api_caching(self, client, mock_db_session, auth_headers):
        """Test API response caching."""
        with patch('routes.business_intelligence.get_db_session') as mock_get_db, \
             patch.object(bi_service, 'get_revenue_metrics') as mock_get_metrics:

            mock_get_db.return_value = mock_db_session
            mock_get_metrics.return_value = {'total_revenue': 50000, 'cached': True}

            # First request
            response1 = client.get('/bi/revenue', headers=auth_headers)
            assert response1.status_code == 200

            # Second request (should use cache if implemented)
            response2 = client.get('/bi/revenue', headers=auth_headers)
            assert response2.status_code == 200

            # Verify same response
            data1 = json.loads(response1.data)
            data2 = json.loads(response2.data)
            assert data1 == data2