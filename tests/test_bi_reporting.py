"""
Tests for Business Intelligence Reporting System
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import os
import tempfile

from utils.bi_reporting import bi_reporting, BusinessIntelligenceReporting
from models.business_intelligence import RevenueStream, CustomerJourney, BusinessKPI
from models.organization import Organization


class TestBusinessIntelligenceReporting:
    """Test cases for BI Reporting System functionality."""

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

    def test_generate_revenue_report(self, mock_db_session, sample_organization):
        """Test revenue report generation."""
        # Mock revenue data
        revenue_streams = []
        for i in range(30):  # 30 days of revenue data
            revenue = Mock(spec=RevenueStream)
            revenue.amount = 1000 + i * 50
            revenue.created_at = datetime.utcnow() - timedelta(days=i)
            revenue.metadata = {
                'source': 'subscription' if i % 2 == 0 else 'one_time',
                'customer_id': f'customer_{i}'
            }
            revenue_streams.append(revenue)

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        result = bi_reporting.generate_revenue_report(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'report_type' in result
        assert 'title' in result
        assert 'data' in result
        assert 'summary' in result
        assert 'charts' in result
        assert 'generated_at' in result

        assert result['report_type'] == 'revenue'
        assert result['title'] == 'Revenue Report'

        # Check data structure
        data = result['data']
        assert 'total_revenue' in data
        assert 'revenue_by_source' in data
        assert 'daily_revenue' in data
        assert 'growth_rate' in data

        # Check summary
        summary = result['summary']
        assert 'period' in summary
        assert 'total_revenue' in summary
        assert 'average_daily_revenue' in summary

    def test_generate_customer_report(self, mock_db_session, sample_organization):
        """Test customer report generation."""
        # Mock customer journey data
        customer_journeys = []
        for i in range(100):  # 100 customers
            journey = Mock(spec=CustomerJourney)
            journey.customer_id = f"customer_{i}"
            journey.total_sessions = 5 + i % 15
            journey.total_actions = 20 + i % 80
            journey.total_time_spent = 300 + i % 3600
            journey.total_conversions = i % 5
            journey.avg_engagement_score = 2.0 + (i % 8)
            journey.lifecycle_stage = ['new', 'active', 'churned'][i % 3]
            journey.segment = ['premium', 'standard', 'basic'][i % 3]
            customer_journeys.append(journey)

        mock_db_session.query.return_value.filter.return_value.all.return_value = customer_journeys

        result = bi_reporting.generate_customer_report(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['report_type'] == 'customer'
        assert result['title'] == 'Customer Analytics Report'

        # Check data structure
        data = result['data']
        assert 'total_customers' in data
        assert 'customer_segments' in data
        assert 'lifecycle_stages' in data
        assert 'engagement_metrics' in data

        # Check summary
        summary = result['summary']
        assert 'period' in summary
        assert 'total_customers' in summary
        assert 'active_customers' in summary

    def test_generate_usage_report(self, mock_db_session, sample_organization):
        """Test usage report generation."""
        # Mock usage pattern data
        usage_patterns = []
        for i in range(50):  # 50 usage patterns
            pattern = Mock()
            pattern.pattern_type = ['daily', 'weekly', 'monthly'][i % 3]
            pattern.frequency = 10 + i % 20
            pattern.avg_duration = 30 + i % 60
            pattern.peak_hours = [9, 10, 11, 14, 15, 16][i % 6]
            pattern.error_rate = 0.02 + (i % 5) * 0.01
            pattern.success_rate = 0.95 - (i % 5) * 0.01
            usage_patterns.append(pattern)

        mock_db_session.query.return_value.filter.return_value.all.return_value = usage_patterns

        result = bi_reporting.generate_usage_report(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['report_type'] == 'usage'
        assert result['title'] == 'Usage Analytics Report'

        # Check data structure
        data = result['data']
        assert 'total_requests' in data
        assert 'usage_patterns' in data
        assert 'peak_hours' in data
        assert 'error_rates' in data

        # Check summary
        summary = result['summary']
        assert 'period' in summary
        assert 'total_requests' in summary
        assert 'average_daily_requests' in summary

    def test_generate_kpi_report(self, mock_db_session, sample_organization):
        """Test KPI report generation."""
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

        result = bi_reporting.generate_kpi_report(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['report_type'] == 'kpi'
        assert result['title'] == 'Key Performance Indicators Report'

        # Check data structure
        data = result['data']
        assert 'kpis' in data
        assert 'kpi_categories' in data
        assert 'overall_performance' in data

        # Check summary
        summary = result['summary']
        assert 'period' in summary
        assert 'total_kpis' in summary
        assert 'kpis_on_target' in summary

    def test_generate_comprehensive_report(self, mock_db_session, sample_organization):
        """Test comprehensive report generation."""
        # Mock all required data
        revenue_streams = [Mock(spec=RevenueStream)]
        revenue_streams[0].amount = 1000
        revenue_streams[0].created_at = datetime.utcnow()
        revenue_streams[0].metadata = {'source': 'subscription'}

        customer_journeys = [Mock(spec=CustomerJourney)]
        customer_journeys[0].customer_id = 'customer_1'
        customer_journeys[0].total_sessions = 10
        customer_journeys[0].lifecycle_stage = 'active'

        kpis = [Mock(spec=BusinessKPI)]
        kpis[0].name = 'Revenue Growth'
        kpis[0].value = 100
        kpis[0].target = 120

        mock_db_session.query.return_value.filter.return_value.all.side_effect = [
            revenue_streams, customer_journeys, kpis
        ]

        result = bi_reporting.generate_comprehensive_report(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert result['report_type'] == 'comprehensive'
        assert result['title'] == 'Comprehensive Business Intelligence Report'

        # Check that all sections are included
        assert 'revenue_section' in result
        assert 'customer_section' in result
        assert 'kpi_section' in result
        assert 'executive_summary' in result

    def test_report_export_formats(self, mock_db_session, sample_organization):
        """Test report export in different formats."""
        # Mock revenue data
        revenue_streams = [Mock(spec=RevenueStream)]
        revenue_streams[0].amount = 1000
        revenue_streams[0].created_at = datetime.utcnow()
        revenue_streams[0].metadata = {'source': 'subscription'}

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        # Generate report
        report = bi_reporting.generate_revenue_report(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        # Test JSON export
        json_export = bi_reporting.export_report(report, 'json')
        assert isinstance(json_export, str)

        # Parse and verify JSON structure
        json_data = json.loads(json_export)
        assert json_data['report_type'] == 'revenue'
        assert json_data['title'] == 'Revenue Report'

        # Test CSV export
        csv_export = bi_reporting.export_report(report, 'csv')
        assert isinstance(csv_export, str)
        assert 'Revenue Report' in csv_export

        # Test HTML export
        html_export = bi_reporting.export_report(report, 'html')
        assert isinstance(html_export, str)
        assert '<html>' in html_export
        assert 'Revenue Report' in html_export

        # Test PDF export (mocked)
        with patch('utils.bi_reporting.generate_pdf_report') as mock_pdf:
            mock_pdf.return_value = b'PDF content'
            pdf_export = bi_reporting.export_report(report, 'pdf')
            assert isinstance(pdf_export, bytes)
            mock_pdf.assert_called_once()

        # Test Excel export (mocked)
        with patch('utils.bi_reporting.generate_excel_report') as mock_excel:
            mock_excel.return_value = b'Excel content'
            excel_export = bi_reporting.export_report(report, 'excel')
            assert isinstance(excel_export, bytes)
            mock_excel.assert_called_once()

    def test_scheduled_reporting(self, mock_db_session, sample_organization):
        """Test scheduled reporting functionality."""
        # Mock data
        revenue_streams = [Mock(spec=RevenueStream)]
        revenue_streams[0].amount = 1000
        revenue_streams[0].created_at = datetime.utcnow()

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        # Test creating a scheduled report
        schedule_config = {
            'report_type': 'revenue',
            'frequency': 'weekly',
            'day_of_week': 1,  # Monday
            'time': '09:00',
            'recipients': ['admin@example.com'],
            'format': 'pdf'
        }

        result = bi_reporting.create_scheduled_report(
            sample_organization.id,
            schedule_config,
            mock_db_session
        )

        assert 'schedule_id' in result
        assert 'status' in result
        assert 'next_run' in result
        assert result['status'] == 'active'

        # Test getting scheduled reports
        schedules = bi_reporting.get_scheduled_reports(sample_organization.id)

        assert isinstance(schedules, list)
        assert len(schedules) > 0

        # Test updating scheduled report
        updated_config = schedule_config.copy()
        updated_config['frequency'] = 'monthly'

        update_result = bi_reporting.update_scheduled_report(
            result['schedule_id'],
            updated_config
        )

        assert update_result['status'] == 'updated'

        # Test deleting scheduled report
        delete_result = bi_reporting.delete_scheduled_report(result['schedule_id'])

        assert delete_result['status'] == 'deleted'

    def test_custom_report_builder(self, mock_db_session, sample_organization):
        """Test custom report builder functionality."""
        # Mock data
        revenue_streams = [Mock(spec=RevenueStream)]
        revenue_streams[0].amount = 1000
        revenue_streams[0].created_at = datetime.utcnow()

        customer_journeys = [Mock(spec=CustomerJourney)]
        customer_journeys[0].customer_id = 'customer_1'
        customer_journeys[0].total_sessions = 10

        mock_db_session.query.return_value.filter.return_value.all.side_effect = [
            revenue_streams, customer_journeys
        ]

        # Test building custom report
        report_config = {
            'title': 'Custom Revenue and Customer Report',
            'sections': [
                {
                    'type': 'revenue',
                    'metrics': ['total_revenue', 'daily_revenue'],
                    'chart_type': 'line'
                },
                {
                    'type': 'customer',
                    'metrics': ['total_customers', 'customer_segments'],
                    'chart_type': 'pie'
                }
            ],
            'date_range': {
                'start_date': datetime.utcnow() - timedelta(days=30),
                'end_date': datetime.utcnow()
            },
            'filters': {
                'revenue_source': 'subscription'
            }
        }

        result = bi_reporting.build_custom_report(
            sample_organization.id,
            report_config,
            mock_db_session
        )

        assert 'report_id' in result
        assert 'title' in result
        assert 'sections' in result
        assert result['title'] == 'Custom Revenue and Customer Report'
        assert len(result['sections']) == 2

    def test_report_templates(self):
        """Test report template functionality."""
        # Test getting available templates
        templates = bi_reporting.get_report_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0
        assert all('id' in template and 'name' in template for template in templates)

        # Test getting specific template
        template = bi_reporting.get_report_template('revenue_monthly')

        assert 'id' in template
        assert 'name' in template
        assert 'sections' in template
        assert 'default_config' in template

        # Test creating custom template
        custom_template = {
            'name': 'Custom Template',
            'description': 'A custom report template',
            'sections': [
                {
                    'type': 'revenue',
                    'title': 'Revenue Overview'
                }
            ]
        }

        result = bi_reporting.create_report_template(custom_template)

        assert 'template_id' in result
        assert 'status' in result
        assert result['status'] == 'created'

    def test_email_delivery(self, mock_db_session, sample_organization):
        """Test email delivery functionality."""
        # Mock report data
        report = {
            'report_type': 'revenue',
            'title': 'Test Revenue Report',
            'data': {'total_revenue': 1000},
            'summary': {'period': '30 days'}
        }

        # Test sending report via email
        with patch('utils.bi_reporting.send_email') as mock_email:
            mock_email.return_value = True

            result = bi_reporting.send_report_via_email(
                report,
                ['admin@example.com'],
                'Test Subject',
                'Test message body'
            )

            assert result['status'] == 'sent'
            mock_email.assert_called_once()

            # Check email parameters
            call_args = mock_email.call_args
            assert 'admin@example.com' in call_args[1]['to']
            assert 'Test Subject' in call_args[1]['subject']

    def test_report_storage(self, mock_db_session, sample_organization):
        """Test report storage and retrieval."""
        # Mock report data
        report = {
            'report_type': 'revenue',
            'title': 'Test Revenue Report',
            'data': {'total_revenue': 1000},
            'summary': {'period': '30 days'}
        }

        # Test storing report
        result = bi_reporting.store_report(
            sample_organization.id,
            report,
            'test_report.pdf'
        )

        assert 'report_id' in result
        assert 'storage_path' in result
        assert 'status' in result
        assert result['status'] == 'stored'

        # Test retrieving report
        retrieved_report = bi_reporting.get_stored_report(result['report_id'])

        assert retrieved_report['report_type'] == 'revenue'
        assert retrieved_report['title'] == 'Test Revenue Report'

        # Test listing stored reports
        reports = bi_reporting.list_stored_reports(sample_organization.id)

        assert isinstance(reports, list)
        assert len(reports) > 0

        # Test deleting stored report
        delete_result = bi_reporting.delete_stored_report(result['report_id'])

        assert delete_result['status'] == 'deleted'

    def test_data_aggregation_methods(self):
        """Test data aggregation helper methods."""
        # Test _aggregate_revenue_by_source
        revenue_streams = [
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream),
            Mock(spec=RevenueStream)
        ]
        revenue_streams[0].amount = 1000
        revenue_streams[0].metadata = {'source': 'subscription'}
        revenue_streams[1].amount = 500
        revenue_streams[1].metadata = {'source': 'one_time'}
        revenue_streams[2].amount = 1500
        revenue_streams[2].metadata = {'source': 'subscription'}

        aggregated = bi_reporting._aggregate_revenue_by_source(revenue_streams)

        assert len(aggregated) == 2
        assert aggregated['subscription'] == 2500
        assert aggregated['one_time'] == 500

        # Test _aggregate_customers_by_segment
        customer_journeys = [
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney),
            Mock(spec=CustomerJourney)
        ]
        customer_journeys[0].segment = 'premium'
        customer_journeys[1].segment = 'standard'
        customer_journeys[2].segment = 'premium'

        aggregated = bi_reporting._aggregate_customers_by_segment(customer_journeys)

        assert len(aggregated) == 2
        assert aggregated['premium'] == 2
        assert aggregated['standard'] == 1

    def test_error_handling(self, mock_db_session, sample_organization):
        """Test error handling in reporting system."""
        # Test with empty data
        mock_db_session.query.return_value.filter.return_value.all.return_value = []

        result = bi_reporting.generate_revenue_report(
            sample_organization.id,
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),
            mock_db_session
        )

        assert 'error' in result
        assert 'No revenue data found' in result['error']

        # Test with invalid date range
        result = bi_reporting.generate_revenue_report(
            sample_organization.id,
            datetime.utcnow(),  # End date before start date
            datetime.utcnow() - timedelta(days=30),
            mock_db_session
        )

        assert 'error' in result
        assert 'Invalid date range' in result['error']

        # Test invalid export format
        report = {'report_type': 'revenue', 'title': 'Test Report'}
        with pytest.raises(ValueError):
            bi_reporting.export_report(report, 'invalid_format')

    def test_report_initialization(self):
        """Test reporting system initialization."""
        assert isinstance(bi_reporting, BusinessIntelligenceReporting)
        assert hasattr(bi_reporting, 'report_templates')
        assert hasattr(bi_reporting, 'scheduled_reports')
        assert hasattr(bi_reporting, 'report_storage')

    def test_report_validation(self):
        """Test report validation functionality."""
        # Test valid report config
        valid_config = {
            'title': 'Test Report',
            'report_type': 'revenue',
            'date_range': {
                'start_date': datetime.utcnow() - timedelta(days=30),
                'end_date': datetime.utcnow()
            }
        }

        is_valid, errors = bi_reporting.validate_report_config(valid_config)
        assert is_valid == True
        assert len(errors) == 0

        # Test invalid report config
        invalid_config = {
            'title': '',  # Empty title
            'report_type': 'invalid_type',
            'date_range': {
                'start_date': datetime.utcnow(),
                'end_date': datetime.utcnow() - timedelta(days=30)  # Invalid range
            }
        }

        is_valid, errors = bi_reporting.validate_report_config(invalid_config)
        assert is_valid == False
        assert len(errors) > 0

    def test_performance_monitoring(self, mock_db_session, sample_organization):
        """Test performance monitoring for report generation."""
        # Mock large dataset
        revenue_streams = []
        for i in range(1000):  # Large dataset
            revenue = Mock(spec=RevenueStream)
            revenue.amount = 1000 + i
            revenue.created_at = datetime.utcnow() - timedelta(days=i % 30)
            revenue.metadata = {'source': 'subscription'}
            revenue_streams.append(revenue)

        mock_db_session.query.return_value.filter.return_value.all.return_value = revenue_streams

        # Test report generation with performance monitoring
        with patch('utils.bi_reporting.time') as mock_time:
            mock_time.time.return_value = 1000  # Start time
            mock_time.time.side_effect = [1000, 1005]  # Start and end time

            result = bi_reporting.generate_revenue_report(
                sample_organization.id,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow(),
                mock_db_session
            )

            assert 'performance_metrics' in result
            assert 'generation_time' in result['performance_metrics']
            assert result['performance_metrics']['generation_time'] == 5.0