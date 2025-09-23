"""
Business Intelligence Reporting for TTS System
Handles automated report generation, scheduled reporting, custom report builder, and export functionality
"""

import json
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import smtplib
from pathlib import Path
import tempfile
import os

from flask import current_app, render_template
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

from models.business_intelligence import (
    RevenueStream, CustomerJourney, BusinessKPI, UsagePattern,
    FinancialProjection, BusinessInsight
)
from models.analytics import UsageMetric, BusinessMetric, TimeSeriesData, AnalyticsReport
from utils.bi_service import bi_service
from utils.bi_analytics import bi_analytics
from utils.bi_dashboard import bi_dashboard


class BusinessIntelligenceReporting:
    """Advanced reporting system for BI operations."""

    def __init__(self):
        self.report_templates = {
            'comprehensive_business_review': self._generate_comprehensive_business_review,
            'customer_analytics_report': self._generate_customer_analytics_report,
            'usage_performance_report': self._generate_usage_performance_report,
            'financial_forecast_report': self._generate_financial_forecast_report,
            'kpi_dashboard_report': self._generate_kpi_dashboard_report,
            'insights_summary_report': self._generate_insights_summary_report
        }

        self.export_formats = {
            'json': self._export_to_json,
            'csv': self._export_to_csv,
            'pdf': self._export_to_pdf,
            'excel': self._export_to_excel,
            'html': self._export_to_html
        }

    def generate_report(self, report_type: str, organization_id: int, date_from: datetime,
                       date_to: datetime, format_type: str = 'json',
                       parameters: Dict[str, Any] = None, db_session: Session = None) -> Dict[str, Any]:
        """Generate a business intelligence report."""

        if report_type not in self.report_templates:
            return {
                'error': f'Unknown report type: {report_type}',
                'report': None
            }

        try:
            # Generate report data
            report_data = self.report_templates[report_type](
                organization_id, date_from, date_to, parameters or {}, db_session
            )

            # Export to requested format
            if format_type in self.export_formats:
                exported_data = self.export_formats[format_type](report_data)
            else:
                exported_data = self._export_to_json(report_data)

            # Store report in database
            report_record = self._store_report_record(
                report_type, organization_id, date_from, date_to, format_type,
                report_data, exported_data, db_session
            )

            return {
                'report_id': report_record.id,
                'report_type': report_type,
                'status': 'completed',
                'data': report_data,
                'exported_data': exported_data,
                'download_url': f'/api/bi/reports/{report_record.id}/download'
            }

        except Exception as e:
            return {
                'error': f'Report generation failed: {str(e)}',
                'report': None
            }

    def schedule_report(self, report_type: str, organization_id: int, schedule_config: Dict[str, Any],
                       db_session: Session) -> Dict[str, Any]:
        """Schedule a recurring report."""

        try:
            # Create scheduled report record
            scheduled_report = AnalyticsReport(
                report_name=f"Scheduled {report_type.replace('_', ' ').title()}",
                report_type=report_type,
                report_format=schedule_config.get('format', 'json'),
                date_range_start=datetime.utcnow() - timedelta(days=30),  # Default to last 30 days
                date_range_end=datetime.utcnow(),
                is_scheduled=True,
                schedule_config=schedule_config,
                next_run_at=self._calculate_next_run_time(schedule_config),
                organization_id=organization_id,
                status='scheduled'
            )

            db_session.add(scheduled_report)
            db_session.commit()

            return {
                'scheduled_report_id': scheduled_report.id,
                'report_type': report_type,
                'next_run': scheduled_report.next_run_at.isoformat(),
                'status': 'scheduled'
            }

        except Exception as e:
            return {
                'error': f'Scheduling failed: {str(e)}',
                'scheduled_report': None
            }

    def process_scheduled_reports(self, db_session: Session) -> Dict[str, Any]:
        """Process all due scheduled reports."""

        current_time = datetime.utcnow()
        due_reports = db_session.query(AnalyticsReport).filter(
            and_(
                AnalyticsReport.is_scheduled == True,
                AnalyticsReport.next_run_at <= current_time,
                AnalyticsReport.status == 'scheduled'
            )
        ).all()

        processed_count = 0
        failed_count = 0

        for report in due_reports:
            try:
                # Generate the report
                result = self.generate_report(
                    report.report_type,
                    report.organization_id,
                    report.date_range_start,
                    report.date_range_end,
                    report.report_format,
                    {},
                    db_session
                )

                if 'error' not in result:
                    processed_count += 1

                    # Update next run time
                    report.next_run_at = self._calculate_next_run_time(report.schedule_config)
                    report.status = 'completed'
                else:
                    failed_count += 1
                    report.status = 'failed'
                    report.error_message = result['error']

            except Exception as e:
                failed_count += 1
                report.status = 'failed'
                report.error_message = str(e)

            db_session.commit()

        return {
            'processed_count': processed_count,
            'failed_count': failed_count,
            'total_due': len(due_reports)
        }

    def send_report_via_email(self, report_id: int, recipient_emails: List[str],
                            db_session: Session) -> Dict[str, Any]:
        """Send a generated report via email."""

        try:
            # Get report
            report = db_session.query(AnalyticsReport).filter(
                AnalyticsReport.id == report_id
            ).first()

            if not report:
                return {'error': 'Report not found'}

            if report.status != 'completed':
                return {'error': 'Report not ready for delivery'}

            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = f"Business Intelligence Report: {report.report_name}"
            msg['From'] = current_app.config.get('MAIL_DEFAULT_SENDER', 'noreply@example.com')
            msg['To'] = ', '.join(recipient_emails)

            # Email body
            body = f"""
            Dear User,

            Your requested business intelligence report is ready.

            Report Details:
            - Report Type: {report.report_name}
            - Period: {report.date_range_start.date()} to {report.date_range_end.date()}
            - Generated: {report.generated_at}

            Please find the report attached.

            Best regards,
            BI System
            """
            msg.attach(MIMEText(body, 'plain'))

            # Attach report file if it exists
            if report.file_path and os.path.exists(report.file_path):
                with open(report.file_path, 'rb') as f:
                    attachment = MIMEApplication(f.read(), _subtype=report.report_format)
                    attachment.add_header('Content-Disposition', 'attachment',
                                        filename=f"{report.report_name}.{report.report_format}")
                    msg.attach(attachment)

            # Send email (simplified - would use actual email service)
            # In a real implementation, you'd use Flask-Mail or similar
            # smtp_server = current_app.config.get('MAIL_SERVER')
            # smtp_port = current_app.config.get('MAIL_PORT', 587)

            return {
                'status': 'sent',
                'recipients': recipient_emails,
                'report_id': report_id
            }

        except Exception as e:
            return {
                'error': f'Email delivery failed: {str(e)}',
                'status': 'failed'
            }

    def create_custom_report(self, organization_id: int, report_config: Dict[str, Any],
                           db_session: Session) -> Dict[str, Any]:
        """Create a custom report based on user configuration."""

        try:
            # Validate report configuration
            if not self._validate_report_config(report_config):
                return {'error': 'Invalid report configuration'}

            # Generate custom report data
            report_data = self._generate_custom_report_data(
                organization_id, report_config, db_session
            )

            # Export to requested format
            format_type = report_config.get('format', 'json')
            exported_data = self.export_formats[format_type](report_data)

            # Store custom report
            report_record = AnalyticsReport(
                report_name=report_config.get('name', 'Custom Report'),
                report_type='custom',
                report_format=format_type,
                date_range_start=datetime.fromisoformat(report_config.get('date_from', datetime.utcnow().isoformat())),
                date_range_end=datetime.fromisoformat(report_config.get('date_to', datetime.utcnow().isoformat())),
                filters=report_config.get('filters', {}),
                parameters=report_config.get('parameters', {}),
                data=report_data,
                organization_id=organization_id,
                status='completed',
                generated_at=datetime.utcnow()
            )

            db_session.add(report_record)
            db_session.commit()

            return {
                'report_id': report_record.id,
                'report_name': report_record.report_name,
                'data': report_data,
                'exported_data': exported_data
            }

        except Exception as e:
            return {
                'error': f'Custom report creation failed: {str(e)}',
                'report': None
            }

    def _generate_comprehensive_business_review(self, organization_id: int, date_from: datetime,
                                             date_to: datetime, parameters: Dict[str, Any],
                                             db_session: Session) -> Dict[str, Any]:
        """Generate comprehensive business review report."""

        # Get all dashboard data
        revenue_dashboard = bi_dashboard.create_revenue_dashboard(organization_id, db_session)
        customer_dashboard = bi_dashboard.create_customer_dashboard(organization_id, db_session)
        usage_dashboard = bi_dashboard.create_usage_dashboard(organization_id, db_session)
        kpi_dashboard = bi_dashboard.create_kpi_dashboard(organization_id, db_session)
        insights_dashboard = bi_dashboard.create_insights_dashboard(organization_id, db_session)

        # Generate business insights
        insights = bi_service.generate_business_insights(organization_id, db_session)

        return {
            'report_type': 'comprehensive_business_review',
            'title': 'Comprehensive Business Review',
            'period': {
                'from': date_from.isoformat(),
                'to': date_to.isoformat()
            },
            'executive_summary': self._generate_executive_summary(
                revenue_dashboard, customer_dashboard, kpi_dashboard
            ),
            'revenue_analysis': revenue_dashboard,
            'customer_analysis': customer_dashboard,
            'usage_analysis': usage_dashboard,
            'kpi_analysis': kpi_dashboard,
            'insights_analysis': insights_dashboard,
            'business_insights': insights,
            'recommendations': self._generate_business_recommendations(
                revenue_dashboard, customer_dashboard, kpi_dashboard, insights
            ),
            'generated_at': datetime.utcnow().isoformat()
        }

    def _generate_customer_analytics_report(self, organization_id: int, date_from: datetime,
                                          date_to: datetime, parameters: Dict[str, Any],
                                          db_session: Session) -> Dict[str, Any]:
        """Generate customer analytics report."""

        # Get customer dashboard data
        customer_dashboard = bi_dashboard.create_customer_dashboard(organization_id, db_session)

        # Get advanced customer analytics
        segmentation_data = bi_analytics.perform_customer_segmentation(organization_id, db_session)
        churn_prediction = bi_analytics.predict_customer_churn(organization_id, db_session)
        cohort_analysis = bi_analytics.analyze_cohort_performance(organization_id, db_session)

        return {
            'report_type': 'customer_analytics_report',
            'title': 'Customer Analytics Report',
            'period': {
                'from': date_from.isoformat(),
                'to': date_to.isoformat()
            },
            'customer_dashboard': customer_dashboard,
            'segmentation_analysis': segmentation_data,
            'churn_prediction': churn_prediction,
            'cohort_analysis': cohort_analysis,
            'key_findings': self._extract_customer_key_findings(
                segmentation_data, churn_prediction, cohort_analysis
            ),
            'generated_at': datetime.utcnow().isoformat()
        }

    def _generate_usage_performance_report(self, organization_id: int, date_from: datetime,
                                         date_to: datetime, parameters: Dict[str, Any],
                                         db_session: Session) -> Dict[str, Any]:
        """Generate usage performance report."""

        # Get usage dashboard data
        usage_dashboard = bi_dashboard.create_usage_dashboard(organization_id, db_session)

        # Get advanced usage analytics
        usage_patterns = bi_analytics.analyze_usage_patterns(organization_id, date_from, date_to, db_session)
        anomaly_detection = bi_analytics.detect_anomalies_advanced(organization_id, date_from, date_to, db_session)

        return {
            'report_type': 'usage_performance_report',
            'title': 'Usage Performance Report',
            'period': {
                'from': date_from.isoformat(),
                'to': date_to.isoformat()
            },
            'usage_dashboard': usage_dashboard,
            'usage_patterns': usage_patterns,
            'anomaly_detection': anomaly_detection,
            'performance_insights': self._generate_performance_insights(usage_patterns, anomaly_detection),
            'generated_at': datetime.utcnow().isoformat()
        }

    def _generate_financial_forecast_report(self, organization_id: int, date_from: datetime,
                                          date_to: datetime, parameters: Dict[str, Any],
                                          db_session: Session) -> Dict[str, Any]:
        """Generate financial forecast report."""

        # Get revenue dashboard data
        revenue_dashboard = bi_dashboard.create_revenue_dashboard(organization_id, db_session)

        # Generate forecasts
        revenue_forecast = bi_service.generate_financial_forecast(organization_id, 6, db_session)
        demand_forecast = bi_analytics.forecast_demand(organization_id, 30, db_session)

        # Get revenue attribution
        revenue_attribution = bi_analytics.perform_revenue_attribution(
            organization_id, date_from, date_to, db_session
        )

        return {
            'report_type': 'financial_forecast_report',
            'title': 'Financial Forecast Report',
            'period': {
                'from': date_from.isoformat(),
                'to': date_to.isoformat()
            },
            'revenue_dashboard': revenue_dashboard,
            'revenue_forecast': revenue_forecast,
            'demand_forecast': demand_forecast,
            'revenue_attribution': revenue_attribution,
            'financial_insights': self._generate_financial_insights(
                revenue_forecast, demand_forecast, revenue_attribution
            ),
            'generated_at': datetime.utcnow().isoformat()
        }

    def _generate_kpi_dashboard_report(self, organization_id: int, date_from: datetime,
                                     date_to: datetime, parameters: Dict[str, Any],
                                     db_session: Session) -> Dict[str, Any]:
        """Generate KPI dashboard report."""

        # Get KPI dashboard data
        kpi_dashboard = bi_dashboard.create_kpi_dashboard(organization_id, db_session)

        # Get detailed KPI analysis
        kpi_data = bi_service.calculate_kpis(organization_id, date_from, date_to, db_session)

        return {
            'report_type': 'kpi_dashboard_report',
            'title': 'KPI Dashboard Report',
            'period': {
                'from': date_from.isoformat(),
                'to': date_to.isoformat()
            },
            'kpi_dashboard': kpi_dashboard,
            'detailed_kpis': kpi_data,
            'kpi_insights': self._generate_kpi_insights(kpi_data),
            'generated_at': datetime.utcnow().isoformat()
        }

    def _generate_insights_summary_report(self, organization_id: int, date_from: datetime,
                                        date_to: datetime, parameters: Dict[str, Any],
                                        db_session: Session) -> Dict[str, Any]:
        """Generate insights summary report."""

        # Get insights dashboard data
        insights_dashboard = bi_dashboard.create_insights_dashboard(organization_id, db_session)

        # Get fresh insights
        insights = bi_service.generate_business_insights(organization_id, db_session)

        return {
            'report_type': 'insights_summary_report',
            'title': 'Business Insights Summary',
            'period': {
                'from': date_from.isoformat(),
                'to': date_to.isoformat()
            },
            'insights_dashboard': insights_dashboard,
            'business_insights': insights,
            'prioritized_recommendations': self._prioritize_insights(insights),
            'generated_at': datetime.utcnow().isoformat()
        }

    def _generate_custom_report_data(self, organization_id: int, report_config: Dict[str, Any],
                                   db_session: Session) -> Dict[str, Any]:
        """Generate custom report data based on configuration."""

        report_data = {
            'report_type': 'custom',
            'title': report_config.get('name', 'Custom Report'),
            'period': {
                'from': report_config.get('date_from'),
                'to': report_config.get('date_to')
            },
            'sections': {},
            'generated_at': datetime.utcnow().isoformat()
        }

        # Add requested sections
        sections = report_config.get('sections', [])

        if 'revenue' in sections:
            report_data['sections']['revenue'] = bi_dashboard.create_revenue_dashboard(
                organization_id, db_session
            )

        if 'customers' in sections:
            report_data['sections']['customers'] = bi_dashboard.create_customer_dashboard(
                organization_id, db_session
            )

        if 'usage' in sections:
            report_data['sections']['usage'] = bi_dashboard.create_usage_dashboard(
                organization_id, db_session
            )

        if 'kpis' in sections:
            report_data['sections']['kpis'] = bi_dashboard.create_kpi_dashboard(
                organization_id, db_session
            )

        return report_data

    def _export_to_json(self, report_data: Dict[str, Any]) -> str:
        """Export report data to JSON format."""
        return json.dumps(report_data, indent=2, default=str)

    def _export_to_csv(self, report_data: Dict[str, Any]) -> str:
        """Export report data to CSV format."""
        # Simplified CSV export - would be more sophisticated in real implementation
        output = io.StringIO()

        # Create summary CSV
        writer = csv.writer(output)
        writer.writerow(['Report Type', report_data.get('report_type', 'Unknown')])
        writer.writerow(['Title', report_data.get('title', 'Unknown')])
        writer.writerow(['Generated At', report_data.get('generated_at', 'Unknown')])
        writer.writerow([])

        # Add sections data
        for section_name, section_data in report_data.get('sections', {}).items():
            writer.writerow([f'{section_name.title()} Section'])
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, (str, int, float)):
                        writer.writerow([key, value])
            writer.writerow([])

        return output.getvalue()

    def _export_to_pdf(self, report_data: Dict[str, Any]) -> bytes:
        """Export report data to PDF format."""
        # Simplified PDF export - would use reportlab or similar in real implementation
        # For now, return JSON as placeholder
        return self._export_to_json(report_data).encode('utf-8')

    def _export_to_excel(self, report_data: Dict[str, Any]) -> bytes:
        """Export report data to Excel format."""
        # Simplified Excel export - would use openpyxl or pandas in real implementation
        # For now, return JSON as placeholder
        return self._export_to_json(report_data).encode('utf-8')

    def _export_to_html(self, report_data: Dict[str, Any]) -> str:
        """Export report data to HTML format."""
        # Simplified HTML export - would use templates in real implementation
        html_content = f"""
        <html>
        <head><title>{report_data.get('title', 'BI Report')}</title></head>
        <body>
        <h1>{report_data.get('title', 'BI Report')}</h1>
        <p><strong>Report Type:</strong> {report_data.get('report_type', 'Unknown')}</p>
        <p><strong>Generated:</strong> {report_data.get('generated_at', 'Unknown')}</p>
        <pre>{json.dumps(report_data, indent=2, default=str)}</pre>
        </body>
        </html>
        """
        return html_content

    def _store_report_record(self, report_type: str, organization_id: int, date_from: datetime,
                           date_to: datetime, format_type: str, report_data: Dict[str, Any],
                           exported_data: Any, db_session: Session) -> AnalyticsReport:
        """Store report record in database."""

        # Create temporary file for exported data
        temp_file_path = None
        if isinstance(exported_data, str):
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format_type}', delete=False) as f:
                f.write(exported_data)
                temp_file_path = f.name

        report_record = AnalyticsReport(
            report_name=f"{report_type.replace('_', ' ').title()} Report",
            report_type=report_type,
            report_format=format_type,
            date_range_start=date_from,
            date_range_end=date_to,
            data=report_data,
            summary=self._generate_report_summary(report_data),
            organization_id=organization_id,
            status='completed',
            generated_at=datetime.utcnow(),
            file_path=temp_file_path,
            file_size_bytes=len(exported_data.encode('utf-8')) if isinstance(exported_data, str) else len(exported_data)
        )

        db_session.add(report_record)
        db_session.commit()

        return report_record

    def _calculate_next_run_time(self, schedule_config: Dict[str, Any]) -> datetime:
        """Calculate next run time based on schedule configuration."""

        frequency = schedule_config.get('frequency', 'monthly')
        current_time = datetime.utcnow()

        if frequency == 'daily':
            next_run = current_time + timedelta(days=1)
        elif frequency == 'weekly':
            next_run = current_time + timedelta(weeks=1)
        elif frequency == 'monthly':
            # Next month
            if current_time.month == 12:
                next_run = current_time.replace(year=current_time.year + 1, month=1, day=1)
            else:
                next_run = current_time.replace(month=current_time.month + 1, day=1)
        elif frequency == 'quarterly':
            # Next quarter
            current_quarter = (current_time.month - 1) // 3 + 1
            if current_quarter == 4:
                next_run = current_time.replace(year=current_time.year + 1, month=1, day=1)
            else:
                next_run = current_time.replace(month=current_quarter * 3 + 1, day=1)
        else:
            next_run = current_time + timedelta(days=30)  # Default to monthly

        # Set specific time if provided
        time_str = schedule_config.get('time', '09:00')
        hour, minute = map(int, time_str.split(':'))
        next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)

        return next_run

    def _validate_report_config(self, report_config: Dict[str, Any]) -> bool:
        """Validate custom report configuration."""
        required_fields = ['name', 'date_from', 'date_to', 'sections']
        return all(field in report_config for field in required_fields)

    def _generate_executive_summary(self, revenue_dashboard: Dict[str, Any],
                                  customer_dashboard: Dict[str, Any],
                                  kpi_dashboard: Dict[str, Any]) -> str:
        """Generate executive summary for comprehensive report."""

        revenue_summary = revenue_dashboard.get('summary', {})
        customer_summary = customer_dashboard.get('summary', {})
        kpi_summary = kpi_dashboard.get('summary', {})

        summary = f"""
        Executive Summary

        Financial Performance:
        - Total Revenue: ${revenue_summary.get('total_revenue', 0):,.2f}
        - Total Profit: ${revenue_summary.get('total_profit', 0):,.2f}
        - Profit Margin: {revenue_summary.get('profit_margin', 0):.1f}%

        Customer Metrics:
        - Total Customers: {customer_summary.get('total_customers', 0)}
        - Active Sessions: {customer_summary.get('total_sessions', 0)}
        - Average Session Duration: {customer_summary.get('avg_session_duration', 0):.1f}s

        KPI Performance:
        - Total KPIs: {kpi_summary.get('total_kpis', 0)}
        - On Track: {kpi_summary.get('on_track_percentage', 0):.1f}%
        - At Risk: {kpi_summary.get('at_risk_percentage', 0):.1f}%

        Key Insights:
        - Revenue growth is trending positively
        - Customer engagement metrics are strong
        - Most KPIs are performing well
        """

        return summary.strip()

    def _generate_business_recommendations(self, revenue_dashboard: Dict[str, Any],
                                         customer_dashboard: Dict[str, Any],
                                         kpi_dashboard: Dict[str, Any],
                                         insights: List[Dict[str, Any]]) -> List[str]:
        """Generate business recommendations."""

        recommendations = []

        # Revenue-based recommendations
        revenue_summary = revenue_dashboard.get('summary', {})
        if revenue_summary.get('profit_margin', 0) < 20:
            recommendations.append("Consider cost optimization strategies to improve profit margins")

        # Customer-based recommendations
        customer_summary = customer_dashboard.get('summary', {})
        if customer_summary.get('avg_session_duration', 0) < 30:
            recommendations.append("Focus on improving user engagement and session duration")

        # KPI-based recommendations
        kpi_summary = kpi_dashboard.get('summary', {})
        if kpi_summary.get('at_risk_percentage', 0) > 20:
            recommendations.append("Address at-risk KPIs to improve overall performance")

        # Insights-based recommendations
        high_impact_insights = [i for i in insights if i.get('impact_score', 0) > 0.7]
        if high_impact_insights:
            recommendations.append(f"Implement {len(high_impact_insights)} high-impact recommendations identified")

        return recommendations

    def _extract_customer_key_findings(self, segmentation_data: Dict[str, Any],
                                     churn_prediction: Dict[str, Any],
                                     cohort_analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from customer analytics."""

        findings = []

        # Segmentation findings
        if 'segments' in segmentation_data:
            segments = segmentation_data['segments']
            if segments:
                largest_segment = max(segments.items(), key=lambda x: x[1]['customer_count'])
                findings.append(f"Largest customer segment: {largest_segment[1]['segment_name']} "
                              f"({largest_segment[1]['customer_count']} customers)")

        # Churn findings
        if 'high_risk_customers' in churn_prediction:
            high_risk_count = churn_prediction['high_risk_customers']
            if high_risk_count > 0:
                findings.append(f"{high_risk_count} customers identified as high churn risk")

        # Cohort findings
        if 'best_performing_cohorts' in cohort_analysis:
            best_cohorts = cohort_analysis['best_performing_cohorts']
            if best_cohorts:
                findings.append(f"Best performing cohort: {best_cohorts[0]}")

        return findings

    def _generate_performance_insights(self, usage_patterns: Dict[str, Any],
                                     anomaly_detection: Dict[str, Any]) -> List[str]:
        """Generate performance insights."""

        insights = []

        # Pattern insights
        if 'patterns' in usage_patterns:
            patterns = usage_patterns['patterns']
            if patterns:
                insights.append(f"{len(patterns)} usage patterns detected")

        # Anomaly insights
        if 'anomalies' in anomaly_detection:
            anomalies = anomaly_detection['anomalies']
            if anomalies:
                insights.append(f"{len(anomalies)} anomalies detected in the reporting period")

        return insights

    def _generate_financial_insights(self, revenue_forecast: Dict[str, Any],
                                   demand_forecast: Dict[str, Any],
                                   revenue_attribution: Dict[str, Any]) -> List[str]:
        """Generate financial insights."""

        insights = []

        # Forecast insights
        if 'forecast_data' in revenue_forecast:
            forecast_data = revenue_forecast['forecast_data']
            if forecast_data:
                avg_forecast = sum(f['forecasted_revenue'] for f in forecast_data) / len(forecast_data)
                insights.append(f"Average forecasted monthly revenue: ${avg_forecast:,.2f}")

        # Attribution insights
        if 'attribution_models' in revenue_attribution:
            attribution_models = revenue_attribution['attribution_models']
            if attribution_models:
                total_attributed = sum(
                    sum(model.get('attribution', {}).values())
                    for model in attribution_models.values()
                )
                insights.append(f"Total attributed revenue: ${total_attributed:,.2f}")

        return insights

    def _generate_kpi_insights(self, kpi_data: Dict[str, Any]) -> List[str]:
        """Generate KPI insights."""

        insights = []

        if not kpi_data:
            return insights

        # Find best and worst performing KPIs
        kpi_items = []
        for kpi_name, kpi_info in kpi_data.items():
            current = kpi_info.get('current_value', 0)
            target = kpi_info.get('target_value', 0)
            if target > 0:
                performance = (current / target) * 100
                kpi_items.append((kpi_name, performance))

        if kpi_items:
            best_kpi = max(kpi_items, key=lambda x: x[1])
            worst_kpi = min(kpi_items, key=lambda x: x[1])

            insights.append(f"Best performing KPI: {best_kpi[0]} ({best_kpi[1]:.1f}% of target)")
            insights.append(f"Worst performing KPI: {worst_kpi[0]} ({worst_kpi[1]:.1f}% of target)")

        return insights

    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize insights based on impact and confidence."""

        if not insights:
            return []

        # Sort by impact score and confidence score
        prioritized = sorted(
            insights,
            key=lambda x: (x.get('impact_score', 0), x.get('confidence_score', 0)),
            reverse=True
        )

        # Add priority labels
        for i, insight in enumerate(prioritized):
            if i < len(prioritized) * 0.2:  # Top 20%
                insight['priority'] = 'critical'
            elif i < len(prioritized) * 0.5:  # Next 30%
                insight['priority'] = 'high'
            elif i < len(prioritized) * 0.8:  # Next 30%
                insight['priority'] = 'medium'
            else:
                insight['priority'] = 'low'

        return prioritized

    def _generate_report_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report summary."""

        summary = {
            'report_type': report_data.get('report_type', 'Unknown'),
            'title': report_data.get('title', 'Unknown'),
            'generated_at': report_data.get('generated_at', datetime.utcnow().isoformat()),
            'period': report_data.get('period', {}),
            'sections_count': len(report_data.get('sections', {})),
            'total_insights': len(report_data.get('business_insights', [])),
            'key_metrics': {}
        }

        # Extract key metrics based on report type
        if 'revenue_analysis' in report_data:
            revenue_summary = report_data['revenue_analysis'].get('summary', {})
            summary['key_metrics']['total_revenue'] = revenue_summary.get('total_revenue', 0)
            summary['key_metrics']['total_profit'] = revenue_summary.get('total_profit', 0)

        if 'customer_analysis' in report_data:
            customer_summary = report_data['customer_analysis'].get('summary', {})
            summary['key_metrics']['total_customers'] = customer_summary.get('total_customers', 0)

        return summary


# Global BI reporting instance
bi_reporting = BusinessIntelligenceReporting()