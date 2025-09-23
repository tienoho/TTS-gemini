"""
Business Intelligence API Routes for TTS System
Provides comprehensive BI endpoints for revenue, customers, usage, KPIs, and reporting
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from flask import Blueprint, jsonify, request, send_file, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
import json

from app.extensions import db
from models import User
from utils.bi_service import bi_service
from utils.bi_analytics import bi_analytics
from utils.bi_dashboard import bi_dashboard
from utils.bi_reporting import bi_reporting
from utils.tenant_manager import tenant_manager

bi_bp = Blueprint('business_intelligence', __name__, url_prefix='/bi')


@bi_bp.route('/revenue', methods=['GET'])
@jwt_required()
def get_revenue_analytics():
    """Get comprehensive revenue analytics.

    Returns:
        JSON response with revenue metrics and analysis
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        days = request.args.get('days', 90, type=int)
        days = min(max(days, 1), 365)  # Limit between 1 and 365 days

        include_forecast = request.args.get('forecast', 'false').lower() == 'true'
        forecast_months = request.args.get('forecast_months', 6, type=int)

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get revenue dashboard
        revenue_dashboard = bi_dashboard.create_revenue_dashboard(user.organization_id, db.session)

        # Get revenue attribution if requested
        attribution_data = None
        if request.args.get('attribution', 'false').lower() == 'true':
            attribution_data = bi_analytics.perform_revenue_attribution(
                user.organization_id, start_date, end_date, db.session
            )

        # Get forecast if requested
        forecast_data = None
        if include_forecast:
            forecast_data = bi_service.generate_financial_forecast(
                user.organization_id, forecast_months, db.session
            )

        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'revenue_dashboard': revenue_dashboard,
            'organization_id': user.organization_id
        }

        if attribution_data:
            response['revenue_attribution'] = attribution_data

        if forecast_data:
            response['revenue_forecast'] = forecast_data

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get revenue analytics',
            'message': str(e)
        }), 500


@bi_bp.route('/customers', methods=['GET'])
@jwt_required()
def get_customer_analytics():
    """Get comprehensive customer analytics.

    Returns:
        JSON response with customer metrics and analysis
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        days = request.args.get('days', 90, type=int)
        days = min(max(days, 1), 365)

        include_segmentation = request.args.get('segmentation', 'false').lower() == 'true'
        include_churn = request.args.get('churn', 'false').lower() == 'true'
        include_cohorts = request.args.get('cohorts', 'false').lower() == 'true'

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get customer dashboard
        customer_dashboard = bi_dashboard.create_customer_dashboard(user.organization_id, db.session)

        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'customer_dashboard': customer_dashboard,
            'organization_id': user.organization_id
        }

        # Add advanced analytics if requested
        if include_segmentation:
            segmentation_data = bi_analytics.perform_customer_segmentation(
                user.organization_id, db.session
            )
            response['customer_segmentation'] = segmentation_data

        if include_churn:
            churn_data = bi_analytics.predict_customer_churn(user.organization_id, db.session)
            response['churn_prediction'] = churn_data

        if include_cohorts:
            cohort_data = bi_analytics.analyze_cohort_performance(user.organization_id, db.session)
            response['cohort_analysis'] = cohort_data

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get customer analytics',
            'message': str(e)
        }), 500


@bi_bp.route('/usage', methods=['GET'])
@jwt_required()
def get_usage_analytics():
    """Get comprehensive usage analytics.

    Returns:
        JSON response with usage metrics and analysis
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        days = request.args.get('days', 30, type=int)
        days = min(max(days, 1), 365)

        include_patterns = request.args.get('patterns', 'false').lower() == 'true'
        include_anomalies = request.args.get('anomalies', 'false').lower() == 'true'

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get usage dashboard
        usage_dashboard = bi_dashboard.create_usage_dashboard(user.organization_id, db.session)

        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'usage_dashboard': usage_dashboard,
            'organization_id': user.organization_id
        }

        # Add advanced analytics if requested
        if include_patterns:
            patterns_data = bi_analytics.analyze_usage_patterns(
                user.organization_id, start_date, end_date, db.session
            )
            response['usage_patterns'] = patterns_data

        if include_anomalies:
            anomalies_data = bi_analytics.detect_anomalies_advanced(
                user.organization_id, start_date, end_date, db.session
            )
            response['anomaly_detection'] = anomalies_data

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get usage analytics',
            'message': str(e)
        }), 500


@bi_bp.route('/kpis', methods=['GET'])
@jwt_required()
def get_kpi_analytics():
    """Get key performance indicators.

    Returns:
        JSON response with KPI metrics and analysis
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        days = request.args.get('days', 30, type=int)
        days = min(max(days, 1), 365)

        include_detailed = request.args.get('detailed', 'false').lower() == 'true'

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get KPI dashboard
        kpi_dashboard = bi_dashboard.create_kpi_dashboard(user.organization_id, db.session)

        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'kpi_dashboard': kpi_dashboard,
            'organization_id': user.organization_id
        }

        # Add detailed KPI calculations if requested
        if include_detailed:
            detailed_kpis = bi_service.calculate_kpis(
                user.organization_id, start_date, end_date, db.session
            )
            response['detailed_kpis'] = detailed_kpis

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get KPI analytics',
            'message': str(e)
        }), 500


@bi_bp.route('/reports', methods=['POST'])
@jwt_required()
def generate_report():
    """Generate a business intelligence report.

    Returns:
        JSON response with report generation status
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get request data
        data = request.get_json() or {}

        # Validate required fields
        if 'report_type' not in data:
            return jsonify({'error': 'report_type is required'}), 400

        if 'date_from' not in data or 'date_to' not in data:
            return jsonify({'error': 'date_from and date_to are required'}), 400

        # Parse dates
        try:
            date_from = datetime.fromisoformat(data['date_from'].replace('Z', '+00:00'))
            date_to = datetime.fromisoformat(data['date_to'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use ISO format.'}), 400

        # Validate date range
        if date_to <= date_from:
            return jsonify({'error': 'date_to must be after date_from'}), 400

        if (date_to - date_from).days > 365:
            return jsonify({'error': 'Date range cannot exceed 365 days'}), 400

        # Get optional parameters
        report_type = data['report_type']
        report_format = data.get('format', 'json')
        parameters = data.get('parameters', {})

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Generate report using BI reporting service
        report_result = bi_reporting.generate_report(
            report_type, user.organization_id, date_from, date_to, report_format, parameters, db.session
        )

        if 'error' in report_result:
            return jsonify(report_result), 400

        return jsonify({
            'message': 'Report generation started',
            'report_id': report_result['report_id'],
            'status': report_result['status'],
            'estimated_completion': 'within_5_minutes'
        }), 202

    except Exception as e:
        return jsonify({
            'error': 'Failed to generate report',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/<int:report_id>', methods=['GET'])
@jwt_required()
def get_report_status(report_id: int):
    """Get report generation status.

    Returns:
        JSON response with report status and data if completed
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get report from database
        from models.analytics import AnalyticsReport
        report = db.session.query(AnalyticsReport).filter(
            AnalyticsReport.id == report_id
        ).first()

        if not report:
            return jsonify({'error': 'Report not found'}), 404

        # Check if user has access to this report
        if report.organization_id != user.organization_id and report.user_id != user.id:
            return jsonify({'error': 'Access denied'}), 403

        # Return report status and data
        response = report.to_dict()

        if report.status == 'completed' and report.data:
            response['download_url'] = f"/api/bi/reports/{report_id}/download"

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get report status',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/<int:report_id>/download', methods=['GET'])
@jwt_required()
def download_report(report_id: int):
    """Download generated report.

    Returns:
        File download response
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get report from database
        from models.analytics import AnalyticsReport
        report = db.session.query(AnalyticsReport).filter(
            AnalyticsReport.id == report_id
        ).first()

        if not report:
            return jsonify({'error': 'Report not found'}), 404

        # Check if user has access to this report
        if report.organization_id != user.organization_id and report.user_id != user.id:
            return jsonify({'error': 'Access denied'}), 403

        # Check if report is completed
        if report.status != 'completed':
            return jsonify({'error': 'Report not ready for download'}), 400

        # Return file download
        if report.file_path and report.report_format in ['csv', 'excel']:
            # For CSV/Excel files, return file download
            try:
                return send_file(
                    report.file_path,
                    as_attachment=True,
                    download_name=f"{report.report_name}.{report.report_format}"
                )
            except FileNotFoundError:
                return jsonify({'error': 'Report file not found'}), 404
        else:
            # For JSON/HTML reports, return JSON data
            return jsonify({
                'report_name': report.report_name,
                'report_type': report.report_type,
                'generated_at': report.generated_at.isoformat() if report.generated_at else None,
                'data': report.data,
                'summary': report.summary
            }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to download report',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/scheduled', methods=['POST'])
@jwt_required()
def schedule_report():
    """Schedule a recurring report.

    Returns:
        JSON response with scheduling status
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get request data
        data = request.get_json() or {}

        # Validate required fields
        if 'report_type' not in data:
            return jsonify({'error': 'report_type is required'}), 400

        if 'schedule_config' not in data:
            return jsonify({'error': 'schedule_config is required'}), 400

        # Get parameters
        report_type = data['report_type']
        schedule_config = data['schedule_config']

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Schedule report using BI reporting service
        schedule_result = bi_reporting.schedule_report(
            report_type, user.organization_id, schedule_config, db.session
        )

        if 'error' in schedule_result:
            return jsonify(schedule_result), 400

        return jsonify({
            'message': 'Report scheduled successfully',
            'scheduled_report_id': schedule_result['scheduled_report_id'],
            'report_type': schedule_result['report_type'],
            'next_run': schedule_result['next_run']
        }), 201

    except Exception as e:
        return jsonify({
            'error': 'Failed to schedule report',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/scheduled/process', methods=['POST'])
@jwt_required()
def process_scheduled_reports():
    """Process all due scheduled reports.

    Returns:
        JSON response with processing results
    """
    try:
        # This endpoint would typically be called by a cron job or task scheduler
        # For now, we'll process reports on demand

        # Process scheduled reports
        processing_result = bi_reporting.process_scheduled_reports(db.session)

        return jsonify({
            'message': 'Scheduled reports processed',
            'processed_count': processing_result['processed_count'],
            'failed_count': processing_result['failed_count'],
            'total_due': processing_result['total_due'],
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to process scheduled reports',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/scheduled', methods=['GET'])
@jwt_required()
def get_scheduled_reports():
    """Get list of scheduled reports.

    Returns:
        JSON response with scheduled reports
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get scheduled reports
        from models.analytics import AnalyticsReport
        scheduled_reports = db.session.query(AnalyticsReport).filter(
            AnalyticsReport.organization_id == user.organization_id,
            AnalyticsReport.is_scheduled == True
        ).order_by(AnalyticsReport.next_run_at).all()

        reports_list = []
        for report in scheduled_reports:
            reports_list.append({
                'id': report.id,
                'report_name': report.report_name,
                'report_type': report.report_type,
                'report_format': report.report_format,
                'schedule_config': report.schedule_config,
                'next_run_at': report.next_run_at.isoformat() if report.next_run_at else None,
                'status': report.status,
                'created_at': report.created_at.isoformat() if report.created_at else None
            })

        return jsonify({
            'scheduled_reports': reports_list,
            'total_count': len(reports_list),
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get scheduled reports',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/scheduled/<int:report_id>', methods=['DELETE'])
@jwt_required()
def delete_scheduled_report(report_id: int):
    """Delete a scheduled report.

    Returns:
        JSON response with deletion status
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get and delete scheduled report
        from models.analytics import AnalyticsReport
        report = db.session.query(AnalyticsReport).filter(
            AnalyticsReport.id == report_id,
            AnalyticsReport.organization_id == user.organization_id,
            AnalyticsReport.is_scheduled == True
        ).first()

        if not report:
            return jsonify({'error': 'Scheduled report not found'}), 404

        db.session.delete(report)
        db.session.commit()

        return jsonify({
            'message': 'Scheduled report deleted successfully',
            'report_id': report_id
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to delete scheduled report',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/custom', methods=['POST'])
@jwt_required()
def create_custom_report():
    """Create a custom report based on user configuration.

    Returns:
        JSON response with custom report
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get request data
        data = request.get_json() or {}

        # Validate required fields
        required_fields = ['name', 'date_from', 'date_to', 'sections']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Create custom report
        custom_report_result = bi_reporting.create_custom_report(
            user.organization_id, data, db.session
        )

        if 'error' in custom_report_result:
            return jsonify(custom_report_result), 400

        return jsonify({
            'message': 'Custom report created successfully',
            'report_id': custom_report_result['report_id'],
            'report_name': custom_report_result['report_name'],
            'data': custom_report_result['data']
        }), 201

    except Exception as e:
        return jsonify({
            'error': 'Failed to create custom report',
            'message': str(e)
        }), 500


@bi_bp.route('/reports/email', methods=['POST'])
@jwt_required()
def send_report_via_email():
    """Send a report via email.

    Returns:
        JSON response with email delivery status
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get request data
        data = request.get_json() or {}

        # Validate required fields
        if 'report_id' not in data:
            return jsonify({'error': 'report_id is required'}), 400

        if 'recipients' not in data:
            return jsonify({'error': 'recipients is required'}), 400

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Send report via email
        email_result = bi_reporting.send_report_via_email(
            data['report_id'], data['recipients'], db.session
        )

        if 'error' in email_result:
            return jsonify(email_result), 400

        return jsonify({
            'message': 'Report sent via email successfully',
            'status': email_result['status'],
            'recipients': email_result['recipients'],
            'report_id': data['report_id']
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to send report via email',
            'message': str(e)
        }), 500


@bi_bp.route('/insights', methods=['GET'])
@jwt_required()
def get_business_insights():
    """Get AI-generated business insights.

    Returns:
        JSON response with business insights and recommendations
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Generate fresh business insights
        insights = bi_service.generate_business_insights(user.organization_id, db.session)

        # Get insights dashboard
        insights_dashboard = bi_dashboard.create_insights_dashboard(user.organization_id, db.session)

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'organization_id': user.organization_id,
            'business_insights': insights,
            'insights_dashboard': insights_dashboard,
            'total_insights': len(insights),
            'high_impact_insights': len([i for i in insights if i.get('impact_score', 0) > 0.7])
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get business insights',
            'message': str(e)
        }), 500


@bi_bp.route('/forecasting', methods=['GET'])
@jwt_required()
def get_forecasting_data():
    """Get forecasting data for revenue and demand.

    Returns:
        JSON response with forecasting data
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        forecast_months = request.args.get('months', 6, type=int)
        forecast_months = min(max(forecast_months, 1), 24)  # Limit between 1 and 24 months

        include_demand = request.args.get('demand', 'false').lower() == 'true'

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get revenue forecast
        revenue_forecast = bi_service.generate_financial_forecast(
            user.organization_id, forecast_months, db.session
        )

        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'organization_id': user.organization_id,
            'forecast_months': forecast_months,
            'revenue_forecast': revenue_forecast
        }

        # Add demand forecast if requested
        if include_demand:
            demand_forecast = bi_analytics.forecast_demand(
                user.organization_id, forecast_months * 30, db.session  # Convert months to days
            )
            response['demand_forecast'] = demand_forecast

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get forecasting data',
            'message': str(e)
        }), 500


@bi_bp.route('/anomalies', methods=['GET'])
@jwt_required()
def get_anomaly_detection():
    """Get anomaly detection results.

    Returns:
        JSON response with anomaly detection data
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        days = request.args.get('days', 30, type=int)
        days = min(max(days, 1), 90)  # Limit to 90 days for performance

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get anomaly detection results
        anomaly_data = bi_analytics.detect_anomalies_advanced(
            user.organization_id, start_date, end_date, db.session
        )

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'organization_id': user.organization_id,
            'anomaly_detection': anomaly_data
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get anomaly detection data',
            'message': str(e)
        }), 500


@bi_bp.route('/segmentation', methods=['GET'])
@jwt_required()
def get_customer_segmentation():
    """Get customer segmentation analysis.

    Returns:
        JSON response with customer segmentation data
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get customer segmentation
        segmentation_data = bi_analytics.perform_customer_segmentation(
            user.organization_id, db.session
        )

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'organization_id': user.organization_id,
            'customer_segmentation': segmentation_data
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get customer segmentation data',
            'message': str(e)
        }), 500


@bi_bp.route('/attribution', methods=['GET'])
@jwt_required()
def get_revenue_attribution():
    """Get revenue attribution analysis.

    Returns:
        JSON response with revenue attribution data
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        days = request.args.get('days', 90, type=int)
        days = min(max(days, 1), 365)

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get revenue attribution
        attribution_data = bi_analytics.perform_revenue_attribution(
            user.organization_id, start_date, end_date, db.session
        )

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'organization_id': user.organization_id,
            'revenue_attribution': attribution_data
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get revenue attribution data',
            'message': str(e)
        }), 500


@bi_bp.route('/churn', methods=['GET'])
@jwt_required()
def get_churn_prediction():
    """Get customer churn prediction.

    Returns:
        JSON response with churn prediction data
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get churn prediction
        churn_data = bi_analytics.predict_customer_churn(user.organization_id, db.session)

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'organization_id': user.organization_id,
            'churn_prediction': churn_data
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get churn prediction data',
            'message': str(e)
        }), 500


@bi_bp.route('/cohorts', methods=['GET'])
@jwt_required()
def get_cohort_analysis():
    """Get customer cohort analysis.

    Returns:
        JSON response with cohort analysis data
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get cohort analysis
        cohort_data = bi_analytics.analyze_cohort_performance(user.organization_id, db.session)

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'organization_id': user.organization_id,
            'cohort_analysis': cohort_data
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get cohort analysis data',
            'message': str(e)
        }), 500