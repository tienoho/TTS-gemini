"""
Analytics Dashboard API Routes for TTS System
Provides comprehensive analytics and reporting endpoints
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from flask import Blueprint, jsonify, request, Response
from sqlalchemy.orm import Session
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.extensions import db
from utils.analytics_service import analytics_service
from models import User

analytics_bp = Blueprint('analytics', __name__, url_prefix='/analytics')


@analytics_bp.route('/overview', methods=['GET'])
@jwt_required()
def get_analytics_overview():
    """Get system overview analytics.

    Returns:
        JSON response with system overview metrics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)  # Limit between 1 hour and 30 days

        # Collect real-time metrics
        real_time_metrics = analytics_service.get_real_time_metrics(db.session)

        # Collect usage metrics
        usage_metrics = analytics_service.collect_usage_metrics(db.session, hours)

        # Collect performance metrics
        performance_metrics = analytics_service.collect_performance_metrics(db.session, hours)

        # Collect user behavior metrics
        user_behavior_metrics = analytics_service.collect_user_behavior_metrics(db.session, hours)

        # Collect business metrics
        business_metrics = analytics_service.collect_business_metrics(db.session, hours)

        # Get active alerts
        active_alerts = analytics_service.check_alerts(db.session)

        # Compile overview response
        overview = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'real_time': real_time_metrics,
            'usage_summary': {
                'total_requests': usage_metrics['total_requests'],
                'total_audio_duration': usage_metrics['total_audio_duration'],
                'total_data_processed': usage_metrics['total_data_processed'],
                'total_errors': usage_metrics['total_errors'],
                'total_success': usage_metrics['total_success'],
                'avg_response_time': usage_metrics['avg_response_time'],
                'throughput': usage_metrics['throughput'],
                'error_rate': (usage_metrics['total_errors'] / max(usage_metrics['total_requests'], 1)) * 100
            },
            'performance_summary': {
                'system_health': performance_metrics.get('current_status', {}),
                'avg_cpu_usage': performance_metrics['system_health'].get('cpu_usage', 0),
                'avg_memory_usage': performance_metrics['system_health'].get('memory_usage', 0),
                'avg_response_time': performance_metrics['response_times'].get('avg', 0),
                'avg_error_rate': performance_metrics['error_rates'].get('error_rate', 0)
            },
            'user_summary': {
                'total_sessions': user_behavior_metrics['total_sessions'],
                'unique_users': user_behavior_metrics['total_users'],
                'avg_session_duration': user_behavior_metrics['avg_session_duration'],
                'total_actions': user_behavior_metrics['total_actions'],
                'engagement_score': user_behavior_metrics.get('engagement_stats', {}).get('mean', 0)
            },
            'business_summary': {
                'total_revenue': business_metrics['revenue']['total'],
                'total_costs': business_metrics['costs']['total'],
                'total_profit': business_metrics['profit']['total'],
                'profit_margin': business_metrics['profit']['margin'],
                'new_customers': business_metrics['customer_metrics']['new'],
                'active_customers': business_metrics['customer_metrics']['active']
            },
            'alerts': {
                'total_active': len(active_alerts),
                'recent_alerts': active_alerts[:5]  # Last 5 alerts
            },
            'top_endpoints': sorted(
                usage_metrics['by_endpoint'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'top_voice_types': sorted(
                usage_metrics['by_voice_type'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'geographic_distribution': dict(list(
                user_behavior_metrics['geographic_distribution'].items()
            )[:10])
        }

        return jsonify(overview)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get analytics overview',
            'message': str(e)
        }), 500


@analytics_bp.route('/usage', methods=['GET'])
@jwt_required()
def get_usage_analytics():
    """Get detailed usage analytics.

    Returns:
        JSON response with usage statistics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        group_by = request.args.get('group_by', 'hour')  # hour, day, week, month

        # Collect usage metrics
        usage_metrics = analytics_service.collect_usage_metrics(db.session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'usage', days=hours//24, group_by=group_by
        )

        # Compile usage analytics response
        usage_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'group_by': group_by,
            'current_period': usage_metrics,
            'historical_trends': historical_trends,
            'breakdown': {
                'by_endpoint': dict(usage_metrics['by_endpoint']),
                'by_voice_type': dict(usage_metrics['by_voice_type']),
                'by_language': dict(usage_metrics['by_language']),
                'by_organization': dict(usage_metrics['by_organization']),
                'by_user': dict(usage_metrics['by_user'])
            },
            'patterns': {
                'hourly_distribution': dict(usage_metrics['hourly_distribution']),
                'error_rate_trend': usage_metrics['error_rate_trend']
            }
        }

        return jsonify(usage_analytics)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get usage analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance_analytics():
    """Get system performance analytics.

    Returns:
        JSON response with performance metrics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        # Collect performance metrics
        performance_metrics = analytics_service.collect_performance_metrics(db.session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'performance', days=hours//24, group_by='hour'
        )

        # Compile performance analytics response
        performance_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'current_status': performance_metrics.get('current_status', {}),
            'system_health': performance_metrics.get('system_health', {}),
            'response_times': performance_metrics.get('response_times', {}),
            'error_rates': performance_metrics.get('error_rates', {}),
            'tts_specific': performance_metrics.get('tts_specific', {}),
            'cache_performance': performance_metrics.get('cache_performance', {}),
            'historical_trends': historical_trends
        }

        return jsonify(performance_analytics)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get performance analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/users', methods=['GET'])
@jwt_required()
def get_user_analytics():
    """Get user behavior and engagement analytics.

    Returns:
        JSON response with user analytics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        # Collect user behavior metrics
        user_behavior_metrics = analytics_service.collect_user_behavior_metrics(db.session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'user_behavior', days=hours//24, group_by='day'
        )

        # Compile user analytics response
        user_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'summary': {
                'total_sessions': user_behavior_metrics['total_sessions'],
                'unique_users': user_behavior_metrics['total_users'],
                'avg_session_duration': user_behavior_metrics['avg_session_duration'],
                'total_actions': user_behavior_metrics['total_actions'],
                'engagement_score': user_behavior_metrics.get('engagement_stats', {}).get('mean', 0)
            },
            'behavior_patterns': {
                'device_breakdown': dict(user_behavior_metrics['device_breakdown']),
                'browser_breakdown': dict(user_behavior_metrics['browser_breakdown']),
                'geographic_distribution': dict(user_behavior_metrics['geographic_distribution']),
                'session_timeline': dict(user_behavior_metrics['session_timeline'])
            },
            'engagement_metrics': {
                'engagement_distribution': dict(user_behavior_metrics['engagement_distribution']),
                'conversion_events': dict(user_behavior_metrics['conversion_events']),
                'retention_categories': dict(user_behavior_metrics['retention_categories'])
            },
            'feature_analysis': {
                'feature_usage': dict(user_behavior_metrics['feature_usage']),
                'error_patterns': dict(user_behavior_metrics['error_patterns'])
            },
            'historical_trends': historical_trends
        }

        return jsonify(user_analytics)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get user analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/business', methods=['GET'])
@jwt_required()
def get_business_analytics():
    """Get business intelligence and revenue analytics.

    Returns:
        JSON response with business metrics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        # Collect business metrics
        business_metrics = analytics_service.collect_business_metrics(db.session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'business', days=hours//24, group_by='day'
        )

        # Compile business analytics response
        business_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'financial_summary': {
                'revenue': business_metrics['revenue'],
                'costs': business_metrics['costs'],
                'profit': business_metrics['profit']
            },
            'customer_metrics': business_metrics['customer_metrics'],
            'usage_metrics': business_metrics['usage_metrics'],
            'growth_metrics': business_metrics['growth_metrics'],
            'product_analysis': {
                'popular_voices': business_metrics['popular_items']['voices'],
                'popular_languages': business_metrics['popular_items']['languages'],
                'feature_adoption': business_metrics['popular_items']['features']
            },
            'organization_breakdown': business_metrics['organization_breakdown'],
            'historical_trends': historical_trends
        }

        return jsonify(business_analytics)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get business analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/reports', methods=['POST'])
@jwt_required()
def generate_report():
    """Generate analytics report.

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
        organization_id = data.get('organization_id', user.organization_id)

        # Generate report using analytics service
        report_result = analytics_service.generate_report(
            db.session, report_type, date_from, date_to, organization_id, report_format
        )

        if 'error' in report_result:
            return jsonify(report_result), 400

        return jsonify({
            'message': 'Report generation started',
            'report_id': report_result['id'],
            'status': report_result['status'],
            'estimated_completion': 'within_5_minutes'
        }), 202

    except Exception as e:
        return jsonify({
            'error': 'Failed to generate report',
            'message': str(e)
        }), 500


@analytics_bp.route('/reports/<int:report_id>', methods=['GET'])
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

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get report status',
            'message': str(e)
        }), 500


@analytics_bp.route('/trends', methods=['GET'])
@jwt_required()
def get_trend_analysis():
    """Get trend analysis for specified metric type.

    Returns:
        JSON response with trend analysis
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        metric_type = request.args.get('type', 'usage')  # usage, performance, business, user_behavior
        days = request.args.get('days', 30, type=int)
        days = min(max(days, 1), 365)

        group_by = request.args.get('group_by', 'day')  # hour, day, week, month

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, metric_type, days, group_by
        )

        # Compile trend analysis response
        trend_analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'metric_type': metric_type,
            'analysis_period_days': days,
            'group_by': group_by,
            'historical_data': historical_trends
        }

        return jsonify(trend_analysis)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get trend analysis',
            'message': str(e)
        }), 500


@analytics_bp.route('/alerts', methods=['GET'])
@jwt_required()
def get_alerts():
    """Get analytics alerts.

    Returns:
        JSON response with alerts
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        status = request.args.get('status', 'active')  # active, resolved, acknowledged, all
        severity = request.args.get('severity')  # low, medium, high, critical
        limit = request.args.get('limit', 50, type=int)
        limit = min(max(limit, 1), 1000)

        # Get alerts from database
        from models.analytics import AnalyticsAlert
        query = db.session.query(AnalyticsAlert)

        # Apply filters
        if status != 'all':
            query = query.filter(AnalyticsAlert.status == status)

        if severity:
            query = query.filter(AnalyticsAlert.severity == severity)

        if user.organization_id:
            query = query.filter(AnalyticsAlert.organization_id == user.organization_id)

        alerts = query.order_by(desc(AnalyticsAlert.triggered_at)).limit(limit).all()

        # Compile alerts response
        alerts_response = {
            'timestamp': datetime.utcnow().isoformat(),
            'filters': {
                'status': status,
                'severity': severity,
                'limit': limit
            },
            'summary': {
                'total': len(alerts),
                'by_status': {},
                'by_severity': {}
            },
            'alerts': [alert.to_dict() for alert in alerts]
        }

        # Calculate summary statistics
        for alert in alerts:
            alerts_response['summary']['by_status'][alert.status] = \
                alerts_response['summary']['by_status'].get(alert.status, 0) + 1
            alerts_response['summary']['by_severity'][alert.severity] = \
                alerts_response['summary']['by_severity'].get(alert.severity, 0) + 1

        return jsonify(alerts_response)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get alerts',
            'message': str(e)
        }), 500


@analytics_bp.route('/alerts/<int:alert_id>/resolve', methods=['POST'])
@jwt_required()
def resolve_alert(alert_id: int):
    """Resolve an analytics alert.

    Returns:
        JSON response with resolution status
    """
    try:
        current_user_id = get_jwt_identity()

        # Get request data
        data = request.get_json() or {}
        resolution_notes = data.get('resolution_notes')

        # Resolve alert using analytics service
        success = analytics_service.resolve_alert(
            db.session, alert_id, resolution_notes
        )

        if not success:
            return jsonify({'error': 'Alert not found or already resolved'}), 404

        return jsonify({
            'message': 'Alert resolved successfully',
            'alert_id': alert_id
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to resolve alert',
            'message': str(e)
        }), 500


@analytics_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@jwt_required()
def acknowledge_alert(alert_id: int):
    """Acknowledge an analytics alert.

    Returns:
        JSON response with acknowledgment status
    """
    try:
        current_user_id = get_jwt_identity()

        # Acknowledge alert using analytics service
        success = analytics_service.acknowledge_alert(
            db.session, alert_id, current_user_id
        )

        if not success:
            return jsonify({'error': 'Alert not found'}), 404

        return jsonify({
            'message': 'Alert acknowledged successfully',
            'alert_id': alert_id
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to acknowledge alert',
            'message': str(e)
        }), 500


@analytics_bp.route('/usage', methods=['GET'])
@jwt_required()
def get_usage_analytics() -> Response:
    """Get detailed usage analytics.

    Returns:
        JSON response with usage statistics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        group_by = request.args.get('group_by', 'hour')  # hour, day, week, month
        endpoint = request.args.get('endpoint')
        voice_type = request.args.get('voice_type')
        language = request.args.get('language')

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Collect usage metrics
        usage_metrics = analytics_service.collect_usage_metrics(db.session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'usage', days=hours//24, group_by=group_by
        )

        # Filter by specific dimensions if provided
        filtered_metrics = usage_metrics.copy()

        if endpoint:
            filtered_metrics['by_endpoint'] = {
                k: v for k, v in filtered_metrics['by_endpoint'].items()
                if k == endpoint
            }

        if voice_type:
            filtered_metrics['by_voice_type'] = {
                k: v for k, v in filtered_metrics['by_voice_type'].items()
                if k == voice_type
            }

        if language:
            filtered_metrics['by_language'] = {
                k: v for k, v in filtered_metrics['by_language'].items()
                if k == language
            }

        # Compile usage analytics response
        usage_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'group_by': group_by,
            'current_period': filtered_metrics,
            'historical_trends': historical_trends,
            'breakdown': {
                'by_endpoint': dict(filtered_metrics['by_endpoint']),
                'by_voice_type': dict(filtered_metrics['by_voice_type']),
                'by_language': dict(filtered_metrics['by_language']),
                'by_organization': dict(filtered_metrics['by_organization']),
                'by_user': dict(filtered_metrics['by_user'])
            },
            'patterns': {
                'hourly_distribution': dict(filtered_metrics['hourly_distribution']),
                'peak_usage_hours': _find_peak_hours(filtered_metrics['hourly_distribution']),
                'error_rate_trend': filtered_metrics['error_rate_trend']
            },
            'insights': _generate_usage_insights(filtered_metrics, historical_trends)
        }

        return jsonify(usage_analytics), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get usage analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance_analytics() -> Response:
    """Get system performance analytics.

    Returns:
        JSON response with performance metrics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        metric_type = request.args.get('metric_type', 'all')  # all, system, api, tts

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Collect performance metrics
        performance_metrics = analytics_service.collect_performance_metrics(db.session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'performance', days=hours//24, group_by='hour'
        )

        # Filter by metric type if specified
        if metric_type != 'all':
            # Apply filtering logic based on metric_type
            filtered_metrics = self._filter_performance_metrics(performance_metrics, metric_type)
        else:
            filtered_metrics = performance_metrics

        # Compile performance analytics response
        performance_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'metric_type': metric_type,
            'current_status': filtered_metrics.get('current_status', {}),
            'system_health': filtered_metrics.get('system_health', {}),
            'response_times': filtered_metrics.get('response_times', {}),
            'error_rates': filtered_metrics.get('error_rates', {}),
            'tts_specific': filtered_metrics.get('tts_specific', {}),
            'cache_performance': filtered_metrics.get('cache_performance', {}),
            'historical_trends': historical_trends,
            'health_score': self._calculate_health_score(filtered_metrics),
            'alerts': self._generate_performance_alerts(filtered_metrics),
            'recommendations': self._generate_performance_recommendations(filtered_metrics)
        }

        return jsonify(performance_analytics), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get performance analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/users', methods=['GET'])
@jwt_required()
def get_user_analytics() -> Response:
    """Get user behavior and engagement analytics.

    Returns:
        JSON response with user analytics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        include_details = request.args.get('details', 'false').lower() == 'true'
        segment = request.args.get('segment')  # new, returning, loyal

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Collect user behavior metrics
        user_behavior_metrics = analytics_service.collect_user_behavior_metrics(db_session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'user_behavior', days=hours//24, group_by='day'
        )

        # Filter by segment if specified
        if segment:
            filtered_metrics = self._filter_user_segment(user_behavior_metrics, segment)
        else:
            filtered_metrics = user_behavior_metrics

        # Compile user analytics response
        user_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'segment': segment,
            'summary': {
                'total_sessions': filtered_metrics['total_sessions'],
                'unique_users': filtered_metrics['total_users'],
                'avg_session_duration': filtered_metrics['avg_session_duration'],
                'total_actions': filtered_metrics['total_actions'],
                'engagement_score': filtered_metrics.get('engagement_stats', {}).get('mean', 0)
            },
            'behavior_patterns': {
                'device_breakdown': dict(filtered_metrics['device_breakdown']),
                'browser_breakdown': dict(filtered_metrics['browser_breakdown']),
                'geographic_distribution': dict(filtered_metrics['geographic_distribution']),
                'session_timeline': dict(filtered_metrics['session_timeline'])
            },
            'engagement_metrics': {
                'engagement_distribution': dict(filtered_metrics['engagement_distribution']),
                'conversion_events': dict(filtered_metrics['conversion_events']),
                'retention_categories': dict(filtered_metrics['retention_categories'])
            },
            'feature_analysis': {
                'feature_usage': dict(filtered_metrics['feature_usage']),
                'error_patterns': dict(filtered_metrics['error_patterns'])
            },
            'historical_trends': historical_trends,
            'insights': self._generate_user_insights(filtered_metrics, historical_trends)
        }

        # Include detailed user journeys if requested
        if include_details:
            user_analytics['user_journeys'] = self._get_user_journeys(db.session, hours)

        return jsonify(user_analytics), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get user analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/business', methods=['GET'])
@jwt_required()
def get_business_analytics() -> Response:
    """Get business intelligence and revenue analytics.

    Returns:
        JSON response with business metrics
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        hours = request.args.get('hours', 24, type=int)
        hours = min(max(hours, 1), 720)

        include_forecast = request.args.get('forecast', 'false').lower() == 'true'
        forecast_days = request.args.get('forecast_days', 7, type=int)

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Collect business metrics
        business_metrics = analytics_service.collect_business_metrics(db_session, hours)

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, 'business', days=hours//24, group_by='day'
        )

        # Generate forecast if requested
        forecast_data = None
        if include_forecast:
            forecast_data = analytics_service.generate_forecast(
                db.session, 'business', forecast_days
            )

        # Compile business analytics response
        business_analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'financial_summary': {
                'revenue': business_metrics['revenue'],
                'costs': business_metrics['costs'],
                'profit': business_metrics['profit']
            },
            'customer_metrics': business_metrics['customer_metrics'],
            'usage_metrics': business_metrics['usage_metrics'],
            'growth_metrics': business_metrics['growth_metrics'],
            'product_analysis': {
                'popular_voices': business_metrics['popular_items']['voices'],
                'popular_languages': business_metrics['popular_items']['languages'],
                'feature_adoption': business_metrics['popular_items']['features']
            },
            'organization_breakdown': business_metrics['organization_breakdown'],
            'historical_trends': historical_trends,
            'insights': self._generate_business_insights(business_metrics, historical_trends)
        }

        # Include forecast data if generated
        if forecast_data:
            business_analytics['forecast'] = forecast_data

        return jsonify(business_analytics), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get business analytics',
            'message': str(e)
        }), 500


@analytics_bp.route('/reports', methods=['POST'])
@jwt_required()
def generate_report() -> Response:
    """Generate analytics report.

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
        organization_id = data.get('organization_id', user.organization_id)
        filters = data.get('filters', {})
        parameters = data.get('parameters', {})

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Generate report using analytics service
        report_result = analytics_service.generate_report(
            db_session, report_type, date_from, date_to, organization_id, report_format
        )

        if 'error' in report_result:
            return jsonify(report_result), 400

        return jsonify({
            'message': 'Report generation started',
            'report_id': report_result['id'],
            'status': report_result['status'],
            'estimated_completion': 'within_5_minutes'
        }), 202

    except Exception as e:
        return jsonify({
            'error': 'Failed to generate report',
            'message': str(e)
        }), 500


@analytics_bp.route('/reports/<int:report_id>', methods=['GET'])
@jwt_required()
def get_report_status(report_id: int) -> Response:
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
            response['download_url'] = f"/api/analytics/reports/{report_id}/download"

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get report status',
            'message': str(e)
        }), 500


@analytics_bp.route('/reports/<int:report_id>/download', methods=['GET'])
@jwt_required()
def download_report(report_id: int) -> Response:
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
            # For JSON reports, return JSON data
            return jsonify({
                'report_name': report.report_name,
                'report_type': report.report_type,
                'generated_at': report.generated_at.isoformat() if report.generated_at else None,
                'data': report.data,
                'summary': report.summary,
                'charts': report.charts,
                'insights': report.insights
            }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to download report',
            'message': str(e)
        }), 500


@analytics_bp.route('/trends', methods=['GET'])
@jwt_required()
def get_trend_analysis() -> Response:
    """Get trend analysis for specified metric type.

    Returns:
        JSON response with trend analysis
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        metric_type = request.args.get('type', 'usage')  # usage, performance, business, user_behavior
        days = request.args.get('days', 30, type=int)
        days = min(max(days, 1), 365)

        group_by = request.args.get('group_by', 'day')  # hour, day, week, month
        include_forecast = request.args.get('forecast', 'false').lower() == 'true'
        forecast_days = request.args.get('forecast_days', 7, type=int)

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get historical trends
        historical_trends = analytics_service.get_historical_trends(
            db.session, metric_type, days, group_by
        )

        # Generate forecast if requested
        forecast_data = None
        if include_forecast:
            forecast_data = analytics_service.generate_forecast(
                db.session, metric_type, forecast_days
            )

        # Compile trend analysis response
        trend_analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'metric_type': metric_type,
            'analysis_period_days': days,
            'group_by': group_by,
            'historical_data': historical_trends,
            'trend_summary': self._summarize_trends(historical_trends),
            'anomalies': self._detect_anomalies(historical_trends),
            'seasonal_patterns': self._identify_seasonal_patterns(historical_trends),
            'insights': self._generate_trend_insights(historical_trends)
        }

        # Include forecast data if generated
        if forecast_data:
            trend_analysis['forecast'] = forecast_data

        return jsonify(trend_analysis), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get trend analysis',
            'message': str(e)
        }), 500


@analytics_bp.route('/alerts', methods=['GET'])
@jwt_required()
def get_alerts() -> Response:
    """Get analytics alerts.

    Returns:
        JSON response with alerts
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user for organization context
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get query parameters
        status = request.args.get('status', 'active')  # active, resolved, acknowledged, all
        severity = request.args.get('severity')  # low, medium, high, critical
        limit = request.args.get('limit', 50, type=int)
        limit = min(max(limit, 1), 1000)

        # Set tenant context if user has organization
        if user.organization_id:
            tenant_context = tenant_manager.get_context(user.organization_id)
            if not tenant_context:
                return jsonify({'error': 'Organization context not found'}), 404

        # Get alerts from database
        from models.analytics import AnalyticsAlert
        query = db.session.query(AnalyticsAlert)

        # Apply filters
        if status != 'all':
            query = query.filter(AnalyticsAlert.status == status)

        if severity:
            query = query.filter(AnalyticsAlert.severity == severity)

        if user.organization_id:
            query = query.filter(AnalyticsAlert.organization_id == user.organization_id)

        alerts = query.order_by(desc(AnalyticsAlert.triggered_at)).limit(limit).all()

        # Compile alerts response
        alerts_response = {
            'timestamp': datetime.utcnow().isoformat(),
            'filters': {
                'status': status,
                'severity': severity,
                'limit': limit
            },
            'summary': {
                'total': len(alerts),
                'by_status': {},
                'by_severity': {}
            },
            'alerts': [alert.to_dict() for alert in alerts]
        }

        # Calculate summary statistics
        for alert in alerts:
            alerts_response['summary']['by_status'][alert.status] = \
                alerts_response['summary']['by_status'].get(alert.status, 0) + 1
            alerts_response['summary']['by_severity'][alert.severity] = \
                alerts_response['summary']['by_severity'].get(alert.severity, 0) + 1

        return jsonify(alerts_response), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get alerts',
            'message': str(e)
        }), 500


@analytics_bp.route('/alerts/<int:alert_id>/resolve', methods=['POST'])
@jwt_required()
def resolve_alert(alert_id: int) -> Response:
    """Resolve an analytics alert.

    Returns:
        JSON response with resolution status
    """
    try:
        current_user_id = get_jwt_identity()

        # Get request data
        data = request.get_json() or {}
        resolution_notes = data.get('resolution_notes')

        # Resolve alert using analytics service
        success = analytics_service.resolve_alert(
            db.session, alert_id, resolution_notes
        )

        if not success:
            return jsonify({'error': 'Alert not found or already resolved'}), 404

        return jsonify({
            'message': 'Alert resolved successfully',
            'alert_id': alert_id
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to resolve alert',
            'message': str(e)
        }), 500


@analytics_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@jwt_required()
def acknowledge_alert(alert_id: int) -> Response:
    """Acknowledge an analytics alert.

    Returns:
        JSON response with acknowledgment status
    """
    try:
        current_user_id = get_jwt_identity()

        # Acknowledge alert using analytics service
        success = analytics_service.acknowledge_alert(
            db.session, alert_id, current_user_id
        )

        if not success:
            return jsonify({'error': 'Alert not found'}), 404

        return jsonify({
            'message': 'Alert acknowledged successfully',
            'alert_id': alert_id
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to acknowledge alert',
            'message': str(e)
        }), 500


# ===== HELPER METHODS =====

def _find_peak_hours(self, hourly_distribution: Dict[str, int]) -> List[Dict[str, Any]]:
    """Find peak usage hours from hourly distribution."""
    if not hourly_distribution:
        return []

    # Sort by usage count
    sorted_hours = sorted(
        hourly_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]  # Top 5 peak hours

    return [
        {
            'hour': hour,
            'requests': count,
            'percentage': (count / sum(hourly_distribution.values())) * 100 if sum(hourly_distribution.values()) > 0 else 0
        }
        for hour, count in sorted_hours
    ]


def _generate_usage_insights(self, usage_metrics: Dict[str, Any], historical_trends: Dict[str, Any]) -> List[str]:
    """Generate insights from usage metrics."""
    insights = []

    # Error rate insights
    error_rate = (usage_metrics['total_errors'] / max(usage_metrics['total_requests'], 1)) * 100
    if error_rate > 5:
        insights.append(f"High error rate detected: {error_rate:.2f}%. Consider investigating API issues.")
    elif error_rate < 1:
        insights.append("Excellent system stability with very low error rate.")

    # Usage volume insights
    if usage_metrics['total_requests'] > 10000:
        insights.append(f"High volume period with {usage_metrics['total_requests']} requests processed.")
    elif usage_metrics['total_requests'] < 100:
        insights.append("Low usage period detected. Consider promotional activities.")

    # Response time insights
    if usage_metrics['avg_response_time'] > 5.0:
        insights.append("Response times are slower than optimal. Consider performance optimization.")
    elif usage_metrics['avg_response_time'] < 0.5:
        insights.append("Excellent response times achieved.")

    # Trend insights
    if historical_trends.get('trend_analysis', {}).get('trend_direction') == 'increasing':
        insights.append("Usage is trending upward. Positive growth indicator.")
    elif historical_trends.get('trend_analysis', {}).get('trend_direction') == 'decreasing':
        insights.append("Usage is declining. Consider investigating causes.")

    return insights


def _filter_performance_metrics(self, performance_metrics: Dict[str, Any], metric_type: str) -> Dict[str, Any]:
    """Filter performance metrics by type."""
    # This is a simplified implementation
    # In a real system, you would have more sophisticated filtering logic
    return performance_metrics


def _calculate_health_score(self, performance_metrics: Dict[str, Any]) -> float:
    """Calculate overall system health score."""
    try:
        # Simple health score calculation
        health_score = 100.0

        # Deduct points for high error rates
        avg_error_rate = performance_metrics.get('error_rates', {}).get('error_rate', 0)
        health_score -= min(50, avg_error_rate * 10)

        # Deduct points for high CPU usage
        avg_cpu = performance_metrics.get('system_health', {}).get('cpu_usage', 0)
        health_score -= min(30, avg_cpu * 0.3)

        # Deduct points for slow response times
        avg_response = performance_metrics.get('response_times', {}).get('avg', 0)
        health_score -= min(20, avg_response * 5)

        return max(0, health_score)
    except:
        return 0.0


def _generate_performance_alerts(self, performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate performance alerts based on current metrics."""
    alerts = []

    # CPU usage alert
    cpu_usage = performance_metrics.get('system_health', {}).get('cpu_usage', 0)
    if cpu_usage > 80:
        alerts.append({
            'type': 'warning',
            'metric': 'cpu_usage',
            'message': f'High CPU usage: {cpu_usage:.1f}%',
            'threshold': 80
        })

    # Memory usage alert
    memory_usage = performance_metrics.get('system_health', {}).get('memory_usage', 0)
    if memory_usage > 1000:  # Assuming MB
        alerts.append({
            'type': 'warning',
            'metric': 'memory_usage',
            'message': f'High memory usage: {memory_usage:.1f} MB',
            'threshold': 1000
        })

    # Error rate alert
    error_rate = performance_metrics.get('error_rates', {}).get('error_rate', 0)
    if error_rate > 5:
        alerts.append({
            'type': 'critical',
            'metric': 'error_rate',
            'message': f'High error rate: {error_rate:.2f}%',
            'threshold': 5
        })

    return alerts


def _generate_performance_recommendations(self, performance_metrics: Dict[str, Any]) -> List[str]:
    """Generate performance recommendations."""
    recommendations = []

    cpu_usage = performance_metrics.get('system_health', {}).get('cpu_usage', 0)
    if cpu_usage > 70:
        recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive operations.")

    memory_usage = performance_metrics.get('system_health', {}).get('memory_usage', 0)
    if memory_usage > 800:
        recommendations.append("Memory usage is high. Consider optimizing memory usage or increasing memory allocation.")

    error_rate = performance_metrics.get('error_rates', {}).get('error_rate', 0)
    if error_rate > 3:
        recommendations.append("Error rate is elevated. Review error logs and implement fixes.")

    response_time = performance_metrics.get('response_times', {}).get('avg', 0)
    if response_time > 3.0:
        recommendations.append("Response times are slow. Consider caching, database optimization, or code optimization.")

    return recommendations


def _filter_user_segment(self, user_behavior_metrics: Dict[str, Any], segment: str) -> Dict[str, Any]:
    """Filter user behavior metrics by segment."""
    # This is a simplified implementation
    # In a real system, you would have more sophisticated segmentation logic
    return user_behavior_metrics


def _generate_user_insights(self, user_behavior_metrics: Dict[str, Any], historical_trends: Dict[str, Any]) -> List[str]:
    """Generate insights from user behavior metrics."""
    insights = []

    total_sessions = user_behavior_metrics.get('total_sessions', 0)
    if total_sessions > 1000:
        insights.append(f"High user engagement with {total_sessions} sessions recorded.")
    elif total_sessions < 10:
        insights.append("Very low user activity. Consider improving user acquisition strategies.")

    avg_duration = user_behavior_metrics.get('avg_session_duration', 0)
    if avg_duration < 30:
        insights.append("User sessions are short. Consider improving user experience and content.")
    elif avg_duration > 300:
        insights.append("Users are spending significant time on the platform. Excellent engagement!")

    engagement_score = user_behavior_metrics.get('engagement_stats', {}).get('mean', 0)
    if engagement_score > 7:
        insights.append("High user engagement scores indicate satisfied users.")
    elif engagement_score < 3:
        insights.append("Low engagement scores suggest users may not be finding value.")

    return insights


def _get_user_journeys(self, db_session: Session, hours: int) -> List[Dict[str, Any]]:
    """Get detailed user journey data."""
    # This is a simplified implementation
    # In a real system, you would have more sophisticated journey tracking
    return []


def _generate_business_insights(self, business_metrics: Dict[str, Any], historical_trends: Dict[str, Any]) -> List[str]:
    """Generate insights from business metrics."""
    insights = []

    profit_margin = business_metrics.get('profit', {}).get('margin', 0)
    if profit_margin < 10:
        insights.append("Profit margins are low. Consider cost optimization strategies.")
    elif profit_margin > 50:
        insights.append("Excellent profit margins achieved. Consider expansion opportunities.")

    new_customers = business_metrics.get('customer_metrics', {}).get('new', 0)
    if new_customers > 50:
        insights.append("Strong customer acquisition this period. Growth trajectory is positive.")
    elif new_customers < 5:
        insights.append("Low customer acquisition. Consider reviewing marketing and sales strategies.")

    growth_rate = business_metrics.get('growth_metrics', {}).get('growth_rate', 0)
    if growth_rate > 20:
        insights.append("Impressive growth rate. Business is expanding rapidly.")
    elif growth_rate < -10:
        insights.append("Negative growth detected. Consider investigating market conditions and competition.")

    return insights


def _summarize_trends(self, historical_trends: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize trend analysis results."""
    trend_analysis = historical_trends.get('trend_analysis', {})

    return {
        'direction': trend_analysis.get('trend_direction', 'stable'),
        'growth_rate': trend_analysis.get('growth_rate_percent', 0),
        'volatility': trend_analysis.get('volatility', 0),
        'data_points': len(historical_trends.get('data_points', [])),
        'periods_analyzed': trend_analysis.get('periods_analyzed', 0)
    }


def _detect_anomalies(self, historical_trends: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect anomalies in trend data."""
    # This is a simplified implementation
    # In a real system, you would use statistical methods like Z-score, IQR, etc.
    return []


def _identify_seasonal_patterns(self, historical_trends: Dict[str, Any]) -> Dict[str, Any]:
    """Identify seasonal patterns in trend data."""
    # This is a simplified implementation
    # In a real system, you would use seasonal decomposition, Fourier analysis, etc.
    return {
        'has_seasonality': False,
        'seasonal_strength': 0.0,
        'peak_periods': [],
        'low_periods': []
    }


def _generate_trend_insights(self, historical_trends: Dict[str, Any]) -> List[str]:
    """Generate insights from trend analysis."""
    insights = []
    trend_summary = self._summarize_trends(historical_trends)

    if trend_summary['direction'] == 'increasing':
        insights.append("Trend is upward. Positive growth indicator.")
    elif trend_summary['direction'] == 'decreasing':
        insights.append("Trend is downward. Consider investigating causes.")
    else:
        insights.append("Trend is stable. Consistent performance observed.")

    if trend_summary['volatility'] > 50:
        insights.append("High volatility detected. Consider stabilization strategies.")
    elif trend_summary['volatility'] < 10:
        insights.append("Low volatility indicates stable and predictable performance.")

    return insights