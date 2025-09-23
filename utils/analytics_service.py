"""
Advanced Analytics Service for TTS System Dashboard
Handles data collection, aggregation, analysis, and reporting
"""

import asyncio
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from decimal import Decimal

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, asc
import pandas as pd

from models.analytics import (
    UsageMetric, UserBehavior, PerformanceMetric, BusinessMetric,
    TimeSeriesData, AnalyticsAlert, AnalyticsReport,
    MetricPeriod, AlertSeverity, ReportType
)
from models import SystemMetric
from utils.tenant_manager import tenant_manager


class AnalyticsService:
    """Advanced analytics service for TTS system."""

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._real_time_metrics = {}
        self._alert_rules = {}

    # ===== DATA COLLECTION AND AGGREGATION =====

    def collect_usage_metrics(self, db_session: Session, hours: int = 24) -> Dict[str, Any]:
        """Collect and aggregate usage metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Get usage metrics
        usage_metrics = db_session.query(UsageMetric).filter(
            UsageMetric.timestamp >= cutoff_time
        ).all()

        # Aggregate by different dimensions
        aggregated = {
            'total_requests': sum(m.request_count for m in usage_metrics),
            'total_audio_duration': sum(m.audio_duration_seconds for m in usage_metrics),
            'total_data_processed': sum(m.data_processed_mb for m in usage_metrics),
            'total_errors': sum(m.error_count for m in usage_metrics),
            'total_success': sum(m.success_count for m in usage_metrics),
            'avg_response_time': 0.0,
            'throughput': 0.0,
            'by_endpoint': defaultdict(int),
            'by_voice_type': defaultdict(int),
            'by_language': defaultdict(int),
            'by_organization': defaultdict(lambda: {'requests': 0, 'duration': 0.0}),
            'by_user': defaultdict(lambda: {'requests': 0, 'duration': 0.0}),
            'hourly_distribution': defaultdict(int),
            'error_rate_trend': []
        }

        if usage_metrics:
            # Calculate averages
            response_times = [m.avg_response_time for m in usage_metrics if m.avg_response_time > 0]
            if response_times:
                aggregated['avg_response_time'] = statistics.mean(response_times)

            # Calculate throughput (requests per hour)
            time_span_hours = max(1, hours)
            aggregated['throughput'] = aggregated['total_requests'] / time_span_hours

            # Aggregate by dimensions
            for metric in usage_metrics:
                aggregated['by_endpoint'][metric.endpoint] += metric.request_count
                aggregated['by_voice_type'][metric.voice_type or 'unknown'] += metric.request_count
                aggregated['by_language'][metric.language or 'unknown'] += metric.request_count

                if metric.organization_id:
                    aggregated['by_organization'][metric.organization_id]['requests'] += metric.request_count
                    aggregated['by_organization'][metric.organization_id]['duration'] += metric.audio_duration_seconds

                if metric.user_id:
                    aggregated['by_user'][metric.user_id]['requests'] += metric.request_count
                    aggregated['by_user'][metric.user_id]['duration'] += metric.audio_duration_seconds

                # Hourly distribution
                hour_key = metric.timestamp.strftime('%Y-%m-%d %H:00')
                aggregated['hourly_distribution'][hour_key] += metric.request_count

                # Error rate trend
                if metric.request_count > 0:
                    error_rate = metric.error_count / metric.request_count
                    aggregated['error_rate_trend'].append({
                        'timestamp': metric.timestamp.isoformat(),
                        'error_rate': error_rate,
                        'requests': metric.request_count
                    })

        return dict(aggregated)

    def collect_performance_metrics(self, db_session: Session, hours: int = 24) -> Dict[str, Any]:
        """Collect and aggregate performance metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Get performance metrics
        perf_metrics = db_session.query(PerformanceMetric).filter(
            PerformanceMetric.timestamp >= cutoff_time
        ).all()

        # Get system metrics
        system_metrics = db_session.query(SystemMetric).filter(
            and_(SystemMetric.timestamp >= cutoff_time,
                 SystemMetric.category.in_(['system', 'api', 'tts']))
        ).all()

        aggregated = {
            'system_health': {
                'cpu_usage': [],
                'memory_usage': [],
                'disk_usage': [],
                'network_io': []
            },
            'response_times': {
                'avg': [],
                'p50': [],
                'p95': [],
                'p99': []
            },
            'error_rates': {
                'error_rate': [],
                'timeout_rate': [],
                'retry_rate': []
            },
            'tts_specific': {
                'processing_time': [],
                'generation_rate': [],
                'success_rate': []
            },
            'cache_performance': {
                'hit_rate': [],
                'size_mb': []
            },
            'current_status': {},
            'alerts': []
        }

        # Process performance metrics
        for metric in perf_metrics:
            aggregated['system_health']['cpu_usage'].append(metric.cpu_usage_percent)
            aggregated['system_health']['memory_usage'].append(metric.memory_usage_mb)
            aggregated['system_health']['disk_usage'].append(metric.disk_usage_mb)
            aggregated['system_health']['network_io'].append(metric.network_io_mb)

            aggregated['response_times']['avg'].append(metric.avg_response_time)
            aggregated['response_times']['p50'].append(metric.p50_response_time)
            aggregated['response_times']['p95'].append(metric.p95_response_time)
            aggregated['response_times']['p99'].append(metric.p99_response_time)

            aggregated['error_rates']['error_rate'].append(metric.error_rate_percent)
            aggregated['error_rates']['timeout_rate'].append(metric.timeout_rate_percent)
            aggregated['error_rates']['retry_rate'].append(metric.retry_rate_percent)

            aggregated['tts_specific']['processing_time'].append(metric.tts_processing_time)
            aggregated['tts_specific']['generation_rate'].append(metric.audio_generation_rate)
            aggregated['tts_specific']['success_rate'].append(metric.voice_cloning_success_rate)

            aggregated['cache_performance']['hit_rate'].append(metric.cache_hit_rate)
            aggregated['cache_performance']['size_mb'].append(metric.cache_size_mb)

        # Process system metrics
        for metric in system_metrics:
            if metric.name == 'cpu_usage':
                aggregated['system_health']['cpu_usage'].append(metric.value)
            elif metric.name == 'memory_usage':
                aggregated['system_health']['memory_usage'].append(metric.value)
            elif metric.name == 'disk_usage':
                aggregated['system_health']['disk_usage'].append(metric.value)
            elif metric.name == 'network_io':
                aggregated['system_health']['network_io'].append(metric.value)

        # Calculate current status (latest values)
        if perf_metrics:
            latest = perf_metrics[-1]
            aggregated['current_status'] = {
                'cpu_usage': latest.cpu_usage_percent,
                'memory_usage': latest.memory_usage_mb,
                'active_connections': latest.active_connections,
                'queue_length': latest.queue_length,
                'avg_response_time': latest.avg_response_time,
                'error_rate': latest.error_rate_percent,
                'cache_hit_rate': latest.cache_hit_rate
            }

        # Calculate averages
        for category in aggregated:
            if isinstance(aggregated[category], dict):
                for metric_name in aggregated[category]:
                    if isinstance(aggregated[category][metric_name], list) and aggregated[category][metric_name]:
                        values = [v for v in aggregated[category][metric_name] if v is not None]
                        if values:
                            aggregated[category][metric_name] = statistics.mean(values)

        return aggregated

    def collect_user_behavior_metrics(self, db_session: Session, hours: int = 24) -> Dict[str, Any]:
        """Collect and analyze user behavior metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Get user behavior data
        behaviors = db_session.query(UserBehavior).filter(
            UserBehavior.session_start >= cutoff_time
        ).all()

        aggregated = {
            'total_sessions': len(behaviors),
            'total_users': len(set(b.user_id for b in behaviors if b.user_id)),
            'avg_session_duration': 0.0,
            'total_actions': sum(b.actions_count for b in behaviors),
            'device_breakdown': defaultdict(int),
            'browser_breakdown': defaultdict(int),
            'geographic_distribution': defaultdict(int),
            'engagement_scores': [],
            'conversion_events': defaultdict(int),
            'retention_categories': defaultdict(int),
            'feature_usage': defaultdict(int),
            'error_patterns': defaultdict(int),
            'session_timeline': defaultdict(int),
            'user_journey_patterns': []
        }

        if behaviors:
            # Calculate session duration average
            durations = [b.session_duration_seconds for b in behaviors if b.session_duration_seconds > 0]
            if durations:
                aggregated['avg_session_duration'] = statistics.mean(durations)

            # Process each behavior record
            for behavior in behaviors:
                # Device and browser breakdown
                aggregated['device_breakdown'][behavior.device_type] += 1
                aggregated['browser_breakdown'][behavior.browser] += 1

                # Geographic distribution
                if behavior.country:
                    aggregated['geographic_distribution'][behavior.country] += 1

                # Engagement scores
                if behavior.engagement_score > 0:
                    aggregated['engagement_scores'].append(behavior.engagement_score)

                # Conversion events
                for event in behavior.conversion_events or []:
                    aggregated['conversion_events'][event] += 1

                # Retention categories
                aggregated['retention_categories'][behavior.retention_category] += 1

                # Feature usage
                for feature in behavior.features_used or []:
                    aggregated['feature_usage'][feature] += 1

                # Error patterns
                for error in behavior.errors_encountered or []:
                    aggregated['error_patterns'][error] += 1

                # Session timeline
                hour_key = behavior.session_start.strftime('%Y-%m-%d %H:00')
                aggregated['session_timeline'][hour_key] += 1

            # Calculate engagement statistics
            if aggregated['engagement_scores']:
                aggregated['engagement_stats'] = {
                    'mean': statistics.mean(aggregated['engagement_scores']),
                    'median': statistics.median(aggregated['engagement_scores']),
                    'std_dev': statistics.stdev(aggregated['engagement_scores']) if len(aggregated['engagement_scores']) > 1 else 0
                }

        return dict(aggregated)

    def collect_business_metrics(self, db_session: Session, hours: int = 24) -> Dict[str, Any]:
        """Collect and analyze business metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Get business metrics
        business_metrics = db_session.query(BusinessMetric).filter(
            BusinessMetric.timestamp >= cutoff_time
        ).all()

        aggregated = {
            'revenue': {'total': 0.0, 'by_period': defaultdict(float)},
            'costs': {'total': 0.0, 'by_period': defaultdict(float)},
            'profit': {'total': 0.0, 'margin': 0.0},
            'customer_metrics': {
                'new': 0,
                'active': 0,
                'churned': 0,
                'total_ltv': 0.0
            },
            'usage_metrics': {
                'total_requests': 0,
                'paid_requests': 0,
                'free_requests': 0,
                'api_calls': 0
            },
            'growth_metrics': {
                'growth_rate': 0.0,
                'retention_rate': 0.0,
                'conversion_rate': 0.0
            },
            'popular_items': {
                'voices': defaultdict(int),
                'languages': defaultdict(int),
                'features': defaultdict(int)
            },
            'organization_breakdown': defaultdict(lambda: {
                'revenue': 0.0,
                'requests': 0,
                'tier': None
            })
        }

        if business_metrics:
            # Aggregate metrics
            for metric in business_metrics:
                aggregated['revenue']['total'] += metric.revenue
                aggregated['costs']['total'] += metric.cost
                aggregated['profit']['total'] += metric.profit

                # Revenue and costs by period
                period_key = metric.timestamp.strftime('%Y-%m-%d')
                aggregated['revenue']['by_period'][period_key] += metric.revenue
                aggregated['costs']['by_period'][period_key] += metric.cost

                # Customer metrics
                aggregated['customer_metrics']['new'] += metric.new_customers
                aggregated['customer_metrics']['active'] += metric.active_customers
                aggregated['customer_metrics']['churned'] += metric.churned_customers
                aggregated['customer_metrics']['total_ltv'] += metric.customer_lifetime_value

                # Usage metrics
                aggregated['usage_metrics']['total_requests'] += metric.total_requests
                aggregated['usage_metrics']['paid_requests'] += metric.paid_requests
                aggregated['usage_metrics']['free_requests'] += metric.free_requests
                aggregated['usage_metrics']['api_calls'] += metric.api_calls

                # Growth metrics
                if metric.growth_rate_percent > 0:
                    aggregated['growth_metrics']['growth_rate'] = metric.growth_rate_percent
                if metric.retention_rate_percent > 0:
                    aggregated['growth_metrics']['retention_rate'] = metric.retention_rate_percent
                if metric.conversion_rate_percent > 0:
                    aggregated['growth_metrics']['conversion_rate'] = metric.conversion_rate_percent

                # Popular items
                for voice in metric.popular_voices or []:
                    aggregated['popular_items']['voices'][voice] += 1
                for language in metric.popular_languages or []:
                    aggregated['popular_items']['languages'][language] += 1
                for feature, usage in metric.feature_adoption.items():
                    aggregated['popular_items']['features'][feature] += usage

                # Organization breakdown
                if metric.organization_id:
                    aggregated['organization_breakdown'][metric.organization_id]['revenue'] += metric.revenue
                    aggregated['organization_breakdown'][metric.organization_id]['requests'] += metric.total_requests
                    aggregated['organization_breakdown'][metric.organization_id]['tier'] = metric.tier

            # Calculate profit margin
            if aggregated['revenue']['total'] > 0:
                aggregated['profit']['margin'] = (aggregated['profit']['total'] / aggregated['revenue']['total']) * 100

        return aggregated

    # ===== REAL-TIME METRICS =====

    async def get_real_time_metrics(self, db_session: Session) -> Dict[str, Any]:
        """Get real-time metrics for dashboard."""
        cache_key = 'real_time_metrics'
        current_time = datetime.utcnow()

        # Check cache
        if cache_key in self._cache:
            cached_data, cache_time = self._cache[cache_key]
            if current_time - cache_time < timedelta(seconds=self._cache_ttl):
                return cached_data

        # Collect fresh data
        real_time_data = {
            'timestamp': current_time.isoformat(),
            'usage': await self._get_real_time_usage(db_session),
            'performance': await self._get_real_time_performance(db_session),
            'alerts': await self._get_active_alerts(db_session),
            'system_health': await self._get_system_health_status(db_session)
        }

        # Cache the data
        self._cache[cache_key] = (real_time_data, current_time)

        return real_time_data

    async def _get_real_time_usage(self, db_session: Session) -> Dict[str, Any]:
        """Get real-time usage metrics."""
        # Get last 5 minutes of usage data
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)

        recent_usage = db_session.query(UsageMetric).filter(
            UsageMetric.timestamp >= cutoff_time
        ).all()

        return {
            'requests_last_5min': sum(m.request_count for m in recent_usage),
            'current_throughput': len(recent_usage) / 5 * 60,  # requests per hour
            'active_users': len(set(m.user_id for m in recent_usage if m.user_id)),
            'error_rate': 0.0
        }

    async def _get_real_time_performance(self, db_session: Session) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        # Get latest performance metrics
        latest_perf = db_session.query(PerformanceMetric).order_by(
            desc(PerformanceMetric.timestamp)
        ).first()

        if latest_perf:
            return {
                'cpu_usage': latest_perf.cpu_usage_percent,
                'memory_usage': latest_perf.memory_usage_mb,
                'response_time': latest_perf.avg_response_time,
                'error_rate': latest_perf.error_rate_percent,
                'active_connections': latest_perf.active_connections,
                'queue_length': latest_perf.queue_length
            }

        return {}

    async def _get_active_alerts(self, db_session: Session) -> List[Dict[str, Any]]:
        """Get active alerts."""
        active_alerts = db_session.query(AnalyticsAlert).filter(
            AnalyticsAlert.status == 'active'
        ).order_by(desc(AnalyticsAlert.triggered_at)).limit(10).all()

        return [alert.to_dict() for alert in active_alerts]

    async def _get_system_health_status(self, db_session: Session) -> Dict[str, str]:
        """Get overall system health status."""
        # Simple health check based on latest metrics
        latest_perf = db_session.query(PerformanceMetric).order_by(
            desc(PerformanceMetric.timestamp)
        ).first()

        if not latest_perf:
            return {'status': 'unknown', 'message': 'No performance data available'}

        # Determine health status
        if (latest_perf.error_rate_percent > 10 or
            latest_perf.cpu_usage_percent > 90 or
            latest_perf.memory_usage_mb > 1000):  # Adjust thresholds as needed
            return {'status': 'critical', 'message': 'System under high load'}
        elif (latest_perf.error_rate_percent > 5 or
              latest_perf.cpu_usage_percent > 70):
            return {'status': 'warning', 'message': 'System experiencing issues'}
        else:
            return {'status': 'healthy', 'message': 'System running normally'}

    # ===== HISTORICAL ANALYSIS =====

    def get_historical_trends(self, db_session: Session, metric_type: str,
                            days: int = 30, group_by: str = 'day') -> Dict[str, Any]:
        """Get historical trends for specified metric type."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        trends = {
            'metric_type': metric_type,
            'period': f'{days}_days',
            'group_by': group_by,
            'data_points': [],
            'trend_analysis': {},
            'seasonal_patterns': {},
            'anomalies': []
        }

        if metric_type == 'usage':
            data = db_session.query(UsageMetric).filter(
                UsageMetric.timestamp >= cutoff_time
            ).order_by(asc(UsageMetric.timestamp)).all()

            # Group by specified period
            grouped_data = self._group_time_series_data(data, group_by)

            # Analyze trends
            trends['data_points'] = self._calculate_usage_trends(grouped_data)
            trends['trend_analysis'] = self._analyze_usage_trends(grouped_data)

        elif metric_type == 'performance':
            data = db_session.query(PerformanceMetric).filter(
                PerformanceMetric.timestamp >= cutoff_time
            ).order_by(asc(PerformanceMetric.timestamp)).all()

            grouped_data = self._group_time_series_data(data, group_by)
            trends['data_points'] = self._calculate_performance_trends(grouped_data)
            trends['trend_analysis'] = self._analyze_performance_trends(grouped_data)

        elif metric_type == 'business':
            data = db_session.query(BusinessMetric).filter(
                BusinessMetric.timestamp >= cutoff_time
            ).order_by(asc(BusinessMetric.timestamp)).all()

            grouped_data = self._group_time_series_data(data, group_by)
            trends['data_points'] = self._calculate_business_trends(grouped_data)
            trends['trend_analysis'] = self._analyze_business_trends(grouped_data)

        return trends

    def _group_time_series_data(self, data: List, group_by: str) -> Dict[str, List]:
        """Group time series data by specified period."""
        grouped = defaultdict(list)

        for item in data:
            if group_by == 'hour':
                key = item.timestamp.strftime('%Y-%m-%d %H:00')
            elif group_by == 'day':
                key = item.timestamp.strftime('%Y-%m-%d')
            elif group_by == 'week':
                key = item.timestamp.strftime('%Y-W%W')
            elif group_by == 'month':
                key = item.timestamp.strftime('%Y-%m')
            else:
                key = item.timestamp.strftime('%Y-%m-%d')

            grouped[key].append(item)

        return dict(grouped)

    def _calculate_usage_trends(self, grouped_data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Calculate usage trends from grouped data."""
        trends = []

        for period, metrics in grouped_data.items():
            total_requests = sum(m.request_count for m in metrics)
            total_duration = sum(m.audio_duration_seconds for m in metrics)
            total_errors = sum(m.error_count for m in metrics)
            avg_response_time = statistics.mean([m.avg_response_time for m in metrics if m.avg_response_time > 0]) if metrics else 0

            trends.append({
                'period': period,
                'requests': total_requests,
                'duration': total_duration,
                'errors': total_errors,
                'response_time': avg_response_time,
                'error_rate': total_errors / total_requests if total_requests > 0 else 0
            })

        return sorted(trends, key=lambda x: x['period'])

    def _calculate_performance_trends(self, grouped_data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Calculate performance trends from grouped data."""
        trends = []

        for period, metrics in grouped_data.items():
            avg_cpu = statistics.mean([m.cpu_usage_percent for m in metrics]) if metrics else 0
            avg_memory = statistics.mean([m.memory_usage_mb for m in metrics]) if metrics else 0
            avg_response = statistics.mean([m.avg_response_time for m in metrics if m.avg_response_time > 0]) if metrics else 0
            avg_error_rate = statistics.mean([m.error_rate_percent for m in metrics]) if metrics else 0

            trends.append({
                'period': period,
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'response_time': avg_response,
                'error_rate': avg_error_rate
            })

        return sorted(trends, key=lambda x: x['period'])

    def _calculate_business_trends(self, grouped_data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Calculate business trends from grouped data."""
        trends = []

        for period, metrics in grouped_data.items():
            total_revenue = sum(m.revenue for m in metrics)
            total_cost = sum(m.cost for m in metrics)
            total_requests = sum(m.total_requests for m in metrics)
            new_customers = sum(m.new_customers for m in metrics)

            trends.append({
                'period': period,
                'revenue': total_revenue,
                'cost': total_cost,
                'profit': total_revenue - total_cost,
                'requests': total_requests,
                'new_customers': new_customers
            })

        return sorted(trends, key=lambda x: x['period'])

    def _analyze_usage_trends(self, grouped_data: Dict[str, List]) -> Dict[str, Any]:
        """Analyze usage trends for patterns and insights."""
        if not grouped_data:
            return {}

        # Extract request counts for trend analysis
        periods = list(grouped_data.keys())
        requests = [sum(m.request_count for m in grouped_data[period]) for period in periods]

        if len(requests) < 2:
            return {'trend': 'insufficient_data'}

        # Calculate linear trend
        x = list(range(len(requests)))
        slope, intercept = np.polyfit(x, requests, 1)

        # Determine trend direction
        if slope > 0.1:
            trend_direction = 'increasing'
        elif slope < -0.1:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'

        # Calculate growth rate
        if requests[0] > 0:
            growth_rate = ((requests[-1] - requests[0]) / requests[0]) * 100
        else:
            growth_rate = 0.0

        return {
            'trend_direction': trend_direction,
            'growth_rate_percent': growth_rate,
            'slope': slope,
            'volatility': statistics.stdev(requests) if len(requests) > 1 else 0,
            'periods_analyzed': len(periods)
        }

    def _analyze_performance_trends(self, grouped_data: Dict[str, List]) -> Dict[str, Any]:
        """Analyze performance trends."""
        if not grouped_data:
            return {}

        # Analyze error rate trend
        periods = list(grouped_data.keys())
        error_rates = [statistics.mean([m.error_rate_percent for m in grouped_data[period]]) for period in periods]

        if len(error_rates) < 2:
            return {'trend': 'insufficient_data'}

        # Calculate error rate trend
        x = list(range(len(error_rates)))
        slope, intercept = np.polyfit(x, error_rates, 1)

        return {
            'error_rate_trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'error_rate_slope': slope,
            'avg_error_rate': statistics.mean(error_rates),
            'error_rate_volatility': statistics.stdev(error_rates) if len(error_rates) > 1 else 0
        }

    def _analyze_business_trends(self, grouped_data: Dict[str, List]) -> Dict[str, Any]:
        """Analyze business trends."""
        if not grouped_data:
            return {}

        # Analyze revenue trend
        periods = list(grouped_data.keys())
        revenues = [sum(m.revenue for m in grouped_data[period]) for period in periods]

        if len(revenues) < 2:
            return {'trend': 'insufficient_data'}

        # Calculate revenue trend
        x = list(range(len(revenues)))
        slope, intercept = np.polyfit(x, revenues, 1)

        return {
            'revenue_trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'revenue_slope': slope,
            'avg_revenue': statistics.mean(revenues),
            'revenue_volatility': statistics.stdev(revenues) if len(revenues) > 1 else 0
        }

    # ===== FORECASTING =====

    def generate_forecast(self, db_session: Session, metric_type: str,
                         forecast_days: int = 7) -> Dict[str, Any]:
        """Generate forecast for specified metric type."""
        # Get historical data
        historical_days = forecast_days * 4  # Use 4x data points for forecasting
        historical_data = self.get_historical_trends(
            db_session, metric_type, historical_days, 'day'
        )

        if not historical_data['data_points']:
            return {'error': 'Insufficient historical data for forecasting'}

        forecast = {
            'metric_type': metric_type,
            'forecast_days': forecast_days,
            'historical_data_points': len(historical_data['data_points']),
            'forecast': [],
            'confidence_intervals': [],
            'method': 'linear_regression',
            'accuracy_metrics': {}
        }

        # Simple linear regression forecast
        if metric_type == 'usage':
            values = [point['requests'] for point in historical_data['data_points']]
        elif metric_type == 'business':
            values = [point['revenue'] for point in historical_data['data_points']]
        else:
            return {'error': 'Forecasting not supported for this metric type'}

        if len(values) < 3:
            return {'error': 'Need at least 3 data points for forecasting'}

        # Calculate linear regression
        x = list(range(len(values)))
        slope, intercept = np.polyfit(x, values, 1)

        # Generate forecast
        last_x = x[-1]
        for i in range(1, forecast_days + 1):
            forecast_x = last_x + i
            forecast_value = slope * forecast_x + intercept

            # Simple confidence interval (gets wider for future predictions)
            confidence_range = max(0.1 * forecast_value, abs(slope) * i * 0.5)
            lower_bound = forecast_value - confidence_range
            upper_bound = forecast_value + confidence_range

            forecast['forecast'].append({
                'day': i,
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'value': max(0, forecast_value),  # Ensure non-negative
                'lower_bound': max(0, lower_bound),
                'upper_bound': upper_bound
            })

        # Calculate accuracy metrics on historical data
        predicted = [slope * i + intercept for i in x]
        mae = statistics.mean([abs(a - p) for a, p in zip(values, predicted)])
        rmse = math.sqrt(statistics.mean([(a - p) ** 2 for a, p in zip(values, predicted)]))

        forecast['accuracy_metrics'] = {
            'mae': mae,
            'rmse': rmse,
            'trend_slope': slope
        }

        return forecast

    # ===== REPORT GENERATION =====

    def generate_report(self, db_session: Session, report_type: str,
                       date_from: datetime, date_to: datetime,
                       organization_id: Optional[int] = None,
                       format: str = 'json') -> Dict[str, Any]:
        """Generate analytics report."""
        # Create report record
        report = AnalyticsReport(
            report_name=f"{report_type.title()} Report - {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}",
            report_type=report_type,
            report_format=format,
            date_range_start=date_from,
            date_range_end=date_to,
            organization_id=organization_id,
            status='generating',
            progress_percent=0.0
        )

        db_session.add(report)
        db_session.commit()

        try:
            # Generate report data based on type
            if report_type == 'usage':
                report_data = self._generate_usage_report(db_session, date_from, date_to, organization_id)
            elif report_type == 'performance':
                report_data = self._generate_performance_report(db_session, date_from, date_to, organization_id)
            elif report_type == 'user_behavior':
                report_data = self._generate_user_behavior_report(db_session, date_from, date_to, organization_id)
            elif report_type == 'business':
                report_data = self._generate_business_report(db_session, date_from, date_to, organization_id)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")

            # Update report with generated data
            report.data = report_data
            report.summary = self._generate_report_summary(report_data, report_type)
            report.charts = self._generate_report_charts(report_data, report_type)
            report.insights = self._generate_report_insights(report_data, report_type)
            report.status = 'completed'
            report.progress_percent = 100.0
            report.generated_at = datetime.utcnow()

            db_session.commit()

            return report.to_dict()

        except Exception as e:
            # Mark report as failed
            report.status = 'failed'
            report.error_message = str(e)
            report.progress_percent = 0.0
            db_session.commit()

            return {
                'error': str(e),
                'report_id': report.id,
                'status': 'failed'
            }

    def _generate_usage_report(self, db_session: Session, date_from: datetime,
                              date_to: datetime, organization_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate usage report."""
        # Get usage metrics for date range
        query = db_session.query(UsageMetric).filter(
            and_(UsageMetric.timestamp >= date_from,
                 UsageMetric.timestamp <= date_to)
        )

        if organization_id:
            query = query.filter(UsageMetric.organization_id == organization_id)

        usage_metrics = query.all()

        # Aggregate data
        report_data = {
            'total_requests': sum(m.request_count for m in usage_metrics),
            'total_audio_duration': sum(m.audio_duration_seconds for m in usage_metrics),
            'total_data_processed': sum(m.data_processed_mb for m in usage_metrics),
            'total_errors': sum(m.error_count for m in usage_metrics),
            'total_success': sum(m.success_count for m in usage_metrics),
            'unique_users': len(set(m.user_id for m in usage_metrics if m.user_id)),
            'unique_organizations': len(set(m.organization_id for m in usage_metrics if m.organization_id)),
            'endpoint_usage': defaultdict(int),
            'voice_type_usage': defaultdict(int),
            'language_usage': defaultdict(int),
            'daily_breakdown': defaultdict(lambda: {
                'requests': 0, 'duration': 0.0, 'errors': 0
            }),
            'hourly_patterns': defaultdict(int)
        }

        # Process metrics
        for metric in usage_metrics:
            report_data['endpoint_usage'][metric.endpoint] += metric.request_count
            report_data['voice_type_usage'][metric.voice_type or 'unknown'] += metric.request_count
            report_data['language_usage'][metric.language or 'unknown'] += metric.request_count

            # Daily breakdown
            day_key = metric.timestamp.strftime('%Y-%m-%d')
            report_data['daily_breakdown'][day_key]['requests'] += metric.request_count
            report_data['daily_breakdown'][day_key]['duration'] += metric.audio_duration_seconds
            report_data['daily_breakdown'][day_key]['errors'] += metric.error_count

            # Hourly patterns
            hour_key = metric.timestamp.strftime('%H:00')
            report_data['hourly_patterns'][hour_key] += metric.request_count

        return dict(report_data)

    def _generate_performance_report(self, db_session: Session, date_from: datetime,
                                   date_to: datetime, organization_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate performance report."""
        # Get performance metrics for date range
        perf_metrics = db_session.query(PerformanceMetric).filter(
            and_(PerformanceMetric.timestamp >= date_from,
                 PerformanceMetric.timestamp <= date_to)
        ).all()

        report_data = {
            'avg_cpu_usage': statistics.mean([m.cpu_usage_percent for m in perf_metrics]) if perf_metrics else 0,
            'avg_memory_usage': statistics.mean([m.memory_usage_mb for m in perf_metrics]) if perf_metrics else 0,
            'avg_response_time': statistics.mean([m.avg_response_time for m in perf_metrics if m.avg_response_time > 0]) if perf_metrics else 0,
            'avg_error_rate': statistics.mean([m.error_rate_percent for m in perf_metrics]) if perf_metrics else 0,
            'total_active_connections': sum(m.active_connections for m in perf_metrics),
            'total_queue_length': sum(m.queue_length for m in perf_metrics),
            'system_health_score': 0,  # Calculated based on thresholds
            'performance_incidents': [],
            'resource_utilization': {
                'cpu_trend': 'stable',
                'memory_trend': 'stable',
                'response_time_trend': 'stable'
            }
        }

        # Calculate system health score
        if perf_metrics:
            health_score = 100
            health_score -= min(50, report_data['avg_error_rate'] * 10)  # Error rate impact
            health_score -= min(30, report_data['avg_cpu_usage'] * 0.3)  # CPU usage impact
            health_score -= min(20, report_data['avg_response_time'] * 5)  # Response time impact
            report_data['system_health_score'] = max(0, health_score)

        return report_data

    def _generate_user_behavior_report(self, db_session: Session, date_from: datetime,
                                     date_to: datetime, organization_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate user behavior report."""
        # Get user behavior data for date range
        query = db_session.query(UserBehavior).filter(
            and_(UserBehavior.session_start >= date_from,
                 UserBehavior.session_start <= date_to)
        )

        if organization_id:
            query = query.filter(UserBehavior.organization_id == organization_id)

        behaviors = query.all()

        report_data = {
            'total_sessions': len(behaviors),
            'unique_users': len(set(b.user_id for b in behaviors if b.user_id)),
            'avg_session_duration': statistics.mean([b.session_duration_seconds for b in behaviors if b.session_duration_seconds > 0]) if behaviors else 0,
            'total_actions': sum(b.actions_count for b in behaviors),
            'device_types': defaultdict(int),
            'browsers': defaultdict(int),
            'countries': defaultdict(int),
            'engagement_distribution': defaultdict(int),
            'conversion_funnel': defaultdict(int),
            'user_retention': {
                'new_users': 0,
                'returning_users': 0,
                'loyal_users': 0
            },
            'feature_adoption': defaultdict(int),
            'error_analysis': defaultdict(int)
        }

        # Process behavior data
        for behavior in behaviors:
            report_data['device_types'][behavior.device_type] += 1
            report_data['browsers'][behavior.browser] += 1
            if behavior.country:
                report_data['countries'][behavior.country] += 1

            # Engagement distribution
            if behavior.engagement_score > 0:
                if behavior.engagement_score < 2:
                    engagement_level = 'low'
                elif behavior.engagement_score < 5:
                    engagement_level = 'medium'
                else:
                    engagement_level = 'high'
                report_data['engagement_distribution'][engagement_level] += 1

            # Conversion funnel
            for event in behavior.conversion_events or []:
                report_data['conversion_funnel'][event] += 1

            # User retention
            report_data['user_retention'][behavior.retention_category] += 1

            # Feature adoption
            for feature in behavior.features_used or []:
                report_data['feature_adoption'][feature] += 1

            # Error analysis
            for error in behavior.errors_encountered or []:
                report_data['error_analysis'][error] += 1

        return dict(report_data)

    def _generate_business_report(self, db_session: Session, date_from: datetime,
                                date_to: datetime, organization_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate business report."""
        # Get business metrics for date range
        query = db_session.query(BusinessMetric).filter(
            and_(BusinessMetric.timestamp >= date_from,
                 BusinessMetric.timestamp <= date_to)
        )

        if organization_id:
            query = query.filter(BusinessMetric.organization_id == organization_id)

        business_metrics = query.all()

        report_data = {
            'total_revenue': sum(m.revenue for m in business_metrics),
            'total_costs': sum(m.cost for m in business_metrics),
            'total_profit': sum(m.profit for m in business_metrics),
            'profit_margin': 0.0,
            'customer_acquisition': {
                'new_customers': sum(m.new_customers for m in business_metrics),
                'customer_lifetime_value': statistics.mean([m.customer_lifetime_value for m in business_metrics if m.customer_lifetime_value > 0]) if business_metrics else 0
            },
            'usage_statistics': {
                'total_requests': sum(m.total_requests for m in business_metrics),
                'paid_requests': sum(m.paid_requests for m in business_metrics),
                'free_requests': sum(m.free_requests for m in business_metrics)
            },
            'growth_metrics': {
                'growth_rate': statistics.mean([m.growth_rate_percent for m in business_metrics if m.growth_rate_percent > 0]) if business_metrics else 0,
                'retention_rate': statistics.mean([m.retention_rate_percent for m in business_metrics if m.retention_rate_percent > 0]) if business_metrics else 0,
                'conversion_rate': statistics.mean([m.conversion_rate_percent for m in business_metrics if m.conversion_rate_percent > 0]) if business_metrics else 0
            },
            'popular_voices': dict(sorted(
                [(k, v) for k, v in defaultdict(int, {voice: count for m in business_metrics for voice, count in (m.popular_voices or [])}).items()],
                key=lambda x: x[1], reverse=True
            )[:10]),
            'popular_languages': dict(sorted(
                [(k, v) for k, v in defaultdict(int, {lang: count for m in business_metrics for lang, count in (m.popular_languages or [])}).items()],
                key=lambda x: x[1], reverse=True
            )[:10])
        }

        # Calculate profit margin
        if report_data['total_revenue'] > 0:
            report_data['profit_margin'] = (report_data['total_profit'] / report_data['total_revenue']) * 100

        return report_data

    def _generate_report_summary(self, report_data: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """Generate report summary."""
        summary = {
            'report_type': report_type,
            'generated_at': datetime.utcnow().isoformat(),
            'key_metrics': {},
            'highlights': [],
            'recommendations': []
        }

        if report_type == 'usage':
            summary['key_metrics'] = {
                'total_requests': report_data.get('total_requests', 0),
                'total_audio_duration': report_data.get('total_audio_duration', 0),
                'error_rate': (report_data.get('total_errors', 0) / max(report_data.get('total_requests', 1), 1)) * 100,
                'unique_users': report_data.get('unique_users', 0)
            }
            summary['highlights'] = [
                f"Processed {report_data.get('total_requests', 0)} requests",
                f"Generated {report_data.get('total_audio_duration', 0):.2f} seconds of audio",
                f"Served {report_data.get('unique_users', 0)} unique users"
            ]

        elif report_type == 'performance':
            summary['key_metrics'] = {
                'avg_response_time': report_data.get('avg_response_time', 0),
                'system_health_score': report_data.get('system_health_score', 0),
                'error_rate': report_data.get('avg_error_rate', 0),
                'resource_utilization': report_data.get('avg_cpu_usage', 0)
            }
            summary['highlights'] = [
                f"Average response time: {report_data.get('avg_response_time', 0):.2f}s",
                f"System health score: {report_data.get('system_health_score', 0):.1f}/100",
                f"Error rate: {report_data.get('avg_error_rate', 0):.2f}%"
            ]

        elif report_type == 'user_behavior':
            summary['key_metrics'] = {
                'total_sessions': report_data.get('total_sessions', 0),
                'unique_users': report_data.get('unique_users', 0),
                'avg_session_duration': report_data.get('avg_session_duration', 0),
                'total_actions': report_data.get('total_actions', 0)
            }
            summary['highlights'] = [
                f"{report_data.get('total_sessions', 0)} user sessions recorded",
                f"Average session duration: {report_data.get('avg_session_duration', 0):.2f} seconds",
                f"{report_data.get('total_actions', 0)} total user actions"
            ]

        elif report_type == 'business':
            summary['key_metrics'] = {
                'total_revenue': report_data.get('total_revenue', 0),
                'total_profit': report_data.get('total_profit', 0),
                'profit_margin': report_data.get('profit_margin', 0),
                'new_customers': report_data.get('customer_acquisition', {}).get('new_customers', 0)
            }
            summary['highlights'] = [
                f"Revenue: ${report_data.get('total_revenue', 0):.2f}",
                f"Profit margin: {report_data.get('profit_margin', 0):.2f}%",
                f"{report_data.get('customer_acquisition', {}).get('new_customers', 0)} new customers acquired"
            ]

        return summary

    def _generate_report_charts(self, report_data: Dict[str, Any], report_type: str) -> List[Dict[str, Any]]:
        """Generate chart configurations for report."""
        charts = []

        if report_type == 'usage':
            charts.extend([
                {
                    'type': 'line',
                    'title': 'Daily Request Volume',
                    'data_key': 'daily_breakdown',
                    'x_axis': 'date',
                    'y_axis': 'requests'
                },
                {
                    'type': 'bar',
                    'title': 'Endpoint Usage',
                    'data_key': 'endpoint_usage',
                    'x_axis': 'endpoint',
                    'y_axis': 'requests'
                },
                {
                    'type': 'pie',
                    'title': 'Voice Type Distribution',
                    'data_key': 'voice_type_usage'
                }
            ])

        elif report_type == 'performance':
            charts.extend([
                {
                    'type': 'line',
                    'title': 'System Performance Over Time',
                    'data_key': 'performance_trends',
                    'x_axis': 'timestamp',
                    'y_axis': 'response_time'
                },
                {
                    'type': 'gauge',
                    'title': 'System Health Score',
                    'data_key': 'system_health_score'
                }
            ])

        elif report_type == 'user_behavior':
            charts.extend([
                {
                    'type': 'bar',
                    'title': 'Device Type Distribution',
                    'data_key': 'device_types'
                },
                {
                    'type': 'line',
                    'title': 'Session Timeline',
                    'data_key': 'session_timeline',
                    'x_axis': 'hour',
                    'y_axis': 'sessions'
                }
            ])

        elif report_type == 'business':
            charts.extend([
                {
                    'type': 'line',
                    'title': 'Revenue Trend',
                    'data_key': 'revenue_trend',
                    'x_axis': 'date',
                    'y_axis': 'revenue'
                },
                {
                    'type': 'bar',
                    'title': 'Popular Voices',
                    'data_key': 'popular_voices'
                }
            ])

        return charts

    def _generate_report_insights(self, report_data: Dict[str, Any], report_type: str) -> List[str]:
        """Generate AI-powered insights for report."""
        insights = []

        if report_type == 'usage':
            total_requests = report_data.get('total_requests', 0)
            error_rate = (report_data.get('total_errors', 0) / max(total_requests, 1)) * 100

            if error_rate > 5:
                insights.append(f"High error rate detected: {error_rate:.2f}%. Consider investigating API issues.")
            if total_requests > 10000:
                insights.append(f"High volume period with {total_requests} requests processed.")
            if report_data.get('unique_users', 0) > 100:
                insights.append(f"Strong user engagement with {report_data.get('unique_users', 0)} unique users.")

        elif report_type == 'performance':
            health_score = report_data.get('system_health_score', 0)
            if health_score < 50:
                insights.append("System health is concerning. Immediate attention required.")
            elif health_score < 80:
                insights.append("System performance is below optimal. Consider optimization.")

        elif report_type == 'user_behavior':
            avg_duration = report_data.get('avg_session_duration', 0)
            if avg_duration < 30:
                insights.append("User sessions are short. Consider improving user experience.")
            if report_data.get('total_sessions', 0) > 1000:
                insights.append("High user activity detected. Platform engagement is strong.")

        elif report_type == 'business':
            profit_margin = report_data.get('profit_margin', 0)
            if profit_margin < 10:
                insights.append("Profit margins are low. Consider cost optimization strategies.")
            if report_data.get('customer_acquisition', {}).get('new_customers', 0) > 50:
                insights.append("Strong customer acquisition this period. Growth trajectory is positive.")

        return insights

    # ===== ALERT SYSTEM =====

    def check_alerts(self, db_session: Session) -> List[Dict[str, Any]]:
        """Check for analytics alerts and create if thresholds exceeded."""
        new_alerts = []

        # Check usage alerts
        usage_metrics = self.collect_usage_metrics(db_session, hours=1)
        if usage_metrics['total_requests'] > 1000:  # Example threshold
            alert = self._create_alert(
                db_session,
                name="High Request Volume",
                alert_type="threshold",
                severity="medium",
                metric_name="requests_per_hour",
                current_value=usage_metrics['total_requests'],
                threshold_value=1000,
                description=f"Request volume exceeded threshold: {usage_metrics['total_requests']}/hour"
            )
            if alert:
                new_alerts.append(alert)

        # Check error rate alerts
        if usage_metrics['total_requests'] > 0:
            error_rate = (usage_metrics['total_errors'] / usage_metrics['total_requests']) * 100
            if error_rate > 5:  # Example threshold
                alert = self._create_alert(
                    db_session,
                    name="High Error Rate",
                    alert_type="threshold",
                    severity="high",
                    metric_name="error_rate_percent",
                    current_value=error_rate,
                    threshold_value=5.0,
                    description=f"Error rate exceeded threshold: {error_rate:.2f}%"
                )
                if alert:
                    new_alerts.append(alert)

        # Check performance alerts
        perf_metrics = self.collect_performance_metrics(db_session, hours=1)
        if perf_metrics['system_health']['cpu_usage']:
            cpu_usage = perf_metrics['system_health']['cpu_usage']
            if cpu_usage > 80:  # Example threshold
                alert = self._create_alert(
                    db_session,
                    name="High CPU Usage",
                    alert_type="threshold",
                    severity="high" if cpu_usage > 90 else "medium",
                    metric_name="cpu_usage_percent",
                    current_value=cpu_usage,
                    threshold_value=80.0,
                    description=f"CPU usage is high: {cpu_usage:.1f}%"
                )
                if alert:
                    new_alerts.append(alert)

        return new_alerts

    def _create_alert(self, db_session: Session, name: str, alert_type: str,
                     severity: str, metric_name: str, current_value: float,
                     threshold_value: float, description: str) -> Optional[AnalyticsAlert]:
        """Create a new analytics alert."""
        # Check if similar alert already exists and is active
        existing_alert = db_session.query(AnalyticsAlert).filter(
            and_(AnalyticsAlert.alert_name == name,
                 AnalyticsAlert.status == 'active',
                 AnalyticsAlert.metric_name == metric_name)
        ).first()

        if existing_alert:
            return None  # Don't create duplicate alerts

        alert = AnalyticsAlert(
            alert_name=name,
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            threshold_value=threshold_value,
            threshold_operator=">",
            current_value=current_value,
            expected_value=threshold_value,
            deviation_percent=((current_value - threshold_value) / threshold_value) * 100,
            description=description,
            status='active'
        )

        db_session.add(alert)
        db_session.commit()

        return alert

    def resolve_alert(self, db_session: Session, alert_id: int, resolution_notes: str = None) -> bool:
        """Resolve an analytics alert."""
        alert = db_session.query(AnalyticsAlert).filter(
            AnalyticsAlert.id == alert_id
        ).first()

        if not alert or alert.status != 'active':
            return False

        alert.status = 'resolved'
        alert.resolved_at = datetime.utcnow()
        if resolution_notes:
            alert.resolution_notes = resolution_notes

        db_session.commit()
        return True

    def acknowledge_alert(self, db_session: Session, alert_id: int, user_id: int) -> bool:
        """Acknowledge an analytics alert."""
        alert = db_session.query(AnalyticsAlert).filter(
            AnalyticsAlert.id == alert_id
        ).first()

        if not alert:
            return False

        alert.status = 'acknowledged'
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = user_id

        db_session.commit()
        return True


# Global analytics service instance
analytics_service = AnalyticsService()