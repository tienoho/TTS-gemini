"""
Business Intelligence Service for TTS System
Handles data aggregation, KPI calculations, trend analysis, and forecasting
"""

import calendar
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import pandas as pd

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, asc

from models.business_intelligence import (
    RevenueStream, CustomerJourney, BusinessKPI, UsagePattern,
    FinancialProjection, BusinessInsight
)
from models.analytics import UsageMetric, BusinessMetric, TimeSeriesData
from models.organization import Organization, OrganizationBilling
from models.user import User
from utils.billing_manager import billing_manager
from config.business_intelligence import bi_config


class BusinessIntelligenceService:
    """Main BI service for data aggregation and analysis."""

    def __init__(self):
        self.config = bi_config
        self.cache = {}

    def calculate_revenue_metrics(self, organization_id: int, start_date: datetime,
                                end_date: datetime, db_session: Session) -> Dict[str, Any]:
        """Calculate comprehensive revenue metrics."""

        # Get revenue streams
        revenue_streams = db_session.query(RevenueStream).filter(
            and_(
                RevenueStream.organization_id == organization_id,
                RevenueStream.recognized_at >= start_date,
                RevenueStream.recognized_at <= end_date,
                RevenueStream.payment_status == 'completed'
            )
        ).all()

        # Calculate totals
        total_revenue = sum(stream.amount for stream in revenue_streams)
        total_cost = sum(stream.cost_of_revenue for stream in revenue_streams)
        total_profit = total_revenue - total_cost

        # Group by stream type
        revenue_by_type = {}
        for stream in revenue_streams:
            stream_type = stream.stream_type
            if stream_type not in revenue_by_type:
                revenue_by_type[stream_type] = {
                    'amount': 0,
                    'count': 0,
                    'avg_amount': 0
                }
            revenue_by_type[stream_type]['amount'] += stream.amount
            revenue_by_type[stream_type]['count'] += 1

        # Calculate averages
        for stream_type in revenue_by_type:
            count = revenue_by_type[stream_type]['count']
            revenue_by_type[stream_type]['avg_amount'] = (
                revenue_by_type[stream_type]['amount'] / count if count > 0 else 0
            )

        # Calculate growth rates
        previous_period_start = start_date - (end_date - start_date)
        previous_revenue = self._get_revenue_for_period(
            organization_id, previous_period_start, start_date, db_session
        )
        revenue_growth = (
            ((total_revenue - previous_revenue) / previous_revenue * 100)
            if previous_revenue > 0 else 0
        )

        return {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'profit_margin': (total_profit / total_revenue * 100) if total_revenue > 0 else 0,
            'revenue_by_type': revenue_by_type,
            'revenue_growth_percent': revenue_growth,
            'average_revenue_per_stream': (
                total_revenue / len(revenue_streams) if revenue_streams else 0
            ),
            'currency': 'USD'  # Default currency
        }

    def calculate_customer_metrics(self, organization_id: int, start_date: datetime,
                                 end_date: datetime, db_session: Session) -> Dict[str, Any]:
        """Calculate customer acquisition and retention metrics."""

        # Get customer journeys
        customer_journeys = db_session.query(CustomerJourney).filter(
            and_(
                CustomerJourney.organization_id == organization_id,
                CustomerJourney.created_at >= start_date,
                CustomerJourney.created_at <= end_date
            )
        ).all()

        # Calculate acquisition metrics
        new_customers = len(set(j.customer_id for j in customer_journeys
                               if j.journey_stage == 'purchase'))

        # Calculate retention metrics
        total_customers = db_session.query(func.count(func.distinct(CustomerJourney.customer_id))).filter(
            CustomerJourney.organization_id == organization_id
        ).scalar()

        # Calculate churn (customers who haven't been active recently)
        churn_threshold = end_date - timedelta(days=30)
        active_customers = db_session.query(func.count(func.distinct(CustomerJourney.customer_id))).filter(
            and_(
                CustomerJourney.organization_id == organization_id,
                CustomerJourney.created_at >= churn_threshold
            )
        ).scalar()

        churned_customers = total_customers - active_customers
        churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0

        # Calculate customer segments
        segment_distribution = {}
        for journey in customer_journeys:
            segment = journey.segment or 'unknown'
            segment_distribution[segment] = segment_distribution.get(segment, 0) + 1

        # Calculate customer lifetime value
        avg_lifetime_value = self._calculate_average_clv(
            organization_id, db_session
        )

        return {
            'new_customers': new_customers,
            'total_customers': total_customers,
            'active_customers': active_customers,
            'churned_customers': churned_customers,
            'churn_rate_percent': churn_rate,
            'segment_distribution': segment_distribution,
            'average_lifetime_value': avg_lifetime_value,
            'customer_acquisition_rate': (
                new_customers / ((end_date - start_date).days / 30) if (end_date - start_date).days > 0 else 0
            )
        }

    def calculate_kpis(self, organization_id: int, period_start: datetime,
                      period_end: datetime, db_session: Session) -> Dict[str, Any]:
        """Calculate key performance indicators."""

        kpis = {}

        # Get existing KPIs or calculate new ones
        existing_kpis = db_session.query(BusinessKPI).filter(
            and_(
                BusinessKPI.organization_id == organization_id,
                BusinessKPI.period_start == period_start,
                BusinessKPI.period_end == period_end
            )
        ).all()

        # If KPIs already exist, return them
        if existing_kpis:
            for kpi in existing_kpis:
                kpis[kpi.kpi_name] = {
                    'current_value': kpi.current_value,
                    'target_value': kpi.target_value,
                    'change_percent': kpi.change_percent,
                    'performance_status': kpi.performance_status
                }
            return kpis

        # Calculate KPIs based on configuration
        for kpi_name, kpi_config in self.config.KPI_DEFINITIONS.items():
            kpi_value = self._calculate_kpi_value(
                kpi_name, kpi_config, organization_id, period_start, period_end, db_session
            )

            kpis[kpi_name] = {
                'current_value': kpi_value,
                'target_value': kpi_config.get('target', 0),
                'change_percent': 0,  # Would need previous period calculation
                'performance_status': self._determine_kpi_status(kpi_value, kpi_config.get('target', 0))
            }

        # Store calculated KPIs
        self._store_kpis(kpis, organization_id, period_start, period_end, db_session)

        return kpis

    def analyze_usage_patterns(self, organization_id: int, start_date: datetime,
                             end_date: datetime, db_session: Session) -> Dict[str, Any]:
        """Analyze usage patterns and detect anomalies."""

        # Get usage metrics
        usage_metrics = db_session.query(UsageMetric).filter(
            and_(
                UsageMetric.organization_id == organization_id,
                UsageMetric.timestamp >= start_date,
                UsageMetric.timestamp <= end_date
            )
        ).all()

        # Convert to time series data
        time_series_data = []
        for metric in usage_metrics:
            time_series_data.append({
                'timestamp': metric.timestamp,
                'requests': metric.request_count,
                'errors': metric.error_count,
                'response_time': metric.avg_response_time
            })

        # Detect patterns
        patterns = self._detect_patterns(time_series_data)

        # Detect anomalies
        anomalies = self._detect_anomalies(time_series_data)

        # Calculate pattern statistics
        pattern_stats = {
            'total_patterns_detected': len(patterns),
            'anomaly_count': len(anomalies),
            'average_requests_per_hour': np.mean([d['requests'] for d in time_series_data]),
            'peak_usage_hour': self._find_peak_usage_hour(time_series_data),
            'error_rate_trend': self._calculate_error_rate_trend(time_series_data)
        }

        return {
            'patterns': patterns,
            'anomalies': anomalies,
            'statistics': pattern_stats,
            'recommendations': self._generate_usage_recommendations(patterns, anomalies)
        }

    def generate_financial_forecast(self, organization_id: int, forecast_months: int,
                                  db_session: Session) -> Dict[str, Any]:
        """Generate financial forecasts using historical data."""

        # Get historical revenue data
        historical_data = self._get_historical_revenue_data(organization_id, 12, db_session)

        if not historical_data:
            return {
                'error': 'Insufficient historical data for forecasting',
                'forecast': None
            }

        # Prepare data for forecasting
        X = np.array(range(len(historical_data))).reshape(-1, 1)
        y = np.array([data['revenue'] for data in historical_data])

        # Simple linear regression for forecasting
        model = LinearRegression()
        model.fit(X, y)

        # Generate forecast
        forecast_periods = np.array(range(len(historical_data),
                                        len(historical_data) + forecast_months)).reshape(-1, 1)
        forecast_values = model.predict(forecast_periods)

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            model, X, y, forecast_periods
        )

        # Create forecast data structure
        forecast_data = []
        base_date = datetime.utcnow().replace(day=1)

        for i, value in enumerate(forecast_values):
            forecast_date = base_date + timedelta(days=30 * i)
            forecast_data.append({
                'period': forecast_date.strftime('%Y-%m'),
                'forecasted_revenue': float(value),
                'confidence_interval_low': confidence_intervals[i][0],
                'confidence_interval_high': confidence_intervals[i][1]
            })

        # Calculate forecast accuracy metrics
        accuracy_metrics = self._calculate_forecast_accuracy(model, X, y)

        return {
            'forecast_method': 'linear_regression',
            'historical_periods': len(historical_data),
            'forecast_periods': forecast_months,
            'forecast_data': forecast_data,
            'accuracy_metrics': accuracy_metrics,
            'model_confidence': model.score(X, y)
        }

    def generate_business_insights(self, organization_id: int, db_session: Session) -> List[Dict[str, Any]]:
        """Generate AI-powered business insights and recommendations."""

        insights = []

        # Revenue insights
        revenue_insights = self._generate_revenue_insights(organization_id, db_session)
        insights.extend(revenue_insights)

        # Customer insights
        customer_insights = self._generate_customer_insights(organization_id, db_session)
        insights.extend(customer_insights)

        # Operational insights
        operational_insights = self._generate_operational_insights(organization_id, db_session)
        insights.extend(operational_insights)

        # Growth insights
        growth_insights = self._generate_growth_insights(organization_id, db_session)
        insights.extend(growth_insights)

        # Sort by impact score and confidence
        insights.sort(key=lambda x: (x['impact_score'], x['confidence_score']), reverse=True)

        # Store insights in database
        self._store_business_insights(insights, organization_id, db_session)

        return insights

    def _get_revenue_for_period(self, organization_id: int, start_date: datetime,
                               end_date: datetime, db_session: Session) -> float:
        """Get total revenue for a specific period."""
        result = db_session.query(func.sum(RevenueStream.amount)).filter(
            and_(
                RevenueStream.organization_id == organization_id,
                RevenueStream.recognized_at >= start_date,
                RevenueStream.recognized_at <= end_date,
                RevenueStream.payment_status == 'completed'
            )
        ).scalar()
        return result or 0.0

    def _calculate_average_clv(self, organization_id: int, db_session: Session) -> float:
        """Calculate average customer lifetime value."""
        # This is a simplified CLV calculation
        # In a real implementation, you'd use more sophisticated models
        avg_monthly_revenue = db_session.query(
            func.avg(RevenueStream.amount)
        ).filter(
            RevenueStream.organization_id == organization_id
        ).scalar() or 0.0

        avg_customer_lifespan = 24  # months (assumed)
        avg_clv = avg_monthly_revenue * avg_customer_lifespan

        return avg_clv

    def _calculate_kpi_value(self, kpi_name: str, kpi_config: Dict[str, Any],
                           organization_id: int, period_start: datetime,
                           period_end: datetime, db_session: Session) -> float:
        """Calculate the value for a specific KPI."""
        # This is a simplified implementation
        # In a real system, you'd have more sophisticated KPI calculation logic

        if kpi_name == 'revenue':
            return self._get_revenue_for_period(organization_id, period_start, period_end, db_session)

        elif kpi_name == 'profit_margin':
            revenue = self._get_revenue_for_period(organization_id, period_start, period_end, db_session)
            # Simplified cost calculation
            cost = revenue * 0.3  # Assume 30% cost
            profit = revenue - cost
            return (profit / revenue * 100) if revenue > 0 else 0

        elif kpi_name == 'customer_acquisition_cost':
            # Simplified calculation
            marketing_cost = 10000  # Would come from actual data
            new_customers = 50  # Would come from actual data
            return marketing_cost / new_customers if new_customers > 0 else 0

        elif kpi_name == 'system_uptime':
            # Simplified calculation - would come from monitoring data
            return 99.9

        elif kpi_name == 'error_rate':
            # Simplified calculation - would come from actual error data
            return 1.0

        else:
            return 0.0

    def _determine_kpi_status(self, current_value: float, target_value: float) -> str:
        """Determine KPI performance status."""
        if current_value >= target_value:
            return 'exceeded'
        elif current_value >= target_value * 0.9:
            return 'on_track'
        elif current_value >= target_value * 0.7:
            return 'at_risk'
        else:
            return 'off_track'

    def _store_kpis(self, kpis: Dict[str, Any], organization_id: int,
                   period_start: datetime, period_end: datetime, db_session: Session):
        """Store calculated KPIs in the database."""
        for kpi_name, kpi_data in kpis.items():
            kpi = BusinessKPI(
                kpi_name=kpi_name,
                kpi_category=self.config.KPI_DEFINITIONS[kpi_name].get('category', 'operational'),
                kpi_type=self.config.KPI_DEFINITIONS[kpi_name].get('type', 'metric'),
                current_value=kpi_data['current_value'],
                target_value=kpi_data['target_value'],
                performance_status=kpi_data['performance_status'],
                period_start=period_start,
                period_end=period_end,
                period_type='monthly',
                organization_id=organization_id
            )
            db_session.add(kpi)

        db_session.commit()

    def _detect_patterns(self, time_series_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in time series data."""
        patterns = []

        if len(time_series_data) < 7:
            return patterns

        # Convert to DataFrame for analysis
        df = pd.DataFrame(time_series_data)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Detect daily patterns
        daily_pattern = df.groupby('hour')['requests'].agg(['mean', 'std']).to_dict()
        if daily_pattern['std']['mean'] > 0:
            patterns.append({
                'type': 'daily_pattern',
                'description': 'Daily usage pattern detected',
                'strength': 0.7,
                'peak_hours': [h for h in range(24) if daily_pattern['mean'][h] > np.mean(list(daily_pattern['mean'].values()))]
            })

        # Detect weekly patterns
        weekly_pattern = df.groupby('day_of_week')['requests'].agg(['mean', 'std']).to_dict()
        if weekly_pattern['std']['mean'] > 0:
            patterns.append({
                'type': 'weekly_pattern',
                'description': 'Weekly usage pattern detected',
                'strength': 0.6,
                'peak_days': [d for d in range(7) if weekly_pattern['mean'][d] > np.mean(list(weekly_pattern['mean'].values()))]
            })

        return patterns

    def _detect_anomalies(self, time_series_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in time series data."""
        anomalies = []

        if len(time_series_data) < 10:
            return anomalies

        # Simple anomaly detection based on standard deviation
        requests = np.array([d['requests'] for d in time_series_data])
        mean_requests = np.mean(requests)
        std_requests = np.std(requests)

        for i, data_point in enumerate(time_series_data):
            if abs(data_point['requests'] - mean_requests) > 2 * std_requests:
                anomalies.append({
                    'timestamp': data_point['timestamp'],
                    'metric': 'requests',
                    'value': data_point['requests'],
                    'expected_range': f'{mean_requests - 2*std_requests:.2f} - {mean_requests + 2*std_requests:.2f}',
                    'severity': 'high' if abs(data_point['requests'] - mean_requests) > 3 * std_requests else 'medium'
                })

        return anomalies

    def _find_peak_usage_hour(self, time_series_data: List[Dict[str, Any]]) -> int:
        """Find the hour with peak usage."""
        if not time_series_data:
            return 0

        df = pd.DataFrame(time_series_data)
        hourly_avg = df.groupby(df['timestamp'].dt.hour)['requests'].mean()
        return hourly_avg.idxmax()

    def _calculate_error_rate_trend(self, time_series_data: List[Dict[str, Any]]) -> str:
        """Calculate error rate trend."""
        if len(time_series_data) < 2:
            return 'stable'

        errors = [d['errors'] for d in time_series_data]
        requests = [d['requests'] for d in time_series_data]
        error_rates = [errors[i] / requests[i] if requests[i] > 0 else 0 for i in range(len(errors))]

        # Simple trend analysis
        if error_rates[-1] > error_rates[0] * 1.1:
            return 'increasing'
        elif error_rates[-1] < error_rates[0] * 0.9:
            return 'decreasing'
        else:
            return 'stable'

    def _generate_usage_recommendations(self, patterns: List[Dict[str, Any]],
                                      anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate usage pattern recommendations."""
        recommendations = []

        if len(anomalies) > 0:
            recommendations.append("Multiple anomalies detected. Consider investigating system stability.")

        for pattern in patterns:
            if pattern['type'] == 'daily_pattern' and len(pattern['peak_hours']) > 0:
                recommendations.append(f"Peak usage hours identified: {pattern['peak_hours']}. Consider scaling resources during these hours.")

        return recommendations

    def _get_historical_revenue_data(self, organization_id: int, months: int,
                                   db_session: Session) -> List[Dict[str, Any]]:
        """Get historical revenue data for forecasting."""
        historical_data = []

        for i in range(months):
            period_start = datetime.utcnow().replace(day=1) - timedelta(days=30 * i)
            period_end = period_start + timedelta(days=30)

            revenue = self._get_revenue_for_period(
                organization_id, period_start, period_end, db_session
            )

            historical_data.append({
                'period': period_start.strftime('%Y-%m'),
                'revenue': revenue
            })

        return list(reversed(historical_data))  # Return in chronological order

    def _calculate_confidence_intervals(self, model, X, y, forecast_periods) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for forecast."""
        # Simplified confidence interval calculation
        predictions = model.predict(forecast_periods)
        std_error = np.sqrt(np.mean((model.predict(X) - y) ** 2))

        confidence_intervals = []
        for pred in predictions:
            confidence_intervals.append((
                pred - 1.96 * std_error,
                pred + 1.96 * std_error
            ))

        return confidence_intervals

    def _calculate_forecast_accuracy(self, model, X, y) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        predictions = model.predict(X)

        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(np.mean((predictions - y) ** 2))
        mape = np.mean(np.abs((predictions - y) / y)) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    def _generate_revenue_insights(self, organization_id: int, db_session: Session) -> List[Dict[str, Any]]:
        """Generate revenue-related insights."""
        insights = []

        # Revenue growth insight
        current_month_revenue = self._get_revenue_for_period(
            organization_id, datetime.utcnow().replace(day=1),
            datetime.utcnow(), db_session
        )

        previous_month_revenue = self._get_revenue_for_period(
            organization_id,
            datetime.utcnow().replace(day=1) - timedelta(days=30),
            datetime.utcnow().replace(day=1), db_session
        )

        if previous_month_revenue > 0:
            growth_rate = ((current_month_revenue - previous_month_revenue) / previous_month_revenue) * 100

            if growth_rate > 20:
                insights.append({
                    'title': 'Strong Revenue Growth',
                    'type': 'opportunity',
                    'category': 'revenue',
                    'description': f'Revenue grew by {growth_rate:.1f}% this month',
                    'confidence_score': 0.9,
                    'impact_score': 0.8,
                    'recommendations': ['Consider increasing marketing spend', 'Expand product offerings']
                })

        return insights

    def _generate_customer_insights(self, organization_id: int, db_session: Session) -> List[Dict[str, Any]]:
        """Generate customer-related insights."""
        insights = []

        # Customer churn insight
        churn_rate = self.calculate_customer_metrics(
            organization_id, datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(), db_session
        )['churn_rate_percent']

        if churn_rate > 5:
            insights.append({
                'title': 'High Customer Churn',
                'type': 'risk',
                'category': 'customer',
                'description': f'Customer churn rate is {churn_rate:.1f}%, above target of 5%',
                'confidence_score': 0.8,
                'impact_score': 0.9,
                'recommendations': ['Implement customer retention programs', 'Improve customer support', 'Conduct customer satisfaction surveys']
            })

        return insights

    def _generate_operational_insights(self, organization_id: int, db_session: Session) -> List[Dict[str, Any]]:
        """Generate operational insights."""
        insights = []

        # System performance insight
        error_rate = self.calculate_kpis(
            organization_id, datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(), db_session
        ).get('error_rate', {}).get('current_value', 0)

        if error_rate > 3:
            insights.append({
                'title': 'System Performance Issues',
                'type': 'risk',
                'category': 'operations',
                'description': f'Error rate is {error_rate:.1f}%, above target of 3%',
                'confidence_score': 0.9,
                'impact_score': 0.7,
                'recommendations': ['Investigate error sources', 'Improve system monitoring', 'Implement automated error recovery']
            })

        return insights

    def _generate_growth_insights(self, organization_id: int, db_session: Session) -> List[Dict[str, Any]]:
        """Generate growth-related insights."""
        insights = []

        # Market opportunity insight
        customer_metrics = self.calculate_customer_metrics(
            organization_id, datetime.utcnow() - timedelta(days=90),
            datetime.utcnow(), db_session
        )

        if customer_metrics['new_customers'] > 100:
            insights.append({
                'title': 'Strong Market Demand',
                'type': 'opportunity',
                'category': 'growth',
                'description': f'Acquired {customer_metrics["new_customers"]} new customers in the last 90 days',
                'confidence_score': 0.8,
                'impact_score': 0.8,
                'recommendations': ['Expand sales team', 'Increase marketing budget', 'Develop new product features']
            })

        return insights

    def _store_business_insights(self, insights: List[Dict[str, Any]],
                               organization_id: int, db_session: Session):
        """Store generated insights in the database."""
        for insight_data in insights:
            insight = BusinessInsight(
                insight_title=insight_data['title'],
                insight_type=insight_data['type'],
                insight_category=insight_data['category'],
                insight_description=insight_data['description'],
                confidence_score=insight_data['confidence_score'],
                impact_score=insight_data['impact_score'],
                recommendations=insight_data['recommendations'],
                time_period_start=datetime.utcnow() - timedelta(days=30),
                time_period_end=datetime.utcnow(),
                organization_id=organization_id,
                priority='high' if insight_data['impact_score'] > 0.8 else 'medium'
            )
            db_session.add(insight)

        db_session.commit()


# Global BI service instance
bi_service = BusinessIntelligenceService()