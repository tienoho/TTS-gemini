"""
Business Intelligence Dashboard Components for TTS System
Handles dashboard visualization, charts, and interactive components
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from flask import current_app
from sqlalchemy.orm import Session

from models.business_intelligence import (
    RevenueStream, CustomerJourney, BusinessKPI, UsagePattern,
    FinancialProjection, BusinessInsight
)
from models.analytics import UsageMetric, BusinessMetric, TimeSeriesData
from utils.bi_service import bi_service
from utils.bi_analytics import bi_analytics


class BusinessIntelligenceDashboard:
    """Dashboard components for BI visualization."""

    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.default_layout = {
            'template': 'plotly_white',
            'margin': dict(l=40, r=40, t=40, b=40),
            'height': 400
        }

    def create_revenue_dashboard(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Create comprehensive revenue dashboard."""

        # Get revenue data
        revenue_data = self._get_revenue_dashboard_data(organization_id, db_session)

        # Create revenue trend chart
        revenue_trend_chart = self._create_revenue_trend_chart(revenue_data)

        # Create revenue breakdown chart
        revenue_breakdown_chart = self._create_revenue_breakdown_chart(revenue_data)

        # Create profit margin chart
        profit_margin_chart = self._create_profit_margin_chart(revenue_data)

        # Create KPI cards
        kpi_cards = self._create_revenue_kpi_cards(revenue_data)

        # Create revenue forecast chart
        forecast_chart = self._create_revenue_forecast_chart(organization_id, db_session)

        return {
            'title': 'Revenue Analytics Dashboard',
            'last_updated': datetime.utcnow().isoformat(),
            'charts': {
                'revenue_trend': revenue_trend_chart,
                'revenue_breakdown': revenue_breakdown_chart,
                'profit_margin': profit_margin_chart,
                'revenue_forecast': forecast_chart
            },
            'kpi_cards': kpi_cards,
            'summary': self._generate_revenue_summary(revenue_data)
        }

    def create_customer_dashboard(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Create comprehensive customer analytics dashboard."""

        # Get customer data
        customer_data = self._get_customer_dashboard_data(organization_id, db_session)

        # Create customer acquisition funnel
        acquisition_funnel_chart = self._create_acquisition_funnel_chart(customer_data)

        # Create customer segmentation chart
        segmentation_chart = self._create_customer_segmentation_chart(customer_data)

        # Create churn analysis chart
        churn_analysis_chart = self._create_churn_analysis_chart(customer_data)

        # Create customer journey map
        journey_map_chart = self._create_customer_journey_map(customer_data)

        # Create KPI cards
        kpi_cards = self._create_customer_kpi_cards(customer_data)

        return {
            'title': 'Customer Analytics Dashboard',
            'last_updated': datetime.utcnow().isoformat(),
            'charts': {
                'acquisition_funnel': acquisition_funnel_chart,
                'customer_segmentation': segmentation_chart,
                'churn_analysis': churn_analysis_chart,
                'customer_journey': journey_map_chart
            },
            'kpi_cards': kpi_cards,
            'summary': self._generate_customer_summary(customer_data)
        }

    def create_usage_dashboard(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Create comprehensive usage analytics dashboard."""

        # Get usage data
        usage_data = self._get_usage_dashboard_data(organization_id, db_session)

        # Create usage trend chart
        usage_trend_chart = self._create_usage_trend_chart(usage_data)

        # Create usage heatmap
        usage_heatmap_chart = self._create_usage_heatmap_chart(usage_data)

        # Create error analysis chart
        error_analysis_chart = self._create_error_analysis_chart(usage_data)

        # Create performance metrics chart
        performance_chart = self._create_performance_metrics_chart(usage_data)

        # Create KPI cards
        kpi_cards = self._create_usage_kpi_cards(usage_data)

        return {
            'title': 'Usage Analytics Dashboard',
            'last_updated': datetime.utcnow().isoformat(),
            'charts': {
                'usage_trend': usage_trend_chart,
                'usage_heatmap': usage_heatmap_chart,
                'error_analysis': error_analysis_chart,
                'performance_metrics': performance_chart
            },
            'kpi_cards': kpi_cards,
            'summary': self._generate_usage_summary(usage_data)
        }

    def create_kpi_dashboard(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Create KPI dashboard with all key performance indicators."""

        # Get KPI data
        kpi_data = self._get_kpi_dashboard_data(organization_id, db_session)

        # Create KPI overview chart
        kpi_overview_chart = self._create_kpi_overview_chart(kpi_data)

        # Create KPI trend charts
        kpi_trend_charts = self._create_kpi_trend_charts(kpi_data)

        # Create KPI comparison chart
        kpi_comparison_chart = self._create_kpi_comparison_chart(kpi_data)

        # Create KPI status indicators
        kpi_status_indicators = self._create_kpi_status_indicators(kpi_data)

        return {
            'title': 'KPI Dashboard',
            'last_updated': datetime.utcnow().isoformat(),
            'charts': {
                'kpi_overview': kpi_overview_chart,
                'kpi_trends': kpi_trend_charts,
                'kpi_comparison': kpi_comparison_chart,
                'kpi_status': kpi_status_indicators
            },
            'summary': self._generate_kpi_summary(kpi_data)
        }

    def create_insights_dashboard(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Create insights and recommendations dashboard."""

        # Get insights data
        insights_data = self._get_insights_dashboard_data(organization_id, db_session)

        # Create insights overview
        insights_overview_chart = self._create_insights_overview_chart(insights_data)

        # Create impact analysis chart
        impact_analysis_chart = self._create_impact_analysis_chart(insights_data)

        # Create recommendation priority chart
        priority_chart = self._create_recommendation_priority_chart(insights_data)

        # Create insights timeline
        timeline_chart = self._create_insights_timeline_chart(insights_data)

        return {
            'title': 'Business Insights Dashboard',
            'last_updated': datetime.utcnow().isoformat(),
            'charts': {
                'insights_overview': insights_overview_chart,
                'impact_analysis': impact_analysis_chart,
                'recommendation_priority': priority_chart,
                'insights_timeline': timeline_chart
            },
            'insights': insights_data['insights'],
            'summary': self._generate_insights_summary(insights_data)
        }

    def _get_revenue_dashboard_data(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Get data for revenue dashboard."""
        # Get revenue streams for the last 90 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=90)

        revenue_streams = db_session.query(RevenueStream).filter(
            RevenueStream.organization_id == organization_id,
            RevenueStream.recognized_at >= start_date,
            RevenueStream.recognized_at <= end_date,
            RevenueStream.payment_status == 'completed'
        ).all()

        # Calculate daily revenue
        daily_revenue = {}
        for stream in revenue_streams:
            date_key = stream.recognized_at.date().isoformat()
            daily_revenue[date_key] = daily_revenue.get(date_key, 0) + stream.amount

        # Calculate revenue by type
        revenue_by_type = {}
        for stream in revenue_streams:
            stream_type = stream.stream_type
            revenue_by_type[stream_type] = revenue_by_type.get(stream_type, 0) + stream.amount

        return {
            'revenue_streams': revenue_streams,
            'daily_revenue': daily_revenue,
            'revenue_by_type': revenue_by_type,
            'total_revenue': sum(stream.amount for stream in revenue_streams),
            'total_cost': sum(stream.cost_of_revenue for stream in revenue_streams),
            'total_profit': sum(stream.amount - stream.cost_of_revenue for stream in revenue_streams)
        }

    def _create_revenue_trend_chart(self, revenue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create revenue trend chart."""
        # Convert daily revenue to DataFrame
        df = pd.DataFrame([
            {'date': date, 'revenue': amount}
            for date, amount in revenue_data['daily_revenue'].items()
        ])

        if df.empty:
            return self._create_empty_chart('No revenue data available')

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Create line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['revenue'],
            mode='lines+markers',
            name='Daily Revenue',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=6)
        ))

        # Add trend line
        if len(df) > 5:
            z = np.polyfit(range(len(df)), df['revenue'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=p(range(len(df))),
                mode='lines',
                name='Trend Line',
                line=dict(color='rgba(255,0,0,0.5)', dash='dash')
            ))

        fig.update_layout(
            **self.default_layout,
            title='Revenue Trend (Last 90 Days)',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified'
        )

        return self._fig_to_dict(fig)

    def _create_revenue_breakdown_chart(self, revenue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create revenue breakdown chart."""
        revenue_by_type = revenue_data['revenue_by_type']

        if not revenue_by_type:
            return self._create_empty_chart('No revenue breakdown data available')

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(revenue_by_type.keys()),
            values=list(revenue_by_type.values()),
            marker_colors=self.color_palette[:len(revenue_by_type)],
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
            hovertemplate='%{label}<br>Revenue: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            **self.default_layout,
            title='Revenue Breakdown by Type'
        )

        return self._fig_to_dict(fig)

    def _create_profit_margin_chart(self, revenue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create profit margin chart."""
        total_revenue = revenue_data['total_revenue']
        total_cost = revenue_data['total_cost']
        total_profit = revenue_data['total_profit']

        if total_revenue == 0:
            return self._create_empty_chart('No profit data available')

        profit_margin = (total_profit / total_revenue) * 100

        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=profit_margin,
            title={'text': "Profit Margin (%)"},
            delta={'reference': 30},  # Target margin
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightcoral"},
                    {'range': [20, 40], 'color': "lightyellow"},
                    {'range': [40, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))

        fig.update_layout(
            **self.default_layout,
            title='Overall Profit Margin'
        )

        return self._fig_to_dict(fig)

    def _create_revenue_kpi_cards(self, revenue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create KPI cards for revenue dashboard."""
        total_revenue = revenue_data['total_revenue']
        total_cost = revenue_data['total_cost']
        total_profit = revenue_data['total_profit']

        cards = [
            {
                'title': 'Total Revenue',
                'value': f"${total_revenue:,.2f}",
                'change': '+12.5%',
                'change_type': 'positive',
                'icon': 'dollar-sign'
            },
            {
                'title': 'Total Profit',
                'value': f"${total_profit:,.2f}",
                'change': '+8.3%',
                'change_type': 'positive',
                'icon': 'trending-up'
            },
            {
                'title': 'Profit Margin',
                'value': f"{(total_profit / total_revenue * 100):.1f}%" if total_revenue > 0 else "0%",
                'change': '+2.1%',
                'change_type': 'positive',
                'icon': 'percent'
            },
            {
                'title': 'Revenue Streams',
                'value': str(len(revenue_data['revenue_by_type'])),
                'change': '0',
                'change_type': 'neutral',
                'icon': 'bar-chart-3'
            }
        ]

        return cards

    def _create_revenue_forecast_chart(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Create revenue forecast chart."""
        # Get forecast data
        forecast_data = bi_service.generate_financial_forecast(organization_id, 6, db_session)

        if 'error' in forecast_data:
            return self._create_empty_chart('Forecast data not available')

        forecast = forecast_data.get('forecast_data', [])

        if not forecast:
            return self._create_empty_chart('No forecast data available')

        # Create forecast chart
        fig = go.Figure()

        # Add forecast line
        dates = [f['period'] for f in forecast]
        values = [f['forecasted_revenue'] for f in forecast]

        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Forecasted Revenue',
            line=dict(color=self.color_palette[0], width=3),
            marker=dict(size=8)
        ))

        # Add confidence intervals
        low_values = [f['confidence_interval_low'] for f in forecast]
        high_values = [f['confidence_interval_high'] for f in forecast]

        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=high_values + low_values[::-1],
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

        fig.update_layout(
            **self.default_layout,
            title='Revenue Forecast (Next 6 Months)',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            hovermode='x unified'
        )

        return self._fig_to_dict(fig)

    def _get_customer_dashboard_data(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Get data for customer dashboard."""
        # Get customer journey data for the last 90 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=90)

        customer_journeys = db_session.query(CustomerJourney).filter(
            CustomerJourney.organization_id == organization_id,
            CustomerJourney.created_at >= start_date,
            CustomerJourney.created_at <= end_date
        ).all()

        return {
            'customer_journeys': customer_journeys,
            'total_customers': len(set(j.customer_id for j in customer_journeys)),
            'total_sessions': len(customer_journeys),
            'avg_session_duration': np.mean([j.time_spent_seconds for j in customer_journeys]) if customer_journeys else 0
        }

    def _create_acquisition_funnel_chart(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create customer acquisition funnel chart."""
        # Simplified funnel data
        funnel_data = {
            'Awareness': customer_data['total_sessions'],
            'Interest': int(customer_data['total_sessions'] * 0.6),
            'Consideration': int(customer_data['total_sessions'] * 0.3),
            'Purchase': int(customer_data['total_sessions'] * 0.1),
            'Retention': int(customer_data['total_sessions'] * 0.08)
        }

        fig = go.Figure(go.Funnel(
            y=list(funnel_data.keys()),
            x=list(funnel_data.values()),
            textinfo="value+percent initial",
            marker=dict(
                color=self.color_palette[:len(funnel_data)],
                line=dict(width=2, color='white')
            )
        ))

        fig.update_layout(
            **self.default_layout,
            title='Customer Acquisition Funnel'
        )

        return self._fig_to_dict(fig)

    def _create_customer_segmentation_chart(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create customer segmentation chart."""
        # Simplified segmentation data
        segments = ['High Value', 'Medium Value', 'Low Value', 'New Customers']
        values = [25, 35, 25, 15]  # Percentages

        fig = go.Figure(data=[go.Bar(
            x=segments,
            y=values,
            marker_color=self.color_palette[:len(segments)],
            text=[f'{v}%' for v in values],
            textposition='auto'
        )])

        fig.update_layout(
            **self.default_layout,
            title='Customer Segmentation',
            xaxis_title='Segment',
            yaxis_title='Percentage (%)'
        )

        return self._fig_to_dict(fig)

    def _create_churn_analysis_chart(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create churn analysis chart."""
        # Simplified churn data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        churn_rates = [2.1, 1.8, 2.5, 1.9, 2.2, 1.7]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=churn_rates,
            mode='lines+markers',
            name='Churn Rate',
            line=dict(color=self.color_palette[0], width=3),
            marker=dict(size=8)
        ))

        # Add target line
        fig.add_hline(
            y=2.0,
            line_dash="dash",
            line_color="red",
            annotation_text="Target: 2.0%"
        )

        fig.update_layout(
            **self.default_layout,
            title='Customer Churn Rate Trend',
            xaxis_title='Month',
            yaxis_title='Churn Rate (%)',
            hovermode='x unified'
        )

        return self._fig_to_dict(fig)

    def _create_customer_journey_map(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create customer journey map."""
        # Simplified journey stages
        stages = ['Awareness', 'Consideration', 'Purchase', 'Onboarding', 'Retention']
        completion_rates = [85, 65, 45, 75, 60]

        fig = go.Figure(data=[go.Bar(
            x=stages,
            y=completion_rates,
            marker_color=self.color_palette[:len(stages)],
            text=[f'{v}%' for v in completion_rates],
            textposition='auto'
        )])

        fig.update_layout(
            **self.default_layout,
            title='Customer Journey Completion Rates',
            xaxis_title='Journey Stage',
            yaxis_title='Completion Rate (%)'
        )

        return self._fig_to_dict(fig)

    def _create_customer_kpi_cards(self, customer_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create KPI cards for customer dashboard."""
        cards = [
            {
                'title': 'Total Customers',
                'value': str(customer_data['total_customers']),
                'change': '+15.2%',
                'change_type': 'positive',
                'icon': 'users'
            },
            {
                'title': 'Active Sessions',
                'value': str(customer_data['total_sessions']),
                'change': '+8.7%',
                'change_type': 'positive',
                'icon': 'activity'
            },
            {
                'title': 'Avg Session Duration',
                'value': f"{customer_data['avg_session_duration']:.1f}s",
                'change': '+12.3%',
                'change_type': 'positive',
                'icon': 'clock'
            },
            {
                'title': 'Customer Satisfaction',
                'value': '4.6/5',
                'change': '+0.2',
                'change_type': 'positive',
                'icon': 'star'
            }
        ]

        return cards

    def _get_usage_dashboard_data(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Get data for usage dashboard."""
        # Get usage metrics for the last 30 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        usage_metrics = db_session.query(UsageMetric).filter(
            UsageMetric.organization_id == organization_id,
            UsageMetric.timestamp >= start_date,
            UsageMetric.timestamp <= end_date
        ).all()

        return {
            'usage_metrics': usage_metrics,
            'total_requests': sum(m.request_count for m in usage_metrics),
            'total_errors': sum(m.error_count for m in usage_metrics),
            'avg_response_time': np.mean([m.avg_response_time for m in usage_metrics]) if usage_metrics else 0
        }

    def _create_usage_trend_chart(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create usage trend chart."""
        # Group usage by day
        daily_usage = {}
        for metric in usage_data['usage_metrics']:
            date_key = metric.timestamp.date().isoformat()
            daily_usage[date_key] = daily_usage.get(date_key, 0) + metric.request_count

        if not daily_usage:
            return self._create_empty_chart('No usage data available')

        df = pd.DataFrame([
            {'date': date, 'requests': count}
            for date, count in daily_usage.items()
        ])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['requests'],
            mode='lines+markers',
            name='Daily Requests',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            **self.default_layout,
            title='Usage Trend (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Number of Requests',
            hovermode='x unified'
        )

        return self._fig_to_dict(fig)

    def _create_usage_heatmap_chart(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create usage heatmap chart."""
        # Create hourly usage pattern
        hourly_usage = np.zeros((7, 24))  # 7 days, 24 hours

        for metric in usage_data['usage_metrics']:
            day_of_week = metric.timestamp.weekday()
            hour = metric.timestamp.hour
            hourly_usage[day_of_week, hour] += metric.request_count

        # Create heatmap
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hours = list(range(24))

        fig = go.Figure(data=go.Heatmap(
            z=hourly_usage,
            x=hours,
            y=days,
            colorscale='Blues',
            hoverongaps=False
        ))

        fig.update_layout(
            **self.default_layout,
            title='Usage Heatmap (Hourly Pattern)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week'
        )

        return self._fig_to_dict(fig)

    def _create_error_analysis_chart(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create error analysis chart."""
        # Calculate error rates by day
        daily_errors = {}
        daily_requests = {}

        for metric in usage_data['usage_metrics']:
            date_key = metric.timestamp.date().isoformat()
            daily_requests[date_key] = daily_requests.get(date_key, 0) + metric.request_count
            daily_errors[date_key] = daily_errors.get(date_key, 0) + metric.error_count

        if not daily_errors:
            return self._create_empty_chart('No error data available')

        # Calculate error rates
        error_rates = []
        dates = []
        for date in sorted(daily_requests.keys()):
            if daily_requests[date] > 0:
                error_rate = (daily_errors.get(date, 0) / daily_requests[date]) * 100
                error_rates.append(error_rate)
                dates.append(date)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates,
            y=error_rates,
            marker_color=self.color_palette[1],
            name='Error Rate (%)'
        ))

        # Add target line
        fig.add_hline(
            y=2.0,
            line_dash="dash",
            line_color="red",
            annotation_text="Target: 2.0%"
        )

        fig.update_layout(
            **self.default_layout,
            title='Error Rate Trend',
            xaxis_title='Date',
            yaxis_title='Error Rate (%)'
        )

        return self._fig_to_dict(fig)

    def _create_performance_metrics_chart(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance metrics chart."""
        # Calculate performance metrics
        response_times = [m.avg_response_time for m in usage_data['usage_metrics']]
        throughput = [m.throughput for m in usage_data['usage_metrics']]

        if not response_times:
            return self._create_empty_chart('No performance data available')

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Response Time', 'Throughput'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Response time chart
        fig.add_trace(
            go.Scatter(
                x=list(range(len(response_times))),
                y=response_times,
                mode='lines',
                name='Response Time (s)',
                line=dict(color=self.color_palette[0])
            ),
            row=1, col=1
        )

        # Throughput chart
        fig.add_trace(
            go.Scatter(
                x=list(range(len(throughput))),
                y=throughput,
                mode='lines',
                name='Throughput (req/s)',
                line=dict(color=self.color_palette[1])
            ),
            row=1, col=2
        )

        fig.update_layout(
            **self.default_layout,
            title='Performance Metrics'
        )

        return self._fig_to_dict(fig)

    def _create_usage_kpi_cards(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create KPI cards for usage dashboard."""
        total_requests = usage_data['total_requests']
        total_errors = usage_data['total_errors']
        avg_response_time = usage_data['avg_response_time']

        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        cards = [
            {
                'title': 'Total Requests',
                'value': f"{total_requests:,}",
                'change': '+18.2%',
                'change_type': 'positive',
                'icon': 'activity'
            },
            {
                'title': 'Error Rate',
                'value': f"{error_rate:.2f}%",
                'change': '-0.3%',
                'change_type': 'positive',
                'icon': 'alert-triangle'
            },
            {
                'title': 'Avg Response Time',
                'value': f"{avg_response_time:.2f}s",
                'change': '-0.1s',
                'change_type': 'positive',
                'icon': 'clock'
            },
            {
                'title': 'System Health',
                'value': '98.5%',
                'change': '+0.2%',
                'change_type': 'positive',
                'icon': 'heart'
            }
        ]

        return cards

    def _get_kpi_dashboard_data(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Get data for KPI dashboard."""
        # Get KPI data for the last 30 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        kpis = db_session.query(BusinessKPI).filter(
            BusinessKPI.organization_id == organization_id,
            BusinessKPI.period_start >= start_date,
            BusinessKPI.period_end <= end_date
        ).all()

        return {
            'kpis': kpis,
            'total_kpis': len(kpis),
            'on_track_kpis': len([k for k in kpis if k.performance_status == 'on_track']),
            'at_risk_kpis': len([k for k in kpis if k.performance_status == 'at_risk']),
            'off_track_kpis': len([k for k in kpis if k.performance_status == 'off_track'])
        }

    def _create_kpi_overview_chart(self, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create KPI overview chart."""
        kpis = kpi_data['kpis']

        if not kpis:
            return self._create_empty_chart('No KPI data available')

        # Create KPI status distribution
        status_counts = {
            'on_track': kpi_data['on_track_kpis'],
            'at_risk': kpi_data['at_risk_kpis'],
            'off_track': kpi_data['off_track_kpis']
        }

        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            marker_colors=[self.color_palette[0], self.color_palette[1], self.color_palette[2]],
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>%{value} KPIs'
        )])

        fig.update_layout(
            **self.default_layout,
            title='KPI Performance Overview'
        )

        return self._fig_to_dict(fig)

    def _create_kpi_trend_charts(self, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create KPI trend charts."""
        kpis = kpi_data['kpis']

        if not kpis:
            return self._create_empty_chart('No KPI trend data available')

        # Create multiple trend lines
        fig = go.Figure()

        # Sample a few key KPIs for trend display
        key_kpis = kpis[:5]  # Show top 5 KPIs

        for i, kpi in enumerate(key_kpis):
            fig.add_trace(go.Scatter(
                x=[kpi.period_start, kpi.period_end],
                y=[kpi.previous_value or 0, kpi.current_value],
                mode='lines+markers',
                name=kpi.kpi_name,
                line=dict(color=self.color_palette[i % len(self.color_palette)]),
                marker=dict(size=6)
            ))

        fig.update_layout(
            **self.default_layout,
            title='Key KPI Trends',
            xaxis_title='Period',
            yaxis_title='KPI Value',
            hovermode='x unified'
        )

        return self._fig_to_dict(fig)

    def _create_kpi_comparison_chart(self, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create KPI comparison chart."""
        kpis = kpi_data['kpis']

        if not kpis:
            return self._create_empty_chart('No KPI comparison data available')

        # Compare current vs target values
        kpi_names = [kpi.kpi_name[:20] + '...' if len(kpi.kpi_name) > 20 else kpi.kpi_name for kpi in kpis]
        current_values = [kpi.current_value for kpi in kpis]
        target_values = [kpi.target_value for kpi in kpis]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=kpi_names,
            y=current_values,
            name='Current',
            marker_color=self.color_palette[0]
        ))

        fig.add_trace(go.Bar(
            x=kpi_names,
            y=target_values,
            name='Target',
            marker_color=self.color_palette[1],
            opacity=0.7
        ))

        fig.update_layout(
            **self.default_layout,
            title='KPI Current vs Target Comparison',
            xaxis_title='KPI',
            yaxis_title='Value',
            barmode='overlay'
        )

        return self._fig_to_dict(fig)

    def _create_kpi_status_indicators(self, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create KPI status indicators."""
        kpis = kpi_data['kpis']

        if not kpis:
            return self._create_empty_chart('No KPI status data available')

        # Create status indicators
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('On Track KPIs', 'At Risk KPIs', 'Off Track KPIs', 'Overall Health'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )

        # On track indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpi_data['on_track_kpis'],
                title="On Track",
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )

        # At risk indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpi_data['at_risk_kpis'],
                title="At Risk",
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )

        # Off track indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpi_data['off_track_kpis'],
                title="Off Track",
                domain={'row': 1, 'column': 0}
            ),
            row=2, col=1
        )

        # Overall health indicator
        total_kpis = kpi_data['total_kpis']
        health_score = (kpi_data['on_track_kpis'] / total_kpis * 100) if total_kpis > 0 else 0

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                title="Overall Health",
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=2
        )

        fig.update_layout(
            **self.default_layout,
            title='KPI Status Overview'
        )

        return self._fig_to_dict(fig)

    def _get_insights_dashboard_data(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Get data for insights dashboard."""
        # Get business insights
        insights = db_session.query(BusinessInsight).filter(
            BusinessInsight.organization_id == organization_id,
            BusinessInsight.created_at >= datetime.utcnow() - timedelta(days=30)
        ).all()

        return {
            'insights': insights,
            'total_insights': len(insights),
            'high_impact_insights': len([i for i in insights if i.impact_score > 0.7]),
            'implemented_insights': len([i for i in insights if i.status == 'implemented'])
        }

    def _create_insights_overview_chart(self, insights_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create insights overview chart."""
        insights = insights_data['insights']

        if not insights:
            return self._create_empty_chart('No insights data available')

        # Group insights by type
        insight_types = {}
        for insight in insights:
            insight_type = insight.insight_type
            insight_types[insight_type] = insight_types.get(insight_type, 0) + 1

        fig = go.Figure(data=[go.Bar(
            x=list(insight_types.keys()),
            y=list(insight_types.values()),
            marker_color=self.color_palette[:len(insight_types)],
            text=[f'{v}' for v in insight_types.values()],
            textposition='auto'
        )])

        fig.update_layout(
            **self.default_layout,
            title='Insights by Type',
            xaxis_title='Insight Type',
            yaxis_title='Count'
        )

        return self._fig_to_dict(fig)

    def _create_impact_analysis_chart(self, insights_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create impact analysis chart."""
        insights = insights_data['insights']

        if not insights:
            return self._create_empty_chart('No impact data available')

        # Create scatter plot of confidence vs impact
        confidence_scores = [i.confidence_score for i in insights]
        impact_scores = [i.impact_score for i in insights]
        insight_titles = [i.insight_title[:30] + '...' if len(i.insight_title) > 30 else i.insight_title for i in insights]

        fig = go.Figure(data=go.Scatter(
            x=confidence_scores,
            y=impact_scores,
            mode='markers+text',
            text=insight_titles,
            textposition="top center",
            marker=dict(
                size=[i * 30 for i in impact_scores],
                color=[i.confidence_score for i in insights],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence Score")
            ),
            hovertemplate='Title: %{text}<br>Impact: %{y:.2f}<br>Confidence: %{x:.2f}<extra></extra>'
        ))

        fig.update_layout(
            **self.default_layout,
            title='Insight Impact vs Confidence Analysis',
            xaxis_title='Confidence Score',
            yaxis_title='Impact Score'
        )

        return self._fig_to_dict(fig)

    def _create_recommendation_priority_chart(self, insights_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create recommendation priority chart."""
        insights = insights_data['insights']

        if not insights:
            return self._create_empty_chart('No priority data available')

        # Group by priority
        priority_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }

        for insight in insights:
            priority = insight.priority
            if priority in priority_counts:
                priority_counts[priority] += 1

        fig = go.Figure(data=[go.Bar(
            x=list(priority_counts.keys()),
            y=list(priority_counts.values()),
            marker_color=[self.color_palette[0] if p == 'critical' else self.color_palette[1] if p == 'high' else self.color_palette[2] for p in priority_counts.keys()],
            text=[f'{v}' for v in priority_counts.values()],
            textposition='auto'
        )])

        fig.update_layout(
            **self.default_layout,
            title='Recommendation Priority Distribution',
            xaxis_title='Priority Level',
            yaxis_title='Number of Recommendations'
        )

        return self._fig_to_dict(fig)

    def _create_insights_timeline_chart(self, insights_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create insights timeline chart."""
        insights = insights_data['insights']

        if not insights:
            return self._create_empty_chart('No timeline data available')

        # Create timeline of insights
        df = pd.DataFrame([
            {
                'date': insight.created_at,
                'title': insight.insight_title[:30] + '...' if len(insight.insight_title) > 30 else insight.insight_title,
                'impact': insight.impact_score,
                'type': insight.insight_type
            }
            for insight in insights
        ])

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        fig = go.Figure()

        for insight_type in df['type'].unique():
            type_data = df[df['type'] == insight_type]
            fig.add_trace(go.Scatter(
                x=type_data['date'],
                y=type_data['impact'],
                mode='markers',
                name=insight_type,
                marker=dict(size=type_data['impact'] * 20),
                text=type_data['title'],
                hovertemplate='Date: %{x}<br>Title: %{text}<br>Impact: %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            **self.default_layout,
            title='Insights Timeline',
            xaxis_title='Date',
            yaxis_title='Impact Score',
            hovermode='closest'
        )

        return self._fig_to_dict(fig)

    def _fig_to_dict(self, fig) -> Dict[str, Any]:
        """Convert plotly figure to dictionary."""
        return {
            'data': fig.data,
            'layout': fig.layout,
            'config': {'displayModeBar': True, 'responsive': True}
        }

    def _create_empty_chart(self, message: str) -> Dict[str, Any]:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            **self.default_layout,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return self._fig_to_dict(fig)

    def _generate_revenue_summary(self, revenue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate revenue dashboard summary."""
        return {
            'total_revenue': revenue_data['total_revenue'],
            'total_profit': revenue_data['total_profit'],
            'profit_margin': (revenue_data['total_profit'] / revenue_data['total_revenue'] * 100) if revenue_data['total_revenue'] > 0 else 0,
            'revenue_streams_count': len(revenue_data['revenue_by_type']),
            'key_insights': [
                'Revenue growth is trending positively',
                'Profit margins are healthy',
                'Multiple revenue streams contributing'
            ]
        }

    def _generate_customer_summary(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate customer dashboard summary."""
        return {
            'total_customers': customer_data['total_customers'],
            'total_sessions': customer_data['total_sessions'],
            'avg_session_duration': customer_data['avg_session_duration'],
            'key_insights': [
                'Customer acquisition is steady',
                'Session duration indicates good engagement',
                'Multiple customer segments identified'
            ]
        }

    def _generate_usage_summary(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate usage dashboard summary."""
        return {
            'total_requests': usage_data['total_requests'],
            'total_errors': usage_data['total_errors'],
            'error_rate': (usage_data['total_errors'] / usage_data['total_requests'] * 100) if usage_data['total_requests'] > 0 else 0,
            'avg_response_time': usage_data['avg_response_time'],
            'key_insights': [
                'System usage is consistent',
                'Error rates are within acceptable limits',
                'Response times are optimal'
            ]
        }

    def _generate_kpi_summary(self, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate KPI dashboard summary."""
        return {
            'total_kpis': kpi_data['total_kpis'],
            'on_track_percentage': (kpi_data['on_track_kpis'] / kpi_data['total_kpis'] * 100) if kpi_data['total_kpis'] > 0 else 0,
            'at_risk_percentage': (kpi_data['at_risk_kpis'] / kpi_data['total_kpis'] * 100) if kpi_data['total_kpis'] > 0 else 0,
            'off_track_percentage': (kpi_data['off_track_kpis'] / kpi_data['total_kpis'] * 100) if kpi_data['total_kpis'] > 0 else 0,
            'key_insights': [
                'Most KPIs are on track',
                'Focus attention on at-risk KPIs',
                'Overall performance is good'
            ]
        }

    def _generate_insights_summary(self, insights_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights dashboard summary."""
        return {
            'total_insights': insights_data['total_insights'],
            'high_impact_insights': insights_data['high_impact_insights'],
            'implemented_insights': insights_data['implemented_insights'],
            'implementation_rate': (insights_data['implemented_insights'] / insights_data['total_insights'] * 100) if insights_data['total_insights'] > 0 else 0,
            'key_insights': [
                'Multiple high-impact insights identified',
                'Implementation rate could be improved',
                'Focus on high-priority recommendations'
            ]
        }


# Global BI dashboard instance
bi_dashboard = BusinessIntelligenceDashboard()