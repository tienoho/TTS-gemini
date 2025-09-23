"""
Dashboard Components for Analytics System
Provides chart components, real-time updates, filters, and export functionality
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from collections import defaultdict
import threading
import time

from flask import Response, request, jsonify
from flask_socketio import emit, join_room, leave_room
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import pandas as pd

from utils.analytics_service import analytics_service
from utils.metrics_collector import metrics_collector


class DashboardComponents:
    """Dashboard components service for analytics visualization."""

    def __init__(self):
        self._active_connections = set()
        self._real_time_data = {}
        self._chart_cache = {}
        self._export_formats = ['json', 'csv', 'excel', 'pdf', 'png']
        self._update_intervals = {
            'real_time': 1,  # 1 second
            'fast': 5,       # 5 seconds
            'normal': 30,    # 30 seconds
            'slow': 300      # 5 minutes
        }
        self._lock = threading.Lock()

    # ===== CHART AND VISUALIZATION COMPONENTS =====

    def create_usage_chart(self, data: Dict[str, Any], chart_type: str = 'line',
                          time_range: str = '24h') -> Dict[str, Any]:
        """Create usage analytics chart."""
        try:
            # Extract data for chart
            if 'historical_trends' in data and 'data_points' in data['historical_trends']:
                chart_data = data['historical_trends']['data_points']
            else:
                chart_data = []

            if not chart_data:
                return self._create_empty_chart('Usage Analytics', 'No data available')

            # Create DataFrame for easier manipulation
            df = pd.DataFrame(chart_data)

            # Create chart based on type
            if chart_type == 'line':
                fig = self._create_line_chart(df, 'Usage Over Time', 'period', 'requests')
            elif chart_type == 'bar':
                fig = self._create_bar_chart(df, 'Usage by Period', 'period', 'requests')
            elif chart_type == 'area':
                fig = self._create_area_chart(df, 'Usage Area Chart', 'period', 'requests')
            else:
                fig = self._create_line_chart(df, 'Usage Over Time', 'period', 'requests')

            # Convert to JSON-serializable format
            chart_json = json.loads(fig.to_json())

            return {
                'chart_type': chart_type,
                'title': 'Usage Analytics',
                'data': chart_json,
                'config': {
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False
                },
                'metadata': {
                    'time_range': time_range,
                    'data_points': len(chart_data),
                    'last_updated': datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            return self._create_error_chart(f'Error creating usage chart: {str(e)}')

    def create_performance_chart(self, data: Dict[str, Any], metric: str = 'cpu_usage',
                               chart_type: str = 'line') -> Dict[str, Any]:
        """Create performance analytics chart."""
        try:
            # Extract performance data
            if 'historical_trends' in data and 'data_points' in data['historical_trends']:
                chart_data = data['historical_trends']['data_points']
            else:
                chart_data = []

            if not chart_data:
                return self._create_empty_chart('Performance Analytics', 'No data available')

            # Create DataFrame
            df = pd.DataFrame(chart_data)

            # Map metric names to display names
            metric_labels = {
                'cpu_usage': 'CPU Usage (%)',
                'memory_usage': 'Memory Usage (MB)',
                'response_time': 'Response Time (s)',
                'error_rate': 'Error Rate (%)'
            }

            display_name = metric_labels.get(metric, metric.replace('_', ' ').title())

            # Create chart
            if chart_type == 'line':
                fig = self._create_line_chart(df, f'Performance: {display_name}', 'period', metric)
            elif chart_type == 'gauge':
                fig = self._create_gauge_chart(data.get('current_status', {}).get(metric, 0), display_name)
            else:
                fig = self._create_line_chart(df, f'Performance: {display_name}', 'period', metric)

            chart_json = json.loads(fig.to_json())

            return {
                'chart_type': chart_type,
                'title': f'Performance: {display_name}',
                'data': chart_json,
                'config': {
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False
                },
                'metadata': {
                    'metric': metric,
                    'data_points': len(chart_data),
                    'last_updated': datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            return self._create_error_chart(f'Error creating performance chart: {str(e)}')

    def create_business_chart(self, data: Dict[str, Any], chart_type: str = 'line') -> Dict[str, Any]:
        """Create business analytics chart."""
        try:
            # Extract business data
            if 'historical_trends' in data and 'data_points' in data['historical_trends']:
                chart_data = data['historical_trends']['data_points']
            else:
                chart_data = []

            if not chart_data:
                return self._create_empty_chart('Business Analytics', 'No data available')

            # Create DataFrame
            df = pd.DataFrame(chart_data)

            # Create multi-line chart for business metrics
            fig = go.Figure()

            # Add revenue line
            if 'revenue' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['period'],
                    y=df['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='green', width=2),
                    marker=dict(size=6)
                ))

            # Add profit line
            if 'profit' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['period'],
                    y=df['profit'],
                    mode='lines+markers',
                    name='Profit',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))

            # Add requests line
            if 'requests' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['period'],
                    y=df['requests'],
                    mode='lines+markers',
                    name='Requests',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6),
                    yaxis='y2'
                ))

            fig.update_layout(
                title='Business Analytics',
                xaxis_title='Period',
                yaxis_title='Amount ($)',
                yaxis2=dict(
                    title='Requests',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                template='plotly_white'
            )

            chart_json = json.loads(fig.to_json())

            return {
                'chart_type': 'multi_line',
                'title': 'Business Analytics',
                'data': chart_json,
                'config': {
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False
                },
                'metadata': {
                    'data_points': len(chart_data),
                    'last_updated': datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            return self._create_error_chart(f'Error creating business chart: {str(e)}')

    def create_user_behavior_chart(self, data: Dict[str, Any], chart_type: str = 'pie') -> Dict[str, Any]:
        """Create user behavior analytics chart."""
        try:
            # Extract user behavior data
            behavior_patterns = data.get('behavior_patterns', {})

            if not behavior_patterns:
                return self._create_empty_chart('User Behavior Analytics', 'No data available')

            # Create device breakdown pie chart
            if 'device_breakdown' in behavior_patterns and chart_type == 'pie':
                device_data = behavior_patterns['device_breakdown']

                fig = px.pie(
                    values=list(device_data.values()),
                    names=list(device_data.keys()),
                    title='Device Type Distribution'
                )

            # Create geographic distribution map
            elif 'geographic_distribution' in behavior_patterns and chart_type == 'map':
                geo_data = behavior_patterns['geographic_distribution']

                # Create simple bar chart for geographic data
                countries = list(geo_data.keys())
                users = list(geo_data.values())

                fig = go.Figure(data=[
                    go.Bar(
                        x=countries,
                        y=users,
                        marker_color='lightblue'
                    )
                ])

                fig.update_layout(
                    title='Geographic Distribution',
                    xaxis_title='Country',
                    yaxis_title='Users'
                )

            # Create session timeline
            elif 'session_timeline' in behavior_patterns and chart_type == 'line':
                session_data = behavior_patterns['session_timeline']

                fig = go.Figure(data=[
                    go.Scatter(
                        x=list(session_data.keys()),
                        y=list(session_data.values()),
                        mode='lines+markers',
                        line=dict(color='purple', width=2),
                        marker=dict(size=6)
                    )
                ])

                fig.update_layout(
                    title='Session Timeline',
                    xaxis_title='Hour',
                    yaxis_title='Sessions'
                )

            else:
                return self._create_empty_chart('User Behavior Analytics', 'No suitable data for chart type')

            chart_json = json.loads(fig.to_json())

            return {
                'chart_type': chart_type,
                'title': 'User Behavior Analytics',
                'data': chart_json,
                'config': {
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False
                },
                'metadata': {
                    'last_updated': datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            return self._create_error_chart(f'Error creating user behavior chart: {str(e)}')

    # ===== CHART CREATION HELPERS =====

    def _create_line_chart(self, df: pd.DataFrame, title: str, x_col: str, y_col: str) -> go.Figure:
        """Create a line chart."""
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template='plotly_white'
        )

        fig.update_layout(
            hovermode='x unified',
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )

        return fig

    def _create_bar_chart(self, df: pd.DataFrame, title: str, x_col: str, y_col: str) -> go.Figure:
        """Create a bar chart."""
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template='plotly_white'
        )

        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )

        return fig

    def _create_area_chart(self, df: pd.DataFrame, title: str, x_col: str, y_col: str) -> go.Figure:
        """Create an area chart."""
        fig = px.area(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template='plotly_white'
        )

        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )

        return fig

    def _create_gauge_chart(self, value: float, title: str) -> go.Figure:
        """Create a gauge chart."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))

        return fig

    def _create_empty_chart(self, title: str, message: str) -> Dict[str, Any]:
        """Create an empty chart with message."""
        fig = go.Figure()

        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )

        fig.update_layout(
            title=title,
            template='plotly_white'
        )

        chart_json = json.loads(fig.to_json())

        return {
            'chart_type': 'empty',
            'title': title,
            'data': chart_json,
            'message': message,
            'metadata': {
                'last_updated': datetime.utcnow().isoformat()
            }
        }

    def _create_error_chart(self, error_message: str) -> Dict[str, Any]:
        """Create an error chart."""
        return self._create_empty_chart('Chart Error', error_message)

    # ===== REAL-TIME DATA UPDATES =====

    def start_real_time_updates(self, room: str, update_type: str = 'normal'):
        """Start real-time data updates for a room."""
        try:
            interval = self._update_intervals.get(update_type, 30)

            # Join the room
            join_room(room)

            # Start background task for updates
            self._start_update_task(room, interval)

            return {
                'status': 'started',
                'room': room,
                'update_interval': interval,
                'update_type': update_type
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def stop_real_time_updates(self, room: str):
        """Stop real-time data updates for a room."""
        try:
            leave_room(room)

            # Stop background task
            self._stop_update_task(room)

            return {
                'status': 'stopped',
                'room': room
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _start_update_task(self, room: str, interval: int):
        """Start background task for real-time updates."""
        def update_task():
            while room in self._active_connections:
                try:
                    # Get fresh data
                    data = self._get_real_time_data()

                    # Emit to room
                    emit('real_time_update', {
                        'room': room,
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    }, room=room)

                    # Wait for next update
                    time.sleep(interval)

                except Exception as e:
                    print(f"Error in real-time update for room {room}: {e}")
                    time.sleep(5)  # Wait before retrying

        # Start task in background thread
        thread = threading.Thread(target=update_task, daemon=True)
        thread.start()

        with self._lock:
            self._active_connections.add(room)

    def _stop_update_task(self, room: str):
        """Stop background task for real-time updates."""
        with self._lock:
            self._active_connections.discard(room)

    def _get_real_time_data(self) -> Dict[str, Any]:
        """Get real-time data for updates."""
        try:
            # Get real-time metrics from analytics service
            real_time_metrics = analytics_service.get_real_time_metrics(None)  # Using None for db_session

            # Get latest system metrics
            system_metrics = metrics_collector.get_metrics_summary(hours=1)

            return {
                'real_time': real_time_metrics,
                'system_summary': system_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    # ===== INTERACTIVE FILTERS =====

    def create_filter_component(self, filter_type: str, options: List[str],
                              default_value: Optional[str] = None) -> Dict[str, Any]:
        """Create an interactive filter component."""
        return {
            'filter_type': filter_type,
            'options': options,
            'default_value': default_value,
            'component_type': 'select',
            'config': {
                'multi': False,
                'searchable': True,
                'clearable': True
            }
        }

    def create_date_range_filter(self, default_days: int = 30) -> Dict[str, Any]:
        """Create a date range filter component."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=default_days)

        return {
            'filter_type': 'date_range',
            'default_start': start_date.isoformat(),
            'default_end': end_date.isoformat(),
            'component_type': 'date_range',
            'config': {
                'min_date': (end_date - timedelta(days=365)).isoformat(),
                'max_date': end_date.isoformat(),
                'format': 'YYYY-MM-DD'
            }
        }

    def apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to analytics data."""
        try:
            filtered_data = data.copy()

            # Apply date range filter
            if 'date_range' in filters:
                start_date = datetime.fromisoformat(filters['date_range']['start'])
                end_date = datetime.fromisoformat(filters['date_range']['end'])

                # Filter historical data points
                if 'historical_trends' in filtered_data and 'data_points' in filtered_data['historical_trends']:
                    original_points = filtered_data['historical_trends']['data_points']
                    filtered_points = [
                        point for point in original_points
                        if start_date <= datetime.fromisoformat(point['period']) <= end_date
                    ]
                    filtered_data['historical_trends']['data_points'] = filtered_points

            # Apply metric type filter
            if 'metric_type' in filters:
                # Filter performance metrics
                if 'performance_summary' in filtered_data:
                    metric_type = filters['metric_type']
                    if metric_type != 'all':
                        # Apply metric-specific filtering logic
                        pass

            # Apply organization filter
            if 'organization_id' in filters:
                org_id = filters['organization_id']
                # Filter data by organization
                if 'organization_breakdown' in filtered_data:
                    if org_id in filtered_data['organization_breakdown']:
                        filtered_data['organization_breakdown'] = {
                            org_id: filtered_data['organization_breakdown'][org_id]
                        }

            return filtered_data

        except Exception as e:
            return {
                'error': f'Error applying filters: {str(e)}',
                'original_data': data
            }

    # ===== EXPORT FUNCTIONALITY =====

    def export_data(self, data: Dict[str, Any], export_format: str = 'json',
                   filename: Optional[str] = None) -> Union[Dict[str, Any], Response]:
        """Export analytics data in specified format."""
        try:
            if export_format not in self._export_formats:
                return {
                    'error': f'Unsupported export format: {export_format}',
                    'supported_formats': self._export_formats
                }

            # Generate filename if not provided
            if not filename:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = f'analytics_export_{timestamp}'

            if export_format == 'json':
                return self._export_json(data, filename)
            elif export_format == 'csv':
                return self._export_csv(data, filename)
            elif export_format == 'excel':
                return self._export_excel(data, filename)
            elif export_format == 'pdf':
                return self._export_pdf(data, filename)
            elif export_format == 'png':
                return self._export_png(data, filename)
            else:
                return {'error': f'Export format {export_format} not implemented'}

        except Exception as e:
            return {
                'error': f'Export failed: {str(e)}'
            }

    def _export_json(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export data as JSON."""
        return {
            'filename': f'{filename}.json',
            'format': 'json',
            'data': data,
            'download_url': f'/api/analytics/export/{filename}.json'
        }

    def _export_csv(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export data as CSV."""
        try:
            # Convert data to CSV format
            csv_data = self._convert_to_csv(data)

            return {
                'filename': f'{filename}.csv',
                'format': 'csv',
                'data': csv_data,
                'download_url': f'/api/analytics/export/{filename}.csv'
            }

        except Exception as e:
            return {
                'error': f'CSV export failed: {str(e)}'
            }

    def _export_excel(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export data as Excel."""
        try:
            # Convert data to Excel format
            excel_data = self._convert_to_excel(data)

            return {
                'filename': f'{filename}.xlsx',
                'format': 'excel',
                'data': excel_data,
                'download_url': f'/api/analytics/export/{filename}.xlsx'
            }

        except Exception as e:
            return {
                'error': f'Excel export failed: {str(e)}'
            }

    def _export_pdf(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export data as PDF."""
        # This would require additional PDF generation library
        return {
            'error': 'PDF export not implemented yet'
        }

    def _export_png(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export chart as PNG."""
        # This would require additional image export functionality
        return {
            'error': 'PNG export not implemented yet'
        }

    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert analytics data to CSV format."""
        try:
            # Flatten the data structure for CSV
            rows = []

            # Add summary data
            if 'usage_summary' in data:
                rows.append(['Metric', 'Value'])
                for key, value in data['usage_summary'].items():
                    rows.append([key, value])

            # Add historical data
            if 'historical_trends' in data and 'data_points' in data['historical_trends']:
                rows.append([])
                rows.append(['Historical Data'])
                if data['historical_trends']['data_points']:
                    headers = list(data['historical_trends']['data_points'][0].keys())
                    rows.append(headers)

                    for point in data['historical_trends']['data_points']:
                        rows.append(list(point.values()))

            # Convert to CSV string
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            for row in rows:
                writer.writerow(row)

            return output.getvalue()

        except Exception as e:
            return f'Error converting to CSV: {str(e)}'

    def _convert_to_excel(self, data: Dict[str, Any]) -> bytes:
        """Convert analytics data to Excel format."""
        try:
            # This would use openpyxl or pandas to create Excel file
            # For now, return placeholder
            return b'Excel export not fully implemented'

        except Exception as e:
            return f'Error converting to Excel: {str(e)}'.encode()

    # ===== RESPONSIVE DESIGN =====

    def get_responsive_config(self, device_type: str = 'desktop') -> Dict[str, Any]:
        """Get responsive configuration for dashboard components."""
        configs = {
            'mobile': {
                'chart_height': 300,
                'font_size': 12,
                'show_legend': False,
                'simplified_ui': True
            },
            'tablet': {
                'chart_height': 400,
                'font_size': 14,
                'show_legend': True,
                'simplified_ui': False
            },
            'desktop': {
                'chart_height': 500,
                'font_size': 16,
                'show_legend': True,
                'simplified_ui': False
            }
        }

        return configs.get(device_type, configs['desktop'])

    def optimize_chart_for_device(self, chart_data: Dict[str, Any],
                                device_type: str = 'desktop') -> Dict[str, Any]:
        """Optimize chart configuration for specific device type."""
        try:
            config = self.get_responsive_config(device_type)

            # Update chart configuration
            if 'config' not in chart_data:
                chart_data['config'] = {}

            chart_data['config'].update({
                'responsive': True,
                'displayModeBar': not config['simplified_ui'],
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d'] if config['simplified_ui'] else []
            })

            # Update layout for mobile
            if device_type == 'mobile':
                chart_data['config']['displayModeBar'] = False
                chart_data['config']['staticPlot'] = False

            return chart_data

        except Exception as e:
            return chart_data

    # ===== CHART CACHING =====

    def get_cached_chart(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached chart data."""
        with self._lock:
            if cache_key in self._chart_cache:
                cached_item = self._chart_cache[cache_key]
                if datetime.utcnow() - cached_item['timestamp'] < timedelta(minutes=5):
                    return cached_item['data']
                else:
                    # Remove expired cache
                    del self._chart_cache[cache_key]

            return None

    def cache_chart(self, cache_key: str, chart_data: Dict[str, Any]):
        """Cache chart data."""
        with self._lock:
            self._chart_cache[cache_key] = {
                'data': chart_data,
                'timestamp': datetime.utcnow()
            }

    def clear_chart_cache(self):
        """Clear all cached charts."""
        with self._lock:
            self._chart_cache.clear()

    # ===== PUBLIC API =====

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about available dashboard components."""
        return {
            'chart_types': ['line', 'bar', 'area', 'pie', 'gauge', 'map', 'multi_line'],
            'export_formats': self._export_formats,
            'update_intervals': self._update_intervals,
            'filter_types': ['date_range', 'select', 'multi_select', 'text'],
            'real_time_enabled': True,
            'responsive_design': True,
            'caching_enabled': True
        }


# Global dashboard components instance
dashboard_components = DashboardComponents()