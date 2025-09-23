"""
Metrics Collector for TTS System Analytics
Handles automatic metrics collection, custom metrics definition, and data aggregation
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
from functools import wraps

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.extensions import db
from models import SystemMetric
from models.analytics import UsageMetric, PerformanceMetric, UserBehavior
from utils.analytics_service import analytics_service


class MetricsCollector:
    """Advanced metrics collector for TTS system."""

    def __init__(self):
        self._collectors = {}
        self._custom_metrics = {}
        self._aggregation_pipelines = {}
        self._performance_monitors = {}
        self._error_trackers = {}
        self._collection_intervals = {
            'system_metrics': 60,  # Every minute
            'usage_metrics': 300,  # Every 5 minutes
            'performance_metrics': 60,  # Every minute
            'user_behavior': 600,  # Every 10 minutes
            'business_metrics': 1800,  # Every 30 minutes
        }
        self._is_running = False
        self._collection_threads = {}
        self._lock = threading.Lock()

    def start_collection(self):
        """Start all metrics collection processes."""
        if self._is_running:
            return

        self._is_running = True

        # Start system metrics collection
        self._start_collector('system_metrics', self._collect_system_metrics)

        # Start usage metrics collection
        self._start_collector('usage_metrics', self._collect_usage_metrics)

        # Start performance metrics collection
        self._start_collector('performance_metrics', self._collect_performance_metrics)

        # Start user behavior collection
        self._start_collector('user_behavior', self._collect_user_behavior_metrics)

        # Start business metrics collection
        self._start_collector('business_metrics', self._collect_business_metrics)

    def stop_collection(self):
        """Stop all metrics collection processes."""
        self._is_running = False

        for thread_name, thread in self._collection_threads.items():
            if thread.is_alive():
                thread.join(timeout=5)

        self._collection_threads.clear()

    def _start_collector(self, name: str, collector_func: Callable, interval: Optional[int] = None):
        """Start a specific metrics collector."""
        if interval is None:
            interval = self._collection_intervals.get(name, 300)

        thread = threading.Thread(
            target=self._collection_worker,
            args=(name, collector_func, interval),
            daemon=True,
            name=f"MetricsCollector-{name}"
        )
        thread.start()
        self._collection_threads[name] = thread

    def _collection_worker(self, name: str, collector_func: Callable, interval: int):
        """Worker function for metrics collection threads."""
        while self._is_running:
            try:
                with self._lock:
                    collector_func()

                # Sleep for the specified interval
                time.sleep(interval)

            except Exception as e:
                print(f"Error in {name} collector: {e}")
                time.sleep(60)  # Wait a minute before retrying

    # ===== SYSTEM METRICS COLLECTION =====

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._store_metric('cpu_usage', cpu_percent, 'gauge', 'system', unit='%')

            # Memory usage
            memory = psutil.virtual_memory()
            self._store_metric('memory_usage', memory.used, 'gauge', 'system', unit='bytes')
            self._store_metric('memory_percent', memory.percent, 'gauge', 'system', unit='%')

            # Disk usage
            disk = psutil.disk_usage('/')
            self._store_metric('disk_usage', disk.used, 'gauge', 'system', unit='bytes')
            self._store_metric('disk_percent', disk.percent, 'gauge', 'system', unit='%')

            # Network I/O
            network = psutil.net_io_counters()
            if network:
                self._store_metric('network_io', network.bytes_sent + network.bytes_recv, 'counter', 'system', unit='bytes')

            # System load
            load_avg = psutil.getloadavg()
            self._store_metric('load_average_1min', load_avg[0], 'gauge', 'system')
            self._store_metric('load_average_5min', load_avg[1], 'gauge', 'system')
            self._store_metric('load_average_15min', load_avg[2], 'gauge', 'system')

            # Process count
            process_count = len(psutil.pids())
            self._store_metric('process_count', process_count, 'gauge', 'system', unit='count')

            # Thread count
            total_threads = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
            self._store_metric('thread_count', total_threads, 'gauge', 'system', unit='count')

        except Exception as e:
            print(f"Error collecting system metrics: {e}")

    # ===== USAGE METRICS COLLECTION =====

    def _collect_usage_metrics(self):
        """Collect usage metrics from recent requests."""
        try:
            # Get recent usage data from the last 5 minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=5)

            # This would typically query your request logs or usage tracking tables
            # For now, we'll create sample data
            self._collect_api_usage_metrics(cutoff_time)
            self._collect_tts_usage_metrics(cutoff_time)
            self._collect_error_metrics(cutoff_time)

        except Exception as e:
            print(f"Error collecting usage metrics: {e}")

    def _collect_api_usage_metrics(self, cutoff_time: datetime):
        """Collect API usage metrics."""
        try:
            # Query recent requests (this is a simplified example)
            # In a real system, you would query your request logs
            request_count = 100  # Sample data
            error_count = 5  # Sample data
            avg_response_time = 0.85  # Sample data

            if request_count > 0:
                error_rate = (error_count / request_count) * 100
                throughput = request_count / 5.0  # requests per minute

                self._store_metric('api_requests', request_count, 'counter', 'api', unit='count')
                self._store_metric('api_errors', error_count, 'counter', 'api', unit='count')
                self._store_metric('api_error_rate', error_rate, 'gauge', 'api', unit='%')
                self._store_metric('api_response_time', avg_response_time, 'gauge', 'api', unit='seconds')
                self._store_metric('api_throughput', throughput, 'gauge', 'api', unit='requests/minute')

        except Exception as e:
            print(f"Error collecting API usage metrics: {e}")

    def _collect_tts_usage_metrics(self, cutoff_time: datetime):
        """Collect TTS-specific usage metrics."""
        try:
            # Sample TTS metrics
            audio_duration = 450.0  # seconds
            data_processed = 25.5  # MB
            tts_requests = 75  # count

            self._store_metric('tts_requests', tts_requests, 'counter', 'tts', unit='count')
            self._store_metric('tts_audio_duration', audio_duration, 'counter', 'tts', unit='seconds')
            self._store_metric('tts_data_processed', data_processed, 'counter', 'tts', unit='MB')

        except Exception as e:
            print(f"Error collecting TTS usage metrics: {e}")

    def _collect_error_metrics(self, cutoff_time: datetime):
        """Collect error and failure metrics."""
        try:
            # Sample error metrics
            timeout_count = 2
            retry_count = 8
            failure_count = 3

            self._store_metric('timeout_count', timeout_count, 'counter', 'system', unit='count')
            self._store_metric('retry_count', retry_count, 'counter', 'system', unit='count')
            self._store_metric('failure_count', failure_count, 'counter', 'system', unit='count')

        except Exception as e:
            print(f"Error collecting error metrics: {e}")

    # ===== PERFORMANCE METRICS COLLECTION =====

    def _collect_performance_metrics(self):
        """Collect detailed performance metrics."""
        try:
            # Database connections
            db_connection_count = self._get_db_connection_count()
            self._store_metric('db_connections', db_connection_count, 'gauge', 'system', unit='count')

            # Active connections
            active_connections = self._get_active_connections()
            self._store_metric('active_connections', active_connections, 'gauge', 'system', unit='count')

            # Queue lengths
            queue_length = self._get_queue_length()
            self._store_metric('queue_length', queue_length, 'gauge', 'system', unit='count')

            # Cache performance
            cache_hit_rate = self._get_cache_hit_rate()
            if cache_hit_rate is not None:
                self._store_metric('cache_hit_rate', cache_hit_rate, 'gauge', 'system', unit='%')

            # Response time percentiles
            response_times = self._get_response_time_percentiles()
            for percentile, value in response_times.items():
                self._store_metric(f'response_time_p{percentile}', value, 'gauge', 'api', unit='seconds')

        except Exception as e:
            print(f"Error collecting performance metrics: {e}")

    def _get_db_connection_count(self) -> int:
        """Get current database connection count."""
        try:
            # This is a simplified implementation
            # In a real system, you would query the actual connection pool
            return 5  # Sample data
        except:
            return 0

    def _get_active_connections(self) -> int:
        """Get count of active connections."""
        try:
            # This would typically query your web server or load balancer
            return 25  # Sample data
        except:
            return 0

    def _get_queue_length(self) -> int:
        """Get current queue length."""
        try:
            # This would query your job queue system
            return 3  # Sample data
        except:
            return 0

    def _get_cache_hit_rate(self) -> Optional[float]:
        """Get cache hit rate percentage."""
        try:
            # This would query your caching system
            return 85.5  # Sample data
        except:
            return None

    def _get_response_time_percentiles(self) -> Dict[str, float]:
        """Get response time percentiles."""
        try:
            # This would typically query your monitoring system or APM
            return {
                '50': 0.45,
                '95': 1.2,
                '99': 2.8
            }
        except:
            return {}

    # ===== USER BEHAVIOR METRICS COLLECTION =====

    def _collect_user_behavior_metrics(self):
        """Collect user behavior metrics."""
        try:
            # Session metrics
            active_sessions = self._get_active_sessions()
            self._store_metric('active_sessions', active_sessions, 'gauge', 'user', unit='count')

            # User engagement
            engagement_score = self._calculate_engagement_score()
            if engagement_score is not None:
                self._store_metric('user_engagement_score', engagement_score, 'gauge', 'user')

            # Geographic distribution
            geo_distribution = self._get_geographic_distribution()
            for country, count in geo_distribution.items():
                self._store_metric(f'users_by_country_{country}', count, 'gauge', 'user', unit='count')

        except Exception as e:
            print(f"Error collecting user behavior metrics: {e}")

    def _get_active_sessions(self) -> int:
        """Get count of active user sessions."""
        try:
            # This would query your session store
            return 42  # Sample data
        except:
            return 0

    def _calculate_engagement_score(self) -> Optional[float]:
        """Calculate overall user engagement score."""
        try:
            # This would analyze user activity patterns
            return 6.8  # Sample data
        except:
            return None

    def _get_geographic_distribution(self) -> Dict[str, int]:
        """Get user distribution by country."""
        try:
            # This would query your user analytics
            return {
                'US': 150,
                'UK': 45,
                'DE': 32,
                'FR': 28,
                'JP': 67
            }
        except:
            return {}

    # ===== BUSINESS METRICS COLLECTION =====

    def _collect_business_metrics(self):
        """Collect business intelligence metrics."""
        try:
            # Revenue metrics
            daily_revenue = self._get_daily_revenue()
            self._store_metric('daily_revenue', daily_revenue, 'gauge', 'business', unit='USD')

            # Customer metrics
            new_customers = self._get_new_customers()
            active_customers = self._get_active_customers()
            churned_customers = self._get_churned_customers()

            self._store_metric('new_customers', new_customers, 'counter', 'business', unit='count')
            self._store_metric('active_customers', active_customers, 'gauge', 'business', unit='count')
            self._store_metric('churned_customers', churned_customers, 'counter', 'business', unit='count')

            # Usage metrics
            total_requests = self._get_total_requests()
            paid_requests = self._get_paid_requests()

            self._store_metric('total_requests', total_requests, 'counter', 'business', unit='count')
            self._store_metric('paid_requests', paid_requests, 'counter', 'business', unit='count')

            # Growth metrics
            growth_rate = self._calculate_growth_rate()
            if growth_rate is not None:
                self._store_metric('growth_rate', growth_rate, 'gauge', 'business', unit='%')

        except Exception as e:
            print(f"Error collecting business metrics: {e}")

    def _get_daily_revenue(self) -> float:
        """Get daily revenue."""
        try:
            # This would query your billing system
            return 1250.75  # Sample data
        except:
            return 0.0

    def _get_new_customers(self) -> int:
        """Get count of new customers today."""
        try:
            # This would query your customer database
            return 12  # Sample data
        except:
            return 0

    def _get_active_customers(self) -> int:
        """Get count of active customers."""
        try:
            # This would query your customer database
            return 2847  # Sample data
        except:
            return 0

    def _get_churned_customers(self) -> int:
        """Get count of churned customers."""
        try:
            # This would query your customer database
            return 3  # Sample data
        except:
            return 0

    def _get_total_requests(self) -> int:
        """Get total requests today."""
        try:
            # This would query your usage logs
            return 15420  # Sample data
        except:
            return 0

    def _get_paid_requests(self) -> int:
        """Get paid requests today."""
        try:
            # This would query your billing system
            return 12850  # Sample data
        except:
            return 0

    def _calculate_growth_rate(self) -> Optional[float]:
        """Calculate business growth rate."""
        try:
            # This would compare current period with previous period
            return 15.7  # Sample data
        except:
            return None

    # ===== METRIC STORAGE =====

    def _store_metric(self, name: str, value: float, metric_type: str,
                     category: str, unit: Optional[str] = None,
                     labels: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Store a metric in the database."""
        try:
            # Create system metric
            metric = SystemMetric.create_gauge(name, value, category)
            if unit:
                metric.unit = unit
            if labels:
                metric.labels = labels
            if metadata:
                metric.metadata = metadata

            # Store in database
            db.session.add(metric)
            db.session.commit()

        except Exception as e:
            print(f"Error storing metric {name}: {e}")
            db.session.rollback()

    # ===== CUSTOM METRICS =====

    def register_custom_metric(self, name: str, collector_func: Callable,
                             category: str = 'custom', unit: Optional[str] = None):
        """Register a custom metric collector."""
        self._custom_metrics[name] = {
            'collector': collector_func,
            'category': category,
            'unit': unit
        }

    def collect_custom_metrics(self):
        """Collect all custom metrics."""
        for name, config in self._custom_metrics.items():
            try:
                value = config['collector']()
                if value is not None:
                    self._store_metric(
                        name, value, 'gauge', config['category'],
                        unit=config['unit']
                    )
            except Exception as e:
                print(f"Error collecting custom metric {name}: {e}")

    # ===== ERROR TRACKING =====

    def track_error(self, error_type: str, error_message: str,
                   context: Optional[Dict[str, Any]] = None):
        """Track an error occurrence."""
        try:
            with self._lock:
                if error_type not in self._error_trackers:
                    self._error_trackers[error_type] = {
                        'count': 0,
                        'last_occurrence': None,
                        'context': {}
                    }

                self._error_trackers[error_type]['count'] += 1
                self._error_trackers[error_type]['last_occurrence'] = datetime.utcnow()
                if context:
                    self._error_trackers[error_type]['context'] = context

                # Store error metric
                self._store_metric(
                    f'error_{error_type}',
                    self._error_trackers[error_type]['count'],
                    'counter',
                    'system',
                    labels={'type': error_type, 'message': error_message[:100]}
                )

        except Exception as e:
            print(f"Error tracking error {error_type}: {e}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error tracking summary."""
        return dict(self._error_trackers)

    # ===== PERFORMANCE MONITORING =====

    def monitor_function(self, func_name: str):
        """Decorator to monitor function performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # Record successful execution
                    execution_time = time.time() - start_time
                    self._store_metric(
                        f'function_execution_time_{func_name}',
                        execution_time,
                        'histogram',
                        'performance',
                        unit='seconds'
                    )

                    return result

                except Exception as e:
                    # Record failed execution
                    execution_time = time.time() - start_time
                    self._store_metric(
                        f'function_execution_time_{func_name}',
                        execution_time,
                        'histogram',
                        'performance',
                        unit='seconds'
                    )
                    self._store_metric(
                        f'function_errors_{func_name}',
                        1,
                        'counter',
                        'performance'
                    )
                    raise e

            return wrapper
        return decorator

    # ===== DATA AGGREGATION =====

    def aggregate_metrics(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Aggregate metrics over a time period."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # Get metrics from database
            metrics = SystemMetric.get_metrics_by_name(metric_name, db.session, hours=hours)

            if not metrics:
                return {}

            values = [m.value for m in metrics]

            return {
                'metric_name': metric_name,
                'count': len(values),
                'sum': sum(values),
                'avg': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'latest': values[-1] if values else 0,
                'period_hours': hours
            }

        except Exception as e:
            print(f"Error aggregating metrics {metric_name}: {e}")
            return {}

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        try:
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'period_hours': hours,
                'system_metrics': {},
                'usage_metrics': {},
                'performance_metrics': {},
                'error_summary': self.get_error_summary()
            }

            # Aggregate key metrics
            key_metrics = [
                'cpu_usage', 'memory_percent', 'api_requests', 'api_errors',
                'tts_requests', 'active_sessions', 'daily_revenue'
            ]

            for metric_name in key_metrics:
                summary['system_metrics'][metric_name] = self.aggregate_metrics(metric_name, hours)

            return summary

        except Exception as e:
            print(f"Error getting metrics summary: {e}")
            return {}

    # ===== CLEANUP =====

    def cleanup_old_metrics(self, days: int = 30):
        """Clean up old metrics from database."""
        try:
            deleted_count = SystemMetric.cleanup_old_metrics(db.session, days=days)
            print(f"Cleaned up {deleted_count} old metrics")
            return deleted_count
        except Exception as e:
            print(f"Error cleaning up old metrics: {e}")
            return 0

    # ===== PUBLIC API =====

    def get_collector_status(self) -> Dict[str, Any]:
        """Get status of all collectors."""
        return {
            'is_running': self._is_running,
            'active_collectors': list(self._collection_threads.keys()),
            'collection_intervals': self._collection_intervals,
            'custom_metrics_count': len(self._custom_metrics),
            'error_trackers_count': len(self._error_trackers)
        }

    def update_collection_interval(self, collector_name: str, interval_seconds: int):
        """Update collection interval for a specific collector."""
        if collector_name in self._collection_intervals:
            self._collection_intervals[collector_name] = interval_seconds
            return True
        return False


# Global metrics collector instance
metrics_collector = MetricsCollector()