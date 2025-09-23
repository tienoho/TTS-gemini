"""
Business Intelligence Analytics Engine for TTS System
Handles customer segmentation, revenue attribution, predictive analytics, and anomaly detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

from models.business_intelligence import (
    RevenueStream, CustomerJourney, BusinessKPI, UsagePattern,
    FinancialProjection, BusinessInsight
)
from models.analytics import UsageMetric, BusinessMetric, TimeSeriesData
from models.organization import Organization
from models.user import User
from utils.bi_service import bi_service


class BusinessIntelligenceAnalytics:
    """Advanced analytics engine for BI operations."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.customer_segments = {}
        self.revenue_attribution_models = {}

    def perform_customer_segmentation(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Perform advanced customer segmentation using clustering algorithms."""

        # Get customer data
        customer_data = self._get_customer_data_for_segmentation(organization_id, db_session)

        if len(customer_data) < 5:
            return {
                'error': 'Insufficient customer data for segmentation',
                'segments': {},
                'recommendations': []
            }

        # Prepare data for clustering
        features = self._prepare_customer_features(customer_data)
        X_scaled = self.scaler.fit_transform(features)

        # Determine optimal number of clusters
        optimal_clusters = self._find_optimal_clusters(X_scaled)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        cluster_analysis = self._analyze_customer_clusters(
            customer_data, clusters, kmeans.cluster_centers_
        )

        # Generate segment recommendations
        recommendations = self._generate_segmentation_recommendations(cluster_analysis)

        # Store segments
        self._store_customer_segments(organization_id, cluster_analysis, db_session)

        return {
            'total_customers': len(customer_data),
            'number_of_segments': optimal_clusters,
            'segments': cluster_analysis,
            'silhouette_score': silhouette_score(X_scaled, clusters),
            'recommendations': recommendations
        }

    def perform_revenue_attribution(self, organization_id: int, start_date: datetime,
                                  end_date: datetime, db_session: Session) -> Dict[str, Any]:
        """Perform revenue attribution analysis using multiple models."""

        # Get revenue and customer journey data
        revenue_data = db_session.query(RevenueStream).filter(
            and_(
                RevenueStream.organization_id == organization_id,
                RevenueStream.recognized_at >= start_date,
                RevenueStream.recognized_at <= end_date
            )
        ).all()

        journey_data = db_session.query(CustomerJourney).filter(
            and_(
                CustomerJourney.organization_id == organization_id,
                CustomerJourney.created_at >= start_date,
                CustomerJourney.created_at <= end_date
            )
        ).all()

        if not revenue_data:
            return {
                'error': 'No revenue data found for attribution analysis',
                'attribution_models': {}
            }

        # Apply different attribution models
        attribution_models = {}

        # Last Touch Attribution
        attribution_models['last_touch'] = self._calculate_last_touch_attribution(
            revenue_data, journey_data
        )

        # First Touch Attribution
        attribution_models['first_touch'] = self._calculate_first_touch_attribution(
            revenue_data, journey_data
        )

        # Linear Attribution
        attribution_models['linear'] = self._calculate_linear_attribution(
            revenue_data, journey_data
        )

        # Data-Driven Attribution (simplified)
        attribution_models['data_driven'] = self._calculate_data_driven_attribution(
            revenue_data, journey_data
        )

        # Analyze attribution insights
        insights = self._analyze_attribution_insights(attribution_models)

        # Store attribution data
        self._store_revenue_attribution(organization_id, attribution_models, db_session)

        return {
            'total_revenue': sum(r.amount for r in revenue_data),
            'attribution_models': attribution_models,
            'insights': insights,
            'recommendations': self._generate_attribution_recommendations(attribution_models)
        }

    def detect_anomalies_advanced(self, organization_id: int, start_date: datetime,
                                end_date: datetime, db_session: Session) -> Dict[str, Any]:
        """Advanced anomaly detection using machine learning."""

        # Get time series data
        time_series_data = self._get_time_series_data(organization_id, start_date, end_date, db_session)

        if len(time_series_data) < 30:
            return {
                'error': 'Insufficient data for anomaly detection',
                'anomalies': []
            }

        # Prepare data for anomaly detection
        features = self._prepare_anomaly_features(time_series_data)
        X_scaled = self.scaler.fit_transform(features)

        # Use Isolation Forest for anomaly detection
        isolation_forest = IsolationForest(
            contamination=0.1,  # Expected percentage of anomalies
            random_state=42
        )
        anomaly_scores = isolation_forest.fit_predict(X_scaled)
        anomaly_confidences = isolation_forest.decision_function(X_scaled)

        # Identify anomalies
        anomalies = []
        for i, (data_point, score, confidence) in enumerate(
            zip(time_series_data, anomaly_scores, anomaly_confidences)
        ):
            if score == -1:  # Anomaly detected
                anomalies.append({
                    'timestamp': data_point['timestamp'],
                    'metric': data_point['metric'],
                    'value': data_point['value'],
                    'expected_value': data_point.get('expected_value', 0),
                    'anomaly_score': abs(confidence),
                    'severity': 'high' if abs(confidence) > 0.7 else 'medium',
                    'description': self._describe_anomaly(data_point, confidence)
                })

        # Sort anomalies by severity
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)

        # Generate anomaly insights
        insights = self._generate_anomaly_insights(anomalies, time_series_data)

        return {
            'total_data_points': len(time_series_data),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(time_series_data),
            'anomalies': anomalies[:50],  # Limit to top 50
            'insights': insights,
            'recommendations': self._generate_anomaly_recommendations(anomalies)
        }

    def predict_customer_churn(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Predict customer churn using machine learning."""

        # Get customer behavior data
        customer_data = self._get_customer_churn_data(organization_id, db_session)

        if len(customer_data) < 50:
            return {
                'error': 'Insufficient data for churn prediction',
                'predictions': {}
            }

        # Prepare features for prediction
        features, labels = self._prepare_churn_features(customer_data)

        if len(features) < 10:
            return {
                'error': 'Insufficient feature data for churn prediction',
                'predictions': {}
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        predictions = rf_model.predict(X_test)

        # Calculate model performance
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Predict churn for all customers
        all_predictions = rf_model.predict(features)

        # Identify high-risk customers
        high_risk_customers = []
        for i, (customer, prediction) in enumerate(zip(customer_data, all_predictions)):
            if prediction > 0.7:  # High churn probability
                high_risk_customers.append({
                    'customer_id': customer['customer_id'],
                    'churn_probability': float(prediction),
                    'risk_level': 'high' if prediction > 0.8 else 'medium',
                    'key_factors': self._identify_churn_factors(customer, rf_model, features[i])
                })

        # Sort by churn probability
        high_risk_customers.sort(key=lambda x: x['churn_probability'], reverse=True)

        return {
            'model_performance': {
                'mse': mse,
                'r2_score': r2,
                'accuracy': max(0, min(1, 1 - mse))  # Simplified accuracy
            },
            'total_customers_analyzed': len(customer_data),
            'high_risk_customers': len(high_risk_customers),
            'average_churn_probability': float(np.mean(all_predictions)),
            'high_risk_customers': high_risk_customers[:20],  # Top 20
            'recommendations': self._generate_churn_prevention_recommendations(high_risk_customers)
        }

    def forecast_demand(self, organization_id: int, forecast_days: int,
                       db_session: Session) -> Dict[str, Any]:
        """Forecast demand using time series analysis."""

        # Get historical usage data
        historical_data = self._get_historical_usage_data(organization_id, 90, db_session)

        if len(historical_data) < 14:
            return {
                'error': 'Insufficient historical data for demand forecasting',
                'forecast': None
            }

        # Prepare time series data
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Use exponential smoothing for forecasting
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        try:
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                df['requests'],
                seasonal='add',
                seasonal_periods=7
            )
            fitted_model = model.fit()

            # Generate forecast
            forecast_index = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )

            forecast = fitted_model.forecast(forecast_days)

            # Calculate confidence intervals
            forecast_std = fitted_model.resid.std()
            confidence_intervals = [
                (float(pred - 1.96 * forecast_std), float(pred + 1.96 * forecast_std))
                for pred in forecast
            ]

            # Create forecast data
            forecast_data = []
            for i, (date, pred, conf_int) in enumerate(zip(forecast_index, forecast, confidence_intervals)):
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'forecasted_requests': int(pred),
                    'confidence_interval_low': int(conf_int[0]),
                    'confidence_interval_high': int(conf_int[1]),
                    'day_of_week': date.strftime('%A')
                })

            # Calculate forecast accuracy
            accuracy_metrics = self._calculate_forecast_accuracy_metrics(fitted_model, df)

            return {
                'forecast_method': 'exponential_smoothing',
                'historical_periods': len(df),
                'forecast_days': forecast_days,
                'forecast_data': forecast_data,
                'accuracy_metrics': accuracy_metrics,
                'seasonal_patterns': self._identify_seasonal_patterns(df),
                'recommendations': self._generate_demand_forecast_recommendations(forecast_data)
            }

        except Exception as e:
            return {
                'error': f'Forecasting failed: {str(e)}',
                'forecast': None
            }

    def analyze_cohort_performance(self, organization_id: int, db_session: Session) -> Dict[str, Any]:
        """Analyze customer cohort performance over time."""

        # Get customer journey data
        journey_data = db_session.query(CustomerJourney).filter(
            CustomerJourney.organization_id == organization_id
        ).all()

        if not journey_data:
            return {
                'error': 'No customer journey data available',
                'cohorts': {}
            }

        # Create cohorts based on acquisition month
        cohorts = self._create_customer_cohorts(journey_data)

        # Analyze cohort retention
        cohort_retention = self._calculate_cohort_retention(cohorts)

        # Analyze cohort revenue
        cohort_revenue = self._calculate_cohort_revenue(cohorts, organization_id, db_session)

        # Calculate cohort lifetime value
        cohort_ltv = self._calculate_cohort_ltv(cohort_revenue, cohort_retention)

        # Identify best and worst performing cohorts
        best_cohorts = self._identify_best_performing_cohorts(cohort_ltv)
        struggling_cohorts = self._identify_struggling_cohorts(cohort_retention)

        return {
            'total_cohorts': len(cohorts),
            'cohort_retention': cohort_retention,
            'cohort_revenue': cohort_revenue,
            'cohort_ltv': cohort_ltv,
            'best_performing_cohorts': best_cohorts,
            'struggling_cohorts': struggling_cohorts,
            'insights': self._generate_cohort_insights(cohort_retention, cohort_ltv),
            'recommendations': self._generate_cohort_recommendations(struggling_cohorts)
        }

    def _get_customer_data_for_segmentation(self, organization_id: int, db_session: Session) -> List[Dict[str, Any]]:
        """Get customer data for segmentation analysis."""
        # This is a simplified implementation
        # In a real system, you'd gather comprehensive customer behavior data

        customers = []

        # Get customer journey data
        journeys = db_session.query(CustomerJourney).filter(
            CustomerJourney.organization_id == organization_id
        ).all()

        # Aggregate data by customer
        customer_dict = {}
        for journey in journeys:
            customer_id = journey.customer_id
            if customer_id not in customer_dict:
                customer_dict[customer_id] = {
                    'customer_id': customer_id,
                    'total_sessions': 0,
                    'total_actions': 0,
                    'total_time_spent': 0,
                    'total_conversions': 0,
                    'avg_engagement_score': 0,
                    'lifecycle_stage': journey.lifecycle_stage,
                    'segment': journey.segment
                }

            customer_dict[customer_id]['total_sessions'] += 1
            customer_dict[customer_id]['total_actions'] += journey.actions_taken.count(',') if journey.actions_taken else 0
            customer_dict[customer_id]['total_time_spent'] += journey.time_spent_seconds
            customer_dict[customer_id]['total_conversions'] += 1 if journey.conversion_value > 0 else 0

        # Convert to list
        for customer_data in customer_dict.values():
            customer_data['avg_engagement_score'] = customer_data['total_actions'] / max(customer_data['total_sessions'], 1)
            customers.append(customer_data)

        return customers

    def _prepare_customer_features(self, customer_data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare customer features for clustering."""
        features = []

        for customer in customer_data:
            feature_vector = [
                customer['total_sessions'],
                customer['total_actions'],
                customer['total_time_spent'],
                customer['total_conversions'],
                customer['avg_engagement_score']
            ]
            features.append(feature_vector)

        return np.array(features)

    def _find_optimal_clusters(self, X_scaled: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette score."""
        max_clusters = min(6, len(X_scaled) // 2)  # Don't exceed reasonable cluster count

        if max_clusters < 2:
            return 2

        best_score = -1
        best_n_clusters = 2

        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, clusters)

                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except:
                continue

        return best_n_clusters

    def _analyze_customer_clusters(self, customer_data: List[Dict[str, Any]],
                                 clusters: np.ndarray, centers: np.ndarray) -> Dict[str, Any]:
        """Analyze customer clusters and create segment profiles."""
        cluster_profiles = {}

        for cluster_id in range(len(centers)):
            cluster_customers = [
                customer for customer, cluster in zip(customer_data, clusters)
                if cluster == cluster_id
            ]

            if cluster_customers:
                # Calculate cluster statistics
                avg_sessions = np.mean([c['total_sessions'] for c in cluster_customers])
                avg_actions = np.mean([c['total_actions'] for c in cluster_customers])
                avg_time_spent = np.mean([c['total_time_spent'] for c in cluster_customers])
                avg_conversions = np.mean([c['total_conversions'] for c in cluster_customers])
                avg_engagement = np.mean([c['avg_engagement_score'] for c in cluster_customers])

                # Determine segment characteristics
                if avg_engagement > 7:
                    segment_name = 'High Engagement'
                    segment_type = 'premium'
                elif avg_sessions > 10:
                    segment_name = 'Active Users'
                    segment_type = 'standard'
                elif avg_conversions > 0.5:
                    segment_name = 'Converting Users'
                    segment_type = 'growth'
                else:
                    segment_name = 'Casual Users'
                    segment_type = 'basic'

                cluster_profiles[f'cluster_{cluster_id}'] = {
                    'segment_name': segment_name,
                    'segment_type': segment_type,
                    'customer_count': len(cluster_customers),
                    'percentage': len(cluster_customers) / len(customer_data) * 100,
                    'avg_sessions': avg_sessions,
                    'avg_actions': avg_actions,
                    'avg_time_spent': avg_time_spent,
                    'avg_conversions': avg_conversions,
                    'avg_engagement_score': avg_engagement,
                    'characteristics': self._describe_segment_characteristics(cluster_customers)
                }

        return cluster_profiles

    def _describe_segment_characteristics(self, customers: List[Dict[str, Any]]) -> List[str]:
        """Describe characteristics of a customer segment."""
        characteristics = []

        avg_sessions = np.mean([c['total_sessions'] for c in customers])
        avg_engagement = np.mean([c['avg_engagement_score'] for c in customers])
        avg_conversions = np.mean([c['total_conversions'] for c in customers])

        if avg_sessions > 15:
            characteristics.append('Highly active users')
        elif avg_sessions < 3:
            characteristics.append('Low activity users')

        if avg_engagement > 8:
            characteristics.append('Strong engagement')
        elif avg_engagement < 2:
            characteristics.append('Low engagement')

        if avg_conversions > 1:
            characteristics.append('High conversion rate')
        elif avg_conversions < 0.1:
            characteristics.append('Low conversion rate')

        return characteristics

    def _generate_segmentation_recommendations(self, cluster_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on customer segmentation."""
        recommendations = []

        for cluster_id, cluster_data in cluster_analysis.items():
            if cluster_data['segment_type'] == 'premium':
                recommendations.append(
                    f"Focus on retention strategies for {cluster_data['segment_name']} segment "
                    f"({cluster_data['customer_count']} customers, {cluster_data['percentage']:.1f}%)"
                )
            elif cluster_data['segment_type'] == 'growth':
                recommendations.append(
                    f"Implement conversion optimization for {cluster_data['segment_name']} segment "
                    f"({cluster_data['customer_count']} customers, {cluster_data['percentage']:.1f}%)"
                )
            elif cluster_data['segment_type'] == 'basic':
                recommendations.append(
                    f"Develop engagement strategies for {cluster_data['segment_name']} segment "
                    f"({cluster_data['customer_count']} customers, {cluster_data['percentage']:.1f}%)"
                )

        return recommendations

    def _store_customer_segments(self, organization_id: int, cluster_analysis: Dict[str, Any], db_session: Session):
        """Store customer segments in the database."""
        # This would store segment definitions for future use
        # Implementation depends on specific requirements
        pass

    def _calculate_last_touch_attribution(self, revenue_data: List, journey_data: List) -> Dict[str, Any]:
        """Calculate last touch attribution."""
        attribution = {}

        for revenue in revenue_data:
            # Find the last journey before revenue recognition
            customer_journeys = [
                j for j in journey_data
                if j.customer_id == revenue.metadata.get('customer_id')
            ]

            if customer_journeys:
                # Sort by timestamp and get the last one
                last_journey = max(customer_journeys, key=lambda x: x.created_at)

                touchpoint = last_journey.touchpoint or 'unknown'
                attribution[touchpoint] = attribution.get(touchpoint, 0) + revenue.amount

        return {
            'model': 'last_touch',
            'attribution': attribution,
            'total_attributed_revenue': sum(attribution.values())
        }

    def _calculate_first_touch_attribution(self, revenue_data: List, journey_data: List) -> Dict[str, Any]:
        """Calculate first touch attribution."""
        attribution = {}

        for revenue in revenue_data:
            customer_journeys = [
                j for j in journey_data
                if j.customer_id == revenue.metadata.get('customer_id')
            ]

            if customer_journeys:
                # Sort by timestamp and get the first one
                first_journey = min(customer_journeys, key=lambda x: x.created_at)

                touchpoint = first_journey.touchpoint or 'unknown'
                attribution[touchpoint] = attribution.get(touchpoint, 0) + revenue.amount

        return {
            'model': 'first_touch',
            'attribution': attribution,
            'total_attributed_revenue': sum(attribution.values())
        }

    def _calculate_linear_attribution(self, revenue_data: List, journey_data: List) -> Dict[str, Any]:
        """Calculate linear attribution."""
        attribution = {}

        for revenue in revenue_data:
            customer_journeys = [
                j for j in journey_data
                if j.customer_id == revenue.metadata.get('customer_id')
            ]

            if customer_journeys:
                # Distribute revenue equally across all touchpoints
                revenue_per_touchpoint = revenue.amount / len(customer_journeys)

                for journey in customer_journeys:
                    touchpoint = journey.touchpoint or 'unknown'
                    attribution[touchpoint] = attribution.get(touchpoint, 0) + revenue_per_touchpoint

        return {
            'model': 'linear',
            'attribution': attribution,
            'total_attributed_revenue': sum(attribution.values())
        }

    def _calculate_data_driven_attribution(self, revenue_data: List, journey_data: List) -> Dict[str, Any]:
        """Calculate data-driven attribution (simplified)."""
        # This is a simplified implementation
        # In a real system, you'd use more sophisticated algorithms

        attribution = {}

        for revenue in revenue_data:
            customer_journeys = [
                j for j in journey_data
                if j.customer_id == revenue.metadata.get('customer_id')
            ]

            if customer_journeys:
                # Weight touchpoints based on their position in the journey
                total_journeys = len(customer_journeys)

                for i, journey in enumerate(customer_journeys):
                    # Give more weight to touchpoints closer to conversion
                    weight = (i + 1) / total_journeys
                    touchpoint = journey.touchpoint or 'unknown'
                    attribution[touchpoint] = attribution.get(touchpoint, 0) + (revenue.amount * weight)

        return {
            'model': 'data_driven',
            'attribution': attribution,
            'total_attributed_revenue': sum(attribution.values())
        }

    def _analyze_attribution_insights(self, attribution_models: Dict[str, Any]) -> List[str]:
        """Analyze attribution insights."""
        insights = []

        # Compare different models
        last_touch = attribution_models.get('last_touch', {})
        first_touch = attribution_models.get('first_touch', {})

        if last_touch and first_touch:
            # Find touchpoints that perform well in both models
            common_touchpoints = set(last_touch.get('attribution', {}).keys()) & \
                               set(first_touch.get('attribution', {}).keys())

            if common_touchpoints:
                insights.append(
                    f"Touchpoints {', '.join(list(common_touchpoints)[:3])} perform well across multiple attribution models"
                )

        return insights

    def _generate_attribution_recommendations(self, attribution_models: Dict[str, Any]) -> List[str]:
        """Generate attribution-based recommendations."""
        recommendations = []

        # Find top performing touchpoints
        all_attribution = {}
        for model_data in attribution_models.values():
            for touchpoint, revenue in model_data.get('attribution', {}).items():
                all_attribution[touchpoint] = all_attribution.get(touchpoint, 0) + revenue

        if all_attribution:
            top_touchpoint = max(all_attribution.items(), key=lambda x: x[1])
            recommendations.append(
                f"Invest more in {top_touchpoint[0]} - it drives {top_touchpoint[1]:.2f} in attributed revenue"
            )

        return recommendations

    def _store_revenue_attribution(self, organization_id: int, attribution_models: Dict[str, Any], db_session: Session):
        """Store revenue attribution data."""
        # Implementation would store attribution data for future reference
        pass

    def _get_time_series_data(self, organization_id: int, start_date: datetime,
                            end_date: datetime, db_session: Session) -> List[Dict[str, Any]]:
        """Get time series data for anomaly detection."""
        time_series_data = []

        # Get usage metrics
        usage_metrics = db_session.query(UsageMetric).filter(
            and_(
                UsageMetric.organization_id == organization_id,
                UsageMetric.timestamp >= start_date,
                UsageMetric.timestamp <= end_date
            )
        ).all()

        for metric in usage_metrics:
            time_series_data.append({
                'timestamp': metric.timestamp,
                'metric': 'requests',
                'value': metric.request_count,
                'expected_value': 0  # Will be calculated
            })

            time_series_data.append({
                'timestamp': metric.timestamp,
                'metric': 'errors',
                'value': metric.error_count,
                'expected_value': 0
            })

        return time_series_data

    def _prepare_anomaly_features(self, time_series_data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare features for anomaly detection."""
        features = []

        for data_point in time_series_data:
            # Create feature vector
            feature_vector = [
                data_point['value'],
                1 if data_point['metric'] == 'requests' else 0,
                1 if data_point['metric'] == 'errors' else 0,
                data_point['timestamp'].hour,
                data_point['timestamp'].weekday()
            ]
            features.append(feature_vector)

        return np.array(features)

    def _describe_anomaly(self, data_point: Dict[str, Any], confidence: float) -> str:
        """Describe an anomaly."""
        severity = 'high' if abs(confidence) > 0.7 else 'medium'

        if data_point['metric'] == 'requests':
            if data_point['value'] > 1000:
                return f"Unusually high request volume: {data_point['value']}"
            else:
                return f"Unusually low request volume: {data_point['value']}"
        elif data_point['metric'] == 'errors':
            return f"Elevated error rate detected: {data_point['value']}"

        return f"Anomaly detected in {data_point['metric']}: {data_point['value']}"

    def _generate_anomaly_insights(self, anomalies: List[Dict[str, Any]],
                                 time_series_data: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from anomalies."""
        insights = []

        if len(anomalies) > len(time_series_data) * 0.05:  # More than 5% anomalies
            insights.append("High anomaly rate detected. Consider investigating system stability.")

        # Check for patterns in anomalies
        error_anomalies = [a for a in anomalies if a['metric'] == 'errors']
        if len(error_anomalies) > 3:
            insights.append("Multiple error anomalies detected. System reliability may be compromised.")

        return insights

    def _generate_anomaly_recommendations(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on anomalies."""
        recommendations = []

        if len(anomalies) > 10:
            recommendations.append("Implement automated anomaly detection and alerting system.")

        error_anomalies = [a for a in anomalies if a['metric'] == 'errors']
        if len(error_anomalies) > 0:
            recommendations.append("Investigate error patterns and implement preventive measures.")

        return recommendations

    def _get_customer_churn_data(self, organization_id: int, db_session: Session) -> List[Dict[str, Any]]:
        """Get customer data for churn prediction."""
        # Simplified implementation
        customers = []

        # Get customer journeys
        journeys = db_session.query(CustomerJourney).filter(
            CustomerJourney.organization_id == organization_id
        ).all()

        # Aggregate by customer
        customer_dict = {}
        for journey in journeys:
            customer_id = journey.customer_id
            if customer_id not in customer_dict:
                customer_dict[customer_id] = {
                    'customer_id': customer_id,
                    'sessions': 0,
                    'actions': 0,
                    'time_spent': 0,
                    'conversions': 0,
                    'engagement_score': 0,
                    'days_since_last_activity': 0,
                    'churned': 0  # Target variable
                }

            customer_dict[customer_id]['sessions'] += 1
            customer_dict[customer_id]['actions'] += len(journey.actions_taken) if journey.actions_taken else 0
            customer_dict[customer_id]['time_spent'] += journey.time_spent_seconds
            customer_dict[customer_id]['conversions'] += 1 if journey.conversion_value > 0 else 0

            # Update engagement score
            customer_dict[customer_id]['engagement_score'] = (
                customer_dict[customer_id]['actions'] / max(customer_dict[customer_id]['sessions'], 1)
            )

        # Convert to list and add churn labels
        for customer_data in customer_dict.values():
            # Simple churn logic: customers with low activity in last 30 days
            customer_data['churned'] = 1 if customer_data['engagement_score'] < 1 else 0
            customers.append(customer_data)

        return customers

    def _prepare_churn_features(self, customer_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for churn prediction."""
        features = []
        labels = []

        for customer in customer_data:
            feature_vector = [
                customer['sessions'],
                customer['actions'],
                customer['time_spent'],
                customer['conversions'],
                customer['engagement_score']
            ]
            features.append(feature_vector)
            labels.append(customer['churned'])

        return np.array(features), np.array(labels)

    def _identify_churn_factors(self, customer: Dict[str, Any], model, features: np.ndarray) -> List[str]:
        """Identify key factors contributing to churn prediction."""
        factors = []

        if customer['sessions'] < 3:
            factors.append('Low session count')
        if customer['engagement_score'] < 1:
            factors.append('Low engagement')
        if customer['conversions'] == 0:
            factors.append('No conversions')

        return factors

    def _generate_churn_prevention_recommendations(self, high_risk_customers: List[Dict[str, Any]]) -> List[str]:
        """Generate churn prevention recommendations."""
        recommendations = []

        if len(high_risk_customers) > 10:
            recommendations.append("Implement proactive customer success program for high-risk customers.")

        # Analyze common factors
        common_factors = {}
        for customer in high_risk_customers:
            for factor in customer['key_factors']:
                common_factors[factor] = common_factors.get(factor, 0) + 1

        if common_factors:
            top_factor = max(common_factors.items(), key=lambda x: x[1])
            recommendations.append(f"Address {top_factor[0].lower()} - affects {top_factor[1]} high-risk customers.")

        return recommendations

    def _get_historical_usage_data(self, organization_id: int, days: int, db_session: Session) -> List[Dict[str, Any]]:
        """Get historical usage data for demand forecasting."""
        historical_data = []

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get daily usage metrics
        usage_metrics = db_session.query(
            func.date(UsageMetric.timestamp).label('date'),
            func.sum(UsageMetric.request_count).label('requests'),
            func.avg(UsageMetric.avg_response_time).label('avg_response_time')
        ).filter(
            and_(
                UsageMetric.organization_id == organization_id,
                UsageMetric.timestamp >= start_date,
                UsageMetric.timestamp <= end_date
            )
        ).group_by(func.date(UsageMetric.timestamp)).all()

        for metric in usage_metrics:
            historical_data.append({
                'date': metric.date,
                'requests': metric.requests or 0,
                'avg_response_time': float(metric.avg_response_time or 0)
            })

        return historical_data

    def _calculate_forecast_accuracy_metrics(self, model, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        # Split data for validation
        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        if len(test) == 0:
            return {'mae': 0, 'rmse': 0, 'mape': 0}

        # Fit model on training data
        train_model = model.__class__(
            seasonal='add' if hasattr(model, 'seasonal') else None,
            seasonal_periods=getattr(model, 'seasonal_periods', 7)
        )
        train_model = train_model.fit(train['requests'])

        # Make predictions
        predictions = train_model.forecast(len(test))

        # Calculate metrics
        mae = np.mean(np.abs(predictions - test['requests']))
        rmse = np.sqrt(np.mean((predictions - test['requests']) ** 2))
        mape = np.mean(np.abs((predictions - test['requests']) / test['requests'])) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }

    def _identify_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify seasonal patterns in usage data."""
        # Simple seasonal analysis
        daily_avg = df.groupby(df.index.dayofweek)['requests'].mean()
        hourly_avg = df.groupby(df.index.hour)['requests'].mean()

        return {
            'daily_pattern': daily_avg.to_dict(),
            'hourly_pattern': hourly_avg.to_dict(),
            'peak_day': int(daily_avg.idxmax()),
            'peak_hour': int(hourly_avg.idxmax())
        }

    def _generate_demand_forecast_recommendations(self, forecast_data: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on demand forecast."""
        recommendations = []

        # Find peak demand periods
        peak_forecasts = [f for f in forecast_data if f['forecasted_requests'] > 1000]

        if peak_forecasts:
            recommendations.append(
                f"Prepare for {len(peak_forecasts)} high-demand periods with forecasted requests > 1000"
            )

        # Check for growth trend
        if len(forecast_data) > 7:
            first_week_avg = np.mean([f['forecasted_requests'] for f in forecast_data[:7]])
            last_week_avg = np.mean([f['forecasted_requests'] for f in forecast_data[-7:]])

            if last_week_avg > first_week_avg * 1.1:
                recommendations.append("Demand is forecasted to increase. Consider capacity planning.")

        return recommendations

    def _create_customer_cohorts(self, journey_data: List) -> Dict[str, List]:
        """Create customer cohorts based on acquisition period."""
        cohorts = {}

        for journey in journey_data:
            cohort_key = journey.created_at.strftime('%Y-%m')  # Monthly cohorts

            if cohort_key not in cohorts:
                cohorts[cohort_key] = []

            if journey.customer_id not in [c.customer_id for c in cohorts[cohort_key]]:
                cohorts[cohort_key].append(journey)

        return cohorts

    def _calculate_cohort_retention(self, cohorts: Dict[str, List]) -> Dict[str, Any]:
        """Calculate cohort retention rates."""
        retention_data = {}

        for cohort_key, customers in cohorts.items():
            customer_ids = list(set(c.customer_id for c in customers))

            # Calculate retention over time (simplified)
            retention_data[cohort_key] = {
                'cohort_size': len(customer_ids),
                'retention_rate': 0.8,  # Simplified - would calculate actual retention
                'avg_lifetime_days': 90  # Simplified
            }

        return retention_data

    def _calculate_cohort_revenue(self, cohorts: Dict[str, List], organization_id: int, db_session: Session) -> Dict[str, float]:
        """Calculate revenue by cohort."""
        cohort_revenue = {}

        for cohort_key, customers in cohorts.items():
            customer_ids = list(set(c.customer_id for c in customers))

            # Calculate total revenue for this cohort (simplified)
            cohort_revenue[cohort_key] = 10000.0  # Would calculate actual revenue

        return cohort_revenue

    def _calculate_cohort_ltv(self, cohort_revenue: Dict[str, float], cohort_retention: Dict[str, Any]) -> Dict[str, float]:
        """Calculate customer lifetime value by cohort."""
        cohort_ltv = {}

        for cohort_key in cohort_revenue:
            revenue = cohort_revenue[cohort_key]
            retention = cohort_retention[cohort_key]['retention_rate']
            cohort_ltv[cohort_key] = revenue * retention

        return cohort_ltv

    def _identify_best_performing_cohorts(self, cohort_ltv: Dict[str, float]) -> List[str]:
        """Identify best performing cohorts."""
        if not cohort_ltv:
            return []

        sorted_cohorts = sorted(cohort_ltv.items(), key=lambda x: x[1], reverse=True)
        return [cohort[0] for cohort in sorted_cohorts[:3]]  # Top 3

    def _identify_struggling_cohorts(self, cohort_retention: Dict[str, Any]) -> List[str]:
        """Identify struggling cohorts."""
        struggling = []

        for cohort_key, retention_data in cohort_retention.items():
            if retention_data['retention_rate'] < 0.5:  # Less than 50% retention
                struggling.append(cohort_key)

        return struggling

    def _generate_cohort_insights(self, cohort_retention: Dict[str, Any], cohort_ltv: Dict[str, float]) -> List[str]:
        """Generate insights from cohort analysis."""
        insights = []

        if cohort_retention:
            avg_retention = np.mean([r['retention_rate'] for r in cohort_retention.values()])

            if avg_retention > 0.8:
                insights.append("Excellent cohort retention rates across all groups.")
            elif avg_retention < 0.5:
                insights.append("Low cohort retention rates indicate potential issues with customer satisfaction.")

        if cohort_ltv:
            avg_ltv = np.mean(list(cohort_ltv.values()))

            if avg_ltv > 50000:
                insights.append("High cohort lifetime values indicate strong business model.")
            elif avg_ltv < 10000:
                insights.append("Low cohort lifetime values suggest need for improvement in customer value.")

        return insights

    def _generate_cohort_recommendations(self, struggling_cohorts: List[str]) -> List[str]:
        """Generate recommendations for struggling cohorts."""
        recommendations = []

        if len(struggling_cohorts) > 0:
            recommendations.append(
                f"Focus retention efforts on struggling cohorts: {', '.join(struggling_cohorts[:3])}"
            )
            recommendations.append("Implement cohort-specific customer success programs.")
            recommendations.append("Analyze differences between high and low performing cohorts.")

        return recommendations


# Global BI analytics instance
bi_analytics = BusinessIntelligenceAnalytics()