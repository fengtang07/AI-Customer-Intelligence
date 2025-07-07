"""
Advanced customer analytics engine
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CustomerAnalyzer:
    """Advanced customer behavior analysis"""
    
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.churn_model = None
        self.segments = None
        
    def set_data(self, df: pd.DataFrame):
        """Set the dataframe for analysis"""
        self.df = df.copy()
        logger.info(f"CustomerAnalyzer loaded {len(self.df)} records")
    
    def analyze_churn_patterns(self) -> Dict[str, Any]:
        """Comprehensive churn pattern analysis"""
        if self.df is None or 'churn' not in self.df.columns:
            return {"error": "No data or churn column available"}
        
        churned = self.df[self.df['churn'] == 1]
        retained = self.df[self.df['churn'] == 0]
        
        analysis = {
            'overview': {
                'total_customers': len(self.df),
                'churned_count': len(churned),
                'retained_count': len(retained),
                'churn_rate': len(churned) / len(self.df) * 100
            },
            'demographic_analysis': self._analyze_demographics(churned, retained),
            'behavioral_analysis': self._analyze_behavior(churned, retained),
            'risk_factors': self._identify_risk_factors(churned, retained),
            'high_risk_customers': self._identify_high_risk_customers(),
            'churn_drivers': self._analyze_churn_drivers()
        }
        
        return analysis
    
    def perform_segmentation(self, n_clusters: int = 4) -> Dict[str, Any]:
        """Perform customer segmentation analysis"""
        if self.df is None:
            return {"error": "No data available"}
        
        # Select features for segmentation
        feature_columns = []
        for col in ['age', 'total_spent', 'monthly_visits', 'satisfaction_score', 'total_orders']:
            if col in self.df.columns:
                feature_columns.append(col)
        
        if len(feature_columns) < 2:
            return {"error": "Insufficient features for segmentation"}
        
        # Prepare data
        X = self.df[feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add segments to dataframe
        df_with_segments = self.df.copy()
        df_with_segments['segment'] = clusters
        
        # Analyze segments
        segment_analysis = {
            'segments': {},
            'segment_summary': self._create_segment_summary(df_with_segments, feature_columns),
            'business_value': self._analyze_segment_value(df_with_segments)
        }
        
        # Create detailed segment profiles
        for segment_id in range(n_clusters):
            segment_data = df_with_segments[df_with_segments['segment'] == segment_id]
            segment_analysis['segments'][f'segment_{segment_id}'] = self._create_segment_profile(
                segment_data, feature_columns, segment_id
            )
        
        self.segments = segment_analysis
        return segment_analysis
    
    def _analyze_demographics(self, churned: pd.DataFrame, retained: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demographic patterns"""
        analysis = {}
        
        # Age analysis
        if 'age' in self.df.columns:
            analysis['age'] = {
                'churned_avg': churned['age'].mean(),
                'retained_avg': retained['age'].mean(),
                'age_groups': self._analyze_age_groups()
            }
        
        # Gender analysis
        if 'gender' in self.df.columns:
            analysis['gender'] = self._analyze_gender_patterns()
        
        return analysis
    
    def _analyze_behavior(self, churned: pd.DataFrame, retained: pd.DataFrame) -> Dict[str, Any]:
        """Analyze behavioral patterns"""
        analysis = {}
        
        behavioral_metrics = ['monthly_visits', 'total_spent', 'satisfaction_score', 'total_orders']
        
        for metric in behavioral_metrics:
            if metric in self.df.columns:
                analysis[metric] = {
                    'churned_avg': churned[metric].mean(),
                    'retained_avg': retained[metric].mean(),
                    'difference_pct': ((churned[metric].mean() / retained[metric].mean()) - 1) * 100
                }
        
        return analysis
    
    def _identify_risk_factors(self, churned: pd.DataFrame, retained: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify key risk factors for churn"""
        risk_factors = []
        
        metrics = ['age', 'total_spent', 'monthly_visits', 'satisfaction_score']
        
        for metric in metrics:
            if metric in self.df.columns:
                churned_mean = churned[metric].mean()
                retained_mean = retained[metric].mean()
                
                if retained_mean != 0:
                    impact = abs((churned_mean - retained_mean) / retained_mean) * 100
                    
                    if impact > 20:  # Significant difference
                        risk_factors.append({
                            'factor': metric,
                            'churned_avg': churned_mean,
                            'retained_avg': retained_mean,
                            'impact_score': impact,
                            'direction': 'lower' if churned_mean < retained_mean else 'higher'
                        })
        
        # Sort by impact
        risk_factors.sort(key=lambda x: x['impact_score'], reverse=True)
        return risk_factors
    
    def _identify_high_risk_customers(self) -> List[Dict[str, Any]]:
        """Identify current customers at high risk of churning"""
        if 'churn' not in self.df.columns:
            return []
        
        # Get current customers (not churned)
        current_customers = self.df[self.df['churn'] == 0]
        
        if len(current_customers) == 0:
            return []
        
        # Define risk thresholds based on churned customer patterns
        churned_customers = self.df[self.df['churn'] == 1]
        
        risk_criteria = {}
        if 'satisfaction_score' in self.df.columns:
            risk_criteria['satisfaction_score'] = churned_customers['satisfaction_score'].quantile(0.75)
        
        if 'monthly_visits' in self.df.columns:
            risk_criteria['monthly_visits'] = churned_customers['monthly_visits'].quantile(0.75)
        
        if 'total_spent' in self.df.columns:
            risk_criteria['total_spent'] = churned_customers['total_spent'].quantile(0.75)
        
        # Score risk for each customer
        high_risk_customers = []
        
        for _, customer in current_customers.iterrows():
            risk_score = 0
            risk_factors = []
            
            for criterion, threshold in risk_criteria.items():
                if criterion in customer and customer[criterion] <= threshold:
                    risk_score += 1
                    risk_factors.append(criterion)
            
            if risk_score >= 2:  # At least 2 risk factors
                high_risk_customers.append({
                    'customer_id': customer['customer_id'],
                    'risk_score': risk_score,
                    'risk_factors': risk_factors,
                    'satisfaction_score': customer.get('satisfaction_score', 'N/A'),
                    'monthly_visits': customer.get('monthly_visits', 'N/A'),
                    'total_spent': customer.get('total_spent', 'N/A')
                })
        
        # Sort by risk score
        high_risk_customers.sort(key=lambda x: x['risk_score'], reverse=True)
        return high_risk_customers[:20]  # Top 20 at-risk customers
    
    def _analyze_churn_drivers(self) -> Dict[str, Any]:
        """Analyze primary drivers of churn using statistical methods"""
        if 'churn' not in self.df.columns:
            return {}
        
        # Calculate correlations
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_columns:
            if col != 'churn' and col != 'customer_id':
                correlation = self.df[col].corr(self.df['churn'])
                if not np.isnan(correlation):
                    correlations[col] = {
                        'correlation': correlation,
                        'strength': abs(correlation),
                        'direction': 'positive' if correlation > 0 else 'negative'
                    }
        
        # Sort by correlation strength
        sorted_correlations = dict(sorted(correlations.items(), 
                                        key=lambda x: x[1]['strength'], 
                                        reverse=True))
        
        return {
            'correlations': sorted_correlations,
            'top_drivers': list(sorted_correlations.keys())[:5]
        }
    
    def _analyze_age_groups(self) -> Dict[str, Any]:
        """Analyze churn by age groups"""
        if 'age' not in self.df.columns:
            return {}
        
        # Create age groups
        age_bins = [0, 25, 35, 45, 55, 100]
        age_labels = ['<25', '25-34', '35-44', '45-54', '55+']
        
        df_temp = self.df.copy()
        df_temp['age_group'] = pd.cut(df_temp['age'], bins=age_bins, labels=age_labels, right=False)
        
        age_analysis = df_temp.groupby('age_group')['churn'].agg(['count', 'sum', 'mean']).round(3)
        age_analysis.columns = ['total', 'churned', 'churn_rate']
        
        return age_analysis.to_dict('index')
    
    def _analyze_gender_patterns(self) -> Dict[str, Any]:
        """Analyze churn patterns by gender"""
        gender_analysis = self.df.groupby('gender')['churn'].agg(['count', 'sum', 'mean']).round(3)
        gender_analysis.columns = ['total', 'churned', 'churn_rate']
        
        # Convert to more readable format
        result = {}
        for gender_code, stats in gender_analysis.iterrows():
            gender_name = 'Female' if gender_code == 0 else 'Male' if gender_code == 1 else f'Gender_{gender_code}'
            result[gender_name] = {
                'total_customers': int(stats['total']),
                'churned_customers': int(stats['churned']),
                'churn_rate': float(stats['churn_rate']) * 100
            }
        
        return result
    
    def _create_segment_summary(self, df_with_segments: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Create high-level segment summary"""
        summary = {}
        
        for segment_id in df_with_segments['segment'].unique():
            segment_data = df_with_segments[df_with_segments['segment'] == segment_id]
            
            summary[f'segment_{segment_id}'] = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(df_with_segments) * 100,
                'churn_rate': segment_data['churn'].mean() * 100 if 'churn' in segment_data.columns else None
            }
        
        return summary
    
    def _analyze_segment_value(self, df_with_segments: pd.DataFrame) -> Dict[str, Any]:
        """Analyze business value of each segment"""
        value_analysis = {}
        
        if 'total_spent' in df_with_segments.columns:
            segment_value = df_with_segments.groupby('segment')['total_spent'].agg(['sum', 'mean', 'count'])
            
            for segment_id, stats in segment_value.iterrows():
                value_analysis[f'segment_{segment_id}'] = {
                    'total_revenue': stats['sum'],
                    'avg_revenue_per_customer': stats['mean'],
                    'customer_count': stats['count'],
                    'revenue_percentage': stats['sum'] / df_with_segments['total_spent'].sum() * 100
                }
        
        return value_analysis
    
    def _create_segment_profile(self, segment_data: pd.DataFrame, features: List[str], segment_id: int) -> Dict[str, Any]:
        """Create detailed profile for a specific segment"""
        profile = {
            'id': segment_id,
            'size': len(segment_data),
            'characteristics': {},
            'business_metrics': {}
        }
        
        # Feature characteristics
        for feature in features:
            if feature in segment_data.columns:
                profile['characteristics'][feature] = {
                    'mean': segment_data[feature].mean(),
                    'median': segment_data[feature].median(),
                    'std': segment_data[feature].std()
                }
        
        # Business metrics
        if 'churn' in segment_data.columns:
            profile['business_metrics']['churn_rate'] = segment_data['churn'].mean() * 100
        
        if 'total_spent' in segment_data.columns:
            profile['business_metrics']['total_revenue'] = segment_data['total_spent'].sum()
            profile['business_metrics']['avg_revenue'] = segment_data['total_spent'].mean()
        
        # Generate segment label
        profile['label'] = self._generate_segment_label(profile['characteristics'], segment_id)
        
        return profile
    
    def _generate_segment_label(self, characteristics: Dict[str, Any], segment_id: int) -> str:
        """Generate human-readable label for segment"""
        # Simple rule-based labeling
        if 'total_spent' in characteristics and 'satisfaction_score' in characteristics:
            spending = characteristics['total_spent']['mean']
            satisfaction = characteristics['satisfaction_score']['mean']
            
            if spending > 1500 and satisfaction > 4.0:
                return "High-Value Loyalists"
            elif spending > 1500 and satisfaction < 3.5:
                return "At-Risk High-Value"
            elif spending < 800 and satisfaction > 4.0:
                return "Satisfied Budget Customers"
            elif satisfaction < 3.0:
                return "Dissatisfied Customers"
            else:
                return f"Standard Segment {segment_id}"
        
        return f"Customer Segment {segment_id}"
