"""
Enhanced AI Customer Intelligence Agent
Core business intelligence engine
"""

import pandas as pd
import numpy as np
import openai
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

from ..utils.data_processor import DataProcessor
from ..analysis.customer_analyzer import CustomerAnalyzer
from config.config import Config

logger = logging.getLogger(__name__)

class CustomerIntelligenceAgent:
    """Advanced AI Customer Intelligence Agent"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.config = Config()
        self.data_path = data_path or self.config.DEFAULT_DATA_PATH
        self.client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.analyzer = CustomerAnalyzer()
        
        # Load data
        self.df = None
        self.load_data()
        
        logger.info(f"CustomerIntelligenceAgent initialized with {len(self.df) if self.df is not None else 0} records")
    
    def load_data(self):
        """Load and prepare customer data"""
        try:
            self.df = self.data_processor.load_data(self.data_path)
            if self.df is not None:
                self.analyzer.set_data(self.df)
                logger.info(f"Data loaded successfully: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.df = None
    
    def query(self, question: str, include_ai_insights: bool = True) -> Dict[str, Any]:
        """
        Process natural language queries about customer data
        
        Args:
            question: Natural language question
            include_ai_insights: Whether to generate AI insights
            
        Returns:
            Dict with analysis results and insights
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        # Classify query type
        query_type = self._classify_query(question)
        
        # Route to appropriate handler
        result = {
            'query': question,
            'query_type': query_type,
            'timestamp': datetime.now().isoformat(),
            'data': {},
            'insights': '',
            'recommendations': []
        }
        
        try:
            if query_type == 'churn_analysis':
                result['data'] = self._handle_churn_analysis()
            elif query_type == 'segmentation':
                result['data'] = self._handle_segmentation()
            elif query_type == 'predictions':
                result['data'] = self._handle_predictions()
            elif query_type == 'recommendations':
                result['data'] = self._handle_recommendations()
            elif query_type == 'general':
                result['data'] = self._handle_general_analysis(question)
            
            # Generate AI insights if requested
            if include_ai_insights and result['data']:
                result['insights'] = self._generate_ai_insights(question, result['data'])
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing query '{question}': {e}")
        
        return result
    
    def _classify_query(self, question: str) -> str:
        """Classify the type of query"""
        question_lower = question.lower()
        
        churn_keywords = ['churn', 'risk', 'leave', 'retention', 'attrition']
        segment_keywords = ['segment', 'group', 'cluster', 'customer types']
        prediction_keywords = ['predict', 'forecast', 'future', 'trend']
        recommendation_keywords = ['recommend', 'suggest', 'action', 'strategy']
        
        if any(keyword in question_lower for keyword in churn_keywords):
            return 'churn_analysis'
        elif any(keyword in question_lower for keyword in segment_keywords):
            return 'segmentation'
        elif any(keyword in question_lower for keyword in prediction_keywords):
            return 'predictions'
        elif any(keyword in question_lower for keyword in recommendation_keywords):
            return 'recommendations'
        else:
            return 'general'
    
    def _handle_churn_analysis(self) -> Dict[str, Any]:
        """Comprehensive churn analysis"""
        return self.analyzer.analyze_churn_patterns()
    
    def _handle_segmentation(self) -> Dict[str, Any]:
        """Customer segmentation analysis"""
        return self.analyzer.perform_segmentation()
    
    def _handle_predictions(self) -> Dict[str, Any]:
        """Predictive analysis"""
        churn_data = self.analyzer.analyze_churn_patterns()
        segments = self.analyzer.perform_segmentation()
        
        return {
            'churn_predictions': churn_data,
            'segment_trends': segments,
            'revenue_forecasts': self._calculate_revenue_forecasts(churn_data)
        }
    
    def _handle_recommendations(self) -> Dict[str, Any]:
        """Strategic recommendations"""
        churn_analysis = self.analyzer.analyze_churn_patterns()
        segments = self.analyzer.perform_segmentation()
        
        return {
            'immediate_actions': self._generate_immediate_actions(churn_analysis),
            'strategic_initiatives': self._generate_strategic_initiatives(segments),
            'investment_priorities': self._generate_investment_priorities(churn_analysis, segments)
        }
    
    def _handle_general_analysis(self, question: str) -> Dict[str, Any]:
        """Handle general data questions"""
        # Basic data overview
        overview = {
            'total_customers': len(self.df),
            'churn_rate': self.df['churn'].mean() * 100 if 'churn' in self.df.columns else None,
            'avg_satisfaction': self.df['satisfaction_score'].mean() if 'satisfaction_score' in self.df.columns else None,
            'data_shape': self.df.shape,
            'columns': list(self.df.columns)
        }
        
        return overview
    
    def _generate_ai_insights(self, question: str, data: Dict[str, Any]) -> str:
        """Generate AI-powered insights"""
        try:
            prompt = f"""
            As a senior customer analytics consultant, analyze this data and provide strategic insights for the question: "{question}"
            
            Data Summary:
            {json.dumps(data, indent=2, default=str)}
            
            Provide:
            1. Key insights (3-5 bullet points)
            2. Business implications
            3. Recommended actions
            4. Success metrics to track
            
            Keep response business-focused and actionable.
            """
            
            response = self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior customer analytics consultant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return "Unable to generate AI insights at this time."
    
    def _calculate_revenue_forecasts(self, churn_data: Dict) -> Dict[str, float]:
        """Calculate revenue impact forecasts"""
        if 'high_risk_customers' not in churn_data:
            return {}
        
        high_risk_revenue = sum([c.get('total_spent', 0) for c in churn_data['high_risk_customers']])
        
        return {
            'revenue_at_risk': high_risk_revenue,
            'potential_loss_3m': high_risk_revenue * 0.3,
            'potential_savings_with_intervention': high_risk_revenue * 0.6
        }
    
    def _generate_immediate_actions(self, churn_analysis: Dict) -> List[str]:
        """Generate immediate action items"""
        actions = []
        
        if churn_analysis.get('overview', {}).get('churn_rate', 0) > 20:
            actions.append("Launch emergency retention campaign for high-risk customers")
        
        if len(churn_analysis.get('high_risk_customers', [])) > 0:
            actions.append("Implement immediate outreach to identified at-risk customers")
        
        actions.extend([
            "Set up automated satisfaction score monitoring",
            "Create early warning alerts for engagement drops",
            "Design targeted offers for high-value at-risk customers"
        ])
        
        return actions
    
    def _generate_strategic_initiatives(self, segments: Dict) -> List[str]:
        """Generate strategic initiatives"""
        return [
            "Develop segment-specific product offerings",
            "Implement personalized customer journey mapping",
            "Create loyalty programs for high-value segments",
            "Design win-back campaigns for churned customers",
            "Build predictive analytics dashboard"
        ]
    
    def _generate_investment_priorities(self, churn_analysis: Dict, segments: Dict) -> List[str]:
        """Generate investment priorities"""
        return [
            "Customer success team expansion",
            "Advanced analytics platform",
            "Personalization engine",
            "Customer feedback systems",
            "Retention marketing automation"
        ]
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "sample_data": self.df.head().to_dict(),
            "basic_stats": self.df.describe().to_dict() if len(self.df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
