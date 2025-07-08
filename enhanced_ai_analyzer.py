# enhanced_ai_analyzer.py - Enhanced version with better output quality
"""
Enhanced AI Customer Intelligence Agent
Features: Better formatting, tool tracking, business insights
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

# Import LangChain components
try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain.agents import AgentType
    from langchain_openai import ChatOpenAI
    from langchain.callbacks import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class ToolTracker(BaseCallbackHandler):
    """Track which tools and operations are used"""
    
    def __init__(self):
        self.tools_used = []
        self.pandas_operations = []
        self.start_time = None
        self.end_time = None
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Record when analysis starts"""
        self.start_time = datetime.now()
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Record tool usage"""
        tool_name = serialized.get("name", "unknown")
        self.tools_used.append({
            "tool": tool_name,
            "input": input_str[:200] + "..." if len(input_str) > 200 else input_str,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Extract pandas operations
        if "python" in tool_name.lower():
            operations = self._extract_pandas_operations(input_str)
            self.pandas_operations.extend(operations)
    
    def on_chain_end(self, outputs, **kwargs):
        """Record when analysis ends"""
        self.end_time = datetime.now()
    
    def _extract_pandas_operations(self, code: str) -> List[str]:
        """Extract pandas operations from code"""
        operations = []
        
        # Common pandas operations to look for
        patterns = [
            r'df\.groupby\([^)]+\)',
            r'df\.agg\([^)]+\)',
            r'df\.mean\(\)',
            r'df\.sum\(\)',
            r'df\.quantile\([^)]+\)',
            r'pd\.cut\([^)]+\)',
            r'df\[.*?\]\.value_counts\(\)',
            r'df\.corr\(\)',
            r'df\.describe\(\)',
            r'df\.pivot_table\([^)]+\)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            operations.extend(matches)
        
        return operations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tool usage"""
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        return {
            "tools_used": self.tools_used,
            "pandas_operations": list(set(self.pandas_operations)),  # Unique operations
            "duration_seconds": round(duration, 2),
            "num_tool_calls": len(self.tools_used)
        }

def create_enhanced_agent(api_key: str, df: pd.DataFrame, model: str = "gpt-4"):
    """Create an enhanced agent with better prompting and tracking"""
    
    # Create callback handler
    tool_tracker = ToolTracker()
    
    # Create LLM with callbacks
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        model=model,
        max_tokens=3000,
        callbacks=[tool_tracker]
    )
    
    # Enhanced system prompt
    enhanced_prompt = """You are an expert data analyst working with customer data.

CRITICAL INSTRUCTIONS:
1. ALWAYS format numbers properly with commas and currency symbols
2. ALWAYS separate text properly (no run-together words)
3. PROVIDE BUSINESS INSIGHTS, not just statistics
4. STRUCTURE your response with clear sections
5. For segments, use MEANINGFUL NAMES (not just Low/Medium/High)

When analyzing segments:
- Use business-friendly names like "Price-Conscious", "Regular Shoppers", "Premium Customers", "VIP Spenders"
- Explain what makes each segment unique
- Provide targeting recommendations

When showing statistics:
- Format currency: $1,234.56
- Format percentages: 45.3%
- Format counts: 1,234 customers
- Use bullet points for clarity

Always end with ACTIONABLE RECOMMENDATIONS."""
    
    # Create agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        prefix=enhanced_prompt,
        suffix="Remember to format all output clearly with proper spacing and provide business insights.",
        max_iterations=15,
        early_stopping_method="force",
        allow_dangerous_code=True,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
            "callbacks": [tool_tracker]
        }
    )
    
    return agent, tool_tracker

def format_analysis_output(raw_output: str) -> str:
    """Fix common formatting issues in the output"""
    
    # Fix run-together words around currency
    raw_output = re.sub(r'(\d+\.?\d*),\*?while', r'\1, while', raw_output)
    raw_output = re.sub(r'(\d+\.?\d*)\*?and\*?(\d+\.?\d*)', r'\1 and \2', raw_output)
    
    # Fix currency formatting
    raw_output = re.sub(r'\$(\d+)\.(\d+)', lambda m: f'${int(m.group(1)):,}.{m.group(2)}', raw_output)
    
    # Fix percentages
    raw_output = re.sub(r'(\d+\.?\d*)%', r'\1%', raw_output)
    
    # Remove weird asterisks
    raw_output = raw_output.replace('*', '')
    
    # Add proper line breaks
    raw_output = raw_output.replace('. ', '.\n')
    
    return raw_output

def analyze_with_enhanced_agent(
    question: str, 
    df: pd.DataFrame, 
    api_key: str,
    model: str = "gpt-3.5-turbo"
) -> Tuple[str, Dict[str, Any]]:
    """Analyze with enhanced agent and return formatted output with tool info"""
    
    if not LANGCHAIN_AVAILABLE:
        return "❌ LangChain not available", {}
    
    try:
        # Create enhanced agent
        agent, tool_tracker = create_enhanced_agent(api_key, df, model)
        
        # Enhance the question based on type
        question_lower = question.lower()
        
        # Question-specific enhancements
        if "segment" in question_lower:
            enhanced_question = f"""{question}

Please:
1. Create meaningful customer segments (e.g., "Budget Shoppers", "Regular Customers", "Premium Buyers", "VIP Clients")
2. Show segment sizes and percentages
3. Describe each segment's characteristics (spending, behavior, demographics)
4. Provide specific targeting strategies for each segment
5. Format all numbers properly with commas and currency symbols"""

        elif "gender" in question_lower:
            enhanced_question = f"""{question}

Please analyze gender differences across:
1. Average spending (formatted as currency)
2. Purchase frequency
3. Satisfaction levels
4. Churn rates
5. Product preferences
6. Statistical significance of differences
7. Business implications and recommendations"""

        elif "spending" in question_lower or "pattern" in question_lower:
            enhanced_question = f"""{question}

Analyze spending patterns including:
1. Total revenue (properly formatted)
2. Customer lifetime value distribution
3. Spending by segments/categories
4. Pareto analysis (80/20 rule)
5. Seasonal or temporal patterns
6. Recommendations to increase revenue"""

        else:
            enhanced_question = f"""{question}

Provide a comprehensive analysis with:
1. Key findings (with properly formatted numbers)
2. Statistical insights
3. Business implications
4. Actionable recommendations"""
        
        # Run analysis
        result = agent.run(enhanced_question)
        
        # Format the output
        formatted_result = format_analysis_output(result)
        
        # Get tool usage summary
        tool_summary = tool_tracker.get_summary()
        
        return formatted_result, tool_summary
        
    except Exception as e:
        return f"❌ Error: {str(e)}", {}

def create_business_focused_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful business segments"""
    
    df = df.copy()
    
    # Value-based segmentation with business names
    if 'total_spent' in df.columns:
        spending_quartiles = df['total_spent'].quantile([0.25, 0.5, 0.75, 0.9])
        
        conditions = [
            df['total_spent'] < spending_quartiles[0.25],
            (df['total_spent'] >= spending_quartiles[0.25]) & (df['total_spent'] < spending_quartiles[0.5]),
            (df['total_spent'] >= spending_quartiles[0.5]) & (df['total_spent'] < spending_quartiles[0.75]),
            (df['total_spent'] >= spending_quartiles[0.75]) & (df['total_spent'] < spending_quartiles[0.9]),
            df['total_spent'] >= spending_quartiles[0.9]
        ]
        
        segment_names = [
            'Price-Conscious Buyers',
            'Occasional Shoppers',
            'Regular Customers',
            'High-Value Customers',
            'VIP Premium Clients'
        ]
        
        df['value_segment'] = np.select(conditions, segment_names, default='Unknown')
    
    # Behavioral segmentation
    if 'satisfaction_score' in df.columns and 'monthly_visits' in df.columns:
        # Create engagement score
        visit_median = df['monthly_visits'].median()
        
        conditions = [
            (df['satisfaction_score'] >= 4) & (df['monthly_visits'] >= visit_median),
            (df['satisfaction_score'] >= 4) & (df['monthly_visits'] < visit_median),
            (df['satisfaction_score'] < 3) & (df['monthly_visits'] >= visit_median),
            (df['satisfaction_score'] < 3) & (df['monthly_visits'] < visit_median)
        ]
        
        behavior_segments = [
            'Loyal Advocates',
            'Satisfied Occasionals',
            'Frequent but Frustrated',
            'At-Risk Detractors'
        ]
        
        df['behavior_segment'] = np.select(conditions, behavior_segments, default='Neutral')
    
    return df

def generate_executive_summary(df: pd.DataFrame, analysis_type: str) -> str:
    """Generate an executive summary based on the data"""
    
    summary = "📊 **EXECUTIVE SUMMARY**\n\n"
    
    # Calculate key metrics
    total_customers = len(df)
    
    if 'total_spent' in df.columns:
        total_revenue = df['total_spent'].sum()
        avg_customer_value = df['total_spent'].mean()
        top_20_percent = df.nlargest(int(0.2 * len(df)), 'total_spent')['total_spent'].sum()
        pareto_ratio = (top_20_percent / total_revenue) * 100
        
        summary += f"**Revenue Metrics:**\n"
        summary += f"• Total Revenue: ${total_revenue:,.2f}\n"
        summary += f"• Average Customer Value: ${avg_customer_value:,.2f}\n"
        summary += f"• Top 20% contribute {pareto_ratio:.1f}% of revenue\n\n"
    
    if 'churn' in df.columns:
        churn_rate = df['churn'].mean() * 100
        at_risk = ((df['churn'] == 0) & (df.get('satisfaction_score', 5) < 3.5)).sum()
        
        summary += f"**Customer Health:**\n"
        summary += f"• Churn Rate: {churn_rate:.1f}%\n"
        summary += f"• At-Risk Customers: {at_risk:,}\n\n"
    
    if 'gender' in df.columns and analysis_type == 'gender':
        gender_insights = []
        for gender in df['gender'].unique():
            gender_data = df[df['gender'] == gender]
            if 'total_spent' in df.columns:
                avg_spend = gender_data['total_spent'].mean()
                gender_insights.append(f"• {gender}: ${avg_spend:,.2f} avg spend")
        
        summary += f"**Gender Insights:**\n"
        summary += '\n'.join(gender_insights) + "\n\n"
    
    return summary

# Enhanced main analysis function
def analyze_with_ai_enhanced(
    question: str, 
    df: pd.DataFrame, 
    api_key: str,
    use_enhanced: bool = True
) -> str:
    """Main entry point for enhanced analysis"""
    
    if not api_key:
        return "⚠️ Please provide an OpenAI API key"
    
    if df is None or len(df) == 0:
        return "⚠️ No data available"
    
    # Add business segments to the dataframe
    df_enhanced = create_business_focused_segments(df)
    
    # Determine analysis type
    question_lower = question.lower()
    if 'gender' in question_lower:
        analysis_type = 'gender'
    elif 'segment' in question_lower:
        analysis_type = 'segment'
    elif 'spending' in question_lower or 'revenue' in question_lower:
        analysis_type = 'revenue'
    else:
        analysis_type = 'general'
    
    # Get executive summary
    exec_summary = generate_executive_summary(df_enhanced, analysis_type)
    
    if use_enhanced and LANGCHAIN_AVAILABLE:
        # Use enhanced agent
        result, tool_info = analyze_with_enhanced_agent(question, df_enhanced, api_key)
        
        # Format final output
        final_output = f"{exec_summary}"
        final_output += f"**AI ANALYSIS:**\n\n{result}\n\n"
        
        # Add tool information
        final_output += "---\n📊 **Analysis Details:**\n"
        final_output += f"• Method: Enhanced Pandas DataFrame Agent\n"
        final_output += f"• Data Points: {len(df):,} customers\n"
        final_output += f"• Duration: {tool_info.get('duration_seconds', 0):.1f} seconds\n"
        final_output += f"• Tool Calls: {tool_info.get('num_tool_calls', 0)}\n\n"
        
        if tool_info.get('pandas_operations'):
            final_output += "**Pandas Operations Used:**\n"
            for op in tool_info['pandas_operations'][:10]:  # Show first 10
                final_output += f"• `{op}`\n"
        
        return final_output
    else:
        # Fallback to simpler analysis
        return analyze_with_precomputed_insights(question, df_enhanced, api_key, exec_summary)

def analyze_with_precomputed_insights(
    question: str, 
    df: pd.DataFrame, 
    api_key: str,
    exec_summary: str
) -> str:
    """Fallback analysis with pre-computed insights"""
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Pre-compute all insights
        insights = {}
        
        # Gender analysis
        if 'gender' in df.columns:
            gender_stats = {}
            for gender in df['gender'].unique():
                gender_df = df[df['gender'] == gender]
                gender_stats[gender] = {
                    'count': len(gender_df),
                    'avg_spent': float(gender_df['total_spent'].mean()) if 'total_spent' in df.columns else 0,
                    'avg_satisfaction': float(gender_df['satisfaction_score'].mean()) if 'satisfaction_score' in df.columns else 0,
                    'churn_rate': float(gender_df['churn'].mean() * 100) if 'churn' in df.columns else 0,
                    'avg_visits': float(gender_df['monthly_visits'].mean()) if 'monthly_visits' in df.columns else 0
                }
            insights['gender_analysis'] = gender_stats
        
        # Segment analysis
        if 'value_segment' in df.columns:
            segment_stats = {}
            for segment in df['value_segment'].unique():
                seg_df = df[df['value_segment'] == segment]
                segment_stats[segment] = {
                    'count': len(seg_df),
                    'percentage': float(len(seg_df) / len(df) * 100),
                    'avg_spent': float(seg_df['total_spent'].mean()) if 'total_spent' in df.columns else 0,
                    'churn_rate': float(seg_df['churn'].mean() * 100) if 'churn' in df.columns else 0
                }
            insights['segment_analysis'] = segment_stats
        
        # Create prompt
        prompt = f"""
Based on this customer data analysis:

{json.dumps(insights, indent=2)}

Question: {question}

Provide a comprehensive business-focused answer with:
1. Key findings with properly formatted numbers (use $ for currency, % for percentages)
2. Statistical insights
3. Business implications
4. Specific, actionable recommendations

Format numbers properly and ensure clear, professional presentation.
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a senior business analyst providing executive-level insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content
        
        return f"""{exec_summary}**AI ANALYSIS:**

{result}

---
📊 **Analysis Details:**
• Method: Pre-computed Insights + OpenAI
• Data Points: {len(df):,} customers
• Operations: Statistical aggregations, segmentation, correlation analysis
"""
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Export the main function
__all__ = ['analyze_with_ai_enhanced', 'create_business_focused_segments']
