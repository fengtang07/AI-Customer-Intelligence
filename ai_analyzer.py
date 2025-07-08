# ai_analyzer.py - Fixed AI Analysis Module
"""
AI Customer Intelligence Agent - Improved AI Analysis Module
Uses pandas dataframe agent for better performance
"""

import pandas as pd
import numpy as np
import sys
import traceback
import platform
from typing import Dict, Any, Optional, List
import warnings

from enhanced_ai_analyzer import (
    analyze_with_ai_enhanced,
    create_business_focused_segments
)

warnings.filterwarnings('ignore')

# Keep your existing debug and test functions
def debug_environment():
    """Debug environment information"""
    try:
        debug_info = f"""
🔧 ENVIRONMENT DEBUG INFO:
Python Version: {platform.python_version()}
Platform: {platform.platform()}
Pandas Version: {pd.__version__}
NumPy Version: {np.__version__}

📦 PACKAGE AVAILABILITY:
"""
        packages = {
            'openai': 'OpenAI API',
            'langchain': 'LangChain Framework',
            'langchain_openai': 'LangChain OpenAI',
            'streamlit': 'Streamlit'
        }

        for package, description in packages.items():
            try:
                __import__(package)
                debug_info += f"✅ {description} ({package}): Available\n"
            except ImportError:
                debug_info += f"❌ {description} ({package}): Not installed\n"

        return debug_info

    except Exception as e:
        return f"❌ Debug error: {str(e)}"

def test_ai_connection(api_key: str) -> Dict[str, Any]:
    """Test AI connection with simple query"""
    if not api_key:
        return {"success": False, "error": "No API key provided"}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'AI connection successful' if you can read this."}
            ],
            max_tokens=50,
            temperature=0
        )

        result = response.choices[0].message.content

        return {
            "success": True,
            "result": f"✅ Connection successful! Response: {result}",
            "model": "gpt-3.5-turbo"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# NEW: Simplified LangChain setup using pandas dataframe agent
def setup_langchain_agent(api_key: str, df: pd.DataFrame):
    """Set up improved LangChain agent for data analysis"""
    try:
        # Fixed import - moved to experimental
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain.agents import AgentType
        from langchain_openai import ChatOpenAI
        
        # Use ChatOpenAI with a good model
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model="gpt-3.5-turbo",  # or "gpt-4" for better results
            max_tokens=2000
        )
        
        # Use pandas dataframe agent - it's MUCH better for data analysis!
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            prefix="""You are a data analyst. The dataframe 'df' contains customer data.
Key columns: customer_id, age, gender, total_spent, monthly_visits, satisfaction_score, churn, product_category.

Always start by understanding the data structure, then answer the specific question with numbers.""",
            max_iterations=10,
            early_stopping_method="force",
            allow_dangerous_code=True,  # Required for pandas agent
            agent_executor_kwargs={
                "handle_parsing_errors": True
            }
        )
        
        return agent
        
    except ImportError as e:
        return f"❌ LangChain import error: {str(e)}"
    except Exception as e:
        return f"❌ Agent setup error: {str(e)}"



# Add this to your existing ai_analyzer.py

import re
from datetime import datetime

def format_agent_output(raw_output: str) -> str:
    """Fix formatting issues in agent output"""
    
    # Fix run-together words
    raw_output = re.sub(r'(\d+\.?\d*),\*?while', r'\1, while ', raw_output)
    raw_output = re.sub(r'spend(\d+)', r'spend $\1', raw_output)
    raw_output = re.sub(r'(\d+)\*?and\*?(\d+)', r'\1 and \2', raw_output)
    
    # Fix currency formatting
    def format_currency(match):
        amount = float(match.group(1))
        return f'${amount:,.2f}'
    
    raw_output = re.sub(r'\$?(\d+\.?\d*)', format_currency, raw_output)
    
    # Remove asterisks
    raw_output = raw_output.replace('*', '')
    
    return raw_output

def analyze_with_langchain_improved(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Improved version with better prompts and formatting"""
    
    if not api_key:
        return "⚠️ No API key provided"
    
    try:
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain.agents import AgentType
        from langchain_openai import ChatOpenAI
        
        # Track start time
        start_time = datetime.now()
        
        # Better model and temperature
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model="gpt-4" if "gpt-4" in api_key else "gpt-3.5-turbo",  # Use GPT-4 if available
            max_tokens=3000
        )
        
        # Create agent with better instructions
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            prefix="""You are a senior data analyst. 

CRITICAL FORMATTING RULES:
- Format all currency with $ and commas: $1,234.56
- Format all percentages with %: 45.3%
- Use proper spacing between words
- Use bullet points for lists
- NO asterisks in output

For customer segments, use business-friendly names like:
- "Budget Conscious" instead of "Low"
- "Regular Shoppers" instead of "Medium"  
- "Premium Customers" instead of "High"
- "VIP Clients" instead of "Very High"

Always provide:
1. Statistical findings
2. Business insights
3. Actionable recommendations""",
            max_iterations=15,
            early_stopping_method="force",
            allow_dangerous_code=True
        )
        
        # Enhance questions for better results
        question_lower = question.lower()
        
        if "segment" in question_lower:
            enhanced_q = f"""{question}

Create meaningful business segments and for each segment provide:
- Segment name (business-friendly)
- Size and percentage of customers
- Average metrics (spending, satisfaction, churn rate)
- Key characteristics
- Marketing recommendations"""
            
        elif "gender" in question_lower:
            enhanced_q = f"""{question}

Analyze gender differences including:
- Spending patterns (with proper $ formatting)
- Behavioral differences
- Statistical significance
- Business implications"""
            
        elif "spending" in question_lower:
            enhanced_q = f"""{question}

Analyze spending with:
- Total revenue (formatted with $ and commas)
- Customer value distribution
- 80/20 analysis
- Category breakdowns
- Growth opportunities"""
        else:
            enhanced_q = question + "\n\nFormat all numbers properly and provide business insights."
        
        # Run analysis
        raw_result = agent.run(enhanced_q)
        
        # Format the output
        formatted_result = format_agent_output(raw_result)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Add analysis metadata
        final_output = f"""📊 **CUSTOMER INTELLIGENCE ANALYSIS**

{formatted_result}

---
📈 **Analysis Details:**
• Method: Pandas DataFrame Agent (Enhanced)
• Model: {llm.model_name}
• Data Points: {len(df):,} customers
• Processing Time: {duration:.1f} seconds
• Key Operations: groupby, aggregation, segmentation, statistical analysis

💡 **Note**: This analysis used automated data exploration to identify patterns and generate insights."""
        
        return final_output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Also create a pre-calculated segment function
def get_better_segments(df: pd.DataFrame) -> dict:
    """Create business-friendly segments"""
    
    segments = {}
    
    if 'total_spent' in df.columns:
        # Calculate spending quartiles
        q1, q2, q3 = df['total_spent'].quantile([0.25, 0.5, 0.75])
        
        # Create segments with meaningful names
        segments['spending_segments'] = {
            'Budget Conscious': {
                'criteria': f'Spending < ${q1:,.2f}',
                'count': len(df[df['total_spent'] < q1]),
                'avg_spent': df[df['total_spent'] < q1]['total_spent'].mean(),
                'profile': 'Price-sensitive customers looking for deals'
            },
            'Occasional Shoppers': {
                'criteria': f'${q1:,.2f} - ${q2:,.2f}',
                'count': len(df[(df['total_spent'] >= q1) & (df['total_spent'] < q2)]),
                'avg_spent': df[(df['total_spent'] >= q1) & (df['total_spent'] < q2)]['total_spent'].mean(),
                'profile': 'Infrequent buyers with moderate budgets'
            },
            'Regular Customers': {
                'criteria': f'${q2:,.2f} - ${q3:,.2f}',
                'count': len(df[(df['total_spent'] >= q2) & (df['total_spent'] < q3)]),
                'avg_spent': df[(df['total_spent'] >= q2) & (df['total_spent'] < q3)]['total_spent'].mean(),
                'profile': 'Consistent shoppers with good spending power'
            },
            'Premium Buyers': {
                'criteria': f'Spending > ${q3:,.2f}',
                'count': len(df[df['total_spent'] >= q3]),
                'avg_spent': df[df['total_spent'] >= q3]['total_spent'].mean(),
                'profile': 'High-value customers seeking quality'
            }
        }
    
    return segments
    

# MAIN FUNCTION: Keep the same interface
def analyze_with_ai(question: str, df: pd.DataFrame, api_key: str, use_langchain: bool = True, response_style: str = 'smart'):
    """Main AI analysis function with improved implementation"""
    if not api_key:
        return "⚠️ Please enter your OpenAI API key to enable AI analysis."
    
    if df is None or len(df) == 0:
        return "⚠️ No data available for analysis. Please upload a dataset first."
    
    if use_langchain:
        try:
            return analyze_with_langchain(question, df, api_key, response_style)
        except Exception as e:
            # Fall back to direct OpenAI if LangChain fails
            fallback_result = analyze_with_direct_openai(question, df, api_key, response_style)
            return f"⚠️ LangChain failed, using direct OpenAI:\n{str(e)}\n\n{fallback_result}"
    else:
        return analyze_with_direct_openai(question, df, api_key, response_style)

# Test function for debugging
def test_both_methods(api_key: str):
    """Test both LangChain and direct OpenAI methods"""
    if not api_key:
        return "❌ No API key provided"
    
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'customer_id': [f'CUST_{i:03d}' for i in range(100)],
        'age': np.random.randint(20, 70, 100),
        'total_spent': np.random.uniform(100, 2000, 100),
        'monthly_visits': np.random.poisson(8, 100),
        'satisfaction_score': np.random.uniform(1, 5, 100),
        'churn': np.random.choice([0, 1], 100, p=[0.75, 0.25]),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], 100)
    })
    
    question = "What are the main drivers of customer churn?"
    
    print("🧪 Testing Improved LangChain Method:")
    print("=" * 50)
    langchain_result = analyze_with_langchain(question, test_df, api_key)
    print(langchain_result)
    
    print("\n🧪 Testing Direct OpenAI Method:")
    print("=" * 50)
    direct_result = analyze_with_direct_openai(question, test_df, api_key)
    print(direct_result)

if __name__ == "__main__":
    print("🔧 AI Analyzer Module (Improved Version)")
    print("Debug Environment:")
    print(debug_environment())
    print("\n✅ Module loaded successfully. Add API key to test AI functions.")

# Function aliases for Streamlit app compatibility
def analyze_with_langchain(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """LangChain analysis - alias for improved version"""
    try:
        return analyze_with_langchain_improved(question, df, api_key, response_style)
    except Exception as e:
        return f"❌ LangChain analysis error: {str(e)}"

def analyze_with_direct_openai(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Direct OpenAI analysis - fallback to main AI function"""
    try:
        return analyze_with_ai(question, df, api_key, use_langchain=False, response_style=response_style)
    except Exception as e:
        return f"❌ Direct OpenAI analysis error: {str(e)}"
