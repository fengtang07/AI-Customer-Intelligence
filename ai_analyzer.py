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

def analyze_with_langchain(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Use enhanced analysis"""
    return analyze_with_ai_enhanced(question, df, api_key, use_enhanced=True)

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
