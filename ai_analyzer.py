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
            agent_executor_kwargs={
                "handle_parsing_errors": True
            }
        )
        
        return agent
        
    except ImportError as e:
        return f"❌ LangChain import error: {str(e)}"
    except Exception as e:
        return f"❌ Agent setup error: {str(e)}"

# IMPROVED: Simplified analysis function
def analyze_with_langchain(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Analyze customer data using improved LangChain agent"""
    if not api_key:
        return "⚠️ No API key provided"
    
    try:
        agent = setup_langchain_agent(api_key, df)
        
        if isinstance(agent, str):
            return f"❌ LangChain setup failed: {agent}"
        
        # Map common questions to specific prompts for better results
        question_lower = question.lower()
        
        if "churn" in question_lower and ("why" in question_lower or "driver" in question_lower):
            enhanced_question = f"""
{question}

To answer this:
1. Calculate the correlation between churn and other numeric features
2. Compare average values between churned (churn=1) and retained (churn=0) customers
3. Identify the top 3 factors that differ most between churned and retained customers
4. Provide specific numbers and percentages
"""
        elif "segment" in question_lower:
            enhanced_question = f"""
{question}

To answer this:
1. Create customer segments based on total_spent using quartiles or meaningful thresholds
2. Show the size of each segment and their characteristics
3. Calculate key metrics (churn rate, avg satisfaction) for each segment
4. Use pandas groupby operations and show specific numbers
"""
        elif "risk" in question_lower or "at risk" in question_lower:
            enhanced_question = f"""
{question}

To answer this:
1. Define at-risk as: active customers (churn=0) with low satisfaction (<3.5) or low engagement
2. Count how many customers meet these criteria
3. Show their average spending and other characteristics
4. Use pandas filtering and provide specific counts
"""
        elif "spending" in question_lower or "revenue" in question_lower or "value" in question_lower:
            enhanced_question = f"""
{question}

To answer this:
1. Calculate total revenue, average customer value, and distribution
2. Find the top 20% customers' contribution to revenue
3. Analyze spending patterns by segments or categories
4. Show specific dollar amounts and percentages
"""
        elif "satisfaction" in question_lower:
            enhanced_question = f"""
{question}

To answer this:
1. Calculate satisfaction statistics (mean, distribution)
2. Show satisfaction by customer segments
3. Analyze correlation between satisfaction and churn
4. Provide specific scores and percentages
"""
        else:
            enhanced_question = f"""
{question}

Please provide specific numbers, calculations, and data-driven insights.
"""
        
        # Run the agent
        result = agent.run(enhanced_question)
        
        # Format the result based on style
        if response_style == 'concise':
            formatted_result = f"""**Analysis Result:**
{result}"""
        else:
            formatted_result = f"""🤖 **LangChain Agent Analysis:**

{result}

✅ **Method:** Pandas DataFrame Agent
📊 **Data Points:** {len(df):,} customers analyzed
"""
        
        return formatted_result
        
    except Exception as e:
        return f"❌ LangChain analysis error: {str(e)}\n\n{traceback.format_exc()}"

# IMPROVED: Direct OpenAI analysis
def analyze_with_direct_openai(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Direct OpenAI analysis - simplified version"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=3)
        
        # Create data summary
        data_summary = f"""
Dataset Overview:
- Total customers: {len(df):,}
- Columns: {list(df.columns)}
"""
        
        if 'churn' in df.columns:
            data_summary += f"- Churn rate: {df['churn'].mean()*100:.1f}%\n"
        
        if 'total_spent' in df.columns:
            data_summary += f"- Avg spending: ${df['total_spent'].mean():.2f}\n"
            
        if 'satisfaction_score' in df.columns:
            data_summary += f"- Avg satisfaction: {df['satisfaction_score'].mean():.2f}/5.0\n"
        
        data_summary += f"\nSample data:\n{df.head(5).to_string()}"
        
        prompt = f"""
You are a data analyst. Based on this customer data:

{data_summary}

Question: {question}

Provide:
1. Direct answer with specific numbers
2. Key insights
3. Actionable recommendations

Be specific and use the actual data provided.
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst specializing in customer analytics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return f"""🤖 **Direct OpenAI Analysis:**

{response.choices[0].message.content}

✅ **Method:** Direct OpenAI API
📊 **Data Points:** {len(df):,} customers analyzed
"""
        
    except Exception as e:
        return f"❌ Direct OpenAI error: {str(e)}"

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
