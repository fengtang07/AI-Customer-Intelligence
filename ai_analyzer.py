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

# Disable enhanced analyzer to avoid formatting issues
ENHANCED_AVAILABLE = False

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
            model="gpt-4o",
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
            "model": "gpt-4o"
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

def fix_text_formatting(text: str) -> str:
    """Fix common text formatting issues"""
    import re
    
    # Fix run-together numbers and text
    text = re.sub(r'(\d+\.?\d*),([a-zA-Z])', r'\1, \2', text)  # Add space after comma+number
    text = re.sub(r'(\d+\.?\d*)([a-zA-Z])', r'\1 \2', text)    # Add space between number and letter
    text = re.sub(r'([a-zA-Z])(\d+\.?\d*)', r'\1 \2', text)    # Add space between letter and number
    
    # Fix specific problematic patterns from segmentation
    text = re.sub(r'(\d+)and(\d+)', r'\1 and \2', text)        # Fix "200and500" -> "200 and 500"
    text = re.sub(r'\$(\d+\.?\d*),with', r'$\1, with', text)
    text = re.sub(r'(\d+\.?\d*),with', r'\1, with', text)
    text = re.sub(r'amountingto\$', r'amounting to $', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
    text = re.sub(r'spending(\d+)', r'spending $\1', text)     # Fix "spending200" -> "spending $200"
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def analyze_with_langchain_improved(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Improved version with simpler, more reliable processing"""
    
    if not api_key:
        return "⚠️ No API key provided"
    
    try:
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain.agents import AgentType
        from langchain_openai import ChatOpenAI
        
        # Track start time
        start_time = datetime.now()
        
        # Use GPT-4 with specific formatting controls
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model="gpt-4o",
            max_tokens=2500,
            model_kwargs={
                "frequency_penalty": 0.1,  # Reduce repetitive text
                "presence_penalty": 0.1    # Encourage diverse responses
            }
        )
        
        # Create agent with enhanced, specific instructions matching Direct OpenAI
        enhanced_prefix = f"""You are a customer data analyst. Your job is to analyze customer data patterns and provide business insights.

Dataset Overview: {len(df):,} customers
Columns available: {', '.join(df.columns)}

CRITICAL ANALYSIS REQUIREMENTS - FOLLOW EXACTLY:

For "at risk" questions:
- Analyze customers with churn=0 who show warning signs (low satisfaction < 3.0, declining engagement, etc.)
- NEVER analyze individual customers by ID - always analyze PATTERNS across customer segments
- Identify segments like "customers with satisfaction < 3.0 AND visits < 5"
- Provide statistical significance and sample sizes

For "segments" questions:
- Analyze the ACTUAL DATA to create comprehensive segmentation
- Calculate quartiles for spending, age groups for demographics, and engagement levels
- Provide SPECIFIC NUMBERS: customer counts, dollar ranges, score ranges
- Create business-meaningful segments with actual data-driven insights
- Don't just list product categories or give generic descriptions

For all questions:
- Use proper spacing in text (write "customers spend $500" NOT "customersspend$500")
- Include confidence levels and statistical tests when possible
- Focus on actionable insights for customer segments with measurable outcomes
- Provide specific numbers, percentages, and dollar amounts
- Give business implications and concrete recommendations

RESPONSE FORMAT:
1. Key findings with specific statistics
2. Segment analysis with sample sizes
3. Statistical insights with confidence levels
4. Business implications for each segment
5. Actionable recommendations with expected outcomes

Remember: ALWAYS analyze customer segments and patterns, NEVER individual customers."""
        
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            prefix=enhanced_prefix,
            max_iterations=8,
            early_stopping_method="force",
            allow_dangerous_code=True
        )
        
        # Run analysis with enhanced question based on type
        question_lower = question.lower()
        
        if "risk" in question_lower:
            enhanced_question = f"""{question}

CRITICAL: Analyze customers with churn=0 who show warning signs. Do NOT analyze individual customers by ID.

STEP 1: First examine the data distribution:
- Check satisfaction_score.describe() for the dataset
- Check monthly_visits.describe() for the dataset  
- Check total_spent.describe() for the dataset

STEP 2: Use DATA-DRIVEN thresholds based on actual distribution:
- Low satisfaction: Below 25th percentile or bottom quartile of satisfaction scores
- Low engagement: Below 25th percentile of monthly visits
- Low spending: Below median total_spent
- Combined risk factors: Multiple warning signs together

STEP 3: If the dataset has generally high satisfaction, use relative thresholds:
- Satisfaction in bottom 20% of all customers with churn=0  
- Visits in bottom 30% of all customers with churn=0
- Any demographic patterns that correlate with higher risk

Always find SOME at-risk segments by adjusting thresholds to the actual data distribution."""
            
        elif "segment" in question_lower:
            enhanced_question = f"""{question}

CRITICAL: Analyze the ACTUAL DATA to create comprehensive customer segmentation.

REQUIRED ANALYSIS STEPS:
1. VALUE SEGMENTATION: Calculate quartiles of total_spent and create 4 spending segments with:
   - Exact dollar ranges (e.g., "$100 to $500")
   - Customer counts for each segment
   - Average spending per segment

2. DEMOGRAPHIC SEGMENTATION: Analyze by age and gender with:
   - Age group breakdowns with customer counts
   - Gender distribution with average metrics
   - Cross-segmentation (age x gender) with specific numbers

3. BEHAVIORAL SEGMENTATION: Analyze engagement patterns with:
   - Visit frequency segments (low/medium/high) with exact visit ranges
   - Satisfaction score segments with score ranges and customer counts
   - Churn patterns by segment

4. BUSINESS VALUE SEGMENTS: Combine multiple factors to create actionable segments like:
   - "High-Value Loyalists" (high spend + high satisfaction)
   - "At-Risk High Spenders" (high spend + low satisfaction)
   - "Growth Potential" (medium spend + high satisfaction)

Provide SPECIFIC NUMBERS for each segment, not generic descriptions."""
            
        else:
            enhanced_question = f"""{question}

Analyze customer patterns and segments with statistical rigor. Provide business insights and actionable recommendations."""
        
        # Run analysis
        raw_result = agent.run(enhanced_question)
        
        # Fix text formatting issues
        raw_result = fix_text_formatting(raw_result)
        
        # Validate result quality and add warnings if needed
        if "risk" in question_lower:
            if "CUST_" in raw_result:
                raw_result = f"""⚠️ ANALYSIS QUALITY WARNING: This response analyzed individual customers instead of segments.

{raw_result}

📋 RECOMMENDED APPROACH: For at-risk analysis, focus on customer segments with specific characteristics (e.g., satisfaction < 3.0, visits < 5) rather than individual customer IDs."""
            
            elif "no customers" in raw_result.lower() or "no at-risk" in raw_result.lower():
                raw_result = f"""⚠️ DATA ANALYSIS WARNING: This response found no at-risk customers, which is unusual for a large dataset.

{raw_result}

📋 RECOMMENDED APPROACH: Use data-driven thresholds based on percentiles (bottom 25% satisfaction, bottom 25% engagement) rather than fixed values. In a dataset of {len(df):,} customers, there should typically be identifiable risk segments."""
        
        # Additional validation for segmentation responses
        if "segment" in question_lower and ("Low Spender:" in raw_result or "spending less than" in raw_result) and len([x for x in raw_result.split('\n') if x.strip() and ('customer' in x.lower() or 'count' in x.lower())]) < 3:
            raw_result = f"""⚠️ SEGMENTATION QUALITY WARNING: This response provided generic segment descriptions without actual data analysis.

{raw_result}

📋 RECOMMENDED APPROACH: Analyze the actual dataset to provide specific customer counts, dollar ranges, and data-driven segment characteristics rather than theoretical descriptions."""
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Add analysis metadata without complex formatting
        final_output = f"""📊 CUSTOMER INTELLIGENCE ANALYSIS

{raw_result}

---
📈 Analysis Details:
• Method: Pandas DataFrame Agent (Enhanced)
• Model: {llm.model_name}
• Data Points: {len(df)} customers
• Processing Time: {duration:.1f} seconds
• Key Operations: groupby, aggregation, segmentation, statistical analysis

💡 Note: This analysis used automated data exploration to identify patterns and generate insights."""
        
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
    

# MAIN FUNCTION: Use Direct OpenAI as default for better results
def analyze_with_ai(question: str, df: pd.DataFrame, api_key: str, use_langchain: bool = False, response_style: str = 'smart'):
    """Main AI analysis function with improved implementation"""
    if not api_key:
        return "⚠️ Please enter your OpenAI API key to enable AI analysis."
    
    if df is None or len(df) == 0:
        return "⚠️ No data available for analysis. Please upload a dataset first."
    
    # Use Direct OpenAI by default for reliability
    if use_langchain:
        try:
            return analyze_with_langchain_improved(question, df, api_key, response_style)
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
        # Fallback to direct OpenAI instead of just showing error
        return analyze_with_direct_openai(question, df, api_key, response_style)

def analyze_with_direct_openai(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Direct OpenAI analysis with reliable prompting"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
        
        # Create clean data summary without complex formatting
        total_customers = len(df)
        total_revenue = df['total_spent'].sum() if 'total_spent' in df.columns else 0
        avg_customer_value = df['total_spent'].mean() if 'total_spent' in df.columns else 0
        churn_rate = df['churn'].mean() * 100 if 'churn' in df.columns else 0
        
        # Enhanced data summary with proper formatting
        churned_count = df['churn'].sum() if 'churn' in df.columns else 0
        active_count = len(df) - churned_count if 'churn' in df.columns else len(df)
        sat_min = df['satisfaction_score'].min() if 'satisfaction_score' in df.columns else 0
        sat_max = df['satisfaction_score'].max() if 'satisfaction_score' in df.columns else 0
        
        data_summary = f"""Customer Database Overview:
Total Customers: {total_customers:,}
Total Revenue: ${total_revenue:,.2f}
Average Customer Value: ${avg_customer_value:.2f}
Churn Rate: {churn_rate:.1f}%
Available columns: {', '.join(df.columns)}

Key Statistics:
- Customers who have churned: {churned_count}
- Active customers: {active_count}
- Satisfaction range: {sat_min:.1f} to {sat_max:.1f}

Sample of first 3 customer records:
{df.head(3).to_string()}

Key Data Distributions for At-Risk Analysis:
- Satisfaction Score Distribution: {df['satisfaction_score'].describe().to_dict() if 'satisfaction_score' in df.columns else 'N/A'}
- Monthly Visits Distribution: {df['monthly_visits'].describe().to_dict() if 'monthly_visits' in df.columns else 'N/A'}
- Total Spent Distribution: {df['total_spent'].describe().to_dict() if 'total_spent' in df.columns else 'N/A'}"""
        
        # Enhanced prompt with specific instructions
        prompt = f"""You are a customer analytics consultant. Analyze this customer data and answer the question with specific insights and recommendations.

{data_summary}

Question: {question}

CRITICAL ANALYSIS REQUIREMENTS:
- For "at risk" questions: Analyze customers with churn=0 who show warning signs. Use DATA-DRIVEN thresholds based on the actual distribution (bottom 25% satisfaction, bottom 25% engagement, etc.) rather than fixed values. NEVER analyze individual customers - always provide segment-level patterns.
- For "churned" questions: Analyze customers with churn=1 to understand patterns of why they left
- For "segments" questions: Provide comprehensive segmentation by demographics, behavior, value, and risk levels
- FORMATTING: CRITICAL - Always separate numbers and text with spaces. Write "customers spend $500" NOT "customersspend$500". Put spaces around dollar amounts: "$1,500, with revenue" NOT "$1,500,withrevenue". Use proper punctuation and spacing between all words.
- STATISTICAL RIGOR: Include confidence levels, sample sizes, and significance tests when possible
- BUSINESS FOCUS: Provide actionable insights for customer segments with measurable expected outcomes

Provide:
1. Key findings with specific numbers and percentages
2. Statistical insights with confidence levels where relevant
3. Business implications for different customer segments
4. Specific, measurable recommendations with expected outcomes

Use clear, professional language with proper spacing between all words.

CRITICAL TEXT FORMATTING RULES:
- Always put spaces around dollar amounts: "$1,500, with total revenue" NOT "$1,500,withtotalrevenue"
- Separate numbers from text: "customers spend $500 on average" NOT "customersspend$500onaverage"  
- Use proper punctuation and spacing between ALL words
- Double-check that no words are run together without spaces"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior customer analytics consultant. Provide clear, data-driven insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        
        # Post-process to fix any remaining formatting issues
        result = fix_text_formatting(result)
        
        return f"""📊 CUSTOMER INTELLIGENCE ANALYSIS

{result}

---
📈 Analysis Details:
• Method: Direct OpenAI Analysis
• Model: gpt-4o
• Data Points: {len(df)} customers
• Revenue Context: {total_revenue:.2f} dollars total customer value
• Focus: Data-driven insights and recommendations"""
        
    except Exception as e:
        return f"❌ Direct OpenAI analysis error: {str(e)}"
