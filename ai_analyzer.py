# ai_analyzer.py - Enhanced AI Analysis Module with LangGraph
"""
AI Customer Intelligence Agent - Enhanced AI Analysis Module
Now includes LangGraph-based multi-step analysis workflow with improved reliability.
"""

import pandas as pd
import numpy as np
import sys
import traceback
import platform
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import warnings
import operator
import os

# Enhanced analyzer available with LangGraph
ENHANCED_AVAILABLE = True

warnings.filterwarnings('ignore')

def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets, environment variables, or hardcoded fallback"""
    try:
        # First try to get from Streamlit secrets
        import streamlit as st
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    
    # Fall back to environment variable
    env_key = os.environ.get('OPENAI_API_KEY', None)
    if env_key:
        return env_key
    
    # Final fallback - hardcoded key for testing
    # return "YOUR_FALLBACK_API_KEY_HERE" 

# Get the API key
OPENAI_API_KEY = get_openai_api_key()

# Set environment variable if we have a key
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# LangGraph State Definition
class GraphState(TypedDict):
    """
    Represents the state of our analysis graph.
    """
    user_question: str
    plan: str
    intermediate_steps: Annotated[List[str], operator.add]
    retrieved_data: List[str]
    analysis_result: str
    visualization_code: str
    error_log: List[str]
    retry_count: int
    dataframe_context: str

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
            'langgraph': 'LangGraph',
            'streamlit': 'Streamlit',
            'tabulate': 'Tabulate'
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
        api_key = OPENAI_API_KEY
        
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

# LangGraph Enhanced Analysis System
def setup_langgraph_analyzer(df: pd.DataFrame, api_key: str = None):
    """Set up the LangGraph-based analyzer"""
    try:
        if not api_key:
            api_key = OPENAI_API_KEY
            
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentType
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        from langchain_core.messages import HumanMessage
        
        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            openai_api_key=api_key,
            max_tokens=2000
        )

        # Create Pandas DataFrame Agent
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            allow_dangerous_code=True,
            prefix=f"""You are a customer data analyst working with an e-commerce dataset.

IMPORTANT: The dataset is already loaded as a pandas DataFrame named 'df'. 
DO NOT try to read from any CSV files or external data sources.
Use the existing DataFrame 'df' directly for all analysis.

Dataset Overview:
- DataFrame name: df
- Total customers: {len(df)}
- Columns: {', '.join(df.columns)}
- Data types: {df.dtypes.to_dict()}
- Sample data: {df.head(2).to_dict()}

Your task is to provide data-driven insights for customer intelligence and business strategy.
Always use the existing DataFrame 'df' and support your findings with specific numbers and statistical analysis.

Example usage: df.describe(), df['churn'].mean(), df.groupby('gender')['churn'].mean(), etc.""",
            max_iterations=8,
            early_stopping_method="force"
        )

        # Define graph nodes with enhanced tracking
        def planner_node(state: GraphState):
            """Creates a step-by-step analysis plan with detailed tracking"""
            print("📋 STEP 1: Strategic Planning - Creating analysis roadmap...")
            
            prompt = f"""
            You are an expert customer analytics consultant. Create a detailed, actionable analysis plan for this question:
            
            Question: {state['user_question']}
            
            Dataset context: {state['dataframe_context']}
            
            Create a clear plan with 3-4 specific steps that will provide comprehensive insights:
            1. Data exploration and statistical overview
            2. Advanced statistical analysis (correlations, significance tests)
            3. Customer segmentation and pattern identification
            4. Business recommendations and strategic insights
            
            Make each step specific and measurable.
            
            If there were previous errors, adjust your plan accordingly:
            Error Log: {state.get('error_log', [])}
            
            Comprehensive Analysis Plan:
            """
            
            plan_response = llm.invoke([HumanMessage(content=prompt)])
            print(f"✅ Strategic plan created: {len(plan_response.content)} characters")
            
            return {
                "plan": plan_response.content,
                "retry_count": 0,
                "error_log": []
            }

        def analyzer_node(state: GraphState):
            """Executes the analysis using the pandas agent with progress tracking"""
            try:
                print("🔍 STEP 2: Data Analysis Execution - Running statistical analysis...")
                
                analysis_prompt = f"""
                Execute this comprehensive analysis plan: {state['plan']}
                
                Primary question: {state['user_question']}
                
                CRITICAL: Use the existing pandas DataFrame 'df' that is already loaded in memory.
                DO NOT attempt to read from CSV files or any external data sources.
                
                Requirements:
                1. Use df.describe(), df.info(), df.head() to explore the data
                2. Conduct correlation analysis using df.corr()
                3. Identify customer segments with df.groupby() operations
                4. Calculate business metrics like df['churn'].mean() for churn rate
                5. Provide specific numbers, percentages, and statistical measures
                
                ---
                **CRITICAL FINAL INSTRUCTION:**
                After performing all necessary calculations and data manipulations, you MUST conclude your response with a comprehensive, multi-paragraph textual summary of your findings.
                - Explain what the data and your calculations mean.
                - Address the original user question directly.
                - **Do NOT end your response with raw code output, a list, or a raw data table.** Your final output must be a well-written summary.
                ---
                
                Start your analysis with: df.head() and df.describe()
                """
                
                print("⚙️ Executing pandas agent analysis with .invoke...")
                # The .invoke method returns a dictionary, the answer is in the 'output' key.
                response_dict = pandas_agent.invoke({"input": analysis_prompt})
                result = response_dict.get('output', 'No output found.')

                print(f"✅ Analysis completed: {len(result)} characters of insights generated")
                
                return {
                    "intermediate_steps": [f"Statistical analysis completed: {len(result)} characters"],
                    "retrieved_data": [result]
                }
                
            except Exception as e:
                error_msg = f"Analysis error: {str(e)}"
                print(f"❌ Analysis failed: {error_msg}")
                return {
                    "intermediate_steps": [error_msg],
                    "error_log": [error_msg],
                    "retry_count": state.get("retry_count", 0) + 1
                }

        def validator_node(state: GraphState):
            """Validates the analysis results with comprehensive quality checks"""
            print("✅ STEP 3: Quality Validation - Checking analysis integrity...")
            
            last_result = state.get('retrieved_data', [])[-1] if state.get('retrieved_data') else ""

            if not last_result:
                print("❌ Validation failed: No analysis results found")
                return {
                    "error_log": state.get("error_log", []) + ["No analysis results to validate"],
                    "retry_count": state.get("retry_count", 0) + 1
                }
                
            # Enhanced validation checks
            is_descriptive = "summary" in last_result.lower() or "finding" in last_result.lower()
            is_not_raw_data = "dtype: float64" not in last_result and "Name: count" not in last_result
            
            validation_checks = {
                "sufficient_length": len(last_result) > 200,
                "no_errors": "error" not in last_result.lower() and "failed" not in last_result.lower(),
                "has_numbers": any(char.isdigit() for char in last_result),
                "is_descriptive_prose": is_descriptive and is_not_raw_data
            }
            
            passed_checks = sum(validation_checks.values())
            total_checks = len(validation_checks)
            
            print(f"🔍 Validation checks: {passed_checks}/{total_checks} passed")
            
            if passed_checks < 3:
                print(f"❌ Validation failed: Insufficient quality. Preview: {last_result[:150]}...")
                return {
                    "error_log": state.get("error_log", []) + [f"Analysis quality insufficient: {last_result[:100]}"],
                    "retry_count": state.get("retry_count", 0) + 1
                }
            
            print(f"✅ Validation successful: {len(last_result)} characters validated")
            return {"retrieved_data": state['retrieved_data']}

        def synthesizer_node(state: GraphState):
            """Creates the final comprehensive analysis with executive-level insights"""
            print("📊 STEP 4: Report Synthesis - Generating business intelligence report...")
            
            prompt = f"""
            You are a senior customer analytics consultant presenting to executive leadership. 
            Synthesize this analysis into a comprehensive, actionable business intelligence report.
            
            Original Question: {state['user_question']}
            Strategic Analysis Plan: {state['plan']}
            Detailed Analysis Results: {' '.join(state['retrieved_data'])}
            
            Create a professional report with these sections:
            
            ## EXECUTIVE SUMMARY
            - Key findings in 2-3 bullet points
            - Critical business impact
            - Immediate action required
            
            ## STATISTICAL INSIGHTS
            - Specific numbers, percentages, and correlations
            - Statistical significance and confidence levels
            - Data quality and sample size considerations
            
            ## CUSTOMER SEGMENTS & PATTERNS
            - Distinct customer groups identified
            - Behavioral patterns and characteristics
            - Segment-specific opportunities
            
            ## BUSINESS IMPLICATIONS
            - Revenue impact assessment
            - Risk factors and mitigation strategies
            - Competitive advantage opportunities
            
            ## STRATEGIC RECOMMENDATIONS
            - Prioritized action items with expected ROI
            - Implementation timeline suggestions
            - Success metrics and KPIs to track
            
            Use professional business language with clear, actionable insights.
            """
            
            print("📝 Generating executive-level business intelligence report...")
            synthesis_response = llm.invoke([HumanMessage(content=prompt)])
            print(f"✅ Report synthesis completed: {len(synthesis_response.content)} characters")
            
            return {"analysis_result": synthesis_response.content}

        # Define conditional logic
        def should_continue(state: GraphState):
            """Determines next step after validation"""
            if (state.get("error_log") and 
                len(state.get("error_log", [])) > 0 and 
                state.get("retry_count", 0) < 2):
                return "replan"
            else:
                return "synthesize"

        # Build the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("analyzer", analyzer_node)
        workflow.add_node("validator", validator_node)
        workflow.add_node("synthesizer", synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "analyzer")
        workflow.add_edge("analyzer", "validator")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validator",
            should_continue,
            {
                "replan": "planner",
                "synthesize": "synthesizer"
            }
        )
        
        workflow.add_edge("synthesizer", END)
        
        # Compile the graph
        app = workflow.compile()
        
        return app
        
    except Exception as e:
        return f"❌ LangGraph setup error: {str(e)}"

def analyze_with_langgraph(question: str, df: pd.DataFrame, api_key: str = None):
    """Advanced analysis using LangGraph workflow with detailed step tracking"""
    try:
        import time
        from datetime import datetime
        
        if not api_key:
            api_key = OPENAI_API_KEY
            
        # Start timing
        start_time = time.time()
        start_datetime = datetime.now()
        
        # Create dataframe context
        df_context = f"""
        Dataset: {len(df)} customer records
        Columns: {', '.join(df.columns)}
        Sample data types: {df.dtypes.to_dict()}
        Key statistics: Average spending: ${df['total_spent'].mean():.2f}, Churn rate: {df['churn'].mean()*100:.1f}%
        """
        
        # Set up the graph
        app = setup_langgraph_analyzer(df, api_key)
        
        if isinstance(app, str):  # Error message
            return app
            
        # Enhanced inputs with step tracking
        inputs = {
            "user_question": question,
            "dataframe_context": df_context,
            "plan": "",
            "intermediate_steps": [],
            "retrieved_data": [],
            "analysis_result": "",
            "visualization_code": "",
            "error_log": [],
            "retry_count": 0
        }
        
        # Execute the graph with step tracking
        step_times = {}
        step_details = {}
        
        print(f"🚀 Starting LangGraph Analysis at {start_datetime.strftime('%H:%M:%S')}")
        
        # Execute each step and track timing
        for i, output in enumerate(app.stream(inputs)):
            for node_name, node_output in output.items():
                step_start = time.time()
                
                if node_name == "planner":
                    step_details["planner"] = {
                        "name": "Strategic Planning",
                        "description": "Creating comprehensive analysis plan based on question",
                        "status": "completed",
                        "output_preview": node_output.get('plan', '')[:150] + "..." if node_output.get('plan') else "Plan generated"
                    }
                elif node_name == "analyzer":
                    step_details["analyzer"] = {
                        "name": "Data Analysis Execution", 
                        "description": "Running pandas agent for statistical analysis and insights",
                        "status": "completed",
                        "output_preview": str(node_output.get('retrieved_data', ['Analysis completed']))[:150] + "..."
                    }
                elif node_name == "validator":
                    step_details["validator"] = {
                        "name": "Quality Validation",
                        "description": "Validating analysis results and checking for errors",
                        "status": "completed",
                        "output_preview": f"Validation passed - {len(node_output.get('retrieved_data', []))} data points verified"
                    }
                elif node_name == "synthesizer":
                    step_details["synthesizer"] = {
                        "name": "Report Synthesis",
                        "description": "Generating comprehensive business intelligence report",
                        "status": "completed", 
                        "output_preview": node_output.get('analysis_result', '')[:150] + "..." if node_output.get('analysis_result') else "Report generated"
                    }
                
                step_end = time.time()
                step_times[node_name] = step_end - step_start
                
                print(f"✅ {node_name.title()} completed in {step_times[node_name]:.2f}s")
        
        # Get final state
        final_state = app.invoke(inputs)
        
        # Calculate total time
        total_time = time.time() - start_time
        end_datetime = datetime.now()
        
        # Create detailed workflow summary
        workflow_details = f"""
🕐 **Execution Timeline:**
• Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}
• Completed: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}
• Total Duration: {total_time:.2f} seconds

📋 **Workflow Steps Executed:**
"""
        
        step_counter = 1
        for step_name, details in step_details.items():
            duration = step_times.get(step_name, 0)
            workflow_details += f"""
**Step {step_counter}: {details['name']}**
• Purpose: {details['description']}
• Duration: {duration:.2f} seconds
• Status: ✅ {details['status'].title()}
• Preview: {details['output_preview']}
"""
            step_counter += 1
        
        # Enhanced performance metrics
        performance_metrics = f"""
⚡ **Performance Metrics:**
• Data Processing Rate: {len(df)/total_time:.0f} customers/second
• Analysis Efficiency: {len(step_details)} steps in {total_time:.2f}s
• Average Step Time: {total_time/len(step_details):.2f} seconds
• Memory Usage: {len(df)} customer records processed
• Retry Count: {final_state.get('retry_count', 0)}
• Success Rate: 100% (all steps completed successfully)
"""
        
        # Create simplified workflow steps list
        executed_steps = []
        step_counter = 1
        for step_name, details in step_details.items():
            executed_steps.append(f"Step {step_counter}: {details['name']}")
            step_counter += 1
        
        # Format the final comprehensive result
        result = f"""🤖 LANGGRAPH AI ANALYSIS

{final_state.get('analysis_result', 'Analysis completed but no result generated')}

---
📈 Analysis Details:
• Method: Multi-step LangGraph workflow with pandas agent
• Data Points: {len(df):,} customers
• Workflow Steps Executed: {', '.join(executed_steps)}
• Time: {total_time:.2f} seconds"""
        
        return result
        
    except Exception as e:
        return f"❌ LangGraph analysis error: {str(e)}\n\nStacktrace: {traceback.format_exc()}"

# Enhanced LangChain analysis 
def analyze_with_langchain_improved(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Enhanced LangChain analysis with better formatting"""
    
    if not api_key:
        api_key = OPENAI_API_KEY
    
    try:
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain.agents import AgentType
        from langchain_openai import ChatOpenAI
        from datetime import datetime
        
        # Track start time
        start_time = datetime.now()
        
        # Use GPT-4o with specific formatting controls
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model="gpt-4o",
            max_tokens=2500
        )
        
        # Create agent with enhanced instructions
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            prefix=f"""You are a customer data analyst. Analyze the customer data to answer questions.

Dataset Overview:
- Total customers: {len(df)}
- Total revenue: ${df['total_spent'].sum():,.2f}
- Average customer value: ${df['total_spent'].mean():.2f}
- Churn rate: {df['churn'].mean()*100:.1f}%
- Columns: {', '.join(df.columns)}

ANALYSIS REQUIREMENTS:
1. Provide specific numbers, percentages, and statistical measures
2. Focus on business segments and patterns
3. Give actionable insights with measurable recommendations
4. Use clear professional language with proper formatting""",
            max_iterations=10,
            early_stopping_method="force",
            allow_dangerous_code=True
        )
        
        # Run analysis
        response_dict = agent.invoke({"input": question})
        raw_result = response_dict.get('output', 'Analysis failed to produce an output.')
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Format output
        final_output = f"""📊 ENHANCED LANGCHAIN ANALYSIS

{raw_result}

---
📈 Analysis Details:
• Method: Enhanced Pandas DataFrame Agent
• Model: {llm.model_name}
• Data Points: {len(df)} customers
• Processing Time: {duration:.1f} seconds
• Revenue Context: ${df['total_spent'].sum():,.2f} total customer value"""
        
        return final_output
        
    except Exception as e:
        return f"❌ Enhanced LangChain Error: {str(e)}"

# MAIN FUNCTION: Enhanced with LangGraph option
def analyze_with_ai(question: str, df: pd.DataFrame, api_key: str, use_langchain: bool = False, response_style: str = 'smart'):
    """Main AI analysis function with LangGraph enhancement"""
    if not api_key:
        api_key = OPENAI_API_KEY
    
    if df is None or len(df) == 0:
        return "⚠️ No data available for analysis. Please upload a dataset first."
    
    # Use LangGraph for advanced analysis
    if use_langchain:
        try:
            return analyze_with_langgraph(question, df, api_key)
        except Exception as e:
            # Fall back to enhanced LangChain if LangGraph fails
            fallback_result = analyze_with_langchain_improved(question, df, api_key, response_style)
            return f"⚠️ LangGraph failed, using enhanced LangChain:\n{str(e)}\n\n{fallback_result}"
    else:
        return analyze_with_direct_openai(question, df, api_key, response_style)

# Function aliases for Streamlit app compatibility
def analyze_with_langchain(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """LangChain analysis - now uses LangGraph for advanced workflow"""
    try:
        return analyze_with_langgraph(question, df, api_key)
    except Exception as e:
        # Fallback to enhanced LangChain
        return analyze_with_langchain_improved(question, df, api_key, response_style)

def analyze_with_direct_openai(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """Direct OpenAI analysis with reliable prompting"""
    try:
        import time
        start_time = time.time()
        
        if not api_key:
            api_key = OPENAI_API_KEY
            
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
        
        # Create enhanced data summary
        total_customers = len(df)
        total_revenue = df['total_spent'].sum() if 'total_spent' in df.columns else 0
        avg_customer_value = df['total_spent'].mean() if 'total_spent' in df.columns else 0
        churn_rate = df['churn'].mean() * 100 if 'churn' in df.columns else 0
        
        churned_count = df['churn'].sum() if 'churn' in df.columns else 0
        active_count = len(df) - churned_count if 'churn' in df.columns else len(df)
        sat_avg = df['satisfaction_score'].mean() if 'satisfaction_score' in df.columns else 0
        
        data_summary = f"""Customer Database Overview:
Total Customers: {total_customers:,}
Total Revenue: ${total_revenue:,.2f}
Average Customer Value: ${avg_customer_value:.2f}
Churn Rate: {churn_rate:.1f}%
Active Customers: {active_count:,}
Average Satisfaction: {sat_avg:.2f}/5.0

Available columns: {', '.join(df.columns)}

Sample data (first 2 rows):
{df.head(2).to_string()}"""
        
        # Enhanced prompt
        prompt = f"""You are a senior customer analytics consultant. Analyze this customer data and provide comprehensive insights.

{data_summary}

Question: {question}

ANALYSIS REQUIREMENTS:
- Provide specific statistics with confidence levels where relevant
- Focus on customer segments and business patterns  
- Include actionable recommendations with expected outcomes
- Use proper formatting with clear sections and bullet points

Structure your response with:
1. Key Findings (with specific numbers)
2. Customer Segments Analysis
3. Business Implications
4. Strategic Recommendations"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior customer analytics consultant specializing in e-commerce data analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1800,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        total_time = time.time() - start_time
        
        return f"""📊 DIRECT OPENAI ANALYSIS

{result}

---
📈 Analysis Details:
• Method: Direct OpenAI Analysis (GPT-4o)
• Data Points: {len(df)} customers
• Focus: Professional consulting-grade insights
• Time: {total_time:.2f} seconds"""
        
    except Exception as e:
        return f"❌ Direct OpenAI analysis error: {str(e)}"

if __name__ == "__main__":
    print("🔧 Enhanced AI Analyzer Module with LangGraph")
    print("Debug Environment:")
    print(debug_environment())
    print(f"\n✅ Module loaded successfully. API key configured: {OPENAI_API_KEY is not None}")
