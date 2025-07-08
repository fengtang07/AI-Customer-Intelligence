# app.py - AI Customer Intelligence Agent
"""
AI Customer Intelligence Agent - Modern Streamlit Application
Features comprehensive e-commerce data analysis, AI chat, and interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import random

warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Customer Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import AI functionality with comprehensive error handling
try:
    from ai_analyzer import (
        analyze_with_ai,
        debug_environment,
        test_ai_connection,
        analyze_with_langchain,
        analyze_with_direct_openai
    )

    AI_AVAILABLE = True
    AI_IMPORT_ERROR = None
except ImportError as e:
    AI_AVAILABLE = False
    AI_IMPORT_ERROR = str(e)
except Exception as e:
    AI_AVAILABLE = False
    AI_IMPORT_ERROR = f"Unexpected error: {str(e)}"

# Modern CSS styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --background-color: #F8F9FA;
        --card-background: #FFFFFF;
        --text-primary: #2C3E50;
        --text-secondary: #7F8C8D;
        --accent-color: #E74C3C;
        --success-color: #27AE60;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2.5rem 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
         .main-header h1 {
         font-size: 2.5rem;
         font-weight: 700;
         margin-bottom: 0.5rem;
         letter-spacing: -0.02em;
         color: white !important;
     }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Tab styling - larger and more visible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding: 1rem 0;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        padding: 1rem 2rem;
        background: var(--card-background);
        border-radius: 12px;
        border: 2px solid #E0E6ED;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white !important;
        border-color: var(--primary-color);
        box-shadow: 0 6px 20px rgba(46, 134, 171, 0.3);
    }
    
    /* Card styling */
    .metric-card {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E0E6ED;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Chat container */
    .chat-container {
        background: var(--card-background);
        border-radius: 12px;
        border: 1px solid #E0E6ED;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    /* Status boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: none;
        color: var(--success-color);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: none;
        color: var(--accent-color);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: none;
        color: var(--primary-color);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: var(--secondary-color);
        border-color: var(--secondary-color);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        color: var(--text-secondary);
        padding: 2rem;
        font-size: 0.9rem;
        border-top: 1px solid #E0E6ED;
        margin-top: 3rem;
    }
    
    /* Remove emoji clutter and improve typography */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Modern metrics */
    [data-testid="metric-container"] {
        background: var(--card-background);
        border: 1px solid #E0E6ED;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables with error handling"""
    session_vars = {
        'chat_history': [],
        'current_data': None,
        'openai_api_key': "",
        'sample_data': None,
        'analysis_method': 'direct',
        'chat_mode': 'smart',
        'data_uploaded': False,
        'last_analysis_time': None
    }

    try:
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    except Exception as e:
        for var, default_value in session_vars.items():
            st.session_state[var] = default_value

# Clean up and initialize
initialize_session_state()

# Function to get API key from secrets or session state
def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets, session state, or hardcoded fallback"""
    try:
        # First try to get from Streamlit secrets
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    
    # Fall back to session state (manual input)
    session_key = st.session_state.get('openai_api_key', '')
    if session_key:
        return session_key
    
    # Final fallback - hardcoded key for testing

# Generate e-commerce data automatically with different data each time
def generate_ecommerce_data():
    """Generate varied e-commerce sample data"""
    # Use current time as seed for variation
    seed = int(datetime.now().timestamp())
    np.random.seed(seed)
    random.seed(seed)
    
    n_customers = random.randint(5000, 6000)  # Generate 5000-6000 customers
    
    # Varied demographics
    age_mean = random.uniform(35, 45)
    spending_scale = random.uniform(5.5, 7.0)
    satisfaction_mean = random.uniform(3.6, 4.2)
    churn_prob = random.uniform(0.15, 0.35)
    
    sample_data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(n_customers)],
        'age': np.random.normal(age_mean, 12, n_customers).astype(int),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'total_spent': np.random.lognormal(spending_scale, 1, n_customers),
        'monthly_visits': np.random.poisson(random.randint(6, 12), n_customers),
        'satisfaction_score': np.random.normal(satisfaction_mean, 0.8, n_customers),
        'churn': np.random.choice([0, 1], n_customers, p=[1-churn_prob, churn_prob]),
        'product_category': np.random.choice([
            'Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports & Outdoors',
            'Beauty & Health', 'Toys & Games', 'Automotive'
        ], n_customers),
        'acquisition_channel': np.random.choice([
            'Organic Search', 'Paid Social', 'Email Marketing', 'Direct', 
            'Referral', 'Display Ads'
        ], n_customers),
        'membership_tier': np.random.choice([
            'Bronze', 'Silver', 'Gold', 'Platinum'
        ], n_customers, p=[0.4, 0.3, 0.2, 0.1])
    }
    
    # Clean up data
    sample_data['age'] = np.clip(sample_data['age'], 18, 80)
    sample_data['satisfaction_score'] = np.clip(sample_data['satisfaction_score'], 1, 5)
    sample_data['total_spent'] = np.round(sample_data['total_spent'], 2)

    df = pd.DataFrame(sample_data)
    return df

# Auto-generate data on startup
if st.session_state.current_data is None:
    with st.spinner("Generating e-commerce customer data..."):
        df = generate_ecommerce_data()
        st.session_state.current_data = df
        st.session_state.sample_data = df
        st.session_state.data_uploaded = True

# Main title with modern styling
st.markdown("""
<div class="main-header">
    <h1>AI Customer Intelligence Agent</h1>
    <p>Transform your e-commerce data into actionable insights with AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh data button (top right)
col1, col2, col3 = st.columns([6, 1, 1])
with col2:
    if st.button("🔄 New Data", help="Generate fresh e-commerce data"):
        with st.spinner("Generating new data..."):
            df = generate_ecommerce_data()
            st.session_state.current_data = df
            st.session_state.sample_data = df
            st.session_state.chat_history = []  # Clear chat history
            st.rerun()

with col3:
    if st.button("🗑️ Clear Chat", help="Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# Get current data
df = st.session_state.current_data

def display_data_overview(df):
    """Display comprehensive data overview"""
    st.markdown("## Data Overview")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")

    with col2:
        churn_rate = df['churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        avg_spending = df['total_spent'].mean()
        st.metric("Average Spending", f"${avg_spending:,.2f}")

    with col4:
        avg_satisfaction = df['satisfaction_score'].mean()
        st.metric("Average Satisfaction", f"{avg_satisfaction:.2f}/5.0")

    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(10), use_container_width=True, height=350)

    # Quick insights
    with st.expander("Key Insights", expanded=True):
        insights = generate_quick_insights(df)
        for insight in insights:
            st.markdown(f"• {insight}")

def generate_quick_insights(df):
    """Generate quick insights from the data"""
    insights = []
    
    # Dataset info
    insights.append(f"Dataset contains {len(df):,} e-commerce customers")
    
    # Churn insights
    churn_rate = df['churn'].mean() * 100
    if churn_rate > 25:
        insights.append(f"High churn rate ({churn_rate:.1f}%) - immediate attention needed")
    elif churn_rate > 15:
        insights.append(f"Moderate churn rate ({churn_rate:.1f}%) - monitor closely")
    else:
        insights.append(f"Low churn rate ({churn_rate:.1f}%) - good retention")

    # Top category
    top_category = df['product_category'].value_counts().index[0]
    insights.append(f"Most popular category: {top_category}")
    
    # Top channel
    top_channel = df['acquisition_channel'].value_counts().index[0]
    insights.append(f"Top acquisition channel: {top_channel}")
    
    # Spending insights
    high_spenders = (df['total_spent'] > df['total_spent'].quantile(0.8)).sum()
    insights.append(f"{high_spenders} high-value customers (top 20% spenders)")

    return insights

def display_ai_chat(df):
    """Display AI chat interface with settings"""
    st.markdown("## AI Customer Analyst")

    # AI Configuration section
    st.markdown("### Configuration")
    
    # AI Analysis Method
    analysis_method = st.selectbox(
        "AI Analysis Method:",
        ["Direct OpenAI", "LangChain Agent"],
        index=1 if st.session_state.analysis_method == 'langchain' else 0,
        help="Direct OpenAI provides comprehensive, well-formatted analysis with GPT-4o. LangChain Agent offers data exploration tools but may have formatting issues."
    )
    
    # Update session state
    if analysis_method == "LangChain Agent":
        st.session_state.analysis_method = 'langchain'
    else:
        st.session_state.analysis_method = 'direct'
    
    # Set default response style since we removed the selector
    st.session_state.chat_mode = "smart"

    # Get API key from secrets or hardcoded
    current_api_key = get_openai_api_key()
    
    # Check prerequisites
    if not AI_AVAILABLE:
        st.error("AI functionality is not available")
        st.error(f"Import error: {AI_IMPORT_ERROR}")
        return

    if not current_api_key:
        st.error("No API key available")
        return

    # Simple status indicators
    st.success("🔑 API Key: Loaded")
    st.success(f"🤖 AI Ready | Method: {st.session_state.analysis_method.title()}")

    # Sample questions
    st.markdown("### Try These Questions")
    
    sample_questions = [
        "Why are customers churning?",
        "What customer segments exist?", 
        "Which customers are at risk?",
        "What drives customer value?",
        "Show satisfaction insights",
        "Analyze spending patterns"
    ]
    
    # Display questions in rows of 3
    for i in range(0, len(sample_questions), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(sample_questions):
                question_text = sample_questions[i + j]
                if col.button(question_text, key=f"sample_q_{i + j}"):
                    process_ai_question(question_text, df)

    # Chat input
    st.markdown("### Ask Your Question")
    
    with st.form(key="ai_chat_form", clear_on_submit=True):
        user_question = st.text_area(
            "What would you like to know about your customer data?",
            height=100,
            placeholder="e.g., What are the main drivers of customer churn?"
        )
        
        ask_submitted = st.form_submit_button("Ask AI", type="primary")

    # Process question
    if ask_submitted and user_question.strip():
        process_ai_question(user_question, df)
    elif ask_submitted and not user_question.strip():
        st.warning("Please enter a question before submitting")

    # Display chat history
    display_chat_history()

def process_ai_question(question, df):
    """Process AI question and generate response"""
    with st.spinner("AI is analyzing your question..."):
        try:
            # Get API key
            api_key = get_openai_api_key()
            
            # Get response style preference
            response_style = st.session_state.get('chat_mode', 'smart')
            
            # Choose analysis method
            if st.session_state.analysis_method == 'langchain':
                response = analyze_with_langchain(question, df, api_key, response_style)
            else:  # direct
                response = analyze_with_direct_openai(question, df, api_key, response_style)

            # Add to chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            chat_entry = {
                'question': question,
                'response': response,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': st.session_state.analysis_method
            }

            st.session_state.chat_history = [chat_entry] + st.session_state.chat_history
            
            # Limit history
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[:10]

            st.success("AI analysis completed")
            st.rerun()

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

def display_chat_history():
    """Display chat history with most recent first"""
    if not st.session_state.chat_history:
        st.info("Your chat history will appear here. Ask a question above to get started.")
        return

    st.markdown("### Chat History")

    # Show most recent first (no need to reverse since we're already storing newest first)
    for i, chat in enumerate(st.session_state.chat_history):
        with st.expander(
                f"{chat['question'][:80]}..." if len(chat['question']) > 80 else chat['question'],
                expanded=(i == 0)  # Expand the most recent (first in list)
        ):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown("**AI Response:**")
            st.markdown(chat['response'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Time: {chat['timestamp']}")
            with col2:
                st.caption(f"Method: {chat.get('method', 'unknown').title()}")

def display_analytics(df):
    """Display analytics with viridis color palette"""
    st.markdown("## Advanced Analytics")

    # Create analytics tabs
    analytics_tabs = st.tabs([
        "Distribution Analysis",
        "Churn Analysis", 
        "Customer Segments",
        "Correlation Analysis"
    ])

    with analytics_tabs[0]:
        display_distribution_analysis(df)

    with analytics_tabs[1]:
        display_churn_analysis(df)

    with analytics_tabs[2]:
        display_segmentation_analysis(df)

    with analytics_tabs[3]:
        display_correlation_analysis(df)

def display_distribution_analysis(df):
    """Display distribution analysis with enhanced viridis styling"""
    st.markdown("### Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution with enhanced styling
        fig = px.histogram(
            df, x='age', 
            title="Customer Age Distribution",
            nbins=25,
            color_discrete_sequence=['#440154']  # Dark purple from viridis
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50')
            ),
            yaxis=dict(
                gridcolor='#E8E8E8', 
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50')
            ),
            margin=dict(t=50, l=40, r=40, b=40)
        )
        fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>Age:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Spending distribution with gradient effect
        fig = px.histogram(
            df, x='total_spent',
            title="Customer Spending Distribution", 
            nbins=25,
            color_discrete_sequence=['#21908C']  # Teal from viridis
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                tickformat='$,.0f'
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50')
            ),
            margin=dict(t=50, l=40, r=40, b=40)
        )
        fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>Spending:</b> $%{x:,.2f}<br><b>Count:</b> %{y}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction distribution with modern styling
        fig = px.histogram(
            df, x='satisfaction_score',
            title="Satisfaction Score Distribution",
            nbins=20,
            color_discrete_sequence=['#5DC863']  # Light green from viridis
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50')
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50')
            ),
            margin=dict(t=50, l=40, r=40, b=40)
        )
        fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>Satisfaction:</b> %{x:.1f}/5.0<br><b>Count:</b> %{y}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Category distribution with beautiful gradient
        category_counts = df['product_category'].value_counts()
        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Customers by Product Category",
            color=category_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Number of Customers'
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Product Category'
            ),
            margin=dict(t=50, l=120, r=60, b=40),
            coloraxis_colorbar=dict(
                title="Customer Count",
                title_font=dict(size=12, color='#2C3E50'),
                tickfont=dict(size=10, color='#2C3E50')
            )
        )
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br><b>Customers:</b> %{x}<extra></extra>',
            marker=dict(line=dict(width=0.5, color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)

def display_churn_analysis(df):
    """Display churn analysis with viridis colors"""
    st.markdown("### Churn Analysis")

    # Churn overview
    churn_rate = df['churn'].mean() * 100
    churned_count = df['churn'].sum()
    active_count = len(df) - churned_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    with col2:
        st.metric("Churned Customers", f"{churned_count:,}")
    with col3:
        st.metric("Active Customers", f"{active_count:,}")

    col1, col2 = st.columns(2)

    with col1:
        # Churn by age groups with enhanced styling
        df_temp = df.copy()
        df_temp['age_group'] = pd.cut(
            df_temp['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['<25', '25-34', '35-44', '45-54', '55+'],
            include_lowest=True
        )

        churn_by_age = df_temp.groupby('age_group', observed=True)['churn'].mean().reset_index()
        churn_by_age['churn_rate'] = churn_by_age['churn'] * 100

        fig = px.bar(
            churn_by_age, x='age_group', y='churn_rate',
            title="Churn Rate by Age Group",
            color='churn_rate',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Age Group'
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Churn Rate (%)'
            ),
            margin=dict(t=50, l=40, r=60, b=40),
            coloraxis_colorbar=dict(
                title="Churn Rate (%)",
                title_font=dict(size=12, color='#2C3E50'),
                tickfont=dict(size=10, color='#2C3E50')
            )
        )
        fig.update_traces(
            hovertemplate='<b>Age Group:</b> %{x}<br><b>Churn Rate:</b> %{y:.1f}%<extra></extra>',
            marker=dict(line=dict(width=0.5, color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Satisfaction vs Churn with modern box plot styling
        fig = px.box(
            df, x='churn', y='satisfaction_score',
            title="Satisfaction Score by Churn Status",
            color='churn',
            color_discrete_sequence=['#21908C', '#440154']  # Viridis colors
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Customer Status'
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Satisfaction Score'
            ),
            margin=dict(t=50, l=40, r=40, b=40)
        )
        fig.update_xaxes(ticktext=['Retained', 'Churned'], tickvals=[0, 1])
        fig.update_traces(
            hovertemplate='<b>Status:</b> %{x}<br><b>Satisfaction:</b> %{y:.2f}/5.0<extra></extra>',
            marker=dict(size=4, line=dict(width=1, color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)

def display_segmentation_analysis(df):
    """Display segmentation analysis with viridis colors"""
    st.markdown("### Customer Segmentation")

    # Value-based segmentation
    st.markdown("#### Value-Based Segments")

    df_temp = df.copy()
    df_temp['spending_quartile'] = pd.qcut(
        df_temp['total_spent'],
        q=4,
        labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
    )

    # Segment summary
    segment_summary = df_temp.groupby('spending_quartile', observed=True).agg({
        'total_spent': ['mean', 'sum', 'count'],
        'churn': 'mean'
    }).round(2)

    segment_summary.columns = ['Avg Spending', 'Total Revenue', 'Customer Count', 'Churn Rate']
    segment_summary['Churn Rate'] = segment_summary['Churn Rate'] * 100

    st.dataframe(segment_summary, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Segment distribution with enhanced pie chart
        segment_counts = df_temp['spending_quartile'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Distribution by Value Segment",
            color_discrete_sequence=['#440154', '#21908C', '#5DC863', '#FDE725']  # Viridis palette
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            margin=dict(t=60, l=40, r=40, b=40)
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=11,
            marker=dict(line=dict(color='white', width=2)),
            hovertemplate='<b>%{label}</b><br><b>Customers:</b> %{value}<br><b>Percentage:</b> %{percent}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Revenue by segment with enhanced bar chart
        revenue_by_segment = df_temp.groupby('spending_quartile', observed=True)['total_spent'].sum()
        fig = px.bar(
            x=revenue_by_segment.index,
            y=revenue_by_segment.values,
            title="Revenue by Segment",
            color=revenue_by_segment.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50', size=12),
            title=dict(font=dict(size=16, color='#2C3E50')),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Customer Segment'
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title_font=dict(size=14, color='#2C3E50'),
                title_text='Total Revenue ($)',
                tickformat='$,.0f'
            ),
            margin=dict(t=50, l=60, r=60, b=40),
            coloraxis_colorbar=dict(
                title="Revenue ($)",
                title_font=dict(size=12, color='#2C3E50'),
                tickfont=dict(size=10, color='#2C3E50'),
                tickformat='$,.0f'
            )
        )
        fig.update_traces(
            hovertemplate='<b>Segment:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>',
            marker=dict(line=dict(width=0.5, color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)

def display_correlation_analysis(df):
    """Display correlation analysis with viridis colors"""
    st.markdown("### Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
        return

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Correlation heatmap with enhanced viridis styling
    fig = px.imshow(
        corr_matrix,
        title="Correlation Heatmap",
        color_continuous_scale='Viridis',
        aspect='auto',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#2C3E50', size=12),
        title=dict(font=dict(size=16, color='#2C3E50')),
        xaxis=dict(
            title_font=dict(size=14, color='#2C3E50'),
            tickfont=dict(size=10, color='#2C3E50')
        ),
        yaxis=dict(
            title_font=dict(size=14, color='#2C3E50'),
            tickfont=dict(size=10, color='#2C3E50')
        ),
        coloraxis_colorbar=dict(
            title="Correlation",
            title_font=dict(size=12, color='#2C3E50'),
            tickfont=dict(size=10, color='#2C3E50'),
            len=0.7
        ),
        margin=dict(t=60, l=60, r=80, b=60)
    )
    fig.update_traces(
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br><b>Correlation:</b> %{z:.3f}<extra></extra>'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Strong correlations
    st.markdown("#### Strong Correlations (|r| > 0.5)")

    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corrs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_val,
                    'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                })

    if strong_corrs:
        strong_corr_df = pd.DataFrame(strong_corrs)
        strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(strong_corr_df, use_container_width=True)
    else:
        st.info("No strong correlations (|r| > 0.5) found between variables")

# Main application flow
def main():
    """Main application flow"""
    
    # Get current data
    df = st.session_state.current_data
    
    if df is not None:
        # Create main tabs with larger, more visible styling
        main_tabs = st.tabs(["📊 Data Overview", "📈 Analytics", "💬 AI Chat"])

        with main_tabs[0]:
            display_data_overview(df)

        with main_tabs[1]:
            display_analytics(df)

        with main_tabs[2]:
            display_ai_chat(df)

# Run the application
main()

# Modern footer
st.markdown("---")
st.markdown("""
<div class="custom-footer">
    <strong>AI Customer Intelligence Agent</strong><br>
    Powered by LangChain & OpenAI | Built with Streamlit | Created by Feng Tang
</div>
""", unsafe_allow_html=True)
