"""
Main Streamlit web application for AI Customer Intelligence Agent
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.customer_agent import CustomerIntelligenceAgent
from config.config import Config

# Page configuration
st.set_page_config(
    page_title="AI Customer Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CustomerIntelligenceApp:
    def __init__(self):
        self.config = Config()
        self.agent = None
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'agent_initialized' not in st.session_state:
            st.session_state.agent_initialized = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
    
    def initialize_agent(self, data_source='sample'):
        """Initialize the AI agent"""
        try:
            if data_source == 'uploaded' and hasattr(st.session_state, 'uploaded_data'):
                # Use uploaded data
                self.agent = CustomerIntelligenceAgent()
                self.agent.df = st.session_state.uploaded_data
                self.agent.analyzer.set_data(st.session_state.uploaded_data)
            else:
                # Use sample data
                self.agent = CustomerIntelligenceAgent()
            
            st.session_state.agent_initialized = True
            st.success("✅ AI Agent initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error initializing agent: {str(e)}")
            return False
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🤖 AI Customer Intelligence Agent</h1>
            <p style="color: #bfdbfe; margin: 0;">Advanced customer analytics powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with navigation and controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Data source selection
        st.sidebar.subheader("📊 Data Source")
        data_option = st.sidebar.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload Your Data"]
        )
        
        if data_option == "Upload Your Data":
            uploaded_file = st.sidebar.file_uploader(
                "Upload Customer Data",
                type=['csv', 'xlsx'],
                help="Upload a CSV or Excel file with customer data"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.sidebar.success(f"✅ Loaded {len(df)} records")
                    st.session_state.uploaded_data = df
                    
                    if st.sidebar.button("Initialize Agent with Uploaded Data"):
                        self.initialize_agent('uploaded')
                        
                except Exception as e:
                    st.sidebar.error(f"Error loading file: {str(e)}")
        else:
            if st.sidebar.button("Initialize Agent with Sample Data"):
                self.initialize_agent('sample')
        
        # Navigation
        st.sidebar.subheader("🧭 Navigation")
        return st.sidebar.selectbox(
            "Choose Analysis View:",
            [
                "🏠 Overview Dashboard",
                "💬 AI Chat Interface",
                "📊 Data Explorer", 
                "🎯 Churn Analysis",
                "👥 Customer Segmentation",
                "📈 Predictive Insights",
                "🚀 Strategic Recommendations"
            ]
        )
    
    def render_overview_dashboard(self):
        """Render main overview dashboard"""
        st.subheader("📊 Customer Intelligence Overview")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Please initialize the AI agent from the sidebar first.")
            return
        
        # Get data summary
        data_summary = self.agent.get_data_summary()
        
        if 'error' in data_summary:
            st.error(data_summary['error'])
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        df = self.agent.df
        
        with col1:
            st.metric(
                "Total Customers",
                f"{len(df):,}",
                help="Total number of customers in dataset"
            )
        
        with col2:
            if 'churn' in df.columns:
                churn_rate = df['churn'].mean() * 100
                st.metric(
                    "Churn Rate",
                    f"{churn_rate:.1f}%",
                    delta=f"{churn_rate-20:.1f}%" if churn_rate < 20 else None,
                    help="Percentage of customers who have churned"
                )
            else:
                st.metric("Churn Rate", "N/A")
        
        with col3:
            if 'total_spent' in df.columns:
                avg_revenue = df['total_spent'].mean()
                st.metric(
                    "Avg Customer Value",
                    f"${avg_revenue:.0f}",
                    help="Average total spending per customer"
                )
            else:
                st.metric("Avg Customer Value", "N/A")
        
        with col4:
            if 'satisfaction_score' in df.columns:
                avg_satisfaction = df['satisfaction_score'].mean()
                st.metric(
                    "Avg Satisfaction",
                    f"{avg_satisfaction:.2f}/5.0",
                    delta=f"{avg_satisfaction-3.5:.2f}" if avg_satisfaction > 3.5 else None,
                    help="Average customer satisfaction score"
                )
            else:
                st.metric("Avg Satisfaction", "N/A")
        
        # Quick insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Quick Insights")
            
            # Generate quick churn analysis
            if st.button("🔍 Analyze Churn Patterns"):
                with st.spinner("Analyzing churn patterns..."):
                    result = self.agent.query("Analyze customer churn patterns", include_ai_insights=True)
                    
                    if 'error' not in result:
                        st.markdown("### Key Findings:")
                        
                        churn_data = result['data']
                        if 'overview' in churn_data:
                            overview = churn_data['overview']
                            st.write(f"• **Churn Rate**: {overview['churn_rate']:.1f}%")
                            st.write(f"• **Customers at Risk**: {len(churn_data.get('high_risk_customers', []))}")
                        
                        if result['insights']:
                            st.markdown("### AI Insights:")
                            st.info(result['insights'])
        
        with col2:
            st.subheader("🎯 Customer Segments")
            
            if st.button("👥 Analyze Customer Segments"):
                with st.spinner("Performing customer segmentation..."):
                    result = self.agent.query("Analyze customer segments", include_ai_insights=True)
                    
                    if 'error' not in result and 'segments' in result['data']:
                        segments = result['data']['segments']
                        
                        for segment_name, segment_info in segments.items():
                            if 'label' in segment_info:
                                st.write(f"• **{segment_info['label']}**: {segment_info['size']} customers")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    def render_chat_interface(self):
        """Render AI chat interface"""
        st.subheader("💬 AI Customer Intelligence Chat")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Please initialize the AI agent from the sidebar first.")
            return
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**👤 You:** {chat['question']}")
                st.markdown(f"**🤖 AI Agent:** {chat['answer']}")
                
                if chat.get('data'):
                    with st.expander("📊 View Detailed Analysis"):
                        st.json(chat['data'])
                
                st.divider()
        
        # Sample questions
        st.markdown("### 💡 Try these sample questions:")
        sample_questions = [
            "Why are customers churning?",
            "What customer segments do we have?",
            "Which customers are at highest risk?",
            "What actions should we take to improve retention?",
            "Show me predictive insights about our customers"
        ]
        
        cols = st.columns(len(sample_questions))
        for i, question in enumerate(sample_questions):
            with cols[i]:
                if st.button(question, key=f"sample_{i}"):
                    st.session_state.current_question = question
        
        # Chat input
        user_question = st.text_input(
            "Ask me anything about your customer data:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., What are the main reasons customers churn?",
            key="chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Ask AI", type="primary")
        
        if ask_button and user_question:
            with st.spinner("🧠 AI is analyzing your question..."):
                try:
                    result = self.agent.query(user_question, include_ai_insights=True)
                    
                    # Add to chat history
                    chat_entry = {
                        'question': user_question,
                        'answer': result.get('insights', 'Analysis completed'),
                        'data': result.get('data', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(chat_entry)
                    
                    # Clear current question
                    if 'current_question' in st.session_state:
                        del st.session_state.current_question
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    def render_data_explorer(self):
        """Render data exploration interface"""
        st.subheader("🔍 Data Explorer")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Please initialize the AI agent from the sidebar first.")
            return
        
        df = self.agent.df
        
        # Data overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Dataset Information")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Column types
            st.markdown("### 📋 Column Types")
            column_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(column_info, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Quick Statistics")
            if len(df.select_dtypes(include=['number']).columns) > 0:
                st.dataframe(df.describe(), use_container_width=True)
        
        # Interactive exploration
        st.markdown("### 🎛️ Interactive Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select columns for analysis
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            selected_numeric = st.selectbox("Select numeric column:", numeric_columns)
            selected_categorical = st.selectbox("Select categorical column:", ['None'] + categorical_columns)
        
        with col2:
            if selected_numeric:
                # Create visualization
                if selected_categorical and selected_categorical != 'None':
                    fig = px.box(df, x=selected_categorical, y=selected_numeric, 
                               title=f"{selected_numeric} by {selected_categorical}")
                else:
                    fig = px.histogram(df, x=selected_numeric, 
                                     title=f"Distribution of {selected_numeric}")
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Data preview with filtering
        st.markdown("### 📋 Data Preview")
        
        # Filters
        if categorical_columns:
            filter_column = st.selectbox("Filter by column:", ['None'] + categorical_columns)
            
            if filter_column and filter_column != 'None':
                unique_values = df[filter_column].unique()
                selected_values = st.multiselect(f"Select {filter_column} values:", unique_values, default=unique_values)
                
                if selected_values:
                    filtered_df = df[df[filter_column].isin(selected_values)]
                else:
                    filtered_df = df
            else:
                filtered_df = df
        else:
            filtered_df = df
        
        st.dataframe(filtered_df, use_container_width=True)
    
    def render_churn_analysis(self):
        """Render churn analysis dashboard"""
        st.subheader("🎯 Customer Churn Analysis")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Please initialize the AI agent from the sidebar first.")
            return
        
        # Get churn analysis
        with st.spinner("Performing churn analysis..."):
            result = self.agent.query("Comprehensive churn analysis", include_ai_insights=True)
        
        if 'error' in result:
            st.error(result['error'])
            return
        
        churn_data = result['data']
        
        # Overview metrics
        if 'overview' in churn_data:
            overview = churn_data['overview']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{overview['total_customers']:,}")
            
            with col2:
                st.metric("Churned", f"{overview['churned_count']:,}")
            
            with col3:
                st.metric("Churn Rate", f"{overview['churn_rate']:.1f}%")
            
            with col4:
                at_risk_count = len(churn_data.get('high_risk_customers', []))
                st.metric("At Risk", f"{at_risk_count:,}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by demographics
            if 'demographic_analysis' in churn_data and 'gender' in churn_data['demographic_analysis']:
                gender_data = churn_data['demographic_analysis']['gender']
                
                # Create gender churn chart
                gender_df = pd.DataFrame.from_dict(gender_data, orient='index').reset_index()
                gender_df.columns = ['Gender', 'Total', 'Churned', 'Churn_Rate']
                
                fig = px.bar(gender_df, x='Gender', y='Churn_Rate', 
                           title="Churn Rate by Gender",
                           labels={'Churn_Rate': 'Churn Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk factors
            if 'risk_factors' in churn_data:
                risk_factors = churn_data['risk_factors']
                
                if risk_factors:
                    # Create risk factors chart
                    risk_df = pd.DataFrame(risk_factors)
                    
                    fig = px.bar(risk_df, x='factor', y='impact_score',
                               title="Top Risk Factors for Churn",
                               labels={'impact_score': 'Impact Score', 'factor': 'Risk Factor'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # AI Insights
        if result['insights']:
            st.markdown("### 🤖 AI-Generated Insights")
            st.info(result['insights'])
        
        # High-risk customers
        if 'high_risk_customers' in churn_data and churn_data['high_risk_customers']:
            st.markdown("### ⚠️ High-Risk Customers")
            
            risk_df = pd.DataFrame(churn_data['high_risk_customers'])
            st.dataframe(risk_df, use_container_width=True)
            
            # Export functionality
            csv = risk_df.to_csv(index=False)
            st.download_button(
                label="📥 Download High-Risk Customer List",
                data=csv,
                file_name=f"high_risk_customers_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def render_segmentation(self):
        """Render customer segmentation analysis"""
        st.subheader("👥 Customer Segmentation Analysis")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Please initialize the AI agent from the sidebar first.")
            return
        
        # Segmentation controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            n_clusters = st.slider("Number of Segments:", 3, 8, 4)
            
            if st.button("🔄 Run Segmentation"):
                with st.spinner("Performing customer segmentation..."):
                    result = self.agent.query("Analyze customer segments", include_ai_insights=True)
                    st.session_state.segmentation_result = result
        
        with col1:
            if 'segmentation_result' in st.session_state:
                result = st.session_state.segmentation_result
                
                if 'error' not in result and 'segments' in result['data']:
                    segments = result['data']['segments']
                    
                    # Create segment summary visualization
                    segment_summary = []
                    for segment_name, segment_info in segments.items():
                        segment_summary.append({
                            'Segment': segment_info.get('label', segment_name),
                            'Size': segment_info['size'],
                            'Churn Rate': segment_info.get('business_metrics', {}).get('churn_rate', 0)
                        })
                    
                    segment_df = pd.DataFrame(segment_summary)
                    
                    # Pie chart of segment sizes
                    fig = px.pie(segment_df, values='Size', names='Segment',
                               title="Customer Segment Distribution")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Segment details
        if 'segmentation_result' in st.session_state:
            result = st.session_state.segmentation_result
            
            if 'error' not in result:
                # AI insights
                if result.get('insights'):
                    st.markdown("### 🤖 Segmentation Insights")
                    st.info(result['insights'])
                
                # Detailed segment profiles
                if 'segments' in result['data']:
                    st.markdown("### 📊 Segment Profiles")
                    
                    segments = result['data']['segments']
                    
                    for segment_name, segment_info in segments.items():
                        with st.expander(f"📋 {segment_info.get('label', segment_name)} ({segment_info['size']} customers)"):
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Characteristics:**")
                                if 'characteristics' in segment_info:
                                    for feature, stats in segment_info['characteristics'].items():
                                        st.write(f"• {feature}: {stats['mean']:.2f} (avg)")
                            
                            with col2:
                                st.markdown("**Business Metrics:**")
                                if 'business_metrics' in segment_info:
                                    metrics = segment_info['business_metrics']
                                    for metric, value in metrics.items():
                                        if isinstance(value, (int, float)):
                                            if 'rate' in metric:
                                                st.write(f"• {metric}: {value:.1f}%")
                                            elif 'revenue' in metric:
                                                st.write(f"• {metric}: ${value:,.0f}")
                                            else:
                                                st.write(f"• {metric}: {value:.2f}")
    
    def render_predictions(self):
        """Render predictive insights dashboard"""
        st.subheader("📈 Predictive Customer Insights")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Please initialize the AI agent from the sidebar first.")
            return
        
        # Generate predictions
        with st.spinner("Generating predictive insights..."):
            result = self.agent.query("Generate predictive insights about customer behavior", include_ai_insights=True)
        
        if 'error' in result:
            st.error(result['error'])
            return
        
        # Display predictions
        st.markdown("### 🔮 Predictive Analysis Results")
        
        if result.get('insights'):
            st.info(result['insights'])
        
        # Revenue forecasts
        prediction_data = result.get('data', {})
        
        if 'revenue_forecasts' in prediction_data:
            forecasts = prediction_data['revenue_forecasts']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Revenue at Risk",
                    f"${forecasts.get('revenue_at_risk', 0):,.0f}",
                    help="Total revenue from high-risk customers"
                )
            
            with col2:
                st.metric(
                    "Potential 3M Loss",
                    f"${forecasts.get('potential_loss_3m', 0):,.0f}",
                    delta=f"-{forecasts.get('potential_loss_3m', 0) / forecasts.get('revenue_at_risk', 1) * 100:.0f}%",
                    help="Estimated revenue loss in 3 months without intervention"
                )
            
            with col3:
                st.metric(
                    "Savings with Intervention",
                    f"${forecasts.get('potential_savings_with_intervention', 0):,.0f}",
                    delta=f"+{forecasts.get('potential_savings_with_intervention', 0) / forecasts.get('revenue_at_risk', 1) * 100:.0f}%",
                    help="Potential revenue protection with intervention"
                )
        
        # Trend analysis (if available)
        df = self.agent.df
        
        if 'signup_date' in df.columns:
            st.markdown("### 📊 Customer Acquisition Trends")
            
            # Convert signup_date to datetime if it's not already
            df['signup_date'] = pd.to_datetime(df['signup_date'])
            
            # Group by month
            monthly_signups = df.groupby(df['signup_date'].dt.to_period('M')).size()
            
            fig = px.line(x=monthly_signups.index.astype(str), y=monthly_signups.values,
                         title="Monthly Customer Acquisitions",
                         labels={'x': 'Month', 'y': 'New Customers'})
            st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations(self):
        """Render strategic recommendations dashboard"""
        st.subheader("🚀 Strategic Recommendations")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Please initialize the AI agent from the sidebar first.")
            return
        
        # Generate recommendations
        with st.spinner("Generating strategic recommendations..."):
            result = self.agent.query("What strategic actions should we take based on customer data?", include_ai_insights=True)
        
        if 'error' in result:
            st.error(result['error'])
            return
        
        # Display AI insights
        if result.get('insights'):
            st.markdown("### 🤖 AI-Generated Strategic Plan")
            st.info(result['insights'])
        
        # Recommendations breakdown
        recommendations_data = result.get('data', {})
        
        if 'immediate_actions' in recommendations_data:
            st.markdown("### 🔥 Immediate Actions (This Week)")
            actions = recommendations_data['immediate_actions']
            
            for i, action in enumerate(actions, 1):
                st.write(f"{i}. {action}")
        
        if 'strategic_initiatives' in recommendations_data:
            st.markdown("### 📈 Strategic Initiatives (Next Quarter)")
            initiatives = recommendations_data['strategic_initiatives']
            
            for i, initiative in enumerate(initiatives, 1):
                st.write(f"{i}. {initiative}")
        
        if 'investment_priorities' in recommendations_data:
            st.markdown("### 💰 Investment Priorities")
            priorities = recommendations_data['investment_priorities']
            
            for i, priority in enumerate(priorities, 1):
                st.write(f"{i}. {priority}")
        
        # Action plan export
        st.markdown("### 📋 Export Action Plan")
        
        if st.button("📥 Generate Action Plan Report"):
            # Create comprehensive report
            report_content = f"""
# Customer Intelligence Action Plan
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{result.get('insights', 'Strategic recommendations based on customer data analysis.')}

## Immediate Actions
"""
            
            if 'immediate_actions' in recommendations_data:
                for i, action in enumerate(recommendations_data['immediate_actions'], 1):
                    report_content += f"{i}. {action}\n"
            
            report_content += "\n## Strategic Initiatives\n"
            
            if 'strategic_initiatives' in recommendations_data:
                for i, initiative in enumerate(recommendations_data['strategic_initiatives'], 1):
                    report_content += f"{i}. {initiative}\n"
            
            report_content += "\n## Investment Priorities\n"
            
            if 'investment_priorities' in recommendations_data:
                for i, priority in enumerate(recommendations_data['investment_priorities'], 1):
                    report_content += f"{i}. {priority}\n"
            
            st.download_button(
                label="📄 Download Action Plan",
                data=report_content,
                file_name=f"customer_intelligence_action_plan_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Route to appropriate page
        if selected_page == "🏠 Overview Dashboard":
            self.render_overview_dashboard()
        elif selected_page == "💬 AI Chat Interface":
            self.render_chat_interface()
        elif selected_page == "📊 Data Explorer":
            self.render_data_explorer()
        elif selected_page == "🎯 Churn Analysis":
            self.render_churn_analysis()
        elif selected_page == "👥 Customer Segmentation":
            self.render_segmentation()
        elif selected_page == "📈 Predictive Insights":
            self.render_predictions()
        elif selected_page == "🚀 Strategic Recommendations":
            self.render_recommendations()

def main():
    """Main entry point"""
    try:
        # Validate configuration
        Config.validate()
        
        # Run application
        app = CustomerIntelligenceApp()
        app.run()
        
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please check your .env file and ensure all required settings are configured.")
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check the logs for more details.")

if __name__ == "__main__":
    main()
