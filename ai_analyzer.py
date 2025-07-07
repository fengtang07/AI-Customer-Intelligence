# ai_analyzer.py - Complete AI Analysis Module with LangChain
"""
AI Customer Intelligence Agent - Complete AI Analysis Module
Features LangChain agent framework with custom tools for sophisticated customer analysis
"""

import pandas as pd
import numpy as np
import sys
import os
import traceback
import platform
from typing import Dict, Any, Optional, List
import warnings

warnings.filterwarnings('ignore')


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

        # Check package availability
        packages = {
            'openai': 'OpenAI API',
            'langchain': 'LangChain Framework',
            'langchain_openai': 'LangChain OpenAI',
            'langchain_experimental': 'LangChain Experimental',
            'streamlit': 'Streamlit'
        }

        for package, description in packages.items():
            try:
                __import__(package)
                debug_info += f"✅ {description} ({package}): Available\n"
            except ImportError:
                debug_info += f"❌ {description} ({package}): Not installed\n"

        debug_info += f"\n🐍 Python Path: {sys.executable}"
        return debug_info

    except Exception as e:
        return f"❌ Debug error: {str(e)}"


def test_ai_connection(api_key: str) -> Dict[str, Any]:
    """Test AI connection with simple query"""
    if not api_key:
        return {"success": False, "error": "No API key provided"}

    try:
        from openai import OpenAI

        # Initialize client with proper error handling
        client = OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=2
        )

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


class CustomerAnalyzerTool:
    """Custom tool for customer data analysis"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.name = "Customer Data Analyzer"
        self.description = """Analyze customer behavior, churn patterns, segmentation, and risk factors. 
        Use for questions about customer insights, churn analysis, segmentation, or risk assessment.
        Input should be a natural language query about customer data."""

    def run(self, query: str) -> str:
        """Run customer analysis based on query"""
        try:
            query_lower = query.lower()

            if 'churn' in query_lower:
                return self._analyze_churn()
            elif 'segment' in query_lower:
                return self._analyze_segments()
            elif 'risk' in query_lower:
                return self._analyze_risk()
            elif 'satisfaction' in query_lower:
                return self._analyze_satisfaction()
            elif ('spending' in query_lower or 'revenue' in query_lower or 
                  'value' in query_lower or 'drives' in query_lower or 
                  'money' in query_lower or 'financial' in query_lower or
                  'profitable' in query_lower or 'worth' in query_lower):
                return self._analyze_spending()
            elif 'age' in query_lower or 'demographic' in query_lower:
                return self._analyze_demographics()
            else:
                return self._general_overview()

        except Exception as e:
            return f"❌ Analysis error: {str(e)}"

    def _analyze_churn(self) -> str:
        """Analyze customer churn patterns with specific insights"""
        try:
            if 'churn' not in self.df.columns:
                return "❌ No churn data available in dataset"

            churn_rate = self.df['churn'].mean() * 100
            churned_customers = self.df[self.df['churn'] == 1]
            active_customers = self.df[self.df['churn'] == 0]

            analysis = f"""
🎯 CHURN ANALYSIS - ACTIONABLE INSIGHTS:

📊 Critical Metrics:
• URGENT: {churn_rate:.1f}% churn rate ({len(churned_customers):,} customers lost)
• Revenue Impact: ${churned_customers['total_spent'].sum():,.2f} lost from churned customers
• Opportunity: ${active_customers['total_spent'].sum():,.2f} revenue at risk from {len(active_customers):,} active customers

💰 Financial Impact Analysis:
"""

            if 'total_spent' in self.df.columns:
                avg_spent_churned = churned_customers['total_spent'].mean()
                avg_spent_active = active_customers['total_spent'].mean()
                revenue_per_churned = churned_customers['total_spent'].sum()
                analysis += f"• Churned customers spent ${avg_spent_churned:.2f} on average (vs ${avg_spent_active:.2f} active)\n"
                analysis += f"• Total revenue lost: ${revenue_per_churned:,.2f}\n"

                # Calculate potential revenue if churn improved
                if churn_rate > 15:
                    potential_savings = len(churned_customers) * 0.5 * avg_spent_churned
                    analysis += f"• POTENTIAL RECOVERY: ${potential_savings:,.2f} if churn reduced by 50%\n"

            analysis += f"\n🔍 Root Cause Analysis:\n"

            if 'satisfaction_score' in self.df.columns:
                low_sat_churned = churned_customers[churned_customers['satisfaction_score'] < 3.0]
                low_sat_churn_rate = len(low_sat_churned) / len(churned_customers) * 100
                analysis += f"• {low_sat_churn_rate:.1f}% of churned customers had low satisfaction (<3.0)\n"

                # Satisfaction comparison
                avg_sat_churned = churned_customers['satisfaction_score'].mean()
                avg_sat_active = active_customers['satisfaction_score'].mean()
                satisfaction_gap = avg_sat_active - avg_sat_churned
                analysis += f"• Satisfaction gap: {satisfaction_gap:.2f} points lower for churned customers\n"

            if 'monthly_visits' in self.df.columns:
                low_engagement = churned_customers[
                    churned_customers['monthly_visits'] < self.df['monthly_visits'].median()]
                low_eng_rate = len(low_engagement) / len(churned_customers) * 100
                analysis += f"• {low_eng_rate:.1f}% of churned customers had below-average engagement\n"

            # Age-based insights
            if 'age' in self.df.columns:
                age_groups = {
                    'Young (<30)': churned_customers[churned_customers['age'] < 30],
                    'Middle (30-49)': churned_customers[
                        (churned_customers['age'] >= 30) & (churned_customers['age'] < 50)],
                    'Senior (50+)': churned_customers[churned_customers['age'] >= 50]
                }

                highest_churn_group = max(age_groups.keys(), key=lambda x: len(age_groups[x]))
                analysis += f"• Highest churn: {highest_churn_group} with {len(age_groups[highest_churn_group])} customers\n"

            analysis += f"""
🚀 IMMEDIATE ACTION PLAN:
• TARGET: {len(low_sat_churned) if 'satisfaction_score' in self.df.columns else 'N/A'} low-satisfaction customers for immediate intervention
• FOCUS: Improve satisfaction by {satisfaction_gap:.1f} points to match active customer levels
• MONITOR: {len(active_customers)} active customers showing similar risk patterns
• ROI: Potential ${potential_savings:,.2f} revenue recovery with 50% churn reduction

⚡ Next Steps: Implement satisfaction surveys, personalized retention campaigns, and engagement programs.
"""

            return analysis

        except Exception as e:
            return f"❌ Churn analysis error: {str(e)}"

    def _analyze_segments(self) -> str:
        """Analyze customer segments"""
        try:
            analysis = "👥 CUSTOMER SEGMENTATION ANALYSIS:\n\n"

            # Value-based segmentation
            if 'total_spent' in self.df.columns:
                spending_q75 = self.df['total_spent'].quantile(0.75)
                spending_q25 = self.df['total_spent'].quantile(0.25)

                high_value = self.df[self.df['total_spent'] > spending_q75]
                medium_value = self.df[
                    (self.df['total_spent'] >= spending_q25) &
                    (self.df['total_spent'] <= spending_q75)
                    ]
                low_value = self.df[self.df['total_spent'] < spending_q25]

                analysis += f"💰 Value Segments:\n"
                analysis += f"• High Value: {len(high_value)} customers ({len(high_value) / len(self.df) * 100:.1f}%)\n"
                analysis += f"• Medium Value: {len(medium_value)} customers ({len(medium_value) / len(self.df) * 100:.1f}%)\n"
                analysis += f"• Low Value: {len(low_value)} customers ({len(low_value) / len(self.df) * 100:.1f}%)\n\n"

            # Satisfaction-based segmentation
            if 'satisfaction_score' in self.df.columns:
                high_satisfaction = self.df[self.df['satisfaction_score'] >= 4.0]
                medium_satisfaction = self.df[
                    (self.df['satisfaction_score'] >= 3.0) &
                    (self.df['satisfaction_score'] < 4.0)
                    ]
                low_satisfaction = self.df[self.df['satisfaction_score'] < 3.0]

                analysis += f"😊 Satisfaction Segments:\n"
                analysis += f"• High Satisfaction: {len(high_satisfaction)} customers ({len(high_satisfaction) / len(self.df) * 100:.1f}%)\n"
                analysis += f"• Medium Satisfaction: {len(medium_satisfaction)} customers ({len(medium_satisfaction) / len(self.df) * 100:.1f}%)\n"
                analysis += f"• Low Satisfaction: {len(low_satisfaction)} customers ({len(low_satisfaction) / len(self.df) * 100:.1f}%)\n\n"

            # Engagement-based segmentation
            if 'monthly_visits' in self.df.columns:
                engagement_median = self.df['monthly_visits'].median()
                high_engagement = self.df[self.df['monthly_visits'] > engagement_median]
                low_engagement = self.df[self.df['monthly_visits'] <= engagement_median]

                analysis += f"📈 Engagement Segments:\n"
                analysis += f"• High Engagement: {len(high_engagement)} customers ({len(high_engagement) / len(self.df) * 100:.1f}%)\n"
                analysis += f"• Low Engagement: {len(low_engagement)} customers ({len(low_engagement) / len(self.df) * 100:.1f}%)\n\n"

            return analysis

        except Exception as e:
            return f"❌ Segmentation analysis error: {str(e)}"

    def _analyze_risk(self) -> str:
        """Analyze customer risk factors"""
        try:
            if 'churn' not in self.df.columns:
                return "❌ No churn data available for risk analysis"

            # Identify at-risk customers (not yet churned but showing risk factors)
            at_risk_conditions = (self.df['churn'] == 0)  # Active customers only

            if 'satisfaction_score' in self.df.columns:
                at_risk_conditions &= (self.df['satisfaction_score'] < 3.5)

            if 'monthly_visits' in self.df.columns:
                at_risk_conditions &= (self.df['monthly_visits'] < self.df['monthly_visits'].median())

            at_risk_customers = self.df[at_risk_conditions]

            analysis = f"""
⚠️ CUSTOMER RISK ANALYSIS:

🎯 At-Risk Customer Identification:
• Customers at risk: {len(at_risk_customers):,} ({len(at_risk_customers) / len(self.df) * 100:.1f}%)
• Risk criteria: Low satisfaction (<3.5) + Low engagement + Still active

📊 Risk Factor Analysis:
"""

            if 'satisfaction_score' in self.df.columns:
                low_sat_count = len(self.df[self.df['satisfaction_score'] < 3.0])
                analysis += f"• Low satisfaction (<3.0): {low_sat_count} customers\n"

            if 'monthly_visits' in self.df.columns:
                low_engagement_count = len(self.df[self.df['monthly_visits'] < self.df['monthly_visits'].median()])
                analysis += f"• Low engagement: {low_engagement_count} customers\n"

            if len(at_risk_customers) > 0:
                analysis += f"\n💡 At-Risk Customer Profile:\n"

                if 'age' in at_risk_customers.columns:
                    analysis += f"• Average age: {at_risk_customers['age'].mean():.1f}\n"

                if 'total_spent' in at_risk_customers.columns:
                    analysis += f"• Average spending: ${at_risk_customers['total_spent'].mean():.2f}\n"

                analysis += "\n🚨 Recommendation: These customers need immediate attention!"

            return analysis

        except Exception as e:
            return f"❌ Risk analysis error: {str(e)}"

    def _analyze_satisfaction(self) -> str:
        """Analyze customer satisfaction patterns"""
        try:
            if 'satisfaction_score' not in self.df.columns:
                return "❌ No satisfaction data available"

            sat_stats = self.df['satisfaction_score'].describe()

            analysis = f"""
😊 CUSTOMER SATISFACTION ANALYSIS:

📊 Satisfaction Statistics:
• Average Score: {sat_stats['mean']:.2f}/5.0
• Median Score: {sat_stats['50%']:.2f}/5.0
• Standard Deviation: {sat_stats['std']:.2f}
• Range: {sat_stats['min']:.1f} - {sat_stats['max']:.1f}

📈 Distribution:
"""

            # Satisfaction distribution
            high_sat = len(self.df[self.df['satisfaction_score'] >= 4.0])
            medium_sat = len(self.df[(self.df['satisfaction_score'] >= 3.0) & (self.df['satisfaction_score'] < 4.0)])
            low_sat = len(self.df[self.df['satisfaction_score'] < 3.0])

            analysis += f"• High (4.0+): {high_sat} customers ({high_sat / len(self.df) * 100:.1f}%)\n"
            analysis += f"• Medium (3.0-3.9): {medium_sat} customers ({medium_sat / len(self.df) * 100:.1f}%)\n"
            analysis += f"• Low (<3.0): {low_sat} customers ({low_sat / len(self.df) * 100:.1f}%)\n"

            # Satisfaction vs other metrics
            if 'churn' in self.df.columns:
                churned_sat = self.df[self.df['churn'] == 1]['satisfaction_score'].mean()
                active_sat = self.df[self.df['churn'] == 0]['satisfaction_score'].mean()
                analysis += f"\n🔍 Satisfaction Impact:\n"
                analysis += f"• Churned customers avg satisfaction: {churned_sat:.2f}\n"
                analysis += f"• Active customers avg satisfaction: {active_sat:.2f}\n"

            return analysis

        except Exception as e:
            return f"❌ Satisfaction analysis error: {str(e)}"

    def _analyze_spending(self) -> str:
        """Analyze customer spending patterns with actionable insights"""
        try:
            if 'total_spent' not in self.df.columns:
                return "❌ No spending data available"

            spending_stats = self.df['total_spent'].describe()
            total_revenue = self.df['total_spent'].sum()

            analysis = f"""
💰 SPENDING ANALYSIS - REVENUE OPTIMIZATION:

💵 Revenue Performance:
• Total Revenue: ${total_revenue:,.2f}
• Average Customer Value: ${spending_stats['mean']:.2f}
• Revenue per Customer Range: ${spending_stats['min']:.2f} - ${spending_stats['max']:.2f}
• Median Spend: ${spending_stats['50%']:.2f} (shows typical customer behavior)

📊 Customer Value Segmentation:
"""

            # Advanced spending segments with business implications
            q25 = spending_stats['25%']
            q75 = spending_stats['75%']
            q90 = self.df['total_spent'].quantile(0.9)

            vip_customers = self.df[self.df['total_spent'] > q90]
            high_spenders = self.df[(self.df['total_spent'] > q75) & (self.df['total_spent'] <= q90)]
            medium_spenders = self.df[(self.df['total_spent'] >= q25) & (self.df['total_spent'] <= q75)]
            low_spenders = self.df[self.df['total_spent'] < q25]

            # Calculate revenue contribution
            vip_revenue = vip_customers['total_spent'].sum()
            high_revenue = high_spenders['total_spent'].sum()
            medium_revenue = medium_spenders['total_spent'].sum()
            low_revenue = low_spenders['total_spent'].sum()

            analysis += f"• VIP Customers (Top 10%): {len(vip_customers)} customers → ${vip_revenue:,.2f} ({vip_revenue / total_revenue * 100:.1f}% of revenue)\n"
            analysis += f"• High Value: {len(high_spenders)} customers → ${high_revenue:,.2f} ({high_revenue / total_revenue * 100:.1f}% of revenue)\n"
            analysis += f"• Medium Value: {len(medium_spenders)} customers → ${medium_revenue:,.2f} ({medium_revenue / total_revenue * 100:.1f}% of revenue)\n"
            analysis += f"• Low Value: {len(low_spenders)} customers → ${low_revenue:,.2f} ({low_revenue / total_revenue * 100:.1f}% of revenue)\n"

            # Calculate average spend per segment
            analysis += f"""
🎯 SEGMENT PERFORMANCE:
• VIP Average: ${vip_customers['total_spent'].mean():.2f} per customer
• High Value Average: ${high_spenders['total_spent'].mean():.2f} per customer  
• Medium Value Average: ${medium_spenders['total_spent'].mean():.2f} per customer
• Low Value Average: ${low_spenders['total_spent'].mean():.2f} per customer
"""

            # Business opportunities
            pareto_customers = len(vip_customers) + len(high_spenders)
            pareto_revenue = vip_revenue + high_revenue
            pareto_percentage = pareto_revenue / total_revenue * 100

            analysis += f"""
💡 BUSINESS OPPORTUNITIES:
• PARETO PRINCIPLE: {pareto_customers} customers ({pareto_customers / len(self.df) * 100:.1f}%) generate ${pareto_revenue:,.2f} ({pareto_percentage:.1f}%) of total revenue
• UPSELL TARGET: {len(medium_spenders)} medium-value customers could increase revenue by ${(high_spenders['total_spent'].mean() - medium_spenders['total_spent'].mean()) * len(medium_spenders):,.2f}
• RETENTION FOCUS: Protect top {len(vip_customers)} VIP customers (worth ${vip_revenue:,.2f} annually)
"""

            # Gender analysis if available
            if 'gender' in self.df.columns:
                male_avg = self.df[self.df['gender'] == 'M']['total_spent'].mean()
                female_avg = self.df[self.df['gender'] == 'F']['total_spent'].mean()
                gender_diff = abs(male_avg - female_avg)
                higher_gender = 'Male' if male_avg > female_avg else 'Female'

                analysis += f"""
👥 GENDER INSIGHTS:
• Male Average Spend: ${male_avg:.2f}
• Female Average Spend: ${female_avg:.2f}
• {higher_gender} customers spend ${gender_diff:.2f} more on average
• OPPORTUNITY: Target lower-spending gender with personalized campaigns
"""

            # Churn impact on revenue
            if 'churn' in self.df.columns:
                churned_revenue_lost = self.df[self.df['churn'] == 1]['total_spent'].sum()
                at_risk_revenue = self.df[self.df['churn'] == 0]['total_spent'].sum()

                analysis += f"""
⚠️ REVENUE AT RISK:
• Lost Revenue: ${churned_revenue_lost:,.2f} from churned customers
• Active Revenue: ${at_risk_revenue:,.2f} needs protection
• Churn Prevention ROI: Every 1% churn reduction = ${at_risk_revenue * 0.01:,.2f} revenue saved
"""

            analysis += f"""
🚀 ACTION PLAN:
1. IMMEDIATE: Launch VIP retention program for top {len(vip_customers)} customers
2. GROWTH: Upsell {len(medium_spenders)} medium-value customers (+${(high_spenders['total_spent'].mean() - medium_spenders['total_spent'].mean()) * len(medium_spenders):,.0f} potential)
3. ENGAGEMENT: Re-activate {len(low_spenders)} low-value customers with targeted offers
4. MONITOR: Track monthly spend changes in each segment

💰 REVENUE IMPACT: Successfully executing this plan could increase total revenue by 15-25%.
"""

            return analysis

        except Exception as e:
            return f"❌ Spending analysis error: {str(e)}"

    def _analyze_demographics(self) -> str:
        """Analyze customer demographics"""
        try:
            analysis = "👥 DEMOGRAPHIC ANALYSIS:\n\n"

            if 'age' in self.df.columns:
                age_stats = self.df['age'].describe()
                analysis += f"📊 Age Distribution:\n"
                analysis += f"• Average Age: {age_stats['mean']:.1f}\n"
                analysis += f"• Age Range: {age_stats['min']:.0f} - {age_stats['max']:.0f}\n"
                analysis += f"• Median Age: {age_stats['50%']:.1f}\n\n"

                # Age groups
                young = len(self.df[self.df['age'] < 30])
                middle = len(self.df[(self.df['age'] >= 30) & (self.df['age'] < 50)])
                senior = len(self.df[self.df['age'] >= 50])

                analysis += f"Age Groups:\n"
                analysis += f"• Young (<30): {young} customers ({young / len(self.df) * 100:.1f}%)\n"
                analysis += f"• Middle (30-49): {middle} customers ({middle / len(self.df) * 100:.1f}%)\n"
                analysis += f"• Senior (50+): {senior} customers ({senior / len(self.df) * 100:.1f}%)\n\n"

            if 'gender' in self.df.columns:
                gender_dist = self.df['gender'].value_counts()
                analysis += f"👫 Gender Distribution:\n"
                for gender, count in gender_dist.items():
                    analysis += f"• {gender}: {count} customers ({count / len(self.df) * 100:.1f}%)\n"

            return analysis

        except Exception as e:
            return f"❌ Demographics analysis error: {str(e)}"

    def _general_overview(self) -> str:
        """General dataset overview"""
        try:
            analysis = f"""
📊 DATASET OVERVIEW:

📈 Basic Metrics:
• Total Customers: {len(self.df):,}
• Data Columns: {len(self.df.columns)}
• Data Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns

📋 Available Columns: {', '.join(self.df.columns)}

"""

            if 'churn' in self.df.columns:
                churn_rate = self.df['churn'].mean() * 100
                analysis += f"• Churn Rate: {churn_rate:.1f}%\n"

            if 'age' in self.df.columns:
                analysis += f"• Average Age: {self.df['age'].mean():.1f}\n"

            if 'total_spent' in self.df.columns:
                analysis += f"• Average Spending: ${self.df['total_spent'].mean():.2f}\n"

            if 'satisfaction_score' in self.df.columns:
                analysis += f"• Average Satisfaction: {self.df['satisfaction_score'].mean():.2f}/5.0\n"

            return analysis

        except Exception as e:
            return f"❌ Overview error: {str(e)}"


class ProductCategoryAnalyzerTool:
    """Custom tool for product category analysis"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.name = "Product Category Analyzer"
        self.description = """Analyze product categories, sales by category, and category performance. 
        Use for questions about product categories, top selling products, category revenue, or product performance."""

    def run(self, query: str) -> str:
        """Run product category analysis based on query"""
        try:
            if 'product_category' not in self.df.columns:
                return "❌ No product category data available in dataset"

            query_lower = query.lower()

            if 'revenue' in query_lower or 'sales' in query_lower or 'top' in query_lower:
                return self._analyze_category_revenue()
            elif 'performance' in query_lower:
                return self._analyze_category_performance()
            elif 'popular' in query_lower or 'selling' in query_lower:
                return self._analyze_popular_categories()
            else:
                return self._general_category_overview()

        except Exception as e:
            return f"❌ Product category analysis error: {str(e)}"

    def _analyze_category_revenue(self) -> str:
        """Analyze revenue by product category"""
        try:
            if 'total_spent' not in self.df.columns:
                return "❌ No spending data available for revenue analysis"

            # Calculate revenue by category
            category_revenue = self.df.groupby('product_category')['total_spent'].agg(['sum', 'count', 'mean']).round(2)
            category_revenue = category_revenue.sort_values('sum', ascending=False)

            total_revenue = self.df['total_spent'].sum()

            analysis = f"""
📊 PRODUCT CATEGORY REVENUE ANALYSIS:

💰 Total Revenue: ${total_revenue:,.2f}

🏆 TOP CATEGORIES BY REVENUE:
"""

            for i, (category, data) in enumerate(category_revenue.head().iterrows(), 1):
                revenue = data['sum']
                customers = int(data['count'])
                avg_spend = data['mean']
                percentage = (revenue / total_revenue) * 100

                analysis += f"""
{i}. {category.upper()}:
   • Revenue: ${revenue:,.2f} ({percentage:.1f}% of total)
   • Customers: {customers:,} purchasers
   • Avg Spend per Customer: ${avg_spend:.2f}
"""

            # Category performance insights
            top_category = category_revenue.index[0]
            top_revenue = category_revenue.iloc[0]['sum']
            top_percentage = (top_revenue / total_revenue) * 100

            analysis += f"""
🔍 KEY INSIGHTS:
• LEADING CATEGORY: {top_category} dominates with ${top_revenue:,.2f} ({top_percentage:.1f}% of revenue)
• REVENUE CONCENTRATION: Top 3 categories generate ${category_revenue.head(3)['sum'].sum():,.2f} ({category_revenue.head(3)['sum'].sum() / total_revenue * 100:.1f}% of total)
• CATEGORY COUNT: {len(category_revenue)} different product categories

🚀 RECOMMENDATIONS:
• FOCUS: Invest more in {top_category} category (highest revenue generator)
• EXPAND: Cross-sell other categories to {top_category} customers
• OPTIMIZE: Improve marketing for underperforming categories
"""

            return analysis

        except Exception as e:
            return f"❌ Category revenue analysis error: {str(e)}"

    def _analyze_category_performance(self) -> str:
        """Analyze category performance metrics"""
        try:
            category_stats = self.df.groupby('product_category').agg({
                'total_spent': ['sum', 'mean', 'count'],
                'satisfaction_score': 'mean' if 'satisfaction_score' in self.df.columns else lambda x: None,
                'churn': 'mean' if 'churn' in self.df.columns else lambda x: None
            }).round(2)

            analysis = f"""
📈 PRODUCT CATEGORY PERFORMANCE:

Category Performance Metrics:
"""

            for category in category_stats.index:
                revenue = category_stats.loc[category, ('total_spent', 'sum')]
                avg_spend = category_stats.loc[category, ('total_spent', 'mean')]
                customers = int(category_stats.loc[category, ('total_spent', 'count')])

                analysis += f"\n{category.upper()}:\n"
                analysis += f"  • Revenue: ${revenue:,.2f}\n"
                analysis += f"  • Customers: {customers:,}\n"
                analysis += f"  • Avg Spend: ${avg_spend:.2f}\n"

                if 'satisfaction_score' in self.df.columns:
                    satisfaction = category_stats.loc[category, ('satisfaction_score', 'mean')]
                    if not pd.isna(satisfaction):
                        analysis += f"  • Satisfaction: {satisfaction:.2f}/5.0\n"

                if 'churn' in self.df.columns:
                    churn_rate = category_stats.loc[category, ('churn', 'mean')] * 100
                    if not pd.isna(churn_rate):
                        analysis += f"  • Churn Rate: {churn_rate:.1f}%\n"

            return analysis

        except Exception as e:
            return f"❌ Category performance analysis error: {str(e)}"

    def _analyze_popular_categories(self) -> str:
        """Analyze most popular categories by customer count"""
        try:
            category_popularity = self.df['product_category'].value_counts()
            total_customers = len(self.df)

            analysis = f"""
🌟 MOST POPULAR PRODUCT CATEGORIES:

📊 Popularity Ranking (by customer count):
"""

            for i, (category, count) in enumerate(category_popularity.head().items(), 1):
                percentage = (count / total_customers) * 100
                analysis += f"{i}. {category}: {count:,} customers ({percentage:.1f}%)\n"

            analysis += f"""
🔍 INSIGHTS:
• Most Popular: {category_popularity.index[0]} with {category_popularity.iloc[0]:,} customers
• Category Spread: {len(category_popularity)} different categories available
• Market Share: Top category captures {category_popularity.iloc[0] / total_customers * 100:.1f}% of customers
"""

            return analysis

        except Exception as e:
            return f"❌ Popular categories analysis error: {str(e)}"

    def _general_category_overview(self) -> str:
        """General overview of product categories"""
        try:
            categories = self.df['product_category'].unique()
            category_counts = self.df['product_category'].value_counts()

            analysis = f"""
🛍️ PRODUCT CATEGORY OVERVIEW:

📋 Available Categories: {len(categories)}
Categories: {', '.join(categories)}

📊 Customer Distribution:
"""

            for category, count in category_counts.items():
                percentage = (count / len(self.df)) * 100
                analysis += f"• {category}: {count:,} customers ({percentage:.1f}%)\n"

            return analysis

        except Exception as e:
            return f"❌ Category overview error: {str(e)}"


class StatisticalAnalyzerTool:
    """Custom tool for statistical analysis"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.name = "Statistical Analyzer"
        self.description = """Perform statistical analysis including correlations, distributions, 
        and numerical summaries. Use for questions about statistical patterns or data distributions."""

    def run(self, query: str) -> str:
        """Run statistical analysis based on query"""
        try:
            query_lower = query.lower()

            if 'correlation' in query_lower:
                return self._correlation_analysis()
            elif 'distribution' in query_lower:
                return self._distribution_analysis()
            elif 'regression' in query_lower or 'model' in query_lower:
                return self._predictive_analysis()
            else:
                return self._general_statistics()

        except Exception as e:
            return f"❌ Statistical analysis error: {str(e)}"

    def _correlation_analysis(self) -> str:
        """Analyze correlations in the data"""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                return "❌ Need at least 2 numeric columns for correlation analysis"

            analysis = "📊 CORRELATION ANALYSIS:\n\n"

            # Correlations with churn (if available)
            if 'churn' in numeric_cols:
                analysis += "🎯 Correlations with Churn:\n"
                correlations = {}

                for col in numeric_cols:
                    if col != 'churn' and col != 'customer_id':
                        corr = self.df[col].corr(self.df['churn'])
                        if abs(corr) > 0.05:  # Only show meaningful correlations
                            correlations[col] = corr

                # Sort by absolute correlation value
                sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

                for col, corr in sorted_corr:
                    direction = "📈 Positive" if corr > 0 else "📉 Negative"
                    strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
                    analysis += f"• {col}: {corr:.3f} ({direction}, {strength})\n"

                analysis += "\n"

            # General correlation insights
            analysis += "🔍 Key Correlation Insights:\n"

            # Find strongest correlations (excluding perfect correlations with self)
            corr_matrix = self.df[numeric_cols].corr()

            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]

                    if abs(corr_val) > 0.3:  # Strong correlation threshold
                        strong_correlations.append((col1, col2, corr_val))

            # Sort by correlation strength
            strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

            if strong_correlations:
                for col1, col2, corr in strong_correlations[:5]:  # Top 5
                    direction = "positively" if corr > 0 else "negatively"
                    analysis += f"• {col1} and {col2} are {direction} correlated ({corr:.3f})\n"
            else:
                analysis += "• No strong correlations (>0.3) found between variables\n"

            return analysis

        except Exception as e:
            return f"❌ Correlation analysis error: {str(e)}"

    def _distribution_analysis(self) -> str:
        """Analyze data distributions"""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                return "❌ No numeric columns found for distribution analysis"

            analysis = "📊 DISTRIBUTION ANALYSIS:\n\n"

            for col in numeric_cols:
                if col not in ['customer_id']:  # Skip ID columns
                    stats = self.df[col].describe()

                    analysis += f"📈 {col.title()}:\n"
                    analysis += f"• Mean: {stats['mean']:.2f}\n"
                    analysis += f"• Median: {stats['50%']:.2f}\n"
                    analysis += f"• Std Dev: {stats['std']:.2f}\n"
                    analysis += f"• Range: {stats['min']:.2f} - {stats['max']:.2f}\n"

                    # Skewness analysis
                    try:
                        from scipy import stats as scipy_stats
                        skewness = scipy_stats.skew(self.df[col].dropna())
                        if abs(skewness) > 1:
                            skew_desc = "highly skewed"
                        elif abs(skewness) > 0.5:
                            skew_desc = "moderately skewed"
                        else:
                            skew_desc = "approximately normal"

                        analysis += f"• Distribution: {skew_desc} (skewness: {skewness:.2f})\n"
                    except ImportError:
                        pass

                    analysis += "\n"

            return analysis

        except Exception as e:
            return f"❌ Distribution analysis error: {str(e)}"

    def _predictive_analysis(self) -> str:
        """Simple predictive analysis"""
        try:
            if 'churn' not in self.df.columns:
                return "❌ No target variable (churn) available for predictive analysis"

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in ['churn', 'customer_id']]

            if len(feature_cols) < 2:
                return "❌ Need at least 2 feature columns for predictive analysis"

            analysis = "🤖 PREDICTIVE ANALYSIS:\n\n"

            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, classification_report

                # Prepare data
                X = self.df[feature_cols].fillna(self.df[feature_cols].median())
                y = self.df['churn']

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Train model
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)

                # Predictions
                y_pred = rf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                analysis += f"🎯 Model Performance:\n"
                analysis += f"• Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)\n"
                analysis += f"• Training samples: {len(X_train)}\n"
                analysis += f"• Test samples: {len(X_test)}\n\n"

                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)

                analysis += "📊 Feature Importance (Top 5):\n"
                for idx, row in feature_importance.head().iterrows():
                    analysis += f"• {row['feature']}: {row['importance']:.3f}\n"

            except ImportError:
                analysis += "❌ Scikit-learn not available for advanced modeling\n"
                analysis += "📊 Basic predictive insights based on correlations:\n"

                # Fallback to correlation analysis
                for col in feature_cols:
                    corr = self.df[col].corr(self.df['churn'])
                    if abs(corr) > 0.1:
                        impact = "increases" if corr > 0 else "decreases"
                        analysis += f"• Higher {col} {impact} churn probability (corr: {corr:.3f})\n"

            return analysis

        except Exception as e:
            return f"❌ Predictive analysis error: {str(e)}"

    def _general_statistics(self) -> str:
        """General statistical summary"""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                return "❌ No numeric columns found for statistical analysis"

            analysis = "📊 GENERAL STATISTICS:\n\n"

            # Dataset overview
            analysis += f"📈 Dataset Summary:\n"
            analysis += f"• Total Records: {len(self.df):,}\n"
            analysis += f"• Numeric Columns: {len(numeric_cols)}\n"
            analysis += f"• Missing Values: {self.df.isnull().sum().sum()}\n\n"

            # Key statistics for main columns
            key_stats = self.df[numeric_cols].describe()

            analysis += "📋 Key Statistics:\n"
            for col in numeric_cols:
                if col not in ['customer_id']:
                    mean_val = key_stats.loc['mean', col]
                    std_val = key_stats.loc['std', col]

                    analysis += f"• {col}: μ={mean_val:.2f}, σ={std_val:.2f}\n"

            return analysis

        except Exception as e:
            return f"❌ General statistics error: {str(e)}"


def setup_langchain_agent(api_key: str, df: pd.DataFrame):
    """
    Set up LangChain agent with custom tools for customer analysis
    """
    try:
        # Import LangChain components with error handling
        try:
            from langchain.agents import initialize_agent, Tool, AgentType
            from langchain_openai import OpenAI
            from langchain.memory import ConversationBufferMemory
        except ImportError:
            # Fallback for newer LangChain versions
            from langchain.agents import create_react_agent, AgentExecutor
            from langchain.tools import Tool

        # Always import ChatOpenAI as fallback
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            pass

        # Try to import PythonREPLTool
        try:
            from langchain_experimental.tools import PythonREPLTool
            use_python_tool = True
        except ImportError:
            use_python_tool = False

        # Initialize LLM with proper configuration
        try:
            from langchain_openai import OpenAI
            llm = OpenAI(
                temperature=0,
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo-instruct",
                max_tokens=1000,
                request_timeout=60
            )
        except:
            # Fallback to ChatOpenAI for newer versions
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    temperature=0,
                    openai_api_key=api_key,
                    model_name="gpt-3.5-turbo",
                    max_tokens=1000,
                    request_timeout=60
                )
            except Exception as llm_error:
                return f"❌ LLM initialization error: {str(llm_error)}"

        # Create custom tools
        customer_tool = CustomerAnalyzerTool(df)
        statistical_tool = StatisticalAnalyzerTool(df)
        product_tool = ProductCategoryAnalyzerTool(df)

        # Create tools list
        tools = [
            Tool(
                name=customer_tool.name,
                func=customer_tool.run,
                description=customer_tool.description
            ),
            Tool(
                name=statistical_tool.name,
                func=statistical_tool.run,
                description=statistical_tool.description
            ),
            Tool(
                name=product_tool.name,
                func=product_tool.run,
                description=product_tool.description
            )
        ]

        # Add Python REPL tool if available
        if use_python_tool:
            tools.append(PythonREPLTool())

        # Initialize agent with error handling for version compatibility
        try:
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True
            )
        except:
            # Simplified agent for compatibility
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent="conversational-react-description",
                verbose=False,
                max_iterations=3
            )

        return agent

    except ImportError as e:
        return f"❌ LangChain import error: {str(e)}"
    except Exception as e:
        return f"❌ Agent setup error: {str(e)}"


def analyze_with_langchain(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """
    Analyze customer data using LangChain agent with enhanced prompting
    """
    if not api_key:
        return "⚠️ No API key provided"

    try:
        # Set up LangChain agent
        agent = setup_langchain_agent(api_key, df)

        if isinstance(agent, str):
            # LangChain failed, fallback to direct analysis
            return f"🔄 **LangChain Unavailable - Using Direct Analysis:**\n\n{analyze_with_direct_openai(question, df, api_key, response_style)}"

        # Define response style for LangChain
        style_instructions = {
            'smart': """
RESPONSE STYLE: Comprehensive and detailed analysis with structured sections and emojis.
FORMAT: Use clear sections with emojis, detailed explanations, and actionable insights.
DETAIL LEVEL: Include all tool outputs with full statistical details and business context.
""",
            'concise': """
RESPONSE STYLE: Brief and to-the-point with essential findings only.
FORMAT: Use bullet points and concise statements. Focus on key metrics and top 3 recommendations.
DETAIL LEVEL: Include only the most critical insights from tools, no lengthy explanations.
""",
            'technical': """
RESPONSE STYLE: Technical analysis with statistical depth and analytical terminology.
FORMAT: Use statistical terminology, include correlation coefficients, confidence intervals, and technical metrics.
DETAIL LEVEL: Include all statistical measures, detailed methodology, and quantitative analysis from tools.
"""
        }

        current_style_instruction = style_instructions.get(response_style, style_instructions['smart'])

        # Enhanced context with specific instructions for actionable insights
        context = f"""
You are a SENIOR CUSTOMER ANALYTICS CONSULTANT providing actionable business insights.

{current_style_instruction}

DATASET CONTEXT:
- Customer Database: {len(df):,} customers
- Columns Available: {list(df.columns)}
- Business Metrics Overview:
"""

        if 'churn' in df.columns:
            churn_rate = df['churn'].mean() * 100
            context += f"  • Churn Rate: {churn_rate:.1f}% ({'CRITICAL' if churn_rate > 20 else 'MODERATE' if churn_rate > 10 else 'GOOD'})\n"
        if 'total_spent' in df.columns:
            total_revenue = df['total_spent'].sum()
            avg_spend = df['total_spent'].mean()
            context += f"  • Total Revenue: ${total_revenue:,.2f}\n"
            context += f"  • Avg Customer Value: ${avg_spend:.2f}\n"
        if 'satisfaction_score' in df.columns:
            avg_satisfaction = df['satisfaction_score'].mean()
            context += f"  • Avg Satisfaction: {avg_satisfaction:.2f}/5.0 ({'EXCELLENT' if avg_satisfaction >= 4.5 else 'GOOD' if avg_satisfaction >= 4.0 else 'NEEDS IMPROVEMENT'})\n"

        context += f"""

ANALYSIS REQUIREMENT:
Question: "{question}"

TOOL SELECTION GUIDELINES:
- For PRODUCT CATEGORY questions (product revenue, top categories, category performance): Use "Product Category Analyzer"
- For CUSTOMER BEHAVIOR questions (churn, segments, demographics, satisfaction): Use "Customer Data Analyzer"  
- For STATISTICAL questions (correlations, distributions): Use "Statistical Analyzer"
- For CALCULATIONS: Use "Python REPL Tool"

CRITICAL INSTRUCTIONS:
1. ALWAYS USE THE APPROPRIATE TOOL FIRST - never answer without using tools
2. **MANDATORY**: COPY AND PASTE THE COMPLETE TOOL OUTPUT into your response exactly as provided
3. DO NOT SUMMARIZE OR PARAPHRASE the tool output - include ALL statistics, percentages, and insights
4. After presenting the complete tool results, add your own strategic business interpretation
5. Use ONLY the specific numbers and insights from the tool outputs
6. Calculate additional FINANCIAL IMPACT based on tool data
7. Identify OPPORTUNITIES and RISKS from the tool analysis
8. Suggest IMPLEMENTATION STRATEGIES based on tool findings

**RESPONSE FORMAT REQUIREMENT:**
```
[TOOL NAME] ANALYSIS:
[PASTE COMPLETE TOOL OUTPUT HERE - DO NOT MODIFY OR SUMMARIZE]

STRATEGIC BUSINESS INTERPRETATION:
[Your analysis and recommendations based on the tool output]

IMMEDIATE ACTION PLAN:
[Specific next steps based on the data]
```

ABSOLUTE REQUIREMENTS:
- NEVER give generic responses like "it is clear that..." or "it is recommended to..."
- ALWAYS include the specific statistics from tool output (percentages, counts, dollar amounts)
- INCLUDE ALL sections from the tool output (statistics, distribution, impact analysis, etc.)
- DO NOT create your own analysis without first presenting the complete tool results

ANSWER THE EXACT QUESTION ASKED. If asked about satisfaction, use Customer Data Analyzer for satisfaction insights and present its complete output.
"""

        # Run the agent with enhanced context with fallback  
        try:
            result = agent.run(context)
        except Exception as agent_error:
            # Agent execution failed, fallback to direct analysis
            return f"🔄 **LangChain Agent Failed - Using Direct Analysis:**\n\n{analyze_with_direct_openai(question, df, api_key, response_style)}"

        # Format the result based on response style
        if response_style == 'concise':
            formatted_result = f"""**Key Insights:**

{result}"""
        elif response_style == 'technical':
            formatted_result = f"""**Technical Analysis:**

{result}

**Methodology:** LangChain ReAct Agent with specialized analytical tools
**Statistical Framework:** Multi-tool analysis with quantitative validation
"""
        else:  # smart
            formatted_result = f"""🤖 **LangChain Agent Analysis:**

{result}

---
✅ **Method:** LangChain ReAct Agent with Enhanced Business Intelligence
🛠️ **Tools Used:** Customer Data Analyzer, Product Category Analyzer, Statistical Analyzer, Python REPL
📊 **Data Points:** {len(df):,} customers analyzed
⚡ **Focus:** Actionable insights with financial impact analysis
"""
        
        return formatted_result

    except Exception as e:
        # Complete fallback to direct OpenAI if everything fails
        return f"🔄 **LangChain Error - Using Direct Analysis:**\n\n{analyze_with_direct_openai(question, df, api_key, response_style)}"


def analyze_with_direct_openai(question: str, df: pd.DataFrame, api_key: str, response_style: str = 'smart'):
    """
    Direct OpenAI analysis with enhanced business intelligence prompting
    """
    try:
        from openai import OpenAI

        # Initialize client with proper configuration
        client = OpenAI(
            api_key=api_key,
            timeout=60.0,
            max_retries=3
        )

        # Create comprehensive data summary with business context
        total_revenue = df['total_spent'].sum() if 'total_spent' in df.columns else 0
        churn_rate = df['churn'].mean() * 100 if 'churn' in df.columns else 0
        avg_satisfaction = df['satisfaction_score'].mean() if 'satisfaction_score' in df.columns else 0

        data_summary = f"""
CUSTOMER ANALYTICS DATABASE - BUSINESS INTELLIGENCE BRIEF:

📊 EXECUTIVE SUMMARY:
- Customer Base: {len(df):,} total customers
- Total Revenue: ${total_revenue:,.2f}
- Churn Rate: {churn_rate:.1f}% ({'🚨 CRITICAL' if churn_rate > 20 else '⚠️ MONITOR' if churn_rate > 10 else '✅ HEALTHY'})
- Customer Satisfaction: {avg_satisfaction:.2f}/5.0 ({'🟢 EXCELLENT' if avg_satisfaction >= 4.5 else '🟡 GOOD' if avg_satisfaction >= 4.0 else '🔴 NEEDS WORK'})

📈 KEY PERFORMANCE INDICATORS:
"""

        if 'total_spent' in df.columns:
            spending_stats = df['total_spent'].describe()
            data_summary += f"- Average Customer Value: ${spending_stats['mean']:.2f}\n"
            data_summary += f"- Customer Value Range: ${spending_stats['min']:.2f} - ${spending_stats['max']:.2f}\n"

            # Value segments
            high_value = len(df[df['total_spent'] > spending_stats['75%']])
            data_summary += f"- High-Value Customers: {high_value} ({high_value / len(df) * 100:.1f}%)\n"

        if 'age' in df.columns:
            age_stats = df['age'].describe()
            data_summary += f"- Customer Age Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years (avg: {age_stats['mean']:.1f})\n"

        if 'gender' in df.columns:
            gender_dist = df['gender'].value_counts()
            data_summary += f"- Gender Split: {gender_dist.to_dict()}\n"

        # Sample data for context
        data_summary += f"\nSAMPLE CUSTOMER RECORDS:\n{df.head(3).to_string()}\n"

        # Define response style parameters
        style_config = {
            'smart': {
                'length': 'comprehensive and detailed',
                'format': 'structured with emojis and sections',
                'tone': 'professional yet engaging',
                'detail_level': 'high detail with examples and explanations'
            },
            'concise': {
                'length': 'brief and to-the-point',
                'format': 'bullet points with key insights only',
                'tone': 'direct and efficient',
                'detail_level': 'essential findings only, no lengthy explanations'
            },
            'technical': {
                'length': 'detailed with statistical depth',
                'format': 'technical analysis with statistical terminology',
                'tone': 'analytical and data-focused',
                'detail_level': 'include statistical measures, correlations, and technical metrics'
            }
        }
        
        current_style = style_config.get(response_style, style_config['smart'])

        # Enhanced prompt for actionable insights with style customization
        prompt = f"""
You are a SENIOR CUSTOMER ANALYTICS CONSULTANT hired to provide strategic business intelligence.

{data_summary}

ANALYSIS REQUEST: "{question}"

RESPONSE STYLE: {current_style['length']} - {current_style['tone']} - {current_style['detail_level']}
FORMAT: {current_style['format']}

REQUIREMENTS - Your response must include:
1. SPECIFIC DATA INSIGHTS: Use exact numbers, percentages, and dollar amounts from the data
2. FINANCIAL IMPACT: Calculate revenue implications, potential losses, and opportunities  
3. CUSTOMER SEGMENTATION: Identify specific groups to target or protect
4. ACTIONABLE RECOMMENDATIONS: Provide concrete steps with timelines
5. BUSINESS PRIORITIES: Rank recommendations by impact and urgency
6. SUCCESS METRICS: Define KPIs to measure implementation success

{'FORMAT YOUR RESPONSE AS:' if response_style != 'concise' else 'PROVIDE:'}
{'🔍 KEY FINDINGS: [Specific data-driven insights with numbers]' if response_style == 'smart' else 'KEY FINDINGS:' if response_style == 'concise' else 'STATISTICAL FINDINGS: [Include correlation coefficients, statistical significance, and confidence intervals]'}
{'💰 FINANCIAL IMPACT: [Revenue/cost implications with calculations]' if response_style == 'smart' else 'FINANCIAL IMPACT:' if response_style == 'concise' else 'QUANTITATIVE IMPACT: [Detailed financial modeling with variance analysis]'}
{'🎯 TARGET SEGMENTS: [Specific customer groups with characteristics]' if response_style == 'smart' else 'TARGET SEGMENTS:' if response_style == 'concise' else 'SEGMENTATION ANALYSIS: [Statistical clustering and demographic profiling]'}
{'🚀 ACTION PLAN: [Prioritized recommendations with implementation steps]' if response_style == 'smart' else 'ACTIONS:' if response_style == 'concise' else 'IMPLEMENTATION STRATEGY: [Detailed methodology with risk assessment]'}
{'📊 SUCCESS METRICS: [KPIs to track progress]' if response_style == 'smart' else 'METRICS:' if response_style == 'concise' else 'STATISTICAL MEASURES: [Control variables and measurement framework]'}

CRITICAL: Never give generic advice. Base everything on the actual data provided. Include specific numbers, percentages, and dollar amounts to demonstrate real business value.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a senior business analytics consultant specializing in customer intelligence and revenue optimization. You provide data-driven, actionable insights that directly impact business performance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )

        result = response.choices[0].message.content

        return f"""🤖 **Direct OpenAI Analysis:**

{result}

---
✅ **Method:** Enhanced Business Intelligence Analysis
📊 **Data Points:** {len(df):,} customers analyzed
💰 **Revenue Context:** ${total_revenue:,.2f} total customer value
🎯 **Focus:** Strategic recommendations with financial impact
"""

    except Exception as e:
        return f"❌ Direct OpenAI error: {str(e)}"


def analyze_with_ai(question: str, df: pd.DataFrame, api_key: str, use_langchain: bool = True, response_style: str = 'smart'):
    """
    Main AI analysis function with fallback options
    """
    if not api_key:
        return "⚠️ Please enter your OpenAI API key to enable AI analysis."

    if df is None or len(df) == 0:
        return "⚠️ No data available for analysis. Please upload a dataset first."

    # Try LangChain first if requested
    if use_langchain:
        try:
            return analyze_with_langchain(question, df, api_key, response_style)
        except Exception as e:
            # Fall back to direct OpenAI if LangChain fails
            fallback_result = analyze_with_direct_openai(question, df, api_key, response_style)
            return f"⚠️ LangChain failed, using direct OpenAI:\n{str(e)}\n\n{fallback_result}"
    else:
        # Use direct OpenAI
        return analyze_with_direct_openai(question, df, api_key, response_style)


# Test and utility functions
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
        'churn': np.random.choice([0, 1], 100, p=[0.75, 0.25])
    })

    question = "What are the main drivers of customer churn?"

    print("🧪 Testing LangChain Method:")
    print("=" * 50)
    langchain_result = analyze_with_langchain(question, test_df, api_key)
    print(langchain_result)

    print("\n🧪 Testing Direct OpenAI Method:")
    print("=" * 50)
    direct_result = analyze_with_direct_openai(question, test_df, api_key)
    print(direct_result)


if __name__ == "__main__":
    # Quick test - only run interactive input in terminal environments
    print("🔧 AI Analyzer Module Test")
    print("Debug Environment:")
    print(debug_environment())
    
    # Check if we have an interactive terminal
    try:
        # This will only work if we have a real interactive terminal
        if sys.stdin.isatty():
            api_key = input("\nEnter OpenAI API key (optional, for testing): ").strip()
            if api_key:
                test_both_methods(api_key)
            else:
                print("✅ Module loaded successfully. Add API key to test AI functions.")
        else:
            print("✅ Module loaded successfully. Add API key to test AI functions.")
    except:
        # If anything fails, just show the success message
        print("✅ Module loaded successfully. Add API key to test AI functions.")