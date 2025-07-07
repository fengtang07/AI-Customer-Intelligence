"""
Data processing utilities for customer intelligence
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.df = None
        
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load customer data from file"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return self._create_sample_data()
            
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # Clean and validate
            df = self._clean_data(df)
            df = self._validate_data(df)
            
            self.df = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the loaded data"""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        if df_clean.shape[0] < initial_shape[0]:
            logger.info(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
        
        # Handle customer_id
        if 'customer_id' in df_clean.columns:
            if df_clean['customer_id'].dtype == 'object':
                # Convert CUST_001 format to numeric if needed
                if df_clean['customer_id'].str.contains('CUST_', na=False).any():
                    df_clean['customer_id_original'] = df_clean['customer_id']
                    df_clean['customer_id'] = df_clean['customer_id'].str.replace('CUST_', '').astype(int)
        
        # Handle gender encoding
        if 'gender' in df_clean.columns:
            if df_clean['gender'].dtype == 'object':
                df_clean['gender_original'] = df_clean['gender']
                df_clean['gender'] = df_clean['gender'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill categorical missing values with mode or 'Unknown'
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
        
        return df_clean
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality"""
        required_columns = ['customer_id']
        recommended_columns = ['age', 'total_spent', 'churn']
        
        # Check required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            logger.warning(f"Missing required columns: {missing_required}")
        
        # Check recommended columns
        missing_recommended = [col for col in recommended_columns if col not in df.columns]
        if missing_recommended:
            logger.info(f"Missing recommended columns: {missing_recommended}")
        
        # Validate data ranges
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 0) | (df['age'] > 120)].shape[0]
            if invalid_ages > 0:
                logger.warning(f"Found {invalid_ages} records with invalid ages")
                df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        
        if 'churn' in df.columns:
            valid_churn_values = df['churn'].isin([0, 1]).all()
            if not valid_churn_values:
                logger.warning("Churn column contains non-binary values")
        
        return df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration"""
        from faker import Faker
        fake = Faker()
        Faker.seed(42)
        np.random.seed(42)
        
        logger.info("Creating sample dataset for demonstration")
        
        n_customers = 1000
        data = []
        
        for i in range(n_customers):
            # Basic demographics
            age = np.random.normal(40, 15)
            age = max(18, min(80, int(age)))
            gender = np.random.choice([0, 1], p=[0.52, 0.48])  # 0=Female, 1=Male
            
            # Behavioral data
            monthly_visits = max(1, int(np.random.poisson(8)))
            time_on_site = max(1, np.random.exponential(10))
            total_orders = max(1, int(np.random.poisson(5)))
            
            # Spending (correlated with age and engagement)
            base_spending = 50 + (age - 18) * 3 + monthly_visits * 20
            total_spent = max(10, base_spending + np.random.normal(0, 200))
            
            # Satisfaction (influences churn)
            satisfaction_score = max(1, min(5, np.random.normal(3.8, 1.0)))
            
            # Churn (based on satisfaction and engagement)
            churn_probability = 0.1
            if satisfaction_score < 3.0:
                churn_probability += 0.4
            if monthly_visits < 5:
                churn_probability += 0.2
            if gender == 1:  # Male customers slightly higher churn in sample
                churn_probability += 0.1
            
            churn = np.random.binomial(1, min(churn_probability, 0.8))
            
            data.append({
                'customer_id': i + 1,
                'age': age,
                'gender': gender,
                'monthly_visits': monthly_visits,
                'time_on_site': round(time_on_site, 2),
                'total_orders': total_orders,
                'total_spent': round(total_spent, 2),
                'satisfaction_score': round(satisfaction_score, 2),
                'churn': churn,
                'signup_date': fake.date_between(start_date='-2y', end_date='today'),
                'last_login': fake.date_between(start_date='-30d', end_date='today')
            })
        
        df = pd.DataFrame(data)
        
        # Save sample data
        from config.config import Config
        config = Config()
        sample_path = config.RAW_DATA_DIR / "sample_customer_data.csv"
        df.to_csv(sample_path, index=False)
        logger.info(f"Sample data saved to {sample_path}")
        
        return df
