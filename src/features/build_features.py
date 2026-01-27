"""
Feature Engineering Module
This module contains functions for building and engineering features
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for feature engineering operations"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.poly_features = None
        self.feature_names = []
    
    def create_income_features(self, df):
        """
        Create income-related features
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Data with new income features
        """
        df = df.copy()
        
        if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
            # Total income
            df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
            
            # Income ratio
            df['IncomeRatio'] = df['ApplicantIncome'] / (df['CoapplicantIncome'] + 1)
            
            # Log transformations
            df['ApplicantIncome_log'] = np.log1p(df['ApplicantIncome'])
            df['CoapplicantIncome_log'] = np.log1p(df['CoapplicantIncome'])
            df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
            
            logger.info("Created income-related features")
        
        return df
    
    def create_loan_features(self, df):
        """
        Create loan-related features
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Data with new loan features
        """
        df = df.copy()
        
        if 'LoanAmount' in df.columns:
            # Log transformation
            df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
            
            # Binning
            df['LoanAmount_binned'] = pd.cut(
                df['LoanAmount'], 
                bins=[0, 100, 200, 300, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
            
            logger.info("Created loan-related features")
        
        if 'Loan_Amount_Term' in df.columns:
            # Convert to years
            df['Loan_Term_Years'] = df['Loan_Amount_Term'] / 12
            
            # Binning
            df['Loan_Term_Category'] = pd.cut(
                df['Loan_Amount_Term'],
                bins=[0, 180, 360, float('inf')],
                labels=['short', 'medium', 'long']
            )
        
        # Loan to income ratio
        if 'LoanAmount' in df.columns and 'TotalIncome' in df.columns:
            df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['TotalIncome'] + 1)
            
            # Monthly payment estimate
            if 'Loan_Amount_Term' in df.columns:
                df['MonthlyPayment'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)
                df['PaymentToIncomeRatio'] = (df['MonthlyPayment'] * 12) / (df['TotalIncome'] + 1)
        
        return df
    
    def create_demographic_features(self, df):
        """
        Create demographic-related features
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Data with new demographic features
        """
        df = df.copy()
        
        # Married with dependents
        if 'Married' in df.columns and 'Dependents' in df.columns:
            df['MarriedWithDependents'] = (
                (df['Married'] == 'Yes') & (df['Dependents'] != '0')
            ).astype(int)
        
        # Family size
        if 'Dependents' in df.columns:
            df['FamilySize'] = df['Dependents'].replace('3+', '3').astype(int) + 1
            if 'Married' in df.columns:
                df['FamilySize'] += (df['Married'] == 'Yes').astype(int)
        
        # Graduate and self-employed
        if 'Education' in df.columns and 'Self_Employed' in df.columns:
            df['GraduateSelfEmployed'] = (
                (df['Education'] == 'Graduate') & (df['Self_Employed'] == 'Yes')
            ).astype(int)
        
        logger.info("Created demographic features")
        
        return df
    
    def create_interaction_features(self, df, columns=None):
        """
        Create interaction features between specified columns
        
        Args:
            df: pandas.DataFrame
            columns: List of column names to create interactions
            
        Returns:
            pandas.DataFrame: Data with interaction features
        """
        df = df.copy()
        
        if columns is None:
            # Default interactions for numerical columns
            columns = df.select_dtypes(include=[np.number]).columns[:3]
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        logger.info(f"Created interaction features for {len(columns)} columns")
        
        return df
    
    def create_polynomial_features(self, df, columns=None, degree=2):
        """
        Create polynomial features
        
        Args:
            df: pandas.DataFrame
            columns: List of columns to create polynomial features
            degree: Degree of polynomial
            
        Returns:
            pandas.DataFrame: Data with polynomial features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns[:3]
        
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            poly_array = self.poly_features.fit_transform(df[columns])
        else:
            poly_array = self.poly_features.transform(df[columns])
        
        # Create feature names
        poly_feature_names = self.poly_features.get_feature_names_out(columns)
        
        # Add only new features (exclude original features)
        new_features = poly_array[:, len(columns):]
        new_feature_names = poly_feature_names[len(columns):]
        
        for i, name in enumerate(new_feature_names):
            df[name] = new_features[:, i]
        
        logger.info(f"Created {len(new_feature_names)} polynomial features")
        
        return df
    
    def create_binned_features(self, df, columns=None, n_bins=5):
        """
        Create binned versions of continuous features
        
        Args:
            df: pandas.DataFrame
            columns: List of columns to bin
            n_bins: Number of bins
            
        Returns:
            pandas.DataFrame: Data with binned features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in df.columns:
                df[f'{col}_binned'] = pd.qcut(
                    df[col], 
                    q=n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
        
        logger.info(f"Created binned features for {len(columns)} columns")
        
        return df
    
    def build_features(self, df, include_interactions=False, include_polynomial=False):
        """
        Build all features
        
        Args:
            df: pandas.DataFrame
            include_interactions: Whether to include interaction features
            include_polynomial: Whether to include polynomial features
            
        Returns:
            pandas.DataFrame: Data with all engineered features
        """
        logger.info("Building features...")
        
        # Create basic features
        df = self.create_income_features(df)
        df = self.create_loan_features(df)
        df = self.create_demographic_features(df)
        
        # Optional advanced features
        if include_interactions:
            numerical_cols = ['ApplicantIncome', 'LoanAmount', 'TotalIncome']
            df = self.create_interaction_features(df, numerical_cols)
        
        if include_polynomial:
            numerical_cols = ['ApplicantIncome', 'LoanAmount']
            df = self.create_polynomial_features(df, numerical_cols, degree=2)
        
        logger.info("Feature building completed")
        
        return df
    
    def get_feature_importance_names(self):
        """Get names of all created features"""
        return self.feature_names

def engineer_features(df, advanced=False):
    """
    Convenience function for feature engineering
    
    Args:
        df: pandas.DataFrame
        advanced: Whether to include advanced features
        
    Returns:
        pandas.DataFrame: Data with engineered features
    """
    engineer = FeatureEngineer()
    df_engineered = engineer.build_features(
        df,
        include_interactions=advanced,
        include_polynomial=advanced
    )
    
    return df_engineered

if __name__ == '__main__':
    # Test feature engineering
    # Create sample data
    sample_data = {
        'ApplicantIncome': [5000, 6000, 7000],
        'CoapplicantIncome': [1500, 2000, 0],
        'LoanAmount': [150, 200, 250],
        'Loan_Amount_Term': [360, 360, 180],
        'Married': ['Yes', 'No', 'Yes'],
        'Dependents': ['0', '1', '2'],
        'Education': ['Graduate', 'Graduate', 'Not Graduate'],
        'Self_Employed': ['No', 'Yes', 'No']
    }
    
    df = pd.DataFrame(sample_data)
    engineer = FeatureEngineer()
    df_engineered = engineer.build_features(df)
    
    print("Original columns:", df.columns.tolist())
    print("Engineered columns:", df_engineered.columns.tolist())
    print(f"Total features: {len(df_engineered.columns)}")
