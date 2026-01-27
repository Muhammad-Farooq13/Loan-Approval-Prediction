"""
Unit tests for feature engineering
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.build_features import FeatureEngineer, engineer_features

class TestFeatureEngineer:
    """Tests for FeatureEngineer class"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_df = pd.DataFrame({
            'ApplicantIncome': [5000, 6000, 7000],
            'CoapplicantIncome': [1500, 2000, 0],
            'LoanAmount': [150, 200, 250],
            'Loan_Amount_Term': [360, 360, 180],
            'Married': ['Yes', 'No', 'Yes'],
            'Dependents': ['0', '1', '2'],
            'Education': ['Graduate', 'Graduate', 'Not Graduate'],
            'Self_Employed': ['No', 'Yes', 'No']
        })
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        assert engineer.poly_features is None
        assert engineer.feature_names == []
    
    def test_create_income_features(self):
        """Test income feature creation"""
        engineer = FeatureEngineer()
        df_features = engineer.create_income_features(self.sample_df)
        
        # Check new features exist
        assert 'TotalIncome' in df_features.columns
        assert 'IncomeRatio' in df_features.columns
        assert 'ApplicantIncome_log' in df_features.columns
        assert 'CoapplicantIncome_log' in df_features.columns
        
        # Check calculations
        assert df_features['TotalIncome'][0] == 6500  # 5000 + 1500
    
    def test_create_loan_features(self):
        """Test loan feature creation"""
        engineer = FeatureEngineer()
        df_features = engineer.create_loan_features(self.sample_df)
        
        # Check new features exist
        assert 'LoanAmount_log' in df_features.columns
        assert 'Loan_Term_Years' in df_features.columns
        
        # Check calculation
        assert df_features['Loan_Term_Years'][0] == 30  # 360 / 12
    
    def test_create_demographic_features(self):
        """Test demographic feature creation"""
        engineer = FeatureEngineer()
        df_features = engineer.create_demographic_features(self.sample_df)
        
        # Check new features exist
        assert 'MarriedWithDependents' in df_features.columns
        assert 'FamilySize' in df_features.columns
        assert 'GraduateSelfEmployed' in df_features.columns
    
    def test_build_features(self):
        """Test complete feature building"""
        engineer = FeatureEngineer()
        df_features = engineer.build_features(self.sample_df)
        
        # Check that original columns still exist
        assert 'ApplicantIncome' in df_features.columns
        assert 'LoanAmount' in df_features.columns
        
        # Check that new features were created
        assert 'TotalIncome' in df_features.columns
        assert 'LoanAmount_log' in df_features.columns
        
        # Check that number of columns increased
        assert len(df_features.columns) > len(self.sample_df.columns)
    
    def test_build_features_with_interactions(self):
        """Test feature building with interactions"""
        engineer = FeatureEngineer()
        df_features = engineer.build_features(
            self.sample_df, 
            include_interactions=True
        )
        
        # Check that interaction features were created
        interaction_cols = [col for col in df_features.columns if '_x_' in col]
        assert len(interaction_cols) > 0
    
    def test_create_binned_features(self):
        """Test binned feature creation"""
        engineer = FeatureEngineer()
        df_features = engineer.create_binned_features(
            self.sample_df, 
            columns=['ApplicantIncome', 'LoanAmount']
        )
        
        # Check that binned features were created
        assert 'ApplicantIncome_binned' in df_features.columns
        assert 'LoanAmount_binned' in df_features.columns

def test_engineer_features_function():
    """Test engineer_features convenience function"""
    df = pd.DataFrame({
        'ApplicantIncome': [5000, 6000, 7000],
        'CoapplicantIncome': [1500, 2000, 0],
        'LoanAmount': [150, 200, 250],
        'Loan_Amount_Term': [360, 360, 180],
        'Married': ['Yes', 'No', 'Yes'],
        'Dependents': ['0', '1', '2'],
        'Education': ['Graduate', 'Graduate', 'Not Graduate'],
        'Self_Employed': ['No', 'Yes', 'No']
    })
    
    df_engineered = engineer_features(df)
    
    # Check that features were engineered
    assert 'TotalIncome' in df_engineered.columns
    assert len(df_engineered.columns) > len(df.columns)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
