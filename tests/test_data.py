"""
Unit tests for data loading and preprocessing
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.load_data import DataLoader, load_loan_data
from src.data.preprocess import DataPreprocessor

class TestDataLoader:
    """Tests for DataLoader class"""
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.data_dir == 'data'
        assert loader.raw_data_dir == 'data/raw'
        assert loader.processed_data_dir == 'data/processed'
    
    def test_get_data_info(self):
        """Test get_data_info method"""
        # Create sample data
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z']
        })
        
        loader = DataLoader()
        info = loader.get_data_info(df)
        
        assert info['rows'] == 3
        assert info['columns'] == 3
        assert 'A' in info['column_names']
        assert 'B' in info['column_names']

class TestDataPreprocessor:
    """Tests for DataPreprocessor class"""
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.target_column == 'Loan_Status'
        assert preprocessor.label_encoders == {}
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        # Create data with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['x', 'y', np.nan, 'z']
        })
        
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.handle_missing_values(df)
        
        # Check no missing values remain
        assert df_processed.isnull().sum().sum() == 0
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        df = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Male'],
            'Education': ['Graduate', 'Not Graduate', 'Graduate'],
            'Value': [1, 2, 3]
        })
        
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical_features(df, fit=True)
        
        # Check that categorical columns are encoded
        assert df_encoded['Gender'].dtype in [np.int32, np.int64]
        assert df_encoded['Education'].dtype in [np.int32, np.int64]
    
    def test_create_features(self):
        """Test feature creation"""
        df = pd.DataFrame({
            'ApplicantIncome': [5000, 6000, 7000],
            'CoapplicantIncome': [1500, 2000, 0],
            'LoanAmount': [150, 200, 250]
        })
        
        preprocessor = DataPreprocessor()
        df_features = preprocessor.create_features(df)
        
        # Check new features are created
        assert 'TotalIncome' in df_features.columns
        assert 'IncomeToLoanRatio' in df_features.columns
        assert 'LoanAmount_log' in df_features.columns
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Create sample data
        df = pd.DataFrame({
            'ApplicantIncome': [5000, 6000, 7000, 8000],
            'CoapplicantIncome': [1500, 2000, 0, 1000],
            'LoanAmount': [150, 200, 250, 180],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Education': ['Graduate', 'Graduate', 'Not Graduate', 'Graduate'],
            'Loan_Status': ['Y', 'N', 'Y', 'N']
        })
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        
        # Check shapes
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

def test_load_loan_data_function():
    """Test load_loan_data convenience function"""
    # This test would require actual data file
    # For now, we just test that the function exists
    assert callable(load_loan_data)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
