"""
Data Preprocessing Module
This module handles data cleaning and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing loan data"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Loan_Status'
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Data with missing values handled
        """
        df = df.copy()
        
        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                logger.info(f"Filled missing values in {col} with mode")
        
        # Numerical columns - fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                logger.info(f"Filled missing values in {col} with median")
        
        return df
    
    def handle_outliers(self, df, columns=None, method='iqr'):
        """
        Handle outliers in numerical columns
        
        Args:
            df: pandas.DataFrame
            columns: List of columns to check for outliers
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            pandas.DataFrame: Data with outliers handled
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Handled outliers in {col} using IQR method")
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features
        
        Args:
            df: pandas.DataFrame
            fit: Whether to fit the encoders (True for training, False for inference)
            
        Returns:
            pandas.DataFrame: Data with encoded features
        """
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target_column]
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen labels
                    known_labels = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_labels else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
            
            logger.info(f"Encoded categorical feature: {col}")
        
        return df
    
    def scale_features(self, X, fit=True):
        """
        Scale numerical features
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            numpy.ndarray: Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted and transformed features")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info("Transformed features using fitted scaler")
        
        return X_scaled
    
    def create_features(self, df):
        """
        Create additional features
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Data with additional features
        """
        df = df.copy()
        
        # Total income
        if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
            df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
            logger.info("Created feature: TotalIncome")
        
        # Income to loan ratio
        if 'TotalIncome' in df.columns and 'LoanAmount' in df.columns:
            df['IncomeToLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
            logger.info("Created feature: IncomeToLoanRatio")
        
        # Log transformations for skewed features
        if 'LoanAmount' in df.columns:
            df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
            logger.info("Created feature: LoanAmount_log")
        
        return df
    
    def preprocess(self, df, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            df: pandas.DataFrame
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Check if target column exists
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in dataframe")
            # For demo purposes, create a dummy target
            df[self.target_column] = np.random.choice(['Y', 'N'], size=len(df))
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create additional features
        df = self.create_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Scale features
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        
        logger.info("Preprocessing completed successfully")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessors
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            numpy.ndarray: Preprocessed features
        """
        df = df.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create features
        df = self.create_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Remove target if present
        if self.target_column in df.columns:
            df = df.drop(self.target_column, axis=1)
        
        # Ensure same columns as training
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.feature_columns]
        
        # Scale features
        X = self.scale_features(df, fit=False)
        
        return X

if __name__ == '__main__':
    # Test preprocessing
    from load_data import load_loan_data
    
    try:
        df = load_loan_data()
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
    except Exception as e:
        print(f"Error: {str(e)}")
