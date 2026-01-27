"""
Configuration Module
This module contains configuration settings for the project
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the loan prediction project"""
    
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    REPORTS_DIR = BASE_DIR / 'reports'
    
    # Data files
    RAW_DATA_FILE = 'loan_data.csv'
    PROCESSED_DATA_FILE = 'processed_loan_data.csv'
    
    # Model settings
    MODEL_NAME = 'loan_model.pkl'
    PREPROCESSOR_NAME = 'preprocessor.pkl'
    
    # Training settings
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Model hyperparameters
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE
    }
    
    LOGISTIC_REGRESSION_PARAMS = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
    
    GRADIENT_BOOSTING_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': RANDOM_STATE
    }
    
    # Feature engineering settings
    NUMERICAL_FEATURES = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History'
    ]
    
    CATEGORICAL_FEATURES = [
        'Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'Property_Area'
    ]
    
    TARGET_COLUMN = 'Loan_Status'
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    API_DEBUG = False
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Preprocessing settings
    MISSING_VALUE_STRATEGY = {
        'numerical': 'median',
        'categorical': 'mode'
    }
    
    OUTLIER_DETECTION_METHOD = 'iqr'
    OUTLIER_THRESHOLD = 1.5
    
    # Feature scaling
    SCALING_METHOD = 'standard'  # 'standard', 'minmax', or 'robust'
    
    # Model evaluation thresholds
    MIN_ACCURACY = 0.70
    MIN_PRECISION = 0.65
    MIN_RECALL = 0.65
    MIN_F1_SCORE = 0.65
    
    # MLOps settings
    MODEL_REGISTRY = 'mlflow'  # 'mlflow', 'dvc', or 'local'
    EXPERIMENT_NAME = 'loan_prediction'
    
    # Drift detection
    DRIFT_THRESHOLD = 0.10
    DRIFT_CHECK_FREQUENCY = 'daily'  # 'hourly', 'daily', 'weekly'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name=None):
        """Get full path to model file"""
        if model_name is None:
            model_name = cls.MODEL_NAME
        return cls.MODELS_DIR / model_name
    
    @classmethod
    def get_data_path(cls, filename=None, processed=False):
        """Get full path to data file"""
        if filename is None:
            filename = cls.PROCESSED_DATA_FILE if processed else cls.RAW_DATA_FILE
        
        data_dir = cls.PROCESSED_DATA_DIR if processed else cls.RAW_DATA_DIR
        return data_dir / filename
    
    @classmethod
    def get_log_path(cls, log_name='app.log'):
        """Get full path to log file"""
        return cls.LOGS_DIR / log_name
    
    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary"""
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(cls, attr)
                if not callable(value):
                    config_dict[attr] = str(value) if isinstance(value, Path) else value
        return config_dict

# Create directories on import
Config.create_directories()

if __name__ == '__main__':
    # Print configuration
    print("Loan Prediction Project Configuration")
    print("=" * 50)
    
    config_dict = Config.to_dict()
    for key, value in config_dict.items():
        print(f"{key}: {value}")
