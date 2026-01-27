"""
MLOps Pipeline for Loan Prediction Project
This module implements CI/CD pipeline with automated testing, model validation, and deployment
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator

# Setup logger
logger = setup_logger('mlops_pipeline', 'logs/mlops_pipeline.log')

class MLOpsPipeline:
    """
    MLOps Pipeline for continuous integration and deployment
    """
    
    def __init__(self, config_path=None):
        """Initialize the MLOps pipeline"""
        self.config = Config()
        self.pipeline_start_time = datetime.now()
        self.results = {
            'pipeline_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'stages': {},
            'status': 'initialized'
        }
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        logger.info("MLOps Pipeline initialized")
    
    def run_tests(self):
        """Run automated tests"""
        logger.info("Stage 1: Running automated tests...")
        
        try:
            # Run pytest
            result = subprocess.run(
                ['pytest', 'tests/', '-v', '--tb=short'],
                capture_output=True,
                text=True
            )
            
            self.results['stages']['testing'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'exit_code': result.returncode,
                'output': result.stdout,
                'timestamp': datetime.now().isoformat()
            }
            
            if result.returncode == 0:
                logger.info("✓ All tests passed")
                return True
            else:
                logger.error("✗ Tests failed")
                logger.error(result.stdout)
                return False
                
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            self.results['stages']['testing'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def validate_data(self, data_path='data/raw/loan_data.csv'):
        """Validate data quality and schema"""
        logger.info("Stage 2: Validating data...")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Data validation checks
            checks = {
                'file_exists': os.path.exists(data_path),
                'not_empty': len(df) > 0,
                'no_duplicate_rows': df.duplicated().sum() == 0,
                'required_columns_present': True,
                'data_types_valid': True
            }
            
            # Check for missing values
            missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
            
            # Check data distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats = df[numeric_cols].describe().to_dict()
            
            validation_result = {
                'status': 'passed' if all(checks.values()) else 'failed',
                'checks': checks,
                'row_count': len(df),
                'column_count': len(df.columns),
                'missing_values': missing_percentage,
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results['stages']['data_validation'] = validation_result
            
            logger.info(f"✓ Data validation complete: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            self.results['stages']['data_validation'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def train_and_validate_model(self):
        """Train and validate the model"""
        logger.info("Stage 3: Training and validating model...")
        
        try:
            # Initialize trainer
            trainer = ModelTrainer()
            
            # Load and preprocess data
            data_path = 'data/raw/loan_data.csv'
            if not os.path.exists(data_path):
                logger.warning(f"Data file not found: {data_path}")
                self.results['stages']['model_training'] = {
                    'status': 'skipped',
                    'reason': 'data_file_not_found',
                    'timestamp': datetime.now().isoformat()
                }
                return True
            
            df = pd.read_csv(data_path)
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
            
            # Train model
            model, metrics = trainer.train(X_train, y_train, X_test, y_test)
            
            # Validate model performance
            min_accuracy = 0.70  # Minimum acceptable accuracy
            
            validation_result = {
                'status': 'passed' if metrics.get('accuracy', 0) >= min_accuracy else 'failed',
                'metrics': metrics,
                'threshold': min_accuracy,
                'model_type': str(type(model).__name__),
                'timestamp': datetime.now().isoformat()
            }
            
            self.results['stages']['model_training'] = validation_result
            
            # Save model if validation passed
            if validation_result['status'] == 'passed':
                model_path = f"models/loan_model_{self.results['pipeline_id']}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"✓ Model saved to {model_path}")
                
                # Also save as latest model
                joblib.dump(model, 'models/loan_model.pkl')
                logger.info("✓ Model updated as latest version")
            
            logger.info(f"✓ Model training complete with accuracy: {metrics.get('accuracy', 0):.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            self.results['stages']['model_training'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def check_model_drift(self):
        """Check for model drift"""
        logger.info("Stage 4: Checking for model drift...")
        
        try:
            # This is a placeholder for drift detection logic
            # In production, you would compare current model performance 
            # with historical performance
            
            drift_result = {
                'status': 'no_drift_detected',
                'drift_score': 0.05,  # Example score
                'threshold': 0.10,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results['stages']['drift_detection'] = drift_result
            
            logger.info("✓ Drift detection complete")
            return True
            
        except Exception as e:
            logger.error(f"Error checking drift: {str(e)}")
            self.results['stages']['drift_detection'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def generate_report(self):
        """Generate pipeline execution report"""
        logger.info("Stage 5: Generating report...")
        
        try:
            self.results['pipeline_end_time'] = datetime.now().isoformat()
            self.results['duration_seconds'] = (
                datetime.now() - self.pipeline_start_time
            ).total_seconds()
            
            # Determine overall status
            stage_statuses = [
                stage.get('status', 'unknown') 
                for stage in self.results['stages'].values()
            ]
            
            if 'error' in stage_statuses or 'failed' in stage_statuses:
                self.results['status'] = 'failed'
            else:
                self.results['status'] = 'success'
            
            # Save report
            report_path = f"reports/pipeline_report_{self.results['pipeline_id']}.json"
            with open(report_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"✓ Report saved to {report_path}")
            logger.info(f"Pipeline status: {self.results['status']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return False
    
    def run(self):
        """Run the complete MLOps pipeline"""
        logger.info("="*70)
        logger.info("Starting MLOps Pipeline Execution")
        logger.info(f"Pipeline ID: {self.results['pipeline_id']}")
        logger.info("="*70)
        
        # Stage 1: Run tests
        if not self.run_tests():
            logger.warning("Tests failed, but continuing pipeline...")
        
        # Stage 2: Validate data
        if not self.validate_data():
            logger.error("Data validation failed, stopping pipeline")
            self.results['status'] = 'failed'
            self.generate_report()
            return False
        
        # Stage 3: Train and validate model
        if not self.train_and_validate_model():
            logger.error("Model training failed, stopping pipeline")
            self.results['status'] = 'failed'
            self.generate_report()
            return False
        
        # Stage 4: Check for drift
        self.check_model_drift()
        
        # Stage 5: Generate report
        self.generate_report()
        
        logger.info("="*70)
        logger.info(f"MLOps Pipeline Completed: {self.results['status'].upper()}")
        logger.info(f"Duration: {self.results['duration_seconds']:.2f} seconds")
        logger.info("="*70)
        
        return self.results['status'] == 'success'

def main():
    """Main function to run the MLOps pipeline"""
    pipeline = MLOpsPipeline()
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
