"""
Unit tests for model training, evaluation, and prediction
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.models.predict import LoanPredictor

class TestModelTrainer:
    """Tests for ModelTrainer class"""
    
    def setup_method(self):
        """Setup test data"""
        X, y = make_classification(
            n_samples=100, 
            n_features=10, 
            n_informative=5,
            random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    def test_trainer_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(model_type='random_forest')
        assert trainer.model_type == 'random_forest'
        assert trainer.model is None
    
    def test_get_model(self):
        """Test get_model method"""
        trainer = ModelTrainer()
        
        # Test different model types
        model_rf = trainer.get_model('random_forest')
        assert model_rf is not None
        
        model_lr = trainer.get_model('logistic_regression')
        assert model_lr is not None
    
    def test_train_model(self):
        """Test model training"""
        trainer = ModelTrainer(model_type='random_forest')
        model, metrics = trainer.train(
            self.X_train, 
            self.y_train, 
            self.X_test, 
            self.y_test
        )
        
        # Check that model was trained
        assert model is not None
        assert trainer.model is not None
        
        # Check that metrics were calculated
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check metric values are reasonable
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        trainer = ModelTrainer(model_type='random_forest')
        trainer.train(self.X_train, self.y_train)
        
        metrics = trainer.evaluate(self.X_test, self.y_test)
        
        # Check all metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_get_param_grid(self):
        """Test parameter grid retrieval"""
        trainer = ModelTrainer()
        
        # Test different model types
        param_grid_rf = trainer.get_param_grid('random_forest')
        assert 'n_estimators' in param_grid_rf
        assert 'max_depth' in param_grid_rf
        
        param_grid_lr = trainer.get_param_grid('logistic_regression')
        assert 'C' in param_grid_lr

class TestModelEvaluator:
    """Tests for ModelEvaluator class"""
    
    def setup_method(self):
        """Setup test data and model"""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(
            n_samples=100, 
            n_features=10, 
            random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator(self.model)
        assert evaluator.model is not None
        assert evaluator.evaluation_results == {}
    
    def test_evaluate(self):
        """Test model evaluation"""
        evaluator = ModelEvaluator(self.model)
        metrics = evaluator.evaluate(self.X_test, self.y_test, save_results=False)
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric values
        assert 0 <= metrics['accuracy'] <= 1
        assert isinstance(metrics['confusion_matrix'], list)

class TestLoanPredictor:
    """Tests for LoanPredictor class"""
    
    def setup_method(self):
        """Setup test data and model"""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(
            n_samples=100, 
            n_features=11, 
            random_state=42
        )
        
        # Train a model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X, y)
        
        # Sample input data
        self.sample_data = {
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': '0',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': 5000,
            'CoapplicantIncome': 1500,
            'LoanAmount': 150,
            'Loan_Amount_Term': 360,
            'Credit_History': 1.0,
            'Property_Area': 'Urban'
        }
    
    def test_predictor_initialization(self):
        """Test LoanPredictor initialization"""
        predictor = LoanPredictor(self.model)
        assert predictor.model is not None
    
    def test_basic_preprocess(self):
        """Test basic preprocessing"""
        predictor = LoanPredictor(self.model)
        df = pd.DataFrame([self.sample_data])
        
        X = predictor._basic_preprocess(df)
        
        # Check output shape
        assert X.shape[0] == 1
        assert X.shape[1] > 0
    
    def test_predict(self):
        """Test prediction"""
        predictor = LoanPredictor(self.model)
        
        predictions, probabilities = predictor.predict(self.sample_data)
        
        # Check outputs
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
        
        if probabilities is not None:
            assert len(probabilities) == 1
            assert 0 <= probabilities[0] <= 1
    
    def test_predict_with_explanation(self):
        """Test prediction with explanation"""
        predictor = LoanPredictor(self.model)
        
        result = predictor.predict_with_explanation(self.sample_data)
        
        # Check result structure
        assert 'prediction' in result
        assert 'prediction_code' in result
        assert result['prediction'] in ['Approved', 'Rejected']
        assert result['prediction_code'] in [0, 1]

def test_model_training_integration():
    """Integration test for model training pipeline"""
    # Create sample data
    X, y = make_classification(
        n_samples=100, 
        n_features=10, 
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    trainer = ModelTrainer(model_type='random_forest')
    model, metrics = trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluator = ModelEvaluator(model)
    eval_metrics = evaluator.evaluate(X_test, y_test, save_results=False)
    
    # Make predictions
    predictor = LoanPredictor(model)
    predictions, probabilities = predictor.predict(X_test)
    
    # Check pipeline worked
    assert model is not None
    assert len(predictions) == len(X_test)
    assert 'accuracy' in eval_metrics

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
