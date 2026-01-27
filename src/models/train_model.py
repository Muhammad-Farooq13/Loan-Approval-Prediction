"""
Model Training Module
This module handles model training and hyperparameter tuning
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training machine learning models"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize ModelTrainer
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.cv_scores = None
    
    def get_model(self, model_type=None):
        """
        Get model instance based on type
        
        Args:
            model_type: Type of model
            
        Returns:
            model: Sklearn model instance
        """
        if model_type is None:
            model_type = self.model_type
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        
        return models.get(model_type, models['random_forest'])
    
    def get_param_grid(self, model_type=None):
        """
        Get parameter grid for hyperparameter tuning
        
        Args:
            model_type: Type of model
            
        Returns:
            dict: Parameter grid
        """
        if model_type is None:
            model_type = self.model_type
        
        param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return param_grids.get(model_type, param_grids['random_forest'])
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            
        Returns:
            tuple: (model, metrics)
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Get model
        self.model = self.get_model()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Cross-validation scores
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring='accuracy'
        )
        
        logger.info(f"Cross-validation accuracy: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        
        # Evaluate on test set if provided
        metrics = {}
        if X_test is not None and y_test is not None:
            metrics = self.evaluate(X_test, y_test)
        
        logger.info("Model training completed")
        
        return self.model, metrics
    
    def train_with_hyperparameter_tuning(
        self, X_train, y_train, X_test=None, y_test=None, 
        search_type='grid', cv=5, n_iter=20
    ):
        """
        Train model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            search_type: 'grid' or 'random'
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            
        Returns:
            tuple: (model, metrics, best_params)
        """
        logger.info(f"Training {self.model_type} with {search_type} search...")
        
        # Get base model and parameter grid
        base_model = self.get_model()
        param_grid = self.get_param_grid()
        
        # Perform hyperparameter search
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=cv, 
                scoring='accuracy', n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv, n_iter=n_iter,
                scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
            )
        
        # Fit search
        search.fit(X_train, y_train)
        
        # Get best model
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Evaluate on test set if provided
        metrics = {}
        if X_test is not None and y_test is not None:
            metrics = self.evaluate(X_test, y_test)
        
        return self.model, metrics, self.best_params
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def save_model(self, filepath='models/loan_model.pkl'):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/loan_model.pkl'):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return self.model
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """
        Compare multiple models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            pandas.DataFrame: Comparison results
        """
        model_types = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
        results = []
        
        for model_type in model_types:
            logger.info(f"Training {model_type}...")
            
            self.model_type = model_type
            model, metrics = self.train(X_train, y_train, X_test, y_test)
            
            results.append({
                'model': model_type,
                **metrics
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        logger.info("\nModel Comparison Results:")
        logger.info(results_df.to_string())
        
        return results_df

if __name__ == '__main__':
    # Test model training
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    trainer = ModelTrainer(model_type='random_forest')
    model, metrics = trainer.train(X_train, y_train, X_test, y_test)
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
