"""
Model Evaluation Module
This module handles model evaluation and performance analysis
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for evaluating trained models"""
    
    def __init__(self, model=None):
        """
        Initialize ModelEvaluator
        
        Args:
            model: Trained model instance
        """
        self.model = model
        self.evaluation_results = {}
    
    def load_model(self, model_path='models/loan_model.pkl'):
        """
        Load a trained model
        
        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def evaluate(self, X_test, y_test, save_results=True):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test target
            save_results: Whether to save results
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded for evaluation")
        
        logger.info("Starting model evaluation...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        self.evaluation_results = metrics
        
        # Log results
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        
        if save_results:
            self.save_results()
        
        return metrics
    
    def plot_confusion_matrix(self, X_test, y_test, save_path='reports/confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            X_test: Test features
            y_test: Test target
            save_path: Path to save the plot
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, X_test, y_test, save_path='reports/roc_curve.png'):
        """
        Plot ROC curve
        
        Args:
            X_test: Test features
            y_test: Test target
            save_path: Path to save the plot
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return
        
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
    
    def plot_feature_importance(self, feature_names=None, top_n=20, 
                                save_path='reports/feature_importance.png'):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not have feature importances")
            return
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Create dataframe
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort and get top N
        feature_imp_df = feature_imp_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {save_path}")
    
    def generate_evaluation_report(self, X_test, y_test):
        """
        Generate comprehensive evaluation report with all plots
        
        Args:
            X_test: Test features
            y_test: Test target
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Create reports directory
        os.makedirs('reports', exist_ok=True)
        
        # Evaluate
        self.evaluate(X_test, y_test)
        
        # Generate plots
        self.plot_confusion_matrix(X_test, y_test)
        self.plot_roc_curve(X_test, y_test)
        self.plot_feature_importance()
        
        logger.info("Evaluation report generated successfully")
    
    def save_results(self, filepath='reports/evaluation_results.txt'):
        """
        Save evaluation results to file
        
        Args:
            filepath: Path to save the results
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results to save")
            return
        
        # Create directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write results
        with open(filepath, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("="*50 + "\n\n")
            
            # Metrics
            f.write("Performance Metrics:\n")
            f.write("-"*50 + "\n")
            for metric, value in self.evaluation_results.items():
                if metric not in ['confusion_matrix', 'classification_report']:
                    f.write(f"{metric}: {value:.4f}\n")
            
            # Confusion Matrix
            f.write("\n\nConfusion Matrix:\n")
            f.write("-"*50 + "\n")
            cm = np.array(self.evaluation_results['confusion_matrix'])
            f.write(str(cm))
            
            # Classification Report
            f.write("\n\n\nClassification Report:\n")
            f.write("-"*50 + "\n")
            report = self.evaluation_results['classification_report']
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    f.write(f"\n{label}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info(f"Evaluation results saved to {filepath}")

if __name__ == '__main__':
    # Test evaluation
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test, save_results=False)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            print(f"{metric}: {value:.4f}")
