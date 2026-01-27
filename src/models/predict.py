"""
Prediction Module
This module handles making predictions with trained models
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class LoanPredictor:
    """Class for making loan predictions"""
    
    def __init__(self, model=None, preprocessor=None):
        """
        Initialize LoanPredictor
        
        Args:
            model: Trained model
            preprocessor: Data preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor
    
    def load_model(self, model_path='models/loan_model.pkl'):
        """
        Load trained model
        
        Args:
            model_path: Path to saved model
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def load_preprocessor(self, preprocessor_path='models/preprocessor.pkl'):
        """
        Load preprocessor
        
        Args:
            preprocessor_path: Path to saved preprocessor
        """
        if not os.path.exists(preprocessor_path):
            logger.warning(f"Preprocessor not found: {preprocessor_path}")
            return
        
        self.preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
    
    def preprocess_input(self, data):
        """
        Preprocess input data for prediction
        
        Args:
            data: Input data (dict or DataFrame)
            
        Returns:
            numpy.ndarray: Preprocessed features
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Use preprocessor if available
        if self.preprocessor is not None:
            X = self.preprocessor.transform(data)
        else:
            # Basic preprocessing
            X = self._basic_preprocess(data)
        
        return X
    
    def _basic_preprocess(self, df):
        """
        Basic preprocessing without fitted preprocessor
        
        Args:
            df: Input DataFrame
            
        Returns:
            numpy.ndarray: Preprocessed features
        """
        df = df.copy()
        
        # Handle categorical columns (simple label encoding)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                # Simple mapping for common values
                if col == 'Gender':
                    df[col] = df[col].map({'Male': 1, 'Female': 0})
                elif col == 'Married':
                    df[col] = df[col].map({'Yes': 1, 'No': 0})
                elif col == 'Education':
                    df[col] = df[col].map({'Graduate': 1, 'Not Graduate': 0})
                elif col == 'Self_Employed':
                    df[col] = df[col].map({'Yes': 1, 'No': 0})
                elif col == 'Property_Area':
                    df[col] = df[col].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
                elif col == 'Dependents':
                    df[col] = df[col].replace('3+', '3').astype(float)
        
        # Fill missing values
        df = df.fillna(df.median(numeric_only=True))
        
        return df.values
    
    def predict(self, data):
        """
        Make predictions
        
        Args:
            data: Input data (dict, DataFrame, or array)
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            logger.error("No model loaded")
            raise ValueError("No model loaded for prediction")
        
        # Preprocess input
        if isinstance(data, (dict, pd.DataFrame)):
            X = self.preprocess_input(data)
        else:
            X = data
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        
        logger.info(f"Made predictions for {len(predictions)} samples")
        
        return predictions, probabilities
    
    def predict_single(self, data):
        """
        Make prediction for a single sample
        
        Args:
            data: Input data (dict or single row DataFrame)
            
        Returns:
            tuple: (prediction, probability)
        """
        predictions, probabilities = self.predict(data)
        
        prediction = predictions[0]
        probability = probabilities[0] if probabilities is not None else None
        
        return prediction, probability
    
    def predict_with_explanation(self, data):
        """
        Make prediction with explanation
        
        Args:
            data: Input data
            
        Returns:
            dict: Prediction results with explanation
        """
        prediction, probability = self.predict_single(data)
        
        result = {
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'prediction_code': int(prediction),
            'confidence': float(probability) if probability is not None else None,
            'risk_level': self._get_risk_level(probability) if probability is not None else 'Unknown'
        }
        
        return result
    
    def _get_risk_level(self, probability):
        """
        Determine risk level based on probability
        
        Args:
            probability: Prediction probability
            
        Returns:
            str: Risk level
        """
        if probability >= 0.8:
            return 'Low Risk'
        elif probability >= 0.6:
            return 'Medium Risk'
        elif probability >= 0.4:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def batch_predict(self, data_list):
        """
        Make predictions for multiple samples
        
        Args:
            data_list: List of input data
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for data in data_list:
            try:
                result = self.predict_with_explanation(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting sample: {str(e)}")
                results.append({
                    'prediction': 'Error',
                    'error': str(e)
                })
        
        return results

def predict_loan_approval(data, model_path='models/loan_model.pkl'):
    """
    Convenience function for loan prediction
    
    Args:
        data: Input data
        model_path: Path to trained model
        
    Returns:
        dict: Prediction result
    """
    predictor = LoanPredictor()
    predictor.load_model(model_path)
    
    result = predictor.predict_with_explanation(data)
    
    return result

if __name__ == '__main__':
    # Test prediction
    sample_data = {
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
    
    # Create a dummy model for testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=11, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    predictor = LoanPredictor(model=model)
    
    try:
        result = predictor.predict_with_explanation(sample_data)
        print("Prediction Result:")
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
