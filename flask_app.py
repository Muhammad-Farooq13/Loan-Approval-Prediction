"""
Flask API for Loan Prediction Model
This module provides a REST API for serving the loan prediction model.
"""

import os
import sys
import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logger
from src.models.predict import LoanPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logger('flask_app', 'logs/flask_app.log')

# Global variables for model
model = None
predictor = None

def load_model():
    """Load the trained model and predictor"""
    global model, predictor
    try:
        model_path = os.path.join('models', 'loan_model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            predictor = LoanPredictor(model)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # Create a dummy predictor for demonstration
            predictor = LoanPredictor(None)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        predictor = LoanPredictor(None)

# Load model on startup
load_model()

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Loan Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'model_info': '/model/info'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }), 200

@app.route('/model/info')
def model_info():
    """Get model information"""
    try:
        info = {
            'model_loaded': model is not None,
            'model_type': str(type(model).__name__) if model else None,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON format:
    {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'failed'
            }), 400
        
        # Validate required fields
        required_fields = [
            'Gender', 'Married', 'Dependents', 'Education', 
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
            'Property_Area'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'failed'
            }), 400
        
        # Make prediction
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'failed'
            }), 500
        
        # Convert data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Get prediction
        prediction, probability = predictor.predict(input_df)
        
        # Prepare response
        response = {
            'prediction': 'Approved' if prediction[0] == 1 else 'Rejected',
            'probability': float(probability[0]),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {response['prediction']} with probability {response['probability']:.4f}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "data": [
            {"Gender": "Male", "Married": "Yes", ...},
            {"Gender": "Female", "Married": "No", ...}
        ]
    }
    """
    try:
        # Get JSON data from request
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({
                'error': 'No data provided',
                'status': 'failed'
            }), 400
        
        data_list = request_data['data']
        
        if not isinstance(data_list, list) or len(data_list) == 0:
            return jsonify({
                'error': 'Data must be a non-empty list',
                'status': 'failed'
            }), 400
        
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'failed'
            }), 500
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data_list)
        
        # Make predictions
        predictions, probabilities = predictor.predict(input_df)
        
        # Prepare response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': i,
                'prediction': 'Approved' if pred == 1 else 'Rejected',
                'probability': float(prob)
            })
        
        response = {
            'predictions': results,
            'count': len(results),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction made for {len(results)} samples")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'failed'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'failed'
    }), 500

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
