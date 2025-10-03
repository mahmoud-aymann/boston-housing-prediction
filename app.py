#!/usr/bin/env python3
"""
Boston Housing Price Prediction Flask Web Application
This Flask app provides a web interface for predicting Boston housing prices
using the trained machine learning model.
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Get script directory (where app.py exists)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "boston_housing_model.pkl")
info_path = os.path.join(script_dir, "model_info.pkl")

# Load the trained model and model info
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    with open(info_path, 'rb') as file:
        model_info = pickle.load(file)
    
    print("Model loaded successfully!")
    print(f"Model type: {model_info['model_type']}")
    print(f"Optimal depth: {model_info['optimal_depth']}")
    print(f"R² Score: {model_info['r2_score']:.4f}")
    print(f"Features: {model_info['feature_names']}")
    
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to create the model files.")
    model = None
    model_info = None

@app.route('/')
def home():
    """Render the main page with the prediction form."""
    return render_template('index.html', 
                         feature_names=model_info['feature_names'] if model_info else [],
                         model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None or model_info is None:
            return jsonify({'error': 'Model not loaded. Please run train_model.py first.'}), 500
        
        # Get input data from the request
        data = request.get_json()
        
        # Extract features
        rm = float(data.get('rm', 0))
        lstat = float(data.get('lstat', 0))
        ptratio = float(data.get('ptratio', 0))
        
        # Validate input ranges
        if not (3 <= rm <= 9):
            return jsonify({'error': 'Number of rooms (RM) should be between 3 and 9'}), 400
        
        if not (1 <= lstat <= 40):
            return jsonify({'error': 'Poverty level (LSTAT) should be between 1% and 40%'}), 400
        
        if not (10 <= ptratio <= 25):
            return jsonify({'error': 'Student-teacher ratio (PTRATIO) should be between 10 and 25'}), 400
        
        # Prepare input for prediction
        input_data = np.array([[rm, lstat, ptratio]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Format the response
        response = {
            'prediction': round(prediction, 2),
            'formatted_prediction': f"${prediction:,.2f}",
            'input_data': {
                'rm': rm,
                'lstat': lstat,
                'ptratio': ptratio
            },
            'model_info': {
                'r2_score': model_info['r2_score'],
                'optimal_depth': model_info['optimal_depth']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/info')
def api_info():
    """Return model information."""
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': model_info['model_type'],
        'feature_names': model_info['feature_names'],
        'target_name': model_info['target_name'],
        'optimal_depth': model_info['optimal_depth'],
        'r2_score': model_info['r2_score']
    })

@app.route('/api/sample_predictions')
def sample_predictions():
    """Return sample predictions for demonstration."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Sample client data
    sample_data = [
        {'name': 'Client 1', 'rm': 5, 'lstat': 17, 'ptratio': 15, 'description': 'Moderate house, moderate neighborhood'},
        {'name': 'Client 2', 'rm': 4, 'lstat': 32, 'ptratio': 22, 'description': 'Small house, high poverty area'},
        {'name': 'Client 3', 'rm': 8, 'lstat': 3, 'ptratio': 12, 'description': 'Large house, affluent neighborhood'}
    ]
    
    predictions = []
    for client in sample_data:
        input_data = np.array([[client['rm'], client['lstat'], client['ptratio']]])
        prediction = model.predict(input_data)[0]
        
        predictions.append({
            'name': client['name'],
            'rm': client['rm'],
            'lstat': client['lstat'],
            'ptratio': client['ptratio'],
            'description': client['description'],
            'predicted_price': round(prediction, 2),
            'formatted_price': f"${prediction:,.2f}"
        })
    
    return jsonify(predictions)

if __name__ == '__main__':
    print("Starting Boston Housing Price Prediction App...")
    print("Visit http://localhost:5000 to use the application")
    app.run(debug=True, host='0.0.0.0', port=5000)
