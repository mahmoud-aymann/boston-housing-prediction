#!/usr/bin/env python3
"""
Boston Housing Price Prediction Model Training Script
This script loads the housing data, trains a decision tree regressor,
and saves the trained model as a pickle file for Flask deployment.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def performance_metric(y_test, y_predict):
    """Calculates and returns the performance score between 
    test and predicted values based on the metric chosen."""
    score = r2_score(y_test, y_predict)
    return score

def create_normalized_pipeline():
    """Create a pipeline with PowerTransformer normalization and decision tree regressor."""
    pipeline = Pipeline([
        ('normalizer', PowerTransformer(method='yeo-johnson')),
        ('regressor', DecisionTreeRegressor())
    ])
    return pipeline

def fit_model(X, y):
    """Performs grid search over the 'max_depth' parameter for a 
    decision tree regressor with normalization trained on the input data [X, y]."""
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # Create a pipeline with normalization and decision tree regressor
    pipeline = create_normalized_pipeline()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'regressor__max_depth': list(range(1, 11))}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(pipeline, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def main():
    print("Loading Boston Housing dataset...")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "housing.csv")

    # Load the Boston housing dataset
    data = pd.read_csv(csv_path)
    
    # Separate features and target
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)
    
    print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"Features: {list(features.columns)}")
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print("="*50)
    print(f"Minimum price: ${prices.min():,.2f}")
    print(f"Maximum price: ${prices.max():,.2f}")
    print(f"Mean price: ${prices.mean():,.2f}")
    print(f"Median price: ${prices.median():,.2f}")
    print(f"Standard deviation: ${prices.std():,.2f}")
    
    # Split the data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, prices, test_size=0.2, random_state=1
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train the model using grid search with normalization
    print("\nTraining model with normalization and grid search...")
    print("Using PowerTransformer to normalize data for better distribution...")
    print("Searching for optimal max_depth parameter...")
    
    reg = fit_model(X_train, y_train)
    
    # Get the optimal parameters
    optimal_depth = reg.get_params()['regressor__max_depth']
    print(f"Optimal max_depth: {optimal_depth}")
    print("Data has been normalized using PowerTransformer")
    
    # Make predictions on test set
    y_pred = reg.predict(X_test)
    
    # Calculate performance metrics
    r2 = performance_metric(y_test, y_pred)
    print(f"R² Score on test set: {r2:.4f}")
    
    # Test predictions on sample client data
    print("\nTesting predictions on sample client data:")
    print("="*50)
    
    client_data = [
        [5, 17, 15],  # Client 1: 5 rooms, 17% poverty, 15:1 ratio
        [4, 32, 22],  # Client 2: 4 rooms, 32% poverty, 22:1 ratio  
        [8, 3, 12]    # Client 3: 8 rooms, 3% poverty, 12:1 ratio
    ]
    
    for i, price in enumerate(reg.predict(client_data)):
        print(f"Client {i+1} predicted price: ${price:,.2f}")
    
    # Save the trained model
    print("\nSaving trained model...")
    model_filename = os.path.join(script_dir, 'boston_housing_model.pkl')
    
    with open(model_filename, 'wb') as file:
        pickle.dump(reg, file)
    
    print(f"Model saved as: {model_filename}")
    
    # Save feature names for Flask app
    feature_names = list(features.columns)
    feature_info = {
        'feature_names': feature_names,
        'target_name': 'MEDV',
        'model_type': 'DecisionTreeRegressor_with_PowerTransformer',
        'optimal_depth': optimal_depth,
        'r2_score': r2,
        'normalization': 'PowerTransformer (yeo-johnson)'
    }
    
    info_filename = os.path.join(script_dir, 'model_info.pkl')
    with open(info_filename, 'wb') as file:
        pickle.dump(feature_info, file)
    
    print(f"Model info saved as: {info_filename}")
    print("\nModel training completed successfully!")
    
    return reg, feature_info

if __name__ == "__main__":
    model, info = main()
