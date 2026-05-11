# src/explainability.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from tensorflow.keras.models import load_model

# Force CPU for SHAP to avoid CUDA conflicts in some environments
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def explain_prediction(model, background_data, test_window, feature_names, save_path=None):
    """
    Generate SHAP explanations for a specific prediction window.
    """
    # Initialize GradientExplainer (more stable for newer TF/LSTM models)
    explainer = shap.GradientExplainer(model, background_data)
    
    # Calculate SHAP values for the test window
    shap_values = explainer.shap_values(test_window)
    
    # SHAP values for a single output are in shap_values[0]
    # Shape is (1, 50, features)
    
    # We aggregate SHAP values over the 50 cycles to see overall feature impact
    # Mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values[0]).mean(axis=1) # (1, features)
    
    # Visualization: Summary Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, mean_shap[0])
    plt.xlabel("Mean Absolute SHAP Value (Impact on Risk)")
    plt.title("Feature Importance Breakdown (SHAP)")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"SHAP Summary Plot saved to {save_path}")
    
    return shap_values

def generate_shap_report():
    """
    Main entry point to generate a SHAP report using real model and dummy/raw data.
    """
    # 1. Load Model and Config
    MODELS_DIR = 'models'
    model = load_model(os.path.join(MODELS_DIR, 'lstm_model.h5'))
    with open(os.path.join(MODELS_DIR, 'config.json')) as f:
        import json
        config = json.load(f)
    
    feature_names = config['sequence_cols']
    seq_len = config['sequence_length']
    
    # 2. Prepare Background and Test Data
    # In a real scenario, we'd use data/processed/features_train.parquet
    # For now, we'll create a small synthetic sample to demonstrate
    num_features = len(feature_names)
    background = np.random.rand(20, seq_len, num_features).astype('float32')
    test_case = np.random.rand(1, seq_len, num_features).astype('float32')
    
    # 3. Explain
    print("Calculating SHAP values (this may take a minute)...")
    save_path = 'reports/shap_summary.png'
    explain_prediction(model, background, test_case, feature_names, save_path)

if __name__ == "__main__":
    generate_shap_report()
