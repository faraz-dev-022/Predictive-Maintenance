# test_inference.py
import numpy as np
import json
import os
from src.predict import predict_failure

def run_test():
    print("--- Testing Predictive Maintenance Inference ---")
    
    # 1. Load config to see what features we need
    config_path = 'models/config.json'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path) as f:
        config = json.load(f)
        
    seq_len = config.get('sequence_length', 50)
    features = config.get('sequence_cols', [])
    num_features = len(features)
    
    print(f"Model expects {seq_len} cycles with {num_features} features.")
    
    # 2. Create a dummy data window (random data for testing)
    # Shape should be (50, num_features)
    dummy_window = np.random.rand(seq_len, num_features)
    
    print("Running prediction...")
    try:
        result = predict_failure(dummy_window)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
        
        if 'error' in result:
            print("\nHint: Make sure lstm_model.h5 and scaler.pkl are in the models/ folder.")
        else:
            print("\nSuccess! The inference pipeline is working correctly.")
            
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")

if __name__ == "__main__":
    run_test()
