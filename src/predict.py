# src/predict.py
import numpy as np
import json
import pickle
import os
import warnings
import logging

# Suppress TensorFlow and Scikit-learn warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_LEVEL'] = 'error'

warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set tensorflow logging to error only
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from tensorflow.keras.models import load_model

# Get absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Global variables for model, scaler and config
model = None
scaler = None
config = None
SEQ_LEN = 50
SEQ_COLS = []
W0 = 15

# Check if files exist before loading to prevent startup crash
model_path = os.path.join(MODELS_DIR, 'lstm_model.h5')
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
config_path = os.path.join(MODELS_DIR, 'config.json')

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(config_path):
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(config_path) as f:
            config = json.load(f)
        
        SEQ_LEN  = config.get('sequence_length', 50)
        SEQ_COLS = config.get('sequence_cols', [])
        W0       = config.get('w0', 15)
        print("Model and config loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
else:
    print(f"Warning: Model files not found in {MODELS_DIR}. Please ensure lstm_model.h5, scaler.pkl, and config.json are present.")

def predict_failure(sensor_window: np.ndarray) -> dict:
    """
    sensor_window: numpy array of shape (50, num_features)
                   — last 50 cycles of sensor readings
    Returns: dict with probability and risk label
    """
    if model is None or scaler is None:
        return {'error': 'Model or scaler not loaded'}
        
    try:
        # The sensor window needs to be scaled using the SAME scaler as training
        # If the input is (50, features), transform expects (n_samples, features)
        scaled = scaler.transform(sensor_window)          # normalise
        
        # Reshape for LSTM: (1, sequence_length, num_features)
        X = scaled.reshape(1, SEQ_LEN, len(SEQ_COLS))
        
        # Get prediction
        prob = float(model.predict(X, verbose=0)[0][0])  # failure probability
        risk = 'HIGH' if prob >= 0.5 else 'LOW'
        
        return {'failure_probability': round(prob, 4), 'risk': risk}
    except Exception as e:
        return {'error': str(e)}
