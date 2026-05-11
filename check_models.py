import pickle, json
import numpy as np
from tensorflow.keras.models import load_model
import os

# Suppress TF logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Checking models...")

model = load_model("models/lstm_model.h5")
print("OK - LSTM loaded - input shape:", model.input_shape)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("OK - Scaler loaded - features:", scaler.n_features_in_)

with open("models/config.json") as f:
    config = json.load(f)
print("OK - Config loaded:")
print(f"   sequence_length : {config['sequence_length']}")
print(f"   n_features      : {len(config['sequence_cols'])}")
print(f"   accuracy        : {config['accuracy']:.2%}")
print(f"   f1 score        : {config['f1']:.2%}")

# Quick inference test
seq_len   = config["sequence_length"]
n_feat    = len(config["sequence_cols"])
dummy     = np.zeros((1, seq_len, n_feat))
pred      = model.predict(dummy, verbose=0)
print(f"\nOK - Test inference passed - output: {pred[0][0]:.4f}")
print("\nAll checks passed! Project is ready.")
