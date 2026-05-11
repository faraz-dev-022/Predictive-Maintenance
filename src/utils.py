import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    logger.warning(f"Model not found at {path}")
    return None
