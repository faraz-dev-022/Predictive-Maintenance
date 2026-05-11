import xgboost as xgb
import joblib
import os
import logging
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import mlflow
import mlflow.xgboost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_xgb_model(X_train, y_train, X_val, y_val, params=None, use_gpu=True, run_name="XGBoost_Baseline"):
    """
    Train an XGBoost Regressor with GPU acceleration and MLflow tracking.
    """
    if params is None:
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    
    if use_gpu:
        logger.info("Enabling GPU acceleration for XGBoost...")
        try:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        except:
            params['tree_method'] = 'gpu_hist'
            
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        
        logger.info(f"Starting model training: {run_name}...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False # Keep MLflow logs clean
        )
        
        # Evaluate
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model")
        
        logger.info(f"Run {run_name} completed. RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
    return model, {"rmse": rmse, "r2": r2}

def save_model_local(model, path="models/xgboost_model.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    logger.info(f"Model saved locally to {path}")
