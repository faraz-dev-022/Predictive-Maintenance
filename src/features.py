import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def get_good_sensors(df, sensor_cols, threshold=0.01):
    """
    Drop sensors with low variance.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[sensor_cols])
    good_sensors = [s for s, v in zip(sensor_cols, selector.get_support()) if v]
    return good_sensors

def fft_feature(series, n=5):
    """
    Calculate mean of top n frequency components using FFT.
    """
    if len(series) < n + 1:
        return 0
    fft_vals = np.abs(np.fft.fft(series.values))
    return fft_vals[1:n+1].mean()

def engineer_features(df):
    """
    Comprehensive feature engineering based on tutorial images.
    """
    sensor_cols = [f's{i}' for i in range(1, 22)]
    
    # 1. Drop low-variance sensors
    good_sensors = get_good_sensors(df, sensor_cols)
    
    # 2. Rolling statistics
    windows = [5, 10, 20]
    for w in windows:
        for sensor in good_sensors:
            df[f'{sensor}_roll_mean_{w}'] = df.groupby('unit')[sensor].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f'{sensor}_roll_std_{w}'] = df.groupby('unit')[sensor].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
            
    # 3. Lag features
    for sensor in good_sensors:
        df[f'{sensor}_lag1'] = df.groupby('unit')[sensor].shift(1).fillna(0)
        df[f'{sensor}_lag3'] = df.groupby('unit')[sensor].shift(3).fillna(0)
        
    # 4. FFT features for vibration-like sensors
    for sensor in ['s2', 's3', 's4']:
        if sensor in df.columns:
            df[f'{sensor}_fft'] = df.groupby('unit')[sensor].transform(lambda x: fft_feature(x))
            
    return df, good_sensors

def scale_features(train_df, test_df, feature_cols):
    """
    Apply Standard Scaling to the feature set.
    """
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    if test_df is not None:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, scaler
