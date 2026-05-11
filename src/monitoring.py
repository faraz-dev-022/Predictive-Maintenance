import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import os

def generate_drift_report(reference_df, current_df, save_path='reports/drift_report.html'):
    """
    Generate an Evidently drift report.
    reference_df: The data used during training (baseline)
    current_df: The new data from production
    """
    # Create the report
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    # Run the report and get the snapshot
    snapshot = drift_report.run(reference_data=reference_df, current_data=current_df)
    
    # Save as HTML
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    snapshot.save_html(save_path)
    print(f"Drift report generated at {save_path}")

def run_monitoring_demo():
    # 1. Create a dummy baseline (Reference)
    cols = [f's{i}' for i in range(1, 22)]
    reference = pd.DataFrame(np.random.normal(0, 1, (100, len(cols))), columns=cols)
    reference['risk_label'] = np.random.randint(0, 2, 100)
    
    # 2. Create a dummy "drifting" production dataset (Current)
    # We simulate drift by changing the mean/std of some sensors
    current = pd.DataFrame(np.random.normal(0.5, 1.2, (100, len(cols))), columns=cols)
    current['risk_label'] = np.random.randint(0, 2, 100)
    
    generate_drift_report(reference, current)

if __name__ == "__main__":
    run_monitoring_demo()
