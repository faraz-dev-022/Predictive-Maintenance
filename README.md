# Predictive Maintenance Project

This project focuses on predicting equipment failure using machine learning.

## Project Structure

```
predictive_maintenance/
│
├── notebooks/
│   ├── 01_eda.ipynb               ← exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_experiment_tracking.ipynb
│   ├── 05_drift_monitoring.ipynb
│   └── 06_demo.ipynb              ← interview walkthrough notebook
│
├── src/
│   ├── __init__.py
│   ├── features.py                ← all feature engineering functions
│   ├── train.py                   ← model training logic
│   ├── predict.py                 ← inference/prediction logic
│   └── utils.py                   ← helper functions (logging, loading, etc.)
│
├── api/
│   ├── main.py                    ← FastAPI app
│   ├── schema.py                  ← Pydantic input/output models
│   └── Dockerfile                 ← containerise the API
│
├── data/
│   ├── raw/                       ← original downloaded dataset
│   └── processed/                 ← cleaned + feature-engineered data (.parquet)
│
├── models/                        ← saved model files (.pkl, .json)
│
├── reports/
│   └── drift_report.html          ← Evidently output
│
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Copy `.env` if needed and update the values.

3. Run the API:
   ```bash
   uvicorn api.main:app --reload
   ```
