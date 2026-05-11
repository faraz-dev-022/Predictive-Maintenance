# Predictive Maintenance System

This repository contains a lightweight predictive maintenance service aligned with a production-style architecture:

- **FastAPI** inference service in `/api`
- **Feature engineering** helpers in `/src/features.py`
- **Prediction logic** with optional TensorFlow/Keras model loading in `/src/prediction.py`
- **Explainability and report artifacts** in `/src/explainability.py`
- **Focused API tests** in `/tests`

## Local setup

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
uvicorn api.main:app --reload
```

## API

- `POST /predict` validates a `50 x 25` sensor window and returns:
  - failure probability
  - predicted remaining useful life (RUL)
  - risk level
  - top feature contributors
- `GET /reports` lists available diagnostic artifacts
- `GET /reports/{name}` serves HTML or PNG reports
