import math
from pathlib import Path
from typing import Iterable

import numpy as np

from src.explainability import summarize_feature_contributions
from src.features import build_feature_frame
from src.utils import clamp_probability, validate_window_shape


class PredictiveMaintenanceModel:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self._keras_model = self._load_keras_model()

    def _load_keras_model(self):
        model_path = self.base_dir / "models" / "lstm_rul.keras"
        if not model_path.exists():
            return None
        try:
            from tensorflow import keras
        except Exception:
            return None
        return keras.models.load_model(model_path)

    def _fallback_predict(self, rows: np.ndarray) -> tuple[float, int]:
        frame = build_feature_frame(rows.tolist())
        sensor_columns = [column for column in frame.columns if column.startswith("sensor_") and "_rolling_" not in column]
        latest_sensor_values = frame.iloc[-1][sensor_columns].to_numpy(dtype=float)
        prior_sensor_values = frame.iloc[-5][sensor_columns].to_numpy(dtype=float)
        trend = np.maximum(latest_sensor_values - prior_sensor_values, 0.0)

        latest_cycle = float(frame.iloc[-1]["normalized_cycle"])
        mean_deviation = float(np.abs(latest_sensor_values - frame[sensor_columns].mean().to_numpy()).mean())
        trend_score = float(trend.mean())
        raw_score = mean_deviation * 0.85 + trend_score * 0.65 + latest_cycle * 1.5

        probability = clamp_probability(1.0 / (1.0 + math.exp(-(raw_score - 1.0))))
        predicted_rul = max(1, int(round(150 * (1.0 - probability))))
        return probability, predicted_rul

    def predict(self, window: Iterable[Iterable[float]]) -> dict[str, object]:
        rows = np.asarray(validate_window_shape(window), dtype=float)

        if self._keras_model is not None:
            output = np.asarray(self._keras_model.predict(rows[np.newaxis, :, :], verbose=0)).reshape(-1)
            if output.size == 1:
                failure_probability = clamp_probability(float(output[0]))
                predicted_rul = max(1, int(round(150 * (1.0 - failure_probability))))
            else:
                predicted_rul = max(1, int(round(float(output[0]))))
                failure_probability = clamp_probability(float(output[1]))
        else:
            failure_probability, predicted_rul = self._fallback_predict(rows)

        risk_level = "low"
        if failure_probability >= 0.75:
            risk_level = "high"
        elif failure_probability >= 0.4:
            risk_level = "medium"

        return {
            "failure_probability": round(failure_probability, 4),
            "predicted_rul": predicted_rul,
            "risk_level": risk_level,
            "top_contributors": summarize_feature_contributions(rows.tolist()),
        }
