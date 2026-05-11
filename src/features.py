from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils import FEATURE_COLUMNS, validate_window_shape


def build_feature_frame(window: Iterable[Iterable[float]]) -> pd.DataFrame:
    rows = validate_window_shape(window)
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    sensor_columns = [column for column in FEATURE_COLUMNS if column.startswith("sensor_")]

    for column in sensor_columns:
        frame[f"{column}_rolling_mean"] = frame[column].rolling(window=5, min_periods=1).mean()
        frame[f"{column}_rolling_std"] = frame[column].rolling(window=5, min_periods=1).std().fillna(0.0)

    return frame


def save_processed_window(window: Iterable[Iterable[float]], output_path: str | Path) -> Path:
    frame = build_feature_frame(window)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return output
