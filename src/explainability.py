import base64
from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils import FEATURE_COLUMNS, validate_window_shape

PNG_PLACEHOLDER = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9s0N0x8AAAAASUVORK5CYII="
)


def summarize_feature_contributions(window: Iterable[Iterable[float]]) -> list[dict[str, float | str]]:
    rows = np.asarray(validate_window_shape(window), dtype=float)
    centered = rows[-1] - rows.mean(axis=0)
    scale = rows.std(axis=0) + 1e-6
    contributions = np.abs(centered / scale)

    ranked = np.argsort(contributions)[::-1][:5]
    return [
        {"feature": FEATURE_COLUMNS[index], "contribution": round(float(contributions[index]), 4)}
        for index in ranked
    ]


def ensure_default_reports(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    html_path = output / "drift_report.html"
    if not html_path.exists():
        html_path.write_text(
            """<html><body><h1>Predictive Maintenance Drift Report</h1><p>No live drift detected in the default baseline sample.</p></body></html>""",
            encoding="utf-8",
        )

    png_path = output / "shap_summary.png"
    if not png_path.exists():
        png_path.write_bytes(PNG_PLACEHOLDER)

    return output
