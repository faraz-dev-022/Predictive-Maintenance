from typing import Iterable

FEATURE_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{index}" for index in range(1, 22)],
    "normalized_cycle",
]
REQUIRED_SEQUENCE_LENGTH = 50
REQUIRED_FEATURE_COUNT = len(FEATURE_COLUMNS)


def validate_window_shape(window: Iterable[Iterable[float]]) -> list[list[float]]:
    rows = [list(row) for row in window]
    if len(rows) != REQUIRED_SEQUENCE_LENGTH:
        raise ValueError(f"Expected {REQUIRED_SEQUENCE_LENGTH} cycles, received {len(rows)}.")
    for row in rows:
        if len(row) != REQUIRED_FEATURE_COUNT:
            raise ValueError(f"Expected {REQUIRED_FEATURE_COUNT} features per cycle, received {len(row)}.")
    return rows


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
