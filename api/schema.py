from typing import List

from pydantic import BaseModel, Field, field_validator

from src.utils import FEATURE_COLUMNS, REQUIRED_FEATURE_COUNT, REQUIRED_SEQUENCE_LENGTH


class PredictionRequest(BaseModel):
    asset_id: str | None = Field(default=None, description="Optional equipment identifier.")
    cycles: List[List[float]] = Field(
        ...,
        description="Last 50 operating cycles with 25 features per cycle.",
    )

    @field_validator("cycles")
    @classmethod
    def validate_cycles(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) != REQUIRED_SEQUENCE_LENGTH:
            raise ValueError(f"Expected {REQUIRED_SEQUENCE_LENGTH} cycles.")
        for row in value:
            if len(row) != REQUIRED_FEATURE_COUNT:
                raise ValueError(f"Each cycle must contain {REQUIRED_FEATURE_COUNT} values.")
        return value


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    failure_probability: float
    predicted_rul: int
    risk_level: str
    top_contributors: List[FeatureContribution]


class ReportListResponse(BaseModel):
    reports: List[str]
    feature_schema: List[str] = FEATURE_COLUMNS
