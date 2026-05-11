from pydantic import BaseModel
from typing import List

class SensorWindow(BaseModel):
    # Expecting a list of 50 cycles, each cycle being a list of feature values
    cycles: List[List[float]]

class PredictionOutput(BaseModel):
    failure_probability: float
    risk: str
