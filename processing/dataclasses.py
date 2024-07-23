import json

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional

from pydantic.main import Model


class SaveLoadBaseModel(BaseModel):
    """Provides save/load convenience function for pydantic models"""
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def load(cls: type[Model], path: str) -> Model:
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())


class LinRegCoefficients(SaveLoadBaseModel):
    """
    Model for storing linear regression parameters, coefficients and intercepts
    for storing regression results across multiple components
    """
    use_intercept: bool = False
    normalize: bool = False
    coeffs: dict[int, dict[str, float]]  # {component: {driver: coefficient}}
    intercepts: Optional[dict[int, float]] = None  # {component: intercept}

    def predict(self, component: int, X: dict[str: pd.DataFrame]) -> pd.DataFrame:
        if self.use_intercept:
            return self.intercepts[component] + np.sum(
                [X[driver] * self.coeffs[component][driver] for driver in X.columns], axis=0
            )
        else:
            return np.sum([X[driver] * self.coeffs[component][driver] for driver in X.columns], axis=0)
