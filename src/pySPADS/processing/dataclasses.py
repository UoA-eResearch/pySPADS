import json

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional, Self


class SaveLoadBaseModel(BaseModel):
    """Provides save/load convenience function for pydantic models"""

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def load(cls: Self, path: str) -> Self:
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())


class LinRegCoefficients(SaveLoadBaseModel):
    """
    Model for storing linear regression parameters, coefficients and intercepts
    for storing regression results across multiple components
    """

    use_intercept: bool = False
    normalize: bool = False
    scalars: Optional[dict[int, dict[str, tuple[float, float]]]] = (
        None  # {component: {driver: (mean, scale)}}
    )
    model: Optional[str] = None  # Which model was used to generate the fit
    coeffs: dict[int, dict[str, float]]  # {component: {driver: coefficient}}
    intercepts: Optional[dict[int, float]] = None  # {component: intercept}

    def predict(self, component: int, X: dict[str : pd.DataFrame]) -> pd.DataFrame:
        if self.normalize:
            for col in X.columns:
                mean, scale = self.scalars[component][col]
                X[col] = (X[col] - mean) / scale

        if self.use_intercept:
            return self.intercepts[component] + np.sum(
                [X[driver] * self.coeffs[component][driver] for driver in X.columns],
                axis=0,
            )
        else:
            return np.sum(
                [X[driver] * self.coeffs[component][driver] for driver in X.columns],
                axis=0,
            )


class TrendModel(SaveLoadBaseModel):
    """Linear regression model for trend prediction"""

    coeff: float = 1.0
    intercept: float = 0.0

    def predict(self, x: np.array) -> np.array:
        """Predict trend for given x, defaults to returning x unmodified"""
        return self.intercept + self.coeff * x
