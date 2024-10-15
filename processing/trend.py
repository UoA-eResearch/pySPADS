import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel


def detect_trend(data: pd.Series) -> LinearModel:
    """Fit linear regression to find long term trend of signal"""
    x = np.array(data.index).reshape(-1, 1)
    y = np.array(data.values).reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True).fit(x, y)

    return reg


def gen_trend(data: pd.Series, reg: LinearModel) -> pd.Series:
    """Generate trend from linear regression model, for given pd.Series"""
    x = np.array(data.index).reshape(-1, 1)
    trend = pd.Series(index=data.index, data=reg.predict(x.astype(float)).flatten())
    return trend
