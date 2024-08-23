import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from PyEMD.checks import whitenoise_check
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel


def decompose(data: pd.Series, noise: float, num_trials: int = 100, progress=False, parallel=True) -> pd.DataFrame:
    """
    Decompose a time series into IMF modes using CEEMDAN
    :param data: the timeseries to decompose
    :param noise: noise level to use for CEEMDAN
    :param num_trials: number of trials to use for CEEMDAN
    :param progress: whether to show a progress bar
    :param parallel: whether to use parallel processing
    :return: a dataframe containing the IMF modes
    """
    assert data.index.inferred_type == 'datetime64', 'Index should be datetime'
    assert data.index.is_monotonic_increasing, 'Data must be sorted by time'

    ceemd = CEEMDAN(trials=num_trials, epsilon=noise, processes=8 if parallel else None, parallel=parallel)
    imfs = ceemd.ceemdan(data.to_numpy(), data.index.to_numpy(), progress=progress)

    return pd.DataFrame(imfs.T, index=data.index)


def reject_noise(imf: pd.DataFrame, noise_threshold=0.95) -> pd.DataFrame:
    """
    Reject components of an IMF that are mostly noise
    :param imf: the input IMF
    :param noise_threshold: threshold for the proportion of noise in a component
    :return: a copy of the input IMF, with columns representing high-noise components dropped
    """
    sig = whitenoise_check(imf.to_numpy().T, alpha=noise_threshold)
    rejects = [k for k, v in sig.items() if v == 0]

    if len(rejects) == len(imf.columns):
        raise Exception(f'All components of imf are noise')  # TODO: report name of imf

    if imf.columns[-1] in rejects:
        print(f'Warning: Trend component of imf is noise')
        rejects.remove(imf.columns[-1])

    # TODO: handle all noise - warn user? set threshold=0 for this channel?
    return imf.drop(columns=[i - 1 for i in rejects])


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
