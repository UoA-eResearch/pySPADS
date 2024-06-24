import pandas as pd
from PyEMD import CEEMDAN
from PyEMD.checks import whitenoise_check


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
    return imf.drop(columns=[i - 1 for i in rejects])
