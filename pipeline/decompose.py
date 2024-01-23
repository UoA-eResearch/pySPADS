from pathlib import Path

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from PyEMD.checks import whitenoise_check


def _interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate time index to daily interval"""
    t_range = pd.date_range(df.index.min(), df.index.max(), freq='D')

    return (
        df
        .reindex(t_range, fill_value=np.nan)
        .interpolate(method='linear')
    )


def load_data_from_csvs(path: Path, time_col: str = 't') -> dict[str, pd.Series]:
    """
    Load data from csv files, interpolate to regular time intervals, and return as a dict of series
    :param path: either a single CSV file, or a directory containing multiple files
    :param time_col: name of the datetime column
    :return: a dict containing a pd.Series for each timeseries found
    """
    if path.is_dir():
        # Load all csv files in directory
        dfs = [pd.read_csv(file, parse_dates=[time_col]).set_index(time_col)
               for file in path.glob('*.csv')]
    else:
        # Load single csv file
        dfs = [pd.read_csv(path, parse_dates=[time_col]).set_index(time_col)]

    # Interpolate to regular time intervals
    dfs = [_interpolate(df) for df in dfs]

    # Convert datetimes to seconds since epoch for internal use
    for df in dfs:
        df.index = df.index.astype(np.int64) // 10 ** 9

    # Note - We can't combine data, as they may have different time ranges
    # Return a dict of series
    out = {}
    for df in dfs:
        for col in df.columns:
            assert col not in out, f'Column {col} already exists'
            out[col] = df[col]

    return out


def imf_filename(output_dir: Path, label: str, noise: float) -> Path:
    """Generate a filename for an IMF file"""
    noise_str = f'{noise:.3f}'.replace('.', '_')
    return output_dir / f'{label}_imf_{noise_str}.csv'


def parse_filename(filename: Path) -> tuple[str, float]:
    """Parse an IMF filename into label and noise"""
    label, noise_str = filename.stem.split('_imf_')
    noise = float(noise_str.replace('_', '.'))
    return label, noise


def decompose(data: pd.Series, noise: float, num_trials: int = 100, progress=False) -> pd.DataFrame:
    """
    Decompose a time series into IMF modes using CEEMDAN
    :param data: the timeseries to decompose
    :param noise: noise level to use for CEEMDAN
    :param num_trials: number of trials to use for CEEMDAN
    :param progress: whether to show a progress bar
    :return: a dataframe containing the IMF modes
    """
    assert data.index.inferred_type == 'integer', 'Index should be integer seconds since epoch'
    assert data.index.is_monotonic_increasing, 'Data must be sorted by time'

    ceemd = CEEMDAN(trials=num_trials, epsilon=noise, processes=8, parallel=True)
    imfs = ceemd.ceemdan(data.to_numpy(), data.index.to_numpy(), progress=progress)

    return pd.DataFrame(imfs.T, index=data.index)


def reject_noise(imfs: dict[str, pd.DataFrame], noise_threshold=0.95) -> dict[str, pd.DataFrame]:
    """
    Reject IMFs which are mostly noise
    :param imfs: dict of input IMFs, with one DataFrame for each input time series
    :param noise_threshold: threshold for the proportion of noise in an IMF
    :return: dict of IMFs with noise removed
    """
    for label, imf_df in imfs.items():
        sig = whitenoise_check(imf_df.to_numpy().T, alpha=noise_threshold)
        rejects = [k for k, v in sig.items() if v == 0]
        imfs[label] = imf_df.drop(columns=[i - 1 for i in rejects])

    return imfs
