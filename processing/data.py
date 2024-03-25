from pathlib import Path

import numpy as np
import pandas as pd

from processing.recomposition import epoch_index_to_datetime


def load_imfs(folder: Path) -> dict[tuple[str, float], pd.DataFrame]:
    """
    Load IMFs from a folder, parse label and noise from filenames
    :param folder: folder containing IMF files
    :return: dict of IMFs, with keys (label, noise)
    """
    assert folder.is_dir(), f'Folder {folder} does not exist'
    imfs = {}
    for file in folder.glob('*.csv'):
        label, noise = parse_filename(file)
        imfs[(label, noise)] = pd.read_csv(file, index_col=0, parse_dates=True)

        # TODO: temporary until data is regenerated
        for key in imfs:
            if imfs[key].index.inferred_type == 'integer':
                imfs[key] = epoch_index_to_datetime(imfs[key])

        # Convert column names to ints
        imfs[(label, noise)].columns = imfs[(label, noise)].columns.astype(int)
    return imfs


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
    # for df in dfs:
    #     df.index = df.index.astype(np.int64) // 10 ** 9

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
