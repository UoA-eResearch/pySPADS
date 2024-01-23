from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.decompose import parse_filename
from processing.recomposition import component_frequencies, nearest_frequency, relative_frequency_difference


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
        imfs[(label, noise)] = pd.read_csv(file, index_col=0)

        # Convert column names to ints
        imfs[(label, noise)].columns = imfs[(label, noise)].columns.astype(int)
    return imfs


def match_frequencies(imfs: dict[str, pd.DataFrame], signal: str, threshold: float) -> pd.DataFrame:
    """
    Match frequencies of each IMF mode in the input data, to each IMF mode in the output data
    :param imfs: dict of input IMFs, with one DataFrame for each input time series
    :param signal: name of the output time series
    :param threshold: maximum relative difference between frequencies
    :return: DataFrame of nearest input IMF mode for each output IMF mode, with one column for each input time series
    """
    # Get the frequency of each component of each imf
    print('Calculating IMF mode frequencies')
    max_mode = max([imf_df.shape[1] for imf_df in imfs.values()])
    freq_df = pd.DataFrame(index=range(max_mode), columns=list(imfs.keys()))
    for label, imf_df in imfs.items():
        freq_df[label] = component_frequencies(imf_df)

    # Find the nearest frequency in each input IMF to the frequency of each output mode
    print('Matching signal modes to nearest frequencies in input modes')
    nearest_freq = nearest_frequency(freq_df[signal], freq_df.drop(columns=[signal]))

    # Check if the nearest frequency is within the threshold
    print('Checking if nearest frequencies are within threshold')
    rel_diff_df = relative_frequency_difference(freq_df[signal], freq_df.drop(columns=[signal]), nearest_freq)

    valid_components = (rel_diff_df < threshold).sum(axis=1)
    if any(valid_components == 0):
        raise ValueError(f'No valid input components for output components: '
                         f'{valid_components[valid_components == 0].index}')

    if any(valid_components < 3):
        print(f'Warning: some output components have less than 3 valid input components: '
              f'{valid_components[valid_components < 3].index}')

    # Show the components which are used for each output component
    print('Components used for each output component:')
    for i in rel_diff_df.index:
        print(f'{i:>5} : {np.sum(rel_diff_df.loc[i] < threshold)}')

    return nearest_freq[rel_diff_df < threshold]
