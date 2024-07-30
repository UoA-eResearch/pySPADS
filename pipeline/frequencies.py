import numpy as np
import pandas as pd
import logging

from processing.recomposition import component_frequencies, nearest_frequency, relative_frequency_difference

logger = logging.getLogger(__name__)


def match_frequencies(imfs: dict[str, pd.DataFrame], signal: str, threshold: float,
                      exclude_trend: bool = False) -> pd.DataFrame:
    """
    Match frequencies of each IMF mode in the input data, to each IMF mode in the output data
    :param imfs: dict of input IMFs, with one DataFrame for each input time series
    :param signal: name of the output time series
    :param threshold: maximum relative difference between frequencies
    :param exclude_trend: whether to exclude the trend component from the input data
    :return: DataFrame of nearest input IMF mode for each output IMF mode, with one column for each input time series
    """
    # Get the frequency of each component of each imf
    logger.info('Calculating IMF mode frequencies')
    max_mode = max([imf_df.shape[1] for imf_df in imfs.values()])
    freq_df = pd.DataFrame(index=range(max_mode), columns=list(imfs.keys()))
    for label, imf_df in imfs.items():
        freq_df[label] = component_frequencies(imf_df)

    # If excluding trend, drop freq data for trend component of the signal
    #  nearest frequencies will then not be generated from driver components
    if exclude_trend:
        trend_col = imfs[signal].columns[-1]
        assert freq_df.loc[trend_col, signal] == 0, 'Trend component must have frequency 0'
        freq_df.loc[trend_col, signal] = np.nan

    # Find the nearest frequency in each input IMF to the frequency of each output mode
    logger.info('Matching signal modes to nearest frequencies in input modes')
    nearest_freq = nearest_frequency(freq_df[signal], freq_df.drop(columns=[signal]))

    # Check if the nearest frequency is within the threshold
    logger.info('Checking if nearest frequencies are within threshold')
    rel_diff_df = relative_frequency_difference(freq_df[signal], freq_df.drop(columns=[signal]), nearest_freq)

    valid_components = (rel_diff_df < threshold).sum(axis=1)
    if any(valid_components == 0):
        raise ValueError(f'No valid input components for output components: '
                         f'{valid_components[valid_components == 0].index}')

    if any(valid_components < 3):
        logger.info(f'Warning: some output components have less than 3 valid input components: '
                    f'{valid_components[valid_components < 3].index}')

    # Show the components which are used for each output component
    logger.info('Components used for each output component:')
    for i in rel_diff_df.index:
        logger.info(f'{i:>5} : {np.sum(rel_diff_df.loc[i] < threshold)}')

    return nearest_freq[rel_diff_df < threshold]
