import pandas as pd
import warnings

from .significance import zero_crossings


def component_frequencies(imfs: pd.DataFrame) -> pd.Series:
    """
    Calculate the frequency of each IMF mode, in cycles per year
    :param imfs: DataFrame of IMF modes, with one column for each mode
    """
    t_range = imfs.index.max() - imfs.index.min()
    return 365 * imfs.apply(zero_crossings, axis=0) / (2 * t_range)


def nearest_frequency(output_freqs: pd.Series, input_freqs: pd.DataFrame) -> pd.DataFrame:
    """
    Find the nearest frequency in each input IMF to the frequency of each output mode
    :param output_freqs: Frequencies of each output IMF mode
    :param input_freqs: Frequencies of each input IMF mode for each input time series
    :return: DataFrame of nearest input IMF mode for each output IMF mode, with one column for each input time series
    """
    input_cols = input_freqs.columns
    output_index = output_freqs.index[~output_freqs.isna()]

    result = pd.DataFrame(index=output_index, columns=input_cols)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        for col in input_cols:
            result[col] = output_freqs.apply(lambda x: (input_freqs[col] - x).abs().argmin(skipna=True))

    return result


def frequency_difference(output_freqs: pd.Series, input_freqs: pd.DataFrame, nearest_freq: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Calculates the difference between each output IMF mode, and each selected matching input IMF mode
    :param output_freqs: Frequencies of each output IMF mode
    :param input_freqs: Frequencies of each input IMF mode for each input time series
    :param nearest_freq: Nearest input IMF mode for each output IMF mode
    :return: DataFrame of frequency differences between each output IMF mode, and each selected matching input IMF mode
    """
    input_cols = input_freqs.columns
    output_index = output_freqs.index[~output_freqs.isna()]

    result = pd.DataFrame(index=output_index, columns=input_cols)
    for col in input_cols:
        diff = (output_freqs[output_index].reset_index(drop=True) -
                input_freqs.loc[nearest_freq[col], col].reset_index(drop=True))
        diff.index = output_index
        result[col] = diff

    return result


def relative_frequency_difference(output_freqs: pd.Series, input_freqs: pd.DataFrame, nearest_freq: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Calculates the relative difference between each output IMF mode, and each selected matching input IMF mode
    :param output_freqs: Frequencies of each output IMF mode
    :param input_freqs: Frequencies of each input IMF mode for each input time series
    :param nearest_freq: Nearest input IMF mode for each output IMF mode
    :return: DataFrame of relative frequency differences between each output IMF mode, and each selected matching
             input IMF mode
    """
    input_cols = input_freqs.columns
    output_index = output_freqs.index[~output_freqs.isna()]

    diff_df = frequency_difference(output_freqs, input_freqs, nearest_freq)

    result = pd.DataFrame(index=output_index, columns=input_cols)
    for col in input_cols:
        result[col] = (diff_df[col] / output_freqs[output_index]).abs()

    # If output has a component with frequency 0, then we can't calculate relative error
    # Instead, check that the input component is < tolerance x next lowest frequency output component
    if output_freqs.min() == 0:
        print('Warning: output has a component with frequency 0, comparing input component to next lowest frequency '
              'output component')
        zero_index = output_freqs.argmin()
        next_lowest = output_freqs[output_freqs > 0].min()
        for col in input_cols:
            result.loc[zero_index, col] = abs(diff_df.loc[zero_index, col] / next_lowest)

    return result
