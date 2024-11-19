import numpy as np
import pandas as pd
import warnings

from .significance import zero_crossings


def epoch_index_to_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the index of a DataFrame from epoch time (in seconds) to days since epoch
    :param df: DataFrame with epoch time index
    :return: DataFrame with index converted to days
    """
    assert all(
        df.index < 10000 * 10**9
    ), "Expecting input timeseries to be in seconds since epoch, data looks like nanoseconds"
    return df.set_index(df.index // (24 * 60 * 60))


def epoch_index_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the index of a DataFrame from epoch time (in seconds) to datetime
    :param df: DataFrame with epoch time index
    :return: DataFrame with index converted to datetime
    """
    assert all(
        df.index < 10000 * 10**9
    ), "Expecting input timeseries to be in seconds since epoch, data looks like nanoseconds"
    return df.set_index(pd.to_datetime(df.index, unit="s"))


def component_frequencies(imfs: pd.DataFrame) -> pd.Series:
    """
    Calculate the frequency of each IMF mode, in cycles per year
    :param imfs: DataFrame of IMF modes, with one column for each mode
    """
    assert (
        imfs.index.inferred_type == "datetime64"
    ), "Expecting input timeseries to be in datetime format"
    t_range = imfs.index.max() - imfs.index.min()
    assert (
        t_range.days == len(imfs) - 1
    ), "Expecting input timeseries to be evenly spaced, with daily frequency"

    return 365 * imfs.apply(zero_crossings, axis=0) / (2 * t_range.days)


def nearest_frequency(target_freq: float, input_freqs: pd.Series) -> int:
    """
    Find the nearest frequency in the input IMF to the target frequency
    :param target_freq: Frequency to match to
    :param input_freqs: Frequencies of each input IMF mode
    :return: Index of the nearest frequency in the input IMF
    """
    # Geometric difference between frequencies
    g_diff = input_freqs / target_freq
    g_diff[g_diff < 1] = 1 / g_diff[g_diff < 1]

    # Return the index of the minimum difference
    return g_diff.idxmin()


def nearest_frequencies(
    output_freqs: pd.Series, input_freqs: pd.DataFrame
) -> pd.DataFrame:
    """
    Find the nearest frequency in each input IMF to the frequency of each output mode
    :param output_freqs: Frequencies of each output IMF mode
    :param input_freqs: Frequencies of each input IMF mode for each input time series
    :return: DataFrame of nearest input IMF mode for each output IMF mode, with one column for each input time series
    """
    input_cols = input_freqs.columns
    output_index = output_freqs.index

    result = pd.DataFrame(index=output_index, columns=input_cols)
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        for col in input_cols:
            result[col] = output_freqs.apply(
                lambda x: nearest_frequency(x, input_freqs[col])
            )

    return result[~output_freqs.isna()].astype(np.int64)


def frequency_difference(
    output_freqs: pd.Series, input_freqs: pd.DataFrame, nearest_freq: pd.DataFrame
) -> pd.DataFrame:
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
        diff = output_freqs[output_index].reset_index(drop=True) - input_freqs.loc[
            nearest_freq[col], col
        ].reset_index(drop=True)
        diff.index = output_index
        result[col] = diff

    return result


def relative_frequency_difference(
    output_freqs: pd.Series, input_freqs: pd.DataFrame, nearest_freq: pd.DataFrame
) -> pd.DataFrame:
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
        print(
            "Warning: output has a component with frequency 0, comparing input component to next lowest frequency "
            "output component"
        )
        zero_index = output_freqs.argmin()
        next_lowest = output_freqs[output_freqs > 0].min()
        for col in input_cols:
            result.loc[zero_index, col] = abs(
                diff_df.loc[zero_index, col] / next_lowest
            )

    return result
