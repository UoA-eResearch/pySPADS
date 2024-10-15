import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from optimization import MReg2
from pipeline.reconstruct import get_X, hindcast_index, get_y
from processing.dataclasses import LinRegCoefficients
from processing.recomposition import component_frequencies, nearest_frequency, relative_frequency_difference
from processing.significance import zero_crossings

import logging

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
        # TODO: make this behaviour a parameter?
        if freq_df.loc[trend_col, signal] == 0:
            freq_df.loc[trend_col, signal] = np.nan
        elif zero_crossings(imfs[signal][trend_col]) <= 2:
            print("Warning: excluding signal trend with <= 2 zero crossings")
            freq_df.loc[trend_col, signal] = np.nan
        else:
            print('Warning: signal has no zero-frequency trend component to exclude')

    # Find the nearest frequency in each input IMF to the frequency of each output mode
    logger.info('Matching signal modes to nearest frequencies in input modes')
    nearest_freq = nearest_frequency(freq_df[signal], freq_df.drop(columns=[signal]))

    # Check if the nearest frequency is within the threshold
    logger.info('Checking if nearest frequencies are within threshold')
    rel_diff_df = relative_frequency_difference(freq_df[signal], freq_df.drop(columns=[signal]), nearest_freq)

    valid_components = (rel_diff_df < threshold).sum(axis=1)
    # TODO: check this behaviour/make it a parameter
    if valid_components.iloc[-1] == 0:
        print('Warning: No valid input components for trend component of output signal, excluding from output')
    if any(valid_components.iloc[:-1] == 0):
        print(f'Warning: No valid input components for non-trend output component: '
                         f'{list(valid_components[valid_components == 0].index)}')
        # raise ValueError(f'No valid input components for non-trend output component: '
        #                  f'{list(valid_components[valid_components == 0].index)}')

    if any(valid_components < 3):
        logger.info(f'Warning: some output components have less than 3 valid input components: '
                    f'{list(valid_components[valid_components < 3].index)}')

    # Show the components which are used for each output component
    logger.info('Components used for each output component:')
    for i in rel_diff_df.index:
        logger.info(f'{i:>5} : {np.sum(rel_diff_df.loc[i] < threshold)}')

    return nearest_freq[rel_diff_df < threshold].dropna(how='all')


def fit(imfs: dict[str, pd.DataFrame], nearest_freqs: pd.DataFrame, signal: str,
        model: str = 'mreg2', fit_intercept: bool = False, normalize: bool = False) -> LinRegCoefficients:
    """
    Fit linear regression model to predict signal components from driver components
    :return: LinRegCoefficients object containing coefficients and intercepts
    """
    index = hindcast_index(imfs, signal)

    # Select model to use
    model_classes = {
        'mreg2': MReg2,
        'linreg': LinearRegression,
        'ridge': Ridge
    }
    assert model in model_classes, f'Model {model} not implemented, available models include {", ".join(model_classes)}'
    model_class = model_classes[model]

    output = {}
    intercepts = {}
    scalars = {}
    for i, component in enumerate(imfs[signal].columns):
        if component not in nearest_freqs.index:
            # Given component had no matching frequencies in drivers
            continue

        X = get_X(imfs, nearest_freqs, signal, component, index)
        y = get_y(imfs, signal, component, index)

        if normalize:
            sc = StandardScaler().set_output(transform='pandas')
            X = sc.fit_transform(X)
            scalars[component] = {label: (sc.mean_[i], sc.scale_[i]) for i, label in enumerate(X.columns)}

        reg = model_class(fit_intercept=fit_intercept).fit(X, y)
        coefs = reg.coef_
        intercept = reg.intercept_

        if fit_intercept:
            intercepts[component] = intercept

        output[component] = {label: coefs[i] for i, label in enumerate(X.columns)}

    return LinRegCoefficients(coeffs=output, intercepts=intercepts if fit_intercept else None,
                              use_intercept=fit_intercept, model=model,
                              normalize=normalize, scalars=scalars if normalize else None)


def predict(imfs: dict[str, pd.DataFrame], nearest_freqs: pd.DataFrame, signal: str,
            coefficients: LinRegCoefficients, start_date: str, end_date: str,
            exclude_trend: bool = False) -> pd.DataFrame:
    """
    For each component, predict a signal from its drivers, given coefficients for each driver
    :return: DataFrame containing predicted components of the signal
    """
    index = pd.date_range(start=start_date, end=end_date, freq='D')
    output_columns = imfs[signal].columns
    if exclude_trend:
        output_columns = output_columns[:-1]
    component_predictions = pd.DataFrame(index=index, columns=output_columns)

    for component in output_columns:
        if component in nearest_freqs.index:
            X = get_X(imfs, nearest_freqs, signal, component, index)

            component_predictions.loc[:, component] = coefficients.predict(component, X)
        else:
            component_predictions = component_predictions.drop(columns=[component])

    return component_predictions
