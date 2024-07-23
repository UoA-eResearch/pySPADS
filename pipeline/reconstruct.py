import numpy as np
import pandas as pd
from optimization import MReg2
from sklearn.linear_model import LinearRegression, Ridge

from processing.dataclasses import LinRegCoefficients


def hindcast_index(imfs, signal):
    # TODO - fix this for actual hindcast, where signal index does not include hindcast period
    not_signal = list(imfs.keys() - {signal})
    return imfs[signal].index[imfs[signal].index.isin(imfs[not_signal[0]].index)]


def get_X(imfs, nearest_freqs, signal, component, hindcast_index):
    not_signal = list(imfs.keys() - {signal})
    X = pd.DataFrame(index=hindcast_index)
    for label in not_signal:
        # Get nearest frequency imf from driver, if we have one
        if not np.isnan(nearest_freqs.loc[component, label]):
            X[label] = imfs[label].loc[imfs[label].index.isin(hindcast_index), nearest_freqs.loc[component, label]]
    return X


def get_y(imfs, signal, component, hindcast_index):
    return imfs[signal].loc[hindcast_index, component]


def fit(imfs: dict[str, pd.DataFrame], nearest_freqs: pd.DataFrame, signal: str,
        model: str = 'mreg2', fit_intercept: bool = False) -> LinRegCoefficients:
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
    for i, component in enumerate(imfs[signal].columns):
        X = get_X(imfs, nearest_freqs, signal, component, index)
        y = get_y(imfs, signal, component, index)

        reg = model_class(fit_intercept=fit_intercept).fit(X, y)
        coefs = reg.coef_
        intercept = reg.intercept_

        if fit_intercept:
            intercepts[component] = intercept

        output[component] = {label: coefs[i] for i, label in enumerate(X.columns)}

    return LinRegCoefficients(coeffs=output, intercepts=intercepts if fit_intercept else None,
                              use_intercept=fit_intercept, model=model)  # TODO: handle normalize
