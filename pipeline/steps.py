import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from optimization import MReg2
from pipeline.reconstruct import get_X, hindcast_index, get_y
from processing.dataclasses import LinRegCoefficients


def fit(imfs: dict[str, pd.DataFrame], nearest_freqs: pd.DataFrame, signal: str,
        model: str = 'mreg2', fit_intercept: bool = False, normalize: bool = False,
        exclude_trend: bool = False) -> LinRegCoefficients:
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
