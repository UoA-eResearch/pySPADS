import pandas as pd

from pipeline.reconstruct import get_X
from processing.dataclasses import LinRegCoefficients


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
