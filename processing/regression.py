import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_imf_components(output_imfs: pd.DataFrame, input_imfs: dict[str, pd.DataFrame], nearest_freq):
    """Fit IMF components to drivers"""
    input_cols = input_imfs.keys()
    output_index = output_imfs.index[~output_imfs.isna()]

    # TODO - set the target time-range explicitly
    hindcast_index = output_imfs.index[output_imfs.index.isin(input_imfs['Hs'].index)]

    # TODO - param or iterate?
    mode = 0

    # Perform linear regression
    X = pd.DataFrame(index=hindcast_index)
    for col in input_cols:
        if not nearest_freq.loc[mode, col].isna():
            X[col] = input_imfs[col].loc[hindcast_index, nearest_freq.loc[mode, col]]

    y = output_imfs.loc[hindcast_index, mode]

    reg = LinearRegression().fit(X, y)
    beta = reg.coef_
    c = reg.intercept_

    return beta, c


def output_imf_from_components():
    # TODO
    pass