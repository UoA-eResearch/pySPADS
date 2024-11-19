import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import floats, lists, composite
from sklearn.linear_model import LinearRegression, Ridge

from pySPADS.optimization import MReg2


@composite
def coeffs_signals_drivers(draw, include_intercept=False, max_coeff=1000):
    """
    Generate random linear coefficients (+ optional intercept), then generate random driver data and a signal using
    those coefficients
    """
    sensible_floats = (
        floats(allow_nan=False, allow_infinity=False, allow_subnormal=False,
               min_value=-max_coeff, max_value=max_coeff)
        .filter(lambda x: x == 0 or np.abs(x) > 1e-4)
    )

    coeffs = draw(lists(sensible_floats, min_size=1, max_size=10))

    if include_intercept:
        intercept = draw(sensible_floats)
    else:
        intercept = 0.0

    num_drivers = len(coeffs)
    num_samples = 1000

    drivers = {f'driver_{i}': np.random.randn(num_samples) for i in range(num_drivers)}
    drivers_df = pd.DataFrame(drivers)

    signal = intercept + np.sum([drivers[f'driver_{i}'] * coeffs[i] for i in range(num_drivers)], axis=0)
    signal_sr = pd.Series(signal)

    return intercept, coeffs, drivers_df, signal_sr


@given(coeffs_signals_drivers(include_intercept=False))
@settings(deadline=1000)
def test_mreg_no_intercept(test_data):
    intercept, coeffs, drivers_df, signal_sr = test_data
    assert intercept == 0, "Not expecting to use intercept"

    model = MReg2(fit_intercept=False).fit(drivers_df, signal_sr)
    _coeffs = model.coef_

    # Check the coefficients are close
    for i in range(len(coeffs)):
        assert np.isclose(_coeffs[i], coeffs[i], rtol=1e-1, atol=1e-1)


# @skip("mreg does not work well with intercept")
@given(coeffs_signals_drivers(include_intercept=True))
@settings(deadline=1000)
def test_mreg_with_intercept(test_data):
    intercept, coeffs, drivers_df, signal_sr = test_data

    model = MReg2(fit_intercept=True).fit(drivers_df, signal_sr)
    _coeffs = model.coef_
    _intercept = model.intercept_

    # Check the coefficients are close
    assert np.isclose(_intercept, intercept, rtol=1e-1, atol=1e-1)
    for i in range(len(coeffs)):
        assert np.isclose(_coeffs[i], coeffs[i], rtol=1e-1, atol=1e-1)


@given(coeffs_signals_drivers(include_intercept=False))
def test_linreg_no_intercept(test_data):
    intercept, coeffs, drivers_df, signal_sr = test_data
    assert intercept == 0, "Not expecting to use intercept"

    reg = LinearRegression(fit_intercept=False).fit(drivers_df, signal_sr)
    _coeffs = reg.coef_

    # Check the coefficients are close
    for i in range(len(coeffs)):
        assert np.isclose(_coeffs[i], coeffs[i], rtol=1e-1, atol=1e-1)


@given(coeffs_signals_drivers(include_intercept=True))
def test_linreg_with_intercept(test_data):
    intercept, coeffs, drivers_df, signal_sr = test_data

    reg = LinearRegression(fit_intercept=True).fit(drivers_df, signal_sr)
    _intercept = reg.intercept_
    _coeffs = reg.coef_

    # Check the coefficients are close
    assert np.isclose(_intercept, intercept, rtol=1e-1, atol=1e-1)
    for i in range(len(coeffs)):
        assert np.isclose(_coeffs[i], coeffs[i], rtol=1e-1, atol=1e-1)


@given(coeffs_signals_drivers(include_intercept=False, max_coeff=50))
def test_ridge_no_intercept(test_data):
    intercept, coeffs, drivers_df, signal_sr = test_data
    assert intercept == 0, "Not expecting to use intercept"

    reg = Ridge(fit_intercept=False).fit(drivers_df, signal_sr)
    _coeffs = reg.coef_

    # Check the coefficients are close
    for i in range(len(coeffs)):
        assert np.isclose(_coeffs[i], coeffs[i], rtol=1e-1, atol=1e-1)


@given(coeffs_signals_drivers(include_intercept=True, max_coeff=50))
def test_ridge_with_intercept(test_data):
    intercept, coeffs, drivers_df, signal_sr = test_data

    reg = Ridge(fit_intercept=True).fit(drivers_df, signal_sr)
    _intercept = reg.intercept_
    _coeffs = reg.coef_

    # Check the coefficients are close
    assert np.isclose(_intercept, intercept, rtol=1e-1, atol=1e-1)
    for i in range(len(coeffs)):
        assert np.isclose(_coeffs[i], coeffs[i], rtol=1e-1, atol=1e-1)
