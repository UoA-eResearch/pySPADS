import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis.strategies import floats

from processing.data import load_imfs
from processing.recomposition import component_frequencies, nearest_frequency
from root import ROOT_DIR


def test_load_imfs():
    """Check that loading IMFs from a folder works as expected"""
    folder = ROOT_DIR / 'tests' / 'data' / 'imfs'
    assert folder.exists()

    imfs = load_imfs(folder)

    expected_series = ['shore', 'Hs', 'Tp', 'Dir',
                       'PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']

    # Check that each expected item was loaded, and looks like an imf
    for label in expected_series:
        key = (label, 0.1)  # include noise value
        assert key in imfs
        assert imfs[key].index.inferred_type == 'datetime64'
        assert imfs[key].columns.inferred_type == 'integer'


@given(floats(min_value=0.01, max_value=182))  # frequencies over 0.5/day can't be detected using zero-crossing method
def test_component_frequencies(freq):
    """Check that component frequencies are calculated correctly"""
    dates = pd.date_range('2020-01-01', periods=15000)
    signal = pd.Series(np.sin(2 * np.pi * freq * np.arange(len(dates)) / 365), index=dates)

    imfs = pd.DataFrame({'imf': signal})
    freqs = component_frequencies(imfs)

    # Expected range - given integer number of zero crossings
    expected_crossings = 2 * freq * (len(dates) - 1) / 365
    max_freq = freq * np.ceil(expected_crossings) / expected_crossings
    min_freq = freq * np.floor(expected_crossings) / expected_crossings

    assert (min_freq <= freqs['imf'] <= max_freq) or \
        np.isclose(freqs['imf'], min_freq) or np.isclose(freqs['imf'], max_freq), \
        f'Expected frequency to be {freq}, in range {min_freq} - {max_freq}, got {freqs["imf"]}'


def test_nearest_frequency():
    """Check that nearest frequency is calculated correctly"""
    # Trivial case
    target_freq = 1.0
    input_freqs = pd.Series([0.5, 0.8, 1.0, 2.0, 3.0])

    nearest = nearest_frequency(target_freq, input_freqs)

    assert nearest == 2, f'Expected index 2, got {nearest}'

    # Non-trivial: NaNs
    target_freq = 1.0
    input_freqs = pd.Series([np.nan, 0.8, 1.0, np.nan, 3.0])

    nearest = nearest_frequency(target_freq, input_freqs)

    assert nearest == 2, f'Expected index 2, got {nearest}'

    # Non-trivial: incomplete index
    target_freq = 1.0
    input_freqs = pd.Series([0.5, 0.8, 1.0, 3.0], index=[2, 3, 4, 7])

    nearest = nearest_frequency(target_freq, input_freqs)

    assert nearest == 4, f'Expected index 4, got {nearest}'

    # Non-trivial: multiple matches (expect first)
    target_freq = 1.0
    input_freqs = pd.Series([0.5, 0.8, 1.0, 1.0, 3.0])

    nearest = nearest_frequency(target_freq, input_freqs)

    assert nearest == 2, f'Expected index 2, got {nearest}'

    # Non-trivial: non exact match
    target_freq = 3.2
    input_freqs = pd.Series([0.5, 2.0, 3.0, 3.6, 7.0])

    nearest = nearest_frequency(target_freq, input_freqs)

    assert nearest == 2, f'Expected index 2, got {nearest}'