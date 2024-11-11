import itertools

import pandas as pd

from pipeline import steps
from processing.data import _interpolate, load_data_from_csvs, imf_filename, parse_filename
from root import ROOT_DIR


def test_interpolate():
    # Construct a dataframe with a datetime index with gaps in it
    df = pd.DataFrame({
        't': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-05']),
        'x': [1, 2, 3.5]
    }).set_index('t')

    # Interpolate to daily intervals
    res = _interpolate(df)

    # Check missing dates are present
    assert len(res) == 5, 'Expected more rows in interpolated data'
    assert pd.to_datetime('2021-01-03') in res.index, 'Interpolated data does not contain expected date: 2021-01-03'
    assert pd.to_datetime('2021-01-04') in res.index, 'Interpolated data does not contain expected date: 2021-01-04'

    # Check for linearly interpolated data for missing dates
    assert res.loc['2021-01-03', 'x'] == 2.5, f'Expected interpolated data to be 2.5, got {res.loc["2021-01-03", "x"]}'
    assert res.loc['2021-01-04', 'x'] == 3.0,f'Expected interpolated data to be 3.0, got {res.loc["2021-01-04", "x"]}'

    # Check original data unchanged
    assert res.loc['2021-01-01', 'x'] == 1, 'Original data should be unchanged'
    assert res.loc['2021-01-02', 'x'] == 2, 'Original data should be unchanged'
    assert res.loc['2021-01-05', 'x'] == 3.5, 'Original data should be unchanged'


def test_load_separate_files():
    folder = ROOT_DIR / 'data' / 'separate_files'
    assert folder.exists()

    data = load_data_from_csvs(folder)

    # Expect one series for each input variable
    expected_series = ['shore', 'Hs', 'Tp', 'Dir',
                       'PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']

    for s in expected_series:
        assert s in data, f'Missing expected series {s}'
        assert s == data[s].name, f'Expected series name to be {s}, got {data[s].name}'

    assert len(data) == len(expected_series), f'Expected {len(expected_series)} series, got {len(data)}'


def test_decompose():
    """Check that decomposition results in expected data"""
    file = ROOT_DIR / 'data' / 'separate_files' / 'shore.csv'
    data = load_data_from_csvs(file)['shore']

    # Use only a small amount of data for testing
    data = data[:1000]

    imf_df = steps.decompose(data, noise=0.1, num_trials=100, progress=False, parallel=False)

    assert all(imf_df.index == data.index), 'Index should be the same as input data'
    # Expect column headers to be integer 0, 1, ...
    for i, col in enumerate(imf_df.columns):
        assert isinstance(col, int), f'Expected column to be integer, got {col} of type {type(col)}'
        assert col == i, f'Expected column to be {i}, got {col}'

    assert not imf_df.isna().any().any(), 'No NaNs should be present in the output data'


def test_imf_filename():
    """Check that the filename is generated correctly"""
    output_dir = ROOT_DIR / 'data' / 'imfs'

    filename = imf_filename(output_dir, 'shore', 0.1)
    assert filename == output_dir / 'shore_imf_0.1.csv', 'Incorrect formatted filename'

    # Check expected noise precision
    filename = imf_filename(output_dir, 'shore', 0.123456789)
    assert filename == output_dir / 'shore_imf_0.123456789.csv', 'Incorrect formatted filename'


def test_parse_filaname():
    """Check that the filename is parsed correctly"""
    filename = ROOT_DIR / 'data' / 'imfs' / 'shore_imf_0.1.csv'
    label, noise = parse_filename(filename)

    assert label == 'shore', 'Incorrect label parsed'
    assert noise == 0.1, 'Incorrect noise parsed'

    for label, noise in itertools.product(
        ['shore', 'test', 'a_', '_b', '_c__'],
        [0, 0.1, 0.123456789]
    ):
        filename = imf_filename(ROOT_DIR / 'data' / 'imfs', label, noise)
        parsed_label, parsed_noise = parse_filename(filename)
        assert label == parsed_label, f'Incorrect label parsed for filename {filename}'
        assert noise == parsed_noise, f'Incorrect noise parsed for filename {filename}'
