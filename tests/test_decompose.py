from unittest import TestCase

import pandas as pd

from pipeline.decompose import load_data_from_csvs, _interpolate, decompose
from root import ROOT_DIR


class Test(TestCase):
    def test_interpolate(self):
        # Construct a dataframe with a datetime index with gaps in it
        df = pd.DataFrame({
            't': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-05']),
            'x': [1, 2, 3.5]
        }).set_index('t')

        # Interpolate to daily intervals
        res = _interpolate(df)

        # Check missing dates are present
        self.assertEqual(len(res), 5)
        self.assertTrue(pd.to_datetime('2021-01-03') in res.index)
        self.assertTrue(pd.to_datetime('2021-01-04') in res.index)

        # Check for linearly interpolated data for missing dates
        self.assertEqual(res.loc['2021-01-03', 'x'], 2.5)
        self.assertEqual(res.loc['2021-01-04', 'x'], 3.0)

        # Check original data unchanged
        self.assertEqual(res.loc['2021-01-01', 'x'], 1)
        self.assertEqual(res.loc['2021-01-02', 'x'], 2)
        self.assertEqual(res.loc['2021-01-05', 'x'], 3.5)

    def test_load_separate_files(self):
        folder = ROOT_DIR / 'data' / 'separate_files'
        self.assertTrue(folder.exists())

        data = load_data_from_csvs(folder)

        # Expect one series for each input variable
        expected_series = ['shore', 'Hs', 'Tp', 'Dir',
                           'PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']

        for s in expected_series:
            self.assertTrue(s in data)
            self.assertEqual(s, data[s].name)

        self.assertEqual(len(data), len(expected_series))

    def test_decompose(self):
        """Check that decomposition results in expected data"""
        file = ROOT_DIR / 'data' / 'separate_files' / 'shore.csv'
        data = load_data_from_csvs(file)['shore']

        # Use only a small amount of data for testing
        data = data[:1000]

        imf_df = decompose(data, noise=0.1, NR=100, progress=False)

        self.assertTrue(all(imf_df.index == data.index))
        # Expect column headers to be integer 0, 1, ...
        self.assertTrue(1 in imf_df.columns)
        self.assertTrue(isinstance(imf_df.columns[1], int))
