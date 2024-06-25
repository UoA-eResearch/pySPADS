from unittest import TestCase

from processing.data import load_imfs
from root import ROOT_DIR


class Test(TestCase):
    def test_load_imfs(self):
        """Check that loading IMFs from a folder works as expected"""
        folder = ROOT_DIR / 'tests' / 'data' / 'imfs'
        self.assertTrue(folder.exists())

        imfs = load_imfs(folder)

        expected_series = ['shore', 'Hs', 'Tp', 'Dir',
                           'PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']

        # Check that each expected item was loaded, and looks like an imf
        for label in expected_series:
            key = (label, 0.1)  # include noise value
            self.assertTrue(key in imfs)
            self.assertTrue(imfs[key].index.inferred_type == 'datetime64')
            self.assertTrue(imfs[key].columns.inferred_type == 'integer')
