from processing.data import load_imfs
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
