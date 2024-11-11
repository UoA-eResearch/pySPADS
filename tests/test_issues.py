from pathlib import Path
from .common import dpath, read_config, nearest_freq


def test_issue_i0():
    folder = dpath('issues/i0')
    config = read_config(folder / 'config.yaml')

    imfs = [
        folder / 'imfs' / f'{label}_imf_0.5.csv'
        for label in ['shore', 'Hs', 'Tp', 'Dir']
    ]

    nearest_freq(0.5, config['signal'], config['frequency_threshold'], config['exclude_trend'], imfs)
