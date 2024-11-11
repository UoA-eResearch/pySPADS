import os
from pathlib import Path


def dpath(path):
    """get the path to a data file (relative to the directory this
    test lives in)"""
    return Path(os.path.realpath(os.path.join(os.path.dirname(__file__), path)))


def read_config(config_file):
    """Replicate snakefile rule read_config.py"""
    import yaml

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def nearest_freq(noise, signal, threshold, exclude_trend, input_imfs, output=None):
    """Replicate snakefile rule nearest_freq.py"""
    from pipeline import steps
    from processing.data import load_imf, parse_filename

    # Load imfs
    imfs = {}
    for fname in input_imfs:
        label, imf_noise = parse_filename(fname)
        assert imf_noise == noise, f'Expected noise {noise} but got {imf_noise}'
        imfs[label] = load_imf(fname)

    # Find nearest frequency
    nearest_freq = steps.match_frequencies(imfs, signal, threshold, exclude_trend)

    # Save
    if output:
        nearest_freq.to_csv(output)

    # Return value for testing
    return nearest_freq
