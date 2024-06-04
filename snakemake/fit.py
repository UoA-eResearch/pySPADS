import json

import numpy as np
import pandas as pd

from pipeline.reconstruct import fit
from processing.data import parse_filename, load_imf

# Parameters
noise = float(snakemake.wildcards.noise)
signal = snakemake.params.c['signal']

# Load imfs
imfs = {}
for fname in snakemake.input.imfs:
    label, imf_noise = parse_filename(fname)
    assert imf_noise == noise, f'Expected noise {noise} but got {imf_noise}'
    imfs[label] = load_imf(fname)

# Load nearest frequency
nearest_freq = pd.read_csv(snakemake.input.freqs, index_col=0)

# Linear regression
coeffs = fit(imfs, nearest_freq, signal)

# Save
with open(snakemake.output[0], 'w') as f:
    json.dump(coeffs, f, default=np.ndarray.tolist, indent=4)
