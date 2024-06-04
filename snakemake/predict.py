import json

import numpy as np
import pandas as pd

from pipeline.reconstruct import get_X
from processing.data import parse_filename, load_imf

# Parameters
noise = float(snakemake.wildcards.noise)
signal = snakemake.params.c['signal']

with open(snakemake.input.dates, 'r') as f:
    dates = json.load(f)

# TODO: should we be predicting from start-end or hindcast-end?
start_date = dates['start']
end_date = dates['end']

# Load imfs
imfs = {}
for fname in snakemake.input.imfs:
    label, imf_noise = parse_filename(fname)
    assert imf_noise == noise, f'Expected noise {noise} but got {imf_noise}'
    imfs[label] = load_imf(fname)

# Load nearest frequency
nearest_freq = pd.read_csv(snakemake.input.freqs, index_col=0)

# Load coefficients
with open(snakemake.input.coeffs, 'r') as f:
    _coeffs = json.load(f)
coeffs = {int(k): v for k, v in _coeffs.items()}  # Convert str keys to int

# Make predictions
output_columns = imfs[signal].columns
index = pd.date_range(start=start_date, end=end_date, freq='D')
predictions = pd.DataFrame(index=index, columns=output_columns)

for component in output_columns:
    X = get_X(imfs, nearest_freq, signal, component, index)

    pred = np.sum([X[c] * coeffs[component][i]
                   for i, c in enumerate(X.columns)], axis=0)
    predictions.loc[:, component] = pred

# Save
predictions.to_csv(snakemake.output[0])
