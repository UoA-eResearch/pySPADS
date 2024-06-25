import json
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

from pipeline.reconstruct import get_X, get_y
from processing.data import parse_filename, load_imf
from processing.dataclasses import LinRegCoefficients

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
coeffs = LinRegCoefficients.load(snakemake.input.coeffs)

# Make predictions
output_columns = imfs[signal].columns
index = pd.date_range(start=start_date, end=end_date, freq='D')
predictions = pd.DataFrame(index=index, columns=output_columns)

for component in output_columns:
    X = get_X(imfs, nearest_freq, signal, component, index)

    pred = coeffs.predict(component, X)
    # pred = np.sum([X[driver] * coeffs.coeffs[component][driver]
    #                for driver in X.columns], axis=0)
    predictions.loc[:, component] = pred

    # Plot prediction vs. signal for debugging
    plt.figure()
    y = get_y(imfs, signal, component, imfs[signal].index)
    plt.plot(y, label='signal')
    plt.plot(index, pred, label='prediction', alpha=0.5)
    plt.legend()
    plt.title(f'Noise {noise}, Component {component}')
    fname = Path(snakemake.output[0]).parent / 'figures' / 'predictions' / f'pred_{noise}_{component}.png'
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)

# Save
predictions.to_csv(snakemake.output[0])
