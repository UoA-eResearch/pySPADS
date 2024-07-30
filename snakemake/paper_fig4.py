import json

import pandas as pd

from processing.data import load_data_from_csvs, imf_filename, load_imf
from visualisation import paper

# Parameters
signal = snakemake.params.c['signal']
noises = snakemake.params.c['noises']
exclude_trend = snakemake.params.c.get('exclude_trend', False)
imf_folder = snakemake.input.imf_folder

with open(snakemake.input.dates, 'r') as f:
    dates = json.load(f)

# Load data
dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])
total = pd.read_csv(snakemake.input.predictions, index_col=0, parse_dates=True)

# DF -> Series
total = total[total.columns[0]]

if exclude_trend:
    # Detrend input signal for plotting
    filename = imf_filename(imf_folder, f'{signal}_full', min(noises))
    signal_full = load_imf(filename)
    plot_signal = dfs[signal] - signal_full.iloc[:, -1]
else:
    plot_signal = dfs[signal]

f = paper.fig4(plot_signal, total, '2000-01-01', '2021-12-30', dates['hindcast'])
f.savefig(snakemake.output[0])
    