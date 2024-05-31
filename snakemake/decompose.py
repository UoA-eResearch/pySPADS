import json

from pipeline.decompose import decompose
from processing.data import load_data_from_csvs
from pathlib import Path

# Load data
dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])

# Select date range (differs if hindcasting signal)
with open(snakemake.input.dates, 'r') as f:
    dates = json.load(f)

label = snakemake.wildcards.label
signal = snakemake.params.c['signal']

if label == signal:
    # TODO: handle both hindcast and forecast - make explicit what date range is considered
    df = dfs[label].loc[dates['start']:dates['hindcast']]
else:
    df = dfs[label].loc[dates['start']:dates['end']]

# Decompose and save
noise = float(snakemake.wildcards.noise)
imf_df = decompose(df, noise=noise, num_trials=100, progress=False, parallel=False)
imf_df.to_csv(snakemake.output[0])