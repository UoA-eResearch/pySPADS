import json

from pipeline import steps
from processing.data import load_data_from_csvs

# Load data
dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])

# Select date range (differs if hindcasting signal)
with open(snakemake.input.dates, 'r') as f:
    dates = json.load(f)

label = snakemake.wildcards.label
signal = snakemake.params.c['signal']

if label == signal and not snakemake.params.full:
    # TODO: handle both hindcast and forecast - make explicit what date range is considered
    df = dfs[label].loc[dates['start']:dates['hindcast']]
else:
    df = dfs[label].loc[dates['start']:dates['end']]

# Decompose
noise = float(snakemake.wildcards.noise)
imf_df = steps.decompose(df, noise=noise, num_trials=100, progress=False, parallel=False)

# Optionally reject modes that are mostly noise
if snakemake.params.c['noise_threshold'] is not None:
    imf_df = steps.reject_noise(imf_df, noise_threshold=snakemake.params.c['noise_threshold'])

# Save result
imf_df.to_csv(snakemake.output[0])
