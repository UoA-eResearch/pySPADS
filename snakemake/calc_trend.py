import json

from processing.data import load_data_from_csvs
from processing.dataclasses import TrendModel
from processing.trend import detect_trend

# Load data
dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])

with open(snakemake.input.dates, 'r') as f:
    dates = json.load(f)

signal = snakemake.params.c['signal']
exclude_trend = snakemake.params.c.get('exclude_trend', False)

# If exlcuding trend, calculate it, otherwise use default no-op trend
if exclude_trend:
    d = dfs[signal].loc[dates['start']:dates['hindcast']]
    signal_trend = detect_trend(d)
else:
    signal_trend = TrendModel()

signal_trend.save(snakemake.output.trend)
