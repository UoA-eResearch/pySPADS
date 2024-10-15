import json
from pathlib import Path

from snakemake.script import snakemake
import pandas as pd

from processing.data import load_data_from_csvs
from processing.dataclasses import TrendModel
from processing.trend import gen_trend
from visualisation import paper

# Parameters
signal = snakemake.params.c['signal']
noises = snakemake.params.c['noises']
exclude_trend = snakemake.params.c.get('exclude_trend', False)
imf_folder = Path(snakemake.input.folder).parent / 'imfs'

with open(snakemake.input.dates, 'r') as f:
    dates = json.load(f)

# Load data
dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])
total = pd.read_csv(snakemake.input.predictions, index_col=0, parse_dates=True)

# DF -> Series
total = total[total.columns[0]]

if exclude_trend:
    signal_trend = TrendModel.load(snakemake.input.trend)
    plot_signal = dfs[signal] + gen_trend(dfs[signal], signal_trend)
else:
    plot_signal = dfs[signal]

f = paper.fig4(plot_signal, total, '2000-01-01', '2021-12-30', dates['hindcast'])
f.savefig(snakemake.output[0])
