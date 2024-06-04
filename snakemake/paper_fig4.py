import pandas as pd

from processing.data import load_data_from_csvs
from visualisation import paper

# Parameters
signal = snakemake.params.c['signal']

# Load data
dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])
total = pd.read_csv(snakemake.input.predictions, index_col=0, parse_dates=True)

# DF -> Series
total = total[total.columns[0]]

f = paper.fig4(dfs[signal], total, '2010-01-01', '2017-01-01')
f.savefig(snakemake.output[0])
