from processing.data import load_data_from_csvs
from visualisation import paper
from pathlib import Path

dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])

f = paper.fig1(dfs['PC0'], dfs['Hs'], dfs['Tp'], dfs['Dir'], '2004-01-01', '2016-01-01')
f.savefig(snakemake.output[0])
