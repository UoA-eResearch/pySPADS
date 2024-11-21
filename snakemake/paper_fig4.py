import json
from pathlib import Path

import pandas as pd

from pySPADS.processing.data import load_data_from_csvs
from pySPADS.visualisation import paper

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Parameters
signal = _snakemake.params.c["signal"]
noises = _snakemake.params.c["noises"]
imf_folder = Path(_snakemake.input.folder).parent / "imfs"

with open(_snakemake.input.dates, "r") as f:
    dates = json.load(f)

# Load data
dfs = load_data_from_csvs(_snakemake.input.folder, _snakemake.params.c["time_col"])
total = pd.read_csv(_snakemake.input.predictions, index_col=0, parse_dates=True)

# DF -> Series
total = total[total.columns[0]]

f = paper.fig4(dfs[signal], total, "2000-01-01", "2021-12-30", dates["hindcast"])
f.savefig(_snakemake.output[0])
