import json

from pySPADS.processing.data import load_data_from_csvs
from pySPADS.processing.dataclasses import TrendModel
from pySPADS.processing.trend import detect_trend

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Load data
dfs = load_data_from_csvs(_snakemake.input.folder, _snakemake.params.c["time_col"])

with open(_snakemake.input.dates, "r") as f:
    dates = json.load(f)

signal = _snakemake.params.c["signal"]
exclude_trend = _snakemake.params.c.get("exclude_trend", False)

# If exlcuding trend, calculate it, otherwise use default no-op trend
if exclude_trend:
    d = dfs[signal].loc[dates["start"] : dates["hindcast"]]
    signal_trend = detect_trend(d)
else:
    signal_trend = TrendModel()

signal_trend.save(_snakemake.output[0])
