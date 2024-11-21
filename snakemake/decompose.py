import json

from pySPADS.pipeline import steps
from pySPADS.processing.data import load_data_from_csvs
from pySPADS.processing.dataclasses import TrendModel
from pySPADS.processing.trend import gen_trend

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Load data
dfs = load_data_from_csvs(_snakemake.input.folder, _snakemake.params.c["time_col"])

# Select date range (differs if hindcasting signal)
with open(_snakemake.input.dates, "r") as f:
    dates = json.load(f)

label = _snakemake.wildcards.label
signal = _snakemake.params.c["signal"]
reject_noise = _snakemake.params.c.get("reject_noise", False)
exclude_trend = _snakemake.params.c.get("exclude_trend", False)

if label == signal and not _snakemake.params.full:
    # TODO: handle both hindcast and forecast - make explicit what date range is considered
    df = dfs[label].loc[dates["start"] : dates["hindcast"]]
else:
    df = dfs[label].loc[dates["start"] : dates["end"]]

if label == signal and exclude_trend:
    signal_trend = TrendModel.load(_snakemake.input.trend)
    df -= gen_trend(df, signal_trend)

# Decompose
noise = float(_snakemake.wildcards.noise)
imf_df = steps.decompose(
    df, noise=noise, num_trials=100, progress=False, parallel=False
)

# Optionally reject modes that are mostly noise
if reject_noise:
    imf_df = steps.reject_noise(
        imf_df, noise_threshold=_snakemake.params.c["noise_threshold"]
    )

# Save result
imf_df.to_csv(_snakemake.output[0])
