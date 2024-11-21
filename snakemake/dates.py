import json
from pySPADS.processing.data import load_data_from_csvs

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Load data
signal = _snakemake.params.c["signal"]
dfs = load_data_from_csvs(_snakemake.input.folder, _snakemake.params.c["time_col"])

# Earliest possible start date, latest common end date
start_date = min([min(df.index) for df in dfs.values()])
end_date = min([max(df.index) for label, df in dfs.items() if label != signal])

# End of historic data to forecast from
forecast_date = max(dfs[signal].index)

# Hindcast date is a config
hindcast_date = _snakemake.params.c["hindcast_date"]

dates = {
    "start": start_date.strftime("%Y-%m-%d"),
    "end": end_date.strftime("%Y-%m-%d"),
    "forecast": forecast_date.strftime("%Y-%m-%d"),
    "hindcast": hindcast_date.strftime("%Y-%m-%d"),
}

with open(_snakemake.output[0], "w") as f:
    json.dump(dates, f)
