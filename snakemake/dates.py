import json
from processing.data import load_data_from_csvs

signal = snakemake.params.c['signal']
dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])
# Earliest possible start date, latest common end date
start_date = min([min(df.index) for df in dfs.values()])
end_date = min([max(df.index) for label, df in dfs.items() if label != signal])
# End of historic data to forecast from
forecast_date = max(dfs[signal].index)
# Hindcast date is a config
hindcast_date = snakemake.params.c['hindcast_date']

dates = {
    'start': start_date.strftime('%Y-%m-%d'),
    'end': end_date.strftime('%Y-%m-%d'),
    'forecast': forecast_date.strftime('%Y-%m-%d'),
    'hindcast': hindcast_date.strftime('%Y-%m-%d')
}

with open(snakemake.output[0], 'w') as f:
    json.dump(dates, f)
