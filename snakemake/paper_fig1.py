from pySPADS.processing.data import load_data_from_csvs
from pySPADS.visualisation import paper

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Load data
dfs = load_data_from_csvs(_snakemake.input.folder, _snakemake.params.c["time_col"])

# Plot
pc0 = dfs["PC0"] if "PC0" in dfs else None
f = paper.fig1(pc0, dfs["Hs"], dfs["Tp"], dfs["Dir"], "2004-01-01", "2016-01-01")

# Save
f.savefig(_snakemake.output[0])
