from pySPADS.processing.data import parse_filename, load_imf
from pySPADS.visualisation.imfs import plot_imfs

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Load imf
fname = _snakemake.input[0]
label, imf_noise = parse_filename(fname)
imf = load_imf(fname)

# Plot and save
plot_imfs(imf.to_numpy().T, label, _snakemake.output[0])
