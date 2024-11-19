from pySPADS.processing.data import parse_filename, load_imf
from pySPADS.visualisation.imfs import plot_imfs

fname = snakemake.input[0]
label, imf_noise = parse_filename(fname)
imf = load_imf(fname)

plot_imfs(imf.to_numpy().T, label, snakemake.output[0])
