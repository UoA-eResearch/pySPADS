import pandas as pd

from pySPADS.processing.data import load_data_from_csvs, parse_filename, load_imf
from pySPADS.visualisation import paper

dfs = load_data_from_csvs(snakemake.input.folder, snakemake.params.c['time_col'])
signal = snakemake.params.c['signal']

# We can get the filenames from the snakemake rule, or construct them
imfs = {}
for fname in snakemake.input.signal_imfs:
    label, noise = parse_filename(fname)
    imfs[noise] = load_imf(fname)

    assert label == snakemake.params.c['signal'], f'Expected signal {signal} but got {label}'
    assert noise in snakemake.params.c['noises'], f'Expected noise in {snakemake.params.c["noises"]} but got {noise}'

# Either output the figure for a single given noise, or averaged over all noises
if snakemake.wildcards.noise == 'mean':
    # Mean imf over noises (note that this may not work well if the modes in each represent different frequencies)
    combined = (
        pd.concat(imfs.values())
        .groupby(level=0)
        .mean()
    )

    # TODO - this date range requires the full signal to be decomposed, not just the start-hindcast period
    f = paper.fig2(dfs[signal], combined, '1999-01-01', '2017-01-01')
    f.savefig(snakemake.output[0])
else:
    noise = float(snakemake.wildcards.noise)
    f = paper.fig2(dfs[signal], imfs[noise], '1999-01-01', '2017-01-01')
    f.savefig(snakemake.output[0])
