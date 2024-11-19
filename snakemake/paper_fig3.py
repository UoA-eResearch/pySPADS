import pandas as pd
from collections import defaultdict
from pySPADS.processing.data import parse_filename, load_imf
from pySPADS.visualisation import paper

# Parameters
signal = snakemake.params.c["signal"]
noise = snakemake.wildcards.noise

if noise == "mean":
    # Mean over all noises
    imfs = defaultdict(dict)
    labels = set()
    for fname in snakemake.input.imfs:
        label, imf_noise = parse_filename(fname)
        imfs[imf_noise][label] = load_imf(fname)

        labels.add(label)
        assert (
            imf_noise in snakemake.params.c["noises"]
        ), f'Expected noise in {snakemake.params.c["noises"]} but got {noise}'

    # For each label, average over all noises
    imfs_mean = {}
    for label in labels:
        imfs_mean[label] = (
            pd.concat([imfs[noise][label] for noise in imfs]).groupby(level=0).mean()
        )

    f = paper.fig3(imfs_mean, signal, "1999-01-01", "2017-01-01")
    f.savefig(snakemake.output[0])

else:
    noise = float(noise)

    imfs = {}
    for fname in snakemake.input.imfs:
        label, imf_noise = parse_filename(fname)
        imfs[label] = load_imf(fname)

        assert imf_noise == noise, f"Expected noise {noise} but got {imf_noise}"

    f = paper.fig3(imfs, signal, "1999-01-01", "2017-01-01")
    f.savefig(snakemake.output[0])
