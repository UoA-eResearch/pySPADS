import pandas as pd
import re

# TODO: pass noise of each file from snakefile, rather than parsing
noise_pattern = re.compile('predictions_(\d+\.\d+).csv')
def parse_noise(fname):
    return float(noise_pattern.search(fname).group(1))

# Parameters
signal = snakemake.params.c['signal']
noises = snakemake.params.c['noises']

# Load predictions by noise
preds = {}
for fname in snakemake.input:
    noise = parse_noise(fname)
    assert noise in noises, f'Expected noise in {noises} but got {noise}'
    preds[noise] = pd.read_csv(fname, index_col=0)

# Combine - from a column for each imf mode to a single column for each noise
pred_by_noise = {
    noise: preds[noise].sum(axis=1)
    for noise in noises
}

# Combine - averaging across noises
total = sum(list(pred_by_noise.values())) / len(pred_by_noise)
total.to_csv(snakemake.output[0])
