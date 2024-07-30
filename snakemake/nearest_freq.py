from pipeline.frequencies import match_frequencies
from processing.data import load_imf, parse_filename

# Parameters
noise = float(snakemake.wildcards.noise)
signal = snakemake.params.c['signal']
threshold = snakemake.params.c['frequency_threshold']
exclude_trend = snakemake.params.c.get('exclude_trend', False)

# Load imfs
imfs = {}
for fname in snakemake.input:
    label, imf_noise = parse_filename(fname)
    assert imf_noise == noise, f'Expected noise {noise} but got {imf_noise}'
    imfs[label] = load_imf(fname)

# Find nearest frequency
nearest_freq = match_frequencies(imfs, signal, threshold, exclude_trend)

# Save
nearest_freq.to_csv(snakemake.output[0])
