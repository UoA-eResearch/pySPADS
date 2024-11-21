from pySPADS.pipeline import steps
from pySPADS.processing.data import load_imf, parse_filename

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Parameters
noise = float(_snakemake.wildcards.noise)
signal = _snakemake.params.c["signal"]
threshold = _snakemake.params.c["frequency_threshold"]
exclude_trend = _snakemake.params.c.get("exclude_trend", False)

# Load imfs
imfs = {}
for fname in _snakemake.input:
    label, imf_noise = parse_filename(fname)
    assert imf_noise == noise, f"Expected noise {noise} but got {imf_noise}"
    imfs[label] = load_imf(fname)

# Find nearest frequency
nearest_freq = steps.match_frequencies(imfs, signal, threshold, exclude_trend)

# Save
nearest_freq.to_csv(_snakemake.output[0])
