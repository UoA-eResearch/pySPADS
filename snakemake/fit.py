import pandas as pd

from pySPADS.pipeline import steps
from pySPADS.processing.data import parse_filename, load_imf
from pySPADS.visualisation.paper import fig_si3

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Parameters
noise = float(_snakemake.wildcards.noise)
signal = _snakemake.params.c["signal"]
exclude_trend = _snakemake.params.c.get("exclude_trend", False)
model = _snakemake.wildcards.model

fit_intercept = "_intercept" in model
if fit_intercept:
    model = model.replace("_intercept", "")

# Load imfs
imfs = {}
for fname in _snakemake.input.imfs:
    label, imf_noise = parse_filename(fname)
    assert imf_noise == noise, f"Expected noise {noise} but got {imf_noise}"
    imfs[label] = load_imf(fname)

# Load nearest frequency
nearest_freq = pd.read_csv(_snakemake.input.freqs, index_col=0)

# Linear regression
coeffs = steps.fit(
    imfs,
    nearest_freq,
    signal,
    model=model,
    fit_intercept=fit_intercept,
    normalize=False,
)

# Save
coeffs.save(_snakemake.output.coeffs)

# Debug figure
f = fig_si3(
    imfs,
    nearest_freq,
    signal,
    coeffs,
    annotate_coeffs=True,
    exclude_trend=exclude_trend,
)
f.savefig(_snakemake.output.figure)
