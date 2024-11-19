import pandas as pd

from pySPADS.pipeline import steps
from pySPADS.processing.data import parse_filename, load_imf
from pySPADS.visualisation.paper import fig_si3

# Parameters
noise = float(snakemake.wildcards.noise)
signal = snakemake.params.c["signal"]
exclude_trend = snakemake.params.c.get("exclude_trend", False)

model = snakemake.wildcards.model
fit_intercept = "_intercept" in model
if fit_intercept:
    model = model.replace("_intercept", "")

# Load imfs
imfs = {}
for fname in snakemake.input.imfs:
    label, imf_noise = parse_filename(fname)
    assert imf_noise == noise, f"Expected noise {noise} but got {imf_noise}"
    imfs[label] = load_imf(fname)

# Load nearest frequency
nearest_freq = pd.read_csv(snakemake.input.freqs, index_col=0)

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
coeffs.save(snakemake.output.coeffs)

# Debug figure
# f = fit_plot_norm(imfs, nearest_freq, signal)
f = fig_si3(
    imfs,
    nearest_freq,
    signal,
    coeffs,
    annotate_coeffs=True,
    exclude_trend=exclude_trend,
)
f.savefig(snakemake.output.figure)
