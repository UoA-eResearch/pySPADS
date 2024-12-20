import json
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

from pySPADS.pipeline import steps
from pySPADS.processing.reconstruct import get_y
from pySPADS.processing.data import parse_filename, load_imf
from pySPADS.processing.dataclasses import LinRegCoefficients

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# Parameters
noise = float(_snakemake.wildcards.noise)
signal = _snakemake.params.c["signal"]
exclude_trend = _snakemake.params.c.get("exclude_trend", False)

# Load data
with open(_snakemake.input.dates, "r") as f:
    dates = json.load(f)

# TODO: should we be predicting from start-end or hindcast-end?
start_date = dates["start"]
end_date = dates["end"]

# Load imfs
imfs = {}
for fname in _snakemake.input.imfs:
    label, imf_noise = parse_filename(fname)
    assert imf_noise == noise, f"Expected noise {noise} but got {imf_noise}"
    imfs[label] = load_imf(fname)

# Load nearest frequency
nearest_freq = pd.read_csv(_snakemake.input.freqs, index_col=0)

# Load coefficients
coeffs = LinRegCoefficients.load(_snakemake.input.coeffs)

# Make predictions
component_predictions = steps.predict(
    imfs, nearest_freq, signal, coeffs, start_date, end_date, exclude_trend
)

# Plot predictions
for component in component_predictions.columns:
    # Plot prediction vs. signal for debugging
    plt.figure()
    y = get_y(imfs, signal, component, imfs[signal].index)
    plt.plot(y, label="signal")
    pred = component_predictions[component]
    plt.plot(pred.index, pred, label="prediction", alpha=0.5)
    plt.legend()
    plt.title(f"Noise {noise}, Component {component}")
    fname = (
        Path(_snakemake.output[0]).parent
        / "figures"
        / "predictions"
        / f"pred_{noise}_{component}.png"
    )
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)

# Save
component_predictions.to_csv(_snakemake.output[0])
