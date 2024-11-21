import pandas as pd
import re

from pySPADS.pipeline import steps
from pySPADS.processing.dataclasses import TrendModel

# snakemake is not defined until runtime, so we need to disable the warning:
_snakemake = snakemake  # noqa: F821

# TODO: pass noise of each file from snakefile, rather than parsing
noise_pattern = re.compile("predictions_(\d+\.\d+).csv")


def parse_noise(fname):
    return float(noise_pattern.search(fname).group(1))


# Parameters
signal = _snakemake.params.c["signal"]
noises = _snakemake.params.c["noises"]

# Load predictions by noise
preds = {}
for fname in _snakemake.input.predictions:
    noise = parse_noise(fname)
    assert noise in noises, f"Expected noise in {noises} but got {noise}"
    preds[noise] = pd.read_csv(fname, index_col=0)

    preds[noise].index = pd.to_datetime(preds[noise].index)

# Combine predictions into single output signal
signal_trend = TrendModel.load(_snakemake.input.trend)
total = steps.combine_predictions(preds, signal_trend)

total.to_csv(_snakemake.output[0])
