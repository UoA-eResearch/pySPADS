import pandas as pd
import re

from pySPADS.pipeline import steps
from pySPADS.processing.dataclasses import TrendModel

# TODO: pass noise of each file from snakefile, rather than parsing
noise_pattern = re.compile("predictions_(\d+\.\d+).csv")


def parse_noise(fname):
    return float(noise_pattern.search(fname).group(1))


# Parameters
signal = snakemake.params.c["signal"]
noises = snakemake.params.c["noises"]

# Load predictions by noise
preds = {}
for fname in snakemake.input.predictions:
    noise = parse_noise(fname)
    assert noise in noises, f"Expected noise in {noises} but got {noise}"
    preds[noise] = pd.read_csv(fname, index_col=0)

    preds[noise].index = pd.to_datetime(preds[noise].index)

# Combine predictions into single output signal
signal_trend = TrendModel.load(snakemake.input.trend)
total = steps.combine_predictions(preds, signal_trend)

total.to_csv(snakemake.output[0])
