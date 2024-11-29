from collections import defaultdict

import click
import pathlib

import pandas as pd
from tqdm import tqdm

from pySPADS.pipeline import steps
from pySPADS.processing.data import load_imfs, load_data_from_csvs, imf_filename
from pySPADS.processing.dataclasses import TrendModel
from pySPADS.util.click import OptionNargs, parse_noise_args
from . import __version__


@click.group()
@click.version_option(__version__)
def cli():
    pass


@cli.command()
@click.argument("files", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=pathlib.Path), nargs=-1)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Output directory. Defaults to current working directory. It is highly recommended that this is a separate directory to that containing the input files",
)
@click.option("--timecol", type=str, default="t", help="Column name of time index")
@click.option(
    "-n",
    "--noise",
    cls=OptionNargs,
    type=tuple[float],
    callback=parse_noise_args,
    help="Noise values to use when decomposing IMFs, e.g. -n 0.1 0.2 0.3",
)
@click.option(
    "--noise-threshold",
    type=float,
    default=None,
    help="Threshold for rejecting IMF modes containing noise. If omitted, no modes will be rejected",
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite existing IMF files in output directory"
)
def decompose(
    files, output, timecol, noise, noise_threshold, overwrite
):
    """
    Decompose input data into IMFs

    FILES expects either a single .csv, a list of .csvs or a directory containing .csv files.
    Files containing more than one timeseries will result in multiple separate output files.

    Resulting files will be named <column_name>_imf_<noise>.csv

    e.g.: if an input file contains columns 'a' and 'b', and noise values 0.1 and 0.2 are specified,
    the output files will be: a_imf_0.1.csv, a_imf_0.2.csv, b_imf_0.1.csv, b_imf_0.2.csv

    Timeseries will be decomposed for the full time range available. If this is not what you intend (e.g.: if
    performing a hindcast where you are training against only part of your signal data, and will use the rest for
    validation), you should ensure that the input files provided are contain the correct subset of data.
    """
    # Load data
    print(f"Loading data from {', '.join([str(f) for f in files])}")
    dfs = {}
    for file in files:
        loaded = load_data_from_csvs(file, timecol)
        for key in loaded:
            assert key not in dfs, f"Duplicate column {key} found in input from file {file}"
        dfs.update(loaded)

    print(
        f'Found {len(dfs)} timeseries in input data, with columns: {", ".join(dfs.keys())}'
    )

    # Output folder
    if output is None:
        output = pathlib.Path.cwd()
    output.mkdir(parents=True, exist_ok=True)

    # Check if files are in output directory
    if any([output == file.parent for file in files if file.is_file()]) \
            or any([output == file for file in files if file.is_dir()]):
        print("WARNING: Some or all input files are in the output directory, this may lead to confusing file names.")

    # Decompose each timeseries and save result
    for col in tqdm(dfs, desc="Decomposing IMFs"):
        for ns in tqdm(noise, desc=f"Decomposing {col}", leave=False):
            filename = imf_filename(output, col, ns)
            if not overwrite and filename.exists():
                tqdm.write(f"IMFs for {col} with noise {ns} already exist, skipping")
                continue
            imf_dfs = steps.decompose(
                dfs[col], noise=ns, num_trials=100, progress=False
            )
            # Optionally reject modes that are mostly noise
            if noise_threshold is not None:
                imf_dfs = steps.reject_noise(imf_dfs, noise_threshold=noise_threshold)

            imf_dfs.to_csv(filename)


@cli.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Output directory",
)
@click.option("-s", "--signal", type=str, help="Column name of signal to fit to")
@click.option(
    "--frequency-threshold",
    type=float,
    default=0.25,
    help="Threshold for accepting IMF modes with similar frequencies to signal frequency",
)
def match(output, signal, reject_noise, noise_threshold, frequency_threshold):
    """Match IMFs to each other"""
    imfs = load_imfs(output / "imfs")

    # Re-organise imfs into dict[noise][label]
    imfs_by_noise = defaultdict(dict)
    for label, noise in imfs.keys():
        imfs_by_noise[noise][label] = imfs[(label, noise)]

    # Match frequencies
    print("Matching frequencies")
    for noise in imfs_by_noise:
        print(f"Noise: {noise}")
        nearest_freq = steps.match_frequencies(
            imfs_by_noise[noise], signal, frequency_threshold
        )
        nearest_freq.to_csv(output / f"frequencies_{noise}.csv")


@cli.command()
@click.option(
    "-w",
    "--working-dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Working directory, defaults to current directory",
)
@click.option("-s", "--signal", type=str, help="Column name of signal to fit to")
def fit(working_dir, signal):
    """Fit IMFs to signal"""
    if working_dir is None:
        working_dir = pathlib.Path.cwd()

    imfs = load_imfs(working_dir / "imfs")




@cli.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Output directory",
)
@click.option("-s", "--signal", type=str, help="Column name of signal to fit to")
def reconstruct(output, signal):
    """Reconstruct signal from IMFs"""
    #
    imfs = load_imfs(output / "imfs")

    imfs_by_noise = defaultdict(dict)
    for label, noise in imfs.keys():
        imfs_by_noise[noise][label] = imfs[(label, noise)]

    nearest_freq = {
        noise: pd.read_csv(output / f"frequencies_{noise}.csv", index_col=0)
        for noise in imfs_by_noise
    }

    coefs = {
        noise: steps.fit(
            imfs_by_noise[noise],
            nearest_freq[noise],
            signal,
            model="mreg2",
            fit_intercept=True,
            normalize=False,
        )
        for noise in imfs_by_noise
    }

    # Reconstruct
    hindcast = {}
    start_date = min([min(df.index) for df in imfs.values()])
    end_date = min([max(df.index) for df in imfs.values()])

    for noise in imfs_by_noise:
        comp_pred = steps.predict(
            imfs_by_noise[noise],
            nearest_freq[noise],
            signal,
            coefs[noise],
            start_date,
            end_date,
        )

        hindcast[noise] = comp_pred
        comp_pred.to_csv(output / f"predictions_{noise}.csv")

    # Reconstructed signal for each noise value
    total = steps.combine_predictions(
        hindcast, trend=TrendModel()
    )  # TODO: implement detrending in CLI
    total.to_csv(output / "reconstructed_total.csv")
