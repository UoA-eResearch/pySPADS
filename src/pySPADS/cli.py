from collections import defaultdict

import click
import pathlib

import pandas as pd
from tqdm import tqdm

from pySPADS.pipeline import steps
from pySPADS.processing.data import load_imfs, load_data_from_csvs, imf_filename
from pySPADS.processing.dataclasses import TrendModel, LinRegCoefficients
from pySPADS.util.click import OptionNargs, parse_noise_args
from . import __version__


@click.group()
@click.version_option(__version__)
def cli():
    pass


@cli.command()
@click.argument(
    "files",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=pathlib.Path),
    nargs=-1,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Output directory. Defaults to ./imfs",
)
@click.option("--timecol", type=str, default="t", help="Column name of time index")
@click.option(
    "-n",
    "--noise",
    cls=OptionNargs,
    type=tuple[float],
    callback=parse_noise_args(default=(0.25,)),
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
def decompose(files, output, timecol, noise, noise_threshold, overwrite):
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
            assert (
                key not in dfs
            ), f"Duplicate column {key} found in input from file {file}"
        dfs.update(loaded)

    print(
        f'Found {len(dfs)} timeseries in input data, with columns: {", ".join(dfs.keys())}'
    )

    # Output folder
    if output is None:
        output = pathlib.Path.cwd() / "imfs"
    output.mkdir(parents=True, exist_ok=True)

    # Check if files are in output directory
    if any([output == file.parent for file in files if file.is_file()]) or any(
        [output == file for file in files if file.is_dir()]
    ):
        print(
            "WARNING: Some or all input files are in the output directory, this may lead to confusing file names."
        )

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
    "-i",
    "--imf_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Directory containing the IMF files, defaults to ./imfs",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Output directory, defaults ./frequencies",
)
@click.option(
    "-s", "--signal", type=str, help="Column name of signal to fit to", required=True
)
@click.option(
    "--frequency-threshold",
    type=float,
    default=0.25,
    help="Threshold for accepting IMF modes with similar frequencies to signal frequency, default=0.25",
)
def match(imf_dir, output, signal, frequency_threshold):
    """
    Matches each component mode of the signal to component modes of each driver with similar frequencies

    IMF_DIR is the directory containing the IMF files, which should be named <column_name>_imf_<noise>.csv
            if omitted, this defaults to ./imfs

    The output will be a CSV file for each noise value, named <output>/frequencies_<noise>.csv
    """
    if imf_dir is None:
        imf_dir = pathlib.Path.cwd() / "imfs"
        assert imf_dir.exists(), f"IMF directory {imf_dir} does not exist"

    # Load IMFs
    imfs = load_imfs(imf_dir)

    # Output folder
    if output is None:
        output = pathlib.Path.cwd() / "frequencies"
    output.mkdir(parents=True, exist_ok=True)

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
    "-i",
    "--imf_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Directory containing the IMF files, defaults to ./imfs",
)
@click.option(
    "-f",
    "--frequency_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Directory containing the frequency files, defaults to ./frequencies",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Output directory, defaults to ./coefficients",
)
@click.option(
    "-s", "--signal", type=str, help="Column name of signal to fit to", required=True
)
@click.option(
    "-n",
    "--noises",
    cls=OptionNargs,
    type=tuple[float],
    callback=parse_noise_args(default=None),
    help="Noise values to use when fitting IMFs, defaults to all noises present in IMF_DIR, e.g. -n 0.1 0.2 0.3",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["mreg2", "linreg", "ridge"]),
    default="mreg2",
    help="Model to use for fitting linear regression, one of mreg2, linreg, ridge",
)
@click.option(
    "--fit-intercept", is_flag=True, help="Fit intercept in linear regression model"
)
@click.option(
    "--normalize", is_flag=True, help="Normalize input data in linear regression model"
)
def fit(
    imf_dir, frequency_dir, output, signal, noises, model, fit_intercept, normalize
):
    """
    Fit a linear model expressing each component of the signal as a linear combination of the components of the drivers

    Results will be saved as a JSON file for each noise value, named <output>/coefficients_<noise>.json

    IMF_DIR is the directory containing the IMF files, which should be named <column_name>_imf_<noise>.csv
            if omitted, this defaults to ./imfs
    FREQUENCY_DIR is the directory containing the frequency files, which should be named frequencies_<noise>.csv
            if omitted, this defaults to ./frequencies
    """
    # Input directories
    if imf_dir is None:
        imf_dir = pathlib.Path.cwd() / "imfs"
        assert imf_dir.exists(), f"IMF directory {imf_dir} does not exist"

    if frequency_dir is None:
        frequency_dir = pathlib.Path.cwd() / "frequencies"
        assert (
            frequency_dir.exists()
        ), f"Frequency directory {frequency_dir} does not exist"

    # Load IMFs
    imfs = load_imfs(imf_dir)

    imfs_by_noise = defaultdict(dict)
    for label, noise in imfs.keys():
        imfs_by_noise[noise][label] = imfs[(label, noise)]

    # Noises to process
    if noises is None:
        noises = list(imfs_by_noise.keys())

    # Load nearest frequencies
    nearest_freq = {
        noise: pd.read_csv(frequency_dir / f"frequencies_{noise}.csv", index_col=0)
        for noise in noises
    }

    # Fit linear models
    coefs = {
        noise: steps.fit(
            imfs_by_noise[noise],
            nearest_freq[noise],
            signal,
            model=model,
            fit_intercept=fit_intercept,
            normalize=normalize,
        )
        for noise in noises
    }

    # Save coefficients
    if output is None:
        output = pathlib.Path.cwd() / "coefficients"
    output.mkdir(parents=True, exist_ok=True)

    for noise in noises:
        coefs[noise].save(output / f"coefficients_{noise}.csv")


@cli.command()
@click.option(
    "-i",
    "--imf_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Directory containing the IMF files, defaults to ./imfs",
)
@click.option(
    "-f",
    "--frequency_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Directory containing the frequency files, defaults to ./frequencies",
)
@click.option(
    "-c",
    "--coeff_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Directory containing the coefficient files, defaults to ./coefficients",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help="Output directory, defaults to current directory",
)
@click.option(
    "-s", "--signal", type=str, help="Column name of signal to fit to", required=True
)
@click.option(
    "-n",
    "--noises",
    cls=OptionNargs,
    type=tuple[float],
    callback=parse_noise_args(default=None),
    help="Noise values to use when fitting IMFs, defaults to all noises present in IMF_DIR, e.g. -n 0.1 0.2 0.3",
)
def reconstruct(imf_dir, frequency_dir, coeff_dir, output, signal, noises):
    """
    Reconstruct signal from IMFs using linear regression coefficients

    Results will be saved as a CSV file for each noise value, named <output>/predictions_<noise>.csv
     and a CSV file for the total signal, named <output>/reconstructed_total.csv

    IMF_DIR is the directory containing the IMF files, which should be named <column_name>_imf_<noise>.csv
            if omitted, this defaults to ./imfs
    FREQUENCY_DIR is the directory containing the frequency files, which should be named frequencies_<noise>.csv
            if omitted, this defaults to ./frequencies
    COEFF_DIR is the directory containing the coefficient files, which should be named coefficients_<noise>.csv
            if omitted, this defaults to ./coefficients
    """
    # Input directories
    if imf_dir is None:
        imf_dir = pathlib.Path.cwd() / "imfs"
        assert imf_dir.exists(), f"IMF directory {imf_dir} does not exist"

    if frequency_dir is None:
        frequency_dir = pathlib.Path.cwd() / "frequencies"
        assert (
            frequency_dir.exists()
        ), f"Frequency directory {frequency_dir} does not exist"

    if coeff_dir is None:
        coeff_dir = pathlib.Path.cwd() / "coefficients"
        assert coeff_dir.exists(), f"Coefficient directory {coeff_dir} does not exist"

    # Load IMFs
    imfs = load_imfs(imf_dir)

    imfs_by_noise = defaultdict(dict)
    for label, noise in imfs.keys():
        imfs_by_noise[noise][label] = imfs[(label, noise)]

    # Noises to process
    if noises is None:
        noises = list(imfs_by_noise.keys())

    # Load coefficients
    coefs = {
        noise: LinRegCoefficients.load(coeff_dir / f"coefficients_{noise}.json")
        for noise in noises
    }

    # Load nearest frequencies
    nearest_freq = {
        noise: pd.read_csv(frequency_dir / f"frequencies_{noise}.csv", index_col=0)
        for noise in noises
    }

    # Ouput directory
    if output is None:
        output = pathlib.Path.cwd()

    output.mkdir(parents=True, exist_ok=True)

    # Reconstruct
    hindcast = {}
    start_date = min([min(df.index) for df in imfs.values()])
    end_date = min([max(df.index) for df in imfs.values()])

    for noise in noises:
        comp_pred = steps.predict(
            imfs_by_noise[noise],
            nearest_freq[noise],
            signal,
            coefs[noise],
            start_date,
            end_date,
        )

        hindcast[noise] = comp_pred
        # Save prediction for each noise value
        comp_pred.to_csv(output / f"predictions_{noise}.csv")

    # Reconstruct total signal
    total = steps.combine_predictions(
        hindcast, trend=TrendModel()
    )  # TODO: implement detrending in CLI

    # Save total signal
    total.to_csv(output / "reconstructed_total.csv")
