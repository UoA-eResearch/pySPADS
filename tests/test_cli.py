import shutil
from pathlib import Path

import pandas as pd
from click.testing import CliRunner
from tests.common import dpath

from pySPADS.cli import cli
from pySPADS.processing.dataclasses import LinRegCoefficients


def _copy_input_data(tmp_path, source_path, dest_path):
    """Copy input data to a temporary folder"""
    input_folder = Path(tmp_path) / dest_path
    input_folder.mkdir(parents=True)
    for file in source_path.glob("*"):
        shutil.copy(file, input_folder / file.name)
    return input_folder


def _column_names(input_dir, time_col):
    """Get the column names from the input data"""
    output = []
    for file in input_dir.glob("*.csv"):
        df = pd.read_csv(file)

        for col in df.columns:
            if col != time_col:
                output.append(col)

    return output


def test_decompose(tmp_path, mocker):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        # Copy input data
        input_folder = _copy_input_data(td, dpath("data/separate_files"), "input")
        expected_series = _column_names(input_folder, "t")

        # Mock decompose function
        mock_decompose = mocker.patch("pySPADS.pipeline.steps.decompose")
        mock_decompose.return_value = pd.DataFrame()

        # Mock reject noise function
        mock_reject_noise = mocker.patch("pySPADS.pipeline.steps.reject_noise")
        mock_reject_noise.return_value = pd.DataFrame()

        # Run the command
        noises = [0.1, 0.2, 0.3]
        result = runner.invoke(
            cli,
            [
                "decompose",
                str(input_folder),
                "-n",
                *[str(n) for n in noises],
            ],
        )

        # Check the output
        assert result.exit_code == 0
        assert mock_decompose.call_count == len(expected_series) * len(
            noises
        ), "decompose should be called for each series and noise"
        assert (
            mock_reject_noise.call_count == 0
        ), "reject_noise should not be called unless option passed"

        output_dir = Path(td) / "imfs"
        assert output_dir.exists()

        # Check that the output files were created
        for s in expected_series:
            for n in noises:
                assert (
                    output_dir / f"{s}_imf_{n}.csv"
                ).exists(), f"Expected series {s} to exist"


def test_match_frequencies(tmp_path, mocker):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        # Copy input data
        _copy_input_data(td, dpath("data/imfs"), "imfs")

        # Mock nearest_freq function
        mock_nearest_freq = mocker.patch("pySPADS.pipeline.steps.match_frequencies")
        mock_nearest_freq.return_value = pd.DataFrame()

        # Run the command
        result = runner.invoke(
            cli,
            [
                "match",
                "--signal",
                "shore",
            ],
        )

        # Check the output
        assert result.exit_code == 0
        assert (
            mock_nearest_freq.call_count == 1
        ), "match_frequencies should be called once"  # test data contains 1 noise level

        output_file = Path(td) / "frequencies" / "frequencies_0.1.csv"
        assert output_file.exists(), "Expected output file to exist"


def test_fit(tmp_path, mocker):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        # Copy input data
        _copy_input_data(td, dpath("data/imfs"), "imfs")
        _copy_input_data(td, dpath("data/frequencies"), "frequencies")

        # Mock fit function
        mock_fit = mocker.patch("pySPADS.pipeline.steps.fit")
        mock_fit.return_value = LinRegCoefficients(coeffs={1: {"PC0": 1}})

        # Run the command
        result = runner.invoke(
            cli,
            [
                "fit",
                "--signal",
                "shore",
            ],
        )

        # Check the output
        assert result.exit_code == 0
        assert mock_fit.call_count == 1, "fit should be called once"

        output_file = Path(td) / "coefficients" / "coefficients_0.1.csv"
        assert output_file.exists(), "Expected output file to exist"


def test_predict(tmp_path, mocker):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        # Copy input data
        _copy_input_data(td, dpath("data/imfs"), "imfs")
        _copy_input_data(td, dpath("data/frequencies"), "frequencies")
        _copy_input_data(td, dpath("data/coefficients"), "coefficients")

        # Mock predict function
        mock_predict = mocker.patch("pySPADS.pipeline.steps.predict")
        mock_predict.return_value = pd.DataFrame()

        # Run the command
        result = runner.invoke(
            cli,
            [
                "reconstruct",
                "--signal",
                "shore",
            ],
        )

        # Check the output
        assert result.exit_code == 0
        assert mock_predict.call_count == 1, "predict should be called once"

        output_file = Path(td) / "predictions_0.1.csv"
        assert output_file.exists(), "Expected per noise output file to exist"

        reconstructed_file = Path(td) / "reconstructed_total.csv"
        assert (
            reconstructed_file.exists()
        ), "Expected reconstructed signal output file to exist"
