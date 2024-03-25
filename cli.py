from collections import defaultdict

import click
import pathlib

import numpy as np
import pandas as pd
from click.testing import CliRunner
from tqdm import tqdm

from pipeline.decompose import decompose as _decompose
from pipeline.frequencies import match_frequencies
from processing.data import load_imfs, load_data_from_csvs, imf_filename
from pipeline.reconstruct import fit, get_X, hindcast_index
from util.click import OptionNargs


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
def run(path):
    """Run the full pipeline"""
    print("Not implemented")


def _parse_noise(ctx, param, value):
    if value is None:
        return (.25,)
    return tuple(float(n) for n in value)


@cli.command()
@click.option('-i', '--input',
              type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=pathlib.Path),
              help='Input data file or directory, expects either a csv, or a directory of csvs',
              required=True)
@click.option('-o', '--output',
              type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
              help='Output directory')
@click.option('-s', '--signal', type=str, help='Column name of signal to fit to')
@click.option('--timecol', type=str, default='t', help='Column name of time index')
@click.option('-n', '--noise', cls=OptionNargs, type=tuple[float], callback=_parse_noise,
              help='Noise values use when decomposing IMFs')
@click.option('--overwrite', is_flag=True, help='Overwrite existing IMF files in output directory')
def decompose(input, output, signal, timecol, noise, overwrite):
    """Decompose input data into IMFs"""
    # Load data
    print(f'Loading data from {input}')
    dfs = load_data_from_csvs(input, timecol)

    assert signal in dfs, f'Column {signal} not found in input data'
    print(f'Found {len(dfs)} timeseries in input data, with columns: {", ".join(dfs.keys())}')

    # Decompose each timeseries and save result
    imf_dir = output / 'imfs'
    imf_dir.mkdir(parents=True, exist_ok=True)
    for col in tqdm(dfs, desc='Decomposing IMFs'):
        for ns in tqdm(noise, desc=f'Decomposing {col}', leave=False):
            filename = imf_filename(imf_dir, col, ns)
            if not overwrite and filename.exists():
                continue
            imf_dfs = _decompose(dfs[col], noise=ns, num_trials=100, progress=False)
            imf_dfs.to_csv(filename)


@cli.command()
@click.option('-o', '--output',
              type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
              help='Output directory')
@click.option('-s', '--signal', type=str, help='Column name of signal to fit to')
@click.option('--reject-noise', is_flag=True, help='Reject IMF modes containing mostly noise')
@click.option('--noise-threshold', type=float, default=0.95, help='Threshold for rejecting IMF modes containing noise')
@click.option('--frequency-threshold', type=float, default=0.25,
              help='Threshold for accepting IMF modes with similar frequencies to signal frequency')
def match(output, signal, reject_noise, noise_threshold, frequency_threshold):
    """Match IMFs to each other"""
    imfs = load_imfs(output / 'imfs')

    # Re-organise imfs into dict[noise][label]
    imfs_by_noise = defaultdict(dict)
    for label, noise in imfs.keys():
        imfs_by_noise[noise][label] = imfs[(label, noise)]

    # Match frequencies
    print('Matching frequencies')
    for noise in imfs_by_noise:
        print(f'Noise: {noise}')
        nearest_freq = match_frequencies(imfs_by_noise[noise], signal, frequency_threshold)
        nearest_freq.to_csv(output / f'frequencies_{noise}.csv')


@cli.command()
@click.option('-o', '--output',
              type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
              help='Output directory')
@click.option('-s', '--signal', type=str, help='Column name of signal to fit to')
def reconstruct(output, signal):
    """Reconstruct signal from IMFs"""
    #
    imfs = load_imfs(output / 'imfs')

    imfs_by_noise = defaultdict(dict)
    for label, noise in imfs.keys():
        imfs_by_noise[noise][label] = imfs[(label, noise)]

    nearest_freq = {
        noise: pd.read_csv(output / f'frequencies_{noise}.csv', index_col=0)
        for noise in imfs_by_noise
    }

    coefs = {
        noise: fit(imfs_by_noise[noise], nearest_freq[noise], signal)
        for noise in imfs_by_noise
    }

    # Reconstruct
    hindcast = defaultdict(dict)
    hindcast_df = {}
    for noise in imfs_by_noise:
        output_columns = imfs_by_noise[noise][signal].columns
        index = hindcast_index(imfs_by_noise[noise], signal)
        predictions = pd.DataFrame(index=index, columns=output_columns)
        for component in imfs_by_noise[noise][signal].columns:
            X = get_X(imfs_by_noise[noise], nearest_freq[noise], signal, component, index)

            pred = np.sum([X[c] * coefs[noise][component][i] for i, c in enumerate(X.columns)], axis=0)
            predictions.loc[:, component] = pred
            hindcast[noise][component] = pred

        hindcast_df[noise] = predictions
        predictions.to_csv(output / f'predictions_{noise}.csv')

    # Reconstructed signal for each noise value
    by_noise = {
        noise: sum(list(hindcast[noise].values()))
        for noise in hindcast
    }
    for noise, series in by_noise.items():
        np.savetxt(output / f'reconstructed_{noise}.csv', series, delimiter=',')

    by_noise_df = {
        noise: hindcast_df[noise].sum(axis=1)
        for noise in hindcast_df
    }

    total = sum(list(by_noise.values())) / len(by_noise)
    np.savetxt(output / 'reconstructed_total.csv', total, delimiter=',')

    total_df = sum(list(by_noise_df.values())) / len(by_noise_df)
    total_df.to_csv(output / 'reconstructed_total_df.csv')


# Don't run this directly - it's here to enable breakpoint debugging
# if __name__ == '__main__':
    # runner = CliRunner()

    # runner.invoke(decompose, ['-i', './data/separate_files', '-o', './test_out', '-s', 'shore', '-n', '0.1', '0.2', '0.3'])
    # runner.invoke(reconstruct, ['-o', './output_test', '-s', 'shore'])
