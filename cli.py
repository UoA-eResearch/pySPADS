import click
import pathlib

from click.testing import CliRunner
from tqdm import tqdm

from pipeline.decompose import load_data_from_csvs, imf_filename
from pipeline.decompose import  decompose as _decompose
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
