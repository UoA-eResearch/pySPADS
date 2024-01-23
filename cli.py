import click
import pathlib


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
def run(path):
    """Run the full pipeline"""
    print("Not implemented")


@cli.command()
@click.option('-i', '--input',
              type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=pathlib.Path),
              help='Input data file or directory, expects either a csv, or a directory of csvs',
              required=True, multiple=True)
@click.option('-o', '--output',
              type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
              help='Output directory')
@click.option('-s', '--signal', type=str, help='Column name of signal to fit to')
@click.option('--overwrite', is_flag=True, help='Overwrite existing IMF files in output directory')
def decompose(input, output, signal, overwrite):
    """Decompose input data into IMFs"""
    print(input)
