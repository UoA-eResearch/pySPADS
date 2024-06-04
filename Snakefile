def configuration(wildcards):
    import yaml
    from pathlib import Path

    cfile = Path('data') / wildcards.folder / 'config.yaml'
    assert cfile.is_file(), f'Configuration file {cfile} not found'

    with open(cfile, 'r') as f:
        config = yaml.safe_load(f)

    return config


def input_files(wildcards):
    import glob
    input_folder = f'data/{wildcards.folder}/input'
    return glob.glob(input_folder + '/*.csv')


def input_columns(wildcards):
    """Get all columns from all input files."""
    import pandas as pd
    files = input_files(wildcards)
    columns = []
    for file in files:
        df = pd.read_csv(file)
        columns.extend(df.columns)

    # Exclude time column
    columns = set(columns)
    config = configuration(wildcards)
    columns.discard(config['time_col'])
    return list(columns)


def expand_config(path, folder=None):
    def _expand(wildcards):
        # if folder:
        #     wildcards['folder'] = folder
        config = configuration(wildcards)
        return expand(path, **config)

    return _expand


rule all_figures:
    input:
        # Fig 1
        'data/{folder}/figures/paper_fig1.png',
        # Fig 2 (noises + mean)
        expand_config('data/{{folder}}/figures/paper_fig2_{noises}.png'),
        'data/{folder}/figures/paper_fig2_mean.png',
    output:
        # Fake output file in order to capture folder wildcard
        touch('data/{folder}/figures/all_figures.mark')


rule dates:
    # Get timespan for analysis
    input:
        folder='data/{folder}/input'
    output:
        'data/{folder}/dates.json'
    params:
        c=configuration
    script:
        'snakemake/dates.py'


rule decompose:
    # Decompose each channel into IMFs
    input:
        folder='data/{folder}/input',
        dates='data/{folder}/dates.json'
    output:
        'data/{folder}/imfs/{label}_imf_{noise}.csv'
    params:
        c=configuration
    script:
        'snakemake/decompose.py'

rule nearest_freq:
    # Find nearest frequencies in drivers to each signal IMF
    input:
        expand('data/{{folder}}/imfs/{label}_imf_{{noise}}.csv', label=input_columns)
    output:
        'data/{folder}/nearest_frequencies_{noise}.csv'
    params:
        c=configuration
    script:
        'snakemake/nearest_freq.py'

rule fit:
    # Fit a model to each signal IMF
    input:
        imfs=expand('data/{{folder}}/imfs/{label}_imf_{{noise}}.csv', label=input_columns),
        freqs='data/{folder}/nearest_frequencies_{noise}.csv'
    output:
        'data/{folder}/coefficients_{noise}.json'
    params:
        c=configuration
    script:
        'snakemake/fit.py'

# Paper figures
rule paper_fig1:
    input:
        folder='data/{folder}/input'
    output:
        'data/{folder}/figures/paper_fig1.png'
    params:
        c=configuration
    script:
        'snakemake/paper_fig1.py'

rule paper_fig2:
    input:
        folder='data/{folder}/input',
        signal_imfs=expand_config('data/{{folder}}/imfs/{signal}_imf_{noises}.csv')
    output:
        'data/{folder}/figures/paper_fig2_{noise}.png',
    params:
        c=configuration
    script:
        'snakemake/paper_fig2.py'
