def configuration(wildcards):
    import yaml
    from pathlib import Path

    cfile = Path('config.yaml')
    assert cfile.is_file(), f'Configuration file {cfile} not found'

    with open(cfile, 'r') as f:
        config = yaml.safe_load(f)

    return config


def input_columns(wildcards):
    """Get all columns from all input files."""
    import pandas as pd
    import glob
    files = glob.glob('input/*.csv')
    columns = []
    for file in files:
        df = pd.read_csv(file)
        columns.extend(df.columns)

    # Exclude time column
    columns = set(columns)
    config = configuration(wildcards)
    columns.discard(config['time_col'])
    return list(columns)


def config_signal(wildcards):
    config = configuration(wildcards)
    return config['signal']


def config_noises(wildcards):
    config = configuration(wildcards)
    return config['noises']


def expand_config(path):
    def _expand(wildcards):
        config = configuration(wildcards)
        return expand(path, **config)

    return _expand


wildcard_constraints:
    # noise=r'\d+\.\d+',
    label=r'[a-zA-Z_\-0-9]+'
    #label=r'^((?!full).)*$'

models = ['mreg2', 'mreg2_intercept', 'linreg', 'linreg_intercept', 'ridge', 'ridge_intercept']

rule all_figures:
    input:
        # Fig 1
        'figures/paper_fig1.png',
        # Fig 2 (noises + mean)
        expand('figures/paper_fig2_{noise}.png', noise=config_noises),
        'figures/paper_fig2_mean.png',
        # Fig 3 (noises + mean)
        expand('figures/paper_fig3_{noise}.png', noise=config_noises),
        'figures/paper_fig3_mean.png',
        # Fig 4
        expand('figures/paper_{model}_fig4.png', model=models),
        # SI Fig 3
        expand('figures/fit_matrix_{model}_{noise}.png', noise=config_noises, model=models),
        # All imfs
        expand('figures/imfs/{label}_imf_{noise}.png', label=input_columns, noise=config_noises),
        expand('imfs/full/{signal}_imf_{noise}.csv', signal=config_signal, noise=config_noises),

rule dates:
    # Get timespan for analysis
    input:
        folder='input'
    output:
        'dates.json'
    params:
        c=configuration
    script:
        'snakemake/dates.py'

rule calc_trend:
    # Calculate signal trend, so it can be removed if needed
    input:
        folder='input',
        dates='dates.json'
    output:
        'trend.json'
    params:
        c=configuration
    script:
        'snakemake/calc_trend.py'

# TODO: this seems too fast - test for all cols/noises?
rule decompose:
    # Decompose each channel into IMFs
    input:
        folder='input',
        dates='dates.json',
        trend='trend.json'
    output:
        'imfs/{label}_imf_{noise}.csv'
    params:
        c=configuration,
        full=False
    script:
        'snakemake/decompose.py'

rule decompose_full_signal:
    # Decompose full signal into IMFs
    input:
        folder='input',
        dates='dates.json',
        trend='trend.json'
    output:
        'imfs/full/{label}_imf_{noise}.csv'
    params:
        c=configuration,
        full=True
    script:
        'snakemake/decompose.py'

rule nearest_freq:
    # Find nearest frequencies in drivers to each signal IMF
    input:
        expand('imfs/{label}_imf_{{noise}}.csv', label=input_columns)
    output:
        'nearest_frequencies_{noise}.csv'
    params:
        c=configuration
    script:
        'snakemake/nearest_freq.py'

rule fit:
    # Fit a model to each signal IMF
    input:
        imfs=expand('imfs/{label}_imf_{{noise}}.csv', label=input_columns),
        freqs='nearest_frequencies_{noise}.csv'
    output:
        coeffs='{model}/coefficients_{noise}.json',
        figure='figures/fit_matrix_{model}_{noise}.png'
    params:
        c=configuration
    script:
        'snakemake/fit.py'

rule predict:
    # Predict each signal IMF
    input:
        imfs=expand('imfs/{label}_imf_{{noise}}.csv', label=input_columns),
        freqs='nearest_frequencies_{noise}.csv',
        coeffs='{model}/coefficients_{noise}.json',
        dates='dates.json'
    output:
        '{model}/predictions_{noise}.csv'
    params:
        c=configuration
    script:
        'snakemake/predict.py'

rule combine_preds:
    # Combine all predictions
    input:
        predictions=expand('{{model}}/predictions_{noise}.csv', noise=config_noises),
        trend='trend.json'
    output:
        '{model}/predictions.csv'
    params:
        c=configuration
    script:
        'snakemake/combine_preds.py'

# Visualisation
rule plot_imf:
    input:
        imf='imfs/{imf_label}_imf_{noise}.csv'
    output:
        'figures/imfs/{imf_label}_imf_{noise}.png'
    script:
        'snakemake/plot_imf.py'

# Paper figures
rule paper_fig1:
    input:
        folder='input'
    output:
        'figures/paper_fig1.png'
    params:
        c=configuration
    script:
        'snakemake/paper_fig1.py'

rule paper_fig2:
    input:
        folder='input',
        signal_imfs=expand_config('imfs/{signal}_imf_{noises}.csv')
    output:
        'figures/paper_fig2_{noise}.png',
    params:
        c=configuration
    script:
        'snakemake/paper_fig2.py'

rule paper_fig3:
    input:
        imfs = expand('imfs/{label}_imf_{{noise}}.csv', label=input_columns),
    output:
        'figures/paper_fig3_{noise}.png',
    params:
        c=configuration
    wildcard_constraints:
        noise=r'\d+\.\d+'
    script:
        'snakemake/paper_fig3.py'

rule paper_fig3_mean:
    input:
        imfs=expand('imfs/{label}_imf_{noise}.csv', label=input_columns, noise=config_noises),
    output:
        'figures/paper_fig3_{noise}.png',
    params:
        c=configuration
    wildcard_constraints:
        noise='mean'
    script:
        'snakemake/paper_fig3.py'

rule paper_fig4:
    input:
        folder='input',
        predictions='{model}/predictions.csv',
        dates='dates.json',
        trend='trend.json'
    output:
        'figures/paper_{model}_fig4.png'
    params:
        c=configuration
    script:
        'snakemake/paper_fig4.py'
