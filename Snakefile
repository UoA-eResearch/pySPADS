def configuration(wildcards):
    import yaml
    from pathlib import Path

    cfile = Path('data') / wildcards.folder / 'config.yaml'
    assert cfile.is_file(), f'Configuration file {cfile} not found'

    with open(cfile, 'r') as f:
        config = yaml.safe_load(f)

    return config

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
    input:
        folder='data/{folder}/input',
        dates='data/{folder}/dates.json'
    output:
        'data/{folder}/imfs/{label}_imf_{noise}.csv'
    params:
        c=configuration
    script:
        'snakemake/decompose.py'


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


def paper2_files(wildcards):
    # IMFs for signal column, for all noises
    config = configuration(wildcards)
    noise_str = lambda n: f'{n:.3f}'.replace('.', '_')
    return ['data/{folder}/imfs/{label}_imf_{noise}.csv'.format(
        folder=wildcards.folder, label=config['signal'], noise=noise_str(n)) for n in config['noises']]


rule paper_fig2:
    input:
        folder='data/{folder}/input',
        signal_imfs=paper2_files
    output:
        'data/{folder}/figures/paper_fig2_{noise}.png',
    params:
        c=configuration
    script:
        'snakemake/paper_fig2.py'
