def configuration(wildcards):
    import yaml
    from pathlib import Path

    cfile = Path('data') / wildcards.folder / 'config.yaml'
    assert cfile.is_file(), f'Configuration file {cfile} not found'

    with open(cfile, 'r') as f:
        config = yaml.safe_load(f)

    return config

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
