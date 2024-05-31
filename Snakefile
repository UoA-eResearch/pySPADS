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
