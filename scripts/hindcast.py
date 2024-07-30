import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.decompose import decompose, reject_noise
from pipeline.frequencies import match_frequencies
from pipeline.reconstruct import fit, hindcast_index, get_X
from processing.data import load_data_from_csvs, imf_filename, load_imfs
from processing.dataclasses import LinRegCoefficients
from visualisation import paper

if __name__ == '__main__':
    run_label = 'run1'

    input_folder = Path(__file__).parent.parent / 'data' / run_label / 'input'
    output_folder = Path(__file__).parent.parent / 'data' / run_label
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load input data
    time_col = 't'
    dfs = load_data_from_csvs(input_folder, time_col)

    # Setup info needed to perform analysis
    signal = 'shore'
    noises = [0.1, 0.2, 0.3, 0.4, 0.5]
    frequency_threshold = 0.25
    noise_threshold = 0.95

    drivers = list(set(dfs.keys()) - {signal})

    # decompose from earliest available date to latest date available for all input data
    start_date = min([min(df.index) for df in dfs.values()])
    end_date = min([max(df.index) for df in dfs.values()])
    # date from which to hindcast signal
    hindcast_date = pd.Timestamp('2012-01-01')

    # Replicate figures from paper
    (output_folder / 'figures').mkdir(parents=True, exist_ok=True)
    if 'PC0' in drivers:  # Skip figure 1 if missing PCA data
        f = paper.fig1(dfs['PC0'], dfs['Hs'], dfs['Tp'], dfs['Dir'], '2004-01-01', '2016-01-01')
        f.savefig(output_folder / 'figures' / 'fig1.png')

    # Perform decomposition (warning: takes ~4 hours)
    imf_dir = output_folder / 'imfs'
    imf_dir.mkdir(parents=True, exist_ok=True)

    for col in tqdm(dfs, desc='Decomposing IMFs'):
        for noise in noises:
            filename = imf_filename(imf_dir, col, noise)
            if filename.exists():
                continue
            # Mask to min/max time range
            if col == signal:
                # For signal, only decompose up to hindcast date
                df = dfs[col].loc[start_date:hindcast_date]
            else:
                df = dfs[col].loc[start_date:end_date]

            imf_df = decompose(df, noise=noise, num_trials=100, progress=False)
            imf_df.to_csv(filename)

    ## Match input frequencies to driver frequencies
    imfs = load_imfs(imf_dir)

    # Drop IMF modes that are mostly noise
    for label, imf_df in imfs.items():
        imfs[label] = reject_noise(imf_df, noise_threshold=noise_threshold)

    # TODO - check which noise value to use
    f = paper.fig2(dfs[signal], imfs[(signal, 0.1)], '1999-01-01', '2017-01-01')
    f.savefig(output_folder / 'figures' / 'fig2.png')

    # Re-organise imfs into dict[noise][label]
    imfs_by_noise: dict[float, dict[str, pd.DataFrame]] = {}
    for label, noise in imfs.keys():
        if noise not in imfs_by_noise:
            imfs_by_noise[noise] = {}
        imfs_by_noise[noise][label] = imfs[(label, noise)]

    if 'PC0' in drivers:  # Skip figure 3 if missing PCA data
        f = paper.fig3(imfs_by_noise[0.1], f'{signal}', '1999-01-01', '2017-01-01')
        f.savefig(output_folder / 'figures' / 'fig3.png')

    ## Reconstruct signal (including hindcast) from drivers
    nearest_freqs: dict[float, pd.DataFrame] = {}
    coefficients: dict[float, LinRegCoefficients] = {}
    component_predictions: dict[float, pd.DataFrame] = {}
    predictions: dict[float, pd.Series] = {}

    for noise in noises:
        # Match signal components to nearest driver components by frequency
        nearest_freqs[noise] = match_frequencies(imfs_by_noise[noise], signal, frequency_threshold)
        nearest_freqs[noise].to_csv(output_folder / f'frequencies_{noise}.csv')

        # Linear regression of decomposed drivers to decomposed signal
        coefficients[noise] = fit(imfs_by_noise[noise], nearest_freqs[noise], signal, model='ridge', fit_intercept=True)

        # SI 3 figure
        f = paper.fig_si3(imfs_by_noise[noise], nearest_freqs[noise], signal, coefficients[noise], annotate_coeffs=True)
        f.savefig(output_folder / 'figures' / f'fig_si3_{noise}.png')

        # Make predictions for each component of the signal
        output_columns = imfs_by_noise[noise][signal].columns
        index = pd.date_range(start=start_date, end=end_date, freq='D')  # Note we're predicting from start date
        comp_preds = pd.DataFrame(index=index, columns=output_columns)

        for component in output_columns:
            X = get_X(imfs_by_noise[noise], nearest_freqs[noise], signal, component, index)

            comp_preds.loc[:, component] = coefficients[noise].predict(component, X)

        component_predictions[noise] = comp_preds
        comp_preds.to_csv(output_folder / f'predictions_{noise}.csv')

        # Sum component predictions to give overall signal prediction (for this noise level)
        predictions[noise] = component_predictions[noise].sum(axis=1)

    # Combine predictions from all noise levels into single output
    total: pd.Series = sum(list(predictions.values())) / len(predictions)
    total.to_csv(output_folder / 'reconstructed_total_df.csv')

    f = paper.fig4(dfs[signal], total, '2000-01-01', '2019-12-31', hindcast_date)
    f.savefig(output_folder / 'figures' / 'fig4.png')
