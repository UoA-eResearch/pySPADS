from pathlib import Path

import pandas as pd
from tqdm import tqdm

from processing.trend import detect_trend, gen_trend
from pipeline import steps
from processing.data import load_data_from_csvs, imf_filename, load_imfs
from processing.dataclasses import LinRegCoefficients, TrendModel
from visualisation import paper
from visualisation.imfs import plot_imfs

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
    exclude_trend = True
    normalize_drivers = False

    drivers = list(set(dfs.keys()) - {signal})

    # decompose from earliest available date to latest date available for all input data
    start_date = min([min(df.index) for df in dfs.values()])
    end_date = min([max(df.index) for df in dfs.values()])
    # date from which to hindcast signal
    hindcast_date = pd.Timestamp('2012-01-01')

    # De-trend signal data, and calculate linear regression to re-trend after prediction
    if exclude_trend:
        # Fit trend based on training data
        d = dfs[signal].loc[start_date:hindcast_date]
        signal_trend = detect_trend(d)

        # Subtract trend from input signal
        dfs[signal] -= gen_trend(dfs[signal], signal_trend)
    else:
        # No-op trend if not excluding
        signal_trend = TrendModel()

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

            imf_df = steps.decompose(df, noise=noise, num_trials=100, progress=False, parallel=False)
            imf_df.to_csv(filename)

            plot_imfs(imf_df.to_numpy().T, f'{col}-{noise}',
                      output_folder / 'figures' / 'imfs' / f'{col}_imf_{noise}.png')

    # Additional full decomposition of signal, for plots
    for noise in noises:
        filename = imf_filename(imf_dir, f'{signal}_full', noise)
        if filename.exists():
            continue

        df = dfs[signal].loc[start_date:end_date]

        imf_df = steps.decompose(df, noise=noise, num_trials=100, progress=False, parallel=False)
        imf_df.to_csv(filename)

        plot_imfs(imf_df.to_numpy().T, f'{signal}-{noise}',
                  output_folder / 'figures' / 'imfs' / f'{signal}_full_imf_{noise}.png')

    ## Load and re-arrange resulting decompositions
    imfs = load_imfs(imf_dir)

    # Filter out signal_full imfs
    imfs = {(label, noise): imf for (label, noise), imf in imfs.items() if label != f'{signal}_full'}

    # Drop IMF modes that are mostly noise
    for label, imf_df in imfs.items():
        imfs[label] = steps.reject_noise(imf_df, noise_threshold=noise_threshold)

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

    for noise in tqdm(noises, desc='Reconstructing signal'):
        # Match signal components to nearest driver components by frequency
        nearest_freqs[noise] = steps.match_frequencies(imfs_by_noise[noise], signal, frequency_threshold,
                                                 exclude_trend=exclude_trend)
        nearest_freqs[noise].to_csv(output_folder / f'frequencies_{noise}.csv')

        # Linear regression of decomposed drivers to decomposed signal
        coefficients[noise] = steps.fit(imfs_by_noise[noise], nearest_freqs[noise], signal, model='mreg2',
                                        fit_intercept=True, normalize=normalize_drivers)

        # SI 3 figure
        f = paper.fig_si3(imfs_by_noise[noise], nearest_freqs[noise], signal, coefficients[noise],
                          annotate_coeffs=True, exclude_trend=exclude_trend)
        f.savefig(output_folder / 'figures' / f'fig_si3_{noise}.png')

        # Make predictions for each component of the signal
        predictions_by_noise = steps.predict(imfs_by_noise[noise], nearest_freqs[noise], signal, coefficients[noise],
                                             start_date, end_date, exclude_trend=exclude_trend)

        component_predictions[noise] = predictions_by_noise
        predictions_by_noise.to_csv(output_folder / f'predictions_{noise}.csv')

    # Combine predictions from all noise levels into single output
    total = steps.combine_predictions(component_predictions, trend=signal_trend)
    total.to_csv(output_folder / 'reconstructed_total_df.csv')

    # Add trend back into signal for plotting
    plot_signal = dfs[signal] + gen_trend(dfs[signal], signal_trend)

    f = paper.fig4(plot_signal, total, '2000-01-01', '2019-12-31', hindcast_date)
    f.savefig(output_folder / 'figures' / 'fig4.png')
