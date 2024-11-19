# Recreating plots from the 2020 paper
import warnings

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime

from pySPADS.processing.reconstruct import hindcast_index, get_y, get_X
from pySPADS.processing.bridge import datenum_to_datetime
import seaborn as sns
import colorcet

from pySPADS.processing.dataclasses import LinRegCoefficients
from pySPADS.processing.recomposition import component_frequencies


def _mask_datetime(df, start, end):
    """Convert datenum index to datetime, and mask to start/end datetimes"""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.map(datenum_to_datetime)
    return out[(out.index >= start) & (out.index <= end)]


def fig1(
    pc0: pd.Series,
    Hs: pd.Series,
    Tp: pd.Series,
    Dir: pd.Series,
    start: datetime,
    end: datetime,
):
    """Figure 1: Model drivers"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if pc0 is not None:
            d0 = _mask_datetime(pc0, start, end)
            sns.scatterplot(x=d0.index, y=d0, ax=axes[0], s=2)
        axes[0].set(ylabel="PC1 [m]")

        d1 = _mask_datetime(Hs, start, end)
        sns.scatterplot(x=d1.index, y=d1, ax=axes[1], s=2)
        axes[1].set(ylabel="Hs [m]")

        d2 = _mask_datetime(Tp, start, end)
        sns.scatterplot(x=d2.index, y=d2, ax=axes[2], s=2)
        axes[2].set(ylabel="Tp [s]")

        d3 = _mask_datetime(Dir, start, end)
        sns.scatterplot(x=d3.index, y=d3, ax=axes[3], s=2)
        axes[3].set(ylabel="Dir [deg]")

        axes[-1].set(xlabel="Time [yr]")

    return fig


def fig2(shore: pd.Series, shore_imf: pd.DataFrame, start: datetime, end: datetime):
    """Figure 2: Shoreline IMFs"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        output = _mask_datetime(shore, start, end)
        trend_col = shore_imf.columns[-1]
        trend = _mask_datetime(shore_imf[trend_col], start, end)
        sns.scatterplot(x=output.index, y=output, ax=axes[0], s=2)
        sns.lineplot(x=trend.index, y=trend, ax=axes[0], color="red")

        for ax in axes[1:]:
            sns.scatterplot(x=output.index, y=output - trend, ax=ax, s=2)

        # Add imf trends to each plot
        d1 = _mask_datetime(shore_imf[5], start, end)
        sns.lineplot(x=d1.index, y=d1, ax=axes[1], color="red")

        d2 = _mask_datetime(shore_imf[6], start, end)
        sns.lineplot(x=d2.index, y=d2, ax=axes[2], color="red")

        d3 = _mask_datetime(shore_imf[7], start, end)
        sns.lineplot(x=d3.index, y=d3, ax=axes[3], color="red")

        for ax in axes:
            ax.set(ylabel="Shoreline [m]")
        axes[-1].set(xlabel="Time [yr]")

    return fig


def fig3(
    all_imfs: dict[str, pd.DataFrame], signal_col: str, start: datetime, end: datetime
):
    """Figure 3: Drivers and shoreline response"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    # TODO: Manually match components by frequency
    #   label period
    #   label E.V.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        s0 = _mask_datetime(all_imfs[signal_col][5], start, end)
        d0 = _mask_datetime(all_imfs["Hs"][5], start, end)
        sns.lineplot(x=s0.index, y=s0, ax=axes[0], color="red")
        ax0 = axes[0].twinx()
        sns.lineplot(x=d0.index, y=d0, ax=ax0, color="cyan")
        axes[0].set(ylabel="IMF_S[m]")
        ax0.set(ylabel="IMF_Hs[m]")

        s1 = _mask_datetime(all_imfs[signal_col][6], start, end)
        d1 = _mask_datetime(all_imfs["Hs"][7], start, end)
        sns.lineplot(x=s1.index, y=s1, ax=axes[1], color="red")
        ax1 = axes[1].twinx()
        sns.lineplot(x=d1.index, y=d1, ax=ax1, color="cyan")
        axes[1].set(ylabel="IMF_S[m]")
        ax1.set(ylabel="IMF_Hs[m]")

        s2 = _mask_datetime(all_imfs[signal_col][8], start, end)
        sns.lineplot(x=s2.index, y=s2, ax=axes[2], color="red")
        ax2 = axes[2].twinx()
        if "PC1" in all_imfs:
            d2 = _mask_datetime(all_imfs["PC1"][8], start, end)
            sns.lineplot(x=d2.index, y=d2, ax=ax2, color="cyan")
        axes[2].set(ylabel="IMF_S[m]")
        ax2.set(ylabel="IMF_PC1[m]")

        s3 = _mask_datetime(all_imfs[signal_col][8], start, end)
        sns.scatterplot(x=s3.index, y=s3, ax=axes[3], color="black", s=2)
        ax3 = axes[3].twinx()
        if "PC1" in all_imfs:
            d3 = _mask_datetime(all_imfs["PC1"].sum(axis=1), start, end)
            sns.lineplot(x=d3.index, y=d3, ax=ax3, color="cyan")
        axes[3].set(ylabel="SOI", xlabel="Time[yr]")
        ax3.set(ylabel="IMF_PC1[m]")

    return fig


def fig4(
    signal: pd.Series,
    imf_predictions: pd.Series,
    start: datetime,
    end: datetime,
    hindcast_date: datetime,
):
    """Figure 4: Shoreline predictions"""
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        signal_data = _mask_datetime(signal, start, end)
        fit_data = _mask_datetime(imf_predictions, start, hindcast_date)
        predict_data = _mask_datetime(imf_predictions, hindcast_date, end)
        # TODO: plot error bounds from multiple noises
        sns.lineplot(
            x=signal_data.index, y=signal_data, ax=axes, color="black", label="Signal"
        )
        sns.lineplot(x=fit_data.index, y=fit_data, ax=axes, color="blue", label="Fit")
        sns.lineplot(
            x=predict_data.index, y=predict_data, ax=axes, color="red", label="Hindcast"
        )
        axes.set(xlabel="Date", ylabel="Shoreline (m)")

    # Legend
    handles, labels = axes.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper right")
    for lh in leg.legendHandles:
        lh.set_linewidth(4.0)

    return fig


def fig_si3(
    imfs: dict[str, pd.DataFrame],
    nearest_freqs: pd.DataFrame,
    signal: str,
    coeffs: LinRegCoefficients,
    annotate_coeffs: bool = False,
    exclude_trend: bool = False,
) -> plt.Figure:
    """SI fig 3: matrix of driver components contributing to signal components"""
    # Set up a plot grid, with an extra cols for summation arrows + result column
    signal_components = [c for c in imfs[signal].columns if c in nearest_freqs.index]
    num_components = len(signal_components)
    num_drivers = len(imfs) - 1

    grid = plt.GridSpec(num_components, num_drivers + 2, hspace=0.5)
    fig = plt.figure(figsize=((num_drivers + 2) * 5, num_components * 5))
    axs = [[None for j in range(num_drivers + 2)] for i in range(num_components)]

    cmap = colorcet.glasbey_hv

    # Plot signal components
    drivers = sorted(list(set(imfs.keys()) - {signal}))
    # TODO: plot for training range, hindcast range, forecast?
    index = hindcast_index(imfs, signal)

    for i, component in enumerate(signal_components):
        X = get_X(imfs, nearest_freqs, signal, component, index)
        y = get_y(imfs, signal, component, index)

        # For each driver (column) and signal frequency (row)
        for j, driver in enumerate(drivers):
            if driver in X.columns:
                ax = plt.subplot(grid[i, j])
                # Plot signal component
                ax.plot(index, y, label="signal", color=cmap[0])
                # Plot driver component
                ax.plot(index, X[driver], label="driver", alpha=0.5, color=cmap[1])
                # Plot prediction component, i.e.: driver * coefficient
                ax.plot(
                    index,
                    coeffs.coeffs[component][driver] * X[driver],
                    label="prediction",
                    alpha=0.5,
                    color=cmap[2],
                )
                # Optionally annotate with coefficient
                if annotate_coeffs:
                    plt.text(
                        0.5,
                        -0.2,
                        f"{coeffs.coeffs[component][driver]:.2f}x",
                        fontsize=30,
                        horizontalalignment="center",
                        verticalalignment="top",
                        transform=ax.transAxes,
                    )
                axs[i][j] = ax
            else:
                # Hide unused components
                axs[i][j] = plt.subplot(grid[i, j])
                axs[i][j].axis("off")

    # Plot signal totals
    for i, component in enumerate(signal_components):
        ax = plt.subplot(grid[i, num_drivers + 1])
        axs[i][num_drivers + 1] = ax

        X = get_X(imfs, nearest_freqs, signal, component, index)
        y = get_y(imfs, signal, component, index)

        ax.plot(index, y, label="signal", color=cmap[0])
        ax.plot(
            index,
            coeffs.predict(component, X),
            label="prediction",
            color=cmap[2],
            alpha=0.5,
        )

        # Summation arrow
        plt.annotate(
            "",
            xy=(-0.3, 0.5),
            xycoords=ax,
            xytext=(1.2, 0.5),
            textcoords=axs[i][num_drivers - 1],
            arrowprops=dict(
                arrowstyle="simple,head_width=2.0,head_length=2.0", color="black"
            ),
        )

    # Label signal periods
    plt.text(
        2.5,
        0.5,
        "component period",
        fontsize=40,
        horizontalalignment="left",
        verticalalignment="center",
        rotation=-90,
        transform=axs[int(num_components / 2)][num_drivers + 1].transAxes,
    )
    frequencies = component_frequencies(imfs[signal][signal_components])
    component_period_days = 365 / frequencies
    max_period_yrs = (imfs[signal].index.max() - imfs[signal].index.min()).days / 365

    def _format_period(period):
        if period < 1:
            return f"{period * 24:.1f} hours"
        elif np.isinf(period):
            return f">{max_period_yrs:.1f} years"
        elif period > 365:
            return f"{period/365:.1f} years"
        return f"{period:.1f} days"

    for i, period in enumerate(component_period_days):
        plt.text(
            1.3,
            0.5,
            _format_period(period),
            fontsize=40,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axs[i][num_drivers + 1].transAxes,
        )

    # Label drivers
    plt.text(
        0.5,
        1.8,
        "Drivers",
        fontsize=40,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axs[0][int((num_drivers + 2) / 2)].transAxes,
    )
    for j, driver in enumerate(drivers):
        plt.text(
            0.5,
            1.15,
            driver,
            fontsize=40,
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=axs[0][j].transAxes,
        )
    plt.text(
        0.5,
        1.15,
        signal,
        fontsize=40,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=axs[0][num_drivers + 1].transAxes,
    )

    # Label components
    plt.text(
        -0.6,
        0.5,
        "Components",
        fontsize=40,
        rotation="vertical",
        horizontalalignment="center",
        verticalalignment="center",
        transform=axs[int(num_components / 2)][0].transAxes,
    )
    for i, component in enumerate(signal_components):
        plt.text(
            -0.3,
            0.5,
            component,
            fontsize=40,
            horizontalalignment="right",
            verticalalignment="center",
            transform=axs[i][0].transAxes,
        )

    # Legend
    handles, labels = axs[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", fontsize=40, ncol=3)
    for lh in leg.legendHandles:
        lh.set_linewidth(8.0)

    return fig
