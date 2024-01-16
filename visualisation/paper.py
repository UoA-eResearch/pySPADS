# Recreating plots from the 2020 paper
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from data_processing.bridge import datenum_to_datetime, datetime_to_datenum
import seaborn as sns


def _mask_datetime(df, start, end):
    """Convert datenum index to datetime, and mask to start/end datetimes"""
    out = df.copy()
    out.index = out.index.map(datenum_to_datetime)
    return out[(out.index >= start) & (out.index <= end)]


def fig1(pc0: pd.Series, Hs: pd.Series, Tp: pd.Series, Dir: pd.Series,
         start: datetime, end: datetime):
    """Figure 1: Model drivers"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        d0 = _mask_datetime(pc0, start, end)
        sns.scatterplot(x=d0.index, y=d0, ax=axes[0], s=2)
        axes[0].set(ylabel='PC1 [m]')

        d1 = _mask_datetime(Hs, start, end)
        sns.scatterplot(x=d1.index, y=d1, ax=axes[1], s=2)
        axes[1].set(ylabel='Hs [m]')

        d2 = _mask_datetime(Tp, start, end)
        sns.scatterplot(x=d2.index, y=d2, ax=axes[2], s=2)
        axes[2].set(ylabel='Tp [s]')

        d3 = _mask_datetime(Dir, start, end)
        sns.scatterplot(x=d3.index, y=d3, ax=axes[3], s=2)
        axes[3].set(ylabel='Dir [deg]')

        axes[-1].set(xlabel='Time [yr]')

    return fig


def fig2(shore: pd.Series, shore_imf: pd.DataFrame, start: datetime, end: datetime):
    """Figure 2: Shoreline IMFs"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        output = _mask_datetime(shore, start, end)
        trend = _mask_datetime(shore_imf[len(shore_imf.columns)], start, end)
        sns.scatterplot(x=output.index, y=output, ax=axes[0], s=2)
        sns.lineplot(x=trend.index, y=trend, ax=axes[0], color='red')

        for ax in axes[1:]:
            sns.scatterplot(x=output.index, y=output - trend, ax=ax, s=2)

        # Add imf trends to each plot
        d1 = _mask_datetime(shore_imf[5], start, end)
        sns.lineplot(x=d1.index, y=d1, ax=axes[1], color='red')

        d2 = _mask_datetime(shore_imf[6], start, end)
        sns.lineplot(x=d2.index, y=d2, ax=axes[2], color='red')

        d3 = _mask_datetime(shore_imf[7], start, end)
        sns.lineplot(x=d3.index, y=d3, ax=axes[3], color='red')

        for ax in axes:
            ax.set(ylabel='Shoreline [m]')
        axes[-1].set(xlabel='Time [yr]')

    return fig


def fig3(all_imfs: dict[str, pd.DataFrame], start: datetime, end: datetime):
    """Figure 3: Drivers and shoreline response"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    # TODO: Manually match components by frequency
    #   label period
    #   label E.V.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        s0 = _mask_datetime(all_imfs['output'][5], start, end)
        d0 = _mask_datetime(all_imfs['Hs'][5], start, end)
        sns.lineplot(x=s0.index, y=s0, ax=axes[0], color='red')
        ax0 = axes[0].twinx()
        sns.lineplot(x=d0.index, y=d0, ax=ax0, color='cyan')
        axes[0].set(ylabel='IMF_S[m]')
        ax0.set(ylabel='IMF_Hs[m]')

        s1 = _mask_datetime(all_imfs['output'][6], start, end)
        d1 = _mask_datetime(all_imfs['Hs'][7], start, end)
        sns.lineplot(x=s1.index, y=s1, ax=axes[1], color='red')
        ax1 = axes[1].twinx()
        sns.lineplot(x=d1.index, y=d1, ax=ax1, color='cyan')
        axes[1].set(ylabel='IMF_S[m]')
        ax1.set(ylabel='IMF_Hs[m]')

        s2 = _mask_datetime(all_imfs['output'][8], start, end)
        d2 = _mask_datetime(all_imfs['PC1'][8], start, end)
        sns.lineplot(x=s2.index, y=s2, ax=axes[2], color='red')
        ax2 = axes[2].twinx()
        sns.lineplot(x=d2.index, y=d2, ax=ax2, color='cyan')
        axes[2].set(ylabel='IMF_S[m]')
        ax2.set(ylabel='IMF_PC1[m]')

        s3 = _mask_datetime(all_imfs['output'][8], start, end)
        d3 = _mask_datetime(all_imfs['PC1'].sum(axis=1), start, end)
        sns.scatterplot(x=s3.index, y=s3, ax=axes[3], color='black', s=2)
        ax3 = axes[3].twinx()
        sns.lineplot(x=d3.index, y=d3, ax=ax3, color='cyan')
        axes[3].set(ylabel='SOI', xlabel='Time[yr]')
        ax3.set(ylabel='IMF_PC1[m]')

    return fig


def fig4(shore: pd.Series, imf_predictions: pd.DataFrame, start: datetime, end: datetime):
    """Figure 4: Shoreline predictions"""
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        s0 = _mask_datetime(shore, start, end)
        p0 = _mask_datetime(imf_predictions.sum(axis=1), start, end)
        sns.lineplot(x=s0.index, y=s0, ax=axes, color='black')
        sns.lineplot(x=p0.index, y=p0, ax=axes, color='red')
        axes.set(xlabel='Date', ylabel='Shoreline (m)')

    return fig
