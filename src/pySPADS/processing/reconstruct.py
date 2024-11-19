import numpy as np
import pandas as pd


def hindcast_index(imfs, signal):
    """Get date index for hindcasting, based on the intersection of the signal and driver IMFs"""
    not_signal = list(imfs.keys() - {signal})
    return imfs[signal].index[imfs[signal].index.isin(imfs[not_signal[0]].index)]


def get_X(imfs, nearest_freqs, signal, component, index):
    """Get X data for given component of each driver"""
    not_signal = list(imfs.keys() - {signal})
    X = pd.DataFrame(index=index)
    for label in not_signal:
        # Get nearest frequency imf from driver, if we have one
        if not np.isnan(nearest_freqs.loc[component, label]):
            X[label] = imfs[label].loc[
                imfs[label].index.isin(index), nearest_freqs.loc[component, label]
            ]
    return X


def get_y(imfs, signal, component, index):
    """Get y data for given component of signal"""
    return imfs[signal].loc[index, component]
