import numpy as np
import pandas as pd


def hindcast_index(imfs, signal):
    # TODO - fix this for actual hindcast, where signal index does not include hindcast period
    not_signal = list(imfs.keys() - {signal})
    return imfs[signal].index[imfs[signal].index.isin(imfs[not_signal[0]].index)]


def get_X(imfs, nearest_freqs, signal, component, hindcast_index):
    not_signal = list(imfs.keys() - {signal})
    X = pd.DataFrame(index=hindcast_index)
    for label in not_signal:
        # Get nearest frequency imf from driver, if we have one
        if not np.isnan(nearest_freqs.loc[component, label]):
            X[label] = imfs[label].loc[imfs[label].index.isin(hindcast_index), nearest_freqs.loc[component, label]]
    return X


def get_y(imfs, signal, component, hindcast_index):
    return imfs[signal].loc[hindcast_index, component]


