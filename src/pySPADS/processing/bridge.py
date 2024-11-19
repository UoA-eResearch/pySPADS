from datetime import datetime, timedelta

import mat73
import numpy as np
import pandas as pd
from scipy.io import loadmat

from pySPADS.root import ROOT_DIR

# This file contains functions to work with data in matlab format, as a bridge to working with raw GIS data
data_dir = ROOT_DIR / "data" / "matlab"


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    """
    days = datenum % 1
    return (
        datetime.fromordinal(int(datenum)) + timedelta(days=days) - timedelta(days=366)
    )


def datetime_to_datenum(time: datetime):
    """
    Convert datetime to Matlab datenum
    """
    datenum = (
        time.toordinal()
        + 366
        + time.hour / 24
        + time.minute / (24 * 60)
        + time.second / (24 * 60 * 60)
    )

    return datenum


def _date_array_to_datetime(row):
    return datetime(*row)


def _loadmat(fname, items: list[tuple], root_object=None, v73=False):
    """Utility function for extracting matlab data in expected format"""
    # Note - won't handle data nested more than 1 level deep
    if not v73:
        # Matlab data from pre v7.3
        data = loadmat(fname, squeeze_me=True)

        result = {}

        # If the data we want is in a parent object, pull it up one level
        if root_object:
            data = data[root_object]
            for item, _ in items:
                result[item] = data[item].item()
        else:
            result = data
    else:
        # Newer Matlab version > v7.3
        result = mat73.loadmat(fname)

        if root_object:
            result = result[root_object]

    # Convert dtypes
    for item, dtype in items:
        if dtype == "datenum":
            # Matlab datenum (float days since 1970)
            value = result[item]
            result[item] = np.array(
                [datenum_to_datetime(t) for t in value.astype(float)]
            )
        elif dtype == "datearray":
            # Date array, columns for year, month, etc.
            value = result[item]
            assert value.shape[1] in [
                3,
                6,
            ], "Expecting 3/6 columns: year, month, day, [hour, minute, second]"
            result[item] = np.apply_along_axis(
                _date_array_to_datetime, axis=1, arr=value.astype(int)
            )
        else:
            # Other binary dtypes, e.g.: f8 -> float
            result[item] = result[item].astype(dtype)

    # Drop unwanted items
    unwanted_keys = set(result.keys()) - set([i[0] for i in items])
    for key in unwanted_keys:
        del result[key]

    return result


def _mean_by_day(df: pd.DataFrame):
    """Reduce time column to days, by taking mean value of other columns in each day"""
    df = df.copy()
    # Create column with integer day number
    df["datenum"] = df["time"].apply(datetime_to_datenum).astype(int)

    # Take mean of other columns for each day
    df = df.groupby("datenum").mean().drop(columns=["time"]).rename_axis("t")
    return df


def load_shorecast():
    fname = data_dir / "Shorecast_complete.mat"
    root_object = "Shore"
    items = [
        ("time", "datenum"),
        ("shore", float),
        ("average", float),
        ("yr", float),
        ("shoreline", float),
        ("alongshore", int),
    ]

    shore_dict = _loadmat(fname, items, root_object)

    y = shore_dict["average"][:-1]  # drop trailing NaN
    t = shore_dict["time"][:-1]
    shore_df = pd.DataFrame({"y": y, "time": t})
    return _mean_by_day(shore_df)


def load_hindcast():
    fname = data_dir / "Wave_hindcast_corrected.mat"
    root_object = "hindcast"
    items = [
        ("time", "datenum"),
        ("Hs", float),
        ("Tp", float),
        ("Dir", float),
        # ('tm01', float),
        # ('tm02', float),
    ]

    hc_dict = _loadmat(fname, items, root_object)
    hc_df = pd.DataFrame(hc_dict)
    return _mean_by_day(hc_df)


def load_SLP():
    fname = data_dir / "PCA_V3_CFS_Tairua_Lon_30_290_Lat_60_-80.mat"
    root_object = "PCA"
    items = [
        ("Media", float),
        ("Desviacion", float),
        ("EOF", float),
        ("PC", float),
        ("variance", float),
        ("Dates_cal", "datearray"),
    ]

    pca = _loadmat(fname, items, root_object, v73=True)
    as_dict = {"time": pca["Dates_cal"]}
    for i in range(10):
        as_dict[f"PC{i}"] = pca["PC"][:, i]
    pca_df = pd.DataFrame(as_dict)
    return _mean_by_day(pca_df)


def load_shore_d():
    # We probably don't need this data
    fname = data_dir / "ShoreD.mat"
    items = [
        ("shoreD", float),
        ("timeD", "datenum"),
    ]

    return _loadmat(fname, items)


def _complete_time_range(*columns: list[pd.Series]) -> list[int]:
    """
    Find the time range covering all DataFrames,
    from the minimum value in the first to the maximum value in the last
    """
    min_time = min(col.min() for col in columns)
    max_time = max(col.max() for col in columns)

    return list(range(int(min_time), int(max_time) + 1))


def _reindex_df(
    df: pd.DataFrame, time_range: list[int], interpolate=True
) -> pd.DataFrame:
    """Reindex DataFrame to fill gaps and interpolate"""
    out = df.copy().reindex(time_range, fill_value=np.nan)

    if interpolate:
        return out.interpolate(method="linear")
    else:
        return out


def load_data():
    """Load all data and do pre-processing"""
    # Target
    shore_df = load_shorecast()

    # Features - hindcast
    hc_df = load_hindcast()

    # Features - PCA
    pca_df = load_SLP()

    # Reindex each DataFrame to fill gaps and interpolate
    time_range = _complete_time_range(shore_df.index)
    shore_df = _reindex_df(shore_df, time_range)
    hc_df = _reindex_df(hc_df, time_range)
    pca_df = _reindex_df(pca_df, time_range)

    # Combine into one DataFrame, keeping only dates that are common to all
    combined = shore_df.merge(
        hc_df, how="inner", left_index=True, right_index=True
    ).merge(pca_df, how="inner", left_index=True, right_index=True)

    return combined
