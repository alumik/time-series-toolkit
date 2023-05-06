import numpy as np
import pandas as pd

from typing import Sequence
from statsmodels.tsa.seasonal import STL

import tskit


def add(
        ts_array: Sequence[tskit.TimeSeries],
        weights: Sequence[float] | None = None,
        standardize_idx: Sequence[int] | None = None,
        index: pd.DatetimeIndex | pd.RangeIndex | None = None,
) -> tskit.TimeSeries:
    """
    Combine multiple time series into one.

    Parameters
    ----------
    ts_array: list of tskit.TimeSeries
        The time series to combine.
    weights: list of float, optional, default: None
        The weights to apply to each time series.
    standardize_idx: list of int, optional, default: None
        The indices of the time series to standardize.
    index: pd.DatetimeIndex or pd.RangeIndex, optional, default: None
        The index of the combined time series. If None, the index of the first time series is used.

    Returns
    -------
    tskit.TimeSeries
        The combined time series.
    """
    if any([len(ts) != len(ts_array[0]) for ts in ts_array]):
        raise ValueError('All time series must have the same length.')
    if weights is None:
        weights = [1.0] * len(ts_array)
    if len(weights) != len(ts_array):
        raise ValueError('Length of weights must be the same as length of time series array.')
    if standardize_idx is None:
        standardize_idx = []
    if index is None:
        index = ts_array[0].index.copy()
    values = np.zeros(len(ts_array[0]))
    for i, ts in enumerate(ts_array):
        if i in standardize_idx:
            values += standardize(ts).values * weights[i]
        else:
            values += ts.values * weights[i]
    return tskit.TimeSeries(
        index=index,
        values=values,
        name=f'[{"+".join([ts.name for ts in ts_array])}]',
    )


def to_shapelet(ts: tskit.TimeSeries, alpha: float = 1.0, inplace: bool = False) -> tskit.TimeSeries:
    """
    Convert the time series to a shapelet.

    A shapelet is a time series whose start and end values are equal.
    This method will convert the time series to a shapelet by gradually minimizing the difference between
    the start and end values along the time series.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The time series to convert.
    alpha: float, optional, default: 1.0
        A non-negative float that controls how much the shapelet deviates from the original time series.
        The smaller the alpha, the more the shapelet deviates from the original time series.
    inplace: bool, optional, default: False
        Whether to modify the time series in place.

    Returns
    -------
    tskit.TimeSeries
        The shapelet.
    """
    if inplace:
        shapelet = ts.values
    else:
        shapelet = ts.values.copy()
    for i in range(len(shapelet)):
        shapelet[i] -= (shapelet[i] - shapelet[0]) * (i / (len(shapelet) - 1)) ** alpha
    if inplace:
        return ts
    return tskit.TimeSeries(index=ts.index.copy(), values=shapelet, name=ts.name + '_shapelet')


def stl_decomposition(
        ts: tskit.TimeSeries,
        **kwargs
):
    """
    Perform seasonal and trend decomposition using LOESS.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The time series to decompose.

    Returns
    -------
    statsmodels.tsa.seasonal.DecomposeResult
        Estimated seasonal, trend, and residual components.

    Other Parameters
    ----------------
    **kwargs
        Other keyword arguments to pass to `statsmodels.tsa.seasonal.STL`.
    """
    stl = STL(pd.Series(ts.values, index=ts.index), **kwargs)
    return stl.fit()


def tile(
        ts: tskit.TimeSeries,
        n: int,
        inplace: bool = False,
) -> tskit.TimeSeries:
    """
    Tile the time series n times.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The time series to tile.
    n: int
        The number of times to tile the time series.
    inplace: bool, optional, default: False
        Whether to modify the time series in place.

    Returns
    -------
    tskit.TimeSeries
    """
    index = tskit.generator.generate_index(start=ts.index[0], length=len(ts.index) * n, freq=ts.freq)
    values = np.tile(ts.values, n)
    if inplace:
        ts.index = index
        ts.values = values
        return ts
    return tskit.TimeSeries(index=index, values=values, name=ts.name + '_tiled')


def standardize(
        ts: tskit.TimeSeries,
        mean: float | None = None,
        std: float | None = None,
        inplace: bool = False) -> tskit.TimeSeries:
    """
    Standardize the time series.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The time series to standardize.
    mean: float, optional, default: None
        The mean to use for standardization. If None, the mean of the time series will be used.
    std: float, optional, default: None
        The standard deviation to use for standardization.
        If None, the standard deviation of the time series will be used.
    inplace: bool, optional, default: False
        Whether to modify the time series in place.

    Returns
    -------
    tskit.TimeSeries
        The standardized time series.
    """
    if mean is None:
        mean = ts.values.mean()
    if std is None:
        std = ts.values.std()
    values = (ts.values - mean) / std
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index,
        values=values,
        name=f'{ts.name}_std',
    )
