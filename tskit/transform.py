import numpy as np
import pandas as pd

import tskit

from typing import *
from statsmodels.tsa.seasonal import STL


def combine(
        ts_array: Sequence[tskit.TimeSeries],
        weights: Optional[Sequence[float]] = None,
        standardize_idx: Optional[Sequence[int]] = None,
        index: Optional[pd.DatetimeIndex | pd.RangeIndex] = None,
) -> tskit.TimeSeries:
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
    stl = STL(pd.Series(ts.values, index=ts.index), **kwargs)
    return stl.fit()


def tile(
        ts: tskit.TimeSeries,
        n: int,
        inplace: bool = False,
) -> tskit.TimeSeries:
    index = tskit.generator.generate_index(start=ts.index[0], length=len(ts.index) * n, freq=ts.freq)
    values = np.tile(ts.values, n)
    if inplace:
        ts.index = index
        ts.values = values
        return ts
    return tskit.TimeSeries(index=index, values=values, name=ts.name + '_tiled')


def standardize(ts: tskit.TimeSeries, inplace: bool = False) -> tskit.TimeSeries:
    values = (ts.values - ts.values.mean()) / ts.values.std()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index,
        values=values,
        name=f'{ts.name}_std',
    )
