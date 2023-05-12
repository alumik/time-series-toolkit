import numpy as np
import pandas as pd

from numpy import typing as npt
from typing import Iterable, Sequence
from statsmodels.tsa.seasonal import STL

import tskit


def add(
        ts_array: Sequence[tskit.TimeSeries],
        weights: Sequence[float] | None = None,
        standardize_idx: Iterable[int] | None = None,
        index: npt.ArrayLike | None = None,
        name: str | None = None,
) -> tskit.TimeSeries:
    """
    Combine multiple time series into one.

    Parameters
    ----------
    ts_array: sequence of tskit.TimeSeries
        The time series to combine.
    weights: sequence of float, optional, default: None
        The weights to apply to each time series.
    standardize_idx: iterable of int, optional, default: None
        The indices of the time series to standardize.
    index: array-like, optional, default: None
        The index of the combined time series.
        If None, the index of the first time series is used.
    name: str, optional, default: None
        The name of the combined time series.
        If None, the names of the input time series are combined.

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
        raise ValueError('`weights` and `ts_array` must be of equal length.')
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
        name=f'[{"+".join([ts.name for ts in ts_array])}]' if name is None else name,
    )


def to_shapelet(
        ts: tskit.TimeSeries,
        alpha: float = 1.0,
        inplace: bool = False,
        name: str | None = None,
) -> tskit.TimeSeries:
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
    name: str, optional, default: None
        The name of the shapelet.
        The default is the name of the original TimeSeries with '_shapelet'.
        This is ignored if `inplace` is True.

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
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=shapelet,
        name=f'{ts.name}_shapelet' if name is None else name,
    )


def stl_decomposition(
        ts: tskit.TimeSeries,
        **kwargs,
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
        interpolate_method: str = 'linear',
        fill_value: float = 0.0,
        period_length: int | pd.offsets.BaseOffset | None = None,
        inplace: bool = False,
        name: str | None = None,
) -> tskit.TimeSeries:
    """
    Tile the time series n times. The time series must be interpolated before tiling.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The time series to tile.
    n: int
        The number of times to tile the time series.
    interpolate_method: str, optional, default: 'linear'
        The interpolation method to use. See `tskit.transform.interpolate` for more details.
    fill_value: float, optional, default: 0.0
        The value to use for filling missing values. This is only used when `interpolate_method` is set to 'constant'.
    period_length: int or pd.offsets.BaseOffset, optional, default: None
        The history period to use for interpolation. This is only used when `method` is set to 'history'.
    inplace: bool, optional, default: False
        Whether to modify the time series in place.
    name: str, optional, default: None
        The name of the tiled time series. The default is the name of the original TimeSeries with '_tiled'.
        This is ignored if `inplace` is True.

    Returns
    -------
    tskit.TimeSeries
        The tiled time series.
    """
    ts_ = tskit.transform.interpolate(ts, method=interpolate_method, fill_value=fill_value, period_length=period_length)
    index = tskit.generators.generate_index(start=ts_.index[0], length=len(ts_.index) * n, freq=ts_.freq)
    values = np.tile(ts_.values, n)
    if inplace:
        ts.assign(values=values, index=index)
        return ts
    return tskit.TimeSeries(index=index, values=values, name=f'{ts_.name}_tiled' if name is None else name)


def standardize(
        ts: tskit.TimeSeries,
        mean: float | None = None,
        std: float | None = None,
        inplace: bool = False,
        name: str | None = None,
) -> tskit.TimeSeries:
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
    name: str, optional, default: None
        The name of the standardized time series. The default is the name of the original TimeSeries with '_std'.
        This is ignored if `inplace` is True.

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
        ts.assign(values)
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_std' if name is None else name,
    )


def interpolate(
        ts: tskit.TimeSeries,
        method: str = 'linear',
        fill_value: float = 0.0,
        period_length: int | pd.offsets.BaseOffset | None = None,
        inplace: bool = False,
        name: str | None = None,
) -> tskit.TimeSeries:
    """
    Interpolate missing index and values in the time series.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The time series to interpolate.
    method: str, optional, default: 'linear'
        The interpolation method to use. See `pandas.Series.interpolate` for more details.
    fill_value: float, optional, default: 0.0
        The value to use for filling missing values. This is only used when `method` is set to 'constant'.
    period_length: int or pandas.offsets.BaseOffset, optional, default: None
        The history period to use for interpolation. This is only used when `method` is set to 'history'.
    inplace: bool, optional, default: False
        Whether to modify the time series in place.
    name: str, optional, default: None
        The name of the interpolated time series. The default is the name of the original TimeSeries with '_interpolated'.
        This is ignored if `inplace` is True.

    Returns
    -------
    tskit.TimeSeries
        The interpolated time series.
    """
    series = pd.Series(ts.values.copy(), index=ts.index.copy())
    new_index = tskit.generators.generate_index(start=ts.index[0], end=ts.index[-1], freq=ts.freq)
    series = series.reindex(new_index)
    if method == 'constant':
        series = series.fillna(fill_value)
    elif method == 'history':
        if isinstance(period_length, int):
            series = series.fillna(series.shift(periods=period_length))
        elif isinstance(period_length, pd.offsets.BaseOffset):
            series = series.fillna(series.shift(freq=period_length))
        else:
            raise ValueError(
                'A valid `period_length` must be provided when using the `history` method. '
                f'Got {period_length}.'
            )
    else:
        series = series.interpolate(method=method)
    if inplace:
        ts.assign(values=series.values, index=series.index)
        return ts
    return tskit.TimeSeries(
        index=series.index,
        values=series.values,
        name=f'{ts.name}_interpolated' if name is None else name,
    )
