import scipy
import numpy as np
import pandas as pd

import tskit


def moving_average(ts: tskit.TimeSeries, window: int, inplace: bool = False) -> tskit.TimeSeries:
    """
    Smooth a TimeSeries using a moving average (MA).

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to smooth.
    window: int
        The window size.
    inplace: bool, optional, default: False
        Whether to smooth the TimeSeries in place.

    Returns
    -------
    tskit.TimeSeries
        The smoothed TimeSeries.
    """
    values = pd.Series(ts.values).rolling(window=window, min_periods=0).mean().to_numpy()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_ma({window})',
    )


def exponential_weighted_moving_average(ts: tskit.TimeSeries, alpha: float, inplace: bool = False) -> tskit.TimeSeries:
    """
    Smooth a TimeSeries using an exponential weighted moving average (EWMA).

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to smooth.
    alpha: float
        The smoothing factor.
    inplace: bool, optional, default: False
        Whether to smooth the TimeSeries in place.

    Returns
    -------
    tskit.TimeSeries
        The smoothed TimeSeries.
    """
    values = pd.Series(ts.values).ewm(alpha=alpha).mean().to_numpy()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_ewma({alpha})',
    )


def median(ts: tskit.TimeSeries, window: int, inplace: bool = False) -> tskit.TimeSeries:
    """
    Smooth a TimeSeries using a median filter.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to smooth.
    window: int
        The window size.
    inplace: bool, optional, default: False
        Whether to smooth the TimeSeries in place.

    Returns
    -------
    tskit.TimeSeries
        The smoothed TimeSeries.
    """
    values = scipy.signal.medfilt(ts.values, window)
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_med({window})',
    )


def savitzky_golay(ts: tskit.TimeSeries, window: int, order: int, inplace: bool = False) -> tskit.TimeSeries:
    """
    Smooth a TimeSeries using a Savitzky-Golay filter.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to smooth.
    window: int
        The window size.
    order: int
        The order of the polynomial used to fit the samples.
    inplace: bool, optional, default: False
        Whether to smooth the TimeSeries in place.

    Returns
    -------
    tskit.TimeSeries
        The smoothed TimeSeries.
    """
    values = scipy.signal.savgol_filter(ts.values, window, order)
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_sg({window},{order})',
    )


ma = moving_average
ewma = exponential_weighted_moving_average
savgol = savitzky_golay
