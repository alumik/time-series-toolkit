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
    values = pd.Series(ts.values).rolling(window=window).mean().to_numpy()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_ma{window}',
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
        name=f'{ts.name}_ewma{alpha}',
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
    values = pd.Series(ts.values).rolling(window=window).median().to_numpy()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_med{window}',
    )


ma = moving_average
ewma = exponential_weighted_moving_average
