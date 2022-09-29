import matplotlib.pyplot as plt

import tskit

from typing import *


def plot(ts: tskit.TimeSeries, figsize: tuple[int] = (12, 2), title: Optional[str] = None):
    """
    Plot a TimeSeries.

    Currently only supports univariate time series.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to plot.
    figsize: tuple, optional, default: (12, 2)
        The size of the figure.
    title: str, optional, default: None
        The title of the plot.
    """
    if ts.values.ndim == 1:
        _plot_univariate_time_series(ts, figsize=figsize, title=title)
    else:
        raise NotImplementedError('Only univariate time series are supported.')


def _plot_univariate_time_series(
        ts: tskit.TimeSeries,
        figsize: tuple[int] = (12, 2),
        title: Optional[str] = None,
):
    """
    Plot a univariate TimeSeries.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to plot.
    figsize: tuple, optional, default: (12, 2)
        The size of the figure.
    title: str, optional, default: None
        The title of the plot.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(ts.index, ts.values)
    title = ts.name if title is None else title
    ax.set_title(title)
    if ts.index.dtype == 'datetime64[ns]':
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.show()
