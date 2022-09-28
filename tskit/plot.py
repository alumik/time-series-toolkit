import matplotlib.pyplot as plt

import tskit

from typing import *


def plot(ts: tskit.TimeSeries, title: Optional[str] = None):
    if ts.values.ndim == 1:
        _plot_univariate_time_series(ts, title=title)
    else:
        raise NotImplementedError('Only univariate time series are supported.')


def _plot_univariate_time_series(
        ts: tskit.TimeSeries,
        figsize: Sequence[int] = (12, 2),
        title: Optional[str] = None,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(ts.index, ts.values)
    title = ts.name if title is None else title
    ax.set_title(title)
    if ts.index.dtype == 'datetime64[ns]':
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.show()
