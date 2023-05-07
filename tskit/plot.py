import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

import tskit


def plot(
        ts: tskit.TimeSeries,
        ax: plt.Axes | None = None,
        tight_layout: bool = True,
        show: bool = True,
        figsize: tuple[int, int] = (12, 2),
        title: str | None = None,
        **kwargs,
):
    """
    Plot a TimeSeries.

    Currently only supports univariate time series.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to plot.
    ax: matplotlib.Axes, optional, default: None
        The axes to plot on. If None, a new figure will be created.
    tight_layout: bool, optional, default: True
        Whether to use tight layout.
    show: bool, optional, default: True
        Whether to show the plot.
    figsize: tuple, optional, default: (12, 2)
        The size of the figure.
    title: str, optional, default: None
        The title of the plot. If None, the name of the time series will be used.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments to pass to the plotting function.
    """
    if ts.values.ndim == 1:
        plotting_func = _plot_univariate_time_series
    else:
        raise NotImplementedError('Only univariate time series are supported at the moment.')
    plotting_func(
        ts,
        ax=ax,
        tight_layout=tight_layout,
        show=show,
        figsize=figsize,
        title=title,
        **kwargs,
    )


def _plot_univariate_time_series(
        ts: tskit.TimeSeries,
        ax: plt.Axes | None = None,
        tight_layout: bool = True,
        show: bool = True,
        figsize: tuple[int, int] = (12, 2),
        title: str | None = None,
        **kwargs,
):
    """
    Plot a univariate TimeSeries.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to plot.
    ax: matplotlib.Axes, optional, default: None
        The axes to plot on. If None, a new figure will be created.
    tight_layout: bool, optional, default: True
        Whether to use tight layout.
    show: bool, optional, default: True
        Whether to show the plot.
    figsize: tuple, optional, default: (12, 2)
        The size of the figure.
    title: str, optional, default: None
        The title of the plot. If None, the name of the time series will be used.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments to pass to the plotting function.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(ts.index, ts.values, **kwargs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(ts.name if title is None else title)

    if ts.index.dtype == 'datetime64[ns]':
        fig.autofmt_xdate()
    if tight_layout:
        fig.tight_layout()
    if show:
        fig.show()
