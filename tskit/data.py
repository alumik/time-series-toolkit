import uuid
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tskit

from typing import *


class TimeSeries:
    """
    TimeSeries class for time series data.

    Parameters
    ----------
    index: pd.RangeIndex or pd.DatetimeIndex
        The index of the time series.
    values: np.ndarray
        The values of the time series.
    name: str, optional, default: None
        The name of the time series.
    """

    def __init__(
            self,
            index: pd.RangeIndex | pd.DatetimeIndex,
            values: np.ndarray,
            name: Optional[str] = None,
    ):
        self.index = index
        self.values = values
        if isinstance(index, pd.DatetimeIndex):
            self.freq = index.freq
        else:
            self.freq = index.step
        self.name = name or str(uuid.uuid4())

    @classmethod
    def from_generators(
            cls,
            generators: Sequence[str | Type['tskit.generator.TimeSeriesGenerator']],
            generator_args: Optional[Sequence[dict]] = None,
            weights: Optional[Sequence[float]] = None,
            standardize_idx: Optional[Sequence[int]] = None,
            start: Optional[pd.Timestamp | int] = 0,
            end: Optional[pd.Timestamp | int] = None,
            length: Optional[int] = None,
            freq: Optional[str | int] = None,
            name: Optional[str] = None,
            dtype: np.dtype = np.float64,
    ):
        """
        Generate a time series from a list of generators.

        Parameters
        ----------
        generators: list of str or list of tskit.generator.TimeSeriesGenerator
            The generators to use.
        generator_args: list of dict, optional, default: None
            The generator specific arguments to pass to the generators.
        weights: list of float, optional, default: None
            The weights to use for combining the generators.
        standardize_idx: list of int, optional, default: None
            The indices of the generators to standardize.
        start: pd.Timestamp or int, optional, default: 0
            The start index of the time series. If a pd.Timestamp is provided, the index will be a pd.DatetimeIndex.
            If an int is provided, the index will be a pd.RangeIndex.
        end: pd.Timestamp or int, optional, default: None
            The end index of the time series.
        length: int, optional, default: None
            The length of the time series. Exactly two of `start`, `end`, `length` must be specified.
            For generating an index with `end` and `length` consider setting `start` to None.
        freq: str or int, optional, default: None
            The frequency of the time series. If the index is pd.DatetimeIndex, this must be a valid frequency string.
            If the index is pd.RangeIndex, this must be an integer.
        name: str, optional, default: None
            The name of the time series. If not specified, a random UUID will be used.
        dtype: np.dtype, optional, default: np.float64
            The dtype of the values.

        Returns
        -------
        tskit.TimeSeries
            The generated time series.
        """
        args = {
            'start': start,
            'end': end,
            'length': length,
            'freq': freq,
            'name': name,
            'dtype': dtype,
        }
        generator_objs = [tskit.utils.deserialize(g, obj_type='generator') for g in generators]
        unknown_generators = [g for obj, g in zip(generator_objs, generators) if obj is None]
        if unknown_generators:
            raise ValueError(f'Unknown generators: {unknown_generators}.')
        if generator_args is None:
            generator_args = [{} for _ in range(len(generator_objs))]
        if len(generator_args) != len(generator_objs):
            raise ValueError('The number of generator arguments must match the number of generators.')
        series = [g(**args, **kwargs)() for g, kwargs in zip(generator_objs, generator_args)]
        series = tskit.transform.combine(series, weights=weights, standardize_idx=standardize_idx)
        return series

    def to_shapelet(self, alpha: float = 1.0) -> 'TimeSeries':
        """
        Convert the time series to a shapelet.

        A shapelet is a time series whose start and end values are equal.
        This method will convert the time series to a shapelet by gradually minimizing the difference between
        the start and end values along the time series.

        Changes to the time series are made in place.

        Parameters
        ----------
        alpha: float, optional, default: 1.0
            A non-negative float that controls how much the shapelet deviates from the original time series.
            The smaller the alpha, the more the shapelet deviates from the original time series.

        Returns
        -------
        tskit.TimeSeries
            The shapelet.
        """
        return tskit.transform.to_shapelet(self, alpha=alpha, inplace=True)

    def add_noise(self, method: str = 'gaussian', amplitude: float = 0.1, **kwargs) -> 'TimeSeries':
        """
        Add noise to the time series.

        Parameters
        ----------
        method: str, optional, default: 'gaussian'
            The noise method to use. ['gaussian', 'uniform', 'perlin'] are currently supported.
        amplitude: float, optional, default: 0.1
            The amplitude of the noise.

        Returns
        -------
        tskit.TimeSeries
            The time series with added noise.

        Other Parameters
        ----------------
        **kwargs:
            Additional keyword arguments to pass to the noise method.
        """
        obj = tskit.utils.deserialize(method, obj_type='noise')
        if obj is None:
            raise ValueError(f'Unknown noise type: {method}.')
        return obj(self, amplitude=amplitude, inplace=True, **kwargs)

    def smooth(self, method: str = 'moving_average', **kwargs) -> 'TimeSeries':
        """
        Smooth the time series.

        Parameters
        ----------
        method: str, optional, default: 'moving_average'
            The smoothing method to use. ['moving_average', 'median', 'exponential_weighted_moving_average']
            are currently supported.
            'ma' is an alias for 'moving_average'. 'ewma' is an alias for 'exponential_weighted_moving_average'.

        Returns
        -------
        tskit.TimeSeries
            The smoothed time series.

        Other Parameters
        ----------------
        **kwargs:
            Additional keyword arguments to pass to the smoothing method.
        """
        obj = tskit.utils.deserialize(method, obj_type='smoother')
        if obj is None:
            raise ValueError(f'Unknown smoothing method: {method}.')
        return obj(self, inplace=True, **kwargs)

    def standardize(self, mean: Optional[float] = None, std: Optional[float] = None) -> 'TimeSeries':
        """
        Standardize the time series.

        Parameters
        ----------
        mean: float, optional, default: None
            The mean to use for standardization. If None, the mean of the time series will be used.
        std: float, optional, default: None
            The standard deviation to use for standardization.
            If None, the standard deviation of the time series will be used.

        Returns
        -------
        tskit.TimeSeries
            The standardized time series.
        """
        return tskit.transform.standardize(self, mean=mean, std=std, inplace=True)

    def tile(self, n: int) -> 'TimeSeries':
        """
        Tile the time series n times.

        Parameters
        ----------
        n: int
            The number of times to tile the time series.

        Returns
        -------
        tskit.TimeSeries
        """
        return tskit.transform.tile(self, n=n, inplace=True)

    def save(self, path: Optional[str] = None, save_format: str = 'csv', **kwargs):
        """
        Save the time series to a file.

        Parameters
        ----------
        path: str, optional, default: None
            The path to save the time series to. If None, the time series name will be used.
        save_format: str, optional, default: 'csv'
            The format to save the time series in. Only 'csv' are currently supported.

        Other Parameters
        ----------------
        **kwargs:
            Additional keyword arguments to pass to the save method.
        """
        if path is None:
            path = tskit.utils.slugify(self.name) + '.' + save_format
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'value': self.values}).set_index(self.index)
        match save_format:
            case 'csv':
                df.to_csv(path, **kwargs)
            case _:
                raise ValueError(f'Unknown format: {save_format}.')

    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            tight_layout: bool = True,
            show: bool = True,
            figsize: tuple[int] = (12, 2),
            title: Optional[str] = None,
            **kwargs,
    ):
        """
            Plot the TimeSeries.

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
                The title of the plot.

            Other Parameters
            ----------------
            **kwargs
                Additional keyword arguments to pass to the plotting function.
            """
        tskit.plot(self, ax=ax, tight_layout=tight_layout, show=show, figsize=figsize, title=title, **kwargs)

    def __len__(self) -> int:
        return len(self.values)
