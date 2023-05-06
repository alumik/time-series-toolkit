import uuid
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Sequence, Union, Self, Any

import tskit


class TimeSeries:
    """
    TimeSeries class for time series data.
    """

    def __init__(
            self,
            index: Sequence[int] | pd.RangeIndex | pd.DatetimeIndex,
            values: Sequence,
            name: str | None = None,
    ):
        """
        Initialize a TimeSeries object.

        Parameters
        ----------
        index: Sequence[int] or pd.RangeIndex or pd.DatetimeIndex
            The index of the time series.
        values: Sequence
            The values of the time series.
        name: str, optional, default: None
            The name of the time series.
        """
        if isinstance(index, pd.RangeIndex) or isinstance(index, pd.DatetimeIndex):
            self.index = index
        elif isinstance(index, Sequence) and all([isinstance(i, int) for i in index]):
            self.index = pd.Index(index)
        else:
            raise TypeError('Index must be a pd.RangeIndex, pd.DatetimeIndex or Sequence.')
        self.values = np.asarray(values)
        self.name = name or str(uuid.uuid4())

        if len(self.index) != len(self.values):
            raise ValueError('Length of index and values must be the same.')

        if isinstance(index, pd.DatetimeIndex):
            self.freq = index.freq
        elif isinstance(index, pd.RangeIndex):
            self.freq = index.step
        else:
            self.freq = None

    @classmethod
    def from_generators(
            cls,
            generators: Sequence[Union[str, 'tskit.generator.TimeSeriesGenerator']],
            generator_args: Sequence[dict] | None = None,
            weights: Sequence[float] | None = None,
            standardize_idx: Sequence[int] | None = None,
            start: pd.Timestamp | int | None = 0,
            end: pd.Timestamp | int | None = None,
            length: int | None = None,
            freq: str | int | None = None,
            name: str | None = None,
            dtype: np.dtype = np.float64,
    ) -> Self:
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
        if unknown_generators := [g for obj, g in zip(generator_objs, generators) if obj is None]:
            raise ValueError(f'Unknown generators: {unknown_generators}.')
        if len(generator_objs) == 0:
            raise ValueError('At least one generator must be specified.')
        if generator_args is None:
            generator_args = [{} for _ in range(len(generator_objs))]
        if len(generator_args) != len(generator_objs):
            raise ValueError('The number of generator arguments must match the number of generators.')
        series = [g(**args, **kwargs)() for g, kwargs in zip(generator_objs, generator_args)]
        series = tskit.transform.add(series, weights=weights, standardize_idx=standardize_idx)
        return series

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """
        Generate a time series from a config dictionary.

        Parameters
        ----------
        config: dict
            The config dictionary.

        Returns
        -------
        tskit.TimeSeries
            The generated time series.
        """
        generators = []
        generator_args = []
        weights = []
        standardize_idx = []
        generator_configs = config.pop('generators')
        for idx, generator_config in enumerate(generator_configs):
            generators.append(generator_config['name'])
            generator_args.append(generator_config.get('args', {}))
            weights.append(generator_config.get('weight', 1.0))
            if generator_config.get('standardize', False):
                standardize_idx.append(idx)
        return cls.from_generators(
            generators=generators,
            generator_args=generator_args,
            weights=weights,
            standardize_idx=standardize_idx,
            **config,
        )

    def to_shapelet(self, alpha: float = 1.0) -> Self:
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

    def add_noise(self, method: str = 'gaussian', amplitude: float = 0.1, **kwargs) -> Self:
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
        noise_func = tskit.utils.deserialize(method, obj_type='noise')
        if noise_func is None:
            raise ValueError(f'Unknown noise type: {method}.')
        return noise_func(self, amplitude=amplitude, inplace=True, **kwargs)

    def smooth(self, method: str = 'moving_average', **kwargs) -> Self:
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

    def standardize(self, mean: float | None = None, std: float | None = None) -> Self:
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

    def tile(self, n: int) -> Self:
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

    def save(self, path: str | None = None, save_format: str = 'csv', **kwargs):
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
            ax: plt.Axes | None = None,
            tight_layout: bool = True,
            show: bool = True,
            figsize: tuple[int] = (12, 2),
            title: str | None = None,
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
