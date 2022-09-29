import uuid
import pathlib
import numpy as np
import pandas as pd

import tskit

from typing import *


class TimeSeries:

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
        return tskit.transform.to_shapelet(self, alpha=alpha, inplace=True)

    def add_noise(self, method: str = 'gaussian', amplitude: float = 0.1, **kwargs) -> 'TimeSeries':
        obj = tskit.utils.deserialize(method, obj_type='noise')
        if obj is None:
            raise ValueError(f'Unknown noise type: {method}.')
        return obj(self, amplitude=amplitude, inplace=True, **kwargs)

    def smooth(self, method: str = 'average', **kwargs) -> 'TimeSeries':
        obj = tskit.utils.deserialize(method, obj_type='smoother')
        if obj is None:
            raise ValueError(f'Unknown smoothing method: {method}.')
        return obj(self, inplace=True, **kwargs)

    def standardize(self) -> 'TimeSeries':
        return tskit.transform.standardize(self, inplace=True)

    def tile(self, n: int) -> 'TimeSeries':
        return tskit.transform.tile(self, n=n, inplace=True)

    def save(self, path: Optional[str] = None, save_format: str = 'csv', **kwargs):
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

    def __len__(self):
        return len(self.values)
