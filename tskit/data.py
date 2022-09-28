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

    def to_shapelet(self, alpha: float = 1.0) -> 'TimeSeries':
        return tskit.transform.to_shapelet(self, alpha=alpha, inplace=True)

    def add_noise(self, method: str = 'gaussian', amplitude: float = 0.1, **kwargs) -> 'TimeSeries':
        match method:
            case 'gaussian':
                return tskit.transform.add_gaussian_noise(self, amplitude=amplitude, inplace=True, **kwargs)
            case _:
                raise ValueError(f'Unknown noise type: {method}.')

    def smooth(self, method: str = 'average', **kwargs) -> 'TimeSeries':
        match method:
            case 'average':
                return tskit.smoothing.average_smoothing(self, inplace=True, **kwargs)
            case 'exponential':
                return tskit.smoothing.exponential_smoothing(self, inplace=True, **kwargs)
            case _:
                raise ValueError(f'Unknown smoothing method: {method}.')

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
