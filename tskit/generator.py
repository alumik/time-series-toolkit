import numpy as np
import pandas as pd

import tskit

from typing import *


def generate_index(
        start: Optional[pd.Timestamp | int] = None,
        end: Optional[pd.Timestamp | int] = None,
        length: Optional[int] = None,
        freq: Optional[str | int] = None,
) -> pd.DatetimeIndex | pd.RangeIndex:
    constructors = [
        arg_name
        for arg, arg_name in zip([start, end, length], ['start', 'end', 'length'])
        if arg is not None
    ]
    if len(constructors) != 2:
        raise ValueError(
            f'Exactly two of `start`, `end`, `length` must be specified, but got {constructors}. '
            'For generating an index with `end` and `length` consider setting `start` to None.'
        )
    if end is not None and start is not None and type(start) != type(end):
        raise ValueError(
            f'`start` and `end` must be of the same type, but got {type(start)} and {type(end)}.'
        )
    if isinstance(start, pd.Timestamp) or isinstance(end, pd.Timestamp):
        index = pd.date_range(
            start=start,
            end=end,
            periods=length,
            freq='D' if freq is None else freq,
            name='timestamp',
        )
    else:
        step = 1 if freq is None else int(freq)
        if start is not None and end is not None and (end - start) % step != 0:
            raise ValueError(
                f'`start - end` must be evenly divisible by `step`, but got {start}, {end}, {step}.'
            )
        index = pd.RangeIndex(
            start=start if start is not None else end - step * length + step,
            stop=end + step if end is not None else start + step * length,
            step=step,
            name='timestamp',
        )
    return index


def random_walk(
        mean: float = 0.0,
        std: float = 1.0,
        start: Optional[pd.Timestamp | int] = 0,
        end: Optional[pd.Timestamp | int] = None,
        length: Optional[int] = None,
        freq: Optional[str | int] = None,
        name: Optional[str] = 'random_walk',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.cumsum(np.random.normal(mean, std, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def sin(
        amplitude: float = 1.0,
        period: float = 1.0,
        phase: float = 0.0,
        start: Optional[pd.Timestamp | int] = 0,
        end: Optional[pd.Timestamp | int] = None,
        length: Optional[int] = None,
        freq: Optional[str | int] = None,
        name: Optional[str] = 'sin',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(amplitude * np.sin(2 * np.pi * np.arange(len(index)) / period + phase), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)
