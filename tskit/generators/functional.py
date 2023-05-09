import numpy as np
import pandas as pd

from typing import Sequence

import tskit


def generate_index(
        start: pd.Timestamp | int | None = None,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
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
        freq = 1 if freq is None else int(freq)
        if start is not None and end is not None and (end - start) % freq != 0:
            raise ValueError(
                f'`start - end` must be evenly divisible by `freq`, but got {start}, {end}, {freq}.'
            )
        index = pd.RangeIndex(
            start=start if start is not None else end - freq * (length - 1),
            stop=end + freq if end is not None else start + freq * length,
            step=freq,
            name='timestamp',
        )
    return index


def random_walk(
        mean: float = 0.0,
        std: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'random_walk',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.cumsum(np.random.normal(mean, std, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def sine(
        amplitude: float = 1.0,
        period: float | None = None,
        phase: float = 0.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'sine',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    if period is None:
        period = len(index)
    values = np.asarray(amplitude * np.sin(2 * np.pi * np.arange(len(index)) / period + phase), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def cosine(
        amplitude: float = 1.0,
        period: float | None = None,
        phase: float = 0.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'cosine',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    if period is None:
        period = len(index)
    values = np.asarray(amplitude * np.cos(2 * np.pi * np.arange(len(index)) / period + phase), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def linear(
        slope: float = 1.0,
        intercept: float = 0.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'linear',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(slope * np.arange(len(index)) + intercept, dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def gaussian(
        mean: float = 0.0,
        std: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'gaussian',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(np.random.normal(mean, std, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def uniform(
        low: float = -1.0,
        high: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'uniform',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(np.random.uniform(low, high, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def step(
        step_size: int = 1,
        delta: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'step',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(delta * np.floor(np.arange(len(index)) / step_size), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def exponential(
        scale: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'exponential',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(np.random.exponential(scale, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def poisson(
        lam: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'poisson',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(np.random.poisson(lam, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def log_normal(
        mean: float = 0.0,
        std: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'log_normal',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(np.random.lognormal(mean, std, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def gamma(
        shape: float = 1.0,
        scale: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'gamma',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(np.random.gamma(shape, scale, size=len(index)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def square(
        amplitude: float = 1.0,
        period: float = 2.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'square',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(amplitude * np.arange(len(index)) / period * 2 % 2 < 1, dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def sawtooth(
        amplitude: float = 1.0,
        period: float = 2.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'sawtooth',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(amplitude * np.arange(len(index)) / period * 2 % 2, dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)


def triangle(
        amplitude: float = 1.0,
        period: float = 2.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'triangle',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray(
        amplitude * np.abs(np.arange(len(index)) / period * 4 % 4 - 2) - amplitude, dtype=dtype
    )
    return tskit.TimeSeries(index=index, values=values, name=name)


def impulse(
        base: float = 0.0,
        impulse_size: float = 1.0,
        impulse_time: Sequence[pd.Timestamp | int] | None = None,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'impulse',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    values = np.asarray([base] * len(index), dtype=dtype)
    if impulse_time is None:
        impulse_time = []
    for time in impulse_time:
        values[index.get_loc(time)] = impulse_size
    return tskit.TimeSeries(index=index, values=values, name=name)


def perlin(
        amplitude: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        theta: float = 1.0,
        factor_x: float = 2.0,
        factor_e: float = 1.0,
        factor_pi: float = 1.0,
        start: pd.Timestamp | int | None = 0,
        end: pd.Timestamp | int | None = None,
        length: int | None = None,
        freq: str | int | None = None,
        name: str | None = 'perlin',
        dtype: np.dtype = np.float64,
) -> tskit.TimeSeries:
    index = generate_index(start=start, end=end, length=length, freq=freq)
    x = np.arange(len(index))
    values = np.asarray(amplitude * (alpha * np.sin(factor_x * x) + beta * np.sin(factor_e * np.e * x) + theta * np.cos(
        factor_pi * np.pi * x)), dtype=dtype)
    return tskit.TimeSeries(index=index, values=values, name=name)
