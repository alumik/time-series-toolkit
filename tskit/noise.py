import numpy as np

import tskit

from typing import *


def add_gaussian_noise(
        ts: tskit.TimeSeries,
        mean: float = 0.0,
        std: float = 1.0,
        amplitude: float = 0.1,
        clip: Optional[Sequence[float]] = None,
        inplace: bool = False,
) -> tskit.TimeSeries:
    noise = amplitude * np.random.normal(mean, std, size=len(ts))
    if clip is not None:
        noise = np.clip(noise, *clip)
    if inplace:
        ts.values += noise
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=ts.values + noise,
        name=ts.name + '_gaussian_noise',
    )


def add_perlin_noise(
        ts: tskit.TimeSeries,
        alpha: float = 1.0,
        beta: float = 1.0,
        theta: float = 1.0,
        factor_x: float = 2.0,
        factor_e: float = 1.0,
        factor_pi: float = 1.0,
        amplitude: float = 0.1,
        clip: Optional[Sequence[float]] = None,
        inplace: bool = False,
) -> tskit.TimeSeries:
    x = np.arange(len(ts))
    noise = amplitude * (alpha * np.sin(factor_x * x) + beta * np.sin(factor_e * np.e * x) + theta * np.cos(
        factor_pi * np.pi * x))
    if clip is not None:
        noise = np.clip(noise, *clip)
    if inplace:
        ts.values += noise
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=ts.values + noise,
        name=ts.name + '_perlin_noise',
    )
