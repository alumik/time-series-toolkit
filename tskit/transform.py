import numpy as np
import pandas as pd

import tskit

from statsmodels.tsa.seasonal import STL


def merge(
        base: tskit.TimeSeries,
        modifier: tskit.TimeSeries,
        base_weight: float = 1.0,
        modifier_weight: float = 1.0,
        standardize_modifier: bool = True,
) -> tskit.TimeSeries:
    if len(base) != len(modifier):
        raise ValueError('Base and modifier must have the same length.')
    if standardize_modifier:
        modifier = tskit.utils.standardize(modifier)
    values = base.values * base_weight + modifier.values * modifier_weight
    return tskit.TimeSeries(
        index=base.index.copy(),
        values=values,
        name=f'[{base.name}*{base_weight}+{modifier.name}*{modifier_weight}]',
    )


def transform_shapelet(ts: tskit.TimeSeries, alpha: float = 1.0, inplace: bool = False) -> tskit.TimeSeries:
    if inplace:
        shapelet = ts.values
    else:
        shapelet = ts.values.copy()
    for i in range(len(shapelet)):
        shapelet[i] -= (shapelet[i] - shapelet[0]) * (i / (len(shapelet) - 1)) ** alpha
    if inplace:
        return ts
    return tskit.TimeSeries(index=ts.index.copy(), values=shapelet, name=ts.name + '_shapelet')


def add_gaussian_noise(
        ts: tskit.TimeSeries,
        mean: float = 0.0,
        std: float = 1.0,
        amplitude: float = 0.1,
        inplace: bool = False,
) -> tskit.TimeSeries:
    noise = np.random.normal(mean, std, size=len(ts))
    if inplace:
        ts.values += noise * amplitude
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=ts.values + noise * amplitude,
        name=ts.name + '_noisy',
    )


def stl_decomposition(
        ts: tskit.TimeSeries,
        **kwargs
):
    stl = STL(pd.Series(ts.values, index=ts.index), **kwargs)
    return stl.fit()
