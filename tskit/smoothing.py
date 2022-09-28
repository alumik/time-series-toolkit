import pandas as pd

import tskit


def average_smoothing(ts: tskit.TimeSeries, window: int, inplace: bool = False) -> tskit.TimeSeries:
    values = pd.Series(ts.values).rolling(window=window).mean().to_numpy()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_avg{window}',
    )


def exponential_smoothing(ts: tskit.TimeSeries, alpha: float, inplace: bool = False) -> tskit.TimeSeries:
    values = pd.Series(ts.values).ewm(alpha=alpha).mean().to_numpy()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=values,
        name=f'{ts.name}_ewm{alpha}',
    )
