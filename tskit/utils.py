import re
import math
import functools
import unicodedata
import pandas as pd

from typing import Callable, Any

import tskit


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Covert a string to a valid path name by removing or replacing invalid characters.

    Parameters
    ----------
    value: str
        The string to convert.
    allow_unicode: bool, optional, default: False
        Whether to allow unicode characters. If False, all non-ASCII characters will be removed.

    Returns
    -------
    str
        The converted string.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = (
            unicodedata.normalize('NFKD', value)
            .encode('ascii', 'ignore')
            .decode('ascii')
        )
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def deserialize(identifier: Any, obj_type: str) -> Callable:
    """
    Deserialize an object from a string identifier.

    Parameters
    ----------
    identifier: any
        The identifier of the object to deserialize. If the identifier is already an object, it will be returned directly.
    obj_type: str
        The type of the object to deserialize. Must be one of 'generator', 'smoother', or 'noise'.

    Returns
    -------
    Callable
        The deserialized object.
    """
    all_objs = {
        'generator': {
            generator.method: generator
            for generator in tskit.generators.TimeSeriesGenerator.__subclasses__()
        },
        'smoother': {
            'moving_average': tskit.smoothing.moving_average,
            'ma': tskit.smoothing.moving_average,
            'exponential_weighted_moving_average': tskit.smoothing.exponential_weighted_moving_average,
            'ewma': tskit.smoothing.exponential_weighted_moving_average,
            'median': tskit.smoothing.median,
            'savitzky_golay': tskit.smoothing.savitzky_golay,
            'savgol': tskit.smoothing.savitzky_golay,
        },
        'noise': {
            'gaussian': tskit.noise.add_gaussian_noise,
            'uniform': tskit.noise.add_uniform_noise,
            'perlin': tskit.noise.add_perlin_noise,
        }
    }
    if isinstance(identifier, str):
        return all_objs.setdefault(obj_type, {}).get(identifier)
    return identifier


def infer_freq(idx: pd.Index | pd.RangeIndex | pd.DatetimeIndex) -> pd.offsets.BaseOffset | int:
    """
    Infer the frequency of a pandas Index.

    Parameters
    ----------
    idx: pd.Index or pd.RangeIndex or pd.DatetimeIndex
        The index to infer the frequency of.

    Returns
    -------
    pd.offsets.BaseOffset or int
        The inferred frequency of the index.
    """
    freq = None
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq
    elif isinstance(idx, pd.RangeIndex):
        freq = idx.step
    if freq is not None:
        return freq
    # TODO: How to check if the index of a `pd.DatetimeIndex is regularly spaced?
    idx = idx.sort_values()
    diffs = idx[1:] - idx[:-1]
    if isinstance(idx, pd.DatetimeIndex):
        freq = diffs.min()
        return pd.tseries.frequencies.to_offset(freq)
    gcd = functools.reduce(math.gcd, diffs)
    return int(gcd)
