import re
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
            for generator in tskit.generator.TimeSeriesGenerator.__subclasses__()
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


def add_freq_to_datetime_index(idx: pd.DatetimeIndex, freq: str = None, inplace: bool = False) -> pd.DatetimeIndex:
    """
    Add a frequency to a pandas DatetimeIndex.

    Parameters
    ----------
    idx: pd.DatetimeIndex
        The DatetimeIndex to add a frequency to.
    freq: str, optional, default: None
        The frequency to add to the DatetimeIndex. If None, the frequency will be inferred.
    inplace: bool, optional, default: False
        Whether to modify the DatetimeIndex inplace.

    Returns
    -------
    pd.DatetimeIndex
        The DatetimeIndex with a frequency. It is
    """
    if not inplace:
        idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError('No discernible frequency found to `idx`. Specify a frequency string with `freq`.')
    return idx
