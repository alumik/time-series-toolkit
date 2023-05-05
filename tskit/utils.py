import re
import unicodedata

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
        The slugified string.
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


def deserialize(identifier: str | tskit.generator.TimeSeriesGenerator, obj_type: str):
    """
    Deserialize an object from a string identifier.

    Parameters
    ----------
    identifier: str | tskit.generator.TimeSeriesGenerator
        The identifier of the object to deserialize.
    obj_type: str
        The type of the object to deserialize. Must be one of 'generator', 'smoother', or 'noise'.

    Returns
    -------
    any
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
    if isinstance(identifier, tskit.generator.TimeSeriesGenerator):
        return identifier
    return all_objs.setdefault(obj_type, {}).get(identifier)
