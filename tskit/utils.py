import re
import unicodedata

import tskit


def slugify(value, allow_unicode=False) -> str:
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
        },
        'noise': {
            'gaussian': tskit.noise.add_gaussian_noise,
            'perlin': tskit.noise.add_perlin_noise,
        }
    }
    if isinstance(identifier, tskit.generator.TimeSeriesGenerator):
        return identifier
    return all_objs.setdefault(obj_type, {}).get(identifier)
