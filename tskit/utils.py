import re
import unicodedata
import numpy as np

import tskit


def repeat(
        ts: tskit.TimeSeries,
        n: int,
        inplace: bool = False,
) -> tskit.TimeSeries:
    index = tskit.generator.generate_index(start=ts.index[0], length=len(ts.index) * n, freq=ts.freq)
    values = np.tile(ts.values, n)
    if inplace:
        ts.index = index
        ts.values = values
        return ts
    return tskit.TimeSeries(index=index, values=values, name=ts.name + '_repeated')


def standardize(ts: tskit.TimeSeries, inplace: bool = False) -> tskit.TimeSeries:
    values = (ts.values - ts.values.mean()) / ts.values.std()
    if inplace:
        ts.values = values
        return ts
    return tskit.TimeSeries(
        index=ts.index,
        values=values,
        name=f'{ts.name}_std',
    )


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
