import pandas as pd
import seaborn as sns

import tskit

sns.set_style('darkgrid')

ts = (
    tskit.TimeSeries.from_generators(
        ['sine', 'square', 'random_walk'],
        generator_args=[{}, {'period': 1440}, {}],
        weights=[1.0, 0.5, 0.8],
        standardize_idx=[2],
        start=pd.Timestamp('2022-09-27 00:00:00'),
        length=1440,
        freq='1min',
    )
    .smooth('ewma', alpha=0.5)
    .to_shapelet(alpha=5)
    .tile(10)
    .add_noise('gaussian', amplitude=0.2)
)
ts.save('out/uts.csv')

tskit.plot(ts)
