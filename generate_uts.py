import pandas as pd
import seaborn as sns

import tskit

sns.set_style('darkgrid')

ts = (
    tskit.TimeSeries.from_generators(
        ['sine', 'random_walk'],
        weights=[1.0, 0.5],
        standardize_idx=[1],
        start=pd.Timestamp('2022-09-27 00:00:00'),
        length=1440,
        freq='1min',
    )
    .smooth('ewma', alpha=0.5)
    .to_shapelet(alpha=5)
    .tile(10)
    .add_noise('gaussian', amplitude=0.1)
)
ts.save('out/uts.csv')

tskit.plot(ts, title=ts.name)
