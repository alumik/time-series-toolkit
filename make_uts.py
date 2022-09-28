import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tskit

sns.set_style('darkgrid')

ts_mod = tskit.generator.random_walk(start=pd.Timestamp('2022-09-27 00:00:00'), length=1440, freq='1min')
ts_base = tskit.generator.sin(start=pd.Timestamp('2022-09-27 00:00:00'), length=1440, freq='1min', period=1440)
ts = tskit.transform.merge([ts_base, ts_mod], weights=[1.0, 0.5], standardize_idx=[1])
ts = (
    ts
    .smooth('exponential', alpha=0.5)
    .to_shapelet(alpha=5)
    .tile(10)
    .add_noise('gaussian')
)
ts.save('out/uts.csv')

tskit.plot(ts, title=ts.name)

stl = tskit.transform.stl_decomposition(ts, period=1440)

stl.plot()
plt.gcf().autofmt_xdate()
plt.gcf().tight_layout()
plt.show()
