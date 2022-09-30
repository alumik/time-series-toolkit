# tskit - Time Series Toolkit

![version-0.0.2](https://img.shields.io/badge/version-0.0.1-blue)
![python-3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
[![license-MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/alumik/time-series-toolkit/blob/main/LICENSE)

`tskit` is a collection of tools for time series data analysis.

Only univariate time series are supported at the moment.

## A Quick Example

```python
import pandas as pd

import tskit

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
```

More Examples:

- [Generate time series](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/generate_time_series.ipynb)
- [Generate time series (step-by-step)](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/generate_time_series_step_by_step.ipynb)
- [Time series generators](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/time_series_generators.ipynb)
- [Smoothing methods](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/smoothing_methods.ipynb)
- [Merge time series](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/merge_time_series.ipynb)

## Tools

- [x] Time series generation
    - [x] From generators
        - [x] Random walk
        - [x] Sine
        - [x] Cosine
        - [x] Linear
        - [x] Gaussian
        - [x] Uniform
        - [x] Step
        - [x] Exponential
        - [x] Poisson
        - [x] LogNormal
        - [x] Gamma
        - [x] Square
        - [x] Sawtooth
        - [x] Triangle
        - [x] Impulse
        - [x] Perlin
    - [ ] From existing time series samples *[Planned for v0.1.0]*
- [x] Add noise to time series
    - [x] Gaussian noise
    - [x] Uniform noise
    - [x] 1-D Perlin noise
- [x] Plot time series
    - [x] Plot univariate time series
- [x] Time series smoothing
    - [x] Moving average (MA)
    - [x] Exponential weighted moving average (EWMA)
    - [x] Median filter
    - [x] Savitzky-Golay filter
- [x] Time series transformation
    - [x] Merge multiple time series
    - [x] Tile time series
    - [x] Shapelet transform
    - [x] Standardize
    - [x] Seasonal-trend decomposition with LOESS (STL)
    - [ ] Split time series into segments *[Planned for v0.1.0]*
    - [ ] Interpolate missing values *[Planned for v0.1.0]*
- [ ] Anomaly/outlier injection *[Planned for v0.1.0]*
    - [ ] Point-wise outlier
        - [ ] Contextual outlier
        - [ ] Global outlier
    - [ ] Pattern-wise outlier
        - [ ] Shapelet outlier
        - [ ] Trend outlier
        - [ ] Seasonal outlier
- [ ] Support for multivariate time series *[Planned for v0.2.0]*
