# tskit - Time Series Toolkit

![version-0.0.1](https://img.shields.io/badge/version-0.0.1-blue)
![python-3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
[![license-MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/alumik/time-series-toolkit/blob/main/LICENSE)

`tskit` is a collection of tools for time series data analysis.

## A Quick Example

```python
import tskit
import pandas as pd

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
```

More Examples:

- [time_series_factory.ipynb](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/time_series_factory.ipynb)
- [generate_uts.ipynb](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/generate_uts.ipynb)
- [generator.ipynb](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/generator.ipynb)
- [merge_uts.ipynb](https://github.com/alumik/time-series-toolkit/blob/master/notebooks/merge_uts.ipynb)

## Tools

- [x] Time series generation
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
    - [ ] Savitzky-Golay filter
- [x] Time series transformation
    - [x] Merge multiple time series
    - [x] Tile time series
    - [x] Shapelet transform
    - [x] Standardize
    - [x] Seasonal-trend decomposition with LOESS (STL)
