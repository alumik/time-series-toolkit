import numpy as np
import pandas as pd

import tskit

from typing import *


class TimeSeriesGenerator:
    method = None

    def __init__(
            self,
            start: Optional[pd.Timestamp | int] = 0,
            end: Optional[pd.Timestamp | int] = None,
            length: Optional[int] = None,
            freq: Optional[str | int] = None,
            name: Optional[str] = None,
            dtype: np.dtype = np.float64,
    ):
        self.start = start
        self.end = end
        self.length = length
        self.freq = freq
        self.name = name or self.__class__.method
        self.dtype = dtype

    def __call__(self, **kwargs) -> tskit.TimeSeries:
        base_config = {
            'start': self.start,
            'end': self.end,
            'length': self.length,
            'freq': self.freq,
            'name': self.name,
            'dtype': self.dtype,
        }
        config = {**base_config, **kwargs}
        return self.call(**config)

    def call(self, **kwargs) -> tskit.TimeSeries:
        raise NotImplementedError


class RandomWalkGenerator(TimeSeriesGenerator):
    method = 'random_walk'

    def __init__(
            self,
            mean: float = 0.0,
            std: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.random_walk(**{'mean': self.mean, 'std': self.std, **kwargs})


class SineGenerator(TimeSeriesGenerator):
    method = 'sine'

    def __init__(
            self,
            amplitude: float = 1.0,
            period: Optional[float] = None,
            phase: float = 0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.sine(**{
            'amplitude': self.amplitude,
            'period': self.period,
            'phase': self.phase,
            **kwargs,
        })


class CosineGenerator(TimeSeriesGenerator):
    method = 'cosine'

    def __init__(
            self,
            amplitude: float = 1.0,
            period: Optional[float] = None,
            phase: float = 0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.cosine(**{
            'amplitude': self.amplitude,
            'period': self.period,
            'phase': self.phase,
            **kwargs,
        })


class LinearGenerator(TimeSeriesGenerator):
    method = 'linear'

    def __init__(
            self,
            slope: float = 1.0,
            intercept: float = 0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.slope = slope
        self.intercept = intercept

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.linear(**{
            'slope': self.slope,
            'intercept': self.intercept,
            **kwargs,
        })


class GaussianGenerator(TimeSeriesGenerator):
    method = 'gaussian'

    def __init__(
            self,
            mean: float = 0.0,
            std: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.gaussian(**{
            'mean': self.mean,
            'std': self.std,
            **kwargs,
        })


class UniformGenerator(TimeSeriesGenerator):
    method = 'uniform'

    def __init__(
            self,
            low: float = -1.0,
            high: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.uniform(**{
            'low': self.low,
            'high': self.high,
            **kwargs,
        })


class StepGenerator(TimeSeriesGenerator):
    method = 'step'

    def __init__(
            self,
            step_size: int = 1,
            delta: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.step_size = step_size
        self.delta = delta

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.step(**{
            'step_size': self.step_size,
            'delta': self.delta,
            **kwargs,
        })


class ExponentialGenerator(TimeSeriesGenerator):
    method = 'exponential'

    def __init__(
            self,
            scale: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.exponential(**{
            'scale': self.scale,
            **kwargs,
        })


class PoissonGenerator(TimeSeriesGenerator):
    method = 'poisson'

    def __init__(
            self,
            lam: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.lam = lam

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.poisson(**{
            'lam': self.lam,
            **kwargs,
        })


class LogNormalGenerator(TimeSeriesGenerator):
    method = 'log_normal'

    def __init__(
            self,
            mean: float = 0.0,
            std: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.log_normal(**{
            'mean': self.mean,
            'std': self.std,
            **kwargs,
        })


class GammaGenerator(TimeSeriesGenerator):
    method = 'gamma'

    def __init__(
            self,
            shape: float = 1.0,
            scale: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.shape = shape
        self.scale = scale

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.gamma(**{
            'shape': self.shape,
            'scale': self.scale,
            **kwargs,
        })


class SquareGenerator(TimeSeriesGenerator):
    method = 'square'

    def __init__(
            self,
            amplitude: float = 1.0,
            period: float = 2.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.period = period

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.square(**{
            'amplitude': self.amplitude,
            'period': self.period,
            **kwargs,
        })


class SawtoothGenerator(TimeSeriesGenerator):
    method = 'sawtooth'

    def __init__(
            self,
            amplitude: float = 1.0,
            period: float = 2.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.period = period

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.sawtooth(**{
            'amplitude': self.amplitude,
            'period': self.period,
            **kwargs,
        })


class TriangleGenerator(TimeSeriesGenerator):
    method = 'triangle'

    def __init__(
            self,
            amplitude: float = 1.0,
            period: float = 2.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.period = period

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.triangle(**{
            'amplitude': self.amplitude,
            'period': self.period,
            **kwargs,
        })


class ImpulseGenerator(TimeSeriesGenerator):
    method = 'impulse'

    def __init__(
            self,
            base: float = 0.0,
            impulse_size: float = 1.0,
            impulse_time: Optional[Sequence[pd.Timestamp | int]] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.base = base
        self.impulse_size = impulse_size
        self.impulse_time = impulse_time

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.impulse(**{
            'base': self.base,
            'impulse_size': self.impulse_size,
            'impulse_time': self.impulse_time,
            **kwargs,
        })


class PerlinGenerator(TimeSeriesGenerator):
    method = 'perlin'

    def __init__(
            self,
            amplitude: float = 1.0,
            alpha: float = 1.0,
            beta: float = 1.0,
            theta: float = 1.0,
            factor_x: float = 2.0,
            factor_e: float = 1.0,
            factor_pi: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.factor_x = factor_x
        self.factor_e = factor_e
        self.factor_pi = factor_pi

    def call(self, **kwargs) -> tskit.TimeSeries:
        return tskit.generator.perlin(**{
            'amplitude': self.amplitude,
            'alpha': self.alpha,
            'beta': self.beta,
            'theta': self.theta,
            'factor_x': self.factor_x,
            'factor_e': self.factor_e,
            'factor_pi': self.factor_pi,
            **kwargs,
        })
