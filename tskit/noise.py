import numpy as np

import tskit


def add_gaussian_noise(
        ts: tskit.TimeSeries,
        mean: float = 0.0,
        std: float = 1.0,
        amplitude: float = 0.1,
        clip: tuple[float, float] | None = None,
        inplace: bool = False,
        name: str | None = None,
) -> tskit.TimeSeries:
    """
    Add Gaussian noise to a TimeSeries.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to add noise to.
    mean: float, optional, default: 0.0
        The mean of the Gaussian distribution.
    std: float, optional, default: 1.0
        The standard deviation of the Gaussian distribution.
    amplitude: float, optional, default: 0.1
        The amplitude of the noise.
    clip: tuple, optional, default: None
        The lower and upper bounds to clip the noise to.
    inplace: bool, optional, default: False
        Whether to add the noise in place.
    name: str, optional, default: None
        The name of the new TimeSeries. The default is the name of the original TimeSeries with '_gaussian_noise'.
        This is ignored if `inplace` is True.

    Returns
    -------
    tskit.TimeSeries
        The TimeSeries with added noise.
    """
    noise = amplitude * np.random.normal(mean, std, size=len(ts))
    if clip is not None:
        noise = np.clip(noise, *clip)
    if inplace:
        ts += noise
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=ts.values + noise,
        name=f'{ts.name}_gaussian_noise' if name is None else name,
    )


def add_uniform_noise(
        ts: tskit.TimeSeries,
        low: float = -1.0,
        high: float = 1.0,
        amplitude: float = 0.1,
        inplace: bool = False,
        name: str | None = None,
) -> tskit.TimeSeries:
    """
    Add uniform noise to a TimeSeries.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to add noise to.
    low: float, optional, default: -1.0
        The lower bound of the uniform distribution.
    high: float, optional, default: 1.0
        The upper bound of the uniform distribution.
    amplitude: float, optional, default: 0.1
        The amplitude of the noise.
    inplace: bool, optional, default: False
        Whether to add the noise in place.
    name: str, optional, default: None
        The name of the new TimeSeries. The default is the name of the original TimeSeries with '_uniform_noise'.
        This is ignored if `inplace` is True.

    Returns
    -------
    tskit.TimeSeries
        The TimeSeries with added noise.
    """
    noise = amplitude * np.random.uniform(low, high, size=len(ts))
    if inplace:
        ts += noise
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=ts.values + noise,
        name=f'{ts.name}_uniform_noise' if name is None else name,
    )


def add_perlin_noise(
        ts: tskit.TimeSeries,
        alpha: float = 1.0,
        beta: float = 1.0,
        theta: float = 1.0,
        factor_x: float = 2.0,
        factor_e: float = 1.0,
        factor_pi: float = 1.0,
        amplitude: float = 0.1,
        clip: tuple[float, float] | None = None,
        inplace: bool = False,
        name: str | None = None,
) -> tskit.TimeSeries:
    """
    Add Perlin noise to a TimeSeries.

    Parameters
    ----------
    ts: tskit.TimeSeries
        The TimeSeries to add noise to.
    alpha: float, optional, default: 1.0
        This parameter will be used in the calculation of the noise: noise = amplitude * (alpha * np.sin(factor_x * x)
        + beta * np.sin(factor_e * np.e * x) + theta * np.cos(factor_pi * np.pi * x))
    beta: float, optional, default: 1.0
        This parameter will be used in the calculation of the noise. See `alpha`.
    theta: float, optional, default: 1.0
        This parameter will be used in the calculation of the noise. See `alpha`.
    factor_x: float, optional, default: 2.0
        This parameter will be used in the calculation of the noise. See `alpha`.
    factor_e: float, optional, default: 1.0
        This parameter will be used in the calculation of the noise. See `alpha`.
    factor_pi: float, optional, default: 1.0
        This parameter will be used in the calculation of the noise. See `alpha`.
    amplitude: float, optional, default: 0.1
        The amplitude of the noise.
    clip: tuple, optional, default: None
        The lower and upper bounds to clip the noise to.
    inplace: bool, optional, default: False
        Whether to add the noise in place.
    name: str, optional, default: None
        The name of the new TimeSeries. The default is the name of the original TimeSeries with '_perlin_noise'.
        This is ignored if `inplace` is True.

    Returns
    -------
    tskit.TimeSeries
        The TimeSeries with added noise.
    """
    x = np.arange(len(ts))
    noise = amplitude * (alpha * np.sin(factor_x * x) + beta * np.sin(factor_e * np.e * x) + theta * np.cos(
        factor_pi * np.pi * x))
    if clip is not None:
        noise = np.clip(noise, *clip)
    if inplace:
        ts += noise
        return ts
    return tskit.TimeSeries(
        index=ts.index.copy(),
        values=ts.values + noise,
        name=f'{ts.name}_perlin_noise' if name is None else name,
    )
