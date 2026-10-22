"""Wall-clock benchmarking with optional explicit device synchronization."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, median_low
from time import perf_counter
from typing import Literal


def bench_with_wall(
    fn: Callable,
    n_repeat: int,
    quantiles: list[float] | None = None,
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
    synchronize: Callable | None = None,
) -> float | list[float]:
    """Time each call in milliseconds, including optional synchronization.

    Synchronize before starting each sample to exclude preceding device work,
    then wait for the timed call to complete before stopping the clock.
    Quantiles use linear interpolation; median returns the lower middle sample,
    matching the GPU benchmark's aggregation conventions.
    """
    times = []
    for _ in range(n_repeat):
        if synchronize is not None:
            synchronize()
        start = perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        times.append((perf_counter() - start) * 1e3)

    if quantiles is not None:
        ordered = sorted(times)
        values = []
        for quantile in quantiles:
            position = quantile * (len(ordered) - 1)
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            values.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
        return values[0] if len(values) == 1 else values
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "median":
        return median_low(times)
    return fmean(times)
