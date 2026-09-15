"""Wall-clock profiling for synchronous CPU callables."""

from contextlib import nullcontext
from functools import partial

import torch

from tilelang.profiler._common import bench_wall

SUPPORTED_METHODS = frozenset({"wall"})


def device_scope(device: torch.device):
    return nullcontext()


def synchronize(device: torch.device) -> None:
    pass


def benchmark(fn, *, device: torch.device, **options):
    return bench_wall(fn, synchronize=partial(synchronize, device), **options)
