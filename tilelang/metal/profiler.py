"""Existing Metal benchmark helper, returning seconds per call."""

import time

import torch


def do_bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / repeats
