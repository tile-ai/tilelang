"""Compare CuTeDSL host launcher paths with a tiny non-TMA kernel.

This script keeps the TileLang call surface fixed and switches only
TILELANG_CUTEDSL_HOST_LAUNCHER between "cpp" and "cutlass". The CUTLASS path
requires TVM-FFI by default so the comparison does not silently fall back to a
Python callable. Use an external timeline profiler such as veloq/nsys around
this process to inspect CPU/GPU bubbles; the script adds NVTX ranges around
each measured loop.

Example:
  TILELANG_DISABLE_CACHE=1 python maint/host_checks/cutedsl_launcher_overhead.py --mode both
"""

from __future__ import annotations

import argparse
import os
import statistics
from contextlib import contextmanager

import torch

import tilelang
import tilelang.language as T


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def make_add_kernel(m: int, n: int, block_m: int, block_n: int, threads: int):
    @T.prim_func
    def add_kernel(
        A: T.Tensor((m, n), T.float32),
        B: T.Tensor((m, n), T.float32),
        C: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
            for local_y, local_x in T.Parallel(block_m, block_n):
                y = by * block_m + local_y
                x = bx * block_n + local_x
                if y < m and x < n:
                    C[y, x] = A[y, x] + B[y, x]

    return add_kernel


def run_mode(fn, args, mode: str, warmup: int, repeat: int) -> float:
    os.environ["TILELANG_CUTEDSL_HOST_LAUNCHER"] = mode
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with nvtx_range(f"tilelang_cutedsl_launcher_{mode}"):
        start.record()
        for _ in range(repeat):
            fn(*args)
        end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpp", "cutlass", "both"), default="both")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=10000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ.setdefault("TILELANG_CUTEDSL_REQUIRE_TVMFFI", "1")

    program = make_add_kernel(args.m, args.n, args.block_m, args.block_n, args.threads)
    fn = tilelang.compile(program, target="cutedsl")

    a = torch.randn((args.m, args.n), device="cuda", dtype=torch.float32)
    b = torch.randn((args.m, args.n), device="cuda", dtype=torch.float32)
    c = torch.empty((args.m, args.n), device="cuda", dtype=torch.float32)
    fn(a, b, c)
    torch.testing.assert_close(c, a + b)

    modes = ("cpp", "cutlass") if args.mode == "both" else (args.mode,)
    samples: dict[str, list[float]] = {}
    for mode in modes:
        samples[mode] = []
        for _ in range(5):
            samples[mode].append(run_mode(fn, (a, b, c), mode, args.warmup, args.repeat))

    for mode in modes:
        values = samples[mode]
        print(
            f"{mode}: mean={statistics.mean(values):.3f} us "
            f"median={statistics.median(values):.3f} us "
            f"min={min(values):.3f} us samples={values}"
        )


if __name__ == "__main__":
    main()
