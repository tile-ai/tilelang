"""oneDNN BF16 and INT8 throughput comparison for ``benchmark_amx.cpp``.

PyTorch's CPU BF16 matmul dispatches to oneDNN on AMX-capable Intel CPUs.  Run
this script under ``taskset`` with one hardware thread per physical core so
that placement and ``--threads`` match the C++ helper benchmark.

The PyTorch BF16 operation stores BF16 output, whereas TileLang stores the FP32
accumulator. This script compares throughput rather than output contracts.
"""

from __future__ import annotations

import argparse
import time

import torch


SHAPES = (
    (16, 16, 32),
    (32, 32, 64),
    (64, 64, 64),
    (64, 64, 256),
    (128, 128, 128),
    (256, 256, 256),
    (128, 1024, 4096),
)


def measure(function, operations: int) -> float:
    for _ in range(10):
        function()
    iterations = 1
    elapsed = 0.0
    while elapsed < 0.2:
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        elapsed = time.perf_counter() - start
        if elapsed < 0.2:
            iterations *= 2
    return operations * iterations / elapsed / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--shape", nargs=3, type=int, action="append", metavar=("M", "N", "K"))
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    shapes = tuple(map(tuple, args.shape)) if args.shape else SHAPES
    for m, n, k in shapes:
        a = torch.randn((m, k), dtype=torch.bfloat16)
        b = torch.randn((k, n), dtype=torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16)
        rate = measure(lambda a=a, b=b, out=out: torch.mm(a, b, out=out), 2 * m * n * k)
        print(f"BF16 {m:4d}x{n:4d}x{k:4d}  {rate:8.2f} GFLOP/s")

    if not hasattr(torch, "_int_mm"):
        return
    for m, n, k in shapes:
        a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        b = torch.randint(-8, 8, (k, n), dtype=torch.int8)
        out = torch.empty((m, n), dtype=torch.int32)
        try:
            rate = measure(lambda a=a, b=b, out=out: torch._int_mm(a, b, out=out), 2 * m * n * k)
        except RuntimeError as error:
            print(f"INT8 reference unavailable: {error}")
            break
        print(f"INT8  {m:4d}x{n:4d}x{k:4d}  {rate:8.2f} GOP/s")


if __name__ == "__main__":
    main()
