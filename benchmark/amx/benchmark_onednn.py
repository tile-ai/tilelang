"""Single-core oneDNN reference for ``benchmark_amx.cpp``.

PyTorch's CPU BF16 matmul dispatches to oneDNN on AMX-capable Intel CPUs.  Run
this script under ``taskset -c <physical-core>`` so that thread placement and
the thread count match the C++ helper benchmark.
"""

from __future__ import annotations

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
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for m, n, k in SHAPES:
        a = torch.randn((m, k), dtype=torch.bfloat16)
        b = torch.randn((k, n), dtype=torch.bfloat16)
        rate = measure(lambda a=a, b=b: torch.mm(a, b), 2 * m * n * k)
        print(f"BF16 {m:4d}x{n:4d}x{k:4d}  {rate:8.2f} GFLOP/s")

    if not hasattr(torch, "_int_mm"):
        return
    for m, n, k in SHAPES:
        a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        b = torch.randint(-8, 8, (k, n), dtype=torch.int8)
        try:
            rate = measure(lambda a=a, b=b: torch._int_mm(a, b), 2 * m * n * k)
        except RuntimeError as error:
            print(f"INT8 reference unavailable: {error}")
            break
        print(f"INT8  {m:4d}x{n:4d}x{k:4d}  {rate:8.2f} GOP/s")


if __name__ == "__main__":
    main()
