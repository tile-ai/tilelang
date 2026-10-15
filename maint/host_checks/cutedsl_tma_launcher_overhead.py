"""Compare CuTeDSL host launcher paths with a small tiled TMA copy kernel.

This script switches only TILELANG_CUTEDSL_HOST_LAUNCHER between "cpp" and
"cutlass". The CUTLASS path requires both direct CUTLASS host launch support
and TVM-FFI conversion by default, so overhead measurements do not silently
fall back to the generated C++ launcher or a Python callable.
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


def make_tma_copy_kernel(m: int, n: int, block_m: int, block_n: int, threads: int):
    @T.prim_func
    def tma_copy_kernel(
        A: T.Tensor((m, n), T.float16),
        B: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_n), T.float16)
            loaded = T.alloc_barrier([32])
            tx = T.get_thread_binding()

            T.use_swizzle(8)

            if tx < 32:
                T.tma_copy(
                    A[
                        by * block_m : (by + 1) * block_m,
                        bx * block_n : (bx + 1) * block_n,
                    ],
                    A_shared,
                    barrier=loaded,
                )
                T.mbarrier_arrive(loaded)

            T.mbarrier_wait_parity(loaded, 0)
            for local_y, local_x in T.Parallel(block_m, block_n):
                B[by * block_m + local_y, bx * block_n + local_x] = A_shared[local_y, local_x]

    return tma_copy_kernel


def run_mode(fn, args, mode: str, warmup: int, repeat: int) -> float:
    os.environ["TILELANG_CUTEDSL_HOST_LAUNCHER"] = mode
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with nvtx_range(f"tilelang_cutedsl_tma_launcher_{mode}"):
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
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=4)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=10000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.threads < 64:
        raise ValueError("--threads must be at least 64 for the two-warp TMA copy kernel")

    os.environ.setdefault("TILELANG_CUTEDSL_REQUIRE_TVMFFI", "1")
    os.environ.setdefault("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")

    program = make_tma_copy_kernel(args.m, args.n, args.block_m, args.block_n, args.threads)
    fn = tilelang.compile(
        program,
        target="cutedsl",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )

    a = torch.randn((args.m, args.n), device="cuda", dtype=torch.float16)
    b = torch.empty((args.m, args.n), device="cuda", dtype=torch.float16)
    fn(a, b)
    torch.testing.assert_close(b, a, rtol=0, atol=0)

    modes = ("cpp", "cutlass") if args.mode == "both" else (args.mode,)
    samples: dict[str, list[float]] = {}
    for mode in modes:
        samples[mode] = []
        for _ in range(5):
            samples[mode].append(run_mode(fn, (a, b), mode, args.warmup, args.repeat))

    for mode in modes:
        values = samples[mode]
        print(
            f"{mode}: mean={statistics.mean(values):.3f} us "
            f"median={statistics.median(values):.3f} us "
            f"min={min(values):.3f} us samples={values}"
        )

    pymodule = fn.adapter.pymodule
    print(f"has_tma_descs={getattr(pymodule, '_has_tma_descs', None)}")
    print(f"cutlass_host_launcher_supported={getattr(pymodule, '_cutlass_host_launcher_supported', None)}")
    print(f"cutlass_tvmffi_launcher={type(getattr(pymodule, '_cutlass_tvmffi_launcher', None)).__name__}")


if __name__ == "__main__":
    main()
