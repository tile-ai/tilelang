"""Transpose template correctness + performance check.

This script mirrors the workflow used in:
- tests/elementwise/test_tl.py
- tests/reduction/test_reduction.py

It validates:
1. The `Transpose` schedule template compiles and runs.
2. Lowered IR contains shared-memory staging and thread launch.
3. Numerical correctness against `torch.transpose(...).contiguous()`.
4. Runtime and effective memory bandwidth.
"""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import s_tir, te, tirx

from tilelang.schedule.gpu.transpose import Transpose  # pylint: disable=import-outside-toplevel


def _build_mod(m: int, n: int, arch: str):
    """Build and lower a scheduled transpose module."""
    a = te.placeholder((m, n), name="a")
    c = te.compute((n, m), lambda i, j: a[j, i], name="c")
    func = te.create_prim_func([a, c])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = Transpose().apply(func, target, False)
    if sch is None:
        raise RuntimeError("Transpose schedule rule returned None for this workload.")

    print("=== Scheduled IR ===")
    print(sch.mod)

    mod = sch.mod
    mod = tirx.transform.Simplify()(mod)
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.CanonicalizeScheduledTIR()(mod)

    print("=== Lowered IR ===")
    print(mod)

    lowered_ir = str(mod)
    if "T.copy(" not in lowered_ir:
        raise RuntimeError("Expected T.copy in lowered IR, but it was not found.")
    if "shared.dyn" not in lowered_ir:
        raise RuntimeError("Expected shared.dyn buffers in lowered IR, but none were found.")
    if "threadIdx.x" not in lowered_ir:
        raise RuntimeError("Expected a threadIdx.x binding in lowered IR, but it was not found.")

    return mod


def build_and_run(m: int, n: int, arch: str, bench_backend: str) -> tuple[float, float]:
    """Build a transpose kernel, run correctness check, and benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod(m, n, arch)
    kernel = tilelang.compile(mod["main"])

    torch.manual_seed(0)
    a_torch = torch.randn((m, n), device="cuda", dtype=torch.float32)
    c_torch = torch.empty((n, m), device="cuda", dtype=torch.float32)
    kernel(a_torch, c_torch)

    @torch.compile()
    def fn_torch(a):
        return a.transpose(0, 1).contiguous()

    torch.testing.assert_close(c_torch, fn_torch(a_torch), rtol=1e-4, atol=1e-4)
    print("\033[92mCorrectness check passed.\033[0m")

    tilelang_time = do_bench(lambda: kernel(a_torch, c_torch), backend=bench_backend)
    torch_time = do_bench(lambda: fn_torch(a_torch), backend=bench_backend)

    total_bytes = (m * n + m * n) * 4  # fp32 read + write
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")

    return tilelang_time, torch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check transpose template correctness and performance.")
    parser.add_argument("--m", type=int, default=1024, help="Input row count.")
    parser.add_argument("--n", type=int, default=2048, help="Input column count.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check schedule + lowering IR; skip CUDA compile/run/benchmark.",
    )
    parser.add_argument(
        "--bench-backend",
        type=str,
        default="event",
        choices=["event", "cupti"],
        help="Profiler backend used by tilelang.profiler.do_bench.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        _build_mod(args.m, args.n, args.arch)
        print("\033[92mSchedule/lowering check passed.\033[0m")
    else:
        build_and_run(args.m, args.n, args.arch, args.bench_backend)
