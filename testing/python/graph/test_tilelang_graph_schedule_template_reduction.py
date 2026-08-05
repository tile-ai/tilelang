"""Reduction template correctness + performance check.

This script mirrors the workflow used in:
- tests/elementwise/test_tl.py
- testing/python/graph/test_tilelang_graph_schedule_template_transpose.py

It validates:
1. The `Reduction` schedule template compiles and runs.
2. Numerical correctness against `torch.sum`.
3. Runtime and effective memory bandwidth.
"""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import s_tir, te, tirx

from tilelang.schedule.gpu.reduction import Reduction  # pylint: disable=import-outside-toplevel


def _build_mod(m: int, n: int, k: int, arch: str):
    """Build and lower a scheduled reduction module."""
    # Define TE reduction: c[i] = sum_j a[i, j]
    a = te.placeholder((m, n, k), name="a")
    rk = te.reduce_axis((0, k), name="rk")
    c = te.compute(
        (
            m,
            n,
        ),
        lambda i, j: te.max(a[i, j, rk], axis=rk),
        name="c",
    )

    func = te.create_prim_func([a, c])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = Reduction().apply(func, target, None)
    if sch is None:
        raise RuntimeError("Reduction schedule rule returned None for this workload.")

    print("=== Scheduled IR ===")
    print(sch.mod)

    # Keep the same style as elementwise/transpose prototype tests.
    mod = sch.mod
    mod = tirx.transform.Simplify()(mod)
    mod = s_tir.transform.LowerInitBlock()(mod)
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.CanonicalizeScheduledTIR()(mod)

    print("=== Lowered IR ===")
    print(mod)

    lowered_ir = str(mod)
    if "T.copy(" not in lowered_ir:
        raise RuntimeError("Expected T.copy in lowered IR, but it was not found.")
    if "T.reduce(" not in lowered_ir:
        raise RuntimeError("Expected T.reduce in lowered IR, but it was not found.")
    if "local.fragment" not in lowered_ir:
        raise RuntimeError("Expected local.fragment buffers in lowered IR, but none were found.")

    return mod


def build_and_run(m: int, n: int, k: int, arch: str, bench_backend: str) -> tuple[float, float]:
    """Build a reduction kernel, run correctness check, and benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")
    mod = _build_mod(m, n, k, arch)
    kernel = tilelang.compile(mod["main"])

    # Correctness
    torch.manual_seed(0)
    a_torch = torch.randn((m, n, k), device="cuda", dtype=torch.float32)
    # b_torch = torch.randn((m, n, k), device="cuda", dtype=torch.float32)
    c_torch = torch.empty((m, n), device="cuda", dtype=torch.float32)
    kernel(a_torch, c_torch)

    @torch.compile()
    def fntorch(a):
        return torch.max(a, dim=2)[0]

    # ref = torch.sum(a_torch, dim=1)
    torch.testing.assert_close(c_torch, fntorch(a_torch), rtol=1e-4, atol=1e-4)
    print("\033[92mCorrectness check passed.\033[0m")

    # Performance
    tilelang_time = do_bench(lambda: kernel(a_torch, c_torch), backend=bench_backend)
    torch_time = do_bench(lambda: fntorch(a_torch), backend=bench_backend)

    total_bytes = (m * n * k + m * n) * 4  # fp32 read + write
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")

    return tilelang_time, torch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check reduction template correctness and performance.")
    parser.add_argument("--m", type=int, default=1024, help="Output length / first input dimension.")
    parser.add_argument("--n", type=int, default=16, help="Output length / second input dimension.")
    parser.add_argument("--k", type=int, default=2048, help="Reduction length / third input dimension.")
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
        _build_mod(args.m, args.n, args.k, args.arch)
        print("\033[92mSchedule/lowering check passed.\033[0m")
    else:
        build_and_run(args.m, args.n, args.k, args.arch, args.bench_backend)
