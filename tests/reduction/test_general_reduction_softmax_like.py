"""GeneralReduction template check for a multi-reduction softmax-like graph.

Graph:
  rmax[i] = max_j a[i, j]
  rsum[i] = sum_j exp(a[i, j] - rmax[i])
  out[i, j] = exp(a[i, j] - rmax[i]) / rsum[i]
"""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tvm import te, tirx

from tilelang.schedule.gpu.general_reduction import GeneralReduction  # pylint: disable=import-outside-toplevel


def _build_mod(m: int, k: int, arch: str):
    a = te.placeholder((m, k), name="a")
    rk0 = te.reduce_axis((0, k), name="rk0")
    rmax = te.compute((m,), lambda i: te.max(a[i, rk0], axis=rk0), name="rmax")
    expv = te.compute((m, k), lambda i, j: te.exp(a[i, j] - rmax[i]), name="expv")
    rk1 = te.reduce_axis((0, k), name="rk1")
    rsum = te.compute((m,), lambda i: te.sum(expv[i, rk1], axis=rk1), name="rsum")
    out = te.compute((m, k), lambda i, j: expv[i, j] / rsum[i], name="out")
    func = te.create_prim_func([a, out])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = GeneralReduction().apply(func, target, False)
    if sch is None:
        raise RuntimeError("GeneralReduction schedule rule returned None for softmax-like workload.")

    print("=== Scheduled IR ===")
    print(sch.mod)

    mod = sch.mod
    scheduled_ir = str(mod)

    # DSL-generated functions (with alloc_reducer / finalize_reducer) must
    # skip schedule-specific transforms that would strip reducer metadata.
    is_dsl_generated = "finalize_reducer" in scheduled_ir

    if not is_dsl_generated:
        mod = tirx.transform.Simplify()(mod)
        mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(mod)
        mod = tilelang.transform.ReserveRootBlock()(mod)

    print("=== Lowered IR ===")
    print(mod)

    lowered_ir = str(mod)
    # Accept either tilelang tile-level patterns (T.copy / local.fragment)
    # or TVM-style thread-cooperative patterns (threadIdx / shared).
    has_tilelang = "T.copy(" in lowered_ir and "local.fragment" in lowered_ir
    has_tvm_style = "threadIdx" in lowered_ir or "shared" in lowered_ir
    if not has_tilelang and not has_tvm_style:
        raise RuntimeError(
            "Expected either tilelang tile-level patterns or TVM-style thread-cooperative patterns in the lowered IR, but found neither."
        )

    return mod


def build_and_run(m: int, k: int, arch: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")
    mod = _build_mod(m, k, arch)
    kernel = tilelang.compile(
        mod["main"],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    torch.manual_seed(0)
    a_torch = torch.randn((m, k), device="cuda", dtype=torch.float32)
    out_torch = torch.empty((m, k), device="cuda", dtype=torch.float32)
    kernel(a_torch, out_torch)

    @torch.compile()
    def fn_torch(a):
        return torch.softmax(a, dim=1)

    torch.testing.assert_close(out_torch, fn_torch(a_torch), rtol=1e-4, atol=1e-4)
    print("\033[92mCorrectness check passed.\033[0m")

    # Performance
    from tilelang.profiler import do_bench

    tilelang_time = do_bench(lambda: kernel(a_torch, out_torch))
    torch_time = do_bench(lambda: fn_torch(a_torch))
    print(f"TileLang time: {tilelang_time:.2f} ms")
    print(f"PyTorch time: {torch_time:.2f} ms")
    print(f"Speedup: {torch_time / tilelang_time:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GeneralReduction on a softmax-like graph.")
    parser.add_argument("--m", type=int, default=128, help="Batch rows.")
    parser.add_argument("--k", type=int, default=1024, help="Reduction/output columns.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check schedule + lowering IR; skip CUDA compile/run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        _build_mod(args.m, args.k, args.arch)
        print("\033[92mSchedule/lowering check passed.\033[0m")
    else:
        build_and_run(args.m, args.k, args.arch)
