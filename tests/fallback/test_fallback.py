"""Generic GPU fallback correctness checks.

Validates:
1. `default_schedule_rules()` keeps `Fallback` as the last rule, so
   multi-output injective graphs can still be scheduled after finer-grained
   templates decline them.
2. `Fallback` directly handles reduction + epilogue graphs and lowers them to
   compilable TileLang IR.
"""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tvm import s_tir, te, tirx as tir

from tilelang.schedule.gpu import Fallback, default_schedule_rules


def _lower_mod(mod):
    mod = tir.transform.Simplify()(mod)
    mod = s_tir.transform.LowerInitBlock()(mod)
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)
    return mod


def _dtype_tolerance(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return 1e-1, 1e-1
    return 1e-4, 1e-4


def _assert_threaded_ir(lowered_ir: str, label: str, expect_onchip_stage: bool = False) -> None:
    if 'launch_thread("threadIdx.x"' not in lowered_ir and "threadIdx.x" not in lowered_ir:
        raise RuntimeError(f"Expected thread binding in {label}, but none was found.")
    if expect_onchip_stage and "local.fragment" not in lowered_ir and 'scope="shared"' not in lowered_ir:
        raise RuntimeError(f"Expected on-chip staging in {label}, but none was found.")


def _build_multi_output_default_stack_mod(m: int, n: int, arch: str):
    a = te.placeholder((m, n), name="A", dtype="float32")
    b = te.compute((m, n), lambda i, j: a[i, j] + tir.const(1.0, "float32"), name="B")
    c = te.compute((m, n), lambda i, j: a[i, j] * tir.const(2.0, "float32"), name="C")

    rules = default_schedule_rules()
    if not isinstance(rules[-1], Fallback):
        raise RuntimeError("default_schedule_rules() must keep Fallback as the final rule.")

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        mod = s_tir.dlight.ApplyDefaultSchedule(*rules)(tvm.IRModule({"main": te.create_prim_func([a, b, c])}))

    mod = _lower_mod(mod)
    print("=== Lowered IR (default-stack multi-output) ===")
    print(mod)
    _assert_threaded_ir(str(mod), "default-stack multi-output")
    return mod


def _build_direct_fallback_epilogue_mod(batch: int, m: int, k_extent: int, arch: str, dtype: str):
    a = te.placeholder((batch, m, k_extent), name="A", dtype=dtype)
    x = te.placeholder((batch, k_extent), name="X", dtype=dtype)
    k = te.reduce_axis((0, k_extent), name="k")
    c = te.compute((batch, m), lambda b, i: te.sum(a[b, i, k] * x[b, k], axis=k), name="C")
    d = te.compute((batch, m), lambda b, i: c[b, i] + tir.const(1.0, dtype), name="D")

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = Fallback().apply(te.create_prim_func([a, x, d]), target, False)
    if sch is None:
        raise RuntimeError("Fallback schedule rule returned None for reduction + epilogue workload.")

    mod = _lower_mod(sch.mod)
    print("=== Lowered IR (direct fallback epilogue) ===")
    print(mod)
    _assert_threaded_ir(str(mod), "direct fallback epilogue", expect_onchip_stage=True)
    return mod


def build_and_run_multi_output_default_stack(m: int, n: int, arch: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_multi_output_default_stack_mod(m, n, arch)
    kernel = tilelang.compile(mod["main"])

    torch.manual_seed(0)
    a_t = torch.randn((m, n), device="cuda", dtype=torch.float32)
    b_t = torch.empty((m, n), device="cuda", dtype=torch.float32)
    c_t = torch.empty((m, n), device="cuda", dtype=torch.float32)

    kernel(a_t, b_t, c_t)

    torch.testing.assert_close(b_t, a_t + 1.0, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(c_t, a_t * 2.0, rtol=1e-4, atol=1e-4)
    print("\033[92mCorrectness check passed (default-stack multi-output).\033[0m")


def build_and_run_direct_fallback_epilogue(
    batch: int,
    m: int,
    k_extent: int,
    arch: str,
    dtype: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_direct_fallback_epilogue_mod(batch, m, k_extent, arch, dtype)
    kernel = tilelang.compile(mod["main"])

    torch_dtype = getattr(torch, dtype)
    rtol, atol = _dtype_tolerance(dtype)

    torch.manual_seed(1)
    a_t = torch.randn((batch, m, k_extent), device="cuda", dtype=torch_dtype)
    x_t = torch.randn((batch, k_extent), device="cuda", dtype=torch_dtype)
    d_t = torch.empty((batch, m), device="cuda", dtype=torch_dtype)

    kernel(a_t, x_t, d_t)

    ref = torch.sum(a_t * x_t[:, None, :], dim=2) + 1.0
    torch.testing.assert_close(d_t, ref, rtol=rtol, atol=atol)
    print("\033[92mCorrectness check passed (direct fallback epilogue).\033[0m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generic GPU fallback scheduling.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument("--m", type=int, default=64, help="Row extent for the multi-output injective case.")
    parser.add_argument("--n", type=int, default=128, help="Column extent for the multi-output injective case.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size for the reduction + epilogue case.")
    parser.add_argument("--epilogue-m", type=int, default=128, help="Output extent for the reduction + epilogue case.")
    parser.add_argument("--k", type=int, default=256, help="Reduction extent for the reduction + epilogue case.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check schedule + lowering IR; skip CUDA compile/run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        _build_multi_output_default_stack_mod(args.m, args.n, args.arch)
        _build_direct_fallback_epilogue_mod(args.batch, args.epilogue_m, args.k, args.arch, args.dtype)
        print("\033[92mSchedule/lowering check passed.\033[0m")
    else:
        build_and_run_multi_output_default_stack(args.m, args.n, args.arch)
        build_and_run_direct_fallback_epilogue(
            args.batch,
            args.epilogue_m,
            args.k,
            args.arch,
            args.dtype,
        )
