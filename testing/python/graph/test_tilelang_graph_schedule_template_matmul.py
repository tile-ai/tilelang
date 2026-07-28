"""Matmul template correctness and lowering checks."""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import s_tir, te, tirx as tir

from tilelang.schedule.gpu import IndexedGEMV, Matmul, default_schedule_rules


def _lower_mod(mod):
    mod = tir.transform.Simplify()(mod)
    mod = s_tir.transform.LowerInitBlock()(mod)
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.CanonicalizeScheduledTIR()(mod)
    return mod


def _dtype_tolerance(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return 1e-1, 1e-1
    return 1e-4, 1e-4


def _assert_lowered_ir(lowered_ir: str, label: str, expect_gemm: bool = False) -> None:
    if "T.copy(" not in lowered_ir:
        raise RuntimeError(f"Expected tiled copies in {label}, but none were found.")
    if "local.fragment" not in lowered_ir:
        raise RuntimeError(f"Expected fragment accumulation in {label}, but none was found.")
    if 'scope="shared"' not in lowered_ir and "shared" not in lowered_ir:
        raise RuntimeError(f"Expected shared-memory staging in {label}, but none was found.")
    if expect_gemm and "T.gemm(" not in lowered_ir and "T.gemm_py(" not in lowered_ir:
        raise RuntimeError(f"Expected a tile-level GEMM op in {label}, but none was found.")
    if expect_gemm and '"num_stages":' not in lowered_ir and "'num_stages':" not in lowered_ir:
        raise RuntimeError(f"Expected a pipelined K loop in {label}, but none was found.")
    if not expect_gemm and "T.parallel(" not in lowered_ir:
        raise RuntimeError(f"Expected tile-parallel loops in {label}, but none were found.")
    if "threadIdx.x" not in lowered_ir:
        raise RuntimeError(f"Expected thread launch in {label}, but none was found.")


def _build_primfunc(
    m: int,
    n: int,
    k_extent: int,
    dtype: str,
    batch: int = 1,
    transposed_b: bool = False,
    epilogue: bool = False,
):
    need_fp32_accum = dtype in ("float16", "bfloat16")
    accum_dtype = "float32" if need_fp32_accum else dtype

    if batch == 1:
        a = te.placeholder((m, k_extent), name="A", dtype=dtype)
        if transposed_b:
            b = te.placeholder((n, k_extent), name="B", dtype=dtype)
            k = te.reduce_axis((0, k_extent), name="k")
            c = te.compute(
                (m, n),
                lambda i, j: te.sum(a[i, k].astype(accum_dtype) * b[j, k].astype(accum_dtype), axis=k),
                name="C",
            )
        else:
            b = te.placeholder((k_extent, n), name="B", dtype=dtype)
            k = te.reduce_axis((0, k_extent), name="k")
            c = te.compute(
                (m, n),
                lambda i, j: te.sum(a[i, k].astype(accum_dtype) * b[k, j].astype(accum_dtype), axis=k),
                name="C",
            )
    else:
        a = te.placeholder((batch, m, k_extent), name="A", dtype=dtype)
        if transposed_b:
            b = te.placeholder((batch, n, k_extent), name="B", dtype=dtype)
            k = te.reduce_axis((0, k_extent), name="k")
            c = te.compute(
                (batch, m, n),
                lambda batch_idx, i, j: te.sum(
                    a[batch_idx, i, k].astype(accum_dtype) * b[batch_idx, j, k].astype(accum_dtype),
                    axis=k,
                ),
                name="C",
            )
        else:
            b = te.placeholder((batch, k_extent, n), name="B", dtype=dtype)
            k = te.reduce_axis((0, k_extent), name="k")
            c = te.compute(
                (batch, m, n),
                lambda batch_idx, i, j: te.sum(
                    a[batch_idx, i, k].astype(accum_dtype) * b[batch_idx, k, j].astype(accum_dtype),
                    axis=k,
                ),
                name="C",
            )

    if epilogue:
        if need_fp32_accum:
            d = te.compute(
                c.shape,
                lambda *idx: tir.max(c[idx], tir.const(0.0, accum_dtype)).astype(dtype),
                name="D",
            )
        else:
            d = te.compute(
                c.shape,
                lambda *idx: tir.max(c[idx], tir.const(0.0, dtype)),
                name="D",
            )
        return te.create_prim_func([a, b, d])

    if need_fp32_accum:
        c_cast = te.compute(c.shape, lambda *idx: c[idx].astype(dtype), name="C_cast")
        return te.create_prim_func([a, b, c_cast])
    return te.create_prim_func([a, b, c])


def _build_mod(
    m: int,
    n: int,
    k_extent: int,
    arch: str,
    dtype: str,
    batch: int = 1,
    transposed_b: bool = False,
    epilogue: bool = False,
):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    func = _build_primfunc(m, n, k_extent, dtype, batch, transposed_b, epilogue)
    sch = Matmul().apply(func, target, False)
    if sch is None:
        raise RuntimeError("Matmul schedule rule returned None for this workload.")

    mod = _lower_mod(sch.mod)
    lowered_ir = str(mod)
    label_parts = ["matmul"]
    if batch != 1:
        label_parts.append("batched")
    if transposed_b:
        label_parts.append("transposed_b")
    if epilogue:
        label_parts.append("epilogue")
    label = " ".join(label_parts)
    expect_gemm = dtype != "float32"

    print(f"=== Lowered IR ({label}) ===")
    print(mod)
    _assert_lowered_ir(lowered_ir, label, expect_gemm=expect_gemm)
    return mod


def _build_with_default_rules(m: int, n: int, k_extent: int, arch: str, dtype: str):
    rules = default_schedule_rules()
    if not isinstance(rules[0], IndexedGEMV) or not isinstance(rules[1], Matmul):
        raise RuntimeError("IndexedGEMV and Matmul must lead the GPU default schedule rules.")
    func = _build_primfunc(m, n, k_extent, dtype)
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        mod = s_tir.dlight.ApplyDefaultSchedule(*rules)(tvm.IRModule({"main": func}))
    mod = _lower_mod(mod)
    _assert_lowered_ir(str(mod), "default-rule matmul", expect_gemm=dtype != "float32")
    return mod


def _reference(a_torch, b_torch, transposed_b: bool, epilogue: bool):
    if transposed_b:
        ref = torch.matmul(a_torch, b_torch.transpose(-1, -2))
    else:
        ref = torch.matmul(a_torch, b_torch)
    if epilogue:
        ref = torch.relu(ref)
    return ref


def _compiled_reference(transposed_b: bool, epilogue: bool):
    @torch.compile()
    def _fn(a_torch, b_torch):
        return _reference(a_torch, b_torch, transposed_b, epilogue)

    return _fn


def build_and_run(
    m: int,
    n: int,
    k_extent: int,
    arch: str,
    dtype: str,
    bench_backend: str,
    batch: int = 1,
    transposed_b: bool = False,
    epilogue: bool = False,
) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod(m, n, k_extent, arch, dtype, batch, transposed_b, epilogue)
    kernel = tilelang.compile(mod["main"])

    torch_dtype = getattr(torch, dtype)
    rtol, atol = _dtype_tolerance(dtype)

    torch.manual_seed(0)
    if batch == 1:
        a_shape = (m, k_extent)
        b_shape = (n, k_extent) if transposed_b else (k_extent, n)
        c_shape = (m, n)
    else:
        a_shape = (batch, m, k_extent)
        b_shape = (batch, n, k_extent) if transposed_b else (batch, k_extent, n)
        c_shape = (batch, m, n)

    a_torch = torch.randn(a_shape, device="cuda", dtype=torch_dtype)
    b_torch = torch.randn(b_shape, device="cuda", dtype=torch_dtype)
    c_torch = torch.empty(c_shape, device="cuda", dtype=torch_dtype)

    kernel(a_torch, b_torch, c_torch)

    ref_fn = _compiled_reference(transposed_b, epilogue)
    ref = ref_fn(a_torch, b_torch)
    torch.testing.assert_close(c_torch, ref, rtol=rtol, atol=atol)

    tilelang_time = do_bench(lambda: kernel(a_torch, b_torch, c_torch), backend=bench_backend)
    torch_time = do_bench(lambda: ref_fn(a_torch, b_torch), backend=bench_backend)

    print(f"TileLang time: {tilelang_time:.6f} ms, torch.compile time: {torch_time:.6f} ms")
    return tilelang_time, torch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TileLang matmul scheduling.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--bench-backend", type=str, default="event", choices=["event", "cuda"])
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check schedule + lowering IR; skip CUDA compile/run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        _build_mod(args.m, args.n, args.k, args.arch, args.dtype)
        _build_mod(args.m, args.n, args.k, args.arch, args.dtype, transposed_b=True)
        _build_mod(args.m, args.n, args.k, args.arch, args.dtype, batch=args.batch)
        _build_mod(
            args.m,
            args.n,
            args.k,
            args.arch,
            args.dtype,
            batch=args.batch,
            epilogue=True,
        )
        _build_with_default_rules(args.m, args.n, args.k, args.arch, args.dtype)
        print("\033[92mSchedule/lowering check passed.\033[0m")
    else:
        build_and_run(args.m, args.n, args.k, args.arch, args.dtype, args.bench_backend)
        build_and_run(
            args.m,
            args.n,
            args.k,
            args.arch,
            args.dtype,
            args.bench_backend,
            transposed_b=True,
        )
        build_and_run(
            args.m,
            args.n,
            args.k,
            args.arch,
            args.dtype,
            args.bench_backend,
            batch=args.batch,
        )
        build_and_run(
            args.m,
            args.n,
            args.k,
            args.arch,
            args.dtype,
            args.bench_backend,
            batch=args.batch,
            epilogue=True,
        )
