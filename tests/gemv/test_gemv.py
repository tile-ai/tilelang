"""GEMV template correctness + performance check.

Validates:
1. The `GEMV` schedule template lowers GEMV TE expressions to tile-style IR.
2. Numerical correctness against PyTorch.
3. Performance comparison against torch.compile.
"""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import s_tir, te, tirx as tir

from tilelang.schedule.gpu.gemv import GEMV


def _lower_mod(mod):
    """Normalize scheduled module into a legal tilelang program.

    Pipeline:
      Simplify → ConvertBlocksToOpaque → CanonicalizeScheduledTIR

    cache_write_at detects init blocks (from decompose_reduction) and
    replaces them with T.fill in-place, so no separate StripBlockInit pass
    is needed.
    """
    mod = tir.transform.Simplify()(mod)
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.CanonicalizeScheduledTIR()(mod)
    return mod


def _dtype_tolerance(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return 1e-1, 1e-1
    return 1e-3, 1e-3


def _assert_lowered_ir(lowered_ir: str, label: str) -> None:
    has_tilelang = "T.copy(" in lowered_ir and "local.fragment" in lowered_ir
    has_thread_allreduce = "tvm_thread_allreduce" in lowered_ir
    if not has_tilelang and not has_thread_allreduce:
        raise RuntimeError(f"Expected either tile-style staging or tvm_thread_allreduce in {label} lowered IR, but found neither.")
    if not has_thread_allreduce and "T.parallel(" not in lowered_ir and "threadIdx.x" not in lowered_ir:
        raise RuntimeError(
            f"Expected either tile-level T.parallel loops or thread-bound lowering in {label} lowered IR, but found neither."
        )


def _build_mod(M: int, N: int, arch: str, dtype: str):
    """Build a scheduled GEMV module: C[i] = sum_k A[i, k] * B[k]."""
    A = te.placeholder((M, N), name="A", dtype=dtype)
    B = te.placeholder((N,), name="B", dtype=dtype)
    k = te.reduce_axis((0, N), name="k")

    if dtype in ("float16", "bfloat16"):
        C_accum = te.compute(
            (M,),
            lambda i: te.sum(A[i, k].astype("float32") * B[k].astype("float32"), axis=k),
            name="C",
        )
        C = te.compute((M,), lambda i: C_accum[i].astype(dtype), name="C_cast")
    else:
        C = te.compute((M,), lambda i: te.sum(A[i, k] * B[k], axis=k), name="C")

    func = te.create_prim_func([A, B, C])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = GEMV().apply(func, target, False)
    if sch is None:
        raise RuntimeError("GEMV schedule rule returned None for this workload.")

    print("=== Scheduled IR ===")
    print(sch.mod)

    mod = _lower_mod(sch.mod)

    print("=== Lowered IR ===")
    print(mod)

    lowered_ir = str(mod)
    _assert_lowered_ir(lowered_ir, "standard GEMV")

    return mod


def _build_mod_transposed(M: int, N: int, arch: str, dtype: str):
    """Build GEMV with transposed weight: C[i] = sum_k A[k] * B[i, k]."""
    A = te.placeholder((N,), name="A", dtype=dtype)
    B = te.placeholder((M, N), name="B", dtype=dtype)
    k = te.reduce_axis((0, N), name="k")

    if dtype in ("float16", "bfloat16"):
        C_accum = te.compute(
            (M,),
            lambda i: te.sum(A[k].astype("float32") * B[i, k].astype("float32"), axis=k),
            name="C",
        )
        C = te.compute((M,), lambda i: C_accum[i].astype(dtype), name="C_cast")
    else:
        C = te.compute((M,), lambda i: te.sum(A[k] * B[i, k], axis=k), name="C")

    func = te.create_prim_func([A, B, C])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = GEMV().apply(func, target, False)
    if sch is None:
        raise RuntimeError("GEMV schedule rule returned None for transposed workload.")

    print("=== Scheduled IR (transposed) ===")
    print(sch.mod)

    mod = _lower_mod(sch.mod)

    print("=== Lowered IR (transposed) ===")
    print(mod)

    lowered_ir = str(mod)
    _assert_lowered_ir(lowered_ir, "transposed GEMV")

    return mod


def _build_mod_batched(B: int, M: int, N: int, arch: str, dtype: str):
    """Build batched GEMV: C[b, i] = sum_k A[b, i, k] * X[b, k]."""
    A_t = te.placeholder((B, M, N), name="A", dtype=dtype)
    X_t = te.placeholder((B, N), name="X", dtype=dtype)
    k = te.reduce_axis((0, N), name="k")

    if dtype in ("float16", "bfloat16"):
        C_accum = te.compute(
            (B, M),
            lambda b, i: te.sum(A_t[b, i, k].astype("float32") * X_t[b, k].astype("float32"), axis=k),
            name="C",
        )
        C_t = te.compute((B, M), lambda b, i: C_accum[b, i].astype(dtype), name="C_cast")
    else:
        C_t = te.compute((B, M), lambda b, i: te.sum(A_t[b, i, k] * X_t[b, k], axis=k), name="C")

    func = te.create_prim_func([A_t, X_t, C_t])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = GEMV().apply(func, target, False)
    if sch is None:
        raise RuntimeError("GEMV schedule rule returned None for batched workload.")

    print("=== Scheduled IR (batched) ===")
    print(sch.mod)

    mod = _lower_mod(sch.mod)

    print("=== Lowered IR (batched) ===")
    print(mod)

    lowered_ir = str(mod)
    _assert_lowered_ir(lowered_ir, "batched GEMV")

    return mod


def _build_mod_epilog(B: int, M: int, N: int, arch: str, dtype: str):
    """Build batched GEMV with epilog: D[b, i] = sum_k A[b, i, k] * X[b, k] + 1.0."""
    A_t = te.placeholder((B, M, N), name="A", dtype=dtype)
    X_t = te.placeholder((B, N), name="X", dtype=dtype)
    k = te.reduce_axis((0, N), name="k")

    if dtype in ("float16", "bfloat16"):
        C_t = te.compute(
            (B, M),
            lambda b, i: te.sum(A_t[b, i, k].astype("float32") * X_t[b, k].astype("float32"), axis=k),
            name="C",
        )
        D_t = te.compute(
            (B, M),
            lambda b, i: C_t[b, i].astype(dtype) + tir.const(1.0, dtype),
            name="D",
        )
    else:
        C_t = te.compute((B, M), lambda b, i: te.sum(A_t[b, i, k] * X_t[b, k], axis=k), name="C")
        D_t = te.compute((B, M), lambda b, i: C_t[b, i] + tir.const(1.0, dtype), name="D")

    func = te.create_prim_func([A_t, X_t, D_t])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = GEMV().apply(func, target, False)
    if sch is None:
        return None

    print("=== Scheduled IR (batched) ===")
    print(sch.mod)

    mod = _lower_mod(sch.mod)

    print("=== Lowered IR (batched) ===")
    print(mod)

    lowered_ir = str(mod)
    _assert_lowered_ir(lowered_ir, "batched GEMV")

    return mod


def build_and_run(M: int, N: int, arch: str, bench_backend: str, dtype: str) -> tuple[float, float]:
    """Build a GEMV kernel, run correctness check, and benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod(M, N, arch, dtype)
    kernel = tilelang.compile(mod["main"], compile_flags=GEMV.COMPILE_FLAGS)

    torch_dtype = getattr(torch, dtype)
    rtol, atol = _dtype_tolerance(dtype)

    torch.manual_seed(0)
    A_torch = torch.randn((M, N), device="cuda", dtype=torch_dtype)
    B_torch = torch.randn((N,), device="cuda", dtype=torch_dtype)
    C_torch = torch.empty((M,), device="cuda", dtype=torch_dtype)

    kernel(A_torch, B_torch, C_torch)

    @torch.compile()
    def fn_torch(A, B):
        return A @ B

    ref = fn_torch(A_torch, B_torch)
    torch.testing.assert_close(C_torch, ref, rtol=rtol, atol=atol)
    print("\033[92mCorrectness check passed (standard GEMV).\033[0m")

    # Benchmark
    torch_time = do_bench(lambda: fn_torch(A_torch, B_torch), backend=bench_backend)
    tilelang_time = do_bench(lambda: kernel(A_torch, B_torch, C_torch), backend=bench_backend)
    torch_no_compile = do_bench(lambda: A_torch @ B_torch, backend=bench_backend)

    total_bytes = (M * N + N + M) * (torch.tensor([], dtype=torch_dtype).element_size())
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Torch (no compile) time: {torch_no_compile:.6f} ms, BW: {total_bytes / (torch_no_compile * 1e-3) / 1e9:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")
    return tilelang_time, torch_time


def build_and_run_transposed(M: int, N: int, arch: str, bench_backend: str, dtype: str) -> tuple[float, float]:
    """Build transposed GEMV: y = W @ x where W is [M, N], x is [N]."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod_transposed(M, N, arch, dtype)
    kernel = tilelang.compile(mod["main"], compile_flags=GEMV.COMPILE_FLAGS)

    torch_dtype = getattr(torch, dtype)
    rtol, atol = _dtype_tolerance(dtype)

    torch.manual_seed(42)
    x_torch = torch.randn((N,), device="cuda", dtype=torch_dtype)
    W_torch = torch.randn((M, N), device="cuda", dtype=torch_dtype)
    y_torch = torch.empty((M,), device="cuda", dtype=torch_dtype)

    kernel(x_torch, W_torch, y_torch)

    @torch.compile()
    def fn_torch(x, W):
        return W @ x

    ref = fn_torch(x_torch, W_torch)
    torch.testing.assert_close(y_torch, ref, rtol=rtol, atol=atol)
    print("\033[92mCorrectness check passed (transposed GEMV).\033[0m")

    tilelang_time = do_bench(lambda: kernel(x_torch, W_torch, y_torch), backend=bench_backend)
    torch_time = do_bench(lambda: fn_torch(x_torch, W_torch), backend=bench_backend)

    total_bytes = (M * N + N + M) * (torch.tensor([], dtype=torch_dtype).element_size())
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")
    return tilelang_time, torch_time


def build_and_run_epilog(B: int, M: int, N: int, arch: str, bench_backend: str, dtype: str) -> tuple[float, float]:
    """Build and run batched GEMV with epilog: D[b, i] = sum_k A[b, i, k] * X[b, k] + 1.0."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod_epilog(B, M, N, arch, dtype)
    if mod is None:
        print("\033[93mSkipped (epilogue not supported in tile schedule path).\033[0m")
        return 0.0, 0.0
    kernel = tilelang.compile(mod["main"], compile_flags=GEMV.COMPILE_FLAGS)

    print(kernel.get_kernel_source())

    torch_dtype = getattr(torch, dtype)
    rtol, atol = _dtype_tolerance(dtype)

    torch.manual_seed(123)
    A_torch = torch.randn((B, M, N), device="cuda", dtype=torch_dtype)
    X_torch = torch.randn((B, N), device="cuda", dtype=torch_dtype)
    D_torch = torch.empty((B, M), device="cuda", dtype=torch_dtype)

    kernel(A_torch, X_torch, D_torch)

    @torch.compile()
    def fn_torch(A, X):
        return A @ X.unsqueeze(2) + 1.0

    ref = fn_torch(A_torch, X_torch).squeeze(2)
    torch.testing.assert_close(D_torch, ref, rtol=rtol, atol=atol)
    print("\033[92mCorrectness check passed (batched GEMV with epilog).\033[0m")

    tilelang_time = do_bench(lambda: kernel(A_torch, X_torch, D_torch), backend=bench_backend)
    torch_time = do_bench(lambda: fn_torch(A_torch, X_torch), backend=bench_backend)

    total_bytes = (B * M * N + B * N + B * M) * (torch.tensor([], dtype=torch_dtype).element_size())
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")
    return tilelang_time, torch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GEMV template correctness and performance.")
    parser.add_argument("--m", type=int, default=4096, help="Number of output elements (rows of weight matrix).")
    parser.add_argument("--n", type=int, default=4096, help="Reduction dimension (cols of weight matrix).")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Input/output dtype used for GEMV correctness and benchmarking.",
    )
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
        print("--- Standard GEMV ---")
        _build_mod(args.m, args.n, args.arch, args.dtype)
        print("\033[92mSchedule/lowering check passed (standard).\033[0m\n")

        print("--- Transposed GEMV ---")
        _build_mod_transposed(args.m, args.n, args.arch, args.dtype)
        print("\033[92mSchedule/lowering check passed (transposed).\033[0m\n")

        print("--- Batched GEMV ---")
        _build_mod_batched(2, args.m, args.n, args.arch, args.dtype)
        print("\033[92mSchedule/lowering check passed (batched).\033[0m\n")

        print("--- Batched GEMV with Epilog ---")
        mod = _build_mod_epilog(2, args.m, args.n, args.arch, args.dtype)
        if mod is None:
            print("\033[93mSkipped (epilogue not supported in tile schedule path).\033[0m\n")
        else:
            print("\033[92mSchedule/lowering check passed (batched epilog).\033[0m\n")
    else:
        print("=== Standard GEMV ===")
        build_and_run(args.m, args.n, args.arch, args.bench_backend, args.dtype)
        print()
        print("=== Transposed GEMV ===")
        build_and_run_transposed(args.m, args.n, args.arch, args.bench_backend, args.dtype)
        print()
        print("=== Batched GEMV with Epilog ===")
        build_and_run_epilog(4, args.m, args.n, args.arch, args.bench_backend, args.dtype)
