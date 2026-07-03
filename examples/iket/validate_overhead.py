#!/usr/bin/env python3
"""Validate TileLang IKET correctness and runtime overhead on representative kernels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


THREADS = 128
GEMM_BLOCK_M = 128
GEMM_BLOCK_N = 128
GEMM_BLOCK_K = 16


@dataclass
class ValidationResult:
    name: str
    workload: str
    baseline_ms: float
    iket_ms: float
    overhead_pct: float
    max_abs_error: float
    correctness: str
    iket_features: str


def vector_add_kernel(n: int, *, instrument: bool, payloads: bool, threads: int = THREADS):
    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
        C: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
            if instrument:
                if payloads:
                    T.iket.range_push("vec_block", payload=T.iket.payload(bx, dtype="int32"))
                else:
                    T.iket.range_push("vec_block")
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n:
                    if instrument:
                        T.iket.mark("vec_load")
                    C[idx] = A[idx] + B[idx]
                    if instrument:
                        if payloads:
                            T.iket.mark("vec_store", payload=T.iket.payload(idx, dtype="int32"))
                        else:
                            T.iket.mark("vec_store")
            if instrument:
                T.iket.range_pop("vec_block")

    return main


def add_scale_kernel(n: int, *, instrument: bool, payloads: bool, threads: int = THREADS):
    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
        Scale: T.Tensor((1,), T.float32),
        C: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
            if instrument:
                T.iket.range_push("scale_block")
                if payloads:
                    T.iket.mark("block_enter", payload=T.iket.payload(bx, dtype="int32"))
                else:
                    T.iket.mark("block_enter")
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n:
                    if instrument:
                        T.iket.mark("load_inputs")
                    C[idx] = (A[idx] + B[idx]) * Scale[0]
                    if instrument:
                        if payloads:
                            T.iket.mark("store_index", payload=T.iket.payload(idx, dtype="int32"))
                        else:
                            T.iket.mark("store_done")
            if instrument:
                if payloads:
                    T.iket.mark("block_exit", payload=T.iket.payload(bx, dtype="int32"))
                else:
                    T.iket.mark("block_exit")
                T.iket.range_pop("scale_block")

    return main


def gemm_kernel(
    m: int,
    n: int,
    k: int,
    *,
    instrument: bool,
    payloads: bool,
):
    thread_row_tiles = 16
    thread_col_tiles = 16

    block_m = GEMM_BLOCK_M
    block_n = GEMM_BLOCK_N
    block_k = GEMM_BLOCK_K
    threads = thread_row_tiles * thread_col_tiles
    local_size_a = block_m // thread_row_tiles
    local_size_b = block_n // thread_col_tiles
    local_size_c = local_size_a * local_size_b
    micro_size_k = 8
    vector_width = 4

    @T.prim_func
    def main(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((n, k), T.float16),
        C: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_n, block_k), T.float16)

            A_local = T.alloc_local((local_size_a, micro_size_k), T.float16)
            B_local = T.alloc_local((local_size_b, micro_size_k), T.float16)
            C_local = T.alloc_local((local_size_c,), T.float32)

            tid = T.get_thread_binding()
            warp_m = tid % thread_row_tiles
            warp_n = tid // thread_row_tiles

            if instrument:
                T.iket.range_push("gemm_tile")
                if payloads:
                    tile_id = by * T.ceildiv(n, block_n) + bx
                    T.iket.mark("tile_begin", payload=T.iket.payload(tile_id, dtype="int32"))
                else:
                    T.iket.mark("tile_begin")

            T.clear(C_local)
            for ko in T.serial(k // block_k):
                for i, kk in T.Parallel(block_m, block_k):
                    A_shared[i, kk] = A[by * block_m + i, ko * block_k + kk]

                for j, kk in T.Parallel(block_n, block_k):
                    B_shared[j, kk] = B[bx * block_n + j, ko * block_k + kk]

                for ki in T.serial(block_k // micro_size_k):
                    for i in T.serial(local_size_a):
                        for mk in T.vectorized(micro_size_k):
                            A_local[i, mk] = A_shared[
                                warp_m * local_size_a + i, ki * micro_size_k + mk
                            ]

                    for j in T.serial(local_size_b):
                        for mk in T.vectorized(micro_size_k):
                            B_local[j, mk] = B_shared[
                                warp_n * local_size_b + j, ki * micro_size_k + mk
                            ]

                    for i, j in T.grid(local_size_a, local_size_b):
                        for mk in T.serial(micro_size_k // vector_width):
                            for vi in T.serial(vector_width):
                                C_local[i * local_size_b + j] += (
                                    A_local[i, mk * vector_width + vi]
                                    * B_local[j, mk * vector_width + vi]
                                )

            for i, j in T.grid(local_size_a, local_size_b):
                C[
                    by * block_m + warp_m * local_size_a + i,
                    bx * block_n + warp_n * local_size_b + j,
                ] = C_local[i * local_size_b + j]

            if instrument:
                T.iket.mark("tile_end")
                T.iket.range_pop("gemm_tile")

    return main


def compile_kernel(
    program_factory: Callable[[], object],
    *,
    output_dir: Path,
    runtime_payloads: bool,
    use_iket: bool,
):
    tilelang.disable_cache()
    try:
        if use_iket:
            with T.iket.session(output_dir=output_dir, runtime_payloads=runtime_payloads):
                program = program_factory()
                return tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
        T.iket.disable()
        T.iket.reset()
        program = program_factory()
        return tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    finally:
        tilelang.enable_cache()


def benchmark(fn: Callable[[], object], *, warmup_ms: float, rep_ms: float) -> float:
    return float(
        do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
            backend="event",
            return_mode="median",
            fast_flush=True,
        )
    )


def validate_vector_add(args, output_dir: Path, *, payloads: bool) -> ValidationResult:
    n = args.vector_n
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    ref = a + b

    label = "vector_add_payloads" if payloads else "vector_add_ranges"
    base = compile_kernel(
        lambda: vector_add_kernel(n, instrument=False, payloads=False),
        output_dir=output_dir / label / "baseline",
        runtime_payloads=False,
        use_iket=False,
    )
    iket = compile_kernel(
        lambda: vector_add_kernel(n, instrument=True, payloads=payloads),
        output_dir=output_dir / label / "iket",
        runtime_payloads=payloads,
        use_iket=True,
    )

    base_out = base(a, b)
    iket_out = iket(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(base_out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(iket_out, ref, rtol=1e-5, atol=1e-5)
    err = float((iket_out - ref).abs().max().item())

    base_ms = benchmark(lambda: base(a, b), warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    iket_ms = benchmark(lambda: iket(a, b), warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    return make_result(
        label,
        f"1D vector add, N={n}",
        base_ms,
        iket_ms,
        err,
        "range + markers" + (" + int32 payloads" if payloads else ""),
    )


def validate_add_scale(args, output_dir: Path) -> ValidationResult:
    n = args.vector_n
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    ref = (a + b) * scale[0]

    base = compile_kernel(
        lambda: add_scale_kernel(n, instrument=False, payloads=False),
        output_dir=output_dir / "add_scale_all_features" / "baseline",
        runtime_payloads=False,
        use_iket=False,
    )
    iket = compile_kernel(
        lambda: add_scale_kernel(n, instrument=True, payloads=True),
        output_dir=output_dir / "add_scale_all_features" / "iket",
        runtime_payloads=True,
        use_iket=True,
    )

    base_out = base(a, b, scale)
    iket_out = iket(a, b, scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(base_out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(iket_out, ref, rtol=1e-5, atol=1e-5)
    err = float((iket_out - ref).abs().max().item())

    base_ms = benchmark(lambda: base(a, b, scale), warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    iket_ms = benchmark(lambda: iket(a, b, scale), warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    return make_result(
        "add_scale_all_features",
        f"add + scale, N={n}",
        base_ms,
        iket_ms,
        err,
        "range + no-payload markers + int32 payloads",
    )


def validate_gemm(args, output_dir: Path) -> ValidationResult:
    m, n, k = args.gemm_m, args.gemm_n, args.gemm_k
    if m % GEMM_BLOCK_M != 0 or n % GEMM_BLOCK_N != 0 or k % GEMM_BLOCK_K != 0:
        raise ValueError(
            "SIMT GEMM validation requires "
            f"M multiple of {GEMM_BLOCK_M}, N multiple of {GEMM_BLOCK_N}, "
            f"and K multiple of {GEMM_BLOCK_K}; got M={m}, N={n}, K={k}"
        )

    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    ref = a.float() @ b.T.float()

    kwargs = {
        "m": m,
        "n": n,
        "k": k,
    }
    base = compile_kernel(
        lambda: gemm_kernel(**kwargs, instrument=False, payloads=False),
        output_dir=output_dir / "gemm_tile_markers" / "baseline",
        runtime_payloads=False,
        use_iket=False,
    )
    iket = compile_kernel(
        lambda: gemm_kernel(**kwargs, instrument=True, payloads=True),
        output_dir=output_dir / "gemm_tile_markers" / "iket",
        runtime_payloads=True,
        use_iket=True,
    )

    base_out = base(a, b)
    iket_out = iket(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(base_out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(iket_out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(iket_out, base_out, rtol=0, atol=0)
    err = float((iket_out.float() - ref.float()).abs().max().item())

    base_ms = benchmark(lambda: base(a, b), warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    iket_ms = benchmark(lambda: iket(a, b), warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
    return make_result(
        "gemm_tile_markers",
        f"SIMT GEMM {m}x{n}x{k}",
        base_ms,
        iket_ms,
        err,
        "tile range + tile_id payload",
    )


def make_result(
    name: str,
    workload: str,
    baseline_ms: float,
    iket_ms: float,
    max_abs_error: float,
    iket_features: str,
) -> ValidationResult:
    overhead_pct = (iket_ms / baseline_ms - 1.0) * 100.0 if baseline_ms else float("nan")
    return ValidationResult(
        name=name,
        workload=workload,
        baseline_ms=baseline_ms,
        iket_ms=iket_ms,
        overhead_pct=overhead_pct,
        max_abs_error=max_abs_error,
        correctness="pass",
        iket_features=iket_features,
    )


def print_markdown(results: list[ValidationResult]) -> None:
    print("\n## IKET validation summary\n")
    print(
        "| case | workload | features | baseline ms | IKET ms | overhead | max abs error | correctness |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        print(
            f"| {row.name} | {row.workload} | {row.iket_features} | "
            f"{row.baseline_ms:.4f} | {row.iket_ms:.4f} | {row.overhead_pct:+.2f}% | "
            f"{row.max_abs_error:.3g} | {row.correctness} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/tilelang_iket_validation")
    parser.add_argument("--vector-n", type=int, default=1 << 20)
    parser.add_argument("--gemm-m", type=int, default=256)
    parser.add_argument("--gemm-n", type=int, default=256)
    parser.add_argument("--gemm-k", type=int, default=256)
    parser.add_argument("--warmup-ms", type=float, default=10.0)
    parser.add_argument("--rep-ms", type=float, default=30.0)
    parser.add_argument("--skip-gemm", action="store_true")
    args = parser.parse_args()

    torch.cuda.init()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    output_dir = Path(args.output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        validate_vector_add(args, output_dir, payloads=False),
        validate_vector_add(args, output_dir, payloads=True),
        validate_add_scale(args, output_dir),
    ]
    if not args.skip_gemm:
        results.append(validate_gemm(args, output_dir))

    print_markdown(results)
    json_path = output_dir / "validation_results.json"
    json_path.write_text(json.dumps([asdict(row) for row in results], indent=2))
    print(f"\nWrote JSON results to {json_path}")


if __name__ == "__main__":
    main()
