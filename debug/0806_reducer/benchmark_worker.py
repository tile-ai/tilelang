"""Run reducer benchmarks against exactly one TileLang checkout.

This file is launched by ``run_benchmarks.py`` in a fresh process.  Keeping the
legacy and v2 imports in different processes is important: both checkouts load
native libraries and register TVM global functions under the same names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from benchmark_cases import CASE_BY_NAME, BenchmarkCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="TileLang checkout to import.")
    parser.add_argument("--variant", choices=("legacy", "v2"), required=True)
    parser.add_argument("--case", action="append", dest="cases", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Worker JSON output path.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-incorrect",
        action="store_true",
        help="Time a kernel even when its output is numerically incorrect.",
    )
    return parser.parse_args()


def git_metadata(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def import_tilelang(repo: Path):
    """Import TileLang from ``repo`` and reject accidental cross-contamination."""

    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import tilelang  # pylint: disable=import-outside-toplevel
    import tilelang.language as T  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    imported_from = Path(tilelang.__file__).resolve()
    try:
        imported_from.relative_to(repo)
    except ValueError as err:
        raise RuntimeError(f"Expected TileLang from {repo}, imported {imported_from}") from err
    return tilelang, T, torch


def make_unique_owner_kernel(T, case: BenchmarkCase, variant: str):
    blocks = case.blocks
    m = case.m
    threads = case.threads
    batch = case.batch

    @T.prim_func
    def kernel(
        A: T.Tensor((blocks, m), T.float32),
        B: T.Tensor((blocks, m), T.float32),
    ):
        with T.Kernel(blocks, threads=threads) as bid:
            if variant == "legacy":
                total = T.alloc_reducer((m,), T.float32, op="sum", replication=case.legacy_replication)
                T.clear(total)
                for i in T.Parallel(m):
                    total[i] += A[bid, i]
                T.finalize_reducer(total, batch=batch)
                T.copy(total, B[bid, 0])
            else:
                total = T.alloc_reducer((m,), T.float32, op="sum")
                T.reducer_init(total)
                for i in T.Parallel(m):
                    T.reducer_update(total[i], A[bid, i])
                result = T.alloc_fragment((m,), T.float32)
                T.finalize_reducer(total, result, batch=batch)
                T.copy(result, B[bid, 0])

    return kernel


def make_row_reduce_kernel(T, case: BenchmarkCase, variant: str):
    blocks = case.blocks
    m = case.m
    k = case.k
    threads = case.threads
    batch = case.batch

    @T.prim_func
    def kernel(
        A: T.Tensor((blocks, m, k), T.float32),
        B: T.Tensor((blocks, m), T.float32),
    ):
        with T.Kernel(blocks, threads=threads) as bid:
            A_shared = T.alloc_shared((m, k), T.float32)
            T.copy(A[bid, 0, 0], A_shared, disable_tma=True)
            A_fragment = T.alloc_fragment((m, k), T.float32)
            T.copy(A_shared, A_fragment)

            if variant == "legacy":
                total = T.alloc_reducer((m,), T.float32, op="sum", replication=case.legacy_replication)
                T.clear(total)
                for i, j in T.Parallel(m, k):
                    total[i] += A_fragment[i, j]
                T.finalize_reducer(total, batch=batch)
                T.copy(total, B[bid, 0])
            else:
                total = T.alloc_reducer((m,), T.float32, op="sum")
                T.reducer_init(total)
                for i, j in T.Parallel(m, k):
                    T.reducer_update(total[i], A_fragment[i, j])
                result = T.alloc_fragment((m,), T.float32)
                T.finalize_reducer(total, result, batch=batch)
                T.copy(result, B[bid, 0])

    return kernel


def make_streaming_gemv_kernel(T, case: BenchmarkCase, variant: str):
    blocks = case.blocks
    m = case.m
    k = case.k
    tile_k = case.tile_k
    num_stages = case.num_stages
    threads = case.threads
    batch = case.batch
    assert tile_k > 0 and k % tile_k == 0

    @T.prim_func
    def kernel(
        A: T.Tensor((blocks, m, k), T.float32),
        X: T.Tensor((blocks, k), T.float32),
        B: T.Tensor((blocks, m), T.float32),
    ):
        with T.Kernel(blocks, threads=threads) as bid:
            if variant == "legacy":
                total = T.alloc_reducer((m,), T.float32, op="sum", replication=case.legacy_replication)
                T.clear(total)
                for ko in T.Pipelined(k // tile_k, num_stages=num_stages):
                    A_shared = T.alloc_shared((m, tile_k), T.float32)
                    T.copy(A[bid, 0, ko * tile_k], A_shared, disable_tma=True)
                    A_fragment = T.alloc_fragment((m, tile_k), T.float32)
                    T.copy(A_shared, A_fragment)
                    X_fragment = T.alloc_fragment((tile_k,), T.float32)
                    T.copy(X[bid, ko * tile_k], X_fragment, disable_tma=True)
                    for i, j in T.Parallel(m, tile_k):
                        total[i] += A_fragment[i, j] * X_fragment[j]
                T.finalize_reducer(total, batch=batch)
                T.copy(total, B[bid, 0])
            else:
                total = T.alloc_reducer((m,), T.float32, op="sum")
                T.reducer_init(total)
                for ko in T.Pipelined(k // tile_k, num_stages=num_stages):
                    A_shared = T.alloc_shared((m, tile_k), T.float32)
                    T.copy(A[bid, 0, ko * tile_k], A_shared, disable_tma=True)
                    A_fragment = T.alloc_fragment((m, tile_k), T.float32)
                    T.copy(A_shared, A_fragment)
                    X_fragment = T.alloc_fragment((tile_k,), T.float32)
                    T.copy(X[bid, ko * tile_k], X_fragment, disable_tma=True)
                    for i, j in T.Parallel(m, tile_k):
                        T.reducer_update(total[i], A_fragment[i, j] * X_fragment[j])
                result = T.alloc_fragment((m,), T.float32)
                T.finalize_reducer(total, result, batch=batch)
                T.copy(result, B[bid, 0])

    return kernel


def make_kernel(T, case: BenchmarkCase, variant: str):
    if case.family == "unique_owner":
        return make_unique_owner_kernel(T, case, variant)
    if case.family == "row_reduce":
        return make_row_reduce_kernel(T, case, variant)
    if case.family == "streaming_gemv":
        return make_streaming_gemv_kernel(T, case, variant)
    raise ValueError(f"Unknown benchmark family: {case.family}")


def make_inputs(torch, case: BenchmarkCase, device: str):
    torch.manual_seed(0)
    if case.family == "unique_owner":
        A = torch.randn((case.blocks, case.m), dtype=torch.float32, device=device)
        B = torch.full_like(A, float("nan"))
        return [A, B], A

    if case.family == "row_reduce":
        shape = (case.blocks, case.m, case.k)
        if case.name.startswith("replica_stress"):
            values = torch.arange(1, case.k + 1, dtype=torch.float32, device=device)
            A = values.reshape(1, 1, case.k).expand(shape).contiguous()
        else:
            A = torch.randn(shape, dtype=torch.float32, device=device)
        B = torch.full((case.blocks, case.m), float("nan"), dtype=torch.float32, device=device)
        return [A, B], A.sum(dim=2)

    if case.family == "streaming_gemv":
        A = torch.randn((case.blocks, case.m, case.k), dtype=torch.float32, device=device)
        X = torch.randn((case.blocks, case.k), dtype=torch.float32, device=device)
        B = torch.full((case.blocks, case.m), float("nan"), dtype=torch.float32, device=device)
        return [A, X, B], (A * X[:, None, :]).sum(dim=2)

    raise ValueError(f"Unknown benchmark family: {case.family}")


def source_metrics(source: str) -> dict[str, int]:
    return {
        "bytes": len(source.encode("utf-8")),
        "all_reduce_calls": source.count("tl::AllReduce<"),
        "run_batch_calls": source.count("::run_batch("),
        "named_barriers": source.count("NamedBarrier<"),
        "sync_threads": source.count("__syncthreads"),
    }


def finite_max(torch, value) -> float | None:
    result = float(value.max().item())
    return result if math.isfinite(result) else None


def sample_values(value) -> list[float | None]:
    samples = []
    for item in value.flatten()[:4].tolist():
        item = float(item)
        samples.append(item if math.isfinite(item) else None)
    return samples


def check_output(torch, actual, expected, case: BenchmarkCase) -> dict[str, Any]:
    difference = (actual - expected).abs()
    denominator = expected.abs().clamp_min(1e-12)
    if case.family == "unique_owner":
        atol, rtol = 1e-5, 1e-5
    elif case.family == "streaming_gemv":
        atol, rtol = 1e-2, 1e-3
    else:
        atol, rtol = 1e-3, 1e-3
    return {
        "correct": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": finite_max(torch, difference),
        "max_rel_error": finite_max(torch, difference / denominator),
        "actual_sample": sample_values(actual),
        "expected_sample": sample_values(expected),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_case(tilelang, T, torch, args: argparse.Namespace, case: BenchmarkCase) -> dict[str, Any]:
    print(f"[{args.variant}] {case.name}: compiling", flush=True)
    started = time.perf_counter()
    kernel = make_kernel(T, case, args.variant)
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    compiled = tilelang.compile(kernel, out_idx=None, pass_configs=pass_configs)
    compile_seconds = time.perf_counter() - started

    source = compiled.get_kernel_source()
    source_path = args.source_dir / args.variant / f"{case.name}.cu"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")

    device = f"cuda:{args.device}"
    inputs, expected = make_inputs(torch, case, device)
    compiled(*inputs)
    torch.cuda.synchronize(args.device)
    actual = inputs[-1]
    correctness = check_output(torch, actual, expected, case)

    latency_ms = None
    benchmark_skipped_reason = None
    if args.no_benchmark:
        benchmark_skipped_reason = "disabled by --no-benchmark"
    elif not correctness["correct"] and not args.benchmark_incorrect:
        benchmark_skipped_reason = "incorrect output"
    else:
        print(f"[{args.variant}] {case.name}: benchmarking", flush=True)
        latency_ms = float(
            compiled.get_profiler().do_bench(
                input_tensors=inputs,
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                backend="event",
                return_mode="median",
                device=args.device,
            )
        )

    print(
        f"[{args.variant}] {case.name}: correct={correctness['correct']} latency_ms={latency_ms}",
        flush=True,
    )
    return {
        "case": case.to_dict(),
        "status": "ok",
        "expected_correct": case.expected_legacy_correct if args.variant == "legacy" else True,
        "compile_seconds": compile_seconds,
        "latency_ms": latency_ms,
        "benchmark_skipped_reason": benchmark_skipped_reason,
        "correctness": correctness,
        "source_path": str(source_path),
        "source_metrics": source_metrics(source),
    }


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    unknown = [name for name in args.cases if name not in CASE_BY_NAME]
    if unknown:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(unknown)}")

    tilelang, T, torch = import_tilelang(args.repo)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for reducer benchmarks")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    # ``importlib.metadata`` may describe an unrelated installed wheel even
    # though this worker imports Python/native code from a source checkout.
    version = getattr(tilelang, "__version__", "unknown")

    payload: dict[str, Any] = {
        "metadata": {
            "variant": args.variant,
            "repo": str(args.repo),
            "tilelang_file": str(Path(tilelang.__file__).resolve()),
            "tilelang_version": version,
            "git": git_metadata(args.repo),
            "device": args.device,
            "gpu_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "benchmark_enabled": not args.no_benchmark,
            "benchmark_incorrect": args.benchmark_incorrect,
        },
        "cases": [],
    }
    write_json(args.output, payload)

    for case_name in args.cases:
        case = CASE_BY_NAME[case_name]
        try:
            result = run_case(tilelang, T, torch, args, case)
        except Exception as err:  # Keep the rest of the matrix usable after one failure.
            result = {
                "case": case.to_dict(),
                "status": "error",
                "expected_correct": (case.expected_legacy_correct if args.variant == "legacy" else True),
                "error": f"{type(err).__name__}: {err}",
                "traceback": traceback.format_exc(),
            }
            print(f"[{args.variant}] {case.name}: ERROR: {result['error']}", flush=True)
        payload["cases"].append(result)
        write_json(args.output, payload)

    failures = sum(result["status"] == "error" for result in payload["cases"])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
