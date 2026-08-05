"""Benchmark the Transpose schedule template across many shapes.

Usage:
    python testing/python/graph/bench_tilelang_graph_schedule_template_transpose.py
    python testing/python/graph/bench_tilelang_graph_schedule_template_transpose.py --arch sm_80
    python testing/python/graph/bench_tilelang_graph_schedule_template_transpose.py --check-only
"""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import te, tirx

from tilelang.schedule.gpu.transpose import Transpose


# Workload shapes: (M, N) input shapes, output is (N, M).
SHAPES = [
    # Square, power-of-2
    (32, 32),
    (64, 64),
    (128, 128),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    # Rectangular, moderate
    (256, 512),
    (512, 256),
    (512, 1024),
    (1024, 512),
    (1024, 2048),
    (2048, 1024),
    # Rectangular, long/tall
    (128, 4096),
    (4096, 128),
    (256, 8192),
    (8192, 256),
    # Non-power-of-2 / tail-heavy
    (96, 160),
    (255, 511),
    (1000, 1536),
    (1536, 1000),
    (1023, 2047),
    (2047, 1023),
    # Small edge cases
    (1, 33),
    (33, 1),
    (17, 65),
    (65, 17),
]


def _build_mod(m: int, n: int, arch: str):
    """Build and lower a scheduled transpose module."""
    a = te.placeholder((m, n), name="a")
    c = te.compute((n, m), lambda i, j: a[j, i], name="c")
    func = te.create_prim_func([a, c])

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    sch = Transpose().apply(func, target, False)
    if sch is None:
        return None, False

    mod = sch.mod
    mod = tirx.transform.Simplify()(mod)
    mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.CanonicalizeScheduledTIR()(mod)

    lowered_ir = str(mod)
    has_shared = "T.copy(" in lowered_ir and "shared.dyn" in lowered_ir
    return mod, has_shared


def _run_one(m: int, n: int, arch: str, bench_backend: str, check_only: bool):
    """Build, verify correctness, and benchmark one shape. Returns a result dict."""
    shape_str = f"({m}, {n})"
    total_elems = m * n
    total_bytes = 2 * total_elems * 4  # fp32 read + write

    try:
        mod, has_shared = _build_mod(m, n, arch)
    except Exception as e:  # pragma: no cover
        return {
            "shape": shape_str,
            "elems": total_elems,
            "shared": "?",
            "status": f"BUILD FAIL: {e}",
            "tl_ms": None,
            "torch_ms": None,
            "tl_gbps": None,
            "torch_gbps": None,
            "speedup": None,
        }

    if mod is None:
        return {
            "shape": shape_str,
            "elems": total_elems,
            "shared": "-",
            "status": "SKIP (rule returned None)",
            "tl_ms": None,
            "torch_ms": None,
            "tl_gbps": None,
            "torch_gbps": None,
            "speedup": None,
        }

    if check_only:
        return {
            "shape": shape_str,
            "elems": total_elems,
            "shared": "Y" if has_shared else "N",
            "status": "OK",
            "tl_ms": None,
            "torch_ms": None,
            "tl_gbps": None,
            "torch_gbps": None,
            "speedup": None,
        }

    try:
        kernel = tilelang.compile(mod["main"])
    except Exception as e:
        return {
            "shape": shape_str,
            "elems": total_elems,
            "shared": "Y" if has_shared else "N",
            "status": f"COMPILE FAIL: {e}",
            "tl_ms": None,
            "torch_ms": None,
            "tl_gbps": None,
            "torch_gbps": None,
            "speedup": None,
        }

    torch.manual_seed(0)
    a_t = torch.randn((m, n), device="cuda", dtype=torch.float32)
    c_t = torch.empty((n, m), device="cuda", dtype=torch.float32)

    try:
        kernel(a_t, c_t)
        ref = a_t.transpose(0, 1).contiguous()
        torch.testing.assert_close(c_t, ref, rtol=1e-4, atol=1e-4)
    except Exception as e:
        return {
            "shape": shape_str,
            "elems": total_elems,
            "shared": "Y" if has_shared else "N",
            "status": f"WRONG: {e}",
            "tl_ms": None,
            "torch_ms": None,
            "tl_gbps": None,
            "torch_gbps": None,
            "speedup": None,
        }

    tl_ms = do_bench(lambda: kernel(a_t, c_t), backend=bench_backend)
    torch_fn = torch.compile(lambda a: a.transpose(0, 1).contiguous())
    torch_fn(a_t)  # warmup compile
    torch_ms = do_bench(lambda: torch_fn(a_t), backend=bench_backend)

    tl_gbps = total_bytes / (tl_ms * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_ms * 1e-3) / 1e9
    speedup = torch_ms / tl_ms if tl_ms > 0 else float("inf")

    return {
        "shape": shape_str,
        "elems": total_elems,
        "shared": "Y" if has_shared else "N",
        "status": "PASS",
        "tl_ms": tl_ms,
        "torch_ms": torch_ms,
        "tl_gbps": tl_gbps,
        "torch_gbps": torch_gbps,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transpose schedule template.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch, e.g. "sm_90a".')
    parser.add_argument("--check-only", action="store_true", help="Only check build/lowering, skip GPU run.")
    parser.add_argument("--bench-backend", type=str, default="event", choices=["event", "cupti"])
    args = parser.parse_args()

    if not args.check_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")

    tilelang.disable_cache()

    results = []
    for m, n in SHAPES:
        result = _run_one(m, n, args.arch, args.bench_backend, args.check_only)
        results.append(result)
        if result["tl_ms"] is not None:
            print(
                f"  {result['shape']:>18s}  shared={result['shared']}  "
                f"tl={result['tl_ms']:8.4f}ms ({result['tl_gbps']:7.1f} GB/s)  "
                f"torch={result['torch_ms']:8.4f}ms ({result['torch_gbps']:7.1f} GB/s)  "
                f"speedup={result['speedup']:.3f}x  {result['status']}"
            )
        else:
            print(f"  {result['shape']:>18s}  shared={result['shared']}  {result['status']}")

    print("\n" + "=" * 108)
    print(
        f"{'Shape':>18s}  {'Shared':>6s}  {'TL (ms)':>10s}  {'TL GB/s':>9s}  "
        f"{'Torch (ms)':>10s}  {'Torch GB/s':>10s}  {'Speedup':>8s}  {'Status':>8s}"
    )
    print("-" * 108)
    for r in results:
        tl_str = f"{r['tl_ms']:10.4f}" if r["tl_ms"] is not None else "       N/A"
        tl_bw = f"{r['tl_gbps']:9.1f}" if r["tl_gbps"] is not None else "      N/A"
        to_str = f"{r['torch_ms']:10.4f}" if r["torch_ms"] is not None else "       N/A"
        to_bw = f"{r['torch_gbps']:10.1f}" if r["torch_gbps"] is not None else "       N/A"
        sp_str = f"{r['speedup']:8.3f}" if r["speedup"] is not None else "     N/A"
        print(f"{r['shape']:>18s}  {r['shared']:>6s}  {tl_str}  {tl_bw}  {to_str}  {to_bw}  {sp_str}  {r['status']:>8s}")
    print("=" * 108)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] not in ("PASS", "OK") and "SKIP" not in r["status"]]
    skipped = [r for r in results if "SKIP" in r["status"]]
    print(f"\nTotal: {len(results)}  Passed: {len(passed)}  Failed: {len(failed)}  Skipped: {len(skipped)}")
    if passed:
        avg_speedup = sum(r["speedup"] for r in passed) / len(passed)
        wins = sum(1 for r in passed if r["speedup"] >= 1.0)
        print(f"Average speedup: {avg_speedup:.3f}x  Wins: {wins}/{len(passed)}")


if __name__ == "__main__":
    main()
