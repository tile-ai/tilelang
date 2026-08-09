"""Alignment and benchmark harness for the SM120 TileLang SageAttention3 raw core.

The default cases are Mayfly's complete 1024x1024 image-generation attention
shapes: 4096 latent tokens plus either 32 or 512 context tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from examples.sage_attention_sm120.sageattn3_fp4 import sage3_packed_fp4_attention_raw_kernel  # noqa: E402

_DEFAULT_SAGE_ROOT = _REPO / "SageAttention" / "sageattention3_blackwell"
_MAYFLY_IMAGE_ROWS = (4096 + 32, 4096 + 512)
_MAYFLY_IMAGE_HEADS = 30


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _import_sageattn3(sage_root: Path) -> dict[str, Any]:
    sage_root = sage_root.resolve()
    if str(sage_root) not in sys.path:
        sys.path.insert(0, str(sage_root))
    try:
        from sageattn3.api import (  # type: ignore
            blockscaled_fp4_attn,
            preprocess_qkv,
            sageattn3_blackwell,
            scale_and_quant_fp4,
            scale_and_quant_fp4_permute,
            scale_and_quant_fp4_transpose,
        )
    except Exception as exc:  # pragma: no cover - depends on local SageAttention checkout
        raise RuntimeError(
            "Unable to import official SageAttention3. Expected SageAttention/sageattention3_blackwell or pass --sage-root."
        ) from exc
    return {
        "blockscaled_fp4_attn": blockscaled_fp4_attn,
        "preprocess_qkv": preprocess_qkv,
        "sageattn3_blackwell": sageattn3_blackwell,
        "scale_and_quant_fp4": scale_and_quant_fp4,
        "scale_and_quant_fp4_permute": scale_and_quant_fp4_permute,
        "scale_and_quant_fp4_transpose": scale_and_quant_fp4_transpose,
    }


def _bench(fn: Callable[[], Any], *, warmup: int, rep: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "min_ms": float(min(times)),
        "p50_ms": float(statistics.median(times)),
        "p90_ms": float(sorted(times)[int(math.ceil(rep * 0.9)) - 1]),
    }


def _bench_with_fresh_k(
    fn: Callable[[torch.Tensor], Any],
    k_base: torch.Tensor,
    *,
    warmup: int,
    rep: int,
) -> tuple[dict[str, float], Any]:
    pool = [k_base.clone() for _ in range(warmup + rep)]
    out = None
    for i in range(warmup):
        out = fn(pool[i])
    torch.cuda.synchronize()
    times: list[float] = []
    for i in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(pool[warmup + i])
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "min_ms": float(min(times)),
        "p50_ms": float(statistics.median(times)),
        "p90_ms": float(sorted(times)[int(math.ceil(rep * 0.9)) - 1]),
    }, out


def _make_inputs(rows: int, heads: int, head_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed + rows)
    shape = (1, heads, rows, head_dim)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    return q, k, v


def _sage_quantize(api: dict[str, Any], q_pre: torch.Tensor, k_pre: torch.Tensor, v_pre: torch.Tensor):
    return (
        api["scale_and_quant_fp4"](q_pre),
        api["scale_and_quant_fp4_permute"](k_pre),
        api["scale_and_quant_fp4_transpose"](v_pre),
    )


def _sage_core_raw(api: dict[str, Any], qlist, klist, vlist, delta_s: torch.Tensor, rows: int):
    return api["blockscaled_fp4_attn"](
        qlist,
        klist,
        vlist,
        delta_s,
        rows,
        False,
        True,
        True,
    )


def _compare(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    err = (actual.to(torch.float32) - reference.to(torch.float32)).abs()
    return {
        "max_abs_err": float(err.max().item()),
        "mean_abs_err": float(err.mean().item()),
    }


def _compile_tilelang_kernel(
    query_tokens: int,
    kv_tokens: int,
    valid_k_tokens: int,
    heads: int,
    head_dim: int,
) -> tuple[Any, dict[str, Any]]:
    import tilelang

    tir_start = time.perf_counter()
    prim_func = sage3_packed_fp4_attention_raw_kernel.get_tir(
        query_tokens,
        kv_tokens,
        valid_k_tokens,
        heads,
        heads,
        head_dim,
    )
    tir_ms = (time.perf_counter() - tir_start) * 1000.0

    compile_start = time.perf_counter()
    kernel = tilelang.compile(
        prim_func,
        target={"kind": "cuda", "arch": "sm_120a"},
        execution_backend="nvrtc",
    )
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    source = kernel.get_kernel_source()
    return kernel, {
        "tir_ms": float(tir_ms),
        "compile_ms": float(compile_ms),
        "source": {
            "tma_load_count": int(source.count("tma_load")),
            "ptx_ldmatrix_count": int(source.count("ptx_ldmatrix")),
            "mma_blockscale_count": int(source.count("mma_sync_blockscale")),
            "tcgen05_count": int(source.count("tcgen05")),
            "partial_sync_count": int(source.count("__sync_thread_partial")),
            "cta_sync_count": int(source.count("__syncthreads")),
        },
    }


def _tilelang_compile_only(*, rows: int, heads: int, head_dim: int) -> dict[str, Any]:
    q_tokens = _ceil_to_multiple(rows, 128)
    k_tokens = _ceil_to_multiple(rows, 128)
    _, compile_info = _compile_tilelang_kernel(q_tokens, k_tokens, rows, heads, head_dim)
    return {
        "mode": "compile_only",
        "shape": {
            "heads": heads,
            "query_tokens_padded": q_tokens,
            "kv_tokens_padded": k_tokens,
            "valid_k_tokens": rows,
            "head_dim": head_dim,
        },
        "compile": compile_info,
    }


def _tilelang_raw_core(
    qlist,
    klist,
    vlist,
    delta_s: torch.Tensor,
    official_raw,
    *,
    rows: int,
    heads: int,
    head_dim: int,
    warmup: int,
    rep: int,
    check: bool,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    q_tokens = int(qlist[0].shape[2])
    k_tokens = int(klist[0].shape[2])
    out = torch.empty((1, heads, q_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((1, heads, q_tokens), device="cuda", dtype=torch.float32)
    kernel, compile_info = _compile_tilelang_kernel(q_tokens, k_tokens, rows, heads, head_dim)

    def run() -> None:
        kernel(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, out, lse)

    run()
    torch.cuda.synchronize()
    raw_out, _ = official_raw
    visible_out = out[:, :, :rows, :]
    visible_ref = raw_out[:, :, :rows, :]
    errors = {
        "raw_out": _compare(out, raw_out),
        "visible_out": _compare(visible_out, visible_ref),
    }
    if check:
        torch.testing.assert_close(visible_out, visible_ref, atol=atol, rtol=rtol)

    torch.cuda.reset_peak_memory_stats()
    timing = _bench(run, warmup=warmup, rep=rep)
    peak = float(torch.cuda.max_memory_allocated() / (1024**3))
    return {
        **timing,
        "peak_gb": peak,
        "shape": {
            "q_tokens_padded": q_tokens,
            "kv_tokens_padded": k_tokens,
            "valid_k_tokens": rows,
            "heads": heads,
            "head_dim": head_dim,
            "launch_threads": 384,
            "compute_threads": 256,
            "tile_group_size": 1,
        },
        "errors": errors,
        "reference": "official blockscaled_fp4_attn raw output on identical packed FP4/FP8/DeltaS inputs",
        "compile": compile_info,
    }


def _run_case(
    api: dict[str, Any],
    *,
    rows: int,
    heads: int,
    head_dim: int,
    warmup: int,
    rep: int,
    seed: int,
    include_tilelang_raw: bool,
    check: bool,
    atol: float,
    rtol: float,
    min_speedup: float,
) -> dict[str, Any]:
    q, k, v = _make_inputs(rows, heads, head_dim, seed)

    def run_full(k_fresh: torch.Tensor) -> torch.Tensor:
        return api["sageattn3_blackwell"](q, k_fresh, v, is_causal=False, per_block_mean=True)

    full_timing, full_once = _bench_with_fresh_k(run_full, k, warmup=warmup, rep=rep)

    q_pre, k_pre, v_pre, delta_s = api["preprocess_qkv"](q, k.clone(), v, True)
    qlist, klist, vlist = _sage_quantize(api, q_pre, k_pre, v_pre)

    native_core_raw = None

    def run_native_core() -> None:
        nonlocal native_core_raw
        native_core_raw = _sage_core_raw(api, qlist, klist, vlist, delta_s, rows)

    core_timing = _bench(run_native_core, warmup=warmup, rep=rep)
    if native_core_raw is None:
        native_core_raw = _sage_core_raw(api, qlist, klist, vlist, delta_s, rows)
    raw_out, _ = native_core_raw

    result: dict[str, Any] = {
        "rows": rows,
        "shape": {"batch": 1, "heads": heads, "query_tokens": rows, "kv_tokens": rows, "head_dim": head_dim},
        "warmup": warmup,
        "rep": rep,
        "sageattn3_full_official": {
            **full_timing,
            "reference": "official sageattn3_blackwell",
        },
        "sageattn3_native_core_official": {
            **core_timing,
            "reference": "official blockscaled_fp4_attn raw call on official packed inputs",
        },
        "full_vs_raw_visible": _compare(raw_out[:, :, :rows, :], full_once),
    }
    if include_tilelang_raw:
        tilelang_raw = _tilelang_raw_core(
            qlist,
            klist,
            vlist,
            delta_s,
            native_core_raw,
            rows=rows,
            heads=heads,
            head_dim=head_dim,
            warmup=warmup,
            rep=rep,
            check=check,
            atol=atol,
            rtol=rtol,
        )
        speedup = core_timing["p50_ms"] / tilelang_raw["p50_ms"]
        partial_sync_count = tilelang_raw["compile"]["source"]["partial_sync_count"]
        error_pass = tilelang_raw["errors"]["visible_out"]["max_abs_err"] <= atol
        tilelang_raw["validation"] = {
            "speedup_vs_official_raw_p50": float(speedup),
            "min_speedup": float(min_speedup),
            "performance_pass": bool(speedup >= min_speedup),
            "partial_sync_count": int(partial_sync_count),
            "partial_sync_pass": partial_sync_count == 0,
            "max_abs_tolerance": float(atol),
            "correctness_pass": bool(error_pass),
        }
        tilelang_raw["status"] = "ok" if speedup >= min_speedup and partial_sync_count == 0 and error_pass else "fail"
        result["tilelang_raw_core"] = tilelang_raw
    return result


def _print_compile_only(item: dict[str, Any]) -> None:
    info = item["tilelang_compile_only"]["compile"]
    src = info["source"]
    print(
        f"rows={item['rows']} compile_only; tilelang_raw "
        f"tir/compile={info['tir_ms']:.1f}/{info['compile_ms']:.1f} ms "
        f"tma={src['tma_load_count']} mma_blockscale={src['mma_blockscale_count']} tcgen05={src['tcgen05_count']}"
    )


def _print_case(item: dict[str, Any]) -> None:
    full = item["sageattn3_full_official"]
    core = item["sageattn3_native_core_official"]
    line = (
        f"rows={item['rows']} sage3_full p50/p90={full['p50_ms']:.3f}/{full['p90_ms']:.3f} ms; "
        f"sage3_core p50/p90={core['p50_ms']:.3f}/{core['p90_ms']:.3f} ms"
    )
    if "tilelang_raw_core" in item:
        tl_raw = item["tilelang_raw_core"]
        err = tl_raw["errors"]["visible_out"]
        line += (
            f"; tilelang_raw p50/p90={tl_raw['p50_ms']:.3f}/{tl_raw['p90_ms']:.3f} ms "
            f"max/mean={err['max_abs_err']:.6f}/{err['mean_abs_err']:.6f} "
            f"speedup={tl_raw['validation']['speedup_vs_official_raw_p50']:.3f}x "
            f"partial_syncs={tl_raw['validation']['partial_sync_count']}"
        )
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="SM120 TileLang SageAttention3 raw-core alignment benchmark.")
    parser.add_argument("--rows", type=int, nargs="+", default=list(_MAYFLY_IMAGE_ROWS))
    parser.add_argument("--heads", type=int, default=_MAYFLY_IMAGE_HEADS)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5060)
    parser.add_argument("--sage-root", type=Path, default=_DEFAULT_SAGE_ROOT)
    parser.add_argument("--include-tilelang-raw", action="store_true")
    parser.add_argument("--compile-only-tilelang", action="store_true")
    parser.add_argument("--check", action="store_true", help="Assert TileLang visible output against the official raw output.")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=0.9,
        help="Minimum TileLang/official raw-core p50 throughput ratio.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.warmup < 1 or args.rep < 2:
        raise ValueError("Use meaningful benchmark settings: warmup >= 1 and rep >= 2")
    if args.head_dim != 128:
        raise ValueError("SageAttention3 Blackwell path is validated here for head_dim=128")
    if args.min_speedup <= 0:
        raise ValueError("--min-speedup must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SM120 SageAttention3 example")

    results = []
    if args.compile_only_tilelang:
        for rows in args.rows:
            results.append(
                {
                    "rows": rows,
                    "tilelang_compile_only": _tilelang_compile_only(rows=rows, heads=args.heads, head_dim=args.head_dim),
                }
            )
        payload = {"status": "ok", "results": results}
    else:
        api = _import_sageattn3(args.sage_root)
        for rows in args.rows:
            results.append(
                _run_case(
                    api,
                    rows=rows,
                    heads=args.heads,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    rep=args.rep,
                    seed=args.seed,
                    include_tilelang_raw=args.include_tilelang_raw,
                    check=args.check,
                    atol=args.atol,
                    rtol=args.rtol,
                    min_speedup=args.min_speedup,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        payload = {
            "status": ("ok" if all(item.get("tilelang_raw_core", {}).get("status", "ok") == "ok" for item in results) else "fail"),
            "baseline_policy": "SOTA baseline is official sageattn3_blackwell; TileLang raw core is compared with official packed raw output",
            "results": results,
        }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for item in results:
            if "tilelang_compile_only" in item:
                _print_compile_only(item)
            else:
                _print_case(item)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
