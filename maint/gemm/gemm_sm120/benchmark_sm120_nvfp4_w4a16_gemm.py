"""SM120 W4A16 NVFP4 GEMM maintenance benchmark (decode-sized batches).

Times the W4A16 NVFP4 GEMM (BF16 activations x NVFP4 weights) of ``sm120_nvfp4_w4a16_streamk.py`` (tuned tables
and stream-K; ``--dispatch example`` times the example's fragment/tile-only heuristic instead) against a BF16 x BF16
cuBLAS GEMM through ``torch.nn.functional.linear`` and, when vLLM is importable, against vLLM's Marlin NVFP4 W4A16
kernel on the same packed weights (built with vLLM 0.29.0's ``prepare_fp4_layer_for_marlin`` and
``apply_fp4_marlin_linear``; these are vLLM internals and may change between versions).

Method: every call is timed inside a CUDA graph (no launch overhead), over rotated weight copies totalling at least
``--min-bytes`` so that the weights stream from DRAM as in a real model rather than from L2, with a tiny separator
kernel between calls (its own cost is measured and subtracted). Each (shape, M) is timed in ``--rounds`` interleaved
rounds, with the method order rotated every round; the table shows the fastest round and the layer-weighted totals
are given for both the fastest and the median rounds. The default shapes are the linear layers of a 27B-class model;
the default batch sizes include the tuned table's breakpoints and points between them.

Run from the repository root:

    python -m maint.gemm.gemm_sm120.benchmark_sm120_nvfp4_w4a16_gemm --ms 1,16,64,256 --verify
"""

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from examples.dequantize_gemm.quantize.nvfp4 import decode_packed_fp4_e2m1
from examples.gemm_sm120 import sm120_nvfp4_w4a16_gemm as example
from maint.gemm.gemm_sm120 import sm120_nvfp4_w4a16_streamk as tuned

DEFAULT_SHAPES = "16384x5120,14336x5120,5120x6144,34816x5120,5120x17408"
DEFAULT_MS = "1,4,8,16,17,24,32,33,40,48,49,56,64,72,96,104,128,144,192,208,232,256,272,304,384,416,512,544,608,768,832,1024"


def _marlin():
    """vLLM's Marlin NVFP4 W4A16 kernel as (prepare, apply), or None when vLLM is not installed."""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )
    except Exception:  # any import failure means "no Marlin baseline"
        return None

    def prepare(packed, scale, alpha):
        from types import SimpleNamespace

        N, K = packed.shape[0], packed.shape[1] * 2
        layer = SimpleNamespace(
            output_size_per_partition=N,
            input_size_per_partition=K,
            params_dtype=torch.bfloat16,
            bias=None,
            weight=packed.view(torch.uint8).clone(),
            weight_scale=scale.clone(),
            # Marlin multiplies by this value (vLLM's compressed-tensors scheme inverts the checkpoint's global scale
            # before it gets here), so it takes alpha itself.
            weight_global_scale=alpha.float().reshape(1).clone(),
        )
        prepare_fp4_layer_for_marlin(layer)
        return layer

    def apply(x, layer):
        N, K = layer.output_size_per_partition, layer.input_size_per_partition
        return apply_fp4_marlin_linear(x, layer.weight, layer.weight_scale, layer.weight_global_scale, layer.workspace, N, K)

    return prepare, apply


def _synthetic_layer(N, K, seed):
    """Random NVFP4 codes with checkpoint-like block scales and global scale."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8, device="cuda", generator=gen)
    scale = (torch.rand(N, K // 16, device="cuda", generator=gen) * 3 + 0.25).to(torch.float8_e4m3fn)
    alpha = torch.tensor(1.0 / 2688.0, device="cuda")
    return packed, scale, alpha


class _GraphTimer:
    def __init__(self, iters=5, repeats=5):
        self.iters, self.repeats = iters, repeats
        self.sep = torch.zeros(1, device="cuda")
        self.sep_us = 0.0
        self.sep_us = self.time([lambda: None] * 8)

    def time(self, fns):
        """Median per-call microseconds of fns (called in turn) replayed from one CUDA graph, minus the separator."""
        fns = [lambda f=f: (f(), self.sep.add_(1.0)) for f in fns]
        for fn in fns:
            fn()
        torch.cuda.synchronize()
        for attempt in range(3):
            graph = torch.cuda.CUDAGraph()
            try:
                # thread_local + retry: capture can fail transiently when another host thread allocates
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    for _ in range(self.iters):
                        for fn in fns:
                            fn()
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(self.repeats):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1e3 / (self.iters * len(fns)))
        return statistics.median(times) - self.sep_us


def _verify(N, K, linear, marlin):
    """Check the timed W4A16 path and, if present, the Marlin baseline against an FP32 reference."""
    packed, scale, alpha = _synthetic_layer(N, K, seed=123)
    weight = example.prepare_nvfp4_weight(packed, scale, alpha)
    layer = marlin[0](packed, scale, alpha) if marlin else None
    w_ref = decode_packed_fp4_e2m1(packed) * scale.float().repeat_interleave(16, dim=1) * alpha
    for M in (1, 17, 64, 200, 1024):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ w_ref.T
        outs = {"W4A16": linear(x, weight)}
        if layer is not None:
            outs["Marlin"] = marlin[1](x, layer)
        for name, y in outs.items():
            err = ((y.float() - ref).norm() / ref.norm()).item()
            if not err <= 2.5e-3:  # also catches NaN / Inf
                raise AssertionError(f"{name} {N}x{K} M={M}: relative error {err:.3e}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma-separated NxK weight shapes")
    parser.add_argument("--ms", default=DEFAULT_MS, help="comma-separated batch sizes M")
    parser.add_argument("--layer-counts", help="comma-separated layer count per shape, for a layer-weighted total")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dispatch", choices=["tuned", "example"], default="tuned", help="W4A16 path to time")
    parser.add_argument("--min-bytes", type=float, default=512e6, help="rotated weight bytes per method and shape")
    parser.add_argument("--no-marlin", action="store_true")
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json", help="append one JSON line per (shape, M) to this file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not example.is_supported_device():
        raise RuntimeError("this benchmark is tested on SM120 (compute capability 12.0) GPUs only")
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    ms = [int(m) for m in args.ms.split(",")]
    counts = [int(c) for c in args.layer_counts.split(",")] if args.layer_counts else None
    assert counts is None or len(counts) == len(shapes)
    marlin = None if args.no_marlin else _marlin()
    w4a16 = tuned if args.dispatch == "tuned" else example
    linear = w4a16.nvfp4_w4a16_linear
    print(
        f"GPU: {torch.cuda.get_device_name()}, dispatch: {args.dispatch}, Marlin baseline: {'yes' if marlin else 'no (vLLM not importable)'}"
    )

    if args.verify:
        for N, K in shapes:
            _verify(N, K, linear, marlin)
        print("Correctness: passed" + (" (TileLang W4A16 and Marlin)" if marlin else " (TileLang W4A16)"))

    # compile every configuration before timing
    for N, K in shapes:
        weight = example.prepare_nvfp4_weight(*_synthetic_layer(N, K, 0))
        for M in ms:
            linear(torch.randn(M, K, device="cuda", dtype=torch.bfloat16), weight)
        del weight
    torch.cuda.synchronize()

    timer = _GraphTimer()
    out = open(args.json, "a") if args.json else None
    results = {}
    header = f"{'N x K':>12} {'M':>5} | {'W4A16 us':>9} {'TFLOPS':>7} {'W GB/s':>7} | {'cuBLAS us':>9} {'speedup':>7}"
    header += f" | {'Marlin us':>9} {'speedup':>7}" if marlin else ""
    print(header)
    for N, K in shapes:
        w_bytes = N * K // 2 + N * K // 16
        copies = max(2, int(args.min_bytes // w_bytes) + 1)
        layers = [_synthetic_layer(N, K, seed) for seed in range(copies)]
        ours = [example.prepare_nvfp4_weight(*layer) for layer in layers]
        marl = [marlin[0](*layer) for layer in layers] if marlin else []
        del layers
        dense = []
        if not args.no_cublas:
            dense = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(max(2, int(args.min_bytes // (N * K * 2)) + 1))]
        for M in ms:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            methods = {"w4a16": [lambda w=w, x=x: linear(x, w) for w in ours]}
            if dense:
                methods["cublas"] = [lambda w=w, x=x: F.linear(x, w) for w in dense]
            if marl:
                methods["marlin"] = [lambda layer=layer, x=x: marlin[1](x, layer) for layer in marl]
            names = list(methods)
            rounds = {k: [] for k in names}
            for r in range(args.rounds):
                for k in names[r % len(names) :] + names[: r % len(names)]:  # rotate the order every round
                    rounds[k].append(timer.time(methods[k]))
            t = {k: min(v) for k, v in rounds.items()}
            results[(N, K, M)] = (t, {k: statistics.median(v) for k, v in rounds.items()})
            flop = 2.0 * M * N * K
            line = f"{f'{N}x{K}':>12} {M:>5} | {t['w4a16']:9.2f} {flop / t['w4a16'] / 1e6:7.1f} {w_bytes / t['w4a16'] / 1e3:7.0f} |"
            line += f" {t['cublas']:9.2f} {t['cublas'] / t['w4a16']:7.3f}" if "cublas" in t else f" {'-':>9} {'-':>7}"
            if "marlin" in t:
                line += f" | {t['marlin']:9.2f} {t['marlin'] / t['w4a16']:7.3f}"
            print(line, flush=True)
            if out:
                row = {"N": N, "K": K, "M": M, "dispatch": args.dispatch, "config": repr(w4a16.get_config(N, K, M))}
                row.update({f"{k}_us": round(v, 2) for k, v in t.items()})
                row.update({f"{k}_rounds": [round(u, 2) for u in v] for k, v in rounds.items() if v})
                out.write(json.dumps(row) + "\n")
                out.flush()
        del ours, marl, dense
        torch.cuda.empty_cache()

    if counts:
        for which, label in ((0, "fastest"), (1, "median")):
            print(f"Layer-weighted totals (us per model step, {label} rounds; % = W4A16 time vs the method):")
            for M in ms:
                keys = results[(*shapes[0], M)][which]
                tot = {k: sum(c * results[(N, K, M)][which][k] for (N, K), c in zip(shapes, counts)) for k in keys}
                line = f"  M={M:>5}: W4A16 {tot['w4a16']:9.1f}"
                for k in ("cublas", "marlin"):
                    if k in tot:
                        line += f" | {k} {tot[k]:9.1f} ({(tot['w4a16'] / tot[k] - 1) * 100:+.1f}%)"
                print(line)


if __name__ == "__main__":
    main()
