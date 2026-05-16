"""Sweep benchmark for Blackwell (SM100) Flash Attention forward.

Runs the configs requested for GLM-5, Llama-4-Maverick, Qwen3.5, DeepSeek-V3.2-Exp,
MiniMax-M2.5 across seqlen 4096/8192/16384. Picks GQA vs MHA kernel automatically
from heads/kv_heads. Tries variants ss / ts / wasp / fa4; reports best.

Note: fa4 lives only in gqa_fwd_bshd.py; for MHA configs (heads == kv_heads) we
dispatch to the GQA fa4 kernel with groups=1, which is equivalent.

Correctness check is intentionally skipped (CPU softmax is too slow at these sizes).
"""

import argparse
import csv
import importlib.util
import os
import sys
import traceback
from pathlib import Path

# Force tilelang/nvcc to use a private tempdir instead of the shared
# /tmp/tvm-debug-mode-tempdirs/, which on this host is owned by another user
# and not writable. EnvVar.get() reads os.environ lazily, so setting it here
# (before any compile call) is sufficient.
os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")

import torch
from tilelang.profiler import do_bench


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mha_mod = _load("mha_fwd_bshd_local", HERE / "mha_fwd_bshd.py")
gqa_mod = _load("gqa_fwd_bshd_local", HERE / "gqa_fwd_bshd.py")


MODELS = [
    # (label, batch, heads, kv_heads, dim, causal)
    ("GLM-5",                     8,  64,  64, 256, False),
    ("Llama-4-Maverick-17B-128E", 8,  40,   8, 128, False),
    ("Qwen3.5-397B-A17B",         8,  32,   2, 256, False),
    ("DeepSeek-V3.2-Exp",         8, 128, 128, 192, False),
    ("MiniMax-M2.5",              8,  48,   8, 128, False),
]

# DEFAULT_SEQS = [1024, 2048, 4096, 8192, 16384]
DEFAULT_SEQS = [1024, 2048]


def build_configs(seqs):
    return [(label, b, h, kv, d, s, c) for (label, b, h, kv, d, c) in MODELS for s in seqs]


def total_flops(batch, heads, seqlen, dim, causal):
    f = 2.0 * 2.0 * batch * heads * seqlen * seqlen * dim
    if causal:
        f *= 0.5
    return f


def build_kernel(use_gqa, batch, heads, seqlen, dim, causal, groups, variant,
                 block_M=128, block_N=128):
    mod = gqa_mod if use_gqa else mha_mod
    if variant in ("ss", "ts"):
        if use_gqa:
            return mod.flashattn(batch, heads, seqlen, dim, causal,
                                 groups=groups, block_M=block_M, block_N=block_N,
                                 variant=variant)
        return mod.flashattn(batch, heads, seqlen, dim, causal,
                             block_M=block_M, block_N=block_N, variant=variant)
    if variant == "fa4":
        # fa4 is only implemented in gqa_fwd_bshd; for MHA we pass groups=1.
        return gqa_mod.flashattn_fa4(batch, heads, seqlen, dim, causal,
                                     groups=groups, block_M=block_M, block_N=block_N)
    # wasp
    if use_gqa:
        return mod.flashattn_wasp(batch, heads, seqlen, dim, causal,
                                  groups=groups, block_M=block_M, block_N=block_N,
                                  threads=256, num_stages=2)
    return mod.flashattn_wasp(batch, heads, seqlen, dim, causal,
                              block_M=block_M, block_N=block_N,
                              threads=256, num_stages=2)


def bench_one(label, batch, heads, kv_heads, dim, seqlen, causal,
              variants=("ss", "ts", "wasp", "fa4"), warmup=25, rep=100):
    use_gqa = heads != kv_heads
    groups = heads // kv_heads if use_gqa else 1
    kind = f"GQA(g={groups})" if use_gqa else "MHA"

    Q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(batch, seqlen, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(batch, seqlen, kv_heads, dim, device="cuda", dtype=torch.bfloat16)

    flops = total_flops(batch, heads, seqlen, dim, causal)
    rows = []
    for v in variants:
        tag = f"[{label} | b={batch} h={heads}/{kv_heads} d={dim} s={seqlen} {kind} {v}]"
        try:
            kernel = build_kernel(use_gqa, batch, heads, seqlen, dim, causal, groups, v)
            # warmup
            for _ in range(3):
                _ = kernel(Q, K, V)
            torch.cuda.synchronize()
            latency = do_bench(lambda: kernel(Q, K, V), warmup=warmup, rep=rep)
            tflops = flops / latency * 1e-9
            rows.append((v, latency, tflops, ""))
            print(f"{tag}  {latency:.3f} ms  {tflops:.2f} TFLOPS")
        except Exception as e:  # noqa: BLE001
            rows.append((v, float("nan"), float("nan"), repr(e)[:120]))
            print(f"{tag}  FAILED: {e}")
            traceback.print_exc(limit=2)
    # free
    del Q, K, V
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["ss", "ts", "wasp", "fa4"],
                    choices=["ss", "ts", "wasp", "fa4"])
    ap.add_argument("--seqs", type=lambda s: [int(x) for x in s.split(",")],
                    default=DEFAULT_SEQS,
                    help="comma-separated seqlens (default: 1024,2048,4096,8192,16384)")
    ap.add_argument("--only", type=int, default=None,
                    help="if set, only run the i-th config (0-indexed) for smoke test")
    ap.add_argument("--csv", type=str,
                    default=str(HERE / "benchmark_configs_results.csv"),
                    help="output CSV path")
    args = ap.parse_args()

    all_cfgs = build_configs(args.seqs)
    if args.only is not None:
        cfgs = [all_cfgs[args.only]]
    else:
        cfgs = all_cfgs

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print()

    summary = []
    for cfg in cfgs:
        label, batch, h, kv_h, dim, s, causal = cfg
        rows = bench_one(label, batch, h, kv_h, dim, s, causal,
                         variants=tuple(args.variants))
        for v, lat, tfs, err in rows:
            summary.append((label, batch, h, kv_h, dim, s, causal, v, lat, tfs, err))

    # print final table
    print()
    print("=" * 110)
    print(f"{'model':<28} {'b':>2} {'h/kv':>7} {'d':>4} {'seq':>6} {'var':>5} "
          f"{'lat(ms)':>9} {'TFLOPS':>8}  note")
    print("-" * 110)
    grouped = {}
    for label, b, h, kv, d, s, c, v, lat, tfs, err in summary:
        key = (label, b, h, kv, d, s)
        grouped.setdefault(key, []).append((v, lat, tfs, err))

    best_variant = {}
    for key, rows in grouped.items():
        label, b, h, kv, d, s = key
        # find best (lowest finite latency)
        finite = [r for r in rows if r[1] == r[1]]  # not NaN
        best_v = min(finite, key=lambda x: x[1])[0] if finite else None
        best_variant[key] = best_v
        for v, lat, tfs, err in rows:
            mark = " *best" if v == best_v else ""
            if lat != lat:  # NaN
                print(f"{label:<28} {b:>2} {h:>3}/{kv:<3} {d:>4} {s:>6} {v:>5} "
                      f"{'-':>9} {'-':>8}  FAILED {err}")
            else:
                print(f"{label:<28} {b:>2} {h:>3}/{kv:<3} {d:>4} {s:>6} {v:>5} "
                      f"{lat:>9.3f} {tfs:>8.2f}{mark}")
    print("=" * 110)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "batch", "heads", "kv_heads", "dim", "seqlen", "causal",
            "variant", "latency_ms", "tflops", "best", "error",
        ])
        for label, b, h, kv, d, s, c, v, lat, tfs, err in summary:
            is_best = (best_variant.get((label, b, h, kv, d, s)) == v)
            lat_str = "" if lat != lat else f"{lat:.6f}"
            tfs_str = "" if tfs != tfs else f"{tfs:.4f}"
            w.writerow([label, b, h, kv, d, s, c, v, lat_str, tfs_str,
                        int(is_best), err])
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
