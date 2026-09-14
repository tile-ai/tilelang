"""Sustained same-data benchmark: TileLang SM120 SageAttention3 forward vs the original kernel.

kernel-only: both kernels consume the same pre-quantized NVFP4 tensors derived from one bf16 Q/K/V
(byte-identical inputs; only the layouts differ). Timing: a CUDA graph of n_unroll launches
replayed for >= sustain_secs (power-wall steady state), TOPS = 4*B*H*N^2*D / t.

  python bench_sageattn3.py --sizes 1024,4096 --num-stages 3          # TileLang only
  python bench_sageattn3.py --arms tl,orig --sustain-secs 20           # both (needs fp4attn_cuda)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402
from examples.attention_sm120.sm120_sageattn3_fwd import prepare_inputs, sm120_sageattn3_fwd  # noqa: E402


def sustained(fn, n_unroll: int, sustain_secs: float, warmup: int = 3) -> float:
    """ms per call: CUDA-graph replay of n_unroll calls for >= sustain_secs."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(n_unroll):
                fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iters = 0
    t0 = time.time()
    start.record()
    while time.time() - t0 < sustain_secs:
        g.replay()
        iters += n_unroll
        if iters % (n_unroll * 4) == 0:
            torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _tl_fn(q, k, v, n, d, num_stages, threads):
    b, h = q.shape[0], q.shape[1]
    _, kargs, _ = prepare_inputs(q, k, v)
    kernel = sm120_sageattn3_fwd(b, h, n, d, num_stages=num_stages, threads=threads)

    def fn():
        return kernel(*kargs)

    return fn


def _orig_fn(q, k, v, n, d):
    import fp4attn_cuda

    q_sm, k_sm, v_p, _, delta_s = sq.preprocess_qkv(q, k, v)
    canon = sq.quantize_canonical(q_sm, k_sm, v_p)
    s = sq.export_for_sage(canon)
    inp = (s["q"], s["k"], s["v"], s["sfq"], s["sfk"], s["sfv"], delta_s)
    scale = d**-0.5

    def fn():
        return fp4attn_cuda.fwd(*inp, n, None, scale, False, True, True)

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--arms", default="tl", help="comma list of tl,orig")
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--sustain-secs", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    b, h, d = args.batch, args.heads, args.dim
    arms = args.arms.split(",")
    rows = []
    print(f"# TileLang SageAttention3 fwd B={b} H={h} D={d} non-causal, sustained {args.sustain_secs}s/arm, cudagraph")
    print(f"# tl: num_stages={args.num_stages} threads={args.threads}")
    for n in (int(x) for x in args.sizes.split(",")):
        torch.manual_seed(args.seed)
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        flops = 4.0 * b * h * n * n * d
        n_unroll = 8 if n <= 4096 else (2 if n <= 16384 else 1)
        line = f"N={n:6d}"
        for arm in arms:
            fn = _tl_fn(q, k, v, n, d, args.num_stages, args.threads) if arm == "tl" else _orig_fn(q, k, v, n, d)
            ms = sustained(fn, n_unroll, args.sustain_secs)
            tops = flops / (ms * 1e-3) / 1e12
            rows.append({"arm": arm, "N": n, "ms": ms, "tops": tops})
            line += f"  {arm}: {ms:9.4f} ms {tops:7.1f} TOPS"
            del fn
            torch.cuda.empty_cache()
        print(line, flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
