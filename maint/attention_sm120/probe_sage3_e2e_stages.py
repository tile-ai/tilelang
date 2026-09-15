"""E0: where does the original SageAttention3 e2e call spend its time?

Times ``sageattn3_blackwell(q, k, v)`` as shipped and each of its stages in isolation, every
arm as its own CUDA graph replayed for ``--secs`` seconds on static inputs (at 1K the whole
call is ~0.25 ms, so eager timing would mostly measure kernel launches). The stage sum is
reported next to the e2e number; the difference is what the stages do not capture
(allocator, graph-boundary effects).

Stages follow ``sageattn3/api.py`` in order:
  kmean   k -= k.mean(dim=-2, keepdim=True)           (bf16, in place)
  pad     pad_128 -> .contiguous() when N % 128 == 0
  qmean   triton_group_mean(q)                        (q_out and qm)
  matmul  delta_s = matmul(qm, k^T)                   (bf16)
  to32    delta_s.to(float32).contiguous()
  quantq  scale_and_quant_fp4(q)
  quantk  scale_and_quant_fp4_permute(k)
  quantv  scale_and_quant_fp4_transpose(v)
  attn    fp4attn_cuda.fwd(...)                       (the kernel-only number)
  slice   o[:, :, :QL, :].contiguous()

Usage: python probe_sage3_e2e_stages.py --sizes 1024,32768 --secs 5
"""

import argparse
import time

import torch

import fp4attn_cuda
from sageattn3 import api as sage_api
from sageattn3 import sageattn3_blackwell


def sustained_ms(fn, secs, n_unroll):
    """ms per call of ``fn`` replayed as one CUDA graph of ``n_unroll`` calls."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(n_unroll):
            fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    iters, t0 = 0, time.time()
    start.record()
    while time.time() - t0 < secs:
        g.replay()
        iters += n_unroll
        if iters % (4 * n_unroll) == 0:
            torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _stages(q, k, v, n, d):
    """Static inputs produced once by the real pipeline, and one timed callable per stage."""
    q_p, k_p, v_p, ds = sage_api.preprocess_qkv(q.clone(), k.clone(), v.clone(), True)
    _, qm = sage_api.triton_group_mean(q.contiguous())
    ds16 = torch.matmul(qm, k_p.transpose(-2, -1))
    q_pk, q_sf = sage_api.scale_and_quant_fp4(q_p)
    k_pk, k_sf = sage_api.scale_and_quant_fp4_permute(k_p)
    v_pk, v_sf = sage_api.scale_and_quant_fp4_transpose(v_p)
    o = fp4attn_cuda.fwd(q_pk, k_pk, v_pk, q_sf, k_sf, v_sf, ds, n, None, d**-0.5, False, True, True)[0]
    k_work = k.clone()
    return {
        "kmean": lambda: k_work.sub_(k_work.mean(dim=-2, keepdim=True)),
        "pad": lambda: (q.contiguous(), k_work.contiguous(), v.contiguous()),
        "qmean": lambda: sage_api.triton_group_mean(q),
        "matmul": lambda: torch.matmul(qm, k_p.transpose(-2, -1)),
        "to32": lambda: ds16.to(torch.float32).contiguous(),
        "quantq": lambda: sage_api.scale_and_quant_fp4(q_p),
        "quantk": lambda: sage_api.scale_and_quant_fp4_permute(k_p),
        "quantv": lambda: sage_api.scale_and_quant_fp4_transpose(v_p),
        "attn": lambda: fp4attn_cuda.fwd(q_pk, k_pk, v_pk, q_sf, k_sf, v_sf, ds, n, None, d**-0.5, False, True, True),
        "slice": lambda: o[:, :, :n, :].contiguous(),
    }


def _measure(n, b, h, d, secs):
    torch.manual_seed(0)
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    flops = 4.0 * b * h * n * n * d
    n_unroll = max(1, min(64, int(200.0 / max(flops / 1.0e15 * 1e3, 1e-3))))
    e2e_k = k.clone()  # the original subtracts the K mean in place
    ms_e2e = sustained_ms(lambda: sageattn3_blackwell(q, e2e_k, v, is_causal=False), secs, n_unroll)
    parts = {}
    for name, fn in _stages(q, k, v, n, d).items():
        parts[name] = sustained_ms(fn, secs, n_unroll)
        torch.cuda.empty_cache()
    return ms_e2e, flops, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--secs", type=float, default=5.0)
    args = ap.parse_args()
    b, h, d = args.batch, args.heads, 128
    print(f"# original SageAttention3 e2e stages, B={b} H={h} D={d}, {args.secs:.0f} s per arm", flush=True)
    for n in (int(x) for x in args.sizes.split(",")):
        if n % 128 != 0:
            raise ValueError(f"seq_len must be a multiple of 128, got {n}")
        ms_e2e, flops, parts = _measure(n, b, h, d, args.secs)
        total = sum(parts.values())
        tops = flops / (ms_e2e * 1e-3) / 1e12
        line = " ".join(f"{name}={ms:.3f}" for name, ms in parts.items())
        print(f"N={n:6d} e2e={ms_e2e:.3f} ms ({tops:.0f} TOPS) stage_sum={total:.3f} residual={ms_e2e - total:+.3f} | {line}", flush=True)
        print("         shares: " + " ".join(f"{name}={100 * ms / total:.1f}%" for name, ms in parts.items()), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
