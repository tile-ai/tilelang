"""S0 baseline: original SageAttention3 (sageattn3_blackwell) on this GPU.

kernel-only = fp4attn_cuda.fwd on pre-quantized inputs; e2e = sageattn3_blackwell(q,k,v).
Timing: CUDA graph of n_unroll calls, replayed for >= sustain_secs (power-wall steady state),
TOPS = 4*B*H*N^2*D / t. Optional NVML clock/power sampling (10 Hz counter -> 20 Hz polling).
"""

import argparse
import json
import statistics
import threading
import time

import torch

import fp4attn_cuda
from sageattn3 import api as sage_api
from sageattn3 import sageattn3_blackwell


def prepare_kernel_inputs(q, k, v):
    q_o, k_o, v_o, ds = sage_api.preprocess_qkv(q.clone(), k.clone(), v.clone(), True)
    q_pk, q_sf = sage_api.scale_and_quant_fp4(q_o)
    k_pk, k_sf = sage_api.scale_and_quant_fp4_permute(k_o)
    v_pk, v_sf = sage_api.scale_and_quant_fp4_transpose(v_o)
    return (q_pk, k_pk, v_pk, q_sf, k_sf, v_sf, ds)


class Sampler(threading.Thread):
    def __init__(self, dev_index, hz=20):
        super().__init__(daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(dev_index)
        self.nv = pynvml
        self.period = 1.0 / hz
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            clk = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
            pwr = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            self.samples.append((clk, pwr))
            time.sleep(self.period)


def sustained(fn, n_unroll, sustain_secs, warmup=3):
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
    return start.elapsed_time(end) / iters  # ms per call


def _make_kernel_fn(q, k, v, n, d):
    inp = prepare_kernel_inputs(q, k, v)
    softmax_scale = d ** (-0.5)

    def fn():
        return fp4attn_cuda.fwd(*inp, n, None, softmax_scale, False, True, True)

    return fn


def _make_e2e_fn(q, k, v):
    def fn():
        return sageattn3_blackwell(q, k, v, is_causal=False)

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--sustain-secs", type=float, default=20.0)
    ap.add_argument("--modes", default="kernel,e2e")
    ap.add_argument("--nvml-index", type=int, default=-1, help="physical GPU index for NVML sampling (-1 = off)")
    ap.add_argument("--label", default="as-shipped")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = args.dim
    results = []
    print(
        f"# SageAttention3 baseline [{args.label}] B={args.batch} H={args.heads} D={d} non-causal, sustained {args.sustain_secs}s/arm",
        flush=True,
    )
    print(f"{'N':>6s} | {'mode':6s} | {'ms/call':>9s} | {'TOPS':>7s} | {'clk_med':>7s} | {'pwr_med':>7s}", flush=True)
    for n in (int(s) for s in args.sizes.split(",")):
        flops = 4.0 * args.batch * args.heads * n * n * d
        q = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=torch.bfloat16)
        ms_per_call_target = flops / 1.0e15 * 1e3  # ~1 PFLOPS guess
        n_unroll = max(1, min(64, int(200.0 / max(ms_per_call_target, 1e-3))))  # ~200 ms per graph
        for mode in args.modes.split(","):
            fn = _make_kernel_fn(q, k, v, n, d) if mode == "kernel" else _make_e2e_fn(q, k, v)
            sampler = Sampler(args.nvml_index) if args.nvml_index >= 0 else None
            if sampler:
                sampler.start()
            ms = sustained(fn, n_unroll, args.sustain_secs)
            if sampler:
                sampler.stop.set()
                sampler.join()
                body = sampler.samples[len(sampler.samples) // 5 : -max(1, len(sampler.samples) // 5)]
                clk = statistics.median(c for c, _ in body)
                pwr = statistics.median(p for _, p in body)
            else:
                clk = pwr = float("nan")
            tops = flops / (ms * 1e-3) / 1e12
            results.append(dict(label=args.label, n=n, mode=mode, ms=ms, tops=tops, clk=clk, pwr=pwr))
            print(f"{n:>6d} | {mode:6s} | {ms:9.4f} | {tops:7.1f} | {clk:7.0f} | {pwr:7.0f}", flush=True)
            del fn
            torch.cuda.empty_cache()
        del q, k, v
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
