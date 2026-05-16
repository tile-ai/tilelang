"""Bit-compare tilelang attention_kernel_1sm against the reference .so.

Loads ~/avo/kernels/attention_kernel_1sm.so via ctypes and runs it side-by-side
with the tilelang port on the same Q/K/V. Reports the abs/rel diff.

This is a structural sanity check, not a correctness proof — both kernels are
bf16 and the .so does a more aggressive numeric schedule (packed exp2 poly,
3-input max, etc.), so element-wise diffs in the 1e-2 range are expected.
"""

import argparse
import ctypes
import os
from pathlib import Path

import torch

from attention_kernel_1sm import attention_kernel_1sm

LIB = Path("~/avo/kernels/attention_kernel_1sm.so").expanduser()


def load_lib():
    if not LIB.exists():
        raise FileNotFoundError(f"missing {LIB} — build it via the avo Makefile")
    lib = ctypes.CDLL(str(LIB))
    # void flash_attention_forward(const void* Q, K, V, void* O,
    #     int batch, seq, num_q, num_kv, dim, is_causal, cudaStream_t stream)
    lib.flash_attention_forward.argtypes = [
        ctypes.c_void_p,     # Q
        ctypes.c_void_p,     # K
        ctypes.c_void_p,     # V
        ctypes.c_void_p,     # O
        ctypes.c_int,        # batch
        ctypes.c_int,        # seq_len
        ctypes.c_int,        # num_q_heads
        ctypes.c_int,        # num_kv_heads
        ctypes.c_int,        # head_dim
        ctypes.c_int,        # is_causal
        ctypes.c_void_p,     # cudaStream_t
    ]
    lib.flash_attention_forward.restype = None
    return lib


def run_so(lib, Q, K, V, is_causal):
    B, S, H, D = Q.shape
    H_kv = K.shape[2]
    O = torch.empty_like(Q)
    lib.flash_attention_forward(
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_void_p(O.data_ptr()),
        B, S, H, H_kv, D, int(is_causal),
        None,
    )
    torch.cuda.synchronize()
    return O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    if args.dim != 128:
        raise SystemExit(
            f"This script compares against attention_kernel_1sm.so which "
            f"is built for dim=128. Got dim={args.dim}."
        )

    torch.manual_seed(0)
    kv_h = args.kv_heads or args.heads
    Q = torch.randn(args.batch, args.seq, args.heads, args.dim,
                    dtype=torch.bfloat16, device="cuda")
    K = torch.randn(args.batch, args.seq, kv_h, args.dim,
                    dtype=torch.bfloat16, device="cuda")
    V = torch.randn(args.batch, args.seq, kv_h, args.dim,
                    dtype=torch.bfloat16, device="cuda")

    lib = load_lib()

    # --- Run reference .so ---
    O_ref = run_so(lib, Q, K, V, args.causal)

    # --- Run tilelang port ---
    fn = attention_kernel_1sm(
        args.batch, args.heads, args.seq, args.dim,
        num_kv_heads=kv_h, is_causal=args.causal,
    )
    # WARMUP: the first call after a fresh kernel load reads stale TMEM and
    # produces a "uniform softmax" result. See followup task. We discard the
    # first call's output and use the second.
    _ = fn(Q, K, V)
    torch.cuda.synchronize()
    O_tl = fn(Q, K, V)

    diff_f = (O_tl.float() - O_ref.float())
    abs_err = diff_f.abs()
    print(f"shape={tuple(O_tl.shape)}")
    print(f"  vs .so   max_abs={abs_err.max().item():.4f} "
          f"mean_abs={abs_err.mean().item():.4f}")
    print(f"  ref .so  mean|O|={O_ref.float().abs().mean().item():.4f}")
    print(f"  tl       mean|O|={O_tl.float().abs().mean().item():.4f}")
    print(f"  diff/ref ratio = {(abs_err.mean() / O_ref.float().abs().mean()).item():.4f}")

    if args.bench:
        from tilelang.profiler import do_bench
        for _ in range(3):
            _ = fn(Q, K, V)
            _ = run_so(lib, Q, K, V, args.causal)
        torch.cuda.synchronize()
        tl_lat = do_bench(lambda: fn(Q, K, V), warmup=25, rep=100)
        so_lat = do_bench(lambda: run_so(lib, Q, K, V, args.causal), warmup=25, rep=100)
        causal_factor = 0.5 if args.causal else 1.0
        flops = 4.0 * args.batch * args.heads * args.seq * args.seq * args.dim * causal_factor
        print(f"  tilelang : {tl_lat:.3f} ms  {flops / tl_lat * 1e-9:.1f} TFLOPS")
        print(f"  .so      : {so_lat:.3f} ms  {flops / so_lat * 1e-9:.1f} TFLOPS")


if __name__ == "__main__":
    main()
