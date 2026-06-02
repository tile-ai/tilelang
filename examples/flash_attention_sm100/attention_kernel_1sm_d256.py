"""TileLang 1SM D256 wrapper for the SM100 FlashAttention kernel.

The actual kernel builder lives in attention_kernel_1sm.py — that file is
parameterized on `dim`, and was modeled to support both d=128 and d=256 (the
existing flashattn_simple_ws example in this directory is benchmarked for both
GLM-5 d=256 and Llama-4-Maverick d=128, so the same shape works).

The D256 shape uses a smaller pipeline (kQStages=1, kKVStages=1) because the
larger SMEM/TMEM footprint leaves less room for deeper pipelining. In this
baseline we already use kQStages=1 in both files, so the main difference here
is the default head_dim and a tighter kv_stages default.

Slice 4 (TMEM aliasing) will let the d=128 variant grow back to kQStages=2.
Slice 3 (TMEM column-slice for chunked O rescale) lets the d=256 variant
reduce its register footprint dramatically — it's the biggest perf win here.
"""

import argparse
from typing import Optional

import torch

from attention_kernel_1sm import attention_kernel_1sm, reference_attention


# ``kv_stages=1`` drops the K/V pipeline depth from 3 (in d=128) to 1 (in
# d=256) because the per-stage SMEM footprint is twice as large. The user can
# override via kwargs if they have enough SMEM headroom.
def attention_kernel_1sm_d256(
    batch: int,
    heads: int,
    seq_len: int,
    num_kv_heads: Optional[int] = None,
    is_causal: bool = False,
    block_M: int = 128,
    block_N: int = 128,
    kv_stages: int = 1,
    # KNOWN LIMITATION: d=256 currently exceeds B200's 228KB dynamic SMEM
    # cap. Paired K/V SMEM buffers make this worse; for block_M=128 d=256
    # they need ~360KB. A compact D256 schedule fits in 192KB by:
    #   (a) single-buffered K/V (kKVStages=1, no pairing), and
    #   (b) aliasing the Q SMEM buffer as the O-epilogue staging buffer
    #       once Q is no longer live.
    # Both need tilelang support that isn't there yet — single-buffered K/V
    # at the prim_func level (Slice 4c idea) and explicit SMEM buffer
    # aliasing (Slice 4 P2a). Until then, this kernel fails at launch with
    # "Failed to set the allowed dynamic shared memory size".
    p_storage: str = "shared",
):
    return attention_kernel_1sm(
        batch,
        heads,
        seq_len,
        dim=256,
        num_kv_heads=num_kv_heads,
        is_causal=is_causal,
        block_M=block_M,
        block_N=block_N,
        kv_stages=kv_stages,
        p_storage=p_storage,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    kv_h = args.kv_heads or args.heads
    Q = torch.randn(args.batch, args.seq, args.heads, 256,
                    dtype=torch.bfloat16, device="cuda")
    K = torch.randn(args.batch, args.seq, kv_h, 256,
                    dtype=torch.bfloat16, device="cuda")
    V = torch.randn(args.batch, args.seq, kv_h, 256,
                    dtype=torch.bfloat16, device="cuda")

    fn = attention_kernel_1sm_d256(
        args.batch,
        args.heads,
        args.seq,
        num_kv_heads=kv_h,
        is_causal=args.causal,
    )
    O = fn(Q, K, V)
    O_ref = reference_attention(Q, K, V, is_causal=args.causal)

    err_abs = (O.to(torch.float32) - O_ref.to(torch.float32)).abs()
    print(
        f"shape={tuple(O.shape)}  "
        f"max_abs={err_abs.max().item():.4f}  "
        f"mean_abs={err_abs.mean().item():.4f}"
    )

    if args.bench:
        from tilelang.profiler import do_bench
        for _ in range(3):
            _ = fn(Q, K, V)
        torch.cuda.synchronize()
        lat = do_bench(lambda: fn(Q, K, V), warmup=25, rep=100)
        causal_factor = 0.5 if args.causal else 1.0
        flops = 2.0 * 2.0 * args.batch * args.heads * args.seq * args.seq * 256 * causal_factor
        tflops = flops / lat * 1e-9
        print(f"latency={lat:.3f} ms  perf={tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
