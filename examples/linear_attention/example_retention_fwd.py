import torch
import tilelang as tl
import tilelang.language as T
from tilelang.profiler import do_bench

import argparse


@tl.jit(out_idx=3, pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def chunk_retention_fwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    dtype: T.dtype = T.float16,
    scale: float = None,
) -> torch.Tensor:
    if scale is None:
        scale = DK**-0.5
    accum_dtype = T.float32

    chunk_size = 64
    BK = BV = 64  # Set to 128 can be faster, but has some numerical differences with FLA
    assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
    NK = tl.cdiv(DK, BK)
    NV = tl.cdiv(DV, BV)
    NT = tl.cdiv(S, chunk_size)

    @T.prim_func
    def chunk_retention_fwd(
        Q: T.Tensor([B, S, H, DK], dtype),  # type: ignore
        K: T.Tensor([B, S, H, DK], dtype),  # type: ignore
        V: T.Tensor([B, S, H, DV], dtype),  # type: ignore
        O: T.Tensor([NK, B, S, H, DV], dtype),  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H
            log_decay = T.alloc_var(T.float32)
            log_decay = T.log2(1 - T.exp2(-5.0 - 1.0 * i_h))  # Head-specific log decay

            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            T.clear(h)

            T.use_swizzle(10)

            for i in T.Pipelined(0, NT):
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(K[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v)

                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(row >= col, s[row, col] * T.exp2((row - col) * log_decay), 0)

                T.copy(h, h_shared)
                T.gemm(q, h_shared, o, clear_accum=True)
                for row, col in T.Parallel(chunk_size, BV):
                    o[row, col] = T.exp2((row + 1) * log_decay) * o[row, col]
                T.gemm(s_shared, v, o)

                for row, col in T.Parallel(chunk_size, BV):
                    v[row, col] = v[row, col] * T.exp2((chunk_size - row - 1) * log_decay)
                for row, col in T.Parallel(BK, BV):
                    h[row, col] = T.exp2(chunk_size * log_decay) * h[row, col]
                T.copy(o, O[i_k, i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV])
                T.gemm(k, v, h, transpose_A=True)

    return chunk_retention_fwd


def postprocess(o):
    return o if o.size(0) == 1 else o.sum(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8, help="Batch size")
    parser.add_argument("--S", type=int, default=4096, help="Seq len")
    parser.add_argument("--H", type=int, default=32, help="Num heads")
    parser.add_argument("--D", type=int, default=128, help="Head dim")
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D
    total_flops = 2.0 * B * S * S * H * D  # causal

    q = torch.randn((B, S, H, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, S, H, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, S, H, D), device="cuda", dtype=torch.float16)

    kernel = chunk_retention_fwd_kernel(B, S, H, D, D)

    t = do_bench(lambda: postprocess(kernel(q, k, v)), warmup=25, rep=100)
    print(f"Tilelang latency: {t:.3f} ms")
    print(f"Tilelang TFLOPs: {total_flops / t * 1e-9}")


if __name__ == "__main__":
    main()
