# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.profiler import do_bench

import argparse
from fla.ops.linear_attn import fused_chunk_linear_attn  # We compare with FLA


@tl.jit(
    out_idx=[4, 5, 6],
    pass_configs={
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True
    })
def chunk_linear_attn_bwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    dtype: str = 'float16',
    scale: float = None,
) -> torch.Tensor:

    if scale is None:
        scale = DK**-0.5
    accum_dtype = 'float'

    chunk_size = 64
    BK = BV = 64  # Set to 128 can be faster, but has some numerical differences with FLA
    assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
    NK = tl.cdiv(DK, BK)
    NV = tl.cdiv(DV, BV)
    NT = tl.cdiv(S, chunk_size)

    @T.prim_func
    def chunk_linear_attn_bwd(
            Q: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            K: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            V: T.Tensor([B, S, H, DV], dtype),  # type: ignore
            dO: T.Tensor([B, S, H, DV], dtype),  # type: ignore
            dQ: T.Tensor([NV, B, S, H, DK], dtype),  # type: ignore
            dK: T.Tensor([NV, B, S, H, DK], dtype),  # type: ignore
            dV: T.Tensor([NK, B, S, H, DV], dtype),  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            ds = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            ds_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            dq = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dk = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dv = T.alloc_fragment([chunk_size, BV], accum_dtype)
            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            do = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BV, BK], accum_dtype)
            h_shared = T.alloc_shared([BV, BK], dtype)
            dh = T.alloc_fragment([BK, BV], accum_dtype)
            dh_shared = T.alloc_shared([BK, BV], dtype)
            T.clear(h)
            T.clear(dh)

            T.annotate_layout({
                ds_shared: tl.layout.make_swizzled_layout(ds_shared),
                q: tl.layout.make_swizzled_layout(q),
                k: tl.layout.make_swizzled_layout(k),
                v: tl.layout.make_swizzled_layout(v),
                do: tl.layout.make_swizzled_layout(do),
                h_shared: tl.layout.make_swizzled_layout(h_shared),
                dh_shared: tl.layout.make_swizzled_layout(dh_shared)
            })
            T.use_swizzle(10)

            # Calculate dQ
            for i in T.Pipelined(0, NT, num_stages=1):
                T.copy(K[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v)
                T.copy(dO[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV],
                       do)

                T.gemm(do, v, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row >= col, ds[row, col], 0)

                T.gemm(ds_shared, k, dq, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(do, h_shared, dq)
                T.gemm(v, k, h, transpose_A=True)
                for row, col in T.Parallel(chunk_size, BK):
                    dq[row, col] *= scale
                T.copy(
                    dq, dQ[i_v, i_b, i * chunk_size:(i + 1) * chunk_size, i_h,
                           i_k * BK:(i_k + 1) * BK])

            # Calculate dK, dV (reversely)
            for i in T.Pipelined(1, NT + 1, num_stages=1):
                start = NT - i
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, start * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(
                    K[i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                      i_k * BK:(i_k + 1) * BK], k)
                T.copy(
                    V[i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                      i_v * BV:(i_v + 1) * BV], v)
                T.copy(
                    dO[i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                       i_v * BV:(i_v + 1) * BV], do)

                # Calculate dk
                T.gemm(
                    v, do, ds, transpose_B=True, clear_accum=True
                )  # ds here actually means `s`, but we simply reuse the buffer `ds`
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
                T.gemm(ds_shared, q, dk, clear_accum=True)
                T.copy(dh, dh_shared)
                T.gemm(v, dh_shared, dk, transpose_B=True)

                # Calculate dv
                T.gemm(k, q, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
                T.gemm(ds_shared, do, dv, clear_accum=True)
                T.gemm(k, dh_shared, dv)

                # Update dh
                T.gemm(q, do, dh, transpose_A=True)

                T.copy(
                    dk, dK[i_v, i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                           i_k * BK:(i_k + 1) * BK])
                T.copy(
                    dv, dV[i_k, i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                           i_v * BV:(i_v + 1) * BV])

    return chunk_linear_attn_bwd


def postprocess(dQ, dK, dV):
    dQ = dQ[0] if dQ.size(0) == 1 else dQ.sum(0)
    dK = dK[0] if dK.size(0) == 1 else dK.sum(0)
    dV = dV[0] if dV.size(0) == 1 else dV.sum(0)
    return dQ, dK, dV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=8, help='Batch size')
    parser.add_argument('--S', type=int, default=4096, help='Seq len')
    parser.add_argument('--H', type=int, default=32, help='Num heads')
    parser.add_argument('--D', type=int, default=256, help='Head dim')
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D

    q = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16, requires_grad=True)
    do = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)

    kernel = chunk_linear_attn_bwd_kernel(B, S, H, D, D)
    dq, dk, dv = postprocess(*kernel(q, k, v, do))
    o_ref, _ = fused_chunk_linear_attn(q, k, v, output_final_state=True, normalize=False)
    o_ref.backward(do, retain_graph=True)
    if torch.allclose(dq, q.grad) and torch.allclose(dk, k.grad) and torch.allclose(dv, v.grad):
        print('Passed all tests!✅')
    else:
        print('Failed some tests!❌')
    t1 = do_bench(lambda: o_ref.backward(do, retain_graph=True), warmup=25, rep=100)
    q.grad = k.grad = v.grad = None
    o_ref, _ = fused_chunk_linear_attn(q, k, v, output_final_state=True, normalize=False)
    t2 = do_bench(lambda: postprocess(*kernel(q, k, v, do)), warmup=25, rep=100)
    print(f'Triton latency: {t1:.3f} ms')
    print(f'TileLang latency: {t2:.3f} ms')
    print(f'Speedup: {t1/t2:.3f}x')


if __name__ == '__main__':
    main()
