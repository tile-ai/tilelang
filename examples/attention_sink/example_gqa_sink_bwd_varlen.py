import torch
import tilelang
from tilelang.profiler import do_bench
import tilelang.language as T
import argparse
from typing import Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../flash_attention"))
from varlen_utils import generate_random_padding_mask, generate_qkv


def get_bwd_configs():
    sm_major, sm_minor = torch.cuda.get_device_capability()
    sm_version = sm_major * 10 + sm_minor
    if sm_version == 80:
        return 64, 32, 1, 128
    else:
        return 128, 32, 2, 256


@tilelang.jit(
    out_idx=[6, 7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_fwd(
    batch_size,
    groups,
    UQ,
    UKV,
    N_CTX,
    heads,
    max_seq_len,
    dim,
    is_causal,
    window_size=None,  # None for full causal attention
    sm_scale=None,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
    dtype=T.float16,
):
    if window_size is not None:
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    o_shape = [UQ, heads, dim]
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(kv_shape, dtype),
        V_unpad: T.Tensor(kv_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        Sinks: T.Tensor([heads], dtype),
        Output_unpad: T.Tensor(o_shape, dtype),
        lse: T.Tensor([batch_size, heads, N_CTX], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(max_seq_len, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            sinks = T.alloc_fragment([block_M], dtype)

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // groups

            q_start_idx = cu_seqlens_q[batch_idx]
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = k_end_idx - kv_start_idx

            T.copy(Q_unpad[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            for i in T.Parallel(block_M):
                sinks[i] = Sinks[head_idx]

            offset = kv_current_seqlen - q_current_seqlen  # always align on the right
            max_visible_k_idx = offset + (bx + 1) * block_M

            # Determine loop range based on causal mask and sliding window
            if is_causal:
                if window_size is not None:
                    start = T.max(0, (offset + bx * block_M - window_size + 1) // block_N)
                    end = T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
                else:
                    start = 0
                    end = T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
            else:
                if window_size is not None:
                    start = T.max(0, (offset + bx * block_M - window_size + 1) // block_N)
                    end = T.ceildiv(kv_current_seqlen, block_N)
                else:
                    start = 0
                    end = T.ceildiv(kv_current_seqlen, block_N)

            loop_range = end - start

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                actual_k = k + start
                T.copy(K_unpad[kv_start_idx + actual_k * block_N : kv_start_idx + (actual_k + 1) * block_N, kv_head_idx, :], K_shared)

                # Build mask considering causal, sliding window, and padding
                if is_causal:
                    if window_size is not None:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + offset
                            k_idx = actual_k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_idx < k_idx)
                                or (q_idx >= k_idx + window_size)
                                or (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                (bx * block_M + i + offset < actual_k * block_N + j)
                                or (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )
                else:
                    if window_size is not None:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + offset
                            k_idx = actual_k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_idx >= k_idx + window_size)
                                or (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(V_unpad[kv_start_idx + actual_k * block_N : kv_start_idx + (actual_k + 1) * block_N, kv_head_idx, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                # Handle case where scores_max is -inf (query sees no keys due to causal mask or sliding window)
                # This can happen when q_len > k_len (offset < 0) in causal attention, or with sliding window
                for i in T.Parallel(block_M):
                    scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            # Attention sink: add sink contribution to logsum
            for i in T.Parallel(block_M):
                logsum[i] += T.exp2(sinks[i] * 1.44269504 - scores_max[i] * scale)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] = 0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]

            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = acc_o[i, d]

            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            for i in T.Parallel(block_M):
                if bx * block_M + i < q_current_seqlen:
                    lse[bz, head_idx, bx * block_M + i] = logsum[i]

    return main


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_preprocess(batch_size, heads, UQ, N_CTX, max_seq_len, dim, dtype: T.dtype = T.float16):
    accum_dtype = T.float32
    shape = [UQ, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        Delta: T.Tensor([batch_size, heads, N_CTX], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(max_seq_len, blk), batch_size) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx

            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                for i, j in T.Parallel(blk, blk):
                    if by * blk + i < q_current_seqlen and k * blk + j < dim:
                        o[i, j] = O[q_start_idx + by * blk + i, bx, k * blk + j]
                        do[i, j] = dO[q_start_idx + by * blk + i, bx, k * blk + j]
                    else:
                        o[i, j] = 0.0
                        do[i, j] = 0.0
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)

            for i in T.Parallel(blk):
                if by * blk + i < q_current_seqlen:
                    Delta[bz, bx, by * blk + i] = delta[i]

    return flash_bwd_prep


def make_dq_layout(dQ):
    # Reorder dq for atomic add: [seq, head, dim] -> permuted layout
    return T.Layout(dQ.shape, lambda l, h, d: [h, l, d])


@tilelang.jit(
    out_idx=[1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_postprocess(UQ, heads, dim, dtype: T.dtype = T.float16):
    accum_dtype = T.float32
    shape = [UQ, heads, dim]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(UQ, blk), heads, threads=128) as (bx, by):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_post


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def flashattn_bwd(
    batch_size,
    groups,
    UQ,
    UKV,
    N_CTX,
    heads,
    max_seq_len,
    dim,
    is_causal,
    window_size=None,
    sm_scale=None,
    dtype=T.float16,
):
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    accum_dtype = T.float32

    block_M, block_N, num_stages, threads = get_bwd_configs()

    if window_size is not None:
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(q_shape, dtype),
        lse: T.Tensor([batch_size, heads, N_CTX], accum_dtype),
        Delta: T.Tensor([batch_size, heads, N_CTX], accum_dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        dQ: T.Tensor(q_shape, accum_dtype),
        dK: T.Tensor(kv_shape, accum_dtype),
        dV: T.Tensor(kv_shape, accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(max_seq_len, block_M), batch_size, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], accum_dtype)
            dk_shared = T.alloc_shared([block_M, dim], accum_dtype)

            q_start_idx = cu_seqlens_q[bz]
            kv_start_idx = cu_seqlens_k[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            k_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = k_end_idx - kv_start_idx

            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                }
            )
            T.copy(K[kv_start_idx + by * block_M : kv_start_idx + (by + 1) * block_M, bx // groups, :], K_shared)
            T.copy(V[kv_start_idx + by * block_M : kv_start_idx + (by + 1) * block_M, bx // groups, :], V_shared)
            T.clear(dv)
            T.clear(dk)

            # For varlen causal attention, we need to account for offset between q and kv lengths
            # In forward: Q at pos q can see KV at pos k if q + offset >= k (where offset = kv_len - q_len)
            # In backward: KV at pos kv_pos is seen by Q at pos q_pos if kv_pos <= q_pos + offset
            offset = kv_current_seqlen - q_current_seqlen

            # loop_st: first Q block that can see this KV block
            # kv_pos <= q_pos + offset => by * block_M <= k * block_N + offset
            # => k >= (by * block_M - offset) / block_N
            loop_st = T.max(0, T.floordiv(by * block_M - offset, block_N)) if is_causal else 0
            loop_ed = (
                T.min(T.ceildiv((by + 1) * block_M - offset + window_size, block_N), T.ceildiv(q_current_seqlen, block_N))
                if window_size is not None
                else T.ceildiv(q_current_seqlen, block_N)
            )

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[q_start_idx + k * block_N : q_start_idx + (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    if window_size is not None:
                        for i, j in T.Parallel(block_M, block_N):
                            # Causal: kv_pos <= q_pos + offset
                            # Sliding window: kv_pos > q_pos + offset - window_size
                            qkT[i, j] = T.if_then_else(
                                (by * block_M + i <= k * block_N + j + offset)
                                and (by * block_M + i > k * block_N + j + offset - window_size)
                                and (by * block_M + i < kv_current_seqlen and k * block_N + j < q_current_seqlen),
                                qkT[i, j],
                                0,
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            # Causal: kv_pos <= q_pos + offset
                            qkT[i, j] = T.if_then_else(
                                (by * block_M + i <= k * block_N + j + offset)
                                and (by * block_M + i < kv_current_seqlen and k * block_N + j < q_current_seqlen),
                                qkT[i, j],
                                0,
                            )
                else:
                    if window_size is not None:
                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(
                                (by * block_M + i > k * block_N + j + offset - window_size)
                                and (by * block_M + i < kv_current_seqlen and k * block_N + j < q_current_seqlen),
                                qkT[i, j],
                                0,
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(
                                by * block_M + i < kv_current_seqlen and k * block_N + j < q_current_seqlen,
                                qkT[i, j],
                                0,
                            )

                T.copy(dO[q_start_idx + k * block_N : q_start_idx + (k + 1) * block_N, bx, :], dst=do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                T.atomic_add(dQ[q_start_idx + k * block_N : q_start_idx + (k + 1) * block_N, bx, :], dq)

            T.copy(dv, dv_shared)
            T.atomic_add(dV[kv_start_idx + by * block_M : kv_start_idx + (by + 1) * block_M, bx // groups, :], dv_shared)
            T.copy(dk, dk_shared)
            T.atomic_add(dK[kv_start_idx + by * block_M : kv_start_idx + (by + 1) * block_M, bx // groups, :], dk_shared)

    return flash_bwd


@tilelang.jit(out_idx=-1)
def flashattn_bwd_dsink(batch_size, heads, N_CTX, max_seq_len, block=256, dtype: T.dtype = T.float16):
    accum_dtype = T.float32
    shape = [batch_size, heads, N_CTX]

    @T.prim_func
    def flash_bwd_dsink(
        Sinks: T.Tensor([heads], dtype),
        Delta: T.Tensor(shape, accum_dtype),
        lse: T.Tensor(shape, accum_dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        dsinks: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(max_seq_len, block), batch_size, threads=256) as (bx, by, bz):
            lse_fragment = T.alloc_fragment([block], accum_dtype)
            delta_fragment = T.alloc_fragment([block], accum_dtype)
            dsink_fragment = T.alloc_fragment([block], dtype)

            # Get actual sequence length for this batch item
            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx

            sink = Sinks[bx]
            T.copy(lse[bz, bx, by * block : (by + 1) * block], lse_fragment)
            T.copy(Delta[bz, bx, by * block : (by + 1) * block], delta_fragment)
            for i in T.Parallel(block):
                # Only compute for valid positions, set 0 for positions beyond sequence length
                dsink_fragment[i] = T.if_then_else(
                    by * block + i < q_current_seqlen,
                    -T.exp2(sink * 1.44269504 - lse_fragment[i]) * delta_fragment[i],
                    0,
                )
            T.copy(dsink_fragment, dsinks[bz, bx, by * block : (by + 1) * block])

    return flash_bwd_dsink


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q_unpad, k_unpad, v_unpad, sinks, cu_seqlens_q, cu_seqlens_k, N_CTX, max_seqlen_q, max_seqlen_k, window_size, groups, is_causal
    ):
        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        q_unpad, k_unpad, v_unpad, sinks = [maybe_contiguous(x) for x in (q_unpad, k_unpad, v_unpad, sinks)]
        UQ, H, D_HEAD = q_unpad.shape
        UKV = k_unpad.shape[0]
        batch_size = cu_seqlens_q.shape[0] - 1
        dtype = T.float16 if q_unpad.dtype == torch.float16 else T.bfloat16

        kernel = flashattn_fwd(
            batch_size,
            groups,
            UQ,
            UKV,
            N_CTX,
            H,
            max_seqlen_q,
            D_HEAD,
            is_causal,
            window_size=window_size,
            block_M=64,
            block_N=64,
            num_stages=1,
            threads=128,
            dtype=dtype,
        )
        o_unpad, lse = kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, sinks)

        ctx.save_for_backward(q_unpad, k_unpad, v_unpad, sinks, o_unpad, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.window_size = window_size
        ctx.groups = groups
        ctx.is_causal = is_causal
        ctx.N_CTX = N_CTX
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.batch_size = batch_size
        return o_unpad

    @staticmethod
    def backward(ctx, do):
        q_unpad, k_unpad, v_unpad, sinks, o_unpad, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        UQ, H, D_HEAD = q_unpad.shape
        UKV = k_unpad.shape[0]
        groups = ctx.groups
        batch_size = ctx.batch_size
        dtype = T.float16 if q_unpad.dtype == torch.float16 else T.bfloat16

        kernel_prep = flashattn_bwd_preprocess(batch_size, H, UQ, ctx.N_CTX, ctx.max_seqlen_q, D_HEAD, dtype=dtype)
        kernel_post = flashattn_bwd_postprocess(UQ, H, D_HEAD, dtype=dtype)
        delta = kernel_prep(o_unpad, do, cu_seqlens_q)

        kernel = flashattn_bwd(
            batch_size,
            groups,
            UQ,
            UKV,
            ctx.N_CTX,
            H,
            ctx.max_seqlen_q,
            D_HEAD,
            ctx.is_causal,
            window_size=ctx.window_size,
            dtype=dtype,
        )

        head_kv = H // groups
        dq = torch.zeros_like(q_unpad, dtype=torch.float32)
        dk = torch.zeros([UKV, head_kv, D_HEAD], dtype=torch.float32, device=q_unpad.device)
        dv = torch.zeros([UKV, head_kv, D_HEAD], dtype=torch.float32, device=q_unpad.device)

        kernel(q_unpad, k_unpad, v_unpad, do, lse, delta, cu_seqlens_q, cu_seqlens_k, dq, dk, dv)
        dq = kernel_post(dq)
        dk = dk.to(q_unpad.dtype)
        dv = dv.to(q_unpad.dtype)

        kernel_dsink = flashattn_bwd_dsink(batch_size, H, ctx.N_CTX, ctx.max_seqlen_q, dtype=dtype)
        dsinks = kernel_dsink(sinks, delta, lse, cu_seqlens_q).sum(0).sum(1)

        return dq, dk, dv, dsinks, None, None, None, None, None, None, None, None


attention = _attention.apply


def ref_program(
    q_unpad: torch.Tensor,
    k_unpad: torch.Tensor,
    v_unpad: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sinks: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    sliding_window: Optional[int] = None,
    groups: int = 1,
) -> torch.Tensor:
    """Reference implementation for varlen attention with sinks."""
    total_q, num_heads, head_dim = q_unpad.shape
    _, num_key_value_heads, _ = k_unpad.shape

    sm_scale = 1.0 / head_dim**0.5

    output = torch.zeros_like(q_unpad)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()

        q_len = q_end - q_start
        k_len = k_end - k_start

        if q_len == 0:
            continue

        q_seq = q_unpad[q_start:q_end]  # [q_len, heads, dim]
        k_seq = k_unpad[k_start:k_end]  # [k_len, head_kv, dim]
        v_seq = v_unpad[k_start:k_end]  # [k_len, head_kv, dim]

        # Reshape for GQA
        q_seq = q_seq.view(q_len, num_key_value_heads, groups, head_dim)
        sinks_expanded = sinks.view(num_key_value_heads, groups, 1, 1).float()

        k_seq = k_seq.unsqueeze(2)  # [k_len, head_kv, 1, dim]
        v_seq = v_seq.unsqueeze(2)  # [k_len, head_kv, 1, dim]

        logits = torch.einsum("qhgd,khgd->hgqk", q_seq.float(), k_seq.float()) * sm_scale

        start_q = k_len - q_len
        pos_keys = torch.arange(k_len, device=q_unpad.device)
        pos_queries = torch.arange(q_len, device=q_unpad.device) + start_q

        if is_causal:
            mask = pos_keys[None, :] > pos_queries[:, None]
            mask = mask.float().masked_fill(mask, float("-inf"))
        else:
            mask = torch.zeros(q_len, k_len, device=q_unpad.device)

        if sliding_window is not None:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = logits + mask[None, None, :, :]

        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
        sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
        unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnormalized_scores / normalizer

        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq.float())
        out = out.reshape(q_len, num_heads, head_dim).to(q_unpad.dtype)

        output[q_start:q_end] = out

    return output


def main(
    batch: int = 1,
    heads: int = 64,
    q_seqlen: int = 2048,
    k_seqlen: int = 2048,
    dim: int = 128,
    groups: int = 16,
    is_causal: bool = True,
    window_size: Optional[int] = None,
):
    assert heads % groups == 0, "heads must be divisible by groups"

    flops_per_matmul = 2.0 * batch * heads * q_seqlen * k_seqlen * dim
    total_flops = 5 * flops_per_matmul  # fwd + bwd

    if is_causal:
        total_flops *= 0.5

    if window_size is not None:
        print(f"Using sliding window attention with window_size={window_size}")
        flops_per_matmul = 2.0 * batch * heads * min(window_size, k_seqlen // 2) * q_seqlen * dim
        total_flops = 5 * flops_per_matmul

    dtype = torch.float16
    device = torch.device("cuda")

    head_kv = heads // groups
    q = torch.randn(batch, q_seqlen, heads, dim, dtype=dtype, device=device)
    k = torch.randn(batch, k_seqlen, head_kv, dim, dtype=dtype, device=device)
    v = torch.randn(batch, k_seqlen, head_kv, dim, dtype=dtype, device=device)
    sinks = torch.randn(heads, dtype=dtype, device=device)

    query_padding_mask = generate_random_padding_mask(q_seqlen, batch, device, mode="random")
    key_padding_mask = generate_random_padding_mask(k_seqlen, batch, device, mode="random")

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        _,
        _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    q_unpad = q_unpad.requires_grad_(True)
    k_unpad = k_unpad.requires_grad_(True)
    v_unpad = v_unpad.requires_grad_(True)
    sinks = sinks.requires_grad_(True)

    dO_unpad = torch.randn_like(q_unpad)

    # TileLang forward + backward
    # N_CTX is the padded sequence length used for tensor allocation
    N_CTX = q_seqlen
    O_unpad = attention(
        q_unpad, k_unpad, v_unpad, sinks, cu_seqlens_q, cu_seqlens_k, N_CTX, max_seqlen_q, max_seqlen_k, window_size, groups, is_causal
    )
    O_unpad.backward(dO_unpad, retain_graph=True)
    dQ, q_unpad.grad = q_unpad.grad.clone(), None
    dK, k_unpad.grad = k_unpad.grad.clone(), None
    dV, v_unpad.grad = v_unpad.grad.clone(), None
    dsinks, sinks.grad = sinks.grad.clone(), None

    # Reference forward + backward
    O_ref_unpad = ref_program(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sinks,
        batch,
        is_causal,
        sliding_window=window_size,
        groups=groups,
    )
    O_ref_unpad.backward(dO_unpad, retain_graph=True)
    dQ_ref, q_unpad.grad = q_unpad.grad.clone(), None
    dK_ref, k_unpad.grad = k_unpad.grad.clone(), None
    dV_ref, v_unpad.grad = v_unpad.grad.clone(), None
    dsinks_ref, sinks.grad = sinks.grad.clone(), None

    # Checks
    # Sliding window attention has slightly higher numerical error due to more complex masking
    rtol, atol = (2e-2, 2e-2) if window_size is not None else (1e-2, 1e-2)
    assert torch.allclose(O_unpad, O_ref_unpad, rtol=rtol, atol=atol), f"O max err: {(O_unpad - O_ref_unpad).abs().max()}"
    assert torch.allclose(dV, dV_ref, rtol=rtol, atol=atol), f"dV max err: {(dV - dV_ref).abs().max()}"
    assert torch.allclose(dK, dK_ref, rtol=rtol, atol=atol), f"dK max err: {(dK - dK_ref).abs().max()}"
    assert torch.allclose(dQ, dQ_ref, rtol=rtol, atol=atol), f"dQ max err: {(dQ - dQ_ref).abs().max()}"
    assert torch.allclose(dsinks, dsinks_ref, rtol=rtol, atol=atol), f"dsinks max err: {(dsinks - dsinks_ref).abs().max()}"

    print("All checks passed for tilelang kernels.✅")

    # Benchmark backward
    def torch_bwd():
        O_ref_unpad.backward(dO_unpad, retain_graph=True)

    def tl_bwd():
        O_unpad.backward(dO_unpad, retain_graph=True)

    latency = do_bench(torch_bwd, warmup=500)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(tl_bwd, warmup=500)
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="query heads")
    parser.add_argument("--groups", type=int, default=16, help="groups")
    parser.add_argument("--q_seqlen", type=int, default=2048, help="query sequence length")
    parser.add_argument("--k_seqlen", type=int, default=2048, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dim")
    parser.add_argument("--is_causal", action="store_true", help="causal attention")
    parser.add_argument("--window_size", type=int, default=None, help="sliding window size (default: None for full attention)")
    args = parser.parse_args()
    main(args.batch, args.heads, args.q_seqlen, args.k_seqlen, args.dim, args.groups, args.is_causal, args.window_size)
