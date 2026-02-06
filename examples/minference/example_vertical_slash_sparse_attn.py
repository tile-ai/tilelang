# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import argparse

import torch
import triton
import triton.language as tl

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit(out_idx=[3])
def _tl_vs_sparse_flashattn(batch, heads, seq_len, dim, vertical_size, slash_size):
    block_M = 64
    block_N = 64
    num_stages = 2
    threads = 128
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, heads, seq_len, dim]

    seq_blocks = (seq_len + block_M - 1) // block_M

    count_shape = [batch, heads, seq_blocks]

    offset_shape = count_shape + [slash_size]
    index_shape = count_shape + [vertical_size]

    vertical_size_round, slash_size_round = tilelang.next_power_of_2(vertical_size), tilelang.next_power_of_2(slash_size)

    dtype = T.float16
    accum_dtype = T.float32
    int_dtype = T.int32

    def kernel_func(block_M, block_N, num_stages, threads):
        @T.macro
        def Prefetch(
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            column_index: T.SharedBuffer([vertical_size_round], int_dtype),
            column_count: T.int32,
            k: T.int32,
            bz: T.int32,
            by: T.int32,
        ):
            with T.attr("default", "async_scope", 1):
                for i, j in T.Parallel(block_N, dim):
                    K_shared[i, j] = T.if_then_else(k + i < column_count, K[bz, by, column_index[k + i], j], 0)

            with T.attr("default", "async_scope", 1):
                for i, j in T.Parallel(block_N, dim):
                    V_shared[i, j] = T.if_then_else(k + i < column_count, V[bz, by, column_index[k + i], j], 0)

            T.ptx_commit_group()

        @T.macro
        def Compute(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            scores_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
            k: T.int32,
            column_count: T.int32,
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),
            logsum: T.FragmentBuffer([block_M], accum_dtype),
            count: T.int32,
        ):
            T.ptx_wait_group(count)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(k + j < column_count, 0, -T.infinity(acc_s.dtype))
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            T.copy(scores_max, scores_max_prev)
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] = acc_o[i, j] * scores_scale[i]

            T.copy(acc_s, acc_s_cast)

            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.reduce_sum(acc_s, scores_sum, dim=1)

            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

        @T.prim_func
        def vs_sparse_flashattn_ws(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
            BlockCount: T.Tensor(count_shape, int_dtype),
            BlockOffset: T.Tensor(offset_shape, int_dtype),
            ColumnCount: T.Tensor(count_shape, int_dtype),
            ColumnIndex: T.Tensor(index_shape, int_dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=256) as (bc, by, bz):
                bx = T.ceildiv(seq_len, block_M) - 1 - bc
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([2, block_N, dim], dtype)
                V_shared = T.alloc_shared([2, block_N, dim], dtype)
                K_shared_1 = T.alloc_shared([block_N, dim], dtype)
                V_shared_1 = T.alloc_shared([block_N, dim], dtype)
                K_shared_2 = T.alloc_shared([block_N, dim], dtype)
                V_shared_2 = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                block_count = T.alloc_var(dtype=int_dtype)
                block_offset = T.alloc_shared([slash_size_round], int_dtype, scope="shared")
                column_count = T.alloc_var(dtype=int_dtype)
                column_index = T.alloc_shared([vertical_size_round], int_dtype, scope="shared")

                T.create_list_of_mbarrier([128] * 9)

                block_count = BlockCount[bz, by, bx]
                column_count = ColumnCount[bz, by, bx]

                for vi in T.Parallel(slash_size_round):
                    if vi < slash_size:
                        block_offset[vi] = BlockOffset[bz, by, bx, vi]

                for vi in T.Parallel(vertical_size_round):
                    if vi < vertical_size:
                        column_index[vi] = ColumnIndex[bz, by, bx, vi]

                tid = T.get_thread_binding()

                if tid >= 128:
                    T.annotate_producer_reg_dealloc()
                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                    T.mbarrier_arrive(mbarrier=8)
                    for bi in T.serial(block_count):
                        k = block_offset[bi]
                        T.mbarrier_wait_parity(mbarrier=bi % 2 + 4, parity=(((bi & 3) >> 1) ^ 1))
                        T.copy(K[bz, by, k : k + block_N, :], K_shared[bi % 2, :, :])
                        T.mbarrier_arrive(mbarrier=bi % 2)
                        T.mbarrier_wait_parity(mbarrier=bi % 2 + 6, parity=(((bi & 3) >> 1) ^ 1))
                        T.copy(V[bz, by, k : k + block_N, :], V_shared[bi % 2, :, :])
                        T.mbarrier_arrive(mbarrier=bi % 2 + 2)
                else:
                    T.annotate_consumer_reg_alloc()
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.mbarrier_wait_parity(mbarrier=8, parity=0)
                    for bi in T.serial(block_count):
                        k = block_offset[bi]
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(bx * block_M + i >= k + j, 0, -T.infinity(acc_s.dtype))

                        T.mbarrier_wait_parity(mbarrier=bi % 2, parity=((bi & 3) >> 1))
                        T.gemm(Q_shared, K_shared[bi % 2, :, :], acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.mbarrier_arrive(mbarrier=bi % 2 + 4)

                        T.copy(scores_max, scores_max_prev)

                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] = acc_o[i, j] * scores_scale[i]

                        T.copy(acc_s, acc_s_cast)
                        T.mbarrier_wait_parity(mbarrier=bi % 2 + 2, parity=((bi & 3) >> 1))
                        T.gemm(acc_s_cast, V_shared[bi % 2, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)

                        T.mbarrier_arrive(mbarrier=bi % 2 + 6)

                        T.reduce_sum(acc_s, scores_sum, dim=1)

                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    if column_count != 0:
                        Prefetch(K, V, K_shared_1, V_shared_1, column_index, column_count, 0, bz, by)
                        for bi in T.serial(T.ceildiv(column_count, block_N) - 1):
                            k = bi * block_N
                            if bi % 2 == 0:
                                Prefetch(K, V, K_shared_2, V_shared_2, column_index, column_count, k + block_N, bz, by)

                                Compute(
                                    acc_s,
                                    acc_s_cast,
                                    acc_o,
                                    scores_max,
                                    scores_max_prev,
                                    k,
                                    column_count,
                                    Q_shared,
                                    K_shared_1,
                                    V_shared_1,
                                    scores_scale,
                                    scores_sum,
                                    logsum,
                                    1,
                                )
                            else:
                                Prefetch(K, V, K_shared_1, V_shared_1, column_index, column_count, k + block_N, bz, by)

                                Compute(
                                    acc_s,
                                    acc_s_cast,
                                    acc_o,
                                    scores_max,
                                    scores_max_prev,
                                    k,
                                    column_count,
                                    Q_shared,
                                    K_shared_2,
                                    V_shared_2,
                                    scores_scale,
                                    scores_sum,
                                    logsum,
                                    1,
                                )
                        if T.ceildiv(column_count, block_N) % 2 == 0:
                            Compute(
                                acc_s,
                                acc_s_cast,
                                acc_o,
                                scores_max,
                                scores_max_prev,
                                T.ceildiv(column_count, block_N) * block_N - block_N,
                                column_count,
                                Q_shared,
                                K_shared_2,
                                V_shared_2,
                                scores_scale,
                                scores_sum,
                                logsum,
                                0,
                            )
                        else:
                            Compute(
                                acc_s,
                                acc_s_cast,
                                acc_o,
                                scores_max,
                                scores_max_prev,
                                T.ceildiv(column_count, block_N) * block_N - block_N,
                                column_count,
                                Q_shared,
                                K_shared_1,
                                V_shared_1,
                                scores_scale,
                                scores_sum,
                                logsum,
                                0,
                            )
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] /= logsum[i]
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

        return vs_sparse_flashattn_ws

    return kernel_func(block_M, block_N, num_stages, threads)


@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q,
    K,
    V,
    seqlens,
    sm_scale,
    block_count,
    block_offset,
    column_count,
    column_index,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    NUM_ROWS,
    NNZ_S,
    NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)  # bx
    off_hz = tl.program_id(1)  # by

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    for start_n in range(0, num_cols, BLOCK_N):  #
        # bi * BLOCK_N: bi * BLOCK_N + BLOCK_N
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_mixed_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q,
        k,
        v,
        seqlens,
        sm_scale,
        block_count,
        block_offset,
        column_count,
        column_index,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        block_count.shape[-1],
        block_offset.shape[-1],
        column_index.shape[-1],
        BLOCK_M=block_size_M,
        BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4,
        num_stages=2,
    )

    return o


def vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    from torch.utils.cpp_extension import load
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.join(current_dir, "ops", "kernels.cpp"), os.path.join(current_dir, "ops", "vertical_slash_index.cu")]
    ops = load(name="convert", sources=sources, verbose=False)
    convert_vertical_slash_indexes = ops.convert_vertical_slash_indexes
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = (block_size_M - context_size) & (block_size_M - 1)
    if pad == block_size_M:
        pad = 0
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]

    seqlens = torch.tensor([context_size] * query.shape[0], dtype=torch.int32, device=query.device)
    sm_scale = head_dim**-0.5
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens,
        v_idx,
        s_idx,
        context_size,
        block_size_M,
        block_size_N,
    )

    tl_kernel = _tl_vs_sparse_flashattn(batch_size, num_heads, context_size, head_dim, v_idx.shape[2], s_idx.shape[2])

    def run(is_triton: bool = True):
        if is_triton:
            out = _triton_mixed_sparse_attention(
                query,
                key,
                value,
                seqlens,
                block_count,
                block_offset,
                column_count,
                column_index,
                sm_scale,
                block_size_M,
                block_size_N,
            )
        else:
            out = tl_kernel(query, key, value, block_count, block_offset, column_count, column_index)
        return out[..., :context_size, :head_dim]

    return run


def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device)  # Zero matrix used for padding
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)  # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1))  # Change the strides
    sum_diags = torch.sum(mat_strided, 2)  # Sums the resulting matrix's columns
    return sum_diags[:, :, 1:]


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--vertical_size", type=int, default=1000)
    parser.add_argument("--slash_size", type=int, default=200)

    args = parser.parse_args(argv)

    BATCH, N_HEADS, SEQ_LEN, D_HEAD = args.batch, args.heads, args.seq_len, args.head_dim

    vertical_size, slash_size = args.vertical_size, args.slash_size

    torch.manual_seed(0)
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)

    q_len = SEQ_LEN

    vertical_size, slash_size = min(q_len, vertical_size), min(q_len, slash_size)
    last_q = 64
    qk = torch.einsum("bhmk, bhnk -> bhmn", q[:, :, -last_q:, :], k)
    arange = torch.arange(last_q, device="cuda")
    qk[:, :, :, -last_q:] = torch.where(arange[None, None, :, None] >= arange[None, None, None, :], qk[:, :, :, -last_q:], -torch.inf)
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices

    slash = sum_all_diagonal_matrix(qk)[..., : -last_q + 1]
    slash[..., -30:] = torch.inf

    slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

    _attn = vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    tilelang_out = _attn(False)
    triton_out = _attn(True)

    torch.testing.assert_close(triton_out, tilelang_out, atol=1e-2, rtol=1e-2)

    triton_time = do_bench(lambda: _attn(True))
    tilelang_time = do_bench(lambda: _attn(False))

    print(f"triton_time: {triton_time:.3f}ms")
    print(f"tilelang_time: {tilelang_time:.3f}ms")
    print(f"speedup: {triton_time / tilelang_time:.2f}x")


def run_regression_perf(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--vertical_size", type=int, default=1000)
    parser.add_argument("--slash_size", type=int, default=200)
    args = parser.parse_args(argv)
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = args.batch, args.heads, args.seq_len, args.head_dim
    vertical_size, slash_size = args.vertical_size, args.slash_size
    torch.manual_seed(0)
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    q_len = SEQ_LEN
    vertical_size, slash_size = min(q_len, vertical_size), min(q_len, slash_size)
    last_q = 64
    qk = torch.einsum("bhmk, bhnk -> bhmn", q[:, :, -last_q:, :], k)
    arange = torch.arange(last_q, device="cuda")
    qk[:, :, :, -last_q:] = torch.where(arange[None, None, :, None] >= arange[None, None, None, :], qk[:, :, :, -last_q:], -torch.inf)
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices
    slash = sum_all_diagonal_matrix(qk)[..., : -last_q + 1]
    slash[..., -30:] = torch.inf
    slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices
    block_size_M = 64
    block_size_N = 64
    query, key, value = q, k, v
    v_idx, s_idx = vertical_topk, slash
    batch_size, num_heads, context_size, head_dim = query.shape
    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    from torch.utils.cpp_extension import load
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.join(current_dir, "ops", "kernels.cpp"), os.path.join(current_dir, "ops", "vertical_slash_index.cu")]
    ops = load(name="convert", sources=sources, verbose=False)
    convert_vertical_slash_indexes = ops.convert_vertical_slash_indexes
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = (block_size_M - context_size) & (block_size_M - 1)
    if pad == block_size_M:
        pad = 0
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])
    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    seqlens = torch.tensor([context_size] * query.shape[0], dtype=torch.int32, device=query.device)
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens,
        v_idx,
        s_idx,
        context_size,
        block_size_M,
        block_size_N,
    )
    tl_kernel = _tl_vs_sparse_flashattn(batch_size, num_heads, context_size, head_dim, vertical_topk.shape[-1], slash.shape[-1])

    def run_kernel_only():
        tl_kernel(query, key, value, block_count, block_offset, column_count, column_index)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    main()
