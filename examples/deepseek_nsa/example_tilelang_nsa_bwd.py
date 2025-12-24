# ruff: noqa
import torch
from typing import Optional, Union
from packaging.version import parse

import torch
import triton

import fla

if parse(fla.__version__) < parse("0.2.1"):
    from fla.ops.common.utils import prepare_token_indices
else:
    from fla.ops.utils import prepare_token_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from reference import naive_nsa
from einops import rearrange
import tilelang


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def tilelang_kernel_fwd(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    scale=None,
    block_size=64,
    groups=1,
    selected_blocks=16,
):
    from tilelang import language as T

    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    o_slc_shape = [batch, seq_len, heads, dim]
    lse_slc_shape = [batch, seq_len, heads]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 0
    threads = 32

    @T.prim_func
    def native_sparse_attention(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        O_slc: T.Tensor(o_slc_shape, dtype),
        LSE_slc: T.Tensor(lse_slc_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, i_t, i_h, i] * BS
                if i_s <= i_t and i_s >= 0:
                    # [BS, BK]
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    if is_causal:
                        for k, j in T.Parallel(G, BS):
                            acc_s[k, j] = T.if_then_else(i_t >= (i_s + j), 0, -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for k in T.Parallel(G):
                        scores_scale[k] = T.exp2(scores_max_prev[k] * scale - scores_max[k] * scale)
                    for k, j in T.Parallel(G, BS):
                        acc_s[k, j] = T.exp2(acc_s[k, j] * scale - scores_max[k] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for k in T.Parallel(G):
                        logsum[k] = logsum[k] * scores_scale[k] + scores_sum[k]
                    T.copy(acc_s, acc_s_cast)

                    # Rescale
                    for k, j in T.Parallel(G, BV):
                        acc_o[k, j] *= scores_scale[k]

                    # V * softmax(Q * K)
                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                O_slc[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV],
            )
            for i in T.Parallel(G):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, LSE_slc[i_b, i_t, i_h * G : (i_h + 1) * G])

    return native_sparse_attention


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def tilelang_kernel_bwd_dkv(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    scale=None,
    block_size=64,
    groups=1,
    selected_blocks=16,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    if scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    else:
        sm_scale = scale

    scale = sm_scale * 1.44269504

    from tilelang import language as T

    B = batch
    BS = block_size
    G = groups
    V = dim
    K = dim
    BK = tilelang.next_power_of_2(K)
    BV = min(128, tilelang.next_power_of_2(dim))
    NS = tilelang.cdiv(seq_len, BS)
    NV = tilelang.cdiv(V, BV)

    heads_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, heads_kv, dim]
    v_shape = [batch, seq_len, heads_kv, dim]
    lse_slc_shape = [batch, seq_len, heads]
    delta_slc_shape = [batch, seq_len, heads]
    o_shape = [batch, heads, seq_len, dim]
    do_slc_shape = [batch, seq_len, heads, dim]
    dk_shape = [NV, batch, seq_len, heads_kv, dim]
    dv_shape = [batch, seq_len, heads_kv, dim]

    block_mask_shape = [batch, seq_len, heads_kv, NS]
    num_threads = 32
    print("NV", NV, "NS", NS, "B", B, "H", H)

    @T.prim_func
    def flash_bwd_dkv(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        LSE_slc: T.Tensor(lse_slc_shape, accum_dtype),
        Delta_slc: T.Tensor(delta_slc_shape, accum_dtype),
        DO_slc: T.Tensor(do_slc_shape, dtype),
        DK: T.Tensor(dk_shape, dtype),
        DV: T.Tensor(dv_shape, dtype),
        BlockMask: T.Tensor(block_mask_shape, T.int32),
    ):
        with T.Kernel(NV, NS, B * H, threads=num_threads) as (i_v, i_s, i_bh):
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            Q_shared = T.alloc_shared([G, BK], dtype)
            qkT = T.alloc_fragment([BS, G], accum_dtype)
            qkT_cast = T.alloc_fragment([BS, G], dtype)
            dsT = T.alloc_fragment([BS, G], accum_dtype)
            dsT_cast = T.alloc_fragment([BS, G], dtype)
            lse_shared = T.alloc_shared([G], accum_dtype)
            delta = T.alloc_shared([G], accum_dtype)

            do = T.alloc_shared([G, BV], dtype)
            dv = T.alloc_fragment([BS, BV], accum_dtype)
            dk = T.alloc_fragment([BS, BK], accum_dtype)
            dq = T.alloc_fragment([BS, G], accum_dtype)

            dv_shared = T.alloc_shared([BS, BV], dtype)
            dk_shared = T.alloc_shared([BS, BK], dtype)

            i_b, i_h = i_bh // H, i_bh % H

            T.copy(K[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BK], K_shared)
            T.copy(V[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BV], V_shared)

            # [BS, BK]
            T.clear(dk)
            # [BS, BV]
            T.clear(dv)

            loop_st = i_s * BS
            loop_ed = seq_len
            for i in T.Pipelined(
                start=loop_st,
                stop=loop_ed,
                num_stages=0,
            ):
                b_m_slc = BlockMask[i_b, i, i_h, i_s]
                if b_m_slc != 0:
                    # [G, BK]
                    T.copy(Q[i_b, i, i_h * G : (i_h + 1) * G, :BK], Q_shared)
                    T.clear(qkT)
                    # [BS, BK] @ [G, BK] -> [BS, G]
                    T.gemm(
                        K_shared,
                        Q_shared,
                        qkT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    # [G]
                    T.copy(LSE_slc[i_b, i, i_h * G : (i_h + 1) * G], lse_shared)

                    for _i, _j in T.Parallel(BS, G):
                        qkT[_i, _j] = T.exp2(qkT[_i, _j] * scale - lse_shared[_j])

                    for _i, _j in T.Parallel(BS, G):
                        qkT[_i, _j] = T.if_then_else(i >= (i_s * BS + _i), qkT[_i, _j], 0)

                    # [G, BV]
                    T.copy(DO_slc[i_b, i, i_h * G : (i_h + 1) * G, :BV], do)
                    T.clear(dsT)
                    # [BS, BV] @ [G, BV] -> [BS, G]
                    T.gemm(
                        V_shared,
                        do,
                        dsT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(qkT, qkT_cast)
                    # [BS, G] @ [G, BV] -> [BS, BV]
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)
                    # [G]
                    T.copy(Delta_slc[i_b, i, i_h * G : (i_h + 1) * G], delta)
                    for i, j in T.Parallel(BS, G):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale

                    # [BS, G] @ [G, BK] -> [BS, BK]
                    T.gemm(dsT_cast, Q_shared, dk, policy=T.GemmWarpPolicy.FullRow)

            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, DV[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BV])
            T.copy(dk_shared, DK[i_v, i_b, i_s * BS : (i_s + 1) * BS, i_h, :BK])

    return flash_bwd_dkv


def make_dq_layout(dQ):
    from tilelang import language as T

    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dQ.shape,
        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def tilelang_kernel_bwd_dqkv(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    scale=None,
    block_size=64,
    groups=1,
    selected_blocks=16,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    if scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    else:
        sm_scale = scale

    scale = sm_scale * 1.44269504

    from tilelang import language as T

    B = batch
    BS = block_size
    G = groups
    V = dim
    K = dim
    BK = tilelang.next_power_of_2(K)
    BV = min(128, tilelang.next_power_of_2(dim))
    NS = tilelang.cdiv(seq_len, BS)
    NV = tilelang.cdiv(V, BV)

    heads_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, heads_kv, dim]
    v_shape = [batch, seq_len, heads_kv, dim]
    lse_slc_shape = [batch, seq_len, heads]
    delta_slc_shape = [batch, seq_len, heads]
    o_shape = [batch, heads, seq_len, dim]
    do_slc_shape = [batch, seq_len, heads, dim]
    dq_shape = [NV, batch, seq_len, heads, dim]
    dk_shape = [NV, batch, seq_len, heads_kv, dim]
    dv_shape = [batch, seq_len, heads_kv, dim]

    block_mask_shape = [batch, seq_len, heads_kv, NS]
    num_threads = 32

    @T.prim_func
    def flash_bwd_dqkv(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        LSE_slc: T.Tensor(lse_slc_shape, accum_dtype),
        Delta_slc: T.Tensor(delta_slc_shape, accum_dtype),
        DO_slc: T.Tensor(do_slc_shape, dtype),
        DQ: T.Tensor(dq_shape, dtype),
        DK: T.Tensor(dk_shape, dtype),
        DV: T.Tensor(dv_shape, dtype),
        BlockMask: T.Tensor(block_mask_shape, T.int32),
    ):
        with T.Kernel(NV, NS, B * H, threads=num_threads) as (i_v, i_s, i_bh):
            K_shared = T.alloc_shared([BS, BK], dtype)
            dsT_shared = T.alloc_shared([BS, G], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            Q_shared = T.alloc_shared([G, BK], dtype)
            qkT = T.alloc_fragment([BS, G], accum_dtype)
            qkT_cast = T.alloc_fragment([BS, G], dtype)
            dsT = T.alloc_fragment([BS, G], accum_dtype)
            dsT_cast = T.alloc_fragment([BS, G], dtype)
            lse_shared = T.alloc_shared([G], accum_dtype)
            delta = T.alloc_shared([G], accum_dtype)

            do = T.alloc_shared([G, BV], dtype)
            dv = T.alloc_fragment([BS, BV], accum_dtype)
            dk = T.alloc_fragment([BS, BK], accum_dtype)
            dq = T.alloc_fragment([G, BK], accum_dtype)

            dv_shared = T.alloc_shared([BS, BV], dtype)
            dk_shared = T.alloc_shared([BS, BK], dtype)

            i_b, i_h = i_bh // H, i_bh % H

            T.copy(K[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BK], K_shared)
            T.copy(V[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BV], V_shared)

            # [BS, BK]
            T.clear(dk)
            # [BS, BV]
            T.clear(dv)

            loop_st = i_s * BS
            loop_ed = seq_len
            for i in T.Pipelined(
                start=loop_st,
                stop=loop_ed,
                num_stages=0,
            ):
                b_m_slc = BlockMask[i_b, i, i_h, i_s]
                if b_m_slc != 0:
                    # [G, BK]
                    T.copy(Q[i_b, i, i_h * G : (i_h + 1) * G, :BK], Q_shared)
                    T.clear(qkT)
                    # [BS, BK] @ [G, BK] -> [BS, G]
                    T.gemm(
                        K_shared,
                        Q_shared,
                        qkT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    # [G]
                    T.copy(LSE_slc[i_b, i, i_h * G : (i_h + 1) * G], lse_shared)

                    for _i, _j in T.Parallel(BS, G):
                        qkT[_i, _j] = T.exp2(qkT[_i, _j] * scale - lse_shared[_j])

                    for _i, _j in T.Parallel(BS, G):
                        qkT[_i, _j] = T.if_then_else(i >= (i_s * BS + _i), qkT[_i, _j], 0)

                    # [G, BV]
                    T.copy(DO_slc[i_b, i, i_h * G : (i_h + 1) * G, :BV], do)
                    T.clear(dsT)
                    # [BS, BV] @ [G, BV] -> [BS, G]
                    T.gemm(
                        V_shared,
                        do,
                        dsT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(qkT, qkT_cast)
                    # [BS, G] @ [G, BV] -> [BS, BV]
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)
                    # [G]
                    T.copy(Delta_slc[i_b, i, i_h * G : (i_h + 1) * G], delta)
                    for _i, _j in T.Parallel(BS, G):
                        dsT_cast[_i, _j] = qkT[_i, _j] * (dsT[_i, _j] - delta[_j]) * sm_scale

                    # [BS, G] @ [G, BK] -> [BS, BK]
                    T.gemm(dsT_cast, Q_shared, dk, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    # [BS, G] * [BS, BK] -> [G, BK]
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                    for _i, _j in T.Parallel(G, BK):
                        T.atomic_add(DQ[i_v, i_b, i, i_h * G + _i, _j], dq[_i, _j])

            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, DV[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BV])
            T.copy(dk_shared, DK[i_v, i_b, i_s * BS : (i_s + 1) * BS, i_h, :BK])

    return flash_bwd_dqkv


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kernel_preprocess(
    batch,
    heads,
    seq_len,
    dim,
    dtype=T.float16,
    accum_dtype=T.float32,
    blk=32,
):
    from tilelang import language as T

    shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([batch, seq_len, heads], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * blk : (by + 1) * blk, bx])

    return flash_bwd_prep


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kernel_block_mask(
    batch,
    heads,
    seq_len,
    selected_blocks,
    block_size,
    dtype=T.int32,
):
    from tilelang import language as T

    block_indices_shape = [batch, seq_len, heads, selected_blocks]
    block_counts_shape = [batch, seq_len, heads]
    S = selected_blocks
    BS = block_size
    NS = tilelang.cdiv(seq_len, BS)

    block_mask_shape = [batch, seq_len, heads, NS]
    USE_BLOCK_COUNTS = block_counts is not None

    @T.prim_func
    def flash_bwd_block_mask(
        BlockIndices: T.Tensor(block_indices_shape, dtype),  # type: ignore
        BlockCounts: T.Tensor(block_counts_shape, dtype),  # type: ignore
        BlockMask: T.Tensor(block_mask_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, batch, heads * S) as (bx, by, bz):
            i_t, i_b, i_hs = bx, by, bz
            i_h, i_s = i_hs // S, i_hs % S
            b_i = BlockIndices[i_b, i_t, i_h, i_s]
            if USE_BLOCK_COUNTS:
                b_m = b_i * BS <= i_t and i_s < BlockCounts[i_b, i_t, i_h].astype(i_s.dtype)
                BlockMask[i_b, i_t, i_h, i_s] = b_m
            else:
                b_m = b_i * BS <= i_t
                BlockMask[i_b, i_t, i_h, i_s] = b_m

    return flash_bwd_block_mask


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_slc: torch.Tensor,
    lse_slc: torch.Tensor,
    do_slc: torch.Tensor,
    o_swa: torch.Tensor,
    lse_swa: torch.Tensor,
    do_swa: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    window_size: int = 0,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    WS = window_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)

    assert window_size == 0, "Window size is not supported yet"
    delta_slc = tilelang_kernel_preprocess(B, HQ, T, K)(o_slc, do_slc)

    dq = torch.zeros(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    block_mask = tilelang_kernel_block_mask(B, H, T, S, BS)(block_indices.to(torch.int32), block_counts.to(torch.int32)).to(torch.bool)

    fused_qkv_bwd_kernel = tilelang_kernel_bwd_dqkv(
        batch=B,
        heads=HQ,
        seq_len=T,
        dim=K,
        is_causal=True,
        block_size=BS,
        groups=G,
        selected_blocks=S,
        scale=scale,
    )
    fused_qkv_bwd_kernel(q, k, v, lse_slc, delta_slc, do_slc, dq, dk, dv, block_mask.to(torch.int32))

    dq = dq.sum(0)
    dk = dk.sum(0)
    return dq, dk, dv


@torch.compile
class ParallelNSAFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        block_indices,
        block_counts,
        block_size,
        window_size,
        scale,
        offsets,
    ):
        ctx.dtype = q.dtype
        assert offsets is None, "Offsets are not supported yet"
        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        B, SEQLEN, HQ, D = q.shape
        H = k.shape[2]
        G = HQ // H
        S = block_indices.shape[-1]
        V = v.shape[-1]
        kernel = tilelang_kernel_fwd(
            batch=B,
            heads=HQ,
            seq_len=SEQLEN,
            dim=D,
            is_causal=True,
            scale=scale,
            block_size=block_size,
            groups=G,
            selected_blocks=S,
        )
        o_slc = torch.empty(B, SEQLEN, HQ, D, dtype=v.dtype, device=q.device)
        lse_slc = torch.empty(B, SEQLEN, HQ, dtype=torch.float, device=q.device)
        kernel(q, k, v, block_indices.to(torch.int32), o_slc, lse_slc)

        ctx.save_for_backward(q, k, v, o_slc, lse_slc)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.window_size = window_size
        ctx.scale = scale
        return o_slc.to(q.dtype), lse_slc.to(torch.float)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do_slc, do_swa):
        q, k, v, o_slc, lse_slc = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o_slc=o_slc,
            o_swa=None,
            lse_slc=lse_slc,
            lse_swa=None,
            do_slc=do_slc,
            do_swa=do_swa,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            window_size=ctx.window_size,
            scale=ctx.scale,
            offsets=ctx.offsets,
            token_indices=ctx.token_indices,
        )
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Optional[Union[torch.LongTensor, int]] = None,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, SEQLEN, HQ, K]` if `head_first=False` else `[B, HQ, SEQLEN, K]`.
        k (torch.Tensor):
            keys of shape `[B, SEQLEN, H, K]` if `head_first=False` else `[B, H, SEQLEN, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, SEQLEN, H, V]` if `head_first=False` else `[B, H, SEQLEN, V]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, SEQLEN, HQ]` if  `head_first=False` else `[B, HQ, SEQLEN]`.
        g_swa (torch.Tensor):
            Gate score for sliding attention of shape `[B, SEQLEN, HQ]` if  `head_first=False` else `[B, HQ, SEQLEN]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, SEQLEN, H, S]` if `head_first=False` else `[B, H, SEQLEN, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, SEQLEN, H]` if `head_first=True` else `[B, SEQLEN, H]`,
            each token can select the same number of blocks.
            If not provided, it will default to `S`, Default: `None`
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, SEQLEN, HQ, V]` if `head_first=False` else `[B, HQ, SEQLEN, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, "b h t d -> b t h d"), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, "b h t -> b t h"), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, "b h t -> b t h")
    assert q.shape[2] % (k.shape[2] * 16) == 0, "Group size must be a multiple of 16 in NSA"

    if isinstance(block_counts, int):
        block_indices = block_indices[:, :, :, :block_counts]
        block_counts = None

    o_slc, o_swa = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size, window_size, scale, cu_seqlens)
    if window_size > 0:
        o = torch.addcmul(o_slc * g_slc.unsqueeze(-1), o_swa, g_swa.unsqueeze(-1))
    else:
        o = o_slc * g_slc.unsqueeze(-1)
    if head_first:
        o = rearrange(o, "b t h d -> b h t d")
    return o


if __name__ == "__main__":
    B, T, H, HQ, D, S, block_size, dtype = 1, 32, 1, 16, 32, 1, 32, torch.float16
    torch.random.manual_seed(0)
    q = torch.randn((B, T, HQ, D), dtype=dtype, device="cuda").requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    g_slc = torch.ones((B, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    g_swa = torch.ones((B, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device="cuda")

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device="cuda")
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    block_counts = torch.randint(1, S + 1, (B, T, H), device="cuda")

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg_slc, g_slc.grad = g_slc.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_size=block_size,
        block_counts=block_counts,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg_slc, g_slc.grad = g_slc.grad.clone(), None

    # assert_close(" o", ref, tri, 0.004)
    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dg_slc, tri_dg_slc, atol=1e-2, rtol=1e-2)
