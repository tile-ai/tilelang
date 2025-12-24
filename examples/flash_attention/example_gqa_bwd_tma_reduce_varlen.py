import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.contrib import nvcc
import argparse
from einops import rearrange, repeat
from bert_padding import pad_input, unpad_input


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    return padding_mask


@tilelang.jit(
    out_idx=[5, 6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_fwd(batch, total_q, total_kv, N_CTX, heads, max_seq_len, dim_qk, dim_v, is_causal, block_M, block_N, groups=1):
    scale = (1.0 / dim_qk) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [total_q, heads, dim_qk]
    k_shape = [total_kv, head_kv, dim_qk]
    v_shape = [total_kv, head_kv, dim_v]
    o_shape = [total_q, heads, dim_v]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], T.int32),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, N_CTX], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(max_seq_len, block_M), heads, batch, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            q_start_idx = cu_seqlens_q[bz]
            k_start_idx = cu_seqlens_k[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            k_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            k_current_seqlen = k_end_idx - k_start_idx

            for i, d in T.Parallel(block_M, dim_qk):
                if bx * block_M + i < q_current_seqlen:
                    Q_shared[i, d] = Q[q_start_idx + bx * block_M + i, by, d]
                else:
                    Q_shared[i, d] = 0.0

            T.fill(acc_o, 0.0)
            T.fill(logsum, 0.0)
            # Warning: in causal/varlen/unaligned seqlen scenarios, the -inf will cause undefined behavior in exp ops
            # We should set it to negative large number instead
            T.fill(scores_max, T.Cast(accum_dtype, -1e30))
            loop_range = T.ceildiv(k_current_seqlen, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                for i, d in T.Parallel(block_N, dim_qk):
                    if k * block_N + i < k_current_seqlen:
                        K_shared[i, d] = K[k_start_idx + k * block_N + i, by // groups, d]
                    else:
                        K_shared[i, d] = 0.0

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i >= k * block_N + j)
                            and (bx * block_M + i < q_current_seqlen and k * block_N + j < k_current_seqlen),
                            0,
                            T.Cast(accum_dtype, -1e30),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i < q_current_seqlen and k * block_N + j < k_current_seqlen, 0, T.Cast(accum_dtype, -1e30)
                        )
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, d in T.Parallel(block_N, dim_v):
                    if k * block_N + i < k_current_seqlen:
                        V_shared[i, d] = V[k_start_idx + k * block_N + i, by // groups, d]
                    else:
                        V_shared[i, d] = 0.0
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] /= logsum[i]

            for i, d in T.Parallel(block_M, dim_v):
                if bx * block_M + i < q_current_seqlen:
                    Output[q_start_idx + bx * block_M + i, by, d] = acc_o[i, d]

            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                if bx * block_M + i < q_current_seqlen:
                    lse[bz, by, bx * block_M + i] = logsum[i]

    return flash_fwd


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_preprocess(batch, heads, total_q, N_CTX, max_seq_len, dim_v):
    dtype = T.float16
    accum_dtype = T.float32
    shape = [total_q, heads, dim_v]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        Delta: T.Tensor([batch, heads, N_CTX], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(max_seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx

            T.clear(acc)
            for k in range(T.ceildiv(dim_v, blk)):
                for i, j in T.Parallel(blk, blk):
                    if by * blk + i < q_current_seqlen and k * blk + j < dim_v:
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
    # bshd -> bhsd to use tma reduction instruction
    return T.Layout(dQ.shape, lambda l, h, d: [h, l, d])


@tilelang.jit(
    out_idx=[3, 4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_postprocess(total_q, total_kv, heads, head_kv, dim_qk, dim_v):
    dtype = T.float16
    accum_dtype = T.float32
    q_shape = [total_q, heads, dim_qk]
    k_shape = [total_kv, head_kv, dim_qk]
    v_shape = [total_kv, head_kv, dim_v]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
        dK: T.Tensor(k_shape, accum_dtype),  # type: ignore
        dV: T.Tensor(v_shape, accum_dtype),  # type: ignore
        dQ_out: T.Tensor(q_shape, dtype),  # type: ignore
        dK_out: T.Tensor(k_shape, dtype),  # type: ignore
        dV_out: T.Tensor(v_shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(total_q, blk), heads, threads=128) as (bx, by):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(dQ[bx * blk : (bx + 1) * blk, by, :], dQ_out[bx * blk : (bx + 1) * blk, by, :])
        with T.Kernel(T.ceildiv(total_kv, blk), head_kv, threads=128) as (bx, by):
            T.annotate_layout(
                {
                    dK: make_dq_layout(dK),
                    dV: make_dq_layout(dV),
                }
            )
            T.copy(dK[bx * blk : (bx + 1) * blk, by, :], dK_out[bx * blk : (bx + 1) * blk, by, :])
            T.copy(dV[bx * blk : (bx + 1) * blk, by, :], dV_out[bx * blk : (bx + 1) * blk, by, :])

    return flash_bwd_post


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def flashattn_bwd_atomic_add(
    batch, total_q, total_kv, N_CTX, heads, max_seq_len, dim_qk, dim_v, is_causal, block_M, block_N, threads=256, num_stages=2, groups=1
):
    sm_scale = (1.0 / dim_qk) ** 0.5
    scale = (1.0 / dim_qk) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [total_q, heads, dim_qk]
    k_shape = [total_kv, head_kv, dim_qk]
    v_shape = [total_kv, head_kv, dim_v]
    do_shape = [total_q, heads, dim_v]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, N_CTX], accum_dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, N_CTX], accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], T.int32),  # type: ignore
        dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
        dK: T.Tensor(k_shape, accum_dtype),  # type: ignore
        dV: T.Tensor(v_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(max_seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim_v], dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dq = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim_v], accum_dtype)
            dk_shared = T.alloc_shared([block_M, dim_qk], accum_dtype)
            dq_shared = T.alloc_shared([block_N, dim_qk], accum_dtype)

            q_start_idx = cu_seqlens_q[bz]
            k_start_idx = cu_seqlens_k[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            k_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            k_current_seqlen = k_end_idx - k_start_idx

            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                    dK: make_dq_layout(dK),
                    dV: make_dq_layout(dV),
                }
            )

            T.copy(K[k_start_idx + by * block_M : k_start_idx + (by + 1) * block_M, bx // groups, :], K_shared)
            T.copy(V[k_start_idx + by * block_M : k_start_idx + (by + 1) * block_M, bx // groups, :], V_shared)

            T.clear(dv)
            T.clear(dk)

            loop_st = T.min(T.floordiv(by * block_M, block_N), T.floordiv(q_current_seqlen, block_N)) if is_causal else 0
            loop_ed = T.ceildiv(q_current_seqlen, block_N)

            for k_base in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[q_start_idx + k_base * block_N : q_start_idx + (k_base + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(lse[bz, bx, k_base * block_N : (k_base + 1) * block_N], lse_shared)

                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            (by * block_M + i <= k_base * block_N + j)
                            and (by * block_M + i < k_current_seqlen and k_base * block_N + j < q_current_seqlen),
                            qkT[i, j],
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            by * block_M + i < k_current_seqlen and k_base * block_N + j < q_current_seqlen, qkT[i, j], 0
                        )

                T.copy(dO[q_start_idx + k_base * block_N : q_start_idx + (k_base + 1) * block_N, bx, :], do)
                T.clear(dsT)
                # dsT: (block_kv, block_q)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k_base * block_N : (k_base + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                T.copy(dq, dq_shared)
                T.atomic_add(
                    dQ[q_start_idx + k_base * block_N : q_start_idx + k_base * block_N + block_N, bx, :],
                    dq_shared,
                    memory_order="relaxed",
                    use_tma=True,
                )

            T.copy(dv, dv_shared)
            T.atomic_add(
                dV[k_start_idx + by * block_M : k_start_idx + by * block_M + block_M, bx // groups, :],
                dv_shared,
                memory_order="relaxed",
                use_tma=True,
            )
            T.copy(dk, dk_shared)
            T.atomic_add(
                dK[k_start_idx + by * block_M : k_start_idx + by * block_M + block_M, bx // groups, :],
                dk_shared,
                memory_order="relaxed",
                use_tma=True,
            )

    return flash_bwd


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def flashattn_bwd_split(
    batch, total_q, total_kv, N_CTX, heads, max_seq_len, dim_qk, dim_v, is_causal, block_M, block_N, threads=256, num_stages=2, groups=1
):
    sm_scale = (1.0 / dim_qk) ** 0.5
    scale = (1.0 / dim_qk) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [total_q, heads, dim_qk]
    k_shape = [total_kv, head_kv, dim_qk]
    v_shape = [total_kv, head_kv, dim_v]
    do_shape = [total_q, heads, dim_v]
    dk_shape = [groups, total_kv, head_kv, dim_qk]  # sum after kernel
    dv_shape = [groups, total_kv, head_kv, dim_v]  # sum after kernel
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, N_CTX], accum_dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, N_CTX], accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], T.int32),  # type: ignore
        dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
        dK: T.Tensor(dk_shape, dtype),  # type: ignore
        dV: T.Tensor(dv_shape, dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(max_seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim_v], dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dq = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim_v], dtype)
            dk_shared = T.alloc_shared([block_M, dim_qk], dtype)

            q_start_idx = cu_seqlens_q[bz]
            k_start_idx = cu_seqlens_k[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            k_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            k_current_seqlen = k_end_idx - k_start_idx

            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                }
            )

            T.copy(K[k_start_idx + by * block_M : k_start_idx + (by + 1) * block_M, bx // groups, :], K_shared)
            T.copy(V[k_start_idx + by * block_M : k_start_idx + (by + 1) * block_M, bx // groups, :], V_shared)

            T.clear(dv)
            T.clear(dk)
            loop_st = T.min(T.floordiv(by * block_M, block_N), T.floordiv(q_current_seqlen, block_N)) if is_causal else 0
            loop_ed = T.ceildiv(q_current_seqlen, block_N)

            for k_base in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                # Note: The padding zero of varlen should be considered in T.copy
                T.copy(Q[q_start_idx + k_base * block_N : q_start_idx + (k_base + 1) * block_N, bx, :], q)

                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dO[q_start_idx + k_base * block_N : q_start_idx + (k_base + 1) * block_N, bx, :], do)

                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(lse[bz, bx, k_base * block_N : (k_base + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            (by * block_M + i <= k_base * block_N + j)
                            and (by * block_M + i < k_current_seqlen and k_base * block_N + j < q_current_seqlen),
                            qkT[i, j],
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            by * block_M + i < k_current_seqlen and k_base * block_N + j < q_current_seqlen, qkT[i, j], 0
                        )
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k_base * block_N : (k_base + 1) * block_N], delta)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim_qk):
                    if k_base * block_N + i < q_current_seqlen:
                        T.atomic_add(dQ[q_start_idx + k_base * block_N + i, bx, j], dq[i, j], memory_order="relaxed")

            T.copy(dv, dv_shared)
            T.copy(dv_shared, dV[bx % groups, k_start_idx + by * block_M : k_start_idx + by * block_M + block_M, bx // groups, :])
            T.copy(dk, dk_shared)
            T.copy(dk_shared, dK[bx % groups, k_start_idx + by * block_M : k_start_idx + by * block_M + block_M, bx // groups, :])

    return flash_bwd


@torch.compile
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, seqlens_q, seqlens_k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, groups=1, use_atomic=True
    ):
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        D_HEAD_V = v.shape[-1]
        block_M = 128
        block_N = 64
        q_unpad, indices_q, _, _ = unpad_input(q, (torch.arange(N_CTX, device=q.device).unsqueeze(0) < seqlens_q.unsqueeze(1)))
        k_unpad, indices_k, _, _ = unpad_input(k, (torch.arange(N_CTX, device=k.device).unsqueeze(0) < seqlens_k.unsqueeze(1)))
        v_unpad, _, _, _ = unpad_input(v, (torch.arange(N_CTX, device=v.device).unsqueeze(0) < seqlens_k.unsqueeze(1)))

        total_q = q_unpad.shape[0]
        total_kv = k_unpad.shape[0]

        mod = flashattn_fwd(BATCH, total_q, total_kv, N_CTX, H, max_seqlen_q, D_HEAD_QK, D_HEAD_V, causal, block_M, block_N, groups)
        o_unpad, lse = mod(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k)
        o = pad_input(o_unpad, indices_q, BATCH, N_CTX)
        ctx.save_for_backward(q_unpad, k_unpad, v_unpad, o_unpad, lse, seqlens_q, seqlens_k, cu_seqlens_q, cu_seqlens_k)
        ctx.batch = BATCH
        ctx.causal = causal
        ctx.use_atomic = use_atomic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.indices_q = indices_q
        ctx.indices_k = indices_k
        return o

    @staticmethod
    def backward(ctx, do):
        N_CTX = do.shape[1]
        q, k, v, o, lse_clone, seqlens_q, seqlens_k, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        # lse_clone = lse.clone()
        do_unpad, _, _, _ = unpad_input(do, (torch.arange(N_CTX, device=do.device).unsqueeze(0) < seqlens_q.unsqueeze(1)))
        total_q, H, D_HEAD_QK = q.shape
        total_kv, HEAD_KV, D_HEAD_V = v.shape
        groups = H // HEAD_KV
        BATCH = len(cu_seqlens_q) - 1

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do_unpad, q, k, v, o)]
        block_M = 128
        block_N = 32
        mod_prep = flashattn_bwd_preprocess(BATCH, H, total_q, N_CTX, ctx.max_seqlen_q, D_HEAD_V)
        mod_post = flashattn_bwd_postprocess(total_q, total_kv, H, HEAD_KV, D_HEAD_QK, D_HEAD_V)
        delta = mod_prep(o, do, cu_seqlens_q)

        if ctx.use_atomic:
            kernel = flashattn_bwd_atomic_add(
                BATCH,
                total_q,
                total_kv,
                N_CTX,
                H,
                ctx.max_seqlen_q,
                D_HEAD_QK,
                D_HEAD_V,
                ctx.causal,
                block_M,
                block_N,
                threads=256,
                num_stages=2,
                groups=groups,
            )
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)
            kernel(q, k, v, do, lse_clone, delta, cu_seqlens_q, cu_seqlens_k, dq, dk, dv)
            dq, dk, dv = mod_post(dq, dk, dv)
        else:
            kernel = flashattn_bwd_split(
                BATCH,
                total_q,
                total_kv,
                N_CTX,
                H,
                ctx.max_seqlen_q,
                D_HEAD_QK,
                D_HEAD_V,
                ctx.causal,
                block_M,
                block_N,
                threads=256,
                num_stages=2,
                groups=groups,
            )
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.empty(groups, *k.shape, dtype=torch.float16, device=q.device)
            dv = torch.empty(groups, *v.shape, dtype=torch.float16, device=q.device)
            kernel(q, k, v, do, lse_clone, delta, cu_seqlens_q, cu_seqlens_k, dq, dk, dv)
            dq, _, _ = mod_post(dq, torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32))
            dk, dv = dk.sum(0), dv.sum(0)

        dq = pad_input(dq, ctx.indices_q, BATCH, N_CTX)
        dk = pad_input(dk, ctx.indices_k, BATCH, N_CTX)
        dv = pad_input(dv, ctx.indices_k, BATCH, N_CTX)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


attention = _attention.apply


def ref_program(Q, K, V, padding_mask, is_causal, groups=1):
    # Q: [B, T, HQ, D_QK]
    # K: [B, T, HK, D_QK]
    # V: [B, T, HV, D_V]
    # HQ = HKV * groups
    # To handle precision issue
    Q, K, V = Q.float(), K.float(), V.float()
    assert Q.size(2) == K.size(2) * groups, f"Q.size(2): {Q.size(2)}, K.size(2): {K.size(2)}, groups: {groups}"
    assert Q.size(2) == V.size(2) * groups, f"Q.size(2): {Q.size(2)}, V.size(2): {V.size(2)}, groups: {groups}"

    dim_qk = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim_qk, dtype=scores.dtype))
    if padding_mask is not None:
        scores.masked_fill_(rearrange(~padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    if padding_mask is not None:
        output.masked_fill_(rearrange(~padding_mask, "b s -> b s 1 1"), 0.0)
    return output


def main(
    BATCH: int = 1,
    H: int = 32,
    N_CTX: int = 256,
    D_HEAD_QK: int = 192,
    D_HEAD_V: int = 128,
    groups: int = 16,
    causal: bool = False,
    use_atomic: bool = True,
):
    flops_per_qk = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD_QK
    flops_per_v = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD_V
    total_flops = 3 * flops_per_qk + 2 * flops_per_v
    if causal:
        total_flops *= 0.5
    Q = torch.empty(BATCH, N_CTX, H, D_HEAD_QK, dtype=torch.half, device="cuda").normal_().requires_grad_()

    head_kv = H // groups
    K = torch.empty(BATCH, N_CTX, head_kv, D_HEAD_QK, dtype=torch.half, device="cuda").normal_().requires_grad_()
    V = torch.empty(BATCH, N_CTX, head_kv, D_HEAD_V, dtype=torch.half, device="cuda").normal_().requires_grad_()
    dO = torch.empty(BATCH, N_CTX, H, D_HEAD_V, dtype=torch.half, device="cuda").normal_().requires_grad_()
    padding_mask = generate_random_padding_mask(N_CTX, BATCH, "cuda", mode="random")
    seqlens_q = padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens_q = F.pad(torch.cumsum(seqlens_q, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen_q = seqlens_q.max().item()

    # In training backward pass, seqlens_k should be the same as seqlens_q
    seqlens_k, cu_seqlens_k, max_seqlen_k = seqlens_q, cu_seqlens_q, max_seqlen_q

    O = attention(Q, K, V, seqlens_q, seqlens_k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, groups, use_atomic)
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None

    O_ref = ref_program(Q, K, V, padding_mask, causal, groups)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None

    def run():
        O_ref.backward(dO, retain_graph=True)

    def run1():
        O.backward(dO, retain_graph=True)

    from tilelang.profiler import do_bench

    latency = do_bench(run, warmup=500)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(run1, warmup=500)
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))

    torch.testing.assert_close(O, O_ref.half(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dQ, dQ_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dK, dK_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dV, dV_ref, rtol=1e-2, atol=1e-2)
    print("All checks passed.✅")
    print(
        "Note: this varlen kernel performance is as good as the non-varlen kernel shown in Nsight-Compute. As you may observe that the TFLOPS is a bit lower, that's because the unpad operation is included in the above benchmark."
    )


def run_regression_perf():
    BATCH = 1
    H = 32
    N_CTX = 256
    D_HEAD_QK = 192
    D_HEAD_V = 128
    groups = 16
    causal = False
    device = "cuda"
    torch.manual_seed(42)
    total_q = BATCH * N_CTX
    total_kv = BATCH * N_CTX
    head_kv = H // groups
    Q = torch.randn(total_q, H, D_HEAD_QK, device=device, dtype=torch.half)
    K = torch.randn(total_kv, head_kv, D_HEAD_QK, device=device, dtype=torch.half)
    V = torch.randn(total_kv, head_kv, D_HEAD_V, device=device, dtype=torch.half)
    O = torch.randn(total_q, H, D_HEAD_V, device=device, dtype=torch.half)
    dO = torch.randn(total_q, H, D_HEAD_V, device=device, dtype=torch.half)
    cu_seqlens_q = torch.arange(0, (BATCH + 1) * N_CTX, N_CTX, device=device, dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_q
    max_seqlen_q = N_CTX
    lse = torch.zeros(BATCH, H, N_CTX, device=device, dtype=torch.float32)
    with torch.no_grad():
        mod_prep = flashattn_bwd_preprocess(BATCH, H, total_q, N_CTX, max_seqlen_q, D_HEAD_V)
        kernel = flashattn_bwd_split(
            BATCH,
            total_q,
            total_kv,
            N_CTX,
            H,
            max_seqlen_q,
            D_HEAD_QK,
            D_HEAD_V,
            causal,
            block_M=128,
            block_N=32,
            threads=256,
            num_stages=2,
            groups=groups,
        )
    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dK = torch.zeros(groups, total_kv, head_kv, D_HEAD_QK, device=device, dtype=torch.float16)
    dV = torch.zeros(groups, total_kv, head_kv, D_HEAD_V, device=device, dtype=torch.float16)
    Delta = mod_prep(O, dO, cu_seqlens_q)
    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(Q, K, V, dO, lse, Delta, cu_seqlens_q, cu_seqlens_k, dQ, dK, dV)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    arch = nvcc.get_target_compute_version()
    print(f"Detected GPU compute capability: {arch}")
    assert float(arch) >= 9.0, "This example only supports GPU with compute capability >= 9.0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--h", type=int, default=32, help="Number of heads")
    parser.add_argument("--n_ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--d_head_qk", type=int, default=192, help="Head dimension for Q/K")
    parser.add_argument("--d_head_v", type=int, default=128, help="Head dimension for V")
    parser.add_argument("--causal", action="store_true", help="Causal flag")
    parser.add_argument("--groups", type=int, default=16, help="groups")
    parser.add_argument("--use_atomic", action="store_true", default=False, help="Use atomic add for dK/dV")
    parser.add_argument("--use_split", action="store_true", default=False, help="Use split for dK/dV")
    args = parser.parse_args()
    # Can be set to True/False for testing
    args.causal = True

    # Handle backward compatibility and logic
    if args.use_split:
        use_atomic = False
    elif args.use_atomic:
        use_atomic = True
    else:
        # Default: use atomic
        use_atomic = True

    main(args.batch, args.h, args.n_ctx, args.d_head_qk, args.d_head_v, args.groups, args.causal, use_atomic)
