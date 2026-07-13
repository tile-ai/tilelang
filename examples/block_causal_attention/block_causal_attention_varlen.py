import itertools

import torch
import tilelang
import tilelang.language as T

from block_causal_attention import (
    LOG2_E,
    _SUPPORTED_DLLM_BLOCKS,
    _check_dllm_block,
    _clone_with_grad,
    _fwd_tile_allowed,
    block_causal_attention_ref,
)


@tilelang.jit
def _fwd_varlen_template(
    batch: int,
    heads: int,
    dim: int,
    dllm_block: int,
    softmax_scale: float,
    block_size: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    _check_dllm_block(dllm_block, block_size)
    scale_log2e = softmax_scale * LOG2_E
    accum_dtype = "float32"
    neg_inf = -1.0e6
    total_tokens = T.dynamic("total_tokens")
    shape = [total_tokens, heads, dim]

    @T.prim_func
    def fwd(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        cu_seqlens: T.Tensor([batch + 1], "int32"),
        max_seqlen: T.int32,
        Output: T.Tensor(shape, dtype),
        LSE: T.Tensor([heads, total_tokens], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen, block_size), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_size, dim], dtype)
            K_shared = T.alloc_shared([block_size, dim], dtype)
            V_shared = T.alloc_shared([block_size, dim], dtype)
            O_shared = T.alloc_shared([block_size, dim], dtype)
            acc_s = T.alloc_fragment([block_size, block_size], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_size, block_size], dtype)
            acc_o = T.alloc_fragment([block_size, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_size], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_size], accum_dtype)
            scores_scale = T.alloc_fragment([block_size], accum_dtype)
            scores_sum = T.alloc_fragment([block_size], accum_dtype)
            logsum = T.alloc_fragment([block_size], accum_dtype)

            seq_start = cu_seqlens[bz]
            seqlen = cu_seqlens[bz + 1] - seq_start
            region_tiles = (seqlen // 2) // block_size
            clean_off = seq_start + region_tiles * block_size
            q_is_noisy = bx < region_tiles

            if bx < region_tiles * 2:
                T.copy(Q[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, neg_inf)

                last_clean_step = T.if_then_else(q_is_noisy, bx, bx - region_tiles)
                num_steps = T.if_then_else(q_is_noisy, bx + 2, (bx - region_tiles) + 1)
                for s in T.Pipelined(num_steps, num_stages=num_stages):
                    is_noisy_diag = T.if_then_else(s > last_clean_step, 1, 0)
                    if is_noisy_diag != 0:
                        T.copy(K[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :], K_shared)
                    else:
                        T.copy(K[clean_off + s * block_size : clean_off + (s + 1) * block_size, by, :], K_shared)

                    needs_mask = is_noisy_diag + T.if_then_else(s == last_clean_step, 1, 0)
                    if needs_mask != 0:
                        for i, j in T.Parallel(block_size, block_size):
                            allowed = _fwd_tile_allowed(i, j, dllm_block, is_noisy_diag, q_is_noisy)
                            acc_s[i, j] = T.if_then_else(allowed > 0, 0, neg_inf)
                    else:
                        T.clear(acc_s)

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, neg_inf)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_size):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_size):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale_log2e - scores_max[i] * scale_log2e)
                    for i, j in T.Parallel(block_size, block_size):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale_log2e - scores_max[i] * scale_log2e)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_size):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_size, dim):
                        acc_o[i, j] *= scores_scale[i]
                    if is_noisy_diag != 0:
                        T.copy(V[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :], V_shared)
                    else:
                        T.copy(V[clean_off + s * block_size : clean_off + (s + 1) * block_size, by, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_size, dim):
                    acc_o[i, j] /= logsum[i] + 1e-30
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :])
                for i in T.Parallel(block_size):
                    logsum[i] = T.log2(logsum[i] + 1e-30) + scores_max[i] * scale_log2e
                T.copy(logsum, LSE[by, seq_start + bx * block_size : seq_start + (bx + 1) * block_size])

    return fwd


@tilelang.jit
def _bwd_preprocess_varlen_template(heads: int, dim: int, block_size: int = 64, dtype: str = "bfloat16"):
    accum_dtype = "float32"
    assert dim % block_size == 0, f"Delta preprocessing requires dim divisible by {block_size}, got {dim}"
    total_tokens = T.dynamic("total_tokens")
    shape = [total_tokens, heads, dim]

    @T.prim_func
    def prep(
        Output: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([heads, total_tokens], accum_dtype),
    ):
        # Delta is a pure per-token row-sum of O*dO, independent of the mask, so
        # we sweep the flat packed token axis (total_tokens is a multiple of block_size).
        with T.Kernel(heads, T.ceildiv(total_tokens, block_size)) as (bx, by):
            o = T.alloc_fragment([block_size, block_size], dtype)
            do = T.alloc_fragment([block_size, block_size], dtype)
            acc = T.alloc_fragment([block_size, block_size], accum_dtype)
            delta = T.alloc_fragment([block_size], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, block_size)):
                T.copy(Output[by * block_size : (by + 1) * block_size, bx, k * block_size : (k + 1) * block_size], o)
                T.copy(dO[by * block_size : (by + 1) * block_size, bx, k * block_size : (k + 1) * block_size], do)
                for i, j in T.Parallel(block_size, block_size):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bx, by * block_size : (by + 1) * block_size])

    return prep


@tilelang.jit
def _bwd_dq_varlen_template(
    batch: int,
    heads: int,
    dim: int,
    dllm_block: int,
    softmax_scale: float,
    block_size: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    """Varlen query-parallel dQ (atomics-free). Mirrors the fixed-length dQ kernel per sequence."""
    _check_dllm_block(dllm_block, block_size)
    sm_scale = softmax_scale
    scale_log2e = sm_scale * LOG2_E
    accum_dtype = "float32"
    total_tokens = T.dynamic("total_tokens")
    shape = [total_tokens, heads, dim]

    @T.prim_func
    def dq_kernel(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        cu_seqlens: T.Tensor([batch + 1], "int32"),
        max_seqlen: T.int32,
        LSE: T.Tensor([heads, total_tokens], accum_dtype),
        Delta: T.Tensor([heads, total_tokens], accum_dtype),
        dQ: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen, block_size), heads, batch, threads=threads) as (bx, by, bz):
            q = T.alloc_shared([block_size, dim], dtype)
            k_shared = T.alloc_shared([block_size, dim], dtype)
            v_shared = T.alloc_shared([block_size, dim], dtype)
            do = T.alloc_shared([block_size, dim], dtype)
            lse = T.alloc_shared([block_size], accum_dtype)
            delta = T.alloc_shared([block_size], accum_dtype)
            qk = T.alloc_fragment([block_size, block_size], accum_dtype)
            ds = T.alloc_fragment([block_size, block_size], accum_dtype)
            ds_cast = T.alloc_fragment([block_size, block_size], dtype)
            dq = T.alloc_fragment([block_size, dim], accum_dtype)
            dq_shared = T.alloc_shared([block_size, dim], dtype)

            seq_start = cu_seqlens[bz]
            seqlen = cu_seqlens[bz + 1] - seq_start
            region_tiles = (seqlen // 2) // block_size
            clean_off = seq_start + region_tiles * block_size
            q_is_noisy = bx < region_tiles

            if bx < region_tiles * 2:
                T.copy(Q[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :], q)
                T.copy(dO[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :], do)
                T.copy(LSE[by, seq_start + bx * block_size : seq_start + (bx + 1) * block_size], lse)
                T.copy(Delta[by, seq_start + bx * block_size : seq_start + (bx + 1) * block_size], delta)
                T.clear(dq)

                last_clean_step = T.if_then_else(q_is_noisy, bx, bx - region_tiles)
                num_steps = T.if_then_else(q_is_noisy, bx + 2, (bx - region_tiles) + 1)
                for s in T.Pipelined(num_steps, num_stages=num_stages):
                    is_noisy_diag = T.if_then_else(s > last_clean_step, 1, 0)
                    if is_noisy_diag != 0:
                        T.copy(K[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :], k_shared)
                        T.copy(V[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :], v_shared)
                    else:
                        T.copy(K[clean_off + s * block_size : clean_off + (s + 1) * block_size, by, :], k_shared)
                        T.copy(V[clean_off + s * block_size : clean_off + (s + 1) * block_size, by, :], v_shared)

                    T.clear(qk)
                    T.gemm(q, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_size, block_size):
                        qk[i, j] = T.exp2(qk[i, j] * scale_log2e - lse[i])

                    needs_mask = is_noisy_diag + T.if_then_else(s == last_clean_step, 1, 0)
                    if needs_mask != 0:
                        for i, j in T.Parallel(block_size, block_size):
                            allowed = _fwd_tile_allowed(i, j, dllm_block, is_noisy_diag, q_is_noisy)
                            qk[i, j] = T.if_then_else(allowed > 0, qk[i, j], 0)

                    T.clear(ds)
                    T.gemm(do, v_shared, ds, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_size, block_size):
                        ds_cast[i, j] = qk[i, j] * (ds[i, j] - delta[i]) * sm_scale
                    T.gemm(ds_cast, k_shared, dq, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dq, dq_shared)
                T.copy(dq_shared, dQ[seq_start + bx * block_size : seq_start + (bx + 1) * block_size, by, :])

    return dq_kernel


@tilelang.jit
def _bwd_dkv_varlen_template(
    batch: int,
    heads: int,
    dim: int,
    dllm_block: int,
    softmax_scale: float,
    block_size: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    """Varlen key-parallel dK/dV (atomics-free).

    The reverse schedule per sequence. `region_tiles` is per-sequence but only drives index arithmetic.
    """
    _check_dllm_block(dllm_block, block_size)
    sm_scale = softmax_scale
    scale_log2e = sm_scale * LOG2_E
    accum_dtype = "float32"
    total_tokens = T.dynamic("total_tokens")
    shape = [total_tokens, heads, dim]

    @T.prim_func
    def dkv_kernel(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        cu_seqlens: T.Tensor([batch + 1], "int32"),
        max_seqlen: T.int32,
        LSE: T.Tensor([heads, total_tokens], accum_dtype),
        Delta: T.Tensor([heads, total_tokens], accum_dtype),
        dK: T.Tensor(shape, dtype),
        dV: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(max_seqlen, block_size), batch, threads=threads) as (bx, by, bz):
            k_shared = T.alloc_shared([block_size, dim], dtype)
            v_shared = T.alloc_shared([block_size, dim], dtype)
            q = T.alloc_shared([block_size, dim], dtype)
            do = T.alloc_shared([block_size, dim], dtype)
            lse = T.alloc_shared([block_size], accum_dtype)
            delta = T.alloc_shared([block_size], accum_dtype)
            qkT = T.alloc_fragment([block_size, block_size], accum_dtype)
            dsT = T.alloc_fragment([block_size, block_size], accum_dtype)
            qkT_cast = T.alloc_fragment([block_size, block_size], dtype)
            dsT_cast = T.alloc_fragment([block_size, block_size], dtype)
            dv = T.alloc_fragment([block_size, dim], accum_dtype)
            dk = T.alloc_fragment([block_size, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_size, dim], dtype)
            dk_shared = T.alloc_shared([block_size, dim], dtype)

            seq_start = cu_seqlens[bz]
            seqlen = cu_seqlens[bz + 1] - seq_start
            region_tiles = (seqlen // 2) // block_size

            if by < region_tiles * 2:
                T.copy(K[seq_start + by * block_size : seq_start + (by + 1) * block_size, bx, :], k_shared)
                T.copy(V[seq_start + by * block_size : seq_start + (by + 1) * block_size, bx, :], v_shared)
                T.clear(dv)
                T.clear(dk)

                # noisy group -- noisy key: just the diagonal {by}; clean key: [local, region_tiles)
                # clean group -- clean key only: [region_tiles + local, 2 * region_tiles)
                key_is_noisy = by < region_tiles
                local = by - region_tiles
                span = region_tiles - local
                noisy_start = T.if_then_else(key_is_noisy, by, local)
                noisy_count = T.if_then_else(key_is_noisy, 1, span)
                clean_start = region_tiles + local
                clean_count = T.if_then_else(key_is_noisy, 0, span)
                is_noisy_diag = T.if_then_else(key_is_noisy, 1, 0)

                # noisy-query group (mask: noisy key always -> block diagonal
                for step in T.Pipelined(noisy_count, num_stages=num_stages):
                    q_tile = noisy_start + step
                    T.copy(Q[seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size, bx, :], q)
                    T.clear(qkT)
                    T.gemm(k_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(LSE[bx, seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size], lse)
                    for i, j in T.Parallel(block_size, block_size):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale_log2e - lse[j])
                    needs_mask = T.if_then_else(key_is_noisy, 1, T.if_then_else(step == 0, 1, 0))
                    if needs_mask != 0:
                        for i, j in T.Parallel(block_size, block_size):
                            allowed = _fwd_tile_allowed(j, i, dllm_block, is_noisy_diag, q_tile < region_tiles)
                            qkT[i, j] = T.if_then_else(allowed > 0, qkT[i, j], 0)
                    T.copy(dO[seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size, bx, :], do)
                    T.clear(dsT)
                    T.gemm(v_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(Delta[bx, seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size], delta)
                    for i, j in T.Parallel(block_size, block_size):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                # clean-query group (clean keys only, boundary = clean diagonal at step 0)
                for step in T.Pipelined(clean_count, num_stages=num_stages):
                    q_tile = clean_start + step
                    T.copy(Q[seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size, bx, :], q)
                    T.clear(qkT)
                    T.gemm(k_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(LSE[bx, seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size], lse)
                    for i, j in T.Parallel(block_size, block_size):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale_log2e - lse[j])
                    needs_mask = T.if_then_else(step == 0, 1, 0)
                    if needs_mask != 0:
                        for i, j in T.Parallel(block_size, block_size):
                            allowed = _fwd_tile_allowed(j, i, dllm_block, is_noisy_diag, q_tile < region_tiles)
                            qkT[i, j] = T.if_then_else(allowed > 0, qkT[i, j], 0)
                    T.copy(dO[seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size, bx, :], do)
                    T.clear(dsT)
                    T.gemm(v_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(Delta[bx, seq_start + q_tile * block_size : seq_start + (q_tile + 1) * block_size], delta)
                    for i, j in T.Parallel(block_size, block_size):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[seq_start + by * block_size : seq_start + (by + 1) * block_size, bx, :])
                T.copy(dk_shared, dK[seq_start + by * block_size : seq_start + (by + 1) * block_size, bx, :])

    return dkv_kernel


class _BlockCausalAttentionVarlenTL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, dllm_block, softmax_scale, max_seqlen, block_size):
        total, heads, dim = q.shape
        batch = cu_seqlens.numel() - 1
        dtype = T.dtype(q.dtype)
        q, k, v = (t.contiguous() for t in (q, k, v))
        fwd = _fwd_varlen_template(batch, heads, dim, dllm_block, softmax_scale, block_size=block_size, dtype=dtype)
        o = torch.empty_like(q)
        lse = torch.empty((heads, total), device=q.device, dtype=torch.float32)
        fwd(q, k, v, cu_seqlens, max_seqlen, o, lse)
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens)
        ctx.dllm_block = dllm_block
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen = max_seqlen
        ctx.block_size = block_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, cu_seqlens = ctx.saved_tensors
        total, heads, dim = q.shape
        batch = cu_seqlens.numel() - 1
        dtype = T.dtype(q.dtype)

        do, q, k, v, o = (t.contiguous() for t in (do, q, k, v, o))
        prep = _bwd_preprocess_varlen_template(heads, dim, block_size=ctx.block_size, dtype=dtype)
        dq_kernel = _bwd_dq_varlen_template(batch, heads, dim, ctx.dllm_block, ctx.softmax_scale, block_size=ctx.block_size, dtype=dtype)
        dkv_kernel = _bwd_dkv_varlen_template(
            batch, heads, dim, ctx.dllm_block, ctx.softmax_scale, block_size=ctx.block_size, num_stages=1, dtype=dtype
        )
        delta = torch.empty((heads, total), device=q.device, dtype=torch.float32)
        prep(o, do, delta)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq_kernel(q, k, v, do, cu_seqlens, ctx.max_seqlen, lse, delta, dq)
        dkv_kernel(q, k, v, do, cu_seqlens, ctx.max_seqlen, lse, delta, dk, dv)
        return dq, dk, dv, None, None, None, None, None


def block_causal_attention_varlen(
    q_unpad, k_unpad, v_unpad, cu_seqlens, dllm_block_size: int, softmax_scale=None, max_seqlen=None, block_size: int = 64
):
    """Varlen dLLM block-causal attention over packed `[total, heads, dim]` tensors.

    `cu_seqlens` is the `[batch + 1]` prefix-sum of full sequence lengths (flash-attn style). Each
    sequence is laid out as [noisy, clean], and every length must be a multiple of `2 * block_size`
    so the halves split on a tile edge. Passing `max_seqlen` skips the host sync that checks this,
    making it the caller's responsibility. `block_size` is the tile size (it must divide `dim` and be
    a multiple of `dllm_block_size`).
    """
    _check_dllm_block(dllm_block_size, block_size)  # fail fast at the API, not deep in a JIT build
    if softmax_scale is None:
        softmax_scale = q_unpad.shape[-1] ** -0.5
    cu_seqlens = cu_seqlens.to(torch.int32)
    if max_seqlen is None:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        for length in lengths:
            assert length % (2 * block_size) == 0, f"each seqlen must be a multiple of 2*block_size ({2 * block_size}); got {length}"
        max_seqlen = max(lengths) if lengths else 0
    return _BlockCausalAttentionVarlenTL.apply(
        q_unpad, k_unpad, v_unpad, cu_seqlens, dllm_block_size, float(softmax_scale), int(max_seqlen), int(block_size)
    )


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------


def block_causal_attention_varlen_ref(q_unpad, k_unpad, v_unpad, cu_seqlens, dllm_block_size: int, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = q_unpad.shape[-1] ** -0.5
    cu = cu_seqlens.tolist()
    outs = []
    for b in range(len(cu) - 1):
        s, e = cu[b], cu[b + 1]
        q = q_unpad[s:e].unsqueeze(0)
        k = k_unpad[s:e].unsqueeze(0)
        v = v_unpad[s:e].unsqueeze(0)
        outs.append(block_causal_attention_ref(q, k, v, dllm_block_size, softmax_scale).squeeze(0))
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def _run_varlen_case(lengths, heads, dim, dllm_block, block_size=64, dtype=torch.float16):
    torch.manual_seed(0)
    cu = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    total = int(cu[-1].item())
    query = torch.randn(total, heads, dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    grad = torch.randn_like(query)

    q_ref, k_ref, v_ref = (_clone_with_grad(t) for t in (query, key, value))
    q_tl, k_tl, v_tl = (_clone_with_grad(t) for t in (query, key, value))

    out_ref = block_causal_attention_varlen_ref(q_ref, k_ref, v_ref, cu, dllm_block)
    out_tl = block_causal_attention_varlen(q_tl, k_tl, v_tl, cu, dllm_block, block_size=block_size)
    torch.testing.assert_close(out_tl, out_ref, atol=2e-2, rtol=2e-2)

    out_ref.backward(grad)
    out_tl.backward(grad)
    torch.testing.assert_close(q_tl.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_tl.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v_tl.grad, v_ref.grad, atol=5e-2, rtol=5e-2)


def test_block_causal_attention_varlen():
    lengths = [128, 256, 384]
    for dllm_block in _SUPPORTED_DLLM_BLOCKS:
        _run_varlen_case(lengths, heads=2, dim=64, dllm_block=dllm_block)
        print(f"[varlen] dllm_block={dllm_block:>2} OK  lengths={lengths}")

    block_size, dllm_block = 32, 16
    _run_varlen_case([128, 256], heads=2, dim=64, dllm_block=dllm_block, block_size=block_size)
    print(f"[varlen] {block_size=} {dllm_block=} OK")


def main():
    test_block_causal_attention_varlen()


if __name__ == "__main__":
    main()
