import os

import torch
import tilelang
import tilelang.language as T

LOG2_E = 1.4426950408889634

_FAST_MATH = os.environ.get("BLOCK_CAUSAL_TL_FAST_MATH", "0") == "1"

_PASS_CFG = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: _FAST_MATH}
_FWD_PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: _FAST_MATH,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}
_DQ_PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: _FAST_MATH,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}


def _diffusion_allowed(q_abs, k_abs, half_len: int, mask_block: int):
    q_clean = T.if_then_else(q_abs >= half_len, 1, 0)
    k_clean = T.if_then_else(k_abs >= half_len, 1, 0)
    q_local = T.if_then_else(q_abs >= half_len, q_abs - half_len, q_abs)
    k_local = T.if_then_else(k_abs >= half_len, k_abs - half_len, k_abs)
    q_block = q_local // mask_block
    k_block = k_local // mask_block

    same_region = T.if_then_else(q_clean == k_clean, 1, 0)
    block_diagonal = T.if_then_else(q_block == k_block, 1, 0) * same_region
    offset_causal = T.if_then_else(q_block > k_block, 1, 0) * (1 - q_clean) * k_clean
    clean_causal = T.if_then_else(q_block >= k_block, 1, 0) * q_clean * k_clean
    return block_diagonal + offset_causal + clean_causal


def _bwd_allowed_64x64_mask32(query_tile, key_tile, query_col, key_row, region_tiles: int):
    key_clean = T.if_then_else(key_tile >= region_tiles, 1, 0)
    query_clean = T.if_then_else(query_tile >= region_tiles, 1, 0)
    key_local_tile = T.if_then_else(key_clean != 0, key_tile - region_tiles, key_tile)
    query_local_tile = T.if_then_else(query_clean != 0, query_tile - region_tiles, query_tile)
    key_half = T.if_then_else(key_row >= 32, 1, 0)
    query_half = T.if_then_else(query_col >= 32, 1, 0)

    noisy_diag = (1 - key_clean) * T.if_then_else(query_half == key_half, 1, 0)
    clean_full = key_clean * T.if_then_else(query_local_tile > key_local_tile, 1, 0)
    clean_boundary = key_clean * T.if_then_else(query_local_tile == key_local_tile, 1, 0) * (
        query_clean * T.if_then_else(query_half >= key_half, 1, 0)
        + (1 - query_clean) * T.if_then_else(query_half > key_half, 1, 0)
    )
    return noisy_diag + clean_full + clean_boundary


def _fwd_template(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    mask_block: int,
    softmax_scale: float,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    assert seq_len % 2 == 0, "seq_len must be noisy|clean halves"
    half_len = seq_len // 2
    assert half_len % block_M == 0, "half_len must be divisible by block_M"
    assert block_M == block_N, "forward uses square tiles"
    region_tiles = half_len // block_M

    scale = softmax_scale
    scale_log2e = scale * LOG2_E
    accum_dtype = "float32"
    neg_inf = -1.0e6
    shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def fwd(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, neg_inf)

            last_clean_step = T.if_then_else(bx < region_tiles, bx, bx - region_tiles)
            num_steps = T.if_then_else(bx < region_tiles, bx + 2, (bx - region_tiles) + 1)

            for s in T.Pipelined(num_steps, num_stages=num_stages):
                is_noisy_diag = T.if_then_else(s > last_clean_step, 1, 0)
                if is_noisy_diag != 0:
                    T.copy(K[bz, bx * block_M : (bx + 1) * block_M, by, :], K_shared)
                else:
                    T.copy(K[bz, half_len + s * block_M : half_len + (s + 1) * block_M, by, :], K_shared)

                is_clean_diag = T.if_then_else(s == last_clean_step, 1, 0)
                needs_mask = is_noisy_diag + is_clean_diag
                if needs_mask != 0:
                    for i, j in T.Parallel(block_M, block_N):
                        if block_M == 64 and block_N == 64 and mask_block == 32:
                            q_hi = T.if_then_else(i >= 32, 1, 0)
                            k_hi = T.if_then_else(j >= 32, 1, 0)
                            noisy_allowed = T.if_then_else(q_hi == k_hi, 1, 0)
                            clean_boundary = T.if_then_else(
                                bx < region_tiles,
                                T.if_then_else(q_hi > k_hi, 1, 0),
                                T.if_then_else(q_hi >= k_hi, 1, 0),
                            )
                            allowed = T.if_then_else(is_noisy_diag == 1, noisy_allowed, clean_boundary)
                        else:
                            k_abs = T.if_then_else(is_noisy_diag == 1, bx * block_M + j, half_len + s * block_M + j)
                            allowed = _diffusion_allowed(bx * block_M + i, k_abs, half_len, mask_block)
                        acc_s[i, j] = T.if_then_else(allowed > 0, 0, neg_inf)
                else:
                    T.clear(acc_s)

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, neg_inf)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale_log2e - scores_max[i] * scale_log2e)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale_log2e - scores_max[i] * scale_log2e)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                if is_noisy_diag != 0:
                    T.copy(V[bz, bx * block_M : (bx + 1) * block_M, by, :], V_shared)
                else:
                    T.copy(V[bz, half_len + s * block_M : half_len + (s + 1) * block_M, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i] + 1e-30
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i] + 1e-30) + scores_max[i] * scale_log2e
            T.copy(logsum, LSE[bz, by, bx * block_M : (bx + 1) * block_M])

    return fwd


@tilelang.jit(out_idx=[2], pass_configs=_PASS_CFG)
def _build_bwd_preprocess(batch: int, heads: int, seq_len: int, dim: int, dtype: str = "bfloat16"):
    accum_dtype = "float32"
    shape = [batch, seq_len, heads, dim]
    block = 64

    @T.prim_func
    def prep(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block), batch) as (bx, by, bz):
            o = T.alloc_fragment([block, block], dtype)
            do = T.alloc_fragment([block, block], dtype)
            acc = T.alloc_fragment([block, block], accum_dtype)
            delta = T.alloc_fragment([block], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, block)):
                T.copy(O[bz, by * block : (by + 1) * block, bx, k * block : (k + 1) * block], o)
                T.copy(dO[bz, by * block : (by + 1) * block, bx, k * block : (k + 1) * block], do)
                for i, j in T.Parallel(block, block):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * block : (by + 1) * block])

    return prep


def _bwd_dq_template(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    mask_block: int,
    softmax_scale: float,
    block: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    assert block == 64 and mask_block == 32, "split dQ is specialized for 64x64 tiles with mask_block=32"
    half_len = seq_len // 2
    region_tiles = half_len // block
    sm_scale = softmax_scale
    scale_log2e = sm_scale * LOG2_E
    accum_dtype = "float32"
    shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def dq_kernel(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block), heads, batch, threads=threads) as (bx, by, bz):
            q = T.alloc_shared([block, dim], dtype)
            k_shared = T.alloc_shared([block, dim], dtype)
            v_shared = T.alloc_shared([block, dim], dtype)
            do = T.alloc_shared([block, dim], dtype)
            lse = T.alloc_shared([block], accum_dtype)
            delta = T.alloc_shared([block], accum_dtype)
            qk = T.alloc_fragment([block, block], accum_dtype)
            ds = T.alloc_fragment([block, block], accum_dtype)
            ds_cast = T.alloc_fragment([block, block], dtype)
            dq = T.alloc_fragment([block, dim], accum_dtype)
            dq_shared = T.alloc_shared([block, dim], dtype)

            T.copy(Q[bz, bx * block : (bx + 1) * block, by, :], q)
            T.copy(dO[bz, bx * block : (bx + 1) * block, by, :], do)
            T.copy(LSE[bz, by, bx * block : (bx + 1) * block], lse)
            T.copy(Delta[bz, by, bx * block : (bx + 1) * block], delta)
            T.clear(dq)

            last_clean_step = T.if_then_else(bx < region_tiles, bx, bx - region_tiles)
            num_steps = T.if_then_else(bx < region_tiles, bx + 2, (bx - region_tiles) + 1)
            for s in T.Pipelined(num_steps, num_stages=num_stages):
                is_noisy_diag = T.if_then_else(s > last_clean_step, 1, 0)
                if is_noisy_diag != 0:
                    T.copy(K[bz, bx * block : (bx + 1) * block, by, :], k_shared)
                    T.copy(V[bz, bx * block : (bx + 1) * block, by, :], v_shared)
                else:
                    T.copy(K[bz, half_len + s * block : half_len + (s + 1) * block, by, :], k_shared)
                    T.copy(V[bz, half_len + s * block : half_len + (s + 1) * block, by, :], v_shared)

                T.clear(qk)
                T.gemm(q, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block, block):
                    qk[i, j] = T.exp2(qk[i, j] * scale_log2e - lse[i])

                needs_mask = is_noisy_diag + T.if_then_else(s == last_clean_step, 1, 0)
                if needs_mask != 0:
                    for i, j in T.Parallel(block, block):
                        q_hi = T.if_then_else(i >= 32, 1, 0)
                        k_hi = T.if_then_else(j >= 32, 1, 0)
                        noisy_allowed = T.if_then_else(q_hi == k_hi, 1, 0)
                        clean_boundary = T.if_then_else(
                            bx < region_tiles,
                            T.if_then_else(q_hi > k_hi, 1, 0),
                            T.if_then_else(q_hi >= k_hi, 1, 0),
                        )
                        allowed = T.if_then_else(is_noisy_diag == 1, noisy_allowed, clean_boundary)
                        qk[i, j] = T.if_then_else(allowed > 0, qk[i, j], 0)

                T.clear(ds)
                T.gemm(do, v_shared, ds, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block, block):
                    ds_cast[i, j] = qk[i, j] * (ds[i, j] - delta[i]) * sm_scale
                T.gemm(ds_cast, k_shared, dq, policy=T.GemmWarpPolicy.FullRow)

            T.copy(dq, dq_shared)
            T.copy(dq_shared, dQ[bz, bx * block : (bx + 1) * block, by, :])

    return dq_kernel


def _bwd_dkv_template(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    mask_block: int,
    softmax_scale: float,
    block: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    assert block == 64 and mask_block == 32, "split dK/dV is specialized for 64x64 tiles with mask_block=32"
    half_len = seq_len // 2
    region_tiles = half_len // block
    sm_scale = softmax_scale
    scale_log2e = sm_scale * LOG2_E
    accum_dtype = "float32"
    shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def dkv_kernel(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dK: T.Tensor(shape, dtype),
        dV: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block), batch, threads=threads) as (bx, by, bz):
            k_shared = T.alloc_shared([block, dim], dtype)
            v_shared = T.alloc_shared([block, dim], dtype)
            q = T.alloc_shared([block, dim], dtype)
            do = T.alloc_shared([block, dim], dtype)
            lse = T.alloc_shared([block], accum_dtype)
            delta = T.alloc_shared([block], accum_dtype)
            qkT = T.alloc_fragment([block, block], accum_dtype)
            dsT = T.alloc_fragment([block, block], accum_dtype)
            qkT_cast = T.alloc_fragment([block, block], dtype)
            dsT_cast = T.alloc_fragment([block, block], dtype)
            dv = T.alloc_fragment([block, dim], accum_dtype)
            dk = T.alloc_fragment([block, dim], accum_dtype)
            dv_shared = T.alloc_shared([block, dim], dtype)
            dk_shared = T.alloc_shared([block, dim], dtype)

            T.copy(K[bz, by * block : (by + 1) * block, bx, :], k_shared)
            T.copy(V[bz, by * block : (by + 1) * block, bx, :], v_shared)
            T.clear(dv)
            T.clear(dk)

            first_q_tile = T.if_then_else(by < region_tiles, by, by - region_tiles)
            clean_span = region_tiles - first_q_tile
            loop_ed = T.if_then_else(by < region_tiles, 1, clean_span * 2)
            for step in T.Pipelined(loop_ed, num_stages=num_stages):
                clean_k = first_q_tile + T.if_then_else(
                    step < clean_span,
                    step,
                    region_tiles + (step - clean_span),
                )
                q_tile = T.if_then_else(by < region_tiles, first_q_tile + step, clean_k)

                T.copy(Q[bz, q_tile * block : (q_tile + 1) * block, bx, :], q)
                T.clear(qkT)
                T.gemm(k_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(LSE[bz, bx, q_tile * block : (q_tile + 1) * block], lse)
                for i, j in T.Parallel(block, block):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale_log2e - lse[j])

                needs_mask = T.if_then_else(by < region_tiles, 1, 0) + T.if_then_else(
                    by >= region_tiles,
                    T.if_then_else(q_tile == first_q_tile, 1, 0)
                    + T.if_then_else(q_tile == region_tiles + first_q_tile, 1, 0),
                    0,
                )
                if needs_mask != 0:
                    for i, j in T.Parallel(block, block):
                        allowed = _bwd_allowed_64x64_mask32(q_tile, by, j, i, region_tiles)
                        qkT[i, j] = T.if_then_else(allowed > 0, qkT[i, j], 0)

                T.copy(dO[bz, q_tile * block : (q_tile + 1) * block, bx, :], do)
                T.clear(dsT)
                T.gemm(v_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, q_tile * block : (q_tile + 1) * block], delta)
                for i, j in T.Parallel(block, block):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block : (by + 1) * block, bx, :])
            T.copy(dk_shared, dK[bz, by * block : (by + 1) * block, bx, :])

    return dkv_kernel


_build_fwd = tilelang.jit(out_idx=[3, 4], pass_configs=_FWD_PASS_CFG)(_fwd_template)
_build_bwd_dq = tilelang.jit(pass_configs=_DQ_PASS_CFG)(_bwd_dq_template)
_build_bwd_dkv = tilelang.jit(pass_configs=_PASS_CFG)(_bwd_dkv_template)

_TL_DTYPE = {torch.bfloat16: "bfloat16", torch.float16: "float16"}
_FWD_CACHE = {}
_BWD_CACHE = {}


def _tl_dtype(tensor: torch.Tensor) -> str:
    if tensor.dtype not in _TL_DTYPE:
        raise TypeError(f"unsupported dtype {tensor.dtype}; use bfloat16 or float16")
    return _TL_DTYPE[tensor.dtype]


def get_fwd_kernel(
    batch,
    heads,
    seq_len,
    dim,
    mask_block,
    softmax_scale,
    dtype,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
):
    key = (batch, heads, seq_len, dim, mask_block, softmax_scale, dtype, block_M, block_N, num_stages, threads, _FAST_MATH)
    if key not in _FWD_CACHE:
        _FWD_CACHE[key] = _build_fwd(
            batch,
            heads,
            seq_len,
            dim,
            mask_block,
            softmax_scale,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
            dtype=dtype,
        )
    return _FWD_CACHE[key]


def get_bwd_kernels(batch, heads, seq_len, dim, mask_block, softmax_scale, dtype, block=64, num_stages=3, threads=128):
    key = (batch, heads, seq_len, dim, mask_block, softmax_scale, dtype, block, num_stages, threads, _FAST_MATH)
    if key not in _BWD_CACHE:
        _BWD_CACHE[key] = (
            _build_bwd_preprocess(batch, heads, seq_len, dim, dtype=dtype),
            _build_bwd_dq(
                batch,
                heads,
                seq_len,
                dim,
                mask_block,
                softmax_scale,
                block=block,
                num_stages=num_stages,
                threads=threads,
                dtype=dtype,
            ),
            _build_bwd_dkv(
                batch,
                heads,
                seq_len,
                dim,
                mask_block,
                softmax_scale,
                block=block,
                num_stages=num_stages,
                threads=threads,
                dtype=dtype,
            ),
        )
    return _BWD_CACHE[key]


class _BlockCausalAttentionTL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask_block, softmax_scale):
        batch, seq_len, heads, dim = q.shape
        dtype = _tl_dtype(q)
        fwd = get_fwd_kernel(batch, heads, seq_len, dim, mask_block, softmax_scale, dtype)
        o, lse = fwd(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.mask_block = mask_block
        ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        batch, seq_len, heads, dim = q.shape
        dtype = _tl_dtype(q)

        def contig(tensor):
            return tensor if tensor.stride(-1) == 1 else tensor.contiguous()

        do, q, k, v, o = (contig(tensor) for tensor in (do, q, k, v, o))
        prep, dq_kernel, dkv_kernel = get_bwd_kernels(batch, heads, seq_len, dim, ctx.mask_block, ctx.softmax_scale, dtype)
        delta = prep(o, do)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq_kernel(q, k, v, do, lse, delta, dq)
        dkv_kernel(q, k, v, do, lse, delta, dk, dv)
        return dq, dk, dv, None, None


def block_causal_attention(query, key, value, mask_block_size: int, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return _BlockCausalAttentionTL.apply(query, key, value, mask_block_size, float(softmax_scale))


def block_causal_attention_ref(query, key, value, mask_block_size: int, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    seq_len = query.shape[1]
    half_len = seq_len // 2
    pos = torch.arange(seq_len, device=query.device)
    q_abs = pos[:, None]
    k_abs = pos[None, :]
    q_clean = q_abs >= half_len
    k_clean = k_abs >= half_len
    q_local = torch.where(q_clean, q_abs - half_len, q_abs)
    k_local = torch.where(k_clean, k_abs - half_len, k_abs)
    q_block = q_local // mask_block_size
    k_block = k_local // mask_block_size

    same_region = q_clean == k_clean
    block_diagonal = (q_block == k_block) & same_region
    offset_causal = (q_block > k_block) & ~q_clean & k_clean
    clean_causal = (q_block >= k_block) & q_clean & k_clean
    attn_mask = block_diagonal | offset_causal | clean_causal

    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * softmax_scale
    scores = scores.masked_fill(~attn_mask[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probs, value.float())
    return output.to(query.dtype)


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_(True)


def test_block_causal_attention_forward_backward():
    batch, seq_len, heads, dim = 2, 256, 2, 64
    mask_block = 32
    dtype = torch.float16
    torch.manual_seed(0)

    query = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    grad = torch.randn_like(query)

    q_ref = _clone_with_grad(query)
    k_ref = _clone_with_grad(key)
    v_ref = _clone_with_grad(value)
    q_tl = _clone_with_grad(query)
    k_tl = _clone_with_grad(key)
    v_tl = _clone_with_grad(value)

    out_ref = block_causal_attention_ref(q_ref, k_ref, v_ref, mask_block)
    out_tl = block_causal_attention(q_tl, k_tl, v_tl, mask_block)

    torch.testing.assert_close(out_tl, out_ref, atol=2e-2, rtol=2e-2)

    out_ref.backward(grad)
    out_tl.backward(grad)

    torch.testing.assert_close(q_tl.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_tl.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v_tl.grad, v_ref.grad, atol=5e-2, rtol=5e-2)


def main():
    test_block_causal_attention_forward_backward()


def run_regression_perf():
    from tilelang.profiler import do_bench

    batch, seq_len, heads, dim = 2, 1024, 8, 64
    mask_block = 32
    dtype = torch.float16
    torch.manual_seed(0)
    query = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    def run_kernel_only():
        block_causal_attention(query, key, value, mask_block)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    main()
