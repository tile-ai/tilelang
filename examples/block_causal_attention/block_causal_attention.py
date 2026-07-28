import torch
import tilelang
import tilelang.language as T

LOG2_E = 1.4426950408889634
_SUPPORTED_DLLM_BLOCKS = (1, 2, 4, 8, 16, 32, 64)


def _check_dllm_block(dllm_block: int, block_size: int):
    assert dllm_block in _SUPPORTED_DLLM_BLOCKS, f"dllm_block must be one of {_SUPPORTED_DLLM_BLOCKS}, got {dllm_block}"
    assert block_size % dllm_block == 0, f"block ({block_size}) must be divisible by dllm_block ({dllm_block})"


# ---------------------------------------------------------------------------
# Tile-local mask helpers
# ---------------------------------------------------------------------------


def _fwd_tile_allowed(i, j, dllm_block: int, is_noisy_diag, q_is_noisy):
    """Mask for a single visited boundary forward tile.

    `i`/`j` are tile-local query/key rows. `is_noisy_diag` marks the noisy-noisy
    diagonal tile. Otherwise this is the clean boundary tile and `q_is_noisy`
    selects between offset_causal and block_causal masking.
    """
    q_blk = i // dllm_block
    k_blk = j // dllm_block
    noisy_allowed = T.if_then_else(q_blk == k_blk, 1, 0)
    clean_boundary = T.if_then_else(
        q_is_noisy,
        T.if_then_else(q_blk > k_blk, 1, 0),
        T.if_then_else(q_blk >= k_blk, 1, 0),
    )
    return T.if_then_else(is_noisy_diag == 1, noisy_allowed, clean_boundary)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[3, 4])
def _fwd_template(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    dllm_block: int,
    softmax_scale: float,
    block_size: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    assert seq_len % 2 == 0, "seq_len must be noisy|clean halves"
    half_len = seq_len // 2
    assert half_len % block_size == 0, "half_len must be divisible by block_size"
    _check_dllm_block(dllm_block, block_size)
    region_tiles = half_len // block_size

    scale_log2e = softmax_scale * LOG2_E
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
        with T.Kernel(T.ceildiv(seq_len, block_size), heads, batch, threads=threads) as (bx, by, bz):
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

            T.copy(Q[bz, bx * block_size : (bx + 1) * block_size, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, neg_inf)

            q_is_noisy = bx < region_tiles
            last_clean_step = T.if_then_else(q_is_noisy, bx, bx - region_tiles)
            num_steps = T.if_then_else(q_is_noisy, bx + 2, (bx - region_tiles) + 1)

            for s in T.Pipelined(num_steps, num_stages=num_stages):
                is_noisy_diag = T.if_then_else(s > last_clean_step, 1, 0)
                if is_noisy_diag != 0:
                    T.copy(K[bz, bx * block_size : (bx + 1) * block_size, by, :], K_shared)
                else:
                    T.copy(K[bz, half_len + s * block_size : half_len + (s + 1) * block_size, by, :], K_shared)

                is_clean_diag = T.if_then_else(s == last_clean_step, 1, 0)
                needs_mask = is_noisy_diag + is_clean_diag
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
                    T.copy(V[bz, bx * block_size : (bx + 1) * block_size, by, :], V_shared)
                else:
                    T.copy(V[bz, half_len + s * block_size : half_len + (s + 1) * block_size, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_size, dim):
                acc_o[i, j] /= logsum[i] + 1e-30
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_size : (bx + 1) * block_size, by, :])
            for i in T.Parallel(block_size):
                logsum[i] = T.log2(logsum[i] + 1e-30) + scores_max[i] * scale_log2e
            T.copy(logsum, LSE[bz, by, bx * block_size : (bx + 1) * block_size])

    return fwd


@tilelang.jit(out_idx=[2])
def _bwd_preprocess_template(batch: int, heads: int, seq_len: int, dim: int, dtype: str = "bfloat16"):
    accum_dtype = "float32"
    shape = [batch, seq_len, heads, dim]
    block_size = 64
    # prep sweeps dim in 64-wide tiles; a non-multiple-of-64 tail would copy past the last column.
    assert dim % block_size == 0, f"Delta preprocessing requires dim divisible by {block_size}, got {dim}"

    @T.prim_func
    def prep(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_size), batch) as (bx, by, bz):
            o = T.alloc_fragment([block_size, block_size], dtype)
            do = T.alloc_fragment([block_size, block_size], dtype)
            acc = T.alloc_fragment([block_size, block_size], accum_dtype)
            delta = T.alloc_fragment([block_size], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, block_size)):
                T.copy(O[bz, by * block_size : (by + 1) * block_size, bx, k * block_size : (k + 1) * block_size], o)
                T.copy(dO[bz, by * block_size : (by + 1) * block_size, bx, k * block_size : (k + 1) * block_size], do)
                for i, j in T.Parallel(block_size, block_size):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * block_size : (by + 1) * block_size])

    return prep


@tilelang.jit
def _bwd_dq_template(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    dllm_block: int,
    softmax_scale: float,
    block_size: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    """Query-parallel dQ.

    Reuses the forward sparse schedule and the forward tile mask,
    accumulates `dq` in registers and writes once (atomics-free)."""
    half_len = seq_len // 2
    assert half_len % block_size == 0, "half_len must be divisible by block_size"
    _check_dllm_block(dllm_block, block_size)
    region_tiles = half_len // block_size
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
        with T.Kernel(T.ceildiv(seq_len, block_size), heads, batch, threads=threads) as (bx, by, bz):
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

            T.copy(Q[bz, bx * block_size : (bx + 1) * block_size, by, :], q)
            T.copy(dO[bz, bx * block_size : (bx + 1) * block_size, by, :], do)
            T.copy(LSE[bz, by, bx * block_size : (bx + 1) * block_size], lse)
            T.copy(Delta[bz, by, bx * block_size : (bx + 1) * block_size], delta)
            T.clear(dq)

            q_is_noisy = bx < region_tiles
            last_clean_step = T.if_then_else(q_is_noisy, bx, bx - region_tiles)
            num_steps = T.if_then_else(q_is_noisy, bx + 2, (bx - region_tiles) + 1)
            for s in T.Pipelined(num_steps, num_stages=num_stages):
                is_noisy_diag = T.if_then_else(s > last_clean_step, 1, 0)
                if is_noisy_diag != 0:
                    T.copy(K[bz, bx * block_size : (bx + 1) * block_size, by, :], k_shared)
                    T.copy(V[bz, bx * block_size : (bx + 1) * block_size, by, :], v_shared)
                else:
                    T.copy(K[bz, half_len + s * block_size : half_len + (s + 1) * block_size, by, :], k_shared)
                    T.copy(V[bz, half_len + s * block_size : half_len + (s + 1) * block_size, by, :], v_shared)

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
            T.copy(dq_shared, dQ[bz, bx * block_size : (bx + 1) * block_size, by, :])

    return dq_kernel


@tilelang.jit
def _bwd_dkv_template(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    dllm_block: int,
    softmax_scale: float,
    block_size: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "bfloat16",
):
    """Key-parallel dK/dV.

    The reverse of the forward schedule -- for each key tile, walk
    exactly the query tiles that attend it, accumulate `dk`/`dv` in
    registers and write once (atomics-free)."""
    half_len = seq_len // 2
    assert half_len % block_size == 0, "half_len must be divisible by block_size"
    _check_dllm_block(dllm_block, block_size)
    region_tiles = half_len // block_size
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
        with T.Kernel(heads, T.ceildiv(seq_len, block_size), batch, threads=threads) as (bx, by, bz):
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

            T.copy(K[bz, by * block_size : (by + 1) * block_size, bx, :], k_shared)
            T.copy(V[bz, by * block_size : (by + 1) * block_size, bx, :], v_shared)
            T.clear(dv)
            T.clear(dk)

            key_is_noisy = by < region_tiles
            first_q_tile = T.if_then_else(key_is_noisy, by, by - region_tiles)
            clean_span = region_tiles - first_q_tile
            loop_ed = T.if_then_else(key_is_noisy, 1, clean_span * 2)
            for step in T.Pipelined(loop_ed, num_stages=num_stages):
                clean_k = first_q_tile + T.if_then_else(step < clean_span, step, region_tiles + (step - clean_span))
                q_tile = T.if_then_else(key_is_noisy, first_q_tile + step, clean_k)

                T.copy(Q[bz, q_tile * block_size : (q_tile + 1) * block_size, bx, :], q)
                T.clear(qkT)
                T.gemm(k_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(LSE[bz, bx, q_tile * block_size : (q_tile + 1) * block_size], lse)
                for i, j in T.Parallel(block_size, block_size):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale_log2e - lse[j])

                q_is_noisy = q_tile < region_tiles
                needs_mask = T.if_then_else(key_is_noisy, 1, 0) + T.if_then_else(
                    key_is_noisy,
                    0,
                    T.if_then_else(q_tile == first_q_tile, 1, 0) + T.if_then_else(q_tile == region_tiles + first_q_tile, 1, 0),
                )
                if needs_mask != 0:
                    for i, j in T.Parallel(block_size, block_size):
                        allowed = _fwd_tile_allowed(j, i, dllm_block, T.if_then_else(key_is_noisy, 1, 0), q_is_noisy)
                        qkT[i, j] = T.if_then_else(allowed > 0, qkT[i, j], 0)

                T.copy(dO[bz, q_tile * block_size : (q_tile + 1) * block_size, bx, :], do)
                T.clear(dsT)
                T.gemm(v_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, q_tile * block_size : (q_tile + 1) * block_size], delta)
                for i, j in T.Parallel(block_size, block_size):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_size : (by + 1) * block_size, bx, :])
            T.copy(dk_shared, dK[bz, by * block_size : (by + 1) * block_size, bx, :])

    return dkv_kernel


class _BlockCausalAttentionTL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dllm_block, softmax_scale):
        batch, seq_len, heads, dim = q.shape
        dtype = T.dtype(q.dtype)
        q, k, v = (t.contiguous() for t in (q, k, v))
        fwd = _fwd_template(batch, heads, seq_len, dim, dllm_block, softmax_scale, dtype=dtype)
        o, lse = fwd(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.dllm_block = dllm_block
        ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        batch, seq_len, heads, dim = q.shape
        dtype = T.dtype(q.dtype)

        do, q, k, v, o = (t.contiguous() for t in (do, q, k, v, o))
        prep = _bwd_preprocess_template(batch, heads, seq_len, dim, dtype=dtype)
        dq_kernel = _bwd_dq_template(batch, heads, seq_len, dim, ctx.dllm_block, ctx.softmax_scale, dtype=dtype)
        dkv_kernel = _bwd_dkv_template(batch, heads, seq_len, dim, ctx.dllm_block, ctx.softmax_scale, dtype=dtype)
        delta = prep(o, do)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq_kernel(q, k, v, do, lse, delta, dq)
        dkv_kernel(q, k, v, do, lse, delta, dk, dv)
        return dq, dk, dv, None, None


def block_causal_attention(query, key, value, dllm_block_size: int, softmax_scale=None):
    """Fixed-length dLLM block-causal attention for any `dllm_block_size`."""
    _check_dllm_block(dllm_block_size, 64)  # fail fast at the API, not deep in a JIT build
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return _BlockCausalAttentionTL.apply(query, key, value, dllm_block_size, float(softmax_scale))


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------


def _dllm_mask(seq_len: int, dllm_block_size: int, device) -> torch.Tensor:
    half_len = seq_len // 2
    pos = torch.arange(seq_len, device=device)
    q_abs = pos[:, None]
    k_abs = pos[None, :]
    q_clean = q_abs >= half_len
    k_clean = k_abs >= half_len
    q_local = torch.where(q_clean, q_abs - half_len, q_abs)
    k_local = torch.where(k_clean, k_abs - half_len, k_abs)
    q_block = q_local // dllm_block_size
    k_block = k_local // dllm_block_size

    block_diagonal = (q_block == k_block) & (~q_clean) & (~k_clean)
    offset_causal = (q_block > k_block) & ~q_clean & k_clean
    clean_causal = (q_block >= k_block) & q_clean & k_clean
    return block_diagonal | offset_causal | clean_causal


def block_causal_attention_ref(query, key, value, dllm_block_size: int, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    seq_len = query.shape[1]
    attn_mask = _dllm_mask(seq_len, dllm_block_size, query.device)
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * softmax_scale
    scores = scores.masked_fill(~attn_mask[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probs, value.float())
    return output.to(query.dtype)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _run_fixed_case(batch, seq_len, heads, dim, dllm_block, dtype=torch.float16):
    torch.manual_seed(0)
    query = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    grad = torch.randn_like(query)

    q_ref, k_ref, v_ref = (_clone_with_grad(t) for t in (query, key, value))
    q_tl, k_tl, v_tl = (_clone_with_grad(t) for t in (query, key, value))

    out_ref = block_causal_attention_ref(q_ref, k_ref, v_ref, dllm_block)
    out_tl = block_causal_attention(q_tl, k_tl, v_tl, dllm_block)
    torch.testing.assert_close(out_tl, out_ref, atol=2e-2, rtol=2e-2)

    out_ref.backward(grad)
    out_tl.backward(grad)
    torch.testing.assert_close(q_tl.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_tl.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v_tl.grad, v_ref.grad, atol=5e-2, rtol=5e-2)


def test_block_causal_attention_all_block_sizes():
    for dllm_block in _SUPPORTED_DLLM_BLOCKS:
        _run_fixed_case(2, 256, 2, 64, dllm_block)
        print(f"[fixed]  dllm_block={dllm_block:>2} OK")


def main():
    test_block_causal_attention_all_block_sizes()


if __name__ == "__main__":
    main()
