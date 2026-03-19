# Adapted from tilelang/examples/flash_attention/example_gqa_bwd.py

import torch
import tilelang
from tilelang.profiler import do_bench
import tilelang.language as T
import argparse
from typing import Optional


def get_bwd_configs():
    sm_major, sm_minor = torch.cuda.get_device_capability()
    sm_version = sm_major * 10 + sm_minor
    if sm_version == 80:
        return 64, 32, 1, 128
    else:
        return 128, 32, 2, 256


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_fwd(
    Q,
    K,
    V,
    Sinks,
    groups: int = 1,
    window_size=None,  # None for full attention
    sm_scale=None,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
):
    batch, heads, seq_len, dim = T.const("batch heads seq_len dim")
    _, head_kv, _, _ = T.const("_ head_kv _ _")
    Q: T.Tensor[[batch, heads, seq_len, dim], T.float16]
    K: T.Tensor[[batch, head_kv, seq_len, dim], T.float16]
    V: T.Tensor[[batch, head_kv, seq_len, dim], T.float16]
    Sinks: T.Tensor[[heads], T.float16]
    Output = T.empty([batch, heads, seq_len, dim], T.float16)
    lse = T.empty([batch, heads, seq_len], T.float32)

    dtype = T.float16
    accum_dtype = T.float32

    if window_size is not None:
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)

    with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
        sinks = T.alloc_fragment([heads], dtype)

        T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))
        for i in T.Parallel(block_M):
            sinks[i] = Sinks[by]

        end = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
        start = T.max(0, (bx * block_M - window_size) // block_N) if window_size is not None else 0

        for k in T.Pipelined(start, end, num_stages=num_stages):
            T.copy(K[bz, by // groups, k * block_N : (k + 1) * block_N, :], K_shared)
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i
                k_idx = k * block_N + j
                if window_size is not None:
                    acc_s[i, j] = T.if_then_else(q_idx >= k_idx and q_idx < k_idx + window_size, 0, -T.infinity(acc_s.dtype))
                else:
                    acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            T.copy(V[bz, by // groups, k * block_N : (k + 1) * block_N, :], V_shared)
            T.copy(scores_max, scores_max_prev)
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # NOTE(wt): check_inf is necessary for sliding window attention.
            for i in T.Parallel(block_M):
                if window_size is not None:
                    scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
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

        for i in T.Parallel(block_M):
            logsum[i] += T.exp2(sinks[i] * 1.44269504 - scores_max[i] * scale)  # The only change for attention sink
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] /= logsum[i]
        T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])
        for i in T.Parallel(block_M):
            logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
        T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return Output, lse


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_preprocess(O, dO):
    batch, heads, seq_len, dim = T.const("batch heads seq_len dim")
    O: T.Tensor[[batch, heads, seq_len, dim], T.float16]
    dO: T.Tensor[[batch, heads, seq_len, dim], T.float16]
    Delta = T.empty([batch, heads, seq_len], T.float32)

    dtype = T.float16
    accum_dtype = T.float32
    blk = 32

    with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
        o = T.alloc_fragment([blk, blk], dtype)
        do = T.alloc_fragment([blk, blk], dtype)
        acc = T.alloc_fragment([blk, blk], accum_dtype)
        delta = T.alloc_fragment([blk], accum_dtype)
        T.clear(acc)
        for k in range(T.ceildiv(dim, blk)):
            T.copy(O[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk], o)
            T.copy(dO[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk], do)
            for i, j in T.Parallel(blk, blk):
                acc[i, j] += o[i, j] * do[i, j]
        T.reduce_sum(acc, delta, 1)
        T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return Delta


def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(dQ.shape, lambda b, h, l, d: [b, h, l // 8, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_postprocess(dQ):
    batch, heads, seq_len, dim = T.const("batch heads seq_len dim")
    dQ: T.Tensor[[batch, heads, seq_len, dim], T.float32]
    dQ_out = T.empty([batch, heads, seq_len, dim], T.float16)

    blk = 64

    with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
        T.annotate_layout({dQ: make_dq_layout(dQ)})
        T.copy(
            dQ[bz, by, bx * blk : (bx + 1) * blk, :],
            dQ_out[bz, by, bx * blk : (bx + 1) * blk, :],
        )

    return dQ_out


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def flashattn_bwd(
    Q,
    K,
    V,
    dO,
    lse,
    Delta,
    dQ,
    dK,
    dV,
    groups: int = 1,
    window_size=None,  # None for full attention
    sm_scale=None,
):
    batch, heads, seq_len, dim = T.const("batch heads seq_len dim")
    _, head_kv, _, _ = T.const("_ head_kv _ _")
    Q: T.Tensor[[batch, heads, seq_len, dim], T.float16]
    K: T.Tensor[[batch, head_kv, seq_len, dim], T.float16]
    V: T.Tensor[[batch, head_kv, seq_len, dim], T.float16]
    dO: T.Tensor[[batch, heads, seq_len, dim], T.float16]
    lse: T.Tensor[[batch, heads, seq_len], T.float32]
    Delta: T.Tensor[[batch, heads, seq_len], T.float32]
    dQ: T.Tensor[[batch, heads, seq_len, dim], T.float32]
    dK: T.Tensor[[batch, head_kv, seq_len, dim], T.float32]
    dV: T.Tensor[[batch, head_kv, seq_len, dim], T.float32]

    block_M, block_N, num_stages, threads = get_bwd_configs()

    dtype = T.float16
    accum_dtype = T.float32

    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)

    if window_size is not None:
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
        K_shared = T.alloc_shared([block_M, dim], dtype)
        dsT_shared = T.alloc_shared([block_M, block_N], dtype)
        q = T.alloc_shared([block_N, dim], dtype)
        V_shared = T.alloc_shared([block_M, dim], dtype)
        qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
        dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
        qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
        dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
        lse_shared = T.alloc_shared([block_N], accum_dtype)
        delta_local = T.alloc_shared([block_N], accum_dtype)
        do = T.alloc_shared([block_N, dim], dtype)
        dv = T.alloc_fragment([block_M, dim], accum_dtype)
        dk = T.alloc_fragment([block_M, dim], accum_dtype)
        dq = T.alloc_fragment([block_N, dim], accum_dtype)
        dv_shared = T.alloc_shared([block_M, dim], accum_dtype)
        dk_shared = T.alloc_shared([block_M, dim], accum_dtype)

        T.annotate_layout(
            {
                dQ: make_dq_layout(dQ),
            }
        )
        T.copy(K[bz, bx // groups, by * block_M : (by + 1) * block_M, :], K_shared)
        T.copy(V[bz, bx // groups, by * block_M : (by + 1) * block_M, :], V_shared)
        T.clear(dv)
        T.clear(dk)

        loop_st = T.floordiv(by * block_M, block_N)
        loop_ed = (
            T.min(T.ceildiv((by + 1) * block_M + window_size, block_N), T.ceildiv(seq_len, block_N))
            if window_size is not None
            else T.ceildiv(seq_len, block_N)
        )

        for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
            T.copy(Q[bz, bx, k * block_N : (k + 1) * block_N, :], q)
            T.clear(qkT)
            T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
            for i, j in T.Parallel(block_M, block_N):
                qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
            for i, j in T.Parallel(block_M, block_N):
                if window_size is not None:
                    qkT[i, j] = T.if_then_else(
                        by * block_M + i <= k * block_N + j and by * block_M + i > k * block_N + j - window_size, qkT[i, j], 0
                    )
                else:
                    qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j], 0)
            T.copy(dO[bz, bx, k * block_N : (k + 1) * block_N, :], dst=do)
            T.clear(dsT)
            T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(qkT, qkT_cast)
            T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

            T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta_local)

            for i, j in T.Parallel(block_M, block_N):
                dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta_local[j]) * sm_scale
            T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

            T.copy(dsT_cast, dsT_shared)
            T.clear(dq)
            T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
            T.atomic_add(dQ[bz, bx, k * block_N : (k + 1) * block_N, :], dq)

        T.copy(dv, dv_shared)
        T.atomic_add(dV[bz, bx // groups, by * block_M : (by + 1) * block_M, :], dv_shared)
        T.copy(dk, dk_shared)
        T.atomic_add(dK[bz, bx // groups, by * block_M : (by + 1) * block_M, :], dk_shared)


@tilelang.jit()
def flashattn_bwd_dsink(Sinks, Delta, lse, block: int = 256):
    heads = T.const("heads")
    batch, _, seq_len = T.const("batch _ seq_len")
    Sinks: T.Tensor[[heads], T.float16]
    Delta: T.Tensor[[batch, heads, seq_len], T.float32]
    lse: T.Tensor[[batch, heads, seq_len], T.float32]
    dsinks = T.empty([batch, heads, seq_len], T.float16)

    accum_dtype = T.float32
    dtype = T.float16

    with T.Kernel(heads, T.ceildiv(seq_len, block), batch, threads=256) as (bx, by, bz):
        lse_fragment = T.alloc_fragment([block], accum_dtype)
        delta_fragment = T.alloc_fragment([block], accum_dtype)
        dsink_fragment = T.alloc_fragment([block], dtype)

        sink = Sinks[bx]
        T.copy(lse[bz, bx, by * block : (by + 1) * block], lse_fragment)
        T.copy(Delta[bz, bx, by * block : (by + 1) * block], delta_fragment)
        for i in T.Parallel(block):
            dsink_fragment[i] = -T.exp2(sink * 1.44269504 - lse_fragment[i]) * delta_fragment[i]
        T.copy(dsink_fragment, dsinks[bz, bx, by * block : (by + 1) * block])

    return dsinks


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, window_size, groups):
        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        q, k, v, sinks = [maybe_contiguous(x) for x in (q, k, v, sinks)]
        o, lse = flashattn_fwd(q, k, v, sinks, groups, window_size)
        ctx.save_for_backward(q, k, v, sinks, o, lse)
        ctx.window_size = window_size
        ctx.groups = groups
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, o, lse = ctx.saved_tensors
        BATCH, H, N_CTX, D_HEAD = q.shape
        groups = ctx.groups

        delta = flashattn_bwd_preprocess(o, do)
        q_shape = [BATCH, H, N_CTX, D_HEAD]
        head_kv = H // groups
        kv_shape = [BATCH, head_kv, N_CTX, D_HEAD]
        dq = torch.zeros(q_shape, dtype=torch.float32, device=q.device)  # acc for atomicAdd
        dk = torch.zeros(kv_shape, dtype=torch.float32, device=q.device)
        dv = torch.zeros(kv_shape, dtype=torch.float32, device=q.device)
        flashattn_bwd(q, k, v, do, lse, delta, dq, dk, dv, groups, ctx.window_size)
        dq = flashattn_bwd_postprocess(dq)

        dsinks = flashattn_bwd_dsink(sinks, delta, lse).sum(0).sum(1)
        return dq, dk, dv, dsinks, None, None


attention = _attention.apply


# Adapted and optimized from
# https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py
def ref_program(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sliding_window: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape
    query = query.transpose(1, 2).contiguous()
    query = query.view(batch_size, query.shape[1], num_key_value_heads, -1, head_dim)
    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape

    start_q = num_keys - num_queries
    sm_scale: float = 1.0 / head_dim**0.5

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups, head_dim).to(dtype)
    return output.transpose(1, 2).contiguous()


def main(
    BATCH: int = 1,
    H: int = 8,
    N_CTX: int = 512,
    D_HEAD: int = 64,
    groups: int = 2,
    window_size: Optional[int] = None,
    dtype: str = "float16",
):
    dtype = T.dtype(dtype)
    torch_dtype = dtype.as_torch()
    if window_size is not None:
        print("Using sliding window attention.")
        assert window_size <= N_CTX
        flops_per_matmul = 2.0 * BATCH * H * min(window_size, N_CTX // 2) * N_CTX * D_HEAD  # just a rough estimation
    else:
        print("Using full attention.")
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD * 0.5
    total_flops = 5 * flops_per_matmul

    Q = torch.randn(BATCH, H, N_CTX, D_HEAD, dtype=torch_dtype, device="cuda").requires_grad_()
    K = torch.randn(BATCH, H // groups, N_CTX, D_HEAD, dtype=torch_dtype, device="cuda").requires_grad_()
    V = torch.randn_like(K).requires_grad_()
    sinks = torch.randn(H, dtype=torch_dtype, device="cuda").requires_grad_()
    dO = torch.randn_like(Q)

    O = attention(Q, K, V, sinks, window_size, groups)
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None
    dsinks, sinks.grad = sinks.grad.clone(), None

    O_ref = ref_program(Q, K, V, sinks, window_size, dtype=torch_dtype)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None
    dsinks_ref, sinks.grad = sinks.grad.clone(), None

    # Checks
    rtol, atol = {
        T.float16: (1e-2, 1e-2),
        T.bfloat16: (2e-2, 2e-2),
    }[dtype]
    assert torch.allclose(O, O_ref, rtol=rtol, atol=atol), f"O max err: {(O - O_ref).abs().max()}"
    assert torch.allclose(dV, dV_ref, rtol=rtol, atol=atol), f"dV max err: {(dV - dV_ref).abs().max()}"
    assert torch.allclose(dK, dK_ref, rtol=rtol, atol=atol), f"dK max err: {(dK - dK_ref).abs().max()}"
    assert torch.allclose(dQ, dQ_ref, rtol=rtol, atol=atol), f"dq max err: {(dQ - dQ_ref).abs().max()}"
    assert torch.allclose(dsinks, dsinks_ref, rtol=rtol, atol=atol), f"dsinks max err: {(dsinks - dsinks_ref).abs().max()}"

    print("All checks passed for tilelang kernels.✅")

    # Only benchmark backward here
    def torch_bwd():
        O_ref.backward(dO, retain_graph=True)

    def tl_bwd():
        O.backward(dO, retain_graph=True)

    latency = do_bench(torch_bwd, warmup=500)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(tl_bwd, warmup=500)
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


def run_regression_perf(
    BATCH: int = 1,
    H: int = 8,
    N_CTX: int = 512,
    D_HEAD: int = 64,
    groups: int = 2,
    window_size: Optional[int] = None,
    dtype: str = "float16",
):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    with torch.no_grad():
        Q = torch.randn(BATCH, H, N_CTX, D_HEAD, dtype=torch_dtype, device="cuda")
        K = torch.randn(BATCH, H // groups, N_CTX, D_HEAD, dtype=torch_dtype, device="cuda")
        V = torch.randn_like(K)
        sinks = torch.randn(H, dtype=torch_dtype, device="cuda")
        dO = torch.randn_like(Q)
        O, lse = flashattn_fwd(Q, K, V, sinks, groups, window_size)

        def maybe_contiguous(x):
            return x if x.stride(-1) == 1 else x.contiguous()

        do, q, k, v, sinks_c, o = [maybe_contiguous(x) for x in (dO, Q, K, V, sinks, O)]
        Delta = flashattn_bwd_preprocess(o, do)
        q_shape = (BATCH, H, N_CTX, D_HEAD)
        head_kv = H // groups
        kv_shape = (BATCH, head_kv, N_CTX, D_HEAD)
        dq = torch.zeros(q_shape, dtype=torch.float32, device="cuda")
        dk = torch.zeros(kv_shape, dtype=torch.float32, device="cuda")
        dv = torch.zeros(kv_shape, dtype=torch.float32, device="cuda")
        flashattn_bwd(q, k, v, do, lse, Delta, dq, dk, dv, groups, window_size)
        _ = flashattn_bwd_dsink(sinks_c, Delta, lse).sum(0).sum(1)

        def run_kernel_only():
            flashattn_bwd(q, k, v, do, lse, Delta, dq, dk, dv, groups, window_size)

        latency_ms = do_bench(run_kernel_only, backend="cupti")
        return latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--h", type=int, default=64, help="Number of heads")
    parser.add_argument("--n_ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--d_head", type=int, default=128, help="Head dimension")
    parser.add_argument("--groups", type=int, default=8, help="Groups")
    parser.add_argument("--window_size", type=int, default=128, help="window size (default: None, which means full attention)")
    parser.add_argument("--dtype", type=str, default="float16", help="dtype, can be float16 or bfloat16")
    args = parser.parse_args()
    main(args.batch, args.h, args.n_ctx, args.d_head, args.groups, args.window_size, args.dtype)
