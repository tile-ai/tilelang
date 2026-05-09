"""Stage-1 kernel: prefill pool-MQA over pooled (blocked) K.

Input: fp8 Q ``[M, H, D]`` + fp8 BlockedK ``[Nb, D]`` + per-block f32 scale
``[Nb]`` + f32 Weights ``[M, H]`` + per-query ``cu_seqlen_blocked_ks/ke [M]``.

For each query ``m`` and pool block ``n`` in ``[cu_seqlen_blocked_ks[m],
cu_seqlen_blocked_ke[m])``:
  ``logits[m, n] = sum_h ReLU(Q[m, h] . BlockedK[n]) * BlockedKScale[n] * Weights[m, h]``

Out-of-range entries in the raw kernel output are undefined — caller should
zero-init the buffer or apply a separate mask kernel.
"""

import tilelang
from tilelang import language as T
from tilelang.profiler import do_bench
import torch

from tilelang_utils import prepare_ks_ke_from_cu_seqlens
from clean_and_maintain_logits import (
    clean_and_maintain_logits_interface,
    ref_clean_and_maintain_logits,
)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def pool_mqa_attn_return_logits_fp8(
    IndexQ,
    IndexBlockedK,
    IndexBlockedKScale,
    Logits,
    Weights,
    CuSeqLenBlockedKS,
    CuSeqLenBlockedKE,
    heads: int = 64,
    index_dim: int = 128,
    block_N: int = 256,
    num_stages: int = 3,
    threads: int = 512,
    block_Q: int = 0,
):
    # block_Q is the tile size for queries; `0` means "derive from heads".
    if block_Q == 0:
        block_Q = 128 // heads
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len, seq_len_blocked_kv = T.const("seq_len, seq_len_blocked_kv")

    IndexQ: T.Tensor[[seq_len * heads, index_dim], fp8_dtype]
    IndexBlockedK: T.Tensor[[seq_len_blocked_kv, index_dim], fp8_dtype]
    IndexBlockedKScale: T.Tensor[[seq_len_blocked_kv], accum_dtype]
    Logits: T.Tensor[[seq_len, seq_len_blocked_kv], accum_dtype]
    Weights: T.Tensor[[seq_len, heads], accum_dtype]
    CuSeqLenBlockedKS: T.Tensor[[seq_len], index_dtype]
    CuSeqLenBlockedKE: T.Tensor[[seq_len], index_dtype]

    with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
        index_q_shared = T.alloc_shared([block_Q * heads, index_dim], fp8_dtype)
        index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
        index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
        s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
        s_reshaped = T.reshape(s, (block_N, block_Q, heads))
        logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
        weights = T.alloc_fragment([block_Q, heads], accum_dtype)

        seq_len_i = bx * block_Q

        cu_k_s_min = T.alloc_var(index_dtype)
        cu_k_e_max = T.alloc_var(index_dtype)
        cu_k_s_min = 2147483647
        cu_k_e_max = -2147483648

        for bq_i in T.serial(block_Q):
            cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenBlockedKS[seq_len_i + bq_i], seq_len_blocked_kv))
        for bq_i in T.serial(block_Q):
            cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenBlockedKE[seq_len_i + bq_i], seq_len_blocked_kv))

        T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
        T.copy(Weights[seq_len_i, 0], weights)

        for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
            T.copy(IndexBlockedK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
            T.copy(IndexBlockedKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

            T.gemm(
                index_k_shared,
                index_q_shared,
                s,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i] * index_k_scale_fragment[bn_i], 0) * weights[bq_i, h_i]

            T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

            for bq_i, bn_i in T.Parallel(block_Q, block_N):
                Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]


def pool_mqa_attn_return_logits_fp8_interface(
    q_fp8: torch.Tensor,
    blocked_kv_fp8: torch.Tensor,
    blocked_kv_scale: torch.Tensor,
    weights_f32: torch.Tensor,
    cu_seqlen_blocked_ks: torch.Tensor,
    cu_seqlen_blocked_ke: torch.Tensor,
    block_N: int = 256,
):
    """Raw kernel invocation; zero-inits logits so positions the kernel
    doesn't touch are 0 (matches the ref)."""
    seq_len, heads, index_dim = q_fp8.shape
    seq_len_blocked_kv = blocked_kv_fp8.shape[0]

    logits = torch.zeros([seq_len, seq_len_blocked_kv], device=q_fp8.device, dtype=torch.float32)
    pool_mqa_attn_return_logits_fp8(
        q_fp8.view(seq_len * heads, index_dim),
        blocked_kv_fp8,
        blocked_kv_scale,
        logits,
        weights_f32,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
        heads=heads,
        index_dim=index_dim,
        block_N=block_N,
    )
    return logits


def ref_pool_mqa_fp8(
    q_fp8: torch.Tensor,
    blocked_kv_fp8: torch.Tensor,
    blocked_kv_scale: torch.Tensor,
    weights_f32: torch.Tensor,
) -> torch.Tensor:
    """Spec: for each (m, n), logits[m, n] = sum_h ReLU(q[m,h] . k[n] * k_scale[n]) * w[m,h].
    Computes the full dense [M, Nb] grid — caller is responsible for any masking."""
    q_f = q_fp8.float()
    k_f = blocked_kv_fp8.float() * blocked_kv_scale[:, None]
    # score[m, n, h] = q[m, h] . k[n]
    s = torch.einsum("mhd,nd->mnh", q_f, k_f)  # [M, Nb, H]
    logits = (s.clamp(min=0) * weights_f32[:, None, :]).sum(dim=-1)  # [M, Nb]
    return logits


def test_pool_mqa_fp8(
    M: int = 32768,
    H: int = 64,
    D: int = 128,
    k_block_size: int = 128,
    block_N: int = 256,
    num_seqs: int = 1,
):
    """Correctness + speed test packing `num_seqs` equal-length causal
    sequences into the [M, H, D] Q tensor.

    Per-query ``cu_seqlen_blocked_ks/ke`` is derived from the raw-token
    packed ``cu_ks / cu_ke`` produced by ``prepare_ks_ke_from_cu_seqlens``
    (floor-divide / ceil-divide by ``k_block_size`` respectively).

    The kernel writes the per-tile ``[cu_k_s_min, cu_k_e_max)`` union of
    visible K ranges — entries inside this union but outside an
    individual query's visible range carry raw (unmasked) dot-product
    values. To make correctness well-defined, we apply the
    ``clean_and_maintain_logits`` mask (-inf for out-of-range, +inf for
    the first/last valid block) to both the kernel output and the torch
    reference before comparing — this mirrors what the hisa pipeline
    does right after this kernel.
    """
    torch.manual_seed(0)
    assert M % num_seqs == 0, f"M ({M}) must be divisible by num_seqs ({num_seqs})"
    per_seq = M // num_seqs
    N_blocked = (M + k_block_size - 1) // k_block_size
    assert N_blocked % block_N == 0, (
        f"N_blocked ({N_blocked}) must be a multiple of block_N ({block_N}). Pick M such that ceildiv(M, k_block_size) % block_N == 0."
    )

    # Per-token packed ks/ke (causal within each sequence), then translate
    # to pool-block coords.
    cu_seqlens = torch.arange(num_seqs + 1, device="cuda", dtype=torch.long) * per_seq
    ks_long, ke_long = prepare_ks_ke_from_cu_seqlens(cu_seqlens)
    cu_ks_token = ks_long.to(torch.int32).contiguous()
    cu_ke_token = ke_long.to(torch.int32).contiguous()
    cu_blocked_ks = (cu_ks_token // k_block_size).contiguous()
    cu_blocked_ke = ((cu_ke_token + k_block_size - 1) // k_block_size).contiguous()

    q_bf16 = torch.randn(M, H, D, device="cuda", dtype=torch.bfloat16)
    q = q_bf16.to(torch.float8_e4m3fn)
    blocked_k_bf16 = torch.randn(N_blocked, D, device="cuda", dtype=torch.bfloat16)
    blocked_k = blocked_k_bf16.to(torch.float8_e4m3fn)
    blocked_k_scale = (0.1 + 0.01 * torch.rand(N_blocked, device="cuda", dtype=torch.float32)).contiguous()
    weights = torch.randn(M, H, device="cuda", dtype=torch.float32)

    # Correctness — kernel + post-mask.
    got = pool_mqa_attn_return_logits_fp8_interface(
        q,
        blocked_k,
        blocked_k_scale,
        weights,
        cu_blocked_ks,
        cu_blocked_ke,
        block_N=block_N,
    )
    clean_and_maintain_logits_interface(got, cu_blocked_ks, cu_blocked_ke)

    ref = ref_pool_mqa_fp8(q, blocked_k, blocked_k_scale, weights)
    ref = ref_clean_and_maintain_logits(ref, cu_blocked_ks, cu_blocked_ke)

    # After the mask, +/-inf positions must agree exactly. Compare the
    # remaining finite values under an fp8×fp8 GEMM tolerance.
    assert torch.equal(torch.isposinf(got), torch.isposinf(ref)), "pos-inf mask differs"
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref)), "neg-inf mask differs"
    finite = torch.isfinite(got) & torch.isfinite(ref)
    torch.testing.assert_close(got[finite], ref[finite], rtol=5e-2, atol=5e-2)
    print(f"  correctness: PASS  (M={M}, H={H}, D={D}, N_blocked={N_blocked}, block_N={block_N}, num_seqs={num_seqs}, per_seq={per_seq})")

    # Speed (kernel only — excludes the post mask).
    def fn():
        return pool_mqa_attn_return_logits_fp8_interface(
            q,
            blocked_k,
            blocked_k_scale,
            weights,
            cu_blocked_ks,
            cu_blocked_ke,
            block_N=block_N,
        )

    ms = do_bench(fn, warmup=50, rep=200)
    # FLOPs: fp8×fp8 GEMM dominates = 2 * M * H * Nb * D (mul+add).
    total_flops = 2 * M * H * N_blocked * D
    tflops = total_flops / (ms * 1e-3) / 1e12
    print(f"  latency: {ms:.4f} ms  ({tflops:.2f} fp8 TFLOPS)")


if __name__ == "__main__":
    # M × k_block_size^-1 must be a multiple of block_N=256.
    # With k_block_size=128 → N_blocked = M/128; need N_blocked % 256 == 0
    # → M % 32768 == 0.
    # (M, H, D, k_block_size, block_N, num_seqs)
    for cfg in [
        (32768, 64, 128, 128, 256, 1),
        (32768, 64, 128, 128, 256, 4),
        (65536, 64, 128, 128, 256, 1),
        (65536, 64, 128, 128, 256, 8),
        (131072, 64, 128, 128, 256, 16),
    ]:
        test_pool_mqa_fp8(*cfg)
