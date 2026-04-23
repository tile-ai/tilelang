import tilelang
from tilelang import language as T
from tilelang.profiler import do_bench
import torch

from tilelang_utils import prepare_ks_ke_from_cu_seqlens


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_sparse_mqa_attn_return_logits(
    IndexQ,
    IndexK,
    IndexKScale,
    TopKBlockIndex,
    Weights,
    CuSeqLenKS,
    CuSeqLenKE,
    heads: int = 64,
    index_dim: int = 128,
    kv_block_size: int = 128,
    topk: int = 64,
    block_N: int = 128,
    num_stages: int = 1,
    threads: int = 256,
):
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32
    topk_index_dtype = T.int64

    seq_len, seq_len_kv = T.const("seq_len, seq_len_kv")

    H_per_block = heads
    block_N = min(block_N, kv_block_size // 2)
    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"

    IndexQ: T.Tensor[[seq_len * heads, index_dim], fp8_dtype]
    IndexK: T.Tensor[[seq_len_kv, index_dim], fp8_dtype]
    IndexKScale: T.Tensor[[seq_len_kv], accum_dtype]
    TopKBlockIndex: T.Tensor[[seq_len, topk], topk_index_dtype]
    Weights: T.Tensor[[seq_len, heads], accum_dtype]
    CuSeqLenKS: T.Tensor[[seq_len], index_dtype]
    CuSeqLenKE: T.Tensor[[seq_len], index_dtype]

    Logits = T.empty((seq_len, topk * kv_block_size), accum_dtype)

    with T.Kernel(seq_len, threads=threads) as bx:
        index_q_shared = T.alloc_shared([H_per_block, index_dim], fp8_dtype)
        index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
        # Shared (zero-init'd) — see note in the hisa source about serial-topk
        # loop making shared slightly faster than fragment here.
        scale_shared = T.alloc_shared([block_N], accum_dtype)

        s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
        s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
        logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
        weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

        seq_len_i = bx

        cu_k_s_min = CuSeqLenKS[seq_len_i]
        cu_k_e_max = CuSeqLenKE[seq_len_i]

        T.copy(IndexQ[seq_len_i * heads : seq_len_i * heads + H_per_block, :], index_q_shared)
        T.copy(Weights[seq_len_i, :], weights)

        for n_i in T.serial(topk):
            topk_block_id = T.cast(TopKBlockIndex[seq_len_i, n_i], index_dtype)
            block_s = topk_block_id * kv_block_size
            for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                block_s_i = block_s + b_i * block_N

                T.copy(IndexK[block_s_i : block_s_i + block_N, :], index_k_shared)
                for bn_i in T.Parallel(block_N):
                    scale_shared[bn_i] = IndexKScale[block_s_i + bn_i]

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i] * scale_shared[bn_i], 0) * weights[bq_i, h_i]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0]

    return Logits


def fp8_native_block_sparse_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    topk_block_index: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    seq_len, heads, index_dim = q.shape
    topk = topk_block_index.shape[1]
    logits = fp8_native_block_sparse_mqa_attn_return_logits(
        q.view(seq_len * heads, index_dim),
        k,
        k_scale,
        topk_block_index,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        heads=heads,
        index_dim=index_dim,
        kv_block_size=kv_block_size,
        topk=topk,
    )
    return logits


def ref_fp8_block_sparse_mqa(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    topk_block_index: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    M, H, D = q_fp8.shape
    N = k_fp8.shape[0]
    topk = topk_block_index.shape[1]

    block_starts = topk_block_index.long() * kv_block_size  # [M, topk]
    pos_in_block = torch.arange(kv_block_size, device=q_fp8.device)
    k_abs = block_starts[..., None] + pos_in_block[None, None, :]  # [M, topk, B]
    k_safe = k_abs.clamp(0, N - 1)

    q_f = q_fp8.float()
    k_f = k_fp8.float() * k_scale[:, None]
    gathered_k = k_f[k_safe.flatten()].reshape(M, topk, kv_block_size, D)

    s = torch.einsum("mhd,mtid->mtih", q_f, gathered_k)  # [M, topk, B, H]
    logits = (s.clamp(min=0) * weights[:, None, None, :]).sum(dim=-1)  # [M, topk, B]

    in_range = (k_abs >= cu_seqlen_ks.long()[:, None, None]) & (k_abs < cu_seqlen_ke.long()[:, None, None]) & (k_abs < N)
    logits = logits.masked_fill(~in_range, float("-inf"))
    return logits.reshape(M, topk * kv_block_size)


def test_fp8_block_sparse_mqa(
    M: int = 1024,
    H: int = 64,
    D: int = 128,
    kv_block_size: int = 128,
    topk: int = 64,
    num_seqs: int = 1,
):
    """Correctness + speed test packing `num_seqs` equal-length causal
    sequences into the [M, H, D] Q and [M, D] K tensors. Each query sees
    only the prefix of its own sequence (``cu_ks = start_of_seq``,
    ``cu_ke = start_of_seq + position_in_seq + 1``).

    ``topk_block_index`` is drawn at random from [0, num_k_blocks) — some
    picks will point to blocks outside the query's own sequence; those
    positions get -inf via the kernel's built-in mask, and the torch ref
    produces the same -inf. Comparison checks both the +/-inf mask
    pattern (exact) and the finite values (fp8 tolerance)."""
    torch.manual_seed(0)
    assert M % num_seqs == 0, f"M ({M}) must be divisible by num_seqs ({num_seqs})"
    N = M  # causal self-attention prefill, packed

    per_seq = M // num_seqs
    cu_seqlens = torch.arange(num_seqs + 1, device="cuda", dtype=torch.long) * per_seq
    ks_long, ke_long = prepare_ks_ke_from_cu_seqlens(cu_seqlens)
    cu_ks = ks_long.to(torch.int32).contiguous()
    cu_ke = ke_long.to(torch.int32).contiguous()

    q_bf16 = torch.randn(M, H, D, device="cuda", dtype=torch.bfloat16)
    q = q_bf16.to(torch.float8_e4m3fn)
    k_bf16 = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    k = k_bf16.to(torch.float8_e4m3fn)
    k_scale = (0.1 + 0.01 * torch.rand(N, device="cuda", dtype=torch.float32)).contiguous()
    weights = torch.randn(M, H, device="cuda", dtype=torch.float32)

    # Random per-query top-k blocks (distinct indices drawn from [0, num_blocks)).
    num_k_blocks = (N + kv_block_size - 1) // kv_block_size
    topk = min(topk, num_k_blocks)
    g = torch.Generator(device="cuda").manual_seed(42)
    topk_block_index = torch.stack([torch.randperm(num_k_blocks, generator=g, device="cuda")[:topk] for _ in range(M)]).to(torch.int64)

    # Correctness.
    got = fp8_native_block_sparse_mqa_attn_return_logits_interface(
        q,
        k,
        k_scale,
        topk_block_index,
        kv_block_size,
        weights,
        cu_ks,
        cu_ke,
    )
    ref = ref_fp8_block_sparse_mqa(
        q,
        k,
        k_scale,
        topk_block_index,
        kv_block_size,
        weights,
        cu_ks,
        cu_ke,
    )
    # The kernel marks out-of-range as -inf. Compare finite positions only —
    # the -inf mask pattern must agree exactly, so we also check that.
    finite = torch.isfinite(got) & torch.isfinite(ref)
    assert torch.equal(torch.isposinf(got), torch.isposinf(ref)), "pos-inf mask differs"
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref)), "neg-inf mask differs"
    torch.testing.assert_close(got[finite], ref[finite], rtol=1e-1, atol=2e-1)
    print(f"  correctness: PASS  (M={M}, H={H}, D={D}, kv_block_size={kv_block_size}, topk={topk}, num_seqs={num_seqs}, per_seq={per_seq})")

    # Speed.
    def fn():
        return fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q,
            k,
            k_scale,
            topk_block_index,
            kv_block_size,
            weights,
            cu_ks,
            cu_ke,
        )

    ms = do_bench(fn, warmup=50, rep=200)
    # FLOPs: M × topk × kv_block_size × H × D (fp8×fp8) × 2 (mul+add).
    total_flops = 2 * M * topk * kv_block_size * H * D
    tflops = total_flops / (ms * 1e-3) / 1e12
    print(f"  latency: {ms:.4f} ms  ({tflops:.2f} fp8 TFLOPS)")


if __name__ == "__main__":
    # Ref path materialises [M, topk, B, D] fp32 gathered_k which is ~M GB at
    # topk=64, kv_block_size=128, D=128. Keep M modest to avoid OOM.
    # (M, H, D, kv_block_size, topk, num_seqs)
    for cfg in [
        (1024, 64, 128, 128, 64, 1),
        (4096, 64, 128, 128, 64, 1),
        (4096, 64, 128, 128, 64, 4),
        (8192, 64, 128, 128, 64, 1),
        (8192, 64, 128, 128, 64, 8),
        (8192, 64, 128, 64, 128, 8),
        (8192, 64, 128, 256, 32, 8),
    ]:
        test_fp8_block_sparse_mqa(*cfg)
        torch.cuda.empty_cache()
