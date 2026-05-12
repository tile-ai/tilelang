import torch
from tilelang.profiler import do_bench

from fp8_block_mean_pooling import fp8_native_block_mean_pooling_interface
from pool_mqa_fp8 import pool_mqa_attn_return_logits_fp8_interface
from block_sparse_mqa_fp8 import fp8_native_block_sparse_mqa_attn_return_logits_interface
from clean_and_maintain_logits import clean_and_maintain_logits_interface
from tilelang_utils import prepare_ks_ke_from_cu_seqlens


def hisa_indexer(
    q: torch.Tensor,  # [M, H, D] fp8_e4m3fn
    k: torch.Tensor,  # [N, D] fp8_e4m3fn
    k_scale: torch.Tensor,  # [N] f32
    weights: torch.Tensor,  # [M, H] f32
    cu_seqlen_ks: torch.Tensor,  # [M] int32 — per-query K start (inclusive)
    cu_seqlen_ke: torch.Tensor,  # [M] int32 — per-query K end   (exclusive)
    *,
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
) -> torch.Tensor:
    """Run the full hisa prefill pipeline.

    Returns: ``[M, topk_tokens]`` int32 — each row is this query's top
    ``topk_tokens`` K positions, expressed as offsets relative to
    ``cu_seqlen_ks[m]`` (so ``0`` means the query's own K start). Slots
    that fell outside ``[cu_seqlen_ks[m], cu_seqlen_ke[m])`` get ``-1``.
    """
    # ------------------------------------------------------------------
    # Stage 0: fp8 mean-pool over K. Groups K into pool blocks of
    # k_block_size tokens each; outputs one fp8 vector + f32 scale per
    # pool block. Grid = (ceil(N/k_block_size),).
    # ------------------------------------------------------------------
    blocked_k_fp8, blocked_k_scale = fp8_native_block_mean_pooling_interface(
        k,
        k_scale,
        k_block_size,
    )  # [Nb, D] fp8, [Nb] f32

    # Translate the per-query K range from flat-token coords to
    # pool-block coords (floor for start, ceil for end).
    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

    # ------------------------------------------------------------------
    # Stage 1: block-level Q·BlockedK score with ReLU + per-head weight
    # reduction. Output is dense (kernel doesn't mask out-of-range).
    # ------------------------------------------------------------------
    block_k_score = pool_mqa_attn_return_logits_fp8_interface(
        q,
        blocked_k_fp8,
        blocked_k_scale,
        weights,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )  # [M, Nb] f32

    # Mask out-of-range entries to -inf and force +inf on first / last
    # valid block so torch.topk picks the boundary blocks.
    clean_and_maintain_logits_interface(
        block_k_score,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )

    # ------------------------------------------------------------------
    # Stage 1.5: top-block_topk selection. bfloat16 + sorted=False is
    # ~40% faster than f32 and the downstream sparse_mqa doesn't rely
    # on order.
    # ------------------------------------------------------------------
    block_topk_eff = min(block_topk, block_k_score.shape[-1])
    topk_block_indices = torch.topk(
        block_k_score.bfloat16(),
        k=block_topk_eff,
        dim=-1,
        sorted=False,
    ).indices  # [M, block_topk_eff] int64

    # ------------------------------------------------------------------
    # Stage 2: fp8 fine-grained Q·K MQA over only the selected
    # blocks' raw tokens (block_topk_eff blocks × k_block_size tokens
    # per query). The kernel writes -inf for positions outside
    # [cu_seqlen_ks[m], cu_seqlen_ke[m]).
    # ------------------------------------------------------------------
    block_sparse_logits = fp8_native_block_sparse_mqa_attn_return_logits_interface(
        q,
        k,
        k_scale,
        topk_block_indices,
        k_block_size,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )  # [M, block_topk_eff * k_block_size] f32

    # ------------------------------------------------------------------
    # Stage 2.5: top-topk_tokens selection over the block_topk_eff
    # × k_block_size candidate tokens. Gives per-query slot ids.
    # ------------------------------------------------------------------
    topk_tokens_eff = min(topk_tokens, block_sparse_logits.shape[-1])
    relevant_topk_indices = torch.topk(
        block_sparse_logits,
        k=topk_tokens_eff,
        dim=-1,
    ).indices  # [M, topk_tokens_eff] int64

    # ------------------------------------------------------------------
    # Stage 3 (post, Python): translate slot ids → absolute K token
    # position → per-query relative offset (matches vLLM indexer
    # output buffer). Slots whose relative offset falls outside the
    # query's visible range are set to -1.
    # ------------------------------------------------------------------
    # slot = block_id_in_topk × k_block_size + offset_in_block
    #      where block_id_in_topk ∈ [0, block_topk_eff)
    # absolute_k = topk_block_indices[m, block_id_in_topk] × k_block_size + offset_in_block
    absolute_topk_block_indices = torch.gather(
        topk_block_indices,
        dim=-1,
        index=(relevant_topk_indices // k_block_size),
    )
    topk_indices = absolute_topk_block_indices * k_block_size + (relevant_topk_indices % k_block_size)
    topk_indices = topk_indices.to(torch.int32)

    # Relative to this query's K start.
    topk_indices -= cu_seqlen_ks[:, None]
    mask_lo = topk_indices >= 0
    mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0
    mask = mask_lo & mask_hi
    topk_indices = topk_indices.masked_fill(~mask, -1)

    return topk_indices


def test_hisa(
    M: int = 1024,
    H: int = 64,
    D: int = 128,
    k_block_size: int = 128,
    block_topk: int = 8,
    topk_tokens: int = 256,
    num_seqs: int = 1,
):
    """End-to-end smoke + speed test packing `num_seqs` equal-length causal
    sequences into the flat [M, H, D] Q and [N=M, D] K tensors.

    Per-token ``cu_ks / cu_ke`` are produced by
    ``prepare_ks_ke_from_cu_seqlens`` so each query sees only the prefix
    of its own sequence. Validity checks are done per-query (so each
    sequence's tail queries have fewer valid candidate slots).
    """
    torch.manual_seed(0)
    assert M % num_seqs == 0, f"M ({M}) must be divisible by num_seqs ({num_seqs})"
    per_seq = M // num_seqs
    N = M  # causal self-attention, packed

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

    topk_indices = hisa_indexer(
        q,
        k,
        k_scale,
        weights,
        cu_ks,
        cu_ke,
        k_block_size=k_block_size,
        block_topk=block_topk,
        topk_tokens=topk_tokens,
    )

    # Sanity checks.
    assert topk_indices.shape == (M, topk_tokens), f"unexpected output shape {tuple(topk_indices.shape)}"
    assert topk_indices.dtype == torch.int32

    # Every non-(-1) offset must be within [0, cu_ke[m] - cu_ks[m]).
    valid = topk_indices >= 0
    spans = (cu_ke - cu_ks)[:, None].expand_as(topk_indices)
    in_range = topk_indices < spans
    assert (valid == (valid & in_range)).all(), "some valid offset falls outside its query's K window"

    # Per-query expected number of valid slots = min(cu_ke[m] - cu_ks[m],
    # topk_tokens) (clipped by K range and by block_topk × k_block_size).
    expected_valid = torch.minimum(
        (cu_ke - cu_ks).clamp(min=0),
        torch.tensor(min(topk_tokens, block_topk * k_block_size), device=cu_ke.device),
    )
    got_valid = valid.sum(dim=-1).to(torch.int32)
    frac_match = (got_valid == expected_valid).float().mean().item()
    print(
        f"  shape: {tuple(topk_indices.shape)}  "
        f"valid_frac: {valid.float().mean().item():.4f}  "
        f"per-query valid count match: {frac_match:.4f}  "
        f"(num_seqs={num_seqs}, per_seq={per_seq})"
    )

    # Speed.
    def fn():
        return hisa_indexer(
            q,
            k,
            k_scale,
            weights,
            cu_ks,
            cu_ke,
            k_block_size=k_block_size,
            block_topk=block_topk,
            topk_tokens=topk_tokens,
        )

    ms = do_bench(fn, warmup=20, rep=50)
    print(
        f"  latency: {ms:.3f} ms  "
        f"(M={M}, H={H}, D={D}, k_block_size={k_block_size}, "
        f"block_topk={block_topk}, topk_tokens={topk_tokens}, num_seqs={num_seqs})"
    )


if __name__ == "__main__":
    # Ref path in block_sparse_mqa materialises [M, topk, kvB, D] fp32 so
    # stay modest on M (reuse the sparse_mqa module's sizing intuition).
    for cfg in [
        dict(M=1024, H=64, D=128, k_block_size=128, block_topk=16, topk_tokens=256, num_seqs=1),
        dict(M=1024, H=64, D=128, k_block_size=128, block_topk=16, topk_tokens=256, num_seqs=4),
        dict(M=4096, H=64, D=128, k_block_size=128, block_topk=32, topk_tokens=1024, num_seqs=1),
        dict(M=4096, H=64, D=128, k_block_size=128, block_topk=32, topk_tokens=1024, num_seqs=4),
        dict(M=8192, H=64, D=128, k_block_size=128, block_topk=64, topk_tokens=2048, num_seqs=1),
        dict(M=8192, H=64, D=128, k_block_size=128, block_topk=64, topk_tokens=2048, num_seqs=8),
    ]:
        test_hisa(**cfg)
        torch.cuda.empty_cache()
