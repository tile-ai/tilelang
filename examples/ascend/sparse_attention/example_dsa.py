"""Causal sparse attention forward over 512-dimensional shared KV on Ascend NPU.

Q/KV are BF16 with 64 query heads; accumulation stays FP32 and the final
output is FP16. The same KV rows serve as both keys and values.

For the causal example, query and KV positions both start at
zero (front-aligned). Restrict candidates to causal positions first, then
select TopK by indexer scores. For query q, select exactly
count = min(q + 1, kv_len, top_k) indices from [0, min(q + 1, kv_len)), write
them to indices[..., :count], and pad the remaining slots with -1. Valid
indices may be in any order within this prefix. The kernel assumes this
count; it does not infer validity from -1 entries, check causality, or
support holes in the valid prefix. The test generator uses random scores
in place of an indexer.

For DSA-v4 integration, the caller can concatenate original and compressed
KV rows into one shared KV pool and remap their selected indices into that
pool. For example, with KV = concat(original_kv, compressed_kv), offset
compressed indices by the number of original KV rows. The selected rows
from both sources then participate in one softmax. Apply model-specific
causal and window rules before remapping: merged pool indices are storage
addresses, not token positions. Pack all valid selections into a prefix.
This mapping must still satisfy the exact count contract above. If the
merged selection has a different valid count, explicit per-query counts
or index masking must be added to the kernel before using that selection.
KV compression and model-specific attention-sink terms are not implemented
by this example.

top_k is a compile-time parameter and must be a positive multiple of 128.
Each iteration processes 128 indices and combines blocks with online
softmax before writing the output. seq_len must be a positive multiple
of the selected Cube core count.
"""

import argparse
import math
from dataclasses import dataclass

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.carver.arch.driver import get_num_cube_cores
from tilelang.layout import make_ascend_compact_nz_layout
from tilelang.profiler import do_bench
from tvm import tirx


NUM_HEADS = 64
HEAD_DIM = 512


@dataclass(frozen=True)
class FwdTiling:
    block_topk: int = 128


def _copy_kv_pair(dst, src, row_stride):
    """Gather two BF16 KV rows in one DMA, with a row-start stride in GM."""
    row_bytes = HEAD_DIM * 2
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
        dst,
        src,
        0,
        2,
        row_bytes,
        0,
        0,
        0,
        0,
        row_stride * row_bytes,
        row_bytes,
    )


def dsa_fwd(
    batch_size,
    seq_len,
    kv_len,
    top_k,
    *,
    num_blocks=None,
    tiling=None,
):
    tiling = tiling or FwdTiling()

    # Capture plain compile-time scalars in the hygienic T.macros below.
    BLOCK_TOP_K = tiling.block_topk
    CHUNK_ROWS = 16
    PIPELINE_STAGES = 2
    HEADS_PER_AIV = NUM_HEADS // 2
    KV_ROWS_PER_AIV = BLOCK_TOP_K // 2
    VECTOR_LANES = 64  # Ascend fp32 SIMD lane count
    SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)
    dtype = "bfloat16"
    accum_dtype = "float32"

    assert batch_size > 0 and seq_len > 0 and kv_len > 0, "batch_size, seq_len and kv_len must be positive"
    # The softmax handles two fp32x64 vectors and packs one bf16x128 vector.
    assert BLOCK_TOP_K == 128, "The current SIMD softmax packing requires block_topk == 128"
    assert top_k > 0 and top_k % BLOCK_TOP_K == 0, "top_k must be a positive multiple of block_topk"
    NUM_BLOCKS = get_num_cube_cores(torch.npu.current_device()) if num_blocks is None else num_blocks
    assert NUM_BLOCKS > 0, "num_blocks must be positive"
    assert seq_len % NUM_BLOCKS == 0, "seq_len must be a multiple of num_blocks"
    NUM_TOPK_BLOCKS = top_k // BLOCK_TOP_K
    MAX_QUERIES_PER_BLOCK = (batch_size * seq_len + NUM_BLOCKS - 1) // NUM_BLOCKS
    OUTPUT_L2_CACHE_CTRL = 4 if MAX_QUERIES_PER_BLOCK > 128 else 0

    # -- shared T.macros --------------------------------------------------

    @T.macro
    def gather_kv(batch_id, valid_count, sid, KV, indices_ub, KV_ub, KV_nz_ub, KV_l1):
        # Each AIV gathers one half of the sparse tile, skipping invalid rows.
        for chunk in T.Serial(
            KV_ROWS_PER_AIV // CHUNK_ROWS,
            annotations={"multi_buffer_eligible": [KV_ub]},
        ):
            row_start = sid * KV_ROWS_PER_AIV + chunk * CHUNK_ROWS
            valid_rows = T.max(T.min(valid_count - row_start, CHUNK_ROWS), 0)
            for pair in T.Serial(valid_rows // 2):
                index0 = T.int32(indices_ub[row_start + 2 * pair])
                index1 = T.int32(indices_ub[row_start + 2 * pair + 1])
                T.assume(index0 >= 0)
                T.assume(index0 < kv_len)
                T.assume(index1 >= 0)
                T.assume(index1 < kv_len)
                if index0 == index1:
                    T.copy(KV[batch_id, index0, :], KV_ub[2 * pair, :])
                    T.copy(KV[batch_id, index1, :], KV_ub[2 * pair + 1, :])
                else:
                    first_index = T.min(index0, index1)
                    row_stride = T.max(index0, index1) - first_index
                    _copy_kv_pair(
                        T.access_ptr(KV_ub[2 * pair, 0], "w"),
                        T.access_ptr(KV[batch_id, first_index, 0], "r"),
                        row_stride,
                    )
            if valid_rows % 2 != 0:
                tail_row = valid_rows - 1
                tail_index = T.int32(indices_ub[row_start + tail_row])
                T.assume(tail_index >= 0)
                T.assume(tail_index < kv_len)
                T.copy(KV[batch_id, tail_index, :], KV_ub[tail_row, :])

            T.copy(KV_ub, KV_nz_ub[:CHUNK_ROWS, :])
            T.copy(KV_nz_ub[:CHUNK_ROWS, :], KV_l1[row_start : row_start + CHUNK_ROWS, :])

    @T.macro
    def softmax_pack(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, valid_count):
        with T.SimdVF(latency=735):
            fp32_mask = S.pset(32, "PAT_ALL")
            scalar_mask = S.pset(32, "PAT_VL1")
            bf16_mask = S.pset(16, "PAT_VL128")
            # A finite negative sentinel makes masked exp(score - max) zero
            # for normal logits; it is not IEEE -inf.
            mask_value = S.vdup(T.float32(-3.402823e38), accum_dtype, fp32_mask)
            p_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", HEADS_PER_AIV + 1, BLOCK_TOP_K),
                dtype,
            )
            nz_stride = T.int32(((HEADS_PER_AIV + 1) << 16) | 1)
            valid_lo = S.update_mask(T.uint32(T.min(valid_count, VECTOR_LANES)), width=32)
            valid_hi = S.update_mask(T.uint32(T.max(valid_count - VECTOR_LANES, 0)), width=32)

            for head in range(HEADS_PER_AIV):
                scores_lo = S.vmuls(S.vld(S_ub[head, 0], "NORM"), T.float32(SOFTMAX_SCALE), fp32_mask)
                scores_hi = S.vmuls(S.vld(S_ub[head, VECTOR_LANES], "NORM"), T.float32(SOFTMAX_SCALE), fp32_mask)
                masked_lo = S.vsel(scores_lo, mask_value, valid_lo)
                masked_hi = S.vsel(scores_hi, mask_value, valid_hi)
                S.vsts(S_ub[head, 0], masked_lo, fp32_mask, "NORM_B32")
                S.vsts(S_ub[head, VECTOR_LANES], masked_hi, fp32_mask, "NORM_B32")
                max_acc = S.alloc_var("float32")
                max_acc = S.vmax(masked_lo, masked_hi, fp32_mask)
                if NUM_TOPK_BLOCKS > 1:
                    # Online softmax: m_new = max(m_old, block_max).
                    old_max = S.vld(m_ub[head], "BRC_B32")
                    S.vsts(old_m_ub[head], old_max, scalar_mask, "ONEPT_B32")
                    max_acc = S.vmax(max_acc, old_max, fp32_mask)
                row_max = S.vcmax(max_acc, fp32_mask)
                S.vsts(m_ub[head], row_max, scalar_mask, "ONEPT_B32")

            S.mem_bar("VST_VLD")

            # Pack unnormalized P = exp(score - m_new) directly in BF16 NZ.
            for head in range(HEADS_PER_AIV):
                row_max = S.vld(m_ub[head], "BRC_B32")
                even_scores, odd_scores = S.vld2(S_ub[head, 0], "DINTLV_B32")
                even_prob = S.vexpdif(even_scores, row_max, fp32_mask)
                odd_prob = S.vexpdif(odd_scores, row_max, fp32_mask)
                sum_acc = S.vadd(even_prob, odd_prob, fp32_mask)
                even_bf16 = S.vcvt(even_prob, dtype, fp32_mask, sat=False, part=0)
                odd_bf16 = S.vcvt(odd_prob, dtype, fp32_mask, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(even_bf16, "uint16x128"),
                    T.reinterpret(odd_bf16, "uint16x128"),
                    bf16_mask,
                )
                p_nz_ptr = S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    p_nz_ptr,
                    nz_stride,
                    bf16_mask,
                    update=True,
                )
                prob_sum = S.vcadd(sum_acc, fp32_mask)
                if NUM_TOPK_BLOCKS > 1:
                    # alpha = exp(m_old - m_new); l_new = alpha * l_old + sum(P).
                    alpha = S.vexpdif(S.vld(old_m_ub[head], "BRC_B32"), row_max, fp32_mask)
                    S.vsts(alpha_ub[head], alpha, scalar_mask, "ONEPT_B32")
                    denominator = S.vadd(
                        S.vmul(S.vld(l_ub[head], "BRC_B32"), alpha, fp32_mask),
                        prob_sum,
                        scalar_mask,
                    )
                    S.vsts(l_ub[head], denominator, scalar_mask, "ONEPT_B32")
                else:
                    S.vsts(l_ub[head], prob_sum, scalar_mask, "ONEPT_B32")

    @T.macro
    def finalize_output(O_ub, O_cast_ub, l_ub, O, batch_id, query_id):
        # Normalize in FP32 and pack FP16 only after all TopK blocks contribute.
        # Ascending reads allow the single-block path to compress in place.
        with T.SimdVF(latency=814):
            normalize_mask = S.pset(32, "PAT_ALL")
            for head in range(HEADS_PER_AIV):
                denominator = S.vld(l_ub[head], "BRC_B32")
                for column in range(0, HEAD_DIM, VECTOR_LANES):
                    normalized = S.vdiv(S.vld(O_ub[head, column], "NORM"), denominator, normalize_mask)
                    packed = S.vcvt(normalized, "float16", normalize_mask, sat=False)
                    S.vsts(O_cast_ub[head, column], packed, normalize_mask, "PK_B32")
        # Shape heuristic: once L2 fills, direct GM stores are preferable.
        # Use mode 4 above 128 queries/core, otherwise mode 0.
        # TODO: Split the query loops: bypass L2 first, then cache the final
        # iterations so L2 fills only at the end.
        T.dual_copy(O_cast_ub[:HEADS_PER_AIV, :], O[batch_id, query_id, :, :], l2_cache_ctrl=OUTPUT_L2_CACHE_CTRL)

    # -- auto-scheduled Cube + Vector kernel ------------------------------

    @T.prim_func
    def main(
        Q: T.Buffer((batch_size, seq_len, NUM_HEADS, HEAD_DIM), dtype),
        KV: T.Buffer((batch_size, kv_len, HEAD_DIM), dtype),
        TopKIndices: T.Buffer((batch_size, seq_len, top_k), "int32"),
        O: T.Buffer((batch_size, seq_len, NUM_HEADS, HEAD_DIM), "float16"),
    ):
        with T.MixedKernel(NUM_BLOCKS) as (core_id, sid):
            num_tasks = T.ceildiv(batch_size * seq_len - core_id, NUM_BLOCKS)
            Q_l1 = T.alloc_l1((NUM_HEADS, HEAD_DIM), dtype)
            KV_l1 = T.alloc_l1((BLOCK_TOP_K, HEAD_DIM), dtype)
            P_l1 = T.alloc_l1((NUM_HEADS, BLOCK_TOP_K), dtype)
            accum_l0c = T.alloc_l0c((NUM_HEADS, HEAD_DIM), accum_dtype)

            KV_ub = T.alloc_shared((CHUNK_ROWS, HEAD_DIM), dtype)
            KV_nz_ub = T.alloc_shared((CHUNK_ROWS + 1, HEAD_DIM), dtype)
            S_ub = T.alloc_shared((HEADS_PER_AIV, BLOCK_TOP_K), accum_dtype)
            P_nz_ub = T.alloc_shared((HEADS_PER_AIV + 1, BLOCK_TOP_K), dtype)
            O_ub = T.alloc_shared((HEADS_PER_AIV, HEAD_DIM), accum_dtype)
            # Reinterpret the same bytes; the packed output uses the first half.
            O_cast_ub = T.view(O_ub, (HEADS_PER_AIV * 2, HEAD_DIM), dtype="float16")
            O_tmp_ub = T.alloc_shared((HEADS_PER_AIV, HEAD_DIM), accum_dtype)
            O_tmp_cast_ub = T.view(O_tmp_ub, (HEADS_PER_AIV * 2, HEAD_DIM), dtype="float16")
            # Per-head running max m, exp sum l, and output rescaling factor alpha.
            m_ub = T.alloc_shared((VECTOR_LANES,), accum_dtype)
            l_ub = T.alloc_shared((VECTOR_LANES,), accum_dtype)
            alpha_ub = T.alloc_shared((VECTOR_LANES,), accum_dtype)
            old_m_ub = T.alloc_shared((VECTOR_LANES,), accum_dtype)
            indices_ub = T.alloc_shared((BLOCK_TOP_K,), "int32")

            T.annotate_layout(
                {
                    KV_nz_ub: make_ascend_compact_nz_layout(KV_nz_ub),
                    P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub),
                }
            )
            T.annotate_buffer_versions(
                {
                    KV_l1: 3,
                    P_l1: 2,
                    accum_l0c: 2,
                    S_ub: 2,
                    KV_ub: 2,
                    KV_nz_ub: 2,
                    P_nz_ub: 2,
                    O_ub: 1,
                    indices_ub: 2,
                    O_tmp_ub: 1,
                }
            )
            if NUM_TOPK_BLOCKS > 1:
                # Keep the loop-carried softmax state in one shared version.
                T.annotate_buffer_versions({m_ub: 1, l_ub: 1})

            for task_block in T.Pipelined(
                num_tasks * NUM_TOPK_BLOCKS,
                num_stages=PIPELINE_STAGES,
                annotations={"enable_offset": True},
            ):
                task = task_block // NUM_TOPK_BLOCKS
                block_id = task_block % NUM_TOPK_BLOCKS
                query = task * NUM_BLOCKS + core_id
                batch_id = query // seq_len
                query_id = query % seq_len
                valid_count = T.min(T.min(query_id + 1, kv_len), top_k)
                block_valid = T.max(T.min(valid_count - block_id * BLOCK_TOP_K, BLOCK_TOP_K), 0)

                # TODO: Specialize empty, partial and full sparse blocks:
                # skip empty-block computation while preserving query state
                # and final output; mask partial blocks and omit masking for full blocks.
                if block_id == 0:
                    T.copy(Q[batch_id, query_id, :, :], Q_l1, l2_cache_ctrl=4)
                    with T.SimdVF():
                        T.fill(KV_ub, T.float32(0.0))
                T.copy(
                    TopKIndices[batch_id, query_id, block_id * BLOCK_TOP_K : (block_id + 1) * BLOCK_TOP_K],
                    indices_ub,
                )
                gather_kv(batch_id, block_valid, sid, KV, indices_ub, KV_ub, KV_nz_ub, KV_l1)

                # QK -> causal-prefix softmax -> PV.
                T.gemm(Q_l1, KV_l1, accum_l0c[:, :BLOCK_TOP_K], transpose_B=True, clear_accum=True)
                T.dual_copy(accum_l0c[:, :BLOCK_TOP_K], S_ub)
                if NUM_TOPK_BLOCKS > 1:  # noqa: SIM102 - specialize before the runtime block check
                    if block_id == 0:
                        # Each query starts with m = -inf and l = 0. The
                        # single-block specialization needs no prior state.
                        with T.SimdVF(latency=445):
                            fp32_mask = S.pset(32, "PAT_ALL")
                            initial_max = S.vdup(-T.infinity(accum_dtype), accum_dtype, fp32_mask)
                            initial_sum = S.vdup(T.float32(0.0), accum_dtype, fp32_mask)
                            S.vsts(m_ub[0], initial_max, fp32_mask, "NORM_B32")
                            S.vsts(l_ub[0], initial_sum, fp32_mask, "NORM_B32")
                softmax_pack(
                    S_ub,
                    P_nz_ub,
                    m_ub,
                    l_ub,
                    alpha_ub,
                    old_m_ub,
                    block_valid,
                )
                T.dual_copy(P_nz_ub[:HEADS_PER_AIV, :], P_l1)
                T.gemm(P_l1, KV_l1, accum_l0c, transpose_B=False, clear_accum=True)
                if block_id == 0:
                    T.dual_copy(accum_l0c, O_ub)
                else:
                    T.dual_copy(accum_l0c, O_tmp_ub)
                    # Rescale the previous numerator to the new softmax max.
                    with T.SimdVF(latency=740):
                        for head, column in T.Parallel(HEADS_PER_AIV, HEAD_DIM):
                            O_ub[head, column] = O_ub[head, column] * alpha_ub[head] + O_tmp_ub[head, column]
                if block_id == NUM_TOPK_BLOCKS - 1:
                    if NUM_TOPK_BLOCKS > 1:
                        # Reuse consumed PV scratch, keeping the running O in FP32.
                        finalize_output(O_ub, O_tmp_cast_ub, l_ub, O, batch_id, query_id)
                    else:
                        finalize_output(O_ub, O_cast_ub, l_ub, O, batch_id, query_id)

    return main


# -- standalone correctness check and benchmark ---------------------------


def make_inputs(batch_size: int, seq_len: int, kv_len: int, top_k: int):
    torch.manual_seed(88888888)
    q = torch.randn((batch_size, seq_len, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    kv = torch.randn((batch_size, kv_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    indices = torch.full((batch_size, seq_len, top_k), -1, dtype=torch.int32, device="npu")
    for query in range(seq_len):
        available = min(query + 1, kv_len)
        count = min(available, top_k)
        scores = torch.rand(batch_size, available, device="npu")
        indices[:, query, :count] = torch.topk(scores, count, dim=-1).indices.to(torch.int32)
    return q, kv, indices


def reference_attention(q, kv, indices):
    batch_size, seq_len, _, dim = q.shape
    scale = 1.0 / math.sqrt(dim)
    valid = indices >= 0
    gather_indices = indices.clamp(min=0)[..., None].expand(-1, -1, -1, dim).long()
    gathered_kv = torch.gather(kv[:, None, :, :].expand(batch_size, seq_len, -1, -1), 2, gather_indices)
    scores = torch.einsum("bqhd,bqkd->bqhk", q.float(), gathered_kv.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bqhk,bqkd->bqhd", probabilities, gathered_kv.float())


def compile_kernel(batch_size, seq_len, kv_len, top_k, *, num_blocks=None, tiling=None):
    return tilelang.compile(
        dsa_fwd(batch_size, seq_len, kv_len, top_k, num_blocks=num_blocks, tiling=tiling),
        target="ascend",
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


def run(batch_size=1, seq_len=448, *, kv_len=4096, top_k=128, num_blocks=None, tiling=None):
    kernel = compile_kernel(batch_size, seq_len, kv_len, top_k, num_blocks=num_blocks, tiling=tiling)
    inputs = make_inputs(batch_size, seq_len, kv_len, top_k)
    expected = reference_attention(*inputs)
    actual = kernel(*inputs)
    torch.npu.synchronize()
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    print(f"Verification passed; max error: {(actual - expected).abs().max().item():.6f}")

    latency_ms = do_bench(lambda: kernel(*inputs), backend="msprof", _n_warmup=30, _n_repeat=100)
    flops = 4 * batch_size * seq_len * NUM_HEADS * HEAD_DIM * top_k
    print(f"{latency_ms * 1000:.2f} us/iter | {flops / (latency_ms * 1e-3) / 1e12:.1f} TFLOPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    run(batch_size=args.batch_size, seq_len=args.seq_len)
