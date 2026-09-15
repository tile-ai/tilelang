"""SM120 SageAttention3-style NVFP4 attention forward in TileLang (v4: two warp groups, rotated GEMMs).

Algorithm (thu-ml/SageAttention `sageattention3_blackwell`, paper arXiv:2505.11594): Q, K, V are
NVFP4 (e2m1 + per-16-element e4m3 scales; K smoothed by its sequence mean, Q by its 128-row block
mean with the compensation term delta_s = q_mean @ K^T), S = FP4MM(Q, K) + delta_s, online softmax
with the two-level P quantization (P2 = P * 448 * 6 folded into exp2, per-16-key e4m3 scale
= e4m3_rn(max16(P2) / 6), P-hat = e2m1_rn(P2 / scale)), O += FP4MM(P-hat, V^T); O /= rowsum(P2).

Data contract (see sageattn3_quant.py): K rows are permuted inside every 32-key group
(PERM32) so that, under the m16n8k64 accumulator layout, every lane holds 8 consecutive original
keys and the C-fragment -> A-fragment relabeling is the identity; V^T and P therefore stay in the
original key order. Inputs: packed fp4 (2 elem/byte, low nibble first) and row-major uint32 scale
words (4 e4m3 bytes = 4 consecutive 16-groups, LSB first).

Kernel structure (FA3-style rotation): 256 threads = two warp groups of 4 warps (FullRow: 16 rows
per warp), one warp of each group per warp scheduler. Both groups run the same softmax /
P-quantization code over the same fragments -- each thread only ever touches its own rows, so the
duplication costs no registers -- but their GEMMs are rotated against each other by a manual
pipeline schedule (T.Pipelined order/stage), so one group issues MMAs while the other runs the
softmax ALU/MUFU work:

    group A:                   softmax(t) | PV(t)  QK(t+1)
    group B:   PV(t-1) QK(t)   | softmax(t)

Group B's PV is rotated one tile back, which gives it a full MMA chunk to issue while group A is
in its softmax section, and lets both groups reach the end of the iteration together. The P scale is folded into the exp2 exponent (P/absmax = exp2((s - smax)
* log2e/sqrt(d) - log2 6)); the P-scale bytes are written with a warp-private shared store
(each warp reads back only its own rows) so no block barrier separates them from the PV GEMM;
the five per-tile copies form one cp.async commit group (one wait + barrier per iteration).

kernel-only usage: python sm120_sageattn3_fwd.py --seq 4096 --verify --bench
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

import tilelang
import tilelang.language as T
from tilelang.layout import Fragment
from tilelang.profiler import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402

LOG2E = 1.4426950408889634
LOG2_P1 = math.log2(1.0 / (448.0 * 6.0))  # two-level: P2 = P * 2688 (softmax_fused.h)
LOG2_FP4 = math.log2(1.0 / 6.0)  # per-16 scale = max16(P2) / 6

# Two fp32 -> one byte of two e2m1 nibbles (even key in the low nibble), via the same
# cvt.rn.satfinite.e2m1x2.f32 the original uses (utils.h packed_float_to_e2m1).
CVT_E2M1_SRC = r"""
__device__ __forceinline__ unsigned int tl_cvt_e2m1x2_rn(float lo, float hi) {
  unsigned int out;
  asm volatile(
      "{\n"
      ".reg .b8 b;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b, %1, %2;\n"
      "cvt.u32.u8 %0, b;\n"
      "}"
      : "=r"(out)
      : "f"(hi), "f"(lo));
  return out;
}

// Warp-private shared-memory byte store for the P-scale bytes: a warp writes and later reads
// only its own rows, so no block barrier is needed. Issued as inline asm so the automatic
// barrier insertion does not see a shared-memory write (a __syncwarp() precedes the PV GEMM).
__device__ __forceinline__ int tl_sts_u8(unsigned char* p, unsigned int v) {
  unsigned int a = static_cast<unsigned int>(__cvta_generic_to_shared(p));
  asm volatile("st.shared.u8 [%0], %1;" :: "r"(a), "r"(v) : "memory");
  return 0;
}
__device__ __forceinline__ int tl_syncwarp() {
  __syncwarp();
  return 0;
}
"""


def _p_group_layout(block_m: int, n_groups: int, warp_rows: int = 32) -> Fragment:
    """Fragment layout of G[row, key_group, key_in_group] (P2 in original key order).

    Thread mapping = the m16n8k64 e2m1 A-operand fragment (FullRow warps of ``warp_rows`` rows):
    for key = 16*group + k, lane 4*(row%8) + (key%32)//8 holds keys 8q..8q+7 (+32) of rows row%8
    and row%8+8. Index = position inside the per-lane A fragment (k64 atom outermost, then the
    16-row atom, then row-half, k-half, key%8), matching make_mma_load_layout's repeat order.
    Written with per-variable floor-divs so the layout checker can prove bijectivity.
    """

    def fwd_thread(i, m, k):
        return (i // warp_rows) * 32 + (i % 8) * 4 + (m % 2) * 2 + k // 8

    def fwd_index(i, m, k):
        return (m // 4) * (warp_rows * 2) + ((i % warp_rows) // 16) * 32 + ((i % 16) // 8) * 8 + ((m % 4) // 2) * 16 + k % 8

    return Fragment((block_m, n_groups, 16), forward_thread_fn=fwd_thread, forward_index_fn=fwd_index)


def _p_half_layout(block_m: int, n_groups: int, warp_rows: int = 32) -> Fragment:
    """Fragment layout of E[row, key_group, t] holding keys 2t (or 2t+1) of the group: G's layout on
    the even (odd) keys, index = G index // 2 (bijective per lane)."""

    def fwd_thread(i, m, t):
        return (i // warp_rows) * 32 + (i % 8) * 4 + (m % 2) * 2 + t // 4

    def fwd_index(i, m, t):
        return (m // 4) * warp_rows + ((i % warp_rows) // 16) * 16 + ((i % 16) // 8) * 4 + ((m % 4) // 2) * 8 + t % 4

    return Fragment((block_m, n_groups, 8), forward_thread_fn=fwd_thread, forward_index_fn=fwd_index)


DEFAULT_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    # The kernel schedules its own two warp groups; the automatic producer/consumer warp
    # specialization is not used (it also does not yet bind annotated fragments to the
    # consumer thread range, see pitfalls in the knowledge base).
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def build_sm120_sageattn3_fwd(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int = 128,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 3,
    threads: int = 256,
    out_dtype=T.bfloat16,
):
    assert dim == 128, "v4 supports head_dim = 128"
    assert block_M == 128 and block_N == 128, "tile contract: 128x128"
    assert threads == 256, "two warp groups of 4 warps (16 rows per warp)"
    # Group B consumes the K/V tile at prefetch distance num_stages-1; at distance 1 the pipeline's
    # cp.async wait is one commit group short for that consumer (silent races).
    assert num_stages >= 3, "num_stages must be >= 3"
    n_warps = threads // 32
    warp_rows = block_M // n_warps  # 16: FullRow policy, every warp owns 16 full rows
    group_threads = threads // 2

    fp4 = T.float4_e2m1fn
    accum = T.float32
    sf_words_qk = dim // 64
    sf_words_pv = block_N // 64
    n_groups = block_N // 16
    softmax_scale = dim**-0.5
    sl2 = softmax_scale * LOG2E
    n_kv_blocks = seq_len // block_N
    assert seq_len % block_N == 0, "seq_len must be a multiple of 128 (pad like the original)"
    g_layout = _p_group_layout(block_M, n_groups, warp_rows)
    ph_layout = _p_half_layout(block_M, n_groups, warp_rows)

    # Manual pipeline schedule. The thread predicates are distributed over the statements before
    # planning, so every statement of a predicated block needs its own entry:
    #   5 copies (stage 0) | B's rotated PV (SA+1) | A's block (SA) | B's seed+QK (SA)
    #   | B's block (SA) | A's seed+QK (SA-1)
    # Two emission-order rules keep every fragment single-versioned (a second version of the S or
    # output accumulator would cost 64 registers per thread): each write of the S accumulator is
    # emitted after all of its reads, and B's rotated PV is emitted before every stage-SA write of
    # the output accumulator.
    SA = num_stages
    n_copy, n_alu_a = 5, 18
    n_alu_b = n_alu_a - 1  # group B's block has no PV
    n_compute = 1 + n_alu_a + 2 + n_alu_b + 2  # each QK is preceded by its delta_s seed
    stage = [0] * n_copy + [SA + 1] + [SA] * n_alu_a + [SA, SA] + [SA] * n_alu_b + [SA - 1, SA - 1]
    order = list(range(n_compute, n_compute + n_copy)) + list(range(n_compute))
    pipe_ann = {  # all five copies in one cp.async commit group
        "software_pipeline_async_producers": [1] * n_copy + [0] * n_compute,
        "software_pipeline_async_producer_groups": [0] * n_copy + [-1] * n_compute,
    }

    @T.macro
    def qk_seed(DS_sh, acc_s):
        # S starts at delta_s (indexed by original key: o(c) = 32*(c//32) + PERM32[c%32]) and the
        # GEMM accumulates on top, so this shared read sits behind the GEMM instead of inside the
        # softmax's dependency chain.
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = DS_sh[(j // 32) * 32 + ((j % 32) // 8) * 2 + ((j % 8) // 2) * 8 + j % 2]

    @T.macro
    def qk_gemm(Q_sh, K_sh, SFQ_sh, SFK_sh, acc_s):
        # Full-tile GEMM inside a thread-predicated block: each executing warp computes its own
        # 16 rows of the FullRow layout (group A -> rows 0-63, group B -> rows 64-127).
        T.mma_gemm_blockscaled(
            Q_sh,
            K_sh,
            acc_s,
            SFQ_sh,
            SFK_sh,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
            clear_accum=False,  # the accumulator already holds delta_s (see qk_seed)
            k_start=0,
            sf_a_granularity_k=16,
            sf_b_granularity_k=16,
            sf_layout="rowmajor",
            scale_dtype="ue4m3",
            sf_b_swizzled=True,
        )

    @T.macro
    def pv_gemm(Pw, V_sh, acc_o, SFP_sh, SFV_sh):
        T.mma_gemm_blockscaled(
            Pw,
            V_sh,
            acc_o,
            SFP_sh,
            SFV_sh,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
            clear_accum=False,
            k_start=0,
            sf_a_granularity_k=16,
            sf_b_granularity_k=16,
            sf_layout="rowmajor",
            scale_dtype="ue4m3",
            a_packed_words=True,
            sf_b_swizzled=True,
        )

    @T.macro
    def softmax_quant(
        with_pv,
        DS_sh,
        V_sh,
        SFV_sh,
        SFP_sh,
        SFP_u8,
        acc_s,
        G,
        smax,
        absmax,
        coff,
        rowsum8,
        Pw,
        Ev,
        Od,
        Es,
        acc_o,
        m_i,
        m_prev,
        ms,
        rescale,
        l_tile,
        l_i,
    ):
        # Regroup S columns (permuted-K order) into original-key 16-groups: a pure in-lane
        # renaming under the K permutation.
        for i, kk, hi, q1, q0, n, jj in T.Parallel(block_M, 2, 2, 2, 2, 4, 2):
            G[i, 4 * kk + 2 * hi + q1, 8 * q0 + 2 * n + jj] = acc_s[i, 64 * kk + 32 * hi + 8 * n + 4 * q1 + 2 * q0 + jj]
        T.reduce_max(G, smax, dim=2, clear=True)  # per-16-key max: 8 in-lane + one xor-1 shuffle
        T.copy(m_i, m_prev)
        T.reduce_max(smax, m_i, dim=1, clear=False)
        for i in T.Parallel(block_M):
            ms[i] = m_i[i] * sl2 + LOG2_P1
            rescale[i] = T.exp2((m_prev[i] - m_i[i]) * sl2)  # same expression order as the original
        for i, m in T.Parallel(block_M, n_groups):
            absmax[i, m] = T.exp2(smax[i, m] * sl2 - ms[i] + LOG2_FP4)  # = max16(P2) / 6
            coff[i, m] = smax[i, m] * sl2 + LOG2_FP4
        # P2 / absmax = exp2((s - smax) * sl2 - log2 6): one exp2 per element, no scaling pass;
        # even/odd key halves so one cvt converts a key pair (one index pattern per loop body).
        for i, m, t in T.Parallel(block_M, n_groups, 8):
            Ev[i, m, t] = T.exp2(G[i, m, 2 * t] * sl2 - coff[i, m])
        for i, m, t in T.Parallel(block_M, n_groups, 8):
            Od[i, m, t] = T.exp2(G[i, m, 2 * t + 1] * sl2 - coff[i, m])
        for i, m, t in T.Parallel(block_M, n_groups, 8):
            Es[i, m, t] = Ev[i, m, t] + Od[i, m, t]
        T.reduce_sum(Es, rowsum8, dim=2, clear=True)
        for i, m in T.Parallel(block_M, n_groups):
            rowsum8[i, m] = rowsum8[i, m] * absmax[i, m]  # rowsum(P2) = sum_m absmax_m * sum_k (P2/absmax)
        T.reduce_sum(rowsum8, l_tile, dim=1, clear=True)
        for i, m, k8 in T.Parallel(block_M, n_groups, 2):  # e2m1 nibbles, 8 keys per uint32
            Pw[i, 2 * m + k8] = 0
            for t2 in T.serial(4):
                Pw[i, 2 * m + k8] = Pw[i, 2 * m + k8] | T.shift_left(
                    T.call_extern("uint32", "tl_cvt_e2m1x2_rn", Ev[i, m, 4 * k8 + t2], Od[i, m, 4 * k8 + t2]),
                    T.cast(8 * t2, T.uint32),
                )
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= rescale[i]
        # P-scale bytes (e4m3 RN satfinite), warp-private store: a warp reads back only its own
        # rows, so no block barrier separates the store from the PV GEMM.
        for i, m in T.Parallel(block_M, n_groups):
            T.evaluate(
                T.call_extern(
                    "int32",
                    "tl_sts_u8",
                    T.address_of(SFP_u8[i, m]),
                    T.cast(T.reinterpret(T.cast(absmax[i, m], T.float8_e4m3fn), T.uint8), T.uint32),
                )
            )
        T.evaluate(T.call_extern("int32", "tl_syncwarp"))
        if with_pv:
            pv_gemm(Pw, V_sh, acc_o, SFP_sh, SFV_sh)
        for i in T.Parallel(block_M):
            l_i[i] = l_i[i] * rescale[i] + l_tile[i]

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), fp4),
        K: T.Tensor((batch, heads, seq_len, dim), fp4),
        VT: T.Tensor((batch, heads, dim, seq_len), fp4),
        SFQ: T.Tensor((batch, heads, seq_len, sf_words_qk), T.uint32),
        SFK: T.Tensor((batch, heads, seq_len, sf_words_qk), T.uint32),
        SFV: T.Tensor((batch, heads, dim, seq_len // 64), T.uint32),
        DS: T.Tensor((batch, heads, seq_len // block_M, seq_len), accum),
        O: T.Tensor((batch, heads, seq_len, dim), out_dtype),
    ):
        with T.Kernel(seq_len // block_M, heads, batch, threads=threads) as (bx, by, bz):
            T.import_source(CVT_E2M1_SRC)
            tx = T.get_thread_binding()
            Q_sh = T.alloc_shared((block_M, dim), fp4)
            K_sh = T.alloc_shared((block_N, dim), fp4)
            V_sh = T.alloc_shared((dim, block_N), fp4)
            SFQ_sh = T.alloc_shared((block_M, sf_words_qk), T.uint32)
            SFK_sh = T.alloc_shared((block_N // 2 + 8, 2 * sf_words_qk), T.uint32)  # swizzled, 4 words/row, one pad row per lane block
            SFV_sh = T.alloc_shared((dim // 2 + 8, 2 * sf_words_pv), T.uint32)  # swizzled, 4 words/row, one pad row per lane block
            SFP_sh = T.alloc_shared((block_M, sf_words_pv), T.uint32)
            SFP_u8 = T.view(SFP_sh, (block_M, n_groups), dtype=T.uint8)
            DS_sh = T.alloc_shared((block_N,), accum)
            O_sh = T.alloc_shared((block_M, dim), out_dtype)

            acc_s = T.alloc_fragment((block_M, block_N), accum)
            G = T.alloc_fragment((block_M, n_groups, 16), accum)  # S regrouped by original 16-key group
            smax = T.alloc_fragment((block_M, n_groups), accum)
            absmax = T.alloc_fragment((block_M, n_groups), accum)
            coff = T.alloc_fragment((block_M, n_groups), accum)  # exponent offset smax*sl2 + log2(6)
            rowsum8 = T.alloc_fragment((block_M, n_groups), accum)
            Pw = T.alloc_fragment((block_M, block_N // 8), T.uint32)  # P-hat words; layout set by the PV GEMM
            Ev = T.alloc_fragment((block_M, n_groups, 8), accum)  # scaled P of even keys
            Od = T.alloc_fragment((block_M, n_groups, 8), accum)  # scaled P of odd keys
            Es = T.alloc_fragment((block_M, n_groups, 8), accum)  # Ev + Od (row-sum partials)
            acc_o = T.alloc_fragment((block_M, dim), accum)
            m_i = T.alloc_fragment((block_M,), accum)
            m_prev = T.alloc_fragment((block_M,), accum)
            ms = T.alloc_fragment((block_M,), accum)
            rescale = T.alloc_fragment((block_M,), accum)
            l_tile = T.alloc_fragment((block_M,), accum)
            l_i = T.alloc_fragment((block_M,), accum)
            T.annotate_layout({G: g_layout, Ev: ph_layout, Od: ph_layout, Es: ph_layout})

            T.copy(Q[bz, by, bx * block_M, 0], Q_sh)
            for r, w in T.Parallel(block_M, sf_words_qk):
                SFQ_sh[r, w] = SFQ[bz, by, bx * block_M + r, w]
            T.fill(acc_o, 0)
            T.fill(acc_s, 0)
            T.fill(Pw, 0)
            T.fill(l_i, 0)
            for i in T.Parallel(block_M):  # explicit (T.fill with -inf is dropped by lowering)
                m_i[i] = -T.infinity(accum)

            for kt in T.Pipelined(n_kv_blocks, order=order, stage=stage, annotations=pipe_ann):
                T.copy(K[bz, by, kt * block_N, 0], K_sh)
                T.copy(VT[bz, by, 0, kt * block_N], V_sh)
                for r, w in T.Parallel(block_N // 2, 2 * sf_words_qk):  # rows already swizzled; pad every 8
                    SFK_sh[(r // 8) * 9 + r % 8, w] = SFK[bz, by, kt * block_N + 2 * r + w // sf_words_qk, w % sf_words_qk]
                for r, w in T.Parallel(dim // 2, 2 * sf_words_pv):
                    SFV_sh[(r // 8) * 9 + r % 8, w] = SFV[bz, by, 2 * r + w // sf_words_pv, kt * sf_words_pv + w % sf_words_pv]
                T.copy(DS[bz, by, bx, kt * block_N : (kt + 1) * block_N], DS_sh)

                if tx // group_threads == 1:  # B: PV of the previous tile, while A is in its softmax
                    pv_gemm(Pw, V_sh, acc_o, SFP_sh, SFV_sh)
                if tx // group_threads == 0:  # A: softmax of this tile, then its own PV
                    softmax_quant(
                        True,
                        DS_sh,
                        V_sh,
                        SFV_sh,
                        SFP_sh,
                        SFP_u8,
                        acc_s,
                        G,
                        smax,
                        absmax,
                        coff,
                        rowsum8,
                        Pw,
                        Ev,
                        Od,
                        Es,
                        acc_o,
                        m_i,
                        m_prev,
                        ms,
                        rescale,
                        l_tile,
                        l_i,
                    )
                if tx // group_threads == 1:  # B: S of this tile, still inside A's softmax section
                    qk_seed(DS_sh, acc_s)
                if tx // group_threads == 1:
                    qk_gemm(Q_sh, K_sh, SFQ_sh, SFK_sh, acc_s)
                if tx // group_threads == 1:  # B: softmax of this tile (its PV runs next iteration)
                    softmax_quant(
                        False,
                        DS_sh,
                        V_sh,
                        SFV_sh,
                        SFP_sh,
                        SFP_u8,
                        acc_s,
                        G,
                        smax,
                        absmax,
                        coff,
                        rowsum8,
                        Pw,
                        Ev,
                        Od,
                        Es,
                        acc_o,
                        m_i,
                        m_prev,
                        ms,
                        rescale,
                        l_tile,
                        l_i,
                    )
                if tx // group_threads == 0:  # A: S of the next tile, while B is in its softmax
                    qk_seed(DS_sh, acc_s)
                if tx // group_threads == 0:
                    qk_gemm(Q_sh, K_sh, SFQ_sh, SFK_sh, acc_s)

            # group B's last PV is drained by the pipeline epilogue (it is the highest stage)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] = acc_o[i, j] / l_i[i]
            T.copy(acc_o, O_sh)
            T.copy(O_sh, O[bz, by, bx * block_M, 0])

    return main


def sm120_sageattn3_fwd(*args, pass_configs: dict | None = None, **kwargs):
    """JIT-compiled kernel; ``pass_configs`` extends DEFAULT_PASS_CONFIGS (experiments/autotune)."""
    cfg = dict(DEFAULT_PASS_CONFIGS)
    if pass_configs:
        cfg.update(pass_configs)
    return tilelang.jit(out_idx=[7], pass_configs=cfg)(build_sm120_sageattn3_fwd)(*args, **kwargs)


# --------------------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------------------
def prepare_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """bf16 [B,H,N,D] -> (canonical dict, TileLang kernel args tuple, delta_s)."""
    q_sm, k_sm, v_p, _, delta_s = sq.preprocess_qkv(q, k, v)
    canon = sq.quantize_canonical(q_sm, k_sm, v_p)
    t = sq.export_for_tilelang(canon)
    args = (
        t["q"].view(torch.int8),
        t["k"].view(torch.int8),
        t["vt"].view(torch.int8),
        t["sfq"],
        t["sfk"],
        t["sfv"],
        delta_s,
    )
    return canon, args, delta_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--dump-source", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    b, h, n, d = args.batch, args.heads, args.seq, args.dim
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    canon, kargs, delta_s = prepare_inputs(q, k, v)
    kernel = sm120_sageattn3_fwd(b, h, n, d, num_stages=args.num_stages, threads=args.threads)
    if args.dump_source:
        print(kernel.get_kernel_source())
    o = kernel(*kargs)
    torch.cuda.synchronize()
    if args.verify:
        o_ref = sq.reference_attention(q, k, v, d**-0.5)
        o_gold = sq.golden_attention(canon, delta_s, d**-0.5)
        print(
            f"TL vs golden: align_ratio={sq.alignment_ratio(o.float(), o_gold, o_ref):.4f} max|diff|={float((o.float() - o_gold).abs().max()):.3e}"
        )
        print(f"TL vs fp64 ref: cos={sq.cos_sim(o.float(), o_ref):.6f} L1={sq.rel_l1(o.float(), o_ref):.4f}")
        try:
            import fp4attn_cuda

            s = sq.export_for_sage(canon)
            o_sage, _ = fp4attn_cuda.fwd(s["q"], s["k"], s["v"], s["sfq"], s["sfk"], s["sfv"], delta_s, n, None, d**-0.5, False, True, True)
            print(
                f"TL vs original kernel: align_ratio={sq.alignment_ratio(o.float(), o_sage.float(), o_ref):.4f} max|diff|={float((o.float() - o_sage.float()).abs().max()):.3e}  (original vs ref cos={sq.cos_sim(o_sage.float(), o_ref):.6f})"
            )
        except ImportError:
            print("original kernel not installed; skipped TL-vs-original")
    if args.bench:
        ms = do_bench(lambda: kernel(*kargs), warmup=25, rep=100, backend="cudagraph", return_mode="median")
        tops = 4.0 * b * h * n * n * d / (ms * 1e-3) / 1e12
        print(f"kernel-only: {ms:.4f} ms  {tops:.1f} TOPS  (B={b} H={h} N={n} D={d})")


if __name__ == "__main__":
    main()
