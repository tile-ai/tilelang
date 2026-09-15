"""SM120 SageAttention3-style NVFP4 attention forward in TileLang (warp-specialized producer/consumer).

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

Kernel structure (persistent + warp specialization): a persistent grid of one CTA per SM, each
running 384 threads = one producer warp group (tx >= 256) and eight consumer warps (tx < 256,
FullRow: 16 rows per warp). Putting the producer at the *high* thread ids keeps the consumer's
thread numbering starting at 0, so every fragment layout is the same as in the non-specialized
kernel.

Each role runs its own T.PersistentTileScheduler over (q_block, head*batch); the default
column-major order makes the q block the fast axis, so the CTAs resident in one wave work on
neighbouring q blocks of the same (head, batch) and share the same K/V stream through L2.

The producer owns all global loads. Q/K/V/delta_s go through TMA (T.tma_copy with a user-managed
mbarrier); the scale words need the per-lane row swizzle, which TMA cannot express, so the
producer gathers those with ordinary vectorized global->shared stores. Synchronization is
numbered mbarrier pairs: loaded/consumed over the num_stages K/V versions (loaded also carries
the TMA transaction bytes) and q_ready/q_free over the q_stages Q versions. Q has to be
multi-buffered across waves -- with a single Q slot the producer stalls at every wave boundary
waiting for the consumer to drain the whole wave, which costs more than the persistent grid
saves.

Both pipelines want to be shallow here: at 4K, num_stages=2 measures 979 TOPS against 948 for
num_stages=3, and num_stages=4 no longer fits the 99 KiB shared-memory budget. q_stages=2 beats
3 and 4 as well.

T.set_max_nreg hands the producer's registers to the consumers. The split must leave at least one
warp granule free: producer_warps*32*producer_regs + consumer_warps*32*consumer_regs must be
strictly below 65536, or setmaxnreg.inc blocks forever and the kernel hangs (24/240, 32/232 and
40/224 all work; 32/240 is exactly 65536 and deadlocks).

The output is stored in store_block_N-wide column chunks: under the producer/consumer branch the
O tile can no longer share its shared-memory address with the Q tile.

Per kv tile the consumer does what the original does, with fewer instructions (702 per lane
against its 860; both issue the same 64 blockscaled MMAs):

* Q and its scale words are copied into register fragments once per q tile and feed the QK
  GEMM from there (``a_packed_words=True`` plus a register SFA operand), instead of being
  reloaded from shared memory for every kv tile.
* P is packed four key pairs at a time into one uint32 word with a single inline-asm sequence
  (four ``cvt.rn.satfinite.e2m1x2.f32`` into byte registers and one concatenating move), the way
  the original packs it.
* The softmax row sum stays in the lane: every lane accumulates its half of each group scaled
  by that group's absmax, and the cross-lane reduction runs once per q tile before the division.
  This changes the float summation order, so the output is no longer bit-identical to
  sm120_sageattn3_fwd.py (max |diff| < 1e-3; golden and original alignment unchanged).

ptxas is compiled with ``--register-usage-level=8``. Levels 6-10 produce the same code and that
code is 0.6-1.1% faster than the default level 5 on this kernel (same arithmetic, bit-identical
output). The issue order ptxas picks for the kv loop moves throughput by a few percent in either
direction and reacts to unrelated edits, so changes to this file should be re-measured rather
than assumed neutral.

kernel-only usage: python sm120_sageattn3_fwd_ws.py --seq 4096 --verify --bench
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import Fragment
from tilelang.profiler import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402

LOG2E = 1.4426950408889634
LOG2_P1 = math.log2(1.0 / (448.0 * 6.0))  # two-level: P2 = P * 2688 (softmax_fused.h)
LOG2_FP4 = math.log2(1.0 / 6.0)  # per-16 scale = max16(P2) / 6

# Device helpers issued as inline asm: P packing, the warp-private P-scale store, and a warp sync.
CVT_E2M1_SRC = r"""
// Four key pairs -> one uint32 of e2m1 nibbles (byte t = keys 2t, 2t+1; even key in the low nibble),
// issued exactly like the original's packed_float_to_e2m1 (utils.h): four e2m1x2 conversions into
// byte registers and a single concatenating move, instead of per-pair zero-extend/shift/or.
__device__ __forceinline__ unsigned int tl_cvt_e2m1x8_rn(float e0, float o0, float e1, float o1,
                                                        float e2, float o2, float e3, float o3) {
  unsigned int out;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(e0), "f"(o0), "f"(e1), "f"(o1), "f"(e2), "f"(o2), "f"(e3), "f"(o3));
  return out;
}

// P-scale byte: e4m3 RN satfinite conversion straight into a warp-private shared-memory store
// (a warp writes and later reads only its own rows; issued as asm so the automatic block barrier
// is not inserted -- the __syncwarp() before the PV GEMM makes the bytes visible). The pair
// converter with both sources = f yields {f, f}; st.shared.u8 keeps the low byte. Replaces the
// reinterpret-cast form, which cost a zero-extension mask and a constant move per byte.
__device__ __forceinline__ int tl_sts_ue4m3(unsigned char* p, float f) {
  unsigned int a = static_cast<unsigned int>(__cvta_generic_to_shared(p));
  asm volatile("{\n.reg .b16 d;\ncvt.rn.satfinite.e4m3x2.f32 d, %1, %1;\nst.shared.u8 [%0], d;\n}" :: "r"(a), "f"(f) : "memory");
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


def _p_gsum_layout(block_m: int, n_groups: int, warp_rows: int = 32) -> Fragment:
    """gs[row, group, half]: the in-lane sum of the P2/absmax values of one key half of a group."""

    def fwd_thread(i, m, h):
        return (i // warp_rows) * 32 + (i % 8) * 4 + (m % 2) * 2 + h

    def fwd_index(i, m, h):
        return (m // 4) * 4 + ((i % 16) // 8) * 2 + (m % 4) // 2

    return Fragment((block_m, n_groups, 2), forward_thread_fn=fwd_thread, forward_index_fn=fwd_index)


def _p_lane_sum_layout(block_m: int, warp_rows: int = 32) -> Fragment:
    """lacc[row, group parity, half]: per-lane running row sum (rows i and i+8 share a lane)."""

    def fwd_thread(i, p, h):
        return (i // warp_rows) * 32 + (i % 8) * 4 + p * 2 + h

    def fwd_index(i, p, h):
        return (i % 16) // 8

    return Fragment((block_m, 2, 2), forward_thread_fn=fwd_thread, forward_index_fn=fwd_index)


DEFAULT_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    # ptxas --register-usage-level=8: +0.6~1.1% over the default 5 at 8K-32K (identical output).
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 8,
    # The kernel writes its own producer/consumer split with explicit mbarriers and
    # T.tma_copy, so the automatic producer/consumer warp specialization stays off.
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def build_sm120_sageattn3_fwd(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int = 128,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 2,
    threads: int = 384,
    producer_regs: int = 24,
    consumer_regs: int = 240,
    store_block_N: int = 64,
    q_stages: int = 2,
    out_dtype=T.bfloat16,
):
    assert dim == 128, "v4 supports head_dim = 128"
    assert block_M == 128 and block_N == 128, "tile contract: 128x128"
    assert threads == 384, "1 producer warpgroup (tx >= 256) + 2 consumer warp groups (tx < 256)"
    assert num_stages >= 2, "num_stages must be >= 2"
    assert dim % store_block_N == 0
    n_consumer_threads = 256
    n_warps = n_consumer_threads // 32  # 8 consumer warps
    warp_rows = block_M // n_warps  # 16: FullRow policy, every warp owns 16 full rows

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
    gs_layout = _p_gsum_layout(block_M, n_groups, warp_rows)
    lacc_layout = _p_lane_sum_layout(block_M, warp_rows)

    @T.macro
    def qk_seed(DS_sh, stage, acc_s):
        # S starts at delta_s (indexed by original key: o(c) = 32*(c//32) + PERM32[c%32]) and the
        # GEMM accumulates on top, so this shared read sits behind the GEMM instead of inside the
        # softmax's dependency chain.
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = DS_sh[stage, (j // 32) * 32 + ((j % 32) // 8) * 2 + ((j % 8) // 2) * 8 + j % 2]

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
            a_packed_words=True,  # Q words live in registers for the whole q tile
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
        gs,
        Pw,
        Ev,
        Od,
        Ev1,
        Ev2,
        Ev3,
        Od1,
        Od2,
        Od3,
        acc_o,
        m_i,
        m_prev,
        ms,
        rescale,
        lacc,
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
        # Row sum kept lane-local, like the original's in-lane row_sum: each lane adds its own half
        # of every group, scaled by that group's absmax; the cross-lane reduction happens once per
        # q tile, just before the division (no per-tile shuffles).
        for i, m, h in T.Parallel(block_M, n_groups, 2):
            gs[i, m, h] = 0.0
            for t4 in T.serial(4):
                gs[i, m, h] = gs[i, m, h] + Ev[i, m, 4 * h + t4] + Od[i, m, 4 * h + t4]
        for i, p, h in T.Parallel(block_M, 2, 2):
            lacc[i, p, h] = lacc[i, p, h] * rescale[i]
            for mm in T.serial(4):
                lacc[i, p, h] = lacc[i, p, h] + gs[i, 2 * mm + p, h] * absmax[i, 2 * mm + p]
        for i, m, k8 in T.Parallel(block_M, n_groups, 2):  # e2m1 nibbles, 8 keys per uint32
            Pw[i, 2 * m + k8] = T.call_extern(
                "uint32",
                "tl_cvt_e2m1x8_rn",
                Ev[i, m, 4 * k8],
                Od[i, m, 4 * k8],
                Ev1[i, m, 4 * k8 + 1],
                Od1[i, m, 4 * k8 + 1],
                Ev2[i, m, 4 * k8 + 2],
                Od2[i, m, 4 * k8 + 2],
                Ev3[i, m, 4 * k8 + 3],
                Od3[i, m, 4 * k8 + 3],
            )
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= rescale[i]
        # P-scale bytes (e4m3 RN satfinite), warp-private store: a warp reads back only its own
        # rows, so no block barrier separates the store from the PV GEMM.
        for i, m in T.Parallel(block_M, n_groups):
            T.evaluate(
                T.call_extern(
                    "int32",
                    "tl_sts_ue4m3",
                    T.address_of(SFP_u8[i, m]),
                    absmax[i, m],
                )
            )
        T.evaluate(T.call_extern("int32", "tl_syncwarp"))
        if with_pv:
            pv_gemm(Pw, V_sh, acc_o, SFP_sh, SFV_sh)

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
        with T.Kernel(driver.get_num_sms(), threads=threads) as block_id:
            T.import_source(CVT_E2M1_SRC)
            tx = T.get_thread_binding()
            Q_sh = T.alloc_shared((q_stages, block_M, dim), fp4)
            Q_words = T.view(Q_sh, (q_stages, block_M, dim // 8), dtype=T.uint32)  # 8 e2m1 elements per word
            K_sh = T.alloc_shared((num_stages, block_N, dim), fp4)
            V_sh = T.alloc_shared((num_stages, dim, block_N), fp4)
            SFQ_sh = T.alloc_shared((q_stages, block_M, sf_words_qk), T.uint32)
            SFK_sh = T.alloc_shared((num_stages, block_N // 2 + 8, 2 * sf_words_qk), T.uint32)
            SFV_sh = T.alloc_shared((num_stages, dim // 2 + 8, 2 * sf_words_pv), T.uint32)
            SFP_sh = T.alloc_shared((block_M, sf_words_pv), T.uint32)
            SFP_u8 = T.view(SFP_sh, (block_M, n_groups), dtype=T.uint8)
            DS_sh = T.alloc_shared((num_stages, block_N), accum)
            O_sh = T.alloc_shared((block_M, store_block_N), out_dtype)

            loaded = T.alloc_barrier([128] * num_stages)
            consumed = T.alloc_barrier([n_consumer_threads] * num_stages)
            q_ready = T.alloc_barrier([128] * q_stages)
            q_free = T.alloc_barrier([n_consumer_threads] * q_stages)

            if tx >= n_consumer_threads:  # ---- producer warp group ----
                if producer_regs > 0:
                    T.set_max_nreg(producer_regs, 0)
                sched_p = T.PersistentTileScheduler(seq_len // block_M, heads * batch, name="sp")
                sched_p.init(block_id)
                while sched_p.valid():
                    bx = sched_p.m_idx
                    by = sched_p.n_idx % heads
                    bz = sched_p.n_idx // heads
                    wave = sched_p.current_iter
                    # Q is multi-buffered across waves: the producer fetches wave w+1's Q while
                    # the consumer is still draining wave w (without this the whole producer
                    # stalls at every wave boundary).
                    qs = wave % q_stages
                    qp = (wave // q_stages) & 1
                    T.barrier_wait(q_free[qs], qp ^ 1)
                    T.tma_copy(Q[bz, by, bx * block_M, 0], Q_sh[qs, :, :], barrier=q_ready[qs], leader_scope_threads=128)
                    for r, w in T.Parallel(block_M, sf_words_qk):
                        SFQ_sh[qs, r, w] = SFQ[bz, by, bx * block_M + r, w]
                    T.barrier_arrive(q_ready[qs])

                    for kt in T.serial(n_kv_blocks):
                        phase = wave * n_kv_blocks + kt
                        stage = phase % num_stages
                        parity = (phase // num_stages) & 1
                        T.barrier_wait(consumed[stage], parity ^ 1)
                        T.tma_copy(K[bz, by, kt * block_N, 0], K_sh[stage, :, :], barrier=loaded[stage], leader_scope_threads=128)
                        T.tma_copy(VT[bz, by, 0, kt * block_N], V_sh[stage, :, :], barrier=loaded[stage], leader_scope_threads=128)
                        T.tma_copy(
                            DS[bz, by, bx, kt * block_N : (kt + 1) * block_N],
                            DS_sh[stage, :],
                            barrier=loaded[stage],
                            leader_scope_threads=128,
                        )
                        for r, w in T.Parallel(block_N // 2, 2 * sf_words_qk):
                            SFK_sh[stage, (r // 8) * 9 + r % 8, w] = SFK[bz, by, kt * block_N + 2 * r + w // sf_words_qk, w % sf_words_qk]
                        for r, w in T.Parallel(dim // 2, 2 * sf_words_pv):
                            SFV_sh[stage, (r // 8) * 9 + r % 8, w] = SFV[
                                bz, by, 2 * r + w // sf_words_pv, kt * sf_words_pv + w % sf_words_pv
                            ]
                        T.barrier_arrive(loaded[stage])
                    sched_p.next_tile()

            else:  # ---- consumer: 8 warps, 16 rows each ----
                if consumer_regs > 0:
                    T.set_max_nreg(consumer_regs, 1)
                acc_s = T.alloc_fragment((block_M, block_N), accum)
                G = T.alloc_fragment((block_M, n_groups, 16), accum)
                smax = T.alloc_fragment((block_M, n_groups), accum)
                absmax = T.alloc_fragment((block_M, n_groups), accum)
                coff = T.alloc_fragment((block_M, n_groups), accum)
                gs = T.alloc_fragment((block_M, n_groups, 2), accum)  # in-lane group half sums
                Pw = T.alloc_fragment((block_M, block_N // 8), T.uint32)
                Qw = T.alloc_fragment((block_M, dim // 8), T.uint32)  # Q words; layout set by the QK GEMM
                SFQw = T.alloc_fragment((block_M, sf_words_qk), T.uint32)  # Q scale words in registers
                Ev = T.alloc_fragment((block_M, n_groups, 8), accum)
                Od = T.alloc_fragment((block_M, n_groups, 8), accum)
                acc_o = T.alloc_fragment((block_M, dim), accum)
                m_i = T.alloc_fragment((block_M,), accum)
                m_prev = T.alloc_fragment((block_M,), accum)
                ms = T.alloc_fragment((block_M,), accum)
                rescale = T.alloc_fragment((block_M,), accum)
                lacc = T.alloc_fragment((block_M, 2, 2), accum)  # per-lane running row sums
                lsum_p = T.alloc_fragment((block_M, 2), accum)
                l_i = T.alloc_fragment((block_M,), accum)
                Ev1 = T.view(Ev, (block_M, n_groups, 8))
                Ev2 = T.view(Ev, (block_M, n_groups, 8))
                Ev3 = T.view(Ev, (block_M, n_groups, 8))
                Od1 = T.view(Od, (block_M, n_groups, 8))
                Od2 = T.view(Od, (block_M, n_groups, 8))
                Od3 = T.view(Od, (block_M, n_groups, 8))
                T.annotate_layout({G: g_layout, Ev: ph_layout, Od: ph_layout, gs: gs_layout, lacc: lacc_layout})

                sched_c = T.PersistentTileScheduler(seq_len // block_M, heads * batch, name="sc")
                sched_c.init(block_id)
                while sched_c.valid():
                    bx = sched_c.m_idx
                    by = sched_c.n_idx % heads
                    bz = sched_c.n_idx // heads
                    wave = sched_c.current_iter
                    T.fill(acc_o, 0)
                    T.fill(acc_s, 0)
                    T.fill(Pw, 0)
                    T.fill(lacc, 0)
                    for i in T.Parallel(block_M):
                        m_i[i] = -T.infinity(accum)
                    qs = wave % q_stages
                    qp = (wave // q_stages) & 1
                    T.barrier_wait(q_ready[qs], qp)
                    T.copy(Q_words[qs, :, :], Qw)  # Q into registers once per q tile (the original does the same)
                    T.copy(SFQ_sh[qs, :, :], SFQw)  # and its scale words (layout set by the QK GEMM)

                    for kt in T.serial(n_kv_blocks):
                        phase = wave * n_kv_blocks + kt
                        stage = phase % num_stages
                        parity = (phase // num_stages) & 1
                        T.barrier_wait(loaded[stage], parity)
                        qk_seed(DS_sh, stage, acc_s)
                        qk_gemm(Qw, K_sh[stage, :, :], SFQw, SFK_sh[stage, :, :], acc_s)
                        softmax_quant(
                            True,
                            DS_sh,
                            V_sh[stage, :, :],
                            SFV_sh[stage, :, :],
                            SFP_sh,
                            SFP_u8,
                            acc_s,
                            G,
                            smax,
                            absmax,
                            coff,
                            gs,
                            Pw,
                            Ev,
                            Od,
                            Ev1,
                            Ev2,
                            Ev3,
                            Od1,
                            Od2,
                            Od3,
                            acc_o,
                            m_i,
                            m_prev,
                            ms,
                            rescale,
                            lacc,
                        )
                        T.barrier_arrive(consumed[stage])

                    T.reduce_sum(lacc, lsum_p, dim=2, clear=True)
                    T.reduce_sum(lsum_p, l_i, dim=1, clear=True)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = acc_o[i, j] / l_i[i]
                    for sb in T.serial(dim // store_block_N):
                        T.copy(acc_o[:, sb * store_block_N : (sb + 1) * store_block_N], O_sh)
                        T.copy(O_sh, O[bz, by, bx * block_M, sb * store_block_N])
                    T.barrier_arrive(q_free[qs])  # this Q slot is free for a later wave
                    sched_c.next_tile()

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
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--threads", type=int, default=384)
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
