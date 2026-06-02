"""TileLang DSL 4-role split implementation of avo/kernels/attention_kernel_1sm.cu.

This file is the primary 1SM parity target. It always builds the split
correction path with the MMA role expressed in DSL and outlined as a
__device__ __noinline__ helper; legacy non-split and monolithic-helper
variants live in sibling files.
"""

import argparse
from typing import Optional

import torch
import tilelang
import tilelang.layout
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Pass config matches the SM100 wasp examples already in this directory.
# TL_DISABLE_WARP_SPECIALIZED=False lets tilelang auto-detect the if-on-tx
# branches and emit per-role setmaxnreg (currently runtime PTX only; Slice 2
# adds the __maxnreg__ launch attribute and JIT setter).
# --------------------------------------------------------------------------- #
PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    "tl.disable_thread_storage_sync": True,
    # Outline each warp-role branch into a separate __device__ __noinline__
    # function. Each device fn gets its own register-allocation budget from
    # ptxas; this is required for the 4-role split DSL path to stay spill-free.
    "tl.outline_warp_spec_branches": True,
}

ATTENTION_1SM_EXTERN_SOURCE = r"""
#include <tl_templates/cuda/copy.h>

namespace tl {

__device__ __forceinline__ void
tcgen05_softmax_pack_4(uint32_t &h0, uint32_t &h1, float const *sv,
                       float &psa0, float &psa1, int elem_base) {
  constexpr int kBlockN = 128;
  constexpr int kEx2EmuFreq = 10;
  constexpr int kEx2EmuRes = 4;
  constexpr int kEx2EmuStartFrg = 1;
  constexpr int kEx2FragSize = 32;
  bfloat16_t h[4];
  #pragma unroll
  for (int i = 0; i < 4; i += 2) {
    int elem = elem_base + i;
    float p0, p1;
    int frag = elem / kEx2FragSize;
    int k_in_frag = elem % kEx2FragSize;
    if (kEx2EmuFreq > 0 && frag >= kEx2EmuStartFrg &&
        frag < (kBlockN / kEx2FragSize - 1) &&
        (k_in_frag % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
      tl::tcgen05_exp2_poly_2(p0, p1, sv[i], sv[i + 1]);
    } else {
      p0 = tl::tcgen05_exp2f_approx(sv[i]);
      p1 = tl::tcgen05_exp2f_approx(sv[i + 1]);
    }
    if (i == 0) {
      psa0 += p0 + p1;
    } else {
      psa1 += p0 + p1;
    }
    h[i] = bfloat16_t(p0);
    h[i + 1] = bfloat16_t(p1);
  }
  h0 = *reinterpret_cast<uint32_t *>(&h[0]);
  h1 = *reinterpret_cast<uint32_t *>(&h[2]);
}

}  // namespace tl
"""


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG, target="cuda -arch=sm_100")
def attention_kernel_1sm(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    num_kv_heads: Optional[int] = None,
    is_causal: bool = False,
    block_M: int = 128,
    block_N: int = 128,
):
    """Build the 4-role split-correction DSL prim_func.

    Role map at threads=512:
        tx in [  0, 128): softmax WG 0
        tx in [128, 256): softmax WG 1
        tx in [256, 384): correction + epilogue WG
        tx in [384, 416): MMA issue WG
        tx in [416, 448): producer/TMA WG
    """
    if num_kv_heads is None:
        num_kv_heads = heads
    if heads % num_kv_heads != 0:
        raise ValueError(f"heads={heads} must be divisible by num_kv_heads={num_kv_heads}")
    groups = heads // num_kv_heads


    # softmax scale baked with log2(e) so exp2 can be used directly
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, num_kv_heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    if dim != block_N:
        raise ValueError(
            f"attention_kernel_1sm split DSL requires dim == block_N; got "
            f"dim={dim}, block_N={block_N}"
        )

    # Avo 1SM split path covers two 128-row Q tiles per CTA.
    kq2_threads = 512
    kq2_block_M = block_M
    kq2_total_M = 2 * kq2_block_M

    # ============================================================
    # 4-role split: correction WG2 separate from mma+TMA+epi WG3.
    # ============================================================
    # Same TMEM / SMEM layout as main_kq2, but redistributes work:
    #   tid [  0, 128): math WG 0 — softmax
    #   tid [128, 256): math WG 1 — softmax
    #   tid [256, 384): correction WG — only the chunked O rescale
    #   tid [384, 512): mma + TMA + epi WG — TMA loads, both gemms,
    #                                         post-loop TMA store
    #
    # New mbarriers:
    #   mbar_corr0, mbar_corr1 (count=4, lane0 arrives from
    #                           correction's 4 warps) — correction →
    #                           mma signal: O is rescaled, safe to
    #                           accumulate next PV into it.
    #
    # The mma -> correction handshake uses an avo-style PV commit barrier:
    # the MMA WG commits once after both O0/O1 PV MMAs, and correction waits
    # on that single completion before reading O TMEM.
    @T.macro
    def pv_mma_128x64_dsl(
        b_smem_ptr,
        a_tmem_addr,
        c_tmem_addr,
        accumulate,
    ):
        b_base = T.tcgen05_smem_base_16b(b_smem_ptr)
        idesc = (
            T.uint32(1 << 4)
            | T.uint32(1 << 7)
            | T.uint32(1 << 10)
            | T.uint32(1 << 16)
            | T.uint32((64 // 8) << 17)
            | T.uint32((128 // 16) << 24)
        )
        b_hi = block_N * 64 * 2
        T.fence_proxy_async()
        for j in T.unroll(0, 128, 16):
            T.tcgen05_mma_ts(
                a_tmem_addr + j // 2,
                T.tcgen05_mk_fast_desc(b_base, b_hi + j * 64 * 2),
                c_tmem_addr + 64,
                idesc,
                T.Select(j == 0, accumulate, T.uint32(1)),
                use_mask=False,
                elect_one=False,
            )
        for j in T.unroll(0, 128, 16):
            T.tcgen05_mma_ts(
                a_tmem_addr + j // 2,
                T.tcgen05_mk_fast_desc(b_base, j * 64 * 2),
                c_tmem_addr,
                idesc,
                T.Select(j == 0, accumulate, T.uint32(1)),
                use_mask=False,
                elect_one=False,
            )

    @T.macro
    def qk_mma_128x128_dsl(
        a_smem_ptr,
        b_smem_ptr,
        c_tmem_addr,
        mbar,
    ):
        a_base = T.tcgen05_smem_base_16b(a_smem_ptr)
        b_base = T.tcgen05_smem_base_16b(b_smem_ptr)
        idesc = (
            T.uint32(1 << 4)
            | T.uint32(1 << 7)
            | T.uint32(1 << 10)
            | T.uint32((128 // 8) << 17)
            | T.uint32((128 // 16) << 24)
        )
        first = T.alloc_var(T.uint32, 1)
        for t in T.unroll(2):
            q_off = t * block_M * 64 * 2
            k_off = t * block_N * 64 * 2
            for j in T.unroll(0, 64, 16):
                T.tcgen05_mma_ss(
                    T.tcgen05_mk_fast_desc(a_base, q_off + j * 2),
                    T.tcgen05_mk_fast_desc(b_base, k_off + j * 2),
                    c_tmem_addr,
                    idesc,
                    T.Select(first == 1, T.uint32(0), T.uint32(1)),
                    use_mask=False,
                    elect_one=False,
                )
                first = 0
        T.tcgen05_commit_1sm(mbar, lane0=True)

    @T.macro
    def softmax_warp_dsl(
        S_tmem: T.Buffer,
        P_tmem: T.Buffer,
        scores_scale_shared: T.SharedBuffer([2, kq2_block_M], accum_dtype),
        logsum_shared: T.SharedBuffer([kq2_block_M], accum_dtype),
        mbar_s: T.Buffer,
        mbar_scale: T.Buffer,
        mbar_p2: T.Buffer,
        mbar_p: T.Buffer,
        loop_extent: T.int32,
        q_row_base: T.int32,
        warp_group_offset: T.int32,
        tid_arg: T.int32,
    ):
        row = tid_arg - warp_group_offset
        warp_row_base = (((row >> 5) & 3) * 32) << 16
        sv = T.alloc_local((128,), accum_dtype)
        psa = T.alloc_local((4,), accum_dtype)
        rmax_local = T.alloc_var(accum_dtype, -T.infinity(accum_dtype))
        rsum_local = T.alloc_var(accum_dtype, 0.0)
        nm = T.alloc_var(accum_dtype)
        rs = T.alloc_var(accum_dtype)
        neg_max_scaled = T.alloc_var(accum_dtype)
        m0 = T.alloc_var(accum_dtype)
        m1 = T.alloc_var(accum_dtype)
        m2 = T.alloc_var(accum_dtype)
        h0 = T.alloc_var(T.uint32)
        h1 = T.alloc_var(T.uint32)
        h2 = T.alloc_var(T.uint32)
        h3 = T.alloc_var(T.uint32)

        for k_soft in T.unroll(loop_extent, explicit=False, unroll_factor=1):
            phase = k_soft & 1
            kv_col_base = k_soft * block_N
            T.mbarrier_wait_parity(mbar_s, phase)
            T.tcgen05_after_thread_sync()

            nm = rmax_local
            for cc in T.unroll(0, 128, 16):
                T.tcgen05_ld(
                    32,
                    16,
                    False,
                    S_tmem[0, 0],
                    warp_row_base + cc,
                    T.access_ptr(sv[cc], "w", 16),
                    emit_fence=False,
                )
            T.tcgen05_fence_tmem_load()

            if is_causal:
                for i in T.unroll(128):
                    sv[i] = T.if_then_else(
                        (kv_col_base + i < seq_len) & (kv_col_base + i <= q_row_base + row),
                        sv[i],
                        -T.infinity(accum_dtype),
                    )
            else:
                remaining = seq_len - kv_col_base
                if remaining < block_N:
                    for i in T.unroll(128):
                        if i >= remaining:
                            sv[i] = -T.infinity(accum_dtype)

            m0 = T.max3(nm, sv[0], sv[1])
            m1 = T.max3(sv[2], sv[3], sv[4])
            m2 = T.max3(sv[5], sv[6], sv[7])
            for i in T.unroll(8, 128, 8):
                m0 = T.max3(m0, sv[i + 0], sv[i + 1])
                m1 = T.max3(m1, sv[i + 2], sv[i + 3])
                m2 = T.max3(m2, sv[i + 4], sv[i + 5])
                nm = T.max3(nm, sv[i + 6], sv[i + 7])
            nm = T.max3(T.fmax2_ftz(m0, m1), m2, nm)

            T.tcgen05_softmax_rescale_update(rs, rmax_local, rsum_local, nm, scale)
            scores_scale_shared[phase, row] = rs
            T.sync_warp()
            T.mbarrier_arrive(T.mbarrier_at(mbar_scale, phase), lane0=True)

            neg_max_scaled = -(rmax_local * scale)
            for cc in T.unroll(0, 128, 16):
                for i in T.unroll(0, 16, 2):
                    T.tcgen05_fma_f32x2(
                        sv[cc + i],
                        sv[cc + i + 1],
                        sv[cc + i],
                        sv[cc + i + 1],
                        scale,
                        scale,
                        neg_max_scaled,
                        neg_max_scaled,
                    )

            for i in T.unroll(4):
                psa[i] = 0.0

            for chunk in T.unroll(8):
                cc = 112 - chunk * 16
                for g_iter in T.unroll(2):
                    g = 8 - g_iter * 8
                    elem_base = cc + g
                    T.call_extern(
                        "void",
                        "tl::tcgen05_softmax_pack_4",
                        h0,
                        h1,
                        T.access_ptr(sv[elem_base], "r", 4),
                        psa[0],
                        psa[1],
                        elem_base,
                    )
                    T.call_extern(
                        "void",
                        "tl::tcgen05_softmax_pack_4",
                        h2,
                        h3,
                        T.access_ptr(sv[elem_base + 4], "r", 4),
                        psa[2],
                        psa[3],
                        elem_base + 4,
                    )
                    T.tcgen05_st_32x32b_x4(
                        P_tmem[0, 0],
                        warp_row_base + (cc + g) // 2,
                        h0,
                        h1,
                        h2,
                        h3,
                    )
                if cc == 32:
                    T.tcgen05_fence_tmem_store()
                    T.mbarrier_arrive(mbar_p2, lane0=True)

            rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3])
            if k_soft == loop_extent - 1:
                logsum_shared[row] = rsum_local

            T.tcgen05_fence_tmem_store()
            T.mbarrier_arrive(mbar_p, lane0=True)


    @T.prim_func
    def main_kq2_split(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, kq2_total_M),
            heads,
            batch,
            threads=kq2_threads,
            cluster_dims=1,
            prelude=ATTENTION_1SM_EXTERN_SOURCE,
        ) as (bx, by, bz):
            T.annotate_min_blocks_per_sm(1)
            Q0_shared = T.alloc_shared([kq2_block_M, dim], dtype)
            Q1_shared = T.alloc_shared([kq2_block_M, dim], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            KV_shared_2 = T.alloc_shared([block_N, dim], dtype)
            O0_shared = T.alloc_shared([kq2_block_M, dim], dtype)
            O1_shared = T.alloc_shared([kq2_block_M, dim], dtype)
            logsum0_shared = T.alloc_shared([kq2_block_M], accum_dtype)
            logsum1_shared = T.alloc_shared([kq2_block_M], accum_dtype)

            S0_tmem = T.alloc_tmem([kq2_block_M, block_N], accum_dtype)
            P0_tmem = T.alloc_tmem(
                [kq2_block_M, block_N], dtype,
                alias=S0_tmem, col_offset=block_N // 2,
            )
            S1_tmem = T.alloc_tmem([kq2_block_M, block_N], accum_dtype)
            P1_tmem = T.alloc_tmem(
                [kq2_block_M, block_N], dtype,
                alias=S1_tmem, col_offset=block_N // 2,
            )
            O0_tmem = T.alloc_tmem([kq2_block_M, dim], accum_dtype)
            O1_tmem = T.alloc_tmem([kq2_block_M, dim], accum_dtype)
            pv_o_layout = T.Layout(
                [kq2_block_M, dim],
                lambda i, j: [i, j],
            )
            pv_p_layout = T.Layout(
                [kq2_block_M, block_N],
                lambda i, j: [i, j],
            )
            T.annotate_layout({
                S0_tmem: pv_p_layout,
                S1_tmem: pv_p_layout,
                O0_tmem: pv_o_layout,
                O1_tmem: pv_o_layout,
                P0_tmem: pv_p_layout,
                P1_tmem: pv_p_layout,
                Q0_shared: tilelang.layout.make_full_bank_swizzled_layout(Q0_shared),
                Q1_shared: tilelang.layout.make_full_bank_swizzled_layout(Q1_shared),
                K_shared_0: tilelang.layout.make_full_bank_swizzled_layout(K_shared_0),
                K_shared_1: tilelang.layout.make_full_bank_swizzled_layout(K_shared_1),
                KV_shared_2: tilelang.layout.make_full_bank_swizzled_layout(KV_shared_2),
            })

            mbar_s0 = T.alloc_barrier(1)
            mbar_s1 = T.alloc_barrier(1)
            # Full-P and partial-P ready signals. Match avo's mb_p/mb_p2:
            # one arrive per softmax warp, issued by lane 0 after P TMEM
            # stores finish and after the halfway point respectively.
            mbar_p0 = T.alloc_barrier(4)
            mbar_p1 = T.alloc_barrier(4)
            mbar_p2_0 = T.alloc_barrier(4)
            mbar_p2_1 = T.alloc_barrier(4)
            mbar_pv = T.alloc_barrier(1)
            # Softmax -> correction scale handoff. Match avo's mb_rs:
            # each of the 4 softmax warps arrives once via lane 0 after
            # writing its 32 row scale values.
            mbar_scale0 = T.alloc_barrier([4, 4])
            mbar_scale1 = T.alloc_barrier([4, 4])
            # Correction -> mma WG. Match avo: one arrive per correction
            # warp, issued by lane 0, after the per-warp x16 TMEM stores.
            mbar_corr0 = T.alloc_barrier(4)
            mbar_corr1 = T.alloc_barrier(4)
            mbar_epi0 = T.alloc_barrier(4)
            mbar_epi1 = T.alloc_barrier(4)
            mbar_q0_load = T.alloc_barrier(1)
            mbar_q1_load = T.alloc_barrier(1)
            mbar_k_load = T.alloc_barrier([1, 1])
            mbar_v_lo_load = T.alloc_barrier([1, 1])
            mbar_v_hi_load = T.alloc_barrier([1, 1])

            scores_scale0_shared = T.alloc_shared([2, kq2_block_M], accum_dtype)
            scores_scale1_shared = T.alloc_shared([2, kq2_block_M], accum_dtype)

            tid = T.get_thread_binding()
            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * kq2_total_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            # Avo-style register donation for the 4-role split:
            #   warps 0-7   softmax    184 regs
            #   warps 8-11  correction  64 regs
            #   warps 12-15 mma/prod/epi/idle 80 regs
            # Avo uses __launch_bounds__(512, 1). Empirically this still
            # preserves the per-role setmaxnreg donation while giving ptxas
            # a slightly better scheduling target than minBlocks=0.
            if tid < 256:
                T.set_max_nreg(184, 1)
            elif tid < 384:
                T.set_max_nreg(64, 0)
            else:
                T.set_max_nreg(80, 0)

            # The DSL softmax body owns the full KV loop internally, like
            # avo's softmax_warp_fn, but exposes the readable loop/max/
            # rescale/store structure instead of hiding it in one helper.
            if tid < 128:
                softmax_warp_dsl(
                    S0_tmem,
                    P0_tmem,
                    scores_scale0_shared,
                    logsum0_shared,
                    mbar_s0,
                    mbar_scale0,
                    mbar_p2_0,
                    mbar_p0,
                    loop_range,
                    bx * kq2_total_M,
                    0,
                    tid,
                )
            elif tid < 256:
                softmax_warp_dsl(
                    S1_tmem,
                    P1_tmem,
                    scores_scale1_shared,
                    logsum1_shared,
                    mbar_s1,
                    mbar_scale1,
                    mbar_p2_1,
                    mbar_p1,
                    loop_range,
                    bx * kq2_total_M + kq2_block_M,
                    128,
                    tid,
                )

            # ================================================================
            # Producer warp 13  --  tid in [416, 448)
            # ================================================================
            if tid >= 416 and tid < 448:
                k_desc = T.create_tma_descriptor(
                    9, 4, T.access_ptr(K, "r"),
                    dim, num_kv_heads, seq_len, batch,
                    2, dim * 2, num_kv_heads * dim * 2,
                    seq_len * num_kv_heads * dim * 2,
                    64, 1, 128, 1,
                    1, 1, 1, 1,
                    0, 3, 2, 0,
                )
                v_desc = T.create_tma_descriptor(
                    9, 4, T.access_ptr(V, "r"),
                    dim, num_kv_heads, seq_len, batch,
                    2, dim * 2, num_kv_heads * dim * 2,
                    seq_len * num_kv_heads * dim * 2,
                    64, 1, 128, 1,
                    1, 1, 1, 1,
                    0, 3, 2, 0,
                )
                with T.device_func():
                    if (T.get_thread_binding() & 31) == 0:
                        k_stage0 = T.access_ptr(K_shared_0, "w")
                        k_stage1 = T.access_ptr(K_shared_1, "w")
                        kv_stage2 = T.access_ptr(KV_shared_2, "w")

                        T.tma_copy(
                            Q[
                                bz,
                                bx * kq2_total_M : bx * kq2_total_M + kq2_block_M,
                                by,
                                :,
                            ],
                            Q0_shared,
                            barrier=mbar_q0_load,
                            annotations={"emit_arrive": 1},
                        )
                        T.tma_copy(
                            Q[
                                bz,
                                bx * kq2_total_M + kq2_block_M : (bx + 1) * kq2_total_M,
                                by,
                                :,
                            ],
                            Q1_shared,
                            barrier=mbar_q1_load,
                            annotations={"emit_arrive": 1},
                        )

                        for k_prefetch in T.unroll(3):
                            if k_prefetch < loop_range:
                                prefetch_stage = k_prefetch % 3
                                stage_ptr = T.tcgen05_reuse3_stage_ptr(
                                    k_stage0,
                                    k_stage1,
                                    kv_stage2,
                                    prefetch_stage,
                                )
                                mbar = T.tcgen05_reuse3_barrier_ref(
                                    mbar_k_load[0],
                                    mbar_k_load[1],
                                    mbar_v_hi_load[0],
                                    prefetch_stage,
                                )
                                T.tcgen05_mbarrier_arrive_expect_tx_ref(
                                    mbar,
                                    block_N * dim * 2,
                                )
                                T.tma_load(
                                    k_desc,
                                    mbar,
                                    stage_ptr,
                                    0,
                                    by // groups,
                                    k_prefetch * block_N,
                                    bz,
                                    0,
                                )
                                T.tma_load(
                                    k_desc,
                                    mbar,
                                    T.tcgen05_smem_ptr_add_bf16(stage_ptr, block_N * 64),
                                    64,
                                    by // groups,
                                    k_prefetch * block_N,
                                    bz,
                                    0,
                                )

                        for k_prod in T.unroll(loop_range, explicit=False, unroll_factor=1):
                            T.tcgen05_wait_barrier(mbar_s1, k_prod & 1)
                            T.tcgen05_after_thread_sync()
                            prod_stage = k_prod % 3
                            v_stage_ptr = T.tcgen05_reuse3_stage_ptr(
                                k_stage0,
                                k_stage1,
                                kv_stage2,
                                prod_stage,
                            )
                            v_mbar = T.tcgen05_reuse3_barrier_ref(
                                mbar_v_lo_load[0],
                                mbar_v_lo_load[1],
                                mbar_v_hi_load[1],
                                prod_stage,
                            )
                            T.tcgen05_mbarrier_arrive_expect_tx_ref(
                                v_mbar,
                                block_N * dim * 2,
                            )
                            T.tma_load(
                                v_desc,
                                v_mbar,
                                v_stage_ptr,
                                0,
                                by // groups,
                                k_prod * block_N,
                                bz,
                                0,
                            )
                            T.tma_load(
                                v_desc,
                                v_mbar,
                                T.tcgen05_smem_ptr_add_bf16(v_stage_ptr, block_N * 64),
                                64,
                                by // groups,
                                k_prod * block_N,
                                bz,
                                0,
                            )
                            if k_prod + 3 < loop_range:
                                T.tcgen05_wait_barrier(mbar_pv, k_prod & 1)
                                T.tcgen05_after_thread_sync()
                                next_k = k_prod + 3
                                next_stage = next_k % 3
                                next_stage_ptr = T.tcgen05_reuse3_stage_ptr(
                                    k_stage0,
                                    k_stage1,
                                    kv_stage2,
                                    next_stage,
                                )
                                next_mbar = T.tcgen05_reuse3_barrier_ref(
                                    mbar_k_load[0],
                                    mbar_k_load[1],
                                    mbar_v_hi_load[0],
                                    next_stage,
                                )
                                T.tcgen05_mbarrier_arrive_expect_tx_ref(
                                    next_mbar,
                                    block_N * dim * 2,
                                )
                                T.tma_load(
                                    k_desc,
                                    next_mbar,
                                    next_stage_ptr,
                                    0,
                                    by // groups,
                                    next_k * block_N,
                                    bz,
                                    0,
                                )
                                T.tma_load(
                                    k_desc,
                                    next_mbar,
                                    T.tcgen05_smem_ptr_add_bf16(next_stage_ptr, block_N * 64),
                                    64,
                                    by // groups,
                                    next_k * block_N,
                                    bz,
                                    0,
                                )

            # ================================================================
            # MMA warp 12  --  tid in [384, 416)
            # ================================================================
            elif tid >= 384 and tid < 416:
                with T.device_func():
                    tid_mma = T.get_thread_binding()
                    if (tid_mma & 31) == 0:
                        q0_ptr = T.access_ptr(Q0_shared, "r")
                        q1_ptr = T.access_ptr(Q1_shared, "r")
                        k0_ptr = T.access_ptr(K_shared_0, "r")
                        k1_ptr = T.access_ptr(K_shared_1, "r")
                        kv2_ptr = T.access_ptr(KV_shared_2, "r")

                        T.tcgen05_wait_barrier(mbar_q0_load, 0)
                        T.tcgen05_wait_barrier(mbar_q1_load, 0)
                        T.tcgen05_after_thread_sync()

                        T.tcgen05_wait_barrier(mbar_k_load[0], 0)
                        T.tcgen05_after_thread_sync()
                        qk_mma_128x128_dsl(
                            q0_ptr,
                            k0_ptr,
                            S0_tmem[0, 0],
                            mbar_s0,
                        )
                        qk_mma_128x128_dsl(
                            q1_ptr,
                            k0_ptr,
                            S1_tmem[0, 0],
                            mbar_s1,
                        )

                        for k_mma in T.unroll(loop_range, explicit=False, unroll_factor=1):
                            stage = k_mma % 3
                            phase = (k_mma // 3) & 1
                            pv_phase = k_mma & 1
                            corr_phase = (k_mma - 1) & 1
                            accum = T.Select(k_mma == 0, T.uint32(0), T.uint32(1))
                            T.tcgen05_wait_barrier(mbar_p2_0, pv_phase)
                            T.tcgen05_wait_barrier_ref(
                                T.tcgen05_reuse3_barrier_ref(
                                    mbar_v_lo_load[0],
                                    mbar_v_lo_load[1],
                                    mbar_v_hi_load[1],
                                    stage,
                                ),
                                phase,
                            )
                            if k_mma > 0:
                                T.tcgen05_wait_barrier(mbar_corr0, corr_phase)
                            T.tcgen05_wait_barrier(mbar_p0, pv_phase)
                            T.tcgen05_after_thread_sync()
                            vptr = T.tcgen05_reuse3_stage_ptr(k0_ptr, k1_ptr, kv2_ptr, stage)
                            pv_mma_128x64_dsl(
                                vptr,
                                P0_tmem[0, 0],
                                O0_tmem[0, 0],
                                accum,
                            )

                            T.tcgen05_wait_barrier(mbar_p2_1, pv_phase)
                            if k_mma > 0:
                                T.tcgen05_wait_barrier(mbar_corr1, corr_phase)
                            T.tcgen05_wait_barrier(mbar_p1, pv_phase)
                            T.tcgen05_after_thread_sync()
                            pv_mma_128x64_dsl(
                                vptr,
                                P1_tmem[0, 0],
                                O1_tmem[0, 0],
                                accum,
                            )
                            T.tcgen05_commit_1sm(mbar_pv, lane0=True)

                            if k_mma + 1 < loop_range:
                                next_stage = (k_mma + 1) % 3
                                next_phase = ((k_mma + 1) // 3) & 1
                                T.tcgen05_wait_barrier_ref(
                                    T.tcgen05_reuse3_barrier_ref(
                                        mbar_k_load[0],
                                        mbar_k_load[1],
                                        mbar_v_hi_load[0],
                                        next_stage,
                                    ),
                                    next_phase,
                                )
                                T.tcgen05_after_thread_sync()
                                kptr = T.tcgen05_reuse3_stage_ptr(
                                    k0_ptr,
                                    k1_ptr,
                                    kv2_ptr,
                                    next_stage,
                                )
                                qk_mma_128x128_dsl(
                                    q0_ptr,
                                    kptr,
                                    S0_tmem[0, 0],
                                    mbar_s0,
                                )
                                qk_mma_128x128_dsl(
                                    q1_ptr,
                                    kptr,
                                    S1_tmem[0, 0],
                                    mbar_s1,
                                )

            # ================================================================
            # Correction WG  --  tid in [256, 384)
            # ================================================================
            elif tid >= 256 and tid < 384:
                row_corr = T.get_thread_binding() - 256
                for k_corr in T.unroll(loop_range, explicit=False, unroll_factor=1):
                    if k_corr > 0:
                        scale_idx = k_corr & 1
                        scale_phase = (k_corr >> 1) & 1
                        pv_phase = (k_corr - 1) & 1

                        T.mbarrier_wait_parity(
                            T.mbarrier_at(mbar_scale0, scale_idx),
                            scale_phase,
                        )
                        T.tcgen05_correction_x16(
                            O0_tmem[0, 0],
                            scores_scale0_shared[scale_idx, row_corr],
                            mbar_pv=mbar_pv,
                            pv_phase=pv_phase,
                            warp_group_offset=256,
                            head_dim=dim,
                        )
                        T.tcgen05_before_thread_sync()
                        T.mbarrier_arrive(mbar_corr0, lane0=True)

                        T.mbarrier_wait_parity(
                            T.mbarrier_at(mbar_scale1, scale_idx),
                            scale_phase,
                        )
                        T.tcgen05_correction_x16(
                            O1_tmem[0, 0],
                            scores_scale1_shared[scale_idx, row_corr],
                            mbar_pv=mbar_pv,
                            pv_phase=pv_phase,
                            warp_group_offset=256,
                            head_dim=dim,
                        )
                        T.tcgen05_before_thread_sync()
                        T.mbarrier_arrive(mbar_corr1, lane0=True)

                final_phase = (loop_range - 1) & 1
                T.mbarrier_wait_parity(mbar_pv, final_phase)
                T.tcgen05_after_thread_sync()
                T.tcgen05_epilogue_store_x16(
                    O0_tmem[0, 0],
                    T.access_ptr(O0_shared, "w"),
                    logsum0_shared[row_corr],
                    warp_group_offset=256,
                    head_dim=dim,
                )
                T.tcgen05_before_thread_sync()
                T.sync_warp()
                T.fence_proxy_async()
                T.mbarrier_arrive(mbar_epi0, lane0=True)

                T.tcgen05_after_thread_sync()
                T.tcgen05_epilogue_store_x16(
                    O1_tmem[0, 0],
                    T.access_ptr(O0_shared, "w", offset=kq2_block_M * dim),
                    logsum1_shared[row_corr],
                    warp_group_offset=256,
                    head_dim=dim,
                )
                T.tcgen05_before_thread_sync()
                T.sync_warp()
                T.fence_proxy_async()
                T.mbarrier_arrive(mbar_epi1, lane0=True)

            else:
                T.evaluate(0)

            # ---- Avo-style split epilogue TMA store ----
            if tid >= 448 and tid < 480:
                output_desc = T.create_tma_descriptor(
                    9, 4, T.access_ptr(Output, "w"),
                    dim, heads, seq_len, batch,
                    2, dim * 2, heads * dim * 2, seq_len * heads * dim * 2,
                    64, 1, 32, 1,
                    1, 1, 1, 1,
                    0, 3, 2, 0,
                )
                with T.device_func():
                    if (T.get_thread_binding() & 31) == 0:
                        T.tcgen05_wait_barrier(mbar_epi0, 0)
                        T.fence_proxy_async()
                        for cw in T.unroll(4):
                            T.tcgen05_epilogue_tma_store_32x128(
                                output_desc,
                                T.access_ptr(O0_shared, "r", offset=cw * 32 * dim),
                                bx * kq2_total_M + cw * 32,
                                by,
                                bz,
                            )
                        T.tma_store_arrive()

                        T.tcgen05_wait_barrier(mbar_epi1, 0)
                        T.fence_proxy_async()
                        for cw in T.unroll(4):
                            T.tcgen05_epilogue_tma_store_32x128(
                                output_desc,
                                T.access_ptr(O1_shared, "r", offset=cw * 32 * dim),
                                bx * kq2_total_M + kq2_block_M + cw * 32,
                                by,
                                bz,
                            )
                        T.tma_store_arrive()
                        T.tma_store_wait(0)

    return main_kq2_split


# --------------------------------------------------------------------------- #
# Reference + driver for smoke testing.
# --------------------------------------------------------------------------- #
def reference_attention(Q, K, V, is_causal=False):
    Q_f = Q.permute(0, 2, 1, 3).to(torch.float32)
    K_f = K.permute(0, 2, 1, 3).to(torch.float32)
    V_f = V.permute(0, 2, 1, 3).to(torch.float32)
    if Q_f.size(1) != K_f.size(1):
        if Q_f.size(1) % K_f.size(1) != 0:
            raise ValueError(
                f"q_heads={Q_f.size(1)} must be divisible by kv_heads={K_f.size(1)}"
            )
        groups = Q_f.size(1) // K_f.size(1)
        K_f = K_f.repeat_interleave(groups, dim=1)
        V_f = V_f.repeat_interleave(groups, dim=1)
    dim = Q.size(-1)
    seq_q = Q.size(1)
    seq_k = K.size(1)
    scores = (Q_f @ K_f.transpose(-1, -2)) * (1.0 / dim**0.5)
    if is_causal:
        mask = torch.tril(torch.ones(seq_q, seq_k, device=Q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    out = weights @ V_f
    return out.permute(0, 2, 1, 3).to(Q.dtype)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument(
        "--skip_ref", action="store_true",
        help="Skip PyTorch reference validation; useful for long-sequence benchmarking.",
    )
    ap.add_argument(
        "--split_correction", action="store_true", help=argparse.SUPPRESS,
    )
    ap.add_argument("--print_source", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    Q = torch.randn(args.batch, args.seq, args.heads, args.dim,
                    dtype=torch.bfloat16, device="cuda")
    kv_h = args.kv_heads or args.heads
    K = torch.randn(args.batch, args.seq, kv_h, args.dim,
                    dtype=torch.bfloat16, device="cuda")
    V = torch.randn(args.batch, args.seq, kv_h, args.dim,
                    dtype=torch.bfloat16, device="cuda")

    fn = attention_kernel_1sm(
        args.batch,
        args.heads,
        args.seq,
        args.dim,
        num_kv_heads=kv_h,
        is_causal=args.causal,
    )
    if args.print_source:
        print(fn.get_kernel_source())

    O = fn(Q, K, V)
    if args.skip_ref:
        print(f"shape={tuple(O.shape)}  correctness=skipped")
    else:
        O_ref = reference_attention(Q, K, V, is_causal=args.causal)
        err_abs = (O.to(torch.float32) - O_ref.to(torch.float32)).abs()
        print(
            f"shape={tuple(O.shape)}  "
            f"max_abs={err_abs.max().item():.4f}  "
            f"mean_abs={err_abs.mean().item():.4f}"
        )

    if args.bench:
        from tilelang.profiler import do_bench
        for _ in range(3):
            _ = fn(Q, K, V)
        torch.cuda.synchronize()
        lat = do_bench(lambda: fn(Q, K, V), warmup=25, rep=100)
        causal_factor = 0.5 if args.causal else 1.0
        flops = 2.0 * 2.0 * args.batch * args.heads * args.seq * args.seq * args.dim * causal_factor
        tflops = flops / lat * 1e-9
        print(f"latency={lat:.3f} ms  perf={tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
