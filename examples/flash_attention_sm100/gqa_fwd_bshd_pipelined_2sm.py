"""TileLang 2SM FlashAttention kernels for SM100.

This is the second CUDA-parity target.  The TileLang DSL owns the launch,
persistent cluster tile loop, memory/barrier topology, role dispatch, and role
schedules.  Small TCGEN05/TMA primitives cover the instruction forms that the
kernel needs directly.
"""

import argparse
from typing import Optional

import torch
import tilelang
import tilelang.layout
import tilelang.language as T
from tilelang.carver.arch import driver


PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    "tl.disable_thread_storage_sync": True,
    "tl.outline_warp_spec_branches": True,
}

ATTENTION_2SM_EXTERN_SOURCE = r"""
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

template <int HeadDim>
__device__ __forceinline__ void
tcgen05_tmem_rescale_row_x16_addr(uint32_t O_base, uint32_t col_offset,
                                  uint32_t row_offset, float rs) {
  uint32_t base = O_base + col_offset + row_offset;
  float buf[2][16];
  int cur = 0;
  tl::tmem_ld_32dp32bNx<false>::copy<16>(base, (uint32_t *)buf[cur]);
  #pragma unroll
  for (int g = 0; g < HeadDim / 16; ++g) {
    tl::fence_view_async_tmem_load();
    int nxt = cur ^ 1;
    if (g + 1 < HeadDim / 16) {
      tl::tmem_ld_32dp32bNx<false>::copy<16>(base + (g + 1) * 16,
                                              (uint32_t *)buf[nxt]);
    }
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      tl::tcgen05_fma_f32x2(buf[cur][i], buf[cur][i + 1],
                            buf[cur][i], buf[cur][i + 1], rs, rs, 0.0f,
                            0.0f);
    }
    tl::tmem_st_32dp32bNx<false>::copy<16>(base + g * 16,
                                            (uint32_t *)buf[cur]);
    cur = nxt;
  }
  tl::fence_view_async_tmem_store();
}

template <int HeadDim>
__device__ __forceinline__ void
tcgen05_tmem_normalize_store_row_bf16_x16_addr(bfloat16_t *epi_stage,
                                               uint32_t O_base,
                                               uint32_t col_offset,
                                               uint32_t row_offset, int lane,
                                               float inv) {
  constexpr int kEpiBlockCols = 64;
  constexpr int kEpiBlockBytes = kEpiBlockCols * 2;
  constexpr int kEpiBlockElems = 32 * kEpiBlockCols;
  uint32_t base = O_base + col_offset + row_offset;
  char *epi_base = reinterpret_cast<char *>(epi_stage);
  char *epi_blk0 = epi_base;
  char *epi_blk1 = epi_base + kEpiBlockElems * 2;
  int row_off = lane * kEpiBlockBytes;
  int swiz = (lane & 7) << 4;

  #pragma unroll
  for (int d = 0; d < HeadDim; d += 16) {
    float t[16];
    tl::tmem_ld_32dp32bNx<false>::copy<16>(base + d, (uint32_t *)t);
    tl::fence_view_async_tmem_load();
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      tl::tcgen05_fma_f32x2(t[i], t[i + 1], t[i], t[i + 1], inv, inv, 0.0f,
                            0.0f);
    }
    bfloat16_t b[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) b[i] = bfloat16_t(t[i]);
    int d_in_blk = d % kEpiBlockCols;
    char *blk = d < kEpiBlockCols ? epi_blk0 : epi_blk1;
    int col0 = d_in_blk * 2;
    int col1 = (d_in_blk + 8) * 2;
    *reinterpret_cast<uint4 *>(blk + row_off + (col0 ^ swiz)) =
        *reinterpret_cast<uint4 *>(&b[0]);
    *reinterpret_cast<uint4 *>(blk + row_off + (col1 ^ swiz)) =
        *reinterpret_cast<uint4 *>(&b[8]);
  }
}

}  // namespace tl
"""


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG, target={"kind": "cuda", "arch": "sm_100"})
def attention_kernel_2sm_d128(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    num_kv_heads: Optional[int] = None,
    is_causal: bool = False,
):
    if dim != 128:
        raise ValueError("attention_kernel_2sm_d128 supports head_dim=128 only")
    if is_causal:
        raise ValueError("attention_kernel_2sm currently implements non-causal attention only")
    if num_kv_heads is None:
        num_kv_heads = heads
    if heads % num_kv_heads != 0:
        raise ValueError(f"heads={heads} must be divisible by num_kv_heads={num_kv_heads}")

    block_m = 256
    block_m_cta = 128
    block_n = 128
    page_rows = 32
    q_stages = 2
    kv_stages = 3
    b_per_cta = 64
    tile_cols = 64
    threads = 512
    q_rows_per_cluster = q_stages * block_m
    q_tiles = T.ceildiv(seq_len, q_rows_per_cluster)
    total_tiles = q_tiles * heads * batch
    sm_num = driver.get_num_sms()
    grid = T.max(2, (T.min(total_tiles * 2, sm_num) // 2) * 2)
    total_clusters = grid // 2
    loop_extent = T.ceildiv(seq_len, block_n)
    scale_log2 = (1.0 / dim) ** 0.5 * 1.44269504089

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, num_kv_heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    q_stage_elems = block_m_cta * dim
    kv_stage_elems = b_per_cta * dim

    @T.macro
    def producer_load_q_dsl(
        q_desc,
        q_stage,
        mbar_q,
        tile_q_col_base,
        tile_q_row_base,
    ):
        q_bytes = 2 * block_m_cta * tile_cols * 2
        for qs in T.unroll(q_stages):
            T.tcgen05_mbarrier_arrive_expect_tx_cluster_lane0_ref(
                T.mbarrier_at(mbar_q, qs),
                q_bytes,
            )
            q_row = tile_q_row_base + qs * block_m
            q_stage_base = qs * q_stage_elems
            T.tma_load_2cta_2d(
                q_desc,
                T.tcgen05_smem_ptr_add_bf16(
                    T.access_ptr(q_stage, "w"),
                    q_stage_base,
                ),
                T.mbarrier_at(mbar_q, qs),
                tile_q_col_base,
                q_row,
            )
            T.tma_load_2cta_2d(
                q_desc,
                T.tcgen05_smem_ptr_add_bf16(
                    T.access_ptr(q_stage, "w"),
                    q_stage_base + block_m_cta * tile_cols,
                ),
                T.mbarrier_at(mbar_q, qs),
                tile_q_col_base + tile_cols,
                q_row,
            )

    @T.macro
    def producer_prefetch_kv_stage_dsl(
        prefetch,
        k_desc,
        v_desc,
        k_stage,
        v_stage,
        mbar_k,
        mbar_v,
        mbar_k_rel,
        mbar_v_rel,
        tile_k_base,
        tile_kv_row_base,
        tile_kv_col_base,
        tile_v_col_base,
        cta_rank,
        loop_extent,
    ):
        kv_bytes = 2 * b_per_cta * tile_cols * 2
        if prefetch < loop_extent:
            tkb = tile_k_base + prefetch
            stage = tkb % kv_stages
            if tkb >= kv_stages:
                prev_phase = ((tkb - kv_stages) // kv_stages) & 1
                T.tcgen05_wait_barrier(
                    T.mbarrier_at(mbar_k_rel, stage),
                    prev_phase,
                )
                T.tcgen05_wait_barrier(
                    T.mbarrier_at(mbar_v_rel, stage),
                    prev_phase,
                )

            k_row = tile_kv_row_base + prefetch * block_n + cta_rank * b_per_cta
            T.tcgen05_mbarrier_arrive_expect_tx_cluster_lane0_ref(
                T.mbarrier_at(mbar_k, stage),
                kv_bytes,
            )
            for t in T.unroll(2):
                k_stage_offset = (
                    stage * kv_stage_elems + t * b_per_cta * tile_cols
                )
                T.tma_load_2cta_2d(
                    k_desc,
                    T.tcgen05_smem_ptr_add_bf16(
                        T.access_ptr(k_stage, "w"),
                        k_stage_offset,
                    ),
                    T.mbarrier_at(mbar_k, stage),
                    tile_kv_col_base + t * tile_cols,
                    k_row,
                )

            v_row = tile_kv_row_base + prefetch * block_n
            T.tcgen05_mbarrier_arrive_expect_tx_cluster_lane0_ref(
                T.mbarrier_at(mbar_v, stage),
                kv_bytes,
            )
            for t in T.unroll(2):
                v_stage_offset = (
                    stage * kv_stage_elems + t * b_per_cta * tile_cols
                )
                T.tma_load_2cta_2d(
                    v_desc,
                    T.tcgen05_smem_ptr_add_bf16(
                        T.access_ptr(v_stage, "w"),
                        v_stage_offset,
                    ),
                    T.mbarrier_at(mbar_v, stage),
                    tile_v_col_base,
                    v_row + t * b_per_cta,
                )

    @T.macro
    def producer_refill_kv_stage_dsl(
        kb,
        k_desc,
        v_desc,
        k_stage,
        v_stage,
        mbar_k,
        mbar_v,
        mbar_k_rel,
        mbar_v_rel,
        tile_k_base,
        tile_kv_row_base,
        tile_kv_col_base,
        tile_v_col_base,
        cta_rank,
        loop_extent,
    ):
        kv_bytes = 2 * b_per_cta * tile_cols * 2
        tkb = tile_k_base + kb
        stage = tkb % kv_stages
        phase = (tkb // kv_stages) & 1
        if kb + kv_stages < loop_extent:
            next_idx = kb + kv_stages
            next_tkb = tile_k_base + next_idx
            next_stage = next_tkb % kv_stages
            T.tcgen05_wait_barrier(
                T.mbarrier_at(mbar_k_rel, stage),
                phase,
            )
            k_row = tile_kv_row_base + next_idx * block_n + cta_rank * b_per_cta
            T.tcgen05_mbarrier_arrive_expect_tx_cluster_lane0_ref(
                T.mbarrier_at(mbar_k, next_stage),
                kv_bytes,
            )
            for t in T.unroll(2):
                k_stage_offset = (
                    next_stage * kv_stage_elems + t * b_per_cta * tile_cols
                )
                T.tma_load_2cta_2d(
                    k_desc,
                    T.tcgen05_smem_ptr_add_bf16(
                        T.access_ptr(k_stage, "w"),
                        k_stage_offset,
                    ),
                    T.mbarrier_at(mbar_k, next_stage),
                    tile_kv_col_base + t * tile_cols,
                    k_row,
                )

        if kb + kv_stages < loop_extent:
            next_idx = kb + kv_stages
            next_tkb = tile_k_base + next_idx
            next_stage = next_tkb % kv_stages
            T.tcgen05_wait_barrier(
                T.mbarrier_at(mbar_v_rel, stage),
                phase,
            )
            v_row = tile_kv_row_base + next_idx * block_n
            T.tcgen05_mbarrier_arrive_expect_tx_cluster_lane0_ref(
                T.mbarrier_at(mbar_v, next_stage),
                kv_bytes,
            )
            for t in T.unroll(2):
                v_stage_offset = (
                    next_stage * kv_stage_elems + t * b_per_cta * tile_cols
                )
                T.tma_load_2cta_2d(
                    v_desc,
                    T.tcgen05_smem_ptr_add_bf16(
                        T.access_ptr(v_stage, "w"),
                        v_stage_offset,
                    ),
                    T.mbarrier_at(mbar_v, next_stage),
                    tile_v_col_base,
                        v_row + t * b_per_cta,
                    )

    @T.macro
    def uma_softmax_role_dsl(
        s_tmem_addr,
        p_tmem_addr,
        rs_stage,
        mbar_s_current,
        mbar_p_current,
        mbar_p2_current,
        loop_extent,
        seq_len,
        tile_k_base,
        softmax_scale_log2,
        qs,
        tid,
        warp,
        cta_rank,
    ):
        sm_tid = tid - T.Select(qs == 1, 128, 0)
        sm_bar = qs * 4 + (warp & 3)
        tmem_row_offset = (cta_rank * block_m_cta + (warp & 3) * 32) * 65536

        sv = T.alloc_local((128,), accum_dtype)
        psa = T.alloc_local((4,), accum_dtype)
        rmax_local = T.alloc_var(accum_dtype, -T.infinity(accum_dtype))
        rsum_local = T.alloc_var(accum_dtype, 0.0)
        m0 = T.alloc_var(accum_dtype)
        m1 = T.alloc_var(accum_dtype)
        m2 = T.alloc_var(accum_dtype)
        m3 = T.alloc_var(accum_dtype)
        block_max = T.alloc_var(accum_dtype)
        new_max = T.alloc_var(accum_dtype)
        rs = T.alloc_var(accum_dtype)
        acc_scale_log2 = T.alloc_var(accum_dtype)
        neg_max_scaled = T.alloc_var(accum_dtype)

        for kb in T.unroll(loop_extent, explicit=False, unroll_factor=1):
            tkb = tile_k_base + kb
            T.tcgen05_wait_barrier(mbar_s_current, tkb & 1)
            T.tcgen05_after_thread_sync()

            for cc in T.unroll(0, block_n, 32):
                T.tcgen05_ld(
                    32,
                    32,
                    False,
                    s_tmem_addr,
                    tmem_row_offset + cc,
                    T.access_ptr(sv[cc], "w", 32),
                    emit_fence=False,
                )
            T.tcgen05_fence_tmem_load()

            remaining = seq_len - kb * block_n
            if remaining < block_n:
                for i in T.unroll(block_n):
                    if i >= remaining:
                        sv[i] = -T.infinity(accum_dtype)

            m0 = T.max3(sv[0], sv[1], sv[2])
            m1 = T.max3(sv[3], sv[4], sv[5])
            m2 = T.max3(sv[6], sv[7], sv[8])
            m3 = sv[9]
            for i in T.unroll(10, block_n, 8):
                m0 = T.max3(m0, sv[i + 0], sv[i + 1])
                m1 = T.max3(m1, sv[i + 2], sv[i + 3])
                m2 = T.max3(m2, sv[i + 4], sv[i + 5])
                m3 = T.max3(m3, sv[i + 6], sv[i + 7])
            block_max = T.fmax2_ftz(
                T.fmax2_ftz(m0, m1),
                T.fmax2_ftz(m2, m3),
            )
            new_max = T.fmax2_ftz(rmax_local, block_max)
            if new_max == -T.infinity(accum_dtype):
                new_max = 0.0
            rs = 1.0
            if kb == 0:
                rmax_local = new_max
            else:
                acc_scale_log2 = (rmax_local - new_max) * softmax_scale_log2
                if acc_scale_log2 < -8.0:
                    rs = T.tcgen05_exp2f_approx(acc_scale_log2)
                    rmax_local = new_max
            rsum_local *= rs

            rs_stage[tkb & 1, qs, sm_tid] = rs
            T.tcgen05_bar_arrive(sm_bar, 64)

            neg_max_scaled = -(rmax_local * softmax_scale_log2)
            for cc in T.unroll(0, block_n, 16):
                for i in T.unroll(0, 16, 2):
                    T.tcgen05_fma_f32x2(
                        sv[cc + i],
                        sv[cc + i + 1],
                        sv[cc + i],
                        sv[cc + i + 1],
                        softmax_scale_log2,
                        softmax_scale_log2,
                        neg_max_scaled,
                        neg_max_scaled,
                    )

            for i in T.unroll(4):
                psa[i] = 0.0

            h0 = T.alloc_var(T.uint32)
            h1 = T.alloc_var(T.uint32)
            h2 = T.alloc_var(T.uint32)
            h3 = T.alloc_var(T.uint32)
            for cc in T.unroll(0, block_n, 16):
                for g_iter in T.unroll(2):
                    g = g_iter * 8
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
                        p_tmem_addr,
                        tmem_row_offset + (cc + g) // 2,
                        h0,
                        h1,
                        h2,
                        h3,
                    )
                if cc == 80:
                    T.tcgen05_fence_tmem_store()
                    T.tcgen05_mbarrier_arrive_cluster_all_ref(mbar_p_current)
            T.tcgen05_fence_tmem_store()
            T.tcgen05_mbarrier_arrive_cluster_all_ref(mbar_p2_current)
            rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3])

        rs_stage[0, qs, sm_tid] = rsum_local
        T.tcgen05_bar_arrive(sm_bar, 64)

    @T.macro
    def uma_qk_mma_2cta_dsl(
        q_base_16b,
        k_base_16b,
        q_off_base,
        k_off_base,
        s_tmem_addr,
        mbar_s_qs,
    ):
        idesc = (
            T.uint32(1 << 4)
            | T.uint32(1 << 7)
            | T.uint32(1 << 10)
            | T.uint32((128 // 8) << 17)
            | T.uint32((256 // 16) << 24)
        )
        first = T.alloc_var(T.uint32, 1)
        for t in T.unroll(2):
            q_off = q_off_base + t * block_m_cta * tile_cols * 2
            k_off = k_off_base + t * b_per_cta * tile_cols * 2
            for j in T.unroll(0, tile_cols, 16):
                T.tcgen05_mma_ss(
                    T.tcgen05_mk_fast_desc(q_base_16b, q_off + j * 2),
                    T.tcgen05_mk_fast_desc(k_base_16b, k_off + j * 2),
                    s_tmem_addr,
                    idesc,
                    T.Select(first == 1, T.uint32(0), T.uint32(1)),
                    cta_group=2,
                    use_mask=False,
                    elect_one=False,
                )
                first = 0
        T.tcgen05_commit_2cta(mbar_s_qs)

    @T.macro
    def uma_pv_mma_2cta_dsl(
        v_base_16b,
        v_stage,
        p_tmem_base,
        o_tmem_addr,
        accumulate,
        mbar_p2_qs,
        phase,
    ):
        idesc = (
            T.uint32(1 << 4)
            | T.uint32(1 << 7)
            | T.uint32(1 << 10)
            | T.uint32(1 << 16)
            | T.uint32((128 // 8) << 17)
            | T.uint32((256 // 16) << 24)
        )
        v_base = v_stage * kv_stage_elems * 2
        v_hi = v_base + b_per_cta * tile_cols * 2

        for j in T.unroll(0, tile_cols, 16):
            T.tcgen05_mma_ts(
                p_tmem_base + j // 2,
                T.tcgen05_mk_fast_desc(v_base_16b, v_base + j * tile_cols * 2),
                o_tmem_addr,
                idesc,
                T.Select(j == 0, accumulate, T.uint32(1)),
                cta_group=2,
                use_mask=False,
                elect_one=False,
            )
        for j in T.unroll(0, tile_cols // 2, 16):
            T.tcgen05_mma_ts(
                p_tmem_base + tile_cols // 2 + j // 2,
                T.tcgen05_mk_fast_desc(v_base_16b, v_hi + j * tile_cols * 2),
                o_tmem_addr,
                idesc,
                T.uint32(1),
                cta_group=2,
                use_mask=False,
                elect_one=False,
            )

        T.tcgen05_wait_barrier(mbar_p2_qs, phase)
        T.tcgen05_after_thread_sync()

        for j in T.unroll(tile_cols // 2, tile_cols, 16):
            T.tcgen05_mma_ts(
                p_tmem_base + tile_cols // 2 + j // 2,
                T.tcgen05_mk_fast_desc(v_base_16b, v_hi + j * tile_cols * 2),
                o_tmem_addr,
                idesc,
                T.uint32(1),
                cta_group=2,
                use_mask=False,
                elect_one=False,
            )

    @T.macro
    def uma_issue_pv_qs_dsl(
        qs,
        tkb,
        v_base_16b,
        s0_tmem_addr,
        mbar_p,
        mbar_p2,
        mbar_v,
        mbar_pv,
        mbar_corr,
        mbar_v_rel,
        wait_corr: bool,
        accumulate,
    ):
        v_stage = tkb % kv_stages
        v_phase = (tkb // kv_stages) & 1
        T.tcgen05_wait_barrier(T.mbarrier_at(mbar_p, qs), tkb & 1)
        T.tcgen05_wait_barrier(T.mbarrier_at(mbar_v, v_stage), v_phase)
        if wait_corr:
            T.tcgen05_wait_barrier(T.mbarrier_at(mbar_corr, qs), tkb & 1)
        T.tcgen05_after_thread_sync()
        uma_pv_mma_2cta_dsl(
            v_base_16b,
            v_stage,
            s0_tmem_addr + 64 + qs * 128,
            s0_tmem_addr + 256 + qs * 128,
            accumulate,
            T.mbarrier_at(mbar_p2, qs),
            tkb & 1,
        )
        if qs == q_stages - 1:
            T.tcgen05_commit_2cta(T.mbarrier_at(mbar_pv, v_stage))
            T.tcgen05_commit_2cta(T.mbarrier_at(mbar_v_rel, v_stage))

    @T.macro
    def uma_issue_next_qk_qs_dsl(
        qs,
        next_tkb,
        q_base_16b,
        k_base_16b,
        s0_tmem_addr,
        mbar_k,
        mbar_s,
        mbar_k_rel,
    ):
        next_stage = next_tkb % kv_stages
        next_phase = (next_tkb // kv_stages) & 1
        if qs == 0:
            T.tcgen05_wait_barrier(T.mbarrier_at(mbar_k, next_stage), next_phase)
            T.tcgen05_after_thread_sync()
        uma_qk_mma_2cta_dsl(
            q_base_16b,
            k_base_16b,
            qs * q_stage_elems * 2,
            next_stage * kv_stage_elems * 2,
            s0_tmem_addr + qs * 128,
            T.mbarrier_at(mbar_s, qs),
        )
        if qs == q_stages - 1:
            T.tcgen05_commit_2cta(T.mbarrier_at(mbar_k_rel, next_stage))

    @T.macro
    def uma_mma_role_dsl(
        cta_rank,
        q_stage,
        k_stage,
        v_stage,
        mbar_q,
        mbar_k,
        mbar_s,
        mbar_p,
        mbar_p2,
        mbar_v,
        mbar_pv,
        mbar_corr,
        mbar_k_rel,
        mbar_v_rel,
        mbar_q_rel,
        mbar_o_tmem_rel,
        s0_tmem_addr,
        loop_extent,
        tile_k_base,
        tile_phase,
    ):
        if cta_rank == 0:
            if T.shuffle_elect(32):
                q_base_16b = T.tcgen05_smem_base_16b(T.access_ptr(q_stage, "r"))
                k_base_16b = T.tcgen05_smem_base_16b(T.access_ptr(k_stage, "r"))
                v_base_16b = T.tcgen05_smem_base_16b(T.access_ptr(v_stage, "r"))

                for qs in T.unroll(q_stages):
                    T.tcgen05_wait_barrier(
                        T.mbarrier_at(mbar_q, qs),
                        tile_phase,
                    )
                T.tcgen05_after_thread_sync()

                kv0_stage = tile_k_base % kv_stages
                kv0_phase = (tile_k_base // kv_stages) & 1
                T.tcgen05_wait_barrier(
                    T.mbarrier_at(mbar_k, kv0_stage),
                    kv0_phase,
                )
                T.tcgen05_after_thread_sync()
                for qs in T.unroll(q_stages):
                    uma_qk_mma_2cta_dsl(
                        q_base_16b,
                        k_base_16b,
                        qs * q_stage_elems * 2,
                        kv0_stage * kv_stage_elems * 2,
                        s0_tmem_addr + qs * 128,
                        T.mbarrier_at(mbar_s, qs),
                    )
                T.tcgen05_commit_2cta(T.mbarrier_at(mbar_k_rel, kv0_stage))

                if loop_extent > 0:
                    if tile_k_base > 0:
                        T.tcgen05_wait_barrier(
                            mbar_o_tmem_rel,
                            tile_phase ^ 1,
                        )
                        T.tcgen05_after_thread_sync()
                    tkb0 = tile_k_base
                    for qs in T.unroll(q_stages):
                        uma_issue_pv_qs_dsl(
                            qs,
                            tkb0,
                            v_base_16b,
                            s0_tmem_addr,
                            mbar_p,
                            mbar_p2,
                            mbar_v,
                            mbar_pv,
                            mbar_corr,
                            mbar_v_rel,
                            False,
                            T.uint32(0),
                        )
                        if 1 < loop_extent:
                            uma_issue_next_qk_qs_dsl(
                                qs,
                                tkb0 + 1,
                                q_base_16b,
                                k_base_16b,
                                s0_tmem_addr,
                                mbar_k,
                                mbar_s,
                                mbar_k_rel,
                            )

                for kb in T.unroll(
                    1,
                    loop_extent,
                    explicit=False,
                    unroll_factor=1,
                ):
                    tkb = tile_k_base + kb
                    for qs in T.unroll(q_stages):
                        uma_issue_pv_qs_dsl(
                            qs,
                            tkb,
                            v_base_16b,
                            s0_tmem_addr,
                            mbar_p,
                            mbar_p2,
                            mbar_v,
                            mbar_pv,
                            mbar_corr,
                            mbar_v_rel,
                            True,
                            T.uint32(1),
                        )
                        if kb + 1 < loop_extent:
                            uma_issue_next_qk_qs_dsl(
                                qs,
                                tkb + 1,
                                q_base_16b,
                                k_base_16b,
                                s0_tmem_addr,
                                mbar_k,
                                mbar_s,
                                mbar_k_rel,
                            )

                T.tcgen05_commit_2cta(mbar_q_rel)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            grid,
            threads=threads,
            cluster_dims=2,
            prelude=ATTENTION_2SM_EXTERN_SOURCE,
        ) as block_id:
            # Use one block per SM while relying on runtime setmaxnreg.inc/dec
            # for role-specific register donation.
            T.annotate_min_blocks_per_sm(1)
            T.use_2cta_tmem(
                mbarrier_init_thread=416,
                tmem_alloc_warp=12,
                compact_shared_state=True,
            )

            Q_shared = T.alloc_shared([q_stages, block_m_cta, dim], dtype)
            O_shared = T.alloc_shared([q_stages, block_m_cta, dim], dtype)
            K_shared = T.alloc_shared([kv_stages, b_per_cta, dim], dtype)
            V_shared = T.alloc_shared([kv_stages, b_per_cta, dim], dtype)
            # Reuse the front of rs_smem for sum_smem after the per-kb rescale
            # loop to stay under the SM100 shared-memory cap.
            rs_shared = T.alloc_shared([2, q_stages, block_m_cta], accum_dtype)

            Base_tmem = T.alloc_tmem([block_m_cta, 512], accum_dtype)
            S0_tmem = T.alloc_tmem([block_m_cta, block_n], accum_dtype, alias=Base_tmem, col_offset=0)
            P0_tmem = T.alloc_tmem([block_m_cta, block_n], dtype, alias=Base_tmem, col_offset=64)
            S1_tmem = T.alloc_tmem([block_m_cta, block_n], accum_dtype, alias=Base_tmem, col_offset=128)
            P1_tmem = T.alloc_tmem([block_m_cta, block_n], dtype, alias=Base_tmem, col_offset=192)
            O0_tmem = T.alloc_tmem([block_m_cta, dim], accum_dtype, alias=Base_tmem, col_offset=256)
            O1_tmem = T.alloc_tmem([block_m_cta, dim], accum_dtype, alias=Base_tmem, col_offset=384)

            base_layout = T.Layout([block_m_cta, 512], lambda i, j: [i, j])
            score_layout = T.Layout([block_m_cta, block_n], lambda i, j: [i, j])
            output_layout = T.Layout([block_m_cta, dim], lambda i, j: [i, j])
            T.annotate_layout({
                Q_shared: tilelang.layout.make_full_bank_swizzled_layout(Q_shared),
                O_shared: tilelang.layout.make_full_bank_swizzled_layout(O_shared),
                K_shared: tilelang.layout.make_full_bank_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_full_bank_swizzled_layout(V_shared),
                Base_tmem: base_layout,
                S0_tmem: score_layout,
                P0_tmem: score_layout,
                S1_tmem: score_layout,
                P1_tmem: score_layout,
                O0_tmem: output_layout,
                O1_tmem: output_layout,
            })

            mb_q = T.alloc_cluster_barrier([2] * q_stages)
            mb_k = T.alloc_cluster_barrier([2] * kv_stages)
            mb_s = T.alloc_cluster_barrier([1] * q_stages)
            mb_p = T.alloc_cluster_barrier([256] * q_stages)
            mb_p2 = T.alloc_cluster_barrier([256] * q_stages)
            mb_v = T.alloc_cluster_barrier([2] * kv_stages)
            mb_k_rel = T.alloc_cluster_barrier([1] * kv_stages)
            mb_v_rel = T.alloc_cluster_barrier([1] * kv_stages)
            mb_pv = T.alloc_cluster_barrier([1] * kv_stages)
            mb_corr = T.alloc_cluster_barrier([256] * q_stages)
            mb_epi = T.alloc_barrier([block_m_cta] * q_stages)
            mb_q_rel = T.alloc_cluster_barrier(1)
            mb_o_rel = T.alloc_barrier(1)
            mb_o_tmem_rel = T.alloc_cluster_barrier(256)

            tid = T.alloc_var(T.int32, T.get_thread_binding())
            warp = T.alloc_var(T.int32, tid // 32)
            warp_group = T.alloc_var(T.int32, warp // 4)
            cta_rank = T.alloc_var(T.int32, T.block_rank_in_cluster())
            cluster_id = T.alloc_var(T.int32, T.cluster_id_x())
            T.assume(cta_rank < 2)

            for tile_iter in T.serial(T.ceildiv(total_tiles, total_clusters)):
                tile_id = tile_iter * total_clusters + cluster_id
                if tile_id < total_tiles:
                    tile_phase = tile_iter & 1
                    tile_k_base = tile_iter * loop_extent
                    tile_mb = T.truncmod(tile_id, q_tiles)
                    tile_head = T.truncmod(T.truncdiv(tile_id, q_tiles), heads)
                    tile_batch = T.truncdiv(tile_id, q_tiles * heads)
                    tile_ms = tile_mb * q_rows_per_cluster + cta_rank * block_m_cta
                    tile_kv_head = T.truncdiv(tile_head * num_kv_heads, heads)
                    tile_q_col_base = tile_head * dim
                    tile_kv_col_base = tile_kv_head * dim
                    tile_q_row_base = tile_batch * seq_len + tile_ms
                    tile_kv_row_base = tile_batch * seq_len
                    tile_v_col_base = tile_kv_col_base + cta_rank * b_per_cta
                    if warp_group == 0:
                        T.set_max_nreg(168, 1)
                        with T.device_func():
                            uma_softmax_role_dsl(
                                S0_tmem[0, 0],
                                P0_tmem[0, 0],
                                rs_shared,
                                mb_s[0],
                                mb_p[0],
                                mb_p2[0],
                                loop_extent,
                                seq_len,
                                tile_k_base,
                                scale_log2,
                                0,
                                tid,
                                warp,
                                cta_rank,
                            )
                    elif warp_group == 1:
                        T.set_max_nreg(168, 1)
                        with T.device_func():
                            uma_softmax_role_dsl(
                                S1_tmem[0, 0],
                                P1_tmem[0, 0],
                                rs_shared,
                                mb_s[1],
                                mb_p[1],
                                mb_p2[1],
                                loop_extent,
                                seq_len,
                                tile_k_base,
                                scale_log2,
                                1,
                                tid,
                                warp,
                                cta_rank,
                            )
                    elif warp_group == 2:
                        T.set_max_nreg(96, 0)
                        corr_tid = tid - 256
                        warp_in_corr = warp & 3
                        tmem_row_base = cta_rank * block_m_cta + warp_in_corr * 32
                        tmem_row_offset = tmem_row_base * 65536

                        for kb in T.unroll(loop_extent, explicit=False, unroll_factor=1):
                            tkb = tile_k_base + kb
                            for qs in T.unroll(q_stages):
                                T.tcgen05_bar_sync(warp_in_corr + qs * 4, 64)
                                rs = rs_shared[tkb & 1, qs, corr_tid]
                                if kb > 0:
                                    if T.ballot(rs < 1.0) != 0:
                                        prev = tkb - 1
                                        pv_stage = prev % kv_stages
                                        pv_phase = (prev // kv_stages) & 1
                                        T.tcgen05_wait_barrier(
                                            T.mbarrier_at(mb_pv, pv_stage),
                                            pv_phase,
                                        )
                                        T.tcgen05_after_thread_sync()
                                        T.call_extern(
                                            "void",
                                            f"tl::tcgen05_tmem_rescale_row_x16_addr<{dim}>",
                                            O0_tmem[0, 0],
                                            qs * 128,
                                            tmem_row_offset,
                                            rs,
                                        )
                                T.tcgen05_mbarrier_arrive_cluster_all_ref(
                                    T.mbarrier_at(mb_corr, qs),
                                )

                        last_tkb = tile_k_base + loop_extent - 1
                        last_v_stage = last_tkb % kv_stages
                        last_v_phase = (last_tkb // kv_stages) & 1
                        if tile_k_base > 0:
                            T.tcgen05_wait_barrier(mb_o_rel, tile_phase ^ 1)

                        for qs in T.unroll(q_stages):
                            T.tcgen05_wait_barrier(
                                T.mbarrier_at(mb_pv, last_v_stage),
                                last_v_phase,
                            )
                            T.tcgen05_after_thread_sync()
                            T.tcgen05_bar_sync(warp_in_corr + qs * 4, 64)
                            rsum = rs_shared[0, qs, corr_tid]
                            inv = T.alloc_var(accum_dtype, 0.0)
                            if rsum > 0.0:
                                inv = T.tcgen05_rcp_approx_ftz(rsum)
                            epi_offset = (
                                qs * block_m_cta * dim
                                + warp_in_corr * 32 * dim
                            )
                            T.call_extern(
                                "void",
                                f"tl::tcgen05_tmem_normalize_store_row_bf16_x16_addr<{dim}>",
                                T.tcgen05_smem_ptr_add_bf16(
                                    T.access_ptr(O_shared, "w"),
                                    epi_offset,
                                ),
                                O0_tmem[0, 0],
                                qs * 128,
                                tmem_row_offset,
                                tid & 31,
                                inv,
                            )
                            T.fence_proxy_async()
                            T.tcgen05_mbarrier_arrive_local_all_ref(
                                T.mbarrier_at(mb_epi, qs),
                            )
                        T.tcgen05_mbarrier_arrive_cluster_all_ref(mb_o_tmem_rel)
                    elif warp == 12:
                        T.set_max_nreg(80, 0)
                        with T.device_func():
                            uma_mma_role_dsl(
                                cta_rank,
                                Q_shared,
                                K_shared,
                                V_shared,
                                mb_q,
                                mb_k,
                                mb_s,
                                mb_p,
                                mb_p2,
                                mb_v,
                                mb_pv,
                                mb_corr,
                                mb_k_rel,
                                mb_v_rel,
                                mb_q_rel,
                                mb_o_tmem_rel,
                                S0_tmem[0, 0],
                                loop_extent,
                                tile_k_base,
                                tile_phase,
                            )
                    elif warp == 13:
                        T.set_max_nreg(80, 0)
                        q_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(Q, "r"),
                            heads * dim, batch * seq_len,
                            2, heads * dim * 2,
                            tile_cols, block_m_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        k_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(K, "r"),
                            num_kv_heads * dim, batch * seq_len,
                            2, num_kv_heads * dim * 2,
                            tile_cols, b_per_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        v_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(V, "r"),
                            num_kv_heads * dim, batch * seq_len,
                            2, num_kv_heads * dim * 2,
                            tile_cols, b_per_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        with T.device_func():
                            if T.shuffle_elect(32):
                                if tile_k_base > 0:
                                    T.tcgen05_wait_barrier(
                                        mb_q_rel,
                                        tile_phase ^ 1,
                                    )
                                producer_load_q_dsl(
                                    q_desc,
                                    Q_shared,
                                    mb_q,
                                    tile_q_col_base,
                                    tile_q_row_base,
                                )

                                for prefetch in T.unroll(kv_stages):
                                    producer_prefetch_kv_stage_dsl(
                                        prefetch,
                                        k_desc,
                                        v_desc,
                                        K_shared,
                                        V_shared,
                                        mb_k,
                                        mb_v,
                                        mb_k_rel,
                                        mb_v_rel,
                                        tile_k_base,
                                        tile_kv_row_base,
                                        tile_kv_col_base,
                                        tile_v_col_base,
                                        cta_rank,
                                        loop_extent,
                                    )

                                for kb in T.unroll(
                                    loop_extent,
                                    explicit=False,
                                    unroll_factor=1,
                                ):
                                    producer_refill_kv_stage_dsl(
                                        kb,
                                        k_desc,
                                        v_desc,
                                        K_shared,
                                        V_shared,
                                        mb_k,
                                        mb_v,
                                        mb_k_rel,
                                        mb_v_rel,
                                        tile_k_base,
                                        tile_kv_row_base,
                                        tile_kv_col_base,
                                        tile_v_col_base,
                                        cta_rank,
                                        loop_extent,
                                    )
                    elif warp == 14:
                        T.set_max_nreg(80, 0)
                        output_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(Output, "w"),
                            heads * dim, batch * seq_len,
                            2, heads * dim * 2,
                            tile_cols, page_rows,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        if T.shuffle_elect(32):
                            for qs in T.unroll(q_stages):
                                T.tcgen05_wait_barrier(
                                    T.mbarrier_at(mb_epi, qs),
                                    tile_phase,
                                )
                                for cw in T.unroll(4):
                                    row_seq = tile_ms + qs * block_m + cw * page_rows
                                    row = tile_batch * seq_len + row_seq
                                    epi_offset = (
                                        qs * block_m_cta * dim
                                        + cw * page_rows * dim
                                    )
                                    epi_base = T.tcgen05_smem_ptr_add_bf16(
                                        T.access_ptr(O_shared, "r"),
                                        epi_offset,
                                    )
                                    in_bounds = row_seq < seq_len
                                    T.tma_store_2d(
                                        output_desc,
                                        epi_base,
                                        tile_q_col_base,
                                        row,
                                        in_bounds,
                                    )
                                    T.tma_store_2d(
                                        output_desc,
                                        T.tcgen05_smem_ptr_add_bf16(
                                            epi_base,
                                            page_rows * tile_cols,
                                        ),
                                        tile_q_col_base + tile_cols,
                                        row,
                                        in_bounds,
                                    )
                                T.tma_store_arrive()
                            T.tma_store_wait(0)
                            T.tcgen05_mbarrier_arrive_local_all_ref(mb_o_rel)
                    else:
                        T.set_max_nreg(80, 0)
                        T.evaluate(0)

            if warp == 12:
                T.deallocate_tmem(Base_tmem)

    return main


ATTENTION_2SM_D256_EXTERN_SOURCE = r"""
#include <tl_templates/cuda/copy.h>

namespace tl {

__device__ __forceinline__ uint32_t tcgen05_tmem_addr_add(uint32_t base,
                                                          uint32_t offset) {
  return base + offset;
}

__device__ __forceinline__ void
tcgen05_softmax_pack_4_plain(uint32_t &h0, uint32_t &h1, float const *sv,
                             float &psa0, float &psa1) {
  bfloat16_t h[4];
  #pragma unroll
  for (int i = 0; i < 4; i += 2) {
    float p0 = tl::tcgen05_exp2f_approx(sv[i]);
    float p1 = tl::tcgen05_exp2f_approx(sv[i + 1]);
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

template <int HeadDim>
__device__ __forceinline__ void
tcgen05_tmem_rescale_row_x16_addr(uint32_t O_base, uint32_t col_offset,
                                  uint32_t row_offset, float rs) {
  uint32_t base = O_base + col_offset + row_offset;
  float buf[2][16];
  int cur = 0;
  tl::tmem_ld_32dp32bNx<false>::copy<16>(base, (uint32_t *)buf[cur]);
  #pragma unroll
  for (int g = 0; g < HeadDim / 16; ++g) {
    tl::fence_view_async_tmem_load();
    int nxt = cur ^ 1;
    if (g + 1 < HeadDim / 16) {
      tl::tmem_ld_32dp32bNx<false>::copy<16>(base + (g + 1) * 16,
                                              (uint32_t *)buf[nxt]);
    }
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      tl::tcgen05_fma_f32x2(buf[cur][i], buf[cur][i + 1],
                            buf[cur][i], buf[cur][i + 1], rs, rs, 0.0f,
                            0.0f);
    }
    tl::tmem_st_32dp32bNx<false>::copy<16>(base + g * 16,
                                            (uint32_t *)buf[cur]);
    cur = nxt;
  }
  tl::fence_view_async_tmem_store();
}

template <int HeadDim>
__device__ __forceinline__ void
tcgen05_tmem_normalize_store_row_bf16_x16_d256_addr(
    bfloat16_t *epi_stage, uint32_t O_base, uint32_t col_offset,
    uint32_t row_offset, int warp_in_corr, int lane, float inv) {
  constexpr int kEpiBlockCols = 64;
  constexpr int kEpiBlockBytes = kEpiBlockCols * 2;
  constexpr int kEpiBlockElems = 32 * kEpiBlockCols;
  uint32_t base = O_base + col_offset + row_offset;
  char *epi_base = reinterpret_cast<char *>(epi_stage);
  int row_off = lane * kEpiBlockBytes;
  int swiz = (lane & 7) << 4;

  #pragma unroll
  for (int d = 0; d < HeadDim; d += 16) {
    float t[16];
    tl::tmem_ld_32dp32bNx<false>::copy<16>(base + d, (uint32_t *)t);
    tl::fence_view_async_tmem_load();
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      tl::tcgen05_fma_f32x2(t[i], t[i + 1], t[i], t[i + 1], inv, inv, 0.0f,
                            0.0f);
    }
    bfloat16_t b[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) b[i] = bfloat16_t(t[i]);
    int blk_idx = d / kEpiBlockCols;
    int d_in_blk = d % kEpiBlockCols;
    char *blk = epi_base + (blk_idx * 4 + warp_in_corr) * kEpiBlockElems * 2;
    int col0 = d_in_blk * 2;
    int col1 = (d_in_blk + 8) * 2;
    *reinterpret_cast<uint4 *>(blk + row_off + (col0 ^ swiz)) =
        *reinterpret_cast<uint4 *>(&b[0]);
    *reinterpret_cast<uint4 *>(blk + row_off + (col1 ^ swiz)) =
        *reinterpret_cast<uint4 *>(&b[8]);
  }
}

}  // namespace tl
"""


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG, target={"kind": "cuda", "arch": "sm_100"})
def attention_kernel_2sm_d256(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    num_kv_heads: Optional[int] = None,
    is_causal: bool = False,
):
    if dim != 256:
        raise ValueError("attention_kernel_2sm_d256 currently supports head_dim=256 only")
    if is_causal:
        raise ValueError("attention_kernel_2sm_d256 currently implements non-causal attention only")
    if num_kv_heads is None:
        num_kv_heads = heads
    if heads % num_kv_heads != 0:
        raise ValueError(f"heads={heads} must be divisible by num_kv_heads={num_kv_heads}")

    block_m = 256
    block_m_cta = 128
    block_n = 128
    page_rows = 32
    q_stages = 1
    kv_stages = 5
    b_per_cta = 64
    tile_cols = 64
    threads = 512
    q_rows_per_cluster = q_stages * block_m
    q_tiles = T.ceildiv(seq_len, q_rows_per_cluster)
    total_tiles = q_tiles * heads * batch
    sm_num = driver.get_num_sms()
    grid = T.max(2, (T.min(total_tiles * 2, sm_num) // 2) * 2)
    total_clusters = grid // 2
    loop_extent = T.ceildiv(seq_len, block_n)
    scale_log2 = (1.0 / dim) ** 0.5 * 1.44269504089

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, num_kv_heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    q_stage_elems = block_m_cta * dim
    kv_stage_elems = b_per_cta * dim
    kv_stage_bytes = kv_stage_elems * 2
    box_bytes = b_per_cta * tile_cols * 2
    v_strip_bytes = 2 * box_bytes
    u64 = T.uint32(64)
    u128 = T.uint32(128)
    u256 = T.uint32(256)
    u384 = T.uint32(384)

    @T.macro
    def tmem_addr(base, offset):
        return T.call_extern("uint32", "tl::tcgen05_tmem_addr_add", base, T.uint32(offset))

    @T.macro
    def producer_load_q_dsl(
        q_desc,
        q_stage,
        mbar_q,
        tile_q_col_base,
        tile_q_row_base,
    ):
        q_bytes = 4 * block_m_cta * tile_cols * 2
        T.tcgen05_mbarrier_arrive_expect_tx_cluster_lane0_ref(
            T.mbarrier_at(mbar_q, 0),
            q_bytes,
        )
        for t in T.unroll(4):
            T.tma_load_2cta_2d(
                q_desc,
                T.tcgen05_smem_ptr_add_bf16(
                    T.access_ptr(q_stage, "w"),
                    t * block_m_cta * tile_cols,
                ),
                T.mbarrier_at(mbar_q, 0),
                tile_q_col_base + t * tile_cols,
                tile_q_row_base,
            )

    @T.macro
    def producer_issue_merged_kv_stage_dsl(
        global_kv_idx,
        is_k,
        kv_block,
        k_desc,
        v_desc,
        kv_stage,
        mbar_kv,
        mbar_kv_rel,
        tile_kv_row_base,
        tile_kv_col_base,
        cta_rank,
    ):
        stage = global_kv_idx % kv_stages
        if global_kv_idx >= kv_stages:
            prev_phase = ((global_kv_idx - kv_stages) // kv_stages) & 1
            T.tcgen05_wait_barrier(T.mbarrier_at(mbar_kv_rel, stage), prev_phase)
        kv_bytes = 4 * b_per_cta * tile_cols * 2
        stage_base = stage * kv_stage_elems
        T.tcgen05_mbarrier_arrive_expect_tx_cluster_lane0_ref(
            T.mbarrier_at(mbar_kv, stage),
            kv_bytes,
        )
        if is_k:
            k_row = tile_kv_row_base + kv_block * block_n + cta_rank * b_per_cta
            for t in T.unroll(2):
                T.tma_load_2cta_2d(
                    k_desc,
                    T.tcgen05_smem_ptr_add_bf16(
                        T.access_ptr(kv_stage, "w"),
                        stage_base + t * b_per_cta * tile_cols,
                    ),
                    T.mbarrier_at(mbar_kv, stage),
                    tile_kv_col_base + t * tile_cols,
                    k_row,
                )
            for t in T.unroll(2):
                T.tma_load_2cta_2d(
                    k_desc,
                    T.tcgen05_smem_ptr_add_bf16(
                        T.access_ptr(kv_stage, "w"),
                        stage_base + (t + 2) * b_per_cta * tile_cols,
                    ),
                    T.mbarrier_at(mbar_kv, stage),
                    tile_kv_col_base + (t + 2) * tile_cols,
                    k_row,
                )
        else:
            v_row_lo = tile_kv_row_base + kv_block * block_n
            v_row_hi = v_row_lo + b_per_cta
            for strip in T.unroll(2):
                for box in T.unroll(2):
                    T.tma_load_2cta_2d(
                        v_desc,
                        T.tcgen05_smem_ptr_add_bf16(
                            T.access_ptr(kv_stage, "w"),
                            stage_base
                            + (strip * 2 + box) * b_per_cta * tile_cols,
                        ),
                        T.mbarrier_at(mbar_kv, stage),
                        tile_kv_col_base + strip * 128 + cta_rank * b_per_cta,
                        T.Select(box == 0, v_row_lo, v_row_hi),
                    )

    @T.macro
    def d256_softmax_role_dsl(
        base_tmem_addr,
        rs_stage,
        mbar_s,
        mbar_p,
        mbar_p2,
        loop_extent,
        seq_len,
        tile_k_base,
        softmax_scale_log2,
        tid,
        warp,
        cta_rank,
    ):
        sm_tid = tid
        sm_bar = warp & 3
        tmem_row_offset = (cta_rank * block_m_cta + (warp & 3) * 32) * 65536

        sv = T.alloc_local((128,), accum_dtype)
        psa = T.alloc_local((4,), accum_dtype)
        rmax_local = T.alloc_var(accum_dtype, -T.infinity(accum_dtype))
        rsum_local = T.alloc_var(accum_dtype, 0.0)
        m0 = T.alloc_var(accum_dtype)
        m1 = T.alloc_var(accum_dtype)
        m2 = T.alloc_var(accum_dtype)
        m3 = T.alloc_var(accum_dtype)
        block_max = T.alloc_var(accum_dtype)
        new_max = T.alloc_var(accum_dtype)
        rs = T.alloc_var(accum_dtype)
        acc_scale_log2 = T.alloc_var(accum_dtype)
        neg_max_scaled = T.alloc_var(accum_dtype)

        for kb in T.unroll(loop_extent, explicit=False, unroll_factor=1):
            tkb = tile_k_base + kb
            buf = tkb & 1
            phase = (tkb // 2) & 1
            buf_cols = T.uint32(buf * 128)
            s_tmem_addr = tmem_addr(base_tmem_addr, buf_cols)
            p_tmem_addr = tmem_addr(base_tmem_addr, buf_cols + u64)
            T.tcgen05_wait_barrier(T.mbarrier_at(mbar_s, buf), phase)
            T.tcgen05_after_thread_sync()

            for cc in T.unroll(0, block_n, 32):
                T.tcgen05_ld(
                    32,
                    32,
                    False,
                    s_tmem_addr,
                    tmem_row_offset + cc,
                    T.access_ptr(sv[cc], "w", 32),
                    emit_fence=False,
                )
            T.tcgen05_fence_tmem_load()

            remaining = seq_len - kb * block_n
            if remaining < block_n:
                for i in T.unroll(block_n):
                    if i >= remaining:
                        sv[i] = -T.infinity(accum_dtype)

            m0 = T.max3(sv[0], sv[1], sv[2])
            m1 = T.max3(sv[3], sv[4], sv[5])
            m2 = T.max3(sv[6], sv[7], sv[8])
            m3 = sv[9]
            for i in T.unroll(10, block_n, 8):
                m0 = T.max3(m0, sv[i + 0], sv[i + 1])
                m1 = T.max3(m1, sv[i + 2], sv[i + 3])
                m2 = T.max3(m2, sv[i + 4], sv[i + 5])
                m3 = T.max3(m3, sv[i + 6], sv[i + 7])
            block_max = T.fmax2_ftz(
                T.fmax2_ftz(m0, m1),
                T.fmax2_ftz(m2, m3),
            )
            new_max = T.fmax2_ftz(rmax_local, block_max)
            if new_max == -T.infinity(accum_dtype):
                new_max = 0.0
            rs = 1.0
            if kb == 0:
                rmax_local = new_max
            else:
                acc_scale_log2 = (rmax_local - new_max) * softmax_scale_log2
                rs = T.tcgen05_exp2f_approx(acc_scale_log2)
                rmax_local = new_max
            rsum_local *= rs

            rs_stage[tkb & 1, sm_tid] = rs
            T.tcgen05_bar_arrive(sm_bar, 64)

            neg_max_scaled = -(rmax_local * softmax_scale_log2)
            for cc in T.unroll(0, block_n, 16):
                for i in T.unroll(0, 16, 2):
                    T.tcgen05_fma_f32x2(
                        sv[cc + i],
                        sv[cc + i + 1],
                        sv[cc + i],
                        sv[cc + i + 1],
                        softmax_scale_log2,
                        softmax_scale_log2,
                        neg_max_scaled,
                        neg_max_scaled,
                    )

            for i in T.unroll(4):
                psa[i] = 0.0

            h0 = T.alloc_var(T.uint32)
            h1 = T.alloc_var(T.uint32)
            h2 = T.alloc_var(T.uint32)
            h3 = T.alloc_var(T.uint32)
            for cc in T.unroll(0, block_n, 16):
                for g_iter in T.unroll(2):
                    g = g_iter * 8
                    elem_base = cc + g
                    T.call_extern(
                        "void",
                        "tl::tcgen05_softmax_pack_4_plain",
                        h0,
                        h1,
                        T.access_ptr(sv[elem_base], "r", 4),
                        psa[0],
                        psa[1],
                    )
                    T.call_extern(
                        "void",
                        "tl::tcgen05_softmax_pack_4_plain",
                        h2,
                        h3,
                        T.access_ptr(sv[elem_base + 4], "r", 4),
                        psa[2],
                        psa[3],
                    )
                    T.tcgen05_st_32x32b_x4(
                        p_tmem_addr,
                        tmem_row_offset + (cc + g) // 2,
                        h0,
                        h1,
                        h2,
                        h3,
                    )
                if cc == 80:
                    T.tcgen05_fence_tmem_store()
                    T.tcgen05_mbarrier_arrive_cluster_all_ref(
                        T.mbarrier_at(mbar_p, buf),
                    )
            T.tcgen05_fence_tmem_store()
            T.tcgen05_mbarrier_arrive_cluster_all_ref(
                T.mbarrier_at(mbar_p2, buf),
            )
            rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3])

        rs_stage[0, sm_tid] = rsum_local
        T.tcgen05_bar_arrive(sm_bar, 64)

    @T.macro
    def uma_qk_mma_2cta_dsl(
        q_base_16b,
        kv_base_16b,
        k_stage,
        s_tmem_addr,
        mbar_s_qs,
    ):
        idesc = (
            T.uint32(1 << 4)
            | T.uint32(1 << 7)
            | T.uint32(1 << 10)
            | T.uint32((128 // 8) << 17)
            | T.uint32((256 // 16) << 24)
        )
        first = T.alloc_var(T.uint32, 1)
        for box in T.unroll(4):
            q_off = box * block_m_cta * tile_cols * 2
            k_off = k_stage * kv_stage_bytes + box * box_bytes
            for j in T.unroll(0, tile_cols, 16):
                T.tcgen05_mma_ss(
                    T.tcgen05_mk_fast_desc(q_base_16b, q_off + j * 2),
                    T.tcgen05_mk_fast_desc(kv_base_16b, k_off + j * 2),
                    s_tmem_addr,
                    idesc,
                    T.Select(first == 1, T.uint32(0), T.uint32(1)),
                    cta_group=2,
                    use_mask=False,
                    elect_one=False,
                )
                first = 0
        T.tcgen05_commit_2cta(mbar_s_qs)

    @T.macro
    def d256_pv_mma_strip_dsl(
        kv_base_16b,
        v_stage,
        p_tmem_base,
        o_tmem_addr,
        strip,
        accumulate,
        mbar_p2_qs,
        phase,
    ):
        idesc = (
            T.uint32(1 << 4)
            | T.uint32(1 << 7)
            | T.uint32(1 << 10)
            | T.uint32(1 << 16)
            | T.uint32((128 // 8) << 17)
            | T.uint32((256 // 16) << 24)
        )
        v_stage_base = v_stage * kv_stage_bytes
        v_base = v_stage_base + strip * v_strip_bytes
        v_hi = v_base + box_bytes

        for j in T.unroll(0, tile_cols, 16):
            T.tcgen05_mma_ts(
                tmem_addr(p_tmem_base, j // 2),
                T.tcgen05_mk_fast_desc(kv_base_16b, v_base + j * tile_cols * 2),
                o_tmem_addr,
                idesc,
                T.Select(j == 0, accumulate, T.uint32(1)),
                cta_group=2,
                use_mask=False,
                elect_one=False,
            )
        for j in T.unroll(0, tile_cols // 2, 16):
            T.tcgen05_mma_ts(
                tmem_addr(p_tmem_base, tile_cols // 2 + j // 2),
                T.tcgen05_mk_fast_desc(kv_base_16b, v_hi + j * tile_cols * 2),
                o_tmem_addr,
                idesc,
                T.uint32(1),
                cta_group=2,
                use_mask=False,
                elect_one=False,
            )

        T.tcgen05_wait_barrier(mbar_p2_qs, phase)
        T.tcgen05_after_thread_sync()

        for j in T.unroll(tile_cols // 2, tile_cols, 16):
            T.tcgen05_mma_ts(
                tmem_addr(p_tmem_base, tile_cols // 2 + j // 2),
                T.tcgen05_mk_fast_desc(kv_base_16b, v_hi + j * tile_cols * 2),
                o_tmem_addr,
                idesc,
                T.uint32(1),
                cta_group=2,
                use_mask=False,
                elect_one=False,
            )

    @T.macro
    def d256_issue_pv_dsl(
        tkb,
        kv_base_16b,
        base_tmem_addr,
        mbar_p,
        mbar_p2,
        mbar_kv,
        mbar_pv,
        mbar_corr,
        mbar_kv_rel,
        wait_corr: bool,
        accumulate,
    ):
        buf = tkb & 1
        v_global = 2 * tkb + 1
        v_stage = v_global % kv_stages
        v_phase = (v_global // kv_stages) & 1
        p_phase = (tkb // 2) & 1
        T.tcgen05_wait_barrier(T.mbarrier_at(mbar_p, buf), p_phase)
        T.tcgen05_wait_barrier(T.mbarrier_at(mbar_kv, v_stage), v_phase)
        if wait_corr:
            T.tcgen05_wait_barrier(T.mbarrier_at(mbar_corr, 0), tkb & 1)
        T.tcgen05_after_thread_sync()
        d256_pv_mma_strip_dsl(
            kv_base_16b,
            v_stage,
            tmem_addr(base_tmem_addr, T.uint32(buf * 128) + u64),
            tmem_addr(base_tmem_addr, u256),
            0,
            accumulate,
            T.mbarrier_at(mbar_p2, buf),
            p_phase,
        )
        d256_pv_mma_strip_dsl(
            kv_base_16b,
            v_stage,
            tmem_addr(base_tmem_addr, T.uint32(buf * 128) + u64),
            tmem_addr(base_tmem_addr, u384),
            1,
            accumulate,
            T.mbarrier_at(mbar_p2, buf),
            p_phase,
        )
        T.tcgen05_commit_2cta(T.mbarrier_at(mbar_pv, v_stage))
        T.tcgen05_commit_2cta(T.mbarrier_at(mbar_kv_rel, v_stage))

    @T.macro
    def d256_issue_qk_dsl(
        next_tkb,
        q_base_16b,
        kv_base_16b,
        base_tmem_addr,
        mbar_kv,
        mbar_s,
        mbar_kv_rel,
    ):
        k_global = 2 * next_tkb
        next_stage = k_global % kv_stages
        next_phase = (k_global // kv_stages) & 1
        next_buf = next_tkb & 1
        T.tcgen05_wait_barrier(T.mbarrier_at(mbar_kv, next_stage), next_phase)
        T.tcgen05_after_thread_sync()
        uma_qk_mma_2cta_dsl(
            q_base_16b,
            kv_base_16b,
            next_stage,
            tmem_addr(base_tmem_addr, T.uint32(next_buf * 128)),
            T.mbarrier_at(mbar_s, next_buf),
        )
        T.tcgen05_commit_2cta(T.mbarrier_at(mbar_kv_rel, next_stage))

    @T.macro
    def uma_mma_role_dsl(
        cta_rank,
        q_stage,
        kv_stage,
        mbar_q,
        mbar_kv,
        mbar_s,
        mbar_p,
        mbar_p2,
        mbar_pv,
        mbar_corr,
        mbar_kv_rel,
        mbar_o_tmem_rel,
        base_tmem_addr,
        loop_extent,
        tile_k_base,
        tile_phase,
    ):
        if cta_rank == 0:
            if T.shuffle_elect(32):
                q_base_16b = T.tcgen05_smem_base_16b(T.access_ptr(q_stage, "r"))
                kv_base_16b = T.tcgen05_smem_base_16b(T.access_ptr(kv_stage, "r"))

                T.tcgen05_wait_barrier(T.mbarrier_at(mbar_q, 0), tile_phase)
                T.tcgen05_after_thread_sync()

                if tile_k_base > 0:
                    T.tcgen05_wait_barrier(
                        T.mbarrier_at(mbar_o_tmem_rel, 0),
                        tile_phase ^ 1,
                    )
                    T.tcgen05_after_thread_sync()

                kv0_global = 2 * tile_k_base
                kv0_stage = kv0_global % kv_stages
                kv0_phase = (kv0_global // kv_stages) & 1
                T.tcgen05_wait_barrier(T.mbarrier_at(mbar_kv, kv0_stage), kv0_phase)
                T.tcgen05_after_thread_sync()
                uma_qk_mma_2cta_dsl(
                    q_base_16b,
                    kv_base_16b,
                    kv0_stage,
                    tmem_addr(base_tmem_addr, T.uint32((tile_k_base & 1) * 128)),
                    T.mbarrier_at(mbar_s, tile_k_base & 1),
                )
                T.tcgen05_commit_2cta(T.mbarrier_at(mbar_kv_rel, kv0_stage))

                if loop_extent > 0:
                    tkb0 = tile_k_base
                    if 1 < loop_extent:
                        d256_issue_qk_dsl(
                            tkb0 + 1,
                            q_base_16b,
                            kv_base_16b,
                            base_tmem_addr,
                            mbar_kv,
                            mbar_s,
                            mbar_kv_rel,
                        )
                    d256_issue_pv_dsl(
                        tkb0,
                        kv_base_16b,
                        base_tmem_addr,
                        mbar_p,
                        mbar_p2,
                        mbar_kv,
                        mbar_pv,
                        mbar_corr,
                        mbar_kv_rel,
                        False,
                        T.uint32(0),
                    )

                for kb in T.unroll(
                    1,
                    loop_extent,
                    explicit=False,
                    unroll_factor=1,
                ):
                    tkb = tile_k_base + kb
                    if kb + 1 < loop_extent:
                        d256_issue_qk_dsl(
                            tkb + 1,
                            q_base_16b,
                            kv_base_16b,
                            base_tmem_addr,
                            mbar_kv,
                            mbar_s,
                            mbar_kv_rel,
                        )
                    d256_issue_pv_dsl(
                        tkb,
                        kv_base_16b,
                        base_tmem_addr,
                        mbar_p,
                        mbar_p2,
                        mbar_kv,
                        mbar_pv,
                        mbar_corr,
                        mbar_kv_rel,
                        True,
                        T.uint32(1),
                    )

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            grid,
            threads=threads,
            cluster_dims=2,
            prelude=ATTENTION_2SM_D256_EXTERN_SOURCE,
        ) as block_id:
            # Use one block per SM while relying on runtime setmaxnreg.inc/dec
            # for role-specific register donation.
            T.annotate_min_blocks_per_sm(1)
            T.use_2cta_tmem(
                mbarrier_init_thread=416,
                tmem_alloc_warp=12,
                compact_shared_state=True,
            )

            # Alias the epilogue staging buffer with Q and use one merged K/V ring.
            QO_shared = T.alloc_shared([block_m_cta, dim], dtype)
            KV_shared = T.alloc_shared([kv_stages, b_per_cta, dim], dtype)
            rs_shared = T.alloc_shared([2, block_m_cta], accum_dtype)

            Base_tmem = T.alloc_tmem([block_m_cta, 512], accum_dtype)
            S0_tmem = T.alloc_tmem([block_m_cta, block_n], accum_dtype, alias=Base_tmem, col_offset=0)
            P0_tmem = T.alloc_tmem([block_m_cta, block_n], dtype, alias=Base_tmem, col_offset=64)
            S1_tmem = T.alloc_tmem([block_m_cta, block_n], accum_dtype, alias=Base_tmem, col_offset=128)
            P1_tmem = T.alloc_tmem([block_m_cta, block_n], dtype, alias=Base_tmem, col_offset=192)
            O0_tmem = T.alloc_tmem([block_m_cta, dim], accum_dtype, alias=Base_tmem, col_offset=256)

            base_layout = T.Layout([block_m_cta, 512], lambda i, j: [i, j])
            score_layout = T.Layout([block_m_cta, block_n], lambda i, j: [i, j])
            output_layout = T.Layout([block_m_cta, dim], lambda i, j: [i, j])
            T.annotate_layout({
                QO_shared: tilelang.layout.make_full_bank_swizzled_layout(QO_shared),
                KV_shared: tilelang.layout.make_full_bank_swizzled_layout(KV_shared),
                Base_tmem: base_layout,
                S0_tmem: score_layout,
                P0_tmem: score_layout,
                S1_tmem: score_layout,
                P1_tmem: score_layout,
                O0_tmem: output_layout,
            })

            mb_q = T.alloc_cluster_barrier(2)
            mb_kv = T.alloc_cluster_barrier([2] * kv_stages)
            mb_s = T.alloc_cluster_barrier([1] * 2)
            mb_p = T.alloc_cluster_barrier([256] * 2)
            mb_p2 = T.alloc_cluster_barrier([256] * 2)
            mb_kv_rel = T.alloc_cluster_barrier([1] * kv_stages)
            mb_pv = T.alloc_cluster_barrier([1] * kv_stages)
            mb_corr = T.alloc_cluster_barrier(256)
            mb_epi = T.alloc_barrier(block_m_cta)
            mb_o_rel = T.alloc_barrier(1)
            mb_o_tmem_rel = T.alloc_cluster_barrier(256)

            tid = T.alloc_var(T.int32, T.get_thread_binding())
            warp = T.alloc_var(T.int32, tid // 32)
            warp_group = T.alloc_var(T.int32, warp // 4)
            cta_rank = T.alloc_var(T.int32, T.block_rank_in_cluster())
            cluster_id = T.alloc_var(T.int32, T.cluster_id_x())
            T.assume(cta_rank < 2)

            for tile_iter in T.serial(T.ceildiv(total_tiles, total_clusters)):
                tile_id = tile_iter * total_clusters + cluster_id
                if tile_id < total_tiles:
                    tile_phase = tile_iter & 1
                    tile_k_base = tile_iter * loop_extent
                    tile_mb = T.truncmod(tile_id, q_tiles)
                    tile_head = T.truncmod(T.truncdiv(tile_id, q_tiles), heads)
                    tile_batch = T.truncdiv(tile_id, q_tiles * heads)
                    tile_ms = tile_mb * q_rows_per_cluster + cta_rank * block_m_cta
                    tile_kv_head = T.truncdiv(tile_head * num_kv_heads, heads)
                    tile_q_col_base = tile_head * dim
                    tile_kv_col_base = tile_kv_head * dim
                    tile_q_row_base = tile_batch * seq_len + tile_ms
                    tile_kv_row_base = tile_batch * seq_len
                    if warp_group == 0:
                        T.set_max_nreg(192, 1)
                        with T.device_func():
                            d256_softmax_role_dsl(
                                S0_tmem[0, 0],
                                rs_shared,
                                mb_s,
                                mb_p,
                                mb_p2,
                                loop_extent,
                                seq_len,
                                tile_k_base,
                                scale_log2,
                                tid,
                                warp,
                                cta_rank,
                            )
                    elif warp_group == 1:
                        T.set_max_nreg(72, 0)
                        T.evaluate(0)
                    elif warp_group == 2:
                        T.set_max_nreg(80, 0)
                        corr_tid = tid - 256
                        warp_in_corr = warp & 3
                        tmem_row_base = cta_rank * block_m_cta + warp_in_corr * 32
                        tmem_row_offset = tmem_row_base * 65536

                        for kb in T.unroll(loop_extent, explicit=False, unroll_factor=1):
                            tkb = tile_k_base + kb
                            T.tcgen05_bar_sync(warp_in_corr, 64)
                            rs = rs_shared[tkb & 1, corr_tid]
                            if kb > 0:
                                if T.ballot(rs < 1.0) != 0:
                                    prev = tkb - 1
                                    prev_v_idx = 2 * prev + 1
                                    pv_stage = prev_v_idx % kv_stages
                                    pv_phase = (prev // kv_stages) & 1
                                    T.tcgen05_wait_barrier(
                                        T.mbarrier_at(mb_pv, pv_stage),
                                        pv_phase,
                                    )
                                    T.tcgen05_after_thread_sync()
                                    T.call_extern(
                                        "void",
                                        f"tl::tcgen05_tmem_rescale_row_x16_addr<{dim}>",
                                        O0_tmem[0, 0],
                                        0,
                                        tmem_row_offset,
                                        rs,
                                    )
                            T.tcgen05_mbarrier_arrive_cluster_all_ref(
                                T.mbarrier_at(mb_corr, 0),
                            )

                        last_tkb = tile_k_base + loop_extent - 1
                        last_v_idx = 2 * last_tkb + 1
                        last_v_stage = last_v_idx % kv_stages
                        last_v_phase = (last_tkb // kv_stages) & 1
                        if tile_k_base > 0:
                            T.tcgen05_wait_barrier(
                                T.mbarrier_at(mb_o_rel, 0),
                                tile_phase ^ 1,
                            )

                        T.tcgen05_wait_barrier(
                            T.mbarrier_at(mb_pv, last_v_stage),
                            last_v_phase,
                        )
                        T.tcgen05_after_thread_sync()
                        T.tcgen05_bar_sync(warp_in_corr, 64)
                        rsum = rs_shared[0, corr_tid]
                        inv = T.alloc_var(accum_dtype, 0.0)
                        if rsum > 0.0:
                            inv = T.tcgen05_rcp_approx_ftz(rsum)
                        T.call_extern(
                            "void",
                            f"tl::tcgen05_tmem_normalize_store_row_bf16_x16_d256_addr<{dim}>",
                            T.access_ptr(QO_shared, "w"),
                            O0_tmem[0, 0],
                            0,
                            tmem_row_offset,
                            warp_in_corr,
                            tid & 31,
                            inv,
                        )
                        T.fence_proxy_async()
                        T.tcgen05_mbarrier_arrive_local_all_ref(
                            T.mbarrier_at(mb_epi, 0),
                        )
                        T.tcgen05_mbarrier_arrive_cluster_all_ref(
                            T.mbarrier_at(mb_o_tmem_rel, 0),
                        )
                    elif warp == 12:
                        T.set_max_nreg(72, 0)
                        with T.device_func():
                            uma_mma_role_dsl(
                                cta_rank,
                                QO_shared,
                                KV_shared,
                                mb_q,
                                mb_kv,
                                mb_s,
                                mb_p,
                                mb_p2,
                                mb_pv,
                                mb_corr,
                                mb_kv_rel,
                                mb_o_tmem_rel,
                                S0_tmem[0, 0],
                                loop_extent,
                                tile_k_base,
                                tile_phase,
                            )
                    elif warp == 13:
                        T.set_max_nreg(72, 0)
                        q_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(Q, "r"),
                            heads * dim, batch * seq_len,
                            2, heads * dim * 2,
                            tile_cols, block_m_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        k_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(K, "r"),
                            num_kv_heads * dim, batch * seq_len,
                            2, num_kv_heads * dim * 2,
                            tile_cols, b_per_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        v_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(V, "r"),
                            num_kv_heads * dim, batch * seq_len,
                            2, num_kv_heads * dim * 2,
                            tile_cols, b_per_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        with T.device_func():
                            if T.shuffle_elect(32):
                                if tile_k_base > 0:
                                    T.tcgen05_wait_barrier(
                                        T.mbarrier_at(mb_o_rel, 0),
                                        tile_phase ^ 1,
                                    )
                                producer_load_q_dsl(
                                    q_desc,
                                    QO_shared,
                                    mb_q,
                                    tile_q_col_base,
                                    tile_q_row_base,
                                )

                                for kv_idx in T.unroll(
                                    2 * loop_extent,
                                    explicit=False,
                                    unroll_factor=1,
                                ):
                                    producer_issue_merged_kv_stage_dsl(
                                        2 * tile_k_base + kv_idx,
                                        (kv_idx & 1) == 0,
                                        kv_idx // 2,
                                        k_desc,
                                        v_desc,
                                        KV_shared,
                                        mb_kv,
                                        mb_kv_rel,
                                        tile_kv_row_base,
                                        tile_kv_col_base,
                                        cta_rank,
                                    )
                    elif warp == 14:
                        T.set_max_nreg(72, 0)
                        output_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(Output, "w"),
                            heads * dim, batch * seq_len,
                            2, heads * dim * 2,
                            tile_cols, page_rows,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        if T.shuffle_elect(32):
                            T.tcgen05_wait_barrier(
                                T.mbarrier_at(mb_epi, 0),
                                tile_phase,
                            )
                            for cw in T.unroll(4):
                                row_seq = tile_ms + cw * page_rows
                                row = tile_batch * seq_len + row_seq
                                in_bounds = row_seq < seq_len
                                for b in T.unroll(4):
                                    epi_offset = (
                                        (b * 4 + cw) * page_rows * tile_cols
                                    )
                                    epi_base = T.tcgen05_smem_ptr_add_bf16(
                                        T.access_ptr(QO_shared, "r"),
                                        epi_offset,
                                    )
                                    T.tma_store_2d(
                                        output_desc,
                                        epi_base,
                                        tile_q_col_base + b * tile_cols,
                                        row,
                                        in_bounds,
                                    )
                            T.tma_store_arrive()
                            T.tma_store_wait(0)
                            T.tcgen05_mbarrier_arrive_local_all_ref(
                                T.mbarrier_at(mb_o_rel, 0),
                            )
                    else:
                        T.set_max_nreg(72, 0)
                        T.evaluate(0)

            if warp == 12:
                T.deallocate_tmem(Base_tmem)

    return main


def attention_kernel_2sm(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    num_kv_heads: Optional[int] = None,
    is_causal: bool = False,
):
    if dim == 128:
        return attention_kernel_2sm_d128(
            batch,
            heads,
            seq_len,
            dim,
            num_kv_heads=num_kv_heads,
            is_causal=is_causal,
        )
    if dim == 256:
        return attention_kernel_2sm_d256(
            batch,
            heads,
            seq_len,
            dim,
            num_kv_heads=num_kv_heads,
            is_causal=is_causal,
        )
    raise ValueError("attention_kernel_2sm supports head_dim=128 or 256 only")


def reference_attention(Q, K, V):
    Q_f = Q.to(torch.float32)
    K_f = K.to(torch.float32)
    V_f = V.to(torch.float32)
    if Q_f.shape[2] != K_f.shape[2]:
        groups = Q_f.shape[2] // K_f.shape[2]
        K_f = K_f.repeat_interleave(groups, dim=2)
        V_f = V_f.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bshd,bthd->bhst", Q_f, K_f) * (1.0 / Q.shape[-1] ** 0.5)
    return torch.einsum("bhst,bthd->bshd", scores.softmax(dim=-1), V_f).to(Q.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--skip_ref", action="store_true")
    ap.add_argument("--print_source", action="store_true")
    ap.add_argument("--compile_only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    kv_heads = args.kv_heads or args.heads
    fn = attention_kernel_2sm(
        args.batch,
        args.heads,
        args.seq,
        args.dim,
        num_kv_heads=kv_heads,
    )
    if args.print_source:
        print(fn.get_kernel_source())
    if args.compile_only:
        return

    Q = torch.randn(args.batch, args.seq, args.heads, args.dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(args.batch, args.seq, kv_heads, args.dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(args.batch, args.seq, kv_heads, args.dim, dtype=torch.bfloat16, device="cuda")

    O = fn(Q, K, V)
    if args.skip_ref:
        torch.cuda.synchronize()
        print(f"shape={tuple(O.shape)}  reference=skipped")
    else:
        O_ref = reference_attention(Q, K, V)
        err_abs = (O.to(torch.float32) - O_ref.to(torch.float32)).abs()
        print(f"shape={tuple(O.shape)}  max_abs={err_abs.max().item():.4f}  mean_abs={err_abs.mean().item():.4f}")

    if args.bench:
        from tilelang.profiler import do_bench

        for _ in range(3):
            _ = fn(Q, K, V)
        torch.cuda.synchronize()
        lat = do_bench(lambda: fn(Q, K, V), warmup=25, rep=100)
        flops = 2.0 * 2.0 * args.batch * args.heads * args.seq * args.seq * args.dim
        print(f"latency={lat:.3f} ms  perf={flops / lat * 1e-9:.2f} TFLOPS")


if __name__ == "__main__":
    main()
