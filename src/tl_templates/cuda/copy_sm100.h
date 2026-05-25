#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "barrier.h"
#include "common.h"
#include "cuda_fp8.h"
#include "tcgen_05.h"
#include "tcgen_05_ld.h"
#include "tcgen_05_st.h"

namespace tl {

// 256-bit load specialization for ulonglong4
__device__ __forceinline__ void global_load_256(ulonglong4 &D, void const *ptr,
                                                bool pred_guard) {
#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %5, 0;\n"
               "  mov.b64 %0, %6;\n"
               "  mov.b64 %1, %7;\n"
               "  mov.b64 %2, %8;\n"
               "  mov.b64 %3, %9;\n"
#if TL_ENABLE_L2_PREFETCH
               "  @p ld.global.L2::128B.v4.u64 {%0, %1, %2, %3}, [%4];\n"
#else
               "  @p ld.global.v4.u64 {%0, %1, %2, %3}, [%4];\n"
#endif
               "}\n"
               : "=l"(D.x), "=l"(D.y), "=l"(D.z), "=l"(D.w)
               : "l"(ptr), "r"((int)pred_guard), "l"(D.x), "l"(D.y), "l"(D.z),
                 "l"(D.w));
#else
  // CUDA < 12.9 fallback: two 128-bit loads (may have performance regression)
  uint4 *data = reinterpret_cast<uint4 *>(&D);
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %9, 0;\n"
               "  mov.b32 %0, %10;\n"
               "  mov.b32 %1, %11;\n"
               "  mov.b32 %2, %12;\n"
               "  mov.b32 %3, %13;\n"
               "  mov.b32 %4, %14;\n"
               "  mov.b32 %5, %15;\n"
               "  mov.b32 %6, %16;\n"
               "  mov.b32 %7, %17;\n"
#if TL_ENABLE_L2_PREFETCH
               "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%8];\n"
               "  @p ld.global.L2::128B.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#else
               "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n"
               "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#endif
               "}\n"
               : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z),
                 "=r"(data[0].w), "=r"(data[1].x), "=r"(data[1].y),
                 "=r"(data[1].z), "=r"(data[1].w)
               : "l"(ptr), "r"((int)pred_guard), "r"(data[0].x), "r"(data[0].y),
                 "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y),
                 "r"(data[1].z), "r"(data[1].w), "l"(((uint8_t *)ptr) + 16));
#endif
}

// Convenience wrapper functions
__device__ __forceinline__ longlong4 load_global_256(const longlong4 *ptr) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, true);
  return *reinterpret_cast<longlong4 *>(&ret);
}

__device__ __forceinline__ ulonglong4 load_global_256(const ulonglong4 *ptr) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, true);
  return ret;
}

// Predicated (conditional) versions
__device__ __forceinline__ longlong4
load_global_256_conditional(const longlong4 *ptr, bool pred) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, pred);
  return *reinterpret_cast<longlong4 *>(&ret);
}

__device__ __forceinline__ ulonglong4
load_global_256_conditional(const ulonglong4 *ptr, bool pred) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, pred);
  return ret;
}

// Generic 256-bit load for FP8 and other types (returns ulonglong4)
template <typename T>
__device__ __forceinline__ ulonglong4 load_global_256(const T *ptr) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, true);
  return ret;
}

template <typename T>
__device__ __forceinline__ ulonglong4 load_global_256_conditional(const T *ptr,
                                                                  bool pred) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, pred);
  return ret;
}

// 256-bit store specialization for ulonglong4
__device__ __forceinline__ void global_store_256(ulonglong4 const &D, void *ptr,
                                                 bool pred_guard) {
#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %5, 0;\n"
               "  @p st.global.v4.u64 [%0], {%1, %2, %3, %4};\n"
               "}\n"
               :
               : "l"(ptr), "l"(D.x), "l"(D.y), "l"(D.z), "l"(D.w),
                 "r"((int)pred_guard));
#else
  // CUDA < 12.9 fallback: two 128-bit stores (may have performance
  // regression)
  uint4 const *data = reinterpret_cast<uint4 const *>(&D);
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %5, 0;\n"
               "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
               "  @p st.global.v4.u32 [%6], {%7, %8, %9, %10};\n"
               "}\n"
               :
               : "l"(ptr), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
                 "r"(data[0].w), "r"((int)pred_guard),
                 "l"(((uint8_t *)ptr) + 16), "r"(data[1].x), "r"(data[1].y),
                 "r"(data[1].z), "r"(data[1].w));
#endif
}

// Convenience wrapper functions for 256-bit store
template <typename T>
__device__ __forceinline__ void store_global_256(void *ptr, const T &val) {
  ulonglong4 const &val_u64 = *reinterpret_cast<ulonglong4 const *>(&val);
  global_store_256(val_u64, ptr, true);
}

template <typename T>
__device__ __forceinline__ void
store_global_256_conditional(void *ptr, const T &val, bool pred) {
  ulonglong4 const &val_u64 = *reinterpret_cast<ulonglong4 const *>(&val);
  global_store_256(val_u64, ptr, pred);
}

__device__ __forceinline__ unsigned long long
pack_bfloat16x4(const bfloat16_t x, const bfloat16_t y, const bfloat16_t z,
                const bfloat16_t w) {
  unsigned long long v0 = *((unsigned short *)&x);
  unsigned long long v1 = *((unsigned short *)&y);
  unsigned long long v2 = *((unsigned short *)&z);
  unsigned long long v3 = *((unsigned short *)&w);
  return (v0 | (v1 << 16) | (v2 << 32) | (v3 << 48));
}

__device__ __forceinline__ unsigned long long
pack_float16x4(const half x, const half y, const half z, const half w) {
  unsigned long long v0 = *((unsigned short *)&x);
  unsigned long long v1 = *((unsigned short *)&y);
  unsigned long long v2 = *((unsigned short *)&z);
  unsigned long long v3 = *((unsigned short *)&w);
  return (v0 | (v1 << 16) | (v2 << 32) | (v3 << 48));
}

// Helper function to find the largest K that 2**K <= N
// Requires N > 0
template <int N, int K = 0>
__device__ __forceinline__ constexpr int get_floor_log2() {
  static_assert(N > 0);
  if constexpr ((1 << (K + 1)) > N)
    return K;
  else
    return get_floor_log2<N, K + 1>();
}

template <typename target_call_cls, int MAX_LOGN, int N, typename dst_t>
__device__ __forceinline__ void tcgen05_ld_core(uint32_t const &tmem_start_col,
                                                dst_t *dst_ptr) {
  static_assert(N > 0);
  constexpr int LOG_N = get_floor_log2<N>();
  constexpr int CUR_SEGMENT_LEN = 1 << (LOG_N > MAX_LOGN ? MAX_LOGN : LOG_N);
  target_call_cls::copy<CUR_SEGMENT_LEN>(tmem_start_col, (uint32_t *)dst_ptr);
  if constexpr (N - CUR_SEGMENT_LEN > 0) {
    tcgen05_ld_core<target_call_cls, MAX_LOGN, N - CUR_SEGMENT_LEN>(
        tmem_start_col + CUR_SEGMENT_LEN, dst_ptr + CUR_SEGMENT_LEN);
  }
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp32bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp32bNx<pack16>, 7, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

// x16-max variant: splits loads into x16 instructions for cross-WG visibility.
// Adds explicit per-warp row offset for correct cross-WG TMEM addressing.
template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp32bNx_x16(uint32_t const &tmem_start_col,
                          uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  uint32_t row_off = (uint32_t)(((threadIdx.x / 32) % 4) * 32) << 16;
  tcgen05_ld_core<tl::tmem_ld_32dp32bNx<pack16>, 4, N>(
      tmem_start_col + tmem_col_offset + row_off, dst_ptr);
  tl::fence_view_async_tmem_load();
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp64bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp64bNx<pack16>, 7, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp128bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp128bNx<pack16>, 6, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp256bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp256bNx<pack16>, 5, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

// NOTE: The column offset increment (CUR_SEGMENT_LEN) assumes each register
// maps to exactly one TMEM column (i.e. unpack::16b is NOT active). If
// unpack::16b were used, each register would expand to 2 columns, requiring
// an increment of 2*CUR_SEGMENT_LEN. Currently the codegen always passes
// unpack16=false for stores (see copy.cc use_pack_unpack_modifier), so this
// is correct. Do not enable unpack for stores without fixing this offset.
template <typename target_call_cls, int MAX_LOGN, int N, typename src_t>
__device__ __forceinline__ void tcgen05_st_core(uint32_t const &tmem_start_col,
                                                src_t const *src_ptr) {
  static_assert(N > 0);
  constexpr int LOG_N = get_floor_log2<N>();
  constexpr int CUR_SEGMENT_LEN = 1 << (LOG_N > MAX_LOGN ? MAX_LOGN : LOG_N);
  target_call_cls::template copy<CUR_SEGMENT_LEN>(tmem_start_col,
                                                  (uint32_t const *)src_ptr);
  if constexpr (N - CUR_SEGMENT_LEN > 0) {
    tcgen05_st_core<target_call_cls, MAX_LOGN, N - CUR_SEGMENT_LEN>(
        tmem_start_col + CUR_SEGMENT_LEN, src_ptr + CUR_SEGMENT_LEN);
  }
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp32bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp32bNx<unpack16>, 7, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

// x16-max variant: splits stores into x16 instructions for cross-WG visibility.
// Does NOT emit per-store wait_st — caller is responsible for calling
// fence_view_async_tmem_store() once after all batched stores complete.
// Adds explicit per-warp row offset for correct cross-WG TMEM addressing.
template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp32bNx_x16(uint32_t const &tmem_start_col,
                          uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  uint32_t row_off = (uint32_t)(((threadIdx.x / 32) % 4) * 32) << 16;
  tcgen05_st_core<tl::tmem_st_32dp32bNx<unpack16>, 4, N>(
      tmem_start_col + tmem_col_offset + row_off, src_ptr);
}

// Per-warp O correction with explicit row offsets (avo-style).
// Each warp in the correction WG handles 32 rows independently.
// tmem_base: base TMEM address for this O tile (e.g., O0_tmem[0])
// scale: per-thread rescale factor (one float per row per thread)
// warp_group_offset: threadIdx.x offset for the correction WG (e.g., 256)
template <int HeadDim>
__device__ __forceinline__ void
tmem_correction_x16(uint32_t tmem_base, float scale, int warp_group_offset) {
  int warp_in_group = ((threadIdx.x - warp_group_offset) / 32) % 4;
  uint32_t tr = (uint32_t)((warp_in_group * 32) << 16);
  uint32_t O_base = tmem_base + tr;
  constexpr int kChunks = HeadDim / 16;
  float buf[2][16];
  int cur = 0;

  // Software-pipelined x16 ld/mul/st matching avo's correction_warp_fn
  tl::tmem_ld_32dp32bNx<false>::copy<16>(O_base, (uint32_t *)buf[0]);
  for (int g = 0; g < kChunks; g++) {
    tl::fence_view_async_tmem_load();
    int nxt = cur ^ 1;
    if (g + 1 < kChunks)
      tl::tmem_ld_32dp32bNx<false>::copy<16>(O_base + (g + 1) * 16,
                                              (uint32_t *)buf[nxt]);
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      float a0 = buf[cur][i] * scale;
      float a1 = buf[cur][i + 1] * scale;
      buf[cur][i] = a0;
      buf[cur][i + 1] = a1;
    }
    tl::tmem_st_32dp32bNx<false>::copy<16>(O_base + g * 16,
                                            (uint32_t const *)buf[cur]);
    cur = nxt;
  }
  tl::fence_view_async_tmem_store();
}

// Non-template wrapper for HeadDim=128 (callable as extern)
__device__ __forceinline__ void
tmem_correction_x16_d128(uint32_t tmem_base, float scale, int warp_group_offset) {
  tl::tmem_correction_x16<128>(tmem_base, scale, warp_group_offset);
}

// ====================================================================
// Avo-exact PV MMA helpers (copied from avo/kernels/fmha_2cta_raw.cuh)
// ====================================================================

// 128×64 BMN TS-MMA: C[128×64] += A_tmem[128×K] @ B_smem[64×K]^T
// A in TMEM (K-major), B in SMEM (MN-major = transposed), C in TMEM (F32)
__device__ __forceinline__ void
tcgen05_mma_1sm_ts_128x64_bmn(uint32_t tmem_c, uint32_t tmem_a,
                               uint64_t desc_b, uint32_t accumulate) {
  uint32_t idesc = (1U << 4)               // c_format = F32
                 | (1U << 7)               // a_format = BF16
                 | (1U << 10)              // b_format = BF16
                 | (0U << 15)              // a_major = K-major
                 | (1U << 16)              // b_major = MN-major
                 | ((64U / 8) << 17)       // n_dim = 8 (N=64)
                 | ((128U / 16) << 24);    // m_dim = 8 (M=128)
  asm volatile("{                                                            \n"
               ".reg .pred p0;                                               \n"
               "setp.ne.b32 p0, %4, 0;                                       \n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p0;   \n"
               "}"
               : : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
                   "r"(accumulate));
}

// Avo-exact commit: NO shared:: qualifier (just .b64 [%0])
// This is different from TileLang's shared::cluster variant!
__device__ __forceinline__ void
tcgen05_commit_1sm(void const *smem_mbar_ptr) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  if (cute::elect_one_sync()) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        : : "r"(p));
  }
}

// Full PV MMA for one qs: avo-exact 128×64 BMN, high D first, low D second.
// V_lo_ptr: pointer to V low-D buffer (bf16, [128, 64], row stride=128B)
// V_hi_ptr: pointer to V high-D buffer (bf16, [128, 64], row stride=128B)
// P_tmem_addr: P tile TMEM base address
// O_tmem_addr: O tile TMEM base address
// accumulate: 0 for k==0 (clear), 1 for k>0 (accumulate into existing O)
// NOTE: Must be called from warp 12 only (codegen adds the guard).
// Uses elect_one_sync internally — only lane 0 of warp 12 issues MMA.
__device__ __forceinline__ void
tcgen05_pv_mma_128x64_avo(void const *V_lo_ptr, void const *V_hi_ptr,
                           uint32_t P_tmem_addr,
                           uint32_t O_tmem_addr, uint32_t accumulate) {
  if (!cute::elect_one_sync()) return;
  constexpr int kBlockN = 128;   // K-rows (seq block)
  constexpr int kTileCols = 64;  // descriptor tile width

  auto *v_lo = (bfloat16_t *)const_cast<void *>(V_lo_ptr);
  auto *v_hi = (bfloat16_t *)const_cast<void *>(V_hi_ptr);
  tl::Tcgen05SMemDescriptor desc_lo, desc_hi;
  tl::initialize_tcgen05_descriptor(
      desc_lo, v_lo, 0, 64, 0, 0, 2);
  tl::initialize_tcgen05_descriptor(
      desc_hi, v_hi, 0, 64, 0, 0, 2);

  constexpr int kStride = 2048;  // bytes per 16 K-rows: 16 * 64 * 2

  // D-tile 1 (HIGH, dim 64-127) FIRST — avo processes high D before low D
  tl::fence_proxy_async();
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr + 64, P_tmem_addr + j * 8,
        uint64_t(desc_hi + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
  // D-tile 0 (LOW, dim 0-63) SECOND
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr, P_tmem_addr + j * 8,
        uint64_t(desc_lo + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp64bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp64bNx<unpack16>, 7, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp128bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp128bNx<unpack16>, 6, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp256bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp256bNx<unpack16>, 5, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

/*q SM100 TMA 2SM load (cta_group::2) */

enum class CacheHintSm100 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3}], [%2], %4;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2, int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2, int32_t const &crd3,
                            int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4),
                 "l"(cache_hint)
               : "memory");
}

} // namespace tl
