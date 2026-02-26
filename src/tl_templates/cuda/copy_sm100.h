#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "cuda_fp8.h"
#include "tcgen_05.h"
#include "tcgen_05_ld.h"
#include "tcgen_05_st.h"
#include "barrier.h"
#include "common.h"

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
                            BarrierType &smem_mbar,
                            void const *const smem_ptr, int32_t const &crd0) {
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
                            BarrierType &smem_mbar,
                            void const *const smem_ptr, int32_t const &crd0,
                            int32_t const &crd1) {
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
                            BarrierType &smem_mbar,
                            void const *const smem_ptr, int32_t const &crd0,
                            int32_t const &crd1, int32_t const &crd2) {
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
                            BarrierType &smem_mbar,
                            void const *const smem_ptr, int32_t const &crd0,
                            int32_t const &crd1, int32_t const &crd2,
                            int32_t const &crd3) {
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
                            BarrierType &smem_mbar,
                            void const *const smem_ptr, int32_t const &crd0,
                            int32_t const &crd1, int32_t const &crd2,
                            int32_t const &crd3, int32_t const &crd4) {
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
