#pragma once
#include "cuda_fp8.h"
#include "tcgen_05.h"
#include "tcgen_05_ld.h"

namespace tl {

// 256-bit load for longlong4
__device__ __forceinline__ longlong4 ld_global_256(const longlong4 *ptr) {
  longlong4 ret;
  asm volatile("ld.global.v4.s64 {%0, %1, %2, %3}, [%4];"
               : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
               : "l"(ptr));
  return ret;
}

// 256-bit load for ulonglong4
__device__ __forceinline__ ulonglong4 ld_global_256(const ulonglong4 *ptr) {
  ulonglong4 ret;
  asm volatile("ld.global.v4.u64 {%0, %1, %2, %3}, [%4];"
               : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
               : "l"(ptr));
  return ret;
}

// Generic 256-bit load for FP8 types (returns ulonglong4)
template <typename T>
__device__ __forceinline__ ulonglong4 ld_global_256(const T *ptr) {
  ulonglong4 ret;
  asm volatile("ld.global.v4.u64 {%0, %1, %2, %3}, [%4];"
               : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
               : "l"(ptr));
  return ret;
}

// 256-bit store for longlong4
__device__ __forceinline__ void st_global_256(longlong4 *ptr, longlong4 &val) {
  asm volatile("st.global.v4.s64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
}

// 256-bit store for ulonglong4 with non-const reference
__device__ __forceinline__ void st_global_256(ulonglong4 *ptr,
                                              ulonglong4 &val) {
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
}

// 256-bit store for ulonglong4 with const reference
// must be const &val, otherwise the compiler will generate a temporary variable
// and compilation will fail if we have st_global_256(ptr, ld_global_256(ptr))
__device__ __forceinline__ void st_global_256(ulonglong4 *ptr,
                                              const ulonglong4 &val) {
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
}

// Generic 256-bit store for FP8 types
template <typename T>
__device__ __forceinline__ void st_global_256(T *ptr, const ulonglong4 &val) {
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
}

// Generic 256-bit store for FP8 types with non-const reference
template <typename T>
__device__ __forceinline__ void st_global_256(T *ptr, T &val) {
  ulonglong4 &val_u64 = *((ulonglong4 *)&val);
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val_u64.x), "l"(val_u64.y), "l"(val_u64.z),
                 "l"(val_u64.w));
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

} // namespace tl
