#pragma once
#include "cuda_fp8.h"
#include "tcgen_05.h"
#include "tcgen_05_ld.h"

/**
 * Load four 64-bit signed integers from global memory into a longlong4.
 * @param ptr Pointer to the source longlong4 in global memory.
 * @returns A longlong4 containing the four loaded 64-bit signed lanes.
 */

/**
 * Store four 64-bit signed integers from a longlong4 into global memory.
 * @param ptr Pointer to the destination longlong4 in global memory.
 * @param val Source longlong4 whose lanes will be written to memory.
 */

/**
 * Load four 64-bit unsigned integers from global memory into a ulonglong4.
 * @param ptr Pointer to the source ulonglong4 in global memory.
 * @returns A ulonglong4 containing the four loaded 64-bit unsigned lanes.
 */

/**
 * Store four 64-bit unsigned integers from a ulonglong4 into global memory.
 * @param ptr Pointer to the destination ulonglong4 in global memory.
 * @param val Const reference to the source ulonglong4 whose lanes will be written to memory.
 *            Must be passed by const reference to avoid generation of a temporary.
 */

/**
 * Load four 64-bit values from global memory (interpreting storage for fp8_e4_32_t)
 * into a ulonglong4.
 * @param ptr Pointer to the source fp8_e4_32_t data in global memory.
 * @returns A ulonglong4 containing the four loaded 64-bit lanes.
 */

/**
 * Store four 64-bit values (interpreting storage for fp8_e4_32_t) from val8 into global memory.
 * @param ptr Pointer to the destination fp8_e4_32_t data in global memory.
 * @param val8 Reference to the source fp8_e4_32_t whose underlying 4x64-bit lanes will be written.
 */

/**
 * Load four 64-bit values from global memory (interpreting storage for fp8_e5_32_t)
 * into a ulonglong4.
 * @param ptr Pointer to the source fp8_e5_32_t data in global memory.
 * @returns A ulonglong4 containing the four loaded 64-bit lanes.
 */

/**
 * Store four 64-bit values (interpreting storage for fp8_e5_32_t) from val8 into global memory.
 * @param ptr Pointer to the destination fp8_e5_32_t data in global memory.
 * @param val8 Reference to the source fp8_e5_32_t whose underlying 4x64-bit lanes will be written.
 */

/**
 * Pack four bfloat16 values into a 64-bit unsigned value with lanes ordered as:
 * low 16 bits = x, next = y, next = z, high 16 bits = w.
 * @param x First bfloat16 lane.
 * @param y Second bfloat16 lane.
 * @param z Third bfloat16 lane.
 * @param w Fourth bfloat16 lane.
 * @returns A 64-bit unsigned integer containing the packed 4x16-bit bfloat16 lanes.
 */

/**
 * Pack four IEEE half-precision values into a 64-bit unsigned value with lanes ordered as:
 * low 16 bits = x, next = y, next = z, high 16 bits = w.
 * @param x First half lane.
 * @param y Second half lane.
 * @param z Third half lane.
 * @param w Fourth half lane.
 * @returns A 64-bit unsigned integer containing the packed 4x16-bit half lanes.
 */

/**
 * Compute the floor of log2(N) at compile time.
 * @tparam N Positive integer input; requires N > 0.
 * @tparam K Internal recursion accumulator; default is 0.
 * @returns The largest integer K such that 2^K <= N.
 */

/**
 * Recursively load N elements from thread memory into dst_ptr using the provided target loader class.
 * The loader class must provide a static template method copy<SEG_LEN>(start_col, dst).
 * @tparam target_call_cls Loader class that implements copy<SEG_LEN>(uint32_t, uint32_t*).
 * @tparam MAX_LOGN Maximum segment log2 to bound per-call segment size.
 * @tparam N Total number of elements to load; must be > 0.
 * @tparam dst_t Destination element type.
 * @param tmem_start_col Starting column index in thread memory for this load.
 * @param dst_ptr Pointer to the destination buffer where loaded elements will be written.
 */

/**
 * Load N elements using the 32dp 32-byte-per-row tmem loader (pack16 selectable) into dst_ptr
 * starting at tmem_start_col + tmem_col_offset, then fence async TMEM loads.
 * @tparam N Number of elements to load.
 * @tparam pack16 If true, use 16-element packing mode in the tmem loader.
 * @tparam dst_t Destination element type.
 * @param tmem_start_col Base starting column in thread memory.
 * @param tmem_col_offset Column offset added to tmem_start_col for this load.
 * @param dst_ptr Pointer to the destination buffer where loaded elements will be written.
 */

/**
 * Load N elements using the 32dp 64-byte-per-row tmem loader (pack16 selectable) into dst_ptr
 * starting at tmem_start_col + tmem_col_offset, then fence async TMEM loads.
 * @tparam N Number of elements to load.
 * @tparam pack16 If true, use 16-element packing mode in the tmem loader.
 * @tparam dst_t Destination element type.
 * @param tmem_start_col Base starting column in thread memory.
 * @param tmem_col_offset Column offset added to tmem_start_col for this load.
 * @param dst_ptr Pointer to the destination buffer where loaded elements will be written.
 */

/**
 * Load N elements using the 32dp 128-byte-per-row tmem loader (pack16 selectable) into dst_ptr
 * starting at tmem_start_col + tmem_col_offset, then fence async TMEM loads.
 * @tparam N Number of elements to load.
 * @tparam pack16 If true, use 16-element packing mode in the tmem loader.
 * @tparam dst_t Destination element type.
 * @param tmem_start_col Base starting column in thread memory.
 * @param tmem_col_offset Column offset added to tmem_start_col for this load.
 * @param dst_ptr Pointer to the destination buffer where loaded elements will be written.
 */

/**
 * Load N elements using the 32dp 256-byte-per-row tmem loader (pack16 selectable) into dst_ptr
 * starting at tmem_start_col + tmem_col_offset, then fence async TMEM loads.
 * @tparam N Number of elements to load.
 * @tparam pack16 If true, use 16-element packing mode in the tmem loader.
 * @tparam dst_t Destination element type.
 * @param tmem_start_col Base starting column in thread memory.
 * @param tmem_col_offset Column offset added to tmem_start_col for this load.
 * @param dst_ptr Pointer to the destination buffer where loaded elements will be written.
 */
namespace tl {

__device__ __forceinline__ longlong4 ld_global_256(const longlong4 *ptr) {
  longlong4 ret;
  asm volatile("ld.global.v4.s64 {%0, %1, %2, %3}, [%4];"
               : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
               : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_global_256(longlong4 *ptr, longlong4 &val) {
  asm volatile("st.global.v4.s64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
}

__device__ __forceinline__ ulonglong4 ld_global_256(const ulonglong4 *ptr) {
  ulonglong4 ret;
  asm volatile("ld.global.v4.u64 {%0, %1, %2, %3}, [%4];"
               : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
               : "l"(ptr));
  return ret;
}

// must be const &val, otherwise the compiler will generate a temporary variable
// and compilation will fail if we have st_global_256(ptr, ld_global_256(ptr))
__device__ __forceinline__ void st_global_256(ulonglong4 *ptr,
                                              const ulonglong4 &val) {
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
}

__device__ __forceinline__ ulonglong4 ld_global_256(const fp8_e4_32_t *ptr) {
  ulonglong4 ret;
  asm volatile("ld.global.v4.u64 {%0, %1, %2, %3}, [%4];"
               : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
               : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_global_256(fp8_e4_32_t *ptr,
                                              fp8_e4_32_t &val8) {
  ulonglong4 &val = *((ulonglong4 *)&val8);
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
}
__device__ __forceinline__ ulonglong4 ld_global_256(const fp8_e5_32_t *ptr) {
  ulonglong4 ret;
  asm volatile("ld.global.v4.u64 {%0, %1, %2, %3}, [%4];"
               : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
               : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_global_256(fp8_e5_32_t *ptr,
                                              fp8_e5_32_t &val8) {
  ulonglong4 &val = *((ulonglong4 *)&val8);
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "l"(val.x), "l"(val.y), "l"(val.z), "l"(val.w));
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