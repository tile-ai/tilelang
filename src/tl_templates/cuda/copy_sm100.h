#pragma once
#include "cuda_fp8.h"
#include "tcgen_05.h"
#include "tcgen_05_ld.h"

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

} // namespace tl
