#pragma once

#include "common.h"

#ifdef __CUDA_ARCH_LIST__
#if __CUDA_ARCH_LIST__ >= 900
#include "copy_sm90.h"
#endif
#if __CUDA_ARCH_LIST__ >= 1000
#include "copy_sm100.h"
#endif
#endif

namespace tl {

TL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> TL_DEVICE void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *global_ptr) {
  static_assert(N == 16 || N == 8 || N == 4);
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *global_ptr, bool cond) {
  static_assert(N == 16 || N == 8 || N == 4);
  int bytes = cond ? N : 0;
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N), "r"(bytes));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N), "r"(bytes));
  }
}

template <int kBytes> struct VecInt {};

template <> struct VecInt<1> {
  using vec_t = int8_t;
};
template <> struct VecInt<2> {
  using vec_t = int16_t;
};
template <> struct VecInt<4> {
  using vec_t = int;
};
template <> struct VecInt<8> {
  using vec_t = int64_t;
};
template <> struct VecInt<16> {
  using vec_t = int4;
};

#define LD_NC_FUNC "ld.nc.global"
#define ST_NA_FUNC "st.global"

template <typename dtype_t> TL_DEVICE dtype_t ld_nc_global(const dtype_t *ptr) {
  auto ret = ld_nc_global(
      reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr));
  return *reinterpret_cast<dtype_t *>(&ret);
}

template <> TL_DEVICE uint8_t ld_nc_global(const uint8_t *ptr) {
  uint16_t ret;
  // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit
  // constraint letter (`h` below means unsigned 16-bit)
  asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return static_cast<uint8_t>(ret);
}

template <> TL_DEVICE int16_t ld_nc_global(const int16_t *ptr) {
  uint16_t ret;
  asm volatile(LD_NC_FUNC ".s16 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return static_cast<int16_t>(ret);
}

template <> TL_DEVICE uint16_t ld_nc_global(const uint16_t *ptr) {
  uint16_t ret;
  asm volatile(LD_NC_FUNC ".u16 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return ret;
}

template <> TL_DEVICE int ld_nc_global(const int *ptr) {
  int ret;
  asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

template <> TL_DEVICE int64_t ld_nc_global(const int64_t *ptr) {
  int64_t ret;
  asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

template <> TL_DEVICE float ld_nc_global(const float *ptr) {
  float ret;
  asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

template <> TL_DEVICE int2 ld_nc_global(const int2 *ptr) {
  int2 ret;
  asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];"
               : "=r"(ret.x), "=r"(ret.y)
               : "l"(ptr));
  return ret;
}

template <> TL_DEVICE int4 ld_nc_global(const int4 *ptr) {
  int4 ret;
  asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

template <typename dtype_t>
TL_DEVICE void st_na_global(const dtype_t *ptr, const dtype_t &value) {
  st_na_global(
      reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr),
      *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(
          &value));
}

template <>
TL_DEVICE void st_na_global(const int16_t *ptr, const int16_t &value) {
  asm volatile(ST_NA_FUNC ".s16 [%0], %1;" ::"l"(ptr), "h"(value));
}

template <>
TL_DEVICE void st_na_global(const uint16_t *ptr, const uint16_t &value) {
  asm volatile(ST_NA_FUNC ".u16 [%0], %1;" ::"l"(ptr), "h"(value));
}

template <> TL_DEVICE void st_na_global(const int *ptr, const int &value) {
  asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
TL_DEVICE void st_na_global(const int64_t *ptr, const int64_t &value) {
  asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <> TL_DEVICE void st_na_global(const float *ptr, const float &value) {
  asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <> TL_DEVICE void st_na_global(const int4 *ptr, const int4 &value) {
  asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
               "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

#define LD_FUNC(ptr) ld_nc_global(ptr)
#define ST_FUNC(ptr, value) st_na_global(ptr, value)

template <int N, int UNROLL_FACTOR, typename dtype_t>
TL_DEVICE void cp_warp_impl(dtype_t const *const dst_addr,
                            dtype_t const *const src_addr) {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  constexpr int kLoopStride = 32 * UNROLL_FACTOR;
  typename std::remove_reference<decltype(LD_FUNC(src_addr + 0))>::type
      unrolled_values[UNROLL_FACTOR];
  auto __src = src_addr;
  auto __dst = dst_addr;
  for (int __i = lane_id; __i < (N / kLoopStride) * kLoopStride;
       __i += kLoopStride) {
    _Pragma("unroll") for (int __j = 0; __j < UNROLL_FACTOR; ++__j)
        unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);
    _Pragma("unroll") for (int __j = 0; __j < UNROLL_FACTOR; ++__j)
        ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);
  }
  for (int __i = (N / kLoopStride) * kLoopStride + lane_id; __i < N; __i += 32)
    ST_FUNC(__dst + __i, LD_FUNC(__src + __i));
}

/**
 * @param enable_aggressive_vectorize If set to true, the copy will be performed
 * with aggressive vectorization (e.g., using int4 for aligned and sized
 * transfers), which requires that both source and destination addresses are
 * 16-byte aligned and N*sizeof(dtype_t) is a multiple of 16 for optimal memory
 * access and throughput. If false, performs a standard element-wise copy.
 */
// todo: support more auto-vectorize later
template <int N, int UNROLL_FACTOR, bool enable_aggressive_vectorize = false,
          typename dtype_t>
TL_DEVICE void cp_warp(dtype_t const *const dst_addr,
                       dtype_t const *const src_addr) {
  if constexpr (enable_aggressive_vectorize) {
    int4 *__restrict__ dst_addr_int4 = (int4 *)dst_addr;
    const int4 *__restrict__ src_addr_int4 = (const int4 *)src_addr;
    constexpr int N_int4 = sizeof(dtype_t) * N / 16;
    cp_warp_impl<N_int4, UNROLL_FACTOR>(dst_addr_int4, src_addr_int4);
  } else {
    cp_warp_impl<N, UNROLL_FACTOR>(dst_addr, src_addr);
  }
}

template <int N, int UNROLL_FACTOR, bool enable_aggressive_vectorize = false,
          typename dtype_t>
TL_DEVICE void cp_warp(uint64_t dst_addr_uint64,
                       dtype_t const *const src_addr) {
  dtype_t *dst_addr = reinterpret_cast<dtype_t *>(dst_addr_uint64);
  if constexpr (enable_aggressive_vectorize) {
    int4 *__restrict__ dst_addr_int4 = (int4 *)dst_addr;
    const int4 *__restrict__ src_addr_int4 = (const int4 *)src_addr;
    constexpr int N_int4 = sizeof(dtype_t) * N / 16;
    cp_warp_impl<N_int4, UNROLL_FACTOR>(dst_addr_int4, src_addr_int4);
  } else {
    cp_warp_impl<N, UNROLL_FACTOR>(dst_addr, src_addr);
  }
}

template <int N, int UNROLL_FACTOR, bool enable_aggressive_vectorize = false,
          typename dtype_t>
TL_DEVICE void cp_warp(dtype_t *const dst_addr, uint64_t src_addr_uint64) {
  const dtype_t *src_addr = reinterpret_cast<const dtype_t *>(src_addr_uint64);
  if constexpr (enable_aggressive_vectorize) {
    int4 *__restrict__ dst_addr_int4 = (int4 *)dst_addr;
    const int4 *__restrict__ src_addr_int4 = (const int4 *)src_addr;
    constexpr int N_int4 = sizeof(dtype_t) * N / 16;
    cp_warp_impl<N_int4, UNROLL_FACTOR>(dst_addr_int4, src_addr_int4);
  } else {
    cp_warp_impl<N, UNROLL_FACTOR>(dst_addr, src_addr);
  }
}

/**
 * Check:
 * nvshmem_src/src/include/non_abi/device/common/nvshmemi_common_device.cuh::nvshmemi_memcpy_threadgroup()
 */
template <int N, typename dtype_t>
TL_DEVICE void nvshmem_cp_threadgroup(dtype_t *__restrict__ _dst,
                                      const dtype_t *__restrict__ _src,
                                      int myIdx, int groupSize) {
  size_t len = N * sizeof(dtype_t);
  void *dst = _dst;
  const void *src = _src;

  /*
   * If src and dst are 16B aligned copy as much as possible using 16B chunks
   */
  if ((uintptr_t)dst % 16 == 0 && (uintptr_t)src % 16 == 0) {
    const size_t nelems = len / 16;

    int4 *__restrict__ dst_p = (int4 *)dst;
    const int4 *__restrict__ src_p = (const int4 *)src;
    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];

    len -= nelems * 16;

    if (0 == len)
      return;

    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  /*
   * If src and dst are 8B aligned copy as much as possible using 8B chunks
   */
  if ((uintptr_t)dst % 8 == 0 && (uintptr_t)src % 8 == 0) {
    uint64_t *__restrict__ dst_p = (uint64_t *)dst;
    const uint64_t *__restrict__ src_p = (const uint64_t *)src;
    const size_t nelems = len / 8;

    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];

    len -= nelems * 8;

    if (0 == len)
      return;

    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  /*
   * If src and dst are 4B aligned copy as much as possible using 4B chunks
   */
  if ((uintptr_t)dst % 4 == 0 && (uintptr_t)src % 4 == 0) {
    uint32_t *__restrict__ dst_p = (uint32_t *)dst;
    const uint32_t *__restrict__ src_p = (const uint32_t *)src;
    const size_t nelems = len / 4;

    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];

    len -= nelems * 4;

    if (0 == len)
      return;

    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  /*
   * If src and dst are 2B aligned copy as much as possible using 2B chunks
   */
  if ((uintptr_t)dst % 2 == 0 && (uintptr_t)src % 2 == 0) {
    uint16_t *__restrict__ dst_p = (uint16_t *)dst;
    const uint16_t *__restrict__ src_p = (const uint16_t *)src;
    const size_t nelems = len / 2;

    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];

    len -= nelems * 2;

    if (0 == len)
      return;

    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  unsigned char *__restrict__ dst_c = (unsigned char *)dst;
  const unsigned char *__restrict__ src_c = (const unsigned char *)src;

  for (size_t i = myIdx; i < len; i += groupSize)
    dst_c[i] = src_c[i];
}

template <int N, typename dtype_t>
TL_DEVICE void nvshmem_cp_warp(dtype_t *__restrict__ dst,
                               const dtype_t *__restrict__ src) {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  nvshmem_cp_threadgroup<N>(dst, src, lane_id, 32);
}

template <int N, typename dtype_t>
TL_DEVICE void nvshmem_cp_block(dtype_t *__restrict__ dst,
                                const dtype_t *__restrict__ src) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y;
  int block_size = blockDim.x * blockDim.y * blockDim.z;
  nvshmem_cp_threadgroup<N>(dst, src, thread_id, block_size);
}

template <int N, typename dtype_t>
TL_DEVICE void cp_block(dtype_t *dst_addr, const dtype_t *src_addr) {
  nvshmem_cp_block<N>(dst_addr, src_addr);
}

template <int N, typename dtype_t>
TL_DEVICE void cp_block(uint64_t dst_addr_uint64, const dtype_t *src_addr) {
  dtype_t *dst_addr = reinterpret_cast<dtype_t *>(dst_addr_uint64);
  nvshmem_cp_block<N>(dst_addr, src_addr);
}

template <int N, typename dtype_t>
TL_DEVICE void cp_block(dtype_t *dst_addr, const uint64_t src_addr_uint64) {
  const dtype_t *src_addr = reinterpret_cast<const dtype_t *>(src_addr_uint64);
  nvshmem_cp_block<N>(dst_addr, src_addr);
}

} // namespace tl
