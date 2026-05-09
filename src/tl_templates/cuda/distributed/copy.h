#pragma once

#include "../copy.h"
#include "../common.h"
#include <cstddef>
#include <cstdint>

namespace tl {

template <typename BarrierType = uint64_t>
TL_DEVICE void tma_load(void *smem_ptr, uint64_t gmem_ptr,
                        BarrierType &smem_mbar, uint32_t size) {
  tma_load(smem_ptr, reinterpret_cast<void const *>(gmem_ptr), smem_mbar,
           size);
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(uint64_t gmem_ptr, void *smem_ptr, uint32_t size) {
  tma_store<cache_hint>(reinterpret_cast<void *>(gmem_ptr), smem_ptr, size);
}

// ---------------------------------------------------------------------------
// cp_warp / cp_block — per-warp and per-block remote memcpy via P2P pointers
// ---------------------------------------------------------------------------

template <int N, int UNROLL_FACTOR, typename dtype_t>
TL_DEVICE void cp_warp_impl(dtype_t const *const dst_addr,
                            dtype_t const *const src_addr) {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  constexpr int kLoopStride = 32 * UNROLL_FACTOR;
  int4 unrolled_values[UNROLL_FACTOR];
  auto __src = reinterpret_cast<const int4 *>(src_addr);
  auto __dst = reinterpret_cast<int4 *>(const_cast<dtype_t *>(dst_addr));
  constexpr int N_int4 = N * sizeof(dtype_t) / sizeof(int4);
  for (int __i = lane_id; __i < (N_int4 / kLoopStride) * kLoopStride;
       __i += kLoopStride) {
    _Pragma("unroll") for (int __j = 0; __j < UNROLL_FACTOR; ++__j)
        unrolled_values[__j] = __src[__i + __j * 32];
    _Pragma("unroll") for (int __j = 0; __j < UNROLL_FACTOR; ++__j)
        __dst[__i + __j * 32] = unrolled_values[__j];
  }
  for (int __i = (N_int4 / kLoopStride) * kLoopStride + lane_id;
       __i < N_int4; __i += 32)
    __dst[__i] = __src[__i];
}

template <int N, int UNROLL_FACTOR, bool enable_aggressive_vectorize = false,
          typename dtype_t>
TL_DEVICE void cp_warp(dtype_t const *const dst_addr,
                       dtype_t const *const src_addr) {
  cp_warp_impl<N, UNROLL_FACTOR>(dst_addr, src_addr);
}

template <int N, int UNROLL_FACTOR, bool enable_aggressive_vectorize = false,
          typename dtype_t>
TL_DEVICE void cp_warp(uint64_t dst_addr_uint64,
                       dtype_t const *const src_addr) {
  dtype_t *dst_addr = reinterpret_cast<dtype_t *>(dst_addr_uint64);
  cp_warp_impl<N, UNROLL_FACTOR>(dst_addr, src_addr);
}

template <int N, int UNROLL_FACTOR, bool enable_aggressive_vectorize = false,
          typename dtype_t>
TL_DEVICE void cp_warp(dtype_t *const dst_addr, uint64_t src_addr_uint64) {
  const dtype_t *src_addr = reinterpret_cast<const dtype_t *>(src_addr_uint64);
  cp_warp_impl<N, UNROLL_FACTOR>(dst_addr, src_addr);
}

// ---------------------------------------------------------------------------
// threadgroup / block copy (nvshmem-style aligned memcpy)
// ---------------------------------------------------------------------------

template <int N, typename dtype_t>
TL_DEVICE void threadgroup_cp(dtype_t *__restrict__ _dst,
                              const dtype_t *__restrict__ _src,
                              int myIdx, int groupSize) {
  size_t len = N * sizeof(dtype_t);
  void *dst = _dst;
  const void *src = _src;

  if ((uintptr_t)dst % 16 == 0 && (uintptr_t)src % 16 == 0) {
    const size_t nelems = len / 16;
    int4 *__restrict__ dst_p = (int4 *)dst;
    const int4 *__restrict__ src_p = (const int4 *)src;
    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];
    len -= nelems * 16;
    if (0 == len) return;
    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  if ((uintptr_t)dst % 8 == 0 && (uintptr_t)src % 8 == 0) {
    uint64_t *__restrict__ dst_p = (uint64_t *)dst;
    const uint64_t *__restrict__ src_p = (const uint64_t *)src;
    const size_t nelems = len / 8;
    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];
    len -= nelems * 8;
    if (0 == len) return;
    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  if ((uintptr_t)dst % 4 == 0 && (uintptr_t)src % 4 == 0) {
    uint32_t *__restrict__ dst_p = (uint32_t *)dst;
    const uint32_t *__restrict__ src_p = (const uint32_t *)src;
    const size_t nelems = len / 4;
    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];
    len -= nelems * 4;
    if (0 == len) return;
    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  if ((uintptr_t)dst % 2 == 0 && (uintptr_t)src % 2 == 0) {
    uint16_t *__restrict__ dst_p = (uint16_t *)dst;
    const uint16_t *__restrict__ src_p = (const uint16_t *)src;
    const size_t nelems = len / 2;
    for (size_t i = myIdx; i < nelems; i += groupSize)
      dst_p[i] = src_p[i];
    len -= nelems * 2;
    if (0 == len) return;
    dst = (void *)(dst_p + nelems);
    src = (void *)(src_p + nelems);
  }

  unsigned char *__restrict__ dst_c = (unsigned char *)dst;
  const unsigned char *__restrict__ src_c = (const unsigned char *)src;
  for (size_t i = myIdx; i < len; i += groupSize)
    dst_c[i] = src_c[i];
}

template <int N, typename dtype_t>
TL_DEVICE void cp_block(dtype_t *dst_addr, const dtype_t *src_addr) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y;
  int block_size = blockDim.x * blockDim.y * blockDim.z;
  threadgroup_cp<N>(dst_addr, src_addr, thread_id, block_size);
}

template <int N, typename dtype_t>
TL_DEVICE void cp_block(uint64_t dst_addr_uint64, const dtype_t *src_addr) {
  dtype_t *dst_addr = reinterpret_cast<dtype_t *>(dst_addr_uint64);
  cp_block<N>(dst_addr, src_addr);
}

template <int N, typename dtype_t>
TL_DEVICE void cp_block(dtype_t *dst_addr, const uint64_t src_addr_uint64) {
  const dtype_t *src_addr = reinterpret_cast<const dtype_t *>(src_addr_uint64);
  cp_block<N>(dst_addr, src_addr);
}

template <typename T>
TL_DEVICE T remote_load(uint64_t addr, T) {
  return *reinterpret_cast<const T *>(addr);
}

template <typename T>
TL_DEVICE void remote_store(uint64_t addr, T value) {
  *reinterpret_cast<T *>(addr) = value;
}

} // namespace tl
