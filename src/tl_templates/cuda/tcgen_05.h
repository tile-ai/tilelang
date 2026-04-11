#pragma once

#include <cstdint>
#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "common.h"
#include <cute/arch/cluster_sm90.hpp>

namespace tl {

template <bool use_2cta = false>
TL_DEVICE void tmem_allocate(void *dst_ptr, int num_columns) {
  uint32_t dst_intptr = smem_ptr_to_uint(dst_ptr);
  if constexpr (use_2cta) {
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(dst_intptr), "r"(num_columns));
  } else {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(dst_intptr), "r"(num_columns));
  }
}

template <bool use_2cta = false>
TL_DEVICE void tmem_deallocate(uint32_t *tmem_ptr, int num_columns) {
  if constexpr (use_2cta) {
    asm volatile("{\n\t"
                 "tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1; \n\t"
                 "}"
                 :
                 : "r"(*tmem_ptr), "r"(num_columns));
  } else {
    asm volatile("{\n\t"
                 "tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t"
                 "}"
                 :
                 : "r"(*tmem_ptr), "r"(num_columns));
  }
}

TL_DEVICE void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

TL_DEVICE void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

TL_DEVICE void fence_view_async_tmem_load() {
  asm volatile("tcgen05.wait::ld.sync.aligned; " ::);
}

TL_DEVICE void fence_view_async_tmem_store() {
  asm volatile("tcgen05.wait::st.sync.aligned; " ::);
}

// Wrapper for CUTLASS umma_arrive: elect one lane, then arrive the mbarrier
template <bool use_2cta = false>
TL_DEVICE void tcgen05_mma_arrive(void const *smem_ptr,
                                  const uint16_t cta_mask = 3) {
  uint32_t bar_intptr = smem_ptr_to_uint(smem_ptr);
  if constexpr (use_2cta) {
    // Adapted from cute::arch::umma_arrive_multicast_2x1SM
    // Arrive at CTAs specified by cta_mask (default to both)
    if (cute::elect_one_sync()) {
      asm volatile("{\n\t"
                   "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::"
                   "cluster.multicast::cluster.b64 [%0], %1; \n\t"
                   "}"
                   :
                   : "r"(bar_intptr), "h"(cta_mask)
                   : "memory");
    }
  } else {
    if (cute::elect_one_sync()) {
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                   "cluster.b64 [%0];"
                   :
                   : "r"(bar_intptr));
    }
  }
}

TL_DEVICE void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

TL_DEVICE void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}


// UTCCP: Copy scale factors from shared memory to tensor memory
// Uses tcgen05.cp.cta_group::1.32x128b.warpx4 instruction
// Must be called by one warp; only one elected thread issues the instruction.
TL_DEVICE void tcgen05_cp(uint64_t const &smem_desc,
                                  uint32_t const &tmem_col) {
  if (cute::elect_one_sync()) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :
                 : "r"(tmem_col), "l"(smem_desc));
  }
}

// Warp-level transpose of 128 uint32 elements in shared memory for UTCCP.
// Each warp (32 threads) transposes a 4x32 block in-place.
// Must be called by exactly one warp. Call __syncwarp() is embedded.
TL_DEVICE void tcgen05_sf_warp_transpose(uint32_t *smem_ptr) {
  const uint32_t lane = threadIdx.x % 32;
  uint32_t values[4];
  #pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
    values[i] = smem_ptr[(i ^ (lane >> 3)) * 32 + lane];
  __syncwarp();
  #pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
    smem_ptr[lane * 4 + (i ^ (lane >> 3))] = values[i];
}

// Build a SMEM descriptor for UTCCP scale factor copy (no swizzle, K-major)
// SBO = 128 bytes (stride between atoms on MN), LBO = 0 (single K atom)
TL_DEVICE uint64_t make_sf_smem_desc(void *smem_ptr) {
  uint32_t uint_ptr = smem_ptr_to_uint(smem_ptr);
  // SmemDescriptor bit layout:
  // [0,14): start_address >> 4
  // [16,30): leading_byte_offset >> 4 = 0
  // [32,46): stride_byte_offset >> 4 = 128/16 = 8
  // [46,48): version = 1 (SM100)
  // [61,64): layout_type = 0 (SWIZZLE_NONE)
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(uint_ptr >> 4) & 0x3FFFull;            // start_address
  // leading_byte_offset = 0 (bits [16,30))
  desc |= static_cast<uint64_t>(8u) << 32;                              // stride_byte_offset >> 4 = 8
  desc |= static_cast<uint64_t>(1u) << 46;                              // version = 1
  // layout_type = 0 (SWIZZLE_NONE), base_offset = 0, lbo_mode = 0
  return desc;
}

} // namespace tl
