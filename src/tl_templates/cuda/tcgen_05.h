#pragma once

#include <cstdint>
#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "common.h"
#include <cute/arch/cluster_sm90.hpp>

namespace tl {

TL_DEVICE void tmem_allocate(void *dst_ptr, int num_columns) {
  uint32_t dst_intptr = smem_ptr_to_uint(dst_ptr);
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
      :
      : "r"(dst_intptr), "r"(num_columns));
}

TL_DEVICE void tmem_deallocate(uint32_t *tmem_ptr, int num_columns) {
  asm volatile("{\n\t"
               "tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t"
               "}"
               :
               : "r"(*tmem_ptr), "r"(num_columns));
}

inline void __device__ fence_view_async_tmem_load() {
  asm volatile("tcgen05.wait::ld.sync.aligned; " ::);
}

inline void __device__ fence_view_async_tmem_store() {
  asm volatile("tcgen05.wait::st.sync.aligned; " ::);
}

template <int M, int N>
inline void __device__ amma_fp16bf16_ss(uint64_t const desc_a,
                                        uint64_t const desc_b,
                                        uint32_t const tmem_c,
                                        uint32_t const idesc,
                                        uint32_t const addC = 1) {
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16 M-mode size should be "
                                     "64 or 128 for 1 CTA cluster MMA.");
  static_assert(
      (M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) ||
          (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
      "SM100_MMA_F16BF16 N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");

  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, "
               "%7, %8}, p; \n\t"
               "}\n"
               :
               : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(addC),
                 "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

// Wrapper for CUTLASS umma_arrive: elect one lane, then arrive the mbarrier
TL_DEVICE void tcgen05_mma_arrive(void const *smem_ptr) {
  uint32_t bar_intptr = smem_ptr_to_uint(smem_ptr);
  if (cute::elect_one_sync()) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                 "cluster.b64 [%0];"
                 :
                 : "r"(bar_intptr));
  }
}

} // namespace tl
