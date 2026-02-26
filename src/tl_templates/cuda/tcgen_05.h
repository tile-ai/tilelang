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
